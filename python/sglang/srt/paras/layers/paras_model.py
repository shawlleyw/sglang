"""
Reusable ParaS model-level mixin.

CausalLM Integration Pattern:
===============================
For a CausalLM class to support ParaS, define these methods:

    def paras_configure_helper(self):
        torch.cuda.synchronize()
        paras_weight_buffer.release_all()

    @paras_func
    def paras_configure_tp(self, paras_tp_size: int, paras_tp_rank: int):
        self.model.paras_configure_tp(paras_tp_size, paras_tp_rank)

    @paras_func
    def paras_configure_ep(self):
        self.model.paras_configure_ep()

Where ``self.model`` is the transformer body (inheriting ParaSModelMixin).
After the switch completes, paras_configure_helper() is called by @paras_func to
synchronize CUDA and free the temporary weight redistribution buffers.
"""

import torch
import torch.distributed as dist

from sglang.srt.paras.paras_memory_manager import get_global_paras_memory_manager
from sglang.srt.paras.paras_parallel_state import get_paras_tp_group, get_paras_tp_size
from sglang.srt.paras.peer_access import init_peer_access
from sglang.srt.paras.utils import paras_func


class ParaSModelMixin:
    """
    Mixin that adds model-level ParaS layer-iteration and conversion strategies.

    The base class must provide:
      - self.layers — list/ModuleList of decoder layers supporting paras methods
    """

    def paras_configure_tp_naive(self, paras_tp_size: int, paras_tp_rank: int):
        """Sequential (non-overlapped) EP→TP conversion for all layers."""
        for layer in self.layers:
            layer.paras_configure_tp_attn(paras_tp_size, paras_tp_rank)
            layer.paras_configure_tp_mlp(paras_tp_size, paras_tp_rank)
            layer.paras_configure_tp(paras_tp_size, paras_tp_rank)

    def paras_configure_tp_overlap(self, paras_tp_size: int, paras_tp_rank: int):
        stream_1 = torch.cuda.Stream()
        stream_2 = torch.cuda.Stream()
        staging_1 = "_1"
        staging_2 = "_2"

        self.layers[0].paras_configure_tp_attn(paras_tp_size, paras_tp_rank)
        last_layer_handles = self.layers[0].paras_configure_tp_mlp_all_gather(
            stream_1, [], async_op=True, staging_suffix=staging_1
        )
        nlayers = len(self.layers)
        for i, layer in enumerate(self.layers):
            not_last_layer = i < nlayers - 1
            if not_last_layer:
                next_layer = self.layers[i + 1]
                next_layer.paras_configure_tp_attn(paras_tp_size, paras_tp_rank)
                new_handles = next_layer.paras_configure_tp_mlp_all_gather(
                    stream_2, last_layer_handles, async_op=True, staging_suffix=staging_2
                )

            layer.paras_configure_tp_mlp_all_to_all(stream_1, last_layer_handles, staging_1)
            layer.paras_configure_tp(paras_tp_size, paras_tp_rank)

            if not_last_layer:
                last_layer_handles = new_handles
                stream_1, stream_2 = stream_2, stream_1
                staging_1, staging_2 = staging_2, staging_1

    def paras_configure_tp_peer_access(self, paras_tp_size: int, paras_tp_rank: int):
        mgr = get_global_paras_memory_manager()

        if not hasattr(self, '_peer_access_ctx') or self._peer_access_ctx is None:
            tp_group_tmp = get_paras_tp_group().device_group
            tp_size_tmp = get_paras_tp_size()
            self._peer_access_ctx = init_peer_access(mgr, tp_group_tmp, tp_size_tmp)

        peer_ctx = self._peer_access_ctx
        dst_base_ptrs = torch.tensor(
            peer_ctx.peer_addresses, dtype=torch.int64, device="cuda"
        )

        paras_tp_group = get_paras_tp_group().device_group
        barrier_tensor = torch.zeros(1, device="cuda")

        for layer in self.layers:
            layer.paras_configure_tp_mlp_fused_peer_access_kernel(peer_ctx, dst_base_ptrs, None)
            dist.all_reduce(barrier_tensor, group=paras_tp_group)

        for layer in self.layers:
            layer.paras_configure_tp_attn(paras_tp_size, paras_tp_rank)
            layer.paras_configure_tp(paras_tp_size, paras_tp_rank)

    def paras_configure_helper(self):
        torch.cuda.synchronize()

    @paras_func
    def paras_configure_tp(
        self, paras_tp_size: int, paras_tp_rank: int, overlap: bool = False, method: str = None
    ):
        if method == "peer_access":
            self.paras_configure_tp_peer_access(paras_tp_size, paras_tp_rank)
        elif method == "overlap" or (method is None and overlap):
            self.paras_configure_tp_overlap(paras_tp_size, paras_tp_rank)
        else:
            self.paras_configure_tp_naive(paras_tp_size, paras_tp_rank)

    @paras_func
    def paras_configure_ep(self):
        """Configure all layers back to EP mode."""
        for layer in self.layers:
            layer.paras_configure_ep()
