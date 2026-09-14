"""
Reusable ParaS model-level mixin.

CausalLM Integration Pattern:
===============================
For a CausalLM class to support ParaS, define these methods:

    def paras_configure_helper(self):
        torch.cuda.synchronize()

    @paras_func
    def paras_configure_tp(self, paras_tp_size: int, paras_tp_rank: int):
        self.model.paras_configure_tp(paras_tp_size, paras_tp_rank)

    @paras_func
    def paras_configure_ep(self):
        self.model.paras_configure_ep()

Where ``self.model`` is the transformer body (inheriting ParaSModelMixin).
After the switch completes, paras_configure_helper() is called by @paras_func to
synchronize CUDA.
"""

import os

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

    def paras_transfer_unified_weights(self, mode: str, rank: int):
        """Transfer one complete layer at a time, without rebinding KV/backend state.

        The scheduler calls the TP direction before migrating KV; graph
        initialization calls it through the ordinary configure methods.
        """
        mgr = get_global_paras_memory_manager()
        if getattr(self, "_unified_weights_mode", "ep") == mode:
            return
        from sglang.srt.paras.attention_transfer import transfer_attention

        group = get_paras_tp_group().device_group
        if getattr(self, "_peer_access_ctx", None) is None:
            self._peer_access_ctx = init_peer_access(mgr, group, get_paras_tp_size())
        if not hasattr(self, "_unified_peer_bases"):
            self._unified_peer_bases = torch.tensor(
                self._peer_access_ctx.peer_addresses, dtype=torch.int64, device="cuda"
            )
            self._unified_fence = torch.zeros(1, device="cuda")
        layers = self.layers if mode == "tp" else reversed(self.layers)
        for layer in layers:
            if mode == "tp":
                layer.paras_configure_tp_mlp_fused_peer_access_kernel(
                    self._peer_access_ctx, self._unified_peer_bases, None
                )
            else:
                layer.paras_configure_ep_mlp_fused_peer_access_kernel(
                    self._peer_access_ctx, self._unified_peer_bases, None
                )
            transfer_attention(
                mgr, layer.mlp._paras_layer_id, mode, rank, self._unified_peer_bases
            )
            dist.all_reduce(self._unified_fence, group=group)
        torch.cuda.synchronize()
        self._unified_weights_mode = mode

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
        if mgr.unified_workspace_enabled:
            self.paras_transfer_unified_weights("tp", paras_tp_rank)
            for layer in self.layers:
                layer.paras_configure_tp_attn(paras_tp_size, paras_tp_rank)
                layer.paras_configure_tp(paras_tp_size, paras_tp_rank)
            return

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

    def paras_finalize_attn_views(self):
        """Pre-allocate TP-mode attention weight/scale Parameters across all
        decoder layers from loaded EP weights. Must be called once after
        weight loading; paras_configure_tp/ep on attention linears then
        become pointer swaps so the captured TP CUDA graph references stable
        data_ptrs across all switches.
        """
        from sglang.srt.paras.paras_parallel_state import (
            get_paras_tp_rank,
            get_paras_tp_size,
        )

        paras_tp_size = get_paras_tp_size()
        paras_tp_rank = get_paras_tp_rank()
        for layer in self.layers:
            if hasattr(layer, "paras_finalize_attn_views"):
                layer.paras_finalize_attn_views(paras_tp_size, paras_tp_rank)

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

    def paras_configure_ep_naive(self):
        """Sequential TP→EP: reverse weight transfer + attn/communicator restore.

        Single pass in reverse layer order, mirroring paras_configure_tp_naive.
        """
        for layer in reversed(self.layers):
            layer.paras_configure_ep_attn()
            layer.paras_configure_ep_mlp_naive()
            layer.paras_configure_ep()

    def paras_configure_ep_peer_access(self):
        """TP→EP via peer access kernels (reverse layer order) + attn/communicator restore."""
        mgr = get_global_paras_memory_manager()
        if mgr.unified_workspace_enabled:
            from sglang.srt.paras.paras_parallel_state import get_paras_tp_rank

            self.paras_transfer_unified_weights("ep", get_paras_tp_rank())
            for layer in self.layers:
                layer.paras_configure_ep_attn()
                layer.paras_configure_ep()
            return

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

        for layer in reversed(self.layers):
            layer.paras_configure_ep_mlp_fused_peer_access_kernel(peer_ctx, dst_base_ptrs, None)
            dist.all_reduce(barrier_tensor, group=paras_tp_group)
            layer.paras_configure_ep_attn()
            layer.paras_configure_ep()

    @paras_func
    def paras_configure_ep(self, method: str = None):
        """Configure all layers back to EP mode."""
        if method is None:
            method = os.environ.get("PARAS_CONFIGURE_METHOD", "peer_access")
        if method == "peer_access":
            self.paras_configure_ep_peer_access()
        else:
            self.paras_configure_ep_naive()
