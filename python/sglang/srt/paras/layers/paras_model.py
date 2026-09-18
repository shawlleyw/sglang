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

import torch
import torch.distributed as dist

from sglang.srt.paras.mode import ParaSMode
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

    def paras_transfer_unified_weights(self, mode: ParaSMode, rank: int):
        """Transfer one complete layer at a time, without rebinding KV/backend state.

        The scheduler calls the TP direction before migrating KV; graph
        initialization calls it through the ordinary configure methods.
        """
        mgr = get_global_paras_memory_manager()
        if getattr(self, "_unified_weights_mode", ParaSMode.EP) == mode:
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
        layers = self.layers if mode == ParaSMode.TP else reversed(self.layers)
        for layer in layers:
            if mode == ParaSMode.TP:
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

    def paras_configure_tp_peer_access(self, paras_tp_size: int, paras_tp_rank: int):
        self.paras_transfer_unified_weights(ParaSMode.TP, paras_tp_rank)
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
        self,
        paras_tp_size: int,
        paras_tp_rank: int,
        overlap: bool = False,
        method: str = None,
    ):
        self.paras_configure_tp_peer_access(paras_tp_size, paras_tp_rank)

    def paras_configure_ep_peer_access(self):
        from sglang.srt.paras.paras_parallel_state import get_paras_tp_rank

        self.paras_transfer_unified_weights(ParaSMode.EP, get_paras_tp_rank())
        for layer in self.layers:
            layer.paras_configure_ep_attn()
            layer.paras_configure_ep()

    @paras_func
    def paras_configure_ep(self, method: str = None):
        self.paras_configure_ep_peer_access()
