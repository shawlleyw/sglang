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

from sglang.srt.paras.paras_memory_manager import get_global_paras_memory_manager
from sglang.srt.paras.paras_parallel_state import (
    get_paras_dp_rank,
    get_paras_dp_size,
    get_paras_ep_group,
    get_paras_ep_rank,
    get_paras_ep_size,
    get_paras_tp_group,
    get_paras_tp_size,
    is_paras_ep_group_node_local,
)
from sglang.srt.paras.peer_access import init_peer_access
from sglang.srt.paras.utils import paras_func
from sglang.srt.paras.weight_transfer import (
    WeightTransferMethod,
    resolve_weight_transfer_method,
)


class ParaSModelMixin:
    """
    Mixin that adds model-level ParaS layer-iteration and conversion strategies.

    The base class must provide:
      - self.layers — list/ModuleList of decoder layers supporting paras methods
    """

    def _paras_transfer_ep_to_tp_nccl(self):
        for layer in reversed(self.layers):
            layer.paras_reshard_ep_to_tp_nccl()

    def _paras_transfer_tp_to_ep_nccl(self):
        for layer in self.layers:
            layer.paras_reshard_tp_to_ep_nccl()

    def _paras_weight_peer_scope(self):
        if is_paras_ep_group_node_local():
            return (
                get_paras_ep_group().device_group,
                get_paras_ep_size(),
            )

        assert (
            get_paras_dp_size() > 1
        ), "ParaS multi-node switching requires dp_size > 1"
        return (
            get_paras_tp_group().device_group,
            get_paras_tp_size(),
        )

    def _paras_weight_peer_context(self):
        mgr = get_global_paras_memory_manager()
        peer_group, peer_size = self._paras_weight_peer_scope()
        if not hasattr(self, "_peer_access_ctx") or self._peer_access_ctx is None:
            self._peer_access_ctx = init_peer_access(mgr, peer_group, peer_size)
        return self._peer_access_ctx, peer_group

    def _paras_transfer_ep_to_tp_multinode(self, dst_base_ptrs):
        tp_group = get_paras_tp_group().device_group
        dp_size = get_paras_dp_size()
        dp_rank = get_paras_dp_rank()
        peer_stream = torch.cuda.Stream()
        gather_stream = torch.cuda.Stream()
        barrier_tensor = torch.zeros(1, device="cuda")
        gather_handles = []

        # Reverse layer order preserves the four-anchor source/destination
        # hazard. The gather stream consumes layer i while the peer stream
        # reshards layer i-1, keeping NIC and NVLink work in flight together.
        for layer in reversed(self.layers):
            with torch.cuda.stream(peer_stream):
                layer.paras_reshard_ep_to_tp_node_peer(
                    dst_base_ptrs, dp_rank, dp_size, peer_stream
                )
                dist.all_reduce(barrier_tensor, group=tp_group)
                peer_done = torch.cuda.Event()
                peer_done.record(peer_stream)

            with torch.cuda.stream(gather_stream):
                gather_stream.wait_event(peer_done)
                gather_handles.extend(layer.paras_all_gather_tp_weights(gather_stream))

        for handle in gather_handles:
            handle.wait()

        current_stream = torch.cuda.current_stream()
        current_stream.wait_stream(peer_stream)
        current_stream.wait_stream(gather_stream)

    def _paras_transfer_ep_to_tp_direct(self):
        peer_ctx, barrier_group = self._paras_weight_peer_context()
        dst_base_ptrs = torch.tensor(
            peer_ctx.peer_addresses, dtype=torch.int64, device="cuda"
        )
        dp_size = get_paras_dp_size()

        if not is_paras_ep_group_node_local():
            self._paras_transfer_ep_to_tp_multinode(dst_base_ptrs)
            return

        barrier_tensor = torch.zeros(1, device="cuda")
        ep_rank = get_paras_ep_rank()
        for layer in reversed(self.layers):
            if dp_size == 1:
                layer.paras_reshard_ep_to_tp_peer(dst_base_ptrs, None)
            else:
                layer.paras_broadcast_ep_to_dptp_peer(
                    dst_base_ptrs, ep_rank, dp_size, None
                )
            dist.all_reduce(barrier_tensor, group=barrier_group)

    def _paras_transfer_tp_to_ep_direct(self):
        peer_ctx, barrier_group = self._paras_weight_peer_context()
        dst_base_ptrs = torch.tensor(
            peer_ctx.peer_addresses, dtype=torch.int64, device="cuda"
        )
        barrier_tensor = torch.zeros(1, device="cuda")
        dp_size = get_paras_dp_size()
        ep_rank = get_paras_ep_rank()
        dp_rank = get_paras_dp_rank()

        for layer in self.layers:
            if dp_size == 1:
                layer.paras_reshard_tp_to_ep_peer(dst_base_ptrs, None)
            elif is_paras_ep_group_node_local():
                layer.paras_reshard_dptp_to_ep_peer(
                    dst_base_ptrs, ep_rank, dp_size, None
                )
            else:
                layer.paras_reshard_tp_to_ep_node_peer(
                    dst_base_ptrs, dp_rank, dp_size, None
                )
            dist.all_reduce(barrier_tensor, group=barrier_group)

    def _paras_activate_tp(self, paras_tp_size: int, paras_tp_rank: int):
        for layer in self.layers:
            layer.paras_configure_tp_attn(paras_tp_size, paras_tp_rank)
            layer.paras_configure_tp(paras_tp_size, paras_tp_rank)

    def _paras_activate_ep(self):
        for layer in self.layers:
            layer.paras_configure_ep_attn()
            layer.paras_configure_ep()

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
    def paras_configure_tp(self, paras_tp_size: int, paras_tp_rank: int, method=None):
        transfer_method = resolve_weight_transfer_method(method)
        if transfer_method is WeightTransferMethod.DIRECT:
            self._paras_transfer_ep_to_tp_direct()
        else:
            self._paras_transfer_ep_to_tp_nccl()
        self._paras_activate_tp(paras_tp_size, paras_tp_rank)

    @paras_func
    def paras_configure_ep(self, method=None):
        transfer_method = resolve_weight_transfer_method(method)
        if transfer_method is WeightTransferMethod.DIRECT:
            self._paras_transfer_tp_to_ep_direct()
        else:
            self._paras_transfer_tp_to_ep_nccl()
        self._paras_activate_ep()
