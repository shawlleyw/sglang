"""
Reusable ParaS model-level mixin.

Weight-transfer topology
========================

EP uses one wide expert-parallel group. TP means a tensor-parallel weight
layout, and ``dp_size`` is the number of independent TP instances. In a
multi-node deployment every TP instance is contained within one node, while
the DP group connects ranks with the same ``tp_rank`` across TP instances.

EP -> TP first reshards the experts owned by ``dp_rank`` within the local TP
group. When ``dp_size > 1``, an in-place DP all-gather then replicates every
expert interval to every TP instance. TP -> EP needs only the inverse local
reshard: each TP instance reads its ``dp_rank``-owned expert interval and
ignores the other replicated experts.

The peer-access context is therefore TP-local even when ``dp_size > 1``. It
contains CUDA IPC mappings only for ranks in one TP instance; the DP all-gather
is the sole cross-DP-rank communication in the EP -> TP direction.

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
    get_paras_tp_group,
    get_paras_tp_size,
)
from sglang.srt.paras.peer_access import init_peer_access
from sglang.srt.paras.utils import paras_func
from sglang.srt.paras.weight_transfer import (
    IntraNodeWeightTransferMethod,
    resolve_intra_node_weight_transfer_method,
)


class ParaSModelMixin:
    """
    Mixin that adds model-level ParaS layer-iteration and conversion strategies.

    The base class must provide:
      - self.layers — list/ModuleList of decoder layers supporting paras methods
    """

    def _paras_intra_node_peer_access_state(self):
        mgr = get_global_paras_memory_manager()
        tp_group = get_paras_tp_group().device_group
        tp_size = get_paras_tp_size()
        if not hasattr(self, "_peer_access_ctx") or self._peer_access_ctx is None:
            self._peer_access_ctx = init_peer_access(mgr, tp_group, tp_size)
        dst_base_ptrs = torch.tensor(
            self._peer_access_ctx.peer_addresses,
            dtype=torch.int64,
            device="cuda",
        )
        return dst_base_ptrs, tp_group

    def _paras_reshard_ep_to_tp_intra_node(
        self,
        layer,
        intra_node_method,
        dp_rank,
        dp_size,
        dst_base_ptrs,
        stream,
    ):
        if intra_node_method is IntraNodeWeightTransferMethod.PEER_ACCESS:
            layer.paras_reshard_ep_to_tp_intra_node_peer_access(
                dst_base_ptrs, dp_rank, dp_size, stream
            )
        else:
            layer.paras_reshard_ep_to_tp_intra_node_nccl(dp_rank, dp_size)

    def _paras_reshard_tp_to_ep_intra_node(
        self,
        layer,
        intra_node_method,
        dp_rank,
        dp_size,
        dst_base_ptrs,
        stream,
    ):
        if intra_node_method is IntraNodeWeightTransferMethod.PEER_ACCESS:
            layer.paras_reshard_tp_to_ep_intra_node_peer_access(
                dst_base_ptrs, dp_rank, dp_size, stream
            )
        else:
            layer.paras_reshard_tp_to_ep_intra_node_nccl(dp_rank, dp_size)

    def _paras_transfer_ep_to_tp(self, intra_node_method):
        dp_rank = get_paras_dp_rank()
        dp_size = get_paras_dp_size()
        dst_base_ptrs = None
        peer_group = None
        barrier_tensor = None
        if intra_node_method is IntraNodeWeightTransferMethod.PEER_ACCESS:
            dst_base_ptrs, peer_group = self._paras_intra_node_peer_access_state()
            barrier_tensor = torch.zeros(1, device="cuda")

        if dp_size == 1:
            for layer in reversed(self.layers):
                self._paras_reshard_ep_to_tp_intra_node(
                    layer,
                    intra_node_method,
                    dp_rank,
                    dp_size,
                    dst_base_ptrs,
                    None,
                )
                if peer_group is not None:
                    dist.all_reduce(barrier_tensor, group=peer_group)
            return

        intra_node_stream = torch.cuda.Stream()
        inter_node_stream = torch.cuda.Stream()
        gather_handles = []

        # Reverse layer order preserves the overlapped source/destination
        # hazard. The DP all-gather consumes layer i while the local reshard
        # prepares layer i-1, overlapping inter-node and intra-node traffic.
        for layer in reversed(self.layers):
            with torch.cuda.stream(intra_node_stream):
                self._paras_reshard_ep_to_tp_intra_node(
                    layer,
                    intra_node_method,
                    dp_rank,
                    dp_size,
                    dst_base_ptrs,
                    intra_node_stream,
                )
                if peer_group is not None:
                    dist.all_reduce(barrier_tensor, group=peer_group)
                intra_node_done = torch.cuda.Event()
                intra_node_done.record(intra_node_stream)

            with torch.cuda.stream(inter_node_stream):
                inter_node_stream.wait_event(intra_node_done)
                gather_handles.extend(
                    layer.paras_all_gather_tp_inter_node(dp_rank, dp_size)
                )

        for handle in gather_handles:
            handle.wait()

        current_stream = torch.cuda.current_stream()
        current_stream.wait_stream(intra_node_stream)
        current_stream.wait_stream(inter_node_stream)

    def _paras_transfer_tp_to_ep(self, intra_node_method):
        dp_rank = get_paras_dp_rank()
        dp_size = get_paras_dp_size()
        dst_base_ptrs = None
        peer_group = None
        barrier_tensor = None
        if intra_node_method is IntraNodeWeightTransferMethod.PEER_ACCESS:
            dst_base_ptrs, peer_group = self._paras_intra_node_peer_access_state()
            barrier_tensor = torch.zeros(1, device="cuda")

        for layer in self.layers:
            self._paras_reshard_tp_to_ep_intra_node(
                layer,
                intra_node_method,
                dp_rank,
                dp_size,
                dst_base_ptrs,
                None,
            )
            if peer_group is not None:
                dist.all_reduce(barrier_tensor, group=peer_group)

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
    def paras_configure_tp(
        self, paras_tp_size: int, paras_tp_rank: int, intra_node_method=None
    ):
        intra_node_method = resolve_intra_node_weight_transfer_method(intra_node_method)
        self._paras_transfer_ep_to_tp(intra_node_method)
        self._paras_activate_tp(paras_tp_size, paras_tp_rank)

    @paras_func
    def paras_configure_ep(self, intra_node_method=None):
        intra_node_method = resolve_intra_node_weight_transfer_method(intra_node_method)
        self._paras_transfer_tp_to_ep(intra_node_method)
        self._paras_activate_ep()
