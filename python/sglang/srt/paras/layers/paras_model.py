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
from sglang.srt.paras.paras_parallel_state import (
    get_paras_dp_size,
    get_paras_ep_group,
    get_paras_ep_size,
    get_paras_tp_group,
    get_paras_tp_size,
    is_paras_ep_group_node_local,
)
from sglang.srt.paras.peer_access import init_peer_access
from sglang.srt.paras.utils import paras_func


class ParaSModelMixin:
    """
    Mixin that adds model-level ParaS layer-iteration and conversion strategies.

    The base class must provide:
      - self.layers — list/ModuleList of decoder layers supporting paras methods
    """

    def paras_configure_tp_naive(self, paras_tp_size: int, paras_tp_rank: int):
        """Sequential (non-overlapped) EP→TP conversion for all layers.

        Reverse layer order: the four-anchor buffer overlaps TP weight layer i
        with EP weight layer i+1, so layer i+1 must be read before layer i's TP
        weights are written.
        """
        for layer in reversed(self.layers):
            layer.paras_configure_tp_attn(paras_tp_size, paras_tp_rank)
            layer.paras_configure_tp_mlp(paras_tp_size, paras_tp_rank)
            layer.paras_configure_tp(paras_tp_size, paras_tp_rank)

    def paras_configure_tp_overlap(self, paras_tp_size: int, paras_tp_rank: int):
        stream_1 = torch.cuda.Stream()
        stream_2 = torch.cuda.Stream()
        staging_1 = "_1"
        staging_2 = "_2"

        # Reverse layer order (N-1..0): TP weight layer i overlaps EP weight
        # layer i+1 in the four-anchor buffer, so i+1 must be fully consumed
        # before i's TP weights are written. The pipeline prefetches the NEXT
        # index in the reversed walk (i-1).
        nlayers = len(self.layers)
        order = list(range(nlayers - 1, -1, -1))
        self.layers[order[0]].paras_configure_tp_attn(paras_tp_size, paras_tp_rank)
        last_layer_handles = self.layers[order[0]].paras_configure_tp_mlp_all_gather(
            stream_1, [], async_op=True, staging_suffix=staging_1
        )
        for pos, i in enumerate(order):
            layer = self.layers[i]
            not_last_layer = pos < nlayers - 1
            if not_last_layer:
                next_layer = self.layers[order[pos + 1]]
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

    def _paras_weight_peer_scope(self):
        if is_paras_ep_group_node_local():
            return (
                get_paras_ep_group().device_group,
                get_paras_ep_size(),
            )

        assert get_paras_dp_size() > 1, (
            "ParaS multi-node switching requires dp_size > 1"
        )
        return (
            get_paras_tp_group().device_group,
            get_paras_tp_size(),
        )

    def _paras_weight_peer_context(self):
        mgr = get_global_paras_memory_manager()
        peer_group, peer_size = self._paras_weight_peer_scope()
        if not hasattr(self, "_peer_access_ctx") or self._peer_access_ctx is None:
            self._peer_access_ctx = init_peer_access(
                mgr, peer_group, peer_size
            )
        return self._peer_access_ctx, peer_group

    def _paras_configure_tp_peer_access_multinode(
        self,
        peer_ctx,
        dst_base_ptrs,
    ):
        """Pipeline node-local NVLink resharding with cross-node DP gathers."""

        tp_group = get_paras_tp_group().device_group
        peer_stream = torch.cuda.Stream()
        gather_stream = torch.cuda.Stream()
        barrier_tensor = torch.zeros(1, device="cuda")
        gather_handles = []

        # Reverse layer order preserves the four-anchor source/destination
        # hazard. The gather stream consumes layer i while the peer stream
        # reshards layer i-1, keeping NIC and NVLink work in flight together.
        for layer in reversed(self.layers):
            with torch.cuda.stream(peer_stream):
                layer.paras_configure_tp_mlp_fused_peer_access_kernel(
                    peer_ctx, dst_base_ptrs, peer_stream
                )
                dist.all_reduce(barrier_tensor, group=tp_group)
                peer_done = torch.cuda.Event()
                peer_done.record(peer_stream)

            with torch.cuda.stream(gather_stream):
                gather_stream.wait_event(peer_done)
                gather_handles.extend(
                    layer.paras_configure_tp_mlp_dp_all_gather(gather_stream)
                )

        for handle in gather_handles:
            handle.wait()

        current_stream = torch.cuda.current_stream()
        current_stream.wait_stream(peer_stream)
        current_stream.wait_stream(gather_stream)

    def paras_configure_tp_peer_access(
        self, paras_tp_size: int, paras_tp_rank: int
    ):
        peer_ctx, barrier_group = self._paras_weight_peer_context()
        dst_base_ptrs = torch.tensor(
            peer_ctx.peer_addresses, dtype=torch.int64, device="cuda"
        )

        if is_paras_ep_group_node_local():
            barrier_tensor = torch.zeros(1, device="cuda")
            # Reverse layer order: TP weight layer i overlaps EP weight layer
            # i+1, so i+1 must be consumed before i is written.
            for layer in reversed(self.layers):
                layer.paras_configure_tp_mlp_fused_peer_access_kernel(
                    peer_ctx, dst_base_ptrs, None
                )
                dist.all_reduce(barrier_tensor, group=barrier_group)
        else:
            self._paras_configure_tp_peer_access_multinode(
                peer_ctx, dst_base_ptrs
            )

        for layer in self.layers:
            layer.paras_configure_tp_attn(
                paras_tp_size, paras_tp_rank
            )
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
        """Sequential TP→EP weight transfer + attn/communicator restore.

        Forward layer order: writing EP weight layer i+1 overlaps TP weight
        layer i in the four-anchor buffer, so layer i must be read (i.e. layer i
        processed) before layer i+1's EP weights are written.
        """
        for layer in self.layers:
            layer.paras_configure_ep_attn()
            layer.paras_configure_ep_mlp_naive()
            layer.paras_configure_ep()

    def paras_configure_ep_peer_access(self):
        """TP→EP via peer access kernels (forward layer order)."""
        peer_ctx, barrier_group = self._paras_weight_peer_context()
        dst_base_ptrs = torch.tensor(
            peer_ctx.peer_addresses, dtype=torch.int64, device="cuda"
        )
        barrier_tensor = torch.zeros(1, device="cuda")

        # Forward layer order: EP weight layer i+1 overlaps TP weight layer i in
        # the four-anchor buffer, so i must be read before i+1's EP is written.
        for layer in self.layers:
            layer.paras_configure_ep_mlp_fused_peer_access_kernel(
                peer_ctx, dst_base_ptrs, None
            )
            dist.all_reduce(barrier_tensor, group=barrier_group)
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
