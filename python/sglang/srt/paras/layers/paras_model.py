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

    def paras_configure_tp_peer_access(self, paras_tp_size: int, paras_tp_rank: int):
        """
        Sequential EP→TP conversion using NVLink peer access per layer.

        Note: Real layer-to-layer overlap would require separating the barrier
        into pre-barrier (after permute) and post-barrier (after write), which
        requires a 2-pass approach. Left as a future optimization.
        """
        from sglang.srt.paras.paras_parallel_state import get_paras_tp_group, get_paras_tp_size
        from sglang.srt.paras.paras_memory_manager import get_global_paras_memory_manager
        from sglang.srt.paras.peer_access import init_peer_access

        mgr = get_global_paras_memory_manager()

        # Initialize peer access if not already done
        if not hasattr(self, '_peer_access_ctx') or self._peer_access_ctx is None:
            tp_group = get_paras_tp_group().device_group
            tp_size = get_paras_tp_size()
            self._peer_access_ctx = init_peer_access(mgr, tp_group, tp_size)

        peer_ctx = self._peer_access_ctx
        packed_plans = {}

        for layer in self.layers:
            layer.paras_configure_tp_attn(paras_tp_size, paras_tp_rank)
            layer.paras_configure_tp_mlp_peer_access(
                peer_ctx, {}, packed_plans, "a", None, []
            )
            layer.paras_configure_tp(paras_tp_size, paras_tp_rank)

    def paras_configure_tp_overlap(self, paras_tp_size: int, paras_tp_rank: int):
        """
        Overlapped EP→TP conversion using dual CUDA streams for pipelining.
        Overlaps the all-gather of layer i+1 with all-to-all of layer i.
        """
        stream_1 = torch.cuda.Stream()
        stream_2 = torch.cuda.Stream()

        self.layers[0].paras_configure_tp_attn(paras_tp_size, paras_tp_rank)
        last_layer_handles = self.layers[0].paras_configure_tp_mlp_all_gather(
            stream_1, [], async_op=True
        )
        nlayers = len(self.layers)
        for i, layer in enumerate(self.layers):
            not_last_layer = i < nlayers - 1
            if not_last_layer:
                next_layer = self.layers[i + 1]
                next_layer.paras_configure_tp_attn(paras_tp_size, paras_tp_rank)
                new_handles = next_layer.paras_configure_tp_mlp_all_gather(
                    stream_2, last_layer_handles, async_op=True
                )

            layer.paras_configure_tp_mlp_all_to_all(stream_1, last_layer_handles)
            layer.paras_configure_tp(paras_tp_size, paras_tp_rank)

            if not_last_layer:
                last_layer_handles = new_handles
                stream_1, stream_2 = stream_2, stream_1

    def paras_configure_tp_fused_peer_access(self, paras_tp_size: int, paras_tp_rank: int):
        from sglang.srt.paras.paras_parallel_state import get_paras_tp_group, get_paras_tp_size
        from sglang.srt.paras.paras_memory_manager import get_global_paras_memory_manager
        from sglang.srt.paras.peer_access import init_peer_access
        import torch
        import time
        import logging
        logger = logging.getLogger(__name__)

        mgr = get_global_paras_memory_manager()

        t0 = time.perf_counter()
        if not hasattr(self, '_peer_access_ctx') or self._peer_access_ctx is None:
            tp_group_tmp = get_paras_tp_group().device_group
            tp_size_tmp = get_paras_tp_size()
            self._peer_access_ctx = init_peer_access(mgr, tp_group_tmp, tp_size_tmp)
        t1 = time.perf_counter()

        peer_ctx = self._peer_access_ctx
        tp_group = get_paras_tp_group().device_group

        dst_base_ptrs = torch.tensor(
            peer_ctx.peer_addresses, dtype=torch.int64, device="cuda"
        )

        t2 = time.perf_counter()
        torch.distributed.barrier(group=tp_group)
        t3 = time.perf_counter()

        for layer in self.layers:
            layer.paras_configure_tp_attn(paras_tp_size, paras_tp_rank)
            layer.paras_configure_tp_mlp_fused_peer_access_kernel(peer_ctx, dst_base_ptrs, None)
        t4 = time.perf_counter()

        torch.cuda.synchronize()
        t5 = time.perf_counter()
        torch.distributed.barrier(group=tp_group)
        t6 = time.perf_counter()

        for layer in self.layers:
            layer.paras_configure_tp_mlp_fused_peer_access_update_views()
            layer.paras_configure_tp(paras_tp_size, paras_tp_rank)
        t7 = time.perf_counter()

        logger.warning(
            f"[fused_peer_access timing] "
            f"init={1000*(t1-t0):.1f}ms "
            f"barrier1={1000*(t3-t2):.1f}ms "
            f"kernels={1000*(t4-t3):.1f}ms "
            f"sync={1000*(t5-t4):.1f}ms "
            f"barrier2={1000*(t6-t5):.1f}ms "
            f"views={1000*(t7-t6):.1f}ms "
            f"total={1000*(t7-t0):.1f}ms"
        )

    def paras_configure_helper(self):
        pass

    @paras_func
    def paras_configure_tp(
        self, paras_tp_size: int, paras_tp_rank: int, overlap: bool = False, method: str = None
    ):
        """
        Configure the model for tensor parallelism.
        Note: the embedding layer stays in DP mode, which also works for TP.

        Args:
            method: "naive", "overlap", "peer_access", or "fused_peer_access". If None, uses overlap flag.
        """
        if method == "peer_access":
            self.paras_configure_tp_peer_access(paras_tp_size, paras_tp_rank)
        elif method == "fused_peer_access":
            self.paras_configure_tp_fused_peer_access(paras_tp_size, paras_tp_rank)
        elif method == "overlap" or (method is None and overlap):
            self.paras_configure_tp_overlap(paras_tp_size, paras_tp_rank)
        else:
            self.paras_configure_tp_naive(paras_tp_size, paras_tp_rank)

    @paras_func
    def paras_configure_ep(self):
        """Configure all layers back to EP mode."""
        for layer in self.layers:
            layer.paras_configure_ep()
