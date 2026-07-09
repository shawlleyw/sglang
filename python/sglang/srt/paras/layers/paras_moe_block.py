"""
Reusable ParaS MoE block mixin.

Extracted from Qwen3MoeSparseMoeBlockParaS to enable any MoE model
to inherit ParaS parallelism-switching logic (EP ↔ TP).
"""

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.paras.paras_parallel_state import (
    get_paras_dp_rank,
    get_paras_dp_size,
    get_paras_tp_group,
    get_paras_tp_rank,
    get_paras_tp_size,
)
from sglang.srt.paras.paras_memory_manager import get_global_paras_memory_manager
from sglang.srt.paras.peer_access import (
    peer_access_fused_transfer_w2_dptp,
    peer_access_fused_transfer_w2_ep,
    peer_access_fused_transfer_w2_ep_dptp,
    peer_access_fused_transfer_w2_v2,
    peer_access_fused_transfer_w13_dptp,
    peer_access_fused_transfer_w13_ep,
    peer_access_fused_transfer_w13_ep_dptp,
    peer_access_fused_transfer_w13_v2,
)
from sglang.srt.paras.utils import paras_func
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, set_weight_attrs


class ParaSMoeBlockMixin:
    """
    Mixin that adds ParaS parallelism-switching (EP → TP) to any MoE block.

    The base class must provide:
      - self.experts       — current expert module (FusedMoE or similar)
      - self.tp_size       — tensor-parallel world size
      - forward_normal(hidden_states, should_allreduce_fusion, use_reduce_scatter)
      - forward_deepep(hidden_states, forward_batch)
    """

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def paras_init_moe(
        self,
        config,
        quant_config,
        prefix,
        layer_id,
        *,
        interleaved_w13: bool = False,
        activation: str = "silu",
        gemm1_alpha: Optional[float] = None,
        gemm1_clamp_limit: Optional[float] = None,
    ):
        """
        Set up EP and TP expert modules for ParaS switching.
        Call this at the end of the subclass ``__init__`` (after ``super().__init__``).

        TP experts will reuse the EP buffer during switching.

        Args:
            interleaved_w13: If True, treat the w13 (gate_up) 2*I axis as
                interleaved [g0, u0, g1, u1, ..., g_{I-1}, u_{I-1}] during
                EP<->TP transport (the GPT-OSS convention, exercised by
                ``swiglu_with_alpha_and_limit``'s ``x[..., ::2]``/``[..., 1::2]``
                split).  If False (default), treat the 2*I axis as
                concatenated [gate(I) | up(I)] (the Qwen3 convention,
                exercised by ``silu_and_mul``'s half-split kernel).  The
                checkpoint layout is model-dependent; each ParaS model
                subclass must set this to match its base model.
            activation, gemm1_alpha, gemm1_clamp_limit: Forwarded to the
                tp_experts FusedMoE so tp mode runs the same nonlinearity
                as ep.  Defaults match the FusedMoE defaults (silu, no
                alpha, no clamp) used by Qwen3; GPT-OSS must pass the
                swiglu + alpha=1.702 + clamp values from its config.
        """
        # Save the EP experts that were created by the base class __init__
        self.ep_experts = self.experts
        self._paras_layer_id = layer_id
        self._paras_interleaved_w13 = interleaved_w13

        # Match ep_experts' with_bias and use_weight_loader_fused so the
        # weight_loader attribute on tp_experts params matches ep_experts'
        # (the base model's _load_normal_weights uses the fused 4-arg
        # signature when bias mappings fire).  Bound-method identity is not
        # preserved across attribute access (each read of
        # ep_experts.weight_loader_fused returns a fresh bound-method
        # wrapper), so compare the underlying functions via __func__.
        ep_with_bias = getattr(
            getattr(self.ep_experts, "quant_method", None), "with_bias", False
        )
        _ep_w13_loader = getattr(
            getattr(self.ep_experts, "w13_weight", None), "weight_loader", None
        )
        _ep_fused_func = getattr(
            getattr(self.ep_experts, "weight_loader_fused", None),
            "__func__",
            None,
        )
        _ep_w13_func = getattr(_ep_w13_loader, "__func__", None)
        ep_uses_fused_loader = (
            _ep_fused_func is not None and _ep_w13_func is _ep_fused_func
        )

        # TP experts: created with skip_weights_init=True because their weights
        # arrive via all-to-all redistribution at runtime (not from checkpoint).
        self.tp_experts = FusedMoE(
            num_experts=config.num_experts
            + get_global_server_args().ep_num_redundant_experts,
            top_k=config.num_experts_per_tok,
            layer_id=layer_id,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
            skip_weights_init=True,
            moe_ep_size_override=1,
            moe_ep_rank_override=0,
            moe_tp_size_override=get_paras_tp_size(),
            moe_tp_rank_override=get_paras_tp_rank(),
            paras_force_standard_dispatcher=True,
            with_bias=ep_with_bias,
            use_weight_loader_fused=ep_uses_fused_loader,
            activation=activation,
            gemm1_alpha=gemm1_alpha,
            gemm1_clamp_limit=gemm1_clamp_limit,
        )

        self.num_global_experts = config.num_experts
        self.num_local_experts = self.num_global_experts // self.tp_size
        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size

        self._paras_has_moe_bias = ep_with_bias and hasattr(
            self.ep_experts, "w13_weight_bias"
        )
        self._scale_param_name_w13 = None
        self._scale_param_name_w2 = None
        self._fp8_block_size = None
        if hasattr(self.ep_experts, "w13_weight_scale_inv"):
            self._scale_param_name_w13 = "w13_weight_scale_inv"
            self._scale_param_name_w2 = "w2_weight_scale_inv"
        elif hasattr(self.ep_experts, "w13_weight_scale"):
            self._scale_param_name_w13 = "w13_weight_scale"
            self._scale_param_name_w2 = "w2_weight_scale"
        if (
            self._scale_param_name_w13 is not None
            and quant_config is not None
            and getattr(quant_config, "weight_block_size", None)
        ):
            self._fp8_block_size = quant_config.weight_block_size[0]

        # Pre-register TP expert weights using TP alias entries.
        # TP aliases point to slot i (one before EP slot i+1), so after the
        # fused peer access transfer writes TP data into slot i, these views
        # are already correct — no update_views() needed.
        mgr = get_global_paras_memory_manager()
        if mgr is not None and mgr.materialized:
            paras_tp_size = get_paras_tp_size()
            tp_inter = self.moe_intermediate_size // paras_tp_size
            tp_w13_name = f"model.layers.{layer_id}.mlp.tp_experts.w13_weight"
            tp_w2_name = f"model.layers.{layer_id}.mlp.tp_experts.w2_weight"

            tp_w13_view = mgr.get_view_as(
                tp_w13_name,
                (self.num_global_experts, 2 * tp_inter, self.hidden_size),
            )
            w13_param = torch.nn.Parameter(tp_w13_view, requires_grad=False)
            set_weight_attrs(w13_param, self.tp_experts.extra_weight_attrs)
            self.tp_experts.register_parameter("w13_weight", w13_param)

            tp_w2_view = mgr.get_view_as(
                tp_w2_name,
                (self.num_global_experts, self.hidden_size, tp_inter),
            )
            w2_param = torch.nn.Parameter(tp_w2_view, requires_grad=False)
            set_weight_attrs(w2_param, self.tp_experts.extra_weight_attrs)
            self.tp_experts.register_parameter("w2_weight", w2_param)

        # Start in EP mode; will switch to TP after paras_configure_tp()
        self.parallelism_config = "ep"

    # ------------------------------------------------------------------
    # Weight redistribution helpers
    # ------------------------------------------------------------------

    def paras_configure_tp_all_gather(self, stream=None, handles=None, async_op=False, staging_suffix=""):
        handles = handles or []

        paras_dp_size = get_paras_dp_size()
        if paras_dp_size > 1:
            raise NotImplementedError(
                "paras dp_size>1 NCCL all-gather path was removed; use the "
                "peer_access dptp kernels (peer_access_fused_transfer_w{13,2}_dptp) instead."
            )

        all_gather_handles = []
        with torch.cuda.stream(stream):
            for handle in handles:
                handle.wait()
            self.w13_ep_gathered = self.ep_experts.w13_weight.data.view(
                self.num_local_experts,
                2 * self.moe_intermediate_size,
                self.hidden_size,
            )
            self.w2_ep_gathered = self.ep_experts.w2_weight.data.view(
                self.num_local_experts,
                self.hidden_size,
                self.moe_intermediate_size,
            )

        if async_op:
            return all_gather_handles
        else:
            for handle in all_gather_handles:
                handle.wait()

    def paras_configure_tp_all_to_all(self, stream=None, handles=None, staging_suffix=""):
        handles = handles or []

        paras_dp_size = get_paras_dp_size()
        if paras_dp_size > 1:
            raise NotImplementedError(
                "paras dp_size>1 NCCL all-to-all path was removed; use the "
                "peer_access dptp kernels (peer_access_fused_transfer_w{13,2}_dptp) instead."
            )

        mgr = get_global_paras_memory_manager()
        paras_tp_size = get_paras_tp_size()
        paras_tp_group = get_paras_tp_group().device_group
        moe_intermediate_size_after_tp = self.moe_intermediate_size // paras_tp_size
        layer_id = self._paras_layer_id
        tp_w13_name = f"model.layers.{layer_id}.mlp.tp_experts.w13_weight"
        tp_w2_name = f"model.layers.{layer_id}.mlp.tp_experts.w2_weight"

        with torch.cuda.stream(stream):
            for handle in handles:
                handle.wait()

            if self._paras_interleaved_w13:
                w13_ep = self.w13_ep_gathered.view(
                    self.num_local_experts,
                    paras_tp_size,
                    2 * moe_intermediate_size_after_tp * self.hidden_size,
                )
                w13_ep_permuted = mgr.get_view(
                    f"staging.w13_pre_permute{staging_suffix}"
                ).view(
                    paras_tp_size,
                    self.num_local_experts,
                    2 * moe_intermediate_size_after_tp * self.hidden_size,
                )
                w13_ep_permuted.copy_(w13_ep.permute(1, 0, 2))
            else:
                w13_ep = self.w13_ep_gathered.view(
                    self.num_local_experts,
                    2,
                    paras_tp_size,
                    moe_intermediate_size_after_tp * self.hidden_size,
                )
                w13_ep_permuted = mgr.get_view(f"staging.w13_pre_permute{staging_suffix}").view(
                    paras_tp_size, self.num_local_experts, 2,
                    moe_intermediate_size_after_tp * self.hidden_size,
                )
                w13_ep_permuted.copy_(w13_ep.permute(2, 0, 1, 3))

            w13_tp = mgr.get_view(tp_w13_name).reshape(self.w13_ep_gathered.shape)
            w13_handle = dist.all_to_all_single(
                output=w13_tp,
                input=w13_ep_permuted.view(self.w13_ep_gathered.shape),
                group=paras_tp_group,
                async_op=True,
            )

            w2_ep = self.w2_ep_gathered.view(
                self.num_local_experts,
                self.hidden_size,
                paras_tp_size,
                moe_intermediate_size_after_tp,
            )
            w2_ep_permuted = mgr.get_view(f"staging.w2_pre_permute{staging_suffix}").view(
                paras_tp_size, self.num_local_experts, self.hidden_size,
                moe_intermediate_size_after_tp,
            )
            w2_ep_permuted.copy_(w2_ep.permute(2, 0, 1, 3))

            w2_tp = mgr.get_view(tp_w2_name).reshape(self.w2_ep_gathered.shape)
            w2_handle = dist.all_to_all_single(
                output=w2_tp,
                input=w2_ep_permuted.view(self.w2_ep_gathered.shape),
                group=paras_tp_group,
                async_op=True,
            )

            w13_handle.wait()
            w2_handle.wait()

    def _paras_gather_full_from_ep(self, local_param: nn.Parameter) -> torch.Tensor:
        """All-gather a per-rank EP-sliced Parameter into a full
        ``(num_global_experts, *trailing)`` tensor.  Caller is responsible
        for the lifetime of the returned tensor; once dropped, its storage
        is released to the caching allocator.
        """
        from sglang.srt.distributed.parallel_state import get_moe_ep_group

        ep_group = get_moe_ep_group().device_group
        local = local_param.data.contiguous()
        full = torch.empty(
            (self.num_global_experts, *local.shape[1:]),
            dtype=local.dtype,
            device=local.device,
        )
        dist.all_gather_into_tensor(full, local, group=ep_group)
        return full

    def paras_finalize_moe_biases(self):
        """Gather the EP-sharded MoE biases from peer ranks and register
        contiguous TP-sliced copies on ``tp_experts``.

        The full ``(num_global_experts, *)`` buffer used for the gather is
        a transient local; once the contiguous TP-slice has been wrapped
        as a ``tp_experts`` Parameter, the full buffer is dropped and its
        storage is released to the caching allocator.

        TP layout:
          * w13 bias: contiguous copy of the ``(num_global_experts,
            2*I/paras_tp)`` interleaved slice over the 2*I axis (GPT-OSS
            ``[g0,u0,g1,u1,...]``).
          * w2 bias: rank-0 wraps the full ``(num_global_experts, H)``
            tensor; non-rank-0 leaves the attribute unregistered so the
            kernel skips the bias add (applied exactly once after the tp
            all-reduce).
        """
        if not self._paras_has_moe_bias:
            return

        paras_tp_size = get_paras_tp_size()
        paras_tp_rank = get_paras_tp_rank()

        full_w13_bias = self._paras_gather_full_from_ep(
            self.ep_experts.w13_weight_bias
        )
        i_per_tp = self.moe_intermediate_size // paras_tp_size
        tp_w13_bias = full_w13_bias[
            :,
            2 * paras_tp_rank * i_per_tp : 2 * (paras_tp_rank + 1) * i_per_tp,
        ].contiguous()
        del full_w13_bias
        tp_w13_param = nn.Parameter(tp_w13_bias, requires_grad=False)
        set_weight_attrs(tp_w13_param, self.tp_experts.extra_weight_attrs)
        self.tp_experts.register_parameter("w13_weight_bias", tp_w13_param)

        if hasattr(self.ep_experts, "w2_weight_bias"):
            full_w2_bias = self._paras_gather_full_from_ep(
                self.ep_experts.w2_weight_bias
            )
            if paras_tp_rank == 0:
                tp_w2_param = nn.Parameter(full_w2_bias, requires_grad=False)
                set_weight_attrs(tp_w2_param, self.tp_experts.extra_weight_attrs)
                self.tp_experts.register_parameter("w2_weight_bias", tp_w2_param)
            del full_w2_bias

    def paras_finalize_moe_scales(self):
        """FP8 sibling of ``paras_finalize_moe_biases``.  Gathers the
        EP-sharded block-quant scales from peer ranks and registers
        contiguous TP-sliced copies on ``tp_experts``.

        The full scale buffer used for the gather is transient; it is
        released as soon as the contiguous TP-slice copies have been
        wrapped as ``tp_experts`` Parameters.

        TP layout depends on the w13 axis convention:
          * Qwen3 (concat ``[gate(I) | up(I)]``): cat of the two
            ``[gate-block-range, up-block-range]`` sub-slices.
          * GPT-OSS (interleaved ``[g0,u0,g1,u1,...]``): contiguous
            block range.
          * w2 scale is sliced on its last dim regardless of layout.

        See ``docs/paras/paras_fp8_support.md`` ("TP Slice Layouts").
        DeepGEMM asserts ``sf.stride(-3) == sf.size(-2) * sf.size(-1)`` so
        the slice must be materialised contiguous.
        """
        if self._scale_param_name_w13 is None or self._fp8_block_size is None:
            return
        assert self._scale_param_name_w2 is not None
        scale_name_w13 = self._scale_param_name_w13
        scale_name_w2 = self._scale_param_name_w2

        paras_tp_size = get_paras_tp_size()
        paras_tp_rank = get_paras_tp_rank()
        B = self._fp8_block_size
        I = self.moe_intermediate_size
        P = paras_tp_size

        if (I // P) % B != 0:
            raise ValueError(
                f"FP8 block alignment failed at layer {self._paras_layer_id}: "
                f"intermediate_size_per_paras_tp = I/P = {I}/{P} = {I // P} "
                f"is not a multiple of block_size B = {B}."
            )
        i_blocks_per_tp = (I // P) // B

        full_w13_scale = self._paras_gather_full_from_ep(
            getattr(self.ep_experts, scale_name_w13)
        )
        if self._paras_interleaved_w13:
            two_i_blocks_per_tp = 2 * i_blocks_per_tp
            tp_w13_scale = full_w13_scale[
                :,
                paras_tp_rank * two_i_blocks_per_tp
                : (paras_tp_rank + 1) * two_i_blocks_per_tp,
                :,
            ].contiguous()
        else:
            i_total_blocks = I // B
            gate_start = paras_tp_rank * i_blocks_per_tp
            up_start = i_total_blocks + paras_tp_rank * i_blocks_per_tp
            tp_w13_scale = torch.cat(
                [
                    full_w13_scale[:, gate_start : gate_start + i_blocks_per_tp, :],
                    full_w13_scale[:, up_start : up_start + i_blocks_per_tp, :],
                ],
                dim=1,
            ).contiguous()
        del full_w13_scale
        tp_w13_param = nn.Parameter(tp_w13_scale, requires_grad=False)
        set_weight_attrs(tp_w13_param, self.tp_experts.extra_weight_attrs)
        self.tp_experts.register_parameter(scale_name_w13, tp_w13_param)

        full_w2_scale = self._paras_gather_full_from_ep(
            getattr(self.ep_experts, scale_name_w2)
        )
        tp_w2_scale = full_w2_scale[
            :,
            :,
            paras_tp_rank * i_blocks_per_tp : (paras_tp_rank + 1) * i_blocks_per_tp,
        ].contiguous()
        del full_w2_scale
        tp_w2_param = nn.Parameter(tp_w2_scale, requires_grad=False)
        set_weight_attrs(tp_w2_param, self.tp_experts.extra_weight_attrs)
        self.tp_experts.register_parameter(scale_name_w2, tp_w2_param)

    def paras_configure_tp_fused_peer_access_kernel(
        self,
        peer_ctx,
        dst_base_ptrs: torch.Tensor,
        stream=None,
    ):
        """Launch NVLink-optimized v2 peer access kernels for this layer. NO barriers — caller manages them.

        Four-anchor layout (EP low / TP high): TP weight layer i overlaps EP
        weight layer i+1, so the caller (paras_model) drives EP->TP in REVERSE
        layer order with a per-layer barrier, ensuring layer i+1 is read before
        layer i's TP weights are written.

        w13 layout dispatch:
          - Qwen3 (concat [g0..g_I, u0..u_I]): num_gates=2, chunk=I'*H per (e, k)
          - GPT-OSS (interleaved [g0,u0,g1,u1,...]): num_gates=1, chunk=2*I'*H per
            (e,) since each rank's slab is 2*I'*H contiguous interleaved bytes.
            The kernel copies contiguous bytes regardless of interpretation, so
            the interleaving is preserved end-to-end with no .cu changes.
        """
        mgr = get_global_paras_memory_manager()
        paras_tp_size = get_paras_tp_size()
        paras_tp_rank = get_paras_tp_rank()
        layer_id = self._paras_layer_id
        moe_intermediate_size_after_tp = self.moe_intermediate_size // paras_tp_size

        ep_w13_entry = mgr._entries[f"model.layers.{layer_id}.mlp.ep_experts.w13_weight"]
        ep_w2_entry = mgr._entries[f"model.layers.{layer_id}.mlp.ep_experts.w2_weight"]

        # TP alias entries — uniform lookup, no layer_id branching
        tp_w13_entry = mgr._entries[f"model.layers.{layer_id}.mlp.tp_experts.w13_weight"]
        tp_w2_entry = mgr._entries[f"model.layers.{layer_id}.mlp.tp_experts.w2_weight"]

        local_buffer_ptr = mgr._buffer.data_ptr()
        dtype_bytes = ep_w13_entry.dtype.itemsize

        if self._paras_interleaved_w13:
            w13_num_gates = 1
            w13_chunk_elems = 2 * moe_intermediate_size_after_tp * self.hidden_size
        else:
            w13_num_gates = 2
            w13_chunk_elems = moe_intermediate_size_after_tp * self.hidden_size

        paras_dp_size = get_paras_dp_size()
        if paras_dp_size > 1:
            # Asymmetric EP -> DP x TP (G = dp_size). E_local is the EP-sharded
            # expert count (num_experts / ep_size), distinct from num_local_experts
            # (num_experts / tp_size). The dptp kernels read each shard once and
            # broadcast to the G dp-replicas.
            e_local = self.num_global_experts // (paras_dp_size * paras_tp_size)
            peer_access_fused_transfer_w13_dptp(
                local_buffer_ptr, dst_base_ptrs,
                ep_w13_entry.offset_bytes, tp_w13_entry.offset_bytes,
                paras_tp_rank, paras_tp_size, paras_dp_size,
                e_local, self.hidden_size, self.moe_intermediate_size,
                num_gates=w13_num_gates, elem_size=dtype_bytes, stream=stream,
            )
            peer_access_fused_transfer_w2_dptp(
                local_buffer_ptr, dst_base_ptrs,
                ep_w2_entry.offset_bytes, tp_w2_entry.offset_bytes,
                paras_tp_rank, paras_tp_size, paras_dp_size,
                e_local, self.hidden_size, self.moe_intermediate_size,
                elem_size=dtype_bytes, stream=stream,
            )
            return

        peer_access_fused_transfer_w13_v2(
            local_buffer_ptr, dst_base_ptrs,
            ep_w13_entry.offset_bytes, tp_w13_entry.offset_bytes,
            paras_tp_rank, paras_tp_size,
            self.num_local_experts,
            w13_chunk_elems,
            num_gates=w13_num_gates, elem_size=dtype_bytes, stream=stream,
        )
        peer_access_fused_transfer_w2_v2(
            local_buffer_ptr, dst_base_ptrs,
            ep_w2_entry.offset_bytes, tp_w2_entry.offset_bytes,
            paras_tp_rank, paras_tp_size,
            self.num_local_experts,
            hidden_size=self.hidden_size,
            full_intermediate=self.moe_intermediate_size,
            tp_intermediate=moe_intermediate_size_after_tp,
            elem_size=dtype_bytes, stream=stream,
        )

    # ------------------------------------------------------------------
    # TP→EP reverse weight redistribution helpers
    # ------------------------------------------------------------------

    def paras_configure_ep_mlp_naive(self):
        """Naive TP→EP reverse weight transfer via NCCL all_to_all (inverse of EP→TP).

        Reads from TP slot[i], all_to_all, inverse permute, writes to EP slot[i+1].
        Currently only supports dp_size==1.
        """
        mgr = get_global_paras_memory_manager()
        paras_tp_size = get_paras_tp_size()
        paras_tp_group = get_paras_tp_group().device_group
        paras_dp_size = get_paras_dp_size()
        assert paras_dp_size == 1, (
            "paras_configure_ep_mlp_naive only supports dp_size==1"
        )

        moe_inter_tp = self.moe_intermediate_size // paras_tp_size
        layer_id = self._paras_layer_id
        tp_w13_name = f"model.layers.{layer_id}.mlp.tp_experts.w13_weight"
        tp_w2_name = f"model.layers.{layer_id}.mlp.tp_experts.w2_weight"

        tp_w13 = mgr.get_view_as(
            tp_w13_name,
            (self.num_global_experts, 2 * moe_inter_tp, self.hidden_size),
        )
        if self._paras_interleaved_w13:
            tp_w13_for_a2a = tp_w13.view(
                paras_tp_size, self.num_local_experts,
                2 * moe_inter_tp * self.hidden_size,
            )
            staging_w13 = mgr.get_view("staging.w13_pre_permute").view(
                paras_tp_size, self.num_local_experts,
                2 * moe_inter_tp * self.hidden_size,
            )
            dist.all_to_all_single(
                output=staging_w13.view(tp_w13.shape),
                input=tp_w13_for_a2a.reshape(tp_w13.shape),
                group=paras_tp_group,
            )
            ep_w13 = self.ep_experts.w13_weight.data.view(
                self.num_local_experts, paras_tp_size,
                2 * moe_inter_tp * self.hidden_size,
            )
            ep_w13.copy_(staging_w13.permute(1, 0, 2))
        else:
            tp_w13_for_a2a = tp_w13.view(
                paras_tp_size, self.num_local_experts, 2,
                moe_inter_tp * self.hidden_size,
            )
            staging_w13 = mgr.get_view("staging.w13_pre_permute").view(
                paras_tp_size, self.num_local_experts, 2,
                moe_inter_tp * self.hidden_size,
            )
            dist.all_to_all_single(
                output=staging_w13.view(tp_w13.shape),
                input=tp_w13_for_a2a.reshape(tp_w13.shape),
                group=paras_tp_group,
            )
            ep_w13 = self.ep_experts.w13_weight.data.view(
                self.num_local_experts, 2, paras_tp_size,
                moe_inter_tp * self.hidden_size,
            )
            ep_w13.copy_(staging_w13.permute(1, 2, 0, 3))

        # --- w2: TP (E_total, H, I') → all_to_all → inv_permute → EP (E_local, H, I) ---
        tp_w2 = mgr.get_view_as(
            tp_w2_name,
            (self.num_global_experts, self.hidden_size, moe_inter_tp),
        )
        tp_w2_for_a2a = tp_w2.view(
            paras_tp_size, self.num_local_experts, self.hidden_size,
            moe_inter_tp,
        )

        staging_w2 = mgr.get_view("staging.w2_pre_permute").view(
            paras_tp_size, self.num_local_experts, self.hidden_size,
            moe_inter_tp,
        )
        dist.all_to_all_single(
            output=staging_w2.view(tp_w2.shape),
            input=tp_w2_for_a2a.reshape(tp_w2.shape),
            group=paras_tp_group,
        )

        ep_w2 = self.ep_experts.w2_weight.data.view(
            self.num_local_experts, self.hidden_size, paras_tp_size,
            moe_inter_tp,
        )
        ep_w2.copy_(staging_w2.permute(1, 2, 0, 3))

        # Biases require no TP->EP transport: ep_experts.w{13,2}_weight_bias
        # are views into self._full_*_bias which was populated in full on
        # every rank at load time.

    def paras_configure_ep_fused_peer_access_kernel(
        self,
        peer_ctx,
        dst_base_ptrs: torch.Tensor,
        stream=None,
    ):
        """Launch TP→EP NVLink peer access kernels for this layer.

        NO barriers — caller manages per-layer synchronization.
        Four-anchor layout: EP weight layer i+1 overlaps TP weight layer i, so
        the caller drives TP->EP in FORWARD (0->N-1) layer order, ensuring layer
        i is read before layer i+1's EP weights are written.

        w13 layout dispatch matches the EP→TP path (see
        ``paras_configure_tp_fused_peer_access_kernel``): GPT-OSS
        interleaved layout uses ``num_gates=1`` and
        ``chunk=2*I'*H`` so the contiguous (g,u) pairs stay contiguous.
        """
        mgr = get_global_paras_memory_manager()
        paras_tp_size = get_paras_tp_size()
        paras_tp_rank = get_paras_tp_rank()
        layer_id = self._paras_layer_id
        moe_intermediate_size_after_tp = self.moe_intermediate_size // paras_tp_size

        # Biases require no TP->EP transport: ep_experts views into
        # self._full_*_bias which was populated in full on every rank at
        # load time.

        tp_w13_entry = mgr._entries[f"model.layers.{layer_id}.mlp.tp_experts.w13_weight"]
        tp_w2_entry = mgr._entries[f"model.layers.{layer_id}.mlp.tp_experts.w2_weight"]
        ep_w13_entry = mgr._entries[f"model.layers.{layer_id}.mlp.ep_experts.w13_weight"]
        ep_w2_entry = mgr._entries[f"model.layers.{layer_id}.mlp.ep_experts.w2_weight"]

        local_buffer_ptr = mgr._buffer.data_ptr()
        dtype_bytes = tp_w13_entry.dtype.itemsize

        if self._paras_interleaved_w13:
            w13_num_gates = 1
            w13_chunk_elems = 2 * moe_intermediate_size_after_tp * self.hidden_size
        else:
            w13_num_gates = 2
            w13_chunk_elems = moe_intermediate_size_after_tp * self.hidden_size

        paras_dp_size = get_paras_dp_size()
        if paras_dp_size > 1:
            # Asymmetric DP x TP -> EP reverse (each dp-replica reassembles EP
            # from its own T tp-peers; redundant replicas dropped).
            e_local = self.num_global_experts // (paras_dp_size * paras_tp_size)
            peer_access_fused_transfer_w13_ep_dptp(
                local_buffer_ptr, dst_base_ptrs,
                tp_w13_entry.offset_bytes, ep_w13_entry.offset_bytes,
                paras_tp_rank, paras_tp_size, paras_dp_size,
                e_local, self.hidden_size, self.moe_intermediate_size,
                num_gates=w13_num_gates, elem_size=dtype_bytes, stream=stream,
            )
            peer_access_fused_transfer_w2_ep_dptp(
                local_buffer_ptr, dst_base_ptrs,
                tp_w2_entry.offset_bytes, ep_w2_entry.offset_bytes,
                paras_tp_rank, paras_tp_size, paras_dp_size,
                e_local, self.hidden_size, self.moe_intermediate_size,
                elem_size=dtype_bytes, stream=stream,
            )
            return

        peer_access_fused_transfer_w13_ep(
            local_buffer_ptr, dst_base_ptrs,
            tp_w13_entry.offset_bytes, ep_w13_entry.offset_bytes,
            paras_tp_rank, paras_tp_size,
            self.num_local_experts,
            w13_chunk_elems,
            num_gates=w13_num_gates, elem_size=dtype_bytes, stream=stream,
        )
        peer_access_fused_transfer_w2_ep(
            local_buffer_ptr, dst_base_ptrs,
            tp_w2_entry.offset_bytes, ep_w2_entry.offset_bytes,
            paras_tp_rank, paras_tp_size,
            self.num_local_experts,
            hidden_size=self.hidden_size,
            full_intermediate=self.moe_intermediate_size,
            tp_intermediate=moe_intermediate_size_after_tp,
            elem_size=dtype_bytes, stream=stream,
        )

    # ------------------------------------------------------------------
    # Parallelism mode switching
    # ------------------------------------------------------------------

    @paras_func
    def paras_configure_tp(self, paras_tp_size: int, paras_tp_rank: int):
        """Configure the block for TP mode."""
        self.parallelism_config = "tp"
        self.tp_size = paras_tp_size
        self.experts = self.tp_experts

    @paras_func
    def paras_configure_ep(self):
        """Configure the block back to EP mode."""
        self.parallelism_config = "ep"
        self.tp_size = 1
        self.experts = self.ep_experts

    def paras_configure_helper(self):
        pass

    # ------------------------------------------------------------------
    # Forward dispatch
    # ------------------------------------------------------------------

    def paras_forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch=None,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        """
        Mode-dispatched forward.

        - EP + DeepEP backend  → forward_deepep
        - EP + normal backend  → forward_normal
        - TP                   → forward_normal
        """
        if self.parallelism_config == "ep":
            if get_moe_a2a_backend().is_deepep():
                return self.forward_deepep(hidden_states, forward_batch)
            else:
                return self.forward_normal(
                    hidden_states, should_allreduce_fusion, use_reduce_scatter
                )
        else:
            return self.forward_normal(
                hidden_states, should_allreduce_fusion, use_reduce_scatter
            )
