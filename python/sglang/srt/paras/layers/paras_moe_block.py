"""
Reusable ParaS MoE block mixin.

Extracted from Qwen3MoeSparseMoeBlockParaS to enable any MoE model
to inherit ParaS parallelism-switching logic (EP ↔ TP).
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.paras.paras_parallel_state import (
    get_paras_dp_group,
    get_paras_tp_group,
    get_paras_tp_rank,
    get_paras_tp_size,
)
from sglang.srt.paras.paras_memory_manager import get_global_paras_memory_manager
from sglang.srt.paras.peer_access import (
    peer_access_reshard_w2_ep_to_tp_intra_node,
    peer_access_reshard_w2_tp_to_ep_intra_node,
    peer_access_reshard_w13_ep_to_tp_intra_node,
    peer_access_reshard_w13_tp_to_ep_intra_node,
)
from sglang.srt.paras.utils import paras_func
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, set_weight_attrs


@dataclass(frozen=True)
class _WeightTransferLayout:
    local_buffer_ptr: int
    ep_w13_offset: int
    ep_w2_offset: int
    tp_w13_offset: int
    tp_w2_offset: int
    tp_size: int
    tp_rank: int
    tp_intermediate: int
    dtype_bytes: int
    w13_num_gates: int
    w13_chunk_elems: int
    hidden_size: int

    @property
    def w13_expert_bytes(self) -> int:
        return self.w13_num_gates * self.w13_chunk_elems * self.dtype_bytes

    @property
    def w2_expert_bytes(self) -> int:
        return self.hidden_size * self.tp_intermediate * self.dtype_bytes


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

    def _paras_ep_weight_views(self, num_local_experts: int):
        return (
            self.ep_experts.w13_weight.data.view(
                num_local_experts,
                2 * self.moe_intermediate_size,
                self.hidden_size,
            ),
            self.ep_experts.w2_weight.data.view(
                num_local_experts,
                self.hidden_size,
                self.moe_intermediate_size,
            ),
        )

    def _paras_tp_weight_views(self, layout: _WeightTransferLayout):
        mgr = get_global_paras_memory_manager()
        layer_id = self._paras_layer_id
        return (
            mgr.get_view_as(
                f"model.layers.{layer_id}.mlp.tp_experts.w13_weight",
                (
                    self.num_global_experts,
                    2 * layout.tp_intermediate,
                    self.hidden_size,
                ),
            ),
            mgr.get_view_as(
                f"model.layers.{layer_id}.mlp.tp_experts.w2_weight",
                (
                    self.num_global_experts,
                    self.hidden_size,
                    layout.tp_intermediate,
                ),
            ),
        )

    def _paras_nccl_staging_views(self, num_local_experts: int):
        mgr = get_global_paras_memory_manager()
        return (
            mgr.get_view_as(
                "staging.w13_pre_permute",
                (
                    num_local_experts,
                    2 * self.moe_intermediate_size,
                    self.hidden_size,
                ),
            ),
            mgr.get_view_as(
                "staging.w2_pre_permute",
                (
                    num_local_experts,
                    self.hidden_size,
                    self.moe_intermediate_size,
                ),
            ),
        )

    def paras_reshard_ep_to_tp_intra_node_nccl(self, dp_rank: int, dp_size: int):
        """Reshard this DP rank's EP experts within its TP group using NCCL."""
        layout = self._paras_weight_transfer_layout()
        (
            num_local_experts,
            dst_expert_start,
            experts_per_dp_rank,
        ) = self._paras_local_expert_interval(layout, dp_rank, dp_size)
        ep_w13, ep_w2 = self._paras_ep_weight_views(num_local_experts)
        tp_w13, tp_w2 = self._paras_tp_weight_views(layout)
        staging_w13, staging_w2 = self._paras_nccl_staging_views(num_local_experts)
        tp_group = get_paras_tp_group().device_group

        if self._paras_interleaved_w13:
            staging_w13.view(
                layout.tp_size,
                num_local_experts,
                2 * layout.tp_intermediate * self.hidden_size,
            ).copy_(
                ep_w13.view(
                    num_local_experts,
                    layout.tp_size,
                    2 * layout.tp_intermediate * self.hidden_size,
                ).permute(1, 0, 2)
            )
        else:
            staging_w13.view(
                layout.tp_size,
                num_local_experts,
                2,
                layout.tp_intermediate * self.hidden_size,
            ).copy_(
                ep_w13.view(
                    num_local_experts,
                    2,
                    layout.tp_size,
                    layout.tp_intermediate * self.hidden_size,
                ).permute(2, 0, 1, 3)
            )

        dist.all_to_all_single(
            output=tp_w13.narrow(0, dst_expert_start, experts_per_dp_rank).reshape(
                ep_w13.shape
            ),
            input=staging_w13,
            group=tp_group,
        )

        staging_w2.view(
            layout.tp_size,
            num_local_experts,
            self.hidden_size,
            layout.tp_intermediate,
        ).copy_(
            ep_w2.view(
                num_local_experts,
                self.hidden_size,
                layout.tp_size,
                layout.tp_intermediate,
            ).permute(2, 0, 1, 3)
        )
        dist.all_to_all_single(
            output=tp_w2.narrow(0, dst_expert_start, experts_per_dp_rank).reshape(
                ep_w2.shape
            ),
            input=staging_w2,
            group=tp_group,
        )

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

        full_w13_bias = self._paras_gather_full_from_ep(self.ep_experts.w13_weight_bias)
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
                paras_tp_rank
                * two_i_blocks_per_tp : (paras_tp_rank + 1)
                * two_i_blocks_per_tp,
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

    def _paras_weight_transfer_layout(self) -> _WeightTransferLayout:
        mgr = get_global_paras_memory_manager()
        tp_size = get_paras_tp_size()
        layer_id = self._paras_layer_id
        if self.moe_intermediate_size % tp_size != 0:
            raise ValueError(
                "ParaS weight transfer requires moe_intermediate_size to be "
                f"divisible by tp_size, got {self.moe_intermediate_size=} and "
                f"{tp_size=}"
            )
        tp_intermediate = self.moe_intermediate_size // tp_size

        ep_w13_entry = mgr._entries[
            f"model.layers.{layer_id}.mlp.ep_experts.w13_weight"
        ]
        ep_w2_entry = mgr._entries[f"model.layers.{layer_id}.mlp.ep_experts.w2_weight"]
        tp_w13_entry = mgr._entries[
            f"model.layers.{layer_id}.mlp.tp_experts.w13_weight"
        ]
        tp_w2_entry = mgr._entries[f"model.layers.{layer_id}.mlp.tp_experts.w2_weight"]

        if self._paras_interleaved_w13:
            w13_num_gates = 1
            w13_chunk_elems = 2 * tp_intermediate * self.hidden_size
        else:
            w13_num_gates = 2
            w13_chunk_elems = tp_intermediate * self.hidden_size

        return _WeightTransferLayout(
            local_buffer_ptr=mgr._buffer.data_ptr(),
            ep_w13_offset=ep_w13_entry.offset_bytes,
            ep_w2_offset=ep_w2_entry.offset_bytes,
            tp_w13_offset=tp_w13_entry.offset_bytes,
            tp_w2_offset=tp_w2_entry.offset_bytes,
            tp_size=tp_size,
            tp_rank=get_paras_tp_rank(),
            tp_intermediate=tp_intermediate,
            dtype_bytes=ep_w13_entry.dtype.itemsize,
            w13_num_gates=w13_num_gates,
            w13_chunk_elems=w13_chunk_elems,
            hidden_size=self.hidden_size,
        )

    def _paras_local_expert_interval(
        self,
        layout: _WeightTransferLayout,
        dp_rank: int,
        dp_size: int,
    ):
        if dp_size <= 0:
            raise ValueError(
                f"ParaS weight transfer requires dp_size > 0, got {dp_size}"
            )
        if not 0 <= dp_rank < dp_size:
            raise ValueError(
                f"ParaS weight transfer requires 0 <= dp_rank < dp_size, "
                f"got {dp_rank=} and {dp_size=}"
            )

        ep_size = dp_size * layout.tp_size
        if self.num_global_experts % ep_size != 0:
            raise ValueError(
                "ParaS weight transfer requires num_global_experts to be "
                f"divisible by dp_size * tp_size, got {self.num_global_experts=}, "
                f"{dp_size=}, and tp_size={layout.tp_size}"
            )

        num_local_experts = self.num_global_experts // ep_size
        experts_per_dp_rank = layout.tp_size * num_local_experts
        expert_start = dp_rank * experts_per_dp_rank
        return num_local_experts, expert_start, experts_per_dp_rank

    def paras_reshard_ep_to_tp_intra_node_peer_access(
        self,
        dst_base_ptrs: torch.Tensor,
        dp_rank: int,
        dp_size: int,
        stream=None,
    ):
        """Reshard this DP rank's EP experts within its local TP group."""
        layout = self._paras_weight_transfer_layout()
        (
            num_local_experts,
            dst_expert_start,
            _,
        ) = self._paras_local_expert_interval(layout, dp_rank, dp_size)

        peer_access_reshard_w13_ep_to_tp_intra_node(
            layout.local_buffer_ptr,
            dst_base_ptrs,
            layout.ep_w13_offset,
            layout.tp_w13_offset + dst_expert_start * layout.w13_expert_bytes,
            layout.tp_rank,
            layout.tp_size,
            num_local_experts,
            layout.w13_chunk_elems,
            num_gates=layout.w13_num_gates,
            elem_size=layout.dtype_bytes,
            stream=stream,
        )
        peer_access_reshard_w2_ep_to_tp_intra_node(
            layout.local_buffer_ptr,
            dst_base_ptrs,
            layout.ep_w2_offset,
            layout.tp_w2_offset + dst_expert_start * layout.w2_expert_bytes,
            layout.tp_rank,
            layout.tp_size,
            num_local_experts,
            hidden_size=layout.hidden_size,
            full_intermediate=self.moe_intermediate_size,
            tp_intermediate=layout.tp_intermediate,
            elem_size=layout.dtype_bytes,
            stream=stream,
        )

    def paras_all_gather_tp_inter_node(self, dp_rank: int, dp_size: int):
        """Replicate one TP expert interval across the DP group."""
        if dp_size <= 1:
            raise ValueError("inter-node TP all-gather requires dp_size > 1")
        layout = self._paras_weight_transfer_layout()
        _, expert_start, experts_per_dp_rank = self._paras_local_expert_interval(
            layout, dp_rank, dp_size
        )
        tp_w13, tp_w2 = self._paras_tp_weight_views(layout)
        dp_group = get_paras_dp_group().device_group

        handles = []
        for output in (tp_w13, tp_w2):
            local = output.narrow(0, expert_start, experts_per_dp_rank)
            handles.append(
                dist.all_gather_into_tensor(
                    output,
                    local,
                    group=dp_group,
                    async_op=True,
                )
            )
        return handles

    # ------------------------------------------------------------------
    # TP→EP reverse weight redistribution helpers
    # ------------------------------------------------------------------

    def paras_reshard_tp_to_ep_intra_node_nccl(self, dp_rank: int, dp_size: int):
        """Reassemble this DP rank's EP experts within its TP group using NCCL."""
        layout = self._paras_weight_transfer_layout()
        (
            num_local_experts,
            src_expert_start,
            experts_per_dp_rank,
        ) = self._paras_local_expert_interval(layout, dp_rank, dp_size)
        ep_w13, ep_w2 = self._paras_ep_weight_views(num_local_experts)
        tp_w13, tp_w2 = self._paras_tp_weight_views(layout)
        staging_w13, staging_w2 = self._paras_nccl_staging_views(num_local_experts)
        tp_group = get_paras_tp_group().device_group

        tp_w13_owned = tp_w13.narrow(0, src_expert_start, experts_per_dp_rank)
        dist.all_to_all_single(
            output=staging_w13,
            input=tp_w13_owned.reshape(ep_w13.shape),
            group=tp_group,
        )
        if self._paras_interleaved_w13:
            ep_w13.view(
                num_local_experts,
                layout.tp_size,
                2 * layout.tp_intermediate * self.hidden_size,
            ).copy_(
                staging_w13.view(
                    layout.tp_size,
                    num_local_experts,
                    2 * layout.tp_intermediate * self.hidden_size,
                ).permute(1, 0, 2)
            )
        else:
            ep_w13.view(
                num_local_experts,
                2,
                layout.tp_size,
                layout.tp_intermediate * self.hidden_size,
            ).copy_(
                staging_w13.view(
                    layout.tp_size,
                    num_local_experts,
                    2,
                    layout.tp_intermediate * self.hidden_size,
                ).permute(1, 2, 0, 3)
            )

        tp_w2_owned = tp_w2.narrow(0, src_expert_start, experts_per_dp_rank)
        dist.all_to_all_single(
            output=staging_w2,
            input=tp_w2_owned.reshape(ep_w2.shape),
            group=tp_group,
        )
        ep_w2.view(
            num_local_experts,
            self.hidden_size,
            layout.tp_size,
            layout.tp_intermediate,
        ).copy_(
            staging_w2.view(
                layout.tp_size,
                num_local_experts,
                self.hidden_size,
                layout.tp_intermediate,
            ).permute(1, 2, 0, 3)
        )

        # Biases require no TP -> EP transport: EP bias views remain populated.

    def paras_reshard_tp_to_ep_intra_node_peer_access(
        self,
        dst_base_ptrs: torch.Tensor,
        dp_rank: int,
        dp_size: int,
        stream=None,
    ):
        """Reassemble this DP rank's EP experts within its local TP group.

        TP weights are replicated across DP ranks. This operation reads only
        the expert interval owned by ``dp_rank``; all other TP experts are
        ignored when the model activates its EP views.
        """
        layout = self._paras_weight_transfer_layout()
        (
            num_local_experts,
            src_expert_start,
            _,
        ) = self._paras_local_expert_interval(layout, dp_rank, dp_size)

        peer_access_reshard_w13_tp_to_ep_intra_node(
            layout.local_buffer_ptr,
            dst_base_ptrs,
            layout.tp_w13_offset + src_expert_start * layout.w13_expert_bytes,
            layout.ep_w13_offset,
            layout.tp_rank,
            layout.tp_size,
            num_local_experts,
            layout.w13_chunk_elems,
            num_gates=layout.w13_num_gates,
            elem_size=layout.dtype_bytes,
            stream=stream,
        )
        peer_access_reshard_w2_tp_to_ep_intra_node(
            layout.local_buffer_ptr,
            dst_base_ptrs,
            layout.tp_w2_offset + src_expert_start * layout.w2_expert_bytes,
            layout.ep_w2_offset,
            layout.tp_rank,
            layout.tp_size,
            num_local_experts,
            hidden_size=layout.hidden_size,
            full_intermediate=self.moe_intermediate_size,
            tp_intermediate=layout.tp_intermediate,
            elem_size=layout.dtype_bytes,
            stream=stream,
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
