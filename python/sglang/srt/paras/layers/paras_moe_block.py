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
    get_paras_dp_group,
    get_paras_dp_rank,
    get_paras_dp_size,
    get_paras_tp_group,
    get_paras_tp_rank,
    get_paras_tp_size,
)
from sglang.srt.paras.paras_memory_manager import get_global_paras_memory_manager
from sglang.srt.paras.peer_access import (
    peer_access_fused_transfer_w13_ep,
    peer_access_fused_transfer_w13_v2,
    peer_access_fused_transfer_w2_ep,
    peer_access_fused_transfer_w2_v2,
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

    _LOADER_PARAM_ATTRS = ("weight_loader", "output_dim", "input_dim")

    def _copy_loader_attrs(self, src_param, dst_param):
        """Propagate weight-loader attributes (weight_loader, output_dim,
        input_dim) from one Parameter to another.  Used when we replace a
        FusedMoE-allocated Parameter via ``register_parameter`` -- the
        replacement would otherwise lose the attributes the checkpoint
        loader depends on.
        """
        for attr in self._LOADER_PARAM_ATTRS:
            if hasattr(src_param, attr):
                setattr(dst_param, attr, getattr(src_param, attr))

    def _build_full_bias(self, ref_bias):
        """Allocate a full ``(num_global_experts, D)`` zero bias on this
        rank and return a Parameter that wraps it.

        The returned Parameter is what the checkpoint loader writes into:
        every rank loads the full bias directly so no transfer is needed
        at switch time.  After load, ``paras_finalize_moe_bias_views``
        replaces this full-shape Parameter with a local-slice view so the
        ep forward path can index it with local expert ids.
        """
        full = torch.zeros(
            self.num_global_experts,
            ref_bias.shape[1],
            dtype=ref_bias.dtype,
            device=ref_bias.device,
        )
        return full, nn.Parameter(full, requires_grad=False)

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

        # Allocate the full (num_global_experts, D) bias tensors on every
        # rank.  ep_experts.w{13,2}_weight_bias is registered as a full-
        # shape Parameter here so the checkpoint loader writes the full
        # bias directly into self._full_*_bias with no transfer.  The
        # ep local-slice view and tp views are built in
        # paras_finalize_moe_bias_views after load completes.
        self._full_w13_bias = None
        self._full_w2_bias = None
        if ep_with_bias and hasattr(self.ep_experts, "w13_weight_bias"):
            _old_w13 = self.ep_experts.w13_weight_bias
            self._full_w13_bias, _ep_w13_param = self._build_full_bias(_old_w13)
            self._copy_loader_attrs(_old_w13, _ep_w13_param)
            self.ep_experts.register_parameter("w13_weight_bias", _ep_w13_param)
        if ep_with_bias and hasattr(self.ep_experts, "w2_weight_bias"):
            _old_w2 = self.ep_experts.w2_weight_bias
            self._full_w2_bias, _ep_w2_param = self._build_full_bias(_old_w2)
            self._copy_loader_attrs(_old_w2, _ep_w2_param)
            self.ep_experts.register_parameter("w2_weight_bias", _ep_w2_param)

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

        paras_dp_group = get_paras_dp_group().device_group
        paras_dp_size = get_paras_dp_size()

        all_gather_handles = []
        with torch.cuda.stream(stream):
            for handle in handles:
                handle.wait()
            if paras_dp_size > 1:
                mgr = get_global_paras_memory_manager()
                w13_ep = self.ep_experts.w13_weight.data.view(
                    self.num_local_experts,
                    2 * self.moe_intermediate_size,
                    self.hidden_size,
                )
                self.w13_ep_gathered = mgr.get_view(f"staging.w13_gather{staging_suffix}").view(
                    self.num_local_experts * paras_dp_size,
                    2 * self.moe_intermediate_size,
                    self.hidden_size,
                )
                all_gather_handles.append(
                    dist.all_gather_into_tensor(
                        self.w13_ep_gathered,
                        w13_ep,
                        group=paras_dp_group,
                        async_op=True,
                    )
                )

                w2_ep = self.ep_experts.w2_weight.data.view(
                    self.num_local_experts,
                    self.hidden_size,
                    self.moe_intermediate_size,
                )
                self.w2_ep_gathered = mgr.get_view(f"staging.w2_gather{staging_suffix}").view(
                    self.num_local_experts * paras_dp_size,
                    self.hidden_size,
                    self.moe_intermediate_size,
                )
                all_gather_handles.append(
                    dist.all_gather_into_tensor(
                        self.w2_ep_gathered,
                        w2_ep,
                        group=paras_dp_group,
                        async_op=True,
                    )
                )

                self.num_local_experts *= paras_dp_size
            else:
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

        mgr = get_global_paras_memory_manager()
        paras_tp_size = get_paras_tp_size()
        paras_dp_size = get_paras_dp_size()
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

            if paras_dp_size > 1:
                w13_tp = self.w13_ep_gathered
            else:
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

            if paras_dp_size > 1:
                w2_tp = self.w2_ep_gathered
            else:
                w2_tp = mgr.get_view(tp_w2_name).reshape(self.w2_ep_gathered.shape)
            w2_handle = dist.all_to_all_single(
                output=w2_tp,
                input=w2_ep_permuted.view(self.w2_ep_gathered.shape),
                group=paras_tp_group,
                async_op=True,
            )

            w13_handle.wait()
            if paras_dp_size > 1:
                tp_w13_shape = (self.num_global_experts, 2 * moe_intermediate_size_after_tp, self.hidden_size)
                w13_post = mgr.get_view(f"staging.w13_pre_permute{staging_suffix}").view(
                    paras_dp_size, paras_tp_size, -1
                )
                w13_post.copy_(
                    w13_tp.view(paras_tp_size, paras_dp_size, -1).transpose(0, 1)
                )
                tp_w13 = mgr.get_view_as(tp_w13_name, tp_w13_shape)
                tp_w13.copy_(w13_post.view_as(tp_w13))

            w2_handle.wait()
            if paras_dp_size > 1:
                tp_w2_shape = (self.num_global_experts, self.hidden_size, moe_intermediate_size_after_tp)
                w2_post = mgr.get_view(f"staging.w2_pre_permute{staging_suffix}").view(
                    paras_dp_size, paras_tp_size, -1
                )
                w2_post.copy_(
                    w2_tp.view(paras_tp_size, paras_dp_size, -1).transpose(0, 1)
                )
                tp_w2 = mgr.get_view_as(tp_w2_name, tp_w2_shape)
                tp_w2.copy_(w2_post.view_as(tp_w2))

    def paras_finalize_moe_bias_views(self):
        """Build ep_experts and tp_experts bias Parameter views after
        load_weights populates ``self._full_*_bias``.

        ep_experts receives a ``(num_local_experts, D)`` local-slice view
        so its forward path indexes with local expert ids.  tp_experts
        receives a slice view sized for the paras TP layout:
          * w13 bias: ``(num_global_experts, 2*I/paras_tp)`` interleaved
            slice of the full (num_global_experts, 2*I) tensor.  For the
            GPT-OSS [g0,u0,g1,u1,...] layout, the slice
            ``[:, 2*tp_start:2*tp_end]`` gives this rank's contiguous
            block of (g_k, u_k) pairs; the triton kernel reads biases
            with explicit stride_bias_n so the strided view is correct.
          * w2 bias: rank-0 registers the full ``(num_global_experts, H)``
            tensor; non-rank-0 leaves the attribute unregistered so the
            kernel skips the bias add (applied exactly once after the
            tp all-reduce).
        """
        ep_rank = self.ep_experts.moe_ep_rank
        ep_n_local = self.ep_experts.num_local_experts
        ep_start = ep_rank * ep_n_local
        ep_end = ep_start + ep_n_local
        for full_bias, ep_param_name in (
            (self._full_w13_bias, "w13_weight_bias"),
            (self._full_w2_bias, "w2_weight_bias"),
        ):
            if full_bias is None:
                continue
            ep_view = nn.Parameter(
                full_bias[ep_start:ep_end], requires_grad=False
            )
            self.ep_experts.register_parameter(ep_param_name, ep_view)

        paras_tp_size = get_paras_tp_size()
        paras_tp_rank = get_paras_tp_rank()
        if self._full_w13_bias is not None:
            i_per_tp = self.moe_intermediate_size // paras_tp_size
            tp_w13_view = self._full_w13_bias[
                :,
                2 * paras_tp_rank * i_per_tp : 2 * (paras_tp_rank + 1) * i_per_tp,
            ]
            tp_w13_param = nn.Parameter(tp_w13_view, requires_grad=False)
            set_weight_attrs(tp_w13_param, self.tp_experts.extra_weight_attrs)
            self.tp_experts.register_parameter("w13_weight_bias", tp_w13_param)
        if self._full_w2_bias is not None and paras_tp_rank == 0:
            tp_w2_param = nn.Parameter(self._full_w2_bias, requires_grad=False)
            set_weight_attrs(tp_w2_param, self.tp_experts.extra_weight_attrs)
            self.tp_experts.register_parameter("w2_weight_bias", tp_w2_param)

    def paras_configure_tp_fused_peer_access_kernel(
        self,
        peer_ctx,
        dst_base_ptrs: torch.Tensor,
        stream=None,
    ):
        """Launch NVLink-optimized v2 peer access kernels for this layer. NO barriers — caller manages them.

        The N+1 slot design guarantees no inter-layer aliasing:
          - Layer i reads local slot[i+1], writes to peer slot[i]
          - Layer i+1 reads local slot[i+2], writes to peer slot[i+1]
          - Different slots → no race → barriers only needed at sweep start/end.
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

        peer_access_fused_transfer_w13_v2(
            local_buffer_ptr, dst_base_ptrs,
            ep_w13_entry.offset_bytes, tp_w13_entry.offset_bytes,
            paras_tp_rank, paras_tp_size,
            self.num_local_experts,
            moe_intermediate_size_after_tp * self.hidden_size,
            num_gates=2, elem_size=dtype_bytes, stream=stream,
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
        """Launch TP→EP reverse NVLink peer access kernels for this layer.

        NO barriers — caller manages per-layer synchronization.
        Layer ordering must be REVERSE (N-1→0) at the model level.
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

        peer_access_fused_transfer_w13_ep(
            local_buffer_ptr, dst_base_ptrs,
            tp_w13_entry.offset_bytes, ep_w13_entry.offset_bytes,
            paras_tp_rank, paras_tp_size,
            self.num_local_experts,
            moe_intermediate_size_after_tp * self.hidden_size,
            num_gates=2, elem_size=dtype_bytes, stream=stream,
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
