"""
Reusable ParaS MoE block mixin.

Extracted from Qwen3MoeSparseMoeBlockParaS to enable any MoE model
to inherit ParaS parallelism-switching logic (EP ↔ TP).
"""

from typing import Optional

import torch
import torch.distributed as dist

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
from sglang.srt.paras.layers.utils import paras_weight_buffer
from sglang.srt.paras.utils import paras_func
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix


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
        self, config, quant_config, prefix, layer_id
    ):
        """
        Set up EP and TP expert modules for ParaS switching.
        Call this at the end of the subclass ``__init__`` (after ``super().__init__``).

        TP experts will reuse the EP buffer during switching.
        """
        # Save the EP experts that were created by the base class __init__
        self.ep_experts = self.experts

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
        )

        self.num_global_experts = config.num_experts
        self.num_local_experts = self.num_global_experts // self.tp_size
        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size

        # Start in EP mode; will switch to TP after paras_configure_tp()
        self.parallelism_config = "ep"

    # ------------------------------------------------------------------
    # Weight redistribution helpers
    # ------------------------------------------------------------------

    def paras_configure_tp_all_gather(self, stream=None, handles=None, async_op=False):
        """
        All-gather EP weights across DP group to prepare for TP conversion.
        EP → DPxEP transformation.
        """
        handles = handles or []

        paras_dp_group = get_paras_dp_group().device_group
        paras_dp_size = get_paras_dp_size()

        all_gather_handles = []
        with torch.cuda.stream(stream):
            for handle in handles:
                handle.wait()
            if paras_dp_size > 1:
                w13_ep = self.ep_experts.w13_weight.data.view(
                    self.num_local_experts,
                    2 * self.moe_intermediate_size,
                    self.hidden_size,
                )
                self.w13_ep_gathered = paras_weight_buffer.get_buffer(
                    (
                        self.num_local_experts * paras_dp_size,
                        2 * self.moe_intermediate_size,
                        self.hidden_size,
                    ),
                    dtype=w13_ep.dtype,
                    device=w13_ep.device,
                )
                all_gather_handles.append(
                    dist.all_gather_into_tensor(
                        self.w13_ep_gathered,
                        w13_ep,
                        group=paras_dp_group,
                        async_op=True,
                    )
                )
                self.ep_experts.paras_drop_params("w13_weight")

                w2_ep = self.ep_experts.w2_weight.data.view(
                    self.num_local_experts,
                    self.hidden_size,
                    self.moe_intermediate_size,
                )
                self.w2_ep_gathered = paras_weight_buffer.get_buffer(
                    (
                        self.num_local_experts * paras_dp_size,
                        self.hidden_size,
                        self.moe_intermediate_size,
                    ),
                    dtype=w2_ep.dtype,
                    device=w2_ep.device,
                )
                all_gather_handles.append(
                    dist.all_gather_into_tensor(
                        self.w2_ep_gathered,
                        w2_ep,
                        group=paras_dp_group,
                        async_op=True,
                    )
                )
                self.ep_experts.paras_drop_params("w2_weight")

                self.num_local_experts *= paras_dp_size
            else:
                w13_ep_gathered = self.ep_experts.w13_weight.data.view(
                    self.num_local_experts,
                    2 * self.moe_intermediate_size,
                    self.hidden_size,
                )
                self.w13_ep_gathered = w13_ep_gathered
                self.ep_experts.paras_drop_params("w13_weight")

                w2_ep_gathered = self.ep_experts.w2_weight.data.view(
                    self.num_local_experts,
                    self.hidden_size,
                    self.moe_intermediate_size,
                )
                self.w2_ep_gathered = w2_ep_gathered
                self.ep_experts.paras_drop_params("w2_weight")

        if async_op:
            return all_gather_handles
        else:
            for handle in all_gather_handles:
                handle.wait()

    def paras_configure_tp_all_to_all(self, stream=None, handles=None):
        """
        All-to-all weight redistribution from DPxEP to DPxTP layout.
        """
        handles = handles or []

        paras_tp_size = get_paras_tp_size()
        paras_dp_size = get_paras_dp_size()
        paras_tp_group = get_paras_tp_group().device_group
        moe_intermediate_size_after_tp = self.moe_intermediate_size // paras_tp_size

        with torch.cuda.stream(stream):
            for handle in handles:
                handle.wait()
            w13_ep = self.w13_ep_gathered.view(
                self.num_local_experts,
                2,
                paras_tp_size,
                moe_intermediate_size_after_tp * self.hidden_size,
            )
            w13_ep_permuted = w13_ep.permute(2, 0, 1, 3).contiguous()
            w13_tp = w13_ep  # reuse memory
            w13_handle = dist.all_to_all_single(
                output=w13_tp,
                input=w13_ep_permuted,
                group=paras_tp_group,
                async_op=True,
            )

            w2_ep = self.w2_ep_gathered.data.view(
                self.num_local_experts,
                self.hidden_size,
                paras_tp_size,
                moe_intermediate_size_after_tp,
            )
            w2_ep_permuted = w2_ep.permute(2, 0, 1, 3).contiguous()
            w2_tp = w2_ep  # reuse memory
            w2_handle = dist.all_to_all_single(
                output=w2_tp,
                input=w2_ep_permuted,
                group=paras_tp_group,
                async_op=True,
            )

            w13_handle.wait()
            if paras_dp_size > 1:
                w13_tp_permuted = w13_ep_permuted.view(
                    paras_dp_size, paras_tp_size, -1
                )
                w13_tp_permuted.copy_(
                    w13_tp.view(paras_tp_size, paras_dp_size, -1).transpose(0, 1)
                )
                w13_tp_weight = w13_tp_permuted
                paras_weight_buffer.put(w13_tp)
            else:
                w13_tp_weight = w13_tp
                paras_weight_buffer.put(w13_ep_permuted)
            self.tp_experts.paras_load_params(
                w13_tp_weight.view(
                    self.num_global_experts,
                    2 * moe_intermediate_size_after_tp,
                    self.hidden_size,
                ),
                "w13_weight",
            )

            w2_handle.wait()
            if paras_dp_size > 1:
                w2_tp_permuted = w2_ep_permuted.view(
                    paras_dp_size, paras_tp_size, -1
                )
                w2_tp_permuted.copy_(
                    w2_tp.view(paras_tp_size, paras_dp_size, -1).transpose(0, 1)
                )
                w2_tp_weight = w2_tp_permuted
                paras_weight_buffer.put(w2_tp)
            else:
                w2_tp_weight = w2_tp
                paras_weight_buffer.put(w2_ep_permuted)
            self.tp_experts.paras_load_params(
                w2_tp_weight.view(
                    self.num_global_experts,
                    self.hidden_size,
                    moe_intermediate_size_after_tp,
                ),
                "w2_weight",
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
        assert False, (
            "ParaSMoeBlockMixin does not support configure back to EP "
            "at this moment"
        )
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
