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
from sglang.srt.paras.paras_memory_manager import get_global_paras_memory_manager
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
        self, config, quant_config, prefix, layer_id
    ):
        """
        Set up EP and TP expert modules for ParaS switching.
        Call this at the end of the subclass ``__init__`` (after ``super().__init__``).

        TP experts will reuse the EP buffer during switching.
        """
        # Save the EP experts that were created by the base class __init__
        self.ep_experts = self.experts
        self._paras_layer_id = layer_id

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

        # Pre-register TP expert weights pointing to EP buffer with TP shape.
        # Data is initially invalid (holds EP weights), but after EP→TP switch
        # the all-to-all writes TP data into the same buffer in-place.
        # This eliminates paras_load_params calls during the switch.
        mgr = get_global_paras_memory_manager()
        if mgr is not None and mgr.materialized:
            paras_tp_size = get_paras_tp_size()
            tp_inter = self.moe_intermediate_size // paras_tp_size
            ep_w13_name = f"model.layers.{layer_id}.mlp.experts.w13_weight"
            ep_w2_name = f"model.layers.{layer_id}.mlp.experts.w2_weight"

            tp_w13_view = mgr.get_view_as(
                ep_w13_name,
                (self.num_global_experts, 2 * tp_inter, self.hidden_size),
            )
            w13_param = torch.nn.Parameter(tp_w13_view, requires_grad=False)
            set_weight_attrs(w13_param, self.tp_experts.extra_weight_attrs)
            self.tp_experts.register_parameter("w13_weight", w13_param)

            tp_w2_view = mgr.get_view_as(
                ep_w2_name,
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
                mgr = get_global_paras_memory_manager()
                w13_ep = self.ep_experts.w13_weight.data.view(
                    self.num_local_experts,
                    2 * self.moe_intermediate_size,
                    self.hidden_size,
                )
                self.w13_ep_gathered = mgr.get_view("staging.w13_a").view(
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
                self.ep_experts.paras_drop_params("w13_weight")

                w2_ep = self.ep_experts.w2_weight.data.view(
                    self.num_local_experts,
                    self.hidden_size,
                    self.moe_intermediate_size,
                )
                self.w2_ep_gathered = mgr.get_view("staging.w2_a").view(
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
        Uses static staging buffers from the memory manager instead of
        dynamic allocation, and reuses the EP managed buffer for TP weights.
        """
        handles = handles or []

        mgr = get_global_paras_memory_manager()
        paras_tp_size = get_paras_tp_size()
        paras_dp_size = get_paras_dp_size()
        paras_tp_group = get_paras_tp_group().device_group
        moe_intermediate_size_after_tp = self.moe_intermediate_size // paras_tp_size

        with torch.cuda.stream(stream):
            for handle in handles:
                handle.wait()

            # -- w13: permute into staging, all-to-all back into gathered buf --
            w13_ep = self.w13_ep_gathered.view(
                self.num_local_experts,
                2,
                paras_tp_size,
                moe_intermediate_size_after_tp * self.hidden_size,
            )
            # Use staging_b (dp>1) or staging_a (dp==1) for the permuted copy
            w13_staging_name = "staging.w13_b" if paras_dp_size > 1 else "staging.w13_a"
            w13_ep_permuted = mgr.get_view(w13_staging_name).view(
                paras_tp_size, self.num_local_experts, 2,
                moe_intermediate_size_after_tp * self.hidden_size,
            )
            w13_ep_permuted.copy_(w13_ep.permute(2, 0, 1, 3))
            w13_tp = self.w13_ep_gathered  # all-to-all writes into gathered buf
            w13_handle = dist.all_to_all_single(
                output=w13_tp,
                input=w13_ep_permuted.view(self.w13_ep_gathered.shape),
                group=paras_tp_group,
                 async_op=True,
            )

            # -- w2: same pattern --
            w2_ep = self.w2_ep_gathered.view(
               self.num_local_experts,
               self.hidden_size,
               paras_tp_size,
               moe_intermediate_size_after_tp,
            )
            w2_staging_name = "staging.w2_b" if paras_dp_size > 1 else "staging.w2_a"
            w2_ep_permuted = mgr.get_view(w2_staging_name).view(
               paras_tp_size, self.num_local_experts, self.hidden_size,
               moe_intermediate_size_after_tp,
            )
            w2_ep_permuted.copy_(w2_ep.permute(2, 0, 1, 3))
            w2_tp = self.w2_ep_gathered
            w2_handle = dist.all_to_all_single(
               output=w2_tp,
               input=w2_ep_permuted.view(self.w2_ep_gathered.shape),
               group=paras_tp_group,
               async_op=True,
            )

            # -- w13 post-processing: reinterpret as TP shape in EP buffer --
            w13_handle.wait()
            if paras_dp_size > 1:
                w13_post = mgr.get_view("staging.w13_b").view(
                    paras_dp_size, paras_tp_size, -1
                )
                w13_post.copy_(
                    w13_tp.view(paras_tp_size, paras_dp_size, -1).transpose(0, 1)
                )
                tp_w13 = mgr.get_view_as(
                    f"model.layers.{self._paras_layer_id}.mlp.experts.w13_weight",
                    (self.num_global_experts, 2 * moe_intermediate_size_after_tp, self.hidden_size),
                )
                tp_w13.copy_(w13_post.view_as(tp_w13))
            else:
                tp_w13 = mgr.get_view_as(
                    f"model.layers.{self._paras_layer_id}.mlp.experts.w13_weight",
                    (self.num_global_experts, 2 * moe_intermediate_size_after_tp, self.hidden_size),
                )

            # -- w2 post-processing --
            w2_handle.wait()
            if paras_dp_size > 1:
                w2_post = mgr.get_view("staging.w2_b").view(
                    paras_dp_size, paras_tp_size, -1
                )
                w2_post.copy_(
                    w2_tp.view(paras_tp_size, paras_dp_size, -1).transpose(0, 1)
                )
                tp_w2 = mgr.get_view_as(
                    f"model.layers.{self._paras_layer_id}.mlp.experts.w2_weight",
                    (self.num_global_experts, self.hidden_size, moe_intermediate_size_after_tp),
                )
                tp_w2.copy_(w2_post.view_as(tp_w2))
            else:
                tp_w2 = mgr.get_view_as(
                    f"model.layers.{self._paras_layer_id}.mlp.experts.w2_weight",
                    (self.num_global_experts, self.hidden_size, moe_intermediate_size_after_tp),
                )

    def paras_configure_tp_peer_access(
        self,
        peer_ctx,           # PeerAccessContext from peer_access.py
        transfer_plans,     # dict[(layer_id, weight_type)] -> List[TransferEntry]
        packed_plans,       # dict[(layer_id, weight_type)] -> dict (packed GPU tensors)
        staging_suffix: str = "a",   # "a" for stream_1, "b" for stream_2 (double-buffering)
        stream=None,
    ):
        """
        EP→TP weight redistribution via NVLink peer access (DP=1 only).

        Phase 1: Permute local EP weights to staging buffer (same as NCCL path)
        Phase 2: barrier() to ensure all ranks finished reading their EP data
        Phase 3: Peer write from staging to remote EP/TP buffers
        Phase 4: Post-processing (reinterpret buffer as TP shape)
        """
        from sglang.srt.paras.peer_access import peer_access_transfer
        from sglang.srt.paras.transfer_plan import pack_transfer_plan

        mgr = get_global_paras_memory_manager()
        paras_tp_size = get_paras_tp_size()
        paras_dp_size = get_paras_dp_size()
        assert paras_dp_size == 1, "peer_access method only supports DP=1"

        moe_intermediate_size_after_tp = self.moe_intermediate_size // paras_tp_size
        paras_tp_group = get_paras_tp_group().device_group
        layer_id = self._paras_layer_id

        with torch.cuda.stream(stream):
            # Phase 1: Permute w13 EP weights → staging
            w13_ep = self.w13_ep_gathered.view(
                self.num_local_experts,
                2,
                paras_tp_size,
                moe_intermediate_size_after_tp * self.hidden_size,
            )
            w13_staging_name = f"staging.w13_{staging_suffix}"
            w13_ep_permuted = mgr.get_view(w13_staging_name).view(
                paras_tp_size, self.num_local_experts, 2,
                moe_intermediate_size_after_tp * self.hidden_size,
            )
            w13_ep_permuted.copy_(w13_ep.permute(2, 0, 1, 3))

            # Permute w2 EP weights → staging
            w2_ep = self.w2_ep_gathered.view(
                self.num_local_experts,
                self.hidden_size,
                paras_tp_size,
                moe_intermediate_size_after_tp,
            )
            w2_staging_name = f"staging.w2_{staging_suffix}"
            w2_ep_permuted = mgr.get_view(w2_staging_name).view(
                paras_tp_size, self.num_local_experts, self.hidden_size,
                moe_intermediate_size_after_tp,
            )
            w2_ep_permuted.copy_(w2_ep.permute(2, 0, 1, 3))

        # Phase 2: Barrier — ensure all ranks finished reading EP data and writing to staging
        if stream is not None:
            stream.synchronize()
        torch.distributed.barrier(group=paras_tp_group)

        # Phase 3: Peer write from local staging to all peer EP buffers
        dst_base_ptrs = torch.tensor(
            peer_ctx.peer_addresses, dtype=torch.int64, device="cuda"
        )
        local_buffer_ptr = mgr._buffer.data_ptr()

        with torch.cuda.stream(stream):
            # Transfer w13
            w13_plan_key = (layer_id, "w13")
            w13_packed = packed_plans.get(w13_plan_key)
            if w13_packed is None:
                from sglang.srt.paras.transfer_plan import build_transfer_plan
                entries = build_transfer_plan(mgr, layer_id, "w13", paras_tp_size, get_paras_tp_rank())
                w13_packed = pack_transfer_plan(entries)
            peer_access_transfer(local_buffer_ptr, dst_base_ptrs, w13_packed, stream)

            # Transfer w2
            w2_plan_key = (layer_id, "w2")
            w2_packed = packed_plans.get(w2_plan_key)
            if w2_packed is None:
                from sglang.srt.paras.transfer_plan import build_transfer_plan
                entries = build_transfer_plan(mgr, layer_id, "w2", paras_tp_size, get_paras_tp_rank())
                w2_packed = pack_transfer_plan(entries)
            peer_access_transfer(local_buffer_ptr, dst_base_ptrs, w2_packed, stream)

        # Phase 4: Wait for peer writes, then barrier to ensure remote GPUs received data
        if stream is not None:
            stream.synchronize()
        torch.distributed.barrier(group=paras_tp_group)

        # Phase 5: Post-processing — reinterpret EP buffer as TP shape
        mgr.get_view_as(
            f"model.layers.{layer_id}.mlp.experts.w13_weight",
            (self.num_global_experts, 2 * moe_intermediate_size_after_tp, self.hidden_size),
        )
        mgr.get_view_as(
            f"model.layers.{layer_id}.mlp.experts.w2_weight",
            (self.num_global_experts, self.hidden_size, moe_intermediate_size_after_tp),
        )

    def paras_configure_tp_combined_peer_access_kernel(
        self,
        peer_ctx,
        dst_base_ptrs: torch.Tensor,
        stream=None,
    ):
        """Launch combined w13+w2 kernel for this layer (192 blocks, single launch)."""
        from sglang.srt.paras.peer_access import peer_access_fused_transfer_combined_single_layer

        mgr = get_global_paras_memory_manager()
        paras_tp_size = get_paras_tp_size()
        paras_tp_rank = get_paras_tp_rank()
        layer_id = self._paras_layer_id
        moe_intermediate_size_after_tp = self.moe_intermediate_size // paras_tp_size

        ep_w13_entry = mgr._entries[f"model.layers.{layer_id}.mlp.experts.w13_weight"]
        ep_w2_entry = mgr._entries[f"model.layers.{layer_id}.mlp.experts.w2_weight"]

        if layer_id == 0:
            tp_w13_entry = mgr._entries["paras.fused_tp_slot0.w13"]
            tp_w2_entry = mgr._entries["paras.fused_tp_slot0.w2"]
        else:
            tp_w13_entry = mgr._entries[f"model.layers.{layer_id - 1}.mlp.experts.w13_weight"]
            tp_w2_entry = mgr._entries[f"model.layers.{layer_id - 1}.mlp.experts.w2_weight"]

        local_buffer_ptr = mgr._buffer.data_ptr()
        elem_size = 2

        peer_access_fused_transfer_combined_single_layer(
            local_buffer_ptr, dst_base_ptrs,
            ep_w13_entry.offset_bytes, tp_w13_entry.offset_bytes,
            ep_w2_entry.offset_bytes, tp_w2_entry.offset_bytes,
            paras_tp_rank, paras_tp_size, self.num_local_experts,
            moe_intermediate_size_after_tp * self.hidden_size,
            num_gates=2, elem_size=elem_size,
            H=self.hidden_size,
            I_full=self.moe_intermediate_size,
            I_prime=moe_intermediate_size_after_tp,
            stream=stream,
        )

    def paras_configure_tp_fused_peer_access_kernel(
        self,
        peer_ctx,
        dst_base_ptrs: torch.Tensor,
        stream=None,
    ):
        """Launch fused peer access kernels for this layer. NO barriers — caller manages them.

        The N+1 slot design guarantees no inter-layer aliasing:
          - Layer i reads local slot[i+1], writes to peer slot[i]
          - Layer i+1 reads local slot[i+2], writes to peer slot[i+1]
          - Different slots → no race → barriers only needed at sweep start/end.
        """
        from sglang.srt.paras.peer_access import peer_access_fused_transfer, peer_access_fused_transfer_w2

        mgr = get_global_paras_memory_manager()
        paras_tp_size = get_paras_tp_size()
        paras_tp_rank = get_paras_tp_rank()
        layer_id = self._paras_layer_id
        moe_intermediate_size_after_tp = self.moe_intermediate_size // paras_tp_size

        ep_w13_entry = mgr._entries[f"model.layers.{layer_id}.mlp.experts.w13_weight"]
        ep_w2_entry = mgr._entries[f"model.layers.{layer_id}.mlp.experts.w2_weight"]

        if layer_id == 0:
            tp_w13_entry = mgr._entries["paras.fused_tp_slot0.w13"]
            tp_w2_entry = mgr._entries["paras.fused_tp_slot0.w2"]
        else:
            tp_w13_entry = mgr._entries[f"model.layers.{layer_id - 1}.mlp.experts.w13_weight"]
            tp_w2_entry = mgr._entries[f"model.layers.{layer_id - 1}.mlp.experts.w2_weight"]

        local_buffer_ptr = mgr._buffer.data_ptr()
        I_prime_H = moe_intermediate_size_after_tp * self.hidden_size
        E_local = self.num_local_experts
        elem_size = 2  # bf16

        peer_access_fused_transfer(
            local_buffer_ptr, dst_base_ptrs,
            ep_w13_entry.offset_bytes, tp_w13_entry.offset_bytes,
            paras_tp_rank, paras_tp_size, E_local, I_prime_H,
            num_gates=2, elem_size=elem_size, stream=stream,
        )
        peer_access_fused_transfer_w2(
            local_buffer_ptr, dst_base_ptrs,
            ep_w2_entry.offset_bytes, tp_w2_entry.offset_bytes,
            paras_tp_rank, paras_tp_size, E_local,
            H=self.hidden_size,
            I_full=self.moe_intermediate_size,
            I_prime=moe_intermediate_size_after_tp,
            elem_size=elem_size, stream=stream,
        )

    def paras_configure_tp_fused_peer_access(self, peer_ctx, stream=None):
        """Convenience wrapper: kernel + barriers + view update for single-layer use (e.g. tests)."""
        paras_tp_group = get_paras_tp_group().device_group
        dst_base_ptrs = torch.tensor(peer_ctx.peer_addresses, dtype=torch.int64, device="cuda")
        torch.distributed.barrier(group=paras_tp_group)
        self.paras_configure_tp_fused_peer_access_kernel(peer_ctx, dst_base_ptrs, stream)
        torch.cuda.synchronize()
        torch.distributed.barrier(group=paras_tp_group)
        self.paras_configure_tp_fused_peer_access_update_views()

    def paras_configure_tp_fused_peer_access_update_views(self):
        """Update TP expert weight views to point to the TP slot after fused transfer."""
        mgr = get_global_paras_memory_manager()
        paras_tp_size = get_paras_tp_size()
        layer_id = self._paras_layer_id
        moe_intermediate_size_after_tp = self.moe_intermediate_size // paras_tp_size
        tp_inter = moe_intermediate_size_after_tp

        if layer_id == 0:
            tp_w13_name = "paras.fused_tp_slot0.w13"
            tp_w2_name = "paras.fused_tp_slot0.w2"
        else:
            tp_w13_name = f"model.layers.{layer_id - 1}.mlp.experts.w13_weight"
            tp_w2_name = f"model.layers.{layer_id - 1}.mlp.experts.w2_weight"

        tp_w13_view = mgr.get_view_as(tp_w13_name, (self.num_global_experts, 2 * tp_inter, self.hidden_size))
        tp_w2_view = mgr.get_view_as(tp_w2_name, (self.num_global_experts, self.hidden_size, tp_inter))
        self.tp_experts.w13_weight = torch.nn.Parameter(tp_w13_view, requires_grad=False)
        self.tp_experts.w2_weight = torch.nn.Parameter(tp_w2_view, requires_grad=False)

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
