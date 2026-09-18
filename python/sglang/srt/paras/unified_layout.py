"""Address-only planning for overlapping weights, KV, and MoE workspace.

This module deliberately has no torch/CUDA imports so the transfer geometry
can be exhaustively checked before allocating GPU memory.
"""

import math
from dataclasses import dataclass

from sglang.srt.paras.mode import ParaSMode

MEMORY_ALIGNMENT = 256


def align_up(n: int, alignment: int = MEMORY_ALIGNMENT) -> int:
    return (n + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class CacheCapacity:
    """Token capacities and reserved K+V slot bytes for each layer."""

    full_tokens: int
    layer_tokens: tuple[int, ...]
    layer_bytes: tuple[int, ...]

    @classmethod
    def from_budget(cls, budget, row_bytes, ratios, page_size):
        """Divide bytes between layers, then fit page-aligned tokens inside them.

        Each layer gets equally sized, 256-byte-aligned K and V slots. Keeping
        slot placement separate from token rounding makes migration geometry
        independent of the EP/TP head counts and page sizes.
        """
        ratio_sum = sum(ratios)
        kv_pair_alignment = 2 * MEMORY_ALIGNMENT  # One slot for K, one for V.
        layer_bytes = tuple(
            int(budget * ratio / ratio_sum) // kv_pair_alignment * kv_pair_alignment
            for ratio in ratios
        )
        full_token_limits = (
            (size // row_bytes - page_size) / ratio
            for size, ratio in zip(layer_bytes, ratios)
        )
        full_tokens = int(min(full_token_limits)) // page_size * page_size
        layer_tokens = tuple(
            int(full_tokens * r) // page_size * page_size for r in ratios
        )
        if min(layer_tokens) < page_size:
            raise ValueError(
                "Unified memory budget cannot hold weights, workspace, and KV"
            )
        return cls(full_tokens, layer_tokens, layer_bytes)

    @property
    def total_bytes(self) -> int:
        return sum(self.layer_bytes)


@dataclass(frozen=True)
class UnifiedLayout:
    num_layers: int
    budget: int
    ep_weight_bytes: int
    tp_weight_bytes: int
    ep_front: int
    tp_tail: int
    ep_cache: CacheCapacity
    tp_cache: CacheCapacity

    def weight_offset(self, mode: ParaSMode, layer: int) -> int:
        if mode == ParaSMode.EP:
            return self.ep_front + layer * self.ep_weight_bytes
        if mode == ParaSMode.TP:
            return layer * self.tp_weight_bytes
        raise ValueError(mode)

    def cache_offset(self, mode: ParaSMode, layer: int) -> int:
        if mode == ParaSMode.EP:
            return self.budget - sum(self.ep_cache.layer_bytes[layer:])
        if mode == ParaSMode.TP:
            return self.num_layers * self.tp_weight_bytes + sum(
                self.tp_cache.layer_bytes[:layer]
            )
        raise ValueError(mode)

    def workspace(self, mode: ParaSMode) -> tuple[int, int]:
        if mode == ParaSMode.EP:
            return 0, self.ep_front
        if mode == ParaSMode.TP:
            return self.budget - self.tp_tail, self.tp_tail
        raise ValueError(mode)


def plan_unified_layout(
    *,
    num_layers: int,
    budget: int,
    ep_weight_bytes: int,
    tp_weight_bytes: int,
    ep_workspace_bytes: int,
    tp_workspace_bytes: int,
    ep_kv_row_bytes: int,
    tp_kv_row_bytes: int,
    page_size: int = 1,
    layer_token_ratios: tuple[float, ...] | None = None,
) -> UnifiedLayout:
    if min(num_layers, ep_kv_row_bytes, tp_kv_row_bytes, page_size) <= 0:
        raise ValueError("Layer count, KV row sizes, and page size must be positive")
    if ep_weight_bytes < tp_weight_bytes:
        raise ValueError("EP-low-weight topology needs a different layout orientation")
    we, wt = align_up(ep_weight_bytes), align_up(tp_weight_bytes)
    budget = budget // MEMORY_ALIGNMENT * MEMORY_ALIGNMENT
    ratios = layer_token_ratios or (1.0,) * num_layers
    assert len(ratios) == num_layers and min(ratios) > 0
    weight_saving = num_layers * (we - wt)
    tp_available = budget - num_layers * wt

    # Let K be EP's total cache budget and r the largest layer's share.
    # TP needs room for K plus its transfer gap r*K, so K <= tp_available/(1+r).
    # EP's front also covers one weight layer, its scratch, and any TP scratch
    # that exceeds the space freed by sharding attention weights.
    largest_layer_share = max(ratios) / sum(ratios)
    cache_transfer_gap = math.ceil(
        tp_available * largest_layer_share / (1 + largest_layer_share)
    )
    front = align_up(
        max(
            wt,
            ep_workspace_bytes,
            tp_workspace_bytes - weight_saving,
            cache_transfer_gap - weight_saving,
        )
    )
    ep = CacheCapacity.from_budget(
        budget - num_layers * we - front, ep_kv_row_bytes, ratios, page_size
    )
    tail = align_up(max(max(ep.layer_bytes), tp_workspace_bytes))
    tp = CacheCapacity.from_budget(
        tp_available - tail, tp_kv_row_bytes, ratios, page_size
    )
    return UnifiedLayout(num_layers, budget, we, wt, front, tail, ep, tp)


# Existing Triton execution limit, shared by workspace planning and runners.
# TP counts input tokens. EP uses this as its reserved row capacity only.
triton_moe_chunk_size: int = 64 * 1024
# The Triton configurations supported by this workspace have BLOCK_SIZE_M <= 256.
MOE_MAX_BLOCK_M = 256


def validate_triton_workspace_blocks(*block_sizes):
    """Reject tuning configurations that exceed the planned padding bound."""
    for size in block_sizes:
        if size is not None and not 0 < size <= MOE_MAX_BLOCK_M:
            raise ValueError(
                f"ParaS Triton workspace requires BLOCK_SIZE_M <= {MOE_MAX_BLOCK_M}, "
                f"but the selected configuration uses {size}"
            )


def bf16_moe_workspace_sizes(
    *,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    tp_size: int,
    dispatch_capacity: int,
) -> tuple[int, int]:
    """BF16 internal scratch; dispatcher inputs and escaped outputs are separate.

    EP reserves space for dispatched rows and the padded LL receive shape;
    larger batches retain the original dynamic allocation. TP follows the
    fused runner's existing input-token chunks.
    """
    ep_rows = max(triton_moe_chunk_size, num_experts * dispatch_capacity)
    ep = align_up(ep_rows * 2 * intermediate_size * 2)
    ep += align_up(ep_rows * intermediate_size * 2)
    tp_inter = intermediate_size // tp_size
    max_input_tokens_per_chunk = triton_moe_chunk_size
    tp_rows = max_input_tokens_per_chunk * top_k + (num_experts + 1) * (
        MOE_MAX_BLOCK_M - 1
    )
    tp = align_up(tp_rows * max(2 * tp_inter, hidden_size) * 2)
    tp += align_up(tp_rows * tp_inter * 2)
    return ep, tp
