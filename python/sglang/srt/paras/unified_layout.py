"""Address-only planning for overlapping weights, KV, and MoE workspace.

This module deliberately has no torch/CUDA imports so the transfer geometry
can be exhaustively checked before allocating GPU memory.
"""

import math
from dataclasses import dataclass

from sglang.srt.paras.mode import ParaSMode


def align_up(n: int, alignment: int = 256) -> int:
    return (n + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class CacheCapacity:
    full_tokens: int
    layer_tokens: tuple[int, ...]
    layer_bytes: tuple[int, ...]

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
    budget = budget // 256 * 256
    front = align_up(
        max(wt, ep_workspace_bytes, tp_workspace_bytes - num_layers * (we - wt))
    )

    ratios = layer_token_ratios or (1.0,) * num_layers
    assert len(ratios) == num_layers and min(ratios) > 0

    def capacity_for_tokens(full_tokens, row_bytes):
        tokens = tuple(
            int(full_tokens * ratio) // page_size * page_size for ratio in ratios
        )
        sizes = tuple(2 * align_up((n + page_size) * row_bytes // 2) for n in tokens)
        return CacheCapacity(full_tokens, tokens, sizes)

    minimum_tokens = math.ceil(page_size / min(ratios) / page_size) * page_size

    def cache_capacity(available, row_bytes):
        # Preserve the configured full/SWA ratio, rounding each pool to pages.
        lo = minimum_tokens // page_size
        hi = max(lo, int(available / row_bytes / sum(ratios)) // page_size)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if capacity_for_tokens(mid * page_size, row_bytes).total_bytes <= available:
                lo = mid
            else:
                hi = mid - 1
        capacity = capacity_for_tokens(lo * page_size, row_bytes)
        if capacity.total_bytes > available:
            raise ValueError(
                "Unified memory budget cannot hold weights, workspace, and KV"
            )
        return capacity

    def capacities(front):
        ep = cache_capacity(budget - front - num_layers * we, ep_kv_row_bytes)
        tail = align_up(max(max(ep.layer_bytes), tp_workspace_bytes))
        tp = cache_capacity(budget - tail - num_layers * wt, tp_kv_row_bytes)
        return ep, tail, tp

    def migration_fits(ep, tp):
        return all(t >= e for e, t in zip(ep.layer_bytes, tp.layer_bytes))

    # Ordered cache migration requires ct >= ce. A large TP workspace can
    # consume more than the weight saving; reserve the difference at EP's
    # front. For small layer counts the cache transfer gap can dominate too.
    # Find the smallest aligned front satisfying both constraints.
    lo = front // 256
    hi = (
        budget
        - num_layers * we
        - capacity_for_tokens(minimum_tokens, ep_kv_row_bytes).total_bytes
    ) // 256
    if hi < lo:
        raise ValueError("Unified memory budget cannot hold weights, workspace, and KV")
    while lo < hi:
        mid = (lo + hi) // 2
        ep, _, tp = capacities(mid * 256)
        if migration_fits(ep, tp):
            hi = mid
        else:
            lo = mid + 1
    front = lo * 256
    ep, tail, tp = capacities(front)
    if not migration_fits(ep, tp):
        raise ValueError("Unified memory budget cannot support ordered cache migration")
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
