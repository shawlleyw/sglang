"""Address-only planning for overlapping weights, KV, and MoE workspace.

This module deliberately has no torch/CUDA imports so the transfer geometry
can be exhaustively checked before allocating GPU memory.
"""

from dataclasses import dataclass


def align_up(n: int, alignment: int = 256) -> int:
    return (n + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class UnifiedLayout:
    num_layers: int
    budget: int
    ep_weight_bytes: int
    tp_weight_bytes: int
    ep_front: int
    tp_tail: int
    ep_cache_bytes: int
    tp_cache_bytes: int
    ep_tokens: int
    tp_tokens: int

    def weight_offset(self, mode: str, layer: int) -> int:
        if mode == "ep":
            return self.ep_front + layer * self.ep_weight_bytes
        if mode == "tp":
            return layer * self.tp_weight_bytes
        raise ValueError(mode)

    def cache_offset(self, mode: str, layer: int) -> int:
        if mode == "ep":
            return self.budget - (self.num_layers - layer) * self.ep_cache_bytes
        if mode == "tp":
            return self.num_layers * self.tp_weight_bytes + layer * self.tp_cache_bytes
        raise ValueError(mode)

    def workspace(self, mode: str) -> tuple[int, int]:
        if mode == "ep":
            return 0, self.ep_front
        if mode == "tp":
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

    def cache_capacity(available, row_bytes):
        tokens = available // num_layers // row_bytes - page_size
        tokens = tokens // page_size * page_size
        if tokens < page_size:
            raise ValueError(
                "Unified memory budget cannot hold weights, workspace, and KV"
            )
        # K and V are individually aligned by all supported head dimensions.
        size = (tokens + page_size) * row_bytes
        if (size // 2) % 256:
            raise ValueError("K/V layer buffers must be individually 256-byte aligned")
        return tokens, size

    def capacities(front):
        ep_tokens, ce = cache_capacity(
            budget - front - num_layers * we, ep_kv_row_bytes
        )
        tail = align_up(max(ce, tp_workspace_bytes))
        tp_tokens, ct = cache_capacity(budget - tail - num_layers * wt, tp_kv_row_bytes)
        return ep_tokens, ce, tail, tp_tokens, ct

    # Ordered cache migration requires ct >= ce. A large TP workspace can
    # consume more than the weight saving; reserve the difference at EP's
    # front. For small layer counts the cache transfer gap can dominate too.
    # Find the smallest aligned front satisfying both constraints.
    lo = front // 256
    hi = (
        budget - num_layers * we - num_layers * 2 * page_size * ep_kv_row_bytes
    ) // 256
    if hi < lo:
        raise ValueError("Unified memory budget cannot hold weights, workspace, and KV")
    while lo < hi:
        mid = (lo + hi) // 2
        _, ce, _, _, ct = capacities(mid * 256)
        if ct >= ce:
            hi = mid
        else:
            lo = mid + 1
    front = lo * 256
    ep_tokens, ce, tail, tp_tokens, ct = capacities(front)
    if ct < ce:
        raise ValueError("Unified memory budget cannot support ordered cache migration")
    return UnifiedLayout(
        num_layers, budget, we, wt, front, tail, ce, ct, ep_tokens, tp_tokens
    )


MOE_CHUNK_ROWS = 65536
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


def triton_workspace_sizes(
    *,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    tp_size: int,
    dispatch_capacity: int,
) -> tuple[int, int]:
    """BF16 internal scratch; dispatcher inputs and escaped outputs are separate.

    EP contiguous work is tiled into MOE_CHUNK_ROWS expert rows. LL uses its
    full padded receive shape. TP follows the fused runner's input-token chunks.
    """
    ep_rows = max(MOE_CHUNK_ROWS, num_experts * dispatch_capacity)
    ep = align_up(ep_rows * 2 * intermediate_size * 2)
    ep += align_up(ep_rows * intermediate_size * 2)
    tp_inter = intermediate_size // tp_size
    tp_rows = MOE_CHUNK_ROWS * top_k + (num_experts + 1) * (MOE_MAX_BLOCK_M - 1)
    tp = align_up(tp_rows * max(2 * tp_inter, hidden_size) * 2)
    tp += align_up(tp_rows * tp_inter * 2)
    return ep, tp
