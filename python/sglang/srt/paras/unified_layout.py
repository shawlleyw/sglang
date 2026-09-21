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
    """Token capacities and reserved K+V bytes for each layer."""

    full_tokens: int
    layer_tokens: tuple[int, ...]
    layer_bytes: tuple[int, ...]

    @classmethod
    def from_budget(cls, budget, row_bytes, ratios, page_size):
        """Divide bytes between layers, then fit page-aligned tokens inside them.

        Each layer gets equally sized, aligned K and V regions. Keeping
        byte placement separate from token rounding makes migration geometry
        independent of the EP/TP head counts and page sizes.
        """
        ratio_sum = sum(ratios)
        kv_pair_alignment = 2 * MEMORY_ALIGNMENT  # One region for K, one for V.
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
    """Plan two interpretations of the same byte budget (before page rounding).

    EP: [front: workspace + transfer headroom][EP weights][EP KV]
    TP: [TP weights][TP KV][tail: workspace + transfer headroom]

    The front must hold one TP weight layer. The tail must hold the largest
    EP KV layer. TP KV regions must be at least as large as EP's, so that
    forward/reverse layer transfers cannot overwrite untransferred data.
    """
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

    # EP_KV = tp_available - weight_saving - front.
    # TP_KV = tp_available - tail. Requiring TP_KV >= EP_KV gives
    # front >= tail - weight_saving. Apply this to both possible tail sizes:
    front_for_tp_workspace = tp_workspace_bytes - weight_saving

    # For a largest EP layer with share r, tail >= r * EP_KV. Substitution
    # gives front >= tp_available * r / (1+r) - weight_saving.
    largest_layer_share = max(ratios) / sum(ratios)
    front_for_kv_transfer = (
        math.ceil(tp_available * largest_layer_share / (1 + largest_layer_share))
        - weight_saving
    )
    front = align_up(
        max(
            wt,
            ep_workspace_bytes,
            front_for_tp_workspace,
            front_for_kv_transfer,
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
# This is a TP input-token limit; EP runners do not chunk their dispatched rows.
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


def tp_moe_workspace_token_capacity(
    *,
    max_prefill_tokens: int | None,
    max_running_requests: int | None,
    chunked_prefill_size: int | None = None,
    cuda_graph_bs: tuple[int, ...] | list[int] = (),
    speculative_num_draft_tokens: int | None = None,
) -> int:
    """Reserve for configured batches, not the kernel's largest possible chunk.

    This is a reservation target, not an admission limit: unchunked prefill can
    admit a first request larger than max_prefill_tokens. Such batches retain
    the runner's existing dynamic-allocation fallback. Automatic request pools
    are capped at 4096, matching ModelRunner's default capacity heuristic.
    """
    prefill = max_prefill_tokens or triton_moe_chunk_size
    if chunked_prefill_size is not None and chunked_prefill_size > 0:
        prefill = min(prefill, chunked_prefill_size)
    tokens_per_request = max(1, speculative_num_draft_tokens or 1)
    decode = (max_running_requests or 4096) * tokens_per_request
    graph = max(cuda_graph_bs, default=0) * tokens_per_request
    return min(triton_moe_chunk_size, max(prefill, decode, graph))


def bf16_moe_workspace_sizes(
    *,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    tp_size: int,
    dispatch_capacity: int,
    tp_input_tokens: int | None = None,
) -> tuple[int, int]:
    """BF16 internal scratch; dispatcher inputs and escaped outputs are separate.

    EP follows DeepEP's low-latency receive shape: local_experts *
    (ep_size * dispatch_capacity). Both EP runners consume that same shape;
    normal dispatch uses its actual received rows and falls back to native
    allocation when larger. TP reserves its configured token target (up to the
    kernel chunk limit) and likewise falls back for larger runtime batches.
    """
    ep_rows = num_experts * dispatch_capacity
    ep = align_up(ep_rows * 2 * intermediate_size * 2)
    ep += align_up(ep_rows * intermediate_size * 2)
    tp_inter = intermediate_size // tp_size
    if tp_input_tokens is not None and tp_input_tokens <= 0:
        raise ValueError("TP workspace token capacity must be positive")
    max_input_tokens_per_chunk = min(
        tp_input_tokens or triton_moe_chunk_size, triton_moe_chunk_size
    )
    tp_rows = max_input_tokens_per_chunk * top_k + (num_experts + 1) * (
        MOE_MAX_BLOCK_M - 1
    )
    tp = align_up(tp_rows * max(2 * tp_inter, hidden_size) * 2)
    tp += align_up(tp_rows * tp_inter * 2)
    return ep, tp
