"""
ParaSMemoryManager: Static contiguous weight/KV buffer for ParaS EP↔TP switching.
Scope: supported ParaS sparse-MoE layouts.
Supported dtypes: BF16/FP16 (unquantized) and FP8.

LIFECYCLE & DESIGN:
  The manager pre-plans one contiguous uint8 buffer containing the persistent
  weights and KV views needed for Expert Parallelism (EP) ↔ Tensor Parallelism
  (TP) switching. This avoids runtime allocation and fragmentation.

  1. plan_*_moe_layout(manager, ...) — records weight and staging metadata
  2. manager.plan_*_kv_capacity(...) — selects EP/TP cache capacities
  3. manager.reserve_kv_cache(...)   — records the selected cache views
  4. manager.materialize()           — assigns offsets and allocates one GPU buffer
  5. manager.get_view("name")        — returns a typed view for model parameters

MEMORY LAYOUT:
  Ordinary weights and staging occupy an aligned prefix. Expert weights and KV
  cache use a combined four-anchor run where inactive EP and TP storage overlaps.
  Every entry is 256-byte aligned.

WHY THIS APPROACH:
  - Single allocation: Avoids GPU memory fragmentation and repeated malloc/free overhead.
  - Deterministic offsets: Reservation order → fixed offsets enables reproducible layouts.
  - Dtype-agnostic storage: uint8 buffer holds any dtype; views reinterpret as needed.
  - Aligned access: 256-byte alignment matches GPU memory coalescing patterns.

HETEROGENEOUS LAYERS (layer_specs):
  When layer_specs differ per layer, each slot is sized for its layer's family
  (full vs SWA). Two families may have different per-layer bytes, but within
  a family all slots have the same bytes per layer.
"""

import json
import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from sglang.srt.paras.weight_transfer import (
    IntraNodeWeightTransferMethod,
    resolve_intra_node_weight_transfer_method,
)

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from sglang.srt.paras.layers.utils import LayerCacheSpec
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LayoutEntry
# ---------------------------------------------------------------------------

@dataclass
class LayoutEntry:
    """Metadata for one reserved tensor inside the contiguous buffer."""

    name: str
    shape: Tuple[int, ...]
    dtype: torch.dtype
    numel: int
    element_size: int
    size_bytes: int
    offset_bytes: int = 0

    def to_dict(self) -> Dict:
        """Return a JSON-serializable dictionary."""
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": str(self.dtype),
            "numel": self.numel,
            "element_size": self.element_size,
            "size_bytes": self.size_bytes,
            "offset_bytes": self.offset_bytes,
        }


@dataclass(frozen=True)
class ParaSKVCapacityPlan:
    """Exact UMM-owned EP/TP weight and KV capacity plan.

    ``manager_budget_bytes`` is the hard UMM limit after non-UMM static memory.
    ``planned_umm_bytes`` is the exact aligned footprint selected below that
    limit. The EP/TP weight and KV fields expose how each mode consumes the
    shared allocation. SWA fields are zero / empty for pure-MHA plans;
    ``layer_specs`` is set only by the SWA planner for downstream
    :meth:`reserve_kv_cache`.
    """

    available_gpu_memory_bytes: int
    total_gpu_memory_bytes: int
    dynamic_reserve_bytes: int
    umm_budget_bytes: int
    non_umm_static_bytes: int
    manager_budget_bytes: int

    fixed_umm_bytes: int
    ep_expert_weight_bytes: int
    tp_expert_weight_bytes: int
    ep_kv_bytes: int
    tp_kv_bytes: int
    planned_umm_bytes: int

    kv_dtype: torch.dtype

    ep_max_tokens: int
    tp_max_tokens: int
    ep_cell_bytes: int
    tp_cell_bytes: int
    ep_kv_heads: int
    tp_kv_heads: int

    full_layers: int = 0
    swa_layers: int = 0
    ep_max_tokens_swa: int = 0
    tp_max_tokens_swa: int = 0

    layer_specs: Optional[List["LayerCacheSpec"]] = None


@dataclass(frozen=True)
class _ParaSRunGeometry:
    """Aligned byte geometry for the combined expert-weight/KV run."""

    we: Tuple[int, ...]
    wt: Tuple[int, ...]
    ce: Tuple[int, ...]
    ct: Tuple[int, ...]
    sum_we: int
    sum_wt: int
    sum_ce: int
    sum_ct: int
    anchor: int
    pad: int
    run_bytes: int


# ---------------------------------------------------------------------------
# Supported dtypes
# ---------------------------------------------------------------------------

_SUPPORTED_DTYPES = {
    torch.bfloat16,
    torch.float16,
    torch.float32,
    torch.float8_e4m3fn,
}


# ---------------------------------------------------------------------------
# V1 scope validation
# ---------------------------------------------------------------------------

def _validate_v1_scope(
    num_fused_shared_experts: int,
    quant_name: Optional[str],
) -> None:
    """Raise if the model config falls outside the V1 ParaS scope."""
    if num_fused_shared_experts > 0:
        raise ValueError(
            "ParaS V1 does not support shared experts "
            f"(num_fused_shared_experts={num_fused_shared_experts}). "
            "Only pure sparse-MoE layers are supported."
        )
    if quant_name not in (None, "", "fp8"):
        raise ValueError(
            f"ParaS V1 does not support quantization format '{quant_name}'. "
            "Only unquantized (BF16/FP16) and FP8 are supported."
        )


def _validate_paras_swa_runtime_scope(server_args, model_config) -> None:
    """Raise if ParaS + SWA + incompatible runtime features are detected.

    Checks for unsupported combinations:
    - G12: FP8-KV + SWA
    - G14: Speculative decoding + SWA
    """
    swa_attention_layer_ids = getattr(model_config, "swa_attention_layer_ids", None)
    if not swa_attention_layer_ids:
        return

    kv_cache_dtype = server_args.kv_cache_dtype
    if kv_cache_dtype == "fp8":
        raise NotImplementedError(
            "ParaS + SWA + FP8-KV not supported in v1 "
            "(see docs/paras/swa_support.md §12). "
            "Disable FP8-KV or use a non-hybrid model."
        )

    speculative_algorithm = server_args.speculative_algorithm
    if speculative_algorithm is not None:
        raise NotImplementedError(
            "ParaS + SWA + speculative decoding not supported in v1."
        )


# ---------------------------------------------------------------------------
# Hybrid KV budget planner
# ---------------------------------------------------------------------------

def plan_hybrid_kv_budget(
    total_tokens: int,
    full_layers_num: int,
    swa_layers_num: int,
    swa_full_tokens_ratio: float,
) -> Tuple[int, int]:
    """Compute per-layer token budgets for a hybrid full/SWA attention model.

    Mirrors the generic branch of ``set_num_token_hybrid`` in
    ``model_runner.py`` (lines 1497-1516).  Pure arithmetic — no tensor
    allocation.

    The two unknowns satisfy:
        swa_max * swa_layers + full_max * full_layers == total_tokens
        swa_max == full_max * swa_full_tokens_ratio

    Returns:
        (full_max_total_num_tokens, swa_max_total_num_tokens)
    """
    if full_layers_num == 0 and swa_layers_num == 0:
        raise ValueError("no layers")
    if swa_layers_num > 0 and swa_full_tokens_ratio <= 0:
        raise ValueError(
            "swa_full_tokens_ratio must be > 0 when SWA layers present"
        )

    # All-MHA shortcut: no SWA layers at all.
    if swa_layers_num == 0:
        return (int(total_tokens / full_layers_num), 0)

    denominator = swa_full_tokens_ratio * swa_layers_num + full_layers_num
    full_max = int(total_tokens / denominator)
    swa_max = int(full_max * swa_full_tokens_ratio)

    if swa_max < 1:
        logging.warning(
            "plan_hybrid_kv_budget: computed swa_max_total_num_tokens < 1 "
            "(ratio=%.4f, full_max=%d). SWA layers will have near-zero budget.",
            swa_full_tokens_ratio,
            full_max,
        )

    return (full_max, swa_max)


# ---------------------------------------------------------------------------
# ParaSMemoryManager
# ---------------------------------------------------------------------------

class ParaSMemoryManager:
    """
    Pre-plans and materialises a single contiguous ``uint8`` buffer that
    holds every weight tensor needed for ParaS EP↔TP switching.

    Lifecycle:
        1. ``reserve()`` — declare each tensor (name, shape, dtype).
        2. ``materialize()`` — compute aligned offsets, allocate buffer.
        3. ``get_view()`` — obtain typed, shaped views into the buffer.
    """

    ALIGNMENT: int = 256  # bytes — keeps GPU loads aligned

    def __init__(
        self,
        *,
        device: str = "cuda",
        gpu_id: Optional[int] = None,
        server_args: Optional["ServerArgs"] = None,
        cpu_group: Optional["ProcessGroup"] = None,
        world_size: int = 1,
    ) -> None:
        self.device = device
        if gpu_id is not None:
            self.gpu_id = gpu_id
        elif device == "cuda":
            self.gpu_id = torch.cuda.current_device()
        else:
            self.gpu_id = 0
        self.server_args = server_args
        self.cpu_group = cpu_group
        self.world_size = world_size

        self._entries: Dict[str, LayoutEntry] = {}
        self._reservation_order: List[str] = []
        self._buffer: Optional[torch.Tensor] = None
        self._materialized: bool = False
        self._total_bytes: int = 0
        self._buffer_start: int = 0
        self._buffer_end: int = 0
        self.ep_max_kv_tokens: int = 0
        self.tp_max_kv_tokens: int = 0
        self.ep_max_kv_tokens_swa: int = 0
        self.tp_max_kv_tokens_swa: int = 0
        self.ep_max_num_reqs: int = 0
        self.tp_max_num_reqs: int = 0
        self.ep_max_running_requests: int = 0
        self.tp_max_running_requests: int = 0
        self._kv_reserved: bool = False
        self._paras_kv_pending: Optional[Dict] = None
        self._paras_moe_pending: Optional[Dict] = None
        self._deferred_weight_bytes: int = 0
        self._planned_umm_bytes_limit: Optional[int] = None
        

    # ----- reservation ----------------------------------------------------

    def reserve(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
    ) -> LayoutEntry:
        """
        Register a tensor to be placed in the contiguous buffer.

        WHY TRACK RESERVATION ORDER:
          Offsets are assigned in the order tensors are reserved. This deterministic
          ordering ensures reproducible memory layouts across runs, which is critical
          for distributed training where all ranks must agree on the buffer structure.
        """
        if self._materialized:
            raise RuntimeError(
                "Cannot reserve after the buffer has been materialized."
            )
        if name in self._entries:
            raise ValueError(f"Duplicate reservation name: '{name}'")
        if dtype not in _SUPPORTED_DTYPES:
            raise ValueError(
                f"Unsupported dtype {dtype}. "
                f"Supported: {_SUPPORTED_DTYPES}"
            )

        numel = 1
        for d in shape:
            numel *= d
        elem_size = dtype.itemsize if hasattr(dtype, "itemsize") else torch.tensor([], dtype=dtype).element_size()
        size_bytes = numel * elem_size

        entry = LayoutEntry(
            name=name,
            shape=shape,
            dtype=dtype,
            numel=numel,
            element_size=elem_size,
            size_bytes=size_bytes,
        )
        self._entries[name] = entry
        self._reservation_order.append(name)  # Preserve order for deterministic offset assignment
        return entry

    # ----- KV cache reservation -------------------------------------------

    def reserve_kv_cache(
        self,
        *,
        num_layers: int,
        ep_max_tokens: int,
        tp_max_tokens: int,
        num_kv_heads: int,
        head_dim: int,
        kv_dtype: torch.dtype,
        tp_size: int = 1,
        page_size: int = 1,
        prefix: str = "model",
        layer_specs: Optional[list] = None,
    ) -> None:
        """
        Reserve KV cache using a contiguous buffer with per-layer offsets.

        Must be called AFTER plan_qwen_moe_layout() and BEFORE materialize().

        Layout (per K and V separately):
          - TP views are packed at the front of the region.
          - EP views are packed after the smallest gap that keeps every
            same-layer EP source disjoint from its TP destination.
          - TP and EP entries have their own UMM-computed shapes, so GQA
            replication and floor effects are represented explicitly instead
            of inferred from the other mode's byte count.

        Actual LayoutEntry objects are created during materialize() so that
        offsets are computed relative to the end of the weight region.
        """
        if self._materialized:
            raise RuntimeError("Cannot reserve KV cache after materialize().")
        if self._kv_reserved:
            raise RuntimeError("KV cache already reserved.")

        self.ep_max_kv_tokens = ep_max_tokens
        self.tp_max_kv_tokens = tp_max_tokens
        self.ep_max_kv_tokens_swa = 0
        self.tp_max_kv_tokens_swa = 0
        self._layer_specs = layer_specs

        elem_size = (
            kv_dtype.itemsize
            if hasattr(kv_dtype, "itemsize")
            else torch.tensor([], dtype=kv_dtype).element_size()
        )
        tp_kv_heads = max(1, num_kv_heads // tp_size)

        if layer_specs is None:
            ep_per_layer_tokens = ep_max_tokens + page_size
            tp_per_layer_tokens = tp_max_tokens + page_size
            ep_per_layer_bytes = (
                ep_per_layer_tokens * num_kv_heads * head_dim * elem_size
            )
            tp_per_layer_bytes = (
                tp_per_layer_tokens * tp_kv_heads * head_dim * elem_size
            )
            layer_ep_shapes = [
                (ep_per_layer_tokens, num_kv_heads, head_dim)
            ] * num_layers
            layer_tp_shapes = [
                (tp_per_layer_tokens, tp_kv_heads, head_dim)
            ] * num_layers
            layer_ep_bytes = [ep_per_layer_bytes] * num_layers
            layer_tp_bytes = [tp_per_layer_bytes] * num_layers
        else:
            layer_ep_shapes = [
                (s.tokens_cap_ep + page_size, s.num_kv_heads, s.head_dim)
                for s in layer_specs
            ]
            layer_tp_shapes = [
                (
                    s.tokens_cap_tp + page_size,
                    max(1, s.num_kv_heads // tp_size),
                    s.head_dim,
                )
                for s in layer_specs
            ]
            layer_ep_bytes = [
                (s.tokens_cap_ep + page_size)
                * s.num_kv_heads
                * s.head_dim
                * elem_size
                for s in layer_specs
            ]
            layer_tp_bytes = [
                (s.tokens_cap_tp + page_size)
                * max(1, s.num_kv_heads // tp_size)
                * s.head_dim
                * elem_size
                for s in layer_specs
            ]
            full_specs = [s for s in layer_specs if s.kind == "full"]
            swa_specs = [s for s in layer_specs if s.kind == "swa"]
            if full_specs:
                self.ep_max_kv_tokens = max(s.tokens_cap_ep for s in full_specs)
                self.tp_max_kv_tokens = max(s.tokens_cap_tp for s in full_specs)
            if swa_specs:
                self.ep_max_kv_tokens_swa = max(s.tokens_cap_ep for s in swa_specs)
                self.tp_max_kv_tokens_swa = max(s.tokens_cap_tp for s in swa_specs)

        # Placed by the deferred four-anchor pass at materialize time, which
        # asserts the ct[i] <= ce[i] precondition its (num_layers+1) kv budget
        # relies on (real configs always satisfy it).
        self._paras_kv_pending = {
            "num_layers": num_layers,
            "prefix": prefix,
            "layer_ep_bytes": layer_ep_bytes,
            "layer_tp_bytes": layer_tp_bytes,
            "layer_ep_shapes": layer_ep_shapes,
            "layer_tp_shapes": layer_tp_shapes,
            "kv_dtype": kv_dtype,
        }

        self._kv_reserved = True

    def _resolve_kv_store_dtype(self) -> torch.dtype:
        s = self.server_args.kv_cache_dtype if self.server_args is not None else "auto"
        if s in ("fp8", "fp8_e4m3fn"):
            return torch.float8_e4m3fn
        return torch.bfloat16

    def _compute_non_umm_static_bytes(self, config) -> int:
        # embed_tokens + lm_head are DP-replicated full vocab tensors (not
        # mode-switching, so they live outside the UMM) but must count
        # against mem-fraction-static so the contract holds at the driver.
        vocab_size = getattr(config, "vocab_size", 0)
        hidden_size = getattr(config, "hidden_size", 0)
        tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
        elem_size = 2
        embed_bytes = vocab_size * hidden_size * elem_size
        lm_head_bytes = 0 if tie_word_embeddings else embed_bytes
        return embed_bytes + lm_head_bytes

    def _compute_umm_budget_bytes(
        self, config=None
    ) -> Tuple[int, int, int, int, int, int, float]:
        from sglang.srt.utils.common import get_available_gpu_memory

        assert self.server_args is not None, (
            "ParaSMemoryManager: server_args required for budget planning. "
            "Construct via ParaSMemoryManager(server_args=...) in model_runner."
        )

        total_gpu_bytes = torch.cuda.get_device_properties(self.gpu_id).total_memory
        avail_now_gib = get_available_gpu_memory(
            self.device,
            self.gpu_id,
            distributed=self.world_size > 1,
            cpu_group=self.cpu_group,
            empty_cache=True,
        )
        avail_now_bytes = int(avail_now_gib * (1 << 30))

        mem_fraction = self.server_args.mem_fraction_static
        assert mem_fraction is not None, "server_args.mem_fraction_static is required"
        dynamic_reserve_bytes = int(total_gpu_bytes * (1.0 - mem_fraction))
        umm_budget_bytes = max(0, avail_now_bytes - dynamic_reserve_bytes)
        non_umm_static_bytes = (
            self._compute_non_umm_static_bytes(config) if config is not None else 0
        )
        manager_budget_bytes = max(0, umm_budget_bytes - non_umm_static_bytes)

        return (
            avail_now_bytes,
            total_gpu_bytes,
            dynamic_reserve_bytes,
            umm_budget_bytes,
            manager_budget_bytes,
            non_umm_static_bytes,
            avail_now_gib,
        )

    def plan_mha_kv_capacity(
        self,
        *,
        config,
        tp_size: int,
        head_dim: int,
    ) -> ParaSKVCapacityPlan:
        """End-to-end MHA KV capacity planner.

        Reads device / gpu_id / server_args / cpu_group / world_size from
        ``self`` (set at construction in model_runner). Calls
        ``get_available_gpu_memory``, computes the UMM byte limit, and derives
        separate EP and TP token caps whose exact four-anchor footprint fits
        that limit. Populates ``self.ep_max_kv_tokens`` and
        ``self.tp_max_kv_tokens``, logs at INFO level, and returns the plan
        including the resolved ``kv_dtype`` for downstream
        ``reserve_kv_cache``.
        """
        kv_dtype = self._resolve_kv_store_dtype()
        elem_size = (
            kv_dtype.itemsize
            if hasattr(kv_dtype, "itemsize")
            else torch.tensor([], dtype=kv_dtype).element_size()
        )

        (
            avail_now_bytes,
            total_gpu_bytes,
            dynamic_reserve_bytes,
            umm_budget_bytes,
            manager_budget_bytes,
            non_umm_static_bytes,
            avail_now_gib,
        ) = self._compute_umm_budget_bytes(config)

        num_layers = config.num_hidden_layers
        num_kv_heads = config.num_key_value_heads
        page_size = getattr(self.server_args, "page_size", 1) or 1

        ep_kv_heads = num_kv_heads
        tp_kv_heads = max(1, num_kv_heads // tp_size)
        ep_cell_bytes = ep_kv_heads * head_dim * num_layers * 2 * elem_size
        tp_cell_bytes = tp_kv_heads * head_dim * num_layers * 2 * elem_size

        def _build_cache(ep_budget_bytes: int, tp_budget_bytes: int):
            def _tokens_for_budget(budget_bytes: int, cell_bytes: int) -> int:
                page_bytes = page_size * cell_bytes
                return max(1, (budget_bytes - page_bytes) // cell_bytes)

            ep_max_tokens = _tokens_for_budget(ep_budget_bytes, ep_cell_bytes)
            tp_max_tokens = _tokens_for_budget(tp_budget_bytes, tp_cell_bytes)
            tp_max_tokens = self._cap_tp_tokens_to_ep_layer(
                ep_tokens=ep_max_tokens,
                tp_tokens=tp_max_tokens,
                ep_kv_heads=ep_kv_heads,
                tp_kv_heads=tp_kv_heads,
                head_dim=head_dim,
                elem_size=elem_size,
                page_size=page_size,
            )
            ep_k_bytes = (
                (ep_max_tokens + page_size)
                * ep_kv_heads
                * head_dim
                * elem_size
            )
            tp_k_bytes = (
                (tp_max_tokens + page_size)
                * tp_kv_heads
                * head_dim
                * elem_size
            )
            return {
                "ep_max_tokens": ep_max_tokens,
                "tp_max_tokens": tp_max_tokens,
                "layer_ep_bytes": [ep_k_bytes] * num_layers,
                "layer_tp_bytes": [tp_k_bytes] * num_layers,
            }

        cache_plan, geometry, fixed_umm_bytes, planned_umm_bytes = (
            self._plan_balanced_kv_footprint(
                manager_budget_bytes,
                _build_cache,
            )
        )
        ep_max_tokens = cache_plan["ep_max_tokens"]
        tp_max_tokens = cache_plan["tp_max_tokens"]
        self.ep_max_kv_tokens = ep_max_tokens
        self.tp_max_kv_tokens = tp_max_tokens

        logger.info(
            f"ParaS KV budget: avail_now={avail_now_gib:.3f}GiB  "
            f"total={total_gpu_bytes / (1 << 30):.3f}GiB  "
            f"dynamic_reserve={dynamic_reserve_bytes / (1 << 30):.3f}GiB  "
            f"umm_budget={umm_budget_bytes / (1 << 30):.3f}GiB  "
            f"non_umm_static={non_umm_static_bytes / (1 << 30):.3f}GiB  "
            f"manager_budget={manager_budget_bytes / (1 << 30):.3f}GiB  "
            f"planned_umm={planned_umm_bytes / (1 << 30):.3f}GiB  "
            f"fixed_umm={fixed_umm_bytes / (1 << 30):.3f}GiB  "
            f"ep_weights={geometry.sum_we / (1 << 30):.3f}GiB  "
            f"tp_weights={geometry.sum_wt / (1 << 30):.3f}GiB  "
            f"ep_kv={geometry.sum_ce / (1 << 30):.3f}GiB  "
            f"tp_kv={geometry.sum_ct / (1 << 30):.3f}GiB  "
            f"ep_max_tokens={ep_max_tokens}  "
            f"tp_max_tokens={tp_max_tokens}  "
            f"ep_kv_heads={ep_kv_heads}  "
            f"tp_kv_heads={tp_kv_heads}"
        )

        return ParaSKVCapacityPlan(
            available_gpu_memory_bytes=avail_now_bytes,
            total_gpu_memory_bytes=total_gpu_bytes,
            dynamic_reserve_bytes=dynamic_reserve_bytes,
            umm_budget_bytes=umm_budget_bytes,
            non_umm_static_bytes=non_umm_static_bytes,
            manager_budget_bytes=manager_budget_bytes,
            fixed_umm_bytes=fixed_umm_bytes,
            ep_expert_weight_bytes=geometry.sum_we,
            tp_expert_weight_bytes=geometry.sum_wt,
            ep_kv_bytes=geometry.sum_ce,
            tp_kv_bytes=geometry.sum_ct,
            planned_umm_bytes=planned_umm_bytes,
            kv_dtype=kv_dtype,
            ep_max_tokens=ep_max_tokens,
            tp_max_tokens=tp_max_tokens,
            ep_cell_bytes=ep_cell_bytes,
            tp_cell_bytes=tp_cell_bytes,
            ep_kv_heads=ep_kv_heads,
            tp_kv_heads=tp_kv_heads,
            full_layers=num_layers,
        )

    def plan_hybrid_swa_kv_capacity(
        self,
        *,
        config,
        tp_size: int,
        head_dim: int,
    ) -> ParaSKVCapacityPlan:
        """End-to-end SWA-aware KV capacity planner for hybrid full / SWA models.

        Same prelude as :meth:`plan_mha_kv_capacity` (reads device, gpu_id,
        cpu_group, world_size, server_args from ``self``). Then splits the
        per-layer-token budget across full and SWA layers using
        ``plan_hybrid_kv_budget`` with ``server_args.swa_full_tokens_ratio``,
        builds per-layer specs via :func:`classify_layers_from_config`, and
        returns a plan whose ``layer_specs`` is consumed downstream by
        :meth:`reserve_kv_cache`.
        """
        from sglang.srt.paras.layers.utils import classify_layers_from_config

        kv_dtype = self._resolve_kv_store_dtype()
        elem_size = (
            kv_dtype.itemsize
            if hasattr(kv_dtype, "itemsize")
            else torch.tensor([], dtype=kv_dtype).element_size()
        )

        (
            avail_now_bytes,
            total_gpu_bytes,
            dynamic_reserve_bytes,
            umm_budget_bytes,
            manager_budget_bytes,
            non_umm_static_bytes,
            avail_now_gib,
        ) = self._compute_umm_budget_bytes(config)

        num_layers = config.num_hidden_layers
        num_kv_heads = config.num_key_value_heads
        page_size = getattr(self.server_args, "page_size", 1) or 1
        layer_types = getattr(config, "layer_types", None) or (
            ["full_attention"] * num_layers
        )
        if len(layer_types) != num_layers:
            raise ValueError(
                "config.layer_types must contain one entry per hidden layer"
            )
        n_full = sum(
            1 for layer_type in layer_types if layer_type == "full_attention"
        )
        n_swa = sum(
            1 for layer_type in layer_types if layer_type == "sliding_attention"
        )

        ep_kv_heads = num_kv_heads
        tp_kv_heads = max(1, num_kv_heads // tp_size)
        ep_layer_cell_bytes = ep_kv_heads * head_dim * 2 * elem_size
        tp_layer_cell_bytes = tp_kv_heads * head_dim * 2 * elem_size
        ep_cell_bytes = ep_layer_cell_bytes * num_layers
        tp_cell_bytes = tp_layer_cell_bytes * num_layers
        swa_ratio = getattr(self.server_args, "swa_full_tokens_ratio", 0.5)

        def _solve_tokens(
            budget_bytes: int, layer_cell_bytes: int
        ) -> Tuple[int, int]:
            page_bytes = page_size * layer_cell_bytes * num_layers
            total_layer_tokens = max(
                1, (budget_bytes - page_bytes) // layer_cell_bytes
            )
            full_tokens, swa_tokens = plan_hybrid_kv_budget(
                total_layer_tokens,
                n_full,
                n_swa,
                swa_ratio,
            )
            if n_full > 0:
                full_tokens = max(1, full_tokens)
            if n_swa > 0:
                swa_tokens = max(1, swa_tokens)
            return full_tokens, swa_tokens

        def _build_cache(ep_budget_bytes: int, tp_budget_bytes: int):
            ep_full, ep_swa = _solve_tokens(ep_budget_bytes, ep_layer_cell_bytes)
            tp_full, tp_swa = _solve_tokens(tp_budget_bytes, tp_layer_cell_bytes)
            if n_full > 0:
                tp_full = self._cap_tp_tokens_to_ep_layer(
                    ep_tokens=ep_full,
                    tp_tokens=tp_full,
                    ep_kv_heads=ep_kv_heads,
                    tp_kv_heads=tp_kv_heads,
                    head_dim=head_dim,
                    elem_size=elem_size,
                    page_size=page_size,
                )
            if n_swa > 0:
                tp_swa = self._cap_tp_tokens_to_ep_layer(
                    ep_tokens=ep_swa,
                    tp_tokens=tp_swa,
                    ep_kv_heads=ep_kv_heads,
                    tp_kv_heads=tp_kv_heads,
                    head_dim=head_dim,
                    elem_size=elem_size,
                    page_size=page_size,
                )
            layer_ep_bytes = []
            layer_tp_bytes = []
            for layer_type in layer_types:
                is_swa = layer_type == "sliding_attention"
                ep_tokens = ep_swa if is_swa else ep_full
                tp_tokens = tp_swa if is_swa else tp_full
                layer_ep_bytes.append(
                    (ep_tokens + page_size)
                    * ep_kv_heads
                    * head_dim
                    * elem_size
                )
                layer_tp_bytes.append(
                    (tp_tokens + page_size)
                    * tp_kv_heads
                    * head_dim
                    * elem_size
                )
            return {
                "ep_max_tokens": ep_full,
                "tp_max_tokens": tp_full,
                "ep_max_tokens_swa": ep_swa,
                "tp_max_tokens_swa": tp_swa,
                "layer_ep_bytes": layer_ep_bytes,
                "layer_tp_bytes": layer_tp_bytes,
            }

        cache_plan, geometry, fixed_umm_bytes, planned_umm_bytes = (
            self._plan_balanced_kv_footprint(
                manager_budget_bytes,
                _build_cache,
            )
        )
        ep_max_tokens = cache_plan["ep_max_tokens"]
        tp_max_tokens = cache_plan["tp_max_tokens"]
        ep_max_tokens_swa = cache_plan["ep_max_tokens_swa"]
        tp_max_tokens_swa = cache_plan["tp_max_tokens_swa"]

        layer_specs = classify_layers_from_config(
            config,
            tp_size=tp_size,
            ep_tokens_full=ep_max_tokens,
            tp_tokens_full=tp_max_tokens,
            ep_tokens_swa=ep_max_tokens_swa,
            tp_tokens_swa=tp_max_tokens_swa,
        )

        self.ep_max_kv_tokens = ep_max_tokens
        self.tp_max_kv_tokens = tp_max_tokens
        self.ep_max_kv_tokens_swa = ep_max_tokens_swa
        self.tp_max_kv_tokens_swa = tp_max_tokens_swa

        logger.info(
            f"ParaS SWA KV budget: avail_now={avail_now_gib:.3f}GiB  "
            f"total={total_gpu_bytes / (1 << 30):.3f}GiB  "
            f"dynamic_reserve={dynamic_reserve_bytes / (1 << 30):.3f}GiB  "
            f"umm_budget={umm_budget_bytes / (1 << 30):.3f}GiB  "
            f"non_umm_static={non_umm_static_bytes / (1 << 30):.3f}GiB  "
            f"manager_budget={manager_budget_bytes / (1 << 30):.3f}GiB  "
            f"planned_umm={planned_umm_bytes / (1 << 30):.3f}GiB  "
            f"fixed_umm={fixed_umm_bytes / (1 << 30):.3f}GiB  "
            f"ep_weights={geometry.sum_we / (1 << 30):.3f}GiB  "
            f"tp_weights={geometry.sum_wt / (1 << 30):.3f}GiB  "
            f"ep_kv={geometry.sum_ce / (1 << 30):.3f}GiB  "
            f"tp_kv={geometry.sum_ct / (1 << 30):.3f}GiB  "
            f"ep_full_tokens={ep_max_tokens}  "
            f"tp_full_tokens={tp_max_tokens}  "
            f"ep_swa_tokens={ep_max_tokens_swa}  "
            f"tp_swa_tokens={tp_max_tokens_swa}  "
            f"layers={num_layers} (full={n_full} swa={n_swa})"
        )

        return ParaSKVCapacityPlan(
            available_gpu_memory_bytes=avail_now_bytes,
            total_gpu_memory_bytes=total_gpu_bytes,
            dynamic_reserve_bytes=dynamic_reserve_bytes,
            umm_budget_bytes=umm_budget_bytes,
            non_umm_static_bytes=non_umm_static_bytes,
            manager_budget_bytes=manager_budget_bytes,
            fixed_umm_bytes=fixed_umm_bytes,
            ep_expert_weight_bytes=geometry.sum_we,
            tp_expert_weight_bytes=geometry.sum_wt,
            ep_kv_bytes=geometry.sum_ce,
            tp_kv_bytes=geometry.sum_ct,
            planned_umm_bytes=planned_umm_bytes,
            kv_dtype=kv_dtype,
            ep_max_tokens=ep_max_tokens,
            tp_max_tokens=tp_max_tokens,
            ep_cell_bytes=ep_cell_bytes,
            tp_cell_bytes=tp_cell_bytes,
            ep_kv_heads=ep_kv_heads,
            tp_kv_heads=tp_kv_heads,
            full_layers=n_full,
            swa_layers=n_swa,
            ep_max_tokens_swa=ep_max_tokens_swa,
            tp_max_tokens_swa=tp_max_tokens_swa,
            layer_specs=layer_specs,
        )

    def plan_req_capacities(
        self,
        *,
        context_len: int,
        ep_max_num_reqs: Optional[int] = None,
        max_running_requests: Optional[int] = None,
        dp_size: int = 1,
    ) -> Tuple[int, int]:
        """Compute EP and TP request pool capacities from UMM token budgets.

        Also derives per-mode ``max_running_requests`` caps. EP has 8 disjoint
        per-rank schedulers, so each cap divides the global CLI value by
        dp_size; TP runs one unified scheduler whose cap equals the full CLI
        value. Both are clamped to the per-mode pool capacity so the
        scheduler never tries to admit more reqs than the pool can hold.
        """

        def _default_num_reqs(max_tokens: int) -> int:
            return min(max(int(max_tokens / context_len * 512), 2048), 4096)

        ep_num_reqs = (
            ep_max_num_reqs
            if ep_max_num_reqs is not None
            else _default_num_reqs(self.ep_max_kv_tokens)
        )
        tp_num_reqs = max(ep_num_reqs, _default_num_reqs(self.tp_max_kv_tokens))

        self.ep_max_num_reqs = ep_num_reqs
        self.tp_max_num_reqs = tp_num_reqs

        if max_running_requests is not None:
            self.ep_max_running_requests = min(
                max(max_running_requests // max(dp_size, 1), 1), ep_num_reqs
            )
            self.tp_max_running_requests = min(max_running_requests, tp_num_reqs)
        else:
            self.ep_max_running_requests = ep_num_reqs
            self.tp_max_running_requests = tp_num_reqs

        return ep_num_reqs, tp_num_reqs

    def get_ep_max_num_reqs(self) -> int:
        return self.ep_max_num_reqs

    def get_tp_max_num_reqs(self) -> int:
        return self.tp_max_num_reqs

    def get_ep_max_running_requests(self) -> int:
        return self.ep_max_running_requests

    def get_tp_max_running_requests(self) -> int:
        return self.tp_max_running_requests

    def get_ep_max_kv_tokens(self, kind: str = "full") -> int:
        if kind == "full":
            return self.ep_max_kv_tokens
        if kind == "swa":
            return self.ep_max_kv_tokens_swa
        raise ValueError(f"Unknown KV token capacity kind: {kind}")

    def get_tp_max_kv_tokens(self, kind: str = "full") -> int:
        if kind == "full":
            return self.tp_max_kv_tokens
        if kind == "swa":
            return self.tp_max_kv_tokens_swa
        raise ValueError(f"Unknown KV token capacity kind: {kind}")

    def has_kv_cache_reserved(self) -> bool:
        return self._kv_reserved

    # ----- exact footprint planning --------------------------------------

    def _ordinary_layout_bytes(self) -> int:
        """Return the aligned prefix occupied outside the four-anchor run."""
        offset = 0
        for name in self._reservation_order:
            entry = self._entries[name]
            offset = self._align_up(offset, self.ALIGNMENT) + entry.size_bytes
        return self._align_up(offset, self.ALIGNMENT)

    def _compute_paras_run_geometry(
        self,
        moe: Optional[Dict],
        kv: Optional[Dict],
    ) -> _ParaSRunGeometry:
        """Compute the exact aligned four-anchor footprint without allocating."""
        meta = moe if moe is not None else kv
        assert meta is not None, "ParaS run geometry requires MoE or KV metadata"
        num_layers = meta["num_layers"]
        if moe is not None and kv is not None:
            assert kv["num_layers"] == num_layers, (
                moe["num_layers"],
                kv["num_layers"],
            )

        def au(value: int) -> int:
            return self._align_up(value, self.ALIGNMENT)

        if moe is not None:
            we_slab = au(moe["ep_w13_bytes"]) + au(moe["ep_w2_bytes"])
            wt_slab = au(moe["tp_w13_bytes"]) + au(moe["tp_w2_bytes"])
            we = (we_slab,) * num_layers
            wt = (wt_slab,) * num_layers
        else:
            we = (0,) * num_layers
            wt = (0,) * num_layers

        if kv is not None:
            assert len(kv["layer_ep_bytes"]) == num_layers
            assert len(kv["layer_tp_bytes"]) == num_layers
            ce = tuple(au(value) * 2 for value in kv["layer_ep_bytes"])
            ct = tuple(au(value) * 2 for value in kv["layer_tp_bytes"])
        else:
            ce = (0,) * num_layers
            ct = (0,) * num_layers

        sum_we = sum(we)
        sum_wt = sum(wt)
        sum_ce = sum(ce)
        sum_ct = sum(ct)

        assert all(ct[i] <= ce[i] for i in range(num_layers)), (
            "four-anchor cache requires ct[i] <= ce[i] (per-layer TP cache "
            f"must not exceed EP cache); ce={ce} ct={ct}"
        )

        anchor = 0
        suffix = 0
        for i in range(num_layers - 1, -1, -1):
            anchor = max(anchor, ct[i] + suffix)
            suffix += ct[i] - ce[i]
        anchor = au(anchor)

        tp_head_bytes = we[0] if we else 0
        pad = au(
            max(
                0,
                tp_head_bytes
                + sum_wt
                - sum_we
                - sum_ce
                + sum_ct
                - anchor,
            )
        )
        run_bytes = sum_we + pad + sum_ce + anchor
        assert run_bytes == max(
            sum_we + sum_ce + anchor,
            tp_head_bytes + sum_wt + sum_ct,
        )

        return _ParaSRunGeometry(
            we=we,
            wt=wt,
            ce=ce,
            ct=ct,
            sum_we=sum_we,
            sum_wt=sum_wt,
            sum_ce=sum_ce,
            sum_ct=sum_ct,
            anchor=anchor,
            pad=pad,
            run_bytes=run_bytes,
        )

    def _cap_tp_tokens_to_ep_layer(
        self,
        *,
        ep_tokens: int,
        tp_tokens: int,
        ep_kv_heads: int,
        tp_kv_heads: int,
        head_dim: int,
        elem_size: int,
        page_size: int,
    ) -> int:
        """Enforce the four-anchor ct <= ce invariant after alignment."""

        def _aligned_k_bytes(tokens: int, heads: int) -> int:
            raw_bytes = (
                (tokens + page_size) * heads * head_dim * elem_size
            )
            return self._align_up(raw_bytes, self.ALIGNMENT)

        ep_bytes = _aligned_k_bytes(ep_tokens, ep_kv_heads)
        while (
            tp_tokens > 1
            and _aligned_k_bytes(tp_tokens, tp_kv_heads) > ep_bytes
        ):
            tp_tokens -= 1

        assert _aligned_k_bytes(tp_tokens, tp_kv_heads) <= ep_bytes, (
            "one TP KV token plus page slots exceeds the EP layer capacity: "
            f"{ep_tokens=} {tp_tokens=} {ep_kv_heads=} {tp_kv_heads=}"
        )
        return tp_tokens

    def _plan_balanced_kv_footprint(self, manager_budget_bytes: int, build_cache):
        """Maximize a shared EP/TP base footprint under one exact byte limit.

        For a candidate base footprint B, EP receives B - sum(we) KV bytes
        while TP receives B - sum(wt). This charges the larger TP expert
        layout against TP cache capacity instead of incorrectly giving both
        modes the same cache-byte budget. The exact four-anchor geometry then
        accounts for its one-layer head/tail overhead and alignment.
        """
        fixed_umm_bytes = self._ordinary_layout_bytes()
        run_budget_bytes = manager_budget_bytes - fixed_umm_bytes
        if run_budget_bytes <= 0:
            raise RuntimeError(
                "ParaS UMM budget is exhausted by fixed weights and staging: "
                f"{manager_budget_bytes=} {fixed_umm_bytes=}"
            )

        weight_geometry = self._compute_paras_run_geometry(
            self._paras_moe_pending,
            None,
        )
        low = 0
        high = run_budget_bytes
        best = None
        while low <= high:
            base_bytes = (low + high) // 2
            ep_cache_budget = max(0, base_bytes - weight_geometry.sum_we)
            tp_cache_budget = max(0, base_bytes - weight_geometry.sum_wt)
            cache_plan = build_cache(ep_cache_budget, tp_cache_budget)
            kv = {
                "num_layers": len(cache_plan["layer_ep_bytes"]),
                "layer_ep_bytes": cache_plan["layer_ep_bytes"],
                "layer_tp_bytes": cache_plan["layer_tp_bytes"],
            }
            geometry = self._compute_paras_run_geometry(
                self._paras_moe_pending,
                kv,
            )
            planned_umm_bytes = fixed_umm_bytes + geometry.run_bytes
            if planned_umm_bytes <= manager_budget_bytes:
                best = (cache_plan, geometry, fixed_umm_bytes, planned_umm_bytes)
                low = base_bytes + 1
            else:
                high = base_bytes - 1

        if best is None:
            raise RuntimeError(
                "ParaS UMM budget cannot fit expert weights and one KV token "
                f"per layer: {manager_budget_bytes=}"
            )

        self._planned_umm_bytes_limit = manager_budget_bytes
        return best

    # ----- materialization ------------------------------------------------

    def materialize(self) -> int:
        """
        Assign aligned offsets in reservation order, then allocate the
        backing ``uint8`` buffer on ``self.device``.

        Returns the total buffer size in bytes.

        WHY UINT8 BUFFER:
          We store raw bytes (uint8) instead of a typed buffer because different tensors
          have different dtypes (BF16, FP8, FP32). A uint8 buffer is dtype-agnostic and
          allows get_view() to reinterpret the same bytes as different types via .view(dtype).

        WHY 256-BYTE ALIGNMENT:
          GPU memory coalescing works best when tensors start at 256-byte boundaries.
          This alignment ensures efficient memory access patterns during kernel execution.

        WHY STORE BUFFER_START/BUFFER_END:
          These pointers enable is_managed() to quickly check if a tensor's data pointer
          falls within our managed buffer. This is used to distinguish managed vs. external
          tensors during parameter wrapping.
        """
        offset = 0
        for name in self._reservation_order:
            entry = self._entries[name]
            entry.offset_bytes = self._align_up(offset, self.ALIGNMENT)
            offset = entry.offset_bytes + entry.size_bytes

        moe_pending = getattr(self, "_paras_moe_pending", None)
        kv_pending = getattr(self, "_paras_kv_pending", None)
        if moe_pending is not None or kv_pending is not None:
            offset = self._place_paras_run(offset, moe_pending, kv_pending)

        self._total_bytes = self._align_up(offset, self.ALIGNMENT)
        if (
            self._planned_umm_bytes_limit is not None
            and self._total_bytes > self._planned_umm_bytes_limit
        ):
            raise RuntimeError(
                "ParaS materialized UMM exceeds its planned static budget: "
                f"total={self._total_bytes} limit={self._planned_umm_bytes_limit}"
            )
        self._buffer = torch.empty(
            self._total_bytes, dtype=torch.uint8, device=self.device
        )
        buf = self._buffer
        assert buf is not None
        self._buffer_start = buf.data_ptr()
        self._buffer_end = self._buffer_start + self._total_bytes
        self._materialized = True
        return self._total_bytes

    # ----- unified four-anchor layout (called from materialize) -----------

    def _register_entry(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        size_bytes: int,
        offset_bytes: int,
    ) -> LayoutEntry:
        numel = 1
        for d in shape:
            numel *= d
        elem_size = size_bytes // numel if numel else 0
        entry = LayoutEntry(
            name=name,
            shape=shape,
            dtype=dtype,
            numel=numel,
            element_size=elem_size,
            size_bytes=size_bytes,
            offset_bytes=offset_bytes,
        )
        self._entries[name] = entry
        return entry

    def _place_paras_run(
        self,
        offset: int,
        moe: Optional[Dict],
        kv: Optional[Dict],
    ) -> int:
        """Place expert weights and KV cache in one four-anchor run.

        Orientation EP-low / TP-high. Per mode, address order is
        ``weights | pad | cache``; the big TP weights overlap the big EP cache
        (EP and TP are never live together), so the buffer is ~one per-mode
        footprint plus one layer. The tail anchor
        ``max_i(ct[i] + sum_{k>i}(ct[k] - ce[k]))`` keeps every layer's EP and TP
        cache disjoint for any layer order. This method requires ``ct[i] <= ce[i]``
        (asserted by the shared geometry helper). Under that precondition the
        anchor reduces to ``max(ct)``; capacity planning charges the exact
        resulting run, including this tail and all alignment, to the UMM limit.
        Weight sub-slabs are ``[w13|w2]`` and cache sub-slabs ``[k|v]``, laid
        identically in both modes so the offset-agnostic transfer kernels stay valid.

        Returns the byte offset past the end of the run.
        """
        A = self.ALIGNMENT
        P = self._align_up(offset, A)

        meta = moe if moe is not None else kv
        assert meta is not None, "_place_paras_run requires moe or kv pending"
        num_layers = meta["num_layers"]
        prefix = meta["prefix"]
        if num_layers == 0:
            return P

        def au(x: int) -> int:
            return self._align_up(x, A)

        geometry = self._compute_paras_run_geometry(moe, kv)
        we = geometry.we
        wt = geometry.wt
        ce = geometry.ce
        ct = geometry.ct
        sum_we = geometry.sum_we
        sum_wt = geometry.sum_wt
        sum_ce = geometry.sum_ce
        sum_ct = geometry.sum_ct
        anchor = geometry.anchor

        w_end = P + sum_we
        tp_w_end = P + we[0] + sum_wt
        PAD = geometry.pad
        EP_end = w_end + PAD + sum_ce
        tc_end = P + geometry.run_bytes
        assert EP_end % A == 0 and tc_end % A == 0, (EP_end, tc_end)

        if moe is not None:
            dtype = moe["dtype"]
            off = P
            for i in range(num_layers):
                lp = f"{prefix}.layers.{i}.mlp.ep_experts"
                self._register_entry(f"{lp}.w13_weight", moe["ep_w13_shape"], dtype, moe["ep_w13_bytes"], off)
                self._register_entry(f"{lp}.w2_weight", moe["ep_w2_shape"], dtype, moe["ep_w2_bytes"], off + au(moe["ep_w13_bytes"]))
                off += we[i]
            assert off == w_end, (off, w_end)

            off = P + we[0]
            for i in range(num_layers):
                lp = f"{prefix}.layers.{i}.mlp.tp_experts"
                self._register_entry(f"{lp}.w13_weight", moe["tp_w13_shape"], dtype, moe["tp_w13_bytes"], off)
                self._register_entry(f"{lp}.w2_weight", moe["tp_w2_shape"], dtype, moe["tp_w2_bytes"], off + au(moe["tp_w13_bytes"]))
                off += wt[i]
            assert off == tp_w_end, (off, tp_w_end)

            for i in range(num_layers):
                self._entries[f"{prefix}.layers.{i}.mlp.experts.w13_weight"] = self._entries[f"{prefix}.layers.{i}.mlp.ep_experts.w13_weight"]
                self._entries[f"{prefix}.layers.{i}.mlp.experts.w2_weight"] = self._entries[f"{prefix}.layers.{i}.mlp.ep_experts.w2_weight"]

        if kv is not None:
            kv_dtype = kv["kv_dtype"]
            off = w_end + PAD
            for i in range(num_layers):
                kb = kv["layer_ep_bytes"][i]
                shp = kv["layer_ep_shapes"][i]
                k = self._register_entry(f"{prefix}.layers.{i}.kv.ep.k", shp, kv_dtype, kb, off)
                v = self._register_entry(f"{prefix}.layers.{i}.kv.ep.v", shp, kv_dtype, kb, off + au(kb))
                self._entries[f"{prefix}.layers.{i}.kv.k"] = k
                self._entries[f"{prefix}.layers.{i}.kv.v"] = v
                off += ce[i]
            assert off == EP_end, (off, EP_end)

            off = tc_end - sum_ct
            for i in range(num_layers):
                kb = kv["layer_tp_bytes"][i]
                shp = kv["layer_tp_shapes"][i]
                self._register_entry(f"{prefix}.layers.{i}.kv.tp.k", shp, kv_dtype, kb, off)
                self._register_entry(f"{prefix}.layers.{i}.kv.tp.v", shp, kv_dtype, kb, off + au(kb))
                off += ct[i]
            assert off == tc_end, (off, tc_end)

        return tc_end

    # ----- view access ----------------------------------------------------

    def get_view(self, name: str) -> torch.Tensor:
        """
        Return a typed, shaped view into the buffer for *name*.

        BYTE_SLICE → VIEW(DTYPE) → RESHAPE CHAIN:
          1. byte_slice: Extract the raw bytes from the uint8 buffer using offsets.
          2. .view(dtype): Reinterpret those bytes as the target dtype (BF16, FP8, etc.).
             This is a zero-copy operation—no data is moved, just reinterpreted.
          3. .reshape(shape): Reshape the flat 1-D tensor to the original shape.
          
          This chain allows a single uint8 buffer to serve tensors of different dtypes
          without duplication or type conversion overhead.
        """
        if not self._materialized:
            raise RuntimeError("Buffer not materialized yet.")
        if name not in self._entries:
            raise KeyError(f"No reservation named '{name}'")

        entry = self._entries[name]
        assert self._buffer is not None
        byte_slice = self._buffer[
            entry.offset_bytes : entry.offset_bytes + entry.size_bytes
        ]
        return byte_slice.view(entry.dtype).reshape(entry.shape)

    def get_view_as(
        self, name: str, shape: tuple, dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """
        Return the same bytes as *name* but with a different shape/dtype.

        Used for TP reuse: the TP experts share the same underlying buffer
        as the EP experts but interpret it with a different view shape.
        Total bytes must match the original reservation.
        """
        if not self._materialized:
            raise RuntimeError("Buffer not materialized yet.")
        entry = self._entries[name]
        target_dtype = dtype or entry.dtype
        assert self._buffer is not None
        byte_slice = self._buffer[
            entry.offset_bytes : entry.offset_bytes + entry.size_bytes
        ]
        return byte_slice.view(target_dtype).reshape(shape)

    # ----- KV cache views -------------------------------------------------

    def get_kv_views(
        self,
        num_layers: int,
        mode: str,
        tp_size: int = 1,
        page_size: int = 1,
        prefix: str = "model",
        layer_ids: Optional[List[int]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Return k_buffers and v_buffers for the KV pool in the given mode.

        EP mode: returns UMM-planned EP views.
        TP mode: returns UMM-planned TP views when available, falling back to
        reinterpreting EP bytes only for legacy layouts.

        When *layer_ids* is provided, iterate over those specific layer
        indices instead of ``range(num_layers)``.
        """
        k_bufs: List[torch.Tensor] = []
        v_bufs: List[torch.Tensor] = []
        iter_ids = layer_ids if layer_ids is not None else list(range(num_layers))
        for layer_id in iter_ids:
            lp = f"{prefix}.layers.{layer_id}"
            k_name = f"{lp}.kv.k"
            v_name = f"{lp}.kv.v"

            if mode == "ep":
                k_bufs.append(self.get_view(k_name))
                v_bufs.append(self.get_view(v_name))
            elif mode == "tp":
                # Prefer dedicated TP entries (contiguous-buffer design) when available.
                tp_k_name = f"{lp}.kv.tp.k"
                tp_v_name = f"{lp}.kv.tp.v"
                if tp_k_name in self._entries:
                    k_bufs.append(self.get_view(tp_k_name))
                    v_bufs.append(self.get_view(tp_v_name))
                else:
                    # Fallback: reinterpret EP bytes as TP shape.
                    k_entry = self._entries[k_name]
                    ep_heads = k_entry.shape[1]
                    tp_heads = max(1, ep_heads // tp_size)
                    tp_shape = (
                        self.tp_max_kv_tokens + page_size,
                        tp_heads,
                        k_entry.shape[2],
                    )
                    k_bufs.append(self.get_view_as(k_name, tp_shape))
                    v_bufs.append(self.get_view_as(v_name, tp_shape))
            else:
                raise ValueError(f"mode must be 'ep' or 'tp', got '{mode}'")

        return k_bufs, v_bufs

    # ----- aliasing -------------------------------------------------------

    def alias(self, alias_name: str, target_name: str) -> LayoutEntry:
        """
        Create an alias entry that points to the same physical memory as *target*.

        Aliases inherit the target's shape, dtype, offset, and size. They enable
        multiple logical names (e.g., EP vs TP views) to map to the same physical
        slot without duplicating buffer space.

        Must be called after ``materialize()`` because offsets are only valid then.
        """
        if not self._materialized:
            raise RuntimeError("alias() can only be called after materialize().")
        if alias_name in self._entries:
            raise ValueError(f"Alias name already exists: '{alias_name}'")
        if target_name not in self._entries:
            raise KeyError(f"Alias target not found: '{target_name}'")

        target = self._entries[target_name]
        entry = LayoutEntry(
            name=alias_name,
            shape=target.shape,
            dtype=target.dtype,
            numel=target.numel,
            element_size=target.element_size,
            size_bytes=target.size_bytes,
            offset_bytes=target.offset_bytes,
        )
        self._entries[alias_name] = entry
        return entry

    # ----- queries --------------------------------------------------------

    def is_managed(self, tensor: torch.Tensor) -> bool:
        """True if *tensor*'s data pointer falls within the managed buffer."""
        if not self._materialized:
            return False
        ptr = tensor.data_ptr()
        return self._buffer_start <= ptr < self._buffer_end

    def dump_layout(self) -> List[Dict]:
        """All entries as JSON-serializable dicts, in reservation order."""
        return [self._entries[n].to_dict() for n in self._reservation_order]

    def dump_layout_json(self) -> str:
        """Pretty-printed JSON of the full layout."""
        return json.dumps(self.dump_layout(), indent=2)

    # ----- properties -----------------------------------------------------

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    @property
    def num_entries(self) -> int:
        return len(self._entries)

    @property
    def materialized(self) -> bool:
        return self._materialized

    @property
    def buffer(self) -> Optional[torch.Tensor]:
        return self._buffer

    # ----- dunder ---------------------------------------------------------

    def __repr__(self) -> str:
        status = "materialized" if self._materialized else "planned"
        mib = self._total_bytes / (1024 * 1024)
        return (
            f"ParaSMemoryManager("
            f"entries={len(self._entries)}, "
            f"total={mib:.2f} MiB, "
            f"status={status})"
        )

    # ----- helpers --------------------------------------------------------

    @staticmethod
    def _align_up(value: int, alignment: int) -> int:
        return (value + alignment - 1) // alignment * alignment


# ---------------------------------------------------------------------------
# Global manager — set during model construction, checked in create_weights
# ---------------------------------------------------------------------------

_global_paras_memory_manager: Optional[ParaSMemoryManager] = None


def set_global_paras_memory_manager(manager: Optional[ParaSMemoryManager]) -> None:
    global _global_paras_memory_manager
    _global_paras_memory_manager = manager


def get_global_paras_memory_manager() -> Optional[ParaSMemoryManager]:
    return _global_paras_memory_manager


# ---------------------------------------------------------------------------
# Qwen MoE layout planning
# ---------------------------------------------------------------------------


def _validate_moe_parallel_layout(
    *,
    num_experts: int,
    intermediate_size: int,
    num_heads: int,
    ep_size: int,
    tp_size: int,
    dp_size: int,
) -> None:
    if min(ep_size, tp_size, dp_size) <= 0:
        raise ValueError(
            "ParaS parallel sizes must be positive, got "
            f"{ep_size=}, {tp_size=}, and {dp_size=}"
        )
    if ep_size != dp_size * tp_size:
        raise ValueError(
            "ParaS requires ep_size == dp_size * tp_size, got "
            f"{ep_size=}, {dp_size=}, and {tp_size=}"
        )
    for name, value, divisor in (
        ("num_experts", num_experts, ep_size),
        ("intermediate_size", intermediate_size, tp_size),
        ("num_heads", num_heads, tp_size),
    ):
        if value % divisor != 0:
            parallel_size = "ep_size" if name == "num_experts" else "tp_size"
            raise ValueError(
                f"ParaS requires {name} to be divisible by {parallel_size}, "
                f"got {value} and {divisor}"
            )


def plan_qwen_moe_layout(
    manager: ParaSMemoryManager,
    *,
    num_layers: int,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    ep_size: int,
    tp_size: int,
    dp_size: int,
    moe_tp_size: int,
    quant_name: Optional[str] = None,
    fp8_block_size: Optional[int] = None,
    num_fused_shared_experts: int = 0,
    intra_node_weight_transfer_method: str = (
        IntraNodeWeightTransferMethod.PEER_ACCESS.value
    ),
    prefix: str = "model",
) -> None:
    """
    Reserve all weight tensors for a Qwen sparse-MoE model.

    This is the main entry point for planning the contiguous buffer layout.
    KV cache is reserved separately via reserve_kv_cache(). Call that before
    materialize() to include KV buffers in the same contiguous allocation.
    After calling this function (and optionally reserve_kv_cache), call
    manager.materialize() to allocate the GPU buffer.

    NAMING CONVENTION:
      Tensor names follow the pattern: {prefix}.layers.{i}.mlp.experts.{ep|tp}.{w13|w2}_weight
      This naming MUST match what consumers (e.g., model loading code) expect to look up.
      Changing names here requires coordinating with all code that wraps these tensors as nn.Parameter.

    QKV TP BUFFER:
      The QKV_TP buffer is separate from QKV_full because TP reconfiguration requires
      copying q/k/v slices from the full QKV weight into this buffer. This avoids
      modifying the original full weight and enables efficient in-place operations.
    """
    _validate_v1_scope(num_fused_shared_experts, quant_name)
    _validate_moe_parallel_layout(
        num_experts=num_experts,
        intermediate_size=intermediate_size,
        num_heads=num_heads,
        ep_size=ep_size,
        tp_size=tp_size,
        dp_size=dp_size,
    )
    intra_node_method = resolve_intra_node_weight_transfer_method(
        intra_node_weight_transfer_method
    )

    is_fp8 = quant_name == "fp8"
    weight_dtype = torch.float8_e4m3fn if is_fp8 else torch.bfloat16
    elem_size = weight_dtype.itemsize
    ep_local_experts = num_experts // ep_size
    tp_inter = intermediate_size // tp_size

    # Shapes match the ParaS forward exactly. EP shards experts across ep_size
    # ranks; each rank holds num_experts/ep_size experts with the FULL
    # intermediate (paras_moe_block EP gathered view). TP holds ALL num_experts
    # with the intermediate sharded by tp_size (the get_view_as TP view). Bytes
    # are equal at G=1 and TP is G=ep_size/tp_size times larger for G>1; the
    # four-anchor layout sizes EP and TP independently.
    ep_w13_shape = (ep_local_experts, 2 * intermediate_size, hidden_size)
    ep_w2_shape = (ep_local_experts, hidden_size, intermediate_size)
    tp_w13_shape = (num_experts, 2 * tp_inter, hidden_size)
    tp_w2_shape = (num_experts, hidden_size, tp_inter)

    def _shape_bytes(shape):
        n = 1
        for d in shape:
            n *= d
        return n * elem_size

    # Expert weights are placed by the deferred four-anchor pass at materialize
    # time (like KV cache), not reserved in _reservation_order. Biases stay
    # replicated per rank (paras_moe_block: self._full_w{13,2}_bias).
    manager._paras_moe_pending = {
        "num_layers": num_layers,
        "prefix": prefix,
        "dtype": weight_dtype,
        "elem_size": elem_size,
        "ep_w13_shape": ep_w13_shape,
        "ep_w2_shape": ep_w2_shape,
        "tp_w13_shape": tp_w13_shape,
        "tp_w2_shape": tp_w2_shape,
        "ep_w13_bytes": _shape_bytes(ep_w13_shape),
        "ep_w2_bytes": _shape_bytes(ep_w2_shape),
        "tp_w13_bytes": _shape_bytes(tp_w13_shape),
        "tp_w2_bytes": _shape_bytes(tp_w2_shape),
    }
    # Only EP weights consume dedicated budget: TP weights overlap the EP cache
    # in the four-anchor run (never live together), so they add no footprint
    # beyond the balanced per-mode budget. This matches the old slot subtraction
    # (~Σwe), so KV token capacity does not regress at G=1.
    manager._deferred_weight_bytes = num_layers * (
        _shape_bytes(ep_w13_shape) + _shape_bytes(ep_w2_shape)
    )

    for i in range(num_layers):
        lp = f"{prefix}.layers.{i}"

        # FP8 weight scales: see docs/paras/paras_fp8_support.md
        # (replicated full-buffer + EP/TP slice Parameter views, not UMM).

        # -- Attention weights ---------------------------------------------
        qkv_out = num_heads * head_dim + 2 * num_kv_heads * head_dim
        manager.reserve(
            f"{lp}.self_attn.qkv_proj.weight",
            (qkv_out, hidden_size),
            weight_dtype,
        )
        manager.reserve(
            f"{lp}.self_attn.o_proj.weight",
            (hidden_size, num_heads * head_dim),
            weight_dtype,
        )

        # -- QKV / O TP buffers -------------------------------------------
        # GQA replication: when tp_size > num_kv_heads, each rank holds 1 KV
        # head (replicated across rank groups). Mirrors model_config.get_num_kv_heads
        # (configs/model_config.py L497) and QKVParallelLinear.__init__ (L818-824).
        tp_q_size = (num_heads // tp_size) * head_dim
        tp_kv_size = max(1, num_kv_heads // tp_size) * head_dim
        tp_attn_out = (num_heads // tp_size) * head_dim
        manager.reserve(
            f"{lp}.self_attn.qkv_proj.tp_weight",
            (tp_q_size + 2 * tp_kv_size, hidden_size),
            weight_dtype,
        )
        manager.reserve(
            f"{lp}.self_attn.o_proj.tp_weight",
            (hidden_size, tp_attn_out),
            weight_dtype,
        )

    if intra_node_method is IntraNodeWeightTransferMethod.NCCL:
        staging_dtype = torch.float8_e4m3fn if is_fp8 else torch.bfloat16
        staging_experts = num_experts // ep_size
        w13_shape = (staging_experts, 2 * intermediate_size, hidden_size)
        w2_shape = (staging_experts, hidden_size, intermediate_size)
        manager.reserve("staging.w13_pre_permute", w13_shape, staging_dtype)
        manager.reserve("staging.w2_pre_permute", w2_shape, staging_dtype)


# ---------------------------------------------------------------------------
# GPT-OSS MoE layout planning
# ---------------------------------------------------------------------------

def plan_gpt_oss_moe_layout(
    manager: ParaSMemoryManager,
    *,
    num_layers: int,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    ep_size: int,
    tp_size: int,
    dp_size: int,
    moe_tp_size: int,
    quant_name: Optional[str] = None,
    fp8_block_size: Optional[int] = None,
    num_fused_shared_experts: int = 0,
    intra_node_weight_transfer_method: str = (
        IntraNodeWeightTransferMethod.PEER_ACCESS.value
    ),
    prefix: str = "model",
) -> None:
    """Reserve all weight tensors for a GPT-OSS sparse-MoE model.

    GPT-OSS shares the Qwen3-MoE layout exactly for the tensors ParaS
    manages: w13/w2 expert weights (four-anchor EP<->TP layout), QKV/O
    attention projections, FP8 weight scales, and the pre-permute buffers
    used by the NCCL fallback. Both
    models are pure sparse MoE with no shared experts and identical
    attention projection geometry.

    Tensors that differ between the two models (GPT-OSS has biases on
    expert weights and the router; Qwen3 does not) are intentionally
    excluded from the ParaS layout in both cases.  ParaS only manages
    tensors that switch between EP and TP modes; biases, norms, and
    other model-static parameters live in regular PyTorch param storage
    allocated by FusedMoE / ReplicatedLinear / RMSNorm directly.  MXFP4
    is a checkpoint format only -- it is decompressed at load time and
    never appears in the ParaS buffer.

    Delegates to ``plan_qwen_moe_layout``.  If GPT-OSS ever needs a
    distinct staging layout (different intermediate_size handling,
    MXFP4-specific buffers, expert-bias switching), specialise here
    instead of polluting the Qwen path.
    """
    plan_qwen_moe_layout(
        manager,
        num_layers=num_layers,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        ep_size=ep_size,
        tp_size=tp_size,
        dp_size=dp_size,
        moe_tp_size=moe_tp_size,
        quant_name=quant_name,
        fp8_block_size=fp8_block_size,
        num_fused_shared_experts=num_fused_shared_experts,
        intra_node_weight_transfer_method=intra_node_weight_transfer_method,
        prefix=prefix,
    )


# ---------------------------------------------------------------------------
# MoE alias creation (call after materialize)
# ---------------------------------------------------------------------------

def create_paras_moe_aliases(
    manager: ParaSMemoryManager,
    num_layers: int,
    prefix: str = "model",
) -> None:
    """Call-order compatibility shim (kept so model files need no change).

    ep_experts/tp_experts entries are now primaries created by the four-anchor
    pass inside materialize(); this validates they exist rather than aliasing
    the removed N+1 slots.
    """
    for i in range(num_layers):
        for role in ("ep_experts", "tp_experts"):
            for w in ("w13_weight", "w2_weight"):
                name = f"{prefix}.layers.{i}.mlp.{role}.{w}"
                if name not in manager._entries:
                    raise KeyError(
                        f"create_paras_moe_aliases: missing '{name}'. "
                        "plan_qwen_moe_layout + materialize() must run first."
                    )
