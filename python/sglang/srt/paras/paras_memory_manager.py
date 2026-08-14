"""
ParaSMemoryManager: Static contiguous weight buffer for ParaS EP↔TP switching.
V1 scope: Qwen sparse-MoE (no shared experts, no dense MLPs).
Supported dtypes: BF16/FP16 (unquantized) and FP8.

LIFECYCLE & DESIGN:
  The manager pre-plans a single contiguous uint8 buffer that holds all weight tensors
  needed for Expert Parallelism (EP) ↔ Tensor Parallelism (TP) switching. This design
  avoids repeated allocations and fragmentation during dynamic reconfiguration.

  1. plan_qwen_moe_layout(manager, ...)  — reserves tensor slots (name, shape, dtype)
  2. manager.materialize()                — computes aligned offsets, allocates one GPU buffer
  3. manager.get_view("name")             — returns typed, shaped view for module to wrap as nn.Parameter

MEMORY LAYOUT EXAMPLE (1 layer, BF16, 8 experts, ep_size=2, tp_size=4):
  [EP_w13 | EP_w2 | TP_w13 | TP_w2 | QKV_full | O_proj | QKV_TP_buf | ...]
   ^-- all in one contiguous torch.uint8 buffer, 256-byte aligned for GPU access

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
    WeightTransferMethod,
    resolve_weight_transfer_method,
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
    """UMM-owned EP/TP KV cache capacity plan.

    SWA fields are zero / empty for pure-MHA plans. ``layer_specs`` is set
    only by the SWA planner for downstream :meth:`reserve_kv_cache`.
    """

    available_gpu_memory_bytes: int
    total_gpu_memory_bytes: int
    dynamic_reserve_bytes: int
    umm_budget_bytes: int
    weights_only_bytes: int
    non_umm_static_bytes: int
    kv_budget_bytes: int

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

    def _compute_kv_budget_bytes(
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
        kv_budget_bytes = max(
            0,
            umm_budget_bytes - self.weights_only_bytes - non_umm_static_bytes,
        )

        return (
            avail_now_bytes,
            total_gpu_bytes,
            dynamic_reserve_bytes,
            umm_budget_bytes,
            kv_budget_bytes,
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
        ``get_available_gpu_memory``, computes the UMM and KV byte budgets,
        derives EP and TP per-token caps, populates ``self.ep_max_kv_tokens``
        and ``self.tp_max_kv_tokens``, logs at INFO level, and returns the
        plan including the resolved ``kv_dtype`` for downstream
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
            kv_budget_bytes,
            non_umm_static_bytes,
            avail_now_gib,
        ) = self._compute_kv_budget_bytes(config)

        num_layers = config.num_hidden_layers
        num_kv_heads = config.num_key_value_heads

        ep_kv_heads = num_kv_heads
        tp_kv_heads = max(1, num_kv_heads // tp_size)
        # (num_layers + 1) reserves one layer's K+V for the four-anchor cache
        # tail overhead (the max(ct) anchor in _place_paras_run). Without it the
        # materialized run exceeds kv_budget by ~one layer.
        ep_cell_bytes = ep_kv_heads * head_dim * (num_layers + 1) * 2 * elem_size
        tp_cell_bytes = tp_kv_heads * head_dim * (num_layers + 1) * 2 * elem_size
        ep_max_tokens = max(1, int(kv_budget_bytes // ep_cell_bytes))
        tp_max_tokens = max(1, int(kv_budget_bytes // tp_cell_bytes))

        self.ep_max_kv_tokens = ep_max_tokens
        self.tp_max_kv_tokens = tp_max_tokens

        logger.info(
            f"ParaS KV budget: avail_now={avail_now_gib:.3f}GiB  "
            f"total={total_gpu_bytes / (1 << 30):.3f}GiB  "
            f"dynamic_reserve={dynamic_reserve_bytes / (1 << 30):.3f}GiB  "
            f"umm_budget={umm_budget_bytes / (1 << 30):.3f}GiB  "
            f"weights_only={self.weights_only_bytes / (1 << 30):.3f}GiB  "
            f"non_umm_static={non_umm_static_bytes / (1 << 30):.3f}GiB  "
            f"kv_budget={kv_budget_bytes / (1 << 30):.3f}GiB  "
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
            weights_only_bytes=self.weights_only_bytes,
            non_umm_static_bytes=non_umm_static_bytes,
            kv_budget_bytes=kv_budget_bytes,
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
            kv_budget_bytes,
            non_umm_static_bytes,
            avail_now_gib,
        ) = self._compute_kv_budget_bytes(config)

        num_layers = config.num_hidden_layers
        num_kv_heads = config.num_key_value_heads

        layer_types = getattr(config, "layer_types", None) or (
            ["full_attention"] * num_layers
        )
        n_full = sum(1 for t in layer_types if t == "full_attention")
        n_swa = sum(1 for t in layer_types if t == "sliding_attention")

        ep_kv_heads = num_kv_heads
        tp_kv_heads = max(1, num_kv_heads // tp_size)
        cell_bytes = num_kv_heads * head_dim * 2 * elem_size

        def _solve_tokens(kv_budget: int) -> Tuple[int, int]:
            tt = max(1, int(kv_budget // cell_bytes))
            if n_swa > 0:
                swa_ratio = getattr(self.server_args, "swa_full_tokens_ratio", 0.5)
                return plan_hybrid_kv_budget(tt, n_full, n_swa, swa_ratio)
            return max(1, tt // num_layers), 0

        # Two-pass: pass 1 estimates full_max_tokens, pass 2 subtracts the
        # four-anchor cache tail overhead (~one max-layer of K+V bytes) and
        # re-solves. Single iteration converges because the overhead << kv_budget.
        full_max_tokens_pass1, _ = _solve_tokens(kv_budget_bytes)
        overlap_gap_bytes = full_max_tokens_pass1 * cell_bytes
        kv_budget_bytes = max(0, kv_budget_bytes - overlap_gap_bytes)
        full_max_tokens, swa_max_tokens = _solve_tokens(kv_budget_bytes)

        ep_max_tokens = full_max_tokens
        tp_max_tokens = full_max_tokens * tp_size
        ep_max_tokens_swa = swa_max_tokens
        tp_max_tokens_swa = swa_max_tokens * tp_size

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

        ep_cell_bytes = ep_kv_heads * head_dim * num_layers * 2 * elem_size
        tp_cell_bytes = tp_kv_heads * head_dim * num_layers * 2 * elem_size

        logger.info(
            f"ParaS SWA KV budget: avail_now={avail_now_gib:.3f}GiB  "
            f"total={total_gpu_bytes / (1 << 30):.3f}GiB  "
            f"dynamic_reserve={dynamic_reserve_bytes / (1 << 30):.3f}GiB  "
            f"umm_budget={umm_budget_bytes / (1 << 30):.3f}GiB  "
            f"weights_only={self.weights_only_bytes / (1 << 30):.3f}GiB  "
            f"non_umm_static={non_umm_static_bytes / (1 << 30):.3f}GiB  "
            f"overlap_gap={overlap_gap_bytes / (1 << 30):.3f}GiB  "
            f"kv_budget={kv_budget_bytes / (1 << 30):.3f}GiB  "
            f"full_max_tokens={full_max_tokens}  swa_max_tokens={swa_max_tokens}  "
            f"layers={num_layers} (full={n_full} swa={n_swa})"
        )

        return ParaSKVCapacityPlan(
            available_gpu_memory_bytes=avail_now_bytes,
            total_gpu_memory_bytes=total_gpu_bytes,
            dynamic_reserve_bytes=dynamic_reserve_bytes,
            umm_budget_bytes=umm_budget_bytes,
            weights_only_bytes=self.weights_only_bytes,
            non_umm_static_bytes=non_umm_static_bytes,
            kv_budget_bytes=kv_budget_bytes,
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
        (asserted below): the address math stays safe without it, but the kv
        budget reserves only one ``max(ct)`` tail layer, which a ``ct > ce`` layer
        would overflow. Under that precondition the anchor reduces to ``max(ct)``.
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

        if moe is not None:
            we_slab = au(moe["ep_w13_bytes"]) + au(moe["ep_w2_bytes"])
            wt_slab = au(moe["tp_w13_bytes"]) + au(moe["tp_w2_bytes"])
            we = [we_slab] * num_layers
            wt = [wt_slab] * num_layers
        else:
            we = [0] * num_layers
            wt = [0] * num_layers

        if kv is not None:
            ce = [au(b) * 2 for b in kv["layer_ep_bytes"]]
            ct = [au(b) * 2 for b in kv["layer_tp_bytes"]]
        else:
            ce = [0] * num_layers
            ct = [0] * num_layers

        sum_we, sum_wt, sum_ce, sum_ct = sum(we), sum(wt), sum(ce), sum(ct)

        # The kv budget (reserve_kv_cache) reserves exactly one tail layer
        # (num_layers + 1 cells) for the cache tail, sized as max(ct). The general
        # anchor below equals max(ct) only when ct[i] <= ce[i]; a ct[i] > ce[i]
        # layer makes the suffix positive and the anchor exceed max(ct), so the
        # materialized run would overflow the reserved kv budget. Enforce the
        # precondition here. Real configs always satisfy it (num_kv_heads divides
        # tp_size or GQA-replicates, page_size >= 1, so ct == ce for uniform
        # attention and ct <= ce for hybrid).
        assert all(ct[i] <= ce[i] for i in range(num_layers)), (
            "four-anchor cache requires ct[i] <= ce[i] (per-layer TP cache must "
            "not exceed EP cache); the (num_layers+1) kv budget reserves only one "
            f"max(ct) tail layer, which a ct>ce layer would overflow. ce={ce} ct={ct}"
        )

        # Cache tail anchor: gap between EP and TP cache bases that keeps every
        # layer's EP and TP cache disjoint for any layer order. Under the
        # asserted ct[i] <= ce[i] the suffix is non-positive and this reduces to
        # max(ct); the general form is kept so the address math stays correct if
        # the precondition is ever relaxed.
        anchor, suffix = 0, 0
        for i in range(num_layers - 1, -1, -1):
            anchor = max(anchor, ct[i] + suffix)
            suffix += ct[i] - ce[i]
        anchor = au(anchor)

        w_end = P + sum_we
        tp_w_end = P + we[0] + sum_wt
        PAD = au(max(0, tp_w_end - w_end - sum_ce + sum_ct - anchor))
        EP_end = w_end + PAD + sum_ce
        tc_end = EP_end + anchor
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

    @property
    def weights_only_bytes(self) -> int:
        """Total weight bytes NOT including KV cache entries (for KV sizing).

        Includes deferred expert weights, which are placed by the four-anchor
        pass at materialize time and so never enter ``_reservation_order``.
        """
        return sum(
            self._entries[n].size_bytes for n in self._reservation_order
        ) + getattr(self, "_deferred_weight_bytes", 0)

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
    configure_method: str = WeightTransferMethod.DIRECT.value,
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
    transfer_method = resolve_weight_transfer_method(configure_method)
    if transfer_method is WeightTransferMethod.NCCL and dp_size != 1:
        raise ValueError(
            "The ParaS NCCL weight transfer supports only dp_size=1; "
            "use method='direct' for DP x TP configurations"
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

    if transfer_method is WeightTransferMethod.NCCL:
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
    configure_method: str = WeightTransferMethod.DIRECT.value,
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
        configure_method=configure_method,
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
