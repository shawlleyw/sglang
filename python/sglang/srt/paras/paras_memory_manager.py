"""Shared BF16 weight, KV, and backend workspace buffer for ParaS.

Qwen3 MoE and GPT-OSS use fixed EP/TP views into one allocation. The planner
keeps each transfer destination disjoint from live source layers. Full and
sliding-window KV pools retain separate capacities within this layout.
"""

import json
import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from sglang.srt.paras.mode import ParaSMode
from sglang.srt.paras.workspace import ModeWorkspaces, WorkspaceRequirement

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from sglang.srt.paras.layers.utils import LayerCacheSpec
    from sglang.srt.paras.unified_layout import UnifiedLayout
    from sglang.srt.paras.workspace import MoEWorkspace
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
class KVCacheReservation:
    """Per-layer K/V shapes and sizes retained until buffer materialization."""

    num_layers: int
    prefix: str
    layer_ep_bytes: List[int]
    layer_tp_bytes: List[int]
    layer_ep_shapes: List[Tuple[int, ...]]
    layer_tp_shapes: List[Tuple[int, ...]]
    kv_dtype: torch.dtype


@dataclass
class UnifiedModeSpec:
    """Weights and operator scratch for one mode of the unified buffer."""

    workspaces: ModeWorkspaces
    weight_names: List[List[str]] = field(default_factory=list)
    # Aligned bytes per layer, populated when weights are registered.
    weight_bytes: int = 0


@dataclass
class UnifiedLayoutSpec:
    """Model dimensions and EP/TP requirements used to build UnifiedLayout."""

    num_layers: int
    prefix: str
    tp_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    ep: UnifiedModeSpec
    tp: UnifiedModeSpec

    def for_mode(self, mode: ParaSMode) -> UnifiedModeSpec:
        if mode == ParaSMode.EP:
            return self.ep
        if mode == ParaSMode.TP:
            return self.tp
        raise ValueError(mode)


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
    if quant_name is not None:
        raise ValueError(
            f"ParaS V1 does not support quantization format '{quant_name}'. "
            "Only unquantized BF16 weights are supported."
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
        context_len: Optional[int] = None,
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
        self.context_len = context_len
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
        self._unified_spec: Optional[UnifiedLayoutSpec] = None
        self._unified_layout: Optional["UnifiedLayout"] = None
        self._paras_kv_pending: Optional[KVCacheReservation] = None

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

    def get_moe_workspace(self, mode: ParaSMode, shapes, dtype, device):
        return self._get_workspace(mode, "moe", shapes, dtype, device)

    def bind_moe_workspace(self, mode: ParaSMode) -> Optional["MoEWorkspace"]:
        """Bind an expert runner to its fixed region without initializing it."""
        from sglang.srt.paras.workspace import MoEWorkspace

        requirement = self._unified_spec.for_mode(mode).workspaces.moe
        if requirement.size_bytes is None:
            return None
        (buffer,) = self.get_moe_workspace(
            mode, [(requirement.reserved_bytes,)], torch.uint8, self.device
        )
        return MoEWorkspace(mode, buffer)

    def get_attention_workspace(self, backend, mode: ParaSMode, shapes, dtype, device):
        """Return typed scratch views, or None when the backend owns its storage."""
        requirement = self._unified_spec.for_mode(mode).workspaces.attention
        if requirement.size_bytes is None:
            return None
        if requirement.backend != backend:
            raise RuntimeError(
                f"Attention workspace planned for {requirement.backend}, got {backend}"
            )
        return self._get_workspace(mode, "attention", shapes, dtype, device)

    def get_attention_workspace_buffer(
        self, backend, mode: ParaSMode
    ) -> Optional[torch.Tensor]:
        """Return the backend's complete numerical workspace as one uint8 buffer.

        The manager owns its planned capacity; callers need neither tensor
        shapes nor knowledge of the multi-view workspace API.
        """
        requirement = self._unified_spec.for_mode(mode).workspaces.attention
        if requirement.size_bytes is None:
            return None
        (workspace_buffer,) = self.get_attention_workspace(
            backend, mode, [(requirement.size_bytes,)], torch.uint8, self.device
        )
        return workspace_buffer

    def initialize_attention_workspace(self, mode: ParaSMode):
        """Call only after migration; the target region may hold source weights."""
        requirement = self._unified_spec.for_mode(mode).workspaces.attention
        workspace_buffer = self.get_attention_workspace_buffer(
            requirement.backend, mode
        )
        if workspace_buffer is not None:
            workspace_buffer.zero_()

    def _get_workspace(self, mode: ParaSMode, kind, shapes, dtype, device):
        """Typed, non-owning scratch views at the mode's fixed endpoint."""
        from sglang.srt.paras.workspace import workspace_views

        if self._buffer is None:
            raise RuntimeError("Workspace requested before materialization")
        offset, _ = self._unified_layout.workspace(mode)
        relative_offset, capacity = self._unified_spec.for_mode(mode).workspaces.region(
            kind
        )
        offset += relative_offset
        return workspace_views(
            self._buffer[offset : offset + capacity],
            shapes,
            dtype,
            device,
            label=f"ParaS {mode.value} {kind} workspace",
        )

    def _materialize_unified(self):
        layout, spec = self._unified_layout, self._unified_spec
        pending = self._paras_kv_pending
        if layout is None or not self._kv_reserved or pending is None:
            raise RuntimeError(
                "Unified layout requires capacity planning and KV reservation"
            )
        if (layout.ep_cache.full_tokens, layout.tp_cache.full_tokens) != (
            self.ep_max_kv_tokens,
            self.tp_max_kv_tokens,
        ):
            raise ValueError("KV reservation differs from the unified capacity plan")
        dtype = pending.kv_dtype
        for mode in (ParaSMode.EP, ParaSMode.TP):
            shapes = (
                pending.layer_ep_shapes
                if mode == ParaSMode.EP
                else pending.layer_tp_shapes
            )
            sizes = (
                pending.layer_ep_bytes
                if mode == ParaSMode.EP
                else pending.layer_tp_bytes
            )
            for i, names in enumerate(spec.for_mode(mode).weight_names):
                offset = layout.weight_offset(mode, i)
                for name in names:
                    entry = self._entries[name]
                    entry.offset_bytes = offset
                    offset += self._align_up(entry.size_bytes, self.ALIGNMENT)
                shape, size = shapes[i], sizes[i]
                cache = layout.ep_cache if mode == ParaSMode.EP else layout.tp_cache
                kv_slot_bytes = cache.layer_bytes[i] // 2
                assert size <= kv_slot_bytes, "KV reservation exceeds capacity plan"
                for side, displacement in (("k", 0), ("v", kv_slot_bytes)):
                    name = f"{spec.prefix}.layers.{i}.kv.{mode.value}.{side}"
                    entry = LayoutEntry(
                        name,
                        shape,
                        dtype,
                        math.prod(shape),
                        dtype.itemsize,
                        size,
                        layout.cache_offset(mode, i) + displacement,
                    )
                    self._entries[name] = entry
                    if mode == ParaSMode.EP:
                        self._entries[f"{spec.prefix}.layers.{i}.kv.{side}"] = entry
        self._total_bytes = layout.budget
        self._buffer = torch.empty(layout.budget, dtype=torch.uint8, device=self.device)
        self._buffer_start = self._buffer.data_ptr()
        self._buffer_end = self._buffer_start + layout.budget
        self._materialized = True
        return layout.budget

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

        # Save metadata for _create_kv_layout (called from materialize).
        self._paras_kv_pending = KVCacheReservation(
            num_layers=num_layers,
            prefix=prefix,
            layer_ep_bytes=layer_ep_bytes,
            layer_tp_bytes=layer_tp_bytes,
            layer_ep_shapes=layer_ep_shapes,
            layer_tp_shapes=layer_tp_shapes,
            kv_dtype=kv_dtype,
        )

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

    def plan_kv_capacity(
        self,
        *,
        config,
        tp_size: int,
        head_dim: int,
    ) -> ParaSKVCapacityPlan:
        """Plan weights, backend scratch, and per-layer full/SWA KV capacities."""
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

        from sglang.srt.paras.unified_layout import plan_unified_layout
        from sglang.srt.paras.workspace import attention_workspace_requirements

        spec = self._unified_spec
        attention = attention_workspace_requirements(
            self.server_args, config, tp_size, head_dim, context_len=self.context_len
        )
        for mode, requirement in zip((ParaSMode.EP, ParaSMode.TP), attention):
            spec.for_mode(mode).workspaces = ModeWorkspaces(
                spec.for_mode(mode).workspaces.moe, requirement
            )
        ep_row = num_kv_heads * head_dim * 2 * elem_size
        tp_row = max(1, num_kv_heads // tp_size) * head_dim * 2 * elem_size
        layer_types = (
            getattr(config, "layer_types", None) or ["full_attention"] * num_layers
        )
        swa_ratio = self.server_args.swa_full_tokens_ratio
        ratios = tuple(
            swa_ratio if kind == "sliding_attention" else 1.0 for kind in layer_types
        )
        layout = plan_unified_layout(
            num_layers=num_layers,
            budget=umm_budget_bytes - non_umm_static_bytes,
            ep_weight_bytes=spec.ep.weight_bytes,
            tp_weight_bytes=spec.tp.weight_bytes,
            ep_workspace_bytes=spec.ep.workspaces.size_bytes,
            tp_workspace_bytes=spec.tp.workspaces.size_bytes,
            ep_kv_row_bytes=ep_row,
            tp_kv_row_bytes=tp_row,
            page_size=self.server_args.page_size,
            layer_token_ratios=ratios,
        )
        self._unified_layout = layout
        self.ep_max_kv_tokens, self.tp_max_kv_tokens = (
            layout.ep_cache.full_tokens,
            layout.tp_cache.full_tokens,
        )
        swa_layer = next(
            (i for i, kind in enumerate(layer_types) if kind == "sliding_attention"),
            None,
        )
        self.ep_max_kv_tokens_swa = (
            layout.ep_cache.layer_tokens[swa_layer] if swa_layer is not None else 0
        )
        self.tp_max_kv_tokens_swa = (
            layout.tp_cache.layer_tokens[swa_layer] if swa_layer is not None else 0
        )
        layer_specs = classify_layers_from_config(
            config,
            tp_size=tp_size,
            ep_tokens_full=self.ep_max_kv_tokens,
            tp_tokens_full=self.tp_max_kv_tokens,
            ep_tokens_swa=self.ep_max_kv_tokens_swa,
            tp_tokens_swa=self.tp_max_kv_tokens_swa,
        )
        logger.info("ParaS unified weights/KV/workspace: %s", layout)
        for mode in (ParaSMode.EP, ParaSMode.TP):
            requirements = spec.for_mode(mode).workspaces
            logger.info(
                "ParaS %s workspace: %s; unused padding=%d bytes",
                mode.value,
                requirements,
                layout.workspace(mode)[1] - requirements.size_bytes,
            )
        return ParaSKVCapacityPlan(
            available_gpu_memory_bytes=avail_now_bytes,
            total_gpu_memory_bytes=total_gpu_bytes,
            dynamic_reserve_bytes=dynamic_reserve_bytes,
            umm_budget_bytes=layout.budget,
            weights_only_bytes=self.weights_only_bytes,
            non_umm_static_bytes=non_umm_static_bytes,
            kv_budget_bytes=layout.ep_cache.total_bytes,
            kv_dtype=kv_dtype,
            ep_max_tokens=layout.ep_cache.full_tokens,
            tp_max_tokens=layout.tp_cache.full_tokens,
            ep_cell_bytes=num_layers * ep_row,
            tp_cell_bytes=num_layers * tp_row,
            ep_kv_heads=num_kv_heads,
            tp_kv_heads=max(1, num_kv_heads // tp_size),
            full_layers=layer_types.count("full_attention"),
            swa_layers=layer_types.count("sliding_attention"),
            ep_max_tokens_swa=self.ep_max_kv_tokens_swa,
            tp_max_tokens_swa=self.tp_max_kv_tokens_swa,
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
        if self._unified_spec is not None:
            return self._materialize_unified()
        offset = 0
        for name in self._reservation_order:
            entry = self._entries[name]
            entry.offset_bytes = self._align_up(offset, self.ALIGNMENT)
            offset = entry.offset_bytes + entry.size_bytes

        pending = getattr(self, "_paras_kv_pending", None)
        if pending is not None:
            offset = self._create_kv_layout(
                offset,
                num_layers=pending.num_layers,
                prefix=pending.prefix,
                layer_ep_bytes=pending.layer_ep_bytes,
                layer_tp_bytes=pending.layer_tp_bytes,
                layer_ep_shapes=pending.layer_ep_shapes,
                layer_tp_shapes=pending.layer_tp_shapes,
                kv_dtype=pending.kv_dtype,
            )

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

    # ----- KV layout creation (called from materialize) -------------------

    def _create_kv_layout(
        self,
        offset: int,
        *,
        num_layers: int,
        prefix: str,
        layer_ep_bytes: List[int],
        layer_tp_bytes: List[int],
        layer_ep_shapes: List[Tuple[int, ...]],
        layer_tp_shapes: List[Tuple[int, ...]],
        kv_dtype: torch.dtype,
    ) -> int:
        """Create per-layer TP and EP LayoutEntry objects at computed offsets.

        Returns the byte offset past the end of the V region.
        """
        if num_layers == 0:
            return offset

        elem_size = (
            kv_dtype.itemsize
            if hasattr(kv_dtype, "itemsize")
            else torch.tensor([], dtype=kv_dtype).element_size()
        )
        tp_prefix = 0
        ep_prefix = 0
        overlap_gap = 0
        for tp_bytes, ep_bytes in zip(layer_tp_bytes, layer_ep_bytes):
            overlap_gap = max(overlap_gap, tp_prefix + tp_bytes - ep_prefix)
            tp_prefix += tp_bytes
            ep_prefix += ep_bytes

        kv_region_bytes = max(sum(layer_tp_bytes), overlap_gap + sum(layer_ep_bytes))

        k_region_start = self._align_up(offset, self.ALIGNMENT)
        v_region_start = self._align_up(
            k_region_start + kv_region_bytes, self.ALIGNMENT
        )

        for side, region_start in [("k", k_region_start), ("v", v_region_start)]:
            tp_prefix = 0
            ep_prefix = 0
            for i in range(num_layers):
                ep_shape = layer_ep_shapes[i]
                tp_shape = layer_tp_shapes[i]
                ep_bytes = layer_ep_bytes[i]
                tp_bytes = layer_tp_bytes[i]
                ep_numel = ep_shape[0] * ep_shape[1] * ep_shape[2]
                tp_numel = tp_shape[0] * tp_shape[1] * tp_shape[2]

                tp_offset = region_start + tp_prefix
                ep_offset = region_start + overlap_gap + ep_prefix

                ep_entry = LayoutEntry(
                    name=f"{prefix}.layers.{i}.kv.ep.{side}",
                    shape=ep_shape,
                    dtype=kv_dtype,
                    numel=ep_numel,
                    element_size=elem_size,
                    size_bytes=ep_bytes,
                    offset_bytes=ep_offset,
                )
                self._entries[f"{prefix}.layers.{i}.kv.ep.{side}"] = ep_entry
                self._entries[f"{prefix}.layers.{i}.kv.{side}"] = ep_entry

                self._entries[f"{prefix}.layers.{i}.kv.tp.{side}"] = LayoutEntry(
                    name=f"{prefix}.layers.{i}.kv.tp.{side}",
                    shape=tp_shape,
                    dtype=kv_dtype,
                    numel=tp_numel,
                    element_size=elem_size,
                    size_bytes=tp_bytes,
                    offset_bytes=tp_offset,
                )

                tp_prefix += tp_bytes
                ep_prefix += ep_bytes

        return v_region_start + kv_region_bytes

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
        self, name: str, shape: tuple, dtype: torch.dtype = None
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
        mode: ParaSMode,
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

            if mode == ParaSMode.EP:
                k_bufs.append(self.get_view(k_name))
                v_bufs.append(self.get_view(v_name))
            elif mode == ParaSMode.TP:
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
                raise ValueError(f"Expected a ParaSMode, got {mode!r}")

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
        """Total reserved bytes NOT including KV cache entries (for KV sizing)."""
        if self._unified_spec is not None:
            return self._unified_spec.num_layers * self._unified_spec.ep.weight_bytes
        return sum(
            self._entries[n].size_bytes
            for n in self._reservation_order
        )

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
    num_fused_shared_experts: int = 0,
    configure_method: str = "peer_access",
    prefix: str = "model",
    top_k: int = 8,
    with_bias: bool = False,
) -> None:
    """Reserve the shared BF16 EP/TP layout for Qwen and GPT-OSS."""
    assert quant_name is None, "ParaS unified layout requires unquantized BF16 weights"
    assert (
        configure_method == "peer_access"
    ), "ParaS unified layout requires peer_access"
    assert moe_tp_size == dp_size == 1, "ParaS requires MoE TP=1 and ParaS DP=1"
    assert (
        ep_size == tp_size and tp_size > 1
    ), "ParaS requires matching EP/TP groups > 1"
    if manager.server_args is not None:
        assert manager.server_args.moe_runner_backend in (
            "auto",
            "triton",
            "deep_gemm",
        ), "ParaS unified layout supports DeepGEMM/Triton EP and Triton TP"
    _validate_v1_scope(num_fused_shared_experts, quant_name)

    from sglang.srt.paras.unified_layout import align_up, bf16_moe_workspace_sizes
    from sglang.srt.utils import get_int_env_var

    if ep_size != tp_size or num_experts % ep_size or intermediate_size % tp_size:
        raise ValueError(
            "Unified workspace requires equal EP/TP groups and divisible experts"
        )
    if num_heads % tp_size or (
        tp_size % num_kv_heads if tp_size >= num_kv_heads else num_kv_heads % tp_size
    ):
        raise ValueError("Attention heads do not divide the configured TP group")
    from sglang.srt.layers.moe.utils import use_deep_gemm_bf16

    ep_backend = (
        "deep_gemm" if use_deep_gemm_bf16(ep_size, with_bias=with_bias) else "triton"
    )
    capacity = get_int_env_var("SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK", 128)
    ep_workspace_bytes, tp_workspace_bytes = bf16_moe_workspace_sizes(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        tp_size=tp_size,
        dispatch_capacity=capacity,
    )
    spec = UnifiedLayoutSpec(
        num_layers=num_layers,
        prefix=prefix,
        ep=UnifiedModeSpec(
            ModeWorkspaces(
                WorkspaceRequirement(ep_backend, ep_workspace_bytes),
                WorkspaceRequirement("unconfigured", None),
            )
        ),
        tp=UnifiedModeSpec(
            ModeWorkspaces(
                WorkspaceRequirement("triton", tp_workspace_bytes),
                WorkspaceRequirement("unconfigured", None),
            )
        ),
        tp_size=tp_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
    )
    for mode in (ParaSMode.EP, ParaSMode.TP):
        mode_spec = spec.for_mode(mode)
        experts = num_experts // ep_size if mode == ParaSMode.EP else num_experts
        inter = (
            intermediate_size if mode == ParaSMode.EP else intermediate_size // tp_size
        )
        q = (
            num_heads * head_dim
            if mode == ParaSMode.EP
            else num_heads // tp_size * head_dim
        )
        kv = (
            num_kv_heads * head_dim
            if mode == ParaSMode.EP
            else max(1, num_kv_heads // tp_size) * head_dim
        )
        for i in range(num_layers):
            lp = f"{prefix}.layers.{i}"
            suffix = "weight" if mode == ParaSMode.EP else "tp_weight"
            entries = [
                (
                    f"{lp}.mlp.{mode.value}_experts.w13_weight",
                    (experts, 2 * inter, hidden_size),
                ),
                (
                    f"{lp}.mlp.{mode.value}_experts.w2_weight",
                    (experts, hidden_size, inter),
                ),
                (f"{lp}.self_attn.qkv_proj.{suffix}", (q + 2 * kv, hidden_size)),
                (f"{lp}.self_attn.o_proj.{suffix}", (hidden_size, q)),
            ]
            names = []
            for name, shape in entries:
                manager.reserve(name, shape, torch.bfloat16)
                names.append(name)
            mode_spec.weight_names.append(names)
            if mode == ParaSMode.EP:
                for weight in ("w13", "w2"):
                    manager._entries[f"{lp}.mlp.experts.{weight}_weight"] = (
                        manager._entries[f"{lp}.mlp.ep_experts.{weight}_weight"]
                    )
        mode_spec.weight_bytes = sum(
            align_up(manager._entries[n].size_bytes) for n in mode_spec.weight_names[0]
        )
    manager._unified_spec = spec


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
    num_fused_shared_experts: int = 0,
    configure_method: str = "peer_access",
    prefix: str = "model",
    top_k: int = 8,
) -> None:
    """GPT-OSS uses the same weight geometry, with separately owned biases."""
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
        num_fused_shared_experts=num_fused_shared_experts,
        configure_method=configure_method,
        prefix=prefix,
        top_k=top_k,
        with_bias=True,
    )


# ---------------------------------------------------------------------------
# MoE alias creation (call after materialize)
# ---------------------------------------------------------------------------

def create_paras_moe_aliases(
    manager: ParaSMemoryManager,
    num_layers: int,
    prefix: str = "model",
) -> None:
    """
    Create ep_experts and tp_experts aliases for the N+1 slot layout.
    Call after materialize().

    ep_experts layer i → slot i+1 (same physical buffer as EP weights)
    tp_experts layer i → slot i   (one slot before EP, for fused transfer)
    """
    if manager._unified_spec is not None:
        return
    for i in range(num_layers):
        manager.alias(f"{prefix}.layers.{i}.mlp.ep_experts.w13_weight", f"paras.moe_slot.{i+1}.w13")
        manager.alias(f"{prefix}.layers.{i}.mlp.ep_experts.w2_weight", f"paras.moe_slot.{i+1}.w2")
        manager.alias(f"{prefix}.layers.{i}.mlp.tp_experts.w13_weight", f"paras.moe_slot.{i}.w13")
        manager.alias(f"{prefix}.layers.{i}.mlp.tp_experts.w2_weight", f"paras.moe_slot.{i}.w2")
