"""Shared BF16 weight, KV, and backend workspace buffer for ParaS.

Qwen3 MoE and GPT-OSS use fixed EP/TP views into one allocation. The planner
keeps each transfer destination disjoint from live source layers. Full and
sliding-window KV pools retain separate capacities within this layout.
"""

import json
import logging
import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from sglang.srt.paras.mode import ParaSMode
from sglang.srt.paras.unified_layout import MEMORY_ALIGNMENT, align_up
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
class UnifiedMemoryPlan:
    """Completed layout, tensor views, and per-layer cache capacities."""

    layout: "UnifiedLayout"
    entries: tuple[LayoutEntry, ...]
    kv_dtype: torch.dtype
    layer_specs: List["LayerCacheSpec"]


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


class ParaSMemoryManager:
    """Declare model tensors, plan their layout, then allocate one shared buffer."""

    ALIGNMENT: int = MEMORY_ALIGNMENT

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
        self._unified_spec: Optional[UnifiedLayoutSpec] = None
        self._unified_layout: Optional["UnifiedLayout"] = None
        self._plan: Optional[UnifiedMemoryPlan] = None

    # ----- reservation ----------------------------------------------------

    def reserve(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
    ) -> LayoutEntry:
        """Declare a weight tensor; planning assigns its offset later."""
        if self._materialized:
            raise RuntimeError("Cannot reserve after the buffer has been materialized.")
        if name in self._entries:
            raise ValueError(f"Duplicate reservation name: '{name}'")
        if dtype not in _SUPPORTED_DTYPES:
            raise ValueError(
                f"Unsupported dtype {dtype}. " f"Supported: {_SUPPORTED_DTYPES}"
            )

        numel = 1
        for d in shape:
            numel *= d
        elem_size = (
            dtype.itemsize
            if hasattr(dtype, "itemsize")
            else torch.tensor([], dtype=dtype).element_size()
        )
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
        return entry

    # ----- Backend workspace views ----------------------------------------

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

    def _plan_tensor_entries(self, layout, kv_dtype, page_size):
        """Assign weight offsets and derive KV views directly from the layout."""
        spec = self._unified_spec
        for mode in (ParaSMode.EP, ParaSMode.TP):
            cache = layout.ep_cache if mode == ParaSMode.EP else layout.tp_cache
            heads = spec.num_kv_heads
            if mode == ParaSMode.TP:
                heads = max(1, heads // spec.tp_size)
            for i, names in enumerate(spec.for_mode(mode).weight_names):
                offset = layout.weight_offset(mode, i)
                for name in names:
                    entry = self._entries[name]
                    entry.offset_bytes = offset
                    offset += align_up(entry.size_bytes)
                shape = (cache.layer_tokens[i] + page_size, heads, spec.head_dim)
                numel = math.prod(shape)
                size = numel * kv_dtype.itemsize
                kv_region_bytes = cache.layer_bytes[i] // 2
                assert size <= kv_region_bytes, "KV view exceeds planned capacity"
                for side, displacement in (("k", 0), ("v", kv_region_bytes)):
                    name = f"{spec.prefix}.layers.{i}.kv.{mode.value}.{side}"
                    entry = LayoutEntry(
                        name,
                        shape,
                        kv_dtype,
                        numel,
                        kv_dtype.itemsize,
                        size,
                        layout.cache_offset(mode, i) + displacement,
                    )
                    self._entries[name] = entry
                    if mode == ParaSMode.EP:
                        self._entries[f"{spec.prefix}.layers.{i}.kv.{side}"] = entry
        return tuple(replace(entry, name=name) for name, entry in self._entries.items())

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

    def _memory_budget(self, config) -> int:
        """Available bytes after native dynamic and unmanaged static allocations."""
        from sglang.srt.utils.common import get_available_gpu_memory

        total = torch.cuda.get_device_properties(self.gpu_id).total_memory
        available = int(
            get_available_gpu_memory(
                self.device,
                self.gpu_id,
                distributed=self.world_size > 1,
                cpu_group=self.cpu_group,
                empty_cache=True,
            )
            * (1 << 30)
        )
        dynamic_reserve = int(total * (1.0 - self.server_args.mem_fraction_static))
        return max(
            0, available - dynamic_reserve - self._compute_non_umm_static_bytes(config)
        )

    def plan_layout(self, config, *, budget: Optional[int] = None) -> UnifiedMemoryPlan:
        """Finish the weight, workspace, and KV plan before allocating storage.

        An explicit byte budget also permits planning without a CUDA device.
        """
        from sglang.srt.paras.layers.utils import classify_layers_from_config

        spec = self._unified_spec
        tp_size, head_dim = spec.tp_size, spec.head_dim
        num_layers, num_kv_heads = spec.num_layers, spec.num_kv_heads
        kv_dtype = self._resolve_kv_store_dtype()
        elem_size = kv_dtype.itemsize
        if budget is None:
            budget = self._memory_budget(config)

        from sglang.srt.paras.unified_layout import plan_unified_layout
        from sglang.srt.paras.workspace import attention_workspace_requirements

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
            budget=budget,
            ep_weight_bytes=spec.ep.weight_bytes,
            tp_weight_bytes=spec.tp.weight_bytes,
            ep_workspace_bytes=spec.ep.workspaces.size_bytes,
            tp_workspace_bytes=spec.tp.workspaces.size_bytes,
            ep_kv_row_bytes=ep_row,
            tp_kv_row_bytes=tp_row,
            page_size=self.server_args.page_size,
            layer_token_ratios=ratios,
        )
        swa_layer = next(
            (i for i, kind in enumerate(layer_types) if kind == "sliding_attention"),
            None,
        )
        layer_specs = classify_layers_from_config(
            config,
            tp_size=tp_size,
            ep_tokens_full=layout.ep_cache.full_tokens,
            tp_tokens_full=layout.tp_cache.full_tokens,
            ep_tokens_swa=(
                layout.ep_cache.layer_tokens[swa_layer] if swa_layer is not None else 0
            ),
            tp_tokens_swa=(
                layout.tp_cache.layer_tokens[swa_layer] if swa_layer is not None else 0
            ),
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
        self._plan = UnifiedMemoryPlan(
            layout=layout,
            entries=self._plan_tensor_entries(
                layout, kv_dtype, self.server_args.page_size
            ),
            kv_dtype=kv_dtype,
            layer_specs=layer_specs,
        )
        return self._plan

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
        if max_running_requests is not None:
            # Match the native request pool's configured cap, including the
            # shared backing allocation, not just scheduler admission. EP's
            # per-rank admission limit is applied separately below.
            ep_num_reqs = min(ep_num_reqs, max_running_requests)
            tp_num_reqs = min(tp_num_reqs, max_running_requests)

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
        return self._plan is not None

    # ----- materialization ------------------------------------------------

    def materialize(self, plan: UnifiedMemoryPlan) -> int:
        """Allocate the completed plan; all view offsets are already fixed."""
        self._plan = plan
        self._unified_layout = plan.layout
        self._entries = {entry.name: entry for entry in plan.entries}
        self.ep_max_kv_tokens = plan.layout.ep_cache.full_tokens
        self.tp_max_kv_tokens = plan.layout.tp_cache.full_tokens
        swa = next((s for s in plan.layer_specs if s.kind == "swa"), None)
        self.ep_max_kv_tokens_swa = swa.tokens_cap_ep if swa is not None else 0
        self.tp_max_kv_tokens_swa = swa.tokens_cap_tp if swa is not None else 0
        self._total_bytes = plan.layout.budget
        self._buffer = torch.empty(
            self._total_bytes, dtype=torch.uint8, device=self.device
        )
        self._buffer_start = self._buffer.data_ptr()
        self._buffer_end = self._buffer_start + self._total_bytes
        self._materialized = True
        return self._total_bytes

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
        prefix: str = "model",
        layer_ids: Optional[List[int]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Return the planned per-layer K/V views for this mode."""
        ids = layer_ids if layer_ids is not None else range(num_layers)
        keys, values = [], []
        for i in ids:
            name = f"{prefix}.layers.{i}.kv.{mode.value}"
            keys.append(self.get_view(f"{name}.k"))
            values.append(self.get_view(f"{name}.v"))
        return keys, values

    # ----- queries --------------------------------------------------------

    def is_managed(self, tensor: torch.Tensor) -> bool:
        """True if *tensor*'s data pointer falls within the managed buffer."""
        if not self._materialized:
            return False
        ptr = tensor.data_ptr()
        return self._buffer_start <= ptr < self._buffer_end

    def dump_layout(self) -> List[Dict]:
        """Planned tensor views and their lookup aliases."""
        return [entry.to_dict() for entry in self._entries.values()]

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


def reserve_model_weights(
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
    """Declare BF16 EP/TP weight tensors and native MoE workspace requirements."""
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

    from sglang.srt.paras.unified_layout import (
        align_up,
        bf16_moe_workspace_sizes,
        tp_moe_workspace_token_capacity,
    )
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
    args = manager.server_args
    graph_bs = ()
    if args is not None and not getattr(args, "disable_cuda_graph", False):
        graph_bs = (
            getattr(args, "paras_tp_cuda_graph_bs", None)
            or getattr(args, "cuda_graph_bs", None)
            or ()
        )
    tp_tokens = tp_moe_workspace_token_capacity(
        max_prefill_tokens=(
            getattr(args, "paras_tp_max_prefill_tokens", None)
            or getattr(args, "max_prefill_tokens", None)
        ),
        max_running_requests=getattr(args, "max_running_requests", None),
        chunked_prefill_size=getattr(args, "chunked_prefill_size", None),
        cuda_graph_bs=graph_bs,
        speculative_num_draft_tokens=getattr(
            args, "speculative_num_draft_tokens", None
        ),
    )
    ep_workspace_bytes, tp_workspace_bytes = bf16_moe_workspace_sizes(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        tp_size=tp_size,
        dispatch_capacity=capacity,
        tp_input_tokens=tp_tokens,
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
