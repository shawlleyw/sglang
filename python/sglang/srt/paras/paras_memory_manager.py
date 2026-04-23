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
from typing import Dict, List, Optional, Tuple

import torch

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

    kv_cache_dtype = getattr(server_args, "kv_cache_dtype", None)
    if kv_cache_dtype == "fp8":
        raise NotImplementedError(
            "ParaS + SWA + FP8-KV not supported in v1 "
            "(see docs/paras/swa_support.md §12). "
            "Disable FP8-KV or use a non-hybrid model."
        )

    speculative_algorithm = getattr(server_args, "speculative_algorithm", None)
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

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self._entries: Dict[str, LayoutEntry] = {}
        self._reservation_order: List[str] = []
        self._buffer: Optional[torch.Tensor] = None
        self._materialized: bool = False
        self._total_bytes: int = 0
        self._buffer_start: int = 0
        self._buffer_end: int = 0
        self.ep_max_kv_tokens: int = 0
        self.tp_max_kv_tokens: int = 0
        self._kv_reserved: bool = False
        

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
        page_size: int = 1,
        prefix: str = "model",
        layer_specs: Optional[list] = None,
    ) -> None:
        """
        Reserve KV cache using a contiguous buffer with per-layer offsets.

        Must be called AFTER plan_qwen_moe_layout() and BEFORE materialize().

        Layout (per K and V separately):
          - Total region = max_L + sum(L_i) bytes
          - TP view for layer i at offset prefix_i = sum(L_j for j<i)
          - EP view for layer i at offset max_L + prefix_i
          - L_i = (tokens_cap_ep_i + page) * num_kv_heads_i * head_dim_i * elem_size
          - max_L = max(L_i)

        EP and TP KV have same total bytes per layer (union layout):
          ep_tokens * ep_kv_heads * head_dim == tp_tokens * tp_kv_heads * head_dim
        so each region entry is stored once and callers use get_view_as()
        for TP access with a different shape.

        Actual LayoutEntry objects are created during materialize() so that
        offsets are computed relative to the end of the weight region.
        """
        if self._materialized:
            raise RuntimeError("Cannot reserve KV cache after materialize().")
        if self._kv_reserved:
            raise RuntimeError("KV cache already reserved.")

        self.ep_max_kv_tokens = ep_max_tokens
        self.tp_max_kv_tokens = tp_max_tokens
        self._layer_specs = layer_specs

        elem_size = (
            kv_dtype.itemsize
            if hasattr(kv_dtype, "itemsize")
            else torch.tensor([], dtype=kv_dtype).element_size()
        )

        if layer_specs is None:
            per_layer_tokens = ep_max_tokens + page_size
            per_layer_bytes = per_layer_tokens * num_kv_heads * head_dim * elem_size
            layer_ep_shapes = [
                (per_layer_tokens, num_kv_heads, head_dim)
            ] * num_layers
            layer_bytes = [per_layer_bytes] * num_layers
        else:
            layer_ep_shapes = [
                (s.tokens_cap_ep + page_size, s.num_kv_heads, s.head_dim)
                for s in layer_specs
            ]
            layer_bytes = [
                (s.tokens_cap_ep + page_size)
                * s.num_kv_heads
                * s.head_dim
                * elem_size
                for s in layer_specs
            ]

        # Save metadata for _create_kv_layout (called from materialize).
        self._paras_kv_pending = {
            "num_layers": num_layers,
            "prefix": prefix,
            "layer_bytes": layer_bytes,
            "layer_ep_shapes": layer_ep_shapes,
            "max_L": max(layer_bytes) if num_layers > 0 else 0,
            "kv_dtype": kv_dtype,
        }

        self._kv_reserved = True

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

        pending = getattr(self, "_paras_kv_pending", None)
        if pending is not None:
            offset = self._create_kv_layout(offset, **pending)

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
        layer_bytes: List[int],
        layer_ep_shapes: List[Tuple[int, ...]],
        max_L: int,
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
        kv_region_bytes = max_L + sum(layer_bytes)

        k_region_start = self._align_up(offset, self.ALIGNMENT)
        v_region_start = self._align_up(
            k_region_start + kv_region_bytes, self.ALIGNMENT
        )

        for side, region_start in [("k", k_region_start), ("v", v_region_start)]:
            prefix_bytes = 0
            for i in range(num_layers):
                li_bytes = layer_bytes[i]
                ep_shape = layer_ep_shapes[i]
                ep_numel = ep_shape[0] * ep_shape[1] * ep_shape[2]

                tp_offset = region_start + prefix_bytes
                ep_offset = region_start + max_L + prefix_bytes

                ep_entry = LayoutEntry(
                    name=f"{prefix}.layers.{i}.kv.ep.{side}",
                    shape=ep_shape,
                    dtype=kv_dtype,
                    numel=ep_numel,
                    element_size=elem_size,
                    size_bytes=li_bytes,
                    offset_bytes=ep_offset,
                )
                self._entries[f"{prefix}.layers.{i}.kv.ep.{side}"] = ep_entry
                self._entries[f"{prefix}.layers.{i}.kv.{side}"] = ep_entry

                self._entries[f"{prefix}.layers.{i}.kv.tp.{side}"] = LayoutEntry(
                    name=f"{prefix}.layers.{i}.kv.tp.{side}",
                    shape=ep_shape,
                    dtype=kv_dtype,
                    numel=ep_numel,
                    element_size=elem_size,
                    size_bytes=li_bytes,
                    offset_bytes=tp_offset,
                )

                prefix_bytes += li_bytes

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
        mode: str,
        tp_size: int = 1,
        page_size: int = 1,
        prefix: str = "model",
        layer_ids: Optional[List[int]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Return k_buffers and v_buffers for the KV pool in the given mode.

        EP mode: returns views with (ep_tokens + page, total_kv_heads, head_dim)
        TP mode: returns views with (tp_tokens + page, total_kv_heads//tp_size, head_dim)
                 using get_view_as to reinterpret the same bytes.

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
                    tp_heads = ep_heads // tp_size
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
        """Total reserved bytes NOT including KV cache entries (for KV sizing)."""
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
    fp8_block_size: Optional[int] = None,
    num_fused_shared_experts: int = 0,
    configure_method: str = "peer_access",
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

    is_fp8 = quant_name == "fp8"
    weight_dtype = torch.float8_e4m3fn if is_fp8 else torch.bfloat16
    ep_local_experts = num_experts // ep_size
    inter_per_partition = intermediate_size // moe_tp_size
    w13_shape = (ep_local_experts, 2 * inter_per_partition, hidden_size)
    w2_shape = (ep_local_experts, hidden_size, inter_per_partition)

    for slot in range(num_layers + 1):
        manager.reserve(f"paras.moe_slot.{slot}.w13", w13_shape, weight_dtype)
        manager.reserve(f"paras.moe_slot.{slot}.w2", w2_shape, weight_dtype)

    # Create 'experts' aliases for create_weights() / weight loading compatibility.
    # These point to the same LayoutEntry objects as slot i+1 — no copy, no duplication.
    # materialize() processes _reservation_order, so aliases won't be double-processed.
    for i in range(num_layers):
        manager._entries[f"{prefix}.layers.{i}.mlp.experts.w13_weight"] = manager._entries[f"paras.moe_slot.{i+1}.w13"]
        manager._entries[f"{prefix}.layers.{i}.mlp.experts.w2_weight"] = manager._entries[f"paras.moe_slot.{i+1}.w2"]

    for i in range(num_layers):
        lp = f"{prefix}.layers.{i}"

        # -- FP8 scale tensors (if applicable) -----------------------------
        if is_fp8:
            elp = f"{lp}.mlp.experts"
            if fp8_block_size is not None and fp8_block_size > 0:
                def _ceil(a: int, b: int) -> int:
                    return (a + b - 1) // b
                w13_scale_shape = (
                    ep_local_experts,
                    _ceil(2 * inter_per_partition, fp8_block_size),
                    _ceil(hidden_size, fp8_block_size),
                )
                w2_scale_shape = (
                    ep_local_experts,
                    _ceil(hidden_size, fp8_block_size),
                    _ceil(inter_per_partition, fp8_block_size),
                )
                manager.reserve(f"{elp}.w13_weight_scale", w13_scale_shape, torch.float32)
                manager.reserve(f"{elp}.w2_weight_scale", w2_scale_shape, torch.float32)
            else:
                manager.reserve(f"{elp}.w13_weight_scale", (ep_local_experts, 2), torch.float32)
                manager.reserve(f"{elp}.w2_weight_scale", (ep_local_experts,), torch.float32)

        # -- Attention weights ---------------------------------------------
        qkv_out = num_heads * head_dim + 2 * num_kv_heads * head_dim
        manager.reserve(
            f"{lp}.self_attn.qkv_proj.weight",
            (qkv_out, hidden_size),
            torch.bfloat16,
        )
        manager.reserve(
            f"{lp}.self_attn.o_proj.weight",
            (hidden_size, num_heads * head_dim),
            torch.bfloat16,
        )

        # -- QKV TP buffer -------------------------------------------------
        tp_q_size = (num_heads // tp_size) * head_dim
        tp_kv_size = (num_kv_heads // tp_size) * head_dim
        manager.reserve(
            f"{lp}.self_attn.qkv_proj.tp_weight",
            (tp_q_size + 2 * tp_kv_size, hidden_size),
            torch.bfloat16,
        )

    if configure_method != "peer_access":
        staging_dtype = torch.float8_e4m3fn if is_fp8 else torch.bfloat16
        staging_experts = (num_experts // ep_size) * dp_size
        w13_shape = (staging_experts, 2 * intermediate_size, hidden_size)
        w2_shape = (staging_experts, hidden_size, intermediate_size)

        if configure_method == "overlap":
            suffixes = ("_1", "_2")
        else:
            suffixes = ("",)

        for sfx in suffixes:
            manager.reserve(f"staging.w13_pre_permute{sfx}", w13_shape, staging_dtype)
            manager.reserve(f"staging.w2_pre_permute{sfx}", w2_shape, staging_dtype)
            if dp_size > 1:
                manager.reserve(f"staging.w13_gather{sfx}", w13_shape, staging_dtype)
                manager.reserve(f"staging.w2_gather{sfx}", w2_shape, staging_dtype)


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
    configure_method: str = "peer_access",
    prefix: str = "model",
) -> None:
    """Reserve all weight tensors for a GPT-OSS sparse-MoE model.

    GPT-OSS shares the Qwen3-MoE layout exactly for the tensors ParaS
    manages: w13/w2 expert weights (N+1 slot layout for EP<->TP switch),
    QKV/O attention projections, FP8 weight scales, and pre-permute /
    gather staging buffers for non-peer_access transfer methods.  Both
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
    """
    Create ep_experts and tp_experts aliases for the N+1 slot layout.
    Call after materialize().

    ep_experts layer i → slot i+1 (same physical buffer as EP weights)
    tp_experts layer i → slot i   (one slot before EP, for fused transfer)
    """
    for i in range(num_layers):
        manager.alias(f"{prefix}.layers.{i}.mlp.ep_experts.w13_weight", f"paras.moe_slot.{i+1}.w13")
        manager.alias(f"{prefix}.layers.{i}.mlp.ep_experts.w2_weight", f"paras.moe_slot.{i+1}.w2")
        manager.alias(f"{prefix}.layers.{i}.mlp.tp_experts.w13_weight", f"paras.moe_slot.{i}.w13")
        manager.alias(f"{prefix}.layers.{i}.mlp.tp_experts.w2_weight", f"paras.moe_slot.{i}.w2")



