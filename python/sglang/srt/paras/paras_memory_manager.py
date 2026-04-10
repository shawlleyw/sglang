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
"""

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


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
    ) -> None:
        """
        Reserve KV cache entries using EP shapes (same bytes as TP via union layout).

        Must be called AFTER plan_qwen_moe_layout() and BEFORE materialize().

        EP and TP KV have same total bytes per layer:
          ep_tokens × ep_kv_heads × head_dim == tp_tokens × tp_kv_heads × head_dim
        so we reserve once in EP shape and use get_view_as() for TP access.
        """
        if self._materialized:
            raise RuntimeError("Cannot reserve KV cache after materialize().")
        if self._kv_reserved:
            raise RuntimeError("KV cache already reserved.")

        self.ep_max_kv_tokens = ep_max_tokens
        self.tp_max_kv_tokens = tp_max_tokens

        for i in range(num_layers):
            lp = f"{prefix}.layers.{i}"
            kv_shape = (ep_max_tokens + page_size, num_kv_heads, head_dim)
            self.reserve(f"{lp}.kv.k", kv_shape, kv_dtype)
            self.reserve(f"{lp}.kv.v", kv_shape, kv_dtype)

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
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Return k_buffers and v_buffers for the KV pool in the given mode.

        EP mode: returns views with (ep_tokens + page, total_kv_heads, head_dim)
        TP mode: returns views with (tp_tokens + page, total_kv_heads//tp_size, head_dim)
                 using get_view_as to reinterpret the same bytes.
        """
        k_bufs: List[torch.Tensor] = []
        v_bufs: List[torch.Tensor] = []
        for i in range(num_layers):
            lp = f"{prefix}.layers.{i}"
            k_name = f"{lp}.kv.k"
            v_name = f"{lp}.kv.v"

            if mode == "ep":
                k_bufs.append(self.get_view(k_name))
                v_bufs.append(self.get_view(v_name))
            elif mode == "tp":
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
        kv_names = {n for n in self._reservation_order if ".kv." in n}
        return sum(
            self._entries[n].size_bytes
            for n in self._reservation_order
            if n not in kv_names
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

def _reserve_moe_weights(
    manager: ParaSMemoryManager,
    prefix: str,
    layer_idx: int,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    ep_size: int,
    moe_tp_size: int,
    use_triton_kernels: bool,
    quant_name: Optional[str],
    fp8_block_size: Optional[int],
) -> None:
    """
    Reserve EP MoE weight tensors (and FP8 scales) for one layer.

    Only EP expert buffers are reserved. TP experts reuse the same buffer
    region with a different view shape via ``get_view_as`` — the total byte
    count is identical when ``ep_size == tp_size``.

    Shape conventions must match what ``FusedMoE.__init__`` → ``create_weights``
    actually produces at runtime:
      - ``intermediate_size_per_partition = intermediate_size // moe_tp_size``
      - BF16/FP16 with triton kernels: w13 = (E, H, 2*I'), w2 = (E, I', H)
      - All other cases (FP8 / non-triton): w13 = (E, 2*I', H), w2 = (E, H, I')

    where E = num_experts // ep_size, H = hidden_size,
    I' = intermediate_size // moe_tp_size.
    """
    is_fp8 = quant_name == "fp8"
    weight_dtype = torch.float8_e4m3fn if is_fp8 else torch.bfloat16
    lp = f"{prefix}.layers.{layer_idx}.mlp.experts"

    # EP experts: subset of experts, intermediate partitioned by moe_tp_size
    ep_local_experts = num_experts // ep_size
    inter_per_partition = intermediate_size // moe_tp_size

    # Shape depends on triton kernel usage (BF16 triton transposes dims)
    if use_triton_kernels and not is_fp8:
        w13_shape = (ep_local_experts, hidden_size, 2 * inter_per_partition)
        w2_shape = (ep_local_experts, inter_per_partition, hidden_size)
    else:
        w13_shape = (ep_local_experts, 2 * inter_per_partition, hidden_size)
        w2_shape = (ep_local_experts, hidden_size, inter_per_partition)

    manager.reserve(f"{lp}.w13_weight", w13_shape, weight_dtype)
    manager.reserve(f"{lp}.w2_weight", w2_shape, weight_dtype)

    # --- FP8 scale tensors ------------------------------------------------
    if is_fp8:
        if fp8_block_size is not None and fp8_block_size > 0:
            # Block-quantised scales — 3-D with ceil division
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
            manager.reserve(
                f"{lp}.w13_weight_scale", w13_scale_shape, torch.float32
            )
            manager.reserve(
                f"{lp}.w2_weight_scale", w2_scale_shape, torch.float32
            )
        else:
            # Per-tensor scales
            manager.reserve(    
                f"{lp}.w13_weight_scale",
                (ep_local_experts, 2),
                torch.float32,
            )
            manager.reserve(
                f"{lp}.w2_weight_scale",
                (ep_local_experts,),
                torch.float32,
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
    use_triton_kernels: bool,
    quant_name: Optional[str] = None,
    fp8_block_size: Optional[int] = None,
    num_fused_shared_experts: int = 0,
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

    for i in range(num_layers):
        # -- MoE weights (EP only — TP reuses same buffer via get_view_as) -
        _reserve_moe_weights(
            manager=manager,
            prefix=prefix,
            layer_idx=i,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            ep_size=ep_size,
            moe_tp_size=moe_tp_size,
            use_triton_kernels=use_triton_kernels,
            quant_name=quant_name,
            fp8_block_size=fp8_block_size,
        )

        lp = f"{prefix}.layers.{i}"

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

    # -- Static staging buffers for EP→TP weight redistribution ------------
    # Shared across all layers (reused layer-by-layer during switch).
    # staging_a: all-gather destination (dp>1) or permuted input (dp==1)
    # staging_b: permuted all-to-all input (dp>1), unused for dp==1
    is_fp8 = quant_name == "fp8"
    staging_dtype = torch.float8_e4m3fn if is_fp8 else torch.bfloat16
    staging_experts = (num_experts // ep_size) * dp_size

    manager.reserve(
        "staging.w13_a",
        (staging_experts, 2 * intermediate_size, hidden_size),
        staging_dtype,
    )
    manager.reserve(
        "staging.w13_b",
        (staging_experts, 2 * intermediate_size, hidden_size),
        staging_dtype,
    )
    manager.reserve(
        "staging.w2_a",
        (staging_experts, hidden_size, intermediate_size),
        staging_dtype,
    )
    manager.reserve(
        "staging.w2_b",
        (staging_experts, hidden_size, intermediate_size),
        staging_dtype,
    )
