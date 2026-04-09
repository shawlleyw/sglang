"""
ParaSMemoryManager: Static contiguous weight buffer for ParaS EP↔TP switching.
V1 scope: Qwen sparse-MoE (no shared experts, no dense MLPs).
Supported dtypes: BF16/FP16 (unquantized) and FP8.
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

    # ----- reservation ----------------------------------------------------

    def reserve(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
    ) -> LayoutEntry:
        """Register a tensor to be placed in the contiguous buffer."""
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
        self._reservation_order.append(name)
        return entry

    # ----- materialization ------------------------------------------------

    def materialize(self) -> int:
        """
        Assign aligned offsets in reservation order, then allocate the
        backing ``uint8`` buffer on ``self.device``.

        Returns the total buffer size in bytes.
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
        """Return a typed, shaped view into the buffer for *name*."""
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
    tp_size: int,
    quant_name: Optional[str],
    fp8_block_size: Optional[int],
) -> None:
    """
    Reserve EP and TP MoE weight tensors (and FP8 scales) for one layer.

    Weight layout conventions follow FusedMoE / sglang:
      - BF16/FP16 (triton): w13 = (E, H, 2*I),  w2 = (E, I, H)
      - FP8:                 w13 = (E, 2*I, H),   w2 = (E, H, I)

    where E is the local expert count, H = hidden_size, I = intermediate_size
    (or intermediate_size // tp_size for the TP variant).
    """
    is_fp8 = quant_name == "fp8"
    weight_dtype = torch.float8_e4m3fn if is_fp8 else torch.bfloat16
    lp = f"{prefix}.layers.{layer_idx}.mlp.experts"

    # --- EP experts -------------------------------------------------------
    ep_experts = num_experts // ep_size

    if is_fp8:
        ep_w13_shape = (ep_experts, 2 * intermediate_size, hidden_size)
        ep_w2_shape = (ep_experts, hidden_size, intermediate_size)
    else:
        ep_w13_shape = (ep_experts, hidden_size, 2 * intermediate_size)
        ep_w2_shape = (ep_experts, intermediate_size, hidden_size)

    manager.reserve(f"{lp}.ep.w13_weight", ep_w13_shape, weight_dtype)
    manager.reserve(f"{lp}.ep.w2_weight", ep_w2_shape, weight_dtype)

    # --- TP experts -------------------------------------------------------
    tp_inter = intermediate_size // tp_size

    if is_fp8:
        tp_w13_shape = (num_experts, 2 * tp_inter, hidden_size)
        tp_w2_shape = (num_experts, hidden_size, tp_inter)
    else:
        tp_w13_shape = (num_experts, hidden_size, 2 * tp_inter)
        tp_w2_shape = (num_experts, tp_inter, hidden_size)

    manager.reserve(f"{lp}.tp.w13_weight", tp_w13_shape, weight_dtype)
    manager.reserve(f"{lp}.tp.w2_weight", tp_w2_shape, weight_dtype)

    # --- FP8 scale tensors ------------------------------------------------
    if is_fp8:
        if fp8_block_size is not None and fp8_block_size > 0:
            # Block-quantised scales — 3-D with ceil division
            def _ceil(a: int, b: int) -> int:
                return (a + b - 1) // b

            # EP scales
            ep_w13_scale_shape = (
                ep_experts,
                _ceil(2 * intermediate_size, fp8_block_size),
                _ceil(hidden_size, fp8_block_size),
            )
            ep_w2_scale_shape = (
                ep_experts,
                _ceil(hidden_size, fp8_block_size),
                _ceil(intermediate_size, fp8_block_size),
            )
            manager.reserve(
                f"{lp}.ep.w13_weight_scale", ep_w13_scale_shape, torch.float32
            )
            manager.reserve(
                f"{lp}.ep.w2_weight_scale", ep_w2_scale_shape, torch.float32
            )

            # TP scales
            tp_w13_scale_shape = (
                num_experts,
                _ceil(2 * tp_inter, fp8_block_size),
                _ceil(hidden_size, fp8_block_size),
            )
            tp_w2_scale_shape = (
                num_experts,
                _ceil(hidden_size, fp8_block_size),
                _ceil(tp_inter, fp8_block_size),
            )
            manager.reserve(
                f"{lp}.tp.w13_weight_scale", tp_w13_scale_shape, torch.float32
            )
            manager.reserve(
                f"{lp}.tp.w2_weight_scale", tp_w2_scale_shape, torch.float32
            )
        else:
            # Per-tensor scales
            manager.reserve(
                f"{lp}.ep.w13_weight_scale",
                (ep_experts, 2),
                torch.float32,
            )
            manager.reserve(
                f"{lp}.ep.w2_weight_scale",
                (ep_experts,),
                torch.float32,
            )
            manager.reserve(
                f"{lp}.tp.w13_weight_scale",
                (num_experts, 2),
                torch.float32,
            )
            manager.reserve(
                f"{lp}.tp.w2_weight_scale",
                (num_experts,),
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
    quant_name: Optional[str] = None,
    fp8_block_size: Optional[int] = None,
    num_fused_shared_experts: int = 0,
    prefix: str = "model",
) -> None:
    """
    Reserve all weight tensors for a Qwen sparse-MoE model.

    This pre-plans the contiguous buffer layout so that
    ``manager.materialize()`` can allocate exactly once.
    """
    _validate_v1_scope(num_fused_shared_experts, quant_name)

    for i in range(num_layers):
        # -- MoE weights ---------------------------------------------------
        _reserve_moe_weights(
            manager=manager,
            prefix=prefix,
            layer_idx=i,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            ep_size=ep_size,
            tp_size=tp_size,
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
