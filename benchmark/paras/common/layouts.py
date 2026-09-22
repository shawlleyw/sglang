"""KV + Weight layout derivation from a `ModelConfig` plus benchmark volume controls.

Cache volume control follows the CLI contract:
    bytes_per_ep_slot = num_kv_heads * head_dim * elem_size * 2 (K+V)
    ep_max_tokens_per_rank = cache_size_gb * 1024**3 / bytes_per_ep_slot
    num_resident_tokens_per_rank = int(ep_max_tokens_per_rank * load)

So `--cache-size-gb 20 --load 0.5` yields ~10 GiB of *resident* KV per GPU,
which is the source-side load measured by the kernel. EP→TP moves N resident tokens per EP source. TP→EP moves W*N/R tokens
per TP source after head-replica deduplication (W ranks, R replicas).
The routing helper rounds N down to a multiple of R.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import math

from .model_configs import ModelConfig


@dataclass
class KVLayout:
    """Per-rank slot pool layout for cache transfer benchmarks.

    EP slots: shape (ep_max_tokens, num_kv_heads, head_dim)
    TP slots: shape (tp_max_tokens, heads_per_rank, head_dim)

    `tp_max_tokens` is sized for the worst direction (transfer destination
    receives `world_size * num_resident_tokens_per_rank` tokens) plus +1 for
    the slot-0 padding convention used by every peer-access kernel.

    R = tp_size / num_kv_heads when num_kv_heads < tp_size. heads_per_rank is
    `max(1, num_kv_heads // tp_size)`; per kernel convention each TP rank
    writes head index `tp_rank * num_kv_heads / tp_size`, so R contiguous
    ranks share an EP head and broadcast the same data in production.
    """

    tp_size: int
    num_kv_heads: int
    head_dim: int
    elem_size: int
    ep_max_tokens: int
    tp_max_tokens: int
    num_resident_tokens: int

    @property
    def heads_per_rank(self) -> int:
        return max(1, self.num_kv_heads // self.tp_size)

    @property
    def replication_factor(self) -> int:
        return max(1, self.tp_size // self.num_kv_heads)

    @property
    def bytes_per_tp_slot(self) -> int:
        return self.heads_per_rank * self.head_dim * self.elem_size

    @property
    def bytes_per_ep_slot(self) -> int:
        return self.num_kv_heads * self.head_dim * self.elem_size

    @property
    def tp_buffer_bytes(self) -> int:
        return self.tp_max_tokens * self.bytes_per_tp_slot

    @property
    def ep_buffer_bytes(self) -> int:
        return self.ep_max_tokens * self.bytes_per_ep_slot


def make_kv_layout(
    model: ModelConfig, tp_size: int, cache_size_gb: float, load: float
) -> KVLayout:
    if cache_size_gb <= 0:
        raise SystemExit(f"--cache-size-gb must be > 0, got {cache_size_gb}")
    if not (0.0 < load <= 1.0):
        raise SystemExit(f"--load must be in (0, 1], got {load}")

    cache_bytes = int(cache_size_gb * (1024**3))
    bytes_per_ep_slot = model.num_kv_heads * model.head_dim * model.elem_size * 2

    ep_max_tokens = max(2, cache_bytes // bytes_per_ep_slot)
    num_resident = max(1, int(ep_max_tokens * load))
    if num_resident >= ep_max_tokens:
        num_resident = ep_max_tokens - 1

    tp_max_tokens = tp_size * num_resident + 1

    return KVLayout(
        tp_size=tp_size,
        num_kv_heads=model.num_kv_heads,
        head_dim=model.head_dim,
        elem_size=model.elem_size,
        ep_max_tokens=ep_max_tokens,
        tp_max_tokens=tp_max_tokens,
        num_resident_tokens=num_resident,
    )


@dataclass
class WeightLayout:
    """MoE weight layout - kernel-source-of-truth shapes.

    EP w13:  (E_local, num_gates, tp_size, I'*H)  -- strided by tp_size
    TP w13:  (tp_size * E_local, num_gates, I'*H)
    EP w2:   (E_local, H, I_full = tp_size * I')
    TP w2:   (tp_size * E_local, H, I')
    """

    tp_size: int
    E_local: int
    H: int
    I_prime: int
    num_gates: int
    elem_size: int

    @property
    def I_full(self) -> int:
        return self.tp_size * self.I_prime

    @property
    def I_prime_H(self) -> int:
        return self.I_prime * self.H

    @property
    def w13_chunk_bytes(self) -> int:
        return self.I_prime_H * self.elem_size

    @property
    def w13_ep_buffer_bytes(self) -> int:
        return self.E_local * self.num_gates * self.tp_size * self.w13_chunk_bytes

    @property
    def w13_tp_buffer_bytes(self) -> int:
        return self.tp_size * self.E_local * self.num_gates * self.w13_chunk_bytes

    @property
    def w2_ep_buffer_bytes(self) -> int:
        return self.E_local * self.H * self.I_full * self.elem_size

    @property
    def w2_tp_buffer_bytes(self) -> int:
        return self.tp_size * self.E_local * self.H * self.I_prime * self.elem_size


def make_weight_layout(model: ModelConfig, tp_size: int) -> WeightLayout:
    """Derive the MoE weight layout from a model config + TP size.

    `E_local = num_experts / tp_size` per the EP convention. If the model has
    `num_experts < tp_size`, the bench falls back to E_local=1 to keep the
    kernel exercisable but flags the configuration.
    """
    if model.num_experts < tp_size:
        E_local = 1
    elif model.num_experts % tp_size != 0:
        raise SystemExit(
            f"num_experts ({model.num_experts}) must be divisible by tp_size ({tp_size})"
        )
    else:
        E_local = model.num_experts // tp_size

    if model.moe_intermediate_size % tp_size != 0:
        raise SystemExit(
            f"moe_intermediate_size ({model.moe_intermediate_size}) must be divisible by tp_size ({tp_size})"
        )
    I_prime = model.moe_intermediate_size // tp_size

    return WeightLayout(
        tp_size=tp_size,
        E_local=E_local,
        H=model.hidden_size,
        I_prime=I_prime,
        num_gates=model.num_gates,
        elem_size=model.elem_size,
    )


def make_resident_kv_layout(model, tp_size, resident_cache_gib, load=1.0):
    """Uniform distinct layers: resident EP K+V volume summed across the model."""
    if not math.isfinite(resident_cache_gib) or resident_cache_gib <= 0:
        raise ValueError("resident cache GiB must be finite and positive")
    if model.num_hidden_layers <= 0 or not 0 < load <= 1:
        raise ValueError("layer count must be positive and load in (0, 1]")
    replication = max(1, tp_size // model.num_kv_heads)
    token_bytes = 2 * model.num_kv_heads * model.head_dim * model.elem_size
    n = int(resident_cache_gib * 2**30) // (model.num_hidden_layers * token_bytes)
    n = n // replication * replication
    if n == 0:
        raise ValueError(
            "resident volume must hold at least one token per replica per layer"
        )
    return KVLayout(
        tp_size=tp_size,
        num_kv_heads=model.num_kv_heads,
        head_dim=model.head_dim,
        elem_size=model.elem_size,
        ep_max_tokens=math.ceil(n / load) + 1,
        tp_max_tokens=tp_size * n + 1,
        num_resident_tokens=n,
    )


def add_volume_args(parser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--cache-size-gb",
        type=float,
        default=None,
        help="Legacy isolated-layer EP capacity in GiB (default 10)",
    )
    group.add_argument(
        "--resident-cache-gib",
        type=float,
        help="Total resident EP K+V GiB per GPU across distinct uniform model layers; no SWA",
    )
    parser.add_argument(
        "--load",
        type=float,
        default=None,
        help="Resident fraction; defaults to 1 for --resident-cache-gib, 0.5 for legacy mode",
    )


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(a[0], b[0]) < min(a[1], b[1])


@dataclass(frozen=True)
class OverlappingCacheLayout:
    """Layer-wise EP/TP aliases with an EP-sized gap and safe migration order.

    TP stride is padded to at least the EP layer size. Callers must fence
    each layer before committing the next destination; prefetching into
    independent staging alone does not satisfy that dependency.
    """

    ep_buffer_bytes: int
    tp_buffer_bytes: int
    num_layers: int

    def __post_init__(self):
        for name in ("ep_buffer_bytes", "tp_buffer_bytes", "num_layers"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.validate_order("ep_to_tp")
        self.validate_order("tp_to_ep")

    @property
    def ep_layer_bytes(self) -> int:
        return 2 * self.ep_buffer_bytes

    @property
    def tp_layer_bytes(self) -> int:
        return 2 * self.tp_buffer_bytes

    @property
    def tp_layer_stride(self) -> int:
        return max(self.ep_layer_bytes, self.tp_layer_bytes)

    @property
    def arena_bytes(self) -> int:
        return self.ep_layer_bytes + self.num_layers * self.tp_layer_stride

    def offsets(self, layer: int) -> dict[str, int]:
        if not isinstance(layer, int) or not 0 <= layer < self.num_layers:
            raise ValueError("layer index out of bounds")
        ep = layer * self.ep_layer_bytes
        tp = self.ep_layer_bytes + layer * self.tp_layer_stride
        return {"ep_k": ep, "ep_v": ep + self.ep_buffer_bytes,
                "tp_k": tp, "tp_v": tp + self.tp_buffer_bytes}

    def _regions(self, layer: int, direction: str):
        offsets = self.offsets(layer)
        ep = (offsets["ep_k"], offsets["ep_k"] + self.ep_layer_bytes)
        tp = (offsets["tp_k"], offsets["tp_k"] + self.tp_layer_bytes)
        if direction == "ep_to_tp":
            return ep, tp
        if direction == "tp_to_ep":
            return tp, ep
        raise ValueError(f"unknown direction: {direction}")

    def layer_order(self, direction: str) -> tuple[int, ...]:
        if direction == "ep_to_tp":
            return tuple(range(self.num_layers - 1, -1, -1))
        if direction == "tp_to_ep":
            return tuple(range(self.num_layers))
        raise ValueError(f"unknown direction: {direction}")

    def validate_order(self, direction: str, order: Iterable[int] | None = None):
        """Reject out-of-bounds views or writes that destroy an unread source."""
        expected = self.layer_order(direction)
        order = expected if order is None else tuple(order)
        if sorted(order) != list(range(self.num_layers)):
            raise ValueError("order must visit every layer exactly once")
        regions = [self._regions(layer, direction) for layer in range(self.num_layers)]
        for layer, (source, destination) in enumerate(regions):
            for start, end in (source, destination):
                if not 0 <= start < end <= self.arena_bytes:
                    raise ValueError(f"layer {layer} view outside arena")
            if _overlaps(source, destination):
                raise ValueError(f"layer {layer} source and destination overlap")
        unread = set(range(self.num_layers))
        for layer in order:
            unread.remove(layer)
            destination = regions[layer][1]
            for other in unread:
                if _overlaps(destination, regions[other][0]):
                    raise ValueError(f"layer {layer} write destroys unread layer {other}")

    def prefetch_hazards(self, direction: str) -> tuple[tuple[int, int], ...]:
        """Return (current, next) pairs needing read-before-next-write fences.

        Pure next-source prefetch into disjoint staging is safe. These hazards
        concern the next destination commit racing a still-active current read.
        """
        order = self.layer_order(direction)
        return tuple((current, nxt) for current, nxt in zip(order, order[1:])
                     if _overlaps(self._regions(current, direction)[0],
                                  self._regions(nxt, direction)[1]))


def make_overlapping_cache_layout(layout, num_layers: int) -> OverlappingCacheLayout:
    """Derive geometry from KVLayout, including its already-reserved slot zero."""
    return OverlappingCacheLayout(layout.ep_buffer_bytes, layout.tp_buffer_bytes, num_layers)
