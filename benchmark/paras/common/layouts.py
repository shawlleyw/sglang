"""KV + Weight layout derivation from a `ModelConfig` plus benchmark volume controls.

Cache volume control follows the CLI contract:
    bytes_per_ep_slot = num_kv_heads * head_dim * elem_size * 2 (K+V)
    ep_max_tokens_per_rank = cache_size_gb * 1024**3 / bytes_per_ep_slot
    num_resident_tokens_per_rank = int(ep_max_tokens_per_rank * load)

So `--cache-size-gb 20 --load 0.5` yields ~10 GiB of *resident* KV per GPU,
which is the source-side load measured by the kernel. Both EP→TP and TP→EP
move `num_resident_tokens_per_rank` tokens per source rank, giving direct
comparability between the two directions.
"""

from __future__ import annotations

from dataclasses import dataclass

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


def make_kv_layout(model: ModelConfig, tp_size: int,
                   cache_size_gb: float, load: float) -> KVLayout:
    if cache_size_gb <= 0:
        raise SystemExit(f"--cache-size-gb must be > 0, got {cache_size_gb}")
    if not (0.0 < load <= 1.0):
        raise SystemExit(f"--load must be in (0, 1], got {load}")

    cache_bytes = int(cache_size_gb * (1024 ** 3))
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


def add_volume_args(parser) -> None:
    parser.add_argument("--cache-size-gb", type=float, default=10.0,
                        help="Per-GPU EP cache capacity in GiB (default 10)")
    parser.add_argument("--load", type=float, default=0.5,
                        help="Fraction of cache resident, in (0, 1]. "
                             "Cache size * load = resident bytes per GPU. (default 0.5)")
