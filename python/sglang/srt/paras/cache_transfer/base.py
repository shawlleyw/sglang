"""Abstract types and shared base for ParaS cache transfer backends.

This module defines the core abstractions -- ``LayerCacheSpec`` (the
per-layer descriptor that drives all per-layer dispatch) and
``CacheTransferBackend`` (the Protocol every backend implements) --
plus ``CacheTransferBase``, the concrete base class that holds all
state and precomputation logic common to both MHA and SWA backends.

Backends live in sibling modules (``mha.py``, ``swa.py``), shared
stateless helpers live in ``utils.py``.
"""

from dataclasses import dataclass
from typing import List, Literal, Optional, Protocol, runtime_checkable

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class LayerCacheSpec:
    """Specification for a single layer's KV cache in ParaS.

    Attributes:
        layer_id: Layer index in the model.
        kind: Attention type -- ``"full"`` for full attention, ``"swa"`` for
            sliding window.
        tokens_cap_ep: Per-layer token capacity for expert parallelism.
        tokens_cap_tp: Per-layer token capacity for tensor parallelism.
        num_kv_heads: Number of key-value heads in this layer.
        head_dim: Dimension of each attention head.
        sliding_window_size: Window size for SWA layers (``None`` for full).
    """
    layer_id: int
    kind: Literal["full", "swa"]
    tokens_cap_ep: int
    tokens_cap_tp: int
    num_kv_heads: int
    head_dim: int
    sliding_window_size: Optional[int]


@runtime_checkable
class CacheTransferBackend(Protocol):
    """Protocol for cache transfer backends (gather/scatter operations).

    Implementations handle moving KV cache data across TP/EP boundaries
    for both full and sliding window attention layers.  Concrete
    backends live in ``mha.py`` and ``swa.py``; each supports two
    transport methods (``"nccl"`` and ``"peer_access"``) selected at
    construction time.

    The manager is responsible for:
      * iterating layers in the correct order (forward for gather,
        reverse for scatter),
      * calling ``dist.all_reduce(barrier_tensor)`` after every layer
        (unconditionally -- ALL ranks must participate each step or NCCL
        deadlocks).
    """

    def gather_one_layer(self, spec: LayerCacheSpec, **kwargs) -> None:
        """Gather KV cache for one layer (EP -> TP)."""
        ...

    def scatter_one_layer(self, spec: LayerCacheSpec, **kwargs) -> None:
        """Scatter KV cache for one layer (TP -> EP)."""
        ...


# ------------------------------------------------------------------
# Shared concrete base class
# ------------------------------------------------------------------


class CacheTransferBase:
    """Shared base for MHA and SWA cache transfer backends.

    Holds all state and precomputation logic common to both
    direction-kind combinations (gather/scatter x nccl/peer_access).
    Subclasses override only:
      - ``gather_one_layer(spec)``: per-layer EP->TP dispatch
      - ``scatter_one_layer(spec)``: per-layer TP->EP dispatch
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        method: Literal["nccl", "peer_access"],
        direction: Literal["gather", "scatter"],
        kv_cache,
        mgr,
        group,
        # -- gather state --
        num_local_tokens: int = 0,
        num_global_tokens: int = 0,
        local_token_indices: Optional[torch.Tensor] = None,
        global_token_indices: Optional[torch.Tensor] = None,
        global_num_tokens: Optional[List[int]] = None,
        layer_specs: Optional[list] = None,
        # -- peer_access state --
        peer_addresses: Optional[List[int]] = None,
        # -- scatter state --
        ep_head_num: int = 0,
        token_partition: Optional[List[List[int]]] = None,
        ep_dst_positions: Optional[torch.Tensor] = None,
        paras_tp_rank: int = 0,
        paras_tp_size: int = 1,
        source_full_to_swa_mapping: Optional[torch.Tensor] = None,
    ):
        self.method = method
        self.direction = direction
        self.kv_cache = kv_cache
        self.mgr = mgr
        self.group = group
        self.num_local_tokens = num_local_tokens
        self.num_global_tokens = num_global_tokens
        self.local_token_indices = local_token_indices
        self.global_token_indices = global_token_indices
        self.global_num_tokens = global_num_tokens
        self.layer_specs = layer_specs
        self.ep_head_num = ep_head_num
        self.token_partition = token_partition
        self.ep_dst_positions = ep_dst_positions
        self.paras_tp_rank = paras_tp_rank
        self.paras_tp_size = paras_tp_size
        self.group_size = group.world_size
        self.source_full_to_swa_mapping = source_full_to_swa_mapping

        if direction == "gather":
            self._precompute_gather(peer_addresses)
        else:
            self._precompute_scatter(peer_addresses)

    # ------------------------------------------------------------------
    # Gather pre-computation
    # ------------------------------------------------------------------

    def _precompute_gather(self, peer_addresses: Optional[List[int]]) -> None:
        if self.method != "peer_access":
            return

        self._tp_rank = dist.get_rank(group=self.group.device_group)
        self._dst_token_start = (
            sum(self.global_num_tokens[: self._tp_rank]) + 1
            if self.global_num_tokens
            else 1
        )
        self._peer_addresses_gpu = torch.tensor(
            peer_addresses, dtype=torch.int64, device="cuda"
        )
        dtype = self.kv_cache.store_dtype
        self._elem_size = dtype.itemsize if hasattr(dtype, "itemsize") else 2
        self._local_buffer_ptr = self.mgr._buffer.data_ptr()

        if (
            self.local_token_indices is not None
            and self.local_token_indices.dtype != torch.int32
        ):
            self.local_token_indices = self.local_token_indices.to(torch.int32)

    # ------------------------------------------------------------------
    # Scatter pre-computation
    # ------------------------------------------------------------------

    def _precompute_scatter(self, peer_addresses: Optional[List[int]]) -> None:
        if self.method == "nccl":
            self._precompute_scatter_nccl()
        else:
            self._precompute_scatter_peer_access(peer_addresses)

    def _precompute_scatter_nccl(self) -> None:
        """Pre-compute NCCL scatter metadata (replication/split-size math)."""
        self._tp_rank = dist.get_rank(group=self.group.device_group)
        self._num_kv_heads = self.ep_head_num
        self._heads_per_rank = self.kv_cache.head_num
        self._head_dim = self.kv_cache.head_dim

        group_size = self.group_size
        num_kv_heads = self._num_kv_heads
        heads_per_rank = self._heads_per_rank

        if num_kv_heads >= group_size:
            self._replication_factor = 1
        else:
            self._replication_factor = group_size // num_kv_heads

        self._per_token_elems = heads_per_rank * 2 * self._head_dim
        self._intra_rank = self._tp_rank % self._replication_factor

        token_partition = self.token_partition
        self._total_global_tokens = sum(
            len(token_partition[e]) for e in range(group_size)
        )

        # -- Send side --
        intra_rank = self._intra_rank
        replication_factor = self._replication_factor
        per_token_elems = self._per_token_elems

        send_token_counts: List[int] = []
        sorted_parts: List[torch.Tensor] = []
        for e in range(group_size):
            full = len(token_partition[e])
            my_start = full * intra_rank // replication_factor
            my_end = full * (intra_rank + 1) // replication_factor
            my_count = my_end - my_start
            send_token_counts.append(my_count)
            if my_count > 0:
                my_indices = token_partition[e][my_start:my_end]
                part_idx = torch.tensor(
                    my_indices,
                    dtype=torch.long,
                    device=self.global_token_indices.device,
                )
                sorted_parts.append(self.global_token_indices[part_idx])

        self._total_send_tokens = sum(send_token_counts)
        self._input_split_sizes = [
            cnt * per_token_elems for cnt in send_token_counts
        ]

        if sorted_parts:
            self._sorted_tp_indices = torch.cat(sorted_parts)
        else:
            self._sorted_tp_indices = torch.empty(
                0,
                dtype=torch.long,
                device=(
                    self.global_token_indices.device
                    if self.global_token_indices is not None
                    else "cuda"
                ),
            )

        # -- Recv side --
        self._recv_full_count = len(token_partition[self._tp_rank])
        recv_token_counts: List[int] = []
        for src in range(group_size):
            src_intra = src % replication_factor
            s = self._recv_full_count * src_intra // replication_factor
            e_idx = self._recv_full_count * (src_intra + 1) // replication_factor
            recv_token_counts.append(e_idx - s)

        self._output_split_sizes = [
            cnt * per_token_elems for cnt in recv_token_counts
        ]
        self._total_recv_elems = sum(self._output_split_sizes)

        self._reassembly_groups = (
            group_size if heads_per_rank > 1 else num_kv_heads
        )

    def _precompute_scatter_peer_access(
        self, peer_addresses: Optional[List[int]]
    ) -> None:
        """Pre-compute peer-access scatter metadata (token-slice/dst-ranks)."""
        self._local_buffer_ptr = self.mgr._buffer.data_ptr()
        self._peer_buffer_ptrs = torch.tensor(
            peer_addresses, dtype=torch.int64, device="cuda"
        )

        self._num_kv_heads = self.ep_head_num
        self._heads_per_rank = self.kv_cache.head_num
        self._head_dim = self.kv_cache.head_dim
        dtype = self.kv_cache.store_dtype
        self._elem_size = dtype.itemsize if hasattr(dtype, "itemsize") else 2

        group_size = self.group_size
        num_kv_heads = self._num_kv_heads
        replication_factor = (
            group_size // num_kv_heads if num_kv_heads < group_size else 1
        )
        intra_rank = self.paras_tp_rank % replication_factor

        token_partition = self.token_partition
        my_global_indices: List[int] = []
        my_dst_ranks: List[int] = []
        my_ep_dst_pos: List[int] = []
        for e in range(group_size):
            full_tokens = token_partition[e]
            full = len(full_tokens)
            my_start = full * intra_rank // replication_factor
            my_end = full * (intra_rank + 1) // replication_factor
            my_slice = full_tokens[my_start:my_end]
            for local_idx, global_idx in enumerate(my_slice):
                my_global_indices.append(global_idx)
                my_dst_ranks.append(e)
                my_ep_dst_pos.append(my_start + local_idx + 1)

        self._num_my_tokens = len(my_global_indices)

        if self._num_my_tokens > 0:
            gi_tensor = torch.tensor(
                my_global_indices, dtype=torch.long, device="cuda"
            )
            self._tp_token_positions = self.global_token_indices[gi_tensor].to(
                torch.int32
            )
            self._token_to_rank = torch.tensor(
                my_dst_ranks, dtype=torch.int32, device="cuda"
            )
            self._ep_dst_pos_all = torch.tensor(
                my_ep_dst_pos, dtype=torch.int32, device="cuda"
            )
        else:
            self._tp_token_positions = torch.empty(
                0, dtype=torch.int32, device="cuda"
            )
            self._token_to_rank = torch.empty(
                0, dtype=torch.int32, device="cuda"
            )
            self._ep_dst_pos_all = torch.empty(
                0, dtype=torch.int32, device="cuda"
            )

    # ------------------------------------------------------------------
    # Per-layer dispatch (subclasses must override)
    # ------------------------------------------------------------------

    def gather_one_layer(self, spec: LayerCacheSpec, **kwargs) -> None:
        raise NotImplementedError("Subclasses must implement gather_one_layer")

    def scatter_one_layer(self, spec: LayerCacheSpec, **kwargs) -> None:
        raise NotImplementedError("Subclasses must implement scatter_one_layer")


# ------------------------------------------------------------------
# Config helpers
# ------------------------------------------------------------------


def classify_layers_from_config(
    hf_config,
    *,
    tp_size: int,
    ep_tokens_full: int,
    tp_tokens_full: int,
    ep_tokens_swa: int,
    tp_tokens_swa: int,
    ratio: float = 1.0,
) -> list[LayerCacheSpec]:
    """Classify layers and generate a ``LayerCacheSpec`` list from an HF config.

    Handles two config styles:
      * GPT-OSS / Gemma: ``config.layer_types = ["full_attention", "sliding_attention", ...]``
      * Default (everything else): all layers are full attention.
    """
    num_layers = hf_config.num_hidden_layers
    num_kv_heads = hf_config.num_key_value_heads
    head_dim = getattr(
        hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads
    )

    if hasattr(hf_config, "layer_types") and hf_config.layer_types is not None:
        layer_types = hf_config.layer_types
    else:
        layer_types = ["full_attention"] * num_layers

    specs = []
    for layer_id, layer_type in enumerate(layer_types):
        is_swa = layer_type == "sliding_attention"
        kind = "swa" if is_swa else "full"

        if is_swa:
            tokens_cap_ep = int(ep_tokens_swa * ratio)
            tokens_cap_tp = int(tp_tokens_swa * ratio)
            sliding_window_size = hf_config.sliding_window - 1
        else:
            tokens_cap_ep = int(ep_tokens_full * ratio)
            tokens_cap_tp = int(tp_tokens_full * ratio)
            sliding_window_size = None

        specs.append(LayerCacheSpec(
            layer_id=layer_id,
            kind=kind,
            tokens_cap_ep=tokens_cap_ep,
            tokens_cap_tp=tokens_cap_tp,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            sliding_window_size=sliding_window_size,
        ))

    validate_layer_specs(specs, tp_size)
    return specs


def validate_layer_specs(specs: list[LayerCacheSpec], tp_size: int) -> None:
    """Validate a list of ``LayerCacheSpec`` for consistency."""
    if not specs:
        return

    num_kv_heads = specs[0].num_kv_heads
    if not all(s.num_kv_heads == num_kv_heads for s in specs):
        raise ValueError(
            "LayerCacheSpec: num_kv_heads must be uniform across all layers"
        )

    has_swa = any(s.kind == "swa" for s in specs)
    if has_swa and tp_size > num_kv_heads:
        raise ValueError(
            "LayerCacheSpec: tp_size must be <= num_kv_heads for SWA layers"
        )


__all__ = [
    "LayerCacheSpec",
    "CacheTransferBackend",
    "CacheTransferBase",
    "classify_layers_from_config",
    "validate_layer_specs",
]
