"""MHA (Multi-Head Attention) cache transfer backend for ParaS.

Implements ``CacheTransferBackend`` with full gather+scatter dispatch
across both ``nccl`` and ``peer_access`` transport methods.  The manager
constructs one ``MHACacheTransfer`` per direction, pre-computes all
derived state in ``__init__``, then calls ``gather_one_layer`` or
``scatter_one_layer`` in a per-layer loop.  Barriers remain the
manager's responsibility.
"""

from typing import List, Literal, Optional, Union

import torch
import torch.distributed as dist

from sglang.srt.paras.cache_transfer.base import CacheTransferBackend, LayerCacheSpec
from sglang.srt.paras.cache_transfer.utils import (
    do_gather_one_layer_nccl,
    do_gather_one_layer_peer_access,
    do_scatter_one_layer_nccl,
    do_scatter_one_layer_peer_access,
)

# Backward-compat re-export consumed by swa.py.
do_gather_one_layer = do_gather_one_layer_nccl


class MHACacheTransfer:
    """MHA cache transfer backend — nccl + peer_access, gather + scatter.

    Implements the ``CacheTransferBackend`` protocol.  Barriers remain the
    caller's responsibility.
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
        # -- peer_access state --
        peer_addresses: Optional[List[int]] = None,
        # -- scatter state --
        ep_head_num: int = 0,
        token_partition: Optional[List[List[int]]] = None,
        ep_dst_positions: Optional[torch.Tensor] = None,
        paras_tp_rank: int = 0,
        paras_tp_size: int = 1,
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
        self.ep_head_num = ep_head_num
        self.token_partition = token_partition
        self.ep_dst_positions = ep_dst_positions
        self.paras_tp_rank = paras_tp_rank
        self.paras_tp_size = paras_tp_size
        self.group_size = group.world_size

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
            sum(self.global_num_tokens[: self._tp_rank])
            if self.global_num_tokens
            else 0
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
        """Mirror of ``_scatter_cache_nccl`` lines 615-703."""
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
        """Mirror of ``_scatter_cache_peer_access`` lines 440-506."""
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
    # Per-layer dispatch: gather
    # ------------------------------------------------------------------

    def gather_one_layer(self, spec: LayerCacheSpec, **kwargs) -> None:
        layer_id = spec.layer_id
        k_buffer = self.kv_cache.get_key_buffer(layer_id)
        v_buffer = self.kv_cache.get_value_buffer(layer_id)

        num_local = self.num_local_tokens
        layer_global_num = self.global_num_tokens
        num_global = self.num_global_tokens
        local_indices = self.local_token_indices

        if self.method == "nccl":
            do_gather_one_layer_nccl(
                k_buffer,
                v_buffer,
                num_local,
                num_global,
                local_indices,
                self.global_token_indices,
                layer_global_num,
                self.group_size,
                self.kv_cache.head_num,
                self.kv_cache.head_dim,
                self.kv_cache.store_dtype,
                self.kv_cache.device,
                self.mgr,
                layer_id,
                self.group,
            )
        else:
            ep_k_name = f"model.layers.{layer_id}.kv.ep.k"
            ep_v_name = f"model.layers.{layer_id}.kv.ep.v"
            tp_k_name = f"model.layers.{layer_id}.kv.tp.k"
            tp_v_name = f"model.layers.{layer_id}.kv.tp.v"

            do_gather_one_layer_peer_access(
                self._local_buffer_ptr,
                self._peer_addresses_gpu,
                self.mgr._entries[ep_k_name].offset_bytes,
                self.mgr._entries[ep_v_name].offset_bytes,
                self.mgr._entries[tp_k_name].offset_bytes,
                self.mgr._entries[tp_v_name].offset_bytes,
                local_indices,
                num_local,
                self._dst_token_start,
                self.kv_cache.head_num,
                self._tp_rank,
                self.group_size,
                self.kv_cache.head_dim,
                self._elem_size,
            )

    # ------------------------------------------------------------------
    # Per-layer dispatch: scatter
    # ------------------------------------------------------------------

    def scatter_one_layer(self, spec: LayerCacheSpec, **kwargs) -> None:
        if self.method == "nccl":
            do_scatter_one_layer_nccl(
                self.kv_cache,
                self.ep_head_num,
                spec.layer_id,
                self.token_partition,
                self.group_size,
                self._intra_rank,
                self._replication_factor,
                self._per_token_elems,
                self.global_token_indices,
                self.ep_dst_positions,
                self._sorted_tp_indices,
                self._total_send_tokens,
                self._input_split_sizes,
                self._recv_full_count,
                self._output_split_sizes,
                self._total_recv_elems,
                self._num_kv_heads,
                self._heads_per_rank,
                self._head_dim,
                self._total_global_tokens,
                self._reassembly_groups,
                self.mgr,
                self.group,
            )
        else:
            tp_k_name = f"model.layers.{spec.layer_id}.kv.tp.k"
            tp_v_name = f"model.layers.{spec.layer_id}.kv.tp.v"
            ep_k_name = f"model.layers.{spec.layer_id}.kv.ep.k"
            ep_v_name = f"model.layers.{spec.layer_id}.kv.ep.v"

            do_scatter_one_layer_peer_access(
                self._local_buffer_ptr,
                self._peer_buffer_ptrs,
                self._tp_token_positions,
                self._token_to_rank,
                self._ep_dst_pos_all,
                self.mgr._entries[tp_k_name].offset_bytes,
                self.mgr._entries[tp_v_name].offset_bytes,
                self.mgr._entries[ep_k_name].offset_bytes,
                self.mgr._entries[ep_v_name].offset_bytes,
                self._num_my_tokens,
                spec.layer_id,
                self._heads_per_rank,
                self._num_kv_heads,
                self.paras_tp_rank,
                self.paras_tp_size,
                self._head_dim,
                self._elem_size,
            )


__all__ = ["MHACacheTransfer", "do_gather_one_layer"]
