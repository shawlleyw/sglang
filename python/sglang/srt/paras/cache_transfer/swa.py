"""SWA (Sliding Window Attention) cache transfer backend for ParaS."""

from typing import List, Optional

import torch
import torch.distributed as dist

from sglang.srt.paras.cache_transfer.base import CacheTransferBase, LayerCacheSpec
from sglang.srt.paras.cache_transfer.utils import (
    do_gather_one_layer_nccl,
    do_gather_one_layer_peer_access,
    gather_tp_kv_and_permute,
    permute_and_scatter_kv_to_ep,
)


# ------------------------------------------------------------------
# SWA-cap-aware scatter helpers
# ------------------------------------------------------------------

def _do_scatter_one_layer_nccl_swa(
    spec: "LayerCacheSpec",
    kv_cache,
    ep_head_num: int,
    token_partition: List[List[int]],
    group_size: int,
    intra_rank: int,
    replication_factor: int,
    per_token_elems: int,
    global_token_indices: torch.Tensor,
    ep_dst_positions: Optional[torch.Tensor],
    recv_full_count: int,
    num_kv_heads: int,
    heads_per_rank: int,
    head_dim: int,
    total_global_tokens: int,
    reassembly_groups: int,
    mgr,
    gather_group,
    full_to_swa_mapping: Optional[torch.Tensor],
) -> None:
    cap = spec.tokens_cap_ep
    layer_id = spec.layer_id

    swa_send_counts: List[int] = []
    swa_parts: List[torch.Tensor] = []
    for e in range(group_size):
        full = len(token_partition[e])
        capped = min(full, cap)
        my_s = capped * intra_rank // replication_factor
        my_e = capped * (intra_rank + 1) // replication_factor
        my_cnt = my_e - my_s
        swa_send_counts.append(my_cnt)
        if my_cnt > 0:
            part_idx = torch.tensor(
                token_partition[e][my_s:my_e],
                dtype=torch.long,
                device=global_token_indices.device,
            )
            swa_parts.append(global_token_indices[part_idx])

    total_send = sum(swa_send_counts)
    input_split = [cnt * per_token_elems for cnt in swa_send_counts]
    tp_indices = (
        torch.cat(swa_parts)
        if swa_parts
        else torch.empty(0, dtype=torch.long, device=global_token_indices.device)
    )

    # P4 fix: translate TP-side indices from full-pool to SWA-pool space
    # before reading from the SWA k/v buffers.
    if full_to_swa_mapping is not None and tp_indices.numel() > 0:
        tp_indices = full_to_swa_mapping[tp_indices].to(torch.int64)

    recv_full = min(recv_full_count, cap)
    swa_recv_counts: List[int] = []
    for src in range(group_size):
        si = src % replication_factor
        s = recv_full * si // replication_factor
        e_i = recv_full * (si + 1) // replication_factor
        swa_recv_counts.append(e_i - s)
    output_split = [cnt * per_token_elems for cnt in swa_recv_counts]
    total_recv = sum(output_split)
    dst_pos = (
        ep_dst_positions[:recv_full]
        if ep_dst_positions is not None and recv_full > 0
        else ep_dst_positions
    )

    # P4 fix: translate EP-side destination positions from full-pool to
    # SWA-pool space before writing into the SWA EP buffer.
    if full_to_swa_mapping is not None and dst_pos is not None and dst_pos.numel() > 0:
        dst_pos = full_to_swa_mapping[dst_pos].to(torch.int64)

    if total_send > 0:
        k_buf = kv_cache.get_key_buffer(layer_id)
        v_buf = kv_cache.get_value_buffer(layer_id)
        send_buf = gather_tp_kv_and_permute(
            k_buf, v_buf, tp_indices,
            num_kv_heads, heads_per_rank, head_dim, group_size,
        )
    else:
        send_buf = torch.empty(
            0, dtype=kv_cache.store_dtype, device=kv_cache.device
        )

    if total_global_tokens > 0:
        recv_buf = torch.empty(
            total_recv,
            dtype=kv_cache.store_dtype,
            device=kv_cache.device,
        )
        dist.all_to_all_single(
            recv_buf, send_buf,
            output_split, input_split,
            group=gather_group.device_group,
        )

        if recv_full > 0:
            ep_k_name = f"model.layers.{layer_id}.kv.ep.k"
            ep_v_name = f"model.layers.{layer_id}.kv.ep.v"
            total_elements = mgr._entries[ep_k_name].numel
            ep_slots = total_elements // (num_kv_heads * head_dim)
            ep_shape = (ep_slots, num_kv_heads, head_dim)
            ep_k = mgr.get_view_as(ep_k_name, ep_shape)
            ep_v = mgr.get_view_as(ep_v_name, ep_shape)
            permute_and_scatter_kv_to_ep(
                recv_buf, ep_k, ep_v, dst_pos,
                recv_full, num_kv_heads, heads_per_rank,
                head_dim, reassembly_groups,
            )


def _do_scatter_one_layer_peer_access_swa(
    spec: "LayerCacheSpec",
    local_buffer_ptr: int,
    peer_buffer_ptrs: torch.Tensor,
    src_k_offset: int,
    src_v_offset: int,
    dst_k_offset: int,
    dst_v_offset: int,
    heads_per_rank: int,
    num_kv_heads: int,
    paras_tp_rank: int,
    paras_tp_size: int,
    head_dim: int,
    elem_size: int,
    token_partition: List[List[int]],
    global_token_indices: torch.Tensor,
    group_size: int,
    full_to_swa_mapping: Optional[torch.Tensor],
) -> None:
    """SWA peer-access scatter with per-destination cap and index translation.

    Recomputes per-destination capped slices at dispatch time (Option Z from
    the task spec) to avoid touching base.py.  Translates both TP source
    positions and EP destination positions from full-pool to SWA-pool space.
    """
    import paras_peer_access_cuda

    cap = spec.tokens_cap_ep
    replication_factor = (
        group_size // num_kv_heads if num_kv_heads < group_size else 1
    )
    intra_rank = paras_tp_rank % replication_factor

    # Recompute per-destination capped slices (mirrors NCCL SWA path).
    capped_global_indices: List[int] = []
    capped_dst_ranks: List[int] = []
    capped_ep_dst_pos: List[int] = []
    for e in range(group_size):
        full_tokens = token_partition[e]
        full = len(full_tokens)
        capped = min(full, cap)
        my_start = capped * intra_rank // replication_factor
        my_end = capped * (intra_rank + 1) // replication_factor
        my_slice = full_tokens[my_start:my_end]
        for local_idx, global_idx in enumerate(my_slice):
            capped_global_indices.append(global_idx)
            capped_dst_ranks.append(e)
            # EP destination position: offset within the destination's allocation.
            # +1 because slot 0 is the padding slot.
            capped_ep_dst_pos.append(my_start + local_idx + 1)

    layer_num = len(capped_global_indices)
    if layer_num > 0:
        device = global_token_indices.device
        gi_tensor = torch.tensor(
            capped_global_indices, dtype=torch.long, device=device
        )
        tp_positions = global_token_indices[gi_tensor].to(torch.int32)
        token_to_rank = torch.tensor(
            capped_dst_ranks, dtype=torch.int32, device=device
        )
        ep_dst_pos = torch.tensor(
            capped_ep_dst_pos, dtype=torch.int32, device=device
        )

        # P5 fix (b): translate from full-pool to SWA-pool index space.
        if full_to_swa_mapping is not None:
            tp_positions = full_to_swa_mapping[tp_positions.to(torch.int64)].to(torch.int32)
            ep_dst_pos = full_to_swa_mapping[ep_dst_pos.to(torch.int64)].to(torch.int32)

        paras_peer_access_cuda.launch_peer_access_kv_scatter(
            local_buffer_ptr,
            peer_buffer_ptrs,
            tp_positions,
            token_to_rank,
            ep_dst_pos,
            src_k_offset,
            src_v_offset,
            dst_k_offset,
            dst_v_offset,
            layer_num,
            heads_per_rank,
            num_kv_heads,
            paras_tp_rank,
            paras_tp_size,
            head_dim,
            elem_size,
            0,  # default stream
        )


class SWACacheTransfer(CacheTransferBase):
    """Cache transfer for sliding-window-attention layers.

    Inherits ``__init__`` and all ``_precompute_*`` methods from
    ``CacheTransferBase``.  Overrides the per-layer gather/scatter
    dispatch to read buffers from the SWA sub-pool and cap token
    counts at ``spec.tokens_cap_ep``.

    Index translation
    -----------------
    ``req_to_token_pool`` stores **full-pool** indices.  SWA buffers
    live in SWA index space (size ``_size_swa < _size_full``).  Attention
    backends translate via ``SWAKVPool.full_to_swa_index_mapping`` at
    attend time; this class does the same for cache transfer.

    ``full_to_swa_index_mapping`` is zero-initialized, so translating
    an unallocated full-pool index returns 0 (a valid but overwritten
    padding slot).  This is harmless by design.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # SWAKVPool.full_to_swa_index_mapping is set by
        # SWATokenToKVPoolAllocator.__init__ (allocator.py:224) and kept in
        # sync on every paras_resize_and_clear (allocator.py:318).
        self._full_to_swa_mapping = getattr(
            self.kv_cache, "full_to_swa_index_mapping", None
        )

    def _full_to_swa(self, full_indices: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Translate full-pool indices to SWA-pool indices.

        Preserves the caller's input dtype.  The peer-access CUDA bindings
        (see ``csrc/binding.cpp``) read token-index tensors as ``int32*`` via
        ``data_ptr<int>()`` with no runtime dtype check.  NCCL gather/scatter
        helpers use torch advanced indexing which needs int64.  Since both
        consumers pass through this helper, preserve whatever dtype the
        caller handed us so each path stays valid.
        """
        if full_indices is None or full_indices.numel() == 0:
            return full_indices
        if self._full_to_swa_mapping is None:
            return full_indices
        original_dtype = full_indices.dtype
        return self._full_to_swa_mapping[full_indices.to(torch.int64)].to(original_dtype)

    def gather_one_layer(self, spec: LayerCacheSpec, **kwargs) -> None:
        layer_id = spec.layer_id
        local_id, _ = self.kv_cache.layers_mapping[layer_id]
        k_buffer = self.kv_cache.swa_kv_pool.k_buffer[local_id]
        v_buffer = self.kv_cache.swa_kv_pool.v_buffer[local_id]

        # Unconditional SWA token capping.
        num_local = min(self.num_local_tokens, spec.tokens_cap_ep)
        layer_global_num = [
            min(n, spec.tokens_cap_ep) for n in self.global_num_tokens
        ]
        num_global = sum(layer_global_num)

        local_indices = (
            self.local_token_indices[:num_local]
            if self.local_token_indices is not None
            and num_local < self.num_local_tokens
            else self.local_token_indices
        )

        # P3 fix: translate local EP indices from full-pool to SWA-pool space
        # before reading from the SWA k/v buffers.
        swa_local = self._full_to_swa(local_indices)

        # P3 fix: build per-rank capped global_token_indices, then translate.
        # self.global_token_indices is the flat concatenation of all EP ranks'
        # tokens (uncapped).  We must slice each rank's chunk to
        # layer_global_num[i] entries before translation.
        if self.global_token_indices is not None and num_global > 0:
            start = 0
            capped_parts = []
            for i, n_full in enumerate(self.global_num_tokens):
                take = layer_global_num[i]
                capped_parts.append(self.global_token_indices[start:start + take])
                start += n_full
            global_indices_full_capped = torch.cat(capped_parts)
            swa_global = self._full_to_swa(global_indices_full_capped)
        else:
            swa_global = self.global_token_indices

        if self.method == "nccl":
            do_gather_one_layer_nccl(
                k_buffer,
                v_buffer,
                num_local,
                num_global,
                swa_local,
                swa_global,
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

            # Peer-access gather: local_token_indices are the READ side
            # (EP buffer positions), so translate to SWA space.
            # The WRITE side uses dst_token_start as a scalar offset into the
            # TP buffer which is addressed by the CUDA kernel directly — no
            # translation needed there.
            do_gather_one_layer_peer_access(
                self._local_buffer_ptr,
                self._peer_addresses_gpu,
                self.mgr._entries[ep_k_name].offset_bytes,
                self.mgr._entries[ep_v_name].offset_bytes,
                self.mgr._entries[tp_k_name].offset_bytes,
                self.mgr._entries[tp_v_name].offset_bytes,
                swa_local,
                num_local,
                self._dst_token_start,
                self.kv_cache.head_num,
                self._tp_rank,
                self.group_size,
                self.kv_cache.head_dim,
                self._elem_size,
            )

    def scatter_one_layer(self, spec: LayerCacheSpec, **kwargs) -> None:
        if self.method == "nccl":
            _do_scatter_one_layer_nccl_swa(
                spec,
                self.kv_cache,
                self.ep_head_num,
                self.token_partition,
                self.group_size,
                self._intra_rank,
                self._replication_factor,
                self._per_token_elems,
                self.global_token_indices,
                self.ep_dst_positions,
                self._recv_full_count,
                self._num_kv_heads,
                self._heads_per_rank,
                self._head_dim,
                self._total_global_tokens,
                self._reassembly_groups,
                self.mgr,
                self.group,
                self._full_to_swa_mapping,
            )
        else:
            tp_k_name = f"model.layers.{spec.layer_id}.kv.tp.k"
            tp_v_name = f"model.layers.{spec.layer_id}.kv.tp.v"
            ep_k_name = f"model.layers.{spec.layer_id}.kv.ep.k"
            ep_v_name = f"model.layers.{spec.layer_id}.kv.ep.v"

            _do_scatter_one_layer_peer_access_swa(
                spec,
                self._local_buffer_ptr,
                self._peer_buffer_ptrs,
                self.mgr._entries[tp_k_name].offset_bytes,
                self.mgr._entries[tp_v_name].offset_bytes,
                self.mgr._entries[ep_k_name].offset_bytes,
                self.mgr._entries[ep_v_name].offset_bytes,
                self._heads_per_rank,
                self._num_kv_heads,
                self.paras_tp_rank,
                self.paras_tp_size,
                self._head_dim,
                self._elem_size,
                self.token_partition,
                self.global_token_indices,
                self.group_size,
                self._full_to_swa_mapping,
            )


__all__ = ["SWACacheTransfer"]
