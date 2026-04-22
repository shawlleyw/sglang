"""Stateless per-layer gather/scatter helpers shared by MHA and SWA backends.

All functions are module-level, stateless, and barrier-free.  Barriers are
the caller's (manager or backend class) responsibility.
"""

from typing import List, Optional, Union

import torch
import torch.distributed as dist


# ------------------------------------------------------------------
# 1. gather_kv_and_permute  (from gather_manager.py:25-41)
# ------------------------------------------------------------------

def gather_kv_and_permute(
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Gather K/V from cache buffers and permute to [heads, tokens, KV, dim].

    Each head's chunk is token-interleaved (t0_K, t0_V, t1_K, t1_V, ...),
    so that after all_to_all splits by head, concatenating received chunks
    gives [total_tokens, KV, heads, dim] which permute_and_scatter_kv expects.
    """
    local_kcache = k_buffer[indices]
    local_vcache = v_buffer[indices]
    local_kvcache = torch.stack([local_kcache, local_vcache], dim=0).view(
        2, -1, k_buffer.shape[1], k_buffer.shape[2]
    )
    return local_kvcache.permute(2, 1, 0, 3).contiguous().flatten()


# ------------------------------------------------------------------
# 2. permute_and_scatter_kv  (from gather_manager.py:44-57)
# ------------------------------------------------------------------

def permute_and_scatter_kv(
    permuted_kvcache: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    indices: torch.Tensor,
    num_tokens: int,
    num_heads: int,
    head_dim: int,
) -> None:
    """Scatter K/V from [total_tokens, KV, heads, dim] layout into cache buffers."""
    kv = permuted_kvcache.view(num_tokens, 2, num_heads, head_dim)
    kv = kv.permute(1, 0, 2, 3).contiguous()
    k_buffer[indices] = kv[0]
    v_buffer[indices] = kv[1]


# ------------------------------------------------------------------
# 3. gather_tp_kv_and_permute  (from scatter_manager.py:96-124)
# ------------------------------------------------------------------

def gather_tp_kv_and_permute(
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    sorted_token_indices: torch.Tensor,
    num_kv_heads: int,
    heads_per_rank: int,
    head_dim: int,
    group_size: int,
) -> torch.Tensor:
    """Gather K/V from TP cache and pack to [tokens, heads_per_rank, KV=2, dim].

    This is the TP->EP counterpart of ``gather_kv_and_permute`` (which outputs
    ``[heads, tokens, KV, dim]`` for EP->TP).

    ``sorted_token_indices`` must be pre-sorted by destination EP rank so
    that the flat output can be split by per-rank token counts for
    ``all_to_all_single``.

    Output layout per destination-rank chunk:
        ``[tokens_for_rank, heads_per_rank, 2, head_dim]``
    """
    if sorted_token_indices.numel() == 0:
        return torch.empty(0, dtype=k_buffer.dtype, device=k_buffer.device)

    kcache = k_buffer[sorted_token_indices]   # [N, heads_per_rank, head_dim]
    vcache = v_buffer[sorted_token_indices]   # [N, heads_per_rank, head_dim]
    # Interleave K and V -> [N, heads_per_rank, 2, head_dim]
    kvcache = torch.stack([kcache, vcache], dim=2)
    return kvcache.contiguous().flatten()


# ------------------------------------------------------------------
# 4. permute_and_scatter_kv_to_ep  (from scatter_manager.py:131-164)
# ------------------------------------------------------------------

def permute_and_scatter_kv_to_ep(
    recv_buf: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    dst_positions: torch.Tensor,
    num_local_tokens: int,
    num_kv_heads: int,
    heads_per_rank: int,
    head_dim: int,
    group_size: int,
) -> None:
    """Unpack all_to_all output and scatter into EP KV buffers.

    This is the TP->EP counterpart of ``permute_and_scatter_kv`` (which handles
    EP->TP).

    Received layout after all_to_all: ``group_size`` contiguous chunks, each
    ``[num_local_tokens, heads_per_rank, 2, head_dim]``.

    Each chunk carries the ``heads_per_rank`` heads owned by a different TP
    rank.  We interleave the head contributions from all ranks to reconstruct
    the full ``num_kv_heads`` for each token, then scatter K and V separately
    into the EP buffer at ``dst_positions``.
    """
    # [group_size, num_local_tokens, heads_per_rank, 2, head_dim]
    kv = recv_buf.view(group_size, num_local_tokens, heads_per_rank, 2, head_dim)
    # Interleave heads from different ranks ->
    #   [tokens, group_size, heads_per_rank, 2, dim]
    kv = kv.permute(1, 0, 2, 3, 4).contiguous()
    # Merge rank and per-rank head dims -> [tokens, num_kv_heads, 2, dim]
    kv = kv.reshape(num_local_tokens, num_kv_heads, 2, head_dim)
    # Scatter K and V into EP buffers
    k_buffer[dst_positions] = kv[:, :, 0, :]
    v_buffer[dst_positions] = kv[:, :, 1, :]


# ------------------------------------------------------------------
# 5. do_gather_one_layer_nccl  (from mha.py:13-88)
# ------------------------------------------------------------------

def do_gather_one_layer_nccl(
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    num_local_tokens: int,
    num_global_tokens: int,
    local_token_indices: torch.Tensor,
    global_token_indices: torch.Tensor,
    global_num_tokens: List[int],
    group_size: int,
    num_heads: int,
    head_dim: int,
    store_dtype: torch.dtype,
    device: Union[str, torch.device],
    mgr,
    layer_id: int,
    gather_group,
) -> None:
    """Stateless NCCL gather for one layer (EP->TP).

    Wraps the all_to_all gather logic.  No barrier inside -- the caller
    is responsible for inter-layer synchronization.
    """
    sharded_num_heads = max(1, num_heads // group_size)
    replication_factor = (
        max(1, group_size // num_heads) if num_heads < group_size else 1
    )
    splited_size_per_token = sharded_num_heads * head_dim

    input_split_sizes = [
        2 * splited_size_per_token * num_local_tokens
    ] * group_size
    output_split_sizes = [
        2 * (splited_size_per_token * n) for n in global_num_tokens
    ]

    if num_local_tokens > 0:
        permuted_local_kvcache = gather_kv_and_permute(
            k_buffer, v_buffer, local_token_indices,
        )
        if replication_factor > 1:
            permuted_local_kvcache = (
                permuted_local_kvcache
                .view(num_heads, num_local_tokens * 2 * head_dim)
                .repeat_interleave(replication_factor, dim=0)
                .flatten()
            )
    else:
        permuted_local_kvcache = torch.empty(
            (0,), dtype=store_dtype, device=device,
        )

    if num_global_tokens > 0:
        gathered_kvcache = torch.empty(
            2 * num_global_tokens * splited_size_per_token,
            dtype=permuted_local_kvcache.dtype,
            device=permuted_local_kvcache.device,
        )
        dist.all_to_all_single(
            gathered_kvcache,
            permuted_local_kvcache,
            output_split_sizes,
            input_split_sizes,
            group=gather_group.device_group,
        )

        tp_k_name = f"model.layers.{layer_id}.kv.tp.k"
        tp_v_name = f"model.layers.{layer_id}.kv.tp.v"
        total_elements = mgr._entries[tp_k_name].numel
        tp_slots = total_elements // (sharded_num_heads * head_dim)
        tp_shape = (tp_slots, sharded_num_heads, head_dim)
        tp_k = mgr.get_view_as(tp_k_name, tp_shape)
        tp_v = mgr.get_view_as(tp_v_name, tp_shape)
        permute_and_scatter_kv(
            gathered_kvcache,
            tp_k,
            tp_v,
            global_token_indices,
            num_global_tokens,
            sharded_num_heads,
            head_dim,
        )


# ------------------------------------------------------------------
# 6. do_gather_one_layer_peer_access  (from gather_manager.py:360-387)
# ------------------------------------------------------------------

def do_gather_one_layer_peer_access(
    local_buffer_ptr: int,
    peer_addresses: torch.Tensor,
    src_k_offset: int,
    src_v_offset: int,
    dst_k_offset: int,
    dst_v_offset: int,
    local_token_indices: torch.Tensor,
    num_local_tokens: int,
    dst_token_start: int,
    num_heads: int,
    tp_rank: int,
    tp_size: int,
    head_dim: int,
    elem_size: int,
) -> None:
    """Stateless peer-access gather for one layer (EP->TP).

    Calls ``peer_access_kv_transfer`` from ``sglang.srt.paras.peer_access``.
    No barrier inside -- the caller is responsible for inter-layer
    synchronization.
    """
    from sglang.srt.paras.peer_access import peer_access_kv_transfer

    if num_local_tokens > 0:
        peer_access_kv_transfer(
            local_buffer_ptr, peer_addresses,
            local_token_indices,
            src_k_offset, src_v_offset,
            dst_k_offset, dst_v_offset,
            num_local_tokens, dst_token_start,
            num_heads, tp_rank, tp_size, head_dim,
            elem_size,
        )


# ------------------------------------------------------------------
# 7. do_scatter_one_layer_nccl  (from scatter_manager.py:705-786)
# ------------------------------------------------------------------

def do_scatter_one_layer_nccl(
    kv_cache,
    ep_head_num: int,
    layer_id: int,
    token_partition: List[List[int]],
    group_size: int,
    intra_rank: int,
    replication_factor: int,
    per_token_elems: int,
    global_token_indices: torch.Tensor,
    ep_dst_positions: Optional[torch.Tensor],
    sorted_tp_indices: torch.Tensor,
    total_send_tokens: int,
    input_split_sizes: List[int],
    recv_full_count: int,
    output_split_sizes: List[int],
    total_recv_elems: int,
    num_kv_heads: int,
    heads_per_rank: int,
    head_dim: int,
    total_global_tokens: int,
    reassembly_groups: int,
    mgr,
    gather_group,
) -> None:
    """Pure full-attention NCCL scatter for one layer (TP->EP).

    SWA cap handling lives in ``swa.py``.  The caller pre-computes all
    partition / split metadata and passes it in.  No barrier inside --
    the caller is responsible for inter-layer synchronization.
    """
    if total_send_tokens > 0:
        k_buf = kv_cache.get_key_buffer(layer_id)
        v_buf = kv_cache.get_value_buffer(layer_id)
        send_buf = gather_tp_kv_and_permute(
            k_buf, v_buf, sorted_tp_indices,
            num_kv_heads, heads_per_rank, head_dim, group_size,
        )
    else:
        send_buf = torch.empty(
            0, dtype=kv_cache.store_dtype, device=kv_cache.device
        )

    if total_global_tokens > 0:
        recv_buf = torch.empty(
            total_recv_elems,
            dtype=kv_cache.store_dtype,
            device=kv_cache.device,
        )
        dist.all_to_all_single(
            recv_buf, send_buf,
            output_split_sizes, input_split_sizes,
            group=gather_group.device_group,
        )

        if recv_full_count > 0:
            ep_k_name = f"model.layers.{layer_id}.kv.ep.k"
            ep_v_name = f"model.layers.{layer_id}.kv.ep.v"
            total_elements = mgr._entries[ep_k_name].numel
            ep_slots = total_elements // (num_kv_heads * head_dim)
            ep_shape = (ep_slots, num_kv_heads, head_dim)
            ep_k = mgr.get_view_as(ep_k_name, ep_shape)
            ep_v = mgr.get_view_as(ep_v_name, ep_shape)
            permute_and_scatter_kv_to_ep(
                recv_buf, ep_k, ep_v, ep_dst_positions,
                recv_full_count, num_kv_heads, heads_per_rank,
                head_dim, reassembly_groups,
            )


# ------------------------------------------------------------------
# 8. do_scatter_one_layer_peer_access  (from scatter_manager.py:531-556)
# ------------------------------------------------------------------

def do_scatter_one_layer_peer_access(
    local_buffer_ptr: int,
    peer_buffer_ptrs: torch.Tensor,
    tp_token_positions: torch.Tensor,
    token_to_rank: torch.Tensor,
    ep_dst_pos_all: torch.Tensor,
    src_k_offset: int,
    src_v_offset: int,
    dst_k_offset: int,
    dst_v_offset: int,
    num_my_tokens: int,
    layer_id: int,
    heads_per_rank: int,
    num_kv_heads: int,
    paras_tp_rank: int,
    paras_tp_size: int,
    head_dim: int,
    elem_size: int,
) -> None:
    """Pure full-attention peer-access scatter for one layer (TP->EP).

    SWA cap handling lives in ``swa.py``.
    Calls ``launch_peer_access_kv_scatter`` from ``paras_peer_access_cuda``.
    No barrier inside -- the caller is responsible for inter-layer
    synchronization.
    """
    import paras_peer_access_cuda

    if num_my_tokens > 0:
        paras_peer_access_cuda.launch_peer_access_kv_scatter(
            local_buffer_ptr,
            peer_buffer_ptrs,
            tp_token_positions[:num_my_tokens],
            token_to_rank[:num_my_tokens],
            ep_dst_pos_all[:num_my_tokens],
            src_k_offset,
            src_v_offset,
            dst_k_offset,
            dst_v_offset,
            num_my_tokens,
            heads_per_rank,
            num_kv_heads,
            paras_tp_rank,
            paras_tp_size,
            head_dim,
            elem_size,
            0,  # default stream
        )
