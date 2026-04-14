"""ParaS TP→EP scatter manager.

Moved from gather_manager.py for code organization.  Contains:
- partition_requests_for_ep  (with extensible strategy pattern)
- gather_tp_kv_and_permute
- permute_and_scatter_kv_to_ep
- _EPCacheView
- ParaSReqScatterManager
- _scatter_cache_nccl  (with KV head duplication support)
"""

from typing import Any, Callable, List, Optional
import os
import torch
import torch.distributed as dist

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.memory_pool import (
    ReqToTokenPool,
    MHATokenToKVPool,
)
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.distributed.parallel_state import GroupCoordinator


# ============================================================
# R2: Extensible partition heuristic
# ============================================================

# Type alias for partition strategies
PartitionStrategy = Callable[[List[Req], int], List[List[Req]]]


def _greedy_partition(global_reqs: List[Req], num_ranks: int) -> List[List[Req]]:
    """Greedy partition: sort by (-seqlen, rid), assign to rank with
    fewest requests then least tokens.

    Pure function: identical output on every rank given the same inputs.
    """
    partitions: List[List[Req]] = [[] for _ in range(num_ranks)]
    token_counts: List[int] = [0] * num_ranks

    sorted_reqs = sorted(global_reqs, key=lambda r: (-r.seqlen, r.rid))

    for req in sorted_reqs:
        # Pick the rank with (fewest requests, least tokens, lowest index)
        best_rank = min(
            range(num_ranks),
            key=lambda i: (len(partitions[i]), token_counts[i], i),
        )
        partitions[best_rank].append(req)
        token_counts[best_rank] += req.seqlen

    return partitions


# Registry of available strategies
PARTITION_STRATEGIES: dict[str, PartitionStrategy] = {
    "greedy": _greedy_partition,
}


def partition_requests_for_ep(
    global_reqs: List[Req],
    num_ranks: int,
    strategy: str = "greedy",
) -> List[List[Req]]:
    """Partition requests across EP ranks using the specified strategy.

    Available strategies: ``"greedy"`` (default).
    New strategies can be registered by adding to ``PARTITION_STRATEGIES``.

    Args:
        global_reqs: All requests gathered from all ranks (identical on each).
        num_ranks: Number of EP ranks to partition across.
        strategy: Partition strategy name (default ``"greedy"``).

    Returns:
        List of ``num_ranks`` lists, where ``partition[i]`` holds rank *i*'s
        assigned requests.
    """
    if num_ranks <= 0:
        return []
    fn = PARTITION_STRATEGIES.get(strategy)
    if fn is None:
        raise ValueError(
            f"Unknown partition strategy '{strategy}'. "
            f"Available: {list(PARTITION_STRATEGIES)}"
        )
    return fn(global_reqs, num_ranks)


# ============================================================
# gather_tp_kv_and_permute
# ============================================================

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

    This is the TP→EP counterpart of ``gather_kv_and_permute`` (which outputs
    ``[heads, tokens, KV, dim]`` for EP→TP).

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
    # Interleave K and V → [N, heads_per_rank, 2, head_dim]
    kvcache = torch.stack([kcache, vcache], dim=2)
    return kvcache.contiguous().flatten()


# ============================================================
# permute_and_scatter_kv_to_ep
# ============================================================

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

    This is the TP→EP counterpart of ``permute_and_scatter_kv`` (which handles
    EP→TP).

    Received layout after all_to_all: ``group_size`` contiguous chunks, each
    ``[num_local_tokens, heads_per_rank, 2, head_dim]``.

    Each chunk carries the ``heads_per_rank`` heads owned by a different TP
    rank.  We interleave the head contributions from all ranks to reconstruct
    the full ``num_kv_heads`` for each token, then scatter K and V separately
    into the EP buffer at ``dst_positions``.
    """
    # [group_size, num_local_tokens, heads_per_rank, 2, head_dim]
    kv = recv_buf.view(group_size, num_local_tokens, heads_per_rank, 2, head_dim)
    # Interleave heads from different ranks →
    #   [tokens, group_size, heads_per_rank, 2, dim]
    kv = kv.permute(1, 0, 2, 3, 4).contiguous()
    # Merge rank and per-rank head dims → [tokens, num_kv_heads, 2, dim]
    kv = kv.reshape(num_local_tokens, num_kv_heads, 2, head_dim)
    # Scatter K and V into EP buffers
    k_buffer[dst_positions] = kv[:, :, 0, :]
    v_buffer[dst_positions] = kv[:, :, 1, :]


# ============================================================
# _EPCacheView
# ============================================================

class _EPCacheView:
    """Proxy providing EP-layout buffer access via the ParaS memory manager.

    ``_scatter_cache_nccl`` needs an ``ep_kv_cache`` with EP-shaped buffers
    for the write destination.  The N+1 slot design stores EP data in
    slot[i+1] and TP data in slot[i] at *different* physical offsets.
    This proxy reads the EP aliases from the global ``ParaSMemoryManager``
    to return buffers at the correct physical location.
    """

    def __init__(self, tp_cache: 'MHATokenToKVPool', ep_head_num: int):
        self.head_num = ep_head_num
        self.head_dim = tp_cache.head_dim
        self.layer_num = tp_cache.layer_num
        self.store_dtype = tp_cache.store_dtype
        self.device = tp_cache.device
        self._tp_cache = tp_cache

    def _get_mgr(self):
        from sglang.srt.paras.paras_memory_manager import get_global_paras_memory_manager
        return get_global_paras_memory_manager()

    def get_key_buffer(self, layer_id: int):
        mgr = self._get_mgr()
        ep_k_name = f"model.layers.{layer_id}.kv.ep.k"
        total_elems = mgr._entries[ep_k_name].numel
        ep_tokens = total_elems // (self.head_num * self.head_dim)
        return mgr.get_view_as(ep_k_name, (ep_tokens, self.head_num, self.head_dim))

    def get_value_buffer(self, layer_id: int):
        mgr = self._get_mgr()
        ep_v_name = f"model.layers.{layer_id}.kv.ep.v"
        total_elems = mgr._entries[ep_v_name].numel
        ep_tokens = total_elems // (self.head_num * self.head_dim)
        return mgr.get_view_as(ep_v_name, (ep_tokens, self.head_num, self.head_dim))

    def paras_resize_cache(self, layer_id: int, new_size: int, new_head_num: int):
        self._tp_cache.paras_resize_cache(layer_id, new_size, new_head_num)


# ============================================================
# ParaSReqScatterManager
# ============================================================

# from TP to EP, requests are partitioned back to EP ranks
class ParaSReqScatterManager:
    """Orchestrates TP→EP request and KV cache redistribution.

    Reverse of ``ParaSReqGatherManager``: partitions the global TP request
    set back to EP ranks, shrinks memory pools to EP capacity, and scatters
    KV cache data from TP layout to EP layout.
    """

    scatter_group: GroupCoordinator
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool_allocator: TokenToKVPoolAllocator

    global_reqs: List[Req]
    local_reqs: List[Req]

    def __init__(
        self,
        global_reqs: List[Req],
        scatter_group: GroupCoordinator,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: TokenToKVPoolAllocator,
        peer_ctx: Optional[Any] = None,
        paras_tp_rank: int = 0,
        paras_tp_size: int = 1,
    ):
        self.global_reqs = global_reqs
        self.scatter_group = scatter_group
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.group_size = scatter_group.world_size
        self.peer_ctx = peer_ctx
        self.paras_tp_rank = paras_tp_rank
        self.paras_tp_size = paras_tp_size
        self.method = os.environ.get("PARAS_KV_TRANSFER_METHOD", "nccl")

        self.local_reqs: List[Req] = []
        self.local_seqlens_list: List[int] = []
        self.num_local_tokens: int = 0
        self.token_partition: Optional[List[List[int]]] = None
        self.ep_dst_positions: Optional[torch.Tensor] = None
        self.new_cache_size: Optional[int] = None

        # In TP mode all ranks have identical req_to_token_pool entries.
        self.global_seqlens_list = [req.seqlen for req in global_reqs]
        # Last output token is not stored in KV cache.
        self.num_global_tokens = sum(s - 1 for s in self.global_seqlens_list)

        # Flatten TP pool positions for all global requests (identical on every rank).
        if self.num_global_tokens > 0:
            parts: List[torch.Tensor] = []
            for req in global_reqs:
                indices = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx
                ][: req.seqlen - 1]
                parts.append(indices)
            self.global_token_indices: Optional[torch.Tensor] = torch.cat(parts, dim=0)
            assert self.global_token_indices.shape[0] == self.num_global_tokens, (
                f"global tokens {self.num_global_tokens}, "
                f"global token indices {self.global_token_indices.shape}"
            )
        else:
            self.global_token_indices = None

    # ------------------------------------------------------------------
    # Step 1: partition global requests to EP ranks
    # ------------------------------------------------------------------

    def partition_requests(self):
        """Partition global TP requests to EP ranks and build token routing."""
        partitions = partition_requests_for_ep(
            self.global_reqs, self.paras_tp_size
        )
        self.local_reqs = partitions[self.paras_tp_rank]
        self.local_seqlens_list = [req.seqlen for req in self.local_reqs]
        self.num_local_tokens = sum(s - 1 for s in self.local_seqlens_list)

        # Map each request to its global-token-index range.
        req_to_offset: dict = {}
        offset = 0
        for req in self.global_reqs:
            num_tokens = req.seqlen - 1
            req_to_offset[req.rid] = (offset, offset + num_tokens)
            offset += num_tokens

        # token_partition[e] = list of global token indices for EP rank e.
        self.token_partition = []
        for rank_reqs in partitions:
            rank_indices: List[int] = []
            for req in rank_reqs:
                start, end = req_to_offset[req.rid]
                rank_indices.extend(range(start, end))
            self.token_partition.append(rank_indices)

    # ------------------------------------------------------------------
    # Step 2: shrink pools to EP capacity
    # ------------------------------------------------------------------

    def reorchestrate_cache_reverse(
        self,
        new_ep_cache_size: Optional[int] = None,
        new_req_pool_size: Optional[int] = None,
    ):
        """Shrink pools to EP capacity and allocate new EP token indices."""
        if new_req_pool_size is None:
            new_req_pool_size = self.req_to_token_pool.size // self.group_size
        if new_ep_cache_size is None:
            new_ep_cache_size = self.token_to_kv_pool_allocator.size // self.group_size
        self.new_cache_size = new_ep_cache_size

        num_local_reqs = len(self.local_reqs)
        assert self.num_local_tokens <= new_ep_cache_size, (
            f"Local tokens {self.num_local_tokens} exceed EP cache {new_ep_cache_size}"
        )
        assert num_local_reqs <= new_req_pool_size, (
            f"Local reqs {num_local_reqs} exceed EP req pool {new_req_pool_size}"
        )

        # Resize and clear allocators.
        self.req_to_token_pool.paras_resize_and_clear(new_req_pool_size)
        self.token_to_kv_pool_allocator.paras_resize_and_clear(new_ep_cache_size)

        # Allocate new EP pool indices for local requests.
        if num_local_reqs > 0:
            req_pool_indices = self.req_to_token_pool.alloc(num_local_reqs)

            if self.num_local_tokens > 0:
                ep_token_indices: torch.Tensor = (
                    self.token_to_kv_pool_allocator.alloc(self.num_local_tokens)
                )
                start_index = 0
                for req, rpi in zip(self.local_reqs, req_pool_indices):
                    end_index = start_index + req.seqlen - 1
                    req.req_pool_idx = rpi
                    self.req_to_token_pool.write(
                        (rpi, slice(0, req.seqlen - 1)),
                        ep_token_indices[start_index:end_index],
                    )
                    start_index = end_index

                self.ep_dst_positions = ep_token_indices
            else:
                for req, rpi in zip(self.local_reqs, req_pool_indices):
                    req.req_pool_idx = rpi
                self.ep_dst_positions = None
        else:
            self.ep_dst_positions = None

    # ------------------------------------------------------------------
    # Step 3: scatter KV cache TP → EP
    # ------------------------------------------------------------------

    def scatter_cache(
        self,
        tp_kv_cache: Optional['MHATokenToKVPool'] = None,
        ep_head_num: Optional[int] = None,
    ):
        """Transfer KV cache from TP layout to EP layout.

        Args:
            tp_kv_cache: Source TP KV cache.  If *None*, obtained from
                ``self.token_to_kv_pool_allocator.get_kvcache()``.
            ep_head_num: Full EP head count.  If *None*, read from
                ``tp_kv_cache._paras_original_head_num``.
        """
        if tp_kv_cache is None:
            tp_kv_cache = self.token_to_kv_pool_allocator.get_kvcache()
        assert isinstance(tp_kv_cache, MHATokenToKVPool), (
            "Only MHATokenToKVPool is supported for now."
        )
        if ep_head_num is None:
            ep_head_num = tp_kv_cache._paras_original_head_num

        if self.method == "peer_access" and self.peer_ctx is not None:
            self._scatter_cache_peer_access(tp_kv_cache, ep_head_num)
        else:
            self._scatter_cache_nccl_impl(tp_kv_cache, ep_head_num)

    def _scatter_cache_nccl_impl(
        self,
        kv_cache: 'MHATokenToKVPool',
        ep_head_num: int,
    ):
        torch.cuda.empty_cache()

        # _scatter_cache_nccl expects separate TP/EP cache objects with
        # different head_num.  Create a lightweight EP view.
        ep_view = _EPCacheView(kv_cache, ep_head_num)

        _scatter_cache_nccl(
            tp_kv_cache=kv_cache,
            ep_kv_cache=ep_view,
            token_partition=self.token_partition,
            global_token_indices=self.global_token_indices,
            ep_dst_positions=self.ep_dst_positions,
            gather_group=self.scatter_group,
            new_ep_cache_size=self.new_cache_size,
        )

        # Restore EP head_num (per-layer buffers already EP-shaped after resize).
        kv_cache.paras_configure_ep()

    def _scatter_cache_peer_access(
        self,
        kv_cache: 'MHATokenToKVPool',
        ep_head_num: int,
    ):
        from sglang.srt.paras.peer_access import peer_access_kv_scatter
        from sglang.srt.paras.paras_memory_manager import get_global_paras_memory_manager

        torch.cuda.empty_cache()
        mgr = get_global_paras_memory_manager()

        num_layers = kv_cache.layer_num
        heads_per_rank = kv_cache.head_num          # TP sharded heads
        head_dim = kv_cache.head_dim
        elem_size = (
            kv_cache.store_dtype.itemsize
            if hasattr(kv_cache.store_dtype, "itemsize")
            else 2
        )

        local_buffer_ptr = mgr._buffer.data_ptr()
        peer_buffer_ptrs = torch.tensor(
            self.peer_ctx.peer_addresses, dtype=torch.int64, device="cuda"
        )

        # -- Replication-aware token selection ----------------------------
        # When num_kv_heads < tp_size, multiple ranks share the same head
        # and hold identical KV data.  Each subgroup member only needs to
        # write its 1/R slice of tokens — same optimisation as the NCCL
        # path.  We build routing tensors for only the sliced tokens so
        # the kernel processes fewer items and NVLink traffic drops by R.
        num_kv_heads = ep_head_num
        replication_factor = (
            self.group_size // num_kv_heads
            if num_kv_heads < self.group_size
            else 1
        )
        intra_rank = self.paras_tp_rank % replication_factor

        # Collect this rank's token slice per destination.
        my_global_indices: List[int] = []   # indices into global_token_indices
        my_dst_ranks: List[int] = []
        my_ep_dst_pos: List[int] = []
        for e in range(self.group_size):
            full_tokens = self.token_partition[e]
            full = len(full_tokens)
            my_start = full * intra_rank // replication_factor
            my_end = full * (intra_rank + 1) // replication_factor
            my_slice = full_tokens[my_start:my_end]
            # ep_dst_positions on dest rank e are contiguous [1, 2, ...].
            # This rank's slice maps to positions [my_start+1, my_end].
            for local_idx, global_idx in enumerate(my_slice):
                my_global_indices.append(global_idx)
                my_dst_ranks.append(e)
                my_ep_dst_pos.append(my_start + local_idx + 1)  # 1-indexed

        num_my_tokens = len(my_global_indices)

        if num_my_tokens > 0:
            # TP pool positions for this rank's token slice only.
            gi_tensor = torch.tensor(
                my_global_indices, dtype=torch.long, device="cuda"
            )
            tp_token_positions = self.global_token_indices[gi_tensor].to(
                torch.int32
            )
            token_to_rank = torch.tensor(
                my_dst_ranks, dtype=torch.int32, device="cuda"
            )
            ep_dst_pos_all = torch.tensor(
                my_ep_dst_pos, dtype=torch.int32, device="cuda"
            )
        else:
            tp_token_positions = torch.empty(0, dtype=torch.int32, device="cuda")
            token_to_rank = torch.empty(0, dtype=torch.int32, device="cuda")
            ep_dst_pos_all = torch.empty(0, dtype=torch.int32, device="cuda")

        # Build per-layer byte offsets (source=TP, dest=EP).
        src_k_offsets: List[int] = []
        src_v_offsets: List[int] = []
        dst_k_offsets: List[int] = []
        dst_v_offsets: List[int] = []
        for layer_id in range(num_layers):
            tp_k = f"model.layers.{layer_id}.kv.tp.k"
            tp_v = f"model.layers.{layer_id}.kv.tp.v"
            ep_k = f"model.layers.{layer_id}.kv.ep.k"
            ep_v = f"model.layers.{layer_id}.kv.ep.v"
            src_k_offsets.append(mgr._entries[tp_k].offset_bytes)
            src_v_offsets.append(mgr._entries[tp_v].offset_bytes)
            dst_k_offsets.append(mgr._entries[ep_k].offset_bytes)
            dst_v_offsets.append(mgr._entries[ep_v].offset_bytes)

        # Launch kernel per layer in REVERSE order with a per-layer
        # all_reduce barrier, matching the EP→TP gather pattern.
        # ALL ranks must participate in the barrier regardless of whether
        # they have tokens — otherwise ranks with empty partitions skip
        # the barrier and cause a deadlock.
        import paras_peer_access_cuda

        barrier_tensor = torch.zeros(1, device="cuda")
        for layer_idx in range(num_layers - 1, -1, -1):
            if num_my_tokens > 0:
                paras_peer_access_cuda.launch_peer_access_kv_scatter(
                    local_buffer_ptr,
                    peer_buffer_ptrs,
                    tp_token_positions,
                    token_to_rank,
                    ep_dst_pos_all,
                    src_k_offsets[layer_idx],
                    src_v_offsets[layer_idx],
                    dst_k_offsets[layer_idx],
                    dst_v_offsets[layer_idx],
                    num_my_tokens,
                    heads_per_rank,
                    self.paras_tp_rank,
                    self.paras_tp_size,
                    head_dim,
                    elem_size,
                    0,  # default stream
                )
            # Per-layer barrier: ALL ranks participate
            dist.all_reduce(
                barrier_tensor, group=self.scatter_group.device_group
            )

        torch.cuda.synchronize()

        # Point KV buffers to EP managed entries.
        for layer_id in range(num_layers):
            local_layer_idx = layer_id - kv_cache.start_layer
            ep_k_name = f"model.layers.{layer_id}.kv.ep.k"
            ep_v_name = f"model.layers.{layer_id}.kv.ep.v"
            total_elements = mgr._entries[ep_k_name].numel
            ep_slots = total_elements // (ep_head_num * head_dim)
            ep_shape = (ep_slots, ep_head_num, head_dim)
            kv_cache.k_buffer[local_layer_idx] = mgr.get_view_as(ep_k_name, ep_shape)
            kv_cache.v_buffer[local_layer_idx] = mgr.get_view_as(ep_v_name, ep_shape)

        # Restore EP head_num.
        kv_cache.paras_configure_ep()


# ============================================================
# _scatter_cache_nccl  (standalone function)
# ============================================================

def _scatter_cache_nccl(
    tp_kv_cache: 'MHATokenToKVPool',
    ep_kv_cache: 'MHATokenToKVPool',
    token_partition: List[List[int]],
    global_token_indices: torch.Tensor,
    ep_dst_positions: torch.Tensor,
    gather_group: 'GroupCoordinator',
    new_ep_cache_size: Optional[int] = None,
) -> None:
    """Scatter TP KV cache to EP KV cache via NCCL all_to_all.

    Inverse of ``ParaSReqGatherManager._gather_cache_nccl``: moves from
    TP layout (``heads_per_rank`` heads, all tokens) to EP layout (all
    ``num_kv_heads`` heads, local tokens only).

    Supports KV head duplication (``num_kv_heads < group_size``): when
    multiple TP ranks share the same KV head, each subgroup member sends
    a disjoint token slice, splitting per-rank NVLink traffic by the
    replication factor.

    Three-step flow per layer:
      1. **Gather + permute**: read from TP K/V buffers, group tokens by
         destination EP rank → flat send buffer with layout
         ``[tokens_for_rank, heads_per_rank, KV=2, head_dim]`` per chunk.
      2. **all_to_all_single**: redistribute.  Each rank sends variable-sized
         chunks (different token counts per destination) and receives
         chunks from all source ranks.
      3. **Permute + scatter**: interleave the head contributions from all
         source ranks to reconstruct full ``num_kv_heads`` per token, then
         scatter K/V into EP buffers at ``ep_dst_positions``.

    Args:
        tp_kv_cache: Source TP KV pool (``head_num == heads_per_rank``).
        ep_kv_cache: Destination EP KV pool (``head_num == num_kv_heads``).
        token_partition: ``token_partition[e]`` is a list of global token
            indices (into ``global_token_indices``) assigned to EP rank *e*.
            Identical on every rank.
        global_token_indices: TP pool slot positions for all global tokens.
        ep_dst_positions: EP pool positions for tokens assigned to **this**
            rank (i.e. ``token_partition[tp_rank]``).
        gather_group: Communication group (all TP/EP ranks).
        new_ep_cache_size: If provided, resize EP cache per layer before
            writing.
    """
    torch.cuda.empty_cache()

    group_size = gather_group.world_size
    tp_rank = dist.get_rank(group=gather_group.device_group)

    num_layers = tp_kv_cache.layer_num
    num_kv_heads = ep_kv_cache.head_num      # EP has all heads
    heads_per_rank = tp_kv_cache.head_num    # TP has subset
    head_dim = tp_kv_cache.head_dim

    # -- R3: Compute replication factor for KV head duplication -----------
    if num_kv_heads >= group_size:
        # Standard case: each rank has unique heads
        replication_factor = 1
        assert num_kv_heads == heads_per_rank * group_size, (
            f"Head count mismatch: EP {num_kv_heads} != "
            f"TP {heads_per_rank} * {group_size}"
        )
    else:
        # Duplication case: multiple ranks share the same KV head
        assert group_size % num_kv_heads == 0, (
            f"group_size ({group_size}) must be divisible by "
            f"num_kv_heads ({num_kv_heads}) for head duplication"
        )
        replication_factor = group_size // num_kv_heads
        assert heads_per_rank == 1, (
            f"Expected heads_per_rank=1 with head duplication, "
            f"got {heads_per_rank}"
        )

    per_token_elems = heads_per_rank * 2 * head_dim
    intra_rank = tp_rank % replication_factor  # 0 when R=1

    # Total global tokens — identical on every rank, safe as all_to_all guard
    total_global_tokens = sum(len(token_partition[e]) for e in range(group_size))

    # -- Send side: each rank sends its 1/R token slice per destination --
    # When R=1 (intra_rank=0): my_start=0, my_end=full → sends all tokens.
    # When R>1: sends a disjoint slice, cutting NVLink traffic by R.
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
                my_indices, dtype=torch.long,
                device=global_token_indices.device,
            )
            sorted_parts.append(global_token_indices[part_idx])

    total_send_tokens = sum(send_token_counts)
    input_split_sizes = [cnt * per_token_elems for cnt in send_token_counts]

    if sorted_parts:
        sorted_tp_indices = torch.cat(sorted_parts)
    else:
        sorted_tp_indices = torch.empty(
            0, dtype=torch.long, device=global_token_indices.device
        )

    # -- Recv side: when R=1 all sources send recv_full_count (uniform);
    #    when R>1 subgroup members send variable slices, but each
    #    subgroup's total is exactly recv_full_count (integer-division
    #    identity).  Since subgroups are contiguous in recv_buf, a reshape
    #    to [num_kv_heads, recv_full_count, ...] naturally groups them.
    recv_full_count = len(token_partition[tp_rank])
    recv_token_counts: List[int] = []
    for src in range(group_size):
        src_intra = src % replication_factor
        s = recv_full_count * src_intra // replication_factor
        e_idx = recv_full_count * (src_intra + 1) // replication_factor
        recv_token_counts.append(e_idx - s)

    output_split_sizes = [cnt * per_token_elems for cnt in recv_token_counts]
    total_recv_elems = sum(output_split_sizes)

    # Reassembly groups: when heads_per_rank > 1, each source contributes
    # multiple unique heads → view by source rank.  When heads_per_rank == 1
    # (including the R>1 duplication case), contiguous R sources carry the
    # same head → view by head (num_kv_heads).  When R=1 and hpr=1,
    # num_kv_heads == group_size so both views are identical.
    reassembly_groups = group_size if heads_per_rank > 1 else num_kv_heads

    # -- Per-layer: gather → all_to_all → scatter ------------------------
    def scatter_one_layer(layer_id: int) -> None:
        if total_send_tokens > 0:
            k_buf = tp_kv_cache.get_key_buffer(layer_id)
            v_buf = tp_kv_cache.get_value_buffer(layer_id)
            send_buf = gather_tp_kv_and_permute(
                k_buf, v_buf, sorted_tp_indices,
                num_kv_heads, heads_per_rank, head_dim, group_size,
            )
        else:
            send_buf = torch.empty(
                0, dtype=tp_kv_cache.store_dtype, device=tp_kv_cache.device
            )

        if new_ep_cache_size is not None:
            ep_kv_cache.paras_resize_cache(
                layer_id, new_ep_cache_size, num_kv_heads
            )

        if total_global_tokens > 0:
            recv_buf = torch.empty(
                total_recv_elems,
                dtype=tp_kv_cache.store_dtype,
                device=tp_kv_cache.device,
            )
            dist.all_to_all_single(
                recv_buf, send_buf,
                output_split_sizes, input_split_sizes,
                group=gather_group.device_group,
            )

            if recv_full_count > 0:
                ep_k = ep_kv_cache.get_key_buffer(layer_id)
                ep_v = ep_kv_cache.get_value_buffer(layer_id)
                permute_and_scatter_kv_to_ep(
                    recv_buf, ep_k, ep_v, ep_dst_positions,
                    recv_full_count, num_kv_heads, heads_per_rank,
                    head_dim, reassembly_groups,
                )

    # Process layers in REVERSE order (N-1, ..., 0) to respect the N+1
    # slot design.  Layer i reads TP slot[i] and writes EP slot[i+1].
    # Slot[i+1] is also layer (i+1)'s TP read source, so we must finish
    # reading layer i+1's TP data before writing layer i's EP data.
    for layer_id in reversed(range(num_layers)):
        scatter_one_layer(layer_id)

    torch.cuda.synchronize()
