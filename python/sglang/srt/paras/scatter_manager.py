"""ParaS TP→EP scatter manager.

Contains:
- partition_requests_for_ep  (with extensible strategy pattern)
- gather_tp_kv_and_permute
- permute_and_scatter_kv_to_ep
- ParaSReqScatterManager
"""

from typing import Any, Callable, List, Optional, Union
import os
import torch
import torch.distributed as dist

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.memory_pool import (
    ReqToTokenPool,
    MHATokenToKVPool,
    SWAKVPool,
)
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator, SWATokenToKVPoolAllocator
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.paras.cache_transfer.base import LayerCacheSpec
from sglang.srt.paras.cache_transfer.mha import MHACacheTransfer
from sglang.srt.paras.cache_transfer.swa import SWACacheTransfer
from sglang.srt.paras.paras_memory_manager import get_global_paras_memory_manager
from sglang.srt.paras.gather_manager import paras_tp_group_all_gather_reqs


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
        layer_specs: Optional[list] = None,
        local_waiting_reqs: Optional[List[Req]] = None,
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
        self.layer_specs = layer_specs

        # Only rank 0 receives requests in TP mode, so only rank 0 has a
        # populated waiting_queue. Broadcast to all ranks via all-gather
        # (other ranks send []) so every rank can deterministically run the
        # same partition algorithm.
        local_waiting_reqs = local_waiting_reqs or []
        gathered_waiting, _ = paras_tp_group_all_gather_reqs(
            local_waiting_reqs, scatter_group
        )
        self.global_waiting_reqs: List[Req] = gathered_waiting or []
        self.local_waiting_reqs_after_partition: List[Req] = []

        self.local_reqs: List[Req] = []
        self.local_seqlens_list: List[int] = []
        self.num_local_tokens: int = 0
        self.token_partition: Optional[List[List[int]]] = None
        self.ep_dst_positions: Optional[torch.Tensor] = None
        self.new_cache_size: Optional[int] = None
        self.source_full_to_swa_mapping: Optional[torch.Tensor] = None

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

        waiting_partitions = partition_requests_for_ep(
            self.global_waiting_reqs, self.paras_tp_size
        )
        self.local_waiting_reqs_after_partition = waiting_partitions[self.paras_tp_rank]

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

    def reorchestrate_cache(
        self,
        new_ep_cache_size: Optional[int] = None,
        new_req_pool_size: Optional[int] = None,
        new_ep_cache_size_swa: Optional[int] = None,
    ):
        """Shrink pools to EP capacity and allocate new EP token indices."""
        mgr = get_global_paras_memory_manager()
        if new_req_pool_size is None:
            new_req_pool_size = mgr.get_ep_max_num_reqs()

        is_swa_alloc = isinstance(self.token_to_kv_pool_allocator, SWATokenToKVPoolAllocator)
        if is_swa_alloc:
            if new_ep_cache_size is None:
                new_ep_cache_size = mgr.get_ep_max_kv_tokens()
            if new_ep_cache_size_swa is None:
                new_ep_cache_size_swa = mgr.get_ep_max_kv_tokens("swa")
        else:
            if new_ep_cache_size is None:
                new_ep_cache_size = mgr.get_ep_max_kv_tokens()
        self.new_cache_size = new_ep_cache_size

        num_local_reqs = len(self.local_reqs)
        assert self.num_local_tokens <= new_ep_cache_size, (
            f"Local tokens {self.num_local_tokens} exceed EP cache {new_ep_cache_size}"
        )
        assert num_local_reqs <= new_req_pool_size, (
            f"Local reqs {num_local_reqs} exceed EP req pool {new_req_pool_size}"
        )

        # Snapshot the source-mode (TP) full_to_swa_index_mapping BEFORE resize.
        # paras_resize_and_clear zero-fills this mapping; without a snapshot,
        # SWACacheTransfer.scatter_one_layer's lookup on TP-side source
        # positions would resolve to slot 0 (padding), causing all SWA layers'
        # K/V to read from / write to the padding slot and producing uniformly
        # noisy decode output post-switch. Consumed via _full_to_swa_source.
        if is_swa_alloc:
            self.source_full_to_swa_mapping = (
                self.token_to_kv_pool_allocator.full_to_swa_index_mapping.clone()
            )
        else:
            self.source_full_to_swa_mapping = None

        # Resize and clear allocators.
        self.req_to_token_pool.paras_resize_and_clear(new_req_pool_size)
        if is_swa_alloc:
            self.token_to_kv_pool_allocator.paras_resize_and_clear(new_ep_cache_size, new_ep_cache_size_swa)
        else:
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

    def get_new_waiting_queue(self) -> List[Req]:
        return list(self.local_waiting_reqs_after_partition)

    # ------------------------------------------------------------------
    # Step 3: build running batch from local partition
    # ------------------------------------------------------------------

    def get_new_running_batch(
        self,
        tokenizer: Any,
        tree_cache: Any,
        model_config: Any,
        enable_overlap: bool,
        spec_algorithm: Any,
        enable_custom_logit_processor: bool,
    ):
        from sglang.srt.managers.schedule_batch import ScheduleBatch
        from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
        from sglang.srt.paras.gather_manager import recover_request

        if not self.local_reqs:
            return ScheduleBatch(reqs=[], batch_is_full=False)

        for req in self.local_reqs:
            recover_request(req, tree_cache, tokenizer)

        batch = ScheduleBatch.init_new(
            self.local_reqs,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            tree_cache,
            model_config,
            enable_overlap,
            spec_algorithm,
            enable_custom_logit_processor,
        )

        device = self.req_to_token_pool.device
        last_token_list = []
        for req in self.local_reqs:
            if len(req.output_ids) > 0:
                last_token_list.append(req.output_ids[-1])
            else:
                last_token_list.append(req.origin_input_ids[-1])

        # See ParaSReqGatherManager.get_new_running_batch for the seqlen - 1 invariant.
        req_pool_indices_list = [req.req_pool_idx for req in self.local_reqs]
        seq_lens_list = [req.seqlen - 1 for req in self.local_reqs]

        batch.output_ids = torch.tensor(last_token_list, dtype=torch.int64, device=device)
        batch.req_pool_indices = torch.tensor(req_pool_indices_list, dtype=torch.int64, device=device)
        batch.seq_lens = torch.tensor(seq_lens_list, dtype=torch.int64, device=device)
        batch.seq_lens_cpu = torch.tensor(seq_lens_list, dtype=torch.int64, device="cpu")
        batch.orig_seq_lens = batch.seq_lens.clone()
        batch.seq_lens_sum = sum(seq_lens_list)
        batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            batch, model_config.vocab_size,
        )
        return batch

    # ------------------------------------------------------------------
    # Step 4: scatter KV cache TP → EP
    # ------------------------------------------------------------------

    def scatter_cache(
        self,
        tp_kv_cache: Optional[Union[MHATokenToKVPool, SWAKVPool]] = None,
        ep_head_num: Optional[int] = None,
    ):
        """Transfer KV cache from TP layout to EP layout.

        Pool-type invariant:
            - MHA-only model (no SWA layers): ``tp_kv_cache`` is a ``MHATokenToKVPool``.
            - Hybrid model (any SWA layer): ``tp_kv_cache`` MUST be an ``SWAKVPool``.
              ``SWAKVPool`` is a container holding both ``full_kv_pool`` and
              ``swa_kv_pool`` (each a plain ``MHATokenToKVPool``) plus the
              ``layers_mapping`` that routes per-layer access.
        """
        if tp_kv_cache is None:
            tp_kv_cache = self.token_to_kv_pool_allocator.get_kvcache()

        # `has_swa` requires BOTH that layer_specs labels some layers "swa"
        # AND that the allocator is a hybrid SWAKVPool. In --disable-hybrid-swa-memory
        # mode the model's `paras_layer_specs` may still label some layers SWA
        # (gpt_oss.py builds the full classification unconditionally), but the
        # allocator is a flat MHATokenToKVPool with no inner swa_kv_pool — so
        # SWACacheTransfer cannot run. Fall back to routing every layer
        # (including SWA-classified ones) through MHACacheTransfer.
        specs_have_swa = self.layer_specs is not None and any(
            s.kind == "swa" for s in self.layer_specs
        )
        pool_is_swa = isinstance(tp_kv_cache, SWAKVPool)
        has_swa = specs_have_swa and pool_is_swa
        assert isinstance(tp_kv_cache, (MHATokenToKVPool, SWAKVPool)), (
            f"Expected MHATokenToKVPool or SWAKVPool, "
            f"got {type(tp_kv_cache).__name__}."
        )

        if ep_head_num is None:
            ep_head_num = tp_kv_cache.full_head_num

        if self.num_global_tokens == 0:
            return

        torch.cuda.empty_cache()
        kv_cache = tp_kv_cache
        mgr = get_global_paras_memory_manager()
        method = self.method
        peer_addresses = (
            self.peer_ctx.peer_addresses
            if method == "peer_access" and self.peer_ctx
            else None
        )

        # Construct MHA backend (handles full layers).
        mha_backend = MHACacheTransfer(
            method=method,
            direction="scatter",
            kv_cache=kv_cache,
            mgr=mgr,
            group=self.scatter_group,
            global_token_indices=self.global_token_indices,
            peer_addresses=peer_addresses,
            ep_head_num=ep_head_num,
            token_partition=self.token_partition,
            ep_dst_positions=self.ep_dst_positions,
            paras_tp_rank=self.paras_tp_rank,
            paras_tp_size=self.paras_tp_size,
        )

        # Construct SWA backend (only when hybrid layers present).
        swa_backend = None
        if has_swa:
            swa_backend = SWACacheTransfer(
                method=method,
                direction="scatter",
                kv_cache=kv_cache,
                mgr=mgr,
                group=self.scatter_group,
                global_token_indices=self.global_token_indices,
                layer_specs=self.layer_specs,
                peer_addresses=peer_addresses,
                ep_head_num=ep_head_num,
                token_partition=self.token_partition,
                ep_dst_positions=self.ep_dst_positions,
                paras_tp_rank=self.paras_tp_rank,
                paras_tp_size=self.paras_tp_size,
                source_full_to_swa_mapping=self.source_full_to_swa_mapping,
            )

        # Per-layer dispatch in REVERSE order (preserves N+1 slot invariant).
        num_layers = kv_cache.layer_num
        barrier_tensor = (
            torch.zeros(1, device="cuda") if method == "peer_access" else None
        )

        for layer_id in range(num_layers - 1, -1, -1):
            if self.layer_specs is not None:
                spec = self.layer_specs[layer_id]
            else:
                spec = LayerCacheSpec(
                    layer_id=layer_id,
                    kind="full",
                    tokens_cap_ep=0,
                    tokens_cap_tp=0,
                    num_kv_heads=ep_head_num,
                    head_dim=kv_cache.head_dim,
                    sliding_window_size=None,
                )

            # Route SWA-classified layers to swa_backend ONLY when it exists
            # (SWAKVPool present). Under --disable-hybrid-swa-memory the
            # SWA-classified layers fall back to mha_backend.
            backend = (
                swa_backend if (spec.kind == "swa" and swa_backend is not None)
                else mha_backend
            )
            backend.scatter_one_layer(spec)

            # peer_access path needs per-layer barrier (ALL ranks participate).
            if method == "peer_access":
                dist.all_reduce(
                    barrier_tensor, group=self.scatter_group.device_group
                )

        torch.cuda.synchronize()
