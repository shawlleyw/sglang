"""ParaS TP→EP scatter manager.

Contains:
- partition_requests_for_ep  (with extensible strategy pattern)
- gather_tp_kv_and_permute
- permute_and_scatter_kv_to_ep
- ParaSReqScatterManager
"""

from typing import Any, Callable, Dict, List, Optional, Union
import os
import torch
import torch.distributed as dist

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import (
    ReqToTokenPool,
    MHATokenToKVPool,
    SWAKVPool,
)
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator, SWATokenToKVPoolAllocator
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.paras.layers.utils import LayerCacheSpec
from sglang.srt.paras.cache_transfer.mha import MHACacheTransfer
from sglang.srt.paras.cache_transfer.swa import SWACacheTransfer
from sglang.srt.paras.paras_memory_manager import get_global_paras_memory_manager
from sglang.srt.paras.gather_manager import paras_tp_group_all_gather_reqs


def recover_request(
    req: Req,
    tree_cache: BasePrefixCache,
    tokenizer: Any,
):
    """Restore prunable fields on a migrated request (TP→EP path).

    With radix-cache migration (T18): after T16's deserialize+rebuild has
    populated this EP rank's partitioned tree, match_prefix on that tree;
    attach req.last_node + prefix_indices on hit. Fallback to root +
    tree_orphaned=True on miss / disable_radix_cache / ChunkCache.
    """
    req.tokenizer = tokenizer
    req.tree_orphaned = False

    if (
        tree_cache is not None
        and getattr(tree_cache, "root_node", None) is not None
        and not getattr(tree_cache, "disable", False)
        and hasattr(req, "fill_ids")
    ):
        try:
            from sglang.srt.mem_cache.radix_cache import RadixKey
            extra_key = getattr(req, "extra_key", None)
            key = RadixKey(list(req.fill_ids), extra_key)
            match = tree_cache.match_prefix(key)
            matched_indices = getattr(match, "device_indices", None)
            last_node = (
                getattr(match, "last_device_node", None)
                or getattr(match, "last_host_node", None)
            )
            if (
                last_node is not None
                and matched_indices is not None
                and len(matched_indices) > 0
            ):
                req.last_node = last_node
                req.last_host_node = getattr(match, "last_host_node", last_node)
                req.prefix_indices = matched_indices
                return
        except Exception:
            pass

    if (
        tree_cache is not None
        and getattr(tree_cache, "root_node", None) is not None
        and not getattr(tree_cache, "disable", False)
    ):
        req.last_node = tree_cache.root_node
        req.last_host_node = tree_cache.root_node
    else:
        req.last_node = None
        req.last_host_node = None
    req.prefix_indices = []
    req.tree_orphaned = True


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
        # T11: per-rank request partitions (populated by partition_requests();
        # used by broadcast_tree_records to filter records by ownership).
        self.partitions: Optional[List[List[Req]]] = None
        self.token_partition: Optional[List[List[int]]] = None
        self.ep_dst_positions: Optional[torch.Tensor] = None
        self.new_cache_size: Optional[int] = None
        self.source_full_to_swa_mapping: Optional[torch.Tensor] = None

        # T7: global old->new slot map for tree-migration deserialize/remap (T16).
        # Built in reorchestrate_cache when a radix tree (not ChunkCache) is active.
        self.old_to_new_slot_map: Optional[Dict[int, int]] = None

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
        self.partitions = partitions
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
        tree_cache=None,  # T7: passed by scheduler for tree-migration; default None preserves caller compat
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

                # T7: build global old->new slot map (TP -> EP direction).
                should_build_map = tree_cache is None or getattr(tree_cache, "root_node", None) is not None
                if should_build_map:
                    if (
                        self.token_partition is not None
                        and self.global_token_indices is not None
                        and self.ep_dst_positions is not None
                    ):
                        local_global_idx = self.token_partition[self.paras_tp_rank]
                        if local_global_idx:
                            old_slots = (
                                self.global_token_indices[local_global_idx]
                                .detach().cpu().tolist()
                            )
                            new_slots = (
                                self.ep_dst_positions.detach().cpu().tolist()
                            )
                            assert len(old_slots) == len(new_slots), (
                                f"Slot map mismatch: {len(old_slots)} vs {len(new_slots)}"
                            )
                            self.old_to_new_slot_map = dict(zip(old_slots, new_slots))
                        else:
                            self.old_to_new_slot_map = {}

                self._tighten_swa_pool_to_in_window()
            else:
                for req, rpi in zip(self.local_reqs, req_pool_indices):
                    req.req_pool_idx = rpi
                self.ep_dst_positions = None
        else:
            self.ep_dst_positions = None

    def _tighten_swa_pool_to_in_window(self) -> None:
        if not isinstance(self.token_to_kv_pool_allocator, SWATokenToKVPoolAllocator):
            return
        if self.layer_specs is None:
            return
        sliding_window_size = next(
            (s.sliding_window_size for s in self.layer_specs
             if s.kind == "swa" and s.sliding_window_size is not None),
            None,
        )
        if sliding_window_size is None:
            return
        for req in self.local_reqs:
            seqlen_no_last = req.seqlen - 1
            in_window_start = max(
                req.swa_evicted_seqlen, seqlen_no_last - sliding_window_size
            )
            if in_window_start > 0:
                oow_full_slots = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, 0:in_window_start
                ]
                self.token_to_kv_pool_allocator.free_swa(oow_full_slots)
                req.swa_evicted_seqlen = in_window_start

    def get_new_waiting_queue(self) -> List[Req]:
        return list(self.local_waiting_reqs_after_partition)

    def build_slot_remap_callback(self):
        """Return a callable that maps old TP-pool slot -> new EP-pool slot.

        Returns -1 when an old slot is unknown (signal: dropped / not migrated).
        Returns identity if no map was built (e.g., ChunkCache path).
        """
        slot_map = self.old_to_new_slot_map
        if slot_map is None:
            return lambda old: old
        return lambda old, _m=slot_map: _m.get(old, -1)

    def broadcast_tree_records(self, local_tree):
        """Rank 0 serializes the canonical TP tree, broadcasts via paras_tp_group.
        Receivers decode and partition by request ownership.

        Args:
            local_tree: this rank's tree_cache (RadixCache or SWARadixCache).
                On rank 0, must be the canonical TP tree. On other ranks, ignored.
                None or ChunkCache means "no tree to broadcast" -> return [].

        Returns:
            List[TreeRecord]: records OWNED by this rank's partition (others dropped).
            A record is OWNED if its full_token_path is a prefix of any req in
            self.partitions[self.paras_tp_rank].
        """
        from sglang.srt.paras.tree_migration import (
            serialize_radix_cache,
            serialize_swa_radix_cache,
            encode_records,
            decode_records,
        )
        import torch.distributed as dist

        if local_tree is None or getattr(local_tree, "root_node", None) is None:
            return []

        rank = self.scatter_group.rank_in_group
        world_size = self.scatter_group.world_size

        if rank == 0:
            if hasattr(local_tree, "sliding_window_size") and getattr(local_tree, "sliding_window_size", None) is not None:
                records = serialize_swa_radix_cache(local_tree)
            else:
                records = serialize_radix_cache(local_tree)
            blob = encode_records(records)
            payload = [blob]
        else:
            payload = [None]

        dist.broadcast_object_list(payload, src=0, group=self.scatter_group.device_group)
        blob = payload[0]
        all_records = decode_records(blob) if blob else []

        if self.partitions is None:
            return all_records

        owned_reqs = self.partitions[self.paras_tp_rank]
        owned_token_lists = [list(req.fill_ids) if hasattr(req, "fill_ids") else [] for req in owned_reqs]

        return self._partition_records_by_ownership(all_records, owned_token_lists)

    @staticmethod
    def _partition_records_by_ownership(records, owned_token_lists):
        """A record is owned if its full_token_path is a prefix of any owned_token_list."""
        owned: list = []
        for rec in records:
            path = rec.full_token_path
            path_len = len(path)
            for tokens in owned_token_lists:
                if len(tokens) >= path_len and tokens[:path_len] == path:
                    owned.append(rec)
                    break
        return owned

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
