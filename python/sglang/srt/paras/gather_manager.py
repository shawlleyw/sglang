from typing import Callable, Dict, List, Optional, Tuple, Any
import os
import torch
import pickle
import numpy as np
import torch.distributed as dist

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.managers.schedule_batch import (
    Req, 
    ScheduleBatch, 
)
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool, MHATokenToKVPool, SWAKVPool
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator, SWATokenToKVPoolAllocator
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.distributed.parallel_state import GroupCoordinator

from sglang.srt.paras.cache_transfer.utils import (
    gather_kv_and_permute,
    permute_and_scatter_kv,
)

from sglang.srt.paras.paras_memory_manager import get_global_paras_memory_manager
from sglang.srt.paras.layers.utils import LayerCacheSpec
from sglang.srt.paras.cache_transfer.mha import MHACacheTransfer
from sglang.srt.paras.cache_transfer.swa import SWACacheTransfer

def prune_request(req: Req):
    req.last_host_node = None
    req.last_node = None
    req.prefix_indices = None
    req.tokenizer = None
    
def recover_request(
    req: Req,
    tree_cache: BasePrefixCache,
    tokenizer: Any,
):
    """Restore prunable fields on a migrated request.

    With radix-cache migration (T17): if the post-migration tree contains a
    prefix matching this req, attach req.last_node + prefix_indices to that
    matched node. Otherwise (or if disable_radix_cache / ChunkCache path),
    fall back to root + tree_orphaned=True.
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
            last_node = getattr(match, "last_device_node", None) or getattr(match, "last_host_node", None)
            if last_node is not None and matched_indices is not None and len(matched_indices) > 0:
                req.last_node = last_node
                req.last_host_node = getattr(match, "last_host_node", last_node)
                req.prefix_indices = matched_indices
                return
        except Exception:
            pass

    if tree_cache is not None and getattr(tree_cache, "root_node", None) is not None and not getattr(tree_cache, "disable", False):
        req.last_node = tree_cache.root_node
        req.last_host_node = tree_cache.root_node
    else:
        req.last_node = None
        req.last_host_node = None
    req.prefix_indices = []
    req.tree_orphaned = True

def paras_tp_group_all_gather_reqs(
    reqs: List[Req],
    group: GroupCoordinator,
) -> Tuple[List[Req], List[int]]:
    device = torch.device("cuda")
    
    num_ranks = group.world_size
    
    # Clean up tensor members to avoid pickle triggering torch device copy, mostly radix cache related stuff
    for req in reqs:
        prune_request(req)

    
    serialized_data = pickle.dumps(reqs)
    size = len(serialized_data)
    tensor_data = torch.ByteTensor(
        np.frombuffer(serialized_data, dtype=np.uint8),
        device="cpu",
    )
    tensor_size = torch.tensor([size], dtype=torch.long, device=device)
    
    gathered_size = torch.empty(num_ranks, dtype=torch.long, device=device)
    group.all_gather_into_tensor(gathered_size, tensor_size)
    gathered_size_list = gathered_size.tolist()
    max_size = max(gathered_size_list)
    if max_size == 0:
        return None, None
    
    padded_tensor_data = torch.empty((max_size,), dtype=torch.uint8, device=device)
    padded_tensor_data[:size].copy_(tensor_data)

    gathered_data: torch.Tensor = torch.empty((max_size * num_ranks), dtype=torch.uint8, device=device)
    group.all_gather_into_tensor(gathered_data, padded_tensor_data)
    
    serialized_data_per_rank = np.split(gathered_data.cpu().numpy(), num_ranks, axis=0)
    
    gathered_reqs = []
    split_sizes = []
    for i in range(num_ranks):
        data = serialized_data_per_rank[i]
        effective_size = gathered_size_list[i]
        remote_reqs = pickle.loads(data[:effective_size]) if effective_size > 0 else []
        gathered_reqs.extend(remote_reqs)
        split_sizes.append(len(remote_reqs))
        
    print(f"metadata sizes: {gathered_size_list}, num_gathered_reqs: {len(gathered_reqs)}, split_sizes: {split_sizes}")
    
    return gathered_reqs, split_sizes

# from EP to TP, requests are all-gathered from all ranks
class ParaSReqGatherManager:
    
    gather_group: GroupCoordinator
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool_allocator: TokenToKVPoolAllocator
    
    local_reqs: List[Req]
    local_seqlens_list: List[int]
    local_token_indices: torch.Tensor
    
    global_reqs: List[Req]
    global_seqlens_list: List[int]
    global_token_indices: torch.Tensor
    
    def __init__(
        self, 
        local_reqs: List[Req],
        gather_group: GroupCoordinator,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: TokenToKVPoolAllocator,
        peer_ctx: Optional[Any] = None,
        method: Optional[str] = None,
        layer_specs: Optional[list] = None,
        local_waiting_reqs: Optional[List[Req]] = None,
    ):
        self.local_reqs = local_reqs
        self.local_waiting_reqs = local_waiting_reqs or []
        self.gather_group = gather_group
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.group_size = gather_group.world_size
        self.peer_ctx = peer_ctx
        self.method = method or os.environ.get("PARAS_KV_TRANSFER_METHOD", "nccl")
        self.layer_specs = layer_specs
        
        self.local_no_reqs = len(local_reqs) == 0
        self.local_seqlens_list = [req.seqlen for req in local_reqs]
        self.num_local_tokens = sum(self.local_seqlens_list) - len(local_reqs) # the last output token is not stored in kv cache

        self.source_full_to_swa_mapping: Optional[torch.Tensor] = None

        # T7: global old->new slot map for tree-migration deserialize/remap (T15).
        # Built in reorchestrate_cache when a radix tree (not ChunkCache) is active.
        self.old_to_new_slot_map: Optional[Dict[int, int]] = None
        
        if self.local_no_reqs:
            self.local_token_indices = None
        else:
            req_to_token_indices = []
            for req in local_reqs:
                indices = self.req_to_token_pool.req_to_token[req.req_pool_idx][ : req.seqlen - 1]
                req_to_token_indices.append(indices)

            self.local_token_indices = torch.cat(req_to_token_indices, dim=0)
            assert self.local_token_indices.shape[0] == self.num_local_tokens, \
                f"local tokens {self.num_local_tokens}, local token indices {self.local_token_indices.shape}"
    
    def gather_global_reqs(self):
        self.global_reqs, self.global_reqs_split_sizes = paras_tp_group_all_gather_reqs(self.local_reqs, self.gather_group)
        self.global_seqlens_list = [req.seqlen for req in self.global_reqs]
        
        start_index = 0
        self.global_num_tokens = []
        for split_size in self.global_reqs_split_sizes:
            end_index = start_index + split_size
            self.global_num_tokens.append(sum(self.global_seqlens_list[start_index:end_index]) - split_size)
            start_index = end_index
            
        self.num_global_tokens = sum(self.global_num_tokens)

        # Gather waiting-queue requests too. They have no KV cache yet — just
        # plain Python objects — so the same all-gather mechanism preserves
        # them across the EP->TP switch without any GPU memory transfer.
        gathered_waiting, _ = paras_tp_group_all_gather_reqs(
            self.local_waiting_reqs, self.gather_group
        )
        self.global_waiting_reqs = gathered_waiting or []

    def get_new_waiting_queue(self, paras_tp_rank: int) -> List[Req]:
        if paras_tp_rank == 0:
            return list(self.global_waiting_reqs)
        return []
        
    def reorchestrate_cache(
        self, 
        new_req_pool_size: Optional[int] = None,
        new_cache_size: Optional[int] = None,
        new_cache_size_swa: Optional[int] = None,
        tree_cache=None,  # T7: passed by scheduler for tree-migration; default None preserves caller compat
    ):
        '''
        Resize request and KV pools to the TP capacities planned by UMM.
        '''
        mgr = get_global_paras_memory_manager()
        if new_req_pool_size is None:
            new_req_pool_size = mgr.get_tp_max_num_reqs()

        is_swa_alloc = isinstance(self.token_to_kv_pool_allocator, SWATokenToKVPoolAllocator)
        if is_swa_alloc:
            if new_cache_size is None:
                new_cache_size = mgr.get_tp_max_kv_tokens()
            if new_cache_size_swa is None:
                new_cache_size_swa = mgr.get_tp_max_kv_tokens("swa")
        else:
            if new_cache_size is None:
                new_cache_size = mgr.get_tp_max_kv_tokens()
        
        assert self.num_global_tokens <= new_cache_size, "The total size of the requests to reorchestrate is greater than the new size of the cache."
        
        self.new_req_pool_size = new_req_pool_size
        self.new_cache_size = new_cache_size
        
        num_reqs = len(self.global_reqs)
        assert num_reqs <= new_req_pool_size, "The number of requests to reorchestrate is greater than the new size of the request to token pool."
        self.req_to_token_pool.paras_resize_and_clear(new_req_pool_size)
        req_pool_indices = self.req_to_token_pool.alloc(num_reqs)

        # INVARIANT (T26): snapshot MUST precede paras_resize_and_clear (which would
        # zero the mapping) AND _tighten_swa_pool_to_in_window (which would free
        # tree-only OOW slots). Future refactors that reorder these will break
        # tree-migration's SWA layer remap.
        # Snapshot the source-mode (EP) full_to_swa_index_mapping BEFORE resize.
        # paras_resize_and_clear zero-fills this mapping; without a snapshot,
        # SWACacheTransfer.gather_one_layer's lookup on EP-side source positions
        # would resolve to slot 0 (padding), corrupting SWA layer K/V transport
        # on the EP->TP direction. Consumed via _full_to_swa_source.
        if is_swa_alloc:
            self.source_full_to_swa_mapping = (
                self.token_to_kv_pool_allocator.full_to_swa_index_mapping.clone()
            )
            self.token_to_kv_pool_allocator.paras_resize_and_clear(new_cache_size, new_cache_size_swa)
        else:
            self.source_full_to_swa_mapping = None
            self.token_to_kv_pool_allocator.paras_resize_and_clear(new_cache_size)

        if self.num_global_tokens > 0:        
            global_token_indices: torch.Tensor = self.token_to_kv_pool_allocator.alloc(self.num_global_tokens)
            start_index = 0
            # TODO: optimize writing to req_to_token_pool and token_to_kv_pool_allocator
            for req, req_pool_idx in zip(self.global_reqs, req_pool_indices):
                end_index = start_index + req.seqlen - 1
                req.req_pool_idx = req_pool_idx
                token_indices = global_token_indices[start_index:end_index]
                self.req_to_token_pool.write((req_pool_idx, slice(0, req.seqlen - 1)), token_indices)
                start_index = end_index
                
            self.global_token_indices = global_token_indices
            assert self.global_token_indices.shape[0] == self.num_global_tokens, "The number of global tokens is not equal to the number of tokens in the global requests."

            # T7: build global old->new slot map for tree-migration remap.
            # Skip for ChunkCache / SWAChunkCache (no tree topology to migrate).
            # ChunkCache lacks root_node attribute; getattr fallback handles it.
            should_build_map = tree_cache is None or getattr(tree_cache, "root_node", None) is not None
            if should_build_map:
                # Compute this rank's offset into global_token_indices.
                # The local rank's portion starts at sum of preceding ranks' token counts.
                rank_in_group = self.gather_group.rank_in_group
                local_offset = sum(self.global_num_tokens[:rank_in_group])
                if self.num_local_tokens > 0 and self.local_token_indices is not None:
                    local_slots_cpu = self.local_token_indices.detach().cpu().tolist()
                    new_slots_cpu = (
                        self.global_token_indices[
                            local_offset : local_offset + self.num_local_tokens
                        ]
                        .detach()
                        .cpu()
                        .tolist()
                    )
                    assert len(local_slots_cpu) == len(new_slots_cpu), (
                        f"Slot map length mismatch: "
                        f"{len(local_slots_cpu)} vs {len(new_slots_cpu)}"
                    )
                    self.old_to_new_slot_map = dict(zip(local_slots_cpu, new_slots_cpu))
                else:
                    self.old_to_new_slot_map = {}

            self._tighten_swa_pool_to_in_window()
        else:
            self.global_token_indices = None

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
        for req in self.global_reqs:
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

    def gather_cache(self) -> None:
        """Transfer KV cache from EP layout to TP layout.

        Pool-type invariant:
            - MHA-only model (no SWA layers): ``kv_cache`` is a ``MHATokenToKVPool``.
            - Hybrid model (any SWA layer): ``kv_cache`` MUST be an ``SWAKVPool``.
              ``SWAKVPool`` is a container holding both ``full_kv_pool`` and
              ``swa_kv_pool`` (each a plain ``MHATokenToKVPool``) plus the
              ``layers_mapping`` that routes per-layer access.
        """
        torch.cuda.empty_cache()
        kv_cache = self.token_to_kv_pool_allocator.get_kvcache()
        mgr = get_global_paras_memory_manager()
        method = self.method

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
        pool_is_swa = isinstance(kv_cache, SWAKVPool)
        has_swa = specs_have_swa and pool_is_swa
        assert isinstance(kv_cache, (MHATokenToKVPool, SWAKVPool)), (
            f"Expected MHATokenToKVPool or SWAKVPool, "
            f"got {type(kv_cache).__name__}."
        )
        peer_addresses = (
            self.peer_ctx.peer_addresses
            if method == "peer_access" and self.peer_ctx
            else None
        )

        # Construct MHA backend (handles full layers, or all layers when no SWA).
        mha_backend = MHACacheTransfer(
            method=method,
            direction="gather",
            kv_cache=kv_cache,
            mgr=mgr,
            group=self.gather_group,
            num_local_tokens=self.num_local_tokens,
            num_global_tokens=self.num_global_tokens,
            local_token_indices=self.local_token_indices,
            global_token_indices=self.global_token_indices,
            global_num_tokens=self.global_num_tokens,
            peer_addresses=peer_addresses,
        )

        # Construct SWA backend (only when hybrid layers present).
        swa_backend = None
        if has_swa:
            swa_backend = SWACacheTransfer(
                method=method,
                direction="gather",
                kv_cache=kv_cache,
                mgr=mgr,
                group=self.gather_group,
                num_local_tokens=self.num_local_tokens,
                num_global_tokens=self.num_global_tokens,
                local_token_indices=self.local_token_indices,
                global_token_indices=self.global_token_indices,
                global_num_tokens=self.global_num_tokens,
                layer_specs=self.layer_specs,
                peer_addresses=peer_addresses,
                source_full_to_swa_mapping=self.source_full_to_swa_mapping,
            )

        # Per-layer dispatch.
        num_layers = kv_cache.layer_num
        barrier_tensor = (
            torch.zeros(1, device="cuda") if method == "peer_access" else None
        )

        for layer_id in range(num_layers):
            if self.layer_specs is not None:
                spec = self.layer_specs[layer_id]
            else:
                spec = LayerCacheSpec(
                    layer_id=layer_id,
                    kind="full",
                    tokens_cap_ep=self.num_local_tokens,
                    tokens_cap_tp=0,
                    num_kv_heads=kv_cache.head_num,
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
            backend.gather_one_layer(spec)

            # peer_access path needs per-layer barrier (ALL ranks participate).
            if method == "peer_access":
                dist.all_reduce(
                    barrier_tensor, group=self.gather_group.device_group
                )

        torch.cuda.synchronize()

    def get_new_running_batch(
        self,
        tokenizer: Any,
        tree_cache: BasePrefixCache,
        model_config: ModelConfig,
        enable_overlap: bool,
        spec_algorithm: SpeculativeAlgorithm,
        enable_custom_logit_processor: bool,
    ) -> ScheduleBatch:
        """
        Create a new ScheduleBatch from the global requests for decode mode.
        All requests are assumed to be in decode status (already have output_ids).
        """
        # Create a ScheduleBatch using init_new
        # tree_cache should be reset to empty state before calling this function
        
        for req in self.global_reqs:
            recover_request(req, tree_cache, tokenizer)

        batch = ScheduleBatch.init_new(
            self.global_reqs,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            tree_cache,
            model_config,
            enable_overlap,
            spec_algorithm,
            enable_custom_logit_processor,
        )
        
        # Set up decode batch fields
        bs = len(self.global_reqs)
        device = self.req_to_token_pool.device
        
        # Get the last token from each request (for decode input)
        # For decode, input_ids should be the last output token, or last input token if no output yet
        last_token_list = []
        for req in self.global_reqs:
            if len(req.output_ids) > 0:
                last_token_list.append(req.output_ids[-1])
            else:
                # No output yet, use last input token
                last_token_list.append(req.origin_input_ids[-1])
        
        # Get req_pool_indices and seq_lens.
        # SGLang convention: batch.seq_lens = K/V cache history length, which is
        # req.seqlen - 1 because req.output_ids[-1] is the most recently sampled
        # token but its K/V is not yet stored (it will be written when this token
        # becomes the input on the next decode iteration via alloc_for_decode).
        req_pool_indices_list = [req.req_pool_idx for req in self.global_reqs]
        seq_lens_list = [req.seqlen - 1 for req in self.global_reqs]

        # Convert to tensors
        batch.output_ids = torch.tensor(last_token_list, dtype=torch.int64, device=device)
        batch.req_pool_indices = torch.tensor(req_pool_indices_list, dtype=torch.int64, device=device)
        batch.seq_lens = torch.tensor(seq_lens_list, dtype=torch.int64, device=device)
        batch.seq_lens_cpu = torch.tensor(seq_lens_list, dtype=torch.int64, device="cpu")
        batch.orig_seq_lens = batch.seq_lens.clone()
        batch.seq_lens_sum = sum(seq_lens_list)
        
        # Create sampling_info before prepare_for_decode (it's required by prepare_for_decode)
        batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            batch,
            model_config.vocab_size,
        )
        return batch

    def update_running_batch_inplace(
        self,
        running_batch: ScheduleBatch,
    ) -> None:
        """
        Directly update the existing running batch with global requests (in-place modification).
        This is simpler than creating a new batch as it preserves existing batch state.
        All requests are assumed to be in decode status (already have output_ids).
        """
        model_config = running_batch.model_config
        device = self.req_to_token_pool.device
        
        # Update requests and metadata (matching init_new behavior)
        running_batch.reqs = self.global_reqs
        running_batch.return_logprob = any(req.return_logprob for req in self.global_reqs)
        running_batch.has_stream = any(req.stream for req in self.global_reqs)
        running_batch.has_grammar = any(req.grammar for req in self.global_reqs)
        running_batch.return_hidden_states = any(req.return_hidden_states for req in self.global_reqs)
        running_batch.chunked_req = None
        
        if running_batch.tree_cache.disable:
            last_node = None
        else:
            last_node = running_batch.tree_cache.root_node
        for req in running_batch.reqs:
            req.last_node = last_node
        
        # Get the last token from each request (for decode input)
        input_ids_list = []
        for req in self.global_reqs:
            if len(req.output_ids) > 0:
                input_ids_list.append(req.output_ids[-1])
            else:
                # No output yet, use last input token
                input_ids_list.append(req.origin_input_ids[-1])
        
        # See get_new_running_batch: batch.seq_lens excludes the last output
        # token whose K/V is not yet stored (it will be written when that token
        # becomes the input on the next decode iteration via alloc_for_decode).
        req_pool_indices_list = [req.req_pool_idx for req in self.global_reqs]
        seq_lens_list = [req.seqlen - 1 for req in self.global_reqs]

        # Update batch tensors
        running_batch.input_ids = torch.tensor(input_ids_list, dtype=torch.int64, device=device)
        running_batch.req_pool_indices = torch.tensor(req_pool_indices_list, dtype=torch.int64, device=device)
        running_batch.seq_lens = torch.tensor(seq_lens_list, dtype=torch.int64, device=device)
        running_batch.seq_lens_cpu = torch.tensor(seq_lens_list, dtype=torch.int64, device="cpu")
        running_batch.orig_seq_lens = running_batch.seq_lens.clone()
        running_batch.seq_lens_sum = sum(seq_lens_list)
        
        # Set output_ids to input_ids (will be used by prepare_for_decode)
        running_batch.output_ids = running_batch.input_ids.clone()
        
        # Recreate sampling_info with new requests
        running_batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            running_batch,
            model_config.vocab_size,
        )

    def build_slot_remap_callback(self):
        """Return a callable that maps old EP-pool slot -> new TP-pool slot.

        Returns -1 when an old slot is unknown (signal: dropped / not migrated).
        Returns identity if no map was built (e.g., ChunkCache path).
        """
        slot_map = self.old_to_new_slot_map
        if slot_map is None:
            return lambda old: old
        return lambda old, _m=slot_map: _m.get(old, -1)

    def gather_tree_records(
        self,
        local_tree,
        in_flight_slot_set,
    ):
        """Serialize local tree, all-gather across paras_tp_group, dedup.

        Args:
            local_tree: this rank's RadixCache or SWARadixCache (pre-tree.reset()).
                None or ChunkCache means "no tree to gather" -> return ([], 0).
            in_flight_slot_set: set[int] of pool slots currently held by in-flight
                requests on this rank. Used as the lock-ref tiebreaker signal.

        Returns:
            Tuple (deduped_records: List[TreeRecord], dropped_count: int)
            dropped_count: number of records lost to dedup.

        Notes:
            - Tiebreaker (Metis Q5): on (token_path, extra_key) collision, prefer
              the record whose value_slots intersects in_flight_slot_set
              (any rank). If none of the colliding records has an in-flight slot,
              fall back to lex-min source rank order.
        """
        from sglang.srt.paras.tree_migration import (
            serialize_radix_cache,
            serialize_swa_radix_cache,
            encode_records,
            decode_records,
        )

        if local_tree is None or getattr(local_tree, "root_node", None) is None:
            return [], 0

        if hasattr(local_tree, "sliding_window_size") and getattr(local_tree, "sliding_window_size", None) is not None:
            local_records = serialize_swa_radix_cache(local_tree)
        else:
            local_records = serialize_radix_cache(local_tree)

        local_blob = encode_records(local_records)

        world_size = self.gather_group.world_size
        gathered_blobs = [None] * world_size
        dist.all_gather_object(gathered_blobs, local_blob, group=self.gather_group.device_group)

        per_rank_records = []
        for rank_idx, blob in enumerate(gathered_blobs):
            if blob is None:
                per_rank_records.append([])
                continue
            per_rank_records.append(decode_records(blob))

        return self._dedup_records_with_lockref(per_rank_records, in_flight_slot_set)

    def _dedup_records_with_lockref(self, per_rank_records, in_flight_slot_set):
        """Dedup by (full_token_path, extra_key); tiebreaker: prefer in-flight-held slots.

        Returns (kept_records, dropped_count).
        """
        from collections import defaultdict
        bucket = defaultdict(list)
        for rank_idx, records in enumerate(per_rank_records):
            for r in records:
                key = (tuple(r.full_token_path), r.extra_key)
                bucket[key].append((rank_idx, r))

        kept = []
        dropped = 0
        for key, candidates in bucket.items():
            if len(candidates) == 1:
                kept.append(candidates[0][1])
                continue

            with_in_flight = [
                (rank_idx, r)
                for rank_idx, r in candidates
                if any(s in in_flight_slot_set for s in r.value_slots)
            ]
            if with_in_flight:
                with_in_flight.sort(key=lambda pair: pair[0])
                kept.append(with_in_flight[0][1])
                dropped += len(candidates) - 1
            else:
                candidates.sort(key=lambda pair: pair[0])
                kept.append(candidates[0][1])
                dropped += len(candidates) - 1
        return kept, dropped
