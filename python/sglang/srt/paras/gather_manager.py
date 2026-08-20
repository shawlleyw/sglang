from typing import List, Optional, Tuple, Any
import gc
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
    if tree_cache.disable:
        req.last_host_node = None
        req.last_node = None
    else:
        req.last_host_node = tree_cache.root_node
        req.last_node = tree_cache.root_node
    req.prefix_indices = []
    req.tokenizer = tokenizer

def paras_tp_group_all_gather_reqs(
    reqs: List[Req],
    group: GroupCoordinator,
) -> Tuple[Optional[List[Req]], Optional[List[int]]]:
    # The 8 x pickle.loads loop below allocates ~2,048 fresh Req objects
    # (256 reqs x 8 ranks, each with nested fields). That allocation rate
    # trips CPython's gen-2 threshold and the resulting heap walk over the
    # long-lived serving heap (radix nodes, req pools, sampler state, etc.)
    # stalls the GIL for ~1.5-2 s with NO Python frame on the stack (so
    # Kineto sees a silent gap). Disable GC for the duration of the call;
    # gc.collect(0) at the end sweeps the young generation cheaply
    # (O(new_objects), not O(heap_size)) so gen0 does not accumulate across
    # switches. Root cause: INVESTIGATION_paras_gather_reqs_slow.md.
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
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
    finally:
        if gc_was_enabled:
            gc.enable()
        gc.collect(0)

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

    def get_new_waiting_queue(self) -> List[Req]:
        # TP-mode invariant: every TP rank must hold an identical waiting_queue
        # so that the next iteration's get_next_batch_to_run picks the same
        # prefill-vs-decode batch on all ranks. Returning the global list only
        # on rank 0 desynchronizes the event loop and deadlocks the first TP
        # collective in run_batch (cf. scatter_manager.get_new_waiting_queue).
        return list(self.global_waiting_reqs)

    def precheck_tp_capacity(
        self,
        new_req_pool_size: Optional[int] = None,
        new_cache_size: Optional[int] = None,
        new_cache_size_swa: Optional[int] = None,
    ) -> Tuple[bool, str, dict]:
        """Read-only precheck for ``reorchestrate_cache``.

        For models where ``num_kv_heads < paras_tp_size`` (e.g. Qwen3-235B has
        ``num_kv_heads=4`` at ``paras_tp_size=8``), KV heads are replicated
        across ranks in TP mode, which roughly halves the effective KV-cache
        capacity vs the EP-mode pool. Under heavy in-flight load, an EP->TP
        switch can therefore overflow the smaller TP pool and the existing
        asserts at the top of ``reorchestrate_cache`` crash the scheduler
        irrecoverably (observed in the N=2048 paras-t128 rollout sweep).

        Must be called AFTER ``gather_global_reqs`` (so ``self.num_global_tokens``
        and ``self.global_reqs`` are populated) and BEFORE
        ``reorchestrate_cache`` (so the caller can abort the switch gracefully
        instead of crashing inside the destructive pool resize).

        Mirrors the capacity computation in ``reorchestrate_cache`` exactly so
        the precheck is a faithful predicate over the same asserts.

        Returns:
            ok: True if the planned TP pools can hold the in-flight workload.
            reason: empty string when ok; human-readable explanation otherwise.
            details: dict with ``num_global_tokens``, ``new_cache_size``,
                ``num_reqs``, ``new_req_pool_size`` for structured logging by
                the caller.

        Does NOT mutate any pool state.
        """
        mgr = get_global_paras_memory_manager()
        if new_req_pool_size is None:
            new_req_pool_size = mgr.get_tp_max_num_reqs()

        is_swa_alloc = isinstance(
            self.token_to_kv_pool_allocator, SWATokenToKVPoolAllocator
        )
        if is_swa_alloc:
            if new_cache_size is None:
                new_cache_size = mgr.get_tp_max_kv_tokens()
            if new_cache_size_swa is None:
                new_cache_size_swa = mgr.get_tp_max_kv_tokens("swa")
        else:
            if new_cache_size is None:
                new_cache_size = mgr.get_tp_max_kv_tokens()

        num_reqs = len(self.global_reqs)
        details = {
            "num_global_tokens": int(self.num_global_tokens),
            "new_cache_size": int(new_cache_size),
            "num_reqs": int(num_reqs),
            "new_req_pool_size": int(new_req_pool_size),
        }

        # Mirror reorchestrate_cache asserts (KV pool + req pool).
        if self.num_global_tokens > new_cache_size:
            return False, (
                f"in-flight KV total ({self.num_global_tokens} tokens) exceeds "
                f"TP-mode cache capacity ({new_cache_size} tokens)"
            ), details

        if num_reqs > new_req_pool_size:
            return False, (
                f"in-flight req count ({num_reqs}) exceeds TP-mode req pool "
                f"size ({new_req_pool_size})"
            ), details

        return True, "", details

    def reorchestrate_cache(
        self, 
        new_req_pool_size: Optional[int] = None,
        new_cache_size: Optional[int] = None,
        new_cache_size_swa: Optional[int] = None,
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

            self._tighten_swa_pool_to_in_window()
        else:
            self.global_token_indices = None

    def _tighten_swa_pool_to_in_window(self) -> None:
        # Drop the SWA backing for every req's out-of-window prefix in a single
        # batched gather + free_swa, rather than scanning bs req CUDA gathers
        # serially. At running=2048 the per-req version takes ~47 ms (~23 us/req,
        # dominated by per-slice CUDA launches + an implicit host sync on each
        # req_to_token[req_pool_idx, 0:in_window_start] view); the vectorized
        # version reduces that to one CUDA gather + one free_swa call.
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

        req_pool_idxs: List[int] = []
        oow_lens: List[int] = []
        for req in self.global_reqs:
            seqlen_no_last = req.seqlen - 1
            in_window_start = max(
                req.swa_evicted_seqlen, seqlen_no_last - sliding_window_size
            )
            if in_window_start > 0:
                req_pool_idxs.append(req.req_pool_idx)
                oow_lens.append(in_window_start)
                req.swa_evicted_seqlen = in_window_start

        if not req_pool_idxs:
            return

        device = self.req_to_token_pool.device
        row_idx_t = torch.tensor(req_pool_idxs, dtype=torch.int64, device=device)
        lens_t = torch.tensor(oow_lens, dtype=torch.int64, device=device)
        flat_rows = torch.repeat_interleave(row_idx_t, lens_t)

        # flat_cols = concat of arange(oow_lens[i]) for each i, built on GPU
        # via a single arange + cumulative-start subtraction (avoids a Python
        # loop over reqs to materialize per-req column tensors).
        total = int(lens_t.sum().item())
        arange_t = torch.arange(total, dtype=torch.int64, device=device)
        cumsum_t = torch.cumsum(lens_t, dim=0)
        starts = torch.cat(
            (torch.zeros(1, dtype=torch.int64, device=device), cumsum_t[:-1])
        )
        starts_per_pos = torch.repeat_interleave(starts, lens_t)
        flat_cols = arange_t - starts_per_pos

        oow_full_slots = self.req_to_token_pool.req_to_token[flat_rows, flat_cols]
        self.token_to_kv_pool_allocator.free_swa(oow_full_slots)

    def gather_cache(self) -> None:
        """Transfer KV cache from EP layout to TP layout.

        Pool-type invariant:
            - MHA-only model (no SWA layers): ``kv_cache`` is a ``MHATokenToKVPool``.
            - Hybrid model (any SWA layer): ``kv_cache`` MUST be an ``SWAKVPool``.
              ``SWAKVPool`` is a container holding both ``full_kv_pool`` and
              ``swa_kv_pool`` (each a plain ``MHATokenToKVPool``) plus the
              ``layers_mapping`` that routes per-layer access.

        Note: this function used to call torch.cuda.empty_cache() up front to
        defragment the caching allocator (~96 ms / call on 8 x A100 + 120B
        weights). It was dropped because (1) the KV pool was already resized
        by paras_resize_and_clear in reorchestrate_cache; (2) the transfer
        kernels write into pre-allocated UMM peer addresses, not into newly
        allocated caching-allocator blocks; and (3) empty_cache only sweeps
        the caching allocator's free list, which does not touch anything the
        transfer reads or writes. If post-switch decode OOMs surface across
        many EP<->TP cycles (fragmentation accumulating), reintroduce
        empty_cache on a background thread off the switch critical path.
        """
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

        for layer_id in range(num_layers - 1, -1, -1):
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
