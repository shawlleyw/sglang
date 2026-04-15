from typing import List, Optional, Tuple, Any
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
from sglang.srt.mem_cache.memory_pool import (
    ReqToTokenPool, 
    MHATokenToKVPool,
)
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.paras.utils import print_class_tensor_member, profile_object_members


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
    req.last_host_node = tree_cache.root_node
    req.last_node = tree_cache.root_node
    req.prefix_indices = []
    req.tokenizer = tokenizer

def paras_tp_group_all_gather_reqs(
    reqs: List[Req],
    group: GroupCoordinator,
) -> Tuple[List[Req], List[int]]:
    device = torch.device("cuda")
    
    num_ranks = group.world_size
    
    # Clean up tensor members to avoid pickle triggering torch device copy, mostly radix cache related stuff
    for req in reqs:
        prune_request(req)
        print_class_tensor_member(req)
        # profile_object_members(req)
        
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
    ):
        self.local_reqs = local_reqs
        self.gather_group = gather_group
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.group_size = gather_group.world_size
        self.peer_ctx = peer_ctx
        self.method = method or os.environ.get("PARAS_KV_TRANSFER_METHOD", "nccl")
        
        self.local_no_reqs = len(local_reqs) == 0
        self.local_seqlens_list = [req.seqlen for req in local_reqs]
        self.num_local_tokens = sum(self.local_seqlens_list) - len(local_reqs) # the last output token is not stored in kv cache
        
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
        
    def reorchestrate_cache(
        self, 
        new_req_pool_size: Optional[int] = None,
        new_cache_size: Optional[int] = None
    ):
        '''
        Heads are sharded by group size, 
        so the size of the request to token pool and the cache need to be multiplied by the group size.
        '''
        if new_req_pool_size is None:
            new_req_pool_size = self.req_to_token_pool.size * self.group_size
        if new_cache_size is None:
            new_cache_size = self.token_to_kv_pool_allocator.size * self.group_size
        
        assert self.num_global_tokens <= new_cache_size, "The total size of the requests to reorchestrate is greater than the new size of the cache."
        
        self.new_req_pool_size = new_req_pool_size
        self.new_cache_size = new_cache_size
        
        num_reqs = len(self.global_reqs)
        assert num_reqs <= new_req_pool_size, "The number of requests to reorchestrate is greater than the new size of the request to token pool."
        self.req_to_token_pool.paras_resize_and_clear(new_req_pool_size)
        req_pool_indices = self.req_to_token_pool.alloc(num_reqs)
        
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
        else:
            self.global_token_indices = None

    def gather_cache(self) -> torch.Tensor:
        if self.method == "peer_access" and self.peer_ctx is not None:
            self._gather_cache_peer_access()
        else:
            self._gather_cache_nccl()

    def _gather_cache_nccl(self):
        torch.cuda.empty_cache()
        kv_cache = self.token_to_kv_pool_allocator.get_kvcache()
        assert isinstance(kv_cache, MHATokenToKVPool), "Only MHATokenToKVPool is supported for now."
        
        num_layers = kv_cache.layer_num
        
        num_heads = kv_cache.head_num
        head_dim = kv_cache.head_dim
        sharded_num_heads = max(1, num_heads // self.group_size)
        splited_size_per_token = sharded_num_heads * head_dim
        
        # When num_heads < group_size, heads must be replicated so that
        # all_to_all has group_size chunks.  We chose Option A (repeat_interleave
        # before all_to_all) over Option B (sub-head all_to_all + intra-group
        # all_gather) for simplicity — see docs/paras/nvlink_peer_access_weight_transfer.md.
        replication_factor = max(1, self.group_size // num_heads) if num_heads < self.group_size else 1
        virtual_heads = num_heads * replication_factor  # == group_size when replicated
        
        input_split_sizes = [2 * splited_size_per_token * self.num_local_tokens] * self.group_size
        output_split_sizes = [2 * (splited_size_per_token * num_tokens_of_rank) for num_tokens_of_rank in self.global_num_tokens]
        
        def gather_one_layer(layer_id: int) -> torch.Tensor:

            if self.num_local_tokens > 0:
                k_buffer = kv_cache.get_key_buffer(layer_id)
                v_buffer = kv_cache.get_value_buffer(layer_id)
                
                # Fused gather and permute using Triton kernel
                # Output layout: [num_heads, num_tokens, 2, head_dim] (token-interleaved per head)
                permuted_local_kvcache = gather_kv_and_permute(k_buffer, v_buffer, self.local_token_indices)
                
                # Replicate heads for all_to_all when num_heads < group_size.
                # E.g. 4 heads / 8 GPUs: repeat_interleave(2) expands [4,N,2,128] → [8,N,2,128]
                # so virtual heads 0,1 are copies of real head 0 → sent to ranks 0,1 (replication).
                if replication_factor > 1:
                    N = self.num_local_tokens
                    permuted_local_kvcache = (
                        permuted_local_kvcache
                        .view(num_heads, N * 2 * head_dim)
                        .repeat_interleave(replication_factor, dim=0)
                        .flatten()
                    )
            else:
                permuted_local_kvcache = torch.empty((0, ), dtype=kv_cache.store_dtype, device=kv_cache.device)
                
            kv_cache.paras_resize_cache(layer_id, self.new_cache_size, sharded_num_heads)
                
            if self.num_global_tokens > 0:
                gathered_kvcache = torch.empty(2 * self.num_global_tokens * splited_size_per_token, dtype=permuted_local_kvcache.dtype, device=permuted_local_kvcache.device)
                torch.distributed.all_to_all_single(gathered_kvcache, permuted_local_kvcache, output_split_sizes, input_split_sizes, group=self.gather_group.device_group)
                
                # Scatter into TP K/V buffers.
                # gather_kv_and_permute outputs [heads, tokens, KV, dim], so after
                # all_to_all the received data is [total_tokens, KV, sharded_heads, dim]
                # which permute_and_scatter_kv handles directly.
                k_buffer = kv_cache.get_key_buffer(layer_id)
                v_buffer = kv_cache.get_value_buffer(layer_id)
                permute_and_scatter_kv(
                    gathered_kvcache, k_buffer, v_buffer, self.global_token_indices,
                    self.num_global_tokens, sharded_num_heads, head_dim
                )
                
        for layer_id in range(num_layers):
            gather_one_layer(layer_id)
            
        torch.cuda.synchronize()

    def _gather_cache_peer_access(self):
        from sglang.srt.paras.peer_access import peer_access_kv_transfer
        from sglang.srt.paras.paras_memory_manager import get_global_paras_memory_manager

        torch.cuda.empty_cache()
        kv_cache = self.token_to_kv_pool_allocator.get_kvcache()
        mgr = get_global_paras_memory_manager()

        num_layers = kv_cache.layer_num
        num_heads = kv_cache.head_num
        head_dim = kv_cache.head_dim
        sharded_num_heads = max(1, num_heads // self.group_size)

        tp_rank = dist.get_rank(group=self.gather_group.device_group)
        # Destination token start = sum of token counts for all ranks before this one
        dst_token_start = sum(self.global_num_tokens[:tp_rank])

        # Build peer addresses tensor on GPU
        dst_base_ptrs = torch.tensor(
            self.peer_ctx.peer_addresses, dtype=torch.int64, device="cuda"
        )

        # Barrier tensor for per-layer sync
        barrier_tensor = torch.zeros(1, device="cuda")

        elem_size = kv_cache.store_dtype.itemsize if hasattr(kv_cache.store_dtype, 'itemsize') else 2
        local_buffer_ptr = mgr._buffer.data_ptr()

        # Token indices: local_token_indices are the EP slot positions
        # If num_local_tokens == 0, we still participate in barriers
        local_token_indices_gpu = self.local_token_indices  # already on GPU
        # Kernel expects int32 — convert if needed
        if local_token_indices_gpu is not None and local_token_indices_gpu.dtype != torch.int32:
            local_token_indices_gpu = local_token_indices_gpu.to(torch.int32)

        for layer_id in range(num_layers):
            ep_k_name = f"model.layers.{layer_id}.kv.ep.k"
            ep_v_name = f"model.layers.{layer_id}.kv.ep.v"
            tp_k_name = f"model.layers.{layer_id}.kv.tp.k"
            tp_v_name = f"model.layers.{layer_id}.kv.tp.v"

            src_k_offset = mgr._entries[ep_k_name].offset_bytes
            src_v_offset = mgr._entries[ep_v_name].offset_bytes
            dst_k_offset = mgr._entries[tp_k_name].offset_bytes
            dst_v_offset = mgr._entries[tp_v_name].offset_bytes

            if self.num_local_tokens > 0:
                peer_access_kv_transfer(
                    local_buffer_ptr, dst_base_ptrs,
                    local_token_indices_gpu,
                    src_k_offset, src_v_offset,
                    dst_k_offset, dst_v_offset,
                    self.num_local_tokens, dst_token_start,
                    num_heads, tp_rank, self.group_size, head_dim,
                    elem_size,
                )

            # Per-layer barrier: ensures all ranks finish writing before next layer reads
            dist.all_reduce(barrier_tensor, group=self.gather_group.device_group)

        torch.cuda.synchronize()

        # Now resize cache buffers to point to TP slots
        for layer_id in range(num_layers):
            kv_cache.paras_resize_cache(layer_id, self.new_cache_size, sharded_num_heads)

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
        
        # Get req_pool_indices and seq_lens
        req_pool_indices_list = [req.req_pool_idx for req in self.global_reqs]
        seq_lens_list = [req.seqlen for req in self.global_reqs]
        
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
        
        for req in running_batch.reqs:
            req.last_node = running_batch.tree_cache.root_node
        
        # Get the last token from each request (for decode input)
        input_ids_list = []
        for req in self.global_reqs:
            if len(req.output_ids) > 0:
                input_ids_list.append(req.output_ids[-1])
            else:
                # No output yet, use last input token
                input_ids_list.append(req.origin_input_ids[-1])
        
        # Get req_pool_indices and seq_lens
        req_pool_indices_list = [req.req_pool_idx for req in self.global_reqs]
        seq_lens_list = [req.seqlen for req in self.global_reqs]
        
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



# Backward compatibility — scatter components moved to scatter_manager.py
from sglang.srt.paras.scatter_manager import (  # noqa: F401, E402
    ParaSReqScatterManager,
    partition_requests_for_ep,
    gather_tp_kv_and_permute,
    permute_and_scatter_kv_to_ep,
    _scatter_cache_nccl,
)