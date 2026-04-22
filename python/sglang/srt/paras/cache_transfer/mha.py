"""MHA (Multi-Head Attention) cache transfer backend for ParaS."""

from typing import List, Optional, Union

import torch
import torch.distributed as dist

from sglang.srt.paras.cache_transfer import CacheTransferBackend, LayerCacheSpec
from sglang.srt.paras.gather_manager import gather_kv_and_permute, permute_and_scatter_kv
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool


def do_gather_one_layer(
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


class MHACacheTransfer:
    """MHA cache transfer backend -- wraps existing per-layer gather/scatter logic.

    Implements the ``CacheTransferBackend`` protocol.  Barriers remain the
    caller's responsibility (AC3).
    """

    def __init__(
        self,
        kv_cache: MHATokenToKVPool,
        mgr,
        group_size: int,
        num_local_tokens: int,
        num_global_tokens: int,
        local_token_indices: torch.Tensor,
        global_token_indices: torch.Tensor,
        global_num_tokens: List[int],
        gather_group,
    ):
        self.kv_cache = kv_cache
        self.mgr = mgr
        self.group_size = group_size
        self.num_local_tokens = num_local_tokens
        self.num_global_tokens = num_global_tokens
        self.local_token_indices = local_token_indices
        self.global_token_indices = global_token_indices
        self.global_num_tokens = global_num_tokens
        self.gather_group = gather_group

    def gather_one_layer(self, spec: LayerCacheSpec, **kwargs) -> None:
        layer_id = spec.layer_id
        k_buffer = self.kv_cache.get_key_buffer(layer_id)
        v_buffer = self.kv_cache.get_value_buffer(layer_id)
        do_gather_one_layer(
            k_buffer, v_buffer,
            self.num_local_tokens, self.num_global_tokens,
            self.local_token_indices, self.global_token_indices,
            self.global_num_tokens, self.group_size,
            self.kv_cache.head_num, self.kv_cache.head_dim,
            self.kv_cache.store_dtype, self.kv_cache.device,
            self.mgr, layer_id, self.gather_group,
        )

    def scatter_one_layer(self, spec: LayerCacheSpec, **kwargs) -> None:
        # T7 wires the full scatter logic through this method.
        pass


__all__ = ["MHACacheTransfer", "do_gather_one_layer"]
