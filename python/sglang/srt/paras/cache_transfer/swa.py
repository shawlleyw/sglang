"""SWA (Sliding Window Attention) cache transfer backend for ParaS."""

from typing import List

import torch

from sglang.srt.paras.cache_transfer import LayerCacheSpec
from sglang.srt.paras.cache_transfer.mha import do_gather_one_layer


class SWACacheTransfer:
    """Like MHACacheTransfer but routes through ``swa_kv_pool`` and caps
    token counts per layer via ``spec.tokens_cap_ep``."""

    def __init__(
        self,
        kv_cache,
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
        local_id, is_swa = self.kv_cache.layers_mapping[layer_id]

        k_buffer = self.kv_cache.swa_kv_pool.k_buffer[local_id]
        v_buffer = self.kv_cache.swa_kv_pool.v_buffer[local_id]

        num_local = min(self.num_local_tokens, spec.tokens_cap_ep)
        local_indices = self.local_token_indices[:num_local]
        layer_global_num = [
            min(n, spec.tokens_cap_ep) for n in self.global_num_tokens
        ]
        num_global = sum(layer_global_num)

        do_gather_one_layer(
            k_buffer, v_buffer,
            num_local, num_global,
            local_indices, self.global_token_indices,
            layer_global_num, self.group_size,
            self.kv_cache.head_num, self.kv_cache.head_dim,
            self.kv_cache.store_dtype, self.kv_cache.device,
            self.mgr, layer_id, self.gather_group,
        )

    def scatter_one_layer(self, spec: LayerCacheSpec, **kwargs) -> None:
        layer_id = spec.layer_id
        local_id, is_swa = self.kv_cache.layers_mapping[layer_id]

        num_local = min(self.num_local_tokens, spec.tokens_cap_ep)
        layer_global_num = [
            min(n, spec.tokens_cap_ep) for n in self.global_num_tokens
        ]
        num_global = sum(layer_global_num)
        sharded_num_heads = max(1, self.kv_cache.head_num // self.group_size)
        head_dim = self.kv_cache.head_dim
        splited_size_per_token = sharded_num_heads * head_dim

        tp_k_name = f"model.layers.{layer_id}.kv.tp.k"
        tp_v_name = f"model.layers.{layer_id}.kv.tp.v"
        total_elements = self.mgr._entries[tp_k_name].numel
        tp_slots = total_elements // (sharded_num_heads * head_dim)
        tp_shape = (tp_slots, sharded_num_heads, head_dim)
        tp_k = self.mgr.get_view_as(tp_k_name, tp_shape)
        tp_v = self.mgr.get_view_as(tp_v_name, tp_shape)

        k_dst = self.kv_cache.swa_kv_pool.k_buffer[local_id]
        v_dst = self.kv_cache.swa_kv_pool.v_buffer[local_id]

        if num_global > 0 and num_local > 0:
            from sglang.srt.paras.gather_manager import gather_kv_and_permute
            import torch.distributed as dist
            from sglang.srt.paras.gather_manager import permute_and_scatter_kv

            send_kvcache = gather_kv_and_permute(
                tp_k.reshape(-1, sharded_num_heads, head_dim),
                tp_v.reshape(-1, sharded_num_heads, head_dim),
                self.global_token_indices[:num_global],
            )

            output_split_sizes = [
                2 * splited_size_per_token * num_local
            ] * self.group_size
            input_split_sizes = [
                2 * (splited_size_per_token * n) for n in layer_global_num
            ]

            recv_kvcache = torch.empty(
                2 * num_local * splited_size_per_token,
                dtype=send_kvcache.dtype,
                device=send_kvcache.device,
            )
            dist.all_to_all_single(
                recv_kvcache,
                send_kvcache,
                output_split_sizes,
                input_split_sizes,
                group=self.gather_group.device_group,
            )

            local_indices = self.local_token_indices[:num_local]
            permute_and_scatter_kv(
                recv_kvcache,
                k_dst,
                v_dst,
                local_indices,
                num_local,
                sharded_num_heads,
                head_dim,
            )


__all__ = ["SWACacheTransfer"]
