"""MHA (Multi-Head Attention) cache transfer backend for ParaS."""

from sglang.srt.paras.cache_transfer.base import CacheTransferBase, LayerCacheSpec
from sglang.srt.paras.cache_transfer.utils import (
    do_gather_one_layer_nccl,
    do_gather_one_layer_peer_access,
    do_scatter_one_layer_nccl,
    do_scatter_one_layer_peer_access,
)

# Backward-compat re-export consumed by swa.py.
do_gather_one_layer = do_gather_one_layer_nccl


class MHACacheTransfer(CacheTransferBase):
    """Cache transfer for full-attention layers.

    Inherits ``__init__`` and all ``_precompute_*`` methods from
    ``CacheTransferBase``.  Overrides only the per-layer
    gather/scatter dispatch.
    """

    def gather_one_layer(self, spec: LayerCacheSpec, **kwargs) -> None:
        layer_id = spec.layer_id
        k_buffer = self.kv_cache.get_key_buffer(layer_id)
        v_buffer = self.kv_cache.get_value_buffer(layer_id)

        num_local = self.num_local_tokens
        layer_global_num = self.global_num_tokens
        num_global = self.num_global_tokens
        local_indices = self.local_token_indices

        if self.method == "nccl":
            do_gather_one_layer_nccl(
                k_buffer,
                v_buffer,
                num_local,
                num_global,
                local_indices,
                self.global_token_indices,
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

            do_gather_one_layer_peer_access(
                self._local_buffer_ptr,
                self._peer_addresses_gpu,
                self.mgr._entries[ep_k_name].offset_bytes,
                self.mgr._entries[ep_v_name].offset_bytes,
                self.mgr._entries[tp_k_name].offset_bytes,
                self.mgr._entries[tp_v_name].offset_bytes,
                local_indices,
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
            do_scatter_one_layer_nccl(
                self.kv_cache,
                self.ep_head_num,
                spec.layer_id,
                self.token_partition,
                self.group_size,
                self._intra_rank,
                self._replication_factor,
                self._per_token_elems,
                self.global_token_indices,
                self.ep_dst_positions,
                self._sorted_tp_indices,
                self._total_send_tokens,
                self._input_split_sizes,
                self._recv_full_count,
                self._output_split_sizes,
                self._total_recv_elems,
                self._num_kv_heads,
                self._heads_per_rank,
                self._head_dim,
                self._total_global_tokens,
                self._reassembly_groups,
                self.mgr,
                self.group,
            )
        else:
            tp_k_name = f"model.layers.{spec.layer_id}.kv.tp.k"
            tp_v_name = f"model.layers.{spec.layer_id}.kv.tp.v"
            ep_k_name = f"model.layers.{spec.layer_id}.kv.ep.k"
            ep_v_name = f"model.layers.{spec.layer_id}.kv.ep.v"

            do_scatter_one_layer_peer_access(
                self._local_buffer_ptr,
                self._peer_buffer_ptrs,
                self._tp_token_positions,
                self._token_to_rank,
                self._ep_dst_pos_all,
                self.mgr._entries[tp_k_name].offset_bytes,
                self.mgr._entries[tp_v_name].offset_bytes,
                self.mgr._entries[ep_k_name].offset_bytes,
                self.mgr._entries[ep_v_name].offset_bytes,
                self._num_my_tokens,
                spec.layer_id,
                self._heads_per_rank,
                self._num_kv_heads,
                self.paras_tp_rank,
                self.paras_tp_size,
                self._head_dim,
                self._elem_size,
            )


__all__ = ["MHACacheTransfer", "do_gather_one_layer"]
