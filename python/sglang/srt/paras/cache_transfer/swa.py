"""SWA (Sliding Window Attention) cache transfer backend for ParaS."""

import warnings
from typing import List, Optional, Tuple

import torch

from sglang.srt.paras.cache_transfer.base import CacheTransferBase, LayerCacheSpec
from sglang.srt.paras.cache_transfer.utils import (
    do_gather_one_layer_nccl,
    do_gather_one_layer_peer_access,
    do_scatter_one_layer_nccl,
    do_scatter_one_layer_peer_access,
)


class SWACacheTransfer(CacheTransferBase):
    """Cache transfer for sliding-window-attention layers.

    Inherits ``__init__`` and all ``_precompute_*`` methods from
    ``CacheTransferBase``.  Overrides the per-layer gather/scatter
    dispatch to read buffers from the SWA sub-pool and cap token
    counts at ``spec.tokens_cap_ep``.

    Index translation
    -----------------
    ``req_to_token_pool`` stores **full-pool** indices.  SWA buffers
    live in SWA index space (size ``_size_swa < _size_full``).  Attention
    backends translate via ``SWAKVPool.full_to_swa_index_mapping`` at
    attend time; this class does the same for cache transfer.

    ``full_to_swa_index_mapping`` is zero-initialized, so translating
    an unallocated full-pool index returns 0 (a valid but overwritten
    padding slot).  This is harmless by design.
    """

    def __init__(
        self,
        *,
        source_full_to_swa_mapping: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.source_full_to_swa_mapping = source_full_to_swa_mapping
        self._warn_if_head_replication()
        # SWAKVPool.full_to_swa_index_mapping is set by
        # SWATokenToKVPoolAllocator.__init__ (allocator.py:224) and kept in
        # sync on every paras_resize_and_clear (allocator.py:318).
        self._full_to_swa_mapping = getattr(
            self.kv_cache, "full_to_swa_index_mapping", None
        )

    def _warn_if_head_replication(self) -> None:
        """Emit a UserWarning when SWA + head replication is in use.

        ``test_swa_kv_cache_transfer_replication.py`` validates the
        replication-aware paths in ``_compute_swa_scatter_*()`` at R=2
        on 4 GPUs, but production traffic at this combination is rare
        (GPT-OSS / Gemma typically have ``num_kv_heads >= paras_tp_size``).
        Surface a warning so deployments that hit this combo can flag
        it for extra scrutiny rather than failing silently if a future
        edge case slips past the tests.
        """
        group_size = self.group_size
        num_kv_heads = (
            self.kv_cache.head_num
            if self.direction == "gather"
            else self.ep_head_num
        )
        if num_kv_heads < group_size:
            warnings.warn(
                f"SWACacheTransfer with head replication "
                f"(num_kv_heads={num_kv_heads}, paras_tp_size={group_size}, "
                f"replication_factor={group_size // num_kv_heads}) is "
                f"gate-tested at R=2 but rarely exercised in production. "
                f"Verify correctness end-to-end before relying on this "
                f"configuration.",
                UserWarning,
                stacklevel=2,
            )

    def _full_to_swa(self, full_indices: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Translate full-pool indices to SWA-pool indices using the LIVE mapping.

        Use ONLY for destination-side positions (post-resize, post-alloc state).
        For source-side positions (TP for scatter, EP for gather) call
        ``_full_to_swa_source`` instead; the live mapping was wiped by
        ``paras_resize_and_clear`` before this backend was constructed and now
        only contains entries for the destination mode's freshly-allocated
        slots.

        Preserves the caller's input dtype.  The peer-access CUDA bindings
        (see ``csrc/binding.cpp``) read token-index tensors as ``int32*`` via
        ``data_ptr<int>()`` with no runtime dtype check.  NCCL gather/scatter
        helpers use torch advanced indexing which needs int64.  Since both
        consumers pass through this helper, preserve whatever dtype the
        caller handed us so each path stays valid.
        """
        if full_indices is None or full_indices.numel() == 0:
            return full_indices
        if self._full_to_swa_mapping is None:
            return full_indices
        original_dtype = full_indices.dtype
        return self._full_to_swa_mapping[full_indices.to(torch.int64)].to(original_dtype)

    def _full_to_swa_source(self, full_indices: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Translate source-side full-pool indices via the pre-resize snapshot.

        For scatter the source mode is TP; for gather it is EP. In both cases
        the source-mode ``full_to_swa_index_mapping`` was zero-filled by
        ``{ParaSReq*Manager}.reorchestrate_cache``'s ``paras_resize_and_clear``
        before scatter_cache / gather_cache constructs this backend. Without a
        snapshot, every source-side lookup returns 0 (padding slot) and SWA
        layers' transport produces uniformly noisy post-switch decode output.
        ``source_full_to_swa_mapping`` is the snapshot taken just before the
        resize wiped the live mapping.
        """
        if full_indices is None or full_indices.numel() == 0:
            return full_indices
        snapshot = self.source_full_to_swa_mapping
        if snapshot is None:
            return self._full_to_swa(full_indices)
        original_dtype = full_indices.dtype
        return snapshot[full_indices.to(torch.int64)].to(original_dtype)

    def _compute_swa_scatter_splits_nccl(
        self, cap: int,
    ) -> Tuple[torch.Tensor, int, List[int], int, List[int], int]:
        """Cap-aware NCCL scatter splits: ``min(full, cap)`` per destination.

        This cannot reuse ``CacheTransferBase._precompute_scatter_nccl``
        because the SWA cap varies per layer (``spec.tokens_cap_ep``),
        while the base precompute runs once in ``__init__``.  Under head
        replication the per-destination slice is
        ``[full*intra/R, full*(intra+1)/R]`` for the base but
        ``[cap*intra/R, cap*(intra+1)/R]`` for the SWA layer, and the
        latter is not a contiguous subset of the former when
        ``intra > 0`` (e.g. full=10, cap=6, R=2, intra=1 → base takes
        tokens [5, 10), SWA takes tokens [3, 6)).  So SWA layers must
        recompute the per-destination slicing at dispatch time.
        """
        group_size = self.group_size
        intra_rank = self._intra_rank
        replication_factor = self._replication_factor
        per_token_elems = self._per_token_elems
        send_counts: List[int] = []
        sorted_parts: List[torch.Tensor] = []
        for e in range(group_size):
            capped = min(len(self.token_partition[e]), cap)
            my_s = capped * intra_rank // replication_factor
            my_e = capped * (intra_rank + 1) // replication_factor
            my_cnt = my_e - my_s
            send_counts.append(my_cnt)
            if my_cnt > 0:
                part_idx = torch.tensor(
                    self.token_partition[e][my_s:my_e],
                    dtype=torch.long,
                    device=self.global_token_indices.device,
                )
                sorted_parts.append(self.global_token_indices[part_idx])

        total_send_tokens = sum(send_counts)
        input_split_sizes = [c * per_token_elems for c in send_counts]

        sorted_tp_indices = (
            torch.cat(sorted_parts) if sorted_parts
            else torch.empty(0, dtype=torch.long,
                             device=self.global_token_indices.device)
        )

        recv_full_capped = min(self._recv_full_count, cap)
        recv_counts: List[int] = []
        for src in range(group_size):
            src_intra = src % replication_factor
            s = recv_full_capped * src_intra // replication_factor
            e_idx = recv_full_capped * (src_intra + 1) // replication_factor
            recv_counts.append(e_idx - s)

        output_split_sizes = [c * per_token_elems for c in recv_counts]
        total_recv_elems = sum(output_split_sizes)

        return (sorted_tp_indices, total_send_tokens, input_split_sizes,
                recv_full_capped, output_split_sizes, total_recv_elems)

    def _compute_swa_scatter_slices_peer_access(
        self, cap: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], int]:
        """Cap-aware per-destination slices returning int32 tensors for peer-access.

        See ``_compute_swa_scatter_splits_nccl`` for why this cannot
        reuse ``CacheTransferBase._precompute_scatter_peer_access``:
        per-layer cap semantics make the per-destination slice a
        non-subset of the base precompute output under head replication.
        """
        group_size = self.group_size
        num_kv_heads = self._num_kv_heads
        replication_factor = (
            group_size // num_kv_heads if num_kv_heads < group_size else 1
        )
        intra_rank = self.paras_tp_rank % replication_factor

        capped_global_indices: List[int] = []
        capped_dst_ranks: List[int] = []
        capped_ep_dst_pos: List[int] = []
        for e in range(group_size):
            full_tokens = self.token_partition[e]
            full = len(full_tokens)
            capped = min(full, cap)
            my_s = capped * intra_rank // replication_factor
            my_e = capped * (intra_rank + 1) // replication_factor
            my_slice = full_tokens[my_s:my_e]
            for local_idx, global_idx in enumerate(my_slice):
                capped_global_indices.append(global_idx)
                capped_dst_ranks.append(e)
                # +1: slot 0 is the padding slot.
                capped_ep_dst_pos.append(my_s + local_idx + 1)

        layer_num = len(capped_global_indices)
        if layer_num == 0:
            return None, None, None, 0

        device = self.global_token_indices.device
        gi_tensor = torch.tensor(
            capped_global_indices, dtype=torch.long, device=device
        )
        tp_positions = self.global_token_indices[gi_tensor].to(torch.int32)
        token_to_rank = torch.tensor(
            capped_dst_ranks, dtype=torch.int32, device=device
        )
        ep_dst_pos = torch.tensor(
            capped_ep_dst_pos, dtype=torch.int32, device=device
        )
        return tp_positions, token_to_rank, ep_dst_pos, layer_num

    def gather_one_layer(self, spec: LayerCacheSpec, **kwargs) -> None:
        layer_id = spec.layer_id
        local_id, _ = self.kv_cache.layers_mapping[layer_id]
        k_buffer = self.kv_cache.swa_kv_pool.k_buffer[local_id]
        v_buffer = self.kv_cache.swa_kv_pool.v_buffer[local_id]

        # Unconditional SWA token capping.
        num_local = min(self.num_local_tokens, spec.tokens_cap_ep)
        layer_global_num = [
            min(n, spec.tokens_cap_ep) for n in self.global_num_tokens
        ]
        num_global = sum(layer_global_num)

        local_indices = (
            self.local_token_indices[:num_local]
            if self.local_token_indices is not None
            and num_local < self.num_local_tokens
            else self.local_token_indices
        )

        # P3 fix: translate local EP indices from full-pool to SWA-pool space
        # before reading from the SWA k/v buffers. EP is the source mode for
        # gather, so use the pre-resize snapshot (reorchestrate_cache wipes the
        # live mapping then re-populates it for the destination TP slots only).
        swa_local = self._full_to_swa_source(local_indices)

        # P3 fix: build per-rank capped global_token_indices, then translate.
        # self.global_token_indices is the flat concatenation of destination TP
        # token slots (uncapped).  We must slice each rank's chunk to
        # layer_global_num[i] entries before translation. These are destination
        # positions, so use the live post-resize mapping.
        if self.global_token_indices is not None and num_global > 0:
            start = 0
            capped_parts = []
            for i, n_full in enumerate(self.global_num_tokens):
                take = layer_global_num[i]
                capped_parts.append(self.global_token_indices[start:start + take])
                start += n_full
            global_indices_full_capped = torch.cat(capped_parts)
            swa_global = self._full_to_swa(global_indices_full_capped)
        else:
            swa_global = self.global_token_indices

        if self.method == "nccl":
            do_gather_one_layer_nccl(
                k_buffer,
                v_buffer,
                num_local,
                num_global,
                swa_local,
                swa_global,
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

            # Peer-access gather: local_token_indices are the READ side
            # (EP buffer positions), so translate to SWA space.
            # The WRITE side uses dst_token_start as a scalar offset into the
            # TP buffer which is addressed by the CUDA kernel directly — no
            # translation needed there.
            do_gather_one_layer_peer_access(
                self._local_buffer_ptr,
                self._peer_addresses_gpu,
                self.mgr._entries[ep_k_name].offset_bytes,
                self.mgr._entries[ep_v_name].offset_bytes,
                self.mgr._entries[tp_k_name].offset_bytes,
                self.mgr._entries[tp_v_name].offset_bytes,
                swa_local,
                num_local,
                self._dst_token_start,
                self.kv_cache.head_num,
                self._tp_rank,
                self.group_size,
                self.kv_cache.head_dim,
                self._elem_size,
            )

    def scatter_one_layer(self, spec: LayerCacheSpec, **kwargs) -> None:
        cap = spec.tokens_cap_ep
        if self.method == "nccl":
            (sorted_tp_indices_full, total_send_tokens, input_split_sizes,
             recv_full_capped, output_split_sizes, total_recv_elems,
             ) = self._compute_swa_scatter_splits_nccl(cap)
            sorted_tp_indices_swa = self._full_to_swa_source(sorted_tp_indices_full)
            if self.ep_dst_positions is not None and recv_full_capped > 0:
                ep_dst_pos_swa = self._full_to_swa(
                    self.ep_dst_positions[:recv_full_capped])
            else:
                ep_dst_pos_swa = self.ep_dst_positions
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
                ep_dst_pos_swa,
                sorted_tp_indices_swa,
                total_send_tokens,
                input_split_sizes,
                recv_full_capped,
                output_split_sizes,
                total_recv_elems,
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
            tp_positions_full, token_to_rank, ep_dst_pos_full, layer_num = \
                self._compute_swa_scatter_slices_peer_access(cap)
            if layer_num == 0:
                return
            tp_positions_swa = self._full_to_swa_source(tp_positions_full)
            # Peer-access writes directly into remote EP buffers. Pass
            # ep_dst_pos_full WITHOUT translation. Applying the local rank's
            # live full->SWA mapping to remote-rank positions would turn valid
            # remote slots into padding slot 0 (the local mapping only
            # describes THIS rank's freshly allocated EP slots).
            #
            # Structural [1..N] guarantee — three invariants from
            # ParaSReqScatterManager.reorchestrate_cache:
            #   (1) paras_resize_and_clear resets each rank's full and SWA
            #       sub-allocators to free_pages = arange(1, new_size+1).
            #   (2) Exactly ONE alloc(num_local_tokens) follows on each rank,
            #       advancing both inner allocators in lockstep so
            #       full_to_swa_index_mapping[i] == i for i in [1..N].
            #   (3) token_partition is built from the same local_reqs ordering
            #       on every rank.
            # Together these guarantee that for token j on destination rank e:
            #   full_slot_e[j] == swa_slot_e[j] == my_s + local_idx + 1
            # so the inferred remote full destination position equals the
            # remote SWA position bit-for-bit. No mapping lookup is needed
            # and the kernel can index the remote rank's SWA pool directly.
            ep_dst_pos_swa = ep_dst_pos_full
            do_scatter_one_layer_peer_access(
                self._local_buffer_ptr,
                self._peer_buffer_ptrs,
                tp_positions_swa,
                token_to_rank,
                ep_dst_pos_swa,
                self.mgr._entries[tp_k_name].offset_bytes,
                self.mgr._entries[tp_v_name].offset_bytes,
                self.mgr._entries[ep_k_name].offset_bytes,
                self.mgr._entries[ep_v_name].offset_bytes,
                layer_num,
                spec.layer_id,
                self._heads_per_rank,
                self._num_kv_heads,
                self.paras_tp_rank,
                self.paras_tp_size,
                self._head_dim,
                self._elem_size,
            )


__all__ = ["SWACacheTransfer"]
