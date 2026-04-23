#!/usr/bin/env python3
"""
Synthetic hybrid EP-TP-EP round-trip coherence test.

Tests per-layer K/V buffer rebinding for a 2-full + 4-SWA hybrid model:
  1. TP rebind routes full layers to full_kv_pool, SWA layers to swa_kv_pool
  2. EP rebind restores original EP buffer pointers
  3. SWA layers have smaller token capacity than full layers
  4. Round-trip EP->TP->EP preserves buffer pointer identity
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from sglang.srt.paras.cache_transfer import LayerCacheSpec
from sglang.srt.paras.paras_memory_manager import ParaSMemoryManager
from sglang.srt.mem_cache.memory_pool import SWAKVPool
from sglang.srt.paras import paras_memory_manager as pmm

FULL_LAYER_IDS = [0, 1]
SWA_LAYER_IDS = [2, 3, 4, 5]
NUM_LAYERS = 6
NUM_KV_HEADS = 4
HEAD_DIM = 64
TP_SIZE = 4
EP_TOKENS_FULL = 1024
TP_TOKENS_FULL = 4096
EP_TOKENS_SWA = 256
TP_TOKENS_SWA = 1024
PAGE_SIZE = 1
KV_DTYPE = torch.bfloat16
DEVICE = 'cpu'


def make_layer_specs():
    specs = []
    for i in range(NUM_LAYERS):
        if i in FULL_LAYER_IDS:
            specs.append(LayerCacheSpec(
                layer_id=i, kind='full',
                tokens_cap_ep=EP_TOKENS_FULL, tokens_cap_tp=TP_TOKENS_FULL,
                num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
                sliding_window_size=None
            ))
        else:
            specs.append(LayerCacheSpec(
                layer_id=i, kind='swa',
                tokens_cap_ep=EP_TOKENS_SWA, tokens_cap_tp=TP_TOKENS_SWA,
                num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
                sliding_window_size=1023
            ))
    return specs


def setup_mgr_and_pool(specs):
    mgr = ParaSMemoryManager(device=DEVICE)
    mgr.reserve_kv_cache(
        num_layers=NUM_LAYERS,
        ep_max_tokens=EP_TOKENS_FULL,
        tp_max_tokens=TP_TOKENS_FULL,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        kv_dtype=KV_DTYPE,
        page_size=PAGE_SIZE,
        layer_specs=specs,
    )
    mgr.materialize()
    pmm._global_paras_memory_manager = mgr

    pool = SWAKVPool(
        size=EP_TOKENS_FULL,
        size_swa=EP_TOKENS_SWA,
        dtype=KV_DTYPE,
        head_num=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        swa_attention_layer_ids=SWA_LAYER_IDS,
        full_attention_layer_ids=FULL_LAYER_IDS,
        enable_kvcache_transpose=False,
        device=DEVICE,
    )
    return mgr, pool


class TestHybridRoundtrip:
    def test_tp_rebind_routes_correctly(self):
        specs = make_layer_specs()
        mgr, pool = setup_mgr_and_pool(specs)

        pool.paras_configure_tp(paras_tp_size=TP_SIZE, layer_specs=specs)

        for g in range(NUM_LAYERS):
            local_id, is_swa = pool.layers_mapping[g]
            tp_k_name = f"model.layers.{g}.kv.tp.k"
            mgr_ptr = mgr.get_view(tp_k_name).data_ptr()
            if is_swa:
                pool_ptr = pool.swa_kv_pool.k_buffer[local_id].data_ptr()
            else:
                pool_ptr = pool.full_kv_pool.k_buffer[local_id].data_ptr()
            assert mgr_ptr == pool_ptr, f"Layer {g}: TP buffer mismatch"
        print("OK: TP rebind routes correctly")

    def test_ep_rebind_routes_correctly(self):
        specs = make_layer_specs()
        mgr, pool = setup_mgr_and_pool(specs)

        pool.paras_configure_ep(layer_specs=specs)

        for g in range(NUM_LAYERS):
            local_id, is_swa = pool.layers_mapping[g]
            ep_k_name = f"model.layers.{g}.kv.ep.k"
            mgr_ptr = mgr.get_view(ep_k_name).data_ptr()
            if is_swa:
                pool_ptr = pool.swa_kv_pool.k_buffer[local_id].data_ptr()
            else:
                pool_ptr = pool.full_kv_pool.k_buffer[local_id].data_ptr()
            assert mgr_ptr == pool_ptr, f"Layer {g}: EP buffer mismatch"
        print("OK: EP rebind routes correctly")

    def test_swa_layers_have_smaller_capacity(self):
        specs = make_layer_specs()
        for spec in specs:
            if spec.kind == 'full':
                assert spec.tokens_cap_ep == EP_TOKENS_FULL
                assert spec.tokens_cap_tp == TP_TOKENS_FULL
            else:
                assert spec.tokens_cap_ep == EP_TOKENS_SWA
                assert spec.tokens_cap_tp == TP_TOKENS_SWA
                assert spec.tokens_cap_ep < EP_TOKENS_FULL
                assert spec.tokens_cap_tp < TP_TOKENS_FULL
        print("OK: SWA layers have smaller capacity")

    def test_roundtrip_ep_tp_ep_buffer_identity(self):
        specs = make_layer_specs()
        mgr, pool = setup_mgr_and_pool(specs)

        ep_ptrs = {}
        for g in range(NUM_LAYERS):
            ep_k_name = f"model.layers.{g}.kv.ep.k"
            ep_ptrs[g] = mgr.get_view(ep_k_name).data_ptr()

        pool.paras_configure_tp(paras_tp_size=TP_SIZE, layer_specs=specs)
        pool.paras_configure_ep(layer_specs=specs)

        for g in range(NUM_LAYERS):
            local_id, is_swa = pool.layers_mapping[g]
            ep_k_name = f"model.layers.{g}.kv.ep.k"
            restored_ptr = mgr.get_view(ep_k_name).data_ptr()
            if is_swa:
                pool_ptr = pool.swa_kv_pool.k_buffer[local_id].data_ptr()
            else:
                pool_ptr = pool.full_kv_pool.k_buffer[local_id].data_ptr()
            assert pool_ptr == restored_ptr, f"Layer {g}: EP buffer not restored after round-trip"
        print("OK: Round-trip EP->TP->EP buffer identity preserved")

    def test_head_count_sharding(self):
        specs = make_layer_specs()
        mgr, pool = setup_mgr_and_pool(specs)

        assert pool.head_num == NUM_KV_HEADS

        pool.paras_configure_tp(paras_tp_size=TP_SIZE, layer_specs=specs)
        assert pool.head_num == NUM_KV_HEADS // TP_SIZE, f"Expected {NUM_KV_HEADS // TP_SIZE}, got {pool.head_num}"

        pool.paras_configure_ep(layer_specs=specs)
        assert pool.head_num == NUM_KV_HEADS, f"Expected {NUM_KV_HEADS}, got {pool.head_num}"
        print("OK: Head count sharding correct")


class TestSWAAllocatorSignature:
    """Test A (P1/P2): SWA allocator two-arg resize works via managers."""

    def test_swa_allocator_paras_resize_accepts_two_args(self):
        from sglang.srt.mem_cache.allocator import SWATokenToKVPoolAllocator
        from sglang.srt.mem_cache.memory_pool import SWAKVPool

        kvcache = SWAKVPool(
            size=64, size_swa=32,
            dtype=KV_DTYPE, head_num=NUM_KV_HEADS, head_dim=HEAD_DIM,
            swa_attention_layer_ids=[0], full_attention_layer_ids=[1],
            enable_kvcache_transpose=False, device=DEVICE,
        )
        alloc = SWATokenToKVPoolAllocator(
            size=64, size_swa=32, dtype=torch.int64,
            device=DEVICE, kvcache=kvcache, need_sort=False,
        )
        alloc.paras_resize_and_clear(128, 64)
        assert alloc._size_full == 128
        assert alloc._size_swa == 64

    def test_gather_manager_reorchestrate_with_swa_allocator(self):
        from unittest.mock import MagicMock
        from sglang.srt.mem_cache.allocator import SWATokenToKVPoolAllocator
        from sglang.srt.mem_cache.memory_pool import SWAKVPool, ReqToTokenPool

        kvcache = SWAKVPool(
            size=64, size_swa=32,
            dtype=KV_DTYPE, head_num=NUM_KV_HEADS, head_dim=HEAD_DIM,
            swa_attention_layer_ids=[0], full_attention_layer_ids=[1],
            enable_kvcache_transpose=False, device=DEVICE,
        )
        alloc = SWATokenToKVPoolAllocator(
            size=64, size_swa=32, dtype=torch.int64,
            device=DEVICE, kvcache=kvcache, need_sort=False,
        )
        req_pool = ReqToTokenPool(size=32, max_context_len=128, device=DEVICE, enable_memory_saver=False)

        group = MagicMock()
        group.world_size = 2

        from sglang.srt.paras.gather_manager import ParaSReqGatherManager
        mgr = ParaSReqGatherManager(
            local_reqs=[], gather_group=group,
            req_to_token_pool=req_pool,
            token_to_kv_pool_allocator=alloc,
        )
        mgr.global_reqs = []
        mgr.global_reqs_split_sizes = []
        mgr.global_seqlens_list = []
        mgr.global_num_tokens = []
        mgr.num_global_tokens = 0
        mgr.reorchestrate_cache()
        assert alloc._size_full == 128
        assert alloc._size_swa == 64

    def test_scatter_manager_reorchestrate_with_swa_allocator(self):
        from unittest.mock import MagicMock
        from sglang.srt.mem_cache.allocator import SWATokenToKVPoolAllocator
        from sglang.srt.mem_cache.memory_pool import SWAKVPool, ReqToTokenPool

        kvcache = SWAKVPool(
            size=128, size_swa=64,
            dtype=KV_DTYPE, head_num=NUM_KV_HEADS, head_dim=HEAD_DIM,
            swa_attention_layer_ids=[0], full_attention_layer_ids=[1],
            enable_kvcache_transpose=False, device=DEVICE,
        )
        alloc = SWATokenToKVPoolAllocator(
            size=128, size_swa=64, dtype=torch.int64,
            device=DEVICE, kvcache=kvcache, need_sort=False,
        )
        req_pool = ReqToTokenPool(size=64, max_context_len=256, device=DEVICE, enable_memory_saver=False)

        group = MagicMock()
        group.world_size = 2

        from sglang.srt.paras.scatter_manager import ParaSReqScatterManager
        mgr = ParaSReqScatterManager(
            global_reqs=[], scatter_group=group,
            req_to_token_pool=req_pool,
            token_to_kv_pool_allocator=alloc,
            paras_tp_rank=0, paras_tp_size=2,
        )
        mgr.local_reqs = []
        mgr.num_local_tokens = 0
        mgr.token_partition = [[], []]
        mgr.reorchestrate_cache()
        assert alloc._size_full == 64
        assert alloc._size_swa == 32


class TestSWAIndexTranslation:
    """Test B (P3): _full_to_swa translates correctly with divergent mapping."""

    def test_full_to_swa_with_divergent_mapping(self):
        from sglang.srt.paras.cache_transfer.swa import SWACacheTransfer

        mapping = torch.zeros(20, dtype=torch.int64)
        mapping[3] = 1
        mapping[7] = 2
        mapping[11] = 3
        mapping[15] = 4

        class FakeKVCache:
            full_to_swa_index_mapping = mapping
            head_num = 4
            head_dim = 64
            store_dtype = torch.bfloat16
            device = 'cpu'
            layer_num = 2

        class FakeGroup:
            world_size = 1
            device_group = None

        backend = object.__new__(SWACacheTransfer)
        backend.kv_cache = FakeKVCache()
        backend._full_to_swa_mapping = mapping

        full_indices = torch.tensor([3, 7, 11, 15], dtype=torch.int64)
        swa_indices = backend._full_to_swa(full_indices)
        assert torch.equal(swa_indices, torch.tensor([1, 2, 3, 4], dtype=torch.int64))

    def test_full_to_swa_unallocated_returns_zero(self):
        from sglang.srt.paras.cache_transfer.swa import SWACacheTransfer

        mapping = torch.zeros(20, dtype=torch.int64)
        mapping[5] = 10

        backend = object.__new__(SWACacheTransfer)
        backend._full_to_swa_mapping = mapping

        full_indices = torch.tensor([5, 8], dtype=torch.int64)
        swa_indices = backend._full_to_swa(full_indices)
        assert swa_indices[0].item() == 10
        assert swa_indices[1].item() == 0

    def test_full_to_swa_empty_tensor(self):
        from sglang.srt.paras.cache_transfer.swa import SWACacheTransfer

        backend = object.__new__(SWACacheTransfer)
        backend._full_to_swa_mapping = torch.zeros(10, dtype=torch.int64)

        empty = torch.empty(0, dtype=torch.int64)
        result = backend._full_to_swa(empty)
        assert result.numel() == 0

    def test_full_to_swa_none_passthrough(self):
        from sglang.srt.paras.cache_transfer.swa import SWACacheTransfer

        backend = object.__new__(SWACacheTransfer)
        backend._full_to_swa_mapping = torch.zeros(10, dtype=torch.int64)

        assert backend._full_to_swa(None) is None

    def test_full_to_swa_no_mapping_passthrough(self):
        from sglang.srt.paras.cache_transfer.swa import SWACacheTransfer

        backend = object.__new__(SWACacheTransfer)
        backend._full_to_swa_mapping = None

        indices = torch.tensor([1, 2, 3], dtype=torch.int64)
        result = backend._full_to_swa(indices)
        assert torch.equal(result, indices)


class TestPeerAccessPerDestCap:
    """Test C (P5): peer-access SWA scatter caps per-destination, not globally.

    After the scatter refactor, SWA-specific capping and index translation
    live in SWACacheTransfer precompute methods.  These tests exercise the
    full scatter_one_layer path with a mocked shared helper.
    """

    def _make_swa_stub_for_peer_access(self, token_partition, global_token_indices,
                                        group_size, num_kv_heads, mapping=None):
        from sglang.srt.paras.cache_transfer.swa import SWACacheTransfer

        stub = SWACacheTransfer.__new__(SWACacheTransfer)
        device = "cpu"
        stub.token_partition = token_partition
        stub.global_token_indices = global_token_indices
        stub.group_size = group_size
        stub._num_kv_heads = num_kv_heads
        stub._replication_factor = (
            group_size // num_kv_heads if num_kv_heads < group_size else 1
        )
        stub.paras_tp_rank = 0
        stub.paras_tp_size = 1
        stub._heads_per_rank = 1
        stub._head_dim = 64
        stub._elem_size = 2
        stub._local_buffer_ptr = 0
        stub._peer_buffer_ptrs = torch.zeros(group_size, dtype=torch.int64)
        stub._full_to_swa_mapping = mapping
        stub.method = "peer_access"
        stub.ep_dst_positions = None

        class _StubEntry:
            offset_bytes = 0
        class _StubMgr:
            _entries = {}
        stub_mgr = _StubMgr()
        for name in [
            "model.layers.0.kv.tp.k", "model.layers.0.kv.tp.v",
            "model.layers.0.kv.ep.k", "model.layers.0.kv.ep.v",
        ]:
            stub_mgr._entries[name] = _StubEntry()
        stub.mgr = stub_mgr
        return stub

    def test_per_destination_capping(self, monkeypatch):
        import sglang.srt.paras.cache_transfer.swa as swa_mod
        from sglang.srt.paras.cache_transfer.base import LayerCacheSpec

        spec = LayerCacheSpec(
            layer_id=0, kind='swa', tokens_cap_ep=3,
            tokens_cap_tp=0, num_kv_heads=4, head_dim=64,
            sliding_window_size=1023,
        )

        token_partition = [
            list(range(0, 5)),
            list(range(5, 10)),
            list(range(10, 15)),
        ]
        global_token_indices = torch.arange(1, 16, dtype=torch.int64)

        captured_args = {}
        def fake_scatter_peer_access(
            local_buffer_ptr, peer_buffer_ptrs, tp_token_positions, token_to_rank,
            ep_dst_pos_all, src_k_offset, src_v_offset, dst_k_offset, dst_v_offset,
            num_my_tokens, layer_id, heads_per_rank, num_kv_heads,
            paras_tp_rank, paras_tp_size, head_dim, elem_size,
        ):
            captured_args['tp_positions'] = tp_token_positions
            captured_args['token_to_rank'] = token_to_rank
            captured_args['ep_dst_pos'] = ep_dst_pos_all
            captured_args['layer_num'] = num_my_tokens

        monkeypatch.setattr(swa_mod, "do_scatter_one_layer_peer_access", fake_scatter_peer_access)

        stub = self._make_swa_stub_for_peer_access(
            token_partition, global_token_indices, group_size=3, num_kv_heads=4,
        )
        stub.scatter_one_layer(spec)

        assert captured_args['layer_num'] == 9, (
            f"Expected 3 dests * 3 tokens/dest = 9, got {captured_args['layer_num']}"
        )

        ranks = captured_args['token_to_rank'].tolist()
        assert ranks.count(0) == 3
        assert ranks.count(1) == 3
        assert ranks.count(2) == 3

    def test_per_destination_capping_with_translation(self, monkeypatch):
        import sglang.srt.paras.cache_transfer.swa as swa_mod
        from sglang.srt.paras.cache_transfer.base import LayerCacheSpec

        spec = LayerCacheSpec(
            layer_id=0, kind='swa', tokens_cap_ep=2,
            tokens_cap_tp=0, num_kv_heads=4, head_dim=64,
            sliding_window_size=1023,
        )

        token_partition = [list(range(0, 4)), list(range(4, 8))]
        global_token_indices = torch.tensor(
            [10, 20, 30, 40, 50, 60, 70, 80], dtype=torch.int64
        )

        mapping = torch.zeros(100, dtype=torch.int64)
        mapping[10] = 1; mapping[20] = 2
        mapping[50] = 3; mapping[60] = 4

        captured_args = {}
        def fake_scatter_peer_access(
            local_buffer_ptr, peer_buffer_ptrs, tp_token_positions, token_to_rank,
            ep_dst_pos_all, src_k_offset, src_v_offset, dst_k_offset, dst_v_offset,
            num_my_tokens, layer_id, heads_per_rank, num_kv_heads,
            paras_tp_rank, paras_tp_size, head_dim, elem_size,
        ):
            captured_args['tp_positions'] = tp_token_positions
            captured_args['ep_dst_pos'] = ep_dst_pos_all
            captured_args['layer_num'] = num_my_tokens

        monkeypatch.setattr(swa_mod, "do_scatter_one_layer_peer_access", fake_scatter_peer_access)

        stub = self._make_swa_stub_for_peer_access(
            token_partition, global_token_indices, group_size=2, num_kv_heads=4,
            mapping=mapping,
        )
        stub.scatter_one_layer(spec)

        assert captured_args['layer_num'] == 4
        tp_pos = captured_args['tp_positions'].tolist()
        assert tp_pos == [1, 2, 3, 4]


class TestSWAFullToSwaDtype:
    """Verify _full_to_swa preserves caller's input dtype.

    Peer-access CUDA bindings read token-index tensors as int32* (see
    csrc/binding.cpp); NCCL paths use torch advanced indexing which needs
    int64.  Both consumers go through _full_to_swa, so it must round-trip
    dtype faithfully.  Regression guard for peer-access SWA corruption.
    """

    def _make_swa_transfer_with_mapping(self):
        """Minimal SWACacheTransfer stub that only needs _full_to_swa_mapping."""
        from sglang.srt.paras.cache_transfer.swa import SWACacheTransfer
        # Bypass __init__ — we only need the mapping attribute for _full_to_swa.
        stub = SWACacheTransfer.__new__(SWACacheTransfer)
        # Non-trivial mapping so we can check translation correctness too.
        # Full index i maps to SWA index (i * 3) % 17.
        stub._full_to_swa_mapping = torch.tensor(
            [(i * 3) % 17 for i in range(32)], dtype=torch.int64
        )
        return stub

    def test_preserves_int32_dtype(self):
        stub = self._make_swa_transfer_with_mapping()
        full_idx = torch.tensor([1, 4, 7], dtype=torch.int32)
        result = stub._full_to_swa(full_idx)
        assert result.dtype == torch.int32, (
            f"Expected int32 (peer-access contract), got {result.dtype}"
        )
        # Correctness: (1*3)%17=3, (4*3)%17=12, (7*3)%17=4.
        assert result.tolist() == [3, 12, 4]

    def test_preserves_int64_dtype(self):
        stub = self._make_swa_transfer_with_mapping()
        full_idx = torch.tensor([1, 4, 7], dtype=torch.int64)
        result = stub._full_to_swa(full_idx)
        assert result.dtype == torch.int64, (
            f"Expected int64 (NCCL torch-indexing contract), got {result.dtype}"
        )
        assert result.tolist() == [3, 12, 4]

    def test_none_and_empty_are_passthrough(self):
        stub = self._make_swa_transfer_with_mapping()
        assert stub._full_to_swa(None) is None
        empty_i32 = torch.tensor([], dtype=torch.int32)
        out = stub._full_to_swa(empty_i32)
        # Empty tensor returned as-is; dtype must survive.
        assert out.numel() == 0
        assert out.dtype == torch.int32

    def test_missing_mapping_is_passthrough(self):
        """If allocator didn't attach a mapping, pass through unchanged."""
        from sglang.srt.paras.cache_transfer.swa import SWACacheTransfer
        stub = SWACacheTransfer.__new__(SWACacheTransfer)
        stub._full_to_swa_mapping = None
        t = torch.tensor([1, 2, 3], dtype=torch.int32)
        out = stub._full_to_swa(t)
        assert out is t  # identity — no copy


class TestSWAScatterRefactor:
    """Verify the refactored SWA scatter delegates to shared helpers correctly."""

    def _make_swa_stub(self, token_partition, mapping_values,
                        global_token_indices=None):
        from sglang.srt.paras.cache_transfer.swa import SWACacheTransfer
        stub = SWACacheTransfer.__new__(SWACacheTransfer)

        device = "cpu"
        n_tokens = sum(len(p) for p in token_partition)
        if global_token_indices is not None:
            stub.global_token_indices = global_token_indices
        else:
            stub.global_token_indices = torch.arange(
                n_tokens, dtype=torch.long, device=device)

        stub.token_partition = token_partition
        stub.ep_dst_positions = torch.arange(
            1, len(token_partition[0]) + 1, dtype=torch.long, device=device,
        )
        stub.group_size = len(token_partition)
        stub._intra_rank = 0
        stub._replication_factor = 1
        stub._per_token_elems = 8
        stub._recv_full_count = len(token_partition[0])
        stub._num_kv_heads = 4
        stub._heads_per_rank = 1
        stub._head_dim = 4
        stub._total_global_tokens = n_tokens
        stub._reassembly_groups = len(token_partition)
        stub.ep_head_num = 4
        stub.paras_tp_rank = 0
        stub.paras_tp_size = 4
        stub.method = "nccl"
        stub._full_to_swa_mapping = torch.tensor(
            mapping_values, dtype=torch.int64, device=device)

        class _StubKv:
            store_dtype = torch.bfloat16
            device = "cpu"
            head_num = 1
            head_dim = 4
        stub.kv_cache = _StubKv()
        stub.mgr = object()
        stub.group = object()

        stub._local_buffer_ptr = 0
        stub._peer_buffer_ptrs = torch.empty(0, dtype=torch.int64, device=device)
        stub._elem_size = 2

        return stub

    def test_nccl_scatter_caps_per_destination(self, monkeypatch):
        import sglang.srt.paras.cache_transfer.swa as swa_mod
        from sglang.srt.paras.cache_transfer.base import LayerCacheSpec

        captured = {}
        def fake_scatter_nccl(
            kv_cache, ep_head_num, layer_id, token_partition, group_size,
            intra_rank, replication_factor, per_token_elems,
            global_token_indices, ep_dst_positions, sorted_tp_indices,
            total_send_tokens, input_split_sizes, recv_full_count,
            output_split_sizes, total_recv_elems,
            num_kv_heads, heads_per_rank, head_dim, total_global_tokens,
            reassembly_groups, mgr, gather_group,
        ):
            captured["input_split_sizes"] = input_split_sizes
            captured["output_split_sizes"] = output_split_sizes
            captured["sorted_tp_indices"] = sorted_tp_indices
            captured["total_send_tokens"] = total_send_tokens
            captured["recv_full_count"] = recv_full_count

        monkeypatch.setattr(swa_mod, "do_scatter_one_layer_nccl", fake_scatter_nccl)

        token_partition = [[0, 1, 2, 3, 4], [5, 6, 7], [8, 9, 10, 11, 12, 13, 14]]
        mapping = list(range(50))
        stub = self._make_swa_stub(token_partition, mapping)
        stub._recv_full_count = 5
        stub.method = "nccl"

        spec = LayerCacheSpec(
            layer_id=0, kind="swa", tokens_cap_ep=3, tokens_cap_tp=0,
            num_kv_heads=4, head_dim=4, sliding_window_size=10,
        )
        stub.scatter_one_layer(spec)

        expected_send = [3 * stub._per_token_elems] * 3
        assert captured["input_split_sizes"] == expected_send, (
            f"Per-destination cap failed: {captured['input_split_sizes']} != {expected_send}"
        )
        assert captured["total_send_tokens"] == 9
        assert captured["recv_full_count"] == 3

    def test_nccl_scatter_translates_indices_to_swa_space(self, monkeypatch):
        import sglang.srt.paras.cache_transfer.swa as swa_mod
        from sglang.srt.paras.cache_transfer.base import LayerCacheSpec

        captured = {}
        def fake_scatter_nccl(*args, **kwargs):
            captured["ep_dst_positions"] = args[9]
            captured["sorted_tp_indices"] = args[10]

        monkeypatch.setattr(swa_mod, "do_scatter_one_layer_nccl", fake_scatter_nccl)

        token_partition = [[0, 1], [2, 3]]
        global_token_indices = torch.tensor([3, 7, 11, 19], dtype=torch.long)
        mapping = [2 * i + 1 for i in range(50)]
        stub = self._make_swa_stub(token_partition, mapping,
                                    global_token_indices=global_token_indices)
        stub._recv_full_count = 2

        spec = LayerCacheSpec(
            layer_id=0, kind="swa", tokens_cap_ep=10, tokens_cap_tp=0,
            num_kv_heads=4, head_dim=4, sliding_window_size=10,
        )
        stub.scatter_one_layer(spec)

        expected_swa = [7, 15, 23, 39]
        actual = captured["sorted_tp_indices"].tolist()
        assert actual == expected_swa, (
            f"sorted_tp_indices not translated to SWA space: {actual} != {expected_swa}"
        )

    def test_peer_access_scatter_delegates_with_int32_indices(self, monkeypatch):
        import sglang.srt.paras.cache_transfer.swa as swa_mod
        from sglang.srt.paras.cache_transfer.base import LayerCacheSpec

        captured = {}
        def fake_scatter_peer_access(
            local_buffer_ptr, peer_buffer_ptrs, tp_token_positions, token_to_rank,
            ep_dst_pos_all, src_k_offset, src_v_offset, dst_k_offset, dst_v_offset,
            num_my_tokens, layer_id, heads_per_rank, num_kv_heads,
            paras_tp_rank, paras_tp_size, head_dim, elem_size,
        ):
            captured["tp_positions"] = tp_token_positions
            captured["token_to_rank"] = token_to_rank
            captured["ep_dst_pos"] = ep_dst_pos_all
            captured["num_my_tokens"] = num_my_tokens

        monkeypatch.setattr(swa_mod, "do_scatter_one_layer_peer_access", fake_scatter_peer_access)

        class _StubEntry:
            offset_bytes = 0
        class _StubMgr:
            _entries = {
                "model.layers.0.kv.tp.k": _StubEntry(),
                "model.layers.0.kv.tp.v": _StubEntry(),
                "model.layers.0.kv.ep.k": _StubEntry(),
                "model.layers.0.kv.ep.v": _StubEntry(),
            }

        token_partition = [[0, 1], [2, 3]]
        global_token_indices = torch.tensor([3, 7, 11, 19], dtype=torch.long)
        mapping = [2 * i + 1 for i in range(50)]
        stub = self._make_swa_stub(token_partition, mapping,
                                    global_token_indices=global_token_indices)
        stub.method = "peer_access"
        stub.mgr = _StubMgr()

        spec = LayerCacheSpec(
            layer_id=0, kind="swa", tokens_cap_ep=10, tokens_cap_tp=0,
            num_kv_heads=4, head_dim=4, sliding_window_size=10,
        )
        stub.scatter_one_layer(spec)

        assert captured["tp_positions"].dtype == torch.int32, (
            f"tp_positions dtype = {captured['tp_positions'].dtype}, expected int32"
        )
        assert captured["token_to_rank"].dtype == torch.int32
        assert captured["ep_dst_pos"].dtype == torch.int32
        assert captured["num_my_tokens"] == 4


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
