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
from sglang.srt.paras.paras_memory_manager import ParaSMemoryManager, create_paras_kv_aliases
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
    create_paras_kv_aliases(mgr, num_layers=NUM_LAYERS)
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
