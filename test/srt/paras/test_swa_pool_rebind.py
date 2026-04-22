#!/usr/bin/env python3
"""
Unit tests for SWAKVPool.paras_configure_tp/ep buffer rebinding.

Tests cover:
  1. TP rebind — after paras_configure_tp(tp_size, layer_specs), verify inner pool
     buffers are rebound to TP alias views from the memory manager
  2. EP rebind — after paras_configure_ep(layer_specs), verify inner pool buffers
     are rebound to EP alias views from the memory manager
  3. Head count update — TP mode shards head_num to head_num // tp_size,
     EP mode restores to full_head_num
  4. full_to_swa_index_mapping untouched — T8 owns this, not modified by rebind
  5. Round-trip EP→TP→EP — buffer pointers preserved at each stage

All tests are CPU-only (no GPU required).

Usage:
  conda run -n sgl_paras python -m pytest test/srt/paras/test_swa_pool_rebind.py -v
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

# Add sglang to path
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.join(_TEST_DIR, "..", "..", "..")
sys.path.insert(0, os.path.join(_ROOT_DIR, "python"))


# =========================================================================
# TEST GROUP 1: TP rebind — buffer pointer verification
# =========================================================================


class TestTPRebindBufferPointers:
    """Verify TP rebind correctly rewires inner pool buffers to TP aliases."""

    def test_tp_rebind_full_layer_buffer_pointers(self):
        """After TP rebind, full layer k/v buffers match TP alias data_ptr()."""
        from sglang.srt.mem_cache.memory_pool import SWAKVPool, MHATokenToKVPool
        from sglang.srt.paras.paras_memory_manager import ParaSMemoryManager

        # Setup: 2 full + 1 SWA layer
        full_layer_ids = [0, 1]
        swa_layer_ids = [2]
        tp_size = 4
        head_num = 8
        head_dim = 128

        # Create SWAKVPool
        pool = SWAKVPool(
            size=1024,
            size_swa=256,
            dtype=torch.bfloat16,
            head_num=head_num,
            head_dim=head_dim,
            swa_attention_layer_ids=swa_layer_ids,
            full_attention_layer_ids=full_layer_ids,
            enable_kvcache_transpose=False,
            device="cpu",
            token_to_kv_pool_class=MHATokenToKVPool,
        )

        # Create mock memory manager with TP aliases
        mgr = MagicMock(spec=ParaSMemoryManager)
        mgr._entries = {}

        # Create TP alias buffers for each layer
        tp_k_buffers = {}
        tp_v_buffers = {}
        for g in [0, 1, 2]:
            tp_k_name = f"model.layers.{g}.kv.tp.k"
            tp_v_name = f"model.layers.{g}.kv.tp.v"
            sharded_head_num = head_num // tp_size
            tp_shape = (1024, sharded_head_num, head_dim)
            tp_k = torch.zeros(tp_shape, dtype=torch.bfloat16)
            tp_v = torch.zeros(tp_shape, dtype=torch.bfloat16)
            tp_k_buffers[tp_k_name] = tp_k
            tp_v_buffers[tp_v_name] = tp_v

            # Mock entry with numel
            mgr._entries[tp_k_name] = MagicMock()
            mgr._entries[tp_k_name].numel = tp_shape[0] * tp_shape[1] * tp_shape[2]
            mgr._entries[tp_v_name] = MagicMock()
            mgr._entries[tp_v_name].numel = tp_shape[0] * tp_shape[1] * tp_shape[2]

        # Mock get_view_as to return the TP buffers
        def mock_get_view_as(name, shape):
            if name in tp_k_buffers:
                return tp_k_buffers[name]
            elif name in tp_v_buffers:
                return tp_v_buffers[name]
            raise KeyError(f"Unknown buffer: {name}")

        mgr.get_view_as = mock_get_view_as

        # Patch global manager
        with patch(
            "sglang.srt.paras.paras_memory_manager.get_global_paras_memory_manager",
            return_value=mgr,
        ):
            # Call paras_configure_tp
            pool.paras_configure_tp(tp_size, layer_specs=None)

        # Verify: full layer buffers match TP aliases
        for g in [0, 1]:
            local_id, is_swa = pool.layers_mapping[g]
            assert not is_swa, f"Layer {g} should be full"
            tp_k_name = f"model.layers.{g}.kv.tp.k"
            tp_v_name = f"model.layers.{g}.kv.tp.v"
            assert pool.full_kv_pool.k_buffer[local_id].data_ptr() == tp_k_buffers[
                tp_k_name
            ].data_ptr(), f"Full layer {g} k_buffer mismatch"
            assert pool.full_kv_pool.v_buffer[local_id].data_ptr() == tp_v_buffers[
                tp_v_name
            ].data_ptr(), f"Full layer {g} v_buffer mismatch"

    def test_tp_rebind_swa_layer_buffer_pointers(self):
        """After TP rebind, SWA layer k/v buffers match TP alias data_ptr()."""
        from sglang.srt.mem_cache.memory_pool import SWAKVPool, MHATokenToKVPool
        from sglang.srt.paras.paras_memory_manager import ParaSMemoryManager

        # Setup: 1 full + 2 SWA layers
        full_layer_ids = [0]
        swa_layer_ids = [1, 2]
        tp_size = 2
        head_num = 8
        head_dim = 128

        pool = SWAKVPool(
            size=1024,
            size_swa=256,
            dtype=torch.bfloat16,
            head_num=head_num,
            head_dim=head_dim,
            swa_attention_layer_ids=swa_layer_ids,
            full_attention_layer_ids=full_layer_ids,
            enable_kvcache_transpose=False,
            device="cpu",
            token_to_kv_pool_class=MHATokenToKVPool,
        )

        mgr = MagicMock(spec=ParaSMemoryManager)
        mgr._entries = {}

        tp_k_buffers = {}
        tp_v_buffers = {}
        for g in [0, 1, 2]:
            tp_k_name = f"model.layers.{g}.kv.tp.k"
            tp_v_name = f"model.layers.{g}.kv.tp.v"
            sharded_head_num = head_num // tp_size
            tp_shape = (512, sharded_head_num, head_dim)
            tp_k = torch.zeros(tp_shape, dtype=torch.bfloat16)
            tp_v = torch.zeros(tp_shape, dtype=torch.bfloat16)
            tp_k_buffers[tp_k_name] = tp_k
            tp_v_buffers[tp_v_name] = tp_v

            mgr._entries[tp_k_name] = MagicMock()
            mgr._entries[tp_k_name].numel = tp_shape[0] * tp_shape[1] * tp_shape[2]
            mgr._entries[tp_v_name] = MagicMock()
            mgr._entries[tp_v_name].numel = tp_shape[0] * tp_shape[1] * tp_shape[2]

        def mock_get_view_as(name, shape):
            if name in tp_k_buffers:
                return tp_k_buffers[name]
            elif name in tp_v_buffers:
                return tp_v_buffers[name]
            raise KeyError(f"Unknown buffer: {name}")

        mgr.get_view_as = mock_get_view_as

        with patch(
            "sglang.srt.paras.paras_memory_manager.get_global_paras_memory_manager",
            return_value=mgr,
        ):
            pool.paras_configure_tp(tp_size, layer_specs=None)

        # Verify: SWA layer buffers match TP aliases
        for g in [1, 2]:
            local_id, is_swa = pool.layers_mapping[g]
            assert is_swa, f"Layer {g} should be SWA"
            tp_k_name = f"model.layers.{g}.kv.tp.k"
            tp_v_name = f"model.layers.{g}.kv.tp.v"
            assert pool.swa_kv_pool.k_buffer[local_id].data_ptr() == tp_k_buffers[
                tp_k_name
            ].data_ptr(), f"SWA layer {g} k_buffer mismatch"
            assert pool.swa_kv_pool.v_buffer[local_id].data_ptr() == tp_v_buffers[
                tp_v_name
            ].data_ptr(), f"SWA layer {g} v_buffer mismatch"


# =========================================================================
# TEST GROUP 2: EP rebind — buffer pointer verification
# =========================================================================


class TestEPRebindBufferPointers:
    """Verify EP rebind correctly rewires inner pool buffers to EP aliases."""

    def test_ep_rebind_full_layer_buffer_pointers(self):
        """After EP rebind, full layer k/v buffers match EP alias data_ptr()."""
        from sglang.srt.mem_cache.memory_pool import SWAKVPool, MHATokenToKVPool
        from sglang.srt.paras.paras_memory_manager import ParaSMemoryManager

        # Setup: 2 full + 1 SWA layer
        full_layer_ids = [0, 1]
        swa_layer_ids = [2]
        head_num = 8
        head_dim = 128

        pool = SWAKVPool(
            size=1024,
            size_swa=256,
            dtype=torch.bfloat16,
            head_num=head_num,
            head_dim=head_dim,
            swa_attention_layer_ids=swa_layer_ids,
            full_attention_layer_ids=full_layer_ids,
            enable_kvcache_transpose=False,
            device="cpu",
            token_to_kv_pool_class=MHATokenToKVPool,
        )

        mgr = MagicMock(spec=ParaSMemoryManager)
        mgr._entries = {}

        ep_k_buffers = {}
        ep_v_buffers = {}
        for g in [0, 1, 2]:
            ep_k_name = f"model.layers.{g}.kv.ep.k"
            ep_v_name = f"model.layers.{g}.kv.ep.v"
            ep_shape = (1024, head_num, head_dim)
            ep_k = torch.zeros(ep_shape, dtype=torch.bfloat16)
            ep_v = torch.zeros(ep_shape, dtype=torch.bfloat16)
            ep_k_buffers[ep_k_name] = ep_k
            ep_v_buffers[ep_v_name] = ep_v

            mgr._entries[ep_k_name] = MagicMock()
            mgr._entries[ep_k_name].numel = ep_shape[0] * ep_shape[1] * ep_shape[2]
            mgr._entries[ep_v_name] = MagicMock()
            mgr._entries[ep_v_name].numel = ep_shape[0] * ep_shape[1] * ep_shape[2]

        def mock_get_view_as(name, shape):
            if name in ep_k_buffers:
                return ep_k_buffers[name]
            elif name in ep_v_buffers:
                return ep_v_buffers[name]
            raise KeyError(f"Unknown buffer: {name}")

        mgr.get_view_as = mock_get_view_as

        with patch(
            "sglang.srt.paras.paras_memory_manager.get_global_paras_memory_manager",
            return_value=mgr,
        ):
            pool.paras_configure_ep(layer_specs=None)

        # Verify: full layer buffers match EP aliases
        for g in [0, 1]:
            local_id, is_swa = pool.layers_mapping[g]
            assert not is_swa, f"Layer {g} should be full"
            ep_k_name = f"model.layers.{g}.kv.ep.k"
            ep_v_name = f"model.layers.{g}.kv.ep.v"
            assert pool.full_kv_pool.k_buffer[local_id].data_ptr() == ep_k_buffers[
                ep_k_name
            ].data_ptr(), f"Full layer {g} k_buffer mismatch"
            assert pool.full_kv_pool.v_buffer[local_id].data_ptr() == ep_v_buffers[
                ep_v_name
            ].data_ptr(), f"Full layer {g} v_buffer mismatch"

    def test_ep_rebind_swa_layer_buffer_pointers(self):
        """After EP rebind, SWA layer k/v buffers match EP alias data_ptr()."""
        from sglang.srt.mem_cache.memory_pool import SWAKVPool, MHATokenToKVPool
        from sglang.srt.paras.paras_memory_manager import ParaSMemoryManager

        # Setup: 1 full + 2 SWA layers
        full_layer_ids = [0]
        swa_layer_ids = [1, 2]
        head_num = 8
        head_dim = 128

        pool = SWAKVPool(
            size=1024,
            size_swa=256,
            dtype=torch.bfloat16,
            head_num=head_num,
            head_dim=head_dim,
            swa_attention_layer_ids=swa_layer_ids,
            full_attention_layer_ids=full_layer_ids,
            enable_kvcache_transpose=False,
            device="cpu",
            token_to_kv_pool_class=MHATokenToKVPool,
        )

        mgr = MagicMock(spec=ParaSMemoryManager)
        mgr._entries = {}

        ep_k_buffers = {}
        ep_v_buffers = {}
        for g in [0, 1, 2]:
            ep_k_name = f"model.layers.{g}.kv.ep.k"
            ep_v_name = f"model.layers.{g}.kv.ep.v"
            ep_shape = (512, head_num, head_dim)
            ep_k = torch.zeros(ep_shape, dtype=torch.bfloat16)
            ep_v = torch.zeros(ep_shape, dtype=torch.bfloat16)
            ep_k_buffers[ep_k_name] = ep_k
            ep_v_buffers[ep_v_name] = ep_v

            mgr._entries[ep_k_name] = MagicMock()
            mgr._entries[ep_k_name].numel = ep_shape[0] * ep_shape[1] * ep_shape[2]
            mgr._entries[ep_v_name] = MagicMock()
            mgr._entries[ep_v_name].numel = ep_shape[0] * ep_shape[1] * ep_shape[2]

        def mock_get_view_as(name, shape):
            if name in ep_k_buffers:
                return ep_k_buffers[name]
            elif name in ep_v_buffers:
                return ep_v_buffers[name]
            raise KeyError(f"Unknown buffer: {name}")

        mgr.get_view_as = mock_get_view_as

        with patch(
            "sglang.srt.paras.paras_memory_manager.get_global_paras_memory_manager",
            return_value=mgr,
        ):
            pool.paras_configure_ep(layer_specs=None)

        # Verify: SWA layer buffers match EP aliases
        for g in [1, 2]:
            local_id, is_swa = pool.layers_mapping[g]
            assert is_swa, f"Layer {g} should be SWA"
            ep_k_name = f"model.layers.{g}.kv.ep.k"
            ep_v_name = f"model.layers.{g}.kv.ep.v"
            assert pool.swa_kv_pool.k_buffer[local_id].data_ptr() == ep_k_buffers[
                ep_k_name
            ].data_ptr(), f"SWA layer {g} k_buffer mismatch"
            assert pool.swa_kv_pool.v_buffer[local_id].data_ptr() == ep_v_buffers[
                ep_v_name
            ].data_ptr(), f"SWA layer {g} v_buffer mismatch"


# =========================================================================
# TEST GROUP 3: Head count update — TP shards, EP restores
# =========================================================================


class TestHeadCountUpdate:
    """Verify head_num is correctly sharded in TP mode and restored in EP mode."""

    def test_tp_rebind_shards_head_num(self):
        """After TP rebind, head_num should be sharded to head_num // tp_size."""
        from sglang.srt.mem_cache.memory_pool import SWAKVPool, MHATokenToKVPool
        from sglang.srt.paras.paras_memory_manager import ParaSMemoryManager

        full_layer_ids = [0]
        swa_layer_ids = [1]
        original_head_num = 8
        tp_size = 4
        head_dim = 128

        pool = SWAKVPool(
            size=1024,
            size_swa=256,
            dtype=torch.bfloat16,
            head_num=original_head_num,
            head_dim=head_dim,
            swa_attention_layer_ids=swa_layer_ids,
            full_attention_layer_ids=full_layer_ids,
            enable_kvcache_transpose=False,
            device="cpu",
            token_to_kv_pool_class=MHATokenToKVPool,
        )

        # Verify initial state
        assert pool.head_num == original_head_num
        assert pool.full_head_num == original_head_num

        mgr = MagicMock(spec=ParaSMemoryManager)
        mgr._entries = {}

        for g in [0, 1]:
            for suffix in ["k", "v"]:
                tp_name = f"model.layers.{g}.kv.tp.{suffix}"
                sharded_head_num = original_head_num // tp_size
                tp_shape = (512, sharded_head_num, head_dim)
                tp_buf = torch.zeros(tp_shape, dtype=torch.bfloat16)
                mgr._entries[tp_name] = MagicMock()
                mgr._entries[tp_name].numel = (
                    tp_shape[0] * tp_shape[1] * tp_shape[2]
                )

                if suffix == "k":
                    mgr._entries[tp_name].tensor = tp_buf
                else:
                    mgr._entries[tp_name].tensor = tp_buf

        def mock_get_view_as(name, shape):
            sharded_head_num = original_head_num // tp_size
            return torch.zeros(shape, dtype=torch.bfloat16)

        mgr.get_view_as = mock_get_view_as

        with patch(
            "sglang.srt.paras.paras_memory_manager.get_global_paras_memory_manager",
            return_value=mgr,
        ):
            pool.paras_configure_tp(tp_size, layer_specs=None)

        # Verify: head_num is sharded
        expected_sharded = original_head_num // tp_size
        assert pool.head_num == expected_sharded, (
            f"After TP: expected head_num={expected_sharded}, got {pool.head_num}"
        )
        # full_head_num should be saved
        assert pool.full_head_num == original_head_num

    def test_ep_rebind_restores_head_num(self):
        """After EP rebind, head_num should be restored to full_head_num."""
        from sglang.srt.mem_cache.memory_pool import SWAKVPool, MHATokenToKVPool
        from sglang.srt.paras.paras_memory_manager import ParaSMemoryManager

        full_layer_ids = [0]
        swa_layer_ids = [1]
        original_head_num = 8
        tp_size = 2
        head_dim = 128

        pool = SWAKVPool(
            size=1024,
            size_swa=256,
            dtype=torch.bfloat16,
            head_num=original_head_num,
            head_dim=head_dim,
            swa_attention_layer_ids=swa_layer_ids,
            full_attention_layer_ids=full_layer_ids,
            enable_kvcache_transpose=False,
            device="cpu",
            token_to_kv_pool_class=MHATokenToKVPool,
        )

        # Manually set head_num to sharded state (simulating after TP)
        pool.head_num = original_head_num // tp_size
        pool.full_head_num = original_head_num

        mgr = MagicMock(spec=ParaSMemoryManager)
        mgr._entries = {}

        for g in [0, 1]:
            for suffix in ["k", "v"]:
                ep_name = f"model.layers.{g}.kv.ep.{suffix}"
                ep_shape = (512, original_head_num, head_dim)
                ep_buf = torch.zeros(ep_shape, dtype=torch.bfloat16)
                mgr._entries[ep_name] = MagicMock()
                mgr._entries[ep_name].numel = (
                    ep_shape[0] * ep_shape[1] * ep_shape[2]
                )

        def mock_get_view_as(name, shape):
            return torch.zeros(shape, dtype=torch.bfloat16)

        mgr.get_view_as = mock_get_view_as

        with patch(
            "sglang.srt.paras.paras_memory_manager.get_global_paras_memory_manager",
            return_value=mgr,
        ):
            pool.paras_configure_ep(layer_specs=None)

        # Verify: head_num is restored
        assert pool.head_num == original_head_num, (
            f"After EP: expected head_num={original_head_num}, got {pool.head_num}"
        )


# =========================================================================
# TEST GROUP 4: full_to_swa_index_mapping untouched
# =========================================================================


class TestFullToSWAIndexMappingUntouched:
    """Verify full_to_swa_index_mapping is not modified by TP/EP rebind."""

    def test_tp_rebind_does_not_modify_mapping(self):
        """TP rebind should not touch full_to_swa_index_mapping."""
        from sglang.srt.mem_cache.memory_pool import SWAKVPool, MHATokenToKVPool
        from sglang.srt.paras.paras_memory_manager import ParaSMemoryManager

        full_layer_ids = [0]
        swa_layer_ids = [1]
        tp_size = 2
        head_num = 8
        head_dim = 128

        pool = SWAKVPool(
            size=1024,
            size_swa=256,
            dtype=torch.bfloat16,
            head_num=head_num,
            head_dim=head_dim,
            swa_attention_layer_ids=swa_layer_ids,
            full_attention_layer_ids=full_layer_ids,
            enable_kvcache_transpose=False,
            device="cpu",
            token_to_kv_pool_class=MHATokenToKVPool,
        )

        # Set a mapping (simulating T8 work)
        original_mapping = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32)
        pool.full_to_swa_index_mapping = original_mapping

        mgr = MagicMock(spec=ParaSMemoryManager)
        mgr._entries = {}

        for g in [0, 1]:
            for suffix in ["k", "v"]:
                tp_name = f"model.layers.{g}.kv.tp.{suffix}"
                sharded_head_num = head_num // tp_size
                tp_shape = (512, sharded_head_num, head_dim)
                mgr._entries[tp_name] = MagicMock()
                mgr._entries[tp_name].numel = (
                    tp_shape[0] * tp_shape[1] * tp_shape[2]
                )

        def mock_get_view_as(name, shape):
            return torch.zeros(shape, dtype=torch.bfloat16)

        mgr.get_view_as = mock_get_view_as

        with patch(
            "sglang.srt.paras.paras_memory_manager.get_global_paras_memory_manager",
            return_value=mgr,
        ):
            pool.paras_configure_tp(tp_size, layer_specs=None)

        # Verify: mapping is unchanged
        assert pool.full_to_swa_index_mapping is original_mapping
        assert torch.equal(pool.full_to_swa_index_mapping, original_mapping)

    def test_ep_rebind_does_not_modify_mapping(self):
        """EP rebind should not touch full_to_swa_index_mapping."""
        from sglang.srt.mem_cache.memory_pool import SWAKVPool, MHATokenToKVPool
        from sglang.srt.paras.paras_memory_manager import ParaSMemoryManager

        full_layer_ids = [0]
        swa_layer_ids = [1]
        head_num = 8
        head_dim = 128

        pool = SWAKVPool(
            size=1024,
            size_swa=256,
            dtype=torch.bfloat16,
            head_num=head_num,
            head_dim=head_dim,
            swa_attention_layer_ids=swa_layer_ids,
            full_attention_layer_ids=full_layer_ids,
            enable_kvcache_transpose=False,
            device="cpu",
            token_to_kv_pool_class=MHATokenToKVPool,
        )

        # Set a mapping
        original_mapping = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32)
        pool.full_to_swa_index_mapping = original_mapping

        mgr = MagicMock(spec=ParaSMemoryManager)
        mgr._entries = {}

        for g in [0, 1]:
            for suffix in ["k", "v"]:
                ep_name = f"model.layers.{g}.kv.ep.{suffix}"
                ep_shape = (512, head_num, head_dim)
                mgr._entries[ep_name] = MagicMock()
                mgr._entries[ep_name].numel = (
                    ep_shape[0] * ep_shape[1] * ep_shape[2]
                )

        def mock_get_view_as(name, shape):
            return torch.zeros(shape, dtype=torch.bfloat16)

        mgr.get_view_as = mock_get_view_as

        with patch(
            "sglang.srt.paras.paras_memory_manager.get_global_paras_memory_manager",
            return_value=mgr,
        ):
            pool.paras_configure_ep(layer_specs=None)

        # Verify: mapping is unchanged
        assert pool.full_to_swa_index_mapping is original_mapping
        assert torch.equal(pool.full_to_swa_index_mapping, original_mapping)


# =========================================================================
# TEST GROUP 5: Round-trip EP→TP→EP preserves buffer pointers
# =========================================================================


class TestRoundTripEPTPEP:
    """Verify buffer pointers are preserved across EP→TP→EP round-trip."""

    def test_roundtrip_ep_tp_ep_buffer_preservation(self):
        """After EP→TP→EP, buffer pointers should match at each stage."""
        from sglang.srt.mem_cache.memory_pool import SWAKVPool, MHATokenToKVPool
        from sglang.srt.paras.paras_memory_manager import ParaSMemoryManager

        full_layer_ids = [0]
        swa_layer_ids = [1]
        tp_size = 2
        head_num = 8
        head_dim = 128

        pool = SWAKVPool(
            size=1024,
            size_swa=256,
            dtype=torch.bfloat16,
            head_num=head_num,
            head_dim=head_dim,
            swa_attention_layer_ids=swa_layer_ids,
            full_attention_layer_ids=full_layer_ids,
            enable_kvcache_transpose=False,
            device="cpu",
            token_to_kv_pool_class=MHATokenToKVPool,
        )

        # Create EP and TP buffers
        ep_k_buffers = {}
        ep_v_buffers = {}
        tp_k_buffers = {}
        tp_v_buffers = {}

        for g in [0, 1]:
            ep_k_name = f"model.layers.{g}.kv.ep.k"
            ep_v_name = f"model.layers.{g}.kv.ep.v"
            tp_k_name = f"model.layers.{g}.kv.tp.k"
            tp_v_name = f"model.layers.{g}.kv.tp.v"

            ep_shape = (512, head_num, head_dim)
            tp_shape = (512, head_num // tp_size, head_dim)

            ep_k_buffers[ep_k_name] = torch.zeros(ep_shape, dtype=torch.bfloat16)
            ep_v_buffers[ep_v_name] = torch.zeros(ep_shape, dtype=torch.bfloat16)
            tp_k_buffers[tp_k_name] = torch.zeros(tp_shape, dtype=torch.bfloat16)
            tp_v_buffers[tp_v_name] = torch.zeros(tp_shape, dtype=torch.bfloat16)

        mgr = MagicMock(spec=ParaSMemoryManager)
        mgr._entries = {}

        for g in [0, 1]:
            for suffix in ["k", "v"]:
                ep_name = f"model.layers.{g}.kv.ep.{suffix}"
                tp_name = f"model.layers.{g}.kv.tp.{suffix}"
                mgr._entries[ep_name] = MagicMock()
                mgr._entries[ep_name].numel = (
                    512 * head_num * head_dim
                )
                mgr._entries[tp_name] = MagicMock()
                mgr._entries[tp_name].numel = (
                    512 * (head_num // tp_size) * head_dim
                )

        def mock_get_view_as(name, shape):
            if name in ep_k_buffers:
                return ep_k_buffers[name]
            elif name in ep_v_buffers:
                return ep_v_buffers[name]
            elif name in tp_k_buffers:
                return tp_k_buffers[name]
            elif name in tp_v_buffers:
                return tp_v_buffers[name]
            raise KeyError(f"Unknown buffer: {name}")

        mgr.get_view_as = mock_get_view_as

        # Initial state: EP
        with patch(
            "sglang.srt.paras.paras_memory_manager.get_global_paras_memory_manager",
            return_value=mgr,
        ):
            pool.paras_configure_ep(layer_specs=None)

        # Capture EP buffer pointers
        ep_full_k_ptr = pool.full_kv_pool.k_buffer[0].data_ptr()
        ep_swa_k_ptr = pool.swa_kv_pool.k_buffer[0].data_ptr()

        # Switch to TP
        with patch(
            "sglang.srt.paras.paras_memory_manager.get_global_paras_memory_manager",
            return_value=mgr,
        ):
            pool.paras_configure_tp(tp_size, layer_specs=None)

        # Capture TP buffer pointers
        tp_full_k_ptr = pool.full_kv_pool.k_buffer[0].data_ptr()
        tp_swa_k_ptr = pool.swa_kv_pool.k_buffer[0].data_ptr()

        # Verify TP pointers differ from EP
        assert tp_full_k_ptr != ep_full_k_ptr, "TP should rebind to different buffer"
        assert tp_swa_k_ptr != ep_swa_k_ptr, "TP should rebind to different buffer"

        # Switch back to EP
        with patch(
            "sglang.srt.paras.paras_memory_manager.get_global_paras_memory_manager",
            return_value=mgr,
        ):
            pool.paras_configure_ep(layer_specs=None)

        # Capture final EP buffer pointers
        final_ep_full_k_ptr = pool.full_kv_pool.k_buffer[0].data_ptr()
        final_ep_swa_k_ptr = pool.swa_kv_pool.k_buffer[0].data_ptr()

        # Verify round-trip: final EP pointers match initial EP pointers
        assert final_ep_full_k_ptr == ep_full_k_ptr, (
            "After EP→TP→EP, full layer k_buffer should match initial EP pointer"
        )
        assert final_ep_swa_k_ptr == ep_swa_k_ptr, (
            "After EP→TP→EP, SWA layer k_buffer should match initial EP pointer"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
