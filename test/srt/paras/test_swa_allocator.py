#!/usr/bin/env python3
"""
Unit tests for SWATokenToKVPoolAllocator.paras_resize_and_clear.

Tests cover:
  1. Resize state: _size_full, _size_swa, mapping size, mapping zeroed
  2. Resize + alloc bijectivity: mapping[full_indices] = swa_indices (unique, non-zero)
  3. Shared tensor: SWAKVPool.full_to_swa_index_mapping shares data_ptr with allocator
  4. Double-resize idempotence: calling paras_resize_and_clear twice yields same state
  5. Inner allocator resize: full_attn_allocator.available_size() == new_full_size after resize
  6. Capacity check: alloc(None) returns None when need_size > available

Usage:
  conda run -n sgl_paras python -m pytest test/srt/paras/test_swa_allocator.py -x -v
"""

import os
import sys

import pytest
import torch

# Add sglang to path
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.join(_TEST_DIR, "..", "..", "..")
sys.path.insert(0, os.path.join(_ROOT_DIR, "python"))


# =========================================================================
# TEST GROUP 1: Resize state — _size_full, _size_swa, mapping size, mapping zeroed
# =========================================================================


class TestResizeState:
    """Verify paras_resize_and_clear sets correct state."""

    def test_resize_state_full_and_swa_sizes(self):
        """After paras_resize_and_clear(200, 80), _size_full==200, _size_swa==80."""
        from sglang.srt.mem_cache.allocator import SWATokenToKVPoolAllocator
        from sglang.srt.mem_cache.memory_pool import SWAKVPool

        # Create SWAKVPool
        kvcache = SWAKVPool(
            size=100,
            size_swa=50,
            dtype=torch.float16,
            head_num=8,
            head_dim=128,
            swa_attention_layer_ids=[1, 3],
            full_attention_layer_ids=[0, 2],
            enable_kvcache_transpose=False,
            device="cpu",
        )

        # Create allocator
        allocator = SWATokenToKVPoolAllocator(
            size=100,
            size_swa=50,
            dtype=torch.int64,
            device="cpu",
            kvcache=kvcache,
            need_sort=False,
        )

        # Resize
        allocator.paras_resize_and_clear(200, 80)

        # Verify sizes
        assert allocator._size_full == 200
        assert allocator._size_swa == 80

    def test_resize_mapping_size_and_zeroed(self):
        """After paras_resize_and_clear(200, 80), mapping has 281 elements, all zeros."""
        from sglang.srt.mem_cache.allocator import SWATokenToKVPoolAllocator
        from sglang.srt.mem_cache.memory_pool import SWAKVPool

        kvcache = SWAKVPool(
            size=100,
            size_swa=50,
            dtype=torch.float16,
            head_num=8,
            head_dim=128,
            swa_attention_layer_ids=[1, 3],
            full_attention_layer_ids=[0, 2],
            enable_kvcache_transpose=False,
            device="cpu",
        )

        allocator = SWATokenToKVPoolAllocator(
            size=100,
            size_swa=50,
            dtype=torch.int64,
            device="cpu",
            kvcache=kvcache,
            need_sort=False,
        )

        # Resize
        allocator.paras_resize_and_clear(200, 80)

        # Verify mapping size: new_full + new_swa + 1 = 200 + 80 + 1 = 281
        assert allocator.full_to_swa_index_mapping.numel() == 281

        # Verify all zeros
        assert torch.all(allocator.full_to_swa_index_mapping == 0)


# =========================================================================
# TEST GROUP 2: Resize + alloc bijectivity
# =========================================================================


class TestResizeAllocBijectivity:
    """Verify alloc after resize creates bijective mapping."""

    def test_resize_alloc_bijectivity(self):
        """After resize(200, 80) + alloc(50), mapping[full_indices] are 50 unique non-zero SWA indices."""
        from sglang.srt.mem_cache.allocator import SWATokenToKVPoolAllocator
        from sglang.srt.mem_cache.memory_pool import SWAKVPool

        kvcache = SWAKVPool(
            size=100,
            size_swa=50,
            dtype=torch.float16,
            head_num=8,
            head_dim=128,
            swa_attention_layer_ids=[1, 3],
            full_attention_layer_ids=[0, 2],
            enable_kvcache_transpose=False,
            device="cpu",
        )

        allocator = SWATokenToKVPoolAllocator(
            size=100,
            size_swa=50,
            dtype=torch.int64,
            device="cpu",
            kvcache=kvcache,
            need_sort=False,
        )

        # Resize to larger capacity
        allocator.paras_resize_and_clear(200, 80)

        # Allocate 50 tokens
        alloc_full_indices = allocator.alloc(50)

        # Verify alloc succeeded
        assert alloc_full_indices is not None
        assert alloc_full_indices.numel() == 50

        # Get mapped SWA indices
        swa_indices = allocator.full_to_swa_index_mapping[alloc_full_indices]

        # Verify all non-zero
        assert torch.all(swa_indices > 0)

        # Verify all unique (bijectivity)
        assert len(torch.unique(swa_indices)) == 50


# =========================================================================
# TEST GROUP 3: Shared tensor with SWAKVPool
# =========================================================================


class TestSharedTensorWithKVPool:
    """Verify SWAKVPool.full_to_swa_index_mapping shares data_ptr with allocator."""

    def test_shared_tensor_data_ptr(self):
        """SWAKVPool.full_to_swa_index_mapping.data_ptr() == allocator.full_to_swa_index_mapping.data_ptr()."""
        from sglang.srt.mem_cache.allocator import SWATokenToKVPoolAllocator
        from sglang.srt.mem_cache.memory_pool import SWAKVPool

        kvcache = SWAKVPool(
            size=100,
            size_swa=50,
            dtype=torch.float16,
            head_num=8,
            head_dim=128,
            swa_attention_layer_ids=[1, 3],
            full_attention_layer_ids=[0, 2],
            enable_kvcache_transpose=False,
            device="cpu",
        )

        allocator = SWATokenToKVPoolAllocator(
            size=100,
            size_swa=50,
            dtype=torch.int64,
            device="cpu",
            kvcache=kvcache,
            need_sort=False,
        )

        # Resize
        allocator.paras_resize_and_clear(200, 80)

        # Verify shared tensor (same data_ptr)
        assert (
            kvcache.full_to_swa_index_mapping.data_ptr()
            == allocator.full_to_swa_index_mapping.data_ptr()
        )


# =========================================================================
# TEST GROUP 4: Double-resize idempotence
# =========================================================================


class TestDoubleResizeIdempotence:
    """Verify calling paras_resize_and_clear twice yields same state."""

    def test_double_resize_idempotent(self):
        """Calling paras_resize_and_clear(200, 80) twice yields identical state."""
        from sglang.srt.mem_cache.allocator import SWATokenToKVPoolAllocator
        from sglang.srt.mem_cache.memory_pool import SWAKVPool

        kvcache = SWAKVPool(
            size=100,
            size_swa=50,
            dtype=torch.float16,
            head_num=8,
            head_dim=128,
            swa_attention_layer_ids=[1, 3],
            full_attention_layer_ids=[0, 2],
            enable_kvcache_transpose=False,
            device="cpu",
        )

        allocator = SWATokenToKVPoolAllocator(
            size=100,
            size_swa=50,
            dtype=torch.int64,
            device="cpu",
            kvcache=kvcache,
            need_sort=False,
        )

        # First resize
        allocator.paras_resize_and_clear(200, 80)
        first_mapping = allocator.full_to_swa_index_mapping.clone()
        first_size_full = allocator._size_full
        first_size_swa = allocator._size_swa
        first_data_ptr = allocator.full_to_swa_index_mapping.data_ptr()

        # Second resize (same parameters)
        allocator.paras_resize_and_clear(200, 80)
        second_mapping = allocator.full_to_swa_index_mapping.clone()
        second_size_full = allocator._size_full
        second_size_swa = allocator._size_swa
        second_data_ptr = allocator.full_to_swa_index_mapping.data_ptr()

        # Verify identical state
        assert first_size_full == second_size_full == 200
        assert first_size_swa == second_size_swa == 80
        assert torch.all(first_mapping == second_mapping)


# =========================================================================
# TEST GROUP 5: Inner allocator resize
# =========================================================================


class TestInnerAllocatorResize:
    """Verify full_attn_allocator.available_size() == new_full_size after resize."""

    def test_inner_allocator_available_size(self):
        """After resize(200, 80), full_attn_allocator.available_size() == 200."""
        from sglang.srt.mem_cache.allocator import SWATokenToKVPoolAllocator
        from sglang.srt.mem_cache.memory_pool import SWAKVPool

        kvcache = SWAKVPool(
            size=100,
            size_swa=50,
            dtype=torch.float16,
            head_num=8,
            head_dim=128,
            swa_attention_layer_ids=[1, 3],
            full_attention_layer_ids=[0, 2],
            enable_kvcache_transpose=False,
            device="cpu",
        )

        allocator = SWATokenToKVPoolAllocator(
            size=100,
            size_swa=50,
            dtype=torch.int64,
            device="cpu",
            kvcache=kvcache,
            need_sort=False,
        )

        # Resize
        allocator.paras_resize_and_clear(200, 80)

        # Verify inner allocator resized
        assert allocator.full_attn_allocator.available_size() == 200
        assert allocator.swa_attn_allocator.available_size() == 80


# =========================================================================
# TEST GROUP 6: Capacity check — alloc returns None when over capacity
# =========================================================================


class TestCapacityCheck:
    """Verify alloc(need_size) returns None when need_size > available."""

    def test_alloc_returns_none_over_capacity(self):
        """alloc(None) returns None when need_size > available_size."""
        from sglang.srt.mem_cache.allocator import SWATokenToKVPoolAllocator
        from sglang.srt.mem_cache.memory_pool import SWAKVPool

        kvcache = SWAKVPool(
            size=100,
            size_swa=50,
            dtype=torch.float16,
            head_num=8,
            head_dim=128,
            swa_attention_layer_ids=[1, 3],
            full_attention_layer_ids=[0, 2],
            enable_kvcache_transpose=False,
            device="cpu",
        )

        allocator = SWATokenToKVPoolAllocator(
            size=100,
            size_swa=50,
            dtype=torch.int64,
            device="cpu",
            kvcache=kvcache,
            need_sort=False,
        )

        # Resize to small capacity
        allocator.paras_resize_and_clear(10, 5)

        # Try to allocate more than available
        result = allocator.alloc(20)  # 20 > 10 (full capacity)

        # Should return None
        assert result is None

    def test_alloc_returns_none_swa_over_capacity(self):
        """alloc(need_size) returns None when need_size > swa_available_size."""
        from sglang.srt.mem_cache.allocator import SWATokenToKVPoolAllocator
        from sglang.srt.mem_cache.memory_pool import SWAKVPool

        kvcache = SWAKVPool(
            size=100,
            size_swa=50,
            dtype=torch.float16,
            head_num=8,
            head_dim=128,
            swa_attention_layer_ids=[1, 3],
            full_attention_layer_ids=[0, 2],
            enable_kvcache_transpose=False,
            device="cpu",
        )

        allocator = SWATokenToKVPoolAllocator(
            size=100,
            size_swa=50,
            dtype=torch.int64,
            device="cpu",
            kvcache=kvcache,
            need_sort=False,
        )

        # Resize with small SWA capacity
        allocator.paras_resize_and_clear(100, 5)

        # Try to allocate more than SWA available
        result = allocator.alloc(10)  # 10 > 5 (SWA capacity)

        # Should return None
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
