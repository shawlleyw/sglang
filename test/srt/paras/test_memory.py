#!/usr/bin/env python3
"""
Memory invariant tests for ParaS TP↔EP switching.

Tests that memory-related state is preserved correctly:
  1. head_num save/restore across TP↔EP cycles
  2. No GPU memory leak after repeated TP↔EP cycles

Usage:
  torchrun --nproc_per_node=4 -m pytest test/srt/paras/test_memory.py -v
"""

import os
import sys

import pytest
import torch

# Add sglang to path
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.join(_TEST_DIR, "..", "..", "..")
sys.path.insert(0, os.path.join(_ROOT_DIR, "python"))


# ---- test constants (Qwen3-30B-A3B) ----
NUM_LAYERS = 3
NUM_KV_HEADS = 4
HEAD_DIM = 128
KV_DTYPE = torch.bfloat16


def _is_distributed():
    """Check if we're running under torchrun."""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


# =========================================================================
# TEST GROUP 1: head_num restoration (4 GPU)
# =========================================================================


@pytest.mark.skipif(not _is_distributed(), reason="Requires torchrun with 4 GPUs")
class TestHeadNumRestoration:
    """head_num save/restore across TP↔EP cycles."""

    def test_head_num_restored_after_ep(self):
        """MHATokenToKVPool head_num should restore after TP→EP round-trip."""
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        original_head_num = NUM_KV_HEADS
        pool = MHATokenToKVPool(
            size=100,
            page_size=1,
            dtype=KV_DTYPE,
            head_num=original_head_num,
            head_dim=HEAD_DIM,
            layer_num=NUM_LAYERS,
            device=f"cuda:{rank}",
            enable_memory_saver=False,
        )

        assert pool.head_num == original_head_num

        # Switch to TP: head_num should be sharded
        pool.paras_configure_tp(tp_size=world_size, paras_tp_rank=rank)
        sharded = original_head_num // world_size
        assert pool.head_num == sharded, (
            f"After TP: expected head_num={sharded}, got {pool.head_num}"
        )

        # Switch back to EP: head_num should restore
        pool.paras_configure_ep()
        assert pool.head_num == original_head_num, (
            f"After EP restore: expected head_num={original_head_num}, "
            f"got {pool.head_num}"
        )


# =========================================================================
# TEST GROUP 2: Memory leak check (4 GPU)
# =========================================================================


@pytest.mark.skipif(not _is_distributed(), reason="Requires torchrun with 4 GPUs")
class TestMemoryLeak:
    """Verify no GPU memory leak from TP↔EP switching."""

    def test_no_memory_leak(self):
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        pool = MHATokenToKVPool(
            size=200,
            page_size=1,
            dtype=KV_DTYPE,
            head_num=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            layer_num=NUM_LAYERS,
            device=f"cuda:{rank}",
            enable_memory_saver=False,
        )

        # Warm up CUDA memory
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated()

        # Cycle TP↔EP multiple times
        for _ in range(5):
            pool.paras_configure_tp(tp_size=world_size, paras_tp_rank=rank)
            pool.paras_configure_ep()

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        mem_after = torch.cuda.memory_allocated()

        # Allow <1% growth (accounting for CUDA internal state)
        if mem_before > 0:
            delta_pct = abs(mem_after - mem_before) / mem_before
            assert delta_pct < 0.01, (
                f"Memory leak: before={mem_before}, after={mem_after}, "
                f"delta={delta_pct:.4%}"
            )
        else:
            # If before was 0 (unlikely), just check after is small
            assert mem_after - mem_before < 1024 * 1024, (
                f"Memory leak: grew by {mem_after - mem_before} bytes from 0"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
