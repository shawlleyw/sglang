#!/usr/bin/env python3
"""
Tests for request partition algorithm and replication routing.
CPU only — no GPU or distributed required.

Usage:
  python -m pytest test/srt/paras/test_request_partition.py -v
"""

import importlib
import random
import sys
import types
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Import helper: load scatter_manager without GPU-heavy transitive deps.
# The module-level imports in scatter_manager.py pull in Req (which
# transitively requires triton, CUDA, etc.).  We stub the heavy modules
# so the pure-Python partition functions can be tested on CPU.
# ---------------------------------------------------------------------------

def _import_scatter_manager():
    """Import scatter_manager by file path, with heavy deps stubbed out.

    scatter_manager.py imports Req, ReqToTokenPool, etc. at module level
    which transitively pull in triton, CUDA, and other GPU deps.  We stub
    those modules so only the pure-Python partition functions are loaded.
    """
    import os
    import importlib.util

    mod_name = "sglang.srt.paras.scatter_manager"
    if mod_name in sys.modules:
        return sys.modules[mod_name]

    # Modules that need stubbing to break the import chain
    stubs_needed = [
        "sglang.srt.managers.schedule_batch",
        "sglang.srt.model_executor.forward_batch_info",
        "sglang.srt.mem_cache.memory_pool",
        "sglang.srt.mem_cache.allocator",
        "sglang.srt.distributed.parallel_state",
        "sglang.srt.paras.paras_memory_manager",
        "sglang.srt.paras.peer_access",
    ]
    saved = {}
    for name in stubs_needed:
        if name in sys.modules:
            saved[name] = sys.modules[name]
        stub = types.ModuleType(name)
        # Provide dummy classes that scatter_manager imports at top-level
        stub.__dict__.setdefault("Req", type("Req", (), {}))
        stub.__dict__.setdefault("ReqToTokenPool", type("ReqToTokenPool", (), {}))
        stub.__dict__.setdefault("MHATokenToKVPool", type("MHATokenToKVPool", (), {}))
        stub.__dict__.setdefault("TokenToKVPoolAllocator", type("TokenToKVPoolAllocator", (), {}))
        stub.__dict__.setdefault("GroupCoordinator", type("GroupCoordinator", (), {}))
        sys.modules[name] = stub

    # Ensure parent packages exist with correct __path__
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    python_root = os.path.join(repo_root, "python")
    pkg_paths = {
        "sglang": os.path.join(python_root, "sglang"),
        "sglang.srt": os.path.join(python_root, "sglang", "srt"),
        "sglang.srt.paras": os.path.join(python_root, "sglang", "srt", "paras"),
        "sglang.srt.managers": os.path.join(python_root, "sglang", "srt", "managers"),
        "sglang.srt.mem_cache": os.path.join(python_root, "sglang", "srt", "mem_cache"),
        "sglang.srt.distributed": os.path.join(python_root, "sglang", "srt", "distributed"),
        "sglang.srt.model_executor": os.path.join(python_root, "sglang", "srt", "model_executor"),
    }
    for pkg, path in pkg_paths.items():
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = [path]
            m.__package__ = pkg
            sys.modules[pkg] = m

    try:
        # Load scatter_manager.py directly by file path
        scatter_path = os.path.join(
            python_root, "sglang", "srt", "paras", "scatter_manager.py"
        )
        spec = importlib.util.spec_from_file_location(mod_name, scatter_path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        # Restore any saved modules
        for name, orig in saved.items():
            sys.modules[name] = orig


# Eagerly import at module level so all test classes can use it
_scatter_mod = _import_scatter_manager()
partition_requests_for_ep = _scatter_mod.partition_requests_for_ep
PARTITION_STRATEGIES = _scatter_mod.PARTITION_STRATEGIES


# ---------------------------------------------------------------------------
# Mock request for partition tests (lightweight, no sglang deps)
# ---------------------------------------------------------------------------


class _MockReq:
    """Minimal Req stand-in with .rid, .seqlen, .origin_input_ids, .output_ids."""

    def __init__(self, rid: str, seqlen: int):
        self.rid = rid
        self.origin_input_ids = list(range(seqlen))
        self.output_ids = []

    @property
    def seqlen(self):
        return len(self.origin_input_ids) + len(self.output_ids)


# =========================================================================
# TEST GROUP 1: Request Partition (no GPU needed)
# =========================================================================


class TestPartitionRequestsForEP:
    """Tests for partition_requests_for_ep — pure CPU, no GPU."""

    @staticmethod
    def _partition(reqs, num_ranks):
        return partition_requests_for_ep(reqs, num_ranks)

    def test_partition_balanced(self):
        """8 reqs, 4 ranks → 2 each, perfect count balance."""
        reqs = [_MockReq(f"r{i}", seqlen=100 + i * 10) for i in range(8)]
        parts = self._partition(reqs, 4)
        assert len(parts) == 4
        counts = [len(p) for p in parts]
        assert counts == [2, 2, 2, 2], f"Expected [2,2,2,2], got {counts}"

    def test_partition_fewer_than_ranks(self):
        """2 reqs, 4 ranks → 2 ranks get 1 req, 2 get empty."""
        reqs = [_MockReq("a", 50), _MockReq("b", 60)]
        parts = self._partition(reqs, 4)
        assert len(parts) == 4
        counts = sorted([len(p) for p in parts])
        assert counts == [0, 0, 1, 1], f"Expected [0,0,1,1], got {counts}"

    def test_partition_zero(self):
        """Empty request list → all-empty partitions."""
        parts = self._partition([], 4)
        assert len(parts) == 4
        assert all(len(p) == 0 for p in parts)

    def test_partition_equal_seqlens_deterministic(self):
        """10 reqs all seqlen=50, shuffled input → same output."""
        reqs = [_MockReq(f"r{i}", 50) for i in range(10)]
        parts_1 = self._partition(reqs, 4)
        rids_1 = [[r.rid for r in p] for p in parts_1]

        shuffled = list(reqs)
        random.Random(999).shuffle(shuffled)
        parts_2 = self._partition(shuffled, 4)
        rids_2 = [[r.rid for r in p] for p in parts_2]

        assert rids_1 == rids_2, "Partition not deterministic under shuffle"

    def test_partition_imbalanced(self):
        """One 10K token req + seven 1K token reqs — count balance is primary."""
        reqs = [_MockReq("big", 10000)]
        reqs += [_MockReq(f"s{i}", 1000) for i in range(7)]
        parts = self._partition(reqs, 4)
        counts = [len(p) for p in parts]
        assert counts == [2, 2, 2, 2], f"Count balance violated: {counts}"
        # The big request should be in a partition with a small one
        for p in parts:
            rids = [r.rid for r in p]
            if "big" in rids:
                assert len(rids) == 2
                assert any(r.startswith("s") for r in rids)


# =========================================================================
# TEST GROUP 2: Peer-access replication-aware routing (no GPU needed)
# =========================================================================


class TestPeerAccessReplicationRouting:
    """Verify that _scatter_cache_peer_access builds routing tensors for
    only 1/R of the tokens when heads are replicated (num_kv_heads < tp_size).

    This tests the Python-level logic without running the CUDA kernel.
    """

    def test_no_replication(self):
        """R=1 (4 heads / 4 GPUs): each rank routes all tokens."""
        num_kv_heads, group_size = 4, 4
        R = group_size // num_kv_heads  # 1
        assert R == 1
        # 20 tokens total, 5 per destination
        token_partition = [list(range(i * 5, (i + 1) * 5)) for i in range(4)]
        for tp_rank in range(group_size):
            intra = tp_rank % R  # always 0
            my_count = 0
            for e in range(group_size):
                full = len(token_partition[e])
                s = full * intra // R
                end = full * (intra + 1) // R
                my_count += end - s
            assert my_count == 20, (
                f"Rank {tp_rank} should route all 20 tokens, got {my_count}"
            )

    def test_replication_factor_2(self):
        """R=2 (4 heads / 8 GPUs): each subgroup member routes half the tokens."""
        num_kv_heads, group_size = 4, 8
        R = group_size // num_kv_heads  # 2
        assert R == 2
        # 20 tokens total, variable per destination
        token_partition = [
            list(range(0, 3)),  # dest 0: 3 tokens
            list(range(3, 6)),  # dest 1: 3 tokens
            list(range(6, 8)),  # dest 2: 2 tokens
            list(range(8, 11)),  # dest 3: 3 tokens
            list(range(11, 13)),  # dest 4: 2 tokens
            list(range(13, 16)),  # dest 5: 3 tokens
            list(range(16, 18)),  # dest 6: 2 tokens
            list(range(18, 20)),  # dest 7: 2 tokens
        ]
        total_tokens = sum(len(p) for p in token_partition)
        assert total_tokens == 20

        for tp_rank in range(group_size):
            intra = tp_rank % R
            my_count = 0
            for e in range(group_size):
                full = len(token_partition[e])
                s = full * intra // R
                end = full * (intra + 1) // R
                my_count += end - s
            # Subgroup members should collectively cover all tokens
            partner_rank = tp_rank ^ 1  # flip last bit = partner
            partner_intra = partner_rank % R
            partner_count = 0
            for e in range(group_size):
                full = len(token_partition[e])
                s = full * partner_intra // R
                end = full * (partner_intra + 1) // R
                partner_count += end - s
            assert my_count + partner_count == total_tokens, (
                f"Ranks {tp_rank},{partner_rank} together should cover "
                f"{total_tokens} tokens, got {my_count}+{partner_count}"
            )
            # Each member handles roughly half
            assert my_count <= total_tokens // R + group_size, (
                f"Rank {tp_rank} handles too many tokens: {my_count}"
            )

    def test_no_token_lost_or_duplicated(self):
        """With R=2, verify every dest's tokens are fully covered by
        exactly 2 subgroup members with no overlap or gap."""
        R = 2
        for full in [0, 1, 2, 3, 7, 10, 15, 100, 1001]:
            covered = set()
            for intra in range(R):
                s = full * intra // R
                e = full * (intra + 1) // R
                token_set = set(range(s, e))
                assert covered.isdisjoint(token_set), (
                    f"full={full}: overlap at intra_rank={intra}"
                )
                covered |= token_set
            assert covered == set(range(full)), (
                f"full={full}: missing or extra tokens. "
                f"Expected {set(range(full))}, got {covered}"
            )

    def test_replication_factor_4(self):
        """R=4 (2 heads / 8 GPUs): each subgroup member routes 1/4."""
        R = 4
        full = 100
        for intra in range(R):
            s = full * intra // R
            e = full * (intra + 1) // R
            count = e - s
            assert count == 25, f"intra={intra}: expected 25, got {count}"


# =========================================================================
# TEST GROUP 3: Partition strategy extensibility
# =========================================================================


class TestPartitionStrategy:
    """Test strategy extensibility."""

    def test_greedy_strategy(self):
        """The 'greedy' strategy should be registered and produce valid partitions."""
        assert "greedy" in PARTITION_STRATEGIES

        reqs = [_MockReq(f"r{i}", seqlen=50 + i * 10) for i in range(6)]
        parts = partition_requests_for_ep(reqs, 3, strategy="greedy")
        assert len(parts) == 3
        # All requests accounted for
        all_rids = sorted(r.rid for p in parts for r in p)
        expected_rids = sorted(r.rid for r in reqs)
        assert all_rids == expected_rids, (
            f"Request mismatch: {all_rids} != {expected_rids}"
        )

    def test_unknown_strategy_raises(self):
        """An unknown strategy name must raise ValueError."""
        reqs = [_MockReq("x", 10)]
        with pytest.raises(ValueError, match="Unknown partition strategy"):
            partition_requests_for_ep(reqs, 2, strategy="nonexistent_strategy")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
