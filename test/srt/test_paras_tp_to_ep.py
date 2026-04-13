#!/usr/bin/env python3
"""
Unit tests for TP→EP switch components:
  1. Request partition algorithm (no GPU needed)
  2. KV scatter NCCL round-trip (4-GPU torchrun)
  3. Weight restoration pointer swap (4-GPU torchrun)
  4. Memory leak check (4-GPU torchrun)

Usage:
  # Partition tests only (no GPU):
  python test_paras_tp_to_ep.py --partition-only

  # All tests (4 GPU):
  torchrun --nproc_per_node=4 -m pytest test_paras_tp_to_ep.py -v
"""

import os
import sys

import pytest
import torch

# Add sglang to path
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.join(_TEST_DIR, "..", "..")
sys.path.insert(0, os.path.join(_ROOT_DIR, "python"))


# ---- test constants (Qwen3-30B-A3B) ----
NUM_LAYERS = 3
NUM_KV_HEADS = 4
HEAD_DIM = 128
KV_DTYPE = torch.bfloat16
PAGE_SIZE = 1
SEED = 42

# MoE constants
NUM_EXPERTS = 64
HIDDEN = 2048
INTERMEDIATE = 1536


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
    """Tests for partition_requests_for_ep — pure CPU, no GPU required."""

    @staticmethod
    def _partition(reqs, num_ranks):
        from sglang.srt.paras.gather_manager import partition_requests_for_ep
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
        import random
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
# Distributed test helpers (shared by GPU tests)
# =========================================================================

def _is_distributed():
    """Check if we're running under torchrun."""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


class _SimpleGroupCoordinator:
    """Minimal GroupCoordinator stand-in for testing."""

    def __init__(self, device_group, world_size, device, rank_in_group=0):
        self.device_group = device_group
        self.world_size = world_size
        self.device = torch.device(device)
        self.rank_in_group = rank_in_group
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = self.rank


def _setup_distributed():
    import torch.distributed as dist
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 4, f"This test requires exactly 4 GPUs, got {world_size}"
    torch.cuda.set_device(rank)
    return rank, world_size


def _teardown_distributed():
    import torch.distributed as dist
    dist.destroy_process_group()


def _setup_paras_state(rank, world_size):
    """Set ParaS parallel state globals without full sglang server init."""
    import torch.distributed as dist
    import sglang.srt.distributed.parallel_state as ps
    import sglang.srt.paras.paras_parallel_state as pps

    tp_group = dist.new_group(ranks=list(range(world_size)))
    tp_coord = _SimpleGroupCoordinator(tp_group, world_size, f"cuda:{rank}", rank_in_group=rank)

    ps._TP = tp_coord

    pps._PARAS_TP = tp_coord
    pps._PARAS_DP = _SimpleGroupCoordinator(None, 1, f"cuda:{rank}", rank_in_group=0)
    pps._PARAS_SELF = _SimpleGroupCoordinator(None, 1, f"cuda:{rank}", rank_in_group=0)

    pps._PARAS_TP_SIZE = world_size
    pps._PARAS_TP_RANK = rank
    pps._PARAS_DP_SIZE = 1
    pps._PARAS_DP_RANK = 0
    pps._PARAS_EP_SIZE = world_size
    pps._PARAS_EP_RANK = rank

    return tp_group


# =========================================================================
# KV cache helpers
# =========================================================================

TOKENS_PER_RANK = [100, 80, 90, 70]

def _build_kv_manager(rank, world_size):
    """Create ParaSMemoryManager with N+1 KV slots."""
    from sglang.srt.paras.paras_memory_manager import (
        ParaSMemoryManager,
        create_paras_kv_aliases,
        set_global_paras_memory_manager,
    )

    ep_max_tokens = max(TOKENS_PER_RANK) + 100
    heads_per_peer = max(1, NUM_KV_HEADS // world_size)
    total_tokens = sum(TOKENS_PER_RANK)
    min_ep_for_tp = (total_tokens * heads_per_peer + NUM_KV_HEADS - 1) // NUM_KV_HEADS
    ep_max_tokens = max(ep_max_tokens, min_ep_for_tp)
    tp_max_tokens = (ep_max_tokens + PAGE_SIZE) * NUM_KV_HEADS // heads_per_peer

    mgr = ParaSMemoryManager(device=f"cuda:{rank}")
    mgr.reserve_kv_cache(
        num_layers=NUM_LAYERS,
        ep_max_tokens=ep_max_tokens,
        tp_max_tokens=tp_max_tokens,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        kv_dtype=KV_DTYPE,
        page_size=PAGE_SIZE,
    )
    mgr.materialize()
    create_paras_kv_aliases(mgr, NUM_LAYERS)
    set_global_paras_memory_manager(mgr)
    return mgr, ep_max_tokens, tp_max_tokens


def _fill_kv_data(mgr, rank):
    """Fill EP KV buffers with rank-deterministic random data."""
    num_tokens = TOKENS_PER_RANK[rank]
    for layer_id in range(NUM_LAYERS):
        ep_k = mgr.get_view(f"model.layers.{layer_id}.kv.ep.k")
        ep_v = mgr.get_view(f"model.layers.{layer_id}.kv.ep.v")
        ep_k.zero_()
        ep_v.zero_()

        gen_k = torch.Generator(device="cpu")
        gen_k.manual_seed(SEED + layer_id * 1000 + rank)
        data_k = torch.randn(
            (num_tokens, NUM_KV_HEADS, HEAD_DIM),
            generator=gen_k, dtype=torch.float32,
        ).to(dtype=KV_DTYPE, device=ep_k.device)
        ep_k[:num_tokens].copy_(data_k)

        gen_v = torch.Generator(device="cpu")
        gen_v.manual_seed(SEED + layer_id * 1000 + rank + 500)
        data_v = torch.randn(
            (num_tokens, NUM_KV_HEADS, HEAD_DIM),
            generator=gen_v, dtype=torch.float32,
        ).to(dtype=KV_DTYPE, device=ep_v.device)
        ep_v[:num_tokens].copy_(data_v)


# =========================================================================
# TEST GROUP 2: KV Scatter NCCL round-trip (4 GPU)
# =========================================================================

@pytest.mark.skipif(not _is_distributed(), reason="Requires torchrun with 4 GPUs")
class TestKVScatterNCCLRoundtrip:
    """EP→TP gather then TP→EP scatter should produce original data."""

    def test_kv_scatter_nccl_roundtrip(self):
        import torch.distributed as dist
        from sglang.srt.paras.gather_manager import (
            gather_kv_and_permute,
            permute_and_scatter_kv,
            gather_tp_kv_and_permute,
            permute_and_scatter_kv_to_ep,
        )

        rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        tp_group = _setup_paras_state(rank, world_size)
        mgr, ep_max_tokens, tp_max_tokens = _build_kv_manager(rank, world_size)

        _fill_kv_data(mgr, rank)

        heads_per_peer = max(1, NUM_KV_HEADS // world_size)
        total_tokens = sum(TOKENS_PER_RANK)
        tp_view_tokens = (ep_max_tokens + PAGE_SIZE) * NUM_KV_HEADS // heads_per_peer
        num_local = TOKENS_PER_RANK[rank]
        splited_size = heads_per_peer * HEAD_DIM

        # --- Snapshot original EP data ---
        orig_ep = {}
        for lid in range(NUM_LAYERS):
            ep_k = mgr.get_view(f"model.layers.{lid}.kv.ep.k")
            ep_v = mgr.get_view(f"model.layers.{lid}.kv.ep.v")
            orig_ep[lid] = (ep_k[:num_local].clone(), ep_v[:num_local].clone())

        # --- Step 1: EP→TP gather (NCCL all_to_all) ---
        local_token_indices = torch.arange(num_local, dtype=torch.int64, device="cuda")
        global_token_indices = torch.arange(total_tokens, dtype=torch.int64, device="cuda")

        input_split_sizes = [2 * splited_size * num_local] * world_size
        output_split_sizes = [2 * splited_size * TOKENS_PER_RANK[r] for r in range(world_size)]

        for lid in range(NUM_LAYERS):
            ep_k = mgr.get_view(f"model.layers.{lid}.kv.ep.k")
            ep_v = mgr.get_view(f"model.layers.{lid}.kv.ep.v")

            permuted = gather_kv_and_permute(ep_k, ep_v, local_token_indices)

            tp_k = mgr.get_view_as(
                f"model.layers.{lid}.kv.tp.k",
                (tp_view_tokens, heads_per_peer, HEAD_DIM),
            )
            tp_v = mgr.get_view_as(
                f"model.layers.{lid}.kv.tp.v",
                (tp_view_tokens, heads_per_peer, HEAD_DIM),
            )

            gathered = torch.empty(
                2 * total_tokens * splited_size,
                dtype=KV_DTYPE, device="cuda",
            )
            dist.all_to_all_single(
                gathered, permuted, output_split_sizes, input_split_sizes,
                group=tp_group,
            )
            permute_and_scatter_kv(
                gathered, tp_k, tp_v, global_token_indices,
                total_tokens, heads_per_peer, HEAD_DIM,
            )

        torch.cuda.synchronize()
        dist.barrier(group=tp_group)

        # --- Step 2: TP→EP scatter (NCCL all_to_all) ---
        # Build token partition: each rank gets its original tokens
        token_partition = []
        offset = 0
        for r in range(world_size):
            token_partition.append(list(range(offset, offset + TOKENS_PER_RANK[r])))
            offset += TOKENS_PER_RANK[r]

        my_token_count = TOKENS_PER_RANK[rank]
        ep_dst_positions = torch.arange(my_token_count, dtype=torch.int64, device="cuda")

        send_token_counts = [len(token_partition[e]) for e in range(world_size)]
        per_token_elems = heads_per_peer * 2 * HEAD_DIM
        scatter_input_split = [cnt * per_token_elems for cnt in send_token_counts]
        scatter_output_split = [my_token_count * per_token_elems] * world_size

        sorted_parts = []
        for e in range(world_size):
            if send_token_counts[e] > 0:
                part_idx = torch.tensor(token_partition[e], dtype=torch.long, device="cuda")
                sorted_parts.append(global_token_indices[part_idx])
        sorted_tp_indices = torch.cat(sorted_parts) if sorted_parts else torch.empty(0, dtype=torch.long, device="cuda")

        for lid in range(NUM_LAYERS):
            tp_k = mgr.get_view_as(
                f"model.layers.{lid}.kv.tp.k",
                (tp_view_tokens, heads_per_peer, HEAD_DIM),
            )
            tp_v = mgr.get_view_as(
                f"model.layers.{lid}.kv.tp.v",
                (tp_view_tokens, heads_per_peer, HEAD_DIM),
            )

            send_buf = gather_tp_kv_and_permute(
                tp_k, tp_v, sorted_tp_indices,
                NUM_KV_HEADS, heads_per_peer, HEAD_DIM, world_size,
            )

            recv_buf = torch.empty(
                my_token_count * world_size * per_token_elems,
                dtype=KV_DTYPE, device="cuda",
            )
            dist.all_to_all_single(
                recv_buf, send_buf,
                scatter_output_split, scatter_input_split,
                group=tp_group,
            )

            # Write back to EP buffers
            ep_k = mgr.get_view(f"model.layers.{lid}.kv.ep.k")
            ep_v = mgr.get_view(f"model.layers.{lid}.kv.ep.v")
            ep_k.zero_()
            ep_v.zero_()
            permute_and_scatter_kv_to_ep(
                recv_buf, ep_k, ep_v, ep_dst_positions,
                my_token_count, NUM_KV_HEADS, heads_per_peer,
                HEAD_DIM, world_size,
            )

        torch.cuda.synchronize()

        # --- Verify round-trip ---
        all_ok = True
        for lid in range(NUM_LAYERS):
            ep_k = mgr.get_view(f"model.layers.{lid}.kv.ep.k")
            ep_v = mgr.get_view(f"model.layers.{lid}.kv.ep.v")
            k_match = torch.equal(orig_ep[lid][0], ep_k[:num_local])
            v_match = torch.equal(orig_ep[lid][1], ep_v[:num_local])
            if not k_match or not v_match:
                all_ok = False

        # Save evidence
        evidence_dir = os.path.join(_ROOT_DIR, ".sisyphus", "evidence")
        os.makedirs(evidence_dir, exist_ok=True)
        if rank == 0:
            with open(os.path.join(evidence_dir, "task-8-correctness-test.txt"), "w") as f:
                f.write(f"KV scatter NCCL round-trip: {'PASS' if all_ok else 'FAIL'}\n")
                f.write(f"Layers: {NUM_LAYERS}, Heads: {NUM_KV_HEADS}, GPUs: {world_size}\n")
                f.write(f"Tokens per rank: {TOKENS_PER_RANK}\n")

        assert all_ok, "KV scatter NCCL round-trip failed: data mismatch"


# =========================================================================
# TEST GROUP 3: Weight restoration pointer swap (4 GPU)
# =========================================================================

@pytest.mark.skipif(not _is_distributed(), reason="Requires torchrun with 4 GPUs")
class TestWeightRestoration:
    """Tests for MoE pointer swap and KV head_num restoration."""

    def test_moe_pointer_swap(self):
        """Create mock MoE, verify experts toggle between ep/tp."""
        from sglang.srt.paras.layers.paras_moe_block import ParaSMoeBlockMixin

        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        _setup_paras_state(rank, world_size)

        # Build minimal mixin
        m = object.__new__(ParaSMoeBlockMixin)
        m._paras_layer_id = 0
        m.num_local_experts = NUM_EXPERTS // world_size
        m.num_global_experts = NUM_EXPERTS
        m.hidden_size = HIDDEN
        m.moe_intermediate_size = INTERMEDIATE

        # Create mock expert objects with distinct identities
        class _MockExperts:
            def __init__(self, tag):
                self.tag = tag
        ep_exp = _MockExperts("ep")
        tp_exp = _MockExperts("tp")

        m.ep_experts = ep_exp
        m.tp_experts = tp_exp
        m.experts = ep_exp
        m.parallelism_config = "ep"
        m.tp_size = 1

        # Switch to TP
        m.paras_configure_tp(world_size, rank)
        assert m.experts is tp_exp, "After configure_tp, experts should be tp_experts"
        assert m.parallelism_config == "tp"

        # Switch back to EP
        m.paras_configure_ep()
        assert m.experts is ep_exp, "After configure_ep, experts should be ep_experts"
        assert m.parallelism_config == "ep"

    def test_head_num_restoration(self):
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
            f"After EP restore: expected head_num={original_head_num}, got {pool.head_num}"
        )


# =========================================================================
# TEST GROUP 4: Memory leak check (4 GPU)
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


# =========================================================================
# Main entry point
# =========================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="ParaS TP→EP switch tests")
    parser.add_argument(
        "--partition-only", action="store_true",
        help="Run only partition tests (no GPU needed)"
    )
    args = parser.parse_args()

    if args.partition_only:
        # Run partition tests without pytest (standalone)
        print("Running partition tests (no GPU)...\n")
        t = TestPartitionRequestsForEP()

        tests = [
            ("test_partition_balanced", t.test_partition_balanced),
            ("test_partition_fewer_than_ranks", t.test_partition_fewer_than_ranks),
            ("test_partition_zero", t.test_partition_zero),
            ("test_partition_equal_seqlens_deterministic", t.test_partition_equal_seqlens_deterministic),
            ("test_partition_imbalanced", t.test_partition_imbalanced),
        ]

        passed = 0
        for name, fn in tests:
            try:
                fn()
                print(f"  [PASS] {name}")
                passed += 1
            except Exception as e:
                print(f"  [FAIL] {name}: {e}")

        print(f"\n{passed}/{len(tests)} partition tests passed")

        # Save evidence
        evidence_dir = os.path.join(_ROOT_DIR, ".sisyphus", "evidence")
        os.makedirs(evidence_dir, exist_ok=True)
        with open(os.path.join(evidence_dir, "task-8-correctness-test.txt"), "w") as f:
            f.write(f"Partition tests: {passed}/{len(tests)} passed\n")
            for name, fn in tests:
                try:
                    fn()
                    f.write(f"  [PASS] {name}\n")
                except Exception as e:
                    f.write(f"  [FAIL] {name}: {e}\n")

        sys.exit(0 if passed == len(tests) else 1)

    # Full test suite under torchrun
    if _is_distributed():
        rank, world_size = _setup_distributed()
        try:
            # Run GPU tests manually (pytest markers handled by torchrun)
            results = []

            # KV scatter round-trip
            print(f"[Rank {rank}] Running KV scatter NCCL round-trip...", flush=True)
            try:
                TestKVScatterNCCLRoundtrip().test_kv_scatter_nccl_roundtrip()
                results.append(("kv_scatter_nccl_roundtrip", True))
                if rank == 0:
                    print("  [PASS] kv_scatter_nccl_roundtrip", flush=True)
            except Exception as e:
                results.append(("kv_scatter_nccl_roundtrip", False))
                if rank == 0:
                    print(f"  [FAIL] kv_scatter_nccl_roundtrip: {e}", flush=True)

            # Weight restoration
            print(f"[Rank {rank}] Running weight restoration tests...", flush=True)
            try:
                TestWeightRestoration().test_moe_pointer_swap()
                results.append(("moe_pointer_swap", True))
                if rank == 0:
                    print("  [PASS] moe_pointer_swap", flush=True)
            except Exception as e:
                results.append(("moe_pointer_swap", False))
                if rank == 0:
                    print(f"  [FAIL] moe_pointer_swap: {e}", flush=True)

            try:
                TestWeightRestoration().test_head_num_restoration()
                results.append(("head_num_restoration", True))
                if rank == 0:
                    print("  [PASS] head_num_restoration", flush=True)
            except Exception as e:
                results.append(("head_num_restoration", False))
                if rank == 0:
                    print(f"  [FAIL] head_num_restoration: {e}", flush=True)

            # Memory leak
            print(f"[Rank {rank}] Running memory leak test...", flush=True)
            try:
                TestMemoryLeak().test_no_memory_leak()
                results.append(("no_memory_leak", True))
                if rank == 0:
                    print("  [PASS] no_memory_leak", flush=True)
            except Exception as e:
                results.append(("no_memory_leak", False))
                if rank == 0:
                    print(f"  [FAIL] no_memory_leak: {e}", flush=True)

            # Summary
            import torch.distributed as dist
            dist.barrier()
            passed = sum(1 for _, ok in results if ok)
            total = len(results)
            if rank == 0:
                print(f"\n{'=' * 60}")
                print(f"RESULTS: {passed}/{total} GPU tests passed")
                print(f"{'=' * 60}")

            _teardown_distributed()
            sys.exit(0 if passed == total else 1)

        except Exception as e:
            print(f"[Rank {rank}] ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()
            try:
                _teardown_distributed()
            except Exception:
                pass
            sys.exit(1)
    else:
        # No torchrun, just run partition tests
        print("No distributed env detected. Running partition tests only.")
        main_args = ["--partition-only"]
        sys.argv = [sys.argv[0]] + main_args
        main()


if __name__ == "__main__":
    main()
