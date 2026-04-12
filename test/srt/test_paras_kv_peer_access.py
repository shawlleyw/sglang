#!/usr/bin/env python3
"""
4-GPU correctness and benchmark test for ParaS KV cache peer access kernel.

Tests that peer_access_kv_transfer produces bitwise-identical results to a
PyTorch all_gather reference for the EP→TP KV redistribution.

Usage:
  torchrun --nproc_per_node=4 test_paras_kv_peer_access.py            # correctness
  torchrun --nproc_per_node=4 test_paras_kv_peer_access.py --benchmark # + timing
"""

import argparse
import os
import subprocess
import sys
import time

import torch
import torch.distributed as dist

# Add sglang to path
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.join(_TEST_DIR, "..", "..")
sys.path.insert(0, os.path.join(_ROOT_DIR, "python"))

# ---- test constants (Qwen3-30B-A3B, 4 GPUs, NO replication) ----
NUM_LAYERS = 3          # enough to test N+1 ordering
NUM_KV_HEADS = 4        # total KV heads (4 heads / 4 GPUs = 1 head per rank)
HEAD_DIM = 128          # Qwen3-30B-A3B head dim
KV_DTYPE = torch.bfloat16
ELEM_SIZE = 2           # bf16

# Variable token counts per rank (make them different for realistic test)
TOKENS_PER_RANK = [100, 80, 90, 70]  # total: 340
EP_MAX_TOKENS = 200     # max tokens any rank could hold
PAGE_SIZE = 1

# TP view tokens: union layout means same bytes, so
# (EP_MAX + PAGE) * total_heads / heads_per_peer = TP tokens in view
# With 4 heads / 4 GPUs = 1 head per peer: (201) * 4 / 1 = 804
# We store tp_max_tokens for the reserve_kv_cache API but actual TP view
# size is derived from EP slot bytes.
_HEADS_PER_PEER = max(1, NUM_KV_HEADS // 4)  # 1 for 4-GPU case
TP_VIEW_TOKENS = (EP_MAX_TOKENS + PAGE_SIZE) * NUM_KV_HEADS // _HEADS_PER_PEER
# tp_max_tokens for reserve_kv_cache (just metadata, doesn't affect alloc)
TP_MAX_TOKENS = TP_VIEW_TOKENS
SEED = 42
BENCHMARK_WARMUP = 3
BENCHMARK_RUNS = 10


# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------

def setup_distributed():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 4, f"This test requires exactly 4 GPUs, got {world_size}"
    torch.cuda.set_device(rank)
    return rank, world_size


def teardown_distributed():
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# ParaS parallel state (lightweight init — no sglang server needed)
# ---------------------------------------------------------------------------

class _SimpleGroupCoordinator:
    """Minimal GroupCoordinator stand-in for testing."""

    def __init__(self, device_group, world_size, device, rank_in_group=0):
        self.device_group = device_group
        self.world_size = world_size
        self.device = torch.device(device)
        self.rank_in_group = rank_in_group
        self.rank = dist.get_rank()
        self.local_rank = dist.get_rank()


def setup_paras_state(rank, world_size):
    """Set ParaS parallel state globals without full sglang server init."""
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


# ---------------------------------------------------------------------------
# Memory manager helpers
# ---------------------------------------------------------------------------

def build_kv_manager(rank, world_size):
    """Create ParaSMemoryManager with N+1 KV slots."""
    from sglang.srt.paras.paras_memory_manager import (
        ParaSMemoryManager,
        create_paras_kv_aliases,
        set_global_paras_memory_manager,
    )

    mgr = ParaSMemoryManager(device=f"cuda:{rank}")
    mgr.reserve_kv_cache(
        num_layers=NUM_LAYERS,
        ep_max_tokens=EP_MAX_TOKENS,
        tp_max_tokens=TP_MAX_TOKENS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        kv_dtype=KV_DTYPE,
        page_size=PAGE_SIZE,
    )
    mgr.materialize()
    create_paras_kv_aliases(mgr, NUM_LAYERS)
    set_global_paras_memory_manager(mgr)
    return mgr


def fill_kv_data(mgr, rank):
    """Fill EP KV buffers with rank-deterministic random data."""
    num_tokens = TOKENS_PER_RANK[rank]
    for layer_id in range(NUM_LAYERS):
        ep_k = mgr.get_view(f"model.layers.{layer_id}.kv.ep.k")
        ep_v = mgr.get_view(f"model.layers.{layer_id}.kv.ep.v")

        # Zero out first, then fill only the tokens this rank owns
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


# ---------------------------------------------------------------------------
# Reference computation
# ---------------------------------------------------------------------------

def compute_reference(mgr, rank, world_size, tp_group):
    """Compute expected TP KV layout via all_gather of EP data.

    After peer access transfer, TP slot[i] for THIS rank contains:
    - tokens from ALL ranks concatenated: [rank0_tokens, rank1_tokens, ...]
    - each token has heads_per_peer heads (head shard for this rank)
    - head shard = ep_heads[rank * heads_per_peer : (rank+1) * heads_per_peer]
    """
    heads_per_peer = max(1, NUM_KV_HEADS // world_size)
    total_tokens = sum(TOKENS_PER_RANK)

    ref_k = {}
    ref_v = {}

    for layer_id in range(NUM_LAYERS):
        # Gather EP K/V from all ranks
        ep_k = mgr.get_view(f"model.layers.{layer_id}.kv.ep.k")  # (EP_MAX+1, H, D)
        ep_v = mgr.get_view(f"model.layers.{layer_id}.kv.ep.v")

        # Each rank sends its entire ep buffer for gathering
        k_list = [torch.zeros_like(ep_k) for _ in range(world_size)]
        v_list = [torch.zeros_like(ep_v) for _ in range(world_size)]
        dist.all_gather(k_list, ep_k, group=tp_group)
        dist.all_gather(v_list, ep_v, group=tp_group)

        # Build expected TP buffer for this rank
        # TP view: (TP_VIEW_TOKENS, heads_per_peer, HEAD_DIM)
        expected_k = torch.zeros(
            (TP_VIEW_TOKENS, heads_per_peer, HEAD_DIM),
            dtype=KV_DTYPE, device=ep_k.device,
        )
        expected_v = torch.zeros_like(expected_k)

        # Head shard for this rank in the EP buffer
        head_start = rank * heads_per_peer

        token_offset = 0
        for src_rank in range(world_size):
            n_tok = TOKENS_PER_RANK[src_rank]
            # From src_rank's EP buffer, extract this rank's head shard
            expected_k[token_offset:token_offset + n_tok] = (
                k_list[src_rank][:n_tok, head_start:head_start + heads_per_peer, :]
            )
            expected_v[token_offset:token_offset + n_tok] = (
                v_list[src_rank][:n_tok, head_start:head_start + heads_per_peer, :]
            )
            token_offset += n_tok

        ref_k[layer_id] = expected_k
        ref_v[layer_id] = expected_v

    return ref_k, ref_v


# ---------------------------------------------------------------------------
# Peer access setup (IPC for multi-process)
# ---------------------------------------------------------------------------

def setup_peer_ctx(mgr, rank, world_size, tp_group):
    """Enable peer access and exchange CUDA IPC handles."""
    from sglang.srt.paras.peer_access import init_peer_access

    return init_peer_access(mgr, tp_group, world_size)


# ---------------------------------------------------------------------------
# Kernel execution
# ---------------------------------------------------------------------------

def run_peer_access_transfer(mgr, peer_ctx, rank, world_size, tp_group):
    """Run the peer_access_kv_transfer kernel for all layers."""
    from sglang.srt.paras.peer_access import peer_access_kv_transfer

    heads_per_peer = max(1, NUM_KV_HEADS // world_size)
    num_local_tokens = TOKENS_PER_RANK[rank]
    dst_token_start = sum(TOKENS_PER_RANK[:rank])

    dst_base_ptrs = torch.tensor(
        peer_ctx.peer_addresses, dtype=torch.int64, device="cuda"
    )
    local_buffer_ptr = mgr._buffer.data_ptr()
    local_token_indices = torch.arange(
        num_local_tokens, dtype=torch.int32, device="cuda"
    )
    barrier_tensor = torch.zeros(1, device="cuda")

    dist.barrier(group=tp_group)

    for layer_id in range(NUM_LAYERS):
        # Compute byte offsets for EP (source) and TP (destination) K/V
        ep_k_entry = mgr._entries[f"model.layers.{layer_id}.kv.ep.k"]
        ep_v_entry = mgr._entries[f"model.layers.{layer_id}.kv.ep.v"]
        tp_k_entry = mgr._entries[f"model.layers.{layer_id}.kv.tp.k"]
        tp_v_entry = mgr._entries[f"model.layers.{layer_id}.kv.tp.v"]

        peer_access_kv_transfer(
            local_buffer_ptr=local_buffer_ptr,
            dst_base_ptrs_tensor=dst_base_ptrs,
            local_token_indices=local_token_indices,
            src_k_offset=ep_k_entry.offset_bytes,
            src_v_offset=ep_v_entry.offset_bytes,
            dst_k_offset=tp_k_entry.offset_bytes,
            dst_v_offset=tp_v_entry.offset_bytes,
            num_local_tokens=num_local_tokens,
            dst_token_start=dst_token_start,
            num_kv_heads=NUM_KV_HEADS,
            tp_rank=rank,
            tp_size=world_size,
            head_dim=HEAD_DIM,
            elem_size=ELEM_SIZE,
        )
        # Barrier after each layer to sync before next layer uses the slot
        dist.all_reduce(barrier_tensor, op=dist.ReduceOp.SUM, group=tp_group)

    torch.cuda.synchronize()


def read_tp_results(mgr, world_size):
    """Read TP KV buffers after transfer."""
    heads_per_peer = max(1, NUM_KV_HEADS // world_size)
    tp_k = {}
    tp_v = {}
    for layer_id in range(NUM_LAYERS):
        # TP uses the same bytes as EP but reshaped for fewer heads per rank
        # Union layout: same bytes, so TP_VIEW_TOKENS = (EP_MAX+PAGE) * total_heads / heads_per_peer
        tp_k[layer_id] = mgr.get_view_as(
            f"model.layers.{layer_id}.kv.tp.k",
            (TP_VIEW_TOKENS, heads_per_peer, HEAD_DIM),
        ).clone()
        tp_v[layer_id] = mgr.get_view_as(
            f"model.layers.{layer_id}.kv.tp.v",
            (TP_VIEW_TOKENS, heads_per_peer, HEAD_DIM),
        ).clone()
    return tp_k, tp_v


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_results(ref_k, ref_v, tp_k, tp_v, rank):
    """Check bit-exact match between reference and peer access results."""
    total_tokens = sum(TOKENS_PER_RANK)
    all_ok = True

    for layer_id in range(NUM_LAYERS):
        # Only compare the token range that was written
        rk = ref_k[layer_id][:total_tokens]
        rv = ref_v[layer_id][:total_tokens]
        tk = tp_k[layer_id][:total_tokens]
        tv = tp_v[layer_id][:total_tokens]

        k_match = torch.equal(rk, tk)
        v_match = torch.equal(rv, tv)

        if not k_match:
            diff = (rk != tk).sum().item()
            print(
                f"[Rank {rank}] FAIL layer={layer_id} K: "
                f"{diff}/{rk.numel()} elements differ",
                flush=True,
            )
            all_ok = False
        elif rank == 0:
            print(f"  [OK] layer={layer_id} K bitwise match", flush=True)

        if not v_match:
            diff = (rv != tv).sum().item()
            print(
                f"[Rank {rank}] FAIL layer={layer_id} V: "
                f"{diff}/{rv.numel()} elements differ",
                flush=True,
            )
            all_ok = False
        elif rank == 0:
            print(f"  [OK] layer={layer_id} V bitwise match", flush=True)

    return all_ok


# ---------------------------------------------------------------------------
# Correctness test
# ---------------------------------------------------------------------------

def run_correctness_test(rank, world_size):
    tp_group = setup_paras_state(rank, world_size)
    mgr = build_kv_manager(rank, world_size)

    fill_kv_data(mgr, rank)
    ref_k, ref_v = compute_reference(mgr, rank, world_size, tp_group)

    peer_ctx = setup_peer_ctx(mgr, rank, world_size, tp_group)
    run_peer_access_transfer(mgr, peer_ctx, rank, world_size, tp_group)
    tp_k, tp_v = read_tp_results(mgr, world_size)

    all_ok = compare_results(ref_k, ref_v, tp_k, tp_v, rank)
    return all_ok, tp_group, mgr, peer_ctx


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def _restore_kv_data(mgr, rank):
    """Refill KV data for benchmark iterations."""
    fill_kv_data(mgr, rank)


def _bench_peer_access(mgr, peer_ctx, rank, world_size, tp_group):
    """Run peer access KV transfer (for benchmarking)."""
    run_peer_access_transfer(mgr, peer_ctx, rank, world_size, tp_group)


def _bench_nccl(mgr, rank, world_size, tp_group):
    """Run NCCL all_gather-based KV transfer (for benchmarking)."""
    heads_per_peer = max(1, NUM_KV_HEADS // world_size)
    barrier_tensor = torch.zeros(1, device="cuda")

    for layer_id in range(NUM_LAYERS):
        ep_k = mgr.get_view(f"model.layers.{layer_id}.kv.ep.k")
        ep_v = mgr.get_view(f"model.layers.{layer_id}.kv.ep.v")

        num_local = TOKENS_PER_RANK[rank]
        head_start = rank * heads_per_peer

        # Extract this rank's head shard from local EP data
        my_k_shard = ep_k[:num_local, head_start:head_start + heads_per_peer, :].contiguous()
        my_v_shard = ep_v[:num_local, head_start:head_start + heads_per_peer, :].contiguous()

        # Pad to max tokens for all_gather (all ranks must send same size)
        max_tokens = max(TOKENS_PER_RANK)
        padded_k = torch.zeros(
            (max_tokens, heads_per_peer, HEAD_DIM), dtype=KV_DTYPE, device="cuda"
        )
        padded_v = torch.zeros_like(padded_k)
        padded_k[:num_local] = my_k_shard
        padded_v[:num_local] = my_v_shard

        gathered_k = [torch.zeros_like(padded_k) for _ in range(world_size)]
        gathered_v = [torch.zeros_like(padded_v) for _ in range(world_size)]
        dist.all_gather(gathered_k, padded_k, group=tp_group)
        dist.all_gather(gathered_v, padded_v, group=tp_group)

        # Scatter into TP buffer
        tp_k = mgr.get_view_as(
            f"model.layers.{layer_id}.kv.tp.k",
            (TP_VIEW_TOKENS, heads_per_peer, HEAD_DIM),
        )
        tp_v = mgr.get_view_as(
            f"model.layers.{layer_id}.kv.tp.v",
            (TP_VIEW_TOKENS, heads_per_peer, HEAD_DIM),
        )

        token_offset = 0
        for src_rank in range(world_size):
            n_tok = TOKENS_PER_RANK[src_rank]
            tp_k[token_offset:token_offset + n_tok] = gathered_k[src_rank][:n_tok]
            tp_v[token_offset:token_offset + n_tok] = gathered_v[src_rank][:n_tok]
            token_offset += n_tok

        dist.all_reduce(barrier_tensor, op=dist.ReduceOp.SUM, group=tp_group)

    torch.cuda.synchronize()


def run_benchmark(rank, world_size, tp_group, mgr, peer_ctx):
    """Time both NCCL and peer_access paths."""

    # --- warmup ---
    for _ in range(BENCHMARK_WARMUP):
        fill_kv_data(mgr, rank)
        _bench_nccl(mgr, rank, world_size, tp_group)
    torch.cuda.synchronize()
    dist.barrier()

    for _ in range(BENCHMARK_WARMUP):
        fill_kv_data(mgr, rank)
        _bench_peer_access(mgr, peer_ctx, rank, world_size, tp_group)
    torch.cuda.synchronize()
    dist.barrier()

    # --- NCCL timing ---
    nccl_times = []
    for _ in range(BENCHMARK_RUNS):
        fill_kv_data(mgr, rank)
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        _bench_nccl(mgr, rank, world_size, tp_group)
        torch.cuda.synchronize()
        nccl_times.append(time.perf_counter() - t0)

    # --- Peer access timing ---
    pa_times = []
    for _ in range(BENCHMARK_RUNS):
        fill_kv_data(mgr, rank)
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        _bench_peer_access(mgr, peer_ctx, rank, world_size, tp_group)
        torch.cuda.synchronize()
        pa_times.append(time.perf_counter() - t0)

    if rank == 0:
        def _stats(t):
            return sum(t) / len(t), min(t), max(t)

        na, nm, nx = _stats(nccl_times)
        pa, pm, px = _stats(pa_times)
        total_tokens = sum(TOKENS_PER_RANK)

        print(f"\n{'=' * 80}")
        print(
            f"KV BENCHMARK ({NUM_LAYERS} layers, {NUM_KV_HEADS} heads, "
            f"head_dim={HEAD_DIM}, tokens={total_tokens}, TP={world_size}, "
            f"runs={BENCHMARK_RUNS})"
        )
        print(f"{'=' * 80}")
        print(f"  {'Method':<14s}  {'avg(ms)':>10s}  {'min(ms)':>10s}  {'max(ms)':>10s}  {'vs nccl':>10s}")
        print(f"  {'nccl':<14s}  {na*1000:10.3f}  {nm*1000:10.3f}  {nx*1000:10.3f}  {'1.00x':>10s}")
        print(f"  {'peer_access':<14s}  {pa*1000:10.3f}  {pm*1000:10.3f}  {px*1000:10.3f}  {na/pa:10.2f}x")
        print(f"{'=' * 80}")

        print(f"\nPer-run times (ms):")
        print(f"  {'Run':>4s}  {'nccl':>10s}  {'peer_access':>12s}")
        for i in range(BENCHMARK_RUNS):
            print(f"  {i:4d}  {nccl_times[i]*1000:10.3f}  {pa_times[i]*1000:12.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ParaS KV cache peer access correctness test and benchmark"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Run benchmark after correctness test"
    )
    args = parser.parse_args()

    # GPU check before any GPU work (rank 0 only before dist init)
    if os.environ.get("RANK", "0") == "0":
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        )
        used = [int(x) for x in result.stdout.strip().split("\n")]
        empty = sum(1 for x in used if x < 100)
        print(f"[rank0] GPU memory check: {used} MiB used, {empty} empty GPUs")
        assert empty >= 4, f"Need 4 empty GPUs, only {empty} available (used: {used})"

    rank, world_size = setup_distributed()

    try:
        ok, tp_group, mgr, peer_ctx = run_correctness_test(rank, world_size)

        if not ok:
            dist.barrier()
            if rank == 0:
                print("\nFAILED: Bitwise mismatch detected!", flush=True)
            teardown_distributed()
            sys.exit(1)

        dist.barrier()
        if rank == 0:
            print(
                f"\nSUCCESS: All {NUM_LAYERS} layers × K/V × "
                f"{world_size} ranks bitwise match (peer_access vs reference)!",
                flush=True,
            )

        if args.benchmark:
            run_benchmark(rank, world_size, tp_group, mgr, peer_ctx)

        dist.barrier()
    except Exception as e:
        print(f"[Rank {rank}] ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        try:
            teardown_distributed()
        except Exception:
            pass
        sys.exit(1)

    teardown_distributed()
    sys.exit(0)


if __name__ == "__main__":
    main()
