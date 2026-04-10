#!/usr/bin/env python3
"""
4-GPU comparison test for ParaS peer access weight transfers.

Tests that paras_configure_tp_peer_access() produces bitwise-identical
results to paras_configure_tp_all_to_all() for DP=1 configuration.

Usage:
  torchrun --nproc_per_node=4 test_paras_peer_access.py           # correctness
  torchrun --nproc_per_node=4 test_paras_peer_access.py --benchmark  # + timing
"""

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist

# Add sglang to path
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.join(_TEST_DIR, "..", "..")
sys.path.insert(0, os.path.join(_ROOT_DIR, "python"))
# CUDA extension path (built with setup.py build_ext --inplace)
sys.path.insert(0, os.path.join(_ROOT_DIR, "python", "sglang", "srt", "paras", "csrc"))

# ---- test constants ----
NUM_LAYERS = 4
HIDDEN = 512
INTERMEDIATE = 512
NUM_EXPERTS = 16
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
    """Set ParaS parallel state globals without full sglang server init.

    Also stubs out sglang.srt.distributed.parallel_state._TP so that
    module-level import code (e.g. get_tensor_model_parallel_rank()) works.
    """
    import sglang.srt.distributed.parallel_state as ps
    import sglang.srt.paras.paras_parallel_state as pps

    # TP group = all ranks (DP=1 means TP covers the entire world)
    tp_group = dist.new_group(ranks=list(range(world_size)))

    tp_coord = _SimpleGroupCoordinator(tp_group, world_size, f"cuda:{rank}", rank_in_group=rank)

    # Stub sglang's global _TP so import-time code in fused_moe_triton
    # (get_tensor_model_parallel_rank, log_info_on_rank0) doesn't crash.
    ps._TP = tp_coord

    # ParaS-specific parallel state
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

def build_manager(rank, world_size):
    """Create ParaSMemoryManager with EP weight + staging buffers."""
    from sglang.srt.paras.paras_memory_manager import (
        ParaSMemoryManager,
        set_global_paras_memory_manager,
    )

    ep_size = world_size
    num_local = NUM_EXPERTS // ep_size

    mgr = ParaSMemoryManager(device=f"cuda:{rank}")

    # Per-layer EP weight buffers (non-triton shape: E_local, 2*I, H / E_local, H, I)
    for layer_id in range(NUM_LAYERS):
        mgr.reserve(
            f"model.layers.{layer_id}.mlp.experts.w13_weight",
            (num_local, 2 * INTERMEDIATE, HIDDEN),
            torch.bfloat16,
        )
        mgr.reserve(
            f"model.layers.{layer_id}.mlp.experts.w2_weight",
            (num_local, HIDDEN, INTERMEDIATE),
            torch.bfloat16,
        )

    # Staging buffers (same layout as plan_qwen_moe_layout for DP=1)
    staging_experts = num_local  # dp_size * (num_experts // ep_size) = 1 * num_local
    mgr.reserve(
        "staging.w13_a",
        (staging_experts, 2 * INTERMEDIATE, HIDDEN),
        torch.bfloat16,
    )
    mgr.reserve(
        "staging.w13_b",
        (staging_experts, 2 * INTERMEDIATE, HIDDEN),
        torch.bfloat16,
    )
    mgr.reserve(
        "staging.w2_a",
        (staging_experts, HIDDEN, INTERMEDIATE),
        torch.bfloat16,
    )
    mgr.reserve(
        "staging.w2_b",
        (staging_experts, HIDDEN, INTERMEDIATE),
        torch.bfloat16,
    )

    mgr.materialize()
    set_global_paras_memory_manager(mgr)
    return mgr, num_local


def fill_ep_weights(mgr, rank):
    """Fill EP weight buffers with rank-deterministic random data."""
    for layer_id in range(NUM_LAYERS):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(SEED + layer_id * 100 + rank)
        w13 = mgr.get_view(f"model.layers.{layer_id}.mlp.experts.w13_weight")
        w2 = mgr.get_view(f"model.layers.{layer_id}.mlp.experts.w2_weight")
        w13.copy_(
            torch.randn(w13.shape, generator=gen, dtype=torch.float32).to(
                dtype=w13.dtype, device=w13.device
            )
        )
        gen2 = torch.Generator(device="cpu")
        gen2.manual_seed(SEED + layer_id * 100 + rank + 50)
        w2.copy_(
            torch.randn(w2.shape, generator=gen2, dtype=torch.float32).to(
                dtype=w2.dtype, device=w2.device
            )
        )


def snapshot_weights(mgr):
    """Clone all EP weight buffers."""
    snap = {}
    for layer_id in range(NUM_LAYERS):
        snap[layer_id] = (
            mgr.get_view(f"model.layers.{layer_id}.mlp.experts.w13_weight").clone(),
            mgr.get_view(f"model.layers.{layer_id}.mlp.experts.w2_weight").clone(),
        )
    return snap


def restore_weights(mgr, snap):
    """Restore all EP weight buffers from snapshot."""
    for layer_id in range(NUM_LAYERS):
        mgr.get_view(f"model.layers.{layer_id}.mlp.experts.w13_weight").copy_(
            snap[layer_id][0]
        )
        mgr.get_view(f"model.layers.{layer_id}.mlp.experts.w2_weight").copy_(
            snap[layer_id][1]
        )


# ---------------------------------------------------------------------------
# Mixin construction + path runners
# ---------------------------------------------------------------------------

def _make_mixin(layer_id, num_local, mgr):
    """Create minimal ParaSMoeBlockMixin — mimics paras_configure_tp_all_gather(DP=1)."""
    from sglang.srt.paras.layers.paras_moe_block import ParaSMoeBlockMixin

    m = object.__new__(ParaSMoeBlockMixin)
    m._paras_layer_id = layer_id
    m.num_local_experts = num_local
    m.num_global_experts = NUM_EXPERTS
    m.hidden_size = HIDDEN
    m.moe_intermediate_size = INTERMEDIATE

    # For DP=1 the all_gather is a no-op: ep_gathered IS the EP buffer view
    w13 = mgr.get_view(f"model.layers.{layer_id}.mlp.experts.w13_weight")
    w2 = mgr.get_view(f"model.layers.{layer_id}.mlp.experts.w2_weight")
    m.w13_ep_gathered = w13.view(num_local, 2 * INTERMEDIATE, HIDDEN)
    m.w2_ep_gathered = w2.view(num_local, HIDDEN, INTERMEDIATE)
    return m


def run_nccl_path(mgr, num_local):
    """Run NCCL all-to-all for all layers. Returns {layer_id: (w13, w2)} clones."""
    results = {}
    for layer_id in range(NUM_LAYERS):
        mixin = _make_mixin(layer_id, num_local, mgr)
        mixin.paras_configure_tp_all_to_all()
        results[layer_id] = (
            mgr.get_view(
                f"model.layers.{layer_id}.mlp.experts.w13_weight"
            ).clone(),
            mgr.get_view(
                f"model.layers.{layer_id}.mlp.experts.w2_weight"
            ).clone(),
        )
    return results


def run_peer_path(mgr, num_local, peer_ctx, packed_plans):
    """Run peer access for all layers. Returns {layer_id: (w13, w2)} clones."""
    results = {}
    for layer_id in range(NUM_LAYERS):
        mixin = _make_mixin(layer_id, num_local, mgr)
        mixin.paras_configure_tp_peer_access(
            peer_ctx=peer_ctx,
            transfer_plans={},
            packed_plans=packed_plans,
            staging_suffix="a",
            stream=None,
        )
        results[layer_id] = (
            mgr.get_view(
                f"model.layers.{layer_id}.mlp.experts.w13_weight"
            ).clone(),
            mgr.get_view(
                f"model.layers.{layer_id}.mlp.experts.w2_weight"
            ).clone(),
        )
    return results


# ---------------------------------------------------------------------------
# Peer access setup
# ---------------------------------------------------------------------------

def setup_peer_ctx(mgr, rank, world_size, tp_group):
    """Enable peer access and exchange CUDA IPC handles for cross-process access.

    In a multi-process setup (torchrun), raw data_ptr() values are only valid
    within their own process.  We use CUDA IPC to map remote buffers into the
    local address space so the peer-access kernel can write directly.
    """
    import ctypes

    from sglang.srt.paras.peer_access import PeerAccessContext, enable_peer_access

    enable_peer_access(list(range(world_size)))

    _cudart = ctypes.CDLL("libcudart.so")

    # --- ctypes types for CUDA IPC ---
    IPC_HANDLE_SIZE = 64

    class CudaIpcMemHandle(ctypes.Structure):
        _fields_ = [("reserved", ctypes.c_ubyte * IPC_HANDLE_SIZE)]

    _cudart.cudaIpcGetMemHandle.argtypes = [
        ctypes.POINTER(CudaIpcMemHandle),
        ctypes.c_void_p,
    ]
    _cudart.cudaIpcGetMemHandle.restype = ctypes.c_int
    _cudart.cudaIpcOpenMemHandle.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        CudaIpcMemHandle,
        ctypes.c_uint,
    ]
    _cudart.cudaIpcOpenMemHandle.restype = ctypes.c_int

    # 1. Get IPC handle for local buffer
    local_handle = CudaIpcMemHandle()
    ret = _cudart.cudaIpcGetMemHandle(
        ctypes.byref(local_handle), ctypes.c_void_p(mgr._buffer.data_ptr())
    )
    assert ret == 0, f"cudaIpcGetMemHandle failed (cuda error {ret})"

    # 2. Exchange handles via all_gather (64 bytes per rank)
    handle_tensor = torch.tensor(
        list(local_handle.reserved), dtype=torch.uint8, device=f"cuda:{rank}"
    )
    assert handle_tensor.numel() == IPC_HANDLE_SIZE
    all_handles = torch.zeros(
        world_size * IPC_HANDLE_SIZE, dtype=torch.uint8, device=f"cuda:{rank}"
    )
    dist.all_gather_into_tensor(all_handles, handle_tensor, group=tp_group)

    # 3. Open remote handles to get local-address mappings
    peer_addresses = []
    for r in range(world_size):
        if r == rank:
            peer_addresses.append(mgr._buffer.data_ptr())
        else:
            raw_list = (
                all_handles[r * IPC_HANDLE_SIZE : (r + 1) * IPC_HANDLE_SIZE]
                .cpu()
                .tolist()
            )
            remote_handle = CudaIpcMemHandle()
            for idx, val in enumerate(raw_list):
                remote_handle.reserved[idx] = val
            remote_ptr = ctypes.c_void_p()
            ret = _cudart.cudaIpcOpenMemHandle(
                ctypes.byref(remote_ptr),
                remote_handle,
                1,  # cudaIpcMemLazyEnablePeerAccess
            )
            assert ret == 0, (
                f"cudaIpcOpenMemHandle for rank {r} failed (cuda error {ret})"
            )
            peer_addresses.append(remote_ptr.value)

    return PeerAccessContext(
        peer_addresses=peer_addresses,
        peer_access_enabled=True,
        tp_group=tp_group,
        tp_size=world_size,
    )


def build_packed_plans(mgr, world_size):
    """Pre-build packed transfer plans for all layers and weight types."""
    from sglang.srt.paras.paras_parallel_state import get_paras_tp_rank
    from sglang.srt.paras.transfer_plan import build_transfer_plan, pack_transfer_plan

    tp_rank = get_paras_tp_rank()
    plans = {}
    for layer_id in range(NUM_LAYERS):
        for wt in ("w13", "w2"):
            entries = build_transfer_plan(mgr, layer_id, wt, world_size, tp_rank)
            plans[(layer_id, wt)] = pack_transfer_plan(entries)
    return plans


# ---------------------------------------------------------------------------
# Comparison test
# ---------------------------------------------------------------------------

def run_comparison_test(rank, world_size):
    """Verify bitwise match between NCCL and peer access paths."""
    tp_group = setup_paras_state(rank, world_size)
    mgr, num_local = build_manager(rank, world_size)

    fill_ep_weights(mgr, rank)
    snap = snapshot_weights(mgr)

    # ---- NCCL path ----
    nccl_results = run_nccl_path(mgr, num_local)

    # ---- Peer access path (restore EP weights first) ----
    restore_weights(mgr, snap)
    peer_ctx = setup_peer_ctx(mgr, rank, world_size, tp_group)
    packed_plans = build_packed_plans(mgr, world_size)
    peer_results = run_peer_path(mgr, num_local, peer_ctx, packed_plans)

    # ---- Compare ----
    all_ok = True
    for layer_id in range(NUM_LAYERS):
        for i, wt in enumerate(("w13", "w2")):
            nccl_t = nccl_results[layer_id][i]
            peer_t = peer_results[layer_id][i]
            if not torch.equal(nccl_t, peer_t):
                diff = (nccl_t != peer_t).sum().item()
                print(
                    f"[Rank {rank}] FAIL layer={layer_id} {wt}: "
                    f"{diff}/{nccl_t.numel()} elements differ",
                    flush=True,
                )
                all_ok = False
            else:
                if rank == 0:
                    print(
                        f"  [OK] layer={layer_id} {wt} bitwise match",
                        flush=True,
                    )

    return all_ok, tp_group, mgr, num_local, snap, peer_ctx, packed_plans


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    rank, world_size, tp_group, mgr, num_local, snap, peer_ctx, packed_plans
):
    """Time NCCL vs peer access and print comparison table."""
    # Warmup
    for _ in range(BENCHMARK_WARMUP):
        restore_weights(mgr, snap)
        run_nccl_path(mgr, num_local)
    for _ in range(BENCHMARK_WARMUP):
        restore_weights(mgr, snap)
        run_peer_path(mgr, num_local, peer_ctx, packed_plans)

    torch.cuda.synchronize()
    dist.barrier()

    # ---- Time NCCL ----
    nccl_times = []
    for _ in range(BENCHMARK_RUNS):
        restore_weights(mgr, snap)
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        run_nccl_path(mgr, num_local)
        torch.cuda.synchronize()
        nccl_times.append(time.perf_counter() - t0)

    # ---- Time peer access ----
    peer_times = []
    for _ in range(BENCHMARK_RUNS):
        restore_weights(mgr, snap)
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        run_peer_path(mgr, num_local, peer_ctx, packed_plans)
        torch.cuda.synchronize()
        peer_times.append(time.perf_counter() - t0)

    if rank == 0:
        na = sum(nccl_times) / len(nccl_times)
        nm = min(nccl_times)
        nx = max(nccl_times)
        pa = sum(peer_times) / len(peer_times)
        pm = min(peer_times)
        px = max(peer_times)
        speedup = na / pa if pa > 0 else float("inf")

        print(f"\n{'=' * 64}")
        print(
            f"BENCHMARK ({NUM_LAYERS} layers, {NUM_EXPERTS} experts, "
            f"hidden={HIDDEN}, inter={INTERMEDIATE}, TP={world_size}, "
            f"runs={BENCHMARK_RUNS})"
        )
        print(f"{'=' * 64}")
        print(
            f"  NCCL:        avg={na * 1000:8.3f}ms  "
            f"min={nm * 1000:8.3f}ms  max={nx * 1000:8.3f}ms"
        )
        print(
            f"  Peer Access: avg={pa * 1000:8.3f}ms  "
            f"min={pm * 1000:8.3f}ms  max={px * 1000:8.3f}ms"
        )
        print(f"  Speedup:     {speedup:.2f}x (peer access vs NCCL avg)")
        print(f"{'=' * 64}")

        # Per-run detail
        print(f"\nPer-run times (ms):")
        print(f"  {'Run':>4s}  {'NCCL':>10s}  {'Peer':>10s}")
        for i in range(BENCHMARK_RUNS):
            print(
                f"  {i:4d}  {nccl_times[i] * 1000:10.3f}  "
                f"{peer_times[i] * 1000:10.3f}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ParaS peer access comparison test and latency benchmark"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Run latency benchmark after correctness test"
    )
    args = parser.parse_args()

    rank, world_size = setup_distributed()

    try:
        result = run_comparison_test(rank, world_size)
        ok = result[0]

        if not ok:
            dist.barrier()
            if rank == 0:
                print("\nFAILED: Bitwise mismatch detected!", flush=True)
            teardown_distributed()
            sys.exit(1)

        dist.barrier()
        if rank == 0:
            print(
                f"\nSUCCESS: All {NUM_LAYERS} layers × 2 weights × "
                f"{world_size} ranks bitwise match!",
                flush=True,
            )

        if args.benchmark:
            _, tp_group, mgr, num_local, snap, peer_ctx, packed_plans = result
            run_benchmark(
                rank, world_size, tp_group, mgr, num_local, snap, peer_ctx, packed_plans
            )

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
