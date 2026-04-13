#!/usr/bin/env python3
"""
4-GPU comparison test for ParaS peer access weight transfers.

Tests that peer access v2 kernels produce bitwise-identical results to
NCCL all-to-all for DP=1 configuration.

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

# ---- test constants (Qwen3-30B-A3B) ----
NUM_LAYERS = 8
HIDDEN = 2048
INTERMEDIATE = 1536
NUM_EXPERTS = 64
SEED = 42
BENCHMARK_WARMUP = 5
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
    """Create ParaSMemoryManager with N+1 slots + staging buffers."""
    from sglang.srt.paras.paras_memory_manager import (
        ParaSMemoryManager,
        create_paras_moe_aliases,
        set_global_paras_memory_manager,
    )

    ep_size = world_size
    num_local = NUM_EXPERTS // ep_size

    mgr = ParaSMemoryManager(device=f"cuda:{rank}")

    # N+1 generic physical slots (non-triton shape: E_local, 2*I, H / E_local, H, I)
    for slot in range(NUM_LAYERS + 1):
        mgr.reserve(
            f"paras.moe_slot.{slot}.w13",
            (num_local, 2 * INTERMEDIATE, HIDDEN),
            torch.bfloat16,
        )
        mgr.reserve(
            f"paras.moe_slot.{slot}.w2",
            (num_local, HIDDEN, INTERMEDIATE),
            torch.bfloat16,
        )

    # 'experts' aliases → slot i+1 (for weight loading / EP access)
    for i in range(NUM_LAYERS):
        mgr._entries[f"model.layers.{i}.mlp.experts.w13_weight"] = mgr._entries[f"paras.moe_slot.{i+1}.w13"]
        mgr._entries[f"model.layers.{i}.mlp.experts.w2_weight"] = mgr._entries[f"paras.moe_slot.{i+1}.w2"]

    staging_experts = num_local
    w13_staging_shape = (staging_experts, 2 * INTERMEDIATE, HIDDEN)
    w2_staging_shape = (staging_experts, HIDDEN, INTERMEDIATE)
    for sfx in ("", "_1", "_2"):
        mgr.reserve(f"staging.w13_pre_permute{sfx}", w13_staging_shape, torch.bfloat16)
        mgr.reserve(f"staging.w2_pre_permute{sfx}", w2_staging_shape, torch.bfloat16)

    mgr.materialize()
    create_paras_moe_aliases(mgr, NUM_LAYERS)
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

class _MockExperts:
    def __init__(self, w13_view, w2_view):
        self.w13_weight = torch.nn.Parameter(w13_view, requires_grad=False)
        self.w2_weight = torch.nn.Parameter(w2_view, requires_grad=False)


def _make_mixin(layer_id, num_local, mgr, set_gathered=True):
    from sglang.srt.paras.layers.paras_moe_block import ParaSMoeBlockMixin

    m = object.__new__(ParaSMoeBlockMixin)
    m._paras_layer_id = layer_id
    m.num_local_experts = num_local
    m.num_global_experts = NUM_EXPERTS
    m.hidden_size = HIDDEN
    m.moe_intermediate_size = INTERMEDIATE

    w13 = mgr.get_view(f"model.layers.{layer_id}.mlp.experts.w13_weight")
    w2 = mgr.get_view(f"model.layers.{layer_id}.mlp.experts.w2_weight")
    m.ep_experts = _MockExperts(w13, w2)

    if set_gathered:
        m.w13_ep_gathered = w13.view(num_local, 2 * INTERMEDIATE, HIDDEN)
        m.w2_ep_gathered = w2.view(num_local, HIDDEN, INTERMEDIATE)
    return m


class _MockLayer:
    """Wraps a ParaSMoeBlockMixin to satisfy the overlap path's layer interface."""

    def __init__(self, mixin):
        self.mlp = mixin

    def paras_configure_tp_attn(self, tp_size, tp_rank):
        pass

    def paras_configure_tp_mlp_all_gather(self, stream, handles, async_op=False, staging_suffix=""):
        return self.mlp.paras_configure_tp_all_gather(stream, handles, async_op, staging_suffix)

    def paras_configure_tp_mlp_all_to_all(self, stream, handles, staging_suffix=""):
        return self.mlp.paras_configure_tp_all_to_all(stream, handles, staging_suffix)

    def paras_configure_tp(self, tp_size, tp_rank):
        pass


def _read_tp_results(mgr, tp_inter):
    results = {}
    for layer_id in range(NUM_LAYERS):
        results[layer_id] = (
            mgr.get_view_as(
                f"model.layers.{layer_id}.mlp.tp_experts.w13_weight",
                (NUM_EXPERTS, 2 * tp_inter, HIDDEN),
            ).clone(),
            mgr.get_view_as(
                f"model.layers.{layer_id}.mlp.tp_experts.w2_weight",
                (NUM_EXPERTS, HIDDEN, tp_inter),
            ).clone(),
        )
    return results


def run_naive_path(mgr, num_local):
    from sglang.srt.paras.paras_parallel_state import get_paras_tp_size

    tp_size = get_paras_tp_size()
    tp_inter = INTERMEDIATE // tp_size
    for layer_id in range(NUM_LAYERS):
        mixin = _make_mixin(layer_id, num_local, mgr)
        mixin.paras_configure_tp_all_to_all()
    return _read_tp_results(mgr, tp_inter)


def run_overlap_path(mgr, num_local):
    from sglang.srt.paras.paras_parallel_state import get_paras_tp_size

    tp_size = get_paras_tp_size()
    tp_inter = INTERMEDIATE // tp_size

    layers = [_MockLayer(_make_mixin(i, num_local, mgr, set_gathered=False)) for i in range(NUM_LAYERS)]

    stream_1 = torch.cuda.Stream()
    stream_2 = torch.cuda.Stream()
    staging_1 = "_1"
    staging_2 = "_2"

    layers[0].paras_configure_tp_attn(tp_size, 0)
    last_layer_handles = layers[0].paras_configure_tp_mlp_all_gather(
        stream_1, [], async_op=True, staging_suffix=staging_1
    )
    nlayers = len(layers)
    for i, layer in enumerate(layers):
        not_last_layer = i < nlayers - 1
        if not_last_layer:
            next_layer = layers[i + 1]
            next_layer.paras_configure_tp_attn(tp_size, 0)
            new_handles = next_layer.paras_configure_tp_mlp_all_gather(
                stream_2, last_layer_handles, async_op=True, staging_suffix=staging_2
            )

        layer.paras_configure_tp_mlp_all_to_all(stream_1, last_layer_handles, staging_1)
        layer.paras_configure_tp(tp_size, 0)

        if not_last_layer:
            last_layer_handles = new_handles
            stream_1, stream_2 = stream_2, stream_1
            staging_1, staging_2 = staging_2, staging_1

    torch.cuda.synchronize()
    return _read_tp_results(mgr, tp_inter)


def run_peer_access_path(mgr, num_local, peer_ctx):
    from sglang.srt.paras.paras_parallel_state import get_paras_tp_size, get_paras_tp_group

    tp_size = get_paras_tp_size()
    tp_inter = INTERMEDIATE // tp_size

    paras_tp_group = get_paras_tp_group().device_group
    dst_base_ptrs = torch.tensor(peer_ctx.peer_addresses, dtype=torch.int64, device="cuda")
    barrier_tensor = torch.zeros(1, device="cuda")
    dist.barrier(group=paras_tp_group)

    for layer_id in range(NUM_LAYERS):
        mixin = _make_mixin(layer_id, num_local, mgr)
        mixin.paras_configure_tp_fused_peer_access_kernel(peer_ctx, dst_base_ptrs, None)
        dist.all_reduce(barrier_tensor, op=dist.ReduceOp.SUM, group=paras_tp_group)

    return _read_tp_results(mgr, tp_inter)


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

    from sglang.srt.paras.peer_access import PeerAccessContext

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



# ---------------------------------------------------------------------------
# Comparison test
# ---------------------------------------------------------------------------

def _compare_results(ref_results, test_results, ref_name, test_name, rank):
    all_ok = True
    for layer_id in range(NUM_LAYERS):
        for i, wt in enumerate(("w13", "w2")):
            ref_flat = ref_results[layer_id][i].reshape(-1)
            test_flat = test_results[layer_id][i].reshape(-1)
            if not torch.equal(ref_flat, test_flat):
                diff = (ref_flat != test_flat).sum().item()
                print(
                    f"[Rank {rank}] FAIL {test_name} layer={layer_id} {wt}: "
                    f"{diff}/{ref_flat.numel()} elements differ",
                    flush=True,
                )
                all_ok = False
            elif rank == 0:
                print(
                    f"  [OK] {test_name} layer={layer_id} {wt} bitwise match vs {ref_name}",
                    flush=True,
                )
    return all_ok


def run_comparison_test(rank, world_size):
    tp_group = setup_paras_state(rank, world_size)
    mgr, num_local = build_manager(rank, world_size)

    fill_ep_weights(mgr, rank)
    snap = snapshot_weights(mgr)

    naive_results = run_naive_path(mgr, num_local)

    restore_weights(mgr, snap)
    peer_ctx = setup_peer_ctx(mgr, rank, world_size, tp_group)
    pa_results = run_peer_access_path(mgr, num_local, peer_ctx)

    all_ok = True

    if rank == 0:
        print("\n--- naive vs peer_access ---", flush=True)
    all_ok &= _compare_results(naive_results, pa_results, "naive", "peer_access", rank)

    restore_weights(mgr, snap)
    overlap_results = run_overlap_path(mgr, num_local)

    if rank == 0:
        print("\n--- naive vs overlap ---", flush=True)
    all_ok &= _compare_results(naive_results, overlap_results, "naive", "overlap", rank)

    del naive_results, overlap_results, pa_results
    torch.cuda.empty_cache()

    return all_ok, tp_group, mgr, num_local, snap, peer_ctx


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------

def _bench_method(name, run_fn, mgr, num_local, snap, peer_ctx=None):
    args = (mgr, num_local, peer_ctx) if peer_ctx else (mgr, num_local)
    for _ in range(BENCHMARK_WARMUP):
        restore_weights(mgr, snap)
        run_fn(*args)
    torch.cuda.synchronize()
    dist.barrier()

    times = []
    for _ in range(BENCHMARK_RUNS):
        restore_weights(mgr, snap)
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        run_fn(*args)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times


def run_benchmark(
    rank, world_size, tp_group, mgr, num_local, snap, peer_ctx,
):
    naive_times = _bench_method("naive", run_naive_path, mgr, num_local, snap)
    overlap_times = _bench_method("overlap", run_overlap_path, mgr, num_local, snap)
    pa_times = _bench_method("peer_access", run_peer_access_path, mgr, num_local, snap, peer_ctx)

    if rank == 0:
        def _stats(t):
            return sum(t) / len(t), min(t), max(t)

        na, nm, nx = _stats(naive_times)
        oa, om, ox = _stats(overlap_times)
        pa, pm, px = _stats(pa_times)

        print(f"\n{'=' * 80}")
        print(
            f"BENCHMARK ({NUM_LAYERS} layers, {NUM_EXPERTS} experts, "
            f"hidden={HIDDEN}, inter={INTERMEDIATE}, TP={world_size}, "
            f"runs={BENCHMARK_RUNS})"
        )
        print(f"{'=' * 80}")
        print(f"  {'Method':<14s}  {'avg':>10s}  {'min':>10s}  {'max':>10s}  {'vs naive':>10s}")
        print(f"  {'naive':<14s}  {na*1000:10.3f}  {nm*1000:10.3f}  {nx*1000:10.3f}  {'1.00x':>10s}")
        print(f"  {'overlap':<14s}  {oa*1000:10.3f}  {om*1000:10.3f}  {ox*1000:10.3f}  {na/oa:10.2f}x")
        print(f"  {'peer_access':<14s}  {pa*1000:10.3f}  {pm*1000:10.3f}  {px*1000:10.3f}  {na/pa:10.2f}x")
        print(f"{'=' * 80}")

        print(f"\nPer-run times (ms):")
        print(f"  {'Run':>4s}  {'naive':>10s}  {'overlap':>10s}  {'peer_access':>12s}")
        for i in range(BENCHMARK_RUNS):
            print(
                f"  {i:4d}  {naive_times[i]*1000:10.3f}  "
                f"{overlap_times[i]*1000:10.3f}  {pa_times[i]*1000:12.3f}"
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
                f"{world_size} ranks bitwise match (naive vs overlap vs peer_access)!",
                flush=True,
            )

        if args.benchmark:
            _, tp_group, mgr, num_local, snap, peer_ctx = result
            run_benchmark(
                rank, world_size, tp_group, mgr, num_local, snap, peer_ctx,
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
