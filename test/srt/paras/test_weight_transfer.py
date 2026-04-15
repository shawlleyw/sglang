#!/usr/bin/env python3
"""
Weight transfer tests for both EP→TP and TP→EP directions.

Tests that weight redistribution produces correct results:
  1. EP→TP: peer_access vs NCCL bitwise comparison (w13, w2 separately)
  2. TP→EP: MoE pointer swap verification
  3. EP→TP→EP: round-trip bitwise match

Usage:
  torchrun --nproc_per_node=4 test/srt/paras/test_weight_transfer.py
"""

import ctypes
import os
import sys

import torch
import torch.distributed as dist

# Add sglang to path
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.join(_TEST_DIR, "..", "..", "..")
sys.path.insert(0, os.path.join(_ROOT_DIR, "python"))

# ---- test constants (Qwen3-30B-A3B) ----
NUM_LAYERS = 8
HIDDEN = 2048
INTERMEDIATE = 1536
NUM_EXPERTS = 64
SEED = 42


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
    tp_coord = _SimpleGroupCoordinator(
        tp_group, world_size, f"cuda:{rank}", rank_in_group=rank
    )

    # Stub sglang's global _TP
    ps._TP = tp_coord

    # ParaS-specific parallel state
    pps._PARAS_TP = tp_coord
    pps._PARAS_DP = _SimpleGroupCoordinator(
        None, 1, f"cuda:{rank}", rank_in_group=0
    )
    pps._PARAS_SELF = _SimpleGroupCoordinator(
        None, 1, f"cuda:{rank}", rank_in_group=0
    )

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
        mgr._entries[f"model.layers.{i}.mlp.experts.w13_weight"] = mgr._entries[
            f"paras.moe_slot.{i + 1}.w13"
        ]
        mgr._entries[f"model.layers.{i}.mlp.experts.w2_weight"] = mgr._entries[
            f"paras.moe_slot.{i + 1}.w2"
        ]

    # Staging buffers for NCCL all-to-all and overlap paths
    staging_experts = num_local
    w13_staging_shape = (staging_experts, 2 * INTERMEDIATE, HIDDEN)
    w2_staging_shape = (staging_experts, HIDDEN, INTERMEDIATE)
    for sfx in ("", "_1", "_2"):
        mgr.reserve(
            f"staging.w13_pre_permute{sfx}", w13_staging_shape, torch.bfloat16
        )
        mgr.reserve(
            f"staging.w2_pre_permute{sfx}", w2_staging_shape, torch.bfloat16
        )

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
            mgr.get_view(
                f"model.layers.{layer_id}.mlp.experts.w13_weight"
            ).clone(),
            mgr.get_view(
                f"model.layers.{layer_id}.mlp.experts.w2_weight"
            ).clone(),
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

    def paras_configure_tp_mlp_all_gather(
        self, stream, handles, async_op=False, staging_suffix=""
    ):
        return self.mlp.paras_configure_tp_all_gather(
            stream, handles, async_op, staging_suffix
        )

    def paras_configure_tp_mlp_all_to_all(
        self, stream, handles, staging_suffix=""
    ):
        return self.mlp.paras_configure_tp_all_to_all(
            stream, handles, staging_suffix
        )

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

    layers = [
        _MockLayer(_make_mixin(i, num_local, mgr, set_gathered=False))
        for i in range(NUM_LAYERS)
    ]

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
                stream_2,
                last_layer_handles,
                async_op=True,
                staging_suffix=staging_2,
            )

        layer.paras_configure_tp_mlp_all_to_all(
            stream_1, last_layer_handles, staging_1
        )
        layer.paras_configure_tp(tp_size, 0)

        if not_last_layer:
            last_layer_handles = new_handles
            stream_1, stream_2 = stream_2, stream_1
            staging_1, staging_2 = staging_2, staging_1

    torch.cuda.synchronize()
    return _read_tp_results(mgr, tp_inter)


def run_peer_access_path(mgr, num_local, peer_ctx):
    from sglang.srt.paras.paras_parallel_state import (
        get_paras_tp_group,
        get_paras_tp_size,
    )

    tp_size = get_paras_tp_size()
    tp_inter = INTERMEDIATE // tp_size

    paras_tp_group = get_paras_tp_group().device_group
    dst_base_ptrs = torch.tensor(
        peer_ctx.peer_addresses, dtype=torch.int64, device="cuda"
    )
    barrier_tensor = torch.zeros(1, device="cuda")
    dist.barrier(group=paras_tp_group)

    for layer_id in range(NUM_LAYERS):
        mixin = _make_mixin(layer_id, num_local, mgr)
        mixin.paras_configure_tp_fused_peer_access_kernel(
            peer_ctx, dst_base_ptrs, None
        )
        dist.all_reduce(
            barrier_tensor, op=dist.ReduceOp.SUM, group=paras_tp_group
        )

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
# Test classes
# ---------------------------------------------------------------------------


class TestEPtoTPWeightTransfer:
    """EP→TP: peer_access vs NCCL comparison (from test_paras_peer_access.py)."""

    def __init__(self, rank, world_size, mgr, num_local, snap, peer_ctx):
        self.rank = rank
        self.world_size = world_size
        self.mgr = mgr
        self.num_local = num_local
        self.snap = snap
        self.peer_ctx = peer_ctx
        self._naive_results = None
        self._pa_results = None

    def _ensure_results(self):
        """Run NCCL naive and peer_access paths, cache results."""
        if self._naive_results is not None:
            return
        restore_weights(self.mgr, self.snap)
        self._naive_results = run_naive_path(self.mgr, self.num_local)
        restore_weights(self.mgr, self.snap)
        self._pa_results = run_peer_access_path(
            self.mgr, self.num_local, self.peer_ctx
        )

    def test_w13_peer_access_vs_nccl(self):
        """w13 weights must be bitwise identical between peer_access and NCCL."""
        self._ensure_results()
        for layer_id in range(NUM_LAYERS):
            ref = self._naive_results[layer_id][0].reshape(-1)
            test = self._pa_results[layer_id][0].reshape(-1)
            if not torch.equal(ref, test):
                diff = (ref != test).sum().item()
                raise AssertionError(
                    f"[Rank {self.rank}] w13 mismatch layer={layer_id}: "
                    f"{diff}/{ref.numel()} elements differ"
                )
        if self.rank == 0:
            print(
                "  [OK] w13 peer_access vs NCCL: bitwise match all layers",
                flush=True,
            )

    def test_w2_peer_access_vs_nccl(self):
        """w2 weights must be bitwise identical between peer_access and NCCL."""
        self._ensure_results()
        for layer_id in range(NUM_LAYERS):
            ref = self._naive_results[layer_id][1].reshape(-1)
            test = self._pa_results[layer_id][1].reshape(-1)
            if not torch.equal(ref, test):
                diff = (ref != test).sum().item()
                raise AssertionError(
                    f"[Rank {self.rank}] w2 mismatch layer={layer_id}: "
                    f"{diff}/{ref.numel()} elements differ"
                )
        if self.rank == 0:
            print(
                "  [OK] w2 peer_access vs NCCL: bitwise match all layers",
                flush=True,
            )


class TestTPtoEPWeightRestore:
    """TP→EP: MoE pointer swap verification."""

    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size

    def test_moe_pointer_swap(self):
        """Create mock MoE, verify experts toggle between ep/tp."""
        from sglang.srt.paras.layers.paras_moe_block import ParaSMoeBlockMixin

        m = object.__new__(ParaSMoeBlockMixin)
        m._paras_layer_id = 0
        m.num_local_experts = NUM_EXPERTS // self.world_size
        m.num_global_experts = NUM_EXPERTS
        m.hidden_size = HIDDEN
        m.moe_intermediate_size = INTERMEDIATE

        class _TaggedExperts:
            def __init__(self, tag):
                self.tag = tag

        ep_exp = _TaggedExperts("ep")
        tp_exp = _TaggedExperts("tp")

        m.ep_experts = ep_exp
        m.tp_experts = tp_exp
        m.experts = ep_exp
        m.parallelism_config = "ep"
        m.tp_size = 1

        # Switch to TP
        m.paras_configure_tp(self.world_size, self.rank)
        assert (
            m.experts is tp_exp
        ), "After configure_tp, experts should be tp_experts"
        assert m.parallelism_config == "tp"

        # Switch back to EP
        m.paras_configure_ep()
        assert (
            m.experts is ep_exp
        ), "After configure_ep, experts should be ep_experts"
        assert m.parallelism_config == "ep"

        if self.rank == 0:
            print(
                "  [OK] MoE pointer swap: ep→tp→ep verified", flush=True
            )


class TestWeightRoundTrip:
    """EP→TP→EP: weight data bitwise match."""

    def __init__(self, rank, world_size, mgr, num_local, snap):
        self.rank = rank
        self.world_size = world_size
        self.mgr = mgr
        self.num_local = num_local
        self.snap = snap

    def test_weight_roundtrip(self):
        """Verify ep_experts data matches after EP→TP→EP cycle."""
        # Restore clean EP weights
        restore_weights(self.mgr, self.snap)

        # EP→TP via NCCL naive all-to-all
        run_naive_path(self.mgr, self.num_local)

        # TP→EP via NCCL naive reverse — MUST be in reversed layer order
        # to respect N+1 slot aliasing (EP slot[i+1] = TP slot[i+1])
        for layer_id in reversed(range(NUM_LAYERS)):
            mixin = _make_mixin(layer_id, self.num_local, self.mgr)
            mixin.paras_configure_ep_mlp_naive()

        # Compare restored EP weights to original snapshot
        for layer_id in range(NUM_LAYERS):
            w13 = self.mgr.get_view(
                f"model.layers.{layer_id}.mlp.experts.w13_weight"
            )
            w2 = self.mgr.get_view(
                f"model.layers.{layer_id}.mlp.experts.w2_weight"
            )
            w13_orig = self.snap[layer_id][0]
            w2_orig = self.snap[layer_id][1]

            if not torch.equal(w13.reshape(-1), w13_orig.reshape(-1)):
                diff = (w13.reshape(-1) != w13_orig.reshape(-1)).sum().item()
                raise AssertionError(
                    f"[Rank {self.rank}] Round-trip w13 mismatch layer={layer_id}: "
                    f"{diff}/{w13.numel()} elements differ"
                )
            if not torch.equal(w2.reshape(-1), w2_orig.reshape(-1)):
                diff = (w2.reshape(-1) != w2_orig.reshape(-1)).sum().item()
                raise AssertionError(
                    f"[Rank {self.rank}] Round-trip w2 mismatch layer={layer_id}: "
                    f"{diff}/{w2.numel()} elements differ"
                )

        if self.rank == 0:
            print(
                "  [OK] EP→TP→EP round-trip: bitwise match all layers",
                flush=True,
            )


class TestEPtoTPGroundTruth:
    """EP→TP: standalone ground-truth verification against independently computed expected weights."""

    def __init__(self, rank, world_size, mgr, num_local, snap, tp_group):
        self.rank = rank
        self.world_size = world_size
        self.mgr = mgr
        self.num_local = num_local
        self.snap = snap
        self.tp_group = tp_group
        self._actual = None
        self._all_ep_w13 = None
        self._all_ep_w2 = None

    def _ensure_results(self):
        """Run EP→TP via NCCL naive and gather all EP data for ground truth."""
        if self._actual is not None:
            return

        # Restore clean EP weights and gather all ranks' EP data via all_gather
        restore_weights(self.mgr, self.snap)

        self._all_ep_w13 = {}
        self._all_ep_w2 = {}
        for layer_id in range(NUM_LAYERS):
            local_w13 = self.snap[layer_id][0]  # (num_local, 2*INTERMEDIATE, HIDDEN)
            local_w2 = self.snap[layer_id][1]   # (num_local, HIDDEN, INTERMEDIATE)

            gathered_w13 = [
                torch.empty_like(local_w13) for _ in range(self.world_size)
            ]
            gathered_w2 = [
                torch.empty_like(local_w2) for _ in range(self.world_size)
            ]
            dist.all_gather(gathered_w13, local_w13, group=self.tp_group)
            dist.all_gather(gathered_w2, local_w2, group=self.tp_group)

            # (NUM_EXPERTS, 2*INTERMEDIATE, HIDDEN)
            self._all_ep_w13[layer_id] = torch.cat(gathered_w13, dim=0)
            # (NUM_EXPERTS, HIDDEN, INTERMEDIATE)
            self._all_ep_w2[layer_id] = torch.cat(gathered_w2, dim=0)

        # Run EP→TP via NCCL naive
        restore_weights(self.mgr, self.snap)
        tp_inter = INTERMEDIATE // self.world_size
        self._actual = run_naive_path(self.mgr, self.num_local)

    def test_w13_ground_truth(self):
        """w13 TP result must match independently computed ground truth."""
        self._ensure_results()
        tp_inter = INTERMEDIATE // self.world_size
        r = self.rank

        for layer_id in range(NUM_LAYERS):
            full_w13 = self._all_ep_w13[layer_id]  # (NUM_EXPERTS, 2*INTERMEDIATE, HIDDEN)

            gate_shard = full_w13[:, r * tp_inter : (r + 1) * tp_inter, :]
            up_shard = full_w13[
                :, INTERMEDIATE + r * tp_inter : INTERMEDIATE + (r + 1) * tp_inter, :
            ]
            expected = torch.cat([gate_shard, up_shard], dim=1)

            actual = self._actual[layer_id][0]  # (NUM_EXPERTS, 2*tp_inter, HIDDEN)
            if not torch.equal(actual, expected):
                diff = (actual.reshape(-1) != expected.reshape(-1)).sum().item()
                raise AssertionError(
                    f"[Rank {self.rank}] w13 ground-truth mismatch layer={layer_id}: "
                    f"{diff}/{actual.numel()} elements differ"
                )

        if self.rank == 0:
            print(
                "  [OK] w13 EP→TP ground truth: bitwise match all layers",
                flush=True,
            )

    def test_w2_ground_truth(self):
        """w2 TP result must match independently computed ground truth."""
        self._ensure_results()
        tp_inter = INTERMEDIATE // self.world_size
        r = self.rank

        for layer_id in range(NUM_LAYERS):
            full_w2 = self._all_ep_w2[layer_id]  # (NUM_EXPERTS, HIDDEN, INTERMEDIATE)

            expected = full_w2[:, :, r * tp_inter : (r + 1) * tp_inter]

            actual = self._actual[layer_id][1]  # (NUM_EXPERTS, HIDDEN, tp_inter)
            if not torch.equal(actual, expected):
                diff = (actual.reshape(-1) != expected.reshape(-1)).sum().item()
                raise AssertionError(
                    f"[Rank {self.rank}] w2 ground-truth mismatch layer={layer_id}: "
                    f"{diff}/{actual.numel()} elements differ"
                )

        if self.rank == 0:
            print(
                "  [OK] w2 EP→TP ground truth: bitwise match all layers",
                flush=True,
            )


class TestTPtoEPGroundTruth:
    """TP→EP reverse: verify EP weights match original after EP→TP→EP with reversed layer order."""

    def __init__(self, rank, world_size, mgr, num_local, snap):
        self.rank = rank
        self.world_size = world_size
        self.mgr = mgr
        self.num_local = num_local
        self.snap = snap

    def test_reverse_naive_vs_original(self):
        """EP→TP then TP→EP in reversed layer order must reproduce original EP weights."""
        # Restore clean EP weights
        restore_weights(self.mgr, self.snap)

        # EP→TP via NCCL naive all-to-all
        run_naive_path(self.mgr, self.num_local)

        # TP→EP via NCCL naive reverse — MUST iterate in reversed layer order
        # N+1 slot aliasing: forward order would corrupt source data for later layers
        for layer_id in reversed(range(NUM_LAYERS)):
            mixin = _make_mixin(layer_id, self.num_local, self.mgr)
            mixin.paras_configure_ep_mlp_naive()

        # Compare restored EP weights to original snapshot
        for layer_id in range(NUM_LAYERS):
            w13 = self.mgr.get_view(
                f"model.layers.{layer_id}.mlp.experts.w13_weight"
            )
            w2 = self.mgr.get_view(
                f"model.layers.{layer_id}.mlp.experts.w2_weight"
            )
            w13_orig = self.snap[layer_id][0]
            w2_orig = self.snap[layer_id][1]

            if not torch.equal(w13, w13_orig):
                diff = (w13.reshape(-1) != w13_orig.reshape(-1)).sum().item()
                raise AssertionError(
                    f"[Rank {self.rank}] TP→EP w13 mismatch layer={layer_id}: "
                    f"{diff}/{w13.numel()} elements differ"
                )
            if not torch.equal(w2, w2_orig):
                diff = (w2.reshape(-1) != w2_orig.reshape(-1)).sum().item()
                raise AssertionError(
                    f"[Rank {self.rank}] TP→EP w2 mismatch layer={layer_id}: "
                    f"{diff}/{w2.numel()} elements differ"
                )

        if self.rank == 0:
            print(
                "  [OK] TP→EP reverse (reversed layer order): bitwise match all layers",
                flush=True,
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    rank, world_size = setup_distributed()
    passed = 0
    failed = 0

    try:
        tp_group = setup_paras_state(rank, world_size)
        mgr, num_local = build_manager(rank, world_size)
        fill_ep_weights(mgr, rank)
        snap = snapshot_weights(mgr)

        # --- EP→TP weight transfer tests ---
        if rank == 0:
            print("\n=== TestEPtoTPWeightTransfer ===", flush=True)
        peer_ctx = setup_peer_ctx(mgr, rank, world_size, tp_group)
        ep_tp = TestEPtoTPWeightTransfer(
            rank, world_size, mgr, num_local, snap, peer_ctx
        )

        for name in ("test_w13_peer_access_vs_nccl", "test_w2_peer_access_vs_nccl"):
            try:
                getattr(ep_tp, name)()
                passed += 1
            except Exception as e:
                print(f"  [FAIL] {name}: {e}", flush=True)
                failed += 1

        # --- TP→EP pointer swap test ---
        if rank == 0:
            print("\n=== TestTPtoEPWeightRestore ===", flush=True)
        tp_ep = TestTPtoEPWeightRestore(rank, world_size)
        try:
            tp_ep.test_moe_pointer_swap()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] test_moe_pointer_swap: {e}", flush=True)
            failed += 1

        # --- Round-trip test ---
        if rank == 0:
            print("\n=== TestWeightRoundTrip ===", flush=True)
        rt = TestWeightRoundTrip(rank, world_size, mgr, num_local, snap)
        try:
            rt.test_weight_roundtrip()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] test_weight_roundtrip: {e}", flush=True)
            failed += 1

        # --- EP→TP ground truth tests ---
        if rank == 0:
            print("\n=== TestEPtoTPGroundTruth ===", flush=True)
        gt_ep_tp = TestEPtoTPGroundTruth(
            rank, world_size, mgr, num_local, snap, tp_group
        )
        for name in ("test_w13_ground_truth", "test_w2_ground_truth"):
            try:
                getattr(gt_ep_tp, name)()
                passed += 1
            except Exception as e:
                print(f"  [FAIL] {name}: {e}", flush=True)
                failed += 1

        # --- TP→EP ground truth test ---
        if rank == 0:
            print("\n=== TestTPtoEPGroundTruth ===", flush=True)
        gt_tp_ep = TestTPtoEPGroundTruth(
            rank, world_size, mgr, num_local, snap
        )
        try:
            gt_tp_ep.test_reverse_naive_vs_original()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] test_reverse_naive_vs_original: {e}", flush=True)
            failed += 1

        # --- Summary ---
        dist.barrier()
        if rank == 0:
            total = passed + failed
            print(f"\n{'=' * 60}")
            print(f"RESULTS: {passed}/{total} passed, {failed}/{total} failed")
            if failed == 0:
                print(
                    f"SUCCESS: All weight transfer tests passed "
                    f"({NUM_LAYERS} layers × {world_size} ranks)!"
                )
            else:
                print("FAILED: Some tests failed!")
            print(f"{'=' * 60}", flush=True)

        if failed > 0:
            teardown_distributed()
            sys.exit(1)

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
