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
    pps._PARAS_EP = tp_coord
    pps._PARAS_TP = tp_coord
    pps._PARAS_DP = _SimpleGroupCoordinator(None, 1, f"cuda:{rank}", rank_in_group=0)
    pps._PARAS_SELF = _SimpleGroupCoordinator(None, 1, f"cuda:{rank}", rank_in_group=0)

    pps._PARAS_TP_SIZE = world_size
    pps._PARAS_TP_RANK = rank
    pps._PARAS_DP_SIZE = 1
    pps._PARAS_DP_RANK = 0
    pps._PARAS_EP_SIZE = world_size
    pps._PARAS_EP_RANK = rank
    pps._PARAS_EP_GROUP_IS_NODE_LOCAL = True
    pps._PARAS_TP_GROUP_IS_NODE_LOCAL = True

    return tp_group


# ---------------------------------------------------------------------------
# Memory manager helpers
# ---------------------------------------------------------------------------


def build_manager(rank, world_size):
    """Create ParaSMemoryManager via the four-anchor plan_qwen_moe_layout API.

    The removed N+1 slot layout has been replaced by the deferred four-anchor
    pass inside materialize(): plan_qwen_moe_layout stashes per-layer sizes,
    materialize() creates the ep_experts / tp_experts primaries plus the
    experts.{w13,w2}_weight aliases, and create_paras_moe_aliases is now a
    validation shim that asserts the primaries exist.

    moe_tp_size=1 keeps each EP expert's full intermediate dimension. The NCCL
    method reserves one pre-permute buffer per weight; peer_access needs
    no staging.
    """
    from sglang.srt.paras.paras_memory_manager import (
        ParaSMemoryManager,
        create_paras_moe_aliases,
        plan_qwen_moe_layout,
        set_global_paras_memory_manager,
    )

    ep_size = world_size
    tp_size = world_size  # DP=1: TP covers the entire world
    num_local = NUM_EXPERTS // ep_size

    # Attention constants for Qwen3-30B-A3B parity. The test never exercises
    # attention weights, but plan_qwen_moe_layout still reserves qkv_proj /
    # o_proj slots, so these must yield valid shapes.
    NUM_HEADS = 32
    NUM_KV_HEADS = 4
    HEAD_DIM = 128

    mgr = ParaSMemoryManager(device=f"cuda:{rank}")

    plan_qwen_moe_layout(
        mgr,
        num_layers=NUM_LAYERS,
        num_experts=NUM_EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        ep_size=ep_size,
        tp_size=tp_size,
        dp_size=1,
        moe_tp_size=1,
        quant_name=None,
        intra_node_weight_transfer_method="nccl",
        prefix="model",
    )

    mgr.materialize()
    create_paras_moe_aliases(mgr, NUM_LAYERS, prefix="model")
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


def _make_mixin(layer_id, num_local, mgr):
    from sglang.srt.paras.layers.paras_moe_block import ParaSMoeBlockMixin

    m = object.__new__(ParaSMoeBlockMixin)
    m._paras_layer_id = layer_id
    m._paras_interleaved_w13 = False
    m.num_local_experts = num_local
    m.num_global_experts = NUM_EXPERTS
    m.hidden_size = HIDDEN
    m.moe_intermediate_size = INTERMEDIATE

    w13 = mgr.get_view(f"model.layers.{layer_id}.mlp.experts.w13_weight")
    w2 = mgr.get_view(f"model.layers.{layer_id}.mlp.experts.w2_weight")
    m.ep_experts = _MockExperts(w13, w2)

    return m


class _ModelLayerAdapter:
    """Expose weight transfers while making mode activation a no-op."""

    def __init__(self, mlp):
        self.mlp = mlp

    def paras_reshard_ep_to_tp_intra_node_nccl(self, dp_rank, dp_size):
        self.mlp.paras_reshard_ep_to_tp_intra_node_nccl(dp_rank, dp_size)

    def paras_reshard_ep_to_tp_intra_node_peer_access(
        self, dst_base_ptrs, dp_rank, dp_size, stream
    ):
        self.mlp.paras_reshard_ep_to_tp_intra_node_peer_access(
            dst_base_ptrs, dp_rank, dp_size, stream
        )

    def paras_reshard_tp_to_ep_intra_node_nccl(self, dp_rank, dp_size):
        self.mlp.paras_reshard_tp_to_ep_intra_node_nccl(dp_rank, dp_size)

    def paras_reshard_tp_to_ep_intra_node_peer_access(
        self, dst_base_ptrs, dp_rank, dp_size, stream
    ):
        self.mlp.paras_reshard_tp_to_ep_intra_node_peer_access(
            dst_base_ptrs, dp_rank, dp_size, stream
        )

    def paras_configure_tp_attn(self, paras_tp_size, paras_tp_rank):
        pass

    def paras_configure_tp(self, paras_tp_size, paras_tp_rank):
        pass

    def paras_configure_ep_attn(self):
        pass

    def paras_configure_ep(self):
        pass


def _make_model(mgr, num_local, peer_ctx=None):
    from sglang.srt.paras.layers.paras_model import ParaSModelMixin

    model = object.__new__(ParaSModelMixin)
    model.layers = [
        _ModelLayerAdapter(_make_mixin(layer_id, num_local, mgr))
        for layer_id in range(NUM_LAYERS)
    ]
    model._peer_access_ctx = peer_ctx
    return model


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


def run_nccl_path(mgr, num_local):
    from sglang.srt.paras.paras_parallel_state import (
        get_paras_tp_rank,
        get_paras_tp_size,
    )

    tp_size = get_paras_tp_size()
    tp_inter = INTERMEDIATE // tp_size
    model = _make_model(mgr, num_local)
    model.paras_configure_tp(tp_size, get_paras_tp_rank(), intra_node_method="nccl")
    return _read_tp_results(mgr, tp_inter)


def run_peer_access_path(mgr, num_local, peer_ctx):
    from sglang.srt.paras.paras_parallel_state import (
        get_paras_tp_rank,
        get_paras_tp_size,
    )

    tp_size = get_paras_tp_size()
    tp_inter = INTERMEDIATE // tp_size
    model = _make_model(mgr, num_local, peer_ctx)
    model.paras_configure_tp(
        tp_size, get_paras_tp_rank(), intra_node_method="peer_access"
    )
    return _read_tp_results(mgr, tp_inter)


def run_peer_access_reverse_path(mgr, num_local, peer_ctx):
    model = _make_model(mgr, num_local, peer_ctx)
    model.paras_configure_ep(intra_node_method="peer_access")
    return _read_ep_results(mgr)


def run_nccl_reverse_path(mgr, num_local):
    model = _make_model(mgr, num_local)
    model.paras_configure_ep(intra_node_method="nccl")
    return _read_ep_results(mgr)


def _read_ep_results(mgr):
    results = {}
    for layer_id in range(NUM_LAYERS):
        results[layer_id] = (
            mgr.get_view(f"model.layers.{layer_id}.mlp.experts.w13_weight").clone(),
            mgr.get_view(f"model.layers.{layer_id}.mlp.experts.w2_weight").clone(),
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
            assert (
                ret == 0
            ), f"cudaIpcOpenMemHandle for rank {r} failed (cuda error {ret})"
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
        assert m.experts is tp_exp, "After configure_tp, experts should be tp_experts"
        assert m.parallelism_config == "tp"

        # Switch back to EP
        m.paras_configure_ep()
        assert m.experts is ep_exp, "After configure_ep, experts should be ep_experts"
        assert m.parallelism_config == "ep"

        if self.rank == 0:
            print("  [OK] MoE pointer swap: ep→tp→ep verified", flush=True)


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

        # EP→TP via NCCL all-to-all (reverse order, handled inside)
        run_nccl_path(self.mgr, self.num_local)

        run_nccl_reverse_path(self.mgr, self.num_local)

        # Compare restored EP weights to original snapshot
        for layer_id in range(NUM_LAYERS):
            w13 = self.mgr.get_view(f"model.layers.{layer_id}.mlp.experts.w13_weight")
            w2 = self.mgr.get_view(f"model.layers.{layer_id}.mlp.experts.w2_weight")
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
    """Verify NCCL and peer_access EP→TP against independent ground truth."""

    def __init__(self, rank, world_size, mgr, num_local, snap, tp_group, peer_ctx):
        self.rank = rank
        self.world_size = world_size
        self.mgr = mgr
        self.num_local = num_local
        self.snap = snap
        self.tp_group = tp_group
        self.peer_ctx = peer_ctx
        self._expected_w13 = None
        self._expected_w2 = None

    def _build_ground_truth(self):
        if self._expected_w13 is not None:
            return

        tp_inter = INTERMEDIATE // self.world_size
        r = self.rank
        self._expected_w13 = {}
        self._expected_w2 = {}

        for layer_id in range(NUM_LAYERS):
            local_w13 = self.snap[layer_id][0]
            local_w2 = self.snap[layer_id][1]
            gathered_w13 = [torch.empty_like(local_w13) for _ in range(self.world_size)]
            gathered_w2 = [torch.empty_like(local_w2) for _ in range(self.world_size)]
            dist.all_gather(gathered_w13, local_w13, group=self.tp_group)
            dist.all_gather(gathered_w2, local_w2, group=self.tp_group)
            full_w13 = torch.cat(gathered_w13, dim=0)
            full_w2 = torch.cat(gathered_w2, dim=0)

            gate_shard = full_w13[:, r * tp_inter : (r + 1) * tp_inter, :]
            up_shard = full_w13[
                :, INTERMEDIATE + r * tp_inter : INTERMEDIATE + (r + 1) * tp_inter, :
            ]
            self._expected_w13[layer_id] = torch.cat([gate_shard, up_shard], dim=1)
            self._expected_w2[layer_id] = full_w2[
                :, :, r * tp_inter : (r + 1) * tp_inter
            ]

    def _verify_against_ground_truth(self, actual, method_name):
        self._build_ground_truth()
        for layer_id in range(NUM_LAYERS):
            aw13, aw2 = actual[layer_id]
            if not torch.equal(aw13, self._expected_w13[layer_id]):
                diff = (
                    (aw13.reshape(-1) != self._expected_w13[layer_id].reshape(-1))
                    .sum()
                    .item()
                )
                raise AssertionError(
                    f"[Rank {self.rank}] {method_name} w13 mismatch layer={layer_id}: "
                    f"{diff}/{aw13.numel()} elements differ"
                )
            if not torch.equal(aw2, self._expected_w2[layer_id]):
                diff = (
                    (aw2.reshape(-1) != self._expected_w2[layer_id].reshape(-1))
                    .sum()
                    .item()
                )
                raise AssertionError(
                    f"[Rank {self.rank}] {method_name} w2 mismatch layer={layer_id}: "
                    f"{diff}/{aw2.numel()} elements differ"
                )
        if self.rank == 0:
            print(
                f"  [OK] EP→TP {method_name}: bitwise match ground truth all layers",
                flush=True,
            )

    def test_nccl_vs_ground_truth(self):
        restore_weights(self.mgr, self.snap)
        actual = run_nccl_path(self.mgr, self.num_local)
        self._verify_against_ground_truth(actual, "NCCL")

    def test_peer_access_vs_ground_truth(self):
        restore_weights(self.mgr, self.snap)
        actual = run_peer_access_path(self.mgr, self.num_local, self.peer_ctx)
        self._verify_against_ground_truth(actual, "peer_access")


class TestTPtoEPGroundTruth:
    """Verify NCCL and peer_access TP→EP recover the original EP weights."""

    def __init__(self, rank, world_size, mgr, num_local, snap, peer_ctx):
        self.rank = rank
        self.world_size = world_size
        self.mgr = mgr
        self.num_local = num_local
        self.snap = snap
        self.peer_ctx = peer_ctx

    def _run_ep_to_tp(self):
        restore_weights(self.mgr, self.snap)
        run_nccl_path(self.mgr, self.num_local)

    def _verify_ep_matches_original(self, method_name):
        actual = _read_ep_results(self.mgr)
        for layer_id in range(NUM_LAYERS):
            w13_orig = self.snap[layer_id][0]
            w2_orig = self.snap[layer_id][1]
            aw13, aw2 = actual[layer_id]
            if not torch.equal(aw13.reshape(-1), w13_orig.reshape(-1)):
                diff = (aw13.reshape(-1) != w13_orig.reshape(-1)).sum().item()
                raise AssertionError(
                    f"[Rank {self.rank}] TP→EP {method_name} w13 mismatch layer={layer_id}: "
                    f"{diff}/{aw13.numel()} elements differ"
                )
            if not torch.equal(aw2.reshape(-1), w2_orig.reshape(-1)):
                diff = (aw2.reshape(-1) != w2_orig.reshape(-1)).sum().item()
                raise AssertionError(
                    f"[Rank {self.rank}] TP→EP {method_name} w2 mismatch layer={layer_id}: "
                    f"{diff}/{aw2.numel()} elements differ"
                )
        if self.rank == 0:
            print(
                f"  [OK] TP→EP {method_name}: bitwise match original EP all layers",
                flush=True,
            )

    def test_nccl_reverse_vs_original(self):
        self._run_ep_to_tp()
        run_nccl_reverse_path(self.mgr, self.num_local)
        self._verify_ep_matches_original("NCCL reverse")

    def test_peer_access_reverse_vs_original(self):
        self._run_ep_to_tp()
        run_peer_access_reverse_path(self.mgr, self.num_local, self.peer_ctx)
        self._verify_ep_matches_original("peer_access reverse")


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

        peer_ctx = setup_peer_ctx(mgr, rank, world_size, tp_group)

        # --- EP→TP ground truth: NCCL + peer_access ---
        if rank == 0:
            print("\n=== TestEPtoTPGroundTruth ===", flush=True)
        gt_ep_tp = TestEPtoTPGroundTruth(
            rank, world_size, mgr, num_local, snap, tp_group, peer_ctx
        )
        for name in ("test_nccl_vs_ground_truth", "test_peer_access_vs_ground_truth"):
            try:
                getattr(gt_ep_tp, name)()
                passed += 1
            except Exception as e:
                print(f"  [FAIL] {name}: {e}", flush=True)
                failed += 1

        # --- TP→EP ground truth: NCCL reverse + peer_access reverse ---
        if rank == 0:
            print("\n=== TestTPtoEPGroundTruth ===", flush=True)
        gt_tp_ep = TestTPtoEPGroundTruth(
            rank, world_size, mgr, num_local, snap, peer_ctx
        )
        for name in (
            "test_nccl_reverse_vs_original",
            "test_peer_access_reverse_vs_original",
        ):
            try:
                getattr(gt_tp_ep, name)()
                passed += 1
            except Exception as e:
                print(f"  [FAIL] {name}: {e}", flush=True)
                failed += 1

        # --- TP→EP pointer swap test (module-level attribute, orthogonal to data) ---
        if rank == 0:
            print("\n=== TestTPtoEPWeightRestore ===", flush=True)
        tp_ep = TestTPtoEPWeightRestore(rank, world_size)
        try:
            tp_ep.test_moe_pointer_swap()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] test_moe_pointer_swap: {e}", flush=True)
            failed += 1

        # --- Round-trip test (full model-level flow) ---
        if rank == 0:
            print("\n=== TestWeightRoundTrip ===", flush=True)
        rt = TestWeightRoundTrip(rank, world_size, mgr, num_local, snap)
        try:
            rt.test_weight_roundtrip()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] test_weight_roundtrip: {e}", flush=True)
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
