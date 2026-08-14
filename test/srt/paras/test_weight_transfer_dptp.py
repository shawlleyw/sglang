#!/usr/bin/env python3
"""Real dp=2 tp=2 ep=4 weight-transfer round-trip test (asymmetric DP x TP).

Validates both dp=2/tp=2 transfer topologies: direct dptp peer access
within one node, and the logical multi-node path that combines node-local
peer access with an in-place all-gather across DP groups.

Topology (4 GPUs):
  global rank R in [0,4);  tp_rank = R % T (=R%2);  dp_rank = R // T (=R//2)
  EP mode:     4 ranks, each holds num_experts/ep_size = num_experts/4 experts,
               full intermediate (moe_tp_size=2 means ep stores 2 shards worth).
  DP x TP mode: each rank holds ALL num_experts experts, tp-sharded on the
               intermediate dim by tp_rank, and REPLICATED across dp_rank
               (ranks R and R+T carry identical TP data).

Forward (EP -> DP x TP), source rank R, shard t, expert e (0..E_local):
  canonical global expert g = R * E_local + e lands at TP slot g on every
  destination rank d*T + t (d in [0,G)); each such rank receives shard t only.
  So rank dr ends up with TP[g] = shard (dr % T) of EP-source(g) for all g,
  identical across the two dp replicas.

Reverse (DP x TP -> EP): each dp-replica reassembles its own EP shard from its
  T tp-peers (replica-local, no broadcast); the redundant replica is dropped.
  A correct round trip must recover the original EP weights bitwise.

With PARAS_TEST_LOGICAL_MULTINODE=0 (default), IPC spans all four GPUs
and exercises the dptp kernels. With it set to 1, IPC is restricted to the
logical TP nodes {0,1}/{2,3}, while DP groups {0,2}/{1,3} perform the
cross-node all-gather.

Usage:
  CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
      test/srt/paras/test_weight_transfer_dptp.py
  PARAS_TEST_LOGICAL_MULTINODE=1 CUDA_VISIBLE_DEVICES=4,5,6,7 \
      torchrun --nproc_per_node=4 test/srt/paras/test_weight_transfer_dptp.py
"""

import os
import sys

import torch
import torch.distributed as dist

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.join(_TEST_DIR, "..", "..", "..")
sys.path.insert(0, os.path.join(_ROOT_DIR, "python"))

# ---- test constants (Qwen3-30B-A3B geometry; H,I are dptp-dispatch presets) ----
NUM_LAYERS = 8
HIDDEN = 2048
INTERMEDIATE = 768  # (H=2048, I=768) is a supported dptp preset
NUM_EXPERTS = 64
SEED = 42

TP_SIZE = 2
DP_SIZE = 2
EP_SIZE = DP_SIZE * TP_SIZE
LOGICAL_MULTINODE = os.environ.get("PARAS_TEST_LOGICAL_MULTINODE", "0") == "1"


def setup_distributed():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == EP_SIZE, f"requires exactly {EP_SIZE} GPUs, got {world_size}"
    torch.cuda.set_device(rank)
    return rank, world_size


def teardown_distributed():
    dist.destroy_process_group()


class _SimpleGroupCoordinator:
    def __init__(self, device_group, world_size, device, rank_in_group=0):
        self.device_group = device_group
        self.world_size = world_size
        self.device = torch.device(device)
        self.rank_in_group = rank_in_group
        self.rank = dist.get_rank()
        self.local_rank = dist.get_rank()


def setup_paras_state(rank, world_size):
    """ParaS state for a physical or logical dp=2/tp=2 topology."""
    import sglang.srt.distributed.parallel_state as ps
    import sglang.srt.paras.paras_parallel_state as pps

    world_group = dist.new_group(ranks=list(range(world_size)))
    world_coord = _SimpleGroupCoordinator(
        world_group, world_size, f"cuda:{rank}", rank_in_group=rank
    )

    tp_coord = None
    for d in range(DP_SIZE):
        ranks = list(range(d * TP_SIZE, (d + 1) * TP_SIZE))
        group = dist.new_group(ranks=ranks)
        if rank in ranks:
            tp_coord = _SimpleGroupCoordinator(
                group,
                TP_SIZE,
                f"cuda:{rank}",
                rank_in_group=ranks.index(rank),
            )

    dp_coord = None
    for t in range(TP_SIZE):
        ranks = list(range(t, world_size, TP_SIZE))
        group = dist.new_group(ranks=ranks)
        if rank in ranks:
            dp_coord = _SimpleGroupCoordinator(
                group,
                DP_SIZE,
                f"cuda:{rank}",
                rank_in_group=ranks.index(rank),
            )

    assert tp_coord is not None and dp_coord is not None
    ps._TP = world_coord
    pps._PARAS_EP = world_coord
    pps._PARAS_TP = tp_coord
    pps._PARAS_DP = dp_coord
    pps._PARAS_SELF = _SimpleGroupCoordinator(None, 1, f"cuda:{rank}", rank_in_group=0)

    pps._PARAS_TP_SIZE = TP_SIZE
    pps._PARAS_TP_RANK = rank % TP_SIZE
    pps._PARAS_DP_SIZE = DP_SIZE
    pps._PARAS_DP_RANK = rank // TP_SIZE
    pps._PARAS_EP_SIZE = EP_SIZE
    pps._PARAS_EP_RANK = rank
    pps._PARAS_EP_GROUP_IS_NODE_LOCAL = not LOGICAL_MULTINODE
    pps._PARAS_TP_GROUP_IS_NODE_LOCAL = True

    return world_group, tp_coord.device_group, dp_coord.device_group


def build_manager(rank, world_size):
    from sglang.srt.paras.paras_memory_manager import (
        ParaSMemoryManager,
        create_paras_moe_aliases,
        plan_qwen_moe_layout,
        set_global_paras_memory_manager,
    )

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
        ep_size=EP_SIZE,
        tp_size=TP_SIZE,
        dp_size=DP_SIZE,
        moe_tp_size=TP_SIZE,
        quant_name=None,
        configure_method="direct",
        prefix="model",
    )
    mgr.materialize()
    create_paras_moe_aliases(mgr, NUM_LAYERS, prefix="model")
    set_global_paras_memory_manager(mgr)

    e_local = NUM_EXPERTS // EP_SIZE
    return mgr, e_local


def fill_ep_weights(mgr, rank):
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
    snap = {}
    for layer_id in range(NUM_LAYERS):
        snap[layer_id] = (
            mgr.get_view(f"model.layers.{layer_id}.mlp.experts.w13_weight").clone(),
            mgr.get_view(f"model.layers.{layer_id}.mlp.experts.w2_weight").clone(),
        )
    return snap


def restore_weights(mgr, snap):
    for layer_id in range(NUM_LAYERS):
        mgr.get_view(f"model.layers.{layer_id}.mlp.experts.w13_weight").copy_(
            snap[layer_id][0]
        )
        mgr.get_view(f"model.layers.{layer_id}.mlp.experts.w2_weight").copy_(
            snap[layer_id][1]
        )


class _MockExperts:
    def __init__(self, w13_view, w2_view):
        self.w13_weight = torch.nn.Parameter(w13_view, requires_grad=False)
        self.w2_weight = torch.nn.Parameter(w2_view, requires_grad=False)


def _make_mixin(layer_id, num_local_ep, mgr):
    from sglang.srt.paras.layers.paras_moe_block import ParaSMoeBlockMixin

    m = object.__new__(ParaSMoeBlockMixin)
    m._paras_layer_id = layer_id
    m._paras_interleaved_w13 = False
    # num_local_experts is the TP-mode count (num_experts/tp_size); the dptp
    # path uses num_global_experts and computes E_local internally.
    m.num_local_experts = NUM_EXPERTS // TP_SIZE
    m.num_global_experts = NUM_EXPERTS
    m.hidden_size = HIDDEN
    m.moe_intermediate_size = INTERMEDIATE

    w13 = mgr.get_view(f"model.layers.{layer_id}.mlp.experts.w13_weight")
    w2 = mgr.get_view(f"model.layers.{layer_id}.mlp.experts.w2_weight")
    m.ep_experts = _MockExperts(w13, w2)
    return m


class _ModelLayerAdapter:
    """Expose the decoder-layer API used by the model-level pipeline."""

    def __init__(self, mlp):
        self.mlp = mlp

    def paras_reshard_ep_to_tp_node_peer(self, dst_base_ptrs, dp_rank, dp_size, stream):
        return self.mlp.paras_reshard_ep_to_tp_node_peer(
            dst_base_ptrs, dp_rank, dp_size, stream
        )

    def paras_broadcast_ep_to_dptp_peer(self, dst_base_ptrs, ep_rank, dp_size, stream):
        self.mlp.paras_broadcast_ep_to_dptp_peer(
            dst_base_ptrs, ep_rank, dp_size, stream
        )

    def paras_all_gather_tp_weights(self, stream):
        return self.mlp.paras_all_gather_tp_weights(stream)

    def paras_reshard_dptp_to_ep_peer(self, dst_base_ptrs, ep_rank, dp_size, stream):
        self.mlp.paras_reshard_dptp_to_ep_peer(dst_base_ptrs, ep_rank, dp_size, stream)

    def paras_reshard_tp_to_ep_node_peer(self, dst_base_ptrs, dp_rank, dp_size, stream):
        self.mlp.paras_reshard_tp_to_ep_node_peer(
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


def _make_model(mgr, num_local_ep, peer_ctx):
    from sglang.srt.paras.layers.paras_model import ParaSModelMixin

    model = object.__new__(ParaSModelMixin)
    model.layers = [
        _ModelLayerAdapter(_make_mixin(layer_id, num_local_ep, mgr))
        for layer_id in range(NUM_LAYERS)
    ]
    model._peer_access_ctx = peer_ctx
    return model


def setup_peer_ctx(mgr, peer_group, peer_size):
    """Exchange IPC handles over the physical node-local peer group."""
    from sglang.srt.paras.peer_access import init_peer_access

    return init_peer_access(mgr, peer_group, peer_size)


def run_forward(mgr, num_local_ep, peer_ctx, world_group):
    """EP -> DP x TP through the selected topology path."""
    dist.barrier(group=world_group)
    model = _make_model(mgr, num_local_ep, peer_ctx)
    model.paras_configure_tp(TP_SIZE, dist.get_rank() % TP_SIZE, method="direct")


def run_reverse(mgr, num_local_ep, peer_ctx, world_group):
    """DP x TP -> EP through the selected topology path."""
    dist.barrier(group=world_group)
    model = _make_model(mgr, num_local_ep, peer_ctx)
    model.paras_configure_ep(method="direct")


def read_tp_results(mgr):
    """TP w13/w2 as (num_experts, 2*I', H) / (num_experts, H, I')."""
    tp_inter = INTERMEDIATE // TP_SIZE
    out = {}
    for layer_id in range(NUM_LAYERS):
        out[layer_id] = (
            mgr.get_view_as(
                f"model.layers.{layer_id}.mlp.tp_experts.w13_weight",
                (NUM_EXPERTS, 2 * tp_inter, HIDDEN),
            ).clone(),
            mgr.get_view_as(
                f"model.layers.{layer_id}.mlp.tp_experts.w2_weight",
                (NUM_EXPERTS, HIDDEN, tp_inter),
            ).clone(),
        )
    return out


def read_ep_results(mgr):
    out = {}
    for layer_id in range(NUM_LAYERS):
        out[layer_id] = (
            mgr.get_view(f"model.layers.{layer_id}.mlp.experts.w13_weight").clone(),
            mgr.get_view(f"model.layers.{layer_id}.mlp.experts.w2_weight").clone(),
        )
    return out


def build_forward_ground_truth(snap, rank, world_group):
    """Expected DP x TP tensors on this rank after the forward switch.

    Each rank holds ALL num_experts experts, tp-sharded by tp_rank = rank % T.
    Global expert g comes from EP source rank g // E_local, local index
    g % E_local. w13 EP layout is (E_local, num_gates=2, T, I', H) flattened to
    (E_local, 2*I, H) with the gate/up concat; tp_rank selects shard
    [tp_rank*I' : (tp_rank+1)*I'] of each of gate and up. w2 EP layout is
    (E_local, H, I) and tp_rank selects columns [tp_rank*I' : (tp_rank+1)*I'].
    Replicas (same tp_rank, different dp_rank) are identical.
    """
    T = TP_SIZE
    E_local = NUM_EXPERTS // EP_SIZE
    tp_rank = rank % T
    tp_inter = INTERMEDIATE // T

    exp_w13, exp_w2 = {}, {}
    for layer_id in range(NUM_LAYERS):
        local_w13 = snap[layer_id][0]
        local_w2 = snap[layer_id][1]
        gathered_w13 = [torch.empty_like(local_w13) for _ in range(EP_SIZE)]
        gathered_w2 = [torch.empty_like(local_w2) for _ in range(EP_SIZE)]
        dist.all_gather(gathered_w13, local_w13, group=world_group)
        dist.all_gather(gathered_w2, local_w2, group=world_group)
        # Global expert g = src_rank * E_local + e, so cat along expert dim in
        # ascending source-rank order reproduces the kernel's canonical slot order.
        full_w13 = torch.cat(gathered_w13, dim=0)
        full_w2 = torch.cat(gathered_w2, dim=0)

        gate_shard = full_w13[:, tp_rank * tp_inter : (tp_rank + 1) * tp_inter, :]
        up_shard = full_w13[
            :,
            INTERMEDIATE + tp_rank * tp_inter : INTERMEDIATE + (tp_rank + 1) * tp_inter,
            :,
        ]
        exp_w13[layer_id] = torch.cat([gate_shard, up_shard], dim=1)
        exp_w2[layer_id] = full_w2[:, :, tp_rank * tp_inter : (tp_rank + 1) * tp_inter]

    return exp_w13, exp_w2, E_local


def _assert_equal(actual, expected, tag, rank, layer_id):
    a = actual.reshape(-1)
    e = expected.reshape(-1)
    if not torch.equal(a, e):
        diff = (a != e).sum().item()
        raise AssertionError(
            f"[Rank {rank}] {tag} mismatch layer={layer_id}: {diff}/{a.numel()} differ"
        )


def main():
    rank, world_size = setup_distributed()
    passed = failed = 0
    try:
        world_group, tp_group, _ = setup_paras_state(rank, world_size)
        mgr, num_local_ep = build_manager(rank, world_size)
        fill_ep_weights(mgr, rank)
        snap = snapshot_weights(mgr)
        peer_group = tp_group if LOGICAL_MULTINODE else world_group
        peer_size = TP_SIZE if LOGICAL_MULTINODE else world_size
        peer_ctx = setup_peer_ctx(mgr, peer_group, peer_size)

        # === Test 1: EP -> DP x TP forward vs independent ground truth ===
        if rank == 0:
            print("\n=== Forward EP -> DPxTP (dptp) vs ground truth ===", flush=True)
        restore_weights(mgr, snap)
        exp_w13, exp_w2, _ = build_forward_ground_truth(snap, rank, world_group)
        run_forward(mgr, num_local_ep, peer_ctx, world_group)
        actual = read_tp_results(mgr)
        try:
            for layer_id in range(NUM_LAYERS):
                _assert_equal(
                    actual[layer_id][0], exp_w13[layer_id], "w13 fwd", rank, layer_id
                )
                _assert_equal(
                    actual[layer_id][1], exp_w2[layer_id], "w2 fwd", rank, layer_id
                )
            if rank == 0:
                print(
                    "  [OK] forward dptp: bitwise match ground truth all layers",
                    flush=True,
                )
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] forward: {e}", flush=True)
            failed += 1

        # === Test 2: dp replicas identical (rank R and R+T carry same TP data) ===
        if rank == 0:
            print("\n=== DP-replica consistency ===", flush=True)
        try:
            for layer_id in range(NUM_LAYERS):
                for which in (0, 1):
                    buf = actual[layer_id][which].contiguous()
                    peer = torch.empty_like(buf)
                    partner = (
                        rank + TP_SIZE
                    ) % world_size  # same tp_rank, other dp replica
                    # exchange with dp partner
                    if rank < partner:
                        dist.send(buf, dst=partner, group=world_group)
                        dist.recv(peer, src=partner, group=world_group)
                    else:
                        dist.recv(peer, src=partner, group=world_group)
                        dist.send(buf, dst=partner, group=world_group)
                    _assert_equal(
                        buf,
                        peer,
                        f"dp-replica w{'13' if which==0 else '2'}",
                        rank,
                        layer_id,
                    )
            if rank == 0:
                print("  [OK] dp replicas bitwise identical all layers", flush=True)
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] dp-replica: {e}", flush=True)
            failed += 1

        # === Test 3: full EP -> DPxTP -> EP round trip recovers original EP ===
        if rank == 0:
            print("\n=== Round trip EP -> DPxTP -> EP (dptp) ===", flush=True)
        restore_weights(mgr, snap)
        run_forward(mgr, num_local_ep, peer_ctx, world_group)
        run_reverse(mgr, num_local_ep, peer_ctx, world_group)
        ep_actual = read_ep_results(mgr)
        try:
            for layer_id in range(NUM_LAYERS):
                _assert_equal(
                    ep_actual[layer_id][0],
                    snap[layer_id][0],
                    "w13 roundtrip",
                    rank,
                    layer_id,
                )
                _assert_equal(
                    ep_actual[layer_id][1],
                    snap[layer_id][1],
                    "w2 roundtrip",
                    rank,
                    layer_id,
                )
            if rank == 0:
                print("  [OK] round trip: EP recovered bitwise all layers", flush=True)
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] roundtrip: {e}", flush=True)
            failed += 1

        dist.barrier()
        if rank == 0:
            total = passed + failed
            print(f"\n{'=' * 60}")
            print(f"RESULTS: {passed}/{total} passed, {failed}/{total} failed")
            topology = "logical multi-node" if LOGICAL_MULTINODE else "single-node dptp"
            print(
                f"SUCCESS: dp=2 tp=2 ep=4 {topology} switch validated!"
                if failed == 0
                else "FAILED"
            )
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
