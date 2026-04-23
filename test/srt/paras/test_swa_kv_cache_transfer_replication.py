#!/usr/bin/env python3
"""
SWA KV cache transfer tests with head replication (num_kv_heads < tp_size).

Exercises SWACacheTransfer gather + scatter with NCCL and peer_access
methods under a 2-full + 4-SWA hybrid model with num_kv_heads=2,
paras_tp_size=4 (replication_factor R=2).

The SWA cap (tokens_cap_ep=8) is smaller than every rank's token count
(10..12), so truncation fires on every rank for SWA layers.

Usage:
  torchrun --nproc_per_node=4 -m pytest test/srt/paras/test_swa_kv_cache_transfer_replication.py -v
"""

import os
import sys
from typing import List

import pytest
import torch
import torch.distributed as dist

try:
    import paras_peer_access_cuda

    _HAS_PEER_ACCESS = True
except ImportError:
    _HAS_PEER_ACCESS = False

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.join(_TEST_DIR, "..", "..", "..")
sys.path.insert(0, os.path.join(_ROOT_DIR, "python"))
sys.path.insert(0, _TEST_DIR)

from test_kv_cache_transfer import (
    DTYPE,
    PAGE_SIZE,
    _is_distributed,
    _ensure_distributed,
    _setup_paras_state,
    setup_peer_ctx,
    _save_evidence,
    _SimpleGroupCoordinator,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_LAYERS = 6
FULL_LAYER_IDS = [0, 1]
SWA_LAYER_IDS = [2, 3, 4, 5]
HEAD_DIM = 64
NUM_KV_HEADS = 2
TOKENS_PER_RANK = [12, 10, 10, 10]  # total = 42; every rank > SWA cap
EP_TOKENS_CAP_FULL = 16  # cap for full layers (no truncation: max tpr=12)
EP_TOKENS_CAP_SWA = 8  # cap for SWA layers (truncation: min tpr=10 > 8)


# ---------------------------------------------------------------------------
# Pattern generator (HEAD_DIM=64 variant)
# ---------------------------------------------------------------------------


def make_swa_pattern(rank, layer, head, num_tokens):
    """Deterministic test data: unique per (rank, layer, head, token, dim).

    Returns: Tensor of shape (num_tokens, HEAD_DIM) in bfloat16.
    Both source and expected go through the same bf16 conversion, so
    torch.equal works even when large floats lose precision.
    """
    base = rank * 1000.0 + layer * 100.0 + head * 10.0
    t = torch.arange(num_tokens, dtype=torch.float32).unsqueeze(1)
    d = torch.arange(HEAD_DIM, dtype=torch.float32).unsqueeze(0) * 0.001
    return (base + t + d).to(DTYPE)


# ---------------------------------------------------------------------------
# Layer spec builder
# ---------------------------------------------------------------------------


def make_layer_specs():
    from sglang.srt.paras.cache_transfer.base import LayerCacheSpec

    specs = []
    for i in range(NUM_LAYERS):
        if i in FULL_LAYER_IDS:
            specs.append(
                LayerCacheSpec(
                    layer_id=i,
                    kind="full",
                    tokens_cap_ep=EP_TOKENS_CAP_FULL,
                    tokens_cap_tp=0,
                    num_kv_heads=NUM_KV_HEADS,
                    head_dim=HEAD_DIM,
                    sliding_window_size=None,
                )
            )
        else:
            specs.append(
                LayerCacheSpec(
                    layer_id=i,
                    kind="swa",
                    tokens_cap_ep=EP_TOKENS_CAP_SWA,
                    tokens_cap_tp=0,
                    num_kv_heads=NUM_KV_HEADS,
                    head_dim=HEAD_DIM,
                    sliding_window_size=1023,
                )
            )
    return specs


# ---------------------------------------------------------------------------
# Memory manager + SWAKVPool setup
# ---------------------------------------------------------------------------


def setup_mgr_and_pool(rank, world_size):
    """Create ParaSMemoryManager (uniform layout) and SWAKVPool.

    Uses a uniform buffer (no layer_specs in reserve_kv_cache) so that
    every layer gets the same large buffer.  This avoids the TP buffer
    under-sizing issue that occurs when layer_specs shrinks SWA layers
    below the space needed for replication gather.

    The SWA token cap is enforced by the layer_specs passed to the
    cache transfer backends, not by the buffer size.
    """
    from sglang.srt.mem_cache.memory_pool import SWAKVPool
    from sglang.srt.paras.paras_memory_manager import (
        ParaSMemoryManager,
        set_global_paras_memory_manager,
    )

    set_global_paras_memory_manager(None)
    device = f"cuda:{rank}"

    # Uniform buffer: large enough for all tokens per rank AND for the
    # TP gather output (total_tokens across all ranks, with head sharding).
    # With num_kv_heads=2, heads_per_rank=1, the TP view has
    # (ep_max+1)*2 token slots.  We need >= sum(TOKENS_PER_RANK)=42.
    # So ep_max >= 42/2 - 1 = 20.  Use 22 for margin.
    ep_max_tokens = max(max(TOKENS_PER_RANK) + 10, sum(TOKENS_PER_RANK) // NUM_KV_HEADS)

    mgr = ParaSMemoryManager(device=device)
    mgr.reserve_kv_cache(
        num_layers=NUM_LAYERS,
        ep_max_tokens=ep_max_tokens,
        tp_max_tokens=0,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        kv_dtype=DTYPE,
        page_size=PAGE_SIZE,
    )
    mgr.materialize()
    set_global_paras_memory_manager(mgr)

    # Get EP views as external buffers for SWAKVPool
    full_ep_k, full_ep_v = mgr.get_kv_views(
        num_layers=len(FULL_LAYER_IDS),
        mode="ep",
        layer_ids=FULL_LAYER_IDS,
    )
    swa_ep_k, swa_ep_v = mgr.get_kv_views(
        num_layers=len(SWA_LAYER_IDS),
        mode="ep",
        layer_ids=SWA_LAYER_IDS,
    )

    pool = SWAKVPool(
        size=ep_max_tokens + PAGE_SIZE,
        size_swa=ep_max_tokens + PAGE_SIZE,
        dtype=DTYPE,
        head_num=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        swa_attention_layer_ids=SWA_LAYER_IDS,
        full_attention_layer_ids=FULL_LAYER_IDS,
        enable_kvcache_transpose=False,
        device=device,
        full_external_k_buffers=full_ep_k,
        full_external_v_buffers=full_ep_v,
        swa_external_k_buffers=swa_ep_k,
        swa_external_v_buffers=swa_ep_v,
    )

    return mgr, pool, ep_max_tokens


def switch_pool_to_tp(pool, mgr, world_size):
    """Switch SWAKVPool to TP mode with correct head sharding for R=2.

    Manually rebinds buffers to TP views with heads_per_rank=1.
    Cannot use pool.paras_configure_tp because it does plain integer
    division (head_num // tp_size = 2 // 4 = 0).
    """
    heads_per_rank = max(1, NUM_KV_HEADS // world_size)
    pool.full_head_num = pool.head_num
    pool.head_num = heads_per_rank
    pool.full_kv_pool.head_num = heads_per_rank
    pool.swa_kv_pool.head_num = heads_per_rank

    for i, lid in enumerate(FULL_LAYER_IDS):
        entry = mgr._entries[f"model.layers.{lid}.kv.tp.k"]
        tp_tokens = entry.numel // (heads_per_rank * HEAD_DIM)
        tp_shape = (tp_tokens, heads_per_rank, HEAD_DIM)
        pool.full_kv_pool.k_buffer[i] = mgr.get_view_as(
            f"model.layers.{lid}.kv.tp.k", tp_shape
        )
        pool.full_kv_pool.v_buffer[i] = mgr.get_view_as(
            f"model.layers.{lid}.kv.tp.v", tp_shape
        )

    for i, lid in enumerate(SWA_LAYER_IDS):
        entry = mgr._entries[f"model.layers.{lid}.kv.tp.k"]
        tp_tokens = entry.numel // (heads_per_rank * HEAD_DIM)
        tp_shape = (tp_tokens, heads_per_rank, HEAD_DIM)
        pool.swa_kv_pool.k_buffer[i] = mgr.get_view_as(
            f"model.layers.{lid}.kv.tp.k", tp_shape
        )
        pool.swa_kv_pool.v_buffer[i] = mgr.get_view_as(
            f"model.layers.{lid}.kv.tp.v", tp_shape
        )


def switch_pool_to_ep(pool, mgr):
    """Switch SWAKVPool back to EP mode."""
    pool.head_num = pool.full_head_num
    pool.full_kv_pool.head_num = pool.full_head_num
    pool.swa_kv_pool.head_num = pool.full_head_num

    for i, lid in enumerate(FULL_LAYER_IDS):
        pool.full_kv_pool.k_buffer[i] = mgr.get_view(
            f"model.layers.{lid}.kv.ep.k"
        )
        pool.full_kv_pool.v_buffer[i] = mgr.get_view(
            f"model.layers.{lid}.kv.ep.v"
        )

    for i, lid in enumerate(SWA_LAYER_IDS):
        pool.swa_kv_pool.k_buffer[i] = mgr.get_view(
            f"model.layers.{lid}.kv.ep.k"
        )
        pool.swa_kv_pool.v_buffer[i] = mgr.get_view(
            f"model.layers.{lid}.kv.ep.v"
        )


# ---------------------------------------------------------------------------
# Fill helpers
# ---------------------------------------------------------------------------


def fill_ep_kv(mgr, rank):
    """Fill EP KV buffers with deterministic patterns.

    K uses make_swa_pattern(rank, layer, head, n).
    V uses make_swa_pattern(rank, layer, head + 50, n).
    """
    num_tokens = TOKENS_PER_RANK[rank]
    device = f"cuda:{rank}"
    for lid in range(NUM_LAYERS):
        ep_k = mgr.get_view(f"model.layers.{lid}.kv.ep.k")
        ep_v = mgr.get_view(f"model.layers.{lid}.kv.ep.v")
        ep_k.zero_()
        ep_v.zero_()
        for h in range(NUM_KV_HEADS):
            ep_k[:num_tokens, h, :] = make_swa_pattern(
                rank, lid, h, num_tokens
            ).to(device)
            ep_v[:num_tokens, h, :] = make_swa_pattern(
                rank, lid, h + 50, num_tokens
            ).to(device)


def fill_tp_kv(mgr, rank, world_size):
    """Fill TP KV buffers with deterministic patterns.

    K uses make_swa_pattern(rank, layer, local_head, total_tokens).
    V uses make_swa_pattern(rank, layer, local_head + 50, total_tokens).
    """
    heads_per_rank = max(1, NUM_KV_HEADS // world_size)
    total_tokens = sum(TOKENS_PER_RANK)
    device = f"cuda:{rank}"
    for lid in range(NUM_LAYERS):
        entry = mgr._entries[f"model.layers.{lid}.kv.tp.k"]
        tp_tokens = entry.numel // (heads_per_rank * HEAD_DIM)
        tp_k = mgr.get_view_as(
            f"model.layers.{lid}.kv.tp.k",
            (tp_tokens, heads_per_rank, HEAD_DIM),
        )
        tp_v = mgr.get_view_as(
            f"model.layers.{lid}.kv.tp.v",
            (tp_tokens, heads_per_rank, HEAD_DIM),
        )
        tp_k.zero_()
        tp_v.zero_()
        for lh in range(heads_per_rank):
            tp_k[:total_tokens, lh, :] = make_swa_pattern(
                rank, lid, lh, total_tokens
            ).to(device)
            tp_v[:total_tokens, lh, :] = make_swa_pattern(
                rank, lid, lh + 50, total_tokens
            ).to(device)


# ---------------------------------------------------------------------------
# Gather (EP → TP)
# ---------------------------------------------------------------------------


def do_swa_gather(mgr, pool, rank, world_size, tp_group, specs,
                  method="nccl", peer_ctx=None):
    """EP→TP gather using MHA backend for full layers, SWA for SWA layers."""
    from sglang.srt.paras.cache_transfer.mha import MHACacheTransfer
    from sglang.srt.paras.cache_transfer.swa import SWACacheTransfer

    num_local = TOKENS_PER_RANK[rank]
    total_tokens = sum(TOKENS_PER_RANK)
    global_num_tokens = list(TOKENS_PER_RANK)

    local_token_indices = torch.arange(
        num_local, dtype=torch.int64, device="cuda"
    )
    global_token_indices = torch.arange(
        total_tokens, dtype=torch.int64, device="cuda"
    )

    peer_addresses = peer_ctx.peer_addresses if peer_ctx else None
    group_coord = _SimpleGroupCoordinator(
        tp_group, world_size, f"cuda:{rank}", rank
    )

    mha_backend = MHACacheTransfer(
        method=method,
        direction="gather",
        kv_cache=pool,
        mgr=mgr,
        group=group_coord,
        num_local_tokens=num_local,
        num_global_tokens=total_tokens,
        local_token_indices=local_token_indices,
        global_token_indices=global_token_indices,
        global_num_tokens=global_num_tokens,
        peer_addresses=peer_addresses,
    )

    swa_backend = SWACacheTransfer(
        method=method,
        direction="gather",
        kv_cache=pool,
        mgr=mgr,
        group=group_coord,
        num_local_tokens=num_local,
        num_global_tokens=total_tokens,
        local_token_indices=local_token_indices,
        global_token_indices=global_token_indices,
        global_num_tokens=global_num_tokens,
        layer_specs=specs,
        peer_addresses=peer_addresses,
    )

    barrier_tensor = torch.zeros(1, device="cuda")

    for spec in specs:
        backend = swa_backend if spec.kind == "swa" else mha_backend
        backend.gather_one_layer(spec)
        dist.all_reduce(barrier_tensor, group=tp_group)

    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Scatter (TP → EP)
# ---------------------------------------------------------------------------


def do_swa_scatter(mgr, pool, rank, world_size, tp_group, specs,
                   method="nccl", peer_ctx=None):
    """TP→EP scatter using MHA backend for full layers, SWA for SWA layers."""
    from sglang.srt.paras.cache_transfer.mha import MHACacheTransfer
    from sglang.srt.paras.cache_transfer.swa import SWACacheTransfer

    total_tokens = sum(TOKENS_PER_RANK)
    global_token_indices = torch.arange(
        total_tokens, dtype=torch.int64, device="cuda"
    )

    # Contiguous token partition
    token_partition: List[List[int]] = []
    offset = 0
    for r in range(world_size):
        n = TOKENS_PER_RANK[r]
        token_partition.append(list(range(offset, offset + n)))
        offset += n

    ep_dst_positions = torch.arange(
        TOKENS_PER_RANK[rank], dtype=torch.int64, device="cuda"
    )

    peer_addresses = peer_ctx.peer_addresses if peer_ctx else None
    group_coord = _SimpleGroupCoordinator(
        tp_group, world_size, f"cuda:{rank}", rank
    )

    mha_backend = MHACacheTransfer(
        method=method,
        direction="scatter",
        kv_cache=pool,
        mgr=mgr,
        group=group_coord,
        global_token_indices=global_token_indices,
        peer_addresses=peer_addresses,
        ep_head_num=NUM_KV_HEADS,
        token_partition=token_partition,
        ep_dst_positions=ep_dst_positions,
        paras_tp_rank=rank,
        paras_tp_size=world_size,
    )

    swa_backend = SWACacheTransfer(
        method=method,
        direction="scatter",
        kv_cache=pool,
        mgr=mgr,
        group=group_coord,
        global_token_indices=global_token_indices,
        layer_specs=specs,
        peer_addresses=peer_addresses,
        ep_head_num=NUM_KV_HEADS,
        token_partition=token_partition,
        ep_dst_positions=ep_dst_positions,
        paras_tp_rank=rank,
        paras_tp_size=world_size,
    )

    barrier_tensor = torch.zeros(1, device="cuda")

    for spec in reversed(specs):
        backend = swa_backend if spec.kind == "swa" else mha_backend
        backend.scatter_one_layer(spec)
        dist.all_reduce(barrier_tensor, group=tp_group)

    torch.cuda.synchronize()
    return token_partition


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------


def verify_gather(mgr, rank, world_size, specs):
    """Verify EP→TP gather result against ground truth patterns.

    For full layers: all tokens transferred, head-sharded under R=2.
    For SWA layers: only first tokens_cap_ep tokens per source rank.
    Returns True if all checks pass.
    """
    R = max(1, world_size // NUM_KV_HEADS)
    real_head = rank // R
    heads_per_rank = max(1, NUM_KV_HEADS // world_size)
    device = f"cuda:{rank}"

    all_ok = True
    for spec in specs:
        lid = spec.layer_id
        entry = mgr._entries[f"model.layers.{lid}.kv.tp.k"]
        tp_tokens = entry.numel // (heads_per_rank * HEAD_DIM)
        tp_k = mgr.get_view_as(
            f"model.layers.{lid}.kv.tp.k",
            (tp_tokens, heads_per_rank, HEAD_DIM),
        )
        tp_v = mgr.get_view_as(
            f"model.layers.{lid}.kv.tp.v",
            (tp_tokens, heads_per_rank, HEAD_DIM),
        )

        global_offset = 0
        for src in range(world_size):
            if spec.kind == "swa":
                n = min(TOKENS_PER_RANK[src], spec.tokens_cap_ep)
            else:
                n = TOKENS_PER_RANK[src]

            expected_k = make_swa_pattern(src, lid, real_head, n).to(device)
            expected_v = make_swa_pattern(src, lid, real_head + 50, n).to(device)
            actual_k = tp_k[global_offset : global_offset + n, 0, :]
            actual_v = tp_v[global_offset : global_offset + n, 0, :]

            if not torch.equal(actual_k, expected_k):
                all_ok = False
                if rank == 0:
                    diff = (actual_k.float() - expected_k.float()).abs().max()
                    print(
                        f"  K mismatch L{lid}({spec.kind}) src={src}: "
                        f"max_diff={diff}"
                    )
            if not torch.equal(actual_v, expected_v):
                all_ok = False
                if rank == 0:
                    diff = (actual_v.float() - expected_v.float()).abs().max()
                    print(
                        f"  V mismatch L{lid}({spec.kind}) src={src}: "
                        f"max_diff={diff}"
                    )
            global_offset += TOKENS_PER_RANK[src]

    return all_ok


def verify_scatter(mgr, rank, world_size, specs, token_partition,
                   dst_offset=0):
    """Verify TP→EP scatter result against ground truth patterns.

    dst_offset: 0 for NCCL (0-indexed), 1 for peer_access (slot 0 = padding).
    Returns True if all checks pass.
    """
    total_tokens = sum(TOKENS_PER_RANK)
    R = max(1, world_size // NUM_KV_HEADS)
    my_tokens = token_partition[rank]
    count = len(my_tokens)
    device = f"cuda:{rank}"

    all_ok = True
    for spec in specs:
        lid = spec.layer_id
        ep_k = mgr.get_view(f"model.layers.{lid}.kv.ep.k")
        ep_v = mgr.get_view(f"model.layers.{lid}.kv.ep.v")

        if spec.kind == "swa":
            check_count = min(count, spec.tokens_cap_ep)
        else:
            check_count = count

        for h in range(NUM_KV_HEADS):
            for ir in range(R):
                start = check_count * ir // R
                end = check_count * (ir + 1) // R
                if start == end:
                    continue
                src_tp_rank = h * R + ir

                full_k = make_swa_pattern(
                    src_tp_rank, lid, 0, total_tokens
                ).to(device)
                full_v = make_swa_pattern(
                    src_tp_rank, lid, 50, total_tokens
                ).to(device)

                global_indices = torch.tensor(
                    my_tokens[start:end], dtype=torch.long, device=device
                )
                expected_k = full_k[global_indices]
                expected_v = full_v[global_indices]
                ep_start = start + dst_offset
                ep_end = end + dst_offset
                actual_k = ep_k[ep_start:ep_end, h, :]
                actual_v = ep_v[ep_start:ep_end, h, :]

                if not torch.equal(actual_k, expected_k):
                    all_ok = False
                    if rank == 0:
                        diff = (
                            actual_k.float() - expected_k.float()
                        ).abs().max()
                        print(
                            f"  K mismatch L{lid}({spec.kind}) h={h} "
                            f"ir={ir}: max_diff={diff}"
                        )
                if not torch.equal(actual_v, expected_v):
                    all_ok = False
                    if rank == 0:
                        diff = (
                            actual_v.float() - expected_v.float()
                        ).abs().max()
                        print(
                            f"  V mismatch L{lid}({spec.kind}) h={h} "
                            f"ir={ir}: max_diff={diff}"
                        )

    return all_ok


# =========================================================================
# EP→TP with SWA + Replication
# =========================================================================


@pytest.mark.skipif(not _is_distributed(), reason="Requires torchrun")
class TestEPtoTPSWAReplication:
    """EP→TP gather with SWA capping and head replication (R=2)."""

    def test_ep_to_tp_nccl_swa_replication(self):
        """NCCL gather: full layers transfer all tokens, SWA layers cap."""
        rank, world_size = _ensure_distributed()
        specs = make_layer_specs()
        tp_group = _setup_paras_state(rank, world_size)
        mgr, pool, _ = setup_mgr_and_pool(rank, world_size)

        fill_ep_kv(mgr, rank)
        do_swa_gather(mgr, pool, rank, world_size, tp_group, specs,
                      method="nccl")
        dist.barrier(group=tp_group)

        all_ok = verify_gather(mgr, rank, world_size, specs)
        _save_evidence("ep_to_tp_swa_replication_nccl", all_ok, rank)
        R = world_size // NUM_KV_HEADS
        assert all_ok, (
            f"EP→TP NCCL SWA (R={R}) failed on rank {rank}"
        )

    @pytest.mark.skipif(
        not _HAS_PEER_ACCESS, reason="paras_peer_access_cuda not available"
    )
    def test_ep_to_tp_peer_access_swa_replication(self):
        """Peer_access gather: full layers transfer all, SWA layers cap."""
        rank, world_size = _ensure_distributed()
        specs = make_layer_specs()
        tp_group = _setup_paras_state(rank, world_size)
        mgr, pool, _ = setup_mgr_and_pool(rank, world_size)
        peer_ctx = setup_peer_ctx(mgr, rank, world_size, tp_group)

        fill_ep_kv(mgr, rank)
        do_swa_gather(mgr, pool, rank, world_size, tp_group, specs,
                      method="peer_access", peer_ctx=peer_ctx)
        dist.barrier(group=tp_group)

        all_ok = verify_gather(mgr, rank, world_size, specs)
        _save_evidence("ep_to_tp_swa_replication_peer_access", all_ok, rank)
        R = world_size // NUM_KV_HEADS
        assert all_ok, (
            f"EP→TP peer_access SWA (R={R}) failed on rank {rank}"
        )


# =========================================================================
# TP→EP with SWA + Replication
# =========================================================================


@pytest.mark.skipif(not _is_distributed(), reason="Requires torchrun")
class TestTPtoEPSWAReplication:
    """TP→EP scatter with SWA capping and head replication (R=2)."""

    def test_tp_to_ep_nccl_swa_replication(self):
        """NCCL scatter: full layers write all, SWA layers write capped."""
        rank, world_size = _ensure_distributed()
        specs = make_layer_specs()
        tp_group = _setup_paras_state(rank, world_size)
        mgr, pool, _ = setup_mgr_and_pool(rank, world_size)

        # Switch pool to TP mode for scatter (source = TP buffers)
        switch_pool_to_tp(pool, mgr, world_size)
        fill_tp_kv(mgr, rank, world_size)

        token_partition = do_swa_scatter(
            mgr, pool, rank, world_size, tp_group, specs, method="nccl"
        )

        # Switch back to EP mode for verification
        switch_pool_to_ep(pool, mgr)

        all_ok = verify_scatter(mgr, rank, world_size, specs, token_partition)
        _save_evidence("tp_to_ep_swa_replication_nccl", all_ok, rank)
        R = world_size // NUM_KV_HEADS
        assert all_ok, (
            f"TP→EP NCCL SWA (R={R}) failed on rank {rank}"
        )

    @pytest.mark.skipif(
        not _HAS_PEER_ACCESS, reason="paras_peer_access_cuda not available"
    )
    def test_tp_to_ep_peer_access_swa_replication(self):
        """Peer_access scatter: full layers write all, SWA write capped."""
        rank, world_size = _ensure_distributed()
        specs = make_layer_specs()
        tp_group = _setup_paras_state(rank, world_size)
        mgr, pool, _ = setup_mgr_and_pool(rank, world_size)
        peer_ctx = setup_peer_ctx(mgr, rank, world_size, tp_group)

        switch_pool_to_tp(pool, mgr, world_size)
        fill_tp_kv(mgr, rank, world_size)

        token_partition = do_swa_scatter(
            mgr, pool, rank, world_size, tp_group, specs,
            method="peer_access", peer_ctx=peer_ctx,
        )

        switch_pool_to_ep(pool, mgr)

        all_ok = verify_scatter(
            mgr, rank, world_size, specs, token_partition, dst_offset=1,
        )
        _save_evidence("tp_to_ep_swa_replication_peer_access", all_ok, rank)
        R = world_size // NUM_KV_HEADS
        assert all_ok, (
            f"TP→EP peer_access SWA (R={R}) failed on rank {rank}"
        )


# =========================================================================
# Round-trip with SWA + Replication
# =========================================================================


@pytest.mark.skipif(not _is_distributed(), reason="Requires torchrun")
class TestSWAReplicationRoundtrip:
    """EP→TP→EP round-trip with SWA capping and head replication."""

    def test_swa_replication_roundtrip(self):
        """Snapshot EP → gather → scatter → compare capped region."""
        rank, world_size = _ensure_distributed()
        specs = make_layer_specs()
        tp_group = _setup_paras_state(rank, world_size)
        mgr, pool, _ = setup_mgr_and_pool(rank, world_size)

        fill_ep_kv(mgr, rank)

        # Snapshot EP data (capped per layer)
        num_local = TOKENS_PER_RANK[rank]
        orig = {}
        for spec in specs:
            lid = spec.layer_id
            if spec.kind == "swa":
                cap = min(num_local, spec.tokens_cap_ep)
            else:
                cap = num_local
            ep_k = mgr.get_view(f"model.layers.{lid}.kv.ep.k")
            ep_v = mgr.get_view(f"model.layers.{lid}.kv.ep.v")
            orig[lid] = (ep_k[:cap].clone(), ep_v[:cap].clone())

        # EP → TP gather
        do_swa_gather(mgr, pool, rank, world_size, tp_group, specs,
                      method="nccl")
        dist.barrier(group=tp_group)

        # Switch to TP mode for scatter
        switch_pool_to_tp(pool, mgr, world_size)

        # TP → EP scatter
        do_swa_scatter(mgr, pool, rank, world_size, tp_group, specs,
                       method="nccl")

        # Switch back to EP for verification
        switch_pool_to_ep(pool, mgr)

        # Compare capped region
        all_ok = True
        for spec in specs:
            lid = spec.layer_id
            if spec.kind == "swa":
                cap = min(num_local, spec.tokens_cap_ep)
            else:
                cap = num_local
            ep_k = mgr.get_view(f"model.layers.{lid}.kv.ep.k")
            ep_v = mgr.get_view(f"model.layers.{lid}.kv.ep.v")
            if not torch.equal(orig[lid][0], ep_k[:cap]):
                all_ok = False
                if rank == 0:
                    diff = (
                        orig[lid][0].float() - ep_k[:cap].float()
                    ).abs().max()
                    print(
                        f"  K round-trip mismatch L{lid}({spec.kind}): "
                        f"max_diff={diff}"
                    )
            if not torch.equal(orig[lid][1], ep_v[:cap]):
                all_ok = False
                if rank == 0:
                    diff = (
                        orig[lid][1].float() - ep_v[:cap].float()
                    ).abs().max()
                    print(
                        f"  V round-trip mismatch L{lid}({spec.kind}): "
                        f"max_diff={diff}"
                    )

        _save_evidence("swa_replication_roundtrip", all_ok, rank)
        R = world_size // NUM_KV_HEADS
        assert all_ok, (
            f"SWA round-trip (R={R}) failed on rank {rank}"
        )
