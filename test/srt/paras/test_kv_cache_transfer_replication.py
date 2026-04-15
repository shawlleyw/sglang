#!/usr/bin/env python3
"""
KV cache transfer tests with head replication (num_kv_heads < tp_size).

Separated from test_kv_cache_transfer.py to avoid test isolation issues
with CUDA IPC handle invalidation when reallocating buffers with different
num_kv_heads in the same process.

Usage:
  torchrun --nproc_per_node=4 -m pytest test/srt/paras/test_kv_cache_transfer_replication.py -v
  torchrun --nproc_per_node=8 -m pytest test/srt/paras/test_kv_cache_transfer_replication.py -v
"""

import os
import sys

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

# Import shared helpers from the non-replication test file
sys.path.insert(0, _TEST_DIR)
from test_kv_cache_transfer import (
    NUM_LAYERS,
    HEAD_DIM,
    DTYPE,
    PAGE_SIZE,
    _tokens_for_world,
    _is_distributed,
    _ensure_distributed,
    _setup_paras_state,
    setup_peer_ctx,
    setup_memory_manager,
    fill_ep_kv,
    fill_tp_kv,
    do_ep_to_tp_gather,
    do_tp_to_ep_scatter,
    do_ep_to_tp_gather_peer_access,
    do_tp_to_ep_scatter_peer_access,
    verify_ep_to_tp,
    verify_tp_to_ep,
    _save_evidence,
)


# =========================================================================
# EP→TP with Replication
# =========================================================================


@pytest.mark.skipif(not _is_distributed(), reason="Requires torchrun")
class TestEPtoTPReplication:
    """EP→TP gather with head replication (num_kv_heads < tp_size)."""

    def test_ep_to_tp_nccl(self):
        """NCCL gather with replication, verified against pattern ground truth."""
        rank, world_size = _ensure_distributed()
        num_kv_heads = world_size // 2
        tokens_per_rank = _tokens_for_world(world_size)

        tp_group = _setup_paras_state(rank, world_size)
        mgr, ep_max, _ = setup_memory_manager(
            rank, world_size, num_kv_heads, tokens_per_rank
        )
        fill_ep_kv(mgr, rank, num_kv_heads, tokens_per_rank)
        tp_view_tokens = do_ep_to_tp_gather(
            mgr, rank, world_size, num_kv_heads, tokens_per_rank,
            ep_max, tp_group,
        )
        dist.barrier(group=tp_group)

        all_ok = verify_ep_to_tp(
            mgr, rank, world_size, num_kv_heads, tokens_per_rank,
            tp_view_tokens,
        )
        _save_evidence("ep_to_tp_replication_nccl", all_ok, rank)
        assert all_ok, f"EP→TP NCCL (R={world_size // num_kv_heads}) failed on rank {rank}"

    @pytest.mark.skipif(
        not _HAS_PEER_ACCESS, reason="paras_peer_access_cuda not available"
    )
    def test_ep_to_tp_peer_access(self):
        """Peer_access gather with replication, verified against pattern ground truth."""
        rank, world_size = _ensure_distributed()
        num_kv_heads = world_size // 2
        tokens_per_rank = _tokens_for_world(world_size)

        tp_group = _setup_paras_state(rank, world_size)
        mgr, ep_max, _ = setup_memory_manager(
            rank, world_size, num_kv_heads, tokens_per_rank
        )
        peer_ctx = setup_peer_ctx(mgr, rank, world_size, tp_group)

        fill_ep_kv(mgr, rank, num_kv_heads, tokens_per_rank)
        tp_view_tokens = do_ep_to_tp_gather_peer_access(
            mgr, rank, world_size, num_kv_heads, tokens_per_rank,
            ep_max, tp_group, peer_ctx,
        )
        dist.barrier(group=tp_group)

        R = world_size // num_kv_heads
        if rank == 0:
            print(
                f"\n  EP→TP peer_access replication: {num_kv_heads} heads / "
                f"{world_size} GPUs → R={R}"
            )

        all_ok = verify_ep_to_tp(
            mgr, rank, world_size, num_kv_heads, tokens_per_rank,
            tp_view_tokens,
        )
        _save_evidence("ep_to_tp_replication_peer_access", all_ok, rank)
        assert all_ok, f"EP→TP peer_access (R={R}) failed on rank {rank}"


# =========================================================================
# TP→EP with Replication
# =========================================================================


@pytest.mark.skipif(not _is_distributed(), reason="Requires torchrun")
class TestTPtoEPReplication:
    """TP→EP scatter with head replication (num_kv_heads < tp_size)."""

    def test_tp_to_ep_nccl(self):
        """NCCL scatter with replication, verified against pattern ground truth."""
        rank, world_size = _ensure_distributed()
        num_kv_heads = world_size // 2
        tokens_per_rank = _tokens_for_world(world_size)
        total_tokens = sum(tokens_per_rank)

        tp_group = _setup_paras_state(rank, world_size)
        mgr, ep_max, _ = setup_memory_manager(
            rank, world_size, num_kv_heads, tokens_per_rank
        )
        heads_per_rank = max(1, num_kv_heads // world_size)
        tp_view_tokens = (
            (ep_max + PAGE_SIZE) * num_kv_heads // heads_per_rank
        )

        fill_tp_kv(
            mgr, rank, world_size, num_kv_heads, total_tokens, tp_view_tokens
        )
        token_partition, _ = do_tp_to_ep_scatter(
            mgr, rank, world_size, num_kv_heads, tokens_per_rank,
            tp_view_tokens, tp_group,
        )

        all_ok = verify_tp_to_ep(
            mgr, rank, world_size, num_kv_heads, tokens_per_rank,
            token_partition,
        )
        _save_evidence("tp_to_ep_replication_nccl", all_ok, rank)
        R = world_size // num_kv_heads
        assert all_ok, f"TP→EP NCCL (R={R}) failed on rank {rank}"

    @pytest.mark.skipif(
        not _HAS_PEER_ACCESS, reason="paras_peer_access_cuda not available"
    )
    def test_tp_to_ep_peer_access(self):
        """Peer_access scatter with replication, verified against pattern ground truth."""
        rank, world_size = _ensure_distributed()
        num_kv_heads = world_size // 2
        tokens_per_rank = _tokens_for_world(world_size)
        total_tokens = sum(tokens_per_rank)

        tp_group = _setup_paras_state(rank, world_size)
        mgr, ep_max, _ = setup_memory_manager(
            rank, world_size, num_kv_heads, tokens_per_rank
        )
        peer_ctx = setup_peer_ctx(mgr, rank, world_size, tp_group)

        heads_per_rank = max(1, num_kv_heads // world_size)
        tp_view_tokens = (
            (ep_max + PAGE_SIZE) * num_kv_heads // heads_per_rank
        )
        R = world_size // num_kv_heads

        fill_tp_kv(
            mgr, rank, world_size, num_kv_heads, total_tokens, tp_view_tokens
        )
        token_partition, _ = do_tp_to_ep_scatter_peer_access(
            mgr, rank, world_size, num_kv_heads, tokens_per_rank,
            tp_view_tokens, tp_group, peer_ctx,
        )

        if rank == 0:
            print(
                f"\n  TP→EP peer_access replication: {num_kv_heads} heads / "
                f"{world_size} GPUs → R={R}"
            )

        all_ok = verify_tp_to_ep(
            mgr, rank, world_size, num_kv_heads, tokens_per_rank,
            token_partition,
        )
        _save_evidence("tp_to_ep_replication_peer_access", all_ok, rank)
        assert all_ok, f"TP→EP peer_access (R={R}) failed on rank {rank}"


# =========================================================================
# Round-trip with Replication
# =========================================================================


@pytest.mark.skipif(not _is_distributed(), reason="Requires torchrun")
class TestKVRoundTripReplication:
    """EP→TP→EP round-trip with head replication."""

    def test_roundtrip(self):
        """EP→TP→EP with R=2: snapshot → gather → scatter → compare."""
        rank, world_size = _ensure_distributed()
        num_kv_heads = world_size // 2
        tokens_per_rank = _tokens_for_world(world_size)

        tp_group = _setup_paras_state(rank, world_size)
        mgr, ep_max, _ = setup_memory_manager(
            rank, world_size, num_kv_heads, tokens_per_rank
        )
        heads_per_rank = max(1, num_kv_heads // world_size)
        tp_view_tokens = (
            (ep_max + PAGE_SIZE) * num_kv_heads // heads_per_rank
        )

        fill_ep_kv(mgr, rank, num_kv_heads, tokens_per_rank)

        # Snapshot original EP data
        num_local = tokens_per_rank[rank]
        orig = {}
        for lid in range(NUM_LAYERS):
            ep_k = mgr.get_view(f"model.layers.{lid}.kv.ep.k")
            ep_v = mgr.get_view(f"model.layers.{lid}.kv.ep.v")
            orig[lid] = (ep_k[:num_local].clone(), ep_v[:num_local].clone())

        # EP→TP gather
        do_ep_to_tp_gather(
            mgr, rank, world_size, num_kv_heads, tokens_per_rank,
            ep_max, tp_group,
        )
        dist.barrier(group=tp_group)

        # TP→EP scatter (writes back to EP slots)
        do_tp_to_ep_scatter(
            mgr, rank, world_size, num_kv_heads, tokens_per_rank,
            tp_view_tokens, tp_group,
        )

        # Compare
        all_ok = True
        for lid in range(NUM_LAYERS):
            ep_tokens = (
                mgr._entries[f"model.layers.{lid}.kv.ep.k"].numel
                // (num_kv_heads * HEAD_DIM)
            )
            ep_k = mgr.get_view_as(
                f"model.layers.{lid}.kv.ep.k",
                (ep_tokens, num_kv_heads, HEAD_DIM),
            )
            ep_v = mgr.get_view_as(
                f"model.layers.{lid}.kv.ep.v",
                (ep_tokens, num_kv_heads, HEAD_DIM),
            )
            if not torch.equal(orig[lid][0], ep_k[:num_local]):
                all_ok = False
            if not torch.equal(orig[lid][1], ep_v[:num_local]):
                all_ok = False

        _save_evidence("roundtrip_replication", all_ok, rank)
        R = world_size // num_kv_heads
        assert all_ok, f"Round-trip (R={R}) failed on rank {rank}"
