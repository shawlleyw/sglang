"""Production MHA scatter with UMM offsets and allocator-valid (nonzero) slots.

torchrun --standalone --nproc_per_node=8 -m pytest -q <this file>
Set PARAS_TEST_KV_HEADS=4 or 16 for replicated or multiple local heads; run each
setting in fresh workers, so CUDA IPC handles cannot outlive a replaced arena.
"""

import os

import pytest
import torch
import torch.distributed as dist

from test.srt.paras.test_kv_cache_transfer import (
    DTYPE,
    HEAD_DIM,
    NUM_LAYERS,
    _MockKVCache,
    _SimpleGroupCoordinator,
    _ensure_distributed,
    _setup_paras_state,
    _tokens_for_world,
    setup_memory_manager,
    setup_peer_ctx,
)


def pattern(rank, layer, kv, capacity, heads):
    elements = torch.arange(capacity * heads * HEAD_DIM)
    return (
        ((elements + rank * 19 + layer * 7 + kv * 31) % 127)
        .to(DTYPE)
        .view(capacity, heads, HEAD_DIM)
    )


@pytest.mark.skipif("RANK" not in os.environ, reason="Requires torchrun")
def test_production_scatter():
    from sglang.srt.paras.cache_transfer.mha import MHACacheTransfer
    from sglang.srt.paras.layers.utils import LayerCacheSpec

    rank, world = _ensure_distributed()
    heads = int(os.environ.get("PARAS_TEST_KV_HEADS", world))
    hpr, replication = max(heads // world, 1), max(world // heads, 1)
    counts = _tokens_for_world(world)
    counts[1] = 0
    total = sum(counts)
    group = _setup_paras_state(rank, world)
    mgr, _, _ = setup_memory_manager(rank, world, heads, counts)
    peers = setup_peer_ctx(mgr, rank, world, group)
    capacity = mgr.get_view("model.layers.0.kv.tp.k").numel() // (hpr * HEAD_DIM)
    generator = torch.Generator().manual_seed(171)
    physical_slots = torch.randperm(total, generator=generator) + 1
    logical_tokens = torch.randperm(total, generator=generator)
    partitions = [part.tolist() for part in logical_tokens.split(counts)]
    for layer in range(NUM_LAYERS):
        for kv, name in enumerate(("k", "v")):
            mgr.get_view(f"model.layers.{layer}.kv.tp.{name}").view(
                capacity, hpr, HEAD_DIM
            ).copy_(pattern(rank, layer, kv, capacity, hpr))

    backend = MHACacheTransfer(
        method="peer_access",
        direction="scatter",
        kv_cache=_MockKVCache(
            mgr,
            hpr,
            HEAD_DIM,
            NUM_LAYERS,
            DTYPE,
            f"cuda:{rank}",
            "tp",
            view_tokens=capacity,
        ),
        mgr=mgr,
        group=_SimpleGroupCoordinator(group, world, f"cuda:{rank}", rank),
        global_token_indices=physical_slots.to("cuda"),
        peer_addresses=peers.peer_addresses,
        ep_head_num=heads,
        token_partition=partitions,
        paras_tp_rank=rank,
        paras_tp_size=world,
    )
    fence = torch.zeros(1, device="cuda")
    dist.all_reduce(fence, group=group)
    for layer in reversed(range(NUM_LAYERS)):
        backend.scatter_one_layer(
            LayerCacheSpec(
                layer_id=layer,
                kind="full",
                tokens_cap_ep=0,
                tokens_cap_tp=0,
                num_kv_heads=heads,
                head_dim=HEAD_DIM,
                sliding_window_size=None,
            )
        )
        dist.all_reduce(fence, group=group)

    ok = True
    count = counts[rank]
    for layer in range(NUM_LAYERS):
        for kv, name in enumerate(("k", "v")):
            expected = torch.empty((count, heads, HEAD_DIM), dtype=DTYPE)
            for sender in range(world):
                replica = sender % replication
                begin, end = (
                    count * replica // replication,
                    count * (replica + 1) // replication,
                )
                source_slots = physical_slots[partitions[rank][begin:end]]
                first_head = sender * heads // world
                expected[begin:end, first_head : first_head + hpr] = pattern(
                    sender, layer, kv, capacity, hpr
                )[source_slots]
            actual = mgr.get_view(f"model.layers.{layer}.kv.ep.{name}")[1 : count + 1]
            ok = ok and torch.equal(actual.cpu(), expected)
    passed = torch.tensor(int(ok), device="cuda")
    dist.all_reduce(passed, op=dist.ReduceOp.MIN, group=group)
    assert passed.item(), "Production UMM scatter differs from CPU reference"
