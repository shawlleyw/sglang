"""CPU checks for the actual NCCL KV packing and reassembly operations."""

import unittest
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.cache_reference import (
    copy_compact_destination,
    pack_gather,
    pack_scatter,
    scatter_token_positions,
    unpack_gather,
    unpack_scatter,
)


class CacheReferenceTest(unittest.TestCase):
    def test_kernel_scatter_descriptors(self):
        from bench_cache import build_scatter_routing, build_source_slots

        for world, heads in ((2, 4), (8, 8), (8, 4), (4, 1)):
            replication = max(1, world // heads)
            n = 3 * replication
            layout = SimpleNamespace(
                num_resident_tokens=n,
                replication_factor=replication,
                ep_max_tokens=n + 1,
                tp_max_tokens=world * n + 1,
            )
            # Full occupancy used to trigger the old reverse-volume truncation.
            for rank in range(world):
                ctx = SimpleNamespace(
                    rank=rank, world_size=world, device=torch.device("cpu")
                )
                source_slots = build_source_slots(ctx, layout, "tp_to_ep", 123)
                src, owners, dst = build_scatter_routing(ctx, layout, source_slots)
                logical = torch.tensor(
                    scatter_token_positions(rank, world, replication, n)
                )
                self.assertTrue(torch.equal(src, source_slots[logical - 1]))
                self.assertFalse(torch.equal(src, logical))
                # Every TP rank sees the same physical token mapping.
                rank_zero = SimpleNamespace(
                    rank=0, world_size=world, device=torch.device("cpu")
                )
                self.assertTrue(
                    torch.equal(
                        source_slots,
                        build_source_slots(rank_zero, layout, "tp_to_ep", 123),
                    )
                )
                self.assertEqual(src.numel(), world * n // replication)
                for owner in range(world):
                    self.assertEqual(
                        dst[owners == owner].tolist(),
                        ((logical[owners == owner] - 1) % n + 1).tolist(),
                    )

    def test_paged_roundtrip(self):
        from bench_cache import build_scatter_routing, build_source_slots

        for world, heads, dim in (
            (2, 4, 128),
            (4, 4, 128),
            (8, 8, 64),
            (8, 4, 128),
            (4, 1, 128),
        ):
            with self.subTest(world=world, heads=heads):
                replication = max(1, world // heads)
                n = 3 * replication
                ep_k, ep_v, slots = [], [], []
                for rank in range(world):
                    # Distinct values across ranks, tokens, heads, and dimensions.
                    k = (
                        torch.arange((2 * n + 1) * heads * dim).reshape(
                            2 * n + 1, heads, dim
                        )
                        + rank * 10000
                    )
                    ep_k.append(k)
                    ep_v.append(-k)
                    slots.append(
                        torch.randperm(
                            2 * n, generator=torch.Generator().manual_seed(rank)
                        )[:n]
                        + 1
                    )
                packed = [
                    pack_gather(k, v, pos, world)
                    for k, v, pos in zip(ep_k, ep_v, slots)
                ]
                expected_k = torch.cat([k[pos] for k, pos in zip(ep_k, slots)])
                expected_v = torch.cat([v[pos] for v, pos in zip(ep_v, slots)])
                tp = []
                layout = SimpleNamespace(
                    num_resident_tokens=n,
                    replication_factor=replication,
                    ep_max_tokens=2 * n + 1,
                    tp_max_tokens=2 * world * n + 1,
                )
                for rank in range(world):
                    received = torch.stack([packed[src][rank] for src in range(world)])
                    result = unpack_gather(received)
                    first = rank * heads // world
                    local_heads = max(1, heads // world)
                    self.assertTrue(
                        torch.equal(
                            result[:, :, 0], expected_k[:, first : first + local_heads]
                        )
                    )
                    self.assertTrue(
                        torch.equal(
                            result[:, :, 1], expected_v[:, first : first + local_heads]
                        )
                    )
                    k = torch.full((layout.tp_max_tokens, local_heads, dim), -1)
                    v = torch.full_like(k, -1)
                    copy_compact_destination(k, v, result)
                    # Both destination padding and unused capacity stay untouched.
                    self.assertTrue(torch.all(k[0] == -1))
                    self.assertTrue(torch.all(k[world * n + 1 :] == -1))
                    self.assertTrue(torch.equal(k[1 : world * n + 1], result[:, :, 0]))
                    self.assertTrue(torch.equal(v[1 : world * n + 1], result[:, :, 1]))
                    # Model a live TP pool after requests have allocated/freed slots.
                    ctx = SimpleNamespace(
                        rank=rank, world_size=world, device=torch.device("cpu")
                    )
                    positions = build_source_slots(ctx, layout, "tp_to_ep", 123)
                    k[positions.long()] = result[:, :, 0]
                    v[positions.long()] = result[:, :, 1]
                    tp.append((ctx, k, v, positions))

                scatter = []
                for ctx, k, v, positions in tp:
                    source, _, _ = build_scatter_routing(ctx, layout, positions)
                    send = torch.empty(
                        (world, n // replication, local_heads, 2, dim), dtype=k.dtype
                    )
                    pack_scatter(send, k, v, source.long())
                    scatter.append(send)
                for rank in range(world):
                    received = torch.stack([scatter[src][rank] for src in range(world)])
                    restored = unpack_scatter(received, heads)
                    self.assertTrue(
                        torch.equal(restored[:, :, 0], ep_k[rank][slots[rank]])
                    )
                    self.assertTrue(
                        torch.equal(restored[:, :, 1], ep_v[rank][slots[rank]])
                    )
                    k, v = torch.full_like(ep_k[rank], -1), torch.full_like(
                        ep_v[rank], -1
                    )
                    copy_compact_destination(k, v, restored)
                    expected_k, expected_v = torch.full_like(k, -1), torch.full_like(
                        v, -1
                    )
                    expected_k[1 : n + 1] = ep_k[rank][slots[rank]]
                    expected_v[1 : n + 1] = ep_v[rank][slots[rank]]
                    self.assertTrue(torch.equal(k, expected_k))
                    self.assertTrue(torch.equal(v, expected_v))

    def test_initialized_sources_transfer_to_fresh_destinations(self):
        from unittest.mock import patch
        from bench_cache import (
            _initialize,
            _verify,
            _views,
            build_scatter_routing,
            build_source_slots,
            offsets_in_arena,
        )
        from common.layouts import KVLayout

        # CPU-emulate the direct kernel's routes and exercise the benchmark's
        # initialization/verifier, including replicated heads and layer identity.
        for world, heads in ((8, 8), (8, 4), (2, 4)):
            n = 4
            layout = KVLayout(world, heads, 5, 2, 2 * n + 1, world * n + 1, n)
            base = offsets_in_arena(layout)
            for direction in ("ep_to_tp", "tp_to_ep"):
                for layer in (0, 1):
                    ranks = []
                    for rank in range(world):
                        ctx = SimpleNamespace(
                            rank=rank,
                            world_size=world,
                            device=torch.device("cpu"),
                            tp_group=None,
                            buf=torch.zeros(base["total"], dtype=torch.uint8),
                        )
                        views = _views(ctx, layout, base)
                        for value in views.values():
                            value.fill_(-1)
                        slots = build_source_slots(ctx, layout, direction, 123)
                        _initialize(ctx, layout, views, direction, slots, layer)
                        ranks.append((ctx, views, slots))
                    source_mode = "ep" if direction == "ep_to_tp" else "tp"
                    snapshots = [
                        [views[f"{source_mode}_{kind}"].clone() for kind in ("k", "v")]
                        for _, views, _ in ranks
                    ]
                    for ctx, views, slots in ranks:
                        if direction == "ep_to_tp":
                            for dest, (_, target, _) in enumerate(ranks):
                                first = dest * heads // world
                                for kind in ("k", "v"):
                                    target[f"tp_{kind}"][
                                        ctx.rank * n + 1 : (ctx.rank + 1) * n + 1
                                    ] = views[f"ep_{kind}"][
                                        slots.long(),
                                        first : first + layout.heads_per_rank,
                                    ]
                        else:
                            src, owners, dst = build_scatter_routing(ctx, layout, slots)
                            first = ctx.rank * heads // world
                            for source, owner, dest in zip(
                                src.tolist(), owners.tolist(), dst.tolist()
                            ):
                                for kind in ("k", "v"):
                                    ranks[owner][1][f"ep_{kind}"][
                                        dest, first : first + layout.heads_per_rank
                                    ] = views[f"tp_{kind}"][source]
                    with patch("bench_cache.dist.all_reduce"):
                        for rank, (ctx, views, _) in enumerate(ranks):
                            _verify(ctx, layout, views, direction, layer)
                            for kind, snapshot in zip(("k", "v"), snapshots[rank]):
                                self.assertTrue(
                                    torch.equal(
                                        views[f"{source_mode}_{kind}"], snapshot
                                    )
                                )
                                self.assertTrue(torch.all(views[f"ep_{kind}"][0] == -1))
                                self.assertTrue(torch.all(views[f"tp_{kind}"][0] == -1))
                            if direction == "tp_to_ep":
                                self.assertTrue(torch.all(views["ep_k"][n + 1 :] == -1))
                                self.assertTrue(torch.all(views["ep_v"][n + 1 :] == -1))
                        target = "tp_k" if direction == "ep_to_tp" else "ep_k"
                        ranks[0][1][target][1] = -1
                        with self.assertRaisesRegex(
                            RuntimeError, "differs from reference"
                        ):
                            _verify(ranks[0][0], layout, ranks[0][1], direction, layer)

    def test_replica_routes_cover_each_token_once_per_head(self):
        world, replication, n = 8, 2, 10
        for group in range(world // replication):
            positions = [
                p
                for rank in range(group * replication, (group + 1) * replication)
                for p in scatter_token_positions(rank, world, replication, n)
            ]
            self.assertEqual(sorted(positions), list(range(1, world * n + 1)))

    def test_invalid_replica_partition(self):
        with self.assertRaises(ValueError):
            scatter_token_positions(0, 8, 2, 3)


if __name__ == "__main__":
    unittest.main()
