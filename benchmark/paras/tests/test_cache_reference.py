"""CPU checks for the actual NCCL KV packing and reassembly operations."""

import unittest
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.cache_reference import (
    pack_gather,
    scatter_token_positions,
    unpack_gather,
    unpack_scatter,
)


class CacheReferenceTest(unittest.TestCase):
    def test_kernel_scatter_descriptors(self):
        from bench_cache import build_scatter_routing
        from common.slot_init import random_resident_slots

        for world, heads in ((2, 4), (8, 4), (4, 1)):
            replication = max(1, world // heads)
            n = 3 * replication
            layout = SimpleNamespace(
                num_resident_tokens=n,
                replication_factor=replication,
                ep_max_tokens=n + 1,
            )
            # Full occupancy used to trigger the old reverse-volume truncation.
            for rank in range(world):
                ctx = SimpleNamespace(
                    rank=rank, world_size=world, device=torch.device("cpu")
                )
                src, owners, dst = build_scatter_routing(ctx, layout, 123)
                self.assertEqual(
                    src.tolist(), scatter_token_positions(rank, world, replication, n)
                )
                self.assertEqual(src.numel(), world * n // replication)
                for owner in range(world):
                    positions = random_resident_slots(n, n + 1, owner, 123)
                    for token, slot in zip(
                        src[owners == owner].tolist(), dst[owners == owner].tolist()
                    ):
                        self.assertEqual(slot, positions[(token - 1) % n].item())

    def test_paged_roundtrip(self):
        for world, heads in ((2, 4), (4, 4), (8, 4), (4, 1)):
            with self.subTest(world=world, heads=heads):
                replication = max(1, world // heads)
                n, dim = 3 * replication, 5
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
                    tp.append(result)

                scatter = []
                for rank in range(world):
                    pos = (
                        torch.tensor(
                            scatter_token_positions(rank, world, replication, n)
                        )
                        - 1
                    )
                    scatter.append(
                        tp[rank][pos].reshape(
                            world, n // replication, local_heads, 2, dim
                        )
                    )
                for rank in range(world):
                    received = torch.stack([scatter[src][rank] for src in range(world)])
                    restored = unpack_scatter(received, heads)
                    self.assertTrue(
                        torch.equal(restored[:, :, 0], ep_k[rank][slots[rank]])
                    )
                    self.assertTrue(
                        torch.equal(restored[:, :, 1], ep_v[rank][slots[rank]])
                    )

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
