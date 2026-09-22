"""Resident volume and layer separation checks without a GPU."""

import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.layouts import (
    KVLayout, OverlappingCacheLayout, make_overlapping_cache_layout,
    make_resident_kv_layout,
)
from common.model_configs import PRESETS
from bench_cache import _pattern, _views, offsets_in_arena
from types import SimpleNamespace


class ResidentCacheVolumeTest(unittest.TestCase):
    def test_requested_total_is_not_multiplied_by_layers(self):
        for preset in ("gpt-oss-120b", "qwen3-235b", "qwen3-30b"):
            model = PRESETS[preset]
            for gib in (10, 20, 30):
                layout = make_resident_kv_layout(model, 8, gib)
                actual = (
                    layout.num_resident_tokens
                    * 2
                    * layout.bytes_per_ep_slot
                    * model.num_hidden_layers
                )
                rounding = (
                    2
                    * layout.bytes_per_ep_slot
                    * model.num_hidden_layers
                    * layout.replication_factor
                )
                self.assertLessEqual(actual, gib * 2**30)
                self.assertLess(gib * 2**30 - actual, rounding)
                self.assertEqual(
                    layout.num_resident_tokens % layout.replication_factor, 0
                )
                self.assertEqual(layout.ep_max_tokens, layout.num_resident_tokens + 1)

    def test_replicated_tp_cache_requires_more_than_a100_capacity_at_30gib(self):
        # Qwen TP8 replicates four KV heads across eight ranks. Even though
        # redundant network sends are removed, both TP replicas remain resident.
        for preset in ("gpt-oss-120b", "qwen3-235b", "qwen3-30b"):
            model = PRESETS[preset]
            layout = make_resident_kv_layout(model, 8, 30)
            arena = offsets_in_arena(layout)["total"] * model.num_hidden_layers
            expected_gib = 60 if preset == "gpt-oss-120b" else 90
            # Whole-token/replica rounding and per-layer padding are sub-MiB
            # for these settings; they do not change the hardware fit boundary.
            self.assertAlmostEqual(arena / 2**30, expected_gib, delta=2 / 1024)
            self.assertEqual(arena > 80 * 2**30, preset != "gpt-oss-120b")

    def test_distinct_layer_views_and_patterns(self):
        model = PRESETS["gpt-oss-120b"]
        layout = make_resident_kv_layout(model, 8, 0.0002)
        base = offsets_in_arena(layout)
        ctx = SimpleNamespace(buf=torch.zeros(2 * base["total"], dtype=torch.uint8))
        first = _views(ctx, layout, base)
        second = _views(ctx, layout, {k: v + base["total"] for k, v in base.items()})
        for value in first.values():
            value.fill_(1)
        for value in second.values():
            self.assertTrue(torch.all(value == 0))
        tokens = torch.arange(8)
        heads = torch.arange(8)
        self.assertFalse(
            torch.equal(
                _pattern(tokens, heads, 64, layer_index=0),
                _pattern(tokens, heads, 64, layer_index=1),
            )
        )

    def test_reject_invalid_volumes(self):
        for volume in (0, -1, float("nan"), float("inf"), 1e-10):
            with self.assertRaises(ValueError):
                make_resident_kv_layout(PRESETS["gpt-oss-120b"], 8, volume)


class OverlappingCacheTest(unittest.TestCase):
    def test_real_layouts_slot_zero_and_ninety_four_layers(self):
        for heads in (8, 4, 1):  # TP8 gives R1, R2, R8.
            for tokens in (8, 128, 8192):
                base = KVLayout(8, heads, 128, 2, tokens + 1, 8 * tokens + 1, tokens)
                plan = make_overlapping_cache_layout(base, 94)
                self.assertEqual(plan.offsets(93)["ep_v"], 187 * base.ep_buffer_bytes)
                self.assertEqual(plan.offsets(93)["tp_v"],
                                 2 * base.ep_buffer_bytes + 93 * plan.tp_layer_stride + base.tp_buffer_bytes)
                self.assertEqual(plan.layer_order("ep_to_tp"), tuple(range(93, -1, -1)))
                self.assertEqual(plan.layer_order("tp_to_ep"), tuple(range(94)))
                self.assertEqual(plan.arena_bytes, plan.ep_layer_bytes + 94 * plan.tp_layer_stride)
                if heads == 8:
                    self.assertLess(plan.tp_layer_bytes, plan.ep_layer_bytes)
                    self.assertEqual(plan.tp_layer_stride, plan.ep_layer_bytes)
                else:
                    self.assertEqual(plan.tp_layer_stride, plan.tp_layer_bytes)

    def test_byte_array_migration_preserves_unread_sources(self):
        # Simulate real writes into one shared allocation, not just inequalities.
        for layers in (1, 2, 5, 12):
            for replication in (1, 2, 8):
                plan = OverlappingCacheLayout(8, 8 * replication, layers)
                for direction in ("ep_to_tp", "tp_to_ep"):
                    arena = bytearray([255]) * plan.arena_bytes
                    for layer in range(layers):
                        source, _ = plan._regions(layer, direction)
                        arena[source[0]:source[1]] = bytes([layer + 1]) * (source[1] - source[0])
                    for layer in plan.layer_order(direction):
                        source, destination = plan._regions(layer, direction)
                        self.assertEqual(set(arena[source[0]:source[1]]), {layer + 1})
                        arena[destination[0]:destination[1]] = bytes([layer + 1]) * (destination[1] - destination[0])
                    for layer in range(layers):
                        _, destination = plan._regions(layer, direction)
                        self.assertEqual(set(arena[destination[0]:destination[1]]), {layer + 1})

    def test_wrong_order_and_incomplete_order_rejected(self):
        plan = OverlappingCacheLayout(64, 128, 94)
        for direction in ("ep_to_tp", "tp_to_ep"):
            with self.assertRaisesRegex(ValueError, "destroys unread"):
                plan.validate_order(direction, reversed(plan.layer_order(direction)))
            with self.assertRaisesRegex(ValueError, "exactly once"):
                plan.validate_order(direction, [0] * 94)

    def test_next_destination_must_not_race_current_source_read(self):
        # With R1 adjacent layouts shift by exactly one layer. Eager destination
        # commits would destroy every preceding in-flight read in both directions.
        plan = OverlappingCacheLayout(8, 8, 4)
        self.assertEqual(plan.prefetch_hazards("ep_to_tp"), ((3, 2), (2, 1), (1, 0)))
        self.assertEqual(plan.prefetch_hazards("tp_to_ep"), ((0, 1), (1, 2), (2, 3)))

    def test_real_qwen_residency_allocation(self):
        # R2 physical TP volume is approximately twice EP; one extra EP layer
        # replaces the previous full additional EP arena, including slot zero.
        for gib in (10, 20, 30, 40, 50, 60):
            tokens = (gib * 2**30) // (94 * 2 * 4 * 128 * 2)
            layout = KVLayout(8, 4, 128, 2, tokens + 1, tokens * 8 + 1, tokens)
            plan = make_overlapping_cache_layout(layout, 94)
            old = 94 * (plan.ep_layer_bytes + plan.tp_layer_bytes)
            self.assertEqual(old - plan.arena_bytes, 93 * plan.ep_layer_bytes)
            self.assertLess(plan.arena_bytes / (94 * plan.ep_layer_bytes), 2.02)

    def test_invalid_arguments(self):
        for args in ((0, 1, 1), (1, -1, 1), (1, 1, 0), (True, 1, 1)):
            with self.assertRaises(ValueError):
                OverlappingCacheLayout(*args)
        plan = OverlappingCacheLayout(1, 1, 1)
        with self.assertRaises(ValueError):
            plan.offsets(1)
        with self.assertRaises(ValueError):
            plan.layer_order("bad")


if __name__ == "__main__":
    unittest.main()
