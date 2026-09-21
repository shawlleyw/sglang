"""Resident volume and layer separation checks without a GPU."""

import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.layouts import make_resident_kv_layout
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


if __name__ == "__main__":
    unittest.main()
