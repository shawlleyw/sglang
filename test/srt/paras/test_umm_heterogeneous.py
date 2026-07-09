#!/usr/bin/env python3
"""
Unit tests for ParaSMemoryManager.reserve_kv_cache with contiguous buffer layout.

Tests cover:
  1. Backward-compat: reserve_kv_cache(layer_specs=None) produces uniform EP/TP shapes
  2. Heterogeneous reservation: 2 full + 4 SWA layers produce correct per-layer shapes
  3. Alias resolution: get_view() returns correct shapes per layer
  4. Alias-name stability: alias names remain consistent across layers
  5. Offset geometry: TP at prefix_i, EP at max_L + prefix_i, no overlap

Usage:
  conda run -n sgl_paras python -m pytest test/srt/paras/test_umm_heterogeneous.py -v
"""

import os
import sys

import pytest
import torch

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.join(_TEST_DIR, "..", "..", "..")
sys.path.insert(0, os.path.join(_ROOT_DIR, "python"))


# =========================================================================
# TEST GROUP 1: Backward-compat — uniform shapes when layer_specs=None
# =========================================================================


class TestBackwardCompatUniformShapes:
    """Verify reserve_kv_cache(layer_specs=None) produces uniform EP/TP shapes."""

    def test_uniform_shapes_no_layer_specs(self):
        from sglang.srt.paras.paras_memory_manager import ParaSMemoryManager

        mgr = ParaSMemoryManager(device="cpu")

        num_layers = 6
        ep_max_tokens = 1024
        tp_max_tokens = 4096
        num_kv_heads = 8
        head_dim = 128
        page_size = 1
        kv_dtype = torch.bfloat16
        elem_size = 2

        mgr.reserve_kv_cache(
            num_layers=num_layers,
            ep_max_tokens=ep_max_tokens,
            tp_max_tokens=tp_max_tokens,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            kv_dtype=kv_dtype,
            page_size=page_size,
            prefix="model",
            layer_specs=None,
        )
        mgr.materialize()

        expected_ep_shape = (ep_max_tokens + page_size, num_kv_heads, head_dim)
        per_layer_bytes = (
            (ep_max_tokens + page_size) * num_kv_heads * head_dim * elem_size
        )
        max_L = per_layer_bytes

        for i in range(num_layers):
            ep_k = mgr.get_view(f"model.layers.{i}.kv.ep.k")
            ep_v = mgr.get_view(f"model.layers.{i}.kv.ep.v")
            tp_k = mgr.get_view(f"model.layers.{i}.kv.tp.k")
            tp_v = mgr.get_view(f"model.layers.{i}.kv.tp.v")

            assert ep_k.shape == expected_ep_shape, (
                f"Layer {i} EP K shape: expected {expected_ep_shape}, got {ep_k.shape}"
            )
            assert ep_v.shape == expected_ep_shape
            assert tp_k.shape == expected_ep_shape
            assert tp_v.shape == expected_ep_shape

            assert ep_k.data_ptr() != tp_k.data_ptr(), (
                f"Layer {i}: EP and TP K should have different offsets"
            )

        ep_k0_entry = mgr._entries["model.layers.0.kv.ep.k"]
        tp_k0_entry = mgr._entries["model.layers.0.kv.tp.k"]
        assert (
            ep_k0_entry.offset_bytes - tp_k0_entry.offset_bytes == max_L
        ), "EP offset should be max_L bytes after TP start"


# =========================================================================
# TEST GROUP 2: Heterogeneous reservation — per-layer shapes
# =========================================================================


class TestHeterogeneousReservation:
    """Verify reserve_kv_cache with layer_specs produces correct per-layer shapes."""

    def test_heterogeneous_2full_4swa(self):
        from sglang.srt.paras.paras_memory_manager import ParaSMemoryManager
        from sglang.srt.paras.cache_transfer import LayerCacheSpec

        mgr = ParaSMemoryManager(device="cpu")

        layer_specs = [
            LayerCacheSpec(
                layer_id=0, kind="full", tokens_cap_ep=1024, tokens_cap_tp=4096,
                num_kv_heads=8, head_dim=128, sliding_window_size=None,
            ),
            LayerCacheSpec(
                layer_id=1, kind="full", tokens_cap_ep=1024, tokens_cap_tp=4096,
                num_kv_heads=8, head_dim=128, sliding_window_size=None,
            ),
            LayerCacheSpec(
                layer_id=2, kind="swa", tokens_cap_ep=256, tokens_cap_tp=1024,
                num_kv_heads=8, head_dim=128, sliding_window_size=2048,
            ),
            LayerCacheSpec(
                layer_id=3, kind="swa", tokens_cap_ep=256, tokens_cap_tp=1024,
                num_kv_heads=8, head_dim=128, sliding_window_size=2048,
            ),
            LayerCacheSpec(
                layer_id=4, kind="swa", tokens_cap_ep=256, tokens_cap_tp=1024,
                num_kv_heads=8, head_dim=128, sliding_window_size=2048,
            ),
            LayerCacheSpec(
                layer_id=5, kind="swa", tokens_cap_ep=256, tokens_cap_tp=1024,
                num_kv_heads=8, head_dim=128, sliding_window_size=2048,
            ),
        ]

        num_layers = len(layer_specs)
        page_size = 1
        kv_dtype = torch.bfloat16
        elem_size = 2

        mgr.reserve_kv_cache(
            num_layers=num_layers,
            ep_max_tokens=1024,
            tp_max_tokens=4096,
            num_kv_heads=8,
            head_dim=128,
            kv_dtype=kv_dtype,
            page_size=page_size,
            prefix="model",
            layer_specs=layer_specs,
        )
        mgr.materialize()

        expected_shapes = {
            0: (1025, 8, 128),
            1: (1025, 8, 128),
            2: (257, 8, 128),
            3: (257, 8, 128),
            4: (257, 8, 128),
            5: (257, 8, 128),
        }

        for i, exp_shape in expected_shapes.items():
            ep_k = mgr.get_view(f"model.layers.{i}.kv.ep.k")
            assert ep_k.shape == exp_shape, (
                f"Layer {i} EP K shape: expected {exp_shape}, got {ep_k.shape}"
            )

        layer_bytes = [
            (s.tokens_cap_ep + page_size) * s.num_kv_heads * s.head_dim * elem_size
            for s in layer_specs
        ]
        max_L = max(layer_bytes)

        prefix_bytes = 0
        for i in range(num_layers):
            tp_entry = mgr._entries[f"model.layers.{i}.kv.tp.k"]
            ep_entry = mgr._entries[f"model.layers.{i}.kv.ep.k"]

            tp_expected = mgr._entries["model.layers.0.kv.tp.k"].offset_bytes + prefix_bytes
            ep_expected = tp_expected + max_L

            assert tp_entry.offset_bytes == tp_expected, (
                f"Layer {i} TP K offset: expected {tp_expected}, got {tp_entry.offset_bytes}"
            )
            assert ep_entry.offset_bytes == ep_expected, (
                f"Layer {i} EP K offset: expected {ep_expected}, got {ep_entry.offset_bytes}"
            )
            prefix_bytes += layer_bytes[i]


# =========================================================================
# TEST GROUP 3: Alias resolution — per-layer model.layers.{i}.kv aliases
# =========================================================================


class TestAliasResolution:
    """Verify get_view() returns correct shapes for model.layers.{i}.kv aliases."""

    def test_alias_resolution_per_layer(self):
        from sglang.srt.paras.paras_memory_manager import ParaSMemoryManager
        from sglang.srt.paras.cache_transfer import LayerCacheSpec

        mgr = ParaSMemoryManager(device="cpu")

        layer_specs = [
            LayerCacheSpec(
                layer_id=0, kind="full", tokens_cap_ep=1024, tokens_cap_tp=4096,
                num_kv_heads=8, head_dim=128, sliding_window_size=None,
            ),
            LayerCacheSpec(
                layer_id=1, kind="full", tokens_cap_ep=1024, tokens_cap_tp=4096,
                num_kv_heads=8, head_dim=128, sliding_window_size=None,
            ),
            LayerCacheSpec(
                layer_id=2, kind="swa", tokens_cap_ep=256, tokens_cap_tp=1024,
                num_kv_heads=8, head_dim=128, sliding_window_size=2048,
            ),
            LayerCacheSpec(
                layer_id=3, kind="swa", tokens_cap_ep=256, tokens_cap_tp=1024,
                num_kv_heads=8, head_dim=128, sliding_window_size=2048,
            ),
        ]

        num_layers = len(layer_specs)
        page_size = 1
        kv_dtype = torch.bfloat16

        mgr.reserve_kv_cache(
            num_layers=num_layers,
            ep_max_tokens=1024,
            tp_max_tokens=4096,
            num_kv_heads=8,
            head_dim=128,
            kv_dtype=kv_dtype,
            page_size=page_size,
            prefix="model",
            layer_specs=layer_specs,
        )
        mgr.materialize()

        expected = {
            0: (1025, 8, 128),
            1: (1025, 8, 128),
            2: (257, 8, 128),
            3: (257, 8, 128),
        }

        for i, exp_shape in expected.items():
            k = mgr.get_view(f"model.layers.{i}.kv.k")
            assert k.shape == exp_shape, (
                f"model.layers.{i}.kv.k shape: expected {exp_shape}, got {k.shape}"
            )

            ep_k = mgr.get_view(f"model.layers.{i}.kv.ep.k")
            assert k.data_ptr() == ep_k.data_ptr(), (
                f"Layer {i}: kv.k and kv.ep.k should share the same data pointer"
            )


# =========================================================================
# TEST GROUP 4: Alias-name stability
# =========================================================================


class TestG10AliasNameStability:
    """Verify alias names remain consistent."""

    def test_alias_names_stable(self):
        from sglang.srt.paras.paras_memory_manager import ParaSMemoryManager
        from sglang.srt.paras.cache_transfer import LayerCacheSpec

        mgr = ParaSMemoryManager(device="cpu")

        layer_specs = [
            LayerCacheSpec(
                layer_id=0, kind="full", tokens_cap_ep=1024, tokens_cap_tp=4096,
                num_kv_heads=8, head_dim=128, sliding_window_size=None,
            ),
            LayerCacheSpec(
                layer_id=1, kind="full", tokens_cap_ep=1024, tokens_cap_tp=4096,
                num_kv_heads=8, head_dim=128, sliding_window_size=None,
            ),
            LayerCacheSpec(
                layer_id=2, kind="swa", tokens_cap_ep=256, tokens_cap_tp=1024,
                num_kv_heads=8, head_dim=128, sliding_window_size=2048,
            ),
            LayerCacheSpec(
                layer_id=3, kind="swa", tokens_cap_ep=256, tokens_cap_tp=1024,
                num_kv_heads=8, head_dim=128, sliding_window_size=2048,
            ),
        ]

        num_layers = len(layer_specs)
        page_size = 1
        kv_dtype = torch.bfloat16

        mgr.reserve_kv_cache(
            num_layers=num_layers,
            ep_max_tokens=1024,
            tp_max_tokens=4096,
            num_kv_heads=8,
            head_dim=128,
            kv_dtype=kv_dtype,
            page_size=page_size,
            prefix="model",
            layer_specs=layer_specs,
        )
        mgr.materialize()

        expected_aliases = []
        for i in range(num_layers):
            expected_aliases.append(f"model.layers.{i}.kv.k")
            expected_aliases.append(f"model.layers.{i}.kv.v")
            expected_aliases.append(f"model.layers.{i}.kv.ep.k")
            expected_aliases.append(f"model.layers.{i}.kv.ep.v")
            expected_aliases.append(f"model.layers.{i}.kv.tp.k")
            expected_aliases.append(f"model.layers.{i}.kv.tp.v")

        for alias_name in expected_aliases:
            assert alias_name in mgr._entries, (
                f"Expected alias '{alias_name}' not found in manager entries"
            )

        for i in range(num_layers):
            layer_k_entry = mgr._entries[f"model.layers.{i}.kv.k"]
            ep_k_entry = mgr._entries[f"model.layers.{i}.kv.ep.k"]
            assert layer_k_entry is ep_k_entry, (
                f"model.layers.{i}.kv.k should be the same object as kv.ep.k"
            )

            layer_v_entry = mgr._entries[f"model.layers.{i}.kv.v"]
            ep_v_entry = mgr._entries[f"model.layers.{i}.kv.ep.v"]
            assert layer_v_entry is ep_v_entry, (
                f"model.layers.{i}.kv.v should be the same object as kv.ep.v"
            )


# =========================================================================
# TEST GROUP 5: Offset geometry — race-free layout invariants
# =========================================================================


class TestOffsetGeometry:
    """Verify TP/EP offsets satisfy the contiguous-buffer layout invariants."""

    def test_no_overlap_heterogeneous(self):
        from sglang.srt.paras.paras_memory_manager import ParaSMemoryManager
        from sglang.srt.paras.cache_transfer import LayerCacheSpec

        mgr = ParaSMemoryManager(device="cpu")

        layer_specs = [
            LayerCacheSpec(
                layer_id=0, kind="swa", tokens_cap_ep=256, tokens_cap_tp=1024,
                num_kv_heads=8, head_dim=128, sliding_window_size=2048,
            ),
            LayerCacheSpec(
                layer_id=1, kind="full", tokens_cap_ep=1024, tokens_cap_tp=4096,
                num_kv_heads=8, head_dim=128, sliding_window_size=None,
            ),
            LayerCacheSpec(
                layer_id=2, kind="full", tokens_cap_ep=1024, tokens_cap_tp=4096,
                num_kv_heads=8, head_dim=128, sliding_window_size=None,
            ),
        ]

        num_layers = len(layer_specs)
        page_size = 1
        kv_dtype = torch.bfloat16
        elem_size = 2

        mgr.reserve_kv_cache(
            num_layers=num_layers,
            ep_max_tokens=1024,
            tp_max_tokens=4096,
            num_kv_heads=8,
            head_dim=128,
            kv_dtype=kv_dtype,
            page_size=page_size,
            prefix="model",
            layer_specs=layer_specs,
        )
        mgr.materialize()

        layer_bytes = [
            (s.tokens_cap_ep + page_size) * s.num_kv_heads * s.head_dim * elem_size
            for s in layer_specs
        ]
        max_L = max(layer_bytes)

        def _collect(mode, side):
            return [
                (mgr._entries[f"model.layers.{i}.kv.{mode}.{side}"].offset_bytes,
                 mgr._entries[f"model.layers.{i}.kv.{mode}.{side}"].offset_bytes
                 + mgr._entries[f"model.layers.{i}.kv.{mode}.{side}"].size_bytes,
                 f"{mode}.{side}.{i}")
                for i in range(num_layers)
            ]

        for side in ("k", "v"):
            tp_intervals = sorted(_collect("tp", side))
            for idx in range(len(tp_intervals) - 1):
                _, end_a, name_a = tp_intervals[idx]
                start_b, _, name_b = tp_intervals[idx + 1]
                assert end_a <= start_b, f"TP overlap: {name_a}→{end_a} vs {name_b}→{start_b}"

            ep_intervals = sorted(_collect("ep", side))
            for idx in range(len(ep_intervals) - 1):
                _, end_a, name_a = ep_intervals[idx]
                start_b, _, name_b = ep_intervals[idx + 1]
                assert end_a <= start_b, f"EP overlap: {name_a}→{end_a} vs {name_b}→{start_b}"

        k_ep_last = max(e.offset_bytes + e.size_bytes for n, e in mgr._entries.items() if ".kv.ep.k" in n)
        v_tp_first = min(e.offset_bytes for n, e in mgr._entries.items() if ".kv.tp.v" in n)
        assert k_ep_last <= v_tp_first, f"K/V regions overlap: K ends {k_ep_last}, V starts {v_tp_first}"

        for i in range(num_layers):
            tp_e = mgr._entries[f"model.layers.{i}.kv.tp.k"]
            ep_e = mgr._entries[f"model.layers.{i}.kv.ep.k"]
            assert ep_e.offset_bytes >= tp_e.offset_bytes + tp_e.size_bytes, (
                f"Layer {i}: same-layer TP and EP overlap"
            )

        for i in range(num_layers):
            tp_e = mgr._entries[f"model.layers.{i}.kv.tp.k"]
            ep_e = mgr._entries[f"model.layers.{i}.kv.ep.k"]
            assert ep_e.offset_bytes - tp_e.offset_bytes == max_L, (
                f"Layer {i}: EP - TP offset should be max_L={max_L}, "
                f"got {ep_e.offset_bytes - tp_e.offset_bytes}"
            )
            assert tp_e.size_bytes == ep_e.size_bytes == layer_bytes[i]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
