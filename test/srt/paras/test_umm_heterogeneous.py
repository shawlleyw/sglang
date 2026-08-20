#!/usr/bin/env python3
"""
Unit tests for ParaSMemoryManager.reserve_kv_cache with contiguous buffer layout.

Tests cover:
  1. Backward-compat: reserve_kv_cache(layer_specs=None) produces uniform EP/TP shapes
  2. Heterogeneous reservation: 2 full + 4 SWA layers produce correct per-layer shapes
  3. Alias resolution: get_view() returns correct shapes per layer
  4. Alias-name stability: alias names remain consistent across layers
  5. Offset geometry: overlapped layout (EP cache low, TP cache high),
     per-layer [k|v] slabs, clobber-safe under switch orders, and exact
     tp_kv_tail_bytes accounting

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


def _align_up(x, a=256):
    return (x + a - 1) // a * a


def _overlap(a, b):
    return a[0] < b[0] + b[1] and b[0] < a[0] + a[1]


def _assert_overlapped_kv_layout(mgr, num_layers, prefix="model"):
    """Assert the overlapped KV invariants over real offsets.

    EP cache low / TP cache high; per-layer [k|v] slabs adjacent; and the
    layout is clobber-safe under the production switch orders (EP->TP cache
    reverse, TP->EP cache forward), verified per-layer-atomically.
    """

    def rng(mode, i, side):
        e = mgr._entries[f"{prefix}.layers.{i}.kv.{mode}.{side}"]
        return (e.offset_bytes, e.size_bytes)

    # Global orientation: the TP cache region base sits above the EP cache base.
    # Per-layer ranges can still overlap across modes; the clobber checks below
    # validate the production transfer orders.
    assert (
        rng("tp", 0, "k")[0] > rng("ep", 0, "k")[0]
    ), "TP cache base must sit above EP"
    for i in range(num_layers):
        ek, ev = rng("ep", i, "k"), rng("ep", i, "v")
        tk, tv = rng("tp", i, "k"), rng("tp", i, "v")
        assert ev[0] == ek[0] + _align_up(ek[1]), f"layer {i}: EP [k|v] not adjacent"
        assert tv[0] == tk[0] + _align_up(tk[1]), f"layer {i}: TP [k|v] not adjacent"
        for off, _ in (ek, ev, tk, tv):
            assert off % 256 == 0

    def slab(mode, i):
        return [rng(mode, i, "k"), rng(mode, i, "v")]

    unread = {i: slab("ep", i) for i in range(num_layers)}
    for i in range(num_layers - 1, -1, -1):
        for d in slab("tp", i):
            for j, ss in unread.items():
                for r in ss:
                    assert not _overlap(d, r), f"EP->TP clobber tp{i} vs ep{j}"
        del unread[i]

    unread = {i: slab("tp", i) for i in range(num_layers)}
    for i in range(num_layers):
        for d in slab("ep", i):
            for j, ss in unread.items():
                for r in ss:
                    assert not _overlap(d, r), f"TP->EP clobber ep{i} vs tp{j}"
        del unread[i]


# =========================================================================
# TEST GROUP 1: Backward-compat — uniform shapes when layer_specs=None
# =========================================================================


class TestBackwardCompatUniformShapes:
    """Verify reserve_kv_cache(layer_specs=None) produces uniform EP/TP shapes."""

    def test_uniform_shapes_no_layer_specs(self):
        from sglang.srt.paras.paras_memory_manager import ParaSMemoryManager

        mgr = ParaSMemoryManager(device="cpu")

        num_layers = 6
        # TP shards num_kv_heads by tp_size and holds tp_size x more tokens, so
        # each TP KV layer fits its EP counterpart, as the planner requires.
        tp_size = 4
        ep_max_tokens = 1024
        tp_max_tokens = ep_max_tokens * tp_size
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
            tp_size=tp_size,
            page_size=page_size,
            prefix="model",
            layer_specs=None,
        )
        mgr.materialize()

        tp_kv_heads = max(1, num_kv_heads // tp_size)
        expected_ep_shape = (ep_max_tokens + page_size, num_kv_heads, head_dim)
        expected_tp_shape = (tp_max_tokens + page_size, tp_kv_heads, head_dim)

        for i in range(num_layers):
            ep_k = mgr.get_view(f"model.layers.{i}.kv.ep.k")
            ep_v = mgr.get_view(f"model.layers.{i}.kv.ep.v")
            tp_k = mgr.get_view(f"model.layers.{i}.kv.tp.k")
            tp_v = mgr.get_view(f"model.layers.{i}.kv.tp.v")

            assert (
                ep_k.shape == expected_ep_shape
            ), f"Layer {i} EP K shape: expected {expected_ep_shape}, got {ep_k.shape}"
            assert ep_v.shape == expected_ep_shape
            assert tp_k.shape == expected_tp_shape
            assert tp_v.shape == expected_tp_shape

            assert (
                ep_k.data_ptr() != tp_k.data_ptr()
            ), f"Layer {i}: EP and TP K should have different offsets"

        _assert_overlapped_kv_layout(mgr, num_layers)


# =========================================================================
# TEST GROUP 2: Heterogeneous reservation — per-layer shapes
# =========================================================================


class TestHeterogeneousReservation:
    """Verify reserve_kv_cache with layer_specs produces correct per-layer shapes."""

    def test_heterogeneous_2full_4swa(self):
        from sglang.srt.paras.paras_memory_manager import ParaSMemoryManager
        from sglang.srt.paras.layers.utils import LayerCacheSpec

        mgr = ParaSMemoryManager(device="cpu")

        layer_specs = [
            LayerCacheSpec(
                layer_id=0,
                kind="full",
                tokens_cap_ep=1024,
                tokens_cap_tp=4096,
                num_kv_heads=8,
                head_dim=128,
                sliding_window_size=None,
            ),
            LayerCacheSpec(
                layer_id=1,
                kind="full",
                tokens_cap_ep=1024,
                tokens_cap_tp=4096,
                num_kv_heads=8,
                head_dim=128,
                sliding_window_size=None,
            ),
            LayerCacheSpec(
                layer_id=2,
                kind="swa",
                tokens_cap_ep=256,
                tokens_cap_tp=1024,
                num_kv_heads=8,
                head_dim=128,
                sliding_window_size=2048,
            ),
            LayerCacheSpec(
                layer_id=3,
                kind="swa",
                tokens_cap_ep=256,
                tokens_cap_tp=1024,
                num_kv_heads=8,
                head_dim=128,
                sliding_window_size=2048,
            ),
            LayerCacheSpec(
                layer_id=4,
                kind="swa",
                tokens_cap_ep=256,
                tokens_cap_tp=1024,
                num_kv_heads=8,
                head_dim=128,
                sliding_window_size=2048,
            ),
            LayerCacheSpec(
                layer_id=5,
                kind="swa",
                tokens_cap_ep=256,
                tokens_cap_tp=1024,
                num_kv_heads=8,
                head_dim=128,
                sliding_window_size=2048,
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
            tp_size=4,
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
            assert (
                ep_k.shape == exp_shape
            ), f"Layer {i} EP K shape: expected {exp_shape}, got {ep_k.shape}"

        _assert_overlapped_kv_layout(mgr, num_layers)


# =========================================================================
# TEST GROUP 3: Alias resolution — per-layer model.layers.{i}.kv aliases
# =========================================================================


class TestAliasResolution:
    """Verify get_view() returns correct shapes for model.layers.{i}.kv aliases."""

    def test_alias_resolution_per_layer(self):
        from sglang.srt.paras.paras_memory_manager import ParaSMemoryManager
        from sglang.srt.paras.layers.utils import LayerCacheSpec

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
            tp_size=4,
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
        from sglang.srt.paras.layers.utils import LayerCacheSpec

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
            tp_size=4,
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
        from sglang.srt.paras.layers.utils import LayerCacheSpec

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
            tp_size=4,
            page_size=page_size,
            prefix="model",
            layer_specs=layer_specs,
        )
        mgr.materialize()

        # Per-side (k, v) within-mode contiguity: no two same-mode same-side
        # slabs overlap.
        def _collect(mode, side):
            return [
                (mgr._entries[f"model.layers.{i}.kv.{mode}.{side}"].offset_bytes,
                 mgr._entries[f"model.layers.{i}.kv.{mode}.{side}"].offset_bytes
                 + mgr._entries[f"model.layers.{i}.kv.{mode}.{side}"].size_bytes,
                 f"{mode}.{side}.{i}")
                for i in range(num_layers)
            ]

        for mode in ("tp", "ep"):
            for side in ("k", "v"):
                intervals = sorted(_collect(mode, side))
                for idx in range(len(intervals) - 1):
                    _, end_a, name_a = intervals[idx]
                    start_b, _, name_b = intervals[idx + 1]
                    assert (
                        end_a <= start_b
                    ), f"{mode} overlap: {name_a}→{end_a} vs {name_b}→{start_b}"

        # EP cache low, TP cache high, per-layer [k|v] slabs, clobber-safe.
        _assert_overlapped_kv_layout(mgr, num_layers)

        # Validate the general tail formula used by the layout geometry.
        # Production inputs make this equal to max(tp_layer_kv_bytes).
        def _slab(mode, i):
            return _align_up(
                mgr._entries[f"model.layers.{i}.kv.{mode}.k"].size_bytes
            ) + _align_up(mgr._entries[f"model.layers.{i}.kv.{mode}.v"].size_bytes)

        tp_layer_kv_bytes = [_slab("tp", i) for i in range(num_layers)]
        ep_layer_kv_bytes = [_slab("ep", i) for i in range(num_layers)]
        tp_kv_tail_bytes, suffix = 0, 0
        for i in range(num_layers - 1, -1, -1):
            tp_kv_tail_bytes = max(tp_kv_tail_bytes, tp_layer_kv_bytes[i] + suffix)
            suffix += tp_layer_kv_bytes[i] - ep_layer_kv_bytes[i]
        tp_kv_tail_bytes = _align_up(tp_kv_tail_bytes)

        ep_end = max(
            e.offset_bytes + e.size_bytes
            for n, e in mgr._entries.items()
            if ".kv.ep." in n
        )
        tc_end = max(
            e.offset_bytes + e.size_bytes
            for n, e in mgr._entries.items()
            if ".kv.tp." in n
        )
        assert (
            tc_end == _align_up(ep_end) + tp_kv_tail_bytes
        ), f"TP KV tail mismatch: {tc_end=} {ep_end=} {tp_kv_tail_bytes=}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
