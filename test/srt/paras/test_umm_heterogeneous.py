#!/usr/bin/env python3
"""
Unit tests for ParaSMemoryManager.reserve_kv_cache with per-layer heterogeneous shapes.

Tests cover:
  1. Backward-compat: reserve_kv_cache(layer_specs=None) produces uniform slot shapes
  2. Heterogeneous reservation: 2 full + 4 SWA layers produce correct per-layer slot shapes
  3. Alias resolution: get_view() returns correct shapes per layer
  4. G10 alias-name stability: alias names remain consistent across layers

Usage:
  conda run -n sgl_paras python -m pytest test/srt/paras/test_umm_heterogeneous.py -v
"""

import os
import sys

import pytest
import torch

# Add sglang to path
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.join(_TEST_DIR, "..", "..", "..")
sys.path.insert(0, os.path.join(_ROOT_DIR, "python"))


# =========================================================================
# TEST GROUP 1: Backward-compat — uniform shapes when layer_specs=None
# =========================================================================


class TestBackwardCompatUniformShapes:
    """Verify reserve_kv_cache(layer_specs=None) produces uniform slot shapes."""

    def test_uniform_shapes_no_layer_specs(self):
        """
        When layer_specs=None, all slots should have uniform shape
        (ep_max_tokens + page_size, num_kv_heads, head_dim).
        """
        from sglang.srt.paras.paras_memory_manager import ParaSMemoryManager

        mgr = ParaSMemoryManager(device="cpu")

        # Reserve KV cache with uniform shapes
        num_layers = 6
        ep_max_tokens = 1024
        tp_max_tokens = 4096
        num_kv_heads = 8
        head_dim = 128
        page_size = 1
        kv_dtype = torch.bfloat16

        mgr.reserve_kv_cache(
            num_layers=num_layers,
            ep_max_tokens=ep_max_tokens,
            tp_max_tokens=tp_max_tokens,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            kv_dtype=kv_dtype,
            page_size=page_size,
            prefix="model",
            layer_specs=None,  # Uniform shapes
        )

        # Materialize to create the buffer
        mgr.materialize()

        # Expected uniform shape
        expected_shape = (ep_max_tokens + page_size, num_kv_heads, head_dim)

        # Verify all slots have the same shape
        for j in range(num_layers + 1):
            k_view = mgr.get_view(f"paras.kv_slot.{j}.k")
            v_view = mgr.get_view(f"paras.kv_slot.{j}.v")

            assert k_view.shape == expected_shape, (
                f"Slot {j} K shape mismatch: expected {expected_shape}, got {k_view.shape}"
            )
            assert v_view.shape == expected_shape, (
                f"Slot {j} V shape mismatch: expected {expected_shape}, got {v_view.shape}"
            )


# =========================================================================
# TEST GROUP 2: Heterogeneous reservation — per-layer shapes
# =========================================================================


class TestHeterogeneousReservation:
    """Verify reserve_kv_cache with layer_specs produces correct per-layer shapes."""

    def test_heterogeneous_2full_4swa(self):
        """
        Reserve KV cache for 2 full + 4 SWA layers.
        Verify each slot has the correct shape per its layer spec.
        """
        from sglang.srt.paras.paras_memory_manager import ParaSMemoryManager
        from sglang.srt.paras.cache_transfer import LayerCacheSpec

        mgr = ParaSMemoryManager(device="cpu")

        # Create layer specs: 2 full (1024 tokens) + 4 SWA (256 tokens)
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

        mgr.reserve_kv_cache(
            num_layers=num_layers,
            ep_max_tokens=1024,  # Unused when layer_specs provided
            tp_max_tokens=4096,  # Unused when layer_specs provided
            num_kv_heads=8,
            head_dim=128,
            kv_dtype=kv_dtype,
            page_size=page_size,
            prefix="model",
            layer_specs=layer_specs,
        )

        # Materialize
        mgr.materialize()

        # Verify slot shapes match layer specs
        # Slot 0: TP landing for layer 0 (full, 1024 tokens)
        k0 = mgr.get_view("paras.kv_slot.0.k")
        assert k0.shape == (1024 + page_size, 8, 128), (
            f"Slot 0 shape mismatch: expected (1025, 8, 128), got {k0.shape}"
        )

        # Slot 1: EP data for layer 0 (full, 1024 tokens)
        k1 = mgr.get_view("paras.kv_slot.1.k")
        assert k1.shape == (1024 + page_size, 8, 128), (
            f"Slot 1 shape mismatch: expected (1025, 8, 128), got {k1.shape}"
        )

        # Slot 2: EP data for layer 1 (full, 1024 tokens)
        k2 = mgr.get_view("paras.kv_slot.2.k")
        assert k2.shape == (1024 + page_size, 8, 128), (
            f"Slot 2 shape mismatch: expected (1025, 8, 128), got {k2.shape}"
        )

        # Slot 3: EP data for layer 2 (SWA, 256 tokens)
        k3 = mgr.get_view("paras.kv_slot.3.k")
        assert k3.shape == (256 + page_size, 8, 128), (
            f"Slot 3 shape mismatch: expected (257, 8, 128), got {k3.shape}"
        )

        # Slot 4: EP data for layer 3 (SWA, 256 tokens)
        k4 = mgr.get_view("paras.kv_slot.4.k")
        assert k4.shape == (256 + page_size, 8, 128), (
            f"Slot 4 shape mismatch: expected (257, 8, 128), got {k4.shape}"
        )

        # Slot 5: EP data for layer 4 (SWA, 256 tokens)
        k5 = mgr.get_view("paras.kv_slot.5.k")
        assert k5.shape == (256 + page_size, 8, 128), (
            f"Slot 5 shape mismatch: expected (257, 8, 128), got {k5.shape}"
        )

        # Slot 6: EP data for layer 5 (SWA, 256 tokens)
        k6 = mgr.get_view("paras.kv_slot.6.k")
        assert k6.shape == (256 + page_size, 8, 128), (
            f"Slot 6 shape mismatch: expected (257, 8, 128), got {k6.shape}"
        )


# =========================================================================
# TEST GROUP 3: Alias resolution — per-layer model.layers.{i}.kv aliases
# =========================================================================


class TestAliasResolution:
    """Verify get_view() returns correct shapes for model.layers.{i}.kv aliases."""

    def test_alias_resolution_per_layer(self):
        """
        Verify that model.layers.{i}.kv.k/v aliases resolve to correct slot shapes.
        """
        from sglang.srt.paras.paras_memory_manager import ParaSMemoryManager
        from sglang.srt.paras.cache_transfer import LayerCacheSpec

        mgr = ParaSMemoryManager(device="cpu")

        # Create layer specs: 2 full + 2 SWA
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

        # Verify model.layers.{i}.kv aliases resolve correctly
        # Layer 0 (full): should map to slot 1 (1024 + 1 tokens)
        k0 = mgr.get_view("model.layers.0.kv.k")
        assert k0.shape == (1024 + page_size, 8, 128), (
            f"model.layers.0.kv.k shape mismatch: expected (1025, 8, 128), got {k0.shape}"
        )

        # Layer 1 (full): should map to slot 2 (1024 + 1 tokens)
        k1 = mgr.get_view("model.layers.1.kv.k")
        assert k1.shape == (1024 + page_size, 8, 128), (
            f"model.layers.1.kv.k shape mismatch: expected (1025, 8, 128), got {k1.shape}"
        )

        # Layer 2 (SWA): should map to slot 3 (256 + 1 tokens)
        k2 = mgr.get_view("model.layers.2.kv.k")
        assert k2.shape == (256 + page_size, 8, 128), (
            f"model.layers.2.kv.k shape mismatch: expected (257, 8, 128), got {k2.shape}"
        )

        # Layer 3 (SWA): should map to slot 4 (256 + 1 tokens)
        k3 = mgr.get_view("model.layers.3.kv.k")
        assert k3.shape == (256 + page_size, 8, 128), (
            f"model.layers.3.kv.k shape mismatch: expected (257, 8, 128), got {k3.shape}"
        )


# =========================================================================
# TEST GROUP 4: G10 alias-name stability
# =========================================================================


class TestG10AliasNameStability:
    """Verify alias names remain consistent (G10 constraint)."""

    def test_alias_names_stable(self):
        """
        Verify that alias names follow the pattern:
        - model.layers.{i}.kv.k/v (weight-loading aliases)
        - paras.kv_slot.{j}.k/v (physical slots)
        """
        from sglang.srt.paras.paras_memory_manager import ParaSMemoryManager
        from sglang.srt.paras.cache_transfer import LayerCacheSpec

        mgr = ParaSMemoryManager(device="cpu")

        # Create layer specs: 2 full + 2 SWA
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

        # Verify that all expected alias names exist
        expected_aliases = []

        # Physical slots: paras.kv_slot.{j}.k/v for j in [0, num_layers]
        for j in range(num_layers + 1):
            expected_aliases.append(f"paras.kv_slot.{j}.k")
            expected_aliases.append(f"paras.kv_slot.{j}.v")

        # Weight-loading aliases: model.layers.{i}.kv.k/v for i in [0, num_layers-1]
        for i in range(num_layers):
            expected_aliases.append(f"model.layers.{i}.kv.k")
            expected_aliases.append(f"model.layers.{i}.kv.v")

        # Check that all expected aliases are in the manager's entries
        for alias_name in expected_aliases:
            assert alias_name in mgr._entries, (
                f"Expected alias '{alias_name}' not found in manager entries"
            )

        # Verify that model.layers.{i}.kv.k/v point to the correct slots
        for i in range(num_layers):
            # model.layers.{i}.kv.k should point to paras.kv_slot.{i+1}.k
            layer_k_entry = mgr._entries[f"model.layers.{i}.kv.k"]
            slot_k_entry = mgr._entries[f"paras.kv_slot.{i+1}.k"]
            assert layer_k_entry is slot_k_entry, (
                f"model.layers.{i}.kv.k should alias to paras.kv_slot.{i+1}.k"
            )

            # model.layers.{i}.kv.v should point to paras.kv_slot.{i+1}.v
            layer_v_entry = mgr._entries[f"model.layers.{i}.kv.v"]
            slot_v_entry = mgr._entries[f"paras.kv_slot.{i+1}.v"]
            assert layer_v_entry is slot_v_entry, (
                f"model.layers.{i}.kv.v should alias to paras.kv_slot.{i+1}.v"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
