#!/usr/bin/env python3
"""
Unit tests for LayerCacheSpec, classify_layers_from_config, and plan_hybrid_kv_budget.

Tests cover:
  1. LayerCacheSpec frozen dataclass immutability
  2. classify_layers_from_config with mixed layer types
  3. classify_layers_from_config with all-full attention
  4. classify_layers_from_config with all-SWA
  5. classify_layers_from_config without layer_types attribute
  6. G15 violation: mismatched num_kv_heads
  7. G7 violation: tp_size > num_kv_heads with SWA
  8. Budget parity: plan_hybrid_kv_budget formula verification

Usage:
  conda run -n sgl_paras python -m pytest test/srt/paras/test_layer_cache_spec.py -v
"""

import os
import sys
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

# Add sglang to path
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.join(_TEST_DIR, "..", "..", "..")
sys.path.insert(0, os.path.join(_ROOT_DIR, "python"))


# =========================================================================
# TEST GROUP 1: LayerCacheSpec frozen dataclass
# =========================================================================


class TestLayerCacheSpecFrozen:
    """Verify LayerCacheSpec is frozen and immutable."""

    def test_layer_cache_spec_is_frozen(self):
        """Mutation of LayerCacheSpec should raise FrozenInstanceError."""
        from sglang.srt.paras.layers.utils import LayerCacheSpec

        spec = LayerCacheSpec(
            layer_id=0,
            kind="full",
            tokens_cap_ep=1024,
            tokens_cap_tp=4096,
            num_kv_heads=8,
            head_dim=128,
            sliding_window_size=None,
        )

        # Verify initial state
        assert spec.layer_id == 0
        assert spec.kind == "full"

        # Attempt to mutate should raise FrozenInstanceError
        with pytest.raises(FrozenInstanceError):
            spec.layer_id = 1

        with pytest.raises(FrozenInstanceError):
            spec.kind = "swa"

        with pytest.raises(FrozenInstanceError):
            spec.tokens_cap_ep = 2048


# =========================================================================
# TEST GROUP 2: classify_layers_from_config with mixed layer types
# =========================================================================


class TestClassifyLayersFromConfigMixed:
    """Test classify_layers_from_config with mixed full + SWA layers."""

    def test_classify_mixed_full_and_swa(self):
        """Classify 6 layers: 2 full + 4 SWA."""
        from sglang.srt.paras.layers.utils import classify_layers_from_config

        # Mock HF config with layer_types
        class MockConfig:
            num_hidden_layers = 6
            num_key_value_heads = 8
            num_attention_heads = 32
            hidden_size = 4096
            layer_types = [
                "full_attention",
                "full_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
            ]
            sliding_window = 4097  # sliding_window_size = 4096

        config = MockConfig()
        specs = classify_layers_from_config(
            config,
            tp_size=4,
            ep_tokens_full=1024,
            tp_tokens_full=4096,
            ep_tokens_swa=256,
            tp_tokens_swa=1024,
            ratio=1.0,
        )

        # Verify count
        assert len(specs) == 6

        # Verify kinds
        assert specs[0].kind == "full"
        assert specs[1].kind == "full"
        assert specs[2].kind == "swa"
        assert specs[3].kind == "swa"
        assert specs[4].kind == "swa"
        assert specs[5].kind == "swa"

        # Verify token capacities
        assert specs[0].tokens_cap_ep == 1024
        assert specs[0].tokens_cap_tp == 4096
        assert specs[2].tokens_cap_ep == 256
        assert specs[2].tokens_cap_tp == 1024

        # Verify sliding_window_size
        assert specs[0].sliding_window_size is None
        assert specs[2].sliding_window_size == 4096

        # Verify uniform num_kv_heads
        for spec in specs:
            assert spec.num_kv_heads == 8
            assert spec.head_dim == 128


# =========================================================================
# TEST GROUP 3: classify_layers_from_config with all-full attention
# =========================================================================


class TestClassifyLayersFromConfigAllFull:
    """Test classify_layers_from_config with all-full attention (no layer_types)."""

    def test_classify_all_full_no_layer_types(self):
        """Classify 4 layers as all-full when layer_types is absent."""
        from sglang.srt.paras.layers.utils import classify_layers_from_config

        class MockConfig:
            num_hidden_layers = 4
            num_key_value_heads = 16
            num_attention_heads = 64
            hidden_size = 8192
            # No layer_types attribute

        config = MockConfig()
        specs = classify_layers_from_config(
            config,
            tp_size=2,
            ep_tokens_full=2048,
            tp_tokens_full=8192,
            ep_tokens_swa=512,
            tp_tokens_swa=2048,
            ratio=1.0,
        )

        # Verify count
        assert len(specs) == 4

        # All should be full
        for i, spec in enumerate(specs):
            assert spec.kind == "full"
            assert spec.layer_id == i
            assert spec.tokens_cap_ep == 2048
            assert spec.tokens_cap_tp == 8192
            assert spec.sliding_window_size is None
            assert spec.num_kv_heads == 16
            assert spec.head_dim == 128


# =========================================================================
# TEST GROUP 4: classify_layers_from_config with all-SWA
# =========================================================================


class TestClassifyLayersFromConfigAllSWA:
    """Test classify_layers_from_config with all-SWA layers."""

    def test_classify_all_swa(self):
        """Classify 3 layers as all-SWA."""
        from sglang.srt.paras.layers.utils import classify_layers_from_config

        class MockConfig:
            num_hidden_layers = 3
            num_key_value_heads = 4
            num_attention_heads = 16
            hidden_size = 2048
            layer_types = ["sliding_attention", "sliding_attention", "sliding_attention"]
            sliding_window = 2049

        config = MockConfig()
        specs = classify_layers_from_config(
            config,
            tp_size=2,
            ep_tokens_full=512,
            tp_tokens_full=2048,
            ep_tokens_swa=128,
            tp_tokens_swa=512,
            ratio=1.0,
        )

        # Verify count
        assert len(specs) == 3

        # All should be SWA
        for i, spec in enumerate(specs):
            assert spec.kind == "swa"
            assert spec.layer_id == i
            assert spec.tokens_cap_ep == 128
            assert spec.tokens_cap_tp == 512
            assert spec.sliding_window_size == 2048
            assert spec.num_kv_heads == 4


# =========================================================================
# TEST GROUP 5: classify_layers_from_config without layer_types attribute
# =========================================================================


class TestClassifyLayersFromConfigNoLayerTypes:
    """Test classify_layers_from_config when layer_types is None."""

    def test_classify_layer_types_none(self):
        """Classify layers when layer_types is explicitly None."""
        from sglang.srt.paras.layers.utils import classify_layers_from_config

        class MockConfig:
            num_hidden_layers = 2
            num_key_value_heads = 8
            num_attention_heads = 32
            hidden_size = 4096
            layer_types = None  # Explicitly None

        config = MockConfig()
        specs = classify_layers_from_config(
            config,
            tp_size=1,
            ep_tokens_full=1024,
            tp_tokens_full=4096,
            ep_tokens_swa=256,
            tp_tokens_swa=1024,
            ratio=1.0,
        )

        # Should default to all-full
        assert len(specs) == 2
        for spec in specs:
            assert spec.kind == "full"
            assert spec.sliding_window_size is None


# =========================================================================
# TEST GROUP 6: G15 violation — mismatched num_kv_heads
# =========================================================================


class TestG15ViolationMismatchedHeads:
    """Test G15 validation: num_kv_heads must be uniform across layers."""

    def test_g15_violation_raises_value_error(self):
        """Mismatched num_kv_heads should raise ValueError with 'uniform' in message."""
        from sglang.srt.paras.layers.utils import validate_layer_specs, LayerCacheSpec

        # Create specs with mismatched num_kv_heads
        specs = [
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
                num_kv_heads=16,  # Mismatch!
                head_dim=128,
                sliding_window_size=None,
            ),
        ]

        # Should raise ValueError with "uniform" in message
        with pytest.raises(ValueError) as exc_info:
            validate_layer_specs(specs, tp_size=4)

        assert "uniform" in str(exc_info.value).lower()


# =========================================================================
# TEST GROUP 7: G7 violation — tp_size > num_kv_heads with SWA
# =========================================================================


class TestG7ViolationTPSizeExceedsHeads:
    """Test G7 validation: tp_size must be <= num_kv_heads for SWA layers."""

    def test_g7_violation_raises_value_error(self):
        """tp_size > num_kv_heads with SWA should raise ValueError with 'tp_size' in message."""
        from sglang.srt.paras.layers.utils import validate_layer_specs, LayerCacheSpec

        # Create specs with SWA and tp_size > num_kv_heads
        specs = [
            LayerCacheSpec(
                layer_id=0,
                kind="full",
                tokens_cap_ep=1024,
                tokens_cap_tp=4096,
                num_kv_heads=4,
                head_dim=128,
                sliding_window_size=None,
            ),
            LayerCacheSpec(
                layer_id=1,
                kind="swa",
                tokens_cap_ep=256,
                tokens_cap_tp=1024,
                num_kv_heads=4,
                head_dim=128,
                sliding_window_size=2048,
            ),
        ]

        # tp_size=8 > num_kv_heads=4 with SWA should raise
        with pytest.raises(ValueError) as exc_info:
            validate_layer_specs(specs, tp_size=8)

        assert "tp_size" in str(exc_info.value).lower()

    def test_g7_violation_in_classify_layers(self):
        """classify_layers_from_config should also enforce G7."""
        from sglang.srt.paras.layers.utils import classify_layers_from_config

        class MockConfig:
            num_hidden_layers = 2
            num_key_value_heads = 2  # Only 2 heads
            num_attention_heads = 8
            hidden_size = 2048
            layer_types = ["full_attention", "sliding_attention"]
            sliding_window = 2049

        config = MockConfig()

        # tp_size=4 > num_kv_heads=2 with SWA should raise
        with pytest.raises(ValueError) as exc_info:
            classify_layers_from_config(
                config,
                tp_size=4,  # Too large!
                ep_tokens_full=512,
                tp_tokens_full=2048,
                ep_tokens_swa=128,
                tp_tokens_swa=512,
                ratio=1.0,
            )

        assert "tp_size" in str(exc_info.value).lower()


# =========================================================================
# TEST GROUP 8: Budget parity — plan_hybrid_kv_budget formula verification
# =========================================================================


class TestPlanHybridKVBudgetParity:
    """Test plan_hybrid_kv_budget matches the formula from model_runner.py:1511-1515."""

    def test_budget_parity_ac8(self):
        """Verify budget formula: denominator = ratio * swa + full, full_max = int(total/denom), swa_max = int(full_max * ratio)."""
        from sglang.srt.paras.paras_memory_manager import plan_hybrid_kv_budget

        # Test case: total=10000, full_layers=10, swa_layers=20, ratio=0.25
        total_tokens = 10000
        full_layers_num = 10
        swa_layers_num = 20
        swa_full_tokens_ratio = 0.25

        full_max, swa_max = plan_hybrid_kv_budget(
            total_tokens, full_layers_num, swa_layers_num, swa_full_tokens_ratio
        )

        # Manual calculation per formula:
        # denominator = 0.25 * 20 + 10 = 5 + 10 = 15
        # full_max = int(10000 / 15) = int(666.666...) = 666
        # swa_max = int(666 * 0.25) = int(166.5) = 166
        expected_full_max = 666
        expected_swa_max = 166

        assert full_max == expected_full_max, (
            f"Expected full_max={expected_full_max}, got {full_max}"
        )
        assert swa_max == expected_swa_max, (
            f"Expected swa_max={expected_swa_max}, got {swa_max}"
        )

    def test_budget_all_full_layers(self):
        """Test budget when there are no SWA layers (all-full case)."""
        from sglang.srt.paras.paras_memory_manager import plan_hybrid_kv_budget

        total_tokens = 10000
        full_layers_num = 10
        swa_layers_num = 0
        swa_full_tokens_ratio = 1.0  # Ignored when swa_layers_num=0

        full_max, swa_max = plan_hybrid_kv_budget(
            total_tokens, full_layers_num, swa_layers_num, swa_full_tokens_ratio
        )

        # All-full shortcut: full_max = total / full_layers, swa_max = 0
        assert full_max == 1000
        assert swa_max == 0

    def test_budget_all_swa_layers(self):
        """Test budget when there are only SWA layers (all-SWA case)."""
        from sglang.srt.paras.paras_memory_manager import plan_hybrid_kv_budget

        total_tokens = 10000
        full_layers_num = 0
        swa_layers_num = 10
        swa_full_tokens_ratio = 0.5

        full_max, swa_max = plan_hybrid_kv_budget(
            total_tokens, full_layers_num, swa_layers_num, swa_full_tokens_ratio
        )

        # denominator = 0.5 * 10 + 0 = 5
        # full_max = int(10000 / 5) = 2000
        # swa_max = int(2000 * 0.5) = 1000
        assert full_max == 2000
        assert swa_max == 1000

    def test_budget_no_layers_raises(self):
        """Test that plan_hybrid_kv_budget raises when no layers are present."""
        from sglang.srt.paras.paras_memory_manager import plan_hybrid_kv_budget

        with pytest.raises(ValueError) as exc_info:
            plan_hybrid_kv_budget(10000, 0, 0, 1.0)

        assert "no layers" in str(exc_info.value).lower()

    def test_budget_invalid_ratio_raises(self):
        """Test that plan_hybrid_kv_budget raises when ratio <= 0 with SWA layers."""
        from sglang.srt.paras.paras_memory_manager import plan_hybrid_kv_budget

        with pytest.raises(ValueError) as exc_info:
            plan_hybrid_kv_budget(10000, 5, 5, 0.0)  # ratio=0 with SWA

        assert "ratio" in str(exc_info.value).lower()

    def test_budget_ratio_scaling(self):
        """Test budget with different ratio values."""
        from sglang.srt.paras.paras_memory_manager import plan_hybrid_kv_budget

        total_tokens = 10000
        full_layers_num = 5
        swa_layers_num = 5

        # Test ratio=0.5
        full_max_half, swa_max_half = plan_hybrid_kv_budget(
            total_tokens, full_layers_num, swa_layers_num, 0.5
        )
        # denominator = 0.5 * 5 + 5 = 7.5
        # full_max = int(10000 / 7.5) = 1333
        # swa_max = int(1333 * 0.5) = 666
        assert full_max_half == 1333
        assert swa_max_half == 666

        # Test ratio=0.25
        full_max_quarter, swa_max_quarter = plan_hybrid_kv_budget(
            total_tokens, full_layers_num, swa_layers_num, 0.25
        )
        # denominator = 0.25 * 5 + 5 = 6.25
        # full_max = int(10000 / 6.25) = 1600
        # swa_max = int(1600 * 0.25) = 400
        assert full_max_quarter == 1600
        assert swa_max_quarter == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
