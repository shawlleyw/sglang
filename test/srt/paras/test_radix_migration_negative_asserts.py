"""
Test negative assertions for radix cache migration under ParaS.

These tests verify that incompatible features (EAGLE, HiRadix, CPP radix, page_size>1)
are properly rejected at startup when radix cache is enabled under ParaS.
"""

import os
import pytest
from sglang.srt.server_args import ServerArgs


class TestRadixMigrationNegativeAsserts:
    """Test suite for negative assertions in ParaS + radix cache configuration."""

    def test_eagle_with_paras_radix_raises(self):
        """Test that EAGLE speculative decoding raises AssertionError with ParaS + radix cache."""
        with pytest.raises(AssertionError, match="EAGLE"):
            ServerArgs(
                enable_paras_moe=True,
                enable_dp_attention=True,
                enable_dp_lm_head=True,
                paras_tp_size=4,
                disable_radix_cache=False,
                speculative_algorithm="EAGLE",
            ).check_server_args()

    def test_hi_radix_with_paras_radix_raises(self):
        """Test that HiRadixCache raises AssertionError with ParaS + radix cache."""
        with pytest.raises(AssertionError, match="hierarchical|HiRadix"):
            ServerArgs(
                enable_paras_moe=True,
                enable_dp_attention=True,
                enable_dp_lm_head=True,
                paras_tp_size=4,
                disable_radix_cache=False,
                enable_hierarchical_cache=True,
            ).check_server_args()

    def test_cpp_radix_with_paras_radix_raises(self, monkeypatch):
        """Test that CPP radix tree raises AssertionError with ParaS + radix cache."""
        monkeypatch.setenv("SGLANG_EXPERIMENTAL_CPP_RADIX_TREE", "1")
        with pytest.raises(AssertionError, match="CPP|cpp"):
            ServerArgs(
                enable_paras_moe=True,
                enable_dp_attention=True,
                enable_dp_lm_head=True,
                paras_tp_size=4,
                disable_radix_cache=False,
            ).check_server_args()

    def test_page_size_gt_1_with_paras_radix_raises(self):
        """Test that page_size > 1 raises AssertionError with ParaS + radix cache."""
        with pytest.raises(AssertionError, match="page_size"):
            ServerArgs(
                enable_paras_moe=True,
                enable_dp_attention=True,
                enable_dp_lm_head=True,
                paras_tp_size=4,
                disable_radix_cache=False,
                page_size=16,
            ).check_server_args()

    def test_new_flags_present(self):
        """Test that new flags are present with correct defaults."""
        args = ServerArgs()
        assert hasattr(args, "paras_radix_preserve_unlocked")
        assert hasattr(args, "paras_radix_migration_strict")
        assert args.paras_radix_preserve_unlocked is False
        assert args.paras_radix_migration_strict == "fail"

    def test_no_asserts_when_disable_radix_cache_true(self):
        """Test that incompatible features don't raise when disable_radix_cache=True."""
        # These should NOT raise because disable_radix_cache=True
        args = ServerArgs(
            enable_paras_moe=True,
            enable_dp_attention=True,
            enable_dp_lm_head=True,
            paras_tp_size=4,
            disable_radix_cache=True,
            enable_hierarchical_cache=True,
            speculative_algorithm="EAGLE",
            page_size=16,
        )
        # Should not raise
        args.check_server_args()
        assert args.disable_radix_cache is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
