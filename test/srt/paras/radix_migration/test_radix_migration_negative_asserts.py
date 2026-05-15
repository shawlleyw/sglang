"""
Test negative assertions for radix cache migration under ParaS.

These tests verify that incompatible features (EAGLE, HiRadix, CPP radix, page_size>1)
are properly rejected when radix cache is enabled under ParaS.

NOTE: ServerArgs.__post_init__ short-circuits when model_path == "dummy",
so we construct with model_path="dummy" and invoke ``_check_paras_config()``
directly to exercise the assertion logic without requiring a real model
config / HF lookup.
"""

import pytest
from sglang.srt.server_args import ServerArgs


def _make_paras_args(**overrides):
    """Build a ServerArgs instance configured for ParaS, with sane defaults
    for fields ``_check_paras_config()`` reads. Caller can override any field
    via kwargs.
    """
    args = ServerArgs(model_path="dummy")
    # Core ParaS prerequisites checked by _check_paras_config
    args.enable_paras_moe = True
    args.enable_dp_attention = True
    args.enable_dp_lm_head = True
    args.paras_tp_size = 4
    args.tp_size = 4
    args.dp_size = 4
    # ParaS requires chunked_prefill_size <= 0 (set to a passing value
    # by default so the 4 negative asserts fire first).
    args.chunked_prefill_size = -1
    # Default radix cache fields to a passing config; tests flip these.
    args.disable_radix_cache = False
    args.enable_hierarchical_cache = False
    args.speculative_algorithm = None
    args.page_size = 1
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


class TestRadixMigrationNegativeAsserts:
    """Test suite for negative assertions in ParaS + radix cache configuration.

    Each test asserts that ``_check_paras_config()`` raises ``AssertionError``
    *and* that the error message uniquely names the offending feature
    (HiRadix / CPP / EAGLE / page_size).
    """

    def test_eagle_with_paras_radix_raises(self):
        """EAGLE + ParaS + radix cache: AssertionError mentions EAGLE/speculative."""
        args = _make_paras_args(speculative_algorithm="EAGLE")
        with pytest.raises(AssertionError, match="(?i)eagle|speculative"):
            args._check_paras_config()

    def test_hi_radix_with_paras_radix_raises(self):
        """HiRadix + ParaS + radix cache: AssertionError mentions hierarchical/HiRadix."""
        args = _make_paras_args(enable_hierarchical_cache=True)
        with pytest.raises(AssertionError, match="(?i)hierarchical|hi.?radix"):
            args._check_paras_config()

    def test_cpp_radix_with_paras_radix_raises(self, monkeypatch):
        """CPP radix env + ParaS + radix cache: AssertionError mentions CPP/cpp."""
        monkeypatch.setenv("SGLANG_EXPERIMENTAL_CPP_RADIX_TREE", "1")
        args = _make_paras_args()
        with pytest.raises(AssertionError, match="(?i)cpp|c\\+\\+|experimental"):
            args._check_paras_config()

    def test_page_size_gt_1_with_paras_radix_raises(self):
        """page_size>1 + ParaS + radix cache: AssertionError mentions page_size."""
        args = _make_paras_args(page_size=16)
        with pytest.raises(AssertionError, match="(?i)page_size|page size"):
            args._check_paras_config()

    def test_new_flags_present(self):
        """New ParaS-radix flags exist on ServerArgs with correct defaults."""
        args = ServerArgs(model_path="dummy")
        assert hasattr(args, "paras_radix_preserve_unlocked")
        assert hasattr(args, "paras_radix_migration_strict")
        assert args.paras_radix_preserve_unlocked is False
        assert args.paras_radix_migration_strict == "fail"

    def test_no_asserts_when_disable_radix_cache_true(self):
        """Incompatible features don't raise when disable_radix_cache=True
        (the 4 conditional asserts are skipped by the guard)."""
        args = _make_paras_args(
            disable_radix_cache=True,
            enable_hierarchical_cache=True,
            speculative_algorithm="EAGLE",
            page_size=16,
        )
        # Should not raise — the guard `if not self.disable_radix_cache:`
        # short-circuits the 4 incompatible-feature asserts.
        args._check_paras_config()
        assert args.disable_radix_cache is True

    def test_radix_cache_works_with_paras_after_lift(self):
        """T30: enable_paras_moe=True + disable_radix_cache=False should be ACCEPTED
        (the original ParaS-blocks-radix-cache assertion was lifted)."""
        args = _make_paras_args()
        # All defaults are migration-compatible; should not raise.
        args._check_paras_config()
        assert args.disable_radix_cache is False
        assert args.enable_paras_moe is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
