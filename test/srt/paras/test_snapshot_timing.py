"""T26: regression test for snapshot timing invariant.

Verifies that in both gather_manager and scatter_manager, the source_full_to_swa_mapping
clone() happens BEFORE paras_resize_and_clear and BEFORE _tighten_swa_pool_to_in_window
in source-line order.
"""
from pathlib import Path
import re


def _line_indices(source: str, patterns: list) -> list:
    """For each regex pattern, return the LINE NUMBER (1-indexed) of the first match."""
    out = []
    for pat in patterns:
        m = re.search(pat, source, flags=re.MULTILINE)
        if m is None:
            out.append(None)
        else:
            out.append(source[: m.start()].count("\n") + 1)
    return out


REPO = Path(__file__).resolve().parents[3]


def test_gather_manager_snapshot_timing():
    src = (REPO / "python/sglang/srt/paras/gather_manager.py").read_text()
    snap, resize, tighten = _line_indices(
        src,
        [
            r"full_to_swa_index_mapping\.clone\(\)",
            r"token_to_kv_pool_allocator\.paras_resize_and_clear",
            r"self\._tighten_swa_pool_to_in_window\(\)",
        ],
    )
    assert snap is not None, "snapshot clone not found in gather_manager.py"
    assert resize is not None, "token_to_kv_pool_allocator.paras_resize_and_clear not found in gather_manager.py"
    if tighten is not None:
        assert snap < tighten, f"snapshot (line {snap}) must precede _tighten (line {tighten})"
    assert snap < resize, f"snapshot (line {snap}) must precede token_to_kv_pool_allocator.paras_resize_and_clear (line {resize})"


def test_scatter_manager_snapshot_timing():
    src = (REPO / "python/sglang/srt/paras/scatter_manager.py").read_text()
    snap, resize, tighten = _line_indices(
        src,
        [
            r"full_to_swa_index_mapping\.clone\(\)",
            r"token_to_kv_pool_allocator\.paras_resize_and_clear",
            r"self\._tighten_swa_pool_to_in_window\(\)",
        ],
    )
    assert snap is not None, "snapshot clone not found in scatter_manager.py"
    assert resize is not None, "token_to_kv_pool_allocator.paras_resize_and_clear not found in scatter_manager.py"
    if tighten is not None:
        assert snap < tighten, f"snapshot (line {snap}) must precede _tighten (line {tighten})"
    assert snap < resize, f"snapshot (line {snap}) must precede token_to_kv_pool_allocator.paras_resize_and_clear (line {resize})"


def test_invariant_comment_present_gather():
    src = (REPO / "python/sglang/srt/paras/gather_manager.py").read_text()
    assert "INVARIANT" in src or "snapshot MUST precede" in src, \
        "gather_manager.py missing invariant comment"


def test_invariant_comment_present_scatter():
    src = (REPO / "python/sglang/srt/paras/scatter_manager.py").read_text()
    assert "INVARIANT" in src or "snapshot MUST precede" in src, \
        "scatter_manager.py missing invariant comment"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
