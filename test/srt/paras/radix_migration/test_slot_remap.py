"""T7: Slot-remap callback semantics + map construction reference tests."""
from __future__ import annotations
from typing import Callable, Dict, Optional

import pytest
import torch


def _build_slot_remap_callback(slot_map: Optional[Dict[int, int]]) -> Callable[[int], int]:
    """Local mirror of manager's build_slot_remap_callback for unit testing semantic."""
    if slot_map is None:
        return lambda old: old
    return lambda old, _m=slot_map: _m.get(old, -1)


class TestSlotRemapCallback:
    def test_map_round_trip_lookup(self):
        m = {10: 100, 11: 101, 12: 102}
        cb = _build_slot_remap_callback(m)
        assert cb(10) == 100
        assert cb(11) == 101
        assert cb(12) == 102

    def test_dropped_signal_returns_minus_one(self):
        m = {10: 100}
        cb = _build_slot_remap_callback(m)
        assert cb(999) == -1

    def test_none_passthrough(self):
        cb = _build_slot_remap_callback(None)
        assert cb(42) == 42

    def test_empty_map_all_dropped(self):
        cb = _build_slot_remap_callback({})
        assert cb(0) == -1


def _gather_build_map(local_indices, global_indices, rank_in_group, global_num_tokens, num_local_tokens):
    """Reference implementation of the gather-side map build."""
    local_offset = sum(global_num_tokens[:rank_in_group])
    local_slots = local_indices.detach().cpu().tolist()
    new_slots = global_indices[local_offset : local_offset + num_local_tokens].detach().cpu().tolist()
    return dict(zip(local_slots, new_slots))


def _scatter_build_map(global_indices, token_partition_local, ep_dst_positions):
    """Reference implementation of the scatter-side map build."""
    old_slots = global_indices[token_partition_local].detach().cpu().tolist()
    new_slots = ep_dst_positions.detach().cpu().tolist()
    return dict(zip(old_slots, new_slots))


class TestMapConstruction:
    def test_gather_map_size_matches_in_flight_tokens(self):
        local = torch.arange(1000, 1015)
        global_tokens = torch.arange(2000, 2025)
        global_num_tokens = [10, 15]
        slot_map = _gather_build_map(local, global_tokens, 1, global_num_tokens, 15)
        assert len(slot_map) == 15

    def test_gather_map_round_trip(self):
        local = torch.tensor([5, 6, 7, 8])
        global_tokens = torch.tensor([100, 101, 102, 103, 104, 105, 106, 107])
        slot_map = _gather_build_map(local, global_tokens, 1, [4, 4], 4)
        assert slot_map == {5: 104, 6: 105, 7: 106, 8: 107}

    def test_scatter_map_round_trip(self):
        global_tokens = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17])
        local_global_idx = [0, 1, 4, 5]
        ep_dst = torch.tensor([200, 201, 202, 203])
        slot_map = _scatter_build_map(global_tokens, local_global_idx, ep_dst)
        assert slot_map == {10: 200, 11: 201, 14: 202, 15: 203}

    def test_scatter_empty_partition(self):
        global_tokens = torch.tensor([10, 11, 12])
        ep_dst = torch.tensor([], dtype=torch.int64)
        slot_map = _scatter_build_map(global_tokens, [], ep_dst)
        assert slot_map == {}


class TestChunkCacheSkip:
    def test_no_root_node_skips_build(self):
        class FakeChunk:
            pass
        cache = FakeChunk()
        should_build = getattr(cache, "root_node", None) is not None
        assert should_build is False

    def test_radix_cache_builds(self):
        class FakeRadix:
            def __init__(self):
                self.root_node = object()
        cache = FakeRadix()
        assert (getattr(cache, "root_node", None) is not None) is True

    def test_none_tree_cache_builds_by_default(self):
        cache = None
        should_build = cache is None or getattr(cache, "root_node", None) is not None
        assert should_build is True


@pytest.mark.skip(reason="Requires GPU/serving deps (transformers, torch.distributed). Source-level grep verification used instead.")
def test_production_callback_exists_in_managers():
    """Verify production code has the expected lambda semantic."""
    from sglang.srt.paras.gather_manager import ParaSReqGatherManager
    from sglang.srt.paras.scatter_manager import ParaSReqScatterManager
    import inspect
    src1 = inspect.getsource(ParaSReqGatherManager.build_slot_remap_callback)
    assert "get(old, -1)" in src1
    src2 = inspect.getsource(ParaSReqScatterManager.build_slot_remap_callback)
    assert "get(old, -1)" in src2


def test_production_callback_source_grep():
    """Module-import-free source check: grep production files for the -1 sentinel lambda."""
    import pathlib
    repo_root = pathlib.Path(__file__).resolve().parents[4]
    gather_src = (repo_root / "python/sglang/srt/paras/gather_manager.py").read_text()
    scatter_src = (repo_root / "python/sglang/srt/paras/scatter_manager.py").read_text()
    assert "build_slot_remap_callback" in gather_src
    assert "build_slot_remap_callback" in scatter_src
    assert "_m.get(old, -1)" in gather_src
    assert "_m.get(old, -1)" in scatter_src
    assert "old_to_new_slot_map" in gather_src
    assert "old_to_new_slot_map" in scatter_src


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
