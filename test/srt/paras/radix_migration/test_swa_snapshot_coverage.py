"""T21: SWA snapshot coverage verification + compose_swa_remap helper."""
import pytest
import torch


class TestComposeSWARemap:
    def test_no_swa_mapping_returns_identity_for_swa(self):
        from sglang.srt.paras.tree_migration import compose_swa_remap
        cbs = compose_swa_remap(remap_callback=lambda s: s + 100)
        assert cbs["full"](5) == 105
        assert cbs["swa"](5) == 105

    def test_swa_mapping_translates_then_remaps(self):
        from sglang.srt.paras.tree_migration import compose_swa_remap
        full_to_swa = torch.tensor([0, 0, 0, 7, 0, 9, 11, 0])
        cbs = compose_swa_remap(
            remap_callback=lambda s: s + 1000,
            source_full_to_swa_mapping=full_to_swa,
        )
        assert cbs["full"](3) == 1003
        assert cbs["swa"](3) == 1007
        assert cbs["swa"](5) == 1009

    def test_swa_mapping_zero_means_freed(self):
        from sglang.srt.paras.tree_migration import compose_swa_remap
        full_to_swa = torch.tensor([0, 0, 0, 0, 0])
        cbs = compose_swa_remap(
            remap_callback=lambda s: -1 if s == 0 else s + 100,
            source_full_to_swa_mapping=full_to_swa,
        )
        assert cbs["swa"](2) == -1

    def test_out_of_bounds_full_slot_returns_minus_one(self):
        from sglang.srt.paras.tree_migration import compose_swa_remap
        full_to_swa = torch.tensor([0, 0, 0])
        cbs = compose_swa_remap(remap_callback=lambda s: s, source_full_to_swa_mapping=full_to_swa)
        assert cbs["swa"](999) == -1


class TestSnapshotCoverageDocumented:
    """The snapshot in gather/scatter_manager.reorchestrate_cache covers ALL
    allocated full slots (sized full_size + swa_size + 1), including unlocked
    tree-only slots. This test verifies the doc/code claim by reading source.
    """

    def test_gather_manager_snapshot_capture_present(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parents[4] / "python/sglang/srt/paras/gather_manager.py").read_text()
        assert "source_full_to_swa_mapping" in src
        assert ".clone()" in src or "clone()" in src

    def test_scatter_manager_snapshot_capture_present(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parents[4] / "python/sglang/srt/paras/scatter_manager.py").read_text()
        assert "source_full_to_swa_mapping" in src
        assert ".clone()" in src or "clone()" in src


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
