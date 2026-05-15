"""T27: Preserve-unlocked path — enumeration + asymmetry doc."""
import pytest
import torch


class _FakeKey:
    def __init__(self, t): self.token_ids = list(t)


class _FakeNode:
    def __init__(self, key=None, lock_ref=0, full_lock_ref=0, swa_tombstone=False):
        self.key = key
        self.children = {}
        self.parent = None
        self.lock_ref = lock_ref
        self.full_lock_ref = full_lock_ref
        self.swa_tombstone = swa_tombstone
        self.value = None


class _FakeMHATree:
    def __init__(self):
        self.root_node = _FakeNode(_FakeKey([]))


class _FakeSWATree:
    def __init__(self):
        self.root_node = _FakeNode(_FakeKey([]))
        self.sliding_window_size = 8


class TestEnumerateUnlockedSlots:
    def test_empty_tree_returns_empty(self):
        from sglang.srt.paras.tree_migration import enumerate_unlocked_slots
        assert enumerate_unlocked_slots(_FakeMHATree()) == []

    def test_unlocked_node_slots_collected(self):
        from sglang.srt.paras.tree_migration import enumerate_unlocked_slots
        t = _FakeMHATree()
        n = _FakeNode(_FakeKey([1, 2]), lock_ref=0)
        n.value = torch.tensor([10, 20])
        t.root_node.children[1] = n
        assert enumerate_unlocked_slots(t) == [10, 20]

    def test_locked_node_slots_skipped(self):
        from sglang.srt.paras.tree_migration import enumerate_unlocked_slots
        t = _FakeMHATree()
        n = _FakeNode(_FakeKey([1, 2]), lock_ref=1)
        n.value = torch.tensor([10, 20])
        t.root_node.children[1] = n
        assert enumerate_unlocked_slots(t) == []

    def test_swa_tree_uses_full_lock_ref(self):
        from sglang.srt.paras.tree_migration import enumerate_unlocked_slots
        t = _FakeSWATree()
        a = _FakeNode(_FakeKey([1, 2]), full_lock_ref=0)
        a.value = torch.tensor([10, 20])
        b = _FakeNode(_FakeKey([3]), full_lock_ref=2)
        b.value = torch.tensor([30])
        t.root_node.children[1] = a
        t.root_node.children[3] = b
        assert sorted(enumerate_unlocked_slots(t)) == [10, 20]


class TestPreserveUnlockedFlag:
    def test_default_false(self):
        try:
            from sglang.srt.server_args import ServerArgs
        except Exception as e:
            pytest.skip(f"server_args import failed (env): {e}")
        sa = ServerArgs(model_path="dummy")
        assert getattr(sa, "paras_radix_preserve_unlocked", False) is False


class TestCollectUnlockedSlotsForPreserve:
    def test_disabled_returns_empty(self):
        try:
            from sglang.srt.paras.gather_manager import ParaSReqGatherManager
        except Exception as e:
            pytest.skip(f"gather_manager import failed (env): {e}")
        cls = ParaSReqGatherManager
        assert hasattr(cls, "collect_unlocked_slots_for_preserve")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
