"""T4: Unit tests for tree_migration serialize/deserialize/rebuild.

Uses lightweight stand-in classes for RadixKey / TreeNode to avoid pulling the
full sglang import graph (which has runtime-only deps like zmq, uvloop, etc.).
The serializer only reads ``.children``, ``.key``, ``.value`` and
``.last_access_time`` from nodes, so a minimal duck-typed mock is sufficient
and -- in fact -- preferable because it isolates serializer correctness from
unrelated infrastructure.
"""
import sys

import pytest
import torch

from sglang.srt.paras.tree_migration import (
    serialize_radix_cache,
    serialize_swa_radix_cache,
    TreeRecord,
)


class FakeKey:
    def __init__(self, token_ids, extra_key=None):
        self.token_ids = list(token_ids)
        self.extra_key = extra_key


class FakeNode:
    def __init__(self, key=None, value=None, last_access_time=0.0):
        self.children = {}
        self.parent = None
        self.key = key
        self.value = value
        self.last_access_time = last_access_time


class FakeTree:
    def __init__(self, root):
        self.root_node = root


def _child_key(token_ids, extra_key=None):
    plain = token_ids[0]
    return plain if extra_key is None else (extra_key, plain)


def _add_child(parent, token_ids, value_list, extra_key=None, last_access_time=0.0):
    child = FakeNode(
        key=FakeKey(token_ids, extra_key=extra_key),
        value=torch.tensor(value_list, dtype=torch.int64),
        last_access_time=last_access_time,
    )
    child.parent = parent
    parent.children[_child_key(token_ids, extra_key)] = child
    return child


class TestSerializeEmptyTree:
    def test_empty_tree_returns_empty_records(self):
        root = FakeNode(key=FakeKey([], extra_key=None), value=[])
        tree = FakeTree(root)
        records = serialize_radix_cache(tree)
        assert records == []


class TestSerializeBranchingTree:
    def test_record_count_and_paths(self):
        root = FakeNode(key=FakeKey([], extra_key=None), value=[])
        shared = _add_child(root, [1, 2], [10, 20])
        _add_child(shared, [3], [30])
        _add_child(shared, [4], [40])
        tree = FakeTree(root)
        records = serialize_radix_cache(tree)
        full_paths = sorted(tuple(r.full_token_path) for r in records)
        assert (1, 2) in full_paths
        assert (1, 2, 3) in full_paths
        assert (1, 2, 4) in full_paths
        assert len(records) == 3


class TestSerializeFullTokenPathAccuracy:
    def test_paths_match_dfs(self):
        root = FakeNode(key=FakeKey([], extra_key=None), value=[])
        node = root
        for tok in [1, 2, 3, 4, 5]:
            node = _add_child(node, [tok], [tok * 10])
        tree = FakeTree(root)
        records = serialize_radix_cache(tree)
        assert len(records) == 5
        max_path_len = max(len(r.full_token_path) for r in records)
        assert max_path_len == 5
        deepest = max(records, key=lambda r: len(r.full_token_path))
        assert deepest.full_token_path == [1, 2, 3, 4, 5]

    def test_value_slots_and_extra_key_preserved(self):
        root = FakeNode(key=FakeKey([], extra_key=None), value=[])
        _add_child(root, [7, 8, 9], [70, 80, 90], extra_key="lora-1",
                   last_access_time=42.5)
        tree = FakeTree(root)
        records = serialize_radix_cache(tree)
        assert len(records) == 1
        rec = records[0]
        assert rec.full_token_path == [7, 8, 9]
        assert rec.value_slots == [70, 80, 90]
        assert rec.extra_key == "lora-1"
        assert rec.last_access_time == 42.5
        assert rec.swa_tombstone is False
        assert rec.host_value is None


class TestSerializeNoRecursion:
    def test_deep_tree_no_stack_overflow(self):
        """Depth-1000 chain forces iterative traversal under recursion-limit-50."""
        root = FakeNode(key=FakeKey([], extra_key=None), value=[])
        node = root
        for i in range(1, 1001):
            node = _add_child(node, [i], [i])
        tree = FakeTree(root)

        old_limit = sys.getrecursionlimit()
        try:
            sys.setrecursionlimit(50)
            records = serialize_radix_cache(tree)
            assert len(records) == 1000
            max_path_len = max(len(r.full_token_path) for r in records)
            assert max_path_len == 1000
            deepest = max(records, key=lambda r: len(r.full_token_path))
            assert deepest.full_token_path == list(range(1, 1001))
        finally:
            sys.setrecursionlimit(old_limit)


class FakeSWANode:
    """SWA-flavored mock node: adds swa_tombstone to the FakeNode shape."""

    def __init__(self, key=None, value=None, last_access_time=0.0, swa_tombstone=False):
        self.children = {}
        self.parent = None
        self.key = key
        self.value = value
        self.last_access_time = last_access_time
        self.swa_tombstone = swa_tombstone


def _add_swa_child(parent, token_ids, value_list, extra_key=None,
                   last_access_time=0.0, swa_tombstone=False):
    child = FakeSWANode(
        key=FakeKey(token_ids, extra_key=extra_key),
        value=torch.tensor(value_list, dtype=torch.int64) if value_list else None,
        last_access_time=last_access_time,
        swa_tombstone=swa_tombstone,
    )
    child.parent = parent
    parent.children[_child_key(token_ids, extra_key)] = child
    return child


class TestSerializeSWA:
    """T5: SWA tree serializer with tombstone preservation.

    Uses lightweight mocks (FakeSWANode) to avoid the SWARadixCache import,
    which pulls in triton via the allocator module. The serializer only reads
    `.children`, `.key`, `.value`, `.last_access_time`, and `.swa_tombstone`
    on nodes, so the mock is faithful to the production behavior.
    """

    def test_serialize_swa_basic(self):
        """No tombstones: every record has swa_tombstone=False."""
        root = FakeSWANode(key=FakeKey([], extra_key=None), value=[])
        _add_swa_child(root, [1, 2, 3, 4], [10, 20, 30, 40])
        tree = FakeTree(root)
        records = serialize_swa_radix_cache(tree)
        assert len(records) == 1
        for r in records:
            assert r.swa_tombstone is False
        assert records[0].full_token_path == [1, 2, 3, 4]
        assert records[0].value_slots == [10, 20, 30, 40]

    def test_serialize_swa_with_tombstones(self):
        """Tombstoned nodes round-trip with swa_tombstone=True; siblings are unaffected."""
        root = FakeSWANode(key=FakeKey([], extra_key=None), value=[])
        # Branch B of PR #17220: tombstone(1..4) -> non-tombstone(5..10)
        tomb = _add_swa_child(root, [1, 2, 3, 4], [10, 20, 30, 40],
                              swa_tombstone=True)
        _add_swa_child(tomb, [5, 6, 7, 8, 9, 10],
                       [50, 60, 70, 80, 90, 100], swa_tombstone=False)
        tree = FakeTree(root)
        records = serialize_swa_radix_cache(tree)
        tombstone_records = [r for r in records if r.swa_tombstone]
        non_tombstone_records = [r for r in records if not r.swa_tombstone]
        assert len(tombstone_records) >= 1, (
            "Expected at least one tombstone record; "
            f"records: {[(r.full_token_path, r.swa_tombstone) for r in records]}"
        )
        assert len(non_tombstone_records) >= 1, (
            "Expected at least one non-tombstone record; "
            f"records: {[(r.full_token_path, r.swa_tombstone) for r in records]}"
        )
        tomb_rec = tombstone_records[0]
        assert tomb_rec.full_token_path == [1, 2, 3, 4]

    def test_serialize_swa_traverses_through_tombstones(self):
        """Tombstone child must NOT short-circuit DFS; descendants still emit records.

        Critical for _match_prefix_helper's W-distance rule: even when the
        immediate prefix is tombstoned, deeper matches may still hit
        in-window slots.
        """
        root = FakeSWANode(key=FakeKey([], extra_key=None), value=[])
        tomb = _add_swa_child(root, [1, 2, 3, 4], [10, 20, 30, 40],
                              swa_tombstone=True)
        _add_swa_child(tomb, [5, 6, 7, 8, 9, 10],
                       [50, 60, 70, 80, 90, 100], swa_tombstone=False)
        tree = FakeTree(root)
        records = serialize_swa_radix_cache(tree)
        full_paths = {tuple(r.full_token_path) for r in records}
        assert (1, 2, 3, 4) in full_paths, "Tombstone record missing"
        assert (1, 2, 3, 4, 5, 6, 7, 8, 9, 10) in full_paths, (
            "Tombstone descendant missing: DFS short-circuited through tombstone"
        )
        assert len(records) == 2

    def test_serialize_swa_no_recursion(self):
        """Depth-1000 tombstone chain forces iterative traversal under recursion-limit-50."""
        root = FakeSWANode(key=FakeKey([], extra_key=None), value=[])
        node = root
        for i in range(1, 1001):
            node = _add_swa_child(node, [i], [i],
                                  swa_tombstone=(i % 2 == 0))
        tree = FakeTree(root)

        old_limit = sys.getrecursionlimit()
        try:
            sys.setrecursionlimit(50)
            records = serialize_swa_radix_cache(tree)
            assert len(records) == 1000
            tomb_count = sum(1 for r in records if r.swa_tombstone)
            assert tomb_count == 500
        finally:
            sys.setrecursionlimit(old_limit)

    def test_serialize_swa_empty_tree(self):
        root = FakeSWANode(key=FakeKey([], extra_key=None), value=[])
        tree = FakeTree(root)
        records = serialize_swa_radix_cache(tree)
        assert records == []

    def test_serialize_swa_preserves_extra_key(self):
        """LoRA / cache_salt key fragments survive serialization."""
        root = FakeSWANode(key=FakeKey([], extra_key=None), value=[])
        _add_swa_child(root, [7, 8, 9], [70, 80, 90], extra_key="lora-2",
                       last_access_time=99.0, swa_tombstone=True)
        tree = FakeTree(root)
        records = serialize_swa_radix_cache(tree)
        assert len(records) == 1
        r = records[0]
        assert r.extra_key == "lora-2"
        assert r.swa_tombstone is True
        assert r.last_access_time == 99.0
        assert r.host_value is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
