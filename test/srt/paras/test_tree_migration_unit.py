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

from sglang.srt.paras.tree_migration import serialize_radix_cache, TreeRecord


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
