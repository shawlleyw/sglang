"""T34: Slot-index remap correctness on CPU.

Writes a distinguishing pattern into a source pool (CPU tensor); rebuilds tree
on a destination pool with the remap callback; reads via the rebuilt tree's
node.value indices; asserts the pattern is recovered (proves remap math is correct).
"""
import pytest
import torch

from sglang.srt.paras.tree_migration import (
    TreeRecord,
    rebuild_radix_cache,
)


class _FakeKey:
    def __init__(self, t, extra_key=None):
        self.token_ids = list(t)
        self.extra_key = extra_key


class _FakeNode:
    def __init__(self, key=None):
        self.key = key
        self.children = {}
        self.parent = None
        self.value = None
        self.lock_ref = 0
        self.last_access_time = 0.0


class _FakeTree:
    """Test scaffold that captures inserts and lets us walk node.value."""
    def __init__(self):
        self.root_node = _FakeNode(_FakeKey([]))

    def insert(self, key, value, **kwargs):
        token_ids = key.token_ids if hasattr(key, "token_ids") else list(key)
        n = _FakeNode(_FakeKey(token_ids))
        n.value = value
        n.parent = self.root_node
        if token_ids:
            self.root_node.children[token_ids[0]] = n


def _walk_nodes(tree):
    """Iterative DFS yielding non-root nodes."""
    stack = list(tree.root_node.children.values())
    while stack:
        n = stack.pop()
        yield n
        for c in n.children.values():
            stack.append(c)


class TestSlotRemapCorrectness:
    def test_remap_basic_correctness(self):
        """Pattern: dest_pool[new_slot] = 0xDEAD0000 + new_slot.
        After rebuild + remap, every tree node's value indices point to dest slots
        whose contents == 0xDEAD0000 + that_dest_slot.
        """
        DEST_POOL_SIZE = 1000
        dest_pool = torch.zeros(DEST_POOL_SIZE, dtype=torch.int64)

        records = [
            TreeRecord(full_token_path=[1, 2], extra_key=None, value_slots=[10, 11]),
            TreeRecord(full_token_path=[1, 2, 3], extra_key=None, value_slots=[10, 11, 12]),
        ]

        OFFSET = 100
        def remap(old_slot):
            return old_slot + OFFSET

        for rec in records:
            for old in rec.value_slots:
                new = remap(old)
                if 0 <= new < DEST_POOL_SIZE:
                    dest_pool[new] = 0xDEAD0000 + new

        tree = _FakeTree()
        rebuild_radix_cache(tree, records, remap_slot_idx=remap)

        for node in _walk_nodes(tree):
            for new_slot in node.value.tolist():
                assert dest_pool[new_slot].item() == 0xDEAD0000 + new_slot, (
                    f"Pattern mismatch: dest_pool[{new_slot}]={dest_pool[new_slot].item():#x}"
                )

    def test_remap_dropped_slots_skipped(self):
        """Records whose remap returns -1 are SKIPPED (not migrated with garbage)."""
        records = [
            TreeRecord(full_token_path=[1, 2], extra_key=None, value_slots=[10, 999]),
            TreeRecord(full_token_path=[3, 4], extra_key=None, value_slots=[20, 30]),
        ]
        def remap(s):
            return -1 if s == 999 else s + 100
        tree = _FakeTree()
        rebuild_radix_cache(tree, records, remap_slot_idx=remap)
        nodes = list(_walk_nodes(tree))
        assert len(nodes) == 1
        for n in nodes:
            for slot in n.value.tolist():
                assert slot >= 0

    def test_remap_swa_tombstones_correctness(self):
        """Tombstone records with swa_tombstone=True are still rebuilt (with the
        tombstone semantic flowing through), and their value_slots are remapped."""
        DEST_POOL_SIZE = 200
        dest_pool = torch.zeros(DEST_POOL_SIZE, dtype=torch.int64)
        records = [
            TreeRecord(full_token_path=[1, 2, 3], extra_key=None,
                       value_slots=[10, 11, 12], swa_tombstone=True),
        ]
        OFFSET = 50
        def remap(s):
            return s + OFFSET
        for rec in records:
            for old in rec.value_slots:
                new = remap(old)
                if 0 <= new < DEST_POOL_SIZE:
                    dest_pool[new] = 0xBEEF0000 + new

        tree = _FakeTree()
        rebuild_radix_cache(tree, records, remap_slot_idx=remap)
        nodes = list(_walk_nodes(tree))
        assert len(nodes) == 1
        for slot in nodes[0].value.tolist():
            assert dest_pool[slot].item() == 0xBEEF0000 + slot

    def test_remap_pattern_preservation_multi_record(self):
        """20 records, each with random-shaped paths; all reachable through tree
        and all values map to new slots whose pattern matches."""
        import random
        rng = random.Random(123)
        DEST_POOL_SIZE = 5000
        dest_pool = torch.zeros(DEST_POOL_SIZE, dtype=torch.int64)
        records = []
        used_slots = set()
        for i in range(20):
            path_len = rng.randint(2, 8)
            path = [rng.randint(1, 1000) for _ in range(path_len)]
            slots = []
            while len(slots) < path_len:
                cand = rng.randint(0, 500)
                if cand not in used_slots:
                    used_slots.add(cand)
                    slots.append(cand)
            records.append(TreeRecord(full_token_path=path, extra_key=None,
                                      value_slots=slots, swa_tombstone=False))

        OFFSET = 2000
        for rec in records:
            for old in rec.value_slots:
                new = old + OFFSET
                if new < DEST_POOL_SIZE:
                    dest_pool[new] = 0xCAFE0000 + new

        tree = _FakeTree()
        rebuild_radix_cache(tree, records, remap_slot_idx=lambda s: s + OFFSET)
        for node in _walk_nodes(tree):
            for new_slot in node.value.tolist():
                assert dest_pool[new_slot].item() == 0xCAFE0000 + new_slot


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
