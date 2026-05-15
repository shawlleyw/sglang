"""T33: SWA round-trip CPU-mock integration test with tombstone preservation."""
import pytest
import torch

from sglang.srt.paras.tree_migration import (
    TreeRecord,
    serialize_swa_radix_cache,
    encode_records,
    decode_records,
    rebuild_radix_cache,
    canonicalize_post_rebuild,
    normalize_lru_lists,
)


class _FakeKey:
    def __init__(self, t, extra_key=None):
        self.token_ids = list(t)
        self.extra_key = extra_key


class _FakeSWANode:
    def __init__(self, key=None, swa_tombstone=False):
        self.key = key
        self.children = {}
        self.parent = None
        self.value = None
        self.full_lock_ref = 0
        self.swa_lock_ref = 0
        self.swa_tombstone = swa_tombstone
        self.last_access_time = 0.0
        self.hash_value = None


class _FakeLRU:
    def __init__(self):
        self.entries = []
    def insert_mru(self, n):
        if n in self.entries:
            self.entries.remove(n)
        self.entries.insert(0, n)
    def remove(self, n):
        if n in self.entries:
            self.entries.remove(n)


class _FakeSWATree:
    def __init__(self, sliding_window_size=8):
        self.root_node = _FakeSWANode(_FakeKey([]))
        self.sliding_window_size = sliding_window_size
        self.disable = False
        self.full_lru_list = _FakeLRU()
        self.swa_lru_list = _FakeLRU()
        self.full_evictable_size_ = 0
        self.swa_evictable_size_ = 0

    def insert(self, key, value, swa_evicted_seqlen: int = 0, **kwargs):
        if hasattr(key, "token_ids"):
            token_ids = list(key.token_ids)
        else:
            token_ids = list(key)
        is_tombstone = swa_evicted_seqlen >= len(token_ids) and swa_evicted_seqlen > 0
        node = _FakeSWANode(_FakeKey(token_ids), swa_tombstone=is_tombstone)
        node.value = value.clone() if hasattr(value, "clone") else torch.tensor(list(value), dtype=torch.int64)
        node.parent = self.root_node
        if token_ids:
            self.root_node.children[token_ids[0]] = node
        self.full_lru_list.insert_mru(node)
        if not is_tombstone:
            self.swa_lru_list.insert_mru(node)
        self.full_evictable_size_ += len(value)
        if not is_tombstone:
            self.swa_evictable_size_ += len(value)

    def evictable_size(self):
        return self.swa_evictable_size_

    def protected_size(self):
        return 0

    def full_evictable_size(self):
        return self.full_evictable_size_

    def full_protected_size(self):
        return 0

    def swa_evictable_size(self):
        return self.swa_evictable_size_

    def swa_protected_size(self):
        return 0


class TestSWARoundTrip:
    def test_tombstone_records_round_trip(self):
        records_pre = [
            TreeRecord(full_token_path=[1, 2, 3, 4],
                       extra_key=None,
                       value_slots=[10, 20, 30, 40],
                       swa_tombstone=True),
            TreeRecord(full_token_path=[5, 6, 7, 8, 9],
                       extra_key=None,
                       value_slots=[50, 60, 70, 80, 90],
                       swa_tombstone=False),
        ]
        decoded = decode_records(encode_records(records_pre))
        assert len(decoded) == 2
        assert decoded[0].swa_tombstone is True
        assert decoded[1].swa_tombstone is False

    def test_rebuild_preserves_tombstone_in_tree(self):
        records = [
            TreeRecord(full_token_path=[1, 2, 3, 4],
                       extra_key=None,
                       value_slots=[10, 20, 30, 40],
                       swa_tombstone=True),
            TreeRecord(full_token_path=[5, 6, 7, 8, 9, 10],
                       extra_key=None,
                       value_slots=[50, 60, 70, 80, 90, 100],
                       swa_tombstone=False),
        ]
        receiver = _FakeSWATree(sliding_window_size=4)
        rebuild_radix_cache(receiver, records, remap_slot_idx=lambda s: s)
        tombstone_count = sum(1 for c in receiver.root_node.children.values() if c.swa_tombstone)
        assert tombstone_count >= 1

    def test_normalize_lru_excludes_tombstones(self):
        records = [
            TreeRecord(full_token_path=[1, 2],
                       extra_key=None,
                       value_slots=[10, 20],
                       swa_tombstone=True),
            TreeRecord(full_token_path=[3, 4],
                       extra_key=None,
                       value_slots=[30, 40],
                       swa_tombstone=False),
        ]
        receiver = _FakeSWATree(sliding_window_size=4)
        rebuild_radix_cache(receiver, records, remap_slot_idx=lambda s: s)
        normalize_lru_lists(receiver)
        for entry in receiver.swa_lru_list.entries:
            assert entry.swa_tombstone is False, "Tombstone in swa_lru_list — invariant violated"

    def test_5_round_trips_preserve_tombstone_count(self):
        records = [
            TreeRecord(full_token_path=[i, i + 1],
                       extra_key=None,
                       value_slots=[10 * i, 10 * i + 1],
                       swa_tombstone=(i % 2 == 0))
            for i in range(1, 11)
        ]
        original_tombstone_count = sum(1 for r in records if r.swa_tombstone)
        current = records
        for _ in range(5):
            current = decode_records(encode_records(current))
        new_tombstone_count = sum(1 for r in current if r.swa_tombstone)
        assert original_tombstone_count == new_tombstone_count


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
