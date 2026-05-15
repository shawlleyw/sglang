"""T32: MHA round-trip CPU-mock integration test.

Exercises the full migration pipeline on CPU with mocked K/V pools:
  serialize → encode → decode → dedup → rebuild → recover_request equivalents
  → lock_ref recompute.

Verifies tree-state preservation across simulated EP↔TP round-trips and that
post-switch match_prefix returns prefixes that existed pre-switch.
"""
import pytest
import torch

from sglang.srt.paras.tree_migration import (
    TreeRecord,
    serialize_radix_cache,
    encode_records,
    decode_records,
    rebuild_radix_cache,
    canonicalize_post_rebuild,
    recompute_lock_refs,
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
        self.hash_value = None


class _FakeRadixCache:
    """Minimal MHA-like tree for round-trip testing."""
    def __init__(self):
        self.root_node = _FakeNode(_FakeKey([]))
        self.disable = False
        self.inc_calls = []

    def insert(self, key, value, **kwargs):
        tokens = list(key.token_ids) if hasattr(key, "token_ids") else list(key)
        if hasattr(value, "tolist"):
            value_list = value.tolist()
        else:
            value_list = list(value)
        path_aligned = len(value_list) == len(tokens)
        node = self.root_node
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok not in node.children:
                leaf_value = value_list[i:] if path_aligned else value_list
                child = _FakeNode(_FakeKey(tokens[i:]))
                child.parent = node
                child.value = torch.tensor(leaf_value, dtype=torch.int64)
                node.children[tok] = child
                return
            child = node.children[tok]
            child_tokens = list(child.key.token_ids)
            cpl = 0
            while (cpl < len(child_tokens) and i + cpl < len(tokens)
                   and child_tokens[cpl] == tokens[i + cpl]):
                cpl += 1
            if cpl == len(child_tokens):
                node = child
                i += cpl
                continue
            existing_value = child.value.tolist() if hasattr(child.value, "tolist") else list(child.value)
            intermediate = _FakeNode(_FakeKey(child_tokens[:cpl]))
            intermediate.parent = node
            intermediate.value = torch.tensor(existing_value[:cpl], dtype=torch.int64)
            child.key = _FakeKey(child_tokens[cpl:])
            child.value = torch.tensor(existing_value[cpl:], dtype=torch.int64)
            child.parent = intermediate
            intermediate.children[child_tokens[cpl]] = child
            node.children[tok] = intermediate
            node = intermediate
            i += cpl

    def match_prefix(self, key):
        node = self.root_node
        matched = []
        tokens = list(key.token_ids if hasattr(key, "token_ids") else key)
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok not in node.children:
                break
            child = node.children[tok]
            child_tokens = list(child.key.token_ids)
            cpl = 0
            while (cpl < len(child_tokens) and i + cpl < len(tokens)
                   and child_tokens[cpl] == tokens[i + cpl]):
                cpl += 1
            if cpl == 0:
                break
            child_value = child.value.tolist() if hasattr(child.value, "tolist") else list(child.value)
            matched.extend(child_value[:cpl])
            node = child
            i += cpl
            if cpl < len(child_tokens):
                break
        m = type("M", (), {})()
        m.device_indices = torch.tensor(matched, dtype=torch.int64)
        m.last_device_node = node
        m.last_host_node = node
        return m

    def inc_lock_ref(self, node):
        self.inc_calls.append(node)
        n = node
        while n is not None and n is not self.root_node:
            n.lock_ref += 1
            n = n.parent
        return None

    def evictable_size(self):
        return 0

    def protected_size(self):
        return 0


def _populate_tree(tree, prefixes_with_value_starts):
    for tokens, vstart in prefixes_with_value_starts:
        tree.insert(_FakeKey(tokens), torch.tensor([vstart + i for i in range(len(tokens))], dtype=torch.int64))


class TestMHARoundTrip:
    def test_serialize_decode_rebuild_round_trip(self):
        sender = _FakeRadixCache()
        _populate_tree(sender, [
            ([1, 2, 3, 4, 5, 6, 7, 8], 1000),  # shared system prompt
            ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 2000),
            ([1, 2, 3, 4, 5, 6, 7, 8, 11, 12], 3000),
        ])
        records = serialize_radix_cache(sender)
        blob = encode_records(records)
        decoded = decode_records(blob)
        assert len(decoded) == len(records)

        receiver = _FakeRadixCache()
        rebuild_radix_cache(receiver, decoded, remap_slot_idx=lambda s: s + 100000)
        canonicalize_post_rebuild(receiver)

        match = receiver.match_prefix(_FakeKey([1, 2, 3, 4, 5, 6, 7, 8]))
        assert len(match.device_indices) >= 8

    def test_post_switch_match_prefix_recovers_shared_prefix(self):
        sender = _FakeRadixCache()
        prefix = list(range(1, 51))  # 50-token shared prefix
        for i in range(8):
            tail = list(range(100 + 10 * i, 100 + 10 * (i + 1)))
            _populate_tree(sender, [(prefix + tail, 1000 * (i + 1))])
        records = serialize_radix_cache(sender)
        blob = encode_records(records)
        decoded = decode_records(blob)

        receiver = _FakeRadixCache()
        rebuild_radix_cache(receiver, decoded, remap_slot_idx=lambda s: s)
        canonicalize_post_rebuild(receiver)

        match = receiver.match_prefix(_FakeKey(prefix))
        assert len(match.device_indices) >= 50, (
            f"Expected >=50 cached prefix tokens post-rebuild; got {len(match.device_indices)}"
        )

    def test_round_trip_5_iterations_stable_record_set(self):
        sender = _FakeRadixCache()
        _populate_tree(sender, [
            ([1, 2, 3], 100),
            ([1, 2, 4], 200),
            ([5, 6, 7], 300),
        ])
        sigs_baseline = sorted(
            (tuple(r.full_token_path), tuple(r.value_slots))
            for r in serialize_radix_cache(sender)
        )

        current = sender
        for round_idx in range(5):
            records = serialize_radix_cache(current)
            decoded = decode_records(encode_records(records))
            new_tree = _FakeRadixCache()
            rebuild_radix_cache(new_tree, decoded, remap_slot_idx=lambda s: s)
            canonicalize_post_rebuild(new_tree)
            current = new_tree
            sigs = sorted(
                (tuple(r.full_token_path), tuple(r.value_slots))
                for r in serialize_radix_cache(current)
            )
            assert sigs == sigs_baseline, f"Drift at round {round_idx}: {len(sigs)} vs {len(sigs_baseline)}"

    def test_recompute_lock_refs_after_rebuild(self):
        sender = _FakeRadixCache()
        _populate_tree(sender, [([1, 2, 3, 4, 5], 100)])
        records = serialize_radix_cache(sender)
        receiver = _FakeRadixCache()
        rebuild_radix_cache(receiver, decode_records(encode_records(records)),
                            remap_slot_idx=lambda s: s)
        canonicalize_post_rebuild(receiver)

        match = receiver.match_prefix(_FakeKey([1, 2, 3, 4, 5]))
        class Req:
            tree_orphaned = False
            cache_protected_len = 0
        r = Req()
        r.last_node = match.last_device_node
        recompute_lock_refs(receiver, [r])

        assert r.last_node.lock_ref >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
