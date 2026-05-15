"""T36: Round-trip migration determinism test."""
import pytest
import torch

from sglang.srt.paras.tree_migration import (
    TreeRecord,
    encode_records,
    decode_records,
    rebuild_radix_cache,
)


def _build_records(seed: int = 42, n_records: int = 100):
    """Build a deterministic synthetic record set seeded by `seed`."""
    import random
    rng = random.Random(seed)
    records = []
    for i in range(n_records):
        path_len = rng.randint(5, 30)
        path = [rng.randint(1, 10000) for _ in range(path_len)]
        records.append(TreeRecord(
            full_token_path=path,
            extra_key=None,
            value_slots=[100000 + i * path_len + j for j in range(path_len)],
            swa_tombstone=(i % 7 == 0),
            last_access_time=0.0,
        ))
    return records


def _record_signature(r):
    """Stable signature ignoring last_access_time."""
    return (tuple(r.full_token_path), r.extra_key, tuple(r.value_slots), r.swa_tombstone)


def _all_signatures(records):
    return sorted(_record_signature(r) for r in records)


class TestRoundTripDeterminism:
    def test_encode_decode_idempotent(self):
        records = _build_records(seed=42, n_records=50)
        once = decode_records(encode_records(records))
        twice = decode_records(encode_records(once))
        assert _all_signatures(once) == _all_signatures(twice)

    def test_5_round_trips_stable(self):
        records = _build_records(seed=99, n_records=100)
        sigs_baseline = _all_signatures(records)
        current = records
        for i in range(5):
            current = decode_records(encode_records(current))
            assert _all_signatures(current) == sigs_baseline, f"Drift at round {i}"

    def test_rebuild_call_pattern_deterministic(self):
        """Rebuild from same records (in different orders) produces same insert call sequence."""
        records = _build_records(seed=7, n_records=30)
        records_alt = list(reversed(records))

        def collect(recs):
            calls = []
            class FT:
                def __init__(self): pass
                def insert(self, key, value, **kwargs):
                    path = list(key.token_ids if hasattr(key, "token_ids") else key)
                    calls.append(tuple(path))
            ft = FT()
            ft.root_node = type("R", (), {"children": {}})()
            rebuild_radix_cache(ft, recs, remap_slot_idx=lambda s: s)
            return calls

        a = collect(records)
        b = collect(records_alt)
        assert a == b, "rebuild call sequence not deterministic across record orderings"

    def test_tombstone_field_preserved_across_rounds(self):
        records = _build_records(seed=1, n_records=20)
        original_tombstones = [(_record_signature(r)[:2], r.swa_tombstone) for r in records]
        current = records
        for _ in range(3):
            current = decode_records(encode_records(current))
        round_tombstones = [(_record_signature(r)[:2], r.swa_tombstone) for r in current]
        assert sorted(original_tombstones) == sorted(round_tombstones)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
