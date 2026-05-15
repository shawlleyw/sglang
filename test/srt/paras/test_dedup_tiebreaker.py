"""T10: EP->TP records dedup with lock-ref tiebreaker."""
import pytest
from sglang.srt.paras.tree_migration import TreeRecord


def _dedup_records_with_lockref(per_rank_records, in_flight_slot_set):
    """Standalone copy of the dedup logic for unit testing without ParaSReqGatherManager."""
    from collections import defaultdict
    bucket = defaultdict(list)
    for rank_idx, records in enumerate(per_rank_records):
        for r in records:
            key = (tuple(r.full_token_path), r.extra_key)
            bucket[key].append((rank_idx, r))

    kept = []
    dropped = 0
    for key, candidates in bucket.items():
        if len(candidates) == 1:
            kept.append(candidates[0][1])
            continue
        with_in_flight = [
            (rank_idx, r)
            for rank_idx, r in candidates
            if any(s in in_flight_slot_set for s in r.value_slots)
        ]
        if with_in_flight:
            with_in_flight.sort(key=lambda pair: pair[0])
            kept.append(with_in_flight[0][1])
            dropped += len(candidates) - 1
        else:
            candidates.sort(key=lambda pair: pair[0])
            kept.append(candidates[0][1])
            dropped += len(candidates) - 1
    return kept, dropped


def _rec(path, slots, extra_key=None, swa_tombstone=False):
    return TreeRecord(
        full_token_path=list(path),
        extra_key=extra_key,
        value_slots=list(slots),
        swa_tombstone=swa_tombstone,
    )


class TestDedupTiebreaker:
    def test_no_collision_keeps_all(self):
        per_rank = [[_rec([1, 2], [10])], [_rec([3, 4], [20])]]
        kept, dropped = _dedup_records_with_lockref(per_rank, set())
        assert len(kept) == 2
        assert dropped == 0

    def test_lockref_preference_basic(self):
        """Two ranks have the same (path, extra_key); only rank 1's slots are in-flight.
        Tiebreaker: rank 1's record wins despite higher rank index."""
        per_rank = [
            [_rec([1, 2], [10])],
            [_rec([1, 2], [20])],
        ]
        kept, dropped = _dedup_records_with_lockref(per_rank, in_flight_slot_set={20})
        assert len(kept) == 1
        assert kept[0].value_slots == [20]
        assert dropped == 1

    def test_no_in_flight_falls_back_to_lex_min_rank(self):
        per_rank = [
            [_rec([1, 2], [10])],
            [_rec([1, 2], [20])],
        ]
        kept, dropped = _dedup_records_with_lockref(per_rank, in_flight_slot_set=set())
        assert len(kept) == 1
        assert kept[0].value_slots == [10]
        assert dropped == 1

    def test_dedup_metric_increments(self):
        """3 ranks with 1 collision: 2 dropped."""
        per_rank = [
            [_rec([1, 2], [10])],
            [_rec([1, 2], [20])],
            [_rec([1, 2], [30])],
        ]
        kept, dropped = _dedup_records_with_lockref(per_rank, in_flight_slot_set=set())
        assert len(kept) == 1
        assert dropped == 2

    def test_extra_key_distinct_no_dedup(self):
        per_rank = [
            [_rec([1, 2], [10], extra_key="lora_a")],
            [_rec([1, 2], [20], extra_key="lora_b")],
        ]
        kept, dropped = _dedup_records_with_lockref(per_rank, in_flight_slot_set=set())
        assert len(kept) == 2
        assert dropped == 0

    def test_in_flight_with_multi_slot_record(self):
        """Record with multiple value_slots: in-flight if ANY slot is in the set."""
        per_rank = [
            [_rec([1, 2], [10, 20, 30])],
            [_rec([1, 2], [40, 50, 60])],
        ]
        kept, _ = _dedup_records_with_lockref(per_rank, in_flight_slot_set={50})
        assert kept[0].value_slots == [40, 50, 60]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
