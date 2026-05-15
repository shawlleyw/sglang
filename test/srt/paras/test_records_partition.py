"""T11: TP→EP rank-0 broadcast partition by req ownership."""
import pytest
from sglang.srt.paras.tree_migration import TreeRecord


def _partition_records_by_ownership(records, owned_token_lists):
    owned = []
    for rec in records:
        path = rec.full_token_path
        path_len = len(path)
        for tokens in owned_token_lists:
            if len(tokens) >= path_len and tokens[:path_len] == path:
                owned.append(rec)
                break
    return owned


def _rec(path, slots=None, extra_key=None, swa_tombstone=False):
    return TreeRecord(
        full_token_path=list(path),
        extra_key=extra_key,
        value_slots=slots or [100 + i for i in range(len(path))],
        swa_tombstone=swa_tombstone,
    )


class TestPartitionByOwnership:
    def test_empty_records_returns_empty(self):
        owned = _partition_records_by_ownership([], [[1, 2, 3]])
        assert owned == []

    def test_full_match_owned(self):
        records = [_rec([1, 2, 3])]
        owned = _partition_records_by_ownership(records, [[1, 2, 3, 4, 5]])
        assert len(owned) == 1
        assert owned[0].full_token_path == [1, 2, 3]

    def test_unrelated_path_dropped(self):
        records = [_rec([1, 2, 3]), _rec([7, 8, 9])]
        owned = _partition_records_by_ownership(records, [[1, 2, 3, 4, 5]])
        assert len(owned) == 1
        assert owned[0].full_token_path == [1, 2, 3]

    def test_no_owned_reqs_drops_all(self):
        records = [_rec([1, 2, 3])]
        owned = _partition_records_by_ownership(records, [])
        assert owned == []

    def test_tombstone_records_partition_correctly(self):
        records = [
            _rec([1, 2, 3], swa_tombstone=True),
            _rec([1, 2, 3, 4], swa_tombstone=False),
        ]
        owned = _partition_records_by_ownership(records, [[1, 2, 3, 4, 5]])
        assert len(owned) == 2

    def test_path_longer_than_owned_dropped(self):
        records = [_rec([1, 2, 3, 4, 5, 6])]
        owned = _partition_records_by_ownership(records, [[1, 2, 3, 4]])
        assert owned == []

    def test_multiple_partitions_match_first(self):
        records = [_rec([5, 6, 7])]
        owned = _partition_records_by_ownership(records, [[1, 2, 3], [5, 6, 7, 8, 9]])
        assert len(owned) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
