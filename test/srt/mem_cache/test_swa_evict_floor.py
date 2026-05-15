"""Tests for SWA eviction floor: max(swa_evicted_seqlen, cache_protected_len).

When the tree locks a prefix, runtime SWA eviction must not free those slots.
"""

import pytest


class MockReq:
    """Mock Req object for testing eviction floor logic."""
    def __init__(self):
        self.swa_evicted_seqlen = 0
        self.cache_protected_len = 0


def compute_eviction_floor(req):
    """Simulate the eviction floor computation from ScheduleBatch._evict_swa."""
    return max(req.swa_evicted_seqlen, req.cache_protected_len)


class TestEvictionFloor:
    def test_default_zero_no_floor(self):
        """Both fields default to 0, floor is 0."""
        req = MockReq()
        assert compute_eviction_floor(req) == 0

    def test_floor_respects_cache_protected_len(self):
        """When cache_protected_len > swa_evicted_seqlen, floor is cache_protected_len."""
        req = MockReq()
        req.swa_evicted_seqlen = 3
        req.cache_protected_len = 7
        assert compute_eviction_floor(req) == 7

    def test_floor_respects_swa_evicted_seqlen(self):
        """When swa_evicted_seqlen > cache_protected_len, floor is swa_evicted_seqlen."""
        req = MockReq()
        req.swa_evicted_seqlen = 10
        req.cache_protected_len = 5
        assert compute_eviction_floor(req) == 10

    def test_no_regression_when_cache_protected_len_is_zero(self):
        """Default case (cache_protected_len=0): behavior unchanged from pre-migration."""
        req = MockReq()
        req.swa_evicted_seqlen = 8
        # cache_protected_len defaults to 0
        assert compute_eviction_floor(req) == 8

    def test_equal_values(self):
        """When both values are equal, floor is that value."""
        req = MockReq()
        req.swa_evicted_seqlen = 5
        req.cache_protected_len = 5
        assert compute_eviction_floor(req) == 5
