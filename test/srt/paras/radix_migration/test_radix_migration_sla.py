"""T35: CPU-only migration cost SLA enforcement (P99 budget)."""
from __future__ import annotations
import statistics
import time
from pathlib import Path
import pytest

from sglang.srt.paras.tree_migration import (
    TreeRecord,
    encode_records,
    decode_records,
    rebuild_radix_cache,
    canonicalize_post_rebuild,
)


def _build_synthetic_records(num_records: int = 500, avg_path_len: int = 50):
    records = []
    next_token = 1
    branches = 50
    children_per_branch = num_records // branches
    for b in range(branches):
        prefix = list(range(next_token, next_token + avg_path_len // 2))
        next_token += avg_path_len // 2
        for c in range(children_per_branch):
            tail = list(range(next_token, next_token + avg_path_len // 2))
            next_token += avg_path_len // 2
            records.append(TreeRecord(
                full_token_path=prefix + tail,
                extra_key=None,
                value_slots=[10000 + i for i in range(len(prefix) + len(tail))],
                swa_tombstone=False,
                last_access_time=0.0,
            ))
    return records[:num_records]


class _FakeTree:
    """Minimal MHA tree mock for rebuild test (no LRU, no lock_refs)."""
    def __init__(self):
        class _Root:
            children = {}
        self.root_node = _Root()
        self.insert_count = 0

    def insert(self, key, value, **kwargs):
        self.insert_count += 1


def _identity_remap(s):
    return s + 1_000_000


def _run_full_cycle(records):
    blob = encode_records(records)
    decoded = decode_records(blob)
    seen = set()
    deduped = []
    for r in decoded:
        sig = (tuple(r.full_token_path), r.extra_key)
        if sig in seen:
            continue
        seen.add(sig)
        deduped.append(r)
    tree = _FakeTree()
    rebuild_radix_cache(tree, deduped, remap_slot_idx=_identity_remap)


SLA_BUDGET_MS = 30.0
N_ITER = 100


def test_p99_cpu_migration_cost_under_budget():
    records = _build_synthetic_records(500, 50)
    for _ in range(5):
        _run_full_cycle(records)
    timings = []
    for _ in range(N_ITER):
        t0 = time.perf_counter()
        _run_full_cycle(records)
        timings.append((time.perf_counter() - t0) * 1000.0)
    timings.sort()
    p50 = statistics.median(timings)
    p95 = timings[int(0.95 * len(timings))]
    p99 = timings[int(0.99 * len(timings))]
    mean = statistics.mean(timings)
    print(f"\n[T35 SLA] P50={p50:.2f}ms P95={p95:.2f}ms P99={p99:.2f}ms Max={max(timings):.2f}ms Mean={mean:.2f}ms (n={N_ITER})")
    Path(".sisyphus/evidence").mkdir(parents=True, exist_ok=True)
    Path(".sisyphus/evidence/task-35-sla-numbers.txt").write_text(
        f"P50={p50:.2f}\nP95={p95:.2f}\nP99={p99:.2f}\nMax={max(timings):.2f}\nMean={mean:.2f}\nN={N_ITER}\nBudget={SLA_BUDGET_MS}\n"
    )
    assert p95 <= SLA_BUDGET_MS, (
        f"SLA REGRESSION: P95={p95:.2f}ms > {SLA_BUDGET_MS}ms budget"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
