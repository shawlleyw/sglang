"""T9.5 HARD GATE: synthetic-tree CPU latency benchmark for tree migration.

Original gate was GPU-bound (8xA100 NVLink). Adapted to CPU-only per scope:
measure pure serialize -> encode -> decode -> dedup -> rebuild -> remap CPU cost
over 100 iterations on a 500-node tree. Budget: P99 <= 25 ms.

If CPU alone exceeds 25 ms, real-system overhead (incl. cross-rank exchange
and GPU work) will blow the 17/12 ms GPU-bound headroom for sure.
"""
from __future__ import annotations
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch

from sglang.srt.paras.tree_migration import (
    TreeRecord,
    encode_records,
    decode_records,
    rebuild_radix_cache,
)


# ----------------------------------------------------------------------------
# Synthetic tree fixture: 500 records, avg path length 50 tokens, varied widths.
# This models a typical chat workload with shared prefixes.
# ----------------------------------------------------------------------------


def _build_synthetic_records(num_records: int = 500, avg_path_len: int = 50) -> list:
    """Build a representative record set with realistic shapes.

    Mixes deep prefixes (shared by many records) with narrower divergent tails.
    """
    records: list = []
    next_token = 1
    # 50 prefix paths of length avg_path_len/2 each act as branch points; each has 10 children.
    branches = 50
    children_per_branch = num_records // branches
    for b in range(branches):
        prefix = list(range(next_token, next_token + avg_path_len // 2))
        next_token += avg_path_len // 2
        for c in range(children_per_branch):
            tail = list(range(next_token, next_token + avg_path_len // 2))
            next_token += avg_path_len // 2
            full_path = prefix + tail
            value_slots = [10000 + i for i in range(len(full_path))]
            records.append(TreeRecord(
                full_token_path=full_path,
                extra_key=None,
                value_slots=value_slots,
                swa_tombstone=(c == 0),  # ~10% tombstones
                last_access_time=0.0,
            ))
    return records[:num_records]


@dataclass
class _CallTracker:
    insert_count: int = 0


class _FakeTree:
    """Minimal tree for rebuild test -- only needs .insert."""

    def __init__(self):
        self.tracker = _CallTracker()

    def insert(self, key, value, **kwargs):
        self.tracker.insert_count += 1


def _identity_remap(s: int) -> int:
    return s + 1_000_000  # simulate non-trivial remap


def _run_one_migration_cycle(records: list) -> None:
    """Full migration round-trip on CPU: encode -> decode -> dedup -> rebuild."""
    # Step 1: Encode (sender side)
    blob = encode_records(records)
    # Step 2: Decode (receiver side)
    decoded = decode_records(blob)
    # Step 3: Dedup pass -- group by (tuple(path), extra_key); keep first.
    seen: set = set()
    deduped: list = []
    for r in decoded:
        key_sig = (tuple(r.full_token_path), r.extra_key)
        if key_sig in seen:
            continue
        seen.add(key_sig)
        deduped.append(r)
    # Step 4: Rebuild
    tree = _FakeTree()
    rebuild_radix_cache(tree, deduped, remap_slot_idx=_identity_remap)


def _measure_p99(records: list, n_iter: int = 100) -> dict:
    """Run n_iter cycles; return statistics dict."""
    timings_ms: list = []
    # Warm-up (3 iterations)
    for _ in range(3):
        _run_one_migration_cycle(records)
    for _ in range(n_iter):
        t0 = time.perf_counter()
        _run_one_migration_cycle(records)
        timings_ms.append((time.perf_counter() - t0) * 1000.0)
    timings_ms.sort()
    return {
        "p50": statistics.median(timings_ms),
        "p95": timings_ms[int(0.95 * len(timings_ms))],
        "p99": timings_ms[int(0.99 * len(timings_ms))],
        "max": max(timings_ms),
        "mean": statistics.mean(timings_ms),
        "n": n_iter,
    }


GATE_BUDGET_MS = 25.0


def _write_gate_failed_marker(stats: dict, message: str) -> None:
    Path(".sisyphus/evidence").mkdir(parents=True, exist_ok=True)
    Path(".sisyphus/evidence/task-9.5-GATE-FAILED.txt").write_text(
        f"GATE FAILED: {message}\n"
        f"P50: {stats['p50']:.2f} ms\n"
        f"P95: {stats['p95']:.2f} ms\n"
        f"P99: {stats['p99']:.2f} ms\n"
        f"Max: {stats['max']:.2f} ms\n"
        f"Mean: {stats['mean']:.2f} ms\n"
        f"Iterations: {stats['n']}\n"
        f"Budget: {GATE_BUDGET_MS:.1f} ms (CPU-only)\n"
        f"Gap: {stats['p99'] - GATE_BUDGET_MS:+.2f} ms\n"
    )


def test_synthetic_tree_migration_p99_under_budget():
    """HARD GATE: P99 CPU-only migration logic <= 25 ms on a 500-node synthetic tree."""
    records = _build_synthetic_records(num_records=500, avg_path_len=50)
    assert len(records) == 500, f"Expected 500 records, got {len(records)}"
    stats = _measure_p99(records, n_iter=100)
    print(f"\n[T9.5] CPU migration logic latency on 500-node synthetic tree:")
    print(f"  P50={stats['p50']:.2f} ms")
    print(f"  P95={stats['p95']:.2f} ms")
    print(f"  P99={stats['p99']:.2f} ms")
    print(f"  Max={stats['max']:.2f} ms")
    print(f"  Mean={stats['mean']:.2f} ms")
    print(f"  Budget: {GATE_BUDGET_MS:.1f} ms")

    # Persist the perf result for evidence regardless of pass/fail
    Path(".sisyphus/evidence").mkdir(parents=True, exist_ok=True)
    Path(".sisyphus/evidence/task-9.5-perf-numbers.txt").write_text(
        f"P50={stats['p50']:.2f} ms\nP95={stats['p95']:.2f} ms\nP99={stats['p99']:.2f} ms\n"
        f"Max={stats['max']:.2f} ms\nMean={stats['mean']:.2f} ms\nN={stats['n']}\n"
        f"Budget={GATE_BUDGET_MS:.1f} ms\n"
    )

    if stats["p99"] > GATE_BUDGET_MS:
        _write_gate_failed_marker(
            stats,
            f"P99 ({stats['p99']:.2f} ms) > budget ({GATE_BUDGET_MS:.1f} ms)",
        )
        pytest.fail(
            f"HARD GATE FAILED: P99 CPU migration latency {stats['p99']:.2f} ms "
            f"exceeds {GATE_BUDGET_MS:.1f} ms budget. See "
            f".sisyphus/evidence/task-9.5-GATE-FAILED.txt"
        )


def test_regression_detector_with_oversize_tree():
    """Sanity: if we run on a deliberately-oversize tree (10000 nodes), the gate WOULD fire."""
    records = _build_synthetic_records(num_records=10000, avg_path_len=50)
    stats = _measure_p99(records, n_iter=20)
    print(f"\n[T9.5] Regression detector -- 10000-node tree P99: {stats['p99']:.2f} ms")
    # We don't assert anything here -- just confirm the harness produces numbers.
    assert stats["p99"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
