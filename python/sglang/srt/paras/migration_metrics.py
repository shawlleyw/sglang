"""ParaS radix-cache migration metrics.

Module-level counters and timing trackers, observable via 
Scheduler.get_internal_state for production monitoring. Single-threaded scheduler
means no locks needed.
"""

from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field
import time
from typing import Dict


@dataclass
class MigrationMetrics:
    """Module-level metrics. Safe to mutate from scheduler event-loop thread only."""
    failures_total: int = 0
    fallbacks_total: int = 0
    serialize_ms_ema: float = 0.0   # exponential moving average
    remap_ms_ema: float = 0.0
    dedup_drop_count: int = 0
    preserve_unlocked_bytes: int = 0
    orphan_req_count: int = 0
    
    # EMA decay factor: new_ema = alpha * sample + (1-alpha) * old_ema
    _ema_alpha: float = 0.2
    
    def update_ema(self, attr: str, sample: float) -> None:
        current = getattr(self, attr)
        if current == 0.0:
            setattr(self, attr, sample)
        else:
            setattr(self, attr, self._ema_alpha * sample + (1 - self._ema_alpha) * current)
    
    def as_dict(self) -> Dict[str, float]:
        """For scheduler.get_internal_state JSON export."""
        return {
            "paras_radix_migration_failures_total": self.failures_total,
            "paras_radix_migration_fallbacks_total": self.fallbacks_total,
            "paras_radix_migration_serialize_ms_ema": self.serialize_ms_ema,
            "paras_radix_migration_remap_ms_ema": self.remap_ms_ema,
            "paras_radix_migration_dedup_drop_count": self.dedup_drop_count,
            "paras_radix_migration_preserve_unlocked_bytes": self.preserve_unlocked_bytes,
            "paras_radix_migration_orphan_req_count": self.orphan_req_count,
        }


# Module-level singleton
metrics = MigrationMetrics()


@contextmanager
def time_block(metric_attr: str):
    """Context manager: time the block and update the EMA on the given metric attribute."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        metrics.update_ema(metric_attr, elapsed_ms)
