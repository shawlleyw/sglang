"""Per-second metrics sampler for ParaS rollout benchmarking.

Spawns a daemon thread on tp_rank==0 that samples the scheduler's
gathered state every 1.0s and writes one CSV row per sample.

Columns:
    timestamp_iso, elapsed_s, mode, running_reqs, waiting_reqs,
    decode_tokens_per_sec, prefill_tokens_per_sec

mode is 'EP' or 'TP' (read from scheduler.paras_parallelism_config).
Throughput is computed from deltas of the monotonic lifetime counters
exposed on scheduler.last_batch.global_total_*_tokens.

Activated by passing --paras-metrics-file <path> to the SGLang server.
On non-zero ranks, start() is a no-op (only rank 0 writes).
"""

from __future__ import annotations

import atexit
import csv
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ParasMetricsSampler:
    CSV_HEADER = [
        "timestamp_iso",
        "elapsed_s",
        "mode",
        "running_reqs",
        "waiting_reqs",
        "decode_tokens_per_sec",
        "prefill_tokens_per_sec",
    ]

    def __init__(self, scheduler, output_path: str, interval_sec: float = 1.0):
        self.scheduler = scheduler
        self.output_path = output_path
        self.interval_sec = interval_sec
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0 = time.time()
        self._prev_decode_total = 0
        self._prev_prefill_total = 0
        self._prev_mode: Optional[str] = None
        self._fh: Any = None
        self._writer: Any = None

    def start(self) -> None:
        # Distributed-correctness gate: only rank 0 writes the metrics file.
        if getattr(self.scheduler, "tp_rank", 0) != 0:
            logger.debug("ParasMetricsSampler: tp_rank != 0, no-op")
            return

        out_dir = os.path.dirname(self.output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        self._fh = open(self.output_path, "w", buffering=1)  # buffering=1 => line-buffered
        self._writer = csv.writer(self._fh)
        self._writer.writerow(self.CSV_HEADER)

        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="ParasMetricsSampler"
        )
        self._thread.start()
        atexit.register(self.stop)
        logger.info(f"ParasMetricsSampler started: writing to {self.output_path}")

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=2.0)
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass
        logger.info(f"ParasMetricsSampler stopped: {self.output_path}")

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_sec):
            try:
                self._sample_once()
            except Exception:
                # Daemon-thread isolation: never let a sampler bug kill the scheduler.
                logger.exception("ParasMetricsSampler: sample failed")

    def _sample_once(self) -> None:
        scheduler = self.scheduler
        batch = getattr(scheduler, "last_batch", None)

        paras_config = getattr(scheduler, "paras_parallelism_config", None)
        if paras_config is not None:
            mode = paras_config
        else:
            # Static mode (no ParaS): enable_dp_attention <=> "EP" shape data plane.
            sa = getattr(scheduler, "server_args", None)
            if sa is not None and getattr(sa, "enable_dp_attention", False):
                mode = "EP"
            else:
                mode = "TP"

        # Source selection mirrors paras_auto_observe in scheduler_paras_mixin:
        # EP mode -> rank 0 holds only its DP slice, so sum the all-gather output;
        # TP mode -> unified data plane, rank 0's local view IS the global view.
        if mode == "EP" and batch is not None and getattr(batch, "global_running_reqs", None):
            running = int(sum(batch.global_running_reqs))
            waiting = int(sum(batch.global_waiting_reqs))
            decode_total = int(sum(batch.global_total_decode_tokens))
            prefill_total = int(sum(batch.global_total_prefill_tokens))
        else:
            rb = getattr(scheduler, "running_batch", None)
            running = len(rb.reqs) if rb is not None else 0
            waiting = len(getattr(scheduler, "waiting_queue", []) or [])
            decode_total = getattr(scheduler, "total_decode_tokens_lifetime", 0) or 0
            prefill_total = getattr(scheduler, "total_prefill_tokens_lifetime", 0) or 0

        # PARAS-BURSTY-PATCH: re-anchor on mode change (incl. first sample).
        # TP mode reads scheduler.total_*_lifetime (one rank's counter);
        # EP mode reads sum(batch.global_total_*) (all 8 ranks' counters).
        # Without re-anchoring, the first sample after a switch computes
        # a delta across two incompatible counter scales -> millions tok/s.
        if mode != self._prev_mode:
            self._prev_decode_total = decode_total
            self._prev_prefill_total = prefill_total
            self._prev_mode = mode

        decode_delta = decode_total - self._prev_decode_total
        prefill_delta = prefill_total - self._prev_prefill_total
        decode_tps = decode_delta / self.interval_sec if decode_delta > 0 else 0.0
        prefill_tps = prefill_delta / self.interval_sec if prefill_delta > 0 else 0.0

        now = time.time()
        row = [
            datetime.now(timezone.utc).isoformat(),
            f"{now - self._t0:.3f}",
            mode,
            running,
            waiting,
            f"{decode_tps:.2f}",
            f"{prefill_tps:.2f}",
        ]
        self._writer.writerow(row)

        self._prev_decode_total = decode_total
        self._prev_prefill_total = prefill_total
