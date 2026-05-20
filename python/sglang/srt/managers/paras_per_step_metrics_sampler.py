"""Per-forward-step metrics sampler for ParaS rollout benchmarking.

Records one CSV row per scheduler forward step, capturing GPU-side forward
latency (time from first kernel to last kernel of the forward, measured
with paired ``torch.cuda.Event``). Event-completion is queried in a
background drain thread; the scheduler's main thread never blocks on
``synchronize``. This makes the measurement valid regardless of whether
``--disable-overlap-schedule`` is set: the event timestamps reflect GPU
execution, not host wall-clock.

Schedule overhead (Python scheduling, batch construction, kernel-launch
dispatch) is excluded by construction: the events bracket only the GPU
kernel sequence on whatever stream the forward runs on.

Columns:
    step_idx                monotonic, from scheduler.forward_ct
    timestamp_iso           UTC ISO timestamp at record() call
    elapsed_s               seconds since sampler.start()
    mode                    "EP" or "TP" (from scheduler.paras_parallelism_config,
                            or inferred from server_args.enable_dp_attention for
                            non-ParaS servers)
    forward_mode            "EXTEND" / "DECODE" / "IDLE" / "MIXED" (from
                            batch.forward_mode.name)
    batch_size_local        len(batch.reqs) at record() call
    batch_size_global       In EP / DP-attention mode, sum(batch.global_running_reqs);
                            otherwise equals batch_size_local
    num_tokens              For EXTEND: batch.extend_num_tokens
                            Otherwise: batch_size_local
    step_latency_ms         start_event.elapsed_time(end_event), in milliseconds.
                            GPU-side timing (first kernel start to last kernel
                            end on the forward stream).
    running_reqs            len(scheduler.running_batch.reqs) at record() call
    waiting_reqs            len(scheduler.waiting_queue) at record() call

Activated by ``--paras-per-step-metrics-file <path>``. Only tp_rank 0
writes: non-zero ranks no-op in ``start()`` and ``record()``.
"""

from __future__ import annotations

import atexit
import collections
import csv
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ParasPerStepMetricsSampler:
    CSV_HEADER = [
        "step_idx",
        "timestamp_iso",
        "elapsed_s",
        "mode",
        "forward_mode",
        "batch_size_local",
        "batch_size_global",
        "num_tokens",
        "step_latency_ms",
        "running_reqs",
        "waiting_reqs",
    ]

    def __init__(self, scheduler, output_path: str):
        self.scheduler = scheduler
        self.output_path = output_path
        self._t0: Optional[float] = None
        self._fh: Any = None
        self._writer: Any = None
        self._active: bool = False
        self._pending: "collections.deque[tuple]" = collections.deque()
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._drain_thread: Optional[threading.Thread] = None

    def is_active(self) -> bool:
        """True iff this rank is the writer rank AND start() succeeded.

        Used by Scheduler.run_batch to gate per-step Event creation so
        non-writer ranks pay zero overhead.
        """
        return self._active

    def start(self) -> None:
        if getattr(self.scheduler, "tp_rank", 0) != 0:
            logger.debug("ParasPerStepMetricsSampler: tp_rank != 0, no-op")
            return

        out_dir = os.path.dirname(self.output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # buffering=1 (line-buffered) so partial progress is visible to
        # tail -f even if the server crashes mid-step.
        self._fh = open(self.output_path, "w", buffering=1)
        self._writer = csv.writer(self._fh)
        self._writer.writerow(self.CSV_HEADER)
        self._t0 = time.time()
        self._active = True

        self._drain_thread = threading.Thread(
            target=self._drain_loop, daemon=True, name="ParasPerStepDrain"
        )
        self._drain_thread.start()
        atexit.register(self.stop)
        logger.info(
            f"ParasPerStepMetricsSampler started: writing per-step CSV to "
            f"{self.output_path}"
        )

    def stop(self) -> None:
        if not self._active:
            return
        self._active = False
        self._stop_evt.set()
        if self._drain_thread is not None:
            self._drain_thread.join(timeout=10.0)
            self._drain_thread = None
        self._force_drain_remaining()
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None
            self._writer = None
        logger.info(
            f"ParasPerStepMetricsSampler stopped: {self.output_path}"
        )

    def record(self, batch, start_event, end_event) -> None:
        """Snapshot batch metadata + enqueue events for deferred drain.

        Called from Scheduler.run_batch immediately after recording
        ``end_event`` on the forward stream. Returns immediately; the GPU
        is NOT synchronized here. The background drain thread polls
        end_event.query() and computes elapsed_time once the GPU has
        actually finished the forward.

        Safe to call on any rank: no-op on non-writer ranks because
        ``start()`` did not set _active there.
        """
        if not self._active:
            return
        try:
            snapshot = self._snapshot(batch)
        except Exception:
            logger.exception("ParasPerStepMetricsSampler: snapshot failed")
            return
        with self._lock:
            self._pending.append((snapshot, start_event, end_event))

    def _snapshot(self, batch) -> dict:
        scheduler = self.scheduler

        paras_config = getattr(scheduler, "paras_parallelism_config", None)
        if paras_config is not None:
            mode = paras_config
        else:
            sa = getattr(scheduler, "server_args", None)
            if sa is not None and getattr(sa, "enable_dp_attention", False):
                mode = "EP"
            else:
                mode = "TP"

        fwd = getattr(batch, "forward_mode", None) if batch is not None else None
        forward_mode = fwd.name if fwd is not None else "NONE"

        reqs = getattr(batch, "reqs", None) if batch is not None else None
        bs_local = len(reqs) if reqs else 0
        bs_global = bs_local
        if mode == "EP" and batch is not None:
            global_running = getattr(batch, "global_running_reqs", None)
            if global_running:
                try:
                    bs_global = int(sum(global_running))
                except Exception:
                    bs_global = bs_local

        if forward_mode == "EXTEND":
            num_tokens = int(getattr(batch, "extend_num_tokens", 0) or 0)
        else:
            num_tokens = bs_local

        rb = getattr(scheduler, "running_batch", None)
        running_reqs = len(rb.reqs) if rb is not None and getattr(rb, "reqs", None) else 0
        waiting_reqs = len(getattr(scheduler, "waiting_queue", []) or [])

        now = time.time()
        elapsed = now - self._t0 if self._t0 is not None else 0.0
        return {
            "step_idx": getattr(scheduler, "forward_ct", 0),
            "timestamp_iso": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": elapsed,
            "mode": mode,
            "forward_mode": forward_mode,
            "batch_size_local": bs_local,
            "batch_size_global": bs_global,
            "num_tokens": num_tokens,
            "running_reqs": running_reqs,
            "waiting_reqs": waiting_reqs,
        }

    def _drain_loop(self) -> None:
        # Lock-free polling: only the drain thread mutates the queue head
        # via popleft, and only after end_event.query() returns True.
        while not self._stop_evt.is_set():
            ready = None
            with self._lock:
                if self._pending and self._pending[0][2].query():
                    ready = self._pending.popleft()
            if ready is None:
                # Empty queue or head event not ready yet. Short sleep
                # keeps polling responsive without burning CPU.
                self._stop_evt.wait(0.005)
                continue
            self._emit_row(ready)

    def _force_drain_remaining(self) -> None:
        # Called from stop(). Any remaining events must be synchronously
        # waited on so we don't lose data, even though it adds host stall.
        with self._lock:
            pending = list(self._pending)
            self._pending.clear()
        for entry in pending:
            (_, _, end_event) = entry
            try:
                end_event.synchronize()
            except Exception:
                logger.exception("force-drain synchronize failed")
                continue
            self._emit_row(entry)

    def _emit_row(self, entry: tuple) -> None:
        (snapshot, start_event, end_event) = entry
        try:
            latency_ms = float(start_event.elapsed_time(end_event))
        except Exception:
            logger.exception("elapsed_time failed; dropping row")
            return
        if self._writer is None:
            return
        try:
            self._writer.writerow([
                snapshot["step_idx"],
                snapshot["timestamp_iso"],
                f'{snapshot["elapsed_s"]:.6f}',
                snapshot["mode"],
                snapshot["forward_mode"],
                snapshot["batch_size_local"],
                snapshot["batch_size_global"],
                snapshot["num_tokens"],
                f"{latency_ms:.4f}",
                snapshot["running_reqs"],
                snapshot["waiting_reqs"],
            ])
        except Exception:
            logger.exception("ParasPerStepMetricsSampler: writerow failed")
