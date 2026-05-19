"""Per-forward-step metrics sampler for ParaS rollout benchmarking.

Records one CSV row per scheduler forward step, capturing GPU-completed
forward latency (host-clock around the forward block + torch.cuda.synchronize).

Unlike ``ParasMetricsSampler`` (which is a 1Hz background daemon),
this writer runs on the scheduler's main thread: ``record()`` is called
synchronously from ``Scheduler.run_batch`` immediately after the GPU
finishes the forward+sample for the current step.

Columns:
    step_idx                monotonic, from scheduler.forward_ct
    timestamp_iso           UTC ISO timestamp at record time
    elapsed_s               seconds since recorder.start()
    mode                    "EP" or "TP" (from scheduler.paras_parallelism_config,
                            or inferred from server_args.enable_dp_attention for
                            non-ParaS servers)
    forward_mode            "EXTEND" / "DECODE" / "IDLE" / "MIXED" / ... (from
                            batch.forward_mode.name)
    batch_size_local        len(batch.reqs)
    batch_size_global       In EP mode, sum(batch.global_running_reqs); else
                            equals batch_size_local
    num_tokens              For EXTEND: batch.extend_num_tokens
                            Otherwise: batch_size_local (one token per req per
                            decode step)
    step_latency_ms         (time.perf_counter() - t0) * 1000 measured around
                            forward + sample, with torch.cuda.synchronize()
                            immediately before t1 to capture GPU completion
    running_reqs            len(scheduler.running_batch.reqs)
    waiting_reqs            len(scheduler.waiting_queue)

Activated by passing ``--paras-per-step-metrics-file <path>`` to the SGLang
server. On non-zero ranks, ``start()`` and ``record()`` are no-ops (only
tp_rank==0 writes).
"""

from __future__ import annotations

import atexit
import csv
import logging
import os
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

    def start(self) -> None:
        # Distributed-correctness gate: only rank 0 writes the metrics file.
        if getattr(self.scheduler, "tp_rank", 0) != 0:
            logger.debug("ParasPerStepMetricsSampler: tp_rank != 0, no-op")
            return

        out_dir = os.path.dirname(self.output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # buffering=1 => line-buffered, so partial progress is visible to
        # the driver/log tail even if the server crashes mid-step.
        self._fh = open(self.output_path, "w", buffering=1)
        self._writer = csv.writer(self._fh)
        self._writer.writerow(self.CSV_HEADER)
        self._t0 = time.time()
        self._active = True
        atexit.register(self.stop)
        logger.info(
            f"ParasPerStepMetricsSampler started: writing per-step CSV to "
            f"{self.output_path}"
        )

    def stop(self) -> None:
        if not self._active:
            return
        self._active = False
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

    def record(self, batch, step_latency_ms: float) -> None:
        """Append one row for this forward step.

        Called from ``Scheduler.run_batch`` immediately after the forward
        block (post-cuda-synchronize). Safe to call on every rank: it is a
        no-op on non-zero ranks because ``start()`` did not initialize the
        writer there.
        """
        if not self._active or self._writer is None:
            return
        try:
            self._record_impl(batch, step_latency_ms)
        except Exception:
            # Recorder-thread isolation: never let a metrics bug kill the
            # scheduler. Log and continue.
            logger.exception("ParasPerStepMetricsSampler: record failed")

    def _record_impl(self, batch, step_latency_ms: float) -> None:
        scheduler = self.scheduler

        # Mode: ParaS servers expose the runtime mode on the scheduler;
        # static servers do not, so infer from server_args.
        paras_config = getattr(scheduler, "paras_parallelism_config", None)
        if paras_config is not None:
            mode = paras_config
        else:
            sa = getattr(scheduler, "server_args", None)
            if sa is not None and getattr(sa, "enable_dp_attention", False):
                mode = "EP"
            else:
                mode = "TP"

        # forward_mode: EXTEND / DECODE / IDLE / MIXED (or whatever the enum is).
        fwd = getattr(batch, "forward_mode", None) if batch is not None else None
        forward_mode = fwd.name if fwd is not None else "NONE"

        # batch sizes: local is rank 0's view, global is the all-ranks sum
        # (only meaningful + only populated in EP / DP-attention mode).
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

        # num_tokens: prefill -> extend_num_tokens; decode -> 1 token per req.
        if forward_mode == "EXTEND":
            num_tokens = int(getattr(batch, "extend_num_tokens", 0) or 0)
        else:
            num_tokens = bs_local

        # Scheduler-level counters (these are rank-0-local, like the existing
        # 1Hz sampler's fallback path).
        rb = getattr(scheduler, "running_batch", None)
        running_reqs = len(rb.reqs) if rb is not None and getattr(rb, "reqs", None) else 0
        waiting_reqs = len(getattr(scheduler, "waiting_queue", []) or [])

        now = time.time()
        elapsed = now - self._t0 if self._t0 is not None else 0.0
        row = [
            getattr(scheduler, "forward_ct", 0),
            datetime.now(timezone.utc).isoformat(),
            f"{elapsed:.6f}",
            mode,
            forward_mode,
            bs_local,
            bs_global,
            num_tokens,
            f"{step_latency_ms:.4f}",
            running_reqs,
            waiting_reqs,
        ]
        self._writer.writerow(row)
