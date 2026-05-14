# Copyright 2024-2025 SGLang Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Batch inference benchmark driver for ParaS rollout workloads.

Submits ``--num-requests`` chat-completion requests concurrently to a running
SGLang server and records end-to-end latency plus aggregate throughput.  The
intent is to mimic the synchronous "submit all prompts at t=0, wait for all
to complete" pattern used by GRPO post-training rollouts so that ParaS
auto-switch policies can be benchmarked under realistic burst load.

Server-side metrics (running batch, queue size, per-second throughput) are
written separately by the server via ``--paras-metrics-file``; this driver
intentionally only records *client-observed* quantities.

Two modes:
  * non-spec (default): every request runs with ``max_completion_tokens =
    --max-completion-tokens-cap`` and ``ignore_eos=False``.
  * spec mode (``--spec-mode``): ``max_completion_tokens`` is taken from each
    JSONL row's ``output_len`` field and ``ignore_eos=True`` so the server
    decodes exactly that many tokens (used for reproducible decode-bound
    micro-benchmarks).

GRPO simulation: ``--group-size G`` replicates each unique prompt ``G`` times,
mimicking the "n samples per prompt" pattern.  Replicas are sent as fully
independent requests (no prefix sharing is assumed) and the replicated list
is shuffled by default so different prompts interleave on the server queue.

Example
-------
::

    python -m sglang.bench_paras \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --dataset-jsonl /data/sampled_8k.jsonl \\
        --num-requests 256 --group-size 8 \\
        --output-dir /tmp/paras_run_001 \\
        --mode-label paras
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm as tqdm_asyncio

logger = logging.getLogger("bench_paras")


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------


@dataclass
class RequestSpec:
    """A single request to submit to the server.

    ``unique_id`` indexes into the K unique prompts sampled from the JSONL
    file; ``replica_id`` distinguishes the G copies inside one group.  Both
    are persisted in ``per_request.jsonl`` for post-hoc grouping.
    """

    request_id: int
    unique_id: int
    replica_id: int
    prompt_text: str
    max_new_tokens: int
    ignore_eos: bool


@dataclass
class RequestRecord:
    """Per-request observations written to ``per_request.jsonl``."""

    request_id: int
    unique_id: int
    replica_id: int
    arrival_t: float
    completion_t: float
    e2e_latency: float
    prompt_len_tokens: int
    output_len_tokens: int
    finish_reason: Optional[str]
    completed: bool
    error: Optional[str] = None
    http_status: Optional[int] = None


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts.

    Blank lines are skipped silently.  Malformed lines raise immediately so
    we don't carry partial datasets into a long benchmark run.
    """
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Failed to parse JSON at {path}:{line_no}: {e}"
                ) from e
    return rows


def build_request_specs(
    entries: List[Dict[str, Any]],
    *,
    num_requests: int,
    group_size: int,
    spec_mode: bool,
    max_completion_tokens_cap: int,
    shuffle: bool,
    seed: int,
) -> List[RequestSpec]:
    """Sample K unique prompts, replicate by ``group_size``, optionally shuffle.

    The K unique prompts are taken in the order they appear in the JSONL
    (or shuffled deterministically by ``seed`` when ``--shuffle``) so that
    re-runs with the same seed produce identical request streams.
    """
    if num_requests % group_size != 0:
        raise ValueError(
            f"--num-requests ({num_requests}) must be divisible by "
            f"--group-size ({group_size})."
        )
    num_unique = num_requests // group_size

    if num_unique > len(entries):
        raise ValueError(
            f"Need {num_unique} unique prompts (num_requests={num_requests} / "
            f"group_size={group_size}) but JSONL only has {len(entries)} rows."
        )

    rng = random.Random(seed)

    # Pick the first K rows in the (optionally shuffled) JSONL order.  We
    # shuffle a copy of the *index* list so the original list stays intact
    # for the run_config record.
    indices = list(range(len(entries)))
    if shuffle:
        rng.shuffle(indices)
    chosen_indices = indices[:num_unique]

    specs: List[RequestSpec] = []
    request_id = 0
    for unique_id, src_idx in enumerate(chosen_indices):
        row = entries[src_idx]
        if "prompt_text" not in row:
            raise ValueError(
                f"JSONL row {src_idx} missing required 'prompt_text' field."
            )
        prompt_text = row["prompt_text"]

        if spec_mode:
            if "output_len" not in row:
                raise ValueError(
                    f"--spec-mode set but JSONL row {src_idx} has no "
                    f"'output_len' field."
                )
            max_new_tokens = int(row["output_len"])
            ignore_eos = True
        else:
            max_new_tokens = max_completion_tokens_cap
            ignore_eos = False

        for replica_id in range(group_size):
            specs.append(
                RequestSpec(
                    request_id=request_id,
                    unique_id=unique_id,
                    replica_id=replica_id,
                    prompt_text=prompt_text,
                    max_new_tokens=max_new_tokens,
                    ignore_eos=ignore_eos,
                )
            )
            request_id += 1

    if shuffle:
        # Shuffle the replicated stream so the server queue interleaves
        # different prompts instead of seeing G copies of prompt 0 first.
        rng.shuffle(specs)

    return specs


# ---------------------------------------------------------------------------
# HTTP request
# ---------------------------------------------------------------------------


async def submit_one_request(
    session: aiohttp.ClientSession,
    *,
    url: str,
    model: str,
    spec: RequestSpec,
    burst_spread_sec: float,
    rng: random.Random,
) -> RequestRecord:
    """Send a single chat-completion request and return a per-request record.

    Non-200 responses and exceptions are caught and recorded as
    ``completed=False`` so a single bad request can't poison the gather.
    """
    if burst_spread_sec > 0:
        # Jitter the POST time within ``[0, burst_spread_sec]`` to avoid a
        # thundering herd on the server's accept loop.  The jitter is drawn
        # from the same seeded RNG as the shuffle, so runs stay reproducible.
        await asyncio.sleep(rng.uniform(0.0, burst_spread_sec))

    body = {
        "model": model,
        "messages": [{"role": "user", "content": spec.prompt_text}],
        "temperature": 1.0,
        "max_completion_tokens": spec.max_new_tokens,
        "ignore_eos": spec.ignore_eos,
        "stream": False,
    }

    arrival_t = time.perf_counter()
    try:
        async with session.post(url, json=body) as response:
            completion_t = time.perf_counter()
            if response.status != 200:
                text = await response.text()
                logger.warning(
                    "request %d failed: HTTP %d: %s",
                    spec.request_id,
                    response.status,
                    text[:200],
                )
                return RequestRecord(
                    request_id=spec.request_id,
                    unique_id=spec.unique_id,
                    replica_id=spec.replica_id,
                    arrival_t=arrival_t,
                    completion_t=completion_t,
                    e2e_latency=completion_t - arrival_t,
                    prompt_len_tokens=0,
                    output_len_tokens=0,
                    finish_reason=None,
                    completed=False,
                    error=f"http_{response.status}",
                    http_status=response.status,
                )

            data = await response.json()
    except (
        asyncio.TimeoutError,
        aiohttp.ClientError,
    ) as e:
        completion_t = time.perf_counter()
        logger.warning("request %d errored: %s", spec.request_id, e)
        return RequestRecord(
            request_id=spec.request_id,
            unique_id=spec.unique_id,
            replica_id=spec.replica_id,
            arrival_t=arrival_t,
            completion_t=completion_t,
            e2e_latency=completion_t - arrival_t,
            prompt_len_tokens=0,
            output_len_tokens=0,
            finish_reason=None,
            completed=False,
            error=type(e).__name__,
            http_status=None,
        )

    try:
        usage = data.get("usage", {}) or {}
        prompt_len_tokens = int(usage.get("prompt_tokens", 0))
        output_len_tokens = int(usage.get("completion_tokens", 0))
        choices = data.get("choices") or []
        finish_reason = (
            choices[0].get("finish_reason") if choices else None
        )
    except (AttributeError, IndexError, TypeError, ValueError) as e:
        logger.warning(
            "request %d returned malformed body: %s; raw=%.200s",
            spec.request_id,
            e,
            json.dumps(data)[:200],
        )
        return RequestRecord(
            request_id=spec.request_id,
            unique_id=spec.unique_id,
            replica_id=spec.replica_id,
            arrival_t=arrival_t,
            completion_t=completion_t,
            e2e_latency=completion_t - arrival_t,
            prompt_len_tokens=0,
            output_len_tokens=0,
            finish_reason=None,
            completed=False,
            error=f"bad_body_{type(e).__name__}",
            http_status=200,
        )

    return RequestRecord(
        request_id=spec.request_id,
        unique_id=spec.unique_id,
        replica_id=spec.replica_id,
        arrival_t=arrival_t,
        completion_t=completion_t,
        e2e_latency=completion_t - arrival_t,
        prompt_len_tokens=prompt_len_tokens,
        output_len_tokens=output_len_tokens,
        finish_reason=finish_reason,
        completed=True,
        error=None,
        http_status=200,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


async def run_benchmark(
    specs: List[RequestSpec],
    *,
    url: str,
    model: str,
    timeout_sec: float,
    burst_spread_sec: float,
    seed: int,
) -> List[RequestRecord]:
    """Submit all requests via ``asyncio.gather`` and return per-request records."""
    connector = aiohttp.TCPConnector(limit=4096)
    timeout = aiohttp.ClientTimeout(total=None if timeout_sec <= 0 else timeout_sec)
    # One RNG per call so jitter is reproducible.  Each task gets a private
    # RNG seeded from the master seed so concurrency doesn't reorder samples.
    master_rng = random.Random(seed)
    per_task_rngs = [
        random.Random(master_rng.randrange(1 << 32)) for _ in specs
    ]

    async with aiohttp.ClientSession(
        connector=connector, timeout=timeout
    ) as session:
        coros = [
            submit_one_request(
                session,
                url=url,
                model=model,
                spec=spec,
                burst_spread_sec=burst_spread_sec,
                rng=rng,
            )
            for spec, rng in zip(specs, per_task_rngs)
        ]
        records = await tqdm_asyncio.gather(*coros, desc="requests")
    return list(records)


# ---------------------------------------------------------------------------
# Aggregation + output
# ---------------------------------------------------------------------------


def build_summary(
    records: List[RequestRecord], args: argparse.Namespace
) -> Dict[str, Any]:
    """Compute the aggregate summary written to ``summary.json``."""
    completed_records = [r for r in records if r.completed]
    failed_records = [r for r in records if not r.completed]
    latencies = [r.e2e_latency for r in completed_records]

    if records:
        t_start = min(r.arrival_t for r in records)
    else:
        t_start = 0.0
    if completed_records:
        t_end = max(r.completion_t for r in completed_records)
    else:
        t_end = t_start
    e2e_time = t_end - t_start

    total_input_tokens = sum(r.prompt_len_tokens for r in completed_records)
    total_output_tokens = sum(r.output_len_tokens for r in completed_records)

    p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
    p99 = float(np.percentile(latencies, 99)) if latencies else 0.0
    mean_latency = float(np.mean(latencies)) if latencies else 0.0

    return {
        "mode_label": args.mode_label,
        "model": args.model,
        "dataset_jsonl": args.dataset_jsonl,
        "spec_mode": args.spec_mode,
        "num_requests": args.num_requests,
        "group_size": args.group_size,
        "completed": len(completed_records),
        "failed": len(failed_records),
        "e2e_time": e2e_time,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "input_throughput": (
            total_input_tokens / e2e_time if e2e_time > 0 else 0.0
        ),
        "output_throughput": (
            total_output_tokens / e2e_time if e2e_time > 0 else 0.0
        ),
        "mean_e2e_latency_s": mean_latency,
        "p50_e2e_latency_s": p50,
        "p99_e2e_latency_s": p99,
    }


def get_git_sha(repo_dir: Optional[str] = None) -> str:
    """Best-effort ``git rev-parse HEAD``; returns ``"unknown"`` on failure."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return "unknown"


def write_outputs(
    output_dir: str,
    *,
    summary: Dict[str, Any],
    records: List[RequestRecord],
    args: argparse.Namespace,
) -> None:
    """Write ``summary.json``, ``per_request.jsonl`` and ``run_config.json``."""
    os.makedirs(output_dir, exist_ok=True)

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    per_request_path = os.path.join(output_dir, "per_request.jsonl")
    with open(per_request_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec)) + "\n")

    run_config = {
        **vars(args),
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": get_git_sha(),
    }
    run_config_path = os.path.join(output_dir, "run_config.json")
    with open(run_config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, sort_keys=True, default=str)
        f.write("\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m sglang.bench_paras",
        description=(
            "Client-side batch inference driver for ParaS rollout benchmarks. "
            "Submits N chat-completion requests concurrently and records "
            "end-to-end timing + throughput."
        ),
    )

    # Required.
    parser.add_argument(
        "--model",
        required=True,
        help="HF model id sent as the 'model' field in the request body.",
    )
    parser.add_argument(
        "--dataset-jsonl",
        required=True,
        help=(
            "Path to a JSONL file. Each row must have 'prompt_text'; spec "
            "mode additionally requires 'output_len'."
        ),
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        required=True,
        help="Total requests to submit. Must be divisible by --group-size.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Directory to write summary.json, per_request.jsonl, "
            "run_config.json. Created if missing; existing files overwritten."
        ),
    )

    # Optional.
    parser.add_argument(
        "--mode-label",
        default=None,
        help="Free-form tag baked into summary.json (e.g. 'paras', 'tp-static').",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=1,
        help="Replicate each unique prompt G times (GRPO simulation).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server host.")
    parser.add_argument("--port", type=int, default=30000, help="Server port.")
    parser.add_argument(
        "--spec-mode",
        action="store_true",
        help=(
            "Use per-row 'output_len' from JSONL and ignore_eos=True. "
            "Yields exact, reproducible decode token counts."
        ),
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="(no-op) Stream is always disabled; kept for CLI clarity.",
    )
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shuffle the replicated prompt list before submission.",
    )
    parser.add_argument(
        "--max-completion-tokens-cap",
        type=int,
        default=32768,
        help="Cap on max_completion_tokens (non-spec mode only).",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=0.0,
        help="Per-request HTTP timeout in seconds. 0 or negative disables the timeout (recommended for large batches where queue wait can exceed any reasonable bound).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="RNG seed for shuffle + jitter."
    )
    parser.add_argument(
        "--burst-spread-sec",
        type=float,
        default=0.0,
        help=(
            "If >0, jitter each POST by Uniform(0, X) seconds to soften the "
            "thundering-herd on the server accept loop."
        ),
    )

    args = parser.parse_args(argv)

    if args.num_requests <= 0:
        parser.error("--num-requests must be > 0")
    if args.group_size <= 0:
        parser.error("--group-size must be > 0")
    if args.num_requests % args.group_size != 0:
        parser.error(
            f"--num-requests ({args.num_requests}) must be divisible by "
            f"--group-size ({args.group_size})"
        )
    if args.burst_spread_sec < 0:
        parser.error("--burst-spread-sec must be >= 0")

    return args


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)

    if not os.path.isfile(args.dataset_jsonl):
        raise FileNotFoundError(
            f"--dataset-jsonl not found: {args.dataset_jsonl}"
        )

    logger.info("loading dataset from %s", args.dataset_jsonl)
    entries = load_jsonl(args.dataset_jsonl)
    logger.info("loaded %d rows", len(entries))

    specs = build_request_specs(
        entries,
        num_requests=args.num_requests,
        group_size=args.group_size,
        spec_mode=args.spec_mode,
        max_completion_tokens_cap=args.max_completion_tokens_cap,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    logger.info(
        "built %d request specs (unique=%d, group_size=%d, spec_mode=%s)",
        len(specs),
        args.num_requests // args.group_size,
        args.group_size,
        args.spec_mode,
    )

    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    timeout_label = "none" if args.timeout_sec <= 0 else f"{args.timeout_sec:.0f}s"
    logger.info("submitting to %s with timeout=%s", url, timeout_label)

    records = asyncio.run(
        run_benchmark(
            specs,
            url=url,
            model=args.model,
            timeout_sec=args.timeout_sec,
            burst_spread_sec=args.burst_spread_sec,
            seed=args.seed,
        )
    )

    summary = build_summary(records, args)
    write_outputs(
        args.output_dir, summary=summary, records=records, args=args
    )

    logger.info(
        "done: completed=%d failed=%d e2e=%.2fs in_tput=%.1f tok/s "
        "out_tput=%.1f tok/s p50=%.2fs p99=%.2fs",
        summary["completed"],
        summary["failed"],
        summary["e2e_time"],
        summary["input_throughput"],
        summary["output_throughput"],
        summary["p50_e2e_latency_s"],
        summary["p99_e2e_latency_s"],
    )
    logger.info("results written under %s", args.output_dir)

    # Non-zero exit if anything failed so CI / wrappers can detect it.
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
