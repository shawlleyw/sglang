"""Single-GPU ablation of production VMM activation, without loading a model.

Default scratch shapes match GPT-OSS-120B: EP/TP 256/2048 tokens,
context 131072, vocabulary 201088. Override dimensions for another setting.
Only disposable logits/KV-index backing is measured, not a full model switch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("zero_two_sync", "no_zero_two_sync", "no_zero_one_sync")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ep-max-tokens", type=int, default=256)
    parser.add_argument("--tp-max-tokens", type=int, default=2048)
    parser.add_argument("--context-length", type=int, default=131072)
    parser.add_argument("--vocab-size", type=int, default=201088)
    parser.add_argument("--resident-gib", type=float, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--profile-iterations", type=int, default=5)
    parser.add_argument(
        "--interval-ms",
        type=float,
        default=0,
        help="Idle interval before each activation, outside timing (0: allocation churn)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for key in (
        "ep_max_tokens",
        "tp_max_tokens",
        "context_length",
        "vocab_size",
        "iterations",
        "rounds",
        "profile_iterations",
    ):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.warmup < 0 or not math.isfinite(args.resident_gib) or args.resident_gib < 0:
        parser.error("warmup and finite resident-gib must be nonnegative")
    if not math.isfinite(args.interval_ms) or args.interval_ms < 0:
        parser.error("interval-ms must be finite and nonnegative")
    if min(args.ep_max_tokens, args.tp_max_tokens) < 8 or args.context_length < 513:
        parser.error("correctness probes require at least 8 tokens and context 513")
    return args


def buffer_shapes(args):
    return {
        mode: {
            "kv_indices": (tokens * args.context_length,),
            "logits": (tokens, args.vocab_size),
        }
        for mode, tokens in (("ep", args.ep_max_tokens), ("tp", args.tp_max_tokens))
    }


def command_output(command):
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def summarize(values):
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "min_ms": ordered[0],
        "p95_ms": ordered[math.ceil(0.95 * len(values)) - 1],
        "max_ms": ordered[-1],
    }


def make_memory(args, variant):
    import torch
    from sglang.srt.paras.mode import ParaSMode
    from sglang.srt.paras.runtime_memory import CudaModeRuntimeMemory

    class AblationMemory(CudaModeRuntimeMemory):
        def activate(self, mode):
            changed = mode != self.active
            super().activate(mode)
            # Isolate clearing from removal of the now-unneeded final wait.
            # The original production path is zero_two_sync (default policy).
            if changed and variant == "no_zero_two_sync":
                self.synchronize()

    memory, buffers = AblationMemory("cuda:0"), {}
    for name, shapes in buffer_shapes(args).items():
        mode = ParaSMode(name)
        memory.activate(mode)
        buffers[mode] = {
            key: memory.zeros(
                mode,
                key,
                shape,
                torch.int64 if key == "kv_indices" else torch.float32,
                zero_on_resume=variant == "zero_two_sync",
            )
            for key, shape in shapes.items()
        }
    memory.activate(ParaSMode.EP)
    return memory, buffers


def release(memory, buffers):
    import torch

    torch.cuda.synchronize()
    buffers.clear()
    for allocations in memory.allocations.values():
        for allocation in allocations.values():
            allocation.close()


def verify_graph_replay(memory, buffers):
    """Poison remapped storage, rebuild indices, replay real Triton attention.

    Capture both modes once. Change sequence lengths/request order after each
    remap. Include tails, zero-length rows and all-empty metadata. Metadata is
    produced outside capture, as in production; attention and logits overwrite
    run inside the retained graph. These are operator checks, not model logits.
    """
    import torch
    from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
    from sglang.srt.layers.attention.triton_ops.decode_attention import (
        decode_attention_fwd,
    )
    from sglang.srt.paras.mode import ParaSMode

    torch.manual_seed(1729)
    modes = (ParaSMode.EP, ParaSMode.TP)
    stream = torch.cuda.Stream()
    states = {}
    for mode, kv_heads, dim in ((modes[0], 4, 128), (modes[1], 1, 64)):
        memory.activate(mode)
        batch, q_heads, splits, context, pool = 8, 8, 4, 513, 2049
        indices = buffers[mode]["kv_indices"]
        logits = buffers[mode]["logits"][:batch]
        state = {
            "indices": indices,
            "logits": logits,
            "requests": torch.arange(batch, dtype=torch.int32, device="cuda"),
            "lengths": torch.ones(batch, dtype=torch.int32, device="cuda"),
            "indptr": torch.arange(batch + 1, dtype=torch.int32, device="cuda"),
            "mapping": torch.randint(
                1, pool, (batch, context), device="cuda", dtype=torch.int32
            ),
            "q": torch.randn(batch, q_heads, dim, dtype=torch.float16, device="cuda"),
            "k": torch.randn(pool, kv_heads, dim, dtype=torch.float16, device="cuda"),
            "v": torch.randn(pool, kv_heads, dim, dtype=torch.float16, device="cuda"),
            "out": torch.empty(batch, q_heads, dim, dtype=torch.float16, device="cuda"),
            "attn_logits": torch.empty(batch, q_heads, splits, dim, device="cuda"),
            "attn_lse": torch.empty(batch, q_heads, splits, device="cuda"),
            "splits": torch.full((batch,), splits, dtype=torch.int32, device="cuda"),
            "logit_source": torch.randn_like(logits),
        }

        def metadata(s):
            create_flashinfer_kv_indices_triton[(batch,)](
                s["mapping"],
                s["requests"],
                s["lengths"],
                s["indptr"],
                None,
                s["indices"],
                context,
            )

        def forward(s):
            decode_attention_fwd(
                s["q"],
                s["k"],
                s["v"],
                s["out"],
                s["indptr"],
                s["indices"],
                s["attn_logits"],
                s["attn_lse"],
                s["splits"],
                splits,
                s["q"].shape[-1] ** -0.5,
            )
            # Same overwrite used by LogitsProcessor before consuming logits.
            s["logits"].copy_(s["logit_source"])

        torch.cuda.synchronize()
        with torch.cuda.stream(stream):
            for _ in range(2):
                metadata(state)
                forward(state)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            forward(state)
        state["graph"] = graph
        states[mode] = state

    checks = []
    patterns = ([513, 3, 31, 1, 65, 0, 1, 7], [1, 0, 17, 513, 2, 33, 1, 0], [0] * 8)
    pointers = {
        mode: {key: t.data_ptr() for key, t in group.items()}
        for mode, group in buffers.items()
    }
    for cycle, lengths in enumerate(patterns):
        for mode in modes:
            memory.activate(mode)
            state = states[mode]
            for key, tensor in buffers[mode].items():
                assert tensor.data_ptr() == pointers[mode][key]
            # Deliberately nonzero/invalid old data proves overwrite-before-read;
            # correctness cannot accidentally depend on driver-cleared pages.
            state["indices"].fill_(-1)
            buffers[mode]["logits"].fill_(float("nan"))
            state["requests"].copy_(torch.arange(8, device="cuda").roll(cycle + 1))
            state["lengths"].copy_(torch.tensor(lengths, device="cuda"))
            state["indptr"][1:].copy_(state["lengths"].cumsum(0))
            state["logit_source"].add_(1)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                metadata(state)
                state["graph"].replay()
            stream.synchronize()
            expected_indices = torch.cat(
                [
                    state["mapping"][request, :length]
                    for request, length in zip(
                        state["requests"].cpu().tolist(), lengths
                    )
                ]
            ).long()
            torch.testing.assert_close(
                state["indices"][: sum(lengths)], expected_indices
            )
            assert (
                (state["indices"][sum(lengths) : sum(lengths) + 16] == -1).all().item()
            )
            torch.testing.assert_close(
                state["logits"], state["logit_source"], rtol=0, atol=0
            )
            offset = 0
            for row, length in enumerate(lengths):
                if length:
                    slots = expected_indices[offset : offset + length]
                    k = (
                        state["k"][slots]
                        .float()
                        .repeat_interleave(8 // state["k"].shape[1], dim=1)
                    )
                    v = (
                        state["v"][slots]
                        .float()
                        .repeat_interleave(8 // state["v"].shape[1], dim=1)
                    )
                    scores = torch.einsum("hd,thd->ht", state["q"][row].float(), k)
                    probabilities = (scores * state["q"].shape[-1] ** -0.5).softmax(-1)
                    expected = torch.einsum("ht,thd->hd", probabilities, v)
                    torch.testing.assert_close(
                        state["out"][row].float(), expected, rtol=0.01, atol=0.003
                    )
                offset += length
            stats = memory.stats()
            assert all(
                s["resident_bytes"] == (s["virtual_bytes"] if name == mode.value else 0)
                for name, s in stats.items()
            )
            checks.append({"mode": mode.value, "lengths": lengths, "passed": True})
    torch.cuda.synchronize()
    # Locals and graph owners are dropped before the caller closes VMM ranges.
    return checks


def install_profiler(memory):
    """Time driver/sync calls without adding synchronization; diagnostic pass only."""
    original_call, original_sync = memory.driver._call, memory.synchronize
    phases = {}

    def record(name, call, *args):
        start = time.perf_counter_ns()
        try:
            return call(*args)
        finally:
            entry = phases.setdefault(name, {"ms": 0.0, "calls": 0})
            entry["ms"] += (time.perf_counter_ns() - start) / 1e6
            entry["calls"] += 1

    memory.driver._call = lambda name, *args: record(name, original_call, name, *args)

    def synchronize():
        name = "sync_before" if "sync_before" not in phases else "sync_after"
        return record(name, original_sync)

    memory.synchronize = synchronize
    return phases


def measure(args, variant, round_id):
    import torch
    from sglang.srt.paras.mode import ParaSMode

    memory, buffers = make_memory(args, variant)
    directions = ((ParaSMode.TP, "ep_to_tp"), (ParaSMode.EP, "tp_to_ep"))
    rows = []
    try:
        for _ in range(args.warmup):
            for mode, _ in directions:
                if args.interval_ms:
                    time.sleep(args.interval_ms / 1000)
                memory.activate(mode)
        for iteration in range(args.iterations):
            for mode, direction in directions:
                if args.interval_ms:
                    time.sleep(args.interval_ms / 1000)
                start = time.perf_counter_ns()
                memory.activate(mode)
                ms = (time.perf_counter_ns() - start) / 1e6
                rows.append(
                    {
                        "variant": variant,
                        "round": round_id,
                        "iteration": iteration,
                        "direction": direction,
                        "ms": ms,
                        "kind": "timing",
                    }
                )
        phases = install_profiler(memory)
        for iteration in range(args.profile_iterations):
            for mode, direction in directions:
                if args.interval_ms:
                    time.sleep(args.interval_ms / 1000)
                phases.clear()
                start = time.perf_counter_ns()
                memory.activate(mode)
                ms = (time.perf_counter_ns() - start) / 1e6
                expected_calls = {
                    "cuMemUnmap": 2,
                    "cuMemCreate": 2,
                    "cuMemMap": 2,
                    "cuMemSetAccess": 2,
                    "cuMemRelease": 2,
                    "sync_before": 1,
                    "cuMemsetD8_v2": 2 if variant == "zero_two_sync" else 0,
                    "sync_after": 0 if variant == "no_zero_one_sync" else 1,
                }
                for name, count in expected_calls.items():
                    assert phases.get(name, {}).get("calls", 0) == count, (
                        variant,
                        name,
                        phases,
                    )
                state = memory.stats()
                assert all(
                    s["resident_bytes"]
                    == (s["virtual_bytes"] if name == mode.value else 0)
                    for name, s in state.items()
                )
                free, total = torch.cuda.mem_get_info()
                rows.append(
                    {
                        "variant": variant,
                        "round": round_id,
                        "iteration": iteration,
                        "direction": direction,
                        "ms": ms,
                        "kind": "profile",
                        "phases": {name: dict(value) for name, value in phases.items()},
                        "residency": state,
                        "driver_used_bytes": total - free,
                    }
                )
        residency = memory.stats()
        assert residency["tp"]["resident_bytes"] == 0
        assert residency["ep"]["resident_bytes"] == residency["ep"]["virtual_bytes"]
        return rows, residency
    finally:
        release(memory, buffers)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    shapes = buffer_shapes(args)
    sources = [
        Path(__file__).resolve(),
        ROOT / "python/sglang/srt/paras/runtime_memory.py",
        ROOT / "python/sglang/srt/layers/attention/triton_backend.py",
        ROOT / "python/sglang/srt/layers/attention/utils.py",
        ROOT / "python/sglang/srt/layers/attention/triton_ops/decode_attention.py",
        ROOT / "python/sglang/srt/model_executor/cuda_graph_runner.py",
        ROOT / "python/sglang/srt/layers/logits_processor.py",
    ]
    manifest = {
        "command": [sys.executable, *sys.argv],
        "args": vars(args) | {"output": str(args.output)},
        "shapes": shapes,
        "variants": VARIANTS,
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_branch": command_output(["git", "branch", "--show-current"]),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "timing": "host wall time of activation; initialization, validation and diagnostic profiling excluded",
        "scope": "Single-GPU scratch activation, not full-model/multi-GPU switch latency. Resident ballast is synthetic, not a real KV cache.",
    }
    (args.output / "tracked_changes.patch").write_text(
        command_output(["git", "diff", "HEAD"])["stdout"]
    )
    for path in sources:
        target = args.output / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    write_json(args.output / "manifest.json", manifest)
    if args.dry_run:
        print(f"Prepared {args.output}; no GPU access", flush=True)
        return
    sys.path.insert(0, str(ROOT / "python"))
    import torch

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES")
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    manifest.update(
        {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(torch.cuda.get_device_properties(0)),
            "gpu_inventory": command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
                    "--format=csv",
                ]
            ),
        }
    )
    write_json(args.output / "manifest.json", manifest)
    ballast_bytes = int(args.resident_gib * 2**30)
    required = (
        ballast_bytes
        + max(
            s["kv_indices"][0] * 8 + math.prod(s["logits"]) * 4 for s in shapes.values()
        )
        + 2**30
    )
    free, _ = torch.cuda.mem_get_info()
    if required > free:
        raise RuntimeError(
            f"Need about {required / 2**30:.2f} GiB; only {free / 2**30:.2f} GiB free"
        )
    ballast = torch.empty(ballast_bytes, dtype=torch.uint8, device="cuda")
    ballast.zero_()
    torch.cuda.synchronize()
    memory, buffers = make_memory(args, "no_zero_one_sync")
    try:
        checks = verify_graph_replay(memory, buffers)
        write_json(args.output / "correctness.json", checks)
        print(f"Passed {len(checks)} poisoned-buffer graph replay checks", flush=True)
    finally:
        release(memory, buffers)
    rows, residency = [], None
    for round_id in range(args.rounds):
        order = (
            VARIANTS[round_id % len(VARIANTS) :] + VARIANTS[: round_id % len(VARIANTS)]
        )
        for variant in order:
            result, residency = measure(args, variant, round_id)
            rows.extend(result)
            with (args.output / "samples.jsonl").open("a") as stream:
                for row in result:
                    stream.write(json.dumps(row) + "\n")
            print(f"Finished round {round_id + 1}/{args.rounds}: {variant}", flush=True)
    summary = []
    for variant in VARIANTS:
        for direction in ("ep_to_tp", "tp_to_ep"):
            selected = [
                r
                for r in rows
                if r["variant"] == variant and r["direction"] == direction
            ]
            diagnostic = [r for r in selected if r["kind"] == "profile"]
            names = sorted({name for r in diagnostic for name in r["phases"]})
            profile = {
                name: {
                    "mean_ms": statistics.mean(
                        r["phases"].get(name, {}).get("ms", 0) for r in diagnostic
                    ),
                    "mean_calls": statistics.mean(
                        r["phases"].get(name, {}).get("calls", 0) for r in diagnostic
                    ),
                }
                for name in names
            }
            summary.append(
                {
                    "variant": variant,
                    "direction": direction,
                    **summarize([r["ms"] for r in selected if r["kind"] == "timing"]),
                    "diagnostic_call_wall_times": profile,
                }
            )
    write_json(
        args.output / "summary.json",
        {"rows": summary, "residency": residency, "passed": True},
    )
    lines = [
        "# Single-GPU VMM activation ablation",
        "",
        manifest["scope"],
        "",
        "| Variant | Direction | n | Mean (ms) | Median (ms) | P95 (ms) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['variant']} | {row['direction']} | {row['n']} | {row['mean_ms']:.3f} | {row['median_ms']:.3f} | {row['p95_ms']:.3f} |"
        )
    lines += [
        "",
        "Diagnostic per-call host times are in summary.json; no extra internal fences were added. Timed samples use uninstrumented driver calls. Variants rotate order across rounds; every sample is retained.",
        "",
    ]
    (args.output / "TABLE.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
