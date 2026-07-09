"""Per-rank kernel categorization from torch.profiler traces (multi-rank).

Input: a directory containing `*_rank<R>_batch<B>_*_<stage>.trace.json.gz`
files emitted by `bench_one_batch --profile` (one per rank). Auto-discovers
configs from filename prefix and groups ranks by (config, batch, stage).

Output: per-rank category breakdown table + load imbalance (CV%) per config.
Optional --raw-out writes a JSON dump for downstream analysis.

Use cases (per analyze-ep-tp skill, "Step 4: Analyze Traces"):
- Detect MoE compute imbalance across EP ranks (CV% > 30 -> pathological skew)
- Compare attention/MoE/comm time across configs at matched equiv batch
- Identify barrier amplification: rank with HEAVIEST upstream MoE has LOWEST
  notify-kernel duration (inverse correlation = barrier wait, not real cost)

For nsys traces (CUDA-graph-aware), use sister script analyze_nsys.py.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

NUM_GPUS_DEFAULT = 8

DEEPEP_BARRIER_KW = ["cached_notify_combine", "notify_dispatch", "notify_combine"]


def classify_kernel(name: str) -> str:
    """Map a kernel name to a fine-grained category.

    Categories follow the analyze-ep-tp skill's "Kernel classification rules"
    section (Phase: Profiling Methodology > Step 4). Sync barrier kernels are
    SEPARATE from data-movement comm kernels because per-call duration includes
    barrier wait time and must be interpreted via per-call min/max, NOT mean.
    """
    n = name
    if any(s in n for s in DEEPEP_BARRIER_KW):
        return "comm_barrier"
    if "deep_ep::" in n and ("dispatch" in n or "combine" in n):
        if "combine" in n:
            return "comm_deepep_combine"
        return "comm_deepep_dispatch"
    if "deep_ep::layout::get_dispatch_layout" in n:
        return "comm_deepep_dispatch"
    if "cross_device_reduce" in n:
        return "comm_allreduce"
    if "ncclDevKernel" in n and "AllReduce" in n:
        return "comm_allreduce"
    if "ncclDevKernel" in n and "AllGather" in n:
        return "comm_allgather"
    if "ncclDevKernel" in n:
        return "comm_nccl"
    if "all_reduce" in n.lower() and "sgl_kernel" in n:
        return "comm_allreduce"

    if "fused_moe_kernel" in n or "inplace_fused_experts" in n:
        return "moe_compute"
    if "moe_sum_reduce" in n:
        return "moe_compute"
    if "topkGatingSoftmax" in n or "topk_softmax" in n:
        return "moe_gate"
    if "moe_align_block_size" in n or "count_and_sort_expert_tokens" in n:
        return "moe_overhead"
    if "_silu_and_mul_masked_kernel" in n or "swiglu_with_alpha_and_limit_masked" in n:
        return "moe_activation"
    if "swiglu_with_alpha_and_limit" in n or "triton_poi_fused_add_clamp_mul_sigmoid" in n:
        return "moe_activation"
    if "_fwd_kernel_ep_scatter" in n or "_fwd_kernel_ep_gather" in n:
        return "moe_overhead"
    if "silu_and_mul" in n and "masked" not in n:
        return "moe_activation"
    if "act_and_mul" in n:
        return "moe_activation"

    if "flashinfer::BatchPrefill" in n or "flashinfer::PersistentVariableLengthMergeStates" in n:
        return "attn_compute"
    if "_fwd_kernel" in n or "_fwd_grouped_kernel_stage" in n:
        return "attn_compute"
    if "flashinfer::BatchQKApplyRotary" in n or "rope" in n.lower() or "apply_rope" in n:
        return "attn_rope"
    if "create_flashinfer_kv_indices" in n:
        return "attn_overhead"
    if "cutlass::Kernel" in n or "ampere_bf16_s16816gemm" in n:
        return "attn_proj"
    if "cublasLt" in n or "cublas" in n.lower():
        return "attn_proj"

    if "RMSNorm" in n or "rmsnorm" in n.lower() or "FusedAddRMSNorm" in n:
        return "norm"
    return "other"


HIGH_LEVEL = {
    "attn_compute": "Attention",
    "attn_rope": "Attention",
    "attn_proj": "Attention",
    "attn_overhead": "Attention",
    "comm_allreduce": "Communication",
    "comm_allgather": "Communication",
    "comm_nccl": "Communication",
    "comm_deepep_dispatch": "Communication",
    "comm_deepep_combine": "Communication",
    "comm_barrier": "Comm-Barrier",
    "moe_compute": "MoE Compute",
    "moe_gate": "MoE Compute",
    "moe_overhead": "MoE Compute",
    "moe_activation": "MoE Compute",
    "norm": "Other",
    "other": "Other",
}


def parse_trace(filepath: Path) -> dict:
    open_func = gzip.open if filepath.suffix == ".gz" else open
    with open_func(filepath, "rt") as f:
        data = json.load(f)
    events = data.get("traceEvents", data if isinstance(data, list) else [])
    kernel_events = [e for e in events if isinstance(e, dict) and e.get("cat") == "kernel"]

    fine: dict[str, dict] = defaultdict(lambda: {"count": 0, "total_us": 0.0})
    for e in kernel_events:
        cat = classify_kernel(e.get("name", ""))
        fine[cat]["count"] += 1
        fine[cat]["total_us"] += e.get("dur", 0)

    agg: dict[str, dict] = defaultdict(lambda: {"count": 0, "total_us": 0.0})
    for cat, info in fine.items():
        high = HIGH_LEVEL.get(cat, "Other")
        agg[high]["count"] += info["count"]
        agg[high]["total_us"] += info["total_us"]
    return {"fine": dict(fine), "agg": dict(agg)}


TRACE_NAME_RE = re.compile(
    r"^(?P<config>.+?)_rank(?P<rank>\d+)_batch(?P<batch>\d+)_.*?_(?P<stage>decode|prefill|extend)\.trace\.json(?:\.gz)?$"
)


def discover_traces(trace_dir: Path) -> list[dict]:
    by_key: dict[tuple, dict[int, str]] = defaultdict(dict)
    for f in sorted(trace_dir.rglob("*.trace.json*")):
        m = TRACE_NAME_RE.match(f.name)
        if not m:
            continue
        cfg = m.group("config")
        batch = int(m.group("batch"))
        rank = int(m.group("rank"))
        stage = m.group("stage")
        by_key[(cfg, batch, stage)][rank] = str(f)
    configs = []
    for (cfg, batch, stage), ranks in sorted(by_key.items()):
        configs.append({"config": cfg, "batch": batch, "stage": stage, "ranks": ranks})
    return configs


def equiv_batch(config: str, batch: int, num_gpus: int) -> int:
    return batch * num_gpus if config.startswith("dp_") else batch


def fmt_us(us: float) -> str:
    if us >= 1000:
        return f"{us/1000:.1f}ms"
    return f"{us:.0f}us"


def print_per_rank(rank_data: dict, config: str, batch: int, stage: str, num_gpus: int) -> None:
    categories = ["Attention", "Communication", "Comm-Barrier", "MoE Compute", "Other"]
    eq = equiv_batch(config, batch, num_gpus)
    print(f"\n{'=' * 110}")
    print(f"  {config.upper()}  batch={batch}  equiv={eq}  {stage}  per-rank kernel time (us)")
    print(f"{'=' * 110}")
    header = f"  {'Rank':<6}" + "".join(f" | {c:>15}" for c in categories) + f" | {'TOTAL':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    cat_vals: dict[str, list[float]] = {c: [] for c in categories}
    rank_totals: list[float] = []
    for rank in sorted(rank_data.keys()):
        agg = rank_data[rank]["agg"]
        line = f"  R{rank:<5}"
        total = 0.0
        for cat in categories:
            us = agg.get(cat, {}).get("total_us", 0)
            cat_vals[cat].append(us)
            total += us
            line += f" | {us:>15.0f}"
        rank_totals.append(total)
        line += f" | {total:>12.0f}"
        print(line)

    n = len(rank_data)
    if n > 1:
        print("  " + "-" * (len(header) - 2))
        for label, fn in (("AVG", lambda v: sum(v) / n), ("SPREAD", lambda v: max(v) - min(v))):
            ln = f"  {label:<6}"
            for cat in categories:
                ln += f" | {fn(cat_vals[cat]):>15.0f}"
            ln += f" | {fn(rank_totals):>12.0f}"
            print(ln)
        ln = f"  {'CV%':<6}"
        mean_total = sum(rank_totals) / n
        for cat in categories:
            v = cat_vals[cat]
            mean = sum(v) / n
            cv = (max(v) - min(v)) / mean * 100 if mean > 0 else 0
            ln += f" | {cv:>14.1f}%"
        cv_total = (max(rank_totals) - min(rank_totals)) / mean_total * 100 if mean_total > 0 else 0
        ln += f" | {cv_total:>11.1f}%"
        print(ln)

    if n > 1:
        compute_max = max(
            sum(rank_data[r]["agg"].get(c, {}).get("total_us", 0) for c in ("Attention", "MoE Compute", "Other"))
            for r in rank_data
        )
        comm_min = min(
            rank_data[r]["agg"].get("Communication", {}).get("total_us", 0)
            for r in rank_data
        )
        eff = compute_max + comm_min
        print(f"  {'EFF':<6}  max(compute) + min(comm) = {fmt_us(eff)}  "
              f"(compute_max={fmt_us(compute_max)}, comm_min={fmt_us(comm_min)})")


def save_raw(all_results: dict, path: Path) -> None:
    serial: dict = {}
    for (cfg, batch, stage), rank_data in all_results.items():
        key = f"{cfg}_batch{batch}_{stage}"
        serial[key] = {f"rank{r}": d for r, d in rank_data.items()}
    path.write_text(json.dumps(serial, indent=2))
    print(f"\nRaw data saved to {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("trace_dir", type=Path, help="Directory containing *.trace.json.gz files (one per rank).")
    ap.add_argument("--num-gpus", type=int, default=NUM_GPUS_DEFAULT,
                    help="dp_size for DP-attention configs (default 8).")
    ap.add_argument("--raw-out", type=Path, default=None,
                    help="Optional JSON dump of per-rank fine+agg data.")
    args = ap.parse_args()

    if not args.trace_dir.is_dir():
        print(f"Not a directory: {args.trace_dir}", file=sys.stderr)
        sys.exit(1)

    configs = discover_traces(args.trace_dir)
    if not configs:
        print(f"No *.trace.json[.gz] files matching expected pattern under {args.trace_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(configs)} (config, batch, stage) groups:")
    for c in configs:
        print(f"  {c['config']:>6} batch={c['batch']:<6} stage={c['stage']:<8} ranks={sorted(c['ranks'].keys())}")

    all_results: dict = {}
    for c in configs:
        rank_data: dict[int, dict] = {}
        for rank, fp in sorted(c["ranks"].items()):
            rank_data[rank] = parse_trace(Path(fp))
        all_results[(c["config"], c["batch"], c["stage"])] = rank_data
        print_per_rank(rank_data, c["config"], c["batch"], c["stage"], args.num_gpus)

    if args.raw_out:
        save_raw(all_results, args.raw_out)


if __name__ == "__main__":
    main()
