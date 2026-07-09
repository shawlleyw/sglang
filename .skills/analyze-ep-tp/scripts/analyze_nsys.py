"""Effective-latency + axis decomposition from nsys traces (multi-rank, CUDA-graph-aware).

Input: a directory of `*.nsys-rep` files, one per (config, batch, stage). Each
trace is expected to be a single-process trace covering all ranks of one run
(captured with `nsys profile -t cuda --cuda-graph-trace=node` per the
analyze-ep-tp skill, "Step 3: nsys Profiling").

Filename convention (auto-discovered):
    <config>_bs<N>_<stage>[_<suffix>].nsys-rep
e.g. tp_tp_bs2048_prefill_sharegpt.nsys-rep,  dp_ep_bs256_decode.nsys-rep.

`config` should be one of: tp_tp, tp_ep, dp_tp, dp_ep.
`stage`  should be one of: prefill, decode.

Output: markdown report with
1. Per-rank kernel time by category (attn / comm / moe / norm / other) per config
2. Effective latency (max for compute, min for comm) per config
3. Cross-config delta vs TP/TP at matched equiv batch
4. CV% (load imbalance) per category per config
5. DeepEP volume-imbalance view (max-rank for comm, upper bound)
6. Top-5 kernels per config

Sister script: analyze_traces.py (torch.profiler, no CUDA graph).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

LABEL = {"tp_tp": "TP/TP", "tp_ep": "TP/EP", "dp_tp": "DP/TP", "dp_ep": "DP/EP"}

DEEPEP_BARRIER_KW = ["cached_notify_combine", "notify_dispatch", "notify_combine"]


def classify(name: str) -> str:
    """Map a kernel name to {comm, comm_barrier, moe, attn, attn_or_lm_head_gemm, norm, other}.

    Per skill (Step 4 + barrier-kernels caveat): barrier kernels are separated
    from data-movement comm because per-kernel nsys duration includes wait time.
    """
    n = name.lower()
    if any(s in n for s in DEEPEP_BARRIER_KW):
        return "comm_barrier"
    if any(s in n for s in [
        "deep_ep::", "ncclkernel", "nccldevkernel", "ncclalldevice",
        "ncclallreduce", "ncclalltoall", "ncclallgather",
        "cross_device_reduce", "ncclredopkernel", "all_reduce", "allreduce",
        "all_gather", "allgather",
    ]):
        return "comm"
    if any(s in n for s in [
        "fused_moe", "topkgating", "topk_softmax",
        "silu_and_mul_masked", "moe_align", "moe_sum_reduce",
        "ep_scatter", "ep_gather", "count_and_sort_expert",
        "_silu_and_mul_kernel", "_silu_and_mul_post_quant",
        "triton_poi_fused_add_clamp_mul_sigmoid", "swiglu_with_alpha",
        "act_and_mul",
    ]):
        return "moe"
    if any(s in n for s in [
        "flashinfer", "batchprefill", "batchqkapply", "rotary",
        "create_flashinfer", "persistentvariable",
        "_fwd_kernel", "_fwd_grouped_kernel",
    ]):
        return "attn"
    if "rmsnorm" in n or "fusedaddrmsnorm" in n or "rms_norm" in n:
        return "norm"
    if any(s in n for s in [
        "cutlass", "ampere_bf16", "cublaslt", "gemm",
        "ampere_fp16", "sm80_xmma",
    ]):
        return "attn_or_lm_head_gemm"
    return "other"


REP_NAME_RE = re.compile(
    r"^(?P<config>tp_tp|tp_ep|dp_tp|dp_ep)_bs(?P<batch>\d+)_(?P<stage>prefill|decode)(?:_.+)?\.nsys-rep$"
)


def discover_reps(trace_dir: Path) -> list[dict]:
    out: list[dict] = []
    for f in sorted(trace_dir.glob("*.nsys-rep")):
        m = REP_NAME_RE.match(f.name)
        if not m:
            continue
        out.append({
            "config": m.group("config"),
            "batch": int(m.group("batch")),
            "stage": m.group("stage"),
            "rep": f,
        })
    return out


def run_nsys_stats(rep: Path) -> Path:
    csv_path = rep.with_suffix("").as_posix() + "_cuda_gpu_trace.csv"
    if Path(csv_path).exists():
        return Path(csv_path)
    print(f"Extracting CSV from {rep.name}...")
    subprocess.run(
        ["nsys", "stats", "-r", "cuda_gpu_trace", str(rep),
         "-o", rep.with_suffix("").as_posix(), "-f", "csv"],
        check=True, capture_output=True,
    )
    return Path(csv_path)


DEVICE_RE = re.compile(r"\((\d+)\)")


def parse_csv(csv_path: Path) -> dict:
    by_dev_kern: dict = defaultdict(lambda: defaultdict(lambda: {"count": 0, "dur_ns": 0.0}))
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            device_str = row.get("Device", "") or ""
            m = DEVICE_RE.search(device_str)
            if not m:
                continue
            device = int(m.group(1))
            name = (row.get("Name") or "").strip()
            if not name or name.startswith("[CUDA mem"):
                continue
            try:
                dur_ns = float(row.get("Duration (ns)") or row.get("Dur (ns)") or 0)
            except ValueError:
                continue
            by_dev_kern[device][name]["count"] += 1
            by_dev_kern[device][name]["dur_ns"] += dur_ns
    return by_dev_kern


def aggregate_per_rank(by_dev_kern: dict) -> dict:
    per_rank: dict = {}
    for device, kernels in by_dev_kern.items():
        cats: dict[str, float] = defaultdict(float)
        kernel_breakdown: dict[str, dict] = {}
        for name, stats in kernels.items():
            cat = classify(name)
            ms = stats["dur_ns"] / 1e6
            cats[cat] += ms
            kernel_breakdown[name] = {"count": stats["count"], "total_ms": ms, "category": cat}
        per_rank[device] = {
            "categories": dict(cats),
            "top_kernels": dict(sorted(kernel_breakdown.items(), key=lambda x: -x[1]["total_ms"])[:15]),
        }
    return per_rank


def effective_latency(per_rank: dict) -> dict:
    """max-rank for compute, min-rank for comm + comm_barrier (skill methodology)."""
    if not per_rank:
        return {}
    cats: set = set()
    for r in per_rank.values():
        cats.update(r["categories"].keys())
    rule = {"comm": min, "comm_barrier": min}
    eff: dict = {}
    for c in sorted(cats):
        vals = [per_rank[r]["categories"].get(c, 0.0) for r in per_rank]
        eff[c] = (rule.get(c, max))(vals)
    return eff


def cv_pct(per_rank: dict, cat: str) -> float:
    vals = [per_rank[r]["categories"].get(cat, 0.0) for r in per_rank]
    if not vals:
        return 0.0
    mean = sum(vals) / len(vals)
    return (max(vals) - min(vals)) / mean * 100 if mean > 0 else 0.0


def build_report(reps: list[dict], num_gpus: int) -> tuple[list[str], dict]:
    out: dict = {"configs": {}}
    by_stage: dict[str, list[dict]] = defaultdict(list)

    for r in reps:
        csv_path = run_nsys_stats(r["rep"])
        by_dev = parse_csv(csv_path)
        per_rank = aggregate_per_rank(by_dev)
        eff = effective_latency(per_rank)
        eq = r["batch"] * num_gpus if r["config"].startswith("dp_") else r["batch"]
        rec = {
            "config": r["config"],
            "batch_per_rank": r["batch"],
            "stage": r["stage"],
            "equiv_batch": eq,
            "rep": str(r["rep"]),
            "csv": str(csv_path),
            "per_rank": per_rank,
            "effective_ms": eff,
        }
        out["configs"][f"{r['config']}_bs{r['batch']}_{r['stage']}"] = rec
        by_stage[r["stage"]].append(rec)

    cats_set: set = set()
    for cfg_data in out["configs"].values():
        for r in cfg_data["per_rank"].values():
            cats_set.update(r["categories"].keys())
    cats = sorted(cats_set)

    md: list[str] = []
    md.append("# nsys Effective-Latency Analysis\n\n")
    md.append("Per analyze-ep-tp skill, Step 4 + Analysis Methodology: ")
    md.append("effective latency = max_rank(compute) + min_rank(comm).\n\n")

    for stage, records in sorted(by_stage.items()):
        md.append(f"\n---\n\n# {stage.upper()}\n\n")

        for rec in records:
            md.append(f"### {LABEL.get(rec['config'], rec['config'])}  "
                      f"bs={rec['batch_per_rank']}  equiv={rec['equiv_batch']}  ({rec['stage']})\n\n")
            per_rank = rec["per_rank"]
            md.append("| Rank | " + " | ".join(cats) + " | total |\n")
            md.append("|---:|" + "---:|" * (len(cats) + 1) + "\n")
            for r_id in sorted(per_rank):
                row = [str(r_id)]
                tot = 0.0
                for c in cats:
                    v = per_rank[r_id]["categories"].get(c, 0.0)
                    row.append(f"{v:.1f}")
                    tot += v
                row.append(f"**{tot:.1f}**")
                md.append("| " + " | ".join(row) + " |\n")
            md.append("\n")

        md.append(f"## {stage.upper()} effective latency per config (ms)\n\n")
        md.append("max_rank for compute, min_rank for comm/comm_barrier.\n\n")
        md.append("| Config | equiv_bs | " + " | ".join(cats) + " | **eff_total** |\n")
        md.append("|---|---:|" + "---:|" * (len(cats) + 1) + "\n")
        for rec in sorted(records, key=lambda r: (r["equiv_batch"], r["config"])):
            cells = [LABEL.get(rec["config"], rec["config"]), str(rec["equiv_batch"])]
            tot = 0.0
            for c in cats:
                v = rec["effective_ms"].get(c, 0.0)
                cells.append(f"{v:.1f}")
                tot += v
            cells.append(f"**{tot:.1f}**")
            md.append("| " + " | ".join(cells) + " |\n")
        md.append("\n")

        baseline = next((r for r in records if r["config"] == "tp_tp"), None)
        if baseline is not None:
            md.append(f"## {stage.upper()} effective savings vs TP/TP (ms; negative = faster)\n\n")
            md.append("| Config | equiv_bs | " + " | ".join(cats) + " | **total** |\n")
            md.append("|---|---:|" + "---:|" * (len(cats) + 1) + "\n")
            for rec in sorted(records, key=lambda r: (r["equiv_batch"], r["config"])):
                if rec["config"] == "tp_tp":
                    continue
                if rec["equiv_batch"] != baseline["equiv_batch"]:
                    continue
                cells = [LABEL.get(rec["config"], rec["config"]), str(rec["equiv_batch"])]
                base_total = 0.0
                rec_total = 0.0
                for c in cats:
                    a = baseline["effective_ms"].get(c, 0.0)
                    b = rec["effective_ms"].get(c, 0.0)
                    cells.append(f"{b - a:+.1f}")
                    base_total += a
                    rec_total += b
                cells.append(f"**{rec_total - base_total:+.1f}**")
                md.append("| " + " | ".join(cells) + " |\n")
            md.append("\n")

        md.append(f"## {stage.upper()} per-rank CV% (load imbalance)\n\n")
        md.append("CV% = (max - min)/mean * 100 across ranks. Per skill: ")
        md.append("expect 5-15% MoE at large batch with real weights; >30% = pathological skew.\n\n")
        md.append("| Config | equiv_bs | " + " | ".join(cats) + " |\n")
        md.append("|---|---:|" + "---:|" * len(cats) + "\n")
        for rec in sorted(records, key=lambda r: (r["equiv_batch"], r["config"])):
            cells = [LABEL.get(rec["config"], rec["config"]), str(rec["equiv_batch"])]
            for c in cats:
                cells.append(f"{cv_pct(rec['per_rank'], c):.1f}%")
            md.append("| " + " | ".join(cells) + " |\n")
        md.append("\n")

        md.append(f"## {stage.upper()} top 5 kernels per config (sum across ranks)\n\n")
        for rec in sorted(records, key=lambda r: (r["equiv_batch"], r["config"])):
            md.append(f"### {LABEL.get(rec['config'], rec['config'])}  equiv={rec['equiv_batch']}\n\n")
            md.append("| Kernel | Category | Total ms (sum ranks) | Calls (rank 0) |\n")
            md.append("|---|---|---:|---:|\n")
            kernel_tot: dict = defaultdict(lambda: {"total_ms": 0.0, "count": 0, "category": ""})
            for r_id, data in rec["per_rank"].items():
                for k, info in data["top_kernels"].items():
                    kernel_tot[k]["total_ms"] += info["total_ms"]
                    kernel_tot[k]["category"] = info["category"]
                    if r_id == 0:
                        kernel_tot[k]["count"] = info["count"]
            top = sorted(kernel_tot.items(), key=lambda x: -x[1]["total_ms"])[:5]
            for k, info in top:
                k_short = k[:80] + "..." if len(k) > 80 else k
                md.append(f"| `{k_short}` | {info['category']} | {info['total_ms']:.1f} | {info['count']} |\n")
            md.append("\n")

    return md, out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("trace_dir", type=Path,
                    help="Directory containing *.nsys-rep files matching <config>_bs<N>_<stage>[_<suffix>].nsys-rep.")
    ap.add_argument("--out-md", type=Path, default=None,
                    help="Markdown report path. Default: <trace_dir>/report_nsys.md.")
    ap.add_argument("--out-json", type=Path, default=None,
                    help="JSON dump path. Default: <trace_dir>/effective_latency.json.")
    ap.add_argument("--num-gpus", type=int, default=8,
                    help="dp_size for DP-attention configs (default 8).")
    args = ap.parse_args()

    if not args.trace_dir.is_dir():
        print(f"Not a directory: {args.trace_dir}", file=sys.stderr)
        sys.exit(1)

    reps = discover_reps(args.trace_dir)
    if not reps:
        print(f"No *.nsys-rep matching pattern in {args.trace_dir}", file=sys.stderr)
        print("Expected: <config>_bs<N>_<stage>[_<suffix>].nsys-rep "
              "(config in {tp_tp,tp_ep,dp_tp,dp_ep}; stage in {prefill,decode})", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(reps)} traces:")
    for r in reps:
        eq = r["batch"] * args.num_gpus if r["config"].startswith("dp_") else r["batch"]
        print(f"  {r['config']:>5} bs={r['batch']:<5} stage={r['stage']:<8} equiv={eq:<5} {r['rep'].name}")

    md, out = build_report(reps, args.num_gpus)
    md_path = args.out_md or (args.trace_dir / "report_nsys.md")
    json_path = args.out_json or (args.trace_dir / "effective_latency.json")
    md_path.write_text("".join(md))
    json_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {md_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
