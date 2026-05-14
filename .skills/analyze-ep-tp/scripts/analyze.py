"""Compute per-metric best-of-N from bench_one_batch JSONL output.

Reads a JSONL file produced by `python -m sglang.bench_one_batch
--result-filename ...`, takes the MAX value across passes per (run_name,
batch_size) for each throughput metric, scales DP-attention configs to system
throughput (multiplied by dp_size), and prints + writes a markdown report
with one table per metric.

Recognised run_names: tp_tp, tp_ep (TP attention -> dp_size=1) and
dp_tp, dp_ep (DP attention -> dp_size=NUM_GPUS, default 8).

Sister scripts: format_bench.py (raw inspection, all passes) and plot.py
(rendered figure).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

DP_SIZE_DEFAULT = {"tp_tp": 1, "tp_ep": 1, "dp_tp": 8, "dp_ep": 8}


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def best_per_metric(rows: list[dict]) -> dict[tuple[str, int], dict]:
    """Per (run_name, batch_size), keep MAX value independently for each throughput metric.

    Per-metric best-of-N is fairer than best-total-latency: cold/warm passes can
    favor prefill or decode differently (DeepEP prefill especially noisy).
    """
    by_key: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in rows:
        by_key[(r["run_name"], r["batch_size"])].append(r)
    best: dict[tuple[str, int], dict] = {}
    for k, lst in by_key.items():
        best[k] = {
            "run_name": k[0],
            "batch_size": k[1],
            "prefill_throughput": max(r["prefill_throughput"] for r in lst),
            "median_decode_throughput": max(r["median_decode_throughput"] for r in lst),
            "overall_throughput": max(r["overall_throughput"] for r in lst),
            "n_passes": len(lst),
        }
    return best


def equiv_batch(run_name: str, batch_size: int, dp_size_map: dict[str, int]) -> int:
    return batch_size * dp_size_map.get(run_name, 1)


def system_throughput(run_name: str, throughput: float, dp_size_map: dict[str, int]) -> float:
    return throughput * dp_size_map.get(run_name, 1)


def fmt_int(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "  -- "
    return f"{int(x):>7,}"


def fmt_ratio(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "  -- "
    return f"{x:.2f}x"


def build_tables(best: dict, dp_size_map: dict[str, int]) -> dict:
    """Return {metric -> {equiv_batch -> {run_name -> system_throughput}}}, metric in {prefill, decode, overall}."""
    tables: dict[str, dict[int, dict[str, float]]] = {
        "prefill": defaultdict(dict),
        "decode": defaultdict(dict),
        "overall": defaultdict(dict),
    }
    for (run_name, bs), row in best.items():
        eb = equiv_batch(run_name, bs, dp_size_map)
        tables["prefill"][eb][run_name] = system_throughput(run_name, row["prefill_throughput"], dp_size_map)
        tables["decode"][eb][run_name] = system_throughput(run_name, row["median_decode_throughput"], dp_size_map)
        tables["overall"][eb][run_name] = system_throughput(run_name, row["overall_throughput"], dp_size_map)
    return tables


def print_table(tables: dict, metric: str, configs: list[str]) -> None:
    eqs = sorted(tables[metric].keys())
    header = f"  {'equiv_bs':>8}  " + "  ".join(f"{c:>10}" for c in configs)
    if "tp_tp" in configs:
        header += "  " + "  ".join(f"{c}/tp_tp" for c in configs if c != "tp_tp")
    print()
    print(f"=== {metric.upper()} (system tok/s) ===")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for eb in eqs:
        cells = [tables[metric][eb].get(c) for c in configs]
        line = f"  {eb:>8}  " + "  ".join(fmt_int(v) for v in cells)
        if "tp_tp" in configs:
            base = tables[metric][eb].get("tp_tp")
            ratios = []
            for c in configs:
                if c == "tp_tp":
                    continue
                v = tables[metric][eb].get(c)
                ratios.append(v / base if base and v else None)
            line += "  " + "  ".join(fmt_ratio(r) for r in ratios)
        print(line)


def write_markdown(tables: dict, runs: list[str], src: Path, n_rows: int, n_best: int, out_md: Path) -> None:
    md: list[str] = []
    md.append(f"# bench_one_batch sweep analysis")
    md.append("")
    md.append(f"Source: `{src}`")
    md.append(f"Total rows: {n_rows}; best-of-N entries: {n_best}")
    md.append("")
    for metric, label in [
        ("prefill", "Prefill"),
        ("decode", "Decode (median per-iter)"),
        ("overall", "Overall (prefill+decode)"),
    ]:
        eqs = sorted(tables[metric].keys())
        md.append(f"## {label} system throughput (tok/s)")
        md.append("")
        cols = ["equiv_batch", *runs]
        if "tp_tp" in runs:
            cols += [f"{c}/tp_tp" for c in runs if c != "tp_tp"]
        md.append("| " + " | ".join(cols) + " |")
        md.append("|" + "|".join(["---:"] * len(cols)) + "|")
        for eb in eqs:
            row_vals = [str(eb)]
            for c in runs:
                v = tables[metric][eb].get(c)
                row_vals.append(f"{int(v):,}" if v else "--")
            base = tables[metric][eb].get("tp_tp")
            for c in runs:
                if c == "tp_tp":
                    continue
                v = tables[metric][eb].get(c)
                row_vals.append(f"{v/base:.2f}x" if base and v else "--")
            md.append("| " + " | ".join(row_vals) + " |")
        md.append("")
    md.append("## Notes")
    md.append("")
    md.append("- System throughput = bench reported * dp_size (= 1 for TP attention, NUM_GPUS for DP attention).")
    md.append("- Best-of-N: batch sizes appearing twice in the bench cmdline produce two rows; MAX kept independently per metric.")
    md.append("- Decode throughput is the median across 10 iters (bench_one_batch default).")
    md.append("")
    out_md.write_text("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results", type=Path, help="Path to bench_one_batch JSONL output.")
    ap.add_argument("--out-md", type=Path, default=None,
                    help="Write a markdown report. Defaults to <results>.report.md.")
    ap.add_argument("--num-gpus", type=int, default=8,
                    help="dp_size for DP-attention configs (default 8).")
    args = ap.parse_args()

    dp_size_map = {**DP_SIZE_DEFAULT, "dp_tp": args.num_gpus, "dp_ep": args.num_gpus}

    rows = load_rows(args.results)
    if not rows:
        print(f"No rows in {args.results}")
        return
    best = best_per_metric(rows)
    tables = build_tables(best, dp_size_map)
    runs = sorted({k[0] for k in best.keys()})
    print(f"Loaded {len(rows)} rows -> {len(best)} best-of-N entries.")
    print(f"Configs found: {runs}")

    for metric in ("prefill", "decode", "overall"):
        print_table(tables, metric, runs)

    out_md = args.out_md or args.results.with_suffix(".report.md")
    write_markdown(tables, runs, args.results, len(rows), len(best), out_md)
    print(f"\nReport written to {out_md}")


if __name__ == "__main__":
    main()
