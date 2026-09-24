"""Plot prefill & decode system throughput vs equiv batch size for the 4 parallelism configs.

Two subplots side-by-side (Prefill | Decode), log-log axes, one line per config
present in the input (TP/TP, TP/EP, DP/TP, DP/EP). Per-metric best-of-N applied.
Saves PNG and PDF.

Sister scripts: analyze.py (tabular best-of-N report) and format_bench.py (raw inspection).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

DP_SIZE_DEFAULT = {"tp_tp": 1, "tp_ep": 1, "dp_tp": 8, "dp_ep": 8}

CONFIG_ORDER = ["tp_tp", "tp_ep", "dp_tp", "dp_ep"]
CONFIG_LABEL = {
    "tp_tp": "TP/TP  (Attn TP, Exp TP)",
    "tp_ep": "TP/EP  (Attn TP, Exp EP-AR)",
    "dp_tp": "DP/TP  (Attn DP, Exp TP)",
    "dp_ep": "DP/EP  (Attn DP, Exp DeepEP)",
}
CONFIG_COLOR = {
    "tp_tp": "#1f77b4",
    "tp_ep": "#ff7f0e",
    "dp_tp": "#2ca02c",
    "dp_ep": "#d62728",
}
CONFIG_MARKER = {"tp_tp": "o", "tp_ep": "s", "dp_tp": "^", "dp_ep": "D"}


def load_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def best_per_metric(rows: list[dict]) -> dict[tuple[str, int], dict]:
    by_key: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in rows:
        by_key[(r["run_name"], r["batch_size"])].append(r)
    return {
        k: {
            "prefill_throughput": max(r["prefill_throughput"] for r in lst),
            "median_decode_throughput": max(r["median_decode_throughput"] for r in lst),
            "overall_throughput": max(r["overall_throughput"] for r in lst),
        }
        for k, lst in by_key.items()
    }


def series(best: dict, run: str, metric: str, dp_size_map: dict[str, int]) -> tuple[list[int], list[float]]:
    dp = dp_size_map.get(run, 1)
    pts = sorted(
        (bs * dp, v[metric] * dp)
        for (rn, bs), v in best.items()
        if rn == run
    )
    if not pts:
        return [], []
    xs, ys = zip(*pts)
    return list(xs), list(ys)


def plot(best: dict, dp_size_map: dict[str, int], out_path: Path, title: str | None = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=False)
    ax_pre, ax_dec = axes

    runs_present = sorted(
        {rn for (rn, _) in best.keys()},
        key=lambda x: CONFIG_ORDER.index(x) if x in CONFIG_ORDER else 99,
    )

    for run in runs_present:
        for ax, metric in [(ax_pre, "prefill_throughput"), (ax_dec, "median_decode_throughput")]:
            xs, ys = series(best, run, metric, dp_size_map)
            if not xs:
                continue
            ax.plot(
                xs, ys,
                marker=CONFIG_MARKER.get(run, "o"),
                color=CONFIG_COLOR.get(run),
                linewidth=2,
                markersize=7,
                label=CONFIG_LABEL.get(run, run),
            )

    for ax, label in [(ax_pre, "Prefill"), (ax_dec, "Decode (median per-iter)")]:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Equivalent global batch size", fontsize=11)
        ax.set_ylabel("System throughput (tok/s)", fontsize=11)
        ax.set_title(label, fontsize=12)
        ax.grid(True, which="both", linestyle=":", alpha=0.6)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
        ax.ticklabel_format(axis="y", style="plain")
        ax.set_ylim(bottom=0)
        all_xs = sorted({bs * dp_size_map.get(rn, 1) for (rn, bs) in best.keys()})
        ax.set_xticks(all_xs)
        ax.set_xticklabels([str(x) for x in all_xs], rotation=0)

    if title:
        fig.suptitle(title, fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    out_pdf = out_path.with_suffix(".pdf")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved {out_pdf}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results", type=Path, help="Path to bench_one_batch JSONL output.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output PNG path. Defaults to <results>.throughput.png.")
    ap.add_argument("--title", default=None,
                    help="Optional figure suptitle.")
    ap.add_argument("--num-gpus", type=int, default=8,
                    help="dp_size for DP-attention configs (default 8).")
    args = ap.parse_args()

    dp_size_map = {**DP_SIZE_DEFAULT, "dp_tp": args.num_gpus, "dp_ep": args.num_gpus}

    rows = load_rows(args.results)
    print(f"Loaded {len(rows)} rows from {args.results}")
    best = best_per_metric(rows)
    runs = sorted({k[0] for k in best.keys()})
    print(f"Configs: {runs}")
    out = args.out or args.results.with_suffix(".throughput.png")
    plot(best, dp_size_map, out, args.title)


if __name__ == "__main__":
    main()
