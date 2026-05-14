"""Pretty-print raw bench_one_batch JSONL output as a readable table.

Reads one or more JSONL files (or stdin), shows ALL passes verbatim grouped by
run_name and equiv batch. Use this to inspect cold-vs-warm spread and decide
whether more passes are needed. For collapsed best-of-N see analyze.py; for
rendered plots see plot.py.

Recognised run_names: tp_tp, tp_ep (TP attention -> dp_size=1) and
dp_tp, dp_ep (DP attention -> dp_size=NUM_GPUS, default 8).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DP_SIZE_DEFAULT = {"tp_tp": 1, "tp_ep": 1, "dp_tp": 8, "dp_ep": 8}


def load(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    if not paths:
        rows = [json.loads(l) for l in sys.stdin if l.strip()]
    else:
        for p in paths:
            with p.open() as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
    return rows


def equiv_batch(r: dict, dp_size_map: dict[str, int]) -> int:
    return r["batch_size"] * dp_size_map.get(r["run_name"], 1)


def system_tps(r: dict, key: str, dp_size_map: dict[str, int]) -> float:
    return r[key] * dp_size_map.get(r["run_name"], 1)


HEADER = (
    f"{'run':>6}  {'eq_bs':>6}  {'bs':>5}  "
    f"{'pre_lat_s':>10}  {'pre_tps':>10}  {'pre_sys':>10}  "
    f"{'dec_lat_s':>10}  {'dec_tps':>10}  {'dec_sys':>10}  "
    f"{'tot_lat_s':>10}  {'ovr_tps':>10}  {'ovr_sys':>10}  pass"
)


def fmt_row(r: dict, pass_idx: int, dp_size_map: dict[str, int]) -> str:
    return (
        f"{r['run_name']:>6}  {equiv_batch(r, dp_size_map):>6}  {r['batch_size']:>5}  "
        f"{r['prefill_latency']:>10.4f}  "
        f"{r['prefill_throughput']:>10,.0f}  "
        f"{system_tps(r, 'prefill_throughput', dp_size_map):>10,.0f}  "
        f"{r['median_decode_latency']:>10.5f}  "
        f"{r['median_decode_throughput']:>10,.0f}  "
        f"{system_tps(r, 'median_decode_throughput', dp_size_map):>10,.0f}  "
        f"{r['total_latency']:>10.4f}  "
        f"{r['overall_throughput']:>10,.0f}  "
        f"{system_tps(r, 'overall_throughput', dp_size_map):>10,.0f}  "
        f"{pass_idx}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="*", type=Path,
                    help="One or more JSONL files. Reads stdin if omitted.")
    ap.add_argument("--csv", action="store_true",
                    help="Emit CSV instead of fixed-width text.")
    ap.add_argument("--sort", choices=["input", "config_eqbs", "eqbs"],
                    default="config_eqbs",
                    help="Row order: input=raw file order; config_eqbs=group by run then eq_bs (default); eqbs=group by eq_bs across configs.")
    ap.add_argument("--num-gpus", type=int, default=8,
                    help="dp_size for DP-attention configs (default 8).")
    args = ap.parse_args()

    dp_size_map = {**DP_SIZE_DEFAULT, "dp_tp": args.num_gpus, "dp_ep": args.num_gpus}

    rows = load(args.paths)
    if not rows:
        print("(no rows)", file=sys.stderr)
        return

    if args.sort == "input":
        ordered = list(enumerate(rows))
    elif args.sort == "config_eqbs":
        ordered = sorted(enumerate(rows), key=lambda ir: (ir[1]["run_name"], equiv_batch(ir[1], dp_size_map)))
    else:
        ordered = sorted(enumerate(rows), key=lambda ir: (equiv_batch(ir[1], dp_size_map), ir[1]["run_name"]))

    seen: dict[tuple, int] = {}
    pass_idx_for_row: dict[int, int] = {}
    for raw_idx, r in enumerate(rows):
        k = (r["run_name"], r["batch_size"])
        seen[k] = seen.get(k, 0) + 1
        pass_idx_for_row[raw_idx] = seen[k]

    if args.csv:
        cols = [
            "run_name", "equiv_batch", "batch_size",
            "prefill_latency", "prefill_throughput", "prefill_system_tps",
            "median_decode_latency", "median_decode_throughput", "decode_system_tps",
            "total_latency", "overall_throughput", "overall_system_tps",
            "pass_idx",
        ]
        print(",".join(cols))
        for orig_idx, r in ordered:
            cells = [
                r["run_name"], equiv_batch(r, dp_size_map), r["batch_size"],
                f"{r['prefill_latency']:.6f}",
                f"{r['prefill_throughput']:.4f}",
                f"{system_tps(r, 'prefill_throughput', dp_size_map):.4f}",
                f"{r['median_decode_latency']:.6f}",
                f"{r['median_decode_throughput']:.4f}",
                f"{system_tps(r, 'median_decode_throughput', dp_size_map):.4f}",
                f"{r['total_latency']:.6f}",
                f"{r['overall_throughput']:.4f}",
                f"{system_tps(r, 'overall_throughput', dp_size_map):.4f}",
                pass_idx_for_row[orig_idx],
            ]
            print(",".join(str(c) for c in cells))
        return

    print(HEADER)
    print("-" * len(HEADER))
    last_run = None
    for orig_idx, r in ordered:
        if args.sort == "config_eqbs" and last_run is not None and r["run_name"] != last_run:
            print()
        print(fmt_row(r, pass_idx_for_row[orig_idx], dp_size_map))
        last_run = r["run_name"]

    print()
    print("Legend:")
    print("  eq_bs   = equivalent global batch (= bs for TP attention; bs * dp_size for DP attention)")
    print("  pre_*   = prefill metrics            dec_* = decode (median per-iter)   ovr_* = overall (prefill+decode)")
    print("  *_tps   = bench_one_batch reported tok/s (per-DP-rank for DP configs)")
    print("  *_sys   = system throughput (per-rank * dp_size for DP configs)")
    print("  pass    = 1st or 2nd occurrence of (run_name, batch_size) in input")


if __name__ == "__main__":
    main()
