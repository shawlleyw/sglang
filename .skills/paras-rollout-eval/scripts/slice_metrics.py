#!/usr/bin/env python3
"""Slice a paras metrics_timeseries.csv into a per-bench row dir.

Server writes ONE continuous CSV across all benches in a server-reuse driver.
This tool extracts the [lines_before+1 .. lines_after] window with header
prepended, producing a per-bench CSV that downstream forensics can consume
directly.

Usage:
    python3 slice_metrics.py \\
        --src <server_dir>/metrics_timeseries.csv \\
        --dst <row_dir>/metrics_timeseries.csv \\
        --lines-before <N1> --lines-after <N2>

If lines_after <= lines_before (no new rows) emits just the header so the
row file is still parseable. If --src is missing, emits an empty file with
the canonical header.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HEADER = "timestamp_iso,elapsed_s,mode,running_reqs,waiting_reqs,decode_tokens_per_sec,prefill_tokens_per_sec\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dst", type=Path, required=True)
    p.add_argument("--lines-before", type=int, required=True, help="wc -l of src BEFORE this bench")
    p.add_argument("--lines-after", type=int, required=True, help="wc -l of src AFTER bench + drain")
    args = p.parse_args()

    args.dst.parent.mkdir(parents=True, exist_ok=True)

    if not args.src.exists():
        args.dst.write_text(HEADER)
        print(f"WARN: src {args.src} missing; wrote header-only dst", file=sys.stderr)
        return 0

    src_lines = args.src.read_text().splitlines()
    if not src_lines:
        args.dst.write_text(HEADER)
        return 0

    header = src_lines[0]
    if args.lines_after <= args.lines_before:
        args.dst.write_text(header + "\n")
        print(f"WARN: lines_after ({args.lines_after}) <= lines_before ({args.lines_before}); "
              f"wrote header-only dst", file=sys.stderr)
        return 0

    # Lines are 1-indexed in the wc -l convention; src_lines is 0-indexed.
    # Window: src_lines[lines_before .. lines_after-1] inclusive.
    slice_lines = src_lines[args.lines_before : args.lines_after]
    out = [header] + slice_lines
    args.dst.write_text("\n".join(out) + "\n")
    n_rows = len(slice_lines)
    print(f"wrote {args.dst}: {n_rows} data rows + header (slice [{args.lines_before+1}, {args.lines_after}])")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
