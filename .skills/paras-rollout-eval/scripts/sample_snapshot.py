#!/usr/bin/env python3
"""Deterministic snapshot creator for paras rollout eval.

Reads ~/paras-workload/<dataset>/spec_8k_<model>.jsonl, filters output_len > 0,
shuffles with --seed, takes first --num-requests rows, writes to
<output_dir>/<model>__<dataset>__n<N>_seed<seed>.jsonl with idx + sample_idx
preserved so every snapshot is fully self-describing.

Usage:
    python3 sample_snapshot.py \\
        --src ~/paras-workload/dapo/spec_8k_gpt-oss-120b.jsonl \\
        --out artifacts/<sweep>/samples/ \\
        --model gpt-oss-120b --dataset dapo \\
        --num-requests 1024 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True, help="Path to spec_8k_<model>.jsonl")
    p.add_argument("--out", type=Path, required=True, help="Output dir (will be created)")
    p.add_argument("--model", required=True, help="Model slug (e.g. gpt-oss-120b)")
    p.add_argument("--dataset", required=True, help="Dataset slug (e.g. dapo)")
    p.add_argument("--num-requests", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--min-output-len", type=int, default=1, help="Filter rows with output_len < this")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"{args.model}__{args.dataset}__n{args.num_requests}_seed{args.seed}.jsonl"

    src = [json.loads(l) for l in args.src.read_text().splitlines() if l.strip()]
    print(f"Source: {args.src.name} ({len(src)} rows)")
    src = [r for r in src if r.get("output_len", 0) >= args.min_output_len]
    print(f"After output_len >= {args.min_output_len} filter: {len(src)} rows")
    assert len(src) >= args.num_requests, f"Filtered pool too small ({len(src)}) < N ({args.num_requests})"

    for i, r in enumerate(src):
        assert "prompt_text" in r, f"row {i} missing prompt_text"
        assert "output_len" in r, f"row {i} missing output_len"
        assert "idx" in r, f"row {i} missing idx (orig spec_8k position)"

    rng = random.Random(args.seed)
    indices = list(range(len(src)))
    rng.shuffle(indices)
    chosen = indices[: args.num_requests]

    out_rows = []
    for new_idx, src_pos in enumerate(chosen):
        row = dict(src[src_pos])
        row["sample_idx"] = new_idx
        out_rows.append(row)

    with out_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    lens = [r["output_len"] for r in out_rows]
    sl = sorted(lens)
    print(f"\nWrote {out_path}:")
    print(f"  n={len(out_rows)}, size={out_path.stat().st_size/1e6:.2f} MB")
    print(f"  output_len: min={min(lens)} p50={sl[len(sl)//2]} p90={sl[int(len(sl)*0.9)]} "
          f"p99={sl[int(len(sl)*0.99)]} max={max(lens)} mean={statistics.mean(lens):.0f}")
    for cap in (8192, 16384, 32768):
        clipped = sum(1 for x in lens if x > cap)
        eff_total = sum(min(x, cap) for x in lens)
        print(f"  cap={cap}: clipped={clipped} ({100*clipped/len(lens):.1f}%), total_decode={eff_total:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
