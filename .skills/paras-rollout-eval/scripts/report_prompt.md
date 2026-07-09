# REPORT.md aggregation subagent prompt template

Fire as `category=writing`, `run_in_background=true` after the last row's forensics has landed. Reads matrix.csv + every forensics.md, produces top-level REPORT.md.

Replace `<ARTIFACTS_DIR>`, `<MODEL>`, `<DATASET>`, `<SWEEP_NAME>`, `<TOTAL_CELLS>`, `<BRANCH_HEAD>` before sending.

```
[TASK] Aggregate the <SWEEP_NAME> results into a comprehensive REPORT.md.

[CONTEXT]
Sweep configuration:
- Model: <MODEL>
- Dataset: <DATASET>
- Spec mode (ignore_eos=True, max_completion_tokens=min(row.output_len, cap))
- Branch: <BRANCH_HEAD>
- Parity contract: every cell with DISABLE_RADIX_CACHE=1 and HYBRID_SWA=1 (axes under test vary at most one knob)

Matrix: <TOTAL_CELLS> cells across {systems} x {N} x {cap}

[INPUT FILES — READ IN FULL]

1. Matrix CSV: <ARTIFACTS_DIR>/matrix.csv (may have duplicate (system, N, cap) keys if retried; keep LATEST per key)
2. Per-row forensics: <ARTIFACTS_DIR>/<model>__<system>/n<N>_cap<CAP>/forensics.md (one per successful row)
3. Per-server launch commands: <ARTIFACTS_DIR>/<model>__<system>/server_launch.cmd.txt (cite if discrepancies between intended and observed config)
4. Live progress log: <ARTIFACTS_DIR>/PROGRESS.md
5. Any investigations under <ARTIFACTS_DIR>/INVESTIGATION_*.md

[EXPECTED OUTPUT] Write <ARTIFACTS_DIR>/REPORT.md (≤300 lines) with:

## 1. Executive Summary
- Headline finding in 1 paragraph (which system wins at which (N, cap))
- 4-D headline table: rows = systems, cols = (N, cap), cells = e2e_s / out_tps or "❌ <failure reason>"

## 2. Results matrix
- All successful benches: completed/failed, e2e_time, output_throughput, p50/p99 latency
- All failed benches: failure mode, root-cause reference (link to INVESTIGATION_*.md or PROGRESS.md log entry)

## 3. Comparative analysis
- Per (N, cap): best static vs paras (speedup ratio)
- Cap effect: 16k vs 32k ratio per system
- N-scaling: N=1024 vs N=2048 per system
- Mode time-share for paras rows (% EP vs % TP from metrics_timeseries.csv)

## 4. Failure summary
- One row per failure with: cell, exit_code or failure mode, root cause (cite docs/code), suggested fix

## 5. Key findings (3-7 bullets)
- Each bullet grounded in a specific number from the matrix or a forensics.md

## 6. Recommendations
- Code fixes (cite file:line)
- Config changes for next sweep
- Open questions

## 7. Methodology + caveats
- Parity contract enforced (DISABLE_RADIX_CACHE=1, HYBRID_SWA=1)
- Sampler patch state (commit 2b485f66d for global metrics in static modes)
- /health 503 race + server_alive_with_retries pattern
- Any unexpected events from PROGRESS.md (concurrent edits, manual interventions)

[REQUIRED TOOLS] Read, Write (REPORT.md), Bash (grep/awk)
[MUST DO] Use tables. Cite numbers from CSV/forensics. Reference investigations by filename. Distinguish "EP wins" from "paras wins" — they are different (paras may run EP some of the time but pays mode-switch + paras-specific overhead).
[MUST NOT DO] Speculate beyond what data supports. Modify benchmark artifacts. Re-investigate root causes (cite existing INVESTIGATION_*.md instead).
[DELIVERABLE] <ARTIFACTS_DIR>/REPORT.md (≤300 lines, well-organized, tables-heavy).
```

## Acceptance for REPORT.md

- ≤300 lines
- Section structure matches above
- Headline table is 1-glance readable (one row per system, one col per (N, cap))
- Every metric has a source citation
- Failure rows reference root-cause docs (not re-derived)
- Recommendations are actionable (file:line citations or concrete config changes)
