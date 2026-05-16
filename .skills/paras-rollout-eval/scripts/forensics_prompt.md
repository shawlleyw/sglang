# Forensics subagent prompt template

Fire as `category=quick`, `run_in_background=true` per completed row. Replace `<ROW_DIR>`, `<SYSTEM_LABEL>`, `<MODEL_NAME>`, `<N>`, `<CAP>`, `<BENCH_WINDOW>`, `<BASELINE_NUMBERS>` before sending.

```
[TASK] Write a forensics report for one completed benchmark row.

[CONTEXT]
Row directory: <ROW_DIR>
Model: <MODEL_NAME>
System: <SYSTEM_LABEL> (e.g. tp-static__swa-on__overlap-off)
N: <N>
Cap: <CAP>
Server log: <ROW_DIR>/../server.log  (continuous; bench window <BENCH_WINDOW>)
Sample: <ABS PATH TO snapshot jsonl>

[EXPECTED OUTCOME] Write <ROW_DIR>/forensics.md with these sections (use tables for numbers):

1. Headline: completed/failed counts, e2e_time, output_throughput, p50/p99 latency (from summary.json).
2. Output-length distribution from per_request.jsonl: min, p50, p90, p99, max of output_len_tokens. CRITICAL: In spec-mode with ignore_eos=True, ALL reqs have finish_reason="length" - that does NOT mean cap was hit. Count rows where output_len_tokens == <CAP> (actual cap hits) separately from total. Also report finish_reason counts as-is.
3. Server-mode timeseries (from <ROW_DIR>/metrics_timeseries.csv): unique modes observed, peak decode_tps, peak prefill_tps, peak running_reqs, time spent per mode.
4. Server log signals (within bench window): count of hard errors (IndexError, AssertionError, OutOfMemoryError, "state was deleted", "device-side assert"), policy fires (paras only) with directions + observation values + timestamps.
5. Anomalies worth flagging. Reference baseline: <BASELINE_NUMBERS> (e.g. "tp-static N=2048 cap=16k got 8545 tok/s; this row got X — comment").

[REQUIRED TOOLS] Bash (grep/awk/wc/python3), Read, Write
[MUST DO] Numbers, tables, under 60 lines. Be precise about spec-mode finish_reason semantics.
[MUST NOT DO] Modify files outside the row dir except creating forensics.md. Do not launch processes. Do not consult external repos or docs.
```

## Acceptance for the forensics report

A good forensics.md is:
- ≤60 lines
- Has 5 sections (above)
- Every number cited has a source (summary.json, per_request.jsonl, metrics_timeseries.csv, or server.log)
- Anomalies are flagged with magnitudes (% deviation from baseline) not just adjectives
- finish_reason semantics correctly reported (the "spec-mode → all length" clarification)

## When to fire

Once per row, after `summary.json` lands. Driver can fire forensics in parallel for up to 4-6 rows at once. Don't block the driver loop on forensics completion.
