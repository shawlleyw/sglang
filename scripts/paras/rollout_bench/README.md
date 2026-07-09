# ParaS Rollout Benchmark

End-to-end batch-inference benchmark for the ParaS `rollout` auto-switch
policy. Drives one or many runs of `python -m sglang.bench_paras` against an
sglang server launched in one of three server modes (`ep-static`, `tp-static`,
`paras`), so that the rollout policy's effect on a GRPO-style burst workload
can be measured.

## Layout

- `run_one.sh` — drive one `(model, server-mode, dataset)` run.
- `run_matrix.sh` — sweep 2 models × 3 datasets × 3 server modes (18 runs).
- `README.md` — this file.

## Prerequisites

- Conda env `sgl_paras` (or equivalent) activated; `python -m sglang.bench_paras`
  must be importable.
- Datasets prepared under `~/paras-workload/{dapo,acecode,eurus2}/`. Each
  dataset directory ships:
  - `sampled_8k.jsonl` — raw prompts (no `output_len` field).
  - `spec_8k_qwen3-30b.jsonl` — same prompts, with `output_len` from a
    reference qwen3-30b run.
  - `spec_8k_gpt-oss-120b.jsonl` — same prompts, with `output_len` from a
    reference gpt-oss-120b run.

  See `~/paras-workload/<dataset>/README.md` for how each set was generated.
- 8 × A100-80GB (or similar capacity). The default `NUM_GPUS=8` and
  `MEM_FRACTION_STATIC=0.85` target this footprint; smaller GPUs need
  overrides.

## Single run

```bash
MODEL_SLUG=qwen3-30b \
MODE=paras \
DATASET=~/paras-workload/dapo/spec_8k_qwen3-30b.jsonl \
OUTPUT_DIR=/tmp/rollout_single \
NUM_REQUESTS=8000 \
GROUP_SIZE=1 \
SPEC_MODE=1 \
bash scripts/paras/rollout_bench/run_one.sh
```

`MODE` ∈ {`ep-static`, `tp-static`, `paras`}:

- `ep-static`: launches `launch_server_dp_ep.sh` without `ENABLE_PARAS`,
  i.e. plain DP attention + EP experts, no switching.
- `tp-static`: launches `launch_server_tp_tp.sh`, i.e. TP attention + TP
  experts, no switching.
- `paras`: launches `launch_server_dp_ep.sh` with `ENABLE_PARAS=1` plus
  `--paras-auto-switch-policy rollout --paras-metrics-file
  $OUTPUT_DIR/metrics_timeseries.csv`.

Outputs land in `$OUTPUT_DIR`:

| File | Source | Description |
|---|---|---|
| `summary.json` | `bench_paras` | Aggregate run metrics (completed, failed, e2e_time, throughput, p50/p99 latency). |
| `per_request.jsonl` | `bench_paras` | One record per request (request_id, latency, prompt/output tokens, finish_reason). |
| `run_config.json` | `bench_paras` | CLI flags + UTC timestamp + git SHA. |
| `server.log` | launcher | Server stdout/stderr. |
| `nvidia_smi.txt` | launcher | Post-run `nvidia-smi` snapshot. |
| `metrics_timeseries.csv` | server (paras mode) | Per-second `ParasMetricsSampler` rows (running / waiting / decode / prefill). |
| `server.tail.log` | launcher (on failure) | Last 200 lines of `server.log`, written only if `bench_paras` exits non-zero. |

## Full matrix sweep

```bash
bash scripts/paras/rollout_bench/run_matrix.sh
```

Runs the cross product `{qwen3-30b, gpt-oss-120b} × {dapo, acecode, eurus2} ×
{ep-static, tp-static, paras}` = 18 runs and writes one row per run to
`$MATRIX_ROOT/matrix.csv` with columns:

```
model, dataset, server_mode, spec_mode, num_requests, group_size,
exit_code, completed, failed, e2e_time, input_throughput, output_throughput,
retried_oom, output_dir
```

Default `MATRIX_ROOT=~/paras-bench-results/<UTC>_matrix`. Each
`(model, dataset, mode)` run lives in
`$MATRIX_ROOT/<model>__<dataset>__<mode>/` and contains the same files as a
single run above.

By default `SPEC_MODE=1` so each run uses
`~/paras-workload/<dataset>/spec_8k_<model>.jsonl` with deterministic
`output_len` (mirrors the bench's `--spec-mode`). Set `SPEC_MODE=0` to switch
to `sampled_8k.jsonl` with eos-stopping.

## Dry run

Both scripts honor `DRY_RUN=1` to print the commands without executing:

```bash
DRY_RUN=1 \
MODEL_SLUG=qwen3-30b MODE=paras \
DATASET=/tmp/fake.jsonl OUTPUT_DIR=/tmp/dryrun \
bash scripts/paras/rollout_bench/run_one.sh
```

Use this to verify env-var wiring (especially the ParaS-mode flag bundle)
before committing GPU time.

## OOM fallback

`run_matrix.sh` retries once with `MEM_FRACTION_STATIC=0.8` when the prior
attempt failed and `server.log` contains an OOM signature (matched as
`out of memory`, `OutOfMemoryError`, or `OOM`). The retry's outcome
overwrites the row's `exit_code` and sets `retried_oom=1`.

## See also

- `python/sglang/bench_paras.py` — the client driver this kit invokes.
- `python/sglang/srt/managers/paras_metrics_sampler.py` — the server-side
  per-second metrics writer (paras mode only).
- `python/sglang/srt/paras/auto_switch_policy.py` —
  `RolloutAutoSwitchPolicy` and the canonical EP↔TP decision logic.
- `scripts/paras/eval/a100/{qwen,gptoss}/launch_server_*.sh` — the
  per-model launchers this kit composes.
