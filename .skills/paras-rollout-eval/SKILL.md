---
name: paras-rollout-eval
description: Drive end-to-end batch-inference evaluation matrices for the ParaS EP↔TP runtime switch. Covers deterministic prompt sampling, server-per-cell vs server-reuse driver patterns, the canonical artifacts layout (variant-tagged dirs, per-bench files including the new outputs.jsonl, matrix.csv, PROGRESS.md, REPORT.md), `/health` 503 race avoidance, per-bench metrics slicing, failure recording, and forensics+REPORT subagent contracts. Default parity contract: every cell launches with `DISABLE_RADIX_CACHE=1` and `HYBRID_SWA=1` so static baselines and paras differ only on the axes under test.
metadata:
  short-description: Standardized matrix sweep for paras vs static baseline rollout benchmarks
---

# ParaS Rollout Evaluation Matrix

End-to-end batch-inference benchmark for the ParaS EP↔TP runtime switch. Run when you want to sweep `{paras, ep-static, tp-static}` × workload (N, cap) × variants (swa, overlap, threshold, ...) and produce a comparable artifacts tree per cell. For a single smoke test (auto-switch, in-flight switch correctness, weight transfer probe) use the sister skills `paras-test-auto-switch`, `paras-test-manual-switch`, `paras-test-peer-access`. For 4-way attention/expert parallelism microbenches use `bench-ep-tp`.

## When to invoke

- "Run paras rollout matrix" / "paras eval" / "benchmark paras vs baselines"
- "Sweep paras at thresholds X, Y" / "compare paras-Tn vs ep-static at N=..."
- "Re-run the 16-bench matrix" / "rollout eval round"
- "Smoke the SWA/overlap variants" / "single-cell paras smoke"

If the request is "test the switch correctness" or "validate peer-access transfers" — use the sister skills instead.

## Canonical runtime options (parity contract — read first)

Every sweep runs every system at these values. Deviations require an explicit reason in the cell label (e.g. `__swa-off` variant tag).

### Shared across ALL systems

| Knob | Value | Layer | Default-in-launcher? |
|---|---|---|---|
| `DISABLE_RADIX_CACHE` | `1` | env | both launchers default `1` ✓ — paras requires; static parity |
| `HYBRID_SWA` | `1` | env | `tp_tp` default `0` (driver must override); `dp_ep` default `auto` → resolves to `1` for paras, `0` for ep-static (driver must override for ep-static) |
| `DISABLE_OVERLAP` | `0` (overlap ON) | env | both launchers default `0` ✓ — paras now supports overlap; no parity penalty on baselines |
| `SYNC_TOKEN_IDS_ACROSS_TP` | `1` (tp-static only) | env | **baked into `launch_server_tp_tp.sh` for both gpt-oss and qwen** (line ~40 of each) — drivers do NOT need to set it. Forces a MIN all-reduce on sampled token-ids so tp-static inherits the same kernel-determinism safety net paras forces in TP mode. |
| `--chunked-prefill-size` | `-1` | CLI | both launchers hardcode ✓ — paras requires; static parity |
| `--max-prefill-tokens` | `8192` | CLI | driver passes |
| `--attention-backend` | `triton` | CLI | gpt-oss launchers hardcode ✓ — required for gpt-oss-120b |
| `--moe-runner-backend` | `triton` | CLI | gpt-oss launchers hardcode ✓ — required for gpt-oss-120b |
| `ENABLE_CUDA_GRAPH` | `1` | env | both launchers default `1` ✓ |
| `CUDA_GRAPH_MAX_BS` | `256` | env | both launchers unset by default (no cap arg) → **driver must set** |
| `MAX_RUNNING_REQUESTS` | `2048` | env | both launchers default `256` → **driver must override to 2048** |
| `NUM_GPUS` | `8` | env | both launchers default `8` ✓ |

### Per-system overrides

| Knob | tp-static | ep-static | paras-t\<N\> |
|---|---|---|---|
| Launch script (gpt-oss-120b) | `a100/gptoss/launch_server_tp_tp.sh` | `a100/gptoss/launch_server_dp_ep.sh` | `a100/gptoss/launch_server_dp_ep.sh` |
| `ENABLE_PARAS` | `0` | `0` | `1` |
| `MEM_FRACTION_STATIC` | **0.80** | **0.75** | **0.75** |
| `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` | — (no DeepEP) | `256` | `256` |
| `--paras-tp-cuda-graph-max-bs` | — | — | `128` |
| `--paras-auto-switch-policy` | — | — | `rollout` |
| `--paras-auto-switch-threshold` | — | — | `N` (encoded in tag, e.g. `paras-t64`) |
| `--paras-auto-switch-window` | — | — | (default; code: `1` for rollout) |
| `--paras-auto-switch-cooldown-sec` | — | — | (default; code: `5` for rollout) |

### Notes

- Earlier paras forced `--disable-overlap-schedule` at the server level. As of commit `adacebf22 fix(paras): port auto-switch observe/signal hook to event_loop_overlap`, paras supports the overlap scheduler. Parity default is therefore overlap-ON for all systems.
- `SYNC_TOKEN_IDS_ACROSS_TP=1` is hardcoded into `launch_server_tp_tp.sh` for both gpt-oss and qwen. Anyone launching tp-static via the canonical script gets parity for free.
- For qwen launchers (`a100/qwen/`, `h200/qwen/`) the same env-var contract applies; only `--attention-backend`/`--moe-runner-backend` differ (qwen does not pin triton).
- gpt-oss `tp_tp` launcher additionally pins `--attention-backend triton --moe-runner-backend triton`; the qwen `tp_tp` launcher does not (qwen defaults work). Skill drivers should pass these flags explicitly only when running gpt-oss against a non-canonical launcher.

## Artifacts layout (canonical)

```
artifacts/<UTC>_<sweep_name>/
├── samples/
│   ├── <model>__<dataset>__n<N>_seed<seed>.jsonl     # deterministic snapshots
│   └── ...
├── matrix.csv                                         # one row per cell
├── PROGRESS.md                                        # append-only event log
├── REPORT.md                                          # final aggregation
├── INVESTIGATION_*.md                                 # per-question deep dives
├── FIX_*.md / FEATURE_*.md                            # code-change writeups
└── <model>__<system>[__<variant>]+/                   # one dir per server-launch
    ├── server_launch.cmd.txt                          # exact bash invocation (env + flags)
    ├── server.log                                     # includes sglang's parsed `server_args=...` dump
    ├── nvidia_smi.txt
    ├── metrics_timeseries.csv                         # full server lifetime
    └── n<N>_cap<CAP>/                                 # one dir per bench (row)
        ├── summary.json                               # bench client aggregate
        ├── per_request.jsonl                          # per-request metrics
        ├── outputs.jsonl                              # NEW: per-request response_text
        ├── run_config.json                            # bench CLI flags + git sha
        ├── metrics_timeseries.csv                     # sliced for this bench window
        ├── bench.log                                  # bench client stdout
        └── forensics.md                               # (optional) per-row subagent output
```

## Variant naming convention

Server dir = `<model>__<system>[__<variantN>]+`. Examples from the current corpus:

| Server dir | Decoded |
|---|---|
| `gpt-oss-120b__tp-static` | gpt-oss-120b on TP-static at parity defaults |
| `gpt-oss-120b__tp-static__swa-on__overlap-off` | TP-static, hybrid SWA on, overlap off (paras-matching) |
| `gpt-oss-120b__tp-static__swa-off__overlap-on` | TP-static, hybrid SWA off, overlap on (legacy default) |
| `gpt-oss-120b__paras-t64` | paras with auto-switch threshold=64 (canonical) |
| `gpt-oss-120b__paras-t256__swa-on__overlap-on` | paras with threshold=256, both knobs on (smoke-only forcing) |
| `gpt-oss-120b__paras-t64__tail-cap8k__swa-on__overlap-on` | paras t64 with workload-tagged variant (cap=8k tail) |

Rules:
- `<system>` = `tp-static` | `ep-static` | `paras-t<threshold>` (paras encodes the threshold in the tag)
- `<variant>` segments are k-v fragments separated by `__`, e.g. `swa-on`, `overlap-off`, `tail-cap8k`
- Order variants alphabetically when stable; functional (workload-specific) variants like `tail-cap8k` go first when tracing pre/post conditions
- Row dir under a server = `n<N>_cap<CAP>`, always

## Per-bench file contract

Every row dir contains:

| File | Schema | Source |
|---|---|---|
| `summary.json` | `{completed, failed, e2e_time, input_throughput, output_throughput, total_input_tokens, total_output_tokens, mean_e2e_latency_s, p50_e2e_latency_s, p99_e2e_latency_s, ...}` | `bench_paras.build_summary` |
| `per_request.jsonl` | `{request_id, unique_id, replica_id, arrival_t, completion_t, e2e_latency, prompt_len_tokens, output_len_tokens, finish_reason, completed, error?, http_status?}` per line | `bench_paras.RequestRecord` |
| **`outputs.jsonl`** | `{request_id, unique_id, replica_id, response_text}` per line | `bench_paras --dump-outputs` (default on) — for downstream eval (BLEU, exact-match, manual inspection) |
| `run_config.json` | `vars(args) + {utc_timestamp, git_sha}` | `bench_paras.write_outputs` |
| `metrics_timeseries.csv` | columns `timestamp_iso, elapsed_s, mode, running_reqs, waiting_reqs, decode_tokens_per_sec, prefill_tokens_per_sec` (per-second) | sliced from server's continuous CSV |
| `bench.log` | bench client stdout/stderr | tee'd by driver |
| `forensics.md` | optional per-row report (headline / output-dist / mode timeseries / log signals / anomalies) | `category=quick` subagent |

Spec-mode `finish_reason` clarification: with `--spec-mode` and `--ignore-eos`, **every** request finishes with `finish_reason="length"` because the only stop condition is `max_completion_tokens = min(row.output_len, cap)`. This does NOT mean cap was hit. Count `output_len_tokens == cap` separately to identify actual cap-clipped rows.

## Two driver patterns

Pick by what changes between cells:

### Pattern A: server-per-cell (`run_smoke.sh`)

Use when cells differ on a **launch-time** flag — SWA toggle, overlap toggle, ParaS auto-switch threshold, paras-on/off. Each cell launches its own server, runs one bench, kills.

Cost: ~5 min boot per cell (gpt-oss-120b BF16). Use for ≤6 cells per sweep.

Template: [`scripts/run_smoke.sh`](scripts/run_smoke.sh)

### Pattern B: server-reuse (`run_sweep.sh`)

Use when cells differ only on **client** args (`--num-requests`, `--max-completion-tokens-cap`, `--seed`, `--dataset-jsonl`). One server amortizes 4-12 benches.

Cost: ~5 min boot ONCE per server. Best for the canonical 4×4 = 16-bench matrix.

Template: [`scripts/run_sweep.sh`](scripts/run_sweep.sh)

Hybrid: outer loop = system (server-per-cell), inner loop = (N, cap) (server-reuse within each system). This is the 20260515 baseline pattern.

## Driver invariants (both patterns)

1. **Kill any leftover sglang** before each launch via `bash scripts/paras/eval/paras_cmd/kill.sh` + 5 s sleep. The `pkill` pattern in that script is the canonical safe pattern.
2. **Use `wait_ready.sh` for the boot probe** (`grep "Application startup complete"` in server.log).
3. **Always use `server_alive_with_retries` AFTER wait_ready**. sglang's `/health` returns 503 during the post-startup `/generate` warmup (~1-2 s window between "Application startup complete" and "fired up and ready"). A single `/health` check can hit this race and falsely report dead. The retry-loop (15 × 2 s) bridges it. See [`scripts/run_sweep.sh`](scripts/run_sweep.sh) for the canonical implementation.
4. **`ulimit -n 1048576`** in the tmux pane. Default 1024 nofile causes ZMQ socket exhaustion mid-sweep.
5. **Per-bench metrics slicing**: server writes ONE continuous `metrics_timeseries.csv`; before each bench record `lines_before = wc -l <csv>`, after bench record `lines_after`, then write `<row_dir>/metrics_timeseries.csv` = header (line 1) + lines `[lines_before+1 .. lines_after]`. For Pattern A (server-per-cell) just `cp` the whole CSV since it belongs to one bench.
6. **10 s sleep between back-to-back benches** on the same server (Pattern B) for tail-drain + final metrics flush.
7. **Failure handling** — RECORD, DO NOT HALT:
   - bench exits non-zero → matrix.csv row gets `exit_code=N, notes="bench_exit=N"`, continue to next bench
   - server dies mid-bench → mark remaining benches in that system as `skipped_server_died`, kill, move to next system
   - server fails to launch → mark all benches in that system as `skipped_server_down`, move on

## Bench client invocation

```bash
python -m sglang.bench_paras \
  --model "$MODEL_NAME" \
  --dataset-jsonl "$SAMPLE_PATH" \
  --mode-label "$CELL_LABEL" \
  --num-requests "$N" \
  --group-size 1 \
  --spec-mode \
  --max-completion-tokens-cap "$CAP" \
  --output-dir "$ROW_DIR" \
  --host 127.0.0.1 --port 30000 \
  --seed 42 \
  --dump-outputs                # NEW: writes outputs.jsonl alongside per_request.jsonl
```

`--spec-mode` is required for reproducibility. It uses `row.output_len` from the snapshot and sets `ignore_eos=True`, so per-request decode counts are deterministic across runs at the same seed.

`--dump-outputs` is on by default after the bench_paras patch lands. Set `--no-dump-outputs` only if disk pressure is critical (outputs.jsonl can be 20-200 MB per bench at N=2048).

## Per-benchmark metrics slicing (critical for server-reuse pattern)

The server writes ONE continuous `metrics_timeseries.csv` for its whole lifetime (rank-0 only, 1 Hz). When a single server runs multiple benchmarks (Pattern B), every bench's metrics get appended to the same file. We must extract per-bench slices into each row dir so downstream forensics + REPORT have one CSV per bench.

### Method: line-offset slicing

Before launching a bench, snapshot the current line count. After bench finishes + tail drain, snapshot again. Write `<row_dir>/metrics_timeseries.csv` = header (line 1) + lines `[lines_before+1 .. lines_after]`. Header repeats per slice so each row CSV is independently parseable.

Reference implementation: [`scripts/slice_metrics.py`](scripts/slice_metrics.py) (standalone helper).

Driver-side usage (excerpt from [`scripts/run_sweep.sh`](scripts/run_sweep.sh)):

```bash
# Before bench
lines_before=$(wc -l < "$SERVER_METRICS" 2>/dev/null || echo 0)
log_progress "BENCH start: $cell_label lines_before=$lines_before"

# Run bench client
python -m sglang.bench_paras ... > "$ROW_DIR/bench.log" 2>&1 || bench_exit=$?

sleep 10  # tail drain — 1 Hz sampler needs to flush final per-step metrics

# After bench
lines_after=$(wc -l < "$SERVER_METRICS" 2>/dev/null || echo 0)
python3 "$SKILL_DIR/scripts/slice_metrics.py" \
    --src "$SERVER_METRICS" \
    --dst "$ROW_DIR/metrics_timeseries.csv" \
    --lines-before "$lines_before" \
    --lines-after "$lines_after"
```

### Server-per-cell (Pattern A) shortcut

When each cell has its own server, the whole `metrics_timeseries.csv` belongs to that single bench. Just `cp` the file:

```bash
cp "$SERVER_METRICS" "$ROW_DIR/metrics_timeseries.csv"
```

No offset bookkeeping needed.

### Watch-outs

- **The 10 s sleep is essential** — the sampler runs at 1 Hz and writes line-buffered. Without the sleep, the last 1-3 seconds of metrics (often the tail-drain throughput collapse) get attributed to the NEXT bench's slice.
- **Concurrent writes**: only rank-0 writes the CSV; no contention. But if you ever scale to multi-node or rebind the sampler, revisit.
- **Empty slice** (`lines_after <= lines_before`): emit just the header so the row file is still parseable. `slice_metrics.py` handles this.
- **Server died mid-bench**: the slice still works — it captures whatever the sampler wrote before exit. The row dir's `metrics_timeseries.csv` will be shorter than the bench's expected duration; use this as a forensics signal.
- **No `gpt-oss-120b__*/metrics_timeseries.csv` for non-paras runs without `--paras-metrics-file`**: pass the flag for ALL systems (after the sampler patch from 20260515; the metrics sampler now correctly reports mode + global counters even for tp-static and ep-static, see commit `2b485f66d`).

## Server launch command tracking

Every server-launch dir contains `server_launch.cmd.txt` with the exact bash invocation (env vars + flags) that started the server. Reconstructable from the file, paste-and-runnable for forensic reproduction. Two views available:

| File | What it shows |
|---|---|
| `server_launch.cmd.txt` | Driver's command BEFORE the server parsed it — useful to verify the driver set what you intended |
| `server.log` | sglang's `server_args=ServerArgs(...)` dump on line ~1-5 — useful as ground truth of what the server actually saw post-parse |

Both should match (modulo defaults). Discrepancies indicate either a typo in the driver or an env var that didn't propagate. Drivers in this skill (`run_sweep.sh`, `run_smoke.sh`) emit `server_launch.cmd.txt` automatically.

## Memory + cuda-graph caps per (hardware, model, server class)

Subset of the "Canonical runtime options" table above, for quick reference when sizing a sweep. Override via env at the driver:

### 8 × A100-80GB · gpt-oss-120b

| Server | MEM_FRACTION_STATIC | MAX_RUNNING_REQUESTS | CUDA_GRAPH_MAX_BS | DeepEP dispatch cap |
|---|---:|---:|---:|---:|
| tp-static | 0.80 | 2048 | 256 | n/a (no DeepEP) |
| ep-static | 0.75 | 2048 | 256 | 256 |
| paras-t* | 0.75 | 2048 | 256 (+ `--paras-tp-cuda-graph-max-bs=128`) | 256 |

gpt-oss-120b OOMs at mfs=0.80 in ep-static / paras configs (DeepEP buffers + dual cuda-graph capture).

### 8 × A100-80GB · Qwen3-30B-A3B

| Server | MEM_FRACTION_STATIC | MAX_RUNNING_REQUESTS | CUDA_GRAPH_MAX_BS | DeepEP dispatch cap |
|---|---:|---:|---:|---:|
| tp-static | 0.85 | 2048 | 256 | n/a |
| ep-static | 0.85 | 2048 | 256 | 512 (DeepEP default for qwen on A100) |
| paras-t* | 0.60 (launcher default; override via driver) | 2048 | 8 (launcher default; override via driver) | 256 |

### 8 × H200 · Qwen3-235B-A22B-Instruct-2507

| Server | MEM_FRACTION_STATIC | MAX_RUNNING_REQUESTS | CUDA_GRAPH_MAX_BS | DeepEP dispatch cap |
|---|---:|---:|---:|---:|
| tp-static | 0.85 | 2048 | 256 | n/a |
| ep-static | 0.85 | 2048 | 256 | 512 |
| paras-t* | 0.70 (launcher default; override via driver) | 2048 | 8 (launcher default; override via driver) | 512 |

### Notes

- `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256` always when paras is enabled (NVSHMEM_QP_DEPTH constraint). Static EP can use 512.
- Launcher paras defaults for `CUDA_GRAPH_MAX_BS=8` are intentionally small — keeps capture cost low at boot when the canonical paras workload is small. Production sweeps override to 256 via driver env. **All paras launchers keep `ENABLE_CUDA_GRAPH=1`**; cuda graphs are always on under paras.
- Launcher paras defaults for `MEM_FRACTION_STATIC` (0.60 a100/qwen, 0.70 h200/qwen, 0.80 a100/gptoss) leave headroom for the dual cuda-graph capture and the UMM contiguous slab. Drivers can raise if validated for a specific workload.

## Pre-sampling (deterministic snapshots)

Inputs at `~/paras-workload/<dataset>/spec_8k_<model>.jsonl` (8000 rows per dataset; pre-computed `output_len` from a reference eos-stopped run).

Filter `output_len > 0` (a few pathological rows have zero), shuffle with a fixed seed, take first N, persist to a snapshot file with `idx` (orig spec_8k position) + `sample_idx` (position in snapshot) preserved.

Template: [`scripts/sample_snapshot.py`](scripts/sample_snapshot.py). Use different seeds for independent N=1024 and N=2048 draws (e.g., 42 and 43); the overlap between two snapshots is the natural binomial expectation (~12-25%).

## Subagent contracts

### Per-row forensics (`category=quick`)

Fire as each row's `summary.json` lands. Reads the row dir, writes `<row_dir>/forensics.md` with 5 sections: headline, output-length distribution, server-mode timeseries (peak decode_tps, modes observed, time per mode), server log signals (hard errors count: IndexError/AssertionError/OutOfMemoryError/"state was deleted"/"device-side assert"), anomalies + comparison to baseline.

Prompt template: [`scripts/forensics_prompt.md`](scripts/forensics_prompt.md).

### REPORT.md aggregation (`category=writing`)

Fire after the last row completes. Reads `matrix.csv` + all `forensics.md` files. Writes top-level `REPORT.md` with: headline table (system × (N, cap) → out_tps or ❌), full results matrix, comparative analysis (speedup vs best-static per cell), failure summary with root-cause references, recommendations, methodology + caveats.

Prompt template: [`scripts/report_prompt.md`](scripts/report_prompt.md).

## Pre-flight checklist

Run all in parallel before launching any sweep:

```bash
cd /home/shaoyuw/sglang
git status --short                                                        # known untracked OK
git log --oneline -5                                                      # confirm branch state
conda info --envs | grep sgl_paras                                        # env exists
nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l         # expect 0
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -8 # expect 0 each
df -h /                                                                   # if > 90%, clean /tmp/paras_configure_profile
ls /data/shaoyuw/models/<model>/                                          # model files present
ls ~/paras-workload/<dataset>/spec_8k_<model>.jsonl                       # source dataset
ls artifacts/<UTC>_<sweep>/samples/                                       # snapshots if pre-built
```

## Known gotchas

- **`/health` returns 503 during the post-startup `/generate` warmup.** Always use `server_alive_with_retries` after `wait_ready`.
- **GC stall in `paras_tp_group_all_gather_reqs`** — 1.8 s of CPython gen-2 collection per switch unless `gc.freeze()` lands at warmup. See `docs/paras/runs/20260516_paras_perf_followup/task_0_gc_freeze_paras_switch.md`.
- **Negative `#swa token` in non-paras hybrid-SWA launches** — `SWARadixCache` bookkeeping drift when radix-on + SWA-on without paras's UMM init. Mitigated by `DISABLE_RADIX_CACHE=1` parity default. The "fix" attempt was reverted (`fd85c367c`); proper fix tracked in `task_0b_streamline_switch_path.md` discussion.
- **CUDA OOB token id crash on paras long-tail TP decode** — BF16 NaN logits → unbounded sampler output → `embed_tokens(input_ids)` crashes at `IndexKernel.cu:113`. Fix candidates in `docs/paras/runs/20260515_rollout_matrix/...INVESTIGATION_cuda_oob_token_id_crash.md`. Apply Fix A+B+C before paras N=2048 sweeps.
- **`/tmp/paras_configure_profile/`** can hold 20+ GB of pre-existing Kineto traces. Clean before sweeps if `/` > 90%.
- **tmux ulimit inherits 1024 nofile**; always `ulimit -n 1048576` in the matrix pane.
- **Disk-full silently corrupts logs via tee ENOSPC drops.** Watch for log files that stop growing while the process is alive.
- **gpt-oss-120b mfs=0.80 OOMs in ep-static and paras** (DeepEP buffers + dual cuda-graph capture). Use 0.75. tp-static can use 0.80.

## References

- 20260515 sweep: `artifacts/20260515_sweep_gptoss_dapo/` + 5 `INVESTIGATION_*.md` files
- 20260516 follow-up tasks: `docs/paras/runs/20260516_paras_perf_followup/`
- Base ParaS design: `docs/paras/parallelism_switch.md`
- ParaS policy: `docs/paras/parallelism_switch_policy.md`
- Bench driver: `python/sglang/bench_paras.py`
- Launchers: `scripts/paras/eval/a100/<gptoss|qwen>/launch_server_*.sh`
- Server probe helpers: `scripts/paras/eval/paras_cmd/`
