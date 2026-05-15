# `scripts/paras/eval/` — ParaS Evaluation Scripts

Scripts for evaluating ParaS (Parallelism Switching) — both **correctness** (unit / integration tests) and **performance** (microbenchmarks, end-to-end serving). Organized by hardware (A100, H200) and model (qwen3, gpt-oss), with a top-level orchestrator for unit tests and a `paras_cmd/` toolkit for live-server interaction.

## Layout

```
scripts/paras/eval/
├── README.md                   # this file
├── lib.sh                      # shared bash helpers (sourced by launch_*/bench_* scripts)
├── run_paras_tests.sh          # correctness test runner (pytest + torchrun)
├── paras_cmd/                  # server interaction toolkit for ParaS EP↔TP smoke tests
│   ├── lib.sh
│   ├── kill.sh                 # steps 1, 14
│   ├── wait_ready.sh           # step 3
│   ├── health.sh               # step 4
│   ├── send_prompts.sh         # steps 5, 8, 10
│   ├── configure.sh            # steps 6, 9
│   ├── inflight_switch.sh      # steps 11, 12
│   ├── check_log.sh            # steps 7, 13
│   └── e2e_test.sh             # orchestrates steps 3–13
├── a100/
│   ├── qwen/                   # Qwen3-30B-A3B on 4×A100-80GB
│   └── gptoss/                 # gpt-oss-120b-BF16 on 4×A100-80GB
└── h200/
    └── qwen/                   # Qwen3-30B-A3B on H200
```

Each `<hw>/<model>/` directory contains six scripts following a fixed naming convention (see [Per-Hardware Launch + Bench Scripts](#per-hardware-launch--bench-scripts) below).

## Top-Level Scripts

### `lib.sh`

Shared bash helpers sourced by every per-hardware launch / bench script. **Do not run directly.** Provides:

| Function | Used by | Purpose |
|---|---|---|
| `paras_default_cvd` | all | If `CUDA_VISIBLE_DEVICES` is unset, set it to `0,1,...,NUM_GPUS-1`. |
| `paras_init_profile` | `bench_one_batch_*` | Builds `LAUNCHER`, `PROFILE_FLAGS`, `LOAD_FORMAT_FLAGS` arrays from `ENABLE_NSYS`, `ENABLE_TORCH_PROFILE`, `LOAD_FORMAT`. |
| `paras_init_cuda_graph` | `launch_server_*` | Builds `CUDA_GRAPH_FLAGS` array from `ENABLE_CUDA_GRAPH`, `CUDA_GRAPH_MAX_BS`. |

### `run_paras_tests.sh`

Drives ParaS correctness tests (pytest + torchrun). Five test groups:

| Group | Source | Runner | Coverage |
|---|---|---|---|
| `partition` | `test/srt/paras/test_request_partition.py` | pytest (CPU) | request partition logic for EP↔TP boundary |
| `kv` | `test/srt/paras/test_kv_cache_transfer.py` | torchrun (multi-GPU) | gather / scatter KV cache between modes |
| `kv-rep` | `test/srt/paras/test_kv_cache_transfer_replication.py` | torchrun (multi-GPU) | KV transfer with replicated layouts |
| `weight` | `test/srt/paras/test_weight_transfer.py` | torchrun (multi-GPU) | MoE weight redistribution |
| `gpt-oss-cuda-graph` | `test/srt/paras/test_paras_gpt_oss_cuda_graph.py` | torchrun (multi-GPU) | per-mode CUDA graph state preservation, gpt-oss layouts |

Usage:

```bash
# Run everything on 4 GPUs (default)
bash scripts/paras/eval/run_paras_tests.sh

# Run only the KV cache transfer suite
ONLY=kv bash scripts/paras/eval/run_paras_tests.sh

# 8 GPUs (positional arg or env var)
bash scripts/paras/eval/run_paras_tests.sh 8
NUM_GPUS=8 bash scripts/paras/eval/run_paras_tests.sh
```

Exits non-zero if any group fails. Prints a `PASS/FAIL/SKIP` summary at the end.

## `paras_cmd/` — Server Interaction Toolkit

Wraps the 14-step manual ParaS smoke-test procedure (live server, real `/v1/completions` requests, real `/paras_configure_*` switches). Caller owns steps 1, 2, 14 (kill, launch, cleanup); `e2e_test.sh` drives steps 3–13.

| Script | Args | Skill step(s) | Action |
|---|---|---|---|
| `kill.sh` | — | 1, 14 | `pkill -9 -f sglang`, `rm $LOG_FILE` |
| `wait_ready.sh` | — | 3 | poll `$LOG_FILE` for "Application startup complete" |
| `health.sh` | — | 4 | HTTP `/health` + parse `Load weight end ... type=...` |
| `send_prompts.sh` | `<LABEL>` | 5, 8, 10 | three canonical prompts (binary search / train math / TCP-vs-UDP) |
| `configure.sh` | `<tp\|ep>` | 6, 9 | hit `/paras_configure_<mode>`, time the call |
| `inflight_switch.sh` | `<tp\|ep>` | 11, 12 | two background completions + mid-decode switch |
| `check_log.sh` | `<timing\|errors\|cuda_graph\|all>` | 7, 13 | grep timing lines, error lines, dual-capture lines |
| `e2e_test.sh` | — | 3–13 | run all of the above in order, stop on first failure |

All scripts source `paras_cmd/lib.sh`, which exposes:

| Env var | Default | Purpose |
|---|---|---|
| `HOST` | `127.0.0.1` | server host |
| `PORT` | `30000` | server port |
| `MODEL_NAME` | `Qwen3-30B-A3B` | name in OpenAI completions request body (override for gpt-oss) |
| `LOG_FILE` | `/tmp/sglang_paras_test.log` | server log path that the launch script `tee`'d |
| `PRINT_CHARS` | `300` | response chars to print per prompt |
| `INFLIGHT_DELAY` | `4` | seconds before triggering the in-flight switch |
| `INFLIGHT_MAX_TOKENS` | `500` | `max_tokens` for in-flight requests |
| `TIMEOUT_TRIES` | `24` | `wait_ready` max attempts |
| `SLEEP_BETWEEN` | `5` | `wait_ready` seconds between attempts |

### Canonical Workflow

**Qwen3-30B-A3B** (FlashInfer, eager — defaults match `lib.sh`, no env overrides required):

```bash
bash scripts/paras/eval/paras_cmd/kill.sh
ENABLE_PARAS=1 NUM_GPUS=4 \
    bash scripts/paras/eval/a100/qwen/launch_server_dp_ep.sh \
    2>&1 | tee /tmp/sglang_paras_test.log &
bash scripts/paras/eval/paras_cmd/e2e_test.sh
bash scripts/paras/eval/paras_cmd/kill.sh
```

**gpt-oss-120b-BF16** (Triton + dual capture — needs longer wait_ready and a different model name):

```bash
export MODEL_NAME=gpt-oss-120b-BF16-unsloth
export LOG_FILE=/tmp/sglang_paras_gptoss.log
export TIMEOUT_TRIES=60
export SLEEP_BETWEEN=10

bash scripts/paras/eval/paras_cmd/kill.sh
ENABLE_PARAS=1 NUM_GPUS=4 \
    bash scripts/paras/eval/a100/gptoss/launch_server_dp_ep.sh \
    2>&1 | tee "$LOG_FILE" &
bash scripts/paras/eval/paras_cmd/e2e_test.sh
bash scripts/paras/eval/paras_cmd/kill.sh
```

Full walkthrough in [`.skills/paras-test-manual-switch/SKILL.md`](../../../.skills/paras-test-manual-switch/SKILL.md) (covers both qwen3 and gpt-oss). Companion skill [`.skills/paras-test-auto-switch/SKILL.md`](../../../.skills/paras-test-auto-switch/SKILL.md) tests the load-driven autoswitch policy.

## Per-Hardware Launch + Bench Scripts

Each `<hw>/<model>/` directory contains six scripts named:

```
<action>_<attn>_<expert>.sh
```

| Action | Purpose |
|---|---|
| `launch_server` | starts a long-lived sglang server you bench against with `python -m sglang.bench_serving` |
| `bench_one_batch` | one-shot microbench via `python -m sglang.bench_one_batch`, returns latency / throughput per `--batch-size` |

| Suffix | Attention parallelism | MoE expert parallelism | Notes |
|---|---|---|---|
| `dp_ep` | DP (`--enable-dp-attention`) | EP (DeepEP) | the ParaS-relevant configuration |
| `dp_tp` | DP | TP (no DeepEP) | bench-only baseline |
| `tp_ep` | TP | EP | bench-only baseline |
| `tp_tp` | TP | TP | the "switched-to" mode in ParaS |

Models present:
- `a100/qwen/` — Qwen3-30B-A3B on A100-80GB
- `a100/gptoss/` — gpt-oss-120b-BF16-unsloth on A100-80GB
- `h200/qwen/` — Qwen3-30B-A3B on H200

### `ENABLE_PARAS=1` toggle

`launch_server_dp_ep.sh` for both qwen and gptoss accepts `ENABLE_PARAS=1`, which:

1. Adds `--enable-paras-moe --paras-tp-size $NUM_GPUS` to the launch command (qwen also adds `--chunked-prefill-size -1 --max-prefill-tokens 32000`). Overlap scheduling remains enabled; `SchedulerParasMixin._paras_drain_overlap_pipeline` drains any in-flight overlap-queued batch before each EP↔TP switch.
2. Sets canonical ParaS env vars (`PARAS_CONFIGURE_METHOD=peer_access`, `PARAS_KV_TRANSFER_METHOD=peer_access`, etc.) — defaults differ slightly between qwen3 (eager) and gpt-oss (cuda-graph dual capture). See each script's header for the full list.
3. User-supplied env vars on the same line override the ParaS defaults.

Without `ENABLE_PARAS=1`, the same script launches a vanilla DP+EP server suitable for serving benchmarks.

### Common Env Var Cheatsheet

Used by **launch + bench scripts**:

| Var | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | model-specific | hf snapshot path (override for non-default checkout) |
| `HOST` | `0.0.0.0` | bind address (launch) |
| `PORT` | `30000` | bind port (launch) |
| `NUM_GPUS` | `8` | tp/dp/ep size; also seeds `CUDA_VISIBLE_DEVICES` |
| `CUDA_VISIBLE_DEVICES` | `0,1,...,N-1` | physical GPUs |
| `MEM_FRACTION_STATIC` | model+mode-specific | sglang `--mem-fraction-static` |
| `MAX_RUNNING_REQUESTS` | model+mode-specific | sglang `--max-running-requests` |
| `ENABLE_CUDA_GRAPH` | `1` | `0` ⇒ pass `--disable-cuda-graph` |
| `CUDA_GRAPH_MAX_BS` | unset | `--cuda-graph-max-bs` (only honored when graph enabled) |

Used by **bench scripts only** (`paras_init_profile`):

| Var | Default | Purpose |
|---|---|---|
| `ENABLE_NSYS` | `0` | wrap with `nsys profile --cuda-graph-trace=node -t cuda` |
| `NSYS_OUTPUT` | `/tmp/nsys_<RUN_NAME>` | nsys output prefix |
| `ENABLE_TORCH_PROFILE` | `0` | pass `--profile --disable-cuda-graph` |
| `SGLANG_TORCH_PROFILER_DIR` | `/tmp/torch_profile_<RUN_NAME>` | torch.profiler output dir |
| `LOAD_FORMAT` | unset | sglang `--load-format` value |
| `BATCH_SIZE` | `"1 8 64 256 1 8 64 256"` | space-separated list of `--batch-size` values |
| `INPUT_LEN`, `OUTPUT_LEN` | `10` | prompt / decode length |
| `RESULT_FILE` | `/tmp/<model>_<config>.jsonl` | bench output |
| `RUN_NAME` | `<config>` | tag in output / profile dir names |

Used by **`launch_server_dp_ep`** (DeepEP path):

| Var | Default | Purpose |
|---|---|---|
| `SGLANG_DEEPEP_BF16_DISPATCH` | `true` | use BF16 in DeepEP dispatch |
| `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` | `512` (`256` under `ENABLE_PARAS=1`) | DeepEP buffer size per rank |
| `NVSHMEM_QP_DEPTH` | `2048` | NVSHMEM queue pair depth |

Used by **ParaS launches only** (`ENABLE_PARAS=1`):

| Var | Default | Purpose |
|---|---|---|
| `PARAS_CONFIGURE_METHOD` | `peer_access` | `peer_access` (default) or `naive` (NCCL all-to-all) |
| `PARAS_KV_TRANSFER_METHOD` | `peer_access` | `peer_access` (default) or `naive` |
| `PARAS_DISABLE_PEER_ACCESS` | `0` (gpt-oss) | force-disable peer-access pre-init at boot |
| `SGLANG_ATTN_MAX_BS` | `256` | attention batch-size cap (must be ≥ `MAX_RUNNING_REQUESTS / NUM_GPUS`) |

## Common Workflows

### 1. Run all correctness tests

```bash
bash scripts/paras/eval/run_paras_tests.sh
```

### 2. Run end-to-end ParaS EP↔TP test (qwen3)

```bash
bash scripts/paras/eval/paras_cmd/kill.sh
ENABLE_PARAS=1 NUM_GPUS=4 \
    bash scripts/paras/eval/a100/qwen/launch_server_dp_ep.sh \
    2>&1 | tee /tmp/sglang_paras_test.log &
bash scripts/paras/eval/paras_cmd/e2e_test.sh
bash scripts/paras/eval/paras_cmd/kill.sh
```

See [`.skills/paras-test-manual-switch/SKILL.md`](../../../.skills/paras-test-manual-switch/SKILL.md) for the full walkthrough, expected baselines, and known failure modes (covers both qwen3 and gpt-oss).

### 3. Microbench one batch (qwen3 dp_ep)

```bash
NUM_GPUS=4 BATCH_SIZE="1 8 64 256" \
    bash scripts/paras/eval/a100/qwen/bench_one_batch_dp_ep.sh
```

Add `ENABLE_NSYS=1` for an nsight-systems profile, `ENABLE_TORCH_PROFILE=1` for a torch.profiler trace.

### 4. Serving benchmark against a launched server

```bash
# In one shell:
NUM_GPUS=4 bash scripts/paras/eval/a100/qwen/launch_server_dp_ep.sh

# In another:
python -m sglang.bench_serving --backend sglang \
    --host 127.0.0.1 --port 30000 \
    --dataset-name sharegpt --num-prompts 1000
```

## See Also

- [`.skills/paras-test-manual-switch/SKILL.md`](../../../.skills/paras-test-manual-switch/SKILL.md) — manual `/paras_configure_*` switch test (covers qwen3 FlashInfer-eager and gpt-oss Triton-dual-capture)
- [`.skills/paras-test-auto-switch/SKILL.md`](../../../.skills/paras-test-auto-switch/SKILL.md) — load-driven autoswitch policy test (cycle 1 light + cooldown + 32-burst + cycle 3 light + assert autoswitch fired)
- [`.skills/paras-test-peer-access/SKILL.md`](../../../.skills/paras-test-peer-access/SKILL.md) — unit-test skill for KV transfer + weight transfer + request partition
- `docs/paras/gpt_oss_support.md` — gpt-oss design doc + bug chronicle
- `test/srt/paras/` — Python test sources driven by `run_paras_tests.sh`
