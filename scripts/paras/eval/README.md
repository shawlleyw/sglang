# `scripts/paras/eval/` — ParaS Evaluation Scripts

Scripts for evaluating ParaS (Parallelism Switching) — both **correctness** (unit / integration tests) and **performance** (microbenchmarks, end-to-end serving). Organized by hardware (A100, H200) and model (qwen3, gpt-oss), with a top-level orchestrator for unit tests and a `paras_cmd/` toolkit for live-server interaction.

## Layout

```
scripts/paras/eval/
├── README.md                   # this file
├── lib.sh                      # shared bash helpers (sourced by launch_*/bench_* scripts)
├── launch_common.sh            # topology-aware setup for launch_server_*.sh
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
| `paras_init_cuda_graph` | `launch_common.sh` | Builds `CUDA_GRAPH_FLAGS` array from `ENABLE_CUDA_GRAPH`, `CUDA_GRAPH_MAX_BS`. |

### `launch_common.sh`

Topology-aware setup sourced by every `launch_server_*.sh`. Provides two public functions; each resolves the parity contract knobs from [`.skills/paras-rollout-eval/SKILL.md`](../../../.skills/paras-rollout-eval/SKILL.md) and auto-sizes `CUDA_GRAPH_MAX_BS` from `MAX_RUNNING_REQUESTS` per topology.

| Function | Used by | Behavior |
|---|---|---|
| `paras_launch_setup_dp_ep` | `launch_server_dp_ep.sh` | DeepEP env exports, `ENABLE_PARAS` toggle → `PARAS_FLAGS`, `HYBRID_SWA_FLAGS` (resolves `auto` → on/off via `ENABLE_PARAS`), `OVERLAP_FLAGS`, `RADIX_FLAGS`, `CUDA_GRAPH_FLAGS`. Default `CUDA_GRAPH_MAX_BS = MAX_RUNNING_REQUESTS / NUM_GPUS` (per-rank attn batch). |
| `paras_launch_setup_tp_tp` | `launch_server_tp_tp.sh` | Unsets stale DeepEP env, exports `SYNC_TOKEN_IDS_ACROSS_TP=1`, builds the same flag arrays. Default `CUDA_GRAPH_MAX_BS = MAX_RUNNING_REQUESTS` (global TP batch). |

**`CUDA_GRAPH_MAX_BS` sizing rule** — applied uniformly to paras and static. User-provided `CUDA_GRAPH_MAX_BS=N` env wins.

| Topology | Default | Reason |
|---|---|---|
| DP/EP | `MAX_RUNNING_REQUESTS / NUM_GPUS` | each DP-attention rank captures graphs for its own ≤ N/8 slice |
| TP/TP | `MAX_RUNNING_REQUESTS` | single global batch dispatched across TP ranks |

Per-model launchers stay thin: they set `MODEL_PATH` and `MEM_FRACTION_STATIC` defaults (with optional `ENABLE_PARAS=1` branch), `HYBRID_SWA` default for gpt-oss, then call the right `paras_launch_setup_*` and splice the resulting arrays into the `python -m sglang.launch_server` invocation. All shared knobs (`MAX_RUNNING_REQUESTS=2048`, `MAX_PREFILL_TOKENS=8192`, `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=MAX_REQ_PER_RANK`, etc.) live in `launch_common.sh` — see the [Override Cheatsheet](#override-cheatsheet) below for the full list.

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

1. Adds `--enable-paras-moe --paras-tp-size $NUM_GPUS` to the launch command. Overlap scheduling remains enabled; `SchedulerParasMixin._paras_drain_overlap_pipeline` drains any in-flight overlap-queued batch before each EP↔TP switch.
2. Sets canonical ParaS env vars (`PARAS_INTRA_NODE_WEIGHT_TRANSFER_METHOD=peer_access`, `PARAS_KV_TRANSFER_METHOD=peer_access`, etc. — see table below).
3. User-supplied env vars on the same line override every ParaS default.
4. `PARAS_AUTO_SWITCH=0` additionally appends `--no-paras-auto-switch` (use this when driving manual switches via `/paras_configure_{tp,ep}`).

Without `ENABLE_PARAS=1`, the same script launches a vanilla DP+EP server suitable for serving benchmarks.

### Override Cheatsheet

Every var below honors the standard `${VAR:-default}` pattern: `VAR=N bash launch_server_*.sh` overrides; the launcher's default applies otherwise. Defaults live in `launch_common.sh` (shared) or the per-model script (model-specific). Anything passed after `--` on the command line (via `"$@"`) reaches `sglang.launch_server` directly and wins via argparse last-value semantics.

#### Set in `launch_common.sh` (shared by every launcher)

| Var | Default | Topology | Purpose |
|---|---|---|---|
| `HOST` | `0.0.0.0` | both | bind address |
| `PORT` | `30000` | both | bind port |
| `NUM_GPUS` | `8` | both | tp/dp/ep size; seeds `CUDA_VISIBLE_DEVICES` |
| `CUDA_VISIBLE_DEVICES` | `0,1,...,NUM_GPUS-1` | both | physical GPUs (only set if unset) |
| `ENABLE_CUDA_GRAPH` | `1` | both | `0` ⇒ pass `--disable-cuda-graph` |
| `MAX_RUNNING_REQUESTS` | `2048` | both | sglang `--max-running-requests`; uniform across all models/servers |
| `MAX_PREFILL_TOKENS` | `8192` | both | sglang `--max-prefill-tokens`; uniform across all models/servers |
| `MAX_REQ_PER_RANK` | `MAX_RUNNING_REQUESTS / NUM_GPUS` | dp_ep | derived; feeds the three per-rank knobs below |
| `CUDA_GRAPH_MAX_BS` | `MAX_REQ_PER_RANK` (dp_ep) / `MAX_RUNNING_REQUESTS` (tp_tp) | both | `--cuda-graph-max-bs` (only honored when graph enabled) |
| `SGLANG_DEEPEP_BF16_DISPATCH` | `true` | dp_ep | BF16 in DeepEP dispatch |
| `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` | `MAX_REQ_PER_RANK` | dp_ep | DeepEP buffer size per rank (NVSHMEM_QP_DEPTH constraint) |
| `NVSHMEM_QP_DEPTH` | `2048` | dp_ep | NVSHMEM queue pair depth |
| `DISABLE_OVERLAP` | `0` | both | `1` adds `--disable-overlap-schedule` |
| `DISABLE_RADIX_CACHE` | `1` | both | `1` (default) adds `--disable-radix-cache` (paras parity) |
| `HYBRID_SWA` | `auto` (dp_ep) / unset (tp_tp) | both | `auto` resolves to `1` under paras, `0` under static. `0` adds `--disable-hybrid-swa-memory`. Only meaningful for gpt-oss. |
| `SYNC_TOKEN_IDS_ACROSS_TP` | `1` (exported) | tp_tp only | MIN all-reduce on sampled token ids; parity with paras post-EP→TP swap |

#### ParaS-only (set by `launch_common.sh` when `ENABLE_PARAS=1`)

| Var | Default | Purpose |
|---|---|---|
| `ENABLE_PARAS` | `0` | `1` enables paras mode |
| `PARAS_AUTO_SWITCH` | `1` | `0` adds `--no-paras-auto-switch` for manual `/paras_configure_*` testing |
| `PARAS_INTRA_NODE_WEIGHT_TRANSFER_METHOD` | `peer_access` | `peer_access` or `nccl`; controls TP-local weight resharding |
| `PARAS_KV_TRANSFER_METHOD` | `peer_access` | `peer_access` or `naive` |
| `PARAS_DISABLE_PEER_ACCESS` | `0` | `1` force-disables peer-access pre-init at boot |
| `SGLANG_ATTN_MAX_BS` | `MAX_REQ_PER_RANK` | attention batch-size cap (must be ≥ per-rank attn batch) |

#### Set in per-model launcher (model-specific)

| Var | Where set | Default |
|---|---|---|
| `MODEL_PATH` | per-model | `a100/gptoss`: `/data/shaoyuw/models/gpt-oss-120b-BF16-unsloth`; `a100/qwen`: `/data/shaoyuw/models/Qwen3-30B-A3B`; `h200/qwen`: `/models/Qwen3-235B-A22B-Instruct-2507` |
| `MEM_FRACTION_STATIC` | per-model | `a100/gptoss` dp_ep: `0.75`; `a100/gptoss` tp_tp: `0.8`; qwen dp_ep paras: `0.6` (a100), `0.85` (h200); qwen static and tp_tp: `0.85` |

#### Used by `bench_one_batch_*` only (out of scope for `launch_common.sh`)

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
