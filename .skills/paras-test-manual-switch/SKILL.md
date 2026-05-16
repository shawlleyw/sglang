---
name: paras-test-manual-switch
description: Test ParaS manual EP↔TP switching for any supported model. Covers Qwen3-30B-A3B (4×A100, FlashInfer, cuda-graph) and gpt-oss-120b-bf16 (4-8×A100, Triton, cuda-graph dual capture). Drives the canonical 14-step procedure via scripts/paras/eval/paras_cmd/e2e_test.sh: kill → launch → wait → health → 5 send_prompts/configure pairs (round-trip pre-switch / TP / round-trip post-switch) → 2 inflight_switch (EP→TP, TP→EP) → check_log → kill. send_prompts and inflight_switch each fire BURST_SIZE=32 parallel diverse prompts and fail-fast on the degenerate-attractor regex. Use after any change to paras/ code or scripts/paras/eval/.
---

# Test ParaS Manual EP↔TP Switch

Verifies ParaS configure_tp / configure_ep manual switches preserve correctness:
weights survive transfer, in-flight requests migrate cleanly, and round-trip
EP→TP→EP is functionally consistent.

## When to use

- After any change to `python/sglang/srt/paras/` (e.g., scheduler mixin, gather/scatter
  manager, cache transfer backends, model-specific `paras_*` modules)
- After any change to ParaS launch flags or eval scripts under `scripts/paras/eval/`
- Before any PR touching ParaS code is sent for review

## Supported models

| Concern | Qwen3-30B-A3B | gpt-oss-120b-bf16 |
|---|---|---|
| Launch script | [`scripts/paras/eval/a100/qwen/launch_server_dp_ep.sh`](file:///home/shaoyuw/sglang/scripts/paras/eval/a100/qwen/launch_server_dp_ep.sh) | [`scripts/paras/eval/a100/gptoss/launch_server_dp_ep.sh`](file:///home/shaoyuw/sglang/scripts/paras/eval/a100/gptoss/launch_server_dp_ep.sh) |
| `MODEL_NAME` (in OpenAI request body) | `Qwen3-30B-A3B` | `gpt-oss-120b-BF16-unsloth` |
| Recommended `LOG_FILE` | `/tmp/sglang_paras_test.log` (lib.sh default) | `/tmp/sglang_paras_gptoss.log` |
| `TIMEOUT_TRIES` for `wait_ready.sh` | 24 (default; ~2 min) | **60** (gpt-oss boots ~5 min) |
| `SLEEP_BETWEEN` for `wait_ready.sh` | 5 (default) | **10** |
| Default GPU count | 4 | 4 (test) or 8 (canonical deployment) |
| Default attention backend | FlashInfer | **Triton** (required by gpt-oss SWA + sinks) |
| Default cuda graph | cuda graph (`ENABLE_CUDA_GRAPH=1`; max-bs auto = `MAX_RUNNING_REQUESTS/NUM_GPUS` = 256) | cuda graph dual capture (`ENABLE_CUDA_GRAPH=1`; max-bs auto = 256) |
| Hybrid SWA + dual-cache pool | n/a (vanilla full attention) | yes (18 full + 18 SWA layers) |
| Per-head attention sinks | n/a | yes |
| MoE w13 layout | concat `[gate..., up...]` | interleaved `[g0, u0, g1, u1, ...]` |
| Expected `transfer_weights` time | ~106 ms | ~270 ms |
| Expected `configure TP` total | ~125 ms | ~310 ms |
| Expected `configure EP` total | ~125 ms | ~290 ms |
| Pass-criterion model type | `Qwen3MoeForCausalLMParaS` | `GptOssForCausalLMParaS` |
| Model file path | `/data/shaoyuw/models/Qwen3-30B-A3B` | `/data/shaoyuw/models/gpt-oss-120b-BF16-unsloth` (the **unsloth** BF16 build, not the older mislabeled fp8 path — see "Model selection" below) |

The 14-step procedure itself is **identical** for both models. Only the launch
invocation and a few env vars differ.

## Prerequisites

- **Conda env**: `sgl_paras`
- **GPUs**: 4× A100-80GB minimum (use `CUDA_VISIBLE_DEVICES=0,1,2,3` or `4,5,6,7`)
  for both models. Optionally 8× A100-80GB for gpt-oss canonical deployment.
- **Working dir**: this repo

## Quick Start (Qwen3-30B-A3B)

```bash
conda activate sgl_paras
cd /home/shaoyuw/sglang
pip install -e python/ -q --no-deps

bash scripts/paras/eval/paras_cmd/kill.sh                            # step 1

# PARAS_AUTO_SWITCH=0 disables load-driven autoswitch so the test exercises
# manual /paras_configure_* endpoints only. The toggle is wired in
# launch_common.sh for ALL dp_ep launchers (qwen + gpt-oss).
ENABLE_PARAS=1 PARAS_AUTO_SWITCH=0 NUM_GPUS=4 MEM_FRACTION_STATIC=0.7 \
    bash scripts/paras/eval/a100/qwen/launch_server_dp_ep.sh \
    2>&1 | tee /tmp/sglang_paras_test.log &                          # step 2

bash scripts/paras/eval/paras_cmd/e2e_test.sh                        # steps 3-13

bash scripts/paras/eval/paras_cmd/kill.sh                            # step 14
```

## Quick Start (gpt-oss-120b-bf16, 4×A100)

```bash
conda activate sgl_paras
cd /home/shaoyuw/sglang
pip install -e python/ -q --no-deps

# gpt-oss-specific env overrides for paras_cmd helpers.
export MODEL_NAME=gpt-oss-120b-BF16-unsloth
export LOG_FILE=/tmp/sglang_paras_gptoss.log
export TIMEOUT_TRIES=60        # gpt-oss boots ~5 min (vs qwen3 ~40 s)
export SLEEP_BETWEEN=10

bash scripts/paras/eval/paras_cmd/kill.sh                            # step 1

# 4-GPU config: bump MEM_FRACTION_STATIC to 0.8 (per-rank weights ≈ 57 GiB).
# PARAS_AUTO_SWITCH=0 disables load-driven autoswitch — REQUIRED for any
# manual-switch correctness test. See "Disabling autoswitch" below.
ENABLE_PARAS=1 PARAS_AUTO_SWITCH=0 NUM_GPUS=4 MEM_FRACTION_STATIC=0.8 \
    bash scripts/paras/eval/a100/gptoss/launch_server_dp_ep.sh \
    2>&1 | tee "$LOG_FILE" &                                         # step 2

bash scripts/paras/eval/paras_cmd/e2e_test.sh                        # steps 3-13

bash scripts/paras/eval/paras_cmd/kill.sh                            # step 14
```

## Quick Start (gpt-oss-120b-bf16, 8×A100 — canonical deployment)

Same as above except the launch line uses 8 GPUs and `MEM_FRACTION_STATIC=0.7`
(per-rank weight footprint halves with TP=8, so 0.7 is sufficient and matches
the qwen3 baseline):

```bash
ENABLE_PARAS=1 PARAS_AUTO_SWITCH=0 NUM_GPUS=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MEM_FRACTION_STATIC=0.7 \
    bash scripts/paras/eval/a100/gptoss/launch_server_dp_ep.sh \
    2>&1 | tee "$LOG_FILE" &
```

### MHA-only path (`--disable-hybrid-swa-memory`)

To force gpt-oss into MHA-only mode (no separate SWA sub-pool — every layer's
KV cache lives in the full pool), pass `--disable-hybrid-swa-memory` after the
launch script's `"$@"` forwarding:

```bash
ENABLE_PARAS=1 PARAS_AUTO_SWITCH=0 NUM_GPUS=8 MEM_FRACTION_STATIC=0.7 \
    bash scripts/paras/eval/a100/gptoss/launch_server_dp_ep.sh \
    --disable-hybrid-swa-memory \
    2>&1 | tee "$LOG_FILE" &
```

This routes ALL 36 gpt-oss layers (full + SWA-classified) through the MHA
cache transfer backend at switch time. Use this configuration to isolate
SWA-cache-transfer-specific bugs from upstream-of-cache-transfer bugs (e.g.,
to confirm whether a bug lives in `SWACacheTransfer` or in shared scheduler /
req-metadata / cuda-graph code).

## Detailed Procedure

`e2e_test.sh` orchestrates steps 3-13 in order and stops on the first non-zero
return. Run individual steps below for partial reruns or debugging. All
helpers live in [`scripts/paras/eval/paras_cmd/`](file:///home/shaoyuw/sglang/scripts/paras/eval/paras_cmd/).

### 1. Kill any existing sglang processes

```bash
bash scripts/paras/eval/paras_cmd/kill.sh
```

`pkill -9 -f sglang`, sleeps 3 s, removes `$LOG_FILE`.

### 2. Install and launch server

The launch is intentionally **not** wrapped by `e2e_test.sh` — the launch
script is model-specific, the log path is model-specific, and the launch is
long-lived (must be backgrounded or run in tmux). See model-specific Quick
Start blocks above for the exact launch command.

Shared defaults live in `launch_common.sh` (sourced by every launcher):

- `MAX_RUNNING_REQUESTS=2048`, `MAX_PREFILL_TOKENS=8192` (uniform across all
  models / servers).
- `CUDA_GRAPH_MAX_BS = MAX_RUNNING_REQUESTS / NUM_GPUS` for DP/EP (per-rank
  attn batch); `= MAX_RUNNING_REQUESTS` for TP/TP (global batch).
- `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK = MAX_REQ_PER_RANK`,
  `SGLANG_ATTN_MAX_BS = MAX_REQ_PER_RANK` (paras-only).
- `DISABLE_RADIX_CACHE=1`, `DISABLE_OVERLAP=0` (overlap is now supported under
  paras via `SchedulerParasMixin._paras_drain_overlap_pipeline`).

Per-model launcher only sets `MODEL_PATH` and `MEM_FRACTION_STATIC`:

- **qwen3 dp_ep paras**: `MEM_FRACTION_STATIC=0.6` default. Override
  `MEM_FRACTION_STATIC=0.7` on the launch line per the standard test config.
- **gpt-oss dp_ep**: `MEM_FRACTION_STATIC=0.75` default (paras + static),
  Triton attention + Triton MoE backend pinned, `HYBRID_SWA=auto` (on under
  paras, off under static). Override `MEM_FRACTION_STATIC=0.8` for 4-GPU
  paras testing (per-rank weights ≈ 57 GiB at TP=4).

Both `launch_server_dp_ep.sh` add `--enable-paras-moe --paras-tp-size $NUM_GPUS
--enable-nan-detection --chunked-prefill-size -1 --disable-radix-cache` under
`ENABLE_PARAS=1`, plus PARAS env vars (`PARAS_CONFIGURE_METHOD=peer_access`,
`PARAS_KV_TRANSFER_METHOD=peer_access`).

### 3. Wait for server ready

```bash
bash scripts/paras/eval/paras_cmd/wait_ready.sh
```

Polls `$LOG_FILE` for "Application startup complete". qwen3 typically ~40 s;
gpt-oss typically ~5 min (raise `TIMEOUT_TRIES=60` and `SLEEP_BETWEEN=10`).

### 4. Verify health and model type

```bash
bash scripts/paras/eval/paras_cmd/health.sh
```

Checks HTTP `/health` → 200 and parses model type from the log. Must show
`Qwen3MoeForCausalLMParaS` or `GptOssForCausalLMParaS` (NOT the bare class
name without `ParaS`).

For the gpt-oss cuda-graph variant, also verify dual capture happened:

```bash
bash scripts/paras/eval/paras_cmd/check_log.sh cuda_graph
# Expected last lines:
#   "ParaS: dual capture complete avail=...GB  #EP graphs=4  #TP graphs=4"
#   "TP capture done (..., pools_differ=True, ...)"
```

`pools_differ=True` confirms each mode owns its own buffer set.

### 5. Send requests in EP mode (pre-switch)

```bash
bash scripts/paras/eval/paras_cmd/send_prompts.sh EP
```

Sends `BURST_SIZE=32` parallel diverse prompts (from
[`paras_cmd/prompts_diverse.txt`](file:///home/shaoyuw/sglang/scripts/paras/eval/paras_cmd/prompts_diverse.txt))
and **fail-fast verifies** no response matches the degenerate-attractor regex
`(\b\w+\b)(\s+\1){5,}` (5+ consecutive identical word repetitions). Returns
non-zero on any degenerate response.

Override `BURST_SIZE=N` for lower concurrency on memory-constrained systems.

### 6. Trigger ParaS EP→TP switch

```bash
bash scripts/paras/eval/paras_cmd/configure.sh tp
# Expected response: "ParaS TP parallelism configured."
```

### 7. Check timing

```bash
bash scripts/paras/eval/paras_cmd/check_log.sh timing
```

Expected baselines (mem-fraction-static=0.7, peer-access transport):

| Metric | Qwen3 (4×A100) | gpt-oss (4×A100) | Pass threshold |
|---|---|---|---|
| `transfer_weights` | ~106 ms | ~270 ms | < 2000 ms |
| `gather_cache` (in-flight) | n/a (eager) | ~8 ms (peer) | < 100 ms |
| `scatter_cache` (in-flight) | n/a (eager) | ~18-24 ms (peer) | < 100 ms |
| `configure TP` total | ~125 ms | ~280-335 ms | < 2500 ms |
| `configure EP` total | ~125 ms | ~290-300 ms | < 2500 ms |

Peer-access is ~2.2-3.4× faster than the legacy `naive` NCCL all-to-all path
(`PARAS_CONFIGURE_METHOD=naive` ~600-900 ms).

### 8. Send same requests in TP mode (post-switch)

```bash
bash scripts/paras/eval/paras_cmd/send_prompts.sh TP
```

Same shape as step 5. **Critical**: must produce 0/32 degenerate. The
canonical gpt-oss failure mode this catches is the SWA layer-specs misdispatch
(Bug B): if `paras_layer_specs` doesn't reach the scatter/gather managers,
hybrid layers go through the wrong cache transfer backend and post-switch
decode silently corrupts.

### 9. Trigger ParaS TP→EP switch (round-trip)

```bash
bash scripts/paras/eval/paras_cmd/configure.sh ep
```

### 10. Send requests in EP mode (post-roundtrip)

```bash
bash scripts/paras/eval/paras_cmd/send_prompts.sh EP-RT
```

**Determinism check**: with cuda-graph state preservation and BF16 precision,
post-roundtrip EP outputs should be functionally equivalent to pre-switch EP
(step 5) outputs. Wording may differ in the last few decoded tokens due to
floating-point rounding; semantics must be the same. The fail-fast check is
"no degeneration" — exact equivalence is too strict.

### 11. In-flight EP→TP switch

```bash
bash scripts/paras/eval/paras_cmd/inflight_switch.sh tp
```

Spawns `BURST_SIZE=32` concurrent `INFLIGHT_MAX_TOKENS=500` completions, waits
`INFLIGHT_DELAY=4` s for them to enter decode, fires
`/paras_configure_tp` mid-decode, waits for completions, fail-fast verifies
no degeneration.

This stresses the gather (EP→TP) cache transfer + per-Req metadata
propagation path. Failure modes caught: stale `req_to_token` mapping, SWA
layer mapping snapshot bugs, captured-graph kv_indptr staleness.

### 12. In-flight TP→EP switch

```bash
bash scripts/paras/eval/paras_cmd/inflight_switch.sh ep
```

Mirror of step 11 in the opposite direction (scatter cache transfer).

### 13. Verify no errors

```bash
bash scripts/paras/eval/paras_cmd/check_log.sh errors
```

Greps `error|exception` from `$LOG_FILE` and filters known-benign warnings
(`opentelemetry`, `WARNING`, `UserWarning`, `Config file`, `import error`,
`warn_only`). Returns non-zero if any unexpected error is found.

### 14. Cleanup

```bash
bash scripts/paras/eval/paras_cmd/kill.sh
```

## Pass / Fail Criteria

| Check | Pass | Fail |
|---|---|---|
| Model type (step 4) | `<Model>ForCausalLMParaS` | bare class name without `ParaS` suffix |
| Dual capture (gpt-oss only, step 4) | `pools_differ=True`, `#EP graphs=4 #TP graphs=4` | missing or `pools_differ=False` |
| send_prompts EP / TP / EP-RT (steps 5/8/10) | 0/32 degenerate, all return ≥ 10 chars | any degenerate response or empty/timeout |
| EP→TP switch latency (step 7) | < 2500 ms | > 2500 ms or HTTP timeout |
| TP→EP switch latency (step 7) | < 2500 ms | > 2500 ms or HTTP timeout |
| inflight_switch tp (step 11) | 0/32 degenerate after migration | any degenerate response or crash |
| inflight_switch ep (step 12) | 0/32 degenerate after migration | any degenerate response or crash |
| Server errors (step 13) | None | any scheduler exception, NVLink IPC error, or CUDA error |

### Disabling autoswitch

`paras_auto_switch=True` is the server default — under load the policy fires
EP↔TP switches automatically, which contaminates manual-switch correctness
tests. ALWAYS disable it for manual-switch tests:

| Mechanism | Where | Notes |
|---|---|---|
| `PARAS_AUTO_SWITCH=0` env var | `launch_common.sh` (`paras_launch_setup_dp_ep`) | Wired for every dp_ep launcher (qwen + gpt-oss); appends `--no-paras-auto-switch` automatically when `ENABLE_PARAS=1`. |
| `--no-paras-auto-switch` CLI flag | sglang server | Direct; works for any model. Pass via `"$@"` to the launch script. |

Confirmed by code audit: setting either reliably stops ALL automatic switches.
The flag results in `_paras_auto_policy = None` in
`scheduler_paras_mixin.init_paras_config()`, and every access path
(`paras_auto_observe`, `paras_auto_pick_signal`,
`_paras_auto_clear_window_on_switch`) returns early when the policy is `None`.
No HTTP runtime toggle exists; the toggle is server-startup-only.

## Important Notes

- **`MEM_FRACTION_STATIC` depends on (model, GPU count)** — always override
  it explicitly on the launch line. The launch script's `ENABLE_PARAS=1`
  defaults differ by model and don't track these constraints. The matrix:

  | Model | GPUs | `MEM_FRACTION_STATIC` |
  |---|---|---|
  | Qwen3-30B-A3B | 4 | **0.7** |
  | Qwen3-30B-A3B | 8 | 0.7 |
  | gpt-oss-120b-bf16 | 4 | **0.8** (model weights ≈ 57 GiB/rank; at 0.7 the static budget = 55.5 GiB → kv_budget = 0) |
  | gpt-oss-120b-bf16 | 8 | **0.7** (per-rank weight footprint halves; 0.7 leaves enough headroom) |

  Symptom of an undersized fraction: server boots, loads all weight shards,
  then dies during `init_memory_pool` with `RuntimeError: Not enough memory`
  and `kv_budget=0.000GiB` in the ParaS GPT-OSS KV budget log line.
- **`--chunked-prefill-size -1` is baked into both launch scripts** (and into
  `ServerArgs._check_paras_config` as a fail-fast assertion). ParaS migration
  cannot preserve mid-chunked-prefill state — `chunked_req` is not part of
  the gather/scatter set.
- **`--disable-radix-cache` is required by ParaS init** (fail-fast asserts it).
  ParaS does not use prefix sharing, and ChunkCache / SWAChunkCache are the
  supported tree-cache backends. The launch scripts pass this flag automatically
  under `ENABLE_PARAS=1`.
- **Overlap scheduler is now supported under paras** (as of commit `adacebf22`,
  which ported the auto-switch observe/signal hook to `event_loop_overlap`;
  `SchedulerParasMixin._paras_drain_overlap_pipeline` drains the in-flight
  overlap batch before every EP↔TP switch). Default is overlap-ON
  (`DISABLE_OVERLAP=0`); set `DISABLE_OVERLAP=1` only to reproduce the legacy
  no-overlap path for forensics.
- **Wording differences EP vs TP**: BF16 floating-point differences cause
  minor sampling divergence at `temperature=0`. Content/meaning must be
  equivalent across phases; exact tokens may not.
- **Slow timing (~3 s instead of ~100 ms)**: GPU likely in bad state from a
  prior OOM. Kill all processes, wait 5 s, retry on a clean GPU.
- **Profiler overhead**: if `transfer_weights` > 500 ms, check that
  `paras_start_profile` / `paras_stop_profile` are not active in
  `scheduler_paras_mixin.py` and `paras_memory_check` is not called in
  `model_runner.py`.
- **PARAS_CONFIGURE_METHOD=peer_access is the canonical default**. To test
  the legacy `naive` (NCCL all-to-all) path, set `PARAS_CONFIGURE_METHOD=naive
  PARAS_KV_TRANSFER_METHOD=naive` on the launch line.

### gpt-oss-specific notes

- **Test harness uses `/v1/chat/completions`** (not `/v1/completions`). gpt-oss
  is a reasoning model that requires the harmony chat template to emit
  `<|return|>` (id 200002) as its EOS. With raw `/v1/completions`, gpt-oss
  is OOD: it never produces `<|return|>`, runs to `max_tokens`, and the tail
  decays into chat-completion overrun loops that the regex falsely flags as
  ParaS bugs. `lib.sh` (paras_cmd_send_completion / paras_cmd_burst_send)
  was switched to `/v1/chat/completions` and reads the response from
  `choices[0].message.reasoning_content + choices[0].message.content` (in
  generation order). The default reasoning effort is `medium` (per
  `chat_template.jinja`), which means at `max_tokens=200` the model often
  cuts off mid-`<|channel|>analysis` rather than reaching the final
  channel — that's intentional and OK for the regex check (cut-off by
  length without a single-word loop is CLEAN by design).
- **Use the unsloth BF16 weights**, not `/data/shaoyuw/models/gpt-oss-120b-bf16`
  (older, mislabeled — its MoE expert weights are stored in `float8_e5m2`,
  forcing a slow per-tensor FP8→BF16 conversion). The unsloth release stores
  every tensor in BF16 (verified: `experts.gate_up_proj.dtype == torch.bfloat16`).
  73 shards, 218 GB on disk. The launch script defaults to this path.
- **Hybrid attention budget**: gpt-oss layers alternate full / sliding-window.
  KV budget split is controlled by `--swa-full-tokens-ratio` (default 0.8).
  At `--mem-fraction-static 0.7` on 4×A100-80GB, expect roughly
  `full_layer_tokens=63987, swa_layer_tokens=51189` (51189/63987 ≈ 0.8).
- **Per-mode CUDA graph state preservation**: dual capture is the canonical
  gpt-oss config. The hooks (`paras_save_cuda_graph_state` /
  `paras_load_cuda_graph_state` on `AttentionBackend`) raise
  `NotImplementedError` if a future backend forgets to override; that
  surfaces as a clean error during dual capture rather than silent garbage at
  first replay.
- **MoE biases are not transferred** during the switch. `_full_w13_bias` and
  `_full_w2_bias` are loaded full on every rank at init time and exposed via
  Parameter views that get rebound on switch (see
  `paras_finalize_moe_bias_views` in `python/sglang/srt/paras/layers/paras_moe_block.py`).

## Known Failure Modes

1. **`TypeError: NoneType - int` after configure_tp**:
   `scheduler_paras_mixin.paras_configure_helper` is missing
   `max_queued_requests` in the tuple unpacking from `get_worker_info()`.
2. **`RuntimeError: shape '[N, 2048]' is invalid`**: FusedMoE
   `no_combine=True` is set on tp_experts. Check that `FusedMoE.__init__`
   skips `no_combine=True` when `paras_force_standard_dispatcher=True`.
3. **Decode degenerates to `\xa0` (qwen3)**: FlashInfer updaters have stale
   `req_to_token` / `num_kv_heads`. Check that
   `FlashInferAttnBackend.paras_configure_tp()` is called from
   `model_runner.paras_configure_tp()`.
4. **`KeyError: "No reservation named '...w13_weight_bias'"`** in
   `test_paras_gpt_oss_cuda_graph.py`: pre-existing test bug; commit
   `f9baf8a` removed UMM bias entries (biases moved to per-rank `_full_*_bias`
   tensors) but the unit test wasn't updated. The NCCL transport itself
   works at the e2e level.
5. **`CUDA error: Invalid access of peer GPU memory over nvlink`** at first
   switch on A100 with `PARAS_CONFIGURE_METHOD=naive` (legacy): use the
   default `peer_access` path.
6. **`OutOfMemoryError` during dual capture (gpt-oss)**: a regression has
   re-introduced a pre-grow somewhere, or `--cuda-graph-max-bs` is too high.
   With the current auto-sized defaults (`--cuda-graph-max-bs 256
   --max-running-requests 2048 --mem-fraction-static 0.75-0.8`) dual capture
   fits in 80 GB. To shrink the capture footprint for forensics, set
   `CUDA_GRAPH_MAX_BS=8 MAX_RUNNING_REQUESTS=1024` on the launch line.
7. **`NotImplementedError: <Backend> does not implement
   paras_save_cuda_graph_state`** during dual capture: the configured
   attention backend (other than Triton or FlashInfer) was selected with
   `--enable-paras-moe`. Switch to a supported backend or implement the two
   hooks on the new backend (`triton_backend.py` is the reference).
8. **Decode garbage like `Photos. 1990. 1990. 1990.` (gpt-oss)**:
   pre-redesign Bug 6 symptom — captured EP graph reading freed `kv_indptr`
   memory after dual capture reallocated it. Confirm HEAD is at or after
   `098b8a37a` (per-mode state preservation).
9. **Single-word loop tails like `It It It It...` or `The The The...`
   (gpt-oss)**: NOT a ParaS bug — `/v1/completions` endpoint artifact. The
   model is OOD without the harmony chat template, never emits `<|return|>`,
   runs to `max_tokens`, tail decays into chat-completion overrun loops that
   the verify regex flags. Fix: ensure `lib.sh` is using
   `/v1/chat/completions` (it does as of commit `099705271`). If you see
   this on a fresh clone, verify by `grep '/v1/chat/completions' lib.sh`
   and `grep '/v1/completions' lib.sh` — only the chat URL should appear.

## Companion Skills

- [`paras-test-auto-switch`](file:///home/shaoyuw/sglang/.skills/paras-test-auto-switch/SKILL.md) —
  load-driven autoswitch policy test. Run after this skill passes to verify
  the autoswitch trigger path independently.
- [`paras-test-peer-access`](file:///home/shaoyuw/sglang/.skills/paras-test-peer-access/SKILL.md) —
  unit tests for KV cache transfer + weight transfer + request partition
  in isolation (no e2e inference required).

## See Also

- [`scripts/paras/eval/paras_cmd/`](file:///home/shaoyuw/sglang/scripts/paras/eval/paras_cmd/) —
  every helper script. Each script's header documents its env-var overrides.
- [`scripts/paras/eval/paras_cmd/lib.sh`](file:///home/shaoyuw/sglang/scripts/paras/eval/paras_cmd/lib.sh) —
  shared helpers: `paras_cmd_load_prompts`, `paras_cmd_burst_send`,
  `paras_cmd_burst_verify`, `paras_cmd_build_payload`,
  `paras_cmd_send_completion`, `paras_cmd_print_completion_file`.
- [`scripts/paras/eval/paras_cmd/prompts_diverse.txt`](file:///home/shaoyuw/sglang/scripts/paras/eval/paras_cmd/prompts_diverse.txt) —
  32 distinct technical prompts used by `send_prompts.sh` and
  `inflight_switch.sh`. Replaces the older 3-fixed-prompts set. (Note: the
  previously documented "~37% deterministic degeneration on `Topic N`" is
  now resolved via the `/v1/chat/completions` switch — see "gpt-oss-specific
  notes" above. The prompts_diverse.txt set is still the recommended default
  for diversity.)
- `docs/paras/gpt_oss_support.md` — full gpt-oss design + Bug chronicle.
- [`test/srt/paras/test_swa_cache_transfer_metadata.py`](file:///home/shaoyuw/sglang/test/srt/paras/test_swa_cache_transfer_metadata.py) —
  unit-level regression gate for the SWA cache transfer source/destination
  invariants on both NCCL and peer-access paths. Run via
  `pytest test/srt/paras/test_swa_cache_transfer_metadata.py` (4 tests, ~2s).
  Must be 4/4 PASS before any PR touching `cache_transfer/swa.py`,
  `gather_manager.py`, or `scatter_manager.py`.
- `test/srt/paras/` — Python unit tests driven by `run_paras_tests.sh`.
