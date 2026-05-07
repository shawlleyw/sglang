---
name: paras-test-qwen3
description: Test ParaS EP↔TP configure on Qwen3-30B-A3B with 4 GPUs. Use to verify ParaS works after code changes. Drives the canonical 14-step procedure via the helper scripts in scripts/paras/eval/paras_cmd/.
---

# Test ParaS with Qwen3-30B-A3B

## Prerequisites

- **Conda env**: `sgl_paras`
- **GPUs**: 4× A100-80GB (use either `CUDA_VISIBLE_DEVICES=0,1,2,3` or `4,5,6,7`)
- **Model**: `/data/shaoyuw/models/Qwen3-30B-A3B`
- **Working dir**: `/home/shaoyuw/sglang_paras_fp8`
- **Helper scripts**: [`scripts/paras/eval/paras_cmd/`](file:///home/shaoyuw/sglang_paras_fp8/scripts/paras/eval/paras_cmd) — every step below maps to a one-line invocation. See [`lib.sh`](file:///home/shaoyuw/sglang_paras_fp8/scripts/paras/eval/paras_cmd/lib.sh) for shared env-var overrides (`HOST`, `PORT`, `MODEL_NAME`, `LOG_FILE`, `PRINT_CHARS`, `INFLIGHT_DELAY`, `INFLIGHT_MAX_TOKENS`, `TIMEOUT_TRIES`, `SLEEP_BETWEEN`).

## Quick Start

The canonical four-call flow. Defaults baked into `lib.sh` (`MODEL_NAME=Qwen3-30B-A3B`, `LOG_FILE=/tmp/sglang_paras_test.log`, `HOST=127.0.0.1`, `PORT=30000`) match the qwen3 launch script — no env overrides required.

```bash
conda activate sgl_paras
cd /home/shaoyuw/sglang_paras_fp8
pip install -e python/ -q --no-deps

bash scripts/paras/eval/paras_cmd/kill.sh                       # step 1

ENABLE_PARAS=1 NUM_GPUS=4 \
    bash scripts/paras/eval/a100/qwen/launch_server_dp_ep.sh \
    2>&1 | tee /tmp/sglang_paras_test.log &                     # step 2

bash scripts/paras/eval/paras_cmd/e2e_test.sh                   # steps 3-13

bash scripts/paras/eval/paras_cmd/kill.sh                       # step 14
```

`e2e_test.sh` orchestrates steps 3–13 in order and stops on the first non-zero return. Run individual steps below for partial reruns or debugging.

## Detailed Procedure

### 1. Kill any existing sglang processes

`kill.sh` does `pkill -9 -f sglang`, sleeps 3s, removes `$LOG_FILE`.

```bash
bash scripts/paras/eval/paras_cmd/kill.sh
```

### 2. Install and launch server

The launch is intentionally **not** wrapped by `e2e_test.sh` — the launch script and log path are model-specific, and the launch is long-lived (must be backgrounded or run in tmux).

```bash
conda activate sgl_paras
cd /home/shaoyuw/sglang_paras_fp8
pip install -e python/ -q --no-deps

ENABLE_PARAS=1 NUM_GPUS=4 \
    bash scripts/paras/eval/a100/qwen/launch_server_dp_ep.sh \
    2>&1 | tee /tmp/sglang_paras_test.log
```

The launch script ([`scripts/paras/eval/a100/qwen/launch_server_dp_ep.sh`](file:///home/shaoyuw/sglang_paras_fp8/scripts/paras/eval/a100/qwen/launch_server_dp_ep.sh)) detects `ENABLE_PARAS=1` and shifts the canonical defaults: `MEM_FRACTION_STATIC=0.6`, `MAX_RUNNING_REQUESTS=1024`, `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256`, `SGLANG_ATTN_MAX_BS=256`, `PARAS_CONFIGURE_METHOD=peer_access`, `ENABLE_CUDA_GRAPH=0` (eager, qwen ParaS canonical). It also adds `--enable-paras-moe --paras-tp-size $NUM_GPUS --disable-overlap-schedule --chunked-prefill-size -1 --max-prefill-tokens 32000`. `CUDA_VISIBLE_DEVICES` defaults to `0,1,2,3` for `NUM_GPUS=4`. Override any of these by setting the env var on the same line.

### 3. Wait for server ready

`wait_ready.sh` polls `$LOG_FILE` for "Application startup complete" — typically ~35–40s for qwen3. Default ceiling is 24×5s = 120s.

```bash
bash scripts/paras/eval/paras_cmd/wait_ready.sh
```

### 4. Verify server is up and model type is correct

`health.sh` checks HTTP /health is 200 and parses the model type from `Load weight end ... type=...` in the log. Must show `Qwen3MoeForCausalLMParaS` (NOT `Qwen3MoeForCausalLM`).

```bash
bash scripts/paras/eval/paras_cmd/health.sh
```

### 5. Send requests in EP mode (before ParaS switch)

`send_prompts.sh <LABEL>` sends three canonical 200/150/200-token prompts (binary search / train math / TCP-vs-UDP) sequentially. Long prompts stress-test multi-step reasoning — short responses can mask partially corrupted weights post-switch.

```bash
bash scripts/paras/eval/paras_cmd/send_prompts.sh EP
```

**Expected**: All three responses are coherent multi-sentence text with no repetition or garbage. Save for comparison.

### 6. Trigger ParaS EP→TP switch

`configure.sh tp` hits `/paras_configure_tp` and prints elapsed ms + the server's response.

```bash
bash scripts/paras/eval/paras_cmd/configure.sh tp
# Expected response: "ParaS TP parallelism configured."
```

### 7. Check timing

`check_log.sh timing` greps `transfer_weights` and `Time taken to configure {TP,EP}` from the log.

```bash
bash scripts/paras/eval/paras_cmd/check_log.sh timing
```

**Baseline performance** (mem-fraction-static=0.6, 4×A100-80GB, measured 2026-04-10):

| Metric | Baseline | Pass threshold |
|--------|----------|----------------|
| `transfer_weights` | ~106ms | < 300ms |
| `configure TP` total | ~122–132ms | < 400ms |

### 8. Send same requests in TP mode (after switch)

```bash
bash scripts/paras/eval/paras_cmd/send_prompts.sh TP
```

**Expected**:
- P1: Coherent Python reasoning about binary search (may differ in wording from EP due to fp precision)
- P2: Correct math approach — relative speed 60+80=140mph, meets at 300/140 ≈ 2.14 hours
- P3: Coherent TCP/UDP explanation covering three-way handshake, reliability, use cases

**Critical check**: All responses must be coherent multi-token text with no degeneration (e.g., repeated `\xa0`, blank spaces, or topic drift). Degeneration after the first token = stale FlashInfer attention backend state.

### 9. Trigger ParaS TP→EP switch (round-trip)

```bash
bash scripts/paras/eval/paras_cmd/configure.sh ep
# Expected response: "ParaS EP parallelism configured."
```

### 10. Send requests in EP mode (after round-trip)

```bash
bash scripts/paras/eval/paras_cmd/send_prompts.sh EP-RT
```

**Expected**: Coherent output, same quality as original EP mode. May have minor wording differences due to BF16 precision.

### 11. Test KV cache coherence: in-flight EP→TP switch

`inflight_switch.sh tp` kicks off two concurrent 500-token completions (hash table, photosynthesis), waits `INFLIGHT_DELAY=4s` for them to enter decode, then triggers `/paras_configure_tp`. Both responses must survive the switch.

```bash
bash scripts/paras/eval/paras_cmd/inflight_switch.sh tp
```

**Expected**: Both outputs are coherent. R1 should discuss hash functions and collision resolution. R2 should discuss light reactions, Calvin cycle, chloroplasts.

### 12. Test KV cache coherence: in-flight TP→EP switch

Same shape, different prompts (primes, recursive fibonacci with memoization).

```bash
bash scripts/paras/eval/paras_cmd/inflight_switch.sh ep
```

**Expected**: Both outputs are coherent multi-sentence text. No degeneration, no garbage, no repeated tokens.

### 13. Verify no errors

`check_log.sh errors` greps for `error|exception` and filters out known benign warnings (`opentelemetry`, `WARNING`, `UserWarning`, `warn_only`, `Config file`, `import error`).

```bash
bash scripts/paras/eval/paras_cmd/check_log.sh errors
# OK: no unexpected errors
```

### 14. Cleanup

```bash
bash scripts/paras/eval/paras_cmd/kill.sh
```

## Pass/Fail Criteria

| Check | Pass | Fail |
|-------|------|------|
| Model type | `Qwen3MoeForCausalLMParaS` | `Qwen3MoeForCausalLM` |
| EP requests (3 prompts) | All coherent, 150-200 tokens | Error, timeout, or garbage |
| EP→TP switch | Returns in < 300ms | Timeout or OOM |
| TP requests (3 prompts) | Coherent, same quality as EP | Garbage, `\xa0`, or repetition |
| TP→EP switch | Returns in < 300ms | Timeout or error |
| EP requests after round-trip | **Identical** to original EP output | Different output or garbage |
| In-flight EP→TP coherence | Both requests coherent after switch | Degeneration or crash |
| In-flight TP→EP coherence | Both requests coherent after switch | Degeneration or crash |
| Server errors | None | Any scheduler/runtime exception |

## Optional: CUDA-Graph-Enabled Variant

The default procedure runs with `ENABLE_CUDA_GRAPH=0` (eager) for a clean path. To exercise the per-mode CUDA graph state preservation hooks (`paras_save_cuda_graph_state` / `paras_load_cuda_graph_state` on `AttentionBackend`) introduced for gpt-oss but applicable to qwen3+FlashInfer too, override the cuda-graph defaults at launch:

```bash
ENABLE_PARAS=1 NUM_GPUS=4 ENABLE_CUDA_GRAPH=1 CUDA_GRAPH_MAX_BS=8 \
    bash scripts/paras/eval/a100/qwen/launch_server_dp_ep.sh \
    2>&1 | tee /tmp/sglang_paras_test.log
```

Then run `e2e_test.sh` as usual. After step 4, also verify dual capture happened:

```bash
bash scripts/paras/eval/paras_cmd/check_log.sh cuda_graph
# Should show 4 lines per phase (one per DP rank), ending in:
#   "ParaS: dual capture complete avail=...GB  #EP graphs=4  #TP graphs=4"
#   "TP capture done (..., pools_differ=True, ...)" — pools_differ=True confirms each mode owns its own buffer set.
```

Pass criteria: same as eager run, plus `pools_differ=True` and `#EP graphs=4 #TP graphs=4` after dual capture.

## See Also

- [`.skills/paras-test-gpt-oss/SKILL.md`](file:///home/shaoyuw/sglang_paras_fp8/.skills/paras-test-gpt-oss/SKILL.md) — parallel test skill for gpt-oss-120b-bf16. The gpt-oss path always runs with CUDA graphs enabled and uses Triton attention; running both eager (FlashInfer) and Triton+graph paths gives full coverage of the ParaS attention-backend matrix.
- [`scripts/paras/eval/README.md`](file:///home/shaoyuw/sglang_paras_fp8/scripts/paras/eval/README.md) — overview of every script under `scripts/paras/eval/`.
- [`scripts/paras/eval/paras_cmd/`](file:///home/shaoyuw/sglang_paras_fp8/scripts/paras/eval/paras_cmd) — per-step helper scripts and `e2e_test.sh` orchestrator. Each script's header comment documents its env-var overrides.

## Important Notes

- **mem-fraction-static=0.75 will OOM** during weight redistribution on A100-80GB. Use 0.6.
- **ParaS configure supports round-trip** (EP→TP→EP→TP...). Use `/paras_configure_tp` and `/paras_configure_ep` endpoints. For the naive weight transfer method, set `PARAS_CONFIGURE_METHOD=naive` (default is `peer_access`).
- **Overlap mode**: To test overlapped conversion, modify `paras/models/qwen3_moe.py` to pass `overlap=True` to `self.model.paras_configure_tp(...)`.
- **Slow timing (~3s instead of ~100ms)**: Likely GPU in bad state from prior OOM. Kill all processes, wait 5 seconds, retry on clean GPU.
- **Profiler overhead**: If `transfer_weights` > 500ms, check that `paras_start_profile`/`paras_stop_profile` are not called in `scheduler_paras_mixin.py` and `paras_memory_check` is not called in `model_runner.py`.
- **Wording differences EP vs TP**: Normal. BF16 floating point differences cause minor sampling divergence at `temperature=0`. The content/meaning must be equivalent, not the exact words.
- **CUDA graph state preservation**: Both FlashInfer and Triton implement the `paras_save_cuda_graph_state` / `paras_load_cuda_graph_state` hooks on `AttentionBackend`. If a future backend forgets to override these, `paras_init_dual_cuda_graphs` raises `NotImplementedError` immediately on first switch. See `docs/paras/gpt_oss_support.md` Bug 5/6 sections for the design rationale.

## Known Failure Modes

1. **`TypeError: NoneType - int`** after configure_tp: `scheduler_paras_mixin.paras_configure_helper` is missing `max_queued_requests` in the tuple unpacking from `get_worker_info()`.
2. **`RuntimeError: shape '[N, 2048]' is invalid`**: FusedMoE `no_combine=True` is set on tp_experts. Check that `FusedMoE.__init__` skips `no_combine=True` when `paras_force_standard_dispatcher=True`.
3. **Decode degenerates to `\xa0`**: FlashInfer updaters have stale `req_to_token` / `num_kv_heads`. Check that `FlashInferAttnBackend.paras_configure_tp()` is called from `model_runner.paras_configure_tp()`.
4. **`NotImplementedError: <Backend> does not implement paras_save_cuda_graph_state`** during dual capture: a non-Triton/non-FlashInfer attention backend was selected with `--enable-paras-moe`. Either switch to a supported backend or implement the two hooks on the new backend (see `python/sglang/srt/layers/attention/triton_backend.py` for a reference implementation).
