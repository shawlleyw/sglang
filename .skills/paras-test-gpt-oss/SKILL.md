---
name: paras-test-gpt-oss
description: Test ParaS EP↔TP configure on gpt-oss-120b-bf16 with 4 GPUs. Use to verify ParaS works after code changes — exercises the Triton attention backend, hybrid (full+SWA) attention, fused MoE checkpoints, and CUDA-graph dual capture path.
---

# Test ParaS with gpt-oss-120b-bf16

## Why This Differs From paras-test-qwen3

| Concern | qwen3-30B-A3B | gpt-oss-120b-bf16 |
|---|---|---|
| Attention backend | FlashInfer (default) | **Triton** (required by gpt-oss SWA + sinks) |
| CUDA graph | Disabled in default test | **Enabled** (`--cuda-graph-max-bs 8`); exercises `paras_save_cuda_graph_state` / `paras_load_cuda_graph_state` per-mode preservation |
| Attention type | Dense full attention | **Hybrid** full + sliding-window per layer |
| MoE biases | None | Per-expert `w13_weight_bias` / `w2_weight_bias` (replicated, see `gpt_oss_support.md`) |
| Attention sinks | None | Per-head `sinks` parameter |
| Checkpoint layout | Per-projection BF16 | Fused `gate_up_proj` / `down_proj`, interleaved w13 |
| Weight transfer method | peer_access (default) | **naive** (`PARAS_CONFIGURE_METHOD=naive`); peer_access kernel does not yet carry biases |

The gpt-oss test thus covers a different code matrix from qwen3 — both should be run when touching shared ParaS code.

## Prerequisites

- **Conda env**: `sgl_paras`
- **GPUs**: 4× A100-80GB (use `CUDA_VISIBLE_DEVICES=0,1,2,3` or `4,5,6,7`)
- **Model**: `/data/shaoyuw/models/gpt-oss-120b-bf16`
- **Working dir**: `/home/shaoyuw/sglang`

## Test Procedure

### 1. Kill any existing sglang processes

```bash
pkill -9 -f "sglang" 2>/dev/null; sleep 5; rm -f /tmp/sglang_paras_gptoss.log
```

### 2. Install and launch server

Use tmux for the server process. Log to a file for inspection.

```bash
conda activate sgl_paras
cd /home/shaoyuw/sglang
pip install -e python/ -q --no-deps

CUDA_VISIBLE_DEVICES=0,1,2,3 \
PARAS_CONFIGURE_METHOD=naive \
PARAS_KV_TRANSFER_METHOD=nccl \
SGLANG_DEEPEP_BF16_DISPATCH=true \
SGLANG_ATTN_MAX_BS=256 \
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256 \
python -m sglang.launch_server \
    --model-path /data/shaoyuw/models/gpt-oss-120b-bf16 --trust-remote-code \
    --tp-size 4 --dp-size 4 --ep-size 4 \
    --enable-dp-attention --enable-dp-lm-head \
    --moe-a2a-backend deepep --deepep-mode auto \
    --enable-paras-moe --paras-tp-size 4 \
    --attention-backend triton \
    --max-running-requests 1024 \
    --cuda-graph-max-bs 8 \
    --mem-fraction-static 0.8 \
    --disable-overlap-schedule \
    --host 0.0.0.0 --port 30000 \
    2>&1 | tee /tmp/sglang_paras_gptoss.log
```

Note: `--max-running-requests 1024` is the canonical value. Pre-redesign (commits before `098b8a37a`) this required `256` to dodge OOM during dual capture; the state-preservation redesign recovered ~3 GB and removed that workaround.

### 3. Wait for server ready

Polling typically takes ~3–4 minutes (model weights ~63 GB, dual capture, JIT compile of MoE + attention kernels).

```bash
for i in $(seq 1 60); do
    sleep 10
    STATUS=$(curl -s --max-time 3 http://localhost:30000/health -o /dev/null -w "%{http_code}" 2>/dev/null)
    if [ "$STATUS" = "200" ]; then
        echo "READY after ${i}x10s"; break
    fi
    TAIL=$(tail -1 /tmp/sglang_paras_gptoss.log 2>/dev/null | cut -c1-100)
    if echo "$TAIL" | grep -qE "OutOfMemoryError|CUDA error|Traceback|RuntimeError"; then
        echo "ERROR: $TAIL"; break
    fi
    echo "[$i/60] HTTP=$STATUS - $TAIL"
done
```

### 4. Verify server is up, model type, and dual capture happened

```bash
# HTTP health
curl -s --max-time 5 http://localhost:30000/health -o /dev/null -w "HTTP %{http_code}\n"
# Should return 200

# Model type
grep "Load weight end" /tmp/sglang_paras_gptoss.log | head -1
# Should show type=GptOssForCausalLMParaS (not GptOssForCausalLM)

# Dual CUDA graph capture
grep -E "ParaS: dual capture complete|saving EP graphs|capturing TP graphs|TP capture done" /tmp/sglang_paras_gptoss.log
# Should show 4 lines per phase (one per DP rank), ending in:
#   "ParaS: dual capture complete avail=...GB  #EP graphs=4  #TP graphs=4"
# and "TP capture done (..., pools_differ=True, ...)" — pools_differ=True confirms each mode owns its own buffer set.
```

### 5. Send requests in EP mode (before ParaS switch)

```bash
# Prompt 1: Roman history — long-context narrative
curl -s --max-time 60 http://localhost:30000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"gpt-oss-120b-bf16","prompt":"Tell me about the history of Rome. The Roman Empire","max_tokens":200,"temperature":0}' \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print('[EP P1]', d['choices'][0]['text'][:300])"

# Prompt 2: data structures — technical multi-step
curl -s --max-time 60 http://localhost:30000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"gpt-oss-120b-bf16","prompt":"Explain how a hash table works.","max_tokens":200,"temperature":0}' \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print('[EP P2]', d['choices'][0]['text'][:300])"

# Prompt 3: code generation — chain-of-thought
curl -s --max-time 60 http://localhost:30000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"gpt-oss-120b-bf16","prompt":"Write a Python recursive Fibonacci function.","max_tokens":200,"temperature":0}' \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print('[EP P3]', d['choices'][0]['text'][:300])"
```

**Expected**: All three responses are coherent multi-sentence text with no repetition or garbage. Save for comparison.

### 6. Trigger ParaS EP→TP switch

```bash
curl -s --max-time 10 http://localhost:30000/paras_configure_tp
# Expected: "ParaS TP parallelism configured."
```

### 7. Check timing

```bash
grep -E "Time taken to configure TP|transfer_weights|gather_cache" /tmp/sglang_paras_gptoss.log | tail -10
```

**Baseline performance** (4×A100-80GB, naive transfer, NCCL KV, measured 2026-04-26):

| Metric | Baseline | Pass threshold |
|---|---|---|
| `transfer_weights` (naive) | ~600–900 ms | < 2000 ms |
| `gather_cache` (no in-flight) | ~10 ms | < 100 ms |
| `configure TP` total | ~700 ms–1 s | < 2500 ms |

Naive transfer is significantly slower than peer_access (which qwen3 uses by default). This is expected — peer_access for gpt-oss is unimplemented (the fused NVLink kernel does not yet carry biases; see `paras_moe_block.py:489-497`).

### 8. Send same requests in TP mode (after switch)

```bash
curl -s --max-time 60 http://localhost:30000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"gpt-oss-120b-bf16","prompt":"Tell me about the history of Rome. The Roman Empire","max_tokens":200,"temperature":0}' \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print('[TP P1]', d['choices'][0]['text'][:300])"

curl -s --max-time 60 http://localhost:30000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"gpt-oss-120b-bf16","prompt":"Explain how a hash table works.","max_tokens":200,"temperature":0}' \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print('[TP P2]', d['choices'][0]['text'][:300])"

curl -s --max-time 60 http://localhost:30000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"gpt-oss-120b-bf16","prompt":"Write a Python recursive Fibonacci function.","max_tokens":200,"temperature":0}' \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print('[TP P3]', d['choices'][0]['text'][:300])"
```

**Expected**:
- P1: Coherent narrative about Augustus, Roman Empire timeline
- P2: Technical explanation of hash table (separate chaining, open addressing, time complexities)
- P3: Recursive Python Fibonacci function with base cases

**Critical check**: All responses must be coherent multi-token text with no degeneration (repeated tokens, garbage like `1990. 1990. 1990.`, or topic drift). The distinct gpt-oss failure mode the redesign closed was a Bug 6 silent garbage / out-of-bounds index from stale captured `kv_indptr` after dual capture.

### 9. Trigger ParaS TP→EP switch (round-trip)

```bash
curl -s --max-time 60 http://localhost:30000/paras_configure_ep
# Expected: "ParaS EP parallelism configured."
```

### 10. Send a request in EP mode (after round-trip)

```bash
curl -s --max-time 60 http://localhost:30000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"gpt-oss-120b-bf16","prompt":"Tell me about the history of Rome. The Roman Empire","max_tokens":200,"temperature":0}' \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print('[EP-RT P1]', d['choices'][0]['text'][:300])"
```

**Expected**: Coherent output, same quality as original EP mode. May have minor wording differences due to BF16 precision and CUDA graph autotune choosing slightly different configs across captures.

### 11. Test KV cache coherence: in-flight EP→TP switch

Server is in EP mode after step 10. Requests started in EP must survive the switch to TP.

```bash
# Start request in EP mode (background) with a long prompt
curl -s --max-time 180 http://localhost:30000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"gpt-oss-120b-bf16","prompt":"Tell me about the history of Rome. The Roman Empire","max_tokens":300,"temperature":0}' > /tmp/gptoss_r1.json &
PID1=$!

sleep 1

# Switch EP→TP mid-generation
curl -s --max-time 30 http://localhost:30000/paras_configure_tp

wait $PID1
python3 -c "import json; d=json.load(open('/tmp/gptoss_r1.json')); print('[EP→TP R1]', d['choices'][0]['text'][:400])"
```

**Expected**: Coherent narrative about Roman Empire continuing through the switch boundary.

### 12. Test KV cache coherence: in-flight TP→EP switch

Server is in TP mode after step 11.

```bash
curl -s --max-time 180 http://localhost:30000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"gpt-oss-120b-bf16","prompt":"Describe photosynthesis in plants step by step.","max_tokens":300,"temperature":0}' > /tmp/gptoss_r2.json &
PID2=$!

sleep 1

curl -s --max-time 30 http://localhost:30000/paras_configure_ep

wait $PID2
python3 -c "import json; d=json.load(open('/tmp/gptoss_r2.json')); print('[TP→EP R2]', d['choices'][0]['text'][:400])"
```

**Expected**: Coherent multi-paragraph explanation of photosynthesis (light-dependent reactions, Calvin cycle, ATP/NADPH).

### 13. Verify no errors

```bash
grep -iE "error|exception|traceback|assert" /tmp/sglang_paras_gptoss.log \
    | grep -v "WARNING server_args\|opentelemetry\|opted out\|FastAPIDeprecationWarning\|orjson\|Application startup\|already free\|Config file\|deprecated\|MoE kernel\|tokenizer"
# Expected: empty
```

### 14. Cleanup

```bash
pkill -9 -f "sglang" 2>/dev/null
rm -f /tmp/sglang_paras_gptoss.log /tmp/gptoss_r*.json
```

## Pass/Fail Criteria

| Check | Pass | Fail |
|---|---|---|
| Model type | `GptOssForCausalLMParaS` | `GptOssForCausalLM` |
| Dual capture log | `pools_differ=True`, `#EP graphs=4 #TP graphs=4` | Missing or `pools_differ=False` |
| EP requests (3 prompts) | All coherent, ~150–200 tokens | Error, timeout, or garbage |
| EP→TP switch | Returns in < 2.5 s (naive) | Timeout or OOM |
| TP requests (3 prompts) | Coherent, same quality as EP | Garbage, repeated tokens, topic drift |
| TP→EP switch | Returns in < 2.5 s | Timeout or error |
| EP requests after round-trip | Coherent (may differ in wording from original due to BF16) | Garbage or completely off-topic |
| In-flight EP→TP coherence | Request continues coherently | Degeneration into repetition or crash |
| In-flight TP→EP coherence | Request continues coherently | Degeneration into repetition or crash |
| Server errors | None | `AssertionError`, `CUDA error`, `NotImplementedError`, scheduler crash |

## Important Notes

- **`--max-running-requests 1024` is canonical**, NOT the older `256` workaround. The state-preservation redesign (commit `098b8a37a`) freed the ~3 GB of permanent GPU memory the pre-grow consumed. If you see OOM at 1024, suspect either a regression in the redesign or an unrelated change to memory accounting.
- **`--cuda-graph-max-bs 8` exercises the per-mode state preservation hooks**. The hooks (`paras_save_cuda_graph_state` / `paras_load_cuda_graph_state` on `AttentionBackend`) raise `NotImplementedError` if a future backend forgets to override; that surfaces as a clean error during dual capture rather than silent garbage at first replay.
- **`PARAS_CONFIGURE_METHOD=naive` is required for gpt-oss**. The fused NVLink peer-access kernel at `paras_moe_block.py:489-497` raises `NotImplementedError` because it does not carry biases, which gpt-oss requires.
- **`PARAS_DISABLE_PEER_ACCESS=1` is the implicit default** for gpt-oss on A100 (set in `python/sglang/srt/paras/models/gpt_oss.py:438-454`). The peer-access pre-init step interacted badly with NCCL on A100 (Bug 7 in `gpt_oss_support.md`); skipping it costs ~6 s of first-switch latency but avoids the crash.
- **Hybrid attention budget**: gpt-oss layers alternate full-attention and sliding-window. The KV budget split is controlled by `swa_full_tokens_ratio` (default 0.8). The scheduler treats full and sliding layers in a single memory pool (`disable_hybrid_swa_memory=True`).
- **CUDA graph round-trip determinism is degraded vs eager**. With cuda graph, EP→TP→EP outputs match for ~30 tokens then diverge into semantically equivalent but different completions. This is expected (Triton autotune picks different configs across captures); functional correctness is unaffected.

## Companion Unit Tests

For weight-transfer and CUDA graph correctness in isolation (no end-to-end inference required), run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m pytest \
    test/srt/paras/test_paras_gpt_oss_cuda_graph.py -v
```

This file covers:
- Class hierarchy (MRO of `GptOssForCausalLMParaS`)
- Dispatch table registration
- EP→TP→EP weight round-trip + CUDA graph replay
- Bias transfer correctness (NCCL + concat + interleaved layouts)
- W13 layout semantics (gpt-oss uses interleaved `[g0, u0, g1, u1, ...]`, qwen3 uses concat `[gate..., up...]`)

## Known Failure Modes

1. **`NotImplementedError: peer_access not supported for GPT-OSS biases`** at first switch: `PARAS_CONFIGURE_METHOD` is unset or set to `peer_access`. Set `PARAS_CONFIGURE_METHOD=naive` (or `overlap`).
2. **`CUDA error: Invalid access of peer GPU memory over nvlink`** at first switch on A100: `PARAS_DISABLE_PEER_ACCESS` was overridden to `0`. Restore the default.
3. **`OutOfMemoryError` during dual capture**: a regression has re-introduced a pre-grow somewhere, or `--cuda-graph-max-bs` is too high. The redesign verified `--cuda-graph-max-bs 8 --max-running-requests 1024 --mem-fraction-static 0.8` fits in 80 GB; larger graph batch sizes may not.
4. **`NotImplementedError: <Backend> does not implement paras_save_cuda_graph_state`** during dual capture: the configured attention backend (other than Triton or FlashInfer) was selected with `--enable-paras-moe`. Switch to `--attention-backend triton` (gpt-oss-required) or implement the two hooks on the new backend (`triton_backend.py:224-249` is the reference).
5. **Decode garbage like `Photos. 1990. 1990. 1990.`**: pre-redesign Bug 6 symptom — captured EP graph reading freed `kv_indptr` memory after dual capture reallocated it. Confirm HEAD is at or after `098b8a37a` (per-mode state preservation).
6. **`AssertionError: This request holds the node from another tree`** in `radix_cache.dec_lock_ref`: documented as a non-reproducing intermittent issue in `docs/paras/runs/2026-04-25-bug3-qwen3-tp-ep-non-reproduction.md`. If you hit it, capture the full traceback and the `metadata sizes` line from the same iteration before killing the server.

## See Also

- `.skills/paras-test-qwen3/SKILL.md` — parallel test for qwen3-30B-A3B (FlashInfer attention; also has an optional cuda-graph-enabled variant)
- `.skills/paras-test-peer-access/SKILL.md` — unit test skill for KV transfer + weight transfer + request partition
- `docs/paras/gpt_oss_support.md` — full gpt-oss design + Bug chronicle (Bug 5/6 sections explain why this skill exists in the form it does)
