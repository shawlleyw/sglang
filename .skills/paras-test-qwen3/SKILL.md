---
name: paras-test-qwen3
description: Test ParaS EP↔TP configure on Qwen3-30B-A3B with 4 GPUs. Use to verify ParaS works after code changes.
---

# Test ParaS with Qwen3-30B-A3B

## Prerequisites

- **Conda env**: `sgl_paras`
- **GPUs**: 4x A100-80GB (use either CUDA_VISIBLE_DEVICES=0,1,2,3 or 4,5,6,7)
- **Model**: `/data/shaoyuw/models/Qwen3-30B-A3B`
- **Working dir**: `/home/shaoyuw/sglang`

## Test Procedure

### 1. Kill any existing sglang processes

```bash
pkill -9 -f "sglang" 2>/dev/null; sleep 3; rm -f /tmp/sglang_paras_test.log
```

### 2. Install and launch server

Use tmux for the server process. Log to a file for inspection.

```bash
conda activate sgl_paras
cd /home/shaoyuw/sglang
pip install -e python/ -q --no-deps

CUDA_VISIBLE_DEVICES=0,1,2,3 \
SGLANG_ATTN_MAX_BS=256 \
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256 \
SGLANG_DEEPEP_BF16_DISPATCH=true \
python -m sglang.launch_server \
    --model /data/shaoyuw/models/Qwen3-30B-A3B --trust-remote-code \
    --chunked-prefill-size -1 --max-prefill-tokens 32000 \
    --mem-fraction-static 0.6 \
    --tp-size 4 --dp-size 4 --ep-size 4 \
    --enable-dp-attention --enable-dp-lm-head \
    --moe-a2a-backend deepep --deepep-mode auto \
    --max-running-requests 1024 \
    --disable-cuda-graph --disable-overlap-schedule \
    --log-level info \
    --enable-paras-moe --paras-tp-size 4 \
    2>&1 | tee /tmp/sglang_paras_test.log
```

### 3. Wait for server ready

Poll until "Application startup complete" appears in the log. Typically ~35-40 seconds.

```bash
for i in $(seq 1 24); do
    sleep 5
    if grep -q "Application startup complete" /tmp/sglang_paras_test.log 2>/dev/null; then
        echo "READY after ${i}x5s"; break
    fi
    echo "Waiting ${i}/24: $(tail -1 /tmp/sglang_paras_test.log 2>/dev/null | cut -c1-80)"
done
```

### 4. Verify server is up and model type is correct

```bash
curl -s --max-time 5 http://localhost:30000/health
# Should return 200 (empty body)

grep "Load weight end" /tmp/sglang_paras_test.log | head -1
# Should show type=Qwen3MoeForCausalLMParaS (not Qwen3MoeForCausalLM)
```

### 5. Send requests in EP mode (before ParaS switch)

Use longer prompts (200 tokens) to stress-test decode correctness — short responses can appear correct even with partially corrupted weights.

```bash
# Prompt 1: code generation — tests multi-step coherent reasoning
curl -s --max-time 60 http://localhost:30000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen3-30B-A3B","prompt":"Write a Python function that implements binary search on a sorted list, with docstring and edge case handling.","max_tokens":200,"temperature":0}' \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print('[EP P1]', d['choices'][0]['text'][:300])"

# Prompt 2: step-by-step math — tests chain-of-thought
curl -s --max-time 60 http://localhost:30000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen3-30B-A3B","prompt":"A train leaves city A at 60mph. Another train leaves city B (300 miles away) at 80mph heading toward A. When do they meet? Show your work step by step.","max_tokens":150,"temperature":0}' \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print('[EP P2]', d['choices'][0]['text'][:300])"

# Prompt 3: structured explanation — stress-tests long decode
curl -s --max-time 60 http://localhost:30000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen3-30B-A3B","prompt":"Explain the difference between TCP and UDP protocols. Include: connection setup, reliability, use cases.","max_tokens":200,"temperature":0}' \
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
grep "Time taken to configure TP\|transfer_weights" /tmp/sglang_paras_test.log
```

**Baseline performance** (mem-fraction-static=0.6, 4×A100-80GB, measured 2026-04-10):

| Metric | Baseline | Pass threshold |
|--------|----------|----------------|
| `transfer_weights` | ~106ms | < 300ms |
| `configure TP` total | ~122–132ms | < 400ms |

### 8. Send same requests in TP mode (after switch)

```bash
# Prompt 1: binary search
curl -s --max-time 60 http://localhost:30000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen3-30B-A3B","prompt":"Write a Python function that implements binary search on a sorted list, with docstring and edge case handling.","max_tokens":200,"temperature":0}' \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print('[TP P1]', d['choices'][0]['text'][:300])"

# Prompt 2: train problem
curl -s --max-time 60 http://localhost:30000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen3-30B-A3B","prompt":"A train leaves city A at 60mph. Another train leaves city B (300 miles away) at 80mph heading toward A. When do they meet? Show your work step by step.","max_tokens":150,"temperature":0}' \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print('[TP P2]', d['choices'][0]['text'][:300])"

# Prompt 3: TCP vs UDP
curl -s --max-time 60 http://localhost:30000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen3-30B-A3B","prompt":"Explain the difference between TCP and UDP protocols. Include: connection setup, reliability, use cases.","max_tokens":200,"temperature":0}' \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print('[TP P3]', d['choices'][0]['text'][:300])"
```

**Expected**:
- P1: Coherent Python reasoning about binary search (may differ in wording from EP due to fp precision)
- P2: Correct math approach — relative speed 60+80=140mph, meets at 300/140 ≈ 2.14 hours
- P3: Coherent TCP/UDP explanation covering three-way handshake, reliability, use cases

**Critical check**: All responses must be coherent multi-token text with no degeneration (e.g., repeated `\xa0`, blank spaces, or topic drift). Degeneration after the first token = stale FlashInfer attention backend state.

### 9. Trigger ParaS TP→EP switch (round-trip)

```bash
curl -s --max-time 60 http://localhost:30000/paras_configure_ep
# Expected: "ParaS EP parallelism configured."
```

### 10. Send requests in EP mode (after round-trip)

```bash
curl -s --max-time 60 http://localhost:30000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen3-30B-A3B","prompt":"Write a Python function that implements binary search on a sorted list, with docstring and edge case handling.","max_tokens":200,"temperature":0}' \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print('[EP-RT P1]', d['choices'][0]['text'][:300])"
```

**Expected**: Coherent output, same quality as original EP mode. May have minor wording differences due to BF16 precision.

### 11. Test KV cache coherence: in-flight EP→TP switch

Server is in EP mode after step 10. Requests started in EP must survive the switch to TP.

```bash
# Start requests in EP mode (background)
curl -s --max-time 120 http://localhost:30000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen3-30B-A3B","prompt":"Explain how a hash table works, including collision resolution strategies like chaining and open addressing.","max_tokens":500,"temperature":0}' > /tmp/r1.json &
PID1=$!

curl -s --max-time 120 http://localhost:30000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen3-30B-A3B","prompt":"Describe the process of photosynthesis in plants, including the light-dependent and light-independent reactions.","max_tokens":500,"temperature":0}' > /tmp/r2.json &
PID2=$!

sleep 4

# Switch EP→TP mid-generation
curl -s --max-time 60 http://localhost:30000/paras_configure_tp

wait $PID1 $PID2

python3 -c "import json; d=json.load(open('/tmp/r1.json')); print('[EP→TP R1]', d['choices'][0]['text'][:300])"
python3 -c "import json; d=json.load(open('/tmp/r2.json')); print('[EP→TP R2]', d['choices'][0]['text'][:300])"
```

**Expected**: Both outputs are coherent. R1 should discuss hash functions, collision resolution. R2 should discuss light reactions, Calvin cycle, chloroplasts.

### 12. Test KV cache coherence: in-flight TP→EP switch

Server is in TP mode after step 11. Requests started in TP must survive the switch to EP.

```bash
# Start requests in TP mode (background)
curl -s --max-time 120 http://localhost:30000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen3-30B-A3B","prompt":"List the first 10 prime numbers and explain why each is prime.","max_tokens":500,"temperature":0}' > /tmp/r3.json &
PID3=$!

curl -s --max-time 120 http://localhost:30000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen3-30B-A3B","prompt":"Write a recursive Fibonacci function in Python with memoization.","max_tokens":500,"temperature":0}' > /tmp/r4.json &
PID4=$!

sleep 4

# Switch TP→EP mid-generation
curl -s --max-time 60 http://localhost:30000/paras_configure_ep

wait $PID3 $PID4

python3 -c "import json; d=json.load(open('/tmp/r3.json')); print('[TP→EP R3]', d['choices'][0]['text'][:300])"
python3 -c "import json; d=json.load(open('/tmp/r4.json')); print('[TP→EP R4]', d['choices'][0]['text'][:300])"
```

**Expected**: Both outputs are coherent multi-sentence text. No degeneration, no garbage, no repeated tokens.

### 13. Verify no errors

```bash
grep -i "error\|exception" /tmp/sglang_paras_test.log | grep -v "import error\|Config file\|opentelemetry\|WARNING"
# Expected: empty
```

### 14. Cleanup

```bash
pkill -9 -f "sglang" 2>/dev/null
rm -f /tmp/sglang_paras_test.log
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

The default procedure runs with `--disable-cuda-graph` for a clean eager path. To exercise the per-mode CUDA graph state preservation hooks (`paras_save_cuda_graph_state` / `paras_load_cuda_graph_state` on `AttentionBackend`) introduced for gpt-oss but applicable to qwen3+FlashInfer too, swap the launch flag:

```diff
-    --disable-cuda-graph --disable-overlap-schedule \
+    --cuda-graph-max-bs 8 --disable-overlap-schedule \
```

Then run the same 14-step procedure. After step 2, also verify dual capture happened:

```bash
grep -E "ParaS: dual capture complete|saving EP graphs|capturing TP graphs|TP capture done" /tmp/sglang_paras_test.log
# Should show 4 lines per phase (one per DP rank), ending in:
# "ParaS: dual capture complete avail=...GB  #EP graphs=4  #TP graphs=4"
# and "TP capture done (..., pools_differ=True, ...)" — pools_differ=True confirms each mode owns its own buffer set.
```

Pass criteria: same as eager run, plus `pools_differ=True` and `#EP graphs=4 #TP graphs=4` after dual capture.

## See Also

- `.skills/paras-test-gpt-oss/SKILL.md` — parallel test skill for gpt-oss-120b-bf16. The gpt-oss path always runs with CUDA graphs enabled and uses Triton attention; runs both eager (FlashInfer) and Triton+graph paths gives full coverage of the ParaS attention-backend matrix.

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
