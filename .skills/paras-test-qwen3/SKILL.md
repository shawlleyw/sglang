---
name: paras-test-qwen3
description: Test ParaS EP↔TP configure on Qwen3-30B-A3B with 4 GPUs. Use to verify ParaS works after code changes.
---

# Test ParaS with Qwen3-30B-A3B

## Prerequisites

- **Conda env**: `sgl_paras`
- **GPUs**: 4x A100-80GB (CUDA_VISIBLE_DEVICES=4,5,6,7)
- **Model**: `/data/shaoyuw/models/Qwen3-30B-A3B`
- **Working dir**: `/home/shaoyuw/sglang`

## Test Procedure

### 1. Kill any existing sglang processes

```bash
pkill -9 -f "sglang" 2>/dev/null; sleep 3
```

### 2. Launch server

Use tmux for the server process. Log to a file for inspection.

```bash
conda activate sgl_paras
cd /home/shaoyuw/sglang
pip install -e python/ -q --no-deps

CUDA_VISIBLE_DEVICES=4,5,6,7 \
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
for i in $(seq 1 20); do
    sleep 5
    if grep -q "Application startup complete" /tmp/sglang_paras_test.log 2>/dev/null; then
        echo "READY"; break
    fi
done
```

### 4. Verify server is up and model type is correct

```bash
curl -s --max-time 5 http://localhost:30000/health
# Should return 200 (empty body)

grep "Load weight end" /tmp/sglang_paras_test.log | head -1
# Should show type=Qwen3MoeForCausalLMParaS (not Qwen3MoeForCausalLM)
```

### 5. Send request in EP mode (before ParaS switch)

```bash
curl -s --max-time 30 http://localhost:30000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "Qwen3-30B-A3B", "prompt": "The capital of China is", "max_tokens": 50, "temperature": 0}'
```

**Expected**: Coherent response mentioning "Beijing". Save this output for comparison.

### 6. Trigger ParaS EP→TP switch

```bash
curl -s --max-time 10 http://localhost:30000/paras_configure_tp
```

**Expected**: Returns `ParaS TP parallelism configured.` within ~1 second.

### 7. Check timing

```bash
grep "Time taken to configure TP\|transfer_weights" /tmp/sglang_paras_test.log
```

**Expected performance** (mem-fraction-static=0.6):

| Metric | Expected |
|--------|----------|
| `transfer_weights` | < 300ms |
| `configure TP` total | < 400ms |

### 8. Send same request in TP mode (after ParaS switch)

```bash
curl -s --max-time 30 http://localhost:30000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "Qwen3-30B-A3B", "prompt": "The capital of China is", "max_tokens": 50, "temperature": 0}'
```

**Expected**: Coherent response mentioning "Beijing". The wording will differ from EP mode due to floating point precision differences, but the answer must be correct and readable.

### 9. Send additional requests to verify decode works

```bash
# Request 2: math
curl -s --max-time 30 http://localhost:30000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "Qwen3-30B-A3B", "prompt": "1+1=", "max_tokens": 20, "temperature": 0}'
# Expected: starts with "2"

# Request 3: code
curl -s --max-time 30 http://localhost:30000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "Qwen3-30B-A3B", "prompt": "Write a Python function to add two numbers:\ndef add(a, b):", "max_tokens": 30, "temperature": 0}'
# Expected: "return a + b" or equivalent
```

**Critical check**: All responses must be coherent multi-token text, NOT degenerated output (e.g., repeated `\xa0` or empty spaces). If decode degenerates after the first token, the FlashInfer attention backend state is stale — check that `paras_configure_tp` on the attention backend is being called.

### 10. Verify no errors

```bash
grep -i "error\|exception" /tmp/sglang_paras_test.log | grep -v "import error\|Config file"
# Expected: empty (no errors)
```

### 11. Cleanup

```bash
pkill -9 -f "sglang" 2>/dev/null
rm -f /tmp/sglang_paras_test.log
```

## Pass/Fail Criteria

| Check | Pass | Fail |
|-------|------|------|
| Model type | `Qwen3MoeForCausalLMParaS` | `Qwen3MoeForCausalLM` |
| EP request | Coherent response | Error or timeout |
| ParaS switch | Returns in < 1s | Timeout or OOM |
| `transfer_weights` | < 300ms | > 1000ms (profiler may be on) |
| TP request (same prompt) | Coherent, mentions "Beijing" | Garbage, `\xa0`, or timeout |
| TP decode (multi-token) | Readable continuation | Degenerates after first token |
| Additional TP requests | All coherent | Any garbage output |
| Server errors | None | Any scheduler/runtime exception |

## Important Notes

- **mem-fraction-static=0.75 will OOM** during weight redistribution on A100-80GB. Use 0.6.
- **ParaS configure is one-way** (EP→TP only). Once configured, you cannot call `/paras_configure_tp` again. Restart the server for a new test.
- **Overlap mode**: To test overlapped conversion (faster by ~30%), modify `paras/models/qwen3_moe.py` to pass `overlap=True` to `self.model.paras_configure_tp(...)`. Expected: ~200ms transfer_weights.
- **Slow timing (~3s instead of ~300ms)**: Likely GPU in bad state from prior OOM. Kill all processes, wait 5 seconds, retry on clean GPU.
- **Profiler overhead**: If `transfer_weights` > 500ms, check that `paras_start_profile`/`paras_stop_profile` are not called in `scheduler_paras_mixin.py` and `paras_memory_check` is not called in `model_runner.py`.

## Known Failure Modes

1. **`TypeError: NoneType - int`** after configure_tp: `scheduler_paras_mixin.paras_configure_helper` is missing `max_queued_requests` in the tuple unpacking from `get_worker_info()`.
2. **`RuntimeError: shape '[N, 2048]' is invalid`**: FusedMoE `no_combine=True` is set on tp_experts. Check that `FusedMoE.__init__` skips `no_combine=True` when `paras_force_standard_dispatcher=True`.
3. **Decode degenerates to `\xa0`**: FlashInfer updaters have stale `req_to_token` / `num_kv_heads`. Check that `FlashInferAttnBackend.paras_configure_tp()` is called from `model_runner.paras_configure_tp()`.
