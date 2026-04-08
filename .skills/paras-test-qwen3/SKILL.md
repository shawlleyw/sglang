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

### 4. Verify server is up

```bash
curl -s --max-time 5 http://localhost:30000/health
# Should return 200 (empty body)
```

### 5. Trigger ParaS EP→TP switch

```bash
curl -s --max-time 10 http://localhost:30000/paras_configure_tp
```

**Expected**: Returns `ParaS TP parallelism configured.` within ~1 second.

### 6. Check timing

```bash
grep "Time taken to configure TP\|transfer_weights" /tmp/sglang_paras_test.log
```

**Expected performance** (mem-fraction-static=0.6):

| Metric | Expected |
|--------|----------|
| `transfer_weights` | < 300ms |
| `configure TP` total | < 400ms |

### 7. Verify model type loaded

```bash
grep "Load weight end" /tmp/sglang_paras_test.log | head -1
```

Should show `type=Qwen3MoeForCausalLMParaS` (not `Qwen3MoeForCausalLM`).

### 8. Cleanup

```bash
pkill -9 -f "sglang" 2>/dev/null
rm -f /tmp/sglang_paras_test.log
```

## Important Notes

- **mem-fraction-static=0.75 will OOM** during weight redistribution (`permute(...).contiguous()` needs 192 MiB, only ~145 MiB free). This is a pre-existing issue (original branch also OOMs). Use 0.6.
- **ParaS configure is one-way** (EP→TP only). Once configured, you cannot call `/paras_configure_tp` again. Restart the server for a new test.
- **Overlap mode**: To test overlapped conversion (faster by ~30%), modify `paras/models/qwen3_moe.py` line 256 to pass `overlap=True` to `self.model.paras_configure_tp(...)`. Expected: ~200ms transfer_weights.
- **The 3s timing anomaly**: If you see ~3s instead of ~300ms, it's likely because the GPU was in a bad state from a previous OOM. Kill all processes, wait 5 seconds, and retry on a clean GPU.

## Quick One-Liner Test

For CI/quick verification after code changes:

```bash
pkill -9 -f sglang 2>/dev/null; sleep 3; \
cd /home/shaoyuw/sglang && conda activate sgl_paras && \
pip install -e python/ -q --no-deps && \
bash /home/shaoyuw/scripts/sglang/launch_paras_sglang.sh 2>&1 | tee /tmp/sglang_paras_test.log &
sleep 45 && curl -s --max-time 10 http://localhost:30000/paras_configure_tp && \
grep "Time taken to configure TP" /tmp/sglang_paras_test.log
```

Note: The launch script uses `mem-fraction-static 0.75` which will OOM. Override with a modified script or use the full command above with `0.6`.
