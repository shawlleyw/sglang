---
name: paras-test-peer-access
description: Run ParaS NVLink peer access correctness and benchmark tests for weight transfer and KV cache transfer. Knows GPU requirements, conda env, torchrun commands, env vars, and how to interpret results.
metadata:
  short-description: Test ParaS peer access weight + KV cache transfer
---

# ParaS Peer Access Tests

Run correctness and benchmark tests for NVLink peer access transfers (weights and KV cache) during EP→TP switching.

## Prerequisites

- Conda env: `sgl_paras`
- Activate: `source /home/shaoyuw/miniconda3/etc/profile.d/conda.sh && conda activate sgl_paras`
- CUDA extension compiled: `cd python/sglang/srt/paras/csrc && pip install -e .`
- Empty GPUs required (check before running)

## GPU Check (ALWAYS run first)

```bash
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
# All values should be < 100 MiB
```

## Test Files

| File | Tests | GPU Requirement |
|------|-------|----------------|
| `test/srt/test_paras_peer_access.py` | Weight transfer (w13, w2): naive vs overlap vs peer_access | Exactly 4 GPUs |
| `test/srt/test_paras_kv_peer_access.py` | KV cache transfer: peer_access vs NCCL all_to_all vs reference | 4 or 8 GPUs |

## Weight Transfer Test

Tests that peer access v2 kernels produce bitwise-identical results to NCCL all-to-all for MoE weight redistribution.

### Correctness

```bash
conda run -n sgl_paras torchrun --nproc_per_node=4 test/srt/test_paras_peer_access.py
```

### Benchmark

```bash
conda run -n sgl_paras torchrun --nproc_per_node=4 test/srt/test_paras_peer_access.py --benchmark
```

**Expected output**: All layers × {w13, w2} bitwise match for naive, overlap, and peer_access paths. Benchmark shows peer_access ~1.5× faster than naive.

### Test Dimensions (Qwen3-30B-A3B)

- 8 layers, 64 experts, hidden=2048, intermediate=1536, bf16
- 4 GPUs: 16 local experts per rank

## KV Cache Transfer Test

Tests that peer_access_kv_transfer produces bitwise-identical results to a PyTorch all_gather reference for EP→TP KV redistribution. Also tests the NCCL all_to_all path and replication consistency.

### Correctness (4 GPUs, no head replication)

```bash
conda run -n sgl_paras torchrun --nproc_per_node=4 test/srt/test_paras_kv_peer_access.py
```

### Correctness (8 GPUs, head replication)

```bash
conda run -n sgl_paras torchrun --nproc_per_node=8 test/srt/test_paras_kv_peer_access.py
```

With 4 KV heads and 8 GPUs, heads are replicated (ranks 0,1 share head 0; ranks 2,3 share head 1; etc.). The test verifies:
1. Peer access output matches reference
2. NCCL all_to_all output matches reference
3. Replicated ranks have identical TP buffers

### Benchmark

```bash
# Small tokens (default: variable 65-100 per rank)
conda run -n sgl_paras torchrun --nproc_per_node=4 test/srt/test_paras_kv_peer_access.py --benchmark

# Large tokens (30k per rank, uniform)
BENCH_TOKENS_PER_RANK=30000 conda run -n sgl_paras torchrun --nproc_per_node=4 test/srt/test_paras_kv_peer_access.py --benchmark

# 8 GPUs with replication + large tokens
BENCH_TOKENS_PER_RANK=30000 conda run -n sgl_paras torchrun --nproc_per_node=8 test/srt/test_paras_kv_peer_access.py --benchmark
```

### Test Dimensions (Qwen3-30B-A3B)

- 3 layers, 4 KV heads, head_dim=128, bf16
- Default token counts: variable per rank (100, 80, 90, 70, 85, 75, 95, 65)
- `BENCH_TOKENS_PER_RANK`: override with uniform count for benchmarking

## Environment Variables

| Variable | Purpose | Default | Notes |
|---|---|---|---|
| `BENCH_TOKENS_PER_RANK` | Uniform token count per rank for KV benchmark | 0 (use variable defaults) | Set to 30000 for realistic benchmark |
| `PARAS_KV_TRANSFER_METHOD` | KV transfer method in production | `nccl` | Set to `peer_access` for NVLink path |

## Correctness Checks Performed

### Weight Transfer (`test_paras_peer_access.py`)
- naive (NCCL sequential) vs peer_access: bitwise match on w13 and w2 per layer
- naive vs overlap (NCCL pipelined): bitwise match

### KV Cache Transfer (`test_paras_kv_peer_access.py`)
- **Reference**: all_gather raw EP data from all ranks, then slice the correct head shard per rank. No permutation, no all_to_all — pure indexing. This is the ground truth.
- **peer_access vs reference**: bitwise match per layer per K/V
- **NCCL all_to_all vs reference**: bitwise match per layer per K/V (uses gather_kv_and_permute + repeat_interleave + all_to_all + permute_and_scatter_kv)
- **Replication consistency** (8-GPU only): ranks sharing the same head have identical TP buffers

## Interpreting Results

### Correctness Output
```
--- peer_access correctness ---
  [OK] layer=0 K bitwise match     ← each layer, K and V separately
  [OK] layer=0 V bitwise match
--- NCCL all_to_all correctness ---
  [OK] layer=0 K bitwise match
  [OK] layer=0 V bitwise match
--- replication consistency ---
  [OK] replicated ranks have identical TP buffers   ← 8-GPU only
SUCCESS: All 3 layers × K/V × 8 ranks bitwise match
```

Any `FAIL` line means data corruption. Common causes:
- CUDA extension not recompiled after kernel changes
- GPU memory contention from other processes
- Wrong `CUDA_VISIBLE_DEVICES`

### Benchmark Output
```
================================================================================
KV BENCHMARK (3 layers, 4 heads, head_dim=128, tokens=240000, TP=8, runs=10)
================================================================================
  Method          avg(ms)      min(ms)      max(ms)       vs nccl
  nccl              4.831        4.756        4.944        1.00x
  peer_access       2.961        2.912        3.040        1.63x
================================================================================
```

Expected speedups:
- Small tokens (~100/rank): 3-5× (NCCL launch overhead dominates)
- Large tokens (30k/rank): 1.5-2× (NVLink bandwidth dominates)

## Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| `Need N empty GPUs, only M available` | Other processes using GPUs | Kill them or wait |
| `cudaIpcOpenMemHandle failed` | GPU memory fragmentation | Restart and clear GPU memory |
| `NCCL timeout` | Leftover NCCL state from crashed run | `pkill -f torchrun` and retry |
| `Import error: paras_peer_access_cuda` | CUDA extension not compiled | `cd python/sglang/srt/paras/csrc && pip install -e .` |
| OOM on 8-GPU benchmark with large tokens | EP_MAX_TOKENS too small for TP view | Auto-handled by `_init_test_params`, but reduce `BENCH_TOKENS_PER_RANK` if needed |
