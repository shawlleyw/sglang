---
name: bench-ep-tp
description: Benchmark sglang MoE model inference with TP (tensor parallel) and/or EP (expert parallel + DP attention + DeepEP) using bench_one_batch. Knows model configs, memory constraints, env vars, and how to compare results.
metadata:
  short-description: Run TP/EP MoE benchmarks with bench_one_batch
---

# Benchmark EP vs TP with bench_one_batch

Run MoE model benchmarks comparing Tensor Parallel (TP) and Expert Parallel (EP with DP attention + DeepEP) modes on A100/H100 GPUs.

## Prerequisites

- Conda env: `sgl_paras` (or whichever env has sglang + deep_ep installed)
- Activate: `source /home/shaoyuw/miniconda3/etc/profile.d/conda.sh && conda activate sgl_paras`
- Models at `/data/shaoyuw/models/` (Qwen3-30B-A3B, Qwen3-235B-A22B, Qwen3-235B-A22B-half)

## Core Concepts

- **TP batch size = EP batch size × NUM_GPUS** for equivalent comparison
- EP throughput must be **multiplied by NUM_GPUS** to get total system throughput
- DeepEP `auto` mode uses normal dispatch for prefill, low-latency for decode
- On A100: `deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM = False` (SM < 90), so triton runner is used for MoE kernels

## TP Command Template

```bash
export NUM_GPUS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m sglang.bench_one_batch \
    --model <MODEL_PATH> --trust-remote-code \
    --tp-size $NUM_GPUS \
    --cuda-graph-bs <BATCH_SIZES> \
    --disable-overlap-schedule \
    --mem-fraction-static <MEM_FRAC> \
    --batch <BATCH_SIZES> \
    --input-len <INPUT_LEN> --output-len <OUTPUT_LEN> \
    --run-name tp
```

## EP Command Template

```bash
export NUM_GPUS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_DEEPEP_BF16_DISPATCH=true
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=<MAX_DISPATCH>
export NVSHMEM_QP_DEPTH=<QP_DEPTH>  # default 1024, increase if needed

python -m sglang.bench_one_batch \
    --model <MODEL_PATH> --trust-remote-code \
    --tp-size $NUM_GPUS --dp-size $NUM_GPUS --ep-size $NUM_GPUS \
    --enable-dp-attention --enable-dp-lm-head \
    --moe-a2a-backend deepep --deepep-mode auto \
    --cuda-graph-bs <BATCH_SIZES> \
    --disable-overlap-schedule \
    --mem-fraction-static <MEM_FRAC> \
    --batch <BATCH_SIZES> \
    --input-len <INPUT_LEN> --output-len <OUTPUT_LEN> \
    --run-name ep
```

## Profiling

Add `--profile` flag and set `SGLANG_TORCH_PROFILER_DIR=<output_dir>`. Generates `.trace.json.gz` files viewable in Perfetto UI (https://ui.perfetto.dev/).

Profile runs should use a **single batch size** per invocation to get clean traces.

## Key Environment Variables

| Variable | Purpose | Default | Notes |
|---|---|---|---|
| `SGLANG_DEEPEP_BF16_DISPATCH` | Use bf16 for DeepEP dispatch (non-FP8 models) | unset | Set to `true` for bf16 models |
| `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` | Max tokens per rank for DeepEP dispatch | — | Must be >= max EP batch size per rank |
| `NVSHMEM_QP_DEPTH` | NVSHMEM queue pair depth for low-latency mode | 1024 | Must be >= `(max_dispatch + 1) * 2`. Set to 2048 for bsz 512 |
| `SGLANG_TORCH_PROFILER_DIR` | Directory for torch profiler output | — | Set when using `--profile` |

## Understanding `--mem-fraction-static`

This controls the fraction of **total GPU memory** reserved for **model weights + KV cache** (the static allocation). The rest is the **runtime pool**.

```
static_budget  = total_gpu_mem × mem_fraction_static   → weights + KV cache
runtime_budget = total_gpu_mem × (1 - mem_fraction_static) → DeepEP buffers, cuda graph, activations, temp buffers
```

**How it works:**
1. `static_budget = total_gpu_mem × mem_fraction_static`
2. Load model weights → consumes W GB from static_budget
3. `kv_cache_slots = (static_budget - W) / per_token_kv_size` — determines max batch × seq_len
4. Runtime pool `(1 - mem_fraction_static) × total_gpu_mem` must fit: DeepEP buffers (normal + low-latency), cuda graph capture memory, activation tensors, and temporary allocations

**Tradeoffs:**
- **Higher** → more KV cache (larger batch/seq capacity) but less runtime headroom → risk OOM during cuda graph capture or DeepEP buffer allocation
- **Lower** → safer runtime but fewer KV cache slots → limits max batch size

**Why EP needs lower mem-fraction than TP:**
EP allocates DeepEP communication buffers from the runtime pool. With `deepep-mode auto`, **both** normal and low-latency buffers are allocated, consuming significantly more runtime memory. Additionally, cuda graph capture for EP graphs requires extra temporary memory. TP only needs NCCL buffers which are much smaller.

## Memory Guidelines (8×A100-80GB)

| Model | TP mem-fraction | EP mem-fraction | Notes |
|---|---|---|---|
| Qwen3-30B-A3B | 0.8 | 0.8 | Comfortable on both |
| Qwen3-235B-A22B | 0.8 | 0.88-0.9 | EP very tight; auto mode needs normal + low-latency buffers in runtime pool |
| Qwen3-235B-A22B-half (47 layers, dummy) | 0.8 | 0.8 | Use `--load-format dummy` for memory testing |

**Debugging OOM:** If `--mem-fraction-static` is too high, you'll see either:
- `RuntimeError: Not enough memory` during KV cache init — the static budget can't even fit weights + minimal cache (need to reduce model size or GPUs)
- `CUDA out of memory` during cuda graph capture or runtime — lower mem-fraction to give more runtime headroom (try 0.05 decrements)

## Typical Batch Sizes

| | Small | Medium | Large | XL |
|---|---|---|---|---|
| **TP** | 8 | 64 | 512 | 2048 |
| **EP** (per rank, 8 GPUs) | 1 | 8 | 64 | 256 |
| **Equivalent total** | 8 | 64 | 512 | 2048 |

## Known Constraints & Gotchas

1. **NVSHMEM_QP_DEPTH**: Low-latency dispatch asserts `qp_depth >= (max_dispatch_tokens + 1) * 2`. Default 1024 only supports up to 511 tokens per rank. Set `NVSHMEM_QP_DEPTH=2048` for 512 per rank.

2. **expert_alignment=128**: Required by `ep_scatter` triton kernel (`BLOCK_E=128`). Do NOT change to 1.

3. **RDMA buffers**: Gated by `deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM` — allocated on H100 (SM>=90), skipped on A100 (SM80) since `get_rdma_buffer_size_hint` fails on A100.

4. **unquant.py dispatch_output**: `dispatch_output.hidden_states` and `topk_output` must only be accessed inside `if _use_aiter:` branch — `DeepEPNormalDispatchOutput` doesn't have these attributes.

5. **Cuda graph + EP**: Works with `--deepep-mode auto`. The low-latency path handles decode within cuda graph. Normal mode (`--deepep-mode normal`) disables cuda graph.

6. **MoE kernel tuning**: Without tuned configs, you'll see "Performance might be sub-optimal!" warnings. Tune with:
   ```bash
   # TP config (E=128, N for model's moe_intermediate_size / tp_size)
   python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py \
       --model <MODEL> --tp-size 8 --tune
   # EP config (E=num_experts/ep_size, N=moe_intermediate_size)
   python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py \
       --model <MODEL> --tp-size 8 --ep-size 8 --tune
   ```
   Use `CUDA_VISIBLE_DEVICES=0,1,2,3` for 4-GPU parallel tuning via Ray.

7. **FP8 mode**: Set `USE_FP8=1`, add `--quantization fp8`, and unset `SGLANG_DEEPEP_BF16_DISPATCH`.

## Comparison Table Format

When presenting results, use two tables (Decode and Prefill) with columns:

```
| Equiv Total Batch | TP Batch | TP Latency | TP Throughput | EP Batch (per rank) | EP Latency | EP Throughput ×N | EP / TP |
```

Where N = number of GPUs. Mark EP/TP > 1.0 with a checkmark.

## Reference Results (Qwen3-30B-A3B, 8×A100, cuda graph + deepep auto)

Decode crossover: EP beats TP at ~1024 equivalent batch size.
- Batch 2048 equiv: EP 1.31x of TP
- Batch 4096 equiv: EP 1.37x of TP
