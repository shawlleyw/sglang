---
name: bench-ep-tp
description: Benchmark sglang MoE model inference with TP (tensor parallel) and/or EP (expert parallel + DP attention + DeepEP) using bench_one_batch. Covers the 4-way comparison (TP/TP, TP/EP, DP/TP, DP/EP), model configs, memory constraints, env vars, cold-run warmup pitfalls, and how to compare results.
metadata:
  short-description: Run 4-way TP/EP MoE benchmarks with bench_one_batch
---

# Benchmark EP vs TP with bench_one_batch

Run MoE model benchmarks comparing the 4 attention/expert parallelism configurations (TP/TP, TP/EP, DP/TP, DP/EP) on A100/H100 GPUs. For the why-analysis of results, see sister skill `analyze-ep-tp`.

## Prerequisites

- Conda env: `sgl_paras` (or whichever env has sglang + deep_ep installed)
- Activate: `source /home/shaoyuw/miniconda3/etc/profile.d/conda.sh && conda activate sgl_paras`
- Models at `/data/shaoyuw/models/` (Qwen3-30B-A3B, Qwen3-235B-A22B, Qwen3-235B-A22B-half)

## Core Concepts

- **Global batch = `--batch-size` × `dp_size`.** `bench_one_batch` reports PER-SCHEDULER batch & throughput. With `--enable-dp-attention --dp-size N`, there are N schedulers. To compare configs fairly, always translate to **global batch** and multiply reported throughput by `dp_size` for system throughput.
- DeepEP `auto` mode uses normal dispatch for prefill, low-latency for decode.
- On A100: `deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM = False` (SM < 90), so triton runner is used for MoE kernels.
- **4 supported configs**: TP/TP (baseline), TP/EP (DeepEP, no DP), DP/TP (DP attn + TP experts), DP/EP (DP attn + DeepEP). See `analyze-ep-tp` for semantics.
- **`--ep-size N` without `--moe-a2a-backend deepep`** is technically valid ("TP/EP*" — EP-sharded experts + AllReduce). On 8×A100 it is a net loss vs TP/TP at every tested batch (masked activation + full-hidden AR outweighs full-N GEMM benefit). Use DeepEP for real EP wins.

## `bench_one_batch` Does NOT Chunk Prefills

`bench_one_batch.extend()` calls `model_runner.forward(forward_batch)` as a **single forward pass** — it bypasses the scheduler entirely. `--chunked-prefill-size` has **no effect** on results from this tool (verified: identical prefill latency at batch 256 with chunk=1024 vs chunk=8192).

`chunked_prefill_size` only matters for `launch_server` + real serving, where the scheduler chunks per request. `server_args.py:1385` does silently divide it by `dp_size` under `--enable-dp-attention`, which matters for production but not for `bench_one_batch`.

**Real benchmark pitfall — cold vs warm runs**: The first `bench_one_batch` invocation in a fresh Python process runs "cold" (CUDA JIT cache, triton autotuning, thermal state). Large prefills (≥ batch 2048) can measure 1.5–1.7× slower on the cold run vs subsequent warm runs. **Always re-run the largest batch once** and compare — discard or investigate if the two differ by >10%.

## 4-Way Command Templates

Run each config as a **single invocation** with all target batches — `bench_one_batch` sweeps `--batch-size` in one process, reusing the loaded model and cuda graphs. Single invocation also amortizes the ~2–3 min model load + graph capture across the whole batch grid.

Common setup:
```bash
export MODEL=/data/shaoyuw/models/Qwen3-235B-A22B-half
export NUM_GPUS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RESULT_FILE=results.jsonl
```

### 1. TP/TP — baseline
```bash
unset SGLANG_DEEPEP_BF16_DISPATCH SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK NVSHMEM_QP_DEPTH
python -m sglang.bench_one_batch \
    --model-path "$MODEL" --trust-remote-code --load-format dummy \
    --disable-overlap-schedule --mem-fraction-static 0.8 \
    --input-len 10 --output-len 10 \
    --tp-size $NUM_GPUS \
    --batch-size 8 64 512 2048 --cuda-graph-bs 8 64 512 2048 \
    --result-filename "$RESULT_FILE" --run-name tp_tp
```

### 2. TP/EP — TP attention + EP experts via DeepEP
```bash
export SGLANG_DEEPEP_BF16_DISPATCH=true
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=2048
export NVSHMEM_QP_DEPTH=4096
python -m sglang.bench_one_batch \
    --model-path "$MODEL" --trust-remote-code --load-format dummy \
    --disable-overlap-schedule --mem-fraction-static 0.8 \
    --input-len 10 --output-len 10 \
    --tp-size $NUM_GPUS \
    --moe-a2a-backend deepep --deepep-mode auto \
    --batch-size 8 64 512 --cuda-graph-bs 8 64 512 \
    --result-filename "$RESULT_FILE" --run-name tp_ep
```
With `dp_size=1` every rank's DeepEP dispatch sees the full batch. Max testable global batch is capped by `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` (must be ≥ max batch), `NVSHMEM_QP_DEPTH` (≥ (max+1)×2), and memory.

### 3. DP/TP — classic v0.4 DeepSeek DP attention, TP experts
```bash
unset SGLANG_DEEPEP_BF16_DISPATCH SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK NVSHMEM_QP_DEPTH
python -m sglang.bench_one_batch \
    --model-path "$MODEL" --trust-remote-code --load-format dummy \
    --disable-overlap-schedule --mem-fraction-static 0.8 \
    --input-len 10 --output-len 10 \
    --tp-size $NUM_GPUS --dp-size $NUM_GPUS \
    --enable-dp-attention --enable-dp-lm-head \
    --batch-size 1 8 64 256 --cuda-graph-bs 1 8 64 256 \
    --result-filename "$RESULT_FILE" --run-name dp_tp
```

### 4. DP/EP — DP attention + DeepEP
```bash
export SGLANG_DEEPEP_BF16_DISPATCH=true
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256
unset NVSHMEM_QP_DEPTH
python -m sglang.bench_one_batch \
    --model-path "$MODEL" --trust-remote-code --load-format dummy \
    --disable-overlap-schedule --mem-fraction-static 0.8 \
    --input-len 10 --output-len 10 \
    --tp-size $NUM_GPUS --dp-size $NUM_GPUS \
    --enable-dp-attention --enable-dp-lm-head \
    --moe-a2a-backend deepep --deepep-mode auto \
    --batch-size 1 8 64 256 --cuda-graph-bs 1 8 64 256 \
    --result-filename "$RESULT_FILE" --run-name dp_ep
```

### Between configs — cleanup stragglers
NCCL teardown occasionally leaves rank processes holding memory. Between runs:
```bash
pkill -9 -f "sglang.bench_one_batch" 2>/dev/null || true
nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ',' | xargs -r kill -9 2>/dev/null || true
sleep 8
```

### Launching in tmux for monitoring
```bash
tmux new-session -d -s bench "bash run_all.sh 2>&1 | tee logs/driver.log"
tmux attach -t bench   # Ctrl-B then D to detach
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
| Qwen3-235B-A22B-half (47 layers, dummy) | 0.8 | 0.8 | Use `--load-format dummy` for memory testing. 0.8 is plenty across all 4 configs. |

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

8. **Chunked prefill does NOT apply to `bench_one_batch`** — see top note. `chunked_prefill_size` is scheduler-level and scheduler is not used here.

9. **Enable-dp-lm-head is coupled to enable-dp-attention**: `server_args.py:1390-1393` asserts. Also forced to False when `dp_size==1`. There is no way to enable DP LM head in TP-attention configs; keep it on DP configs and note the <2% decode impact in reports.

10. **Cold-run noise on first large-prefill invocation**: triton autotuning and CUDA JIT warmup can inflate the first batch-2048 prefill by 40-70%. Re-run to confirm; discrepancies >10% between back-to-back runs indicate the first was cold.

## Comparison Table Format

Present decode + prefill tables at the same global batches for all 4 configs, with system throughput (DP configs × `dp_size`) and a ratio vs TP/TP baseline:

```
| Global Batch | TP/TP | TP/EP | DP/TP | DP/EP |
|---:|---:|---:|---:|---:|
| 8 | 1.00× | ... | ... | ... |
| 2048 | 1.00× | ... | ... | ... |
```

See sister skill `analyze-ep-tp` §Axis Decomposition for how to read these.

## Reference Workspaces

Completed 4-way benchmark workspaces (driver scripts, results, reports):

| Path | Description |
|---|---|
| `~/qwen235b_4way_analysis/run_all_v2.sh` | 8-GPU 4-way driver for Qwen3-235B-A22B-half with duplicate-batch warmup |
| `~/qwen235b_4way_analysis/analyze.py` | Best-of-2 analysis → decode/prefill/ratio tables + axis decomposition |
| `~/qwen235b_4way_analysis/report.md` | Full 4-way analysis with interpretation |

## Reference Results (quick)

- **Qwen3-30B-A3B, 8×A100 (2-way)**: DP/EP decode crossover ~1024; 1.50× TP/TP at 2048, 1.37× at 4096.
- **Qwen3-235B-A22B-half, 8×A100 (4-way, dummy)**: see `analyze-ep-tp` §Reference Results.
