---
name: bench-ep-tp
description: Benchmark sglang MoE model inference across the 4 attention/expert parallelism configurations (TP/TP, TP/EP, DP/TP, DP/EP) using bench_one_batch. Only DP/EP uses DeepEP; TP/EP uses AllReduce-based EP (--ep-size N, no DeepEP). Covers model configs, memory constraints, env vars, cold-run warmup pitfalls, and how to compare results.
metadata:
  short-description: Run 4-way attention/expert parallelism MoE benchmarks with bench_one_batch
---

# Benchmark EP vs TP with bench_one_batch

Run MoE model benchmarks comparing the 4 attention/expert parallelism configurations (TP/TP, TP/EP, DP/TP, DP/EP) on A100/H100 GPUs. For the why-analysis of results, see sister skill `analyze-ep-tp`.

## Prerequisites

- Conda env: `sgl_paras` (or whichever env has sglang + deep_ep installed)
- Activate: `source /home/shaoyuw/miniconda3/etc/profile.d/conda.sh && conda activate sgl_paras`
- Models at `/data/shaoyuw/models/` (Qwen3-30B-A3B, Qwen3-235B-A22B, Qwen3-235B-A22B-half)

## Core Concepts

- **Global batch = `--batch-size` × `dp_size`.** `bench_one_batch` reports PER-SCHEDULER batch & throughput. With `--enable-dp-attention --dp-size N`, there are N schedulers. To compare configs fairly, always translate to **global batch** and multiply reported throughput by `dp_size` for system throughput.
- **4 supported configs**: TP/TP (baseline), TP/EP (EP-sharded experts + AllReduce, no DeepEP), DP/TP (DP attn + TP experts), DP/EP (DP attn + DeepEP). See `analyze-ep-tp` for semantics.
- **Only DP/EP uses DeepEP.** With TP attention, tokens are replicated across ranks after the post-attention AllReduce; DeepEP dispatch would be redundant. TP/EP therefore uses `--ep-size N` (no `--moe-a2a-backend`) — experts are partitioned across ranks, each rank runs its local experts on the full batch, results combined via full-hidden AllReduce.
- DeepEP `--deepep-mode auto` uses normal dispatch for prefill, low-latency for decode.
- On A100: `deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM = False` (SM < 90), so triton runner is used for MoE kernels.
- **TP/EP is a net loss** at every tested batch on 8×A100 (each rank iterates `E/ep_size` experts over the full batch with masking; extra MoE work not offset by any comm saving). Documented as an anti-pattern for production serving.

## `bench_one_batch` Does NOT Chunk Prefills

`bench_one_batch.extend()` calls `model_runner.forward(forward_batch)` as a **single forward pass** — it bypasses the scheduler entirely. `--chunked-prefill-size` has **no effect** on results from this tool (verified: identical prefill latency at batch 256 with chunk=1024 vs chunk=8192).

`chunked_prefill_size` only matters for `launch_server` + real serving, where the scheduler chunks per request. `server_args.py:1385` does silently divide it by `dp_size` under `--enable-dp-attention`, which matters for production but not for `bench_one_batch`.

**Real benchmark pitfall — cold vs warm runs**: The first `bench_one_batch` invocation in a fresh Python process runs "cold" (CUDA JIT cache, triton autotuning, thermal state). Large prefills (≥ batch 2048) can measure 1.5–1.7× slower on the cold run vs subsequent warm runs. **Always re-run the largest batch once** and compare — discard or investigate if the two differ by >10%.

## 4-Way Command Templates

Run each config as a **single invocation** with all target batches — `bench_one_batch` sweeps `--batch-size` in one process, reusing the loaded model and cuda graphs. Single invocation also amortizes the ~2–3 min model load + graph capture across the whole batch grid.

The four configs ship as scripts under [`scripts/paras/eval/a100/<model>/`](file:///home/shaoyuw/sglang/scripts/paras/eval/a100). Use `qwen/` for Qwen3-30B-A3B (default) or any Qwen3-MoE family member, and `gptoss/` for gpt-oss-120b-bf16. Each script reads env-var overrides; defaults match what each config needs (DeepEP env for DP/EP, unset for the AllReduce configs, `--attention-backend triton` for gpt-oss, etc.). All scripts source the shared helpers in [`scripts/paras/eval/lib.sh`](file:///home/shaoyuw/sglang/scripts/paras/eval/lib.sh).

Common setup (every script reads these):
```bash
export MODEL_PATH=/data/shaoyuw/models/Qwen3-235B-A22B-half
export NUM_GPUS=8
export RESULT_FILE=results.jsonl
export LOAD_FORMAT=dummy            # for memory testing; omit for real weights
export MEM_FRACTION_STATIC=0.8      # tune per model — see Memory Guidelines below
```
The scripts default `CUDA_VISIBLE_DEVICES=0,1,...,NUM_GPUS-1`; override only for non-contiguous GPU sets.

### 1. TP/TP — baseline
```bash
RUN_NAME=tp_tp bash scripts/paras/eval/a100/qwen/bench_one_batch_tp_tp.sh
```
Default batch grid `8 64 512 2048` (per-rank == global because no DP). DeepEP env is unset by the script.

### 2. TP/EP — TP attention + EP-sharded experts via AllReduce
```bash
RUN_NAME=tp_ep bash scripts/paras/eval/a100/qwen/bench_one_batch_tp_ep.sh
```
No DeepEP: experts are partitioned across ranks (`--ep-size N`), each rank runs its local experts on the full token batch with masking for non-routed tokens, results combined via full-hidden AllReduce. No dispatch cap, all batch sizes testable. This config is an anti-pattern (net loss vs TP/TP); useful only as a reference point in 4-way analysis.

### 3. DP/TP — classic v0.4 DeepSeek DP attention, TP experts
```bash
RUN_NAME=dp_tp bash scripts/paras/eval/a100/qwen/bench_one_batch_dp_tp.sh
```
Default batch grid `1 8 64 256` (per-DP-rank; equivalent global = ×`NUM_GPUS`). `--enable-dp-attention --enable-dp-lm-head`; experts stay TP-sharded with AllReduce.

### 4. DP/EP — DP attention + DeepEP
```bash
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256 \
RUN_NAME=dp_ep bash scripts/paras/eval/a100/qwen/bench_one_batch_dp_ep.sh
```
The script exports `SGLANG_DEEPEP_BF16_DISPATCH=true` and `NVSHMEM_QP_DEPTH=2048` by default. Override `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` if your equivalent global batch ≥ 512×`NUM_GPUS`.

For gpt-oss-120b-bf16 swap `qwen/` → `gptoss/` in any of the script paths above; the gpt-oss scripts add `--attention-backend triton --moe-runner-backend triton` and (for non-ParaS configs) `--disable-hybrid-swa-memory`.

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

The bench scripts expose two profile toggles (default off):

```bash
# Torch profiler: adds --profile + --disable-cuda-graph (kernels outside cuda graph only).
ENABLE_TORCH_PROFILE=1 SGLANG_TORCH_PROFILER_DIR=/path/to/torch_traces \
    BATCH_SIZE=2048 CUDA_GRAPH_BS=2048 RUN_NAME=tp_tp \
    bash scripts/paras/eval/a100/qwen/bench_one_batch_tp_tp.sh

# nsys: wraps the python invocation with nsys profile --cuda-graph-trace=node -t cuda.
ENABLE_NSYS=1 NSYS_OUTPUT=/path/to/trace_prefix \
    BATCH_SIZE=2048 CUDA_GRAPH_BS=2048 RUN_NAME=tp_tp \
    bash scripts/paras/eval/a100/qwen/bench_one_batch_tp_tp.sh
```

**Torch profiler**: generates `.trace.json.gz` viewable in Perfetto. The toggle automatically adds `--disable-cuda-graph` because torch profiler can't see in-graph kernels. Use a **single batch size** (`BATCH_SIZE=2048 CUDA_GRAPH_BS=2048`) per invocation for clean traces.

**nsys** (preferred, CUDA-graph-aware): captures kernels inside CUDA graphs correctly. The script auto-creates `TMPDIR` next to `NSYS_OUTPUT` to avoid `/tmp/nvidia` permission errors. For finer scoping, pass `--profile-activities CUDA_PROFILER --capture-range=cudaProfilerApi --profile-stage decode|prefill` via additional sglang flags (script doesn't yet expose these; edit if needed).

## Key Environment Variables (DP/EP only — the other configs don't use DeepEP)

| Variable | Purpose | Default | Notes |
|---|---|---|---|
| `SGLANG_DEEPEP_BF16_DISPATCH` | Use bf16 for DeepEP dispatch (non-FP8 models) | unset | Set to `true` for bf16 models in DP/EP |
| `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` | Max tokens per rank for DeepEP dispatch | — | Must be >= `global_batch / dp_size` for DP/EP. On 8 GPUs at global batch 2048 → set to 256 |
| `NVSHMEM_QP_DEPTH` | NVSHMEM queue pair depth for low-latency mode | 1024 | Must be >= `(max_dispatch + 1) * 2`. Default 1024 suffices for ≤511 tokens/rank; set 2048 for 512 |
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

**Why DP/EP may need lower mem-fraction than other configs:**
DP/EP allocates DeepEP communication buffers from the runtime pool. With `--deepep-mode auto`, **both** normal and low-latency buffers are allocated, consuming significantly more runtime memory than NCCL buffers. TP/TP, TP/EP, and DP/TP use AllReduce-based collectives (NCCL buffers only, much smaller). DP/EP cuda graph capture also requires extra temporary memory for DeepEP kernels.

## Memory Guidelines (8×A100-80GB)

All three AllReduce-based configs (TP/TP, TP/EP, DP/TP) use the same `--mem-fraction-static`. Only DP/EP may need a lower value for the DeepEP buffer headroom.

| Model | AR configs (TP/TP, TP/EP, DP/TP) | DP/EP | Notes |
|---|---|---|---|
| Qwen3-30B-A3B | 0.8 | 0.8 | Comfortable on both |
| Qwen3-235B-A22B | 0.8 | 0.88–0.9 | DP/EP very tight; `--deepep-mode auto` needs normal + low-latency buffers in runtime pool |
| Qwen3-235B-A22B-half (47 layers, dummy) | 0.8 | 0.8 | Use `--load-format dummy` for memory testing. 0.8 is plenty across all 4 configs. |

**Debugging OOM:** If `--mem-fraction-static` is too high, you'll see either:
- `RuntimeError: Not enough memory` during KV cache init — the static budget can't even fit weights + minimal cache (need to reduce model size or GPUs)
- `CUDA out of memory` during cuda graph capture or runtime — lower mem-fraction to give more runtime headroom (try 0.05 decrements)

## Typical Batch Sizes

`--batch-size` is per-scheduler. TP-attention configs (TP/TP, TP/EP) have one scheduler (dp_size=1), so `--batch-size` = global batch. DP-attention configs (DP/TP, DP/EP) have `dp_size` schedulers, so `--batch-size` = global batch / dp_size.

| | Small | Medium | Large | XL |
|---|---|---|---|---|
| **TP/TP, TP/EP** (`--batch-size`) | 8 | 64 | 512 | 2048 |
| **DP/TP, DP/EP** (`--batch-size`, 8 GPUs) | 1 | 8 | 64 | 256 |
| **Equivalent global total** | 8 | 64 | 512 | 2048 |

## Known Constraints & Gotchas

1. **NVSHMEM_QP_DEPTH**: Low-latency dispatch asserts `qp_depth >= (max_dispatch_tokens + 1) * 2`. Default 1024 only supports up to 511 tokens per rank. Set `NVSHMEM_QP_DEPTH=2048` for 512 per rank.

2. **expert_alignment=128**: Required by `ep_scatter` triton kernel (`BLOCK_E=128`). Do NOT change to 1.

3. **RDMA buffers**: Gated by `deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM` — allocated on H100 (SM>=90), skipped on A100 (SM80) since `get_rdma_buffer_size_hint` fails on A100.

4. **unquant.py dispatch_output**: `dispatch_output.hidden_states` and `topk_output` must only be accessed inside `if _use_aiter:` branch — `DeepEPNormalDispatchOutput` doesn't have these attributes.

5. **Cuda graph + EP**: Works with `--deepep-mode auto`. The low-latency path handles decode within cuda graph. Normal mode (`--deepep-mode normal`) disables cuda graph.

6. **MoE kernel tuning**: Without tuned configs, you'll see "Performance might be sub-optimal!" warnings. Tune with:
   ```bash
   # TP/TP and DP/TP use TP-sharded experts (E=128, N=moe_intermediate_size / tp_size):
   python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py \
       --model <MODEL> --tp-size 8 --tune
   # TP/EP and DP/EP use EP-sharded experts (E=num_experts/ep_size, N=moe_intermediate_size):
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

- **Qwen3-30B-A3B, 8×A100** (DP/EP vs TP/TP decode): crossover ~1024; 1.50× at 2048, 1.37× at 4096.
- **Qwen3-235B-A22B-half, 8×A100** (4-way, dummy weights): see `analyze-ep-tp` §Reference Results. Headline: DP/EP 1.28× TP/TP decode at 2048; TP/EP is a net loss at every tested batch.
