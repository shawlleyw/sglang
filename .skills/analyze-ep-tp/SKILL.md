---
name: analyze-ep-tp
description: Analyze WHY TP or EP is faster at a given batch size for MoE models. Covers the 3 performance axes (communication, MoE compute/GEMM shape, workflow overhead), profiling methodology, per-rank load imbalance detection, and further confirmation experiments.
metadata:
  short-description: Root-cause analysis of TP vs EP performance differences
---

# Analyzing TP vs EP Performance Differences

## Mental Model: The 3 Performance Axes

Every TP-vs-EP comparison decomposes into three axes. At any batch size, the winner is determined by which axis dominates.

### Axis 1: Communication Pattern

| | TP | EP |
|---|---|---|
| Attention comm | AllReduce per layer (full hidden) | **None** (DP attention is local) |
| MoE comm | AllReduce per layer (full hidden) | DeepEP dispatch + combine (all-to-all) |
| Comm ops/layer | 2× AllReduce | 1× dispatch + 1× combine |
| Scaling with batch | **Linear** — volume = `batch × hidden_size` per AllReduce | **Sub-linear** — per-rank volume is `batch/dp_size × hidden` |

**At small batch**: TP AllReduce is cheap (custom `cross_device_reduce_1stage` handles small tensors in <1ms). DeepEP dispatch+combine has ~5ms fixed cost in low-latency RDMA mode (buffer sync, metadata exchange).

**At large batch**: TP AllReduce grows linearly (28× from batch 8→2048 in our measurements). DeepEP stays relatively flat (2.3×). EP also saves the entire attention AllReduce.

**Confirmation method**: Profile with CUDA graphs enabled and look at:
- `cross_device_reduce_1stage` / `cross_device_reduce_2stage` / `ncclDevKernel_AllReduce` durations (TP)
- `deep_ep::internode_ll::dispatch` + `deep_ep::internode_ll::combine` durations (EP decode)
- `deep_ep::intranode::dispatch` + `deep_ep::intranode::combine` durations (EP prefill)

### Axis 2: MoE Compute / GEMM Shape Efficiency

Both modes execute the same total FLOPs. The difference is GEMM shape.

| | TP | EP |
|---|---|---|
| Experts per rank | All E experts | E/ep_size experts |
| GEMM N dimension | `intermediate_size / tp_size` | `intermediate_size` (full) |
| GEMM M dimension | Full batch | ~batch (routed tokens to this rank) |

**Key insight**: A100/H100 tensor cores achieve peak utilization when both M and N are large. TP shrinks N by tp_size (e.g., Qwen3-30B: N=768/8=96 — pathologically narrow). EP keeps N full but depends on batch size for M.

**At small batch**: Both have small M. TP's narrow N doesn't matter much because M is the bottleneck. EP's full N doesn't help. But EP has additional overhead from masked activation kernels and scatter/gather.

**At large batch**: EP's full N achieves significantly better tensor core utilization. In our measurements, EP fused_moe is 17% faster at equiv batch=2048 (Qwen3-30B, N=768 vs N=96).

**Confirmation methods**:
1. Compare `fused_moe_kernel` total duration across modes at each batch size
2. Check `_silu_and_mul_masked_kernel` overhead in EP (adds ~1ms at small batch, negligible at large)
3. Run MoE kernel microbenchmark: `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` with both TP and EP configs

### Axis 3: Workflow Overhead & Per-Rank Data Volume

EP processes batch/dp_size tokens through attention, norms, and elementwise ops. TP processes the full batch.

| Component | TP per rank | EP per rank |
|---|---|---|
| Attention tokens | batch | batch/8 |
| RMSNorm tokens | batch | batch/8 |
| Elementwise ops | batch | batch/8 |
| Kernel launches/layer | ~10 | ~20+ (dispatch setup, scatter, gather, masked ops) |

**At small batch**: EP's extra kernel launches dominate (dispatch buffer management alone is ~2ms). This is 20-40% of the total gap.

**At large batch**: EP's 8× reduction in norm/elementwise data volume saves significant time. The extra launches are amortized by CUDA graphs.

**CUDA graph amplification**: EP benefits disproportionately from CUDA graphs because it has more kernel launches that get amortized. In our measurements: EP is 1.50× faster WITH cuda graphs but only 1.05× WITHOUT at equiv batch=2048.

**Confirmation method**: Compare with and without `--disable-cuda-graph` to isolate launch overhead.

---

## Profiling Methodology

### Step 1: Latency Benchmarks (CUDA Graphs ON, No Profiling)

Get clean production numbers first.

```bash
# TP
python -m sglang.bench_one_batch \
    --model-path <MODEL> --trust-remote-code \
    --tp-size 8 \
    --cuda-graph-bs 8 64 512 2048 4096 \
    --disable-overlap-schedule --mem-fraction-static 0.8 \
    --batch-size 8 64 512 2048 4096 \
    --input-len 10 --output-len 10 --run-name tp

# EP
export SGLANG_DEEPEP_BF16_DISPATCH=true
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512
export NVSHMEM_QP_DEPTH=2048
python -m sglang.bench_one_batch \
    --model-path <MODEL> --trust-remote-code \
    --tp-size 8 --dp-size 8 --ep-size 8 \
    --enable-dp-attention --enable-dp-lm-head \
    --moe-a2a-backend deepep --deepep-mode auto \
    --cuda-graph-bs 1 8 64 256 512 \
    --disable-overlap-schedule --mem-fraction-static 0.8 \
    --batch-size 1 8 64 256 512 \
    --input-len 10 --output-len 10 --run-name ep
```

Remember: TP batch = EP batch × NUM_GPUS for equivalent comparison.

### Step 2: Profile Traces (CUDA Graphs OFF, All Ranks)

`bench_one_batch.py` has been patched (line 717) to profile ALL ranks instead of rank 0 only. Use `--disable-cuda-graph` so the profiler captures every kernel (CUDA graph replay hides kernels from torch.profiler).

```bash
export SGLANG_TORCH_PROFILER_DIR=<output_dir>
python -m sglang.bench_one_batch \
    ... \
    --disable-cuda-graph \
    --profile --profile-filename-prefix <prefix> \
    --batch-size <SINGLE_BATCH_SIZE>    # one batch size per run for clean traces
```

This produces `<prefix>_rank{0-7}_batch{N}_..._decode.trace.json.gz` files for each rank.

**Caveat**: Without CUDA graphs, latencies are 10-30% higher and EP is penalized more (more kernel launches). The *compute-only* breakdown (Attention + MoE Compute) is reliable. Communication times include synchronization waiting and are inflated.

### Step 3: nsys Profiling (if available)

nsys captures CUDA graph replay kernels correctly:

```bash
nsys profile \
    --trace-fork-before-exec=true \
    --cuda-graph-trace=node \
    -t cuda -f true \
    -o <output_prefix> \
    python -m sglang.bench_one_batch ...
```

Use `nsys stats -r cuda_gpu_trace <file>.nsys-rep -o <base> -f csv` to extract per-device kernel data.

### Step 4: Analyze Traces

Use `analyze_multirank.py` (in `~/qwen-30b-analysis/`) or `analyze_traces.py` (in `~/sglang_profile_ep_tp/`).

Kernel classification rules:
- **Attention**: flashinfer kernels, cutlass/ampere GEMMs (QKV/O projections), RoPE
- **Communication**: `cross_device_reduce`, `ncclDevKernel`, `deep_ep::*::dispatch`, `deep_ep::*::combine`
- **MoE Compute**: `fused_moe_kernel`, `topkGatingSoftmax`, `moe_align_block_size`, `silu_and_mul`, `moe_sum_reduce`, `ep_scatter/gather`
- **Other**: RMSNorm, elementwise, fills, copies

---

## Load Imbalance Detection (EP Only)

EP routes different tokens to different expert-owning ranks. With real model weights, routing is non-uniform.

**What to measure**: Per-rank `moe_compute` time across all 8 ranks for the same batch.

**Metrics**:
- **Spread** = max - min across ranks
- **CV%** = spread / mean × 100

**Expected behavior**:
- Small batch (≤64 equiv): CV 15-25% — few tokens, high routing variance
- Large batch (≥512 equiv): CV 5-10% — law of large numbers smooths routing
- If CV > 30% at large batch → expert routing is pathologically skewed, consider EPLB (expert load balancing)

**Our measurements (Qwen3-30B-A3B, real weights)**:
| Equiv Batch | MoE Compute CV% |
|---|---|
| 64 | 21.9% |
| 2048 | 7.1% |

TP has zero compute imbalance by construction (all ranks process identical data).

---

## Diagnostic Decision Tree

```
Is EP faster than TP at this batch size?
├── NO (TP wins) → Check:
│   ├── Communication: Is DeepEP dispatch+combine > TP AllReduce?
│   │   └── YES → Fixed cost of DeepEP dominates at small batch
│   ├── MoE Compute: Is EP fused_moe slower than TP?
│   │   └── YES → Poor GEMM utilization (small M, EP overhead)
│   ├── Load Imbalance: Is EP MoE CV% > 20%?
│   │   └── YES → Routing skew, consider EPLB or batch size increase
│   └── Overhead: Is EP "other" time >> TP?
│       └── YES → Kernel launch overhead, dispatch buffer management
│
├── YES (EP wins) → Check:
│   ├── Communication: Is TP AllReduce >> EP DeepEP?
│   │   └── YES → AllReduce scaling is the primary driver
│   ├── Attention: Is TP attention >> EP attention?
│   │   └── YES → DP attention savings (batch/8 per rank)
│   ├── MoE Compute: Is EP fused_moe faster?
│   │   └── YES → Better GEMM shape (full N width)
│   └── Other: Is EP norm/elementwise << TP?
│       └── YES → batch/8 per rank reduces overhead
│
└── CLOSE (within 10%) → The crossover region
    └── Run more batch sizes (e.g., 768, 1024, 1536) to pinpoint crossover
```

---

## Further Confirmation Experiments

### 1. Pinpoint the Crossover

Run additional batch sizes between the last TP-wins and first EP-wins:
```bash
# If crossover is between 512 and 2048:
--batch-size 768 1024 1536  # TP
--batch-size 96 128 192     # EP (equiv / 8)
```

### 2. Isolate Communication Cost

Run with `--disable-cuda-graph` and compare comm kernel times directly. Or if nsys is available, use `--cuda-graph-trace=node` for accurate in-graph timing.

### 3. MoE Kernel Microbenchmark

```bash
# TP config: E=128, N=intermediate/tp_size
python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py \
    --model <MODEL> --tp-size 8 --tune

# EP config: E=num_experts/ep_size, N=intermediate
python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py \
    --model <MODEL> --tp-size 8 --ep-size 8 --tune
```

This tunes AND benchmarks the fused_moe kernel for both configs. Compare the tuned throughput directly.

### 4. Measure Load Imbalance Impact

Profile EP at a single batch size with all ranks. Compare per-rank `fused_moe_kernel` durations. If spread is large, the slowest rank is the bottleneck (all-to-all combine waits for the slowest rank).

### 5. CUDA Graph Impact

Run the same config with and without `--disable-cuda-graph`:
```bash
# With CUDA graphs (production)
python -m sglang.bench_one_batch ... --cuda-graph-bs <BS>

# Without CUDA graphs (profiler-friendly)
python -m sglang.bench_one_batch ... --disable-cuda-graph
```

The difference reveals how much kernel launch overhead matters. EP benefits more from CUDA graphs.

### 6. H100 vs A100 Comparison

On A100 (SM80): `deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM = False`, triton runner used for MoE.
On H100 (SM90): DeepGemm JIT enabled for EP, potentially shifting the crossover point.

### 7. Real vs Dummy Weights

Dummy weights produce uniform routing. Real weights may skew routing, increasing EP load imbalance. Always validate with real weights for production conclusions.

---

## Reference Results

### Qwen3-30B-A3B (8×A100, real weights, CUDA graphs)

| Equiv Batch | EP/TP (decode) |
|---|---|
| 8 | 0.42× (TP wins) |
| 64 | 0.49× |
| 512 | 0.76× |
| 2048 | **1.50×** (EP wins) |
| 4096 | **1.37×** |

Crossover: ~1024 equiv batch.

### Qwen3-235B-A22B-half (8×A100, dummy weights, CUDA graphs)

| Equiv Batch | EP/TP (decode) |
|---|---|
| 8 | 0.39× |
| 64 | 0.60× |
| 512 | 0.91× |
| 2048 | **1.23×** |

Crossover: ~1024 equiv batch.

---

## Existing Analysis Scripts & Data

| Path | Description |
|---|---|
| `~/qwen-30b-analysis/report.md` | Full analysis report (30B real weights) |
| `~/qwen-30b-analysis/analyze_multirank.py` | Multi-rank trace parser (per-rank breakdown + load imbalance) |
| `~/qwen-30b-analysis/multirank_analysis_raw.json` | Raw per-rank data |
| `~/qwen-30b-analysis/torch_tp/` | TP per-rank traces (batch=64, 2048) |
| `~/qwen-30b-analysis/torch_ep/` | EP per-rank traces (batch=8, 256 per-rank) |
| `~/sglang_profile_ep_tp/analyze_traces.py` | Single-rank trace parser (with CUDA graphs) |
| `~/sglang_profile_ep_tp/analysis_report.md` | 235B-half analysis report |
