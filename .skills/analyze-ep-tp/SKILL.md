---
name: analyze-ep-tp
description: Analyze WHY one attention/expert parallelism combination is faster at a given batch size for MoE models. Covers the 4 supported configs (TP/TP, TP/EP, DP/TP, DP/EP), the 3 performance axes (communication, MoE GEMM shape, workflow overhead), profiling methodology, per-rank load imbalance detection, and confirmation experiments.
metadata:
  short-description: Root-cause analysis of attention-TP/DP × experts-TP/EP performance differences
---

# Analyzing Attention × Experts Parallelism Performance Differences

## The 4 Supported Configurations

SGLang supports all four combinations of {attention: TP | DP} × {experts: TP | EP}. Pick the right one for your batch size.

| Tag | Attention | Experts | CLI flags (tp=N) | MoE comm pattern |
|---|---|---|---|---|
| **TP/TP** | TP | TP | `--tp N` | AllReduce (full hidden) |
| **TP/EP** | TP | EP | `--tp N --ep-size N` | AllReduce (full hidden) |
| **DP/TP** | DP | TP | `--tp N --dp N --enable-dp-attention --enable-dp-lm-head` | AllReduce + `dp_gather`/`dp_scatter` |
| **DP/EP** | DP | EP | `--tp N --dp N --enable-dp-attention --enable-dp-lm-head --moe-a2a-backend deepep --deepep-mode auto` | DeepEP all-to-all |

**Key semantics:**
- **Only DP/EP uses DeepEP.** With TP attention, tokens are replicated across ranks after the post-attention AllReduce; routing them via DeepEP dispatch would be redundant (each rank would dispatch the same token set). TP/EP therefore uses AllReduce-based expert parallelism: experts are partitioned across ranks (`--ep-size N`), each rank runs its local experts on the full token batch with masking for non-routed tokens, and results are combined via full-hidden AllReduce.
- `--moe-a2a-backend deepep` (only used in DP/EP) auto-sets `ep_size = tp_size` (`server_args._handle_a2a_moe`); do NOT also set `--ep-size`.
- `--enable-dp-attention` requires `tp_size % dp_size == 0`. The typical choice is `dp_size == tp_size` (attention is pure DP, no intra-node TP). DeepSeek-V2/V3 and Qwen MoE models support DP attention.
- `--deepep-mode auto` is required when prefilling with DeepEP: `low_latency` covers decode only; `normal` disables CUDA graphs. Auto routes prefill → NORMAL and decode → LOW_LATENCY.

**What decides the winner**: each config makes a different trade-off along 3 axes below. The dominant axis changes with batch size, GPU count, and model shape.

## Mental Model: The 3 Performance Axes

### Axis 1: Communication Pattern

| | TP/TP | TP/EP | DP/TP | DP/EP |
|---|---|---|---|---|
| Attention comm | AllReduce per layer | AllReduce per layer | **None** (local DP) | **None** (local DP) |
| MoE comm | AllReduce (full hidden) | AllReduce (full hidden) | AllReduce + dp_gather/scatter | DeepEP dispatch+combine |
| Comm ops/layer | 2× AllReduce | 2× AllReduce | 2× AR + gather/scatter | 1× dispatch + 1× combine |
| Scales with batch | **Linear** `B×H` | **Linear** `B×H` | Linear `B×H` for MoE AR | Sub-linear per-rank `(B/dp)×H` |

**At small batch**: TP AllReduce is cheap (custom `cross_device_reduce_1stage` handles small tensors in <1ms). DeepEP dispatch+combine has ~5ms fixed cost in low-latency RDMA mode (buffer sync, metadata exchange). TP/TP wins because comm overhead is dominated by the small-tensor fast path.

**At large batch**: TP AllReduce grows linearly (28× from batch 8→2048 in our 8-GPU measurements). DeepEP stays relatively flat (2.3×). EP also saves the entire attention AllReduce when combined with DP. DP/EP usually wins at the largest batches.

**Confirmation method**: Profile with CUDA graphs enabled and look at:
- `cross_device_reduce_1stage` / `cross_device_reduce_2stage` / `ncclDevKernel_AllReduce` durations (TP configs)
- `deep_ep::internode_ll::dispatch` + `deep_ep::internode_ll::combine` durations (EP decode)
- `deep_ep::intranode::dispatch` + `deep_ep::intranode::combine` durations (EP prefill)

### Axis 2: MoE Compute / GEMM Shape Efficiency

Both TP and EP execute the same total FLOPs. The difference is GEMM shape.

| | TP (experts) | EP (experts) |
|---|---|---|
| Experts per rank | All E experts | E/ep_size experts |
| GEMM N dimension | `intermediate_size / tp_size` | `intermediate_size` (full) |
| GEMM M dimension | Full batch | ~batch (tokens routed to this rank) |

**Key insight**: A100/H100 tensor cores achieve peak utilization when both M and N are large. TP shrinks N by tp_size (e.g., Qwen3-30B: N=768/8=96 — pathologically narrow). EP keeps N full but depends on batch size for M.

**At small batch**: Both have small M. TP's narrow N doesn't matter much because M is the bottleneck. EP's full N doesn't help. But EP has additional overhead from masked activation kernels and scatter/gather.

**At large batch**: EP's full N achieves significantly better tensor core utilization. In our measurements, EP fused_moe is 17% faster at equiv batch=2048 (Qwen3-30B, N=768 vs N=96 on 8 GPUs).

**4-GPU caveat**: With `tp_size=4`, TP's N shrinks only 4× (N=192), halving the GEMM-shape penalty. This is why EP's crossover is higher on fewer GPUs — see Reference Results.

**Confirmation methods**:
1. Compare `fused_moe_kernel` total duration across configs at each batch size
2. Check `_silu_and_mul_masked_kernel` overhead in EP (adds ~1ms at small batch, negligible at large)
3. Run MoE kernel microbenchmark: `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` with both TP and EP configs

### Axis 3: Workflow Overhead & Per-Rank Data Volume

DP attention processes `batch/dp_size` tokens through attention, norms, and elementwise ops. TP attention processes the full batch.

| Component | TP attn | DP attn (dp=N) |
|---|---|---|
| Attention tokens/rank | batch | batch/N |
| RMSNorm tokens/rank | batch | batch/N (before dp_gather) |
| KV cache/rank | replicated | **partitioned** (huge saving) |
| MoE dispatch buffer | full batch (TP/EP) | batch/N (DP/EP) |

EP adds its own kernel launches:

| | TP experts | EP experts |
|---|---|---|
| Kernel launches/layer | ~10 | ~20+ (dispatch setup, scatter, gather, masked ops) |

**At small batch**: EP's extra kernel launches dominate (dispatch buffer management alone is ~2ms). This is 20-40% of the total gap. DP attention doesn't help because there's not enough work to amortize the gather/scatter.

**At large batch**: DP attention's `batch/dp_size` reduction in attention + norm + KV traffic is huge. EP's extra launches are amortized by CUDA graphs. Both DP/TP and DP/EP pull ahead.

**CUDA graph amplification**: EP benefits disproportionately from CUDA graphs because it has more kernel launches that get amortized. In our 8-GPU measurements: EP is 1.50× faster WITH cuda graphs but only 1.05× WITHOUT at equiv batch=2048.

**Confirmation method**: Compare with and without `--disable-cuda-graph` to isolate launch overhead.

---

## Analysis Methodology: Effective Latency Aggregation

When aggregating per-rank profile data into a wall-clock latency, **do NOT sum across ranks** — that measures total GPU-seconds of work, not end-to-end latency. All ranks run in parallel; the wall-clock is governed by the slowest rank's compute and the actual comm work time.

| Category | Aggregation | Why |
|---|---|---|
| Attention | **max** across ranks | Slowest rank's compute defines when the next collective can start |
| MoE | **max** across ranks | Same — slowest rank bottlenecks the collective |
| Other (norms, sample, fills) | **max** across ranks | Serialized with compute |
| Communication (AR, DeepEP dispatch/combine) | **min** across ranks (caveats below) | The collective kernel's duration on each rank includes wait-for-slowest-arrival. The fastest-arriving rank has minimal wait → its duration ≈ pure comm work |

**Effective latency per forward pass**:
```
latency ≈ max_rank(attn) + max_rank(moe) + min_rank(comm) + max_rank(other)
```

This cleanly separates compute imbalance (shows up as larger `max` terms) from pure comm work (`min` term) **for AllReduce-based collectives**. Wait-time on slow-arriving ranks is attributed to upstream compute imbalance, not to comm itself.

**DeepEP caveat — comm CV has two independent sources:**

| Source | AllReduce | DeepEP dispatch/combine |
|---|:-:|:-:|
| 1. Arrival-time variance (upstream compute wait) | ✅ | ✅ |
| 2. **Intrinsic volume imbalance** (different ranks exchange different amounts) | ❌ | ✅ |

For AllReduce every rank exchanges identical payload → `min_comm` is clean pure-work.
For DeepEP, the rank owning hot experts receives more tokens (dispatch) and sends more results (combine). Even with zero arrival variance, kernel durations differ because comm work differs.

Consequently **`min(DeepEP comm)` is a lower bound on the critical-path comm cost**, representing the lightly-loaded rank. The true latency contribution on the slowest rank is somewhere in `[min_comm, max_comm]`. With dummy weights (uniform routing), volume imbalance ~0 and `min ≈ max ≈ actual`. With real weights, busy ranks carry ~10–20% more combine work — `min_comm` under-estimates real-weight DP/EP cost by ~5–10%.

Volume-imbalance and MoE compute-imbalance are **correlated** (same routing skew drives both): busy MoE rank is also busy combine rank. Our `max_moe + min_comm` formulation slightly under-counts the critical path but does NOT double-count.

**Anti-pattern**: summing kernel durations across all N ranks gives N× the correct latency (total work done, not wall-clock). At 8 ranks, this makes MoE look 8× more costly than it is relative to min-comm.

**Validation**: effective latency should match `bench_one_batch`'s reported latency within ~5 ms (nsys overhead + CPU-side launch). Our Qwen3-235B-half 8×A100 run showed exactly this:

| Config | nsys effective (ms) | bench (ms) |
|---|---:|---:|
| TP/TP decode b=2048 | 91.4 | 96.9 |
| DP/EP decode b=2048 | 74.8 | 75.5 |

### CV% interpretation

- **AllReduce comm CV% is NOT work imbalance.** All ranks do identical AR work, yet CV can be 5–10% due to arrival-time jitter from upstream compute imbalance. Taking `min` strips this.
- **DeepEP comm CV% has two components** (see DeepEP caveat above): (1) arrival-time wait (same as AR) + (2) genuine volume imbalance when routing is skewed. Dummy-weight CV ≈ (1) only; real-weight CV = (1)+(2) compounded.
- **MoE CV% in EP configs at large batch** — genuine expert-routing imbalance. With dummy weights near-uniform; with real weights typically 5–15%. This is the same skew that drives the DeepEP volume-imbalance component.
- **DP attention CV% >2%** usually indicates `dp_gather`/`dp_scatter` sync leaking into the attention measurement window, not compute asymmetry (each DP rank has identical token count by construction).

---

## Profiling Methodology

### Step 1: Latency Benchmarks — All 4 Configs (CUDA Graphs ON)

Get clean production numbers first. Run all 4 configs on the same hardware, same model, same batch grid, then compare at **equivalent global batch**.

**Critical equiv-batch math**: `bench_one_batch --batch-size` is the PER-SCHEDULER batch. With DP attention there are `dp_size` schedulers, so:
- TP-attention configs (TP/TP, TP/EP): global batch = `--batch-size`
- DP-attention configs (DP/TP, DP/EP): global batch = `--batch-size × dp_size`

For an 8-way equiv-batch sweep of 8/64/512/2048/4096 on 8 GPUs (dp_size=8):
- TP configs: `--batch-size 8 64 512 2048 4096`
- DP configs: `--batch-size 1 8 64 256 512` (global = 8, 64, 512, 2048, 4096)

And the reported `throughput` is PER-SCHEDULER too. Multiply by `dp_size` to get system throughput.

```bash
# Common setup
export MODEL=<path to MoE model>
export RESULT_FILE=results.jsonl
export COMMON="--model-path $MODEL --trust-remote-code --disable-overlap-schedule \
               --mem-fraction-static 0.8 --input-len 10 --output-len 10 \
               --result-filename $RESULT_FILE"

# ===== 1. TP/TP — Attn TP + Exp TP (default baseline) =====
unset SGLANG_DEEPEP_BF16_DISPATCH SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK NVSHMEM_QP_DEPTH
python -m sglang.bench_one_batch $COMMON --run-name tp_tp \
    --tp-size 8 \
    --batch-size 8 64 512 2048 4096 \
    --cuda-graph-bs 8 64 512 2048 4096

# ===== 2. TP/EP — Attn TP + Exp EP (DeepEP, no DP attn) =====
# Note: max batch capped by SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK because
# with dp_size=1 the full batch lands on every rank's DeepEP dispatch.
export SGLANG_DEEPEP_BF16_DISPATCH=true
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512
export NVSHMEM_QP_DEPTH=2048
python -m sglang.bench_one_batch $COMMON --run-name tp_ep \
    --tp-size 8 --moe-a2a-backend deepep --deepep-mode auto \
    --batch-size 8 64 512 \
    --cuda-graph-bs 8 64 512

# ===== 3. DP/TP — Attn DP + Exp TP (classic v0.4 DeepSeek DP attention) =====
unset SGLANG_DEEPEP_BF16_DISPATCH SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK NVSHMEM_QP_DEPTH
python -m sglang.bench_one_batch $COMMON --run-name dp_tp \
    --tp-size 8 --dp-size 8 --enable-dp-attention --enable-dp-lm-head \
    --batch-size 1 8 64 256 512 \
    --cuda-graph-bs 1 8 64 256 512

# ===== 4. DP/EP — Attn DP + Exp EP (DP attn + DeepEP) =====
export SGLANG_DEEPEP_BF16_DISPATCH=true
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512
export NVSHMEM_QP_DEPTH=2048
python -m sglang.bench_one_batch $COMMON --run-name dp_ep \
    --tp-size 8 --dp-size 8 --enable-dp-attention --enable-dp-lm-head \
    --moe-a2a-backend deepep --deepep-mode auto \
    --batch-size 1 8 64 256 512 \
    --cuda-graph-bs 1 8 64 256 512
```

**Required `bench_one_batch.py` behavior** (main branch as of this writing):
1. `_maybe_prepare_mlp_sync_batch` must compute `attn_tp_size = tp_size // dp_size`, NOT hard-code 1. Otherwise TP/EP fails with `all_gather_into_tensor` shape mismatch because `require_mlp_sync` returns True (via `require_attn_tp_gather`) whenever `moe_a2a_backend != "none"`, even without DP attention.
2. Result rows should be written **incrementally** (one append per batch_size completion), not batched at end — otherwise a late crash (OOM, DeepEP dispatch cap, NCCL teardown hang) loses all prior data in the sweep.

**Practical failure modes to handle** in a driver script:
- **CUDA OOM**: decrement `--mem-fraction-static` by 0.05 (floor 0.60) and retry.
- **DeepEP dispatch cap**: prefill with `batch × input_len > num_max_dispatch_tokens_per_rank` requires `--deepep-mode auto` (routes prefill to NORMAL, no cap). `low_latency` alone fails on prefill.
- **Ranks hang at NCCL teardown**: kill leftover processes between runs:
  ```bash
  pkill -9 -f "sglang.bench_one_batch" || true
  nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ',' | xargs -r kill -9 || true
  sleep 5
  ```

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

Rank the 4 configs by decode system throughput at the target batch. The winner's advantage maps to a specific axis:

```
Which config wins at this equiv batch?
│
├── TP/TP wins (typical at small batch, few GPUs)
│   ├── DeepEP dispatch+combine > TP AllReduce for MoE?  → EP fixed cost dominates
│   ├── EP fused_moe slower than TP?                      → Poor GEMM utilization (small M)
│   ├── DP attention adds more overhead than savings?     → Too-small per-rank batch
│   └── Overhead: extra EP launches >> AR cost?           → Kernel-launch ceiling
│
├── DP/TP wins (typical at mid-large batch, DP-attention-capable models)
│   ├── TP attention >> DP attention time?                → Axis 3 win: batch/dp_size per rank
│   ├── MoE AllReduce still cheap?                        → AR not yet linear-growth bottleneck
│   ├── KV cache pressure relieved?                       → Partitioned KV (no replication)
│   └── Prefill throughput jump?                          → Fewer attention tokens/rank in extend
│
├── DP/EP wins (typical at very large batch, many GPUs)
│   ├── TP AllReduce >> DeepEP dispatch/combine for MoE?  → Axis 1 win: comm scaling crossover
│   ├── Attention time much lower than TP configs?        → DP attention savings on top
│   ├── EP fused_moe faster than TP fused_moe?            → Axis 2 win: full N GEMM shape
│   └── CV% of per-rank MoE compute < 10%?                → Routing is balanced
│
├── TP/EP wins (unusual — model not DP-capable, or DP broken)
│   ├── Model architecturally rejects DP attention?       → Only valid reason to pick this over DP/EP
│   └── Otherwise → something is wrong; DP/EP should dominate TP/EP
│
└── CLOSE (within 10% of top config) → Crossover region
    └── Run more batch sizes to pinpoint (e.g., 768, 1024, 1536)
```

**Anti-patterns to check first** (they produce misleading rankings):

1. **Per-scheduler vs system throughput**: `bench_one_batch` reports per-DP-rank. Multiply DP configs by `dp_size` before comparing. Forgetting this makes DP configs look 4-8× worse than they are.
2. **Unequal equiv batches**: `--batch-size 512` for TP/TP is NOT the same global batch as `--batch-size 512` for DP/EP with dp=8 (the latter is global=4096). Always translate to equiv batch before comparing.
3. **Dummy weights with EP**: routing becomes uniform, hiding the real-weights load imbalance problem. Always validate with real weights for production conclusions.
4. **Short prefill (`input_len=10`)**: dominated by kernel launch overhead, not compute. Prefill numbers are unreliable for MoE GEMM-shape conclusions; use decode or larger `input_len`.
5. **Cold-run noise on large prefills** (`bench_one_batch`): the FIRST invocation in a session runs cold (CUDA JIT cache, triton autotuning, thermal). Large prefills (batch ≥ 2048) can measure 40-70% slower on the cold run than on warm re-runs. Always re-run the largest batch once and compare; if the two differ by >10%, the first was cold and should be discarded. Note: `--chunked-prefill-size` does NOT affect `bench_one_batch` (scheduler-level setting, scheduler is not invoked by this tool — verified with identical prefill latencies at chunk=1024 vs chunk=8192).
6. **Summing kernel time across ranks** instead of using effective latency aggregation (see §Analysis Methodology). Summing measures total GPU-seconds of work, not wall-clock latency. Correct rule: `max` for compute categories, `min` for collectives. Our Qwen3-235B-half decode b=2048 analysis flipped from "MoE 18% of DP/EP gain" (summed) to "MoE 8%, Comm 63%" (effective) when the correction was applied.

---

## Further Confirmation Experiments

### 1. Pinpoint the Crossover

Run additional batch sizes between the last TP/TP-wins and first DP/EP-wins:
```bash
# If crossover is between 512 and 2048 on 8 GPUs:
--batch-size 768 1024 1536         # TP/TP and TP/EP
--batch-size 96 128 192            # DP/TP and DP/EP (equiv / 8)
```

### 1b. Isolate Which Axis Drove the Flip (DP vs EP contribution)

When DP/EP wins, both DP attention (Axis 3) and EP MoE (Axes 1+2) contribute. To split the gain:

- Run **DP/TP** at the same equiv batch → isolates the DP-attention contribution (same MoE as TP/TP, different attention).
- Run **TP/EP** at the same equiv batch → isolates the EP contribution (same attention as TP/TP, different MoE).
- If `DP/TP ≈ TP/TP` but `DP/EP >> TP/TP`, the win is mostly from EP (Axes 1+2).
- If `DP/TP >> TP/TP` and `DP/EP ≈ DP/TP`, the win is mostly from DP attention (Axis 3).
- If both beat TP/TP and DP/EP is best, both axes contribute.

This is the primary value of running all 4 configs instead of just 2.

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

### Qwen3-30B-A3B (8×A100, real weights, CUDA graphs) — 2-way comparison

| Equiv Batch | DP/EP vs TP/TP (decode) |
|---|---|
| 8 | 0.42× (TP/TP wins) |
| 64 | 0.49× |
| 512 | 0.76× |
| 2048 | **1.50×** (DP/EP wins) |
| 4096 | **1.37×** |

Crossover: ~1024 equiv batch.

### Qwen3-235B-A22B-half (8×A100, dummy weights, CUDA graphs) — 4-way comparison

All 4 configs run with `--input-len 10 --output-len 10`, `--load-format dummy`, `--mem-fraction-static 0.8`. **Each batch measured twice per invocation** (duplicated `--batch-size` list); best-of-2 latency reported to mitigate cold-cache noise.

**Decode — system throughput ratio vs TP/TP**:

| Equiv Batch | TP/TP | TP/EP | DP/TP | DP/EP | Winner |
|---:|---:|---:|---:|---:|---|
| 8    | 1.00× | 0.89× | 0.68× | 0.38× | TP/TP |
| 64   | 1.00× | 0.77× | 0.84× | 0.60× | TP/TP |
| 512  | 1.00× | 0.89× | 0.96× | 0.88× | TP/TP |
| 2048 | 1.00× | 0.90× | 1.17× | **1.28×** | **DP/EP** |

**Prefill — system throughput ratio vs TP/TP** (input_len=10):

| Equiv Batch | TP/TP | TP/EP | DP/TP | DP/EP |
|---:|---:|---:|---:|---:|
| 8    | 1.00× | 0.98× | 0.77× | 0.25× |
| 64   | 1.00× | 0.98× | 0.79× | 0.50× |
| 512  | 1.00× | 0.89× | 1.07× | 0.99× |
| 2048 | 1.00× | 0.86× | 1.08× | **1.45×** |

**Axis decomposition at batch 2048 (effective-latency kernel analysis)**:

| Axis | Metric | Prefill b=2048 | Decode b=2048 |
|---|---|---:|---:|
| 1. Communication | min-rank comm savings | 18% of DP/EP gain | **63%** of DP/EP gain |
| 2. MoE GEMM shape | max-rank MoE savings | **69%** | 8% |
| 3. DP attention | max-rank attn savings | 13% | 33% |

**The dominant axis flips between prefill and decode**. At prefill, large-M GEMM with full-N experts yields massive MoE reduction (422→252 ms max-rank). At decode, per-rank MoE work is similar across configs; the win is almost entirely from cheaper DeepEP comm.

**8-GPU takeaways:**
- **TP/TP wins at batch ≤ 512**. DP/EP's ~5–10ms DeepEP fixed cost dominates at small batch.
- **Crossover at batch ~1024** (bracketed by 512→0.88× and 2048→1.28× decode).
- **DP/TP is a strong middle ground**: 1.17× decode + 1.08× prefill at 2048, without DeepEP.
- **TP/EP (AllReduce-based EP) is uniformly worse than TP/TP** — each rank iterates `E/ep_size` experts over the full batch with masking; extra MoE work is not offset by comm savings. Do not use.
- **Prefill at batch 2048** is DP/EP's strongest win (1.45×); decode at 1.22× (effective latency) or 1.28× (throughput).
- **DeepEP prefill variance**: on second measurement of the same batch, DP/EP prefill can regress 2–4× (observed 0.52s → 2.34s at batch 256). Run at least 2 trials and report best-of-2 for DP/EP.

Data/scripts: `~/qwen235b_4way_analysis/` (`results_v2.jsonl`, `analyze.py`, `run_all_v2.sh`, `report.md`) + `~/qwen235b_4way_analysis/analysis/` (nsys profiles, effective-latency analysis).

### Qwen3-30B-A3B (4×A100, real weights, CUDA graphs) — 4-way comparison

System decode throughput ratio vs TP/TP baseline (higher = faster):

| Equiv Batch | TP/TP | TP/EP | DP/TP | DP/EP | Winner |
|---:|---:|---:|---:|---:|---|
| 4    | 1.00× | 0.35× | 0.80× | 0.36× | TP/TP |
| 32   | 1.00× | 0.42× | 0.85× | 0.42× | TP/TP |
| 128  | 1.00× | 0.50× | 0.90× | 0.51× | TP/TP |
| 512  | 1.00× | 0.62× | 0.90× | 0.74× | TP/TP |
| 2048 | 1.00× | —     | **1.16×** | 0.99× | **DP/TP** |

System prefill throughput ratio vs TP/TP baseline (input_len=10):

| Equiv Batch | TP/TP | TP/EP | DP/TP | DP/EP |
|---:|---:|---:|---:|---:|
| 4    | 1.00× | 0.47× | 0.74× | 0.73× |
| 32   | 1.00× | 0.53× | 0.79× | 0.63× |
| 128  | 1.00× | 0.48× | 0.75× | 0.64× |
| 512  | 1.00× | 0.75× | **1.10×** | 0.97× |
| 2048 | 1.00× | —     | **1.07×** | **1.07×** |

TP/EP at 2048 not collected (run was skipped; would be informative to add).

**4-GPU takeaways:**
- **TP/TP wins ≤ 512** equiv batch. On 4 GPUs the TP AllReduce fast path is cheap and TP's narrow-N GEMM penalty is only 4× (vs 8× on 8 GPUs), so the overhead of every alternative dominates.
- **DP/TP wins at 2048** (decode 1.16×, prefill 1.07×). DP attention cuts per-rank attention/KV work 4×; MoE AllReduce stays cheap because experts stay TP-sharded.
- **DP/EP ties at 2048** but doesn't overtake on 4 GPUs. DeepEP fixed cost ~5ms still hurts.
- **TP/EP loses everywhere** on 4 GPUs within the tested range. TP/EP uses AllReduce-based EP (`--ep-size N`, no DeepEP); each rank iterates `E/ep_size` experts over the full batch, adding MoE work without comm savings.
- **Production rule of thumb for 4-GPU Qwen3-30B-A3B**: use TP/TP for interactive/low-latency, switch to DP/TP at batch ≥ 512, DP/EP only makes sense once batch ≥ 4096 or scaling to 8+ GPUs.

---

## Existing Analysis Scripts & Data

| Path | Description |
|---|---|
| `~/qwen-30b-analysis/report.md` | Full analysis report (30B real weights, 8×A100, 2-way) |
| `~/qwen-30b-analysis/analyze_multirank.py` | Multi-rank trace parser (per-rank breakdown + load imbalance) |
| `~/qwen-30b-analysis/multirank_analysis_raw.json` | Raw per-rank data |
| `~/qwen-30b-analysis/torch_tp/` | TP per-rank traces (batch=64, 2048) |
| `~/qwen-30b-analysis/torch_ep/` | EP per-rank traces (batch=8, 256 per-rank) |
| `~/sglang_profile_ep_tp/analyze_traces.py` | Single-rank trace parser (with CUDA graphs) |
| `~/sglang_profile_ep_tp/analysis_report.md` | 235B-half analysis report |
| `~/qwen30b_4gpu_analysis/report.md` | **4-way comparison on 4×A100 Qwen3-30B-A3B** (TP/TP, TP/EP, DP/TP, DP/EP) |
| `~/qwen30b_4gpu_analysis/analyze.py` | 4-way analysis script: parses `results.jsonl` → tables + ratios + plots |
| `~/qwen30b_4gpu_analysis/run_all.sh` | 4-way driver script with OOM retry + GPU cleanup |
| `~/qwen30b_4gpu_analysis/results.jsonl` | Raw per-config/batch measurements |
| `~/qwen235b_4way_analysis/report.md` | **4-way comparison on 8×A100 Qwen3-235B-A22B-half** (dummy weights, best-of-2) |
| `~/qwen235b_4way_analysis/run_all_v2.sh` | 8-GPU 4-way driver with duplicate-batch warmup |
| `~/qwen235b_4way_analysis/analyze.py` | Best-of-2 analysis script for bench_one_batch latency sweep |
| `~/qwen235b_4way_analysis/analysis/report.md` | Kernel root-cause analysis (nsys, effective-latency) |
| `~/qwen235b_4way_analysis/analysis/scripts/effective_latency.py` | max-compute / min-comm aggregation reference implementation |
| `~/qwen235b_4way_analysis/results_v2.jsonl` | Raw 32 rows (4 configs × 4 batches × 2 passes) |
