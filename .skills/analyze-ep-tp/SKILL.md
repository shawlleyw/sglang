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
- **DeepEP data-movement CV% has two components** (see DeepEP caveat above): (1) arrival-time wait (same as AR) + (2) genuine volume imbalance when routing is skewed. Dummy-weight CV ≈ (1) only; real-weight CV = (1)+(2) compounded.
- **MoE CV% in EP configs at large batch** — genuine expert-routing imbalance. With dummy weights NOT near-uniform — see Anti-pattern #3 below. With real weights typically 5–30% depending on top_k and prompt diversity.
- **DP attention CV% >2%** usually indicates `dp_gather`/`dp_scatter` sync leaking into the attention measurement window, not compute asymmetry (each DP rank has identical token count by construction).

### ⚠️ Critical: barrier kernels' nsys duration ≠ pure kernel cost

**`deep_ep::intranode::cached_notify_combine` and similar `notify_*` kernels are NOT data-movement kernels — they are cross-rank barriers.** Per upstream source ([DeepEP commit `b306af06`](https://github.com/deepseek-ai/DeepEP/blob/b306af06afd412c88e51e71802951606e40b7358/csrc/kernels/legacy/intranode.cu#L626-L703)), block 0 does a global barrier (zeros metadata, second barrier), other blocks scan/reconstruct `send_head`. The barrier uses `atomicAdd_system` + spin-wait, which counts as "GPU active" but is actually busy-waiting for the slowest rank to arrive.

This means **nsys with `--cuda-graph-trace=node` reports per-kernel duration that includes barrier wait time** — not just compute. A reported "8 ms kernel" can be 15 µs of actual work + ~8 ms of busy-wait absorbing upstream compute imbalance.

**Diagnostic**: query per-call MIN/MAX duration *across ranks within a single barrier call* (not per-rank averages — those are uniform and misleading because all ranks unblock together):

```sql
WITH ranked AS (
  SELECT k.deviceId, k.start, k.end-k.start AS dur,
         ROW_NUMBER() OVER (PARTITION BY k.deviceId ORDER BY k.start) AS layer_idx
  FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id
  WHERE s.value LIKE '%cached_notify_combine%'
)
SELECT layer_idx, MIN(dur)/1000.0 AS min_us, MAX(dur)/1000.0 AS max_us
FROM ranked GROUP BY layer_idx ORDER BY layer_idx LIMIT 10;
```

If `min ≪ max` within the same barrier call, it is a barrier absorbing upstream imbalance, NOT a slow kernel. **The min duration is the pure kernel work**; the max duration is the slowest waiter.

For gpt-oss-120b-bf16 prefill at bs=2048 (real weights):
- min per-call: 15-20 µs (pure kernel work — rank that arrives last)
- max per-call: 14,000-18,000 µs (barrier wait — fast ranks waiting)
- avg per-rank: 6,800-8,400 µs (misleading "kernel cost" — barrier-amplified)

Same kernel with dummy weights (extreme imbalance):
- min: 15 µs (rank 0, the slowest, does all the work)
- max: 27,000 µs (light ranks wait far longer)

**Inverse correlation pattern** (signature of barrier amplification):
- Rank with HEAVIEST upstream compute → LOWEST barrier kernel duration (arrives last, no wait)
- Rank with LIGHTEST upstream compute → HIGHEST barrier kernel duration (waits longest)

If you see this inverse correlation in your trace, **the "slow kernel" is a sync point, not a real bottleneck**. Fix the upstream load imbalance (via EPLB or routing-aware batching), not the comm kernel itself.

The same caveat applies (less dramatically) to `deep_ep::internode_ll::combine` in decode and to `ncclDevKernel_AllGather_RING_LL` (the lm_head DP gather): all are sync points where upstream skew leaks into reported kernel duration.

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

**⚠️ nsys overhead is NOT symmetric across configs.** nsys's per-kernel CUPTI tracing has a small fixed cost per launched kernel. DeepEP-heavy configs (DP/EP, TP/EP-DeepEP) launch many small kernels per layer (`deep_ep::dispatch`, `combine`, `cached_notify_combine`, `notify_dispatch`, ...) — order of 10+ per layer × tens of layers. AllReduce-only configs (TP/TP, DP/TP) launch ~2 large NCCL kernels per layer. Net effect on gpt-oss-120b prefill at equiv batch 2048 measured under matched nsys flags:

| Config | bench-only warm prefill | nsys-attached warm prefill | nsys overhead |
|---|---:|---:|---:|
| TP/TP | 587.6 ms | 566.9 ms | ~0% (slightly faster, within noise) |
| DP/EP | 411.1 ms | 485.4 ms | **~+18%** |

Consequence: **the kernel-level effective-latency ratio measured under nsys (1.23× here) is compressed relative to the bench-only wall-clock ratio (1.43× here)** by ~16%. The qualitative axis decomposition is reliable (which categories favor which config and roughly by how much), but the absolute "DP/EP is X× faster" number derived purely from nsys totals will under-state the production speedup whenever DeepEP is in play. Always quote the bench-only number as the headline; use nsys for the WHY breakdown only. If you need a tighter quantitative match, use `torch.profiler` (lower per-kernel overhead) or scope nsys with `-t cuda` plus careful capture-range selection.

### Step 4: Analyze Traces

Use `analyze_multirank.py` (in `~/qwen-30b-analysis/`) or `analyze_traces.py` (in `~/sglang_profile_ep_tp/`).

Kernel classification rules:
- **Attention**: flashinfer kernels, `_fwd_kernel`, `_fwd_grouped_kernel_stage*`, cutlass/ampere GEMMs (QKV/O projections), RoPE (`BatchQKApplyRotary*`)
- **Communication (data movement)**: `cross_device_reduce`, `ncclDevKernel_AllReduce`, `ncclDevKernel_AllGather`, `deep_ep::intranode::dispatch`, `deep_ep::intranode::combine` (prefill), `deep_ep::internode_ll::dispatch`, `deep_ep::internode_ll::combine` (decode)
- **Communication (sync/notify barriers)** — these are NOT data-movement kernels; they're cross-rank barriers: `deep_ep::intranode::cached_notify_combine` (prefill, dominant cost — see caveat in §Analysis Methodology), `deep_ep::intranode::notify_dispatch`. Treat per-kernel duration with care — see "barrier kernels" caveat below.
- **MoE Compute**: `fused_moe_kernel`, `topkGatingSoftmax`, `moe_align_block_size`, `silu_and_mul`, `_silu_and_mul_masked_kernel`, `swiglu_with_alpha_and_limit*` (gpt-oss clamped SwiGLU), `triton_poi_fused_*` (torch.compile-generated activation), `moe_sum_reduce`, `count_and_sort_expert_tokens`, `_fwd_kernel_ep_scatter*`, `_fwd_kernel_ep_gather`
- **Other**: RMSNorm, `FusedAddRMSNormKernel`, elementwise, fills, copies

**Watch out for `triton_poi_fused_*`**: these are torch.compile-Inductor-generated kernels (e.g., gpt-oss's clamped SwiGLU). They can have pathologically bad launch configs for certain shapes (16-32× thread oversubscription). See §gpt-oss-120b reference for the specific bug.

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

**gpt-oss-120b-bf16, real weights, bs=2048 prefill**: ~30% CV. Per-rank fused_moe totals ranged 156–208 ms (rank 3 = busiest at 208 ms, rank 2 = lightest at 156 ms). Top-k=4 amplifies imbalance vs Qwen3 top-k=8 because each token visits fewer experts → less averaging.

TP has zero compute imbalance by construction (all ranks process identical data).

### Diagnostic: how MoE imbalance propagates to comm "slowdown"

When MoE rank-imbalance is present, downstream sync barriers (`cached_notify_combine`, AllGather, etc.) absorb the wait time. To diagnose whether your "slow comm kernel" is real or just barrier-amplified imbalance:

1. Query per-rank fused_moe_kernel total time → identifies the SLOWEST upstream rank.
2. Query per-rank `cached_notify_combine` (or the suspect comm kernel) average duration.
3. If the SLOWEST upstream rank has the LOWEST comm-kernel duration → barrier amplification confirmed. The fix is upstream balancing, not the comm kernel.

For gpt-oss-120b-bf16 prefill at bs=2048 (real weights, after the two fixes in §gpt-oss-120b reference):
| rank | MoE total ms | cached_notify_combine avg µs |
|---:|---:|---:|
| 0 | 160 (lightest) | **8,215 (highest — waits longest)** |
| 3 | 208 (heaviest) | **6,802 (lowest — barely waits)** |

The inverse correlation between MoE and notify is the smoking gun.

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
3. **Dummy weights produce EXTREME imbalance, not uniform routing** (counter-intuitive!): untrained gate_proj weights produce near-identical activations across diverse inputs → all tokens stampede to the same top-k experts → some ranks idle while one does all MoE work. We measured **438× MoE imbalance** (rank 0 = 777 ms, rank 5 = 1.66 ms) with `--load-format dummy` on gpt-oss-120b. Real weights produce ~30% imbalance. Conclusion: **dummy weights are NOT a uniform-routing baseline** — never use them to "isolate kernel cost from imbalance". Use real weights to validate production-relevant conclusions.
4. **Short prefill (`input_len=10`)**: dominated by kernel launch overhead, not compute. Prefill numbers are unreliable for MoE GEMM-shape conclusions; use decode or larger `input_len`.
5. **Cold-run noise on large prefills** (`bench_one_batch`): the FIRST invocation in a session runs cold (CUDA JIT cache, triton autotuning, thermal). Large prefills (batch ≥ 2048) can measure 40-70% slower on the cold run than on warm re-runs. Always re-run the largest batch once and compare; if the two differ by >10%, the first was cold and should be discarded. Note: `--chunked-prefill-size` does NOT affect `bench_one_batch` (scheduler-level setting, scheduler is not invoked by this tool — verified with identical prefill latencies at chunk=1024 vs chunk=8192).
6. **Summing kernel time across ranks** instead of using effective latency aggregation (see §Analysis Methodology). Summing measures total GPU-seconds of work, not wall-clock latency. Correct rule: `max` for compute categories, `min` for collectives. Our Qwen3-235B-half decode b=2048 analysis flipped from "MoE 18% of DP/EP gain" (summed) to "MoE 8%, Comm 63%" (effective) when the correction was applied.
7. **Confusing nsys per-kernel duration with pure kernel cost for sync kernels** (see §Analysis Methodology "barrier kernels" caveat). `cached_notify_combine`, `internode_ll::combine`, AllGather around lm_head, etc. all report duration that includes barrier wait time. A "8 ms cached_notify_combine" can be 15 µs work + 8 ms wait. Diagnostic: per-call min/max across ranks. Inverse correlation between rank's upstream MoE work and barrier kernel duration = barrier amplification, not slow kernel.
8. **Synthetic random IDs from `bench_one_batch` only sample vocab[0:10000]**: `prepare_synthetic_inputs_for_latency_test` uses `np.random.randint(0, 10000)` regardless of model vocab size. For models with vocab ≥ 50k (most modern models), this samples only 5–20% of the vocabulary, mostly common subwords. Activations and routing patterns don't reflect realistic distributions, which biases MoE-routing measurements. **Use the new `--dataset-name sharegpt --dataset-path /path/to/ShareGPT_V3_unfiltered_cleaned_split.json` flag** added to `bench_one_batch.py`; the loader is shared with `bench_serving.get_dataset` and pads/truncates each sampled prompt to `--input-len`. All tp_ranks see identical inputs (deterministic `random.seed(0)` save/restore around the dataset load to prevent module-level shuffle divergence across `multiprocessing` workers). With real prompts on gpt-oss-120b at equiv batch 2048, DP/EP prefill went from 0.90× (synthetic baseline) to **1.43× TP/TP** — see §gpt-oss-120b reference (sharegpt subsection) for the full headline + kernel breakdown. The original handoff is at `~/bench_one_batch_dataset_input.md`.

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

⚠️ **Dummy weights do NOT produce uniform routing — they produce EXTREME routing imbalance** (counter-intuitive but confirmed empirically).

Untrained `gate_proj` weights from `torch.empty` / random init produce near-identical activation magnitudes across diverse inputs (no carefully-tuned variance scaling like a trained model has). After 36 layers, all tokens converge to similar gate-projection outputs → top-k routing always picks the same experts → all tokens stampede to a few ranks. Measured imbalance:

- Qwen3 dummy weights: routing-imbalance hidden by top_k=8 averaging — moderate effect
- gpt-oss-120b dummy weights: **438× MoE compute imbalance** at prefill bs=2048 (rank 0 = 777 ms vs rank 5 = 1.66 ms). With dummy weights, decode at bs=256 saw rank 0 = 64 ms vs rank 5 = 1.7 ms → **8× imbalance**.

Real weights produce ~30% imbalance (rank-busiest 30% slower than rank-lightest), which is realistic for production.

**Implications**:
- **Don't use dummy weights to "isolate kernel cost from imbalance"** — you'll get the opposite of uniform routing.
- **Always validate with real weights for production conclusions.**
- **For uniform-routing baselines**, you'd need explicit hash-based or round-robin routing in the model code, not just dummy weights.

This was the experiment that confirmed the load-imbalance hypothesis for gpt-oss-120b's `cached_notify_combine` 8 ms wait: with dummy weights, the inverse correlation between MoE compute and barrier kernel duration became extreme (rank 0 with most MoE work had 16 µs notify; light ranks had 22 ms notify). Real weights showed the same pattern with smaller magnitude.

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

### gpt-oss-120b-bf16 (8×A100, real weights, CUDA graphs) — 2-way comparison + two bugs found

**Model shape**: 36 MoE layers (hybrid full + sliding-window attention with sinks), `intermediate_size=2880`, `num_experts=128`, **top_k=4** (vs Qwen3 top-k=8 — half the per-token expert work). On 8 GPUs: TP/8 → N=360 (less pathological than Qwen3-30B's N=96), EP/8 → 16 experts/rank with full N=2880.

#### Bug 1 found and fixed: missing masked variant of clamped SwiGLU

- **File**: `python/sglang/srt/layers/moe/moe_runner/triton.py:213` (sglang main, before fix)
- **Symptom**: For Qwen3 (vanilla `silu`, `gemm1_alpha is None`), the runner uses `silu_and_mul_masked_fwd` which only processes valid tokens per expert. For gpt-oss (clamped SwiGLU with `gemm1_alpha=1.702`), no masked variant existed — fell to unmasked path. The unmasked path processed the FULL padded buffer (`E_local=16 × max_tokens=4096 = 65,536 rows` even when only 1024 were real, 64× oversubscription).
- **Diagnostic signature**: `triton_poi_fused_add_clamp_mul_sigmoid_0` (torch.compile-generated kernel) showed grid=`[368640, 1, 1]` block=`[256, 1, 1]` = 94M threads for 5.9M elements (16-32× oversubscription). 700 µs/call regardless of batch size (input-size-independent — clue that it processed padded buffer).
- **Fix**: added `_swiglu_with_alpha_and_limit_masked_kernel` Triton kernel + masked branch in runner (`gemm1_alpha is not None and masked_m is not None` → masked path).
- **Impact**: kernel time 700 µs/call → 24 µs/call (28× speedup); decode bs=2048 throughput **23,353 → 33,597 tok/s** (+44%); DP/EP now wins TP/TP at decode bs=2048 (1.18×).

#### Bug 2 found and fixed: Inductor over-parallelization of unmasked swiglu

- **File**: `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`
- **Symptom**: `@torch.compile`-decorated `swiglu_with_alpha_and_limit` generates a kernel for one shape, reuses it for all shapes. For TP-shaped `[8192, 720]`: 15 µs/call (good). For DP/EP-prefill-shaped `[1024, 5760]` (or padded `[65536, 5760]`): 800+ µs/call. Same kernel binary, different launch config result. Inductor's dynamic-shape heuristic generates shape-generic kernels with bad block tiling for wide-N narrow-M tensors.
- **Fix**: replaced `@torch.compile` with hand-written Triton kernel using grid=`(M, ceil(N/BLOCK_N))`, `BLOCK_N=512`, `num_warps=4` — proper sizing.
- **Impact**: prefill performance recovered (mask-only fix had regressed prefill -14%); within 2-6% of pre-fix baseline.

#### Post-fix headline (DP/EP system tok/s, bs grid 8/64/512/2048)

**Decode**:
| Equiv Batch | TP/TP | DP/EP (post-fix) | Ratio |
|---:|---:|---:|---:|
| 8    | 1,034  | 400    | 0.39× |
| 64   | 4,743  | 2,545  | 0.53× |
| 512  | 15,583 | 14,185 | 0.91× |
| **2048** | **28,591** | **33,597** | **1.18× (DP/EP wins)** |

**Prefill**:
| Equiv Batch | TP/TP | DP/EP (post-fix) | Ratio |
|---:|---:|---:|---:|
| 8    | 1,750  | 1,204  | 0.69× |
| 64   | 13,646 | 9,129  | 0.67× |
| 512  | 35,063 | 30,949 | 0.88× |
| 2048 | 38,812 | 35,122 | **0.90× (DP/EP loses)** |

#### Remaining unfixed bottleneck — `cached_notify_combine` barrier amplification

DP/EP prefill at bs=2048 is 0.90× of TP/TP (DP/EP loses), in contrast to Qwen3-235B-half which is 1.45× (DP/EP wins). The dominant prefill kernel is `deep_ep::intranode::cached_notify_combine<8>` averaging **8.2 ms/call × 36 layers = 295 ms per prefill step** (50% of total prefill kernel time on rank 0).

**Per the §Analysis Methodology "barrier kernels" caveat**, this 8.2 ms is NOT pure kernel cost. Per-call min/max analysis revealed:
- Pure kernel work: **15-20 µs** (the rank that arrives last at the barrier)
- Slowest waiter: **14,000-18,000 µs** (fast ranks waiting for the slow one)
- Inverse correlation observed: rank with heaviest MoE (rank 3 = 208 ms) had LOWEST notify duration (6.8 ms — barely waits); rank with lightest MoE (rank 0 = 160 ms) had HIGHEST notify duration (8.2 ms — waits longest)

**Root cause**: ~30% MoE compute imbalance from real-weight expert-routing skew, amplified by `cached_notify_combine`'s `atomicAdd_system`-based barrier (per [DeepEP source](https://github.com/deepseek-ai/DeepEP/blob/b306af06afd412c88e51e71802951606e40b7358/csrc/kernels/legacy/intranode.cu#L626-L703)). Matches [DeepEP issue #575](https://github.com/deepseek-ai/DeepEP/issues/575).

**Why this hits gpt-oss harder than Qwen3**:
- Top_k=4 (vs Qwen3's top_k=8) → half the per-token expert work → smaller MoE GEMM savings from EP (Axis 2 advantage halved)
- DeepEP NORMAL `cached_notify_combine` cost is fixed per-layer regardless of top_k
- Net: gpt-oss's smaller MoE savings can't offset the comm overhead, while Qwen3's larger savings can

**Fix direction (not yet implemented)**: EPLB (expert load balancing) to reduce upstream MoE imbalance. The `cached_notify_combine` kernel itself is fast (~15 µs); the remaining 8 ms is barrier wait absorbing imbalance. Fix is upstream, not at the kernel level.

#### Post-fix headline with sharegpt (real-prompt) inputs

The `cached_notify_combine` barrier amplification IS the remaining bottleneck, but the magnitude of that bottleneck depends on upstream MoE imbalance, which depends on input distribution. With `np.random.randint(0, 10000)` inputs (anti-pattern #8), only the bottom ~6% of gpt-oss's vocab is sampled, producing artificially skewed routing. With real ShareGPT V3 prompts (via the new `--dataset-name sharegpt` flag), routing is more uniform and the barrier wait shrinks. Same hardware, same weights, same nsys recipe — only the input source changes.

**Per-rank `fused_moe_kernel` total at DP/EP prefill bs=256 (equiv 2048)**:

| Input source | per-rank fused_moe_kernel (ms) | mean | CV% | max/min |
|---|---|---:|---:|---:|
| Synthetic | 160, 191, 156, 208, 157, 180, 169, 196 | 177.2 | **10.34%** | 1.33 |
| ShareGPT  | 140, 164, 158, 141, 160, 131, 154, 150 | 149.7 | **7.26%** | 1.26 |

Sharegpt cuts CV% by ~30% and total MoE compute by ~16%. The reduction in upstream imbalance reduces barrier-wait at `cached_notify_combine`.

**Headline throughput (8×A100, real weights, best-of-2)**:

| Equiv Batch | TP/TP decode | DP/EP decode | DP/EP / TP/TP | TP/TP prefill | DP/EP prefill | DP/EP / TP/TP |
|---:|---:|---:|---:|---:|---:|---:|
|    8 |    943 |    450 | 0.48× |  1,826 |  1,179 | 0.65× |
|   64 |  3,243 |  2,776 | 0.86× | 12,068 |  6,687 | 0.55× |
|  512 | 11,120 | 13,673 | **1.23×** | 28,539 | 29,950 | 1.05× |
| 2048 | 24,662 | 34,965 | **1.42×** | 34,852 | 49,818 | **1.43×** |

**Key finding — the production recommendation flips at prefill bs=2048**:

| Configuration | Synthetic baseline (Apr 27) | Sharegpt (Apr 29) |
|---|---:|---:|
| DP/EP / TP/TP prefill @ equiv 2048 | 0.90× (DP/EP loses) | **1.43× (DP/EP wins)** |
| DP/EP / TP/TP decode @ equiv 2048 | 1.18× | **1.42×** |

**Effective-latency axis decomposition (sharegpt, equiv 2048 prefill, max compute + min comm rule)**:

| Axis | TP/TP ms | DP/EP ms | Δ | Reading |
|---|---:|---:|---:|---|
| Attention      |  83.3 |  66.9 | **−16.4** | Axis 3: DP attention reduces per-rank attn batch (256 vs 2048) |
| MoE            | 348.3 | 181.0 | **−167.3** | Axis 2: EP full-N GEMM + per-rank routing → ~half critical-path MoE time |
| Comm (min-rank)|  84.0 | 170.1 | +86.1 | DeepEP fixed cost (dispatch + combine + cached_notify_combine) > AllReduce |
| **Total (effective)** | **550.0** | **447.5** | **−102.5** | DP/EP wins; magnitude is nsys-overhead-compressed (skill §Step 3 caveat) |

**Crossover ≈ equiv batch 256–512** with sharegpt (TP/TP wins below, DP/EP wins above), down from "DP/EP wins decode at 2048 only" with synthetic.

**Production rule of thumb update for gpt-oss-120b on 8×A100 with realistic conversational inputs**: use TP/TP for batch ≤ 256, switch to DP/EP at batch ≥ 512 — DP/EP wins both prefill and decode there (1.42–1.43× at equiv 2048).

Data/scripts: `~/gpt_oss_ep_tp_analysis/results_sharegpt.jsonl` (32 rows; only `tp_tp` / `dp_ep` rows used in the headline), `analyze_sharegpt.py`, `analyze_nsys_sharegpt.py`, `report_sharegpt.md`, and nsys traces `dp_ep_bs256_prefill_sharegpt.{nsys-rep,sqlite}` + `tp_tp_bs2048_prefill_sharegpt.nsys-rep`.

#### Methodology lessons learned

1. **Always verify nsys per-kernel time with per-call min/max** when sync barriers (`*_notify_*`, AllGather) are involved. The reported duration may be barrier wait, not work.
2. **Dummy weights worsen MoE imbalance** (untrained gate_proj → all tokens to same experts → 438× imbalance for gpt-oss). Don't use them as a uniform-routing baseline.
3. **bench_one_batch's synthetic inputs (`np.random.randint(0, 10000)`) only sample 5% of the vocab** for modern models, biasing MoE routing measurements. Realistic comparisons need real prompts (sharegpt) — see anti-pattern #8 for the now-implemented `--dataset-name sharegpt` flag. On gpt-oss-120b at equiv 2048 prefill, real prompts cut MoE CV% by ~30% (10.34% → 7.26%) and flipped DP/EP from losing prefill (0.90×) to winning decisively (1.43×).
4. **nsys profiling overhead is asymmetric across configs.** DeepEP's many-small-kernels pattern pays per-kernel CUPTI overhead far more than AllReduce-only configs. Measured impact on gpt-oss-120b prefill at equiv 2048: bench-only DP/EP 411 ms → nsys-attached 485 ms (+18%); TP/TP unchanged. The kernel-breakdown ratio understates the production speedup; always use bench-only for headline numbers and nsys for the WHY breakdown only — see §Step 3 for the full caveat.

Data/scripts: `~/gpt_oss_ep_tp_analysis/` (results.jsonl, run_all.sh, profile_nsys*.sh, nsys_traces/, torch_profiler_traces/) + bug report at `~/gpt_oss_dpep_prefill_comm_bug.md` + bench_one_batch dataset-input handoff at `~/bench_one_batch_dataset_input.md`.

---

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
| `~/gpt_oss_ep_tp_analysis/run_all.sh` | **8-GPU 2-way driver for gpt-oss-120b-bf16** (TP/TP, DP/EP, real weights) |
| `~/gpt_oss_ep_tp_analysis/profile_nsys.sh` | nsys decode profile capture (cudaProfilerApi-scoped) |
| `~/gpt_oss_ep_tp_analysis/profile_nsys_prefill.sh` | nsys prefill profile capture |
| `~/gpt_oss_ep_tp_analysis/profile_torch.sh` | torch.profiler decode capture (eager mode) |
| `~/gpt_oss_ep_tp_analysis/profile_torch_cudagraph.sh` | torch.profiler with CUDA graphs (sees only kernels outside graph) |
| `~/gpt_oss_ep_tp_analysis/results.jsonl` | gpt-oss-120b post-fix headline sweep (16 rows) |
| `~/gpt_oss_ep_tp_analysis/results.prefix.jsonl` | gpt-oss-120b pre-fix baseline |
| `~/gpt_oss_ep_tp_analysis/nsys_traces/` | gpt-oss nsys traces (decode + prefill, both configs) |
| `~/gpt_oss_ep_tp_analysis_dummy_weights/` | Same setup with `--load-format dummy` — confirms 438× MoE imbalance from random init (don't use for production conclusions) |
| `~/gpt_oss_ep_tp_analysis/results_sharegpt.jsonl` | **gpt-oss + ShareGPT V3 inputs**: 4-config sweep at equiv batches 8/64/512/2048 (only `tp_tp` and `dp_ep` rows used in the analysis; the other two collected as a byproduct of an aborted 4-way attempt) |
| `~/gpt_oss_ep_tp_analysis/run_4way_sharegpt.sh` | 4-way driver script with `--dataset-name sharegpt`; comment-out blocks if only running 2-way |
| `~/gpt_oss_ep_tp_analysis/analyze_sharegpt.py` | 2-way headline analysis (best-of-2 → sys throughput tables → ratios → markdown report) |
| `~/gpt_oss_ep_tp_analysis/analyze_nsys_sharegpt.py` | nsys per-rank kernel categorization → effective latency → axis decomposition → markdown |
| `~/gpt_oss_ep_tp_analysis/report_sharegpt.md` | **Headline + breakdown report**: TP/TP vs DP/EP with sharegpt, includes the bench-vs-nsys reconciliation block |
| `~/gpt_oss_ep_tp_analysis/nsys_traces/{tp_tp_bs2048,dp_ep_bs256}_prefill_sharegpt.{nsys-rep,sqlite}` | nsys traces for the 2-way prefill comparison at equiv batch 2048 |
| `~/gpt_oss_ep_tp_analysis/nsys_traces/report_nsys_sharegpt.md` | Per-rank kernel-time tables, effective latency per category, CV% by category, top 5 kernels per config |
| `~/gpt_oss_dpep_prefill_comm_bug.md` | **Bug report**: `cached_notify_combine` barrier amplification + `internode_ll::combine` analysis. For another agent to fix at the DeepEP layer. |
| `~/bench_one_batch_dataset_input.md` | **Handoff (now implemented)**: original spec for adding real-dataset input support to `bench_one_batch`. The `--dataset-name sharegpt` flag is live in `python/sglang/bench_one_batch.py` (see anti-pattern #8). |
