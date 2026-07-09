# Parallelism Configuration for MoE Inference

## 1. Introduction

Mixture-of-Experts (MoE) models expose two independent parallelism choices for multi-GPU inference: how to parallelize **attention** and how to parallelize **expert computation**. Each choice is binary — Tensor Parallel (TP) or Data Parallel (DP) for attention, TP or Expert Parallel (EP) for experts — yielding four combinations. This document analyzes all four, explains why only two are practical, and provides trace-level evidence for the recommendation.

Throughout this document we use **Qwen3-235B-A22B-half on 8×A100-80GB** as a running example. The model has 47 MoE layers, 128 experts with top-8 routing, and `moe_intermediate_size=1536`. Results were collected with `bench_one_batch` (CUDA graphs ON, `--load-format dummy`) and nsys kernel profiling. The qualitative conclusions generalize to other MoE models (e.g., DeepSeek-V2/V3, Qwen3-30B-A3B) on similar hardware; absolute numbers and crossover points will shift with model size, GPU count, and interconnect bandwidth.

## 2. The 4 Configurations

| Tag | Attention | Experts | CLI flags (`N`=num_gpus) | MoE comm |
|---|---|---|---|---|
| **TP/TP** | TP | TP | `--tp N` | AllReduce |
| **TP/EP** | TP | EP | `--tp N --ep-size N` | AllReduce |
| **DP/TP** | DP | TP | `--tp N --dp N --enable-dp-attention` | AllReduce + dp_gather/scatter |
| **DP/EP** | DP | EP | `--tp N --dp N --moe-a2a-backend deepep --deepep-mode auto` | DeepEP all-to-all |

**TP attention** replicates all tokens across ranks after a post-attention AllReduce. Every rank sees the full batch.

**DP attention** partitions the batch across ranks. Each rank processes `batch/dp_size` tokens through attention, norms, and KV cache — then exchanges tokens before the MoE stage.

**TP experts** shard every expert's weight matrices along the intermediate dimension (`N_per_rank = intermediate_size / tp_size`). All ranks process the full token set through all experts, synchronized by AllReduce.

**EP experts** assign each rank a disjoint subset of complete experts (`num_local_experts = num_experts / ep_size`). Tokens are routed to expert-owning ranks via either AllReduce (TP/EP) or DeepEP all-to-all dispatch (DP/EP).

**Only DP/EP uses DeepEP.** With TP attention, tokens are replicated across ranks after the post-attention AllReduce; routing them via all-to-all dispatch would be redundant since every rank already holds the same token set. TP/EP therefore uses AllReduce-based expert parallelism: each rank runs its local `E/ep_size` experts on the full batch, masking non-routed tokens, and combines results via full-hidden AllReduce.

## 3. When to Use What

The practical choice is between two configurations:

| Batch regime | Recommendation | Why |
|---|---|---|
| Small (≤ 512 equiv) | **TP/TP** | Small-tensor AllReduce is in the sub-millisecond fast path. Every alternative adds fixed overhead (DeepEP ~5–10ms, DP gather/scatter, EP masking) that dominates at low batch. |
| Large (≥ 1024 equiv) | **DP/EP** | DP attention reduces per-rank work by `dp_size`×. DeepEP dispatch+combine is cheaper than AllReduce on large tensors. EP's full-width GEMM achieves better tensor-core utilization at compute-bound prefill. |

Taking Qwen3-235B-A22B-half on 8×A100 as an example, the system decode throughput ratio vs TP/TP:

| Global batch | TP/TP | TP/EP | DP/TP | DP/EP |
|---:|---:|---:|---:|---:|
| 8 | 1.00× | 0.89× | 0.68× | 0.38× |
| 64 | 1.00× | 0.77× | 0.84× | 0.60× |
| 512 | 1.00× | 0.89× | 0.96× | 0.88× |
| 2048 | 1.00× | 0.90× | 1.17× | **1.28×** |

At prefill (batch 2048, `input_len=10`), DP/EP achieves 1.45× TP/TP throughput.

The crossover between TP/TP and DP/EP is around **~1024 equivalent batch** (bracketed by 512→0.88× and 2048→1.28× on this hardware). Below the crossover, TP/TP wins; above, DP/EP wins. The two "cross" configurations — TP/EP and DP/TP — never beat both TP/TP and DP/EP at any tested batch size.

## 4. Why TP/EP Is Suboptimal

TP/EP combines TP attention (each rank sees the full batch) with EP experts (each rank owns `E/ep_size` complete experts). This pairing is architecturally awkward: tokens are replicated across ranks after the post-attention AllReduce, so every rank feeds the **full batch** into the MoE kernel — but only a fraction of experts are local.

### 4.1. MoE kernel overhead: full batch × sparse experts

In TP/TP, the `fused_moe_kernel` processes `num_tokens` tokens through all 128 experts, each with narrow GEMM width `N = intermediate_size / tp_size`. In TP/EP, the same kernel receives the same `num_tokens` but only 16 experts are local (`E / ep_size = 128 / 8 = 16`). The `StandardDispatcher` remaps non-local expert IDs to -1, and the kernel skips their GEMMs — but the overhead of sorting and aligning ALL `num_tokens × topk` token-expert pairs across ALL 128 expert slots remains.

Taking Qwen3-235B-A22B-half prefill at batch 2048 on 8×A100 as an example, trace evidence (rank 0, warm calls only):

| Kernel | TP/TP | TP/EP | DP/EP |
|---|---:|---:|---:|
| `fused_moe_kernel` (94 calls) | 382.2 ms (4,550 μs/call) | 495.2 ms (5,268 μs/call) | 237.0 ms (2,821 μs/call) |
| `count_and_sort_expert_tokens` | 1.8 ms (47 μs/call) | 7.1 ms (151 μs/call) | 0.9 ms (24 μs/call) |
| `moe_align_block_size` | 1.7 ms (47 μs/call) | 2.1 ms (44 μs/call) | 0.4 ms (11 μs/call) |
| `moe_sum_reduce` | 31.7 ms | 39.9 ms | — |

The `fused_moe_kernel` in TP/EP is **14% slower per call** than TP/TP (5,268 vs 4,550 μs), despite having full-width expert GEMMs (`N=1536` vs `N=192`). The reason is that each rank still processes ALL 2048 input tokens: `moe_align_block_size` sorts 16,384 token-expert pairs (`2048 × topk=8`) across 128 expert slots. Only 16 slots are populated, so the kernel iterates 112 empty, padded expert blocks. The `count_and_sort_expert_tokens` kernel is 3.2× slower (151 vs 47 μs/call) because it counts across 128 slots instead of focusing on 16 populated ones.

The full-N GEMM shape advantage (wider GEMMs per expert) is real but is overwhelmed by the overhead of processing the full batch through 128 expert slots with 87.5% sparsity. By contrast, DP/EP's `fused_moe_kernel` receives only ~256 dispatched tokens and sorts ~2,048 pairs across 16 expert slots — no wasted iteration.

### 4.2. Communication: no savings over TP/TP

TP/EP uses the same full-hidden AllReduce as TP/TP for the MoE combine step. Both execute `tensor_model_parallel_all_reduce(final_hidden_states)` on a `[num_tokens, hidden_dim]` tensor. TP/EP additionally pays for the attention AllReduce (same as TP/TP). There is no communication advantage.

### 4.3. Net result

TP/EP is a net loss at **every tested batch size** on 8×A100. It pays TP/TP's full communication cost, adds MoE kernel overhead from sparse expert iteration, and gains nothing from the full-N GEMM shape because the input token count per rank is not reduced. The GEMM shape benefit only materializes when combined with dispatch (as in DP/EP), which reduces the input token count per rank and focuses the kernel on populated expert slots.

## 5. Why DP/TP Is Suboptimal

DP/TP combines DP attention (each rank processes `batch/dp_size` tokens) with TP experts (each rank holds all experts, sharded along the intermediate dimension). This pairing gains from DP attention's per-rank work reduction but misses the MoE GEMM shape advantage.

### 5.1. The dp_gather/dp_scatter overhead

After DP attention, each rank holds only `batch/dp_size` tokens. Before the TP-sharded MoE, tokens must be gathered to the full batch (`dp_gather` via AllGather), then scattered back after the MoE (`dp_scatter` via ReduceScatter). These collectives add per-layer overhead.

Taking the same Qwen3-235B-A22B-half decode at batch 2048 as an example:

| Kernel | DP/TP | DP/EP | Notes |
|---|---:|---:|---|
| `ncclAllGather` | 9.4 ms | — | Gather tokens before TP-MoE |
| `ncclReduceScatter` | 7.5 ms | — | Scatter results after TP-MoE |
| `deep_ep::ll::dispatch` | — | 7.3 ms | DeepEP replaces both |
| `deep_ep::ll::combine` | — | 7.0 ms | |
| **Comm total (MoE path)** | **16.9 ms** | **14.3 ms** | DP/EP saves 2.6 ms |

DP/TP's AllGather+ReduceScatter pair (16.9 ms) is slightly more expensive than DP/EP's DeepEP dispatch+combine (14.3 ms) at decode batch 2048. The gap widens at prefill: DP/TP uses full-hidden AllReduce (136.6 ms) while DP/EP uses DeepEP intranode dispatch+combine (76.8 ms) — a 59.8 ms saving.

### 5.2. MoE GEMM shape: narrow N, no improvement over TP/TP

DP/TP's experts are TP-sharded, identical to TP/TP: each rank runs all 128 experts with `N = intermediate_size / tp_size = 192`. The full batch (after `dp_gather`) passes through the same narrow-GEMM kernel.

Trace evidence at prefill batch 2048:

| Config | `fused_moe_kernel` per call | Expert GEMM N | Input tokens |
|---|---:|---:|---:|
| DP/TP | 4,527 μs | 192 (narrow) | 2048 (gathered) |
| DP/EP | 2,821 μs | 1536 (full) | ~256 (dispatched) |

DP/TP's `fused_moe_kernel` runs at the same speed as TP/TP (4,527 vs 4,550 μs) — the `dp_gather` reconstitutes the full batch before the MoE stage, so the kernel sees identical input. DP/EP's kernel is **1.60× faster** because it processes ~256 dispatched tokens through 16 experts with full-width GEMMs.

### 5.3. Where DP/TP does help: attention and norms

DP attention reduces per-rank attention + norm work by `dp_size`×. At prefill batch 2048 on 8×A100:

| Kernel | TP/TP | DP/TP | Saving |
|---|---:|---:|---:|
| `FusedAddRMSNorm` | 33.1 ms | 1.7 ms | −31.4 ms |
| `ampere gemm` (QKV/O projections) | 67.2 ms | 64.0 ms | −3.2 ms |
| `flashinfer prefill` | 3.5 ms | 3.4 ms | −0.1 ms |

The attention savings are real (+17% decode throughput vs TP/TP at batch 2048). But DP/EP gets the **same** DP attention savings **plus** the MoE GEMM shape advantage and cheaper communication. DP/TP's attention win is a strict subset of DP/EP's win.

### 5.4. Net result

DP/TP is viable as a middle ground (1.17× TP/TP decode at batch 2048) when DeepEP is unavailable. But it can never match DP/EP because:
1. MoE GEMM is identical to TP/TP (narrow N, full batch after gather) — no compute improvement.
2. Comm overhead from dp_gather + dp_scatter + AllReduce is higher than DeepEP dispatch+combine.
3. DP attention savings are shared with DP/EP — not a differentiator.

At decode batch 2048, the per-kernel gap between DP/TP and DP/EP:

| Source | Saving (DP/EP − DP/TP) | Notes |
|---|---:|---|
| `fused_moe_kernel` | −2.2 ms | Memory-bound; small shape effect at decode |
| `moe_sum_reduce` eliminated | −3.3 ms | DP/EP uses ep_scatter/gather instead |
| AllGather+ReduceScatter → DeepEP | −2.6 ms | Comm swap |
| EP overhead (masked kernel, count_sort) | +1.9 ms | |
| Other | +0.7 ms | |
| **Net** | **−5.5 ms** | |

## 6. Root-Cause Analysis: What Drives DP/EP's Win Over TP/TP

The dominant performance axis **flips between prefill and decode**.

### 6.1. Prefill (compute-bound, large M per expert)

At prefill with batch 2048 (`input_len=10`, 20,480 total tokens), the MoE stage is compute-bound. The `fused_moe_kernel` GEMM shape determines performance.

Taking Qwen3-235B-A22B-half prefill at batch 2048 as an example (rank 0, warm calls only):

| Kernel | TP/TP | DP/EP | Saving |
|---|---:|---:|---:|
| `fused_moe_kernel` | 382.2 ms | 237.0 ms | **−145.2 ms** |
| AllReduce → DeepEP | 130.1 ms → 76.8 ms | | **−53.3 ms** |
| `moe_sum_reduce` → ep_scatter/gather | 31.7 ms → 13.4 ms | | **−18.3 ms** |
| `FusedAddRMSNorm` | 33.1 ms → 4.0 ms | | **−29.1 ms** |
| **Total** | 670.7 ms | 424.9 ms | **−245.8 ms** |

**Savings breakdown**: MoE GEMM shape 59% · Communication 22% · Norm/attention 12% · MoE overhead 7%.

The GEMM shape effect: TP/TP runs 128 experts × `[~128 tokens, N=192]` GEMMs (narrow). DP/EP runs 16 experts × `[~128 tokens, N=1536]` GEMMs (full width). Both process the same total FLOPs, but fewer wider GEMMs achieve better A100 tensor-core occupancy. The per-call speedup is 1.61× (4,550 → 2,821 μs).

### 6.2. Decode (memory-bound, small M per expert)

At decode with batch 2048, each forward pass processes 2,048 tokens (one per request). The `fused_moe_kernel` is memory-bandwidth-bound: per-expert M ≈ 128 tokens, and the kernel spends most time loading expert weights rather than computing. GEMM width N is irrelevant.

| Kernel | TP/TP | DP/EP | Saving |
|---|---:|---:|---:|
| `fused_moe_kernel` | 40.6 ms | 38.5 ms | **−2.1 ms** |
| AllReduce → DeepEP | 22.9 ms → 14.3 ms | | **−8.6 ms** |
| `moe_sum_reduce` | 3.3 ms → 0 | | **−3.3 ms** |
| `FusedAddRMSNorm` | 3.2 ms → 0.8 ms | | **−2.4 ms** |
| **Total** | 91.3 ms | 74.5 ms | **−16.8 ms** |

**Savings breakdown**: Communication 51% · MoE overhead (sum_reduce) 20% · Attention/norms 14% · fused_moe 13%.

At decode, the GEMM shape advantage nearly vanishes (484 vs 459 μs per call, 5% difference). The win is dominated by communication savings: DeepEP's low-latency dispatch+combine (14.3 ms) vs TP/TP's AllReduce (22.9 ms).

### 6.3. Summary: axis dominance by stage

| Axis | Prefill contribution | Decode contribution |
|---|---:|---:|
| MoE GEMM shape (EP full-N vs TP narrow-N) | **59%** | 13% |
| Communication (DeepEP vs AllReduce) | 22% | **51%** |
| DP attention + norms (batch/dp_size per rank) | 12% | 14% |
| MoE overhead (sum_reduce, sorting) | 7% | 20% |

## 7. Load Imbalance Under Expert Parallelism

EP routes different tokens to different expert-owning ranks. Non-uniform routing creates load imbalance: the slowest rank's MoE compute determines the effective layer latency, and DeepEP combine must wait for all ranks.

### 7.1. Measurements (dummy weights — lower bound)

With `--load-format dummy`, gate weights are `uniform(−1e−3, 1e−3)`, producing near-uniform routing. Per-rank MoE time for DP/EP decode at batch 2048:

| Rank | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MoE (ms) | 38.5 | 40.0 | 36.0 | 36.9 | 40.6 | 38.6 | 38.7 | 36.9 |

Spread 12.1%, CV 3.9%. Max rank is 5.6% above mean.

### 7.2. DeepEP communication imbalance

DeepEP comm CV has two independent sources:
1. **Arrival-time variance**: upstream compute imbalance causes some ranks to arrive at the collective later. Present in all collectives.
2. **Volume imbalance** (DeepEP only): ranks with hot experts receive more tokens on dispatch and send more on combine. AllReduce does not have this — every rank exchanges identical payload.

With dummy weights, source (2) is near-zero. With real weights, both sources compound.

### 7.3. Real-weight expectation

Reference measurements on Qwen3-30B-A3B with real weights show MoE CV of 7.1% at batch 2048 (vs our dummy-weight 3.9%). The slowest rank would be 10–20% above mean rather than 6%, adding ~5–10 ms per MoE layer. DP/EP's 1.28× decode throughput win would likely shrink to ~1.20× with real weights.

## 8. Practical Guidance

**For production serving on 8×A100 with MoE models:**

| Scenario | Configuration | Why |
|---|---|---|
| Latency-sensitive, small batch (≤512) | TP/TP | Lowest per-request latency; AllReduce overhead negligible |
| Throughput-oriented, large batch (≥1024) | DP/EP | 1.2–1.5× higher system throughput |
| DeepEP unavailable | DP/TP at large batch | 85% of DP/EP's decode win from DP attention alone |
| Runtime batch fluctuation | ParaS (see [parallelism_switch.md](parallelism_switch.md)) | Switch between TP/TP and DP/EP without restarting |

**TP/EP should not be used in production.** It is a net loss at every tested batch size, useful only as a reference point for isolating EP's GEMM-shape contribution in analysis.

**DP/TP is a fallback** when DeepEP infrastructure (NVSHMEM, deep_ep library) is not available. It captures the DP attention benefit but misses the MoE GEMM shape advantage and pays higher communication cost than DP/EP.

**Always validate with real weights** before drawing production conclusions from dummy-weight benchmarks. Expert routing imbalance under real weights increases EP's load imbalance and may shift the crossover point.

## 9. Reference Data

- Benchmark workspace: `~/qwen235b_4way_analysis/` (`results_v2.jsonl`, `run_all_v2.sh`, `analyze.py`)
- Kernel analysis workspace: `~/qwen235b_4way_analysis/analysis/` (nsys profiles, `effective_latency.py`, `report.md`)
- Analysis methodology and additional reference results: see skill `analyze-ep-tp`
