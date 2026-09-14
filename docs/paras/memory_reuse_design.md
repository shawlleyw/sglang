# Attention layout switching and MoE workspace reuse

Status: implemented for unquantized Qwen3 MoE with equal EP/TP groups,
`peer_access`, and Triton MoE in both modes. The DeepGEMM integration below
remains a design proposal. All memory numbers are per GPU.

## Implemented Triton layout

`unified_layout.py` plans one allocation; `paras_memory_manager.py` exposes
weight, KV, and workspace views. `attention_transfer.py` reconstructs full
attention from TP shards, including replicated KV heads. No full attention
backup survives in TP mode. Switching moves complete layer bundles with a
cross-rank fence after each layer: weights then KV forward for EP→TP, KV
then weights backward for TP→EP. The scheduler drains inference before
transferring and activates the target metadata afterward.

The EP Triton runner places gate/up and activation intermediates in the
front region. Normal dispatch is processed in chunks of at most 65,536
expert rows; masked decode reserves its padded receive-row bound. The TP
fused runner retains its 65,536-input-token chunk bound, aliases gate/up
and down intermediates, and places activation scratch alongside them in
the tail. Every suballocation is aligned to 256 bytes. Workspace ownership
uses both the expert weight address and shape: opposite-mode weights may
have identical addresses. Views are fixed before CUDA graph capture.

DeepEP communication buffers, dispatch/permutation metadata, inputs, and
returned outputs remain outside this internal-scratch allocation. In
particular, an output survives subsequent workspace reuse. Independent
overlapping MoE compute streams would require separate workspace lanes;
the supported scheduler serializes layer computation. Quantized models,
other runners, and other switching transports retain their existing layout.

For Qwen3-30B-A3B BF16 on four A100-80GB GPUs, with the manual-switch
launcher's static fraction 0.7 and 2,048 total running requests, the planner
reported the following allocation (GiB; small rounding seams omitted):

```
EP: [front/workspace 1.059124][weights 15.187500][KV 36.618713]
TP: [weights 13.921875][KV 36.618759][tail/workspace 2.324749]
     <---------------- total 52.865405 GiB ---------------->
```

EP internal scratch needs 288 MiB; TP needs 2,380.543 MiB including the
conservative Triton padding bound. TP scratch exceeds the 1,296 MiB total
attention saving, so the EP front must grow to 1,084.543 MiB to keep TP KV
bytes per layer at least as large as EP's. Merely taking `max(EP workspace,
TP layer weights)` would make this layout unsafe. The planner finds the
smallest aligned EP front satisfying that migration constraint, including
page rounding. This reserves headroom once, not once per layer.

For Qwen3-235B-A22B BF16, DEP8↔TP8, the same Triton execution bounds give
576 MiB EP scratch and 4.450233 GiB TP scratch. With **130 GiB available to
the combined allocation**, the corresponding illustration is:

```
EP: [front/workspace 0.580078][weights 65.359375][KV 64.060413]
TP: [weights 54.527344][KV 71.022408][tail/workspace 4.450233]
     <---------------- total 130 GiB ---------------------->
```

If 130 GiB caps *all* static allocations, first subtract external static
allocations from that budget. These 235B figures are planner calculations,
not an H200 runtime measurement. The smaller workspace examples below are
historical design illustrations, not the implemented Triton maxima.

Validation commands and results are recorded in
`artifacts/paras_unified_memory/validation.md`.

The eight-GPU Qwen3-30B follow-up also passed both idle and in-flight
DEP8↔TP8 switches, with four KV heads replicated across TP rank pairs.
A real eight-GPU IPC test reconstructs attention exactly after destroying
full copies and poisoning redundant KV replicas. See
`artifacts/paras_unified_memory/tp8/validation.md` for the measured layout,
timings, rank mapping, and reproduction commands.

## Proposed unified layout after the backend audit

Use one allocation with two complete mode-specific interpretations:

```
Low address                                                   High address

EP: [DeepGEMM workspace / transfer gap][EP weights][rounding][EP KV]
TP: [TP weights][TP KV][rounding][Triton workspace / transfer gap]
```

The same endpoint bytes serve computation workspace while their mode runs
and transfer headroom during switching. Do not add another MoE-only slot or
another allocation for workspace already assigned here. This replaces the
old identical N+1-slot abstraction with explicit per-mode byte offsets.

Each weight block consists of 94 layer bundles containing w13, w2, QKV,
and O weights. Each cache block consists of 94 K+V layer bundles. Each
endpoint workspace is shared across sequential layers; reserve extra lanes
only for operations whose GPU lifetimes can overlap. Final EP/TP tensor
views and workspace addresses are established before graph capture.

### Plan workspace before cache capacity

Let M be the budget available to this combined region. For a 130 GiB ceiling
on all static memory, M is 130 GiB minus other static allocations, including
embeddings, LM head, routers, norms, and any externally owned buffers charged
to that ceiling. Do not subtract workspace a second time once it is in M.

The backend supplies W_EP and W_TP from its actual supported execution
limits and tensor-lifetime plan:

```
W_EP = workspace for the largest EP execution:
       DeepGEMM masked decode or contiguous prefill,
       including chosen preprocessing/output tensors and concurrent lanes

W_TP = workspace for the largest TP execution:
       the actual Triton-family backend, its batch/chunk and kernel padding,
       including chosen routing/output tensors and concurrent lanes
```

The planner must place simultaneous tensors in disjoint aligned ranges and
alias tensors with non-overlapping lifetimes. A workspace size is a bound
on the resulting layout, not the sum of every allocation in the function.
The EP estimate must include normal-mode prefill; it cannot use the 352 MiB
DeepGEMM masked-decode core number as the complete maximum. The TP estimate
cannot use the smaller EP scheduler batch limit.

For uniform Qwen3-235B BF16 layers, using bytes throughout:

```
N  = 94
we = 712 MiB                     # one EP layer's combined weights
wt = 594 MiB                     # one TP layer's combined weights

P_EP = align_up(max(W_EP, wt, W_TP - N*(we-wt)), workspace_alignment)
ce   = floor_to_EP_cache_granularity((M - P_EP - N*we) / N)

P_TP = align_up(max(W_TP, ce), workspace_alignment)
ct   = floor_to_TP_cache_granularity((M - P_TP - N*wt) / N)
```

Reject a budget with nonpositive usable cache capacity. Validate we >= wt
and ct >= ce; if the latter fails, increase P_EP to the smallest aligned
value that satisfies it and recompute both capacities. At page size one, BF16 K+V rows use
2,048 bytes in EP and 512 bytes in TP. Pool sentinel/page reservations count
toward these sizes. Backend alignment and page-size constraints may require
more rounding than the row-only example below.

Use these actual byte addresses, which keep token rounding explicit:

```
EP workspace: [0, P_EP)
EP W[i]:     [P_EP + i*we, P_EP + (i+1)*we)
EP KV[i]:    [M - N*ce + i*ce, M - N*ce + (i+1)*ce)

TP W[i]:     [i*wt, (i+1)*wt)
TP KV[i]:    [N*wt + i*ct, N*wt + (i+1)*ct)
TP workspace:[M - P_TP, M)
```

Rounding leaves a small EP weight/cache seam and a small TP cache/workspace
seam. No view may consume those bytes without registering an explicit
suballocation. Increasing W_TP consumes TP cache capacity and can require
reducing EP capacity to preserve ct >= ce. Increasing W_EP primarily consumes EP cache
and can also change the minimum cache-transfer gap required at the TP end.

### Illustrative 130 GiB allocation

For W_EP=512 MiB and W_TP=1,280 MiB, the exact row-rounded layout is:

```
EP:
[workspace 594 MiB][weights 65.359375 GiB][seam 140 KiB][KV 64.060413 GiB]

TP:
[weights 54.527344 GiB][KV 74.222614 GiB][seam 44 KiB][workspace 1.25 GiB]
```

Both interpretations use exactly 130 GiB. These workspace requirements are
illustrations of planner inputs, not validated execution bounds. DeepGEMM
decode at dispatch capacity 256 could place a 256 MiB gate/up-or-down area
and a 96 MiB activation area inside the EP endpoint, leaving 242 MiB for
other compatible uses. Triton TP at 16,384 input tokens has a 1,024 MiB
gate/up-or-down area and a 48 MiB activation area before kernel padding and
metadata; the 1,280 MiB example leaves 208 MiB for such additions. These
remainders are not evidence that every backend configuration fits.

### Switch ownership and integration contract

1. Drain inference and wait for all consumers of the active workspace,
   including DeepEP gather/combine and other streams.
2. EP -> TP transfers weights in forward layer order, then cache in forward
   order. TP -> EP transfers cache in reverse layer order, then weights in
   reverse order. Fence each layer's peer reads/writes before overlapping
   source storage is reused.
3. The endpoint may become a destination during transfer. No forward
   workspace lease is valid in this phase. Transfer kernels must either
   write directly to proven-disjoint destinations or use separately proven
   phase-specific scratch; the inference workspace cannot be borrowed
   unconditionally during the switch.
4. Attention belongs to the weight-transfer phase. TP -> EP reconstructs
   full attention projections from TP shards and deduplicates replicated
   K/V heads; there is no preserved local full-attention copy in TP mode.
5. After all transfers complete, activate the target's preplanned views and
   permit target-mode graph replay and workspace use.

A backend workspace plan should report tensor names, byte sizes, alignment,
offsets, and last consumers, with bounds for every supported execution
shape. Integrate explicit output arguments in Triton/DeepGEMM and adapt
DeepGEMM's `dispose_tensor` calls so permanent views are not detached. Keep
DeepEP/NVSHMEM-owned transport storage outside this first integration.

DeepGEMM warmup is a separate allocation phase. Precompile before filling
the static budget where possible, or explicitly budget its dummy tensors
and any inference buffers still live when compilation starts. Do not
silently assume the ordinary endpoint workspace accommodates warmup.

The exact 130 GiB example was checked on CPU for endpoint/workspace
isolation and every destination against every unread source in both
directions, including BF16 row rounding. This remains a layout design;
backend workspace bounds, GPU transfers, and graph replay need runtime
validation before implementation can be considered complete.

## Latest discussion: size padding for each mode's MoE workspace

The endpoint padding can serve as MoE workspace. It must be sized from each
mode's maximum supported execution shape, including padded routing rows and
concurrent workspace users. TP has a larger supported batch per GPU; its
workspace must not be sized from the EP local batch limit.

This refines the equal-payload/equal-gap examples below. They are valid
illustrations, but unnecessarily reduce EP cache when TP needs more scratch.
Use independent EP-front and TP-end reservations instead.

### Source audit: BF16 Triton execution paths

The current branch and `paras_epdptp` have the same versions of the two
Triton runner files checked here.

- Standard TP dispatch reaches `fused_experts_impl` in
  `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`. It already
  aliases gate/up and down intermediates through `cache`, with activation
  output allocated separately. Its chunk size is 65,536 input tokens.
- DeepEP reaches `TritonRunnerCore.run` in
  `python/sglang/srt/layers/moe/moe_runner/triton.py`. Gate/up, activation,
  and down intermediates are allocated separately. In the `no_combine`
  path, GEMM2 writes directly to `out_hidden_states`, leaving the allocated
  `intermediate_cache3` unused. Removing that allocation saves memory before
  introducing UMM reuse.
- DeepEP normal-mode pre-permutation allocates an expert-sorted input and
  routing metadata. Its row count is the sum of received expert assignments.
  It cannot be bounded by average expert balance.
- DeepEP low-latency input has shape
  `[local_experts, ranks * dispatch_capacity, hidden]`, as documented by the
  [DeepEP legacy Buffer API](https://github.com/deepseek-ai/DeepEP/blob/main/deep_ep/buffers/legacy.py).
  The runner allocates for this padded shape, not the count of valid rows.

For TP8, top-k=8, H=4096 and local intermediate size I=192. Ignoring TMA
padding and metadata, the existing fused path needs:

```
W_TP_core = min(input_tokens, 65536) * 8 * (4096 + 192) * 2 bytes
```

| TP input tokens | Shared gate/up-or-down | Activation | Core scratch |
|---:|---:|---:|---:|
| 2,048 | 128 MiB | 6 MiB | 134 MiB |
| 8,192 | 512 MiB | 24 MiB | 536 MiB |
| 16,384 | 1,024 MiB | 48 MiB | 1,072 MiB |
| 65,536 | 4,096 MiB | 192 MiB | 4,288 MiB |

These are not complete workspace bounds: the code can add TMA-padded rows
and routing metadata, and separate outputs remain live where applicable.
The earlier 696.76 MiB gap is already insufficient for the 16,384-token
case before those additions.

The H200 launch scripts default to 2,048 global running requests and set
the low-latency dispatch capacity to 256 per EP rank. For EP8 this gives
16 * 8 * 256 = 32,768 padded expert rows. At I=1536, current allocations
are 192 MiB gate/up, 96 MiB activation, 256 MiB unused down intermediate,
and 256 MiB returned output. Removing the unused intermediate leaves
288 MiB internal scratch plus 256 MiB output. Reusing gate/up storage for
the returned output after activation would reduce their combined reservation
to 352 MiB, but requires keeping the workspace lease until combine finishes.
That 352 MiB figure is a proposed reuse result, not current measured usage.

Prefill must also be included. The scripts use `max_prefill_tokens=8192`
and disable chunked prefill. `schedule_policy.py` permits a first request
larger than the remaining prefill target, so 8192 is not a hard workspace
bound. Either size for the admitted maximum request/batch and received
expert rows, or add bounded internal MoE tiling with correct output and
combine lifetimes. Scheduler chunked prefill support is a separate question.

### DeepGEMM workspace audit

DeepGEMM also needs managed computation space. In this checkout,
`fused_moe_triton/layer.py` selects DeepGEMM for unquantized weights when
expert parallel size is greater than one, DeepEP is the dispatcher, and
JIT DeepGEMM is available/enabled. ParaS TP experts override expert parallel
size to one and use standard dispatch, so they fall back to the configured
Triton family runner. The BF16 workspace plan must therefore support an
EP DeepGEMM / TP Triton combination. The ordinary standard-to-DeepGEMM
preprocessor quantizes activations to FP8, which is why the BF16 path is
explicitly restricted to DeepEP here.

The current and `paras_epdptp` versions of `moe_runner/deep_gemm.py` match.
The BF16 contiguous and masked routines each explicitly allocate:

| Tensor | Contiguous layout | Masked layout | BF16 bytes |
|---|---|---|---:|
| `gateup_output` | (R, 2I) | (E_local, M_padded, 2I) | 4RI |
| `down_input` | (R, I) | (E_local, M_padded, I) | 2RI |
| `down_output` | (R, H) | (E_local, M_padded, H) | 2RH |

For masked GEMM, R=E_local*M_padded, even when most rows are masked out.
`expected_m` is passed to kernel selection; it does not determine the
allocated tensor shape. For contiguous GEMM, R is the sum of received
expert assignments. There is no 65,536-token chunk loop in these DeepGEMM
runner methods.

`gateup_output` dies after SiLU. `down_input` remains live until GEMM2
finishes; `down_output` remains live through post-permutation/DeepEP combine.
The two core lifetime peaks are 6RI and 2R(I+H) bytes. A simple two-region
workspace aliases gate/up and down output, keeping activation separate:

```
core reservation = 2R * (max(2I, H) + I)
```

For EP Qwen, I=1536 and H=4096, this is 11 KiB per padded/received expert
row. With the H200 scripts' dispatch capacity 256, R=16*8*256=32,768:

```
gate/up output: 192 MiB
activation:     96 MiB
down output:   256 MiB

activation phase: 192 + 96 = 288 MiB
down GEMM phase:   96 + 256 = 352 MiB
managed core/output reservation: 352 MiB
```

At dispatch capacity 512, the same reservation grows to 704 MiB. These
numbers exclude received input, normal-mode expert-sorted input, routing
metadata, transport buffers, and post-permutation outputs. Normal-mode
preprocessing currently also allocates scale/dummy-scale buffers even for
BF16; they must be included or removed by a separately validated change.
Original returned buffers cannot be counted as reusable until their last
consumer completes. If the sorted input is also manager-owned, its overlap
with GEMM1 output introduces an additional lifetime peak.

The DeepGEMM wrapper takes explicit output tensors for both grouped BF16
calls, allowing UMM views to be supplied. These SGLang wrapper signatures
do not take an additional workspace argument; this audit does not establish
the native library's complete internal allocation behavior. FP8 routines
add quantized activations and scales while retaining BF16 GEMM outputs, so
their workspace must be planned separately rather than halving BF16 sizes.

`dispose_tensor` in `utils/common.py` uses `set_` to detach the tensor from
its storage. Adapt that lifecycle before supplying cached manager views:
detaching a permanent view invalidates that tensor object for subsequent
calls, and detaching a view does not release the UMM backing allocation.
Use explicit workspace lifetimes through combine, with stable capture views.

JIT precompilation is a distinct startup peak. In
`deep_gemm_wrapper/compile_utils.py`, `_GroupedMaskedBf16WarmupExecutor`
allocates dummy LHS, RHS, and output tensors and destroys the executor after
warmup. For 16 experts, max_m=8192, H=4096, I=1536, gate/up warmup allocates
1,024 + 384 + 768 = 2,176 MiB plus a tiny mask; down warmup allocates
384 + 192 + 1,024 = 1,600 MiB plus a mask. The code runs these kernel-type
warmups separately. Handle their temporary peak separately from the runtime
workspace, particularly if compilation starts after UMM materialization.

These findings are source-derived and arithmetic-checked; no DeepGEMM GPU
profiling or runtime allocation changes were performed.

### Independent endpoint sizing

Define M as the available combined-region budget, W_EP and W_TP as complete
workspace requirements for each mode, and A as the required alignment.
The UMM currently aligns entries to 256 bytes; backend-specific requirements
must also be respected. Align individual suballocations as well as endpoints.

For the uniform Qwen profile, in bytes:

```
EP_front = align_up(max(594 MiB, W_EP), A)
EP_KV_per_layer = floor_to_cache_granularity(
    (M - EP_front - 94 * 712 MiB) / 94)

TP_end = align_up(max(EP_KV_per_layer, W_TP), A)
TP_KV_per_layer = floor_to_cache_granularity(
    (M - TP_end - 94 * 594 MiB) / 94)
```

Place EP weights after `EP_front`, EP cache at the high end, TP weights at
the low end, and TP cache after TP weights. Assign token-rounding leftovers
to endpoint/seam slack and check the actual aligned addresses. The transfer
geometry requires EP weight bytes >= TP weight bytes and EP cache bytes <=
TP cache bytes per layer. If workspace reservations reverse the latter
inequality, this orientation needs replanning with general prefix/suffix
bounds rather than bypassing the check.

```
TP: [TP weights][TP KV sized for its budget][TP_end: TP workspace]
EP: [EP_front: EP workspace][EP weights][EP KV sized for its budget]
```

For M=130 GiB, an illustrative *complete* workspace budget of W_EP=512 MiB
and W_TP=1,280 MiB gives, before token flooring:

| Per-GPU region | EP | TP |
|---|---:|---:|
| Front workspace/padding | 594 MiB | 0 |
| Weights | 65.359375 GiB | 54.52734375 GiB |
| KV | 64.060546875 GiB | 74.22265625 GiB |
| End workspace/padding | 0 | 1.25 GiB |
| Total | 130 GiB | 130 GiB |

The 512/1,280 MiB workspace budgets are examples, not validated maxima for
the current scripts. The planner should derive them from backend sizing and
execution limits. Larger TP workspace consumes TP KV capacity; it need not
consume EP KV capacity too. Physical allocation remains bounded by M.

Workspace lifetime extends through the last consumer, not just the Python
runner call. Keep DeepEP/NVSHMEM-owned transport buffers separate initially;
do not alias returned tensors until gather/combine is finished. Concurrent
microbatches or compute streams require disjoint lanes or serialization.
Capture each mode's CUDA graphs using its fixed endpoint views. Replanning
after capture requires rebuilding affected graphs, not resizing live storage.

Validation: source-derived scratch arithmetic and 1,000 randomized uniform
asymmetric-endpoint layouts checked on CPU against every unread source in
both directions. No runtime allocation changes or GPU profiling were done;
the active Python environment has no PyTorch, Triton, or DeepEP installation.

## Discussion update: asymmetric attention, experts, and KV

The `paras_epdptp` branch has a newer design in
`docs/paras/unified_memory_ep_tp.md`: overlapping complete weight/cache
payloads with separate EP and TP sizes. The calculation below extends that
idea to attention. It is an alternative to the local attention permutation
proposal later in this document, not an implementation of either option.

### Actual Qwen3-235B BF16 sizes for DEP8 and TP8

Here EP means attention DP8 plus expert EP8, and TP means one TP8 instance.
There are 94 identical layers. Assume BF16 KV as well as BF16 weights.

| Per-layer component | EP, MiB | TP, MiB |
|---|---:|---:|
| Expert gate/up | 384 | 384 |
| Expert down | 192 | 192 |
| Attention QKV | 72 | 10 |
| Attention O | 64 | 8 |
| **Combined layer weights** | **712** | **594** |

Total layer weights are 66,928 MiB = 65.359375 GiB in EP and
55,836 MiB = 54.52734375 GiB in TP. TP frees 11,092 MiB =
10.83203125 GiB for cache if its inactive full attention weights are no longer
retained locally. Embeddings, LM head, routers, norms, communication buffers,
and other allocations are outside these totals.

This reverses the branch's original assumption that TP weights grow and TP
cache shrinks. Its current fixed EP-low / TP-high placement and cache-size
assertion cannot simply be reused after adding attention to layer weights.
For this topology, place TP low and EP high instead.

### Balanced payload and endpoint placement

Let C be one EP layer's reserved K+V cache size in MiB. Give both modes the
same total active payload B, excluding transfer slack:

```
EP per layer: 712 MiB weights + C MiB KV
TP per layer: 594 MiB weights + (C + 118) MiB KV
B = 94 * (712 + C) MiB
S = max(594, C) MiB
allocation = B + S
```

For Qwen's uniform layers, the branch's internal seam padding can be moved
to the endpoints without increasing the allocation:

```
TP: [weights: 55,836 MiB][KV: 94*(C+118) MiB][free end: S MiB]
EP: [free front: S MiB][weights: 66,928 MiB][KV: 94*C MiB]
```

These are two interpretations of one allocation. Each weight block contains
94 consecutive layer bundles, with expert and attention tensors inside each
bundle. Cache is likewise arranged in layer bundles containing K and V.
Offsets are:

```
TP_W[i] = i * 594 MiB
EP_W[i] = S + i * 712 MiB
TP_C[i] = 94 * 594 MiB + i * (C + 118) MiB
EP_C[i] = S + 94 * 712 MiB + i * C MiB
```

Transfer TP -> EP: caches in reverse layer order, then weights in reverse
layer order. Transfer EP -> TP: weights in forward layer order, then caches
in forward layer order. Fence peer reads/writes before reusing an earlier
layer's source storage. This is the opposite direction assignment from the
branch's original EP-low layout.

The weight non-overlap condition is S >= 594 MiB; later layers have more
separation because EP layer weights are larger. The cache non-overlap
condition is S >= C; the last layer is the tightest constraint. Thus one
contiguous workspace of size S is available at each mode's endpoint, after
the entire transition and all consumers have completed.

If preserving the branch's exact fixed-head geometry instead, EP has a
594 MiB front gap and max(0, C-594) MiB internal seam; TP has a C MiB end
gap and max(0, 594-C) MiB internal seam. The endpoint placement above
coalesces that seam with the appropriate endpoint and has the same total
allocation. This simplification relies on uniform balanced per-layer sizes;
heterogeneous models need the full prefix/suffix non-clobber calculation.

### Numerical illustration: C = 512 MiB

The model and dtype alone do not determine cache capacity. As an explicit
example, choose 512 MiB of EP K+V per layer; TP then gets 630 MiB per layer.
This makes B = 112.359375 GiB and S = 594 MiB, for a total allocation of
112.939453125 GiB. This is the combined region, not a claim about available
memory on any particular GPU.

```
One GPU, one allocation: 115,650 MiB = 112.939453125 GiB
(widths are schematic)

TP mode:
+-----------------------+-----------------------+----------------+
| Weights: 55,836 MiB    | K+V: 59,220 MiB        | FREE: 594 MiB  |
| MoE:     54,144 MiB    | 94 * 630 MiB           | end workspace  |
| Attn:     1,692 MiB    |                       |                |
+-----------------------+-----------------------+----------------+
0                    55,836                  115,056          115,650 MiB

EP mode:
+----------------+-----------------------+-----------------------+
| FREE: 594 MiB  | Weights: 66,928 MiB    | K+V: 48,128 MiB        |
| front workspace| MoE:     54,144 MiB    | 94 * 512 MiB           |
|                | Attn:    12,784 MiB    |                       |
+----------------+-----------------------+-----------------------+
0              594                   67,522                  115,650 MiB
```

For another capacity, C=768 MiB, the reusable endpoint grows to 768 MiB.
For C=128 MiB, it remains 594 MiB. In this layout the spare region is not
necessarily the old 576 MiB MoE-only slot.

### H200 example: 130 GiB static-memory ceiling

Interpret the requested 130G as 130 GiB. If all of this is available to the
combined layer-weight/KV region, the ceiling includes the safety gap; it is
not the active payload B. The equation is:

```
130 * 1024 = 94 * (712 + C) + max(594, C)       # MiB
C = (133120 - 66928) / 95 = 696.7578947 MiB
S = C
```

Flooring EP cache capacity to whole 2,048-byte K+V rows gives:

| Component, per GPU | EP | TP |
|---|---:|---:|
| Layer weights | 65.359375 GiB | 54.52734375 GiB |
| K+V cache | 63.960189819 GiB | 74.792221069 GiB |
| K+V per layer | 696.7578125 MiB | 814.7578125 MiB |
| Free front | 696.7578125 MiB | 0 |
| Free end | 0 | 696.7578125 MiB |

```
TP: [weights 54.527 GiB][KV 74.792 GiB][free end 0.680 GiB]
EP: [free front 0.680 GiB][weights 65.359 GiB][KV 63.960 GiB]
```

Both occupy 129.999992371 GiB, leaving 8 KiB of rounding room under the
130 GiB ceiling. The cache reservations hold 356,740 EP rows and 1,668,624
TP rows per layer; reserved padding reduces the usable-token counts.
Additional page-size constraints can require further flooring.

The reusable endpoint is therefore approximately 696.76 MiB, larger than
the old 576 MiB expert-only slot. It remains a transfer safety region and
can replace suitable forward scratch only when their lifetimes do not overlap.

If 130 GiB is the ceiling for *all* static allocations, subtract O bytes for
embeddings, LM head, routers, norms, and other static allocations outside this
region before solving: use M = 130 GiB - O on the left-hand side. The table
is the O=0 upper-bound illustration, not a measured whole-model allocation.
For a literal decimal 130 GB combined-region ceiling instead, the endpoint
is approximately 600.52 MiB.

### Capacity versus live cache and attention reconstruction

BF16 K+V costs 2,048 bytes per EP token per layer (4 KV heads) and 512
bytes per TP token per layer (1 replicated KV head). The illustration
corresponds to 262,144 EP cache rows and 1,290,240 TP cache rows, including
any reserved padding rows. Usable tokens depend on page/padding rules.
These are maximum capacities, not a promise that a full EP workload fits
after switching: TP merges the requests from eight EP ranks. At equal
global live-token count, TP stores twice the per-GPU KV bytes of balanced
EP because four KV heads are replicated over eight ranks. Retain the
pre-switch capacity check.

Unlike the local reversible permutation option below, this layout lets TP
KV overwrite inactive full attention weights. TP -> EP must reconstruct
full attention weights from TP shards across the TP group, selecting one
copy of replicated K/V heads. A pointer swap or local inverse permutation
cannot restore discarded weights. Transfers must respect the whole-layer
source lifetime, including all attention collectives and peer fences.

Validation so far: the branch's standalone layout checker passes after
swapping the mode roles for C=128, 512, and 768 MiB. An independent interval
checker also verified the coalesced-endpoint placement for these three
Qwen profiles and 1,000 random uniform balanced profiles, checking every
destination against every unread source in both switch directions. This
proves the tested address geometry only; it does not validate GPU transfer
kernels or CUDA graph replay.

## Objective and accounting

The current allocator reserves full DP attention weights and separate TP QKV
and O-projection weights. For 94 layers, the additional TP allocations are
940 MiB of QKV and 752 MiB of O weights. The N+1 expert layout adds one
576 MiB slot (384 MiB w13 + 192 MiB w2).

The proposal has two parts:

1. Store attention in one full-size region per projection, with different
   physical layouts in DEP and TP modes. This removes 1,692 MiB of allocations.
2. Use the inactive expert slot for MoE computation scratch and attention
   repacking scratch. This replaces separate forward-workspace allocations;
   the extra physical expert slot still exists.

If W bytes of otherwise necessary, lifetime-compatible forward workspace can
be moved into the slot, the additional saving is min(W, 576 MiB). Eliminating
the entire 2,268 MiB relative to DEP8 therefore requires at least 576 MiB of
replaceable workspace at the relevant memory peak. Small batches may leave
some of the slot unused. Moving workspace into UMM must remove its external
allocation and its corresponding budget charge, rather than count it twice.

## Attention: reversible local packing

Attention is already allocated by UMM. The change is to alias its TP views
into the full-weight regions and change the contents during switching.

Reserve only these full-size regions for each layer:

| Projection | DEP shape | TP view at the same base address | Region size |
|---|---|---|---:|
| QKV | (9216, 4096) | (1280, 4096) | 72 MiB |
| O | (4096, 8192) | (4096, 1024) | 64 MiB |

Both mode-specific Parameters are created once, before CUDA graph capture.
Their addresses, shapes, and strides remain stable. The DEP Parameter is not
usable while the storage contains the TP packing, and vice versa.

### QKV packing

For rank r, select 8 query heads and KV head floor(r / 2), because four KV
heads are replicated across eight ranks. Each head has 128 rows.

Construct a row permutation:

```
[local Q rows | local K rows | local V rows | every remaining row]
```

The first 1280 rows are the exact contiguous TP QKV weight. Every other row
is preserved once in the remainder. Inverse permutation restores the full
DEP QKV weight bit for bit. A direct overlapping copy of the selected rows
into the prefix would destroy values and is not sufficient.

### O-projection packing

The local TP weight consists of 1024 columns of the full O matrix. Pack the
matrix as two consecutive flattened row-major matrices:

```
[flatten(O[:, local columns]) | flatten(O[:, remaining columns])]
```

The prefix is a contiguous (4096, 1024) matrix. Inverse packing restores the
original (4096, 8192) row-major matrix. Merely moving local columns to the
front of each row does not produce a contiguous TP weight.

### Scratch and cost

Process projections sequentially. The simple implementation packs one
projection into scratch and copies it back into its original region. It
needs at most 72 MiB, which fits inside the inactive 576 MiB expert slot.
Use explicit kernel destinations; avoid `.contiguous()` or advanced indexing
that silently allocates a temporary GPU tensor.

This is a lossless local permutation. No attention all-gather is required
on the reverse switch, and the DEP attention storage is retained locally.
The cost is additional HBM traffic on every switch. A full pack plus copy-back
reads and writes all attention bytes twice: approximately 49.94 GiB of HBM
traffic per GPU per direction. This is a conservative implementation cost,
not a latency measurement; tiled swaps or cycle permutations could reduce it.

## Expert slots: mode-specific workspace views

For 94 layers, the existing mapping is:

```
DEP: [workspace | EP layer 0 | ... | EP layer 93]
TP:  [TP layer 0 | ... | TP layer 93 | workspace]
       slot 0                           slot 94
```

Expose fixed UMM workspace views for each mode:

- `workspace.ep` aliases the complete slot 0 byte range.
- `workspace.tp` aliases the complete slot 94 byte range.

Do not enlarge every expert slot to accommodate a large workspace: that
would multiply workspace padding by 95. Keep the current expert slot size
and use planned overflow storage or bounded computation chunks when needed.
An individual tensor must fit in a contiguous available range; total free
bytes alone do not guarantee that all scratch tensors can be placed.

During a transfer, the free slot moves between layers. It is unavailable to
forward computation until the transfer is complete. A Python mode flag alone
cannot establish that remote GPU reads have finished.

## Switching order

Use one model-level transition owner to enforce these phases:

1. Drain inference and wait for all users of the current workspace, including
   CUDA graphs and dispatch/combine streams.
2. Redistribute all MoE layers in the existing direction and layer order.
3. Complete local and peer transfer dependencies. The target-mode endpoint
   slot is now free: slot 94 for TP, slot 0 for DEP.
4. Use that slot to pack or restore each attention projection sequentially.
5. Commit the target attention Parameters, expert Parameters, workspace
   views, and runtime mode before resuming inference.

The forward peer-access path already separates MoE transfer and attention
configuration. The reverse path currently interleaves them; separate those
phases before borrowing the endpoint slot. The naive and overlap methods
also need explicit ordering changes before enabling this reuse.

Maintain source, switching, and target states. Repeated configuration of an
already-active mode must not apply the permutation again. A failed transition
must not resume inference with a mixture of layouts. CUDA graph warmup and
capture must invoke the real packing transitions, not just rebind Parameters.

## MoE runner integration

The allocation sites are backend-specific. For the BF16 Triton path, both
`moe_runner/triton.py` and the standard-dispatch fused implementation in
`fused_moe_triton/fused_moe.py` need coverage; modifying only one misses a
runtime path.

Start with internal GEMM and activation scratch. Gate/up output dies after
activation, so its storage can subsequently hold the down-projection
intermediate. Activation output must remain separate while the down GEMM
reads it. For R routed rows, local expert width I, hidden size H, and BF16,
this two-region scratch arrangement takes approximately:

```
2 * R * (max(2 * I, H) + I) bytes
```

For TP8, I=192 and H=4096, giving 8,576 bytes per routed row. For DEP8,
I=1536, giving 11,264 bytes per routed row. These are illustrative internal
scratch sizes, excluding alignment, routing, dispatch buffers, padding, and
returned outputs. DEP routed rows must use actual received/padded expert
work, not simply the local scheduler batch size. Direct-output/no-combine
paths have different liveness and can require less internal scratch.

Returned outputs and tensors still consumed by DeepEP combine cannot be
reused at the end of the runner call. Keep them separate initially. The
manager should supply an explicit workspace handle to each mode's runner;
it should not intercept arbitrary `torch.empty` calls globally.

Size workspace for configured execution and capture capacities before
materialization. Captured graphs bind directly to the appropriate mode's
workspace addresses. Concurrent compute streams or overlapping microbatches
need disjoint workspace lanes or explicit serialization. Scheduler overlap
alone is not evidence that scratch lifetimes are disjoint.

## Implementation and validation sequence

1. Add planned subregion aliases and mode-specific workspace metadata to UMM.
   Preserve the expert-slot offsets used by peer-access kernels.
2. Add BF16 attention pack/unpack kernels with explicit scratch destinations.
   Replace the TP attention reservations with prefix aliases, and remove
   initialization copies into those aliases until the first real switch.
3. Centralize transition ordering and integrate both attention directions.
4. Pass managed scratch into both BF16 Triton execution paths. Account for
   any external spill before assigning the remaining memory to KV cache.
5. Validate bitwise attention round trips on all eight ranks; unchanged
   GEMM outputs; expert-weight checksums after workspace use and switching;
   repeated DEP/TP graph replays; overflow and stream-lifetime cases; and
   end-to-end output agreement with the existing implementation.
6. Measure live and peak allocations, graph-private memory, and switch
   latency. Confirm that attention reservations shrink by exactly 1,692 MiB
   and report workspace savings at actual batch sizes separately.

A CPU integer-index prototype checked the proposed attention permutations
for all ranks with (Q heads, KV heads, TP) = (64,4,8), (16,4,2), (16,4,4),
and (8,2,1). TP prefixes matched the current slice definitions and inverse
packing restored every element. This validates the indexing concept only;
GPU kernels, synchronization, performance, and graph replay remain untested.

## Relevant current code

- `python/sglang/srt/paras/paras_memory_manager.py`: weight reservations,
  physical offsets, typed views, and KV capacity budgeting.
- `python/sglang/srt/layers/linear.py`: QKV/row-parallel attention finalization
  currently allocates and populates persistent TP Parameters.
- `python/sglang/srt/paras/layers/paras_model.py`: model-level switch order.
- `python/sglang/srt/paras/scheduler_paras_mixin.py`: overlap drain and switch
  boundaries.
- `python/sglang/srt/layers/moe/moe_runner/triton.py`: DeepEP Triton runner
  intermediates and returned-output lifetimes.
- `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`: standard fused
  path, already sharing gate/up and down intermediate storage.
