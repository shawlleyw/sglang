# ParaS unified memory manager

This is the current reference for ParaS buffer ownership, layout, workspace
sizing, and migration safety. The implementation is in
[`paras_memory_manager.py`](../../python/sglang/srt/paras/paras_memory_manager.py),
[`unified_layout.py`](../../python/sglang/srt/paras/unified_layout.py), and
[`workspace.py`](../../python/sglang/srt/paras/workspace.py).

ParaS uses this layout for unquantized **BF16 Qwen3 MoE and GPT-OSS**. It
requires equal EP/TP groups greater than one, MoE TP=1, ParaS DP=1, and
`PARAS_CONFIGURE_METHOD=peer_access` for weight switching. Quantized weights
(including FP8 and MXFP4), non-BF16 weights, and shared experts are rejected.
KV dtype is a separate setting: the planner handles BF16 and FP8 KV storage;
hybrid SWA with FP8 KV is outside the supported scope.

## Ownership and layout

One `torch.empty(budget, dtype=torch.uint8)` allocation has two overlapping
interpretations. Only the active mode's contents are valid. EP uses DP
attention; TP shards attention as well as experts.

```text
low address                                                        high address
EP: [MoE scratch][attention scratch][padding][EP weights][seam][EP KV]
    |<----------- front F ---------------->|
TP: [TP weights][TP KV][seam][MoE scratch][attention scratch][padding]
                            |<-------------- tail T -------------->|
```

Each mode's weight region contains N layer bundles, in layer order:
`[expert w13][expert w2][attention QKV][attention O]`. The expert weight bytes
are equal between modes when EP=TP. Attention weight bytes differ: EP stores
full projections, TP stores shards, with replicated K/V heads when there
are fewer KV heads than TP ranks. TP retains no separate full-attention
backup. Layer bundle sizes therefore differ between modes.

Each mode's KV region contains N K/V pairs. EP KV is packed against the end
of the allocation; TP KV starts immediately after TP weights. Token/page
rounding can leave a small seam beside the KV region, and unused bytes
inside each reserved K/V region.

The front and tail serve two purposes:

- **Scratch** is the sum of separately aligned MoE and attention workspace
  reservations. They are disjoint and reused across sequential layers.
- **Padding** is the remainder of the endpoint after scratch. Transfer
  geometry can require more space than the backends need. If scratch is
  larger, the planner expands the endpoint and reduces KV capacity.

The same endpoint bytes provide migration headroom after inference drains.
There is no separately allocated extra layer slot, permanent transfer
staging buffer, or alternate N+1-slot allocator. Transfer headroom still
costs capacity, but backend scratch can use it during inference.

The allocation does **not** include all GPU memory. Embeddings, LM head,
norms, routing weights, GPT-OSS biases/sinks, DeepEP transport buffers,
request/index metadata, CUDA graph allocations, and outputs that escape an
operator remain outside. Backend-specific exclusions are listed below.

## Capacity calculation

All sizes below are per GPU. `align(x)` rounds up to 256 bytes. Let:

| Symbol | Meaning |
| --- | --- |
| `B` | UMM byte budget, rounded down to 256 bytes |
| `N` | Number of layers |
| `we`, `wt` | EP/TP bytes per weight bundle, summing individually aligned tensors |
| `S_EP`, `S_TP` | Sum of aligned managed MoE and attention scratch in each mode |
| `D = N * (we - wt)` | Weight bytes saved in TP mode |
| `A = B - N * wt` | TP bytes available after weights |
| `r` | Largest layer's KV budget share: `max(ratios) / sum(ratios)` |

The planner computes the endpoints directly, without searching:

```text
F = align(max(wt, S_EP, S_TP - D, ceil(A*r/(1+r)) - D))
EP cache budget = B - N*we - F

T = align(max(largest reserved EP K+V layer, S_TP))
TP cache budget = A - T
```

`F >= wt` provides space for a TP weight layer. `T` holds at least the
largest EP K/V layer. The other bounds ensure each TP KV region is at least
as large as its EP counterpart, so ordered migration never overwrites an
unread source layer. For uniform attention `r=1/N`. GPT-OSS uses ratio 1
for full-attention layers and `swa_full_tokens_ratio` for sliding layers.

`CacheCapacity.from_budget` divides each mode's cache budget by those
ratios and rounds each layer's K+V reservation down to 512 bytes. K and V
get equal, individually 256-byte-aligned halves. V starts at the midpoint
of the **reserved region**, which may be after the logical K tensor ends.

Token capacities fit inside those regions, round down to pages, and reserve
one additional page per tensor. A layer's logical K or V shape is:

```text
(layer_token_capacity + page_size, local_KV_heads, head_dim)
EP local_KV_heads = num_kv_heads
TP local_KV_heads = max(1, num_kv_heads // tp_size)
```

TP token capacity is not necessarily `EP_capacity * tp_size`. For TP8 with
four KV heads, each TP rank holds one head and each head has two replicas.
Even equal KV bytes give only four times the EP per-rank token capacity.
Migration also checks that live requests fit the target mode.

### Runtime budget and overhead accounting

Without an explicit planning budget, the manager uses:

```text
B = available_GPU_bytes
    - total_GPU_bytes * (1 - mem_fraction_static)
    - replicated_embedding_and_lm_head_bytes
```

Available memory is measured before materializing the model, taking the
minimum across ranks. Tied embeddings are counted once. The formula leaves
the configured dynamic reserve; it does not itemize every later external
allocation. An illustrative budget for the UMM alone is different from a
cap on all static memory.

Unused endpoint padding is `F - S_EP` in EP or `T - S_TP` in TP. It
measures transfer headroom beyond the **reserved** scratch. It is not the
complete overhead relative to an original single-mode server.

For a matched total-memory budget, compare the endpoint against scratch the
baseline actually allocates: `endpoint - baseline_scratch`, then account
for alignment/page slack and external differences such as dual graphs and
replicated auxiliary tensors. In particular, TP reserves for a full Triton
64K-token chunk, while the native backend sizes temporaries to
`min(actual_input_tokens, 65536)` and the selected kernel's padding. A
smaller runtime batch can therefore leave unused space **inside** the TP
scratch reservation even when endpoint padding is zero. Backend execution
behavior is preserved, but worst-case reservation is not the same as the
baseline's live allocation. Total overhead needs a matched runtime
measurement; the diagrams alone cannot establish it.

## Backend workspace

`WorkspaceRequirement` records a backend name and `size_bytes`. `None`
means the backend allocates externally; it does not mean zero workspace.
`ModeWorkspaces` places MoE first and attention second. Neither can consume
the other's reservation or the unused padding.

### MoE

Qwen EP selects BF16 DeepGEMM where the existing DeepEP backend supports
it, otherwise Triton. TP uses fused Triton. GPT-OSS uses Triton for its
biases and activation. Reservations follow each backend's existing
execution policy.

For hidden size `H`, expert intermediate size `I`, expert count `E`, top-k
`K`, TP size `P`, and DeepEP dispatch capacity `C`, BF16 scratch is:

```text
EP rows = E * C
EP scratch = align(EP_rows * 2*I * 2) + align(EP_rows * I * 2)

TP I = I / P
TP rows = 65536*K + (E+1)*(256-1)
TP scratch = align(TP_rows * max(2*TP_I, H) * 2)
           + align(TP_rows * TP_I * 2)
```

EP reserves gate/up and activation intermediates for DeepEP's low-latency
padded receive shape: `local_experts * (EP_size * C)`. Masked DeepGEMM
consumes that shape directly; Triton EP flattens it. Normal dispatch uses
actual received rows. EP has no 64K chunk limit or reservation floor.

TP preserves `triton_moe_chunk_size = 64 * 1024` **input tokens per chunk**.
It includes top-k expansion and the fused TMA padding bound
`MOE_MAX_BLOCK_M = 256`. Gate/up and down share the first scratch tensor,
so its width is `max(2*TP_I, H)`; activation occupies a separate tensor.
This reservation does not shrink to the decode batch size.

`SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` supplies `C` (dispatcher
default 128). The [shared launch settings](../../scripts/paras/eval/launch_common.sh)
set it to `MAX_RUNNING_REQUESTS / NUM_GPUS` unless overridden: 256 for
2,048 requests on eight GPUs. TP can process the full global batch.

When actual MoE intermediates exceed their reservation, both views fall
back to the backend's original dynamic allocations at the original call
sites. Dispatch inputs, transport/routing/permutation buffers, and returned
outputs remain external. Thus the UMM does not guarantee an allocation-free
forward pass.

### Attention

| Backend | Managed storage | Sizing |
| --- | --- | --- |
| FlashInfer | Numerical/float workspace shared by its wrappers | Configured capacity, normally 384 MiB for Qwen3 MoE; 2 GiB in deterministic mode |
| Triton | FP32 partial output and LSE tensors | `align(tokens*local_query_heads*splits*head_dim*4) + align(tokens*local_query_heads*splits*4)` |
| Other backends, including FlashAttention | External | No caller-owned workspace integration here |

FlashInfer integer plans, pinned CPU staging, KV indices, and graph metadata
remain external. The numerical workspace size is configured through
`SGLANG_FLASHINFER_WORKSPACE_SIZE`; the existing architecture-specific
512 MiB overrides are preserved.

Triton uses the resolved model context length when calculating splits,
including RoPE scaling and context overrides. With CUDA graphs enabled,
each mode follows its own graph allocation capacity:
`G_EP = max(EP graph batches)` and `G_TP = max(TP graph batches)`.
Scratch tokens are `max(G_EP, max_running_requests / P)` in EP and
`max(G_TP, max_running_requests)` in TP; query heads are full in EP and
sharded in TP. Without graphs, both graph capacities are zero.

Graph inputs, attention metadata, and logits are also saved separately by
mode. Ordinary Triton decode no longer allocates a verification-only custom
mask. The opt-in [runtime-state VMM path](runtime_state_vmm.md) releases
inactive KV-index/logits backing while keeping captured virtual addresses.
These allocations remain outside the UMM budget.

Composite/mixed prefill-decode configurations, speculative workers, PDMux,
and two-batch overlap retain external attention allocation. Triton without
an explicit maximum running-request count also remains external. Marking
a workspace external does not imply that all such configurations support
ParaS switching.

### Qwen3-235B BF16 example: EP8 / TP8

This is a **planner calculation**, not a measured H200 footprint. Assume
130 GiB is available to the UMM **after external allocations**, BF16 KV,
page size 1, 94 layers, `H=4096`, `I=1536`, 128 experts, top-k 8, 64 query
heads, four KV heads, and head dimension 128. Use the H200 launch request
settings above (`C=256`) and, for Triton attention, eight KV splits and a
maximum graph batch of 256 in EP and 2,048 in TP. All diagram sizes below
are **MiB per GPU**:

```text
FlashInfer attention:
EP: [MoE 288][attention 384][padding 0][weights 66,928][seam 0.0264][KV 65,519.9736]
    |<--------- front 672 --------->|
TP: [weights 55,836][KV 72,342.9600][seam 0.0012][MoE 4,557.0388][attention 384][padding 0]
                                               |<---------- tail 4,941.0388 ---------->|

Triton attention:
EP: [MoE 288][attention 64.5][padding 241.5][weights 66,928][seam 0.0449][KV 65,597.9551]
    |<-------------- front 594 ------------->|
TP: [weights 55,836][KV 72,662.4590][seam 0.0022][MoE 4,557.0388][attention 64.5][padding 0]
                                               |<---------- tail 4,621.5388 ---------->|
```

Weights are 576 MiB of experts plus 136 MiB of attention per EP layer;
TP has the same experts plus 18 MiB of attention per layer. The diagrams
show reserved KV bytes; page-rounded usable tokens are 356,873 EP /
1,576,152 TP for FlashInfer and 357,298 EP / 1,583,113 TP for Triton.
These are local EP versus global TP token capacities.

FlashInfer has two workspace-bound endpoints. Triton's EP front is instead
bounded by one TP weight layer and has 241.5 MiB of transfer headroom beyond
scratch. The TP reservation still covers 65,536 input tokens even though
the example's decode batch tops out at 2,048. This reservation slack and
external allocations must be included in a comparison with native TP;
zero endpoint padding does not mean zero ParaS overhead. Other settings
can also leave nonzero endpoint padding.

## Initialization and views

1. The model creates and registers a `ParaSMemoryManager`.
2. `reserve_model_weights` declares both modes' weight shapes and native
   MoE requirements. GPT-OSS passes `with_bias=True` for backend selection;
   bias storage remains external.
3. `manager.plan_layout(config)` resolves budget and attention requirements,
   computes `UnifiedLayout`, and derives tensor offsets and `LayerCacheSpec`
   capacities. It returns a typed `UnifiedMemoryPlan` before allocating GPU
   storage. An explicit `budget=` is available for offline planning.
4. `manager.materialize(plan)` allocates the buffer and publishes the planned
   entries. It does not recalculate geometry or place tensors by reservation
   order.
5. Weight construction in `unquant.py` uses managed views instead of allocating
   those weights. The model loads EP weights and establishes stable per-mode
   Parameters. Peer IPC mappings are initialized once before switching.
6. `ModelRunner` binds the planned K/V views to `MHATokenToKVPool` or the
   full/SWA sub-pools of `SWAKVPool`. Each expert runner holds a fixed
   `MoEWorkspace`; attention backends bind named workspace views.

Important types are `UnifiedLayoutSpec` / `UnifiedModeSpec` for requirements,
`UnifiedLayout` / `CacheCapacity` for geometry, and `UnifiedMemoryPlan` for
completed tensor entries and cache specs. Mode arguments use `ParaSMode.EP`
and `ParaSMode.TP`; string values are used only in names/configuration.

```python
from sglang.srt.paras.mode import ParaSMode

# Given an initialized manager:
ep_k, ep_v = manager.get_kv_views(num_layers, ParaSMode.EP)
tp_k, tp_v = manager.get_kv_views(num_layers, ParaSMode.TP)
ep_weights = manager.get_view("model.layers.0.mlp.ep_experts.w13_weight")
tp_weights = manager.get_view("model.layers.0.mlp.tp_experts.w13_weight")
```

Checkpoint-loading names `mlp.experts.*` alias the EP entries. Likewise,
`kv.k/v` alias `kv.ep.k/v`. Explicit `kv.tp.k/v` entries already have their
TP shapes and offsets; `get_kv_views` does not reconstruct them by
reinterpreting EP storage. Summing all entries in `dump_layout()` would
double-count overlapping views and aliases; use `total_bytes` for the
allocation size.

## Migration safety

The scheduler drains outstanding inference, including the overlap pipeline,
before migration. Logical weights/cache state can overlap the destination's
scratch, so binding a view must not initialize it prematurely.

```text
EP -> TP: weights layer 0..N-1 -> KV layer 0..N-1 -> finish reconfiguration
TP -> EP: KV layer N-1..0 -> weights layer N-1..0 -> finish reconfiguration
```

Each weight step transfers a complete bundle: expert w13/w2 plus attention
QKV/O, followed by a cross-rank fence. KV transfers also fence per layer.
Layer order and weight-versus-KV order are both required by the overlap
geometry. The scheduler's early EP→TP weight transfer leaves EP cache/backend
metadata intact until KV migration has finished.

Attention EP→TP copies this rank's Q/K/V slices and O columns from the local
full projection into the planned TP views. TP→EP reads peer TP shards
directly to reconstruct full projections. Q and O use all ranks; replicated
K/V use one representative per head. With TP8 and four KV heads, those
representatives are ranks 0, 2, 4, and 6. No persistent full-weight backup or
full-size gather staging buffer is needed.

Backend reconfiguration changes references without writing target scratch.
Attention scratch is initialized only after migration, then the target
mode's graph metadata is restored. Both modes retain stable addresses
across graph capture and repeated switching; stable views alone do not
preserve inactive-mode contents.

See [parallelism switching](parallelism_switch.md) for request/control flow,
[weight transfer](nvlink_peer_access_weight_transfer.md) for expert kernels,
and [KV transfer](nvlink_peer_access_kv_cache_transfer.md) for routing.
Weight switching requires peer access. KV transport is independently selected
by `PARAS_KV_TRANSFER_METHOD`; its NCCL path still exists and allocates
transient communication buffers outside the UMM. The launch scripts select
peer access for both.

## Validation references

Existing focused tests include `test_unified_workspace_layout.py`,
`test_unified_attention_workspace.py`, `test_unified_workspace_views.py`,
`test_unified_triton_workspace.py`, `test_unified_deepgemm_workspace.py`, and
`test_unified_attention_transfer_tp8.py` under
[`test/srt/paras`](../../test/srt/paras/). They cover geometry, backend sizing,
workspace views, operator results, and attention reconstruction with replicated
KV heads. The manual switch procedure is
[`e2e_test.sh`](../../scripts/paras/eval/paras_cmd/e2e_test.sh).

[Memory analysis](memory_analysis.md) records measurements of the older slot
allocator. Its overhead totals and removed APIs are historical and must not
be used to describe this layout.
