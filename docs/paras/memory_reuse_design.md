# Unified weights, KV, and backend workspace

This document describes the mandatory layout for BF16 Qwen3 MoE and GPT-OSS
with equal EP/TP groups (>1), MoE TP=1, ParaS DP=1, and peer-access switching.
Initialization rejects quantized weights and non-BF16 dtype. Qwen EP uses
DeepGEMM where its existing DeepEP path selects it; TP uses Triton. GPT-OSS
uses Triton in both modes to preserve its biases and activation.

## Layout and capacity

One allocation has two interpretations. Small page-rounding seams between
weights and KV or KV and workspace are omitted:

```
EP: [MoE scratch][attention scratch][padding][EP weights][EP KV]
TP: [TP weights][TP KV][MoE scratch][attention scratch][padding]

W_mode = align_up(MoE_bytes, 256) + align_up(attention_bytes, 256)
EP_front >= max(one_TP_weight_layer, W_EP)
TP_tail   = align_up(max(largest_EP_KV_layer, W_TP), 256)
padding   = endpoint_capacity - W_mode
```

Attention and MoE occupy separate subregions, preserving their native
allocation policies. Each endpoint is reused across sequential layers.
The same bytes provide transfer headroom once inference is drained.
Requirements smaller than the transfer gap leave padding; larger ones
expand the endpoint before KV capacity is calculated.

`unified_layout.py` calculates the endpoints directly. Let `D` be the total
EP-to-TP weight saving, `A = budget - TP_weights`, and `r` the largest layer's
share of the KV budget (1/N for uniform attention). Then:

```
EP_front = align_up(max(TP_weight_layer, W_EP, W_TP - D, ceil(A*r/(1+r)) - D))
```

The last term reserves room for one EP cache layer during migration:
`EP_KV * (1+r) <= A`. After dividing the remaining EP budget into per-layer
regions, TP's tail is the larger of the largest EP region and TP scratch. The
remaining bytes form TP's KV budget, which is at least EP's budget.

Each layer has equal 256-byte-aligned K and V regions. Token capacities are
rounded down to pages within these regions, preserving `swa_full_tokens_ratio`
for GPT-OSS. Placement is independent of token rounding; TP regions are at least
as large as their EP counterparts. Neither calculation needs a search.

Weights include expert w13/w2 and attention QKV/O. TP retains no full DP
attention backup. Per-mode views and offsets are fixed before graph capture.
Managed workspace is charged inside the UMM budget. Its replaced external
allocation is omitted; the configured dynamic reserve remains unchanged.
If a budget caps all static memory, external static allocations must first
be subtracted from the amount available to the UMM.

## Initialization

1. `reserve_model_weights` declares per-mode weight shapes and MoE requirements.
   Qwen and GPT-OSS share this declaration, with GPT-OSS specifying its biases.
2. `plan_layout` resolves the budget and attention requirements, calculates the
   unified layout, then derives every weight/KV view's shape and offset. It
   returns one typed `UnifiedMemoryPlan`, including full/SWA layer capacities.
3. `materialize(plan)` allocates the backing buffer and exposes the planned views.

There is no second KV reservation pass or alternate slot allocator. Transfer
headroom is part of the front/tail regions shown above.

## Backend requirements

`workspace.py` contains the named MoE scratch pair and declares numerical scratch requirements before backend
construction. Each requirement records the backend and a byte count;
`None` means externally allocated, not zero scratch. The manager validates
backend identity, alignment, and each subregion's capacity when returning
views. Neither operator can consume the other's region or unused padding.

| Backend | Managed numerical scratch | Sizing |
| --- | --- | --- |
| BF16 DeepGEMM / Triton MoE, EP | Gate/up and activation intermediates | DeepEP padded receive shape: `local_experts * EP_size * dispatch_tokens_per_rank` rows |
| Triton MoE, TP | Gate/up and down share storage; activation is separate | Up to 65,536 input tokens per chunk, routed top-k rows, and conservative block padding |
| FlashInfer attention | Partial outputs and normalization statistics | Configured capacity: normally 384 MiB for Qwen3 MoE; 2 GiB in deterministic mode |
| Triton attention | FP32 partial outputs and LSE | Separately aligned tensors with payload `tokens * local_query_heads * KV_splits * (head_dim + 1) * 4` bytes |

TP retains SGLang's existing `triton_moe_chunk_size = 64 * 1024`
input-token loop. EP has no corresponding chunk limit. Its scratch follows
DeepEP's configured low-latency receive shape, consumed directly by masked
DeepGEMM and flattened by Triton EP. Normal dispatch uses the actual number
of received expert rows. Calls larger than the reserved scratch retain the
backend's original temporary allocations and allocation timing; those
allocations consume dynamic memory outside the unified buffer. No 64K
reservation floor is imposed on either EP backend.

Only fused Triton TMA calls validate the padding bound `MOE_MAX_BLOCK_M = 256`,
including the final chunk's configuration. Shared workspace views check byte
capacity independently of kernel tuning.

Triton attention planning and execution share the split-count calculation
and use ModelRunner's resolved context length, including RoPE scaling and
context overrides. The existing graph runner reserves for max(EP graph
capacity, TP graph capacity) in both modes. This policy is preserved:
Qwen235B with 2,048 graph tokens, eight splits and head dimension 128 needs
516 MiB EP / 64.5 MiB TP attention scratch.

FlashInfer wrappers share the current mode's numerical workspace. Integer
plans, pinned CPU staging, KV indices, and other attention metadata remain
external. FlashAttention has no caller-owned scratch binding here and
remains external, as do composite backends, speculative workers, PDMux,
and two-batch overlap. Triton attention without an explicit maximum
running-request count also remains external.

DeepEP communication buffers, dispatch/permutation metadata, inputs, and
outputs that outlive an operator remain external. Overlapping compute
streams would require independent workspace lanes; the supported scheduler
serializes layer computation.

## Switching invariants

1. Drain outstanding inference before transferring any weights or KV.
2. EP→TP moves complete layer weight bundles forward, then KV forward.
   TP→EP moves KV backward, then weight bundles backward. Each transferred
   layer has a cross-rank fence before its source storage can be reused.
3. Attention EP→TP slices local replicated weights. TP→EP reconstructs
   Q/O from all TP shards and K/V from representative ranks when KV heads
   are replicated. With TP8 and four KV heads, K/V use ranks 0, 2, 4, 6.
4. Backend reconfiguration binds views without writing them: the target
   workspace can still contain live source weights. Initialize scratch only
   after migration, then restore the target mode's graph metadata.
5. Captured graphs retain stable per-mode addresses. Each EP/TP expert runner
   holds its own `MoEWorkspace` binding in `MoeRunnerConfig`; switching selects
   the corresponding expert object. Fused custom ops receive the scratch tensor
   explicitly and declare its writes. No weight-address lookup is needed.

These invariants apply to both startup dual capture and runtime switching.
Scratch contents are disposable; plans and escaped outputs are not.

## Validation

The focused tests cover capacity and ordered-transfer geometry, disjoint
physical views, alignment and overflow rejection, rebind ordering, resolved
context sizing, and exact managed-versus-original Triton MoE results:

```bash
python -m pytest -q \
  test/srt/paras/test_unified_attention_workspace.py \
  test/srt/paras/test_unified_workspace_layout.py \
  test/srt/paras/test_unified_workspace_views.py \
  test/srt/paras/test_unified_triton_workspace.py
```

The manual-switch procedure is `scripts/paras/eval/paras_cmd/e2e_test.sh`.
Qwen3-30B BF16 passed EP8↔TP8 with both FlashInfer and Triton attention,
including dual CUDA graph capture and in-flight requests (160 responses per
backend). Earlier [four-GPU validation](../../artifacts/paras_unified_memory/validation.md)
and [eight-GPU validation](../../artifacts/paras_unified_memory/tp8/validation.md)
record attention transfer results before attention workspace integration.
The TP8 IPC test reconstructs weights exactly after destroying full copies
and poisoning redundant KV replicas. Qwen235B sizes are planner calculations;
no 235B/H200 runtime measurement is claimed.
