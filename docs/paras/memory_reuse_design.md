# Unified weights, KV, and backend workspace

This document describes the implemented layout for unquantized Qwen3 MoE
with equal EP/TP groups, peer-access switching, and Triton MoE. Other
configurations retain their existing layout. Historical alternatives and
DeepGEMM proposals are deferred; they remain available in the
[previous revision](https://github.com/shawlleyw/sglang/blob/b7d8f3d5cb604815e1393602a2b3dc9f396ae24b/docs/paras/memory_reuse_design.md#proposed-unified-layout-after-the-backend-audit).

## Layout and capacity

One allocation has two interpretations. Small page-rounding seams between
weights and KV or KV and workspace are omitted:

```
EP: [MoE scratch][attention scratch][padding][EP weights][EP KV]
TP: [TP weights][TP KV][MoE scratch][attention scratch][padding]

W_mode = align_up(MoE_bytes, 256) + align_up(attention_bytes, 256)
EP_front >= max(one_TP_weight_layer, W_EP)
TP_tail   = align_up(max(one_EP_KV_layer, W_TP), 256)
padding   = endpoint_capacity - W_mode
```

Attention and MoE occupy separate subregions, preserving their native
allocation policies. Each endpoint is reused across sequential layers.
The same bytes provide transfer headroom once inference is drained.
Requirements smaller than the transfer gap leave padding; larger ones
expand the endpoint before KV capacity is calculated.

`unified_layout.py` also enforces TP KV bytes/layer >= EP KV bytes/layer.
When TP workspace exceeds the weight savings, this can enlarge the EP
front. The planner finds the smallest aligned front satisfying this
constraint and recomputes the TP gap from the resulting EP KV capacity.

Weights include expert w13/w2 and attention QKV/O. TP retains no full DP
attention backup. Per-mode views and offsets are fixed before graph capture.
Managed workspace is charged inside the UMM budget. Its replaced external
allocation is omitted; the configured dynamic reserve remains unchanged.
If a budget caps all static memory, external static allocations must first
be subtracted from the amount available to the UMM.

## Backend requirements

`workspace.py` declares numerical scratch requirements before backend
construction. Each requirement records the backend and a byte count;
`None` means externally allocated, not zero scratch. The manager validates
backend identity, alignment, and each subregion's capacity when returning
views. Neither operator can consume the other's region or unused padding.

| Backend | Managed numerical scratch | Sizing |
| --- | --- | --- |
| Triton MoE, EP | Gate/up and activation intermediates | Up to 65,536 dispatched expert rows per chunk, or the larger padded masked-decode receive bound |
| Triton MoE, TP | Gate/up and down share storage; activation is separate | Up to 65,536 input tokens per chunk, routed top-k rows, and conservative block padding |
| FlashInfer attention | Partial outputs and normalization statistics | Configured capacity: normally 384 MiB for Qwen3 MoE; 2 GiB in deterministic mode |
| Triton attention | FP32 partial outputs and LSE | Separately aligned tensors with payload `tokens * local_query_heads * KV_splits * (head_dim + 1) * 4` bytes |

MoE planning and runners share `MOE_CHUNK_ROWS`. Managed calls validate the
selected `BLOCK_SIZE_M` against `MOE_MAX_BLOCK_M = 256`, including the TP
final chunk and down-projection configuration. Kernel tuning beyond that
bound fails explicitly before using the workspace.

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
running-request count also remains external. DeepGEMM is not integrated.

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
5. Captured graphs retain stable per-mode addresses. Workspace ownership
   uses both weight address and shape to disambiguate overlapping EP/TP views.

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
