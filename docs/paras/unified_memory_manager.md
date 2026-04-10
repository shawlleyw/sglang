# ParaS Unified Memory Manager

## Overview

The ParaS Unified Memory Manager (`ParaSMemoryManager`) is a static, contiguous GPU memory allocator that owns **all** persistent memory for ParaS-enabled MoE models: expert weights, attention weights, staging buffers, and KV cache — in a single `torch.empty(..., dtype=torch.uint8)` allocation.

This design eliminates GPU memory fragmentation and enables zero-allocation parallelism switching between Expert Parallelism (EP) and Tensor Parallelism (TP) at runtime.

## Motivation

### The Problem

Without the memory manager, a ParaS-enabled model allocates memory in dozens of independent `torch.empty` calls:

- Each `FusedMoE` layer allocates `w13_weight` and `w2_weight` separately
- Each attention layer allocates `qkv_proj.weight` and `o_proj.weight` separately
- The KV cache pool allocates per-layer K and V buffers separately
- During EP→TP switching, new weight buffers are allocated for the TP layout, old EP buffers are freed

This leads to:

1. **Memory fragmentation** — hundreds of small allocations create gaps that can't be reused
2. **Allocation overhead during switch** — `torch.empty` calls during the critical switching path add latency
3. **Unpredictable addresses** — makes zero-copy RDMA/NCCL transfers difficult
4. **Double memory peaks** — during the switch, both EP and TP buffers exist briefly

### The Solution

One contiguous buffer. All tensors are views into it. The EP→TP switch overwrites the same bytes with a different interpretation — no allocation, no deallocation, no fragmentation.

## Architecture

### Buffer Layout (Qwen3-30B-A3B Example)

For a 2-GPU setup with `ep_size=2, tp_size=2, paras_tp_size=2`:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Contiguous uint8 Buffer (~65 GiB)            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─── Per-Layer Weights (×48 layers) ────────────────────────┐  │
│  │  EP MoE w13_weight  (64, 2048, 1536) bf16   ~386 MB      │  │
│  │  EP MoE w2_weight   (64, 768, 2048)  bf16   ~193 MB      │  │
│  │  QKV full weight    (2560, 2048)     bf16   ~10 MB       │  │
│  │  O_proj weight      (2048, 2048)     bf16   ~8 MB        │  │
│  │  QKV TP buffer      (640, 2048)      bf16   ~2.5 MB      │  │
│  │  KV cache K         (ep_tokens+1, 4, 128) bf16           │  │
│  │  KV cache V         (ep_tokens+1, 4, 128) bf16           │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─── Staging Buffers (shared across all layers) ────────────┐  │
│  │  staging.w13_a      (64, 1536, 2048) bf16   ~386 MB      │  │
│  │  staging.w13_b      (64, 1536, 2048) bf16   ~386 MB      │  │
│  │  staging.w2_a       (64, 2048, 768)  bf16   ~193 MB      │  │
│  │  staging.w2_b       (64, 2048, 768)  bf16   ~193 MB      │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

Each entry is 256-byte aligned. All tensors share one backing allocation.

### Union Layout: EP and TP Views Over the Same Bytes

The key insight is that EP and TP modes use the **same total bytes** per module when `ep_size == tp_size`:

**MoE Weights:**
```
EP: (num_experts / ep_size) × hidden × 2 × intermediate  elements
TP: num_experts × hidden × 2 × (intermediate / tp_size)  elements
    ────────────────────────────────────────────────────
    Equal when ep_size == tp_size
```

**KV Cache:**
```
EP: ep_tokens × total_kv_heads × head_dim  elements
TP: tp_tokens × (total_kv_heads / tp_size) × head_dim  elements
    where tp_tokens = ep_tokens × tp_size
    ────────────────────────────────────────────────────
    Equal (same total bytes, different shape interpretation)
```

This enables the "union layout" — one buffer region serves both modes via `get_view_as()`:

```python
# EP mode: (64 experts, 2048 hidden, 1536 intermediate)
ep_view = manager.get_view("model.layers.0.mlp.experts.w13_weight")
# shape: (64, 2048, 1536)

# TP mode: same bytes, different shape (128 experts, 768 intermediate)
tp_view = manager.get_view_as(
    "model.layers.0.mlp.experts.w13_weight",
    (128, 768, 2048),
)
# Same data_ptr, same byte count, different shape
```

## Lifecycle

### 1. Planning Phase (model `__init__`)

```
plan_qwen_moe_layout(manager, ...)
    │
    ├── For each layer:
    │   ├── Reserve EP MoE weights (w13, w2)
    │   ├── Reserve attention weights (QKV full, O_proj, QKV TP buffer)
    │   └── (FP8: also reserve scale tensors)
    │
    └── Reserve staging buffers (w13_a/b, w2_a/b)

reserve_kv_cache(manager, ...)
    │
    └── For each layer:
        ├── Reserve K buffer in EP shape
        └── Reserve V buffer in EP shape
```

### 2. Materialization

```python
manager.materialize()
# → Assigns 256-byte-aligned offsets in reservation order
# → Allocates one torch.empty(total_bytes, dtype=uint8, device="cuda")
# → Records buffer pointer range for is_managed() checks
```

### 3. Weight Allocation via Global Intercept

The manager is set as a global singleton. During model construction, `create_weights()` in `unquant.py` checks the global manager:

```python
# In UnquantizedLinearMethod.create_weights():
mgr = get_global_paras_memory_manager()
if mgr and mgr.materialized:
    entry_name = f"{layer.prefix}.weight"
    if entry_name in mgr._entries:
        weight = Parameter(mgr.get_view(entry_name), requires_grad=False)
        # → No torch.empty, weight is a view into the contiguous buffer
```

This intercept is transparent — non-ParaS models see `mgr=None` and take the normal `torch.empty` path.

Similarly for MoE weights in `UnquantizedFusedMoEMethod.create_weights()`:

```python
if use_manager:
    w13_name = f"model.layers.{layer.layer_id}.mlp.experts.w13_weight"
    if w13_name in mgr._entries:
        w13_weight = Parameter(mgr.get_view(w13_name), requires_grad=False)
```

### 4. KV Cache Integration

`MHATokenToKVPool` accepts optional `external_k_buffers` / `external_v_buffers` from the manager:

```python
# In model_runner.py:
k_bufs, v_bufs = manager.get_kv_views(num_layers, mode="ep")
pool = MHATokenToKVPool(..., external_k_buffers=k_bufs, external_v_buffers=v_bufs)
```

The pool's `data_ptrs` and `data_strides` are built from these external buffers. The pool's token allocation logic (free slots, eviction, etc.) is unchanged.

### 5. EP→TP Switch

During the switch, the buffer contents are overwritten in-place:

```
                    EP Buffer Contents                 TP Buffer Contents
                    ──────────────────                 ──────────────────
MoE w13_weight:     (64, H, 2*I) EP data      →       (128, 2*I_tp, H) TP data
MoE w2_weight:      (64, I, H) EP data        →       (128, H, I_tp) TP data
KV K per layer:     (ep_tok+1, 4, 128)        →       (tp_tok+1, 2, 128)
KV V per layer:     (ep_tok+1, 4, 128)        →       (tp_tok+1, 2, 128)
Attention weights:  unchanged (full, unsharded)
Staging buffers:    reused as scratch during switch
```

**MoE weight redistribution flow:**

```
1. all_gather (dp>1 only):
   EP buffer ──all-gather──→ staging.w13_a (gathered from all DP ranks)

2. all_to_all:
   staging.w13_a ──permute──→ staging.w13_b (contiguous for all-to-all)
   staging.w13_b ──all-to-all──→ EP buffer (overwritten with TP layout)

3. TP experts already registered:
   tp_experts.w13_weight points to EP buffer via get_view_as()
   → Data is valid immediately after all-to-all, no re-registration needed
```

**KV cache migration flow:**

```
1. gather_cache() calls paras_resize_cache() per layer:
   → Uses mgr.get_view_as() to reinterpret EP KV region as TP shape
   → gather_kv_and_permute() + all_to_all + permute_and_scatter_kv()
   → TP KV data written into the same managed buffer region
```

## Concrete Numbers (Qwen3-30B-A3B, 2×H100 80GB)

| Component | EP Mode | TP Mode | Bytes |
|-----------|---------|---------|-------|
| MoE w13 per layer | (64, 2048, 1536) bf16 | (128, 768, 2048) bf16 | 402 MB |
| MoE w2 per layer | (64, 768, 2048) bf16 | (128, 2048, 384) bf16 | 201 MB |
| QKV full per layer | (2560, 2048) bf16 | same | 10 MB |
| O_proj per layer | (2048, 2048) bf16 | same (slice) | 8 MB |
| QKV TP buffer per layer | (640, 2048) bf16 | used during switch | 2.5 MB |
| KV K per layer | (361K+1, 4, 128) bf16 | (722K+1, 2, 128) bf16 | ~370 MB |
| KV V per layer | same | same | ~370 MB |
| **Total per layer** | | | **~1.36 GB** |
| **48 layers** | | | **~65 GB** |
| Staging (shared) | 4 buffers | | ~1.16 GB |
| **Grand total** | | | **~65 GiB** |

With `mem_fraction_static=0.8`:
- Buffer: 64.9 GiB (340 entries)
- EP KV tokens: 361K
- TP KV tokens: 722K
- Switch time: 170ms weights + 181ms total
- Available GPU memory after allocation: ~10.8 GiB

## API Reference

### ParaSMemoryManager

```python
manager = ParaSMemoryManager(device="cuda")

# Planning phase
manager.reserve(name, shape, dtype) -> LayoutEntry
manager.reserve_kv_cache(num_layers, ep_max_tokens, tp_max_tokens, ...)

# Materialization
manager.materialize() -> int  # returns total bytes

# View access
manager.get_view(name) -> torch.Tensor           # typed view with reserved shape
manager.get_view_as(name, shape, dtype) -> Tensor # same bytes, different shape

# KV cache views
manager.get_kv_views(num_layers, mode="ep"|"tp", tp_size, page_size)
    -> (List[Tensor], List[Tensor])  # k_buffers, v_buffers

# Queries
manager.is_managed(tensor) -> bool
manager.dump_layout() -> List[Dict]
manager.weights_only_bytes -> int  # excludes KV entries
manager.ep_max_kv_tokens -> int
manager.tp_max_kv_tokens -> int
```

### Global Accessor

```python
set_global_paras_memory_manager(manager)  # called during model init
get_global_paras_memory_manager() -> Optional[ParaSMemoryManager]
```

### MHATokenToKVPool Extensions

```python
# External buffer support (new params)
pool = MHATokenToKVPool(...,
    external_k_buffers=List[Tensor],  # from manager.get_kv_views()
    external_v_buffers=List[Tensor],
)

# Buffer swap during switch (new method)
pool.replace_buffers(new_k_buffers, new_v_buffers, new_size)
```

## File Map

| File | Role |
|------|------|
| `paras/paras_memory_manager.py` | Core manager: reserve, materialize, get_view, layout planning |
| `paras/models/qwen3_moe.py` | Creates manager, plans layout, computes KV budget, sets global |
| `paras/layers/paras_moe_block.py` | EP→TP switch logic using staging buffers and get_view_as |
| `layers/quantization/unquant.py` | Intercepts create_weights for both linear and MoE modules |
| `mem_cache/memory_pool.py` | MHATokenToKVPool external buffer support + replace_buffers |
| `model_executor/model_runner.py` | Wires KV pool to manager, uses manager token counts |
| `layers/linear.py` | Stores `self.prefix` on LinearBase for manager lookups |

## Future Improvements

### 1. Eliminate the QKV row_stack Copy

Currently, QKV TP reconfiguration still copies q/k/v slices from the full weight into a separate `qkv_proj.tp_weight` buffer via `torch.row_stack`. This is because Q, K, and V are interleaved in the full weight and need to be re-sliced for the TP shard.

**Improvement**: Store the full QKV weight in a layout where the TP shard is a contiguous slice (e.g., store Q heads, then K heads, then V heads in head-major order instead of interleaved). Then TP reconfiguration becomes a single `view()` instead of a copy. This would also eliminate the separate `qkv_proj.tp_weight` reservation, saving ~2.5 MB per layer (120 MB for 48 layers).

### 2. Carve Staging Buffers from KV Cache Region

The 4 staging buffers currently occupy ~1.16 GiB permanently. Since the EP→TP switch only happens when requests are paused, the KV cache region has unused capacity during the switch.

**Improvement**: Instead of separate staging reservations, dynamically carve scratch space from the end of the KV cache region during the switch. This reclaims ~1.16 GiB for KV tokens during normal operation. The manager would need a `get_scratch(size_bytes)` method that returns a view into the KV tail.

### 3. TP→EP Reverse Switch

The current implementation only supports EP→TP. The reverse switch would:
- Scatter TP MoE weights back to EP layout via all-to-all (reverse direction)
- Resize KV pool back to EP shape (fewer tokens, more heads per rank)
- Reload full attention weights if they were modified

**Improvement**: Add `paras_configure_ep()` support with reverse all-to-all into the same managed buffer. The union layout already supports this — the buffer bytes are the same, just the view shapes change.

### 4. Support Shared Experts

Qwen models can have fused shared experts that run alongside routed experts. These are currently excluded from the manager (V1 scope).

**Improvement**: Reserve shared expert weights in the managed buffer. They don't participate in EP→TP redistribution (they're replicated on all ranks), so they just need a fixed reservation with no union layout.

### 5. Fused Cross-Rank Weight Transfer

The contiguous, deterministic layout enables fused NCCL/RDMA transfers during the switch. Instead of per-tensor all-to-all calls, a single large transfer could move all weights for all layers at once.

**Improvement**: Since all MoE weights across all layers are contiguous in the buffer, a single `all_to_all` on the entire MoE weight region could replace the per-layer loop. This would reduce NCCL launch overhead from 48x2 kernel launches to 2 (one for w13, one for w2).

### 6. Profile-Guided Buffer Sizing

Currently, KV token counts are computed from `mem_fraction_static * total_gpu_memory - weight_bytes`. This doesn't account for PyTorch's CUDA allocator overhead, activation memory, or other runtime allocations.

**Improvement**: Run a short profiling step (like the existing `profile_max_num_token`) that actually allocates the buffer, runs a dummy forward pass, and measures remaining free memory. This would give a more accurate KV budget.

### 7. Remove Legacy ParaSWeightBuffer

The old `ParaSWeightBuffer` class in `paras/layers/utils.py` is a dynamic pool that's no longer used for MoE weight redistribution (replaced by static staging buffers). It may still be referenced elsewhere.

**Improvement**: Audit all usages and remove the class entirely once all callers use the manager's staging buffers.

### 8. FP8 KV Cache Support

The manager supports FP8 weight dtypes but the KV cache reservation currently uses BF16. FP8 KV cache would halve the KV memory, doubling the token capacity.

**Improvement**: Wire `kv_cache_dtype=fp8` through to `reserve_kv_cache()` and ensure the pool's `store_dtype` (uint8 for FP8) matches the manager's reservation dtype.

### 9. Multi-Model / Pipeline Parallelism Support

The current global manager pattern supports one model per process. Pipeline parallelism with different model shards per rank would need per-shard managers.

**Improvement**: Replace the global singleton with a registry keyed by model instance or pipeline stage. The `create_weights` intercept would look up the appropriate manager for the current module.

### 10. Memory Accounting and Monitoring

The manager tracks byte offsets but doesn't expose runtime memory accounting (how much of the KV region is actually in use, fragmentation metrics, etc.).

**Improvement**: Add a `status()` method that reports: total buffer bytes, weight bytes, KV bytes reserved, KV bytes in use (from pool allocator), staging bytes, and waste from alignment padding. This would help with capacity planning and debugging OOM issues.
