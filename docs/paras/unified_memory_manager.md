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
│                    Contiguous uint8 Buffer                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─── MoE Weight Slots (×(N+1) = 49 for 48 layers) ─────────┐  │
│  │  paras.moe_slot.0.w13  (16, 3072, 2048) bf16  ~192 MB    │  │
│  │  paras.moe_slot.0.w2   (16, 2048, 1536) bf16  ~96 MB     │  │
│  │  ...                                                       │  │
│  │  paras.moe_slot.48.w13 (16, 3072, 2048) bf16  ~192 MB    │  │
│  │  paras.moe_slot.48.w2  (16, 2048, 1536) bf16  ~96 MB     │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─── Per-Layer Attention + KV (×48 layers) ─────────────────┐  │
│  │  QKV full weight    (2560, 2048)     bf16   ~10 MB        │  │
│  │  O_proj weight      (2048, 2048)     bf16   ~8 MB         │  │
│  │  QKV TP buffer      (640, 2048)      bf16   ~2.5 MB       │  │
│  │  KV cache K         (ep_tokens+1, 4, 128) bf16            │  │
│  │  KV cache V         (ep_tokens+1, 4, 128) bf16            │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─── Staging Buffers (NCCL path only, optional) ────────────┐  │
│  │  staging.w13_pre_permute / gather and                     │  │
│  │  staging.w2_pre_permute / gather  ~1.16 GB total         │  │
│  │  (Skipped when using peer_access method)                   │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

Each entry is 256-byte aligned. All tensors share one backing allocation.

### N+1 Slot Layout with EP/TP Aliases

EP and TP modes use the **same total bytes** per MoE module when `ep_size == tp_size`. However, they cannot share the same physical memory during the transfer — one GPU reads its EP data while another writes TP data simultaneously. The N+1 slot design solves this.

> **EP↔DP×TP:** This slot design assumes `ep_size == tp_size`, so EP and TP occupy equal bytes per layer. When the switch target is a `G·T` grid the per-GPU sizes differ (EP weights small, DP×TP weights `G×` larger; EP cache large, DP×TP cache smaller), and the equal-size slots no longer apply. The generalization replaces slots with direct per-`(mode, layer)` byte offsets in one overlapping buffer; see [unified_memory_epdptp.md](unified_memory_epdptp.md).

**Physical slots**: The manager reserves N+1 identical MoE weight slots (`paras.moe_slot.0` through `paras.moe_slot.N`) for a model with N layers.

**Virtual-to-physical mapping** (via `alias()`):
```
Physical slots:  [ slot 0 | slot 1 | slot 2 | ... | slot N ]

experts     (weight loading):  layer i → slot i+1
ep_experts  (explicit EP):     layer i → slot i+1
tp_experts  (explicit TP):     layer i → slot i
```

- `model.layers.{i}.mlp.experts.w13_weight` → slot i+1 (for `create_weights()` and checkpoint loading)
- `model.layers.{i}.mlp.ep_experts.w13_weight` → slot i+1 (explicit EP alias)
- `model.layers.{i}.mlp.tp_experts.w13_weight` → slot i (explicit TP alias)

The `experts` aliases are created before `materialize()` (as dict references to the same `LayoutEntry` objects) so that `create_weights()` in `unquant.py` can find them during model construction. The `ep_experts` and `tp_experts` aliases are created after `materialize()` via `create_paras_moe_aliases()`.

Both EP and TP views are established at model init time and **never change** — eliminating the need for `update_views()` after weight transfer.

```python
# EP mode: layer 0's weights at slot 1
ep_view = manager.get_view("model.layers.0.mlp.ep_experts.w13_weight")

# TP mode: layer 0's weights at slot 0 (different physical memory)
tp_view = manager.get_view("model.layers.0.mlp.tp_experts.w13_weight")

# ep_view.data_ptr() != tp_view.data_ptr() — separate slots, no aliasing
```

## Lifecycle

### 1. Planning Phase (model `__init__`)

```
plan_qwen_moe_layout(manager, ...)                 # or plan_gpt_oss_moe_layout(...)
    │
    ├── Reserve N+1 MoE weight slots:
    │   └── paras.moe_slot.{0..N}.w13, paras.moe_slot.{0..N}.w2
    │
    ├── Create 'experts' aliases (before materialize):
    │   └── model.layers.{i}.mlp.experts.w13_weight → slot i+1
    │
    ├── For each layer:
    │   └── Reserve attention weights (QKV full, O_proj, QKV TP buffer)
    │
    ├── If configure_method != "peer_access":
    │   └── Reserve staging buffers (pre_permute, gather; overlap uses _1/_2)
    │
    └── FP8 models also reserve scale tensors per layer

reserve_kv_cache(manager, ..., layer_specs=optional)
    └── Reserve per-layer K/V slots and register kv.ep / kv.tp entries

manager.materialize()
    ├── Assign aligned offsets in reservation order
    ├── Allocate one torch.empty(total_bytes, dtype=uint8, device="cuda")
    └── Build typed views for all weight and KV entries

create_paras_moe_aliases(manager, num_layers)
    ├── ep_experts alias: layer i → slot i+1
    └── tp_experts alias: layer i → slot i
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

ParaS can wire either the MHA-only pool or the hybrid SWA pool into managed buffers:

```python
# MHA-only models (Qwen3-MoE)
k_bufs, v_bufs = manager.get_kv_views(num_layers, mode="ep")
pool = MHATokenToKVPool(..., external_k_buffers=k_bufs, external_v_buffers=v_bufs)

# Hybrid full + sliding-window models (GPT-OSS)
pool = SWAKVPool(
    ...,
    full_external_k_buffers=full_k_bufs,
    full_external_v_buffers=full_v_bufs,
    swa_external_k_buffers=swa_k_bufs,
    swa_external_v_buffers=swa_v_bufs,
)
```

`MHATokenToKVPool` uses one external K/V buffer pair per layer. `SWAKVPool` uses two sub-pools: one for full-attention layers and one for sliding-window layers. Each layer is routed by its `LayerCacheSpec.kind`. The allocator logic (free slots, eviction, resize/rebind) is unchanged; only the backing buffers come from the manager.

### 5. EP→TP Switch

Three methods are available, selected via `PARAS_CONFIGURE_METHOD` environment variable:

**Method: `naive` (NCCL sequential)**

Per layer, sequentially:
1. All-gather EP weights across DP group (DP>1 only; no-op for DP=1)
2. Permute EP weights → staging buffer
3. NCCL `all_to_all_single`: staging → TP slot (slot i) directly
4. Reconfigure attention and switch mode

**Method: `overlap` (NCCL pipelined)**

Same as naive but with dual CUDA streams:
- Stream 1: all-to-all for layer i
- Stream 2: all-gather for layer i+1 (overlapped)
- Streams swap between layers

**Method: `peer_access` (NVLink direct, fastest)**

Per layer, no barriers between layers:
1. Custom CUDA kernel reads EP slot (i+1) with strided access
2. Kernel writes directly to peer GPU's TP slot (i) via NVLink
3. No staging buffer, no NCCL overhead

See `nvlink_peer_access_weight_transfer.md` for kernel design details.

All three methods write TP data to the TP slot (slot i), where `tp_experts` already points from init. No `update_views()` is needed.

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
plan_qwen_moe_layout(...) -> None
plan_gpt_oss_moe_layout(...) -> None
manager.reserve_kv_cache(num_layers, ep_max_tokens, tp_max_tokens, ..., layer_specs=None)

# Materialization
manager.materialize() -> int  # returns total bytes

# View access
manager.get_view(name) -> torch.Tensor           # typed view with reserved shape
manager.get_view_as(name, shape, dtype) -> Tensor # same bytes, different shape

# KV cache views
manager.get_kv_views(num_layers, mode="ep"|"tp", tp_size, page_size)
    -> (List[Tensor], List[Tensor])  # k_buffers, v_buffers

# Aliasing (post-materialize for MoE; KV aliases are registered during reserve_kv_cache/materialize)
manager.alias(alias_name, target_name)    # create entry sharing target's offset
create_paras_moe_aliases(manager, num_layers)  # create ep_experts + tp_experts aliases

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

### KV Pool Extensions

```python
# MHA-only external buffer support
pool = MHATokenToKVPool(...,
    external_k_buffers=List[Tensor],  # from manager.get_kv_views()
    external_v_buffers=List[Tensor],
)

# Hybrid SWA pool support
pool = SWAKVPool(...,
    full_external_k_buffers=List[Tensor],
    full_external_v_buffers=List[Tensor],
    swa_external_k_buffers=List[Tensor],
    swa_external_v_buffers=List[Tensor],
)

# Buffer swap during switch / pool resize
pool.replace_buffers(...)
```

## File Map

| File | Role |
|------|------|
| `paras/paras_memory_manager.py` | Core manager: reserve, materialize, get_view, `plan_qwen_moe_layout`, `plan_gpt_oss_moe_layout`, KV layout registration |
| `paras/models/qwen3_moe.py` | Creates manager, plans Qwen3 layout, computes KV budget, sets global |
| `paras/models/gpt_oss.py` | Creates manager, plans GPT-OSS layout, computes heterogeneous full+SWA KV budget, sets global |
| `paras/layers/paras_moe_block.py` | EP→TP switch logic using staging buffers and get_view_as |
| `paras/cache_transfer/{base,mha,swa,utils}.py` | Cache-transfer backends and shared per-layer gather/scatter helpers |
| `layers/quantization/unquant.py` | Intercepts create_weights for both linear and MoE modules |
| `mem_cache/memory_pool.py` | `MHATokenToKVPool` and `SWAKVPool` external buffer support + rebind helpers |
| `model_executor/model_runner.py` | Wires KV pool to manager, uses manager token counts |
| `layers/linear.py` | Stores `self.prefix` on LinearBase for manager lookups |

## Future Improvements

### 1. Eliminate the QKV row_stack Copy

Currently, QKV TP reconfiguration still copies q/k/v slices from the full weight into a separate `qkv_proj.tp_weight` buffer via `torch.row_stack`. This is because Q, K, and V are interleaved in the full weight and need to be re-sliced for the TP shard.

**Improvement**: Store the full QKV weight in a layout where the TP shard is a contiguous slice (e.g., store Q heads, then K heads, then V heads in head-major order instead of interleaved). Then TP reconfiguration becomes a single `view()` instead of a copy. This would also eliminate the separate `qkv_proj.tp_weight` reservation, saving ~2.5 MB per layer (120 MB for 48 layers).

### 2. Carve Staging Buffers from KV Cache Region

Staging buffers are now conditional via `configure_method`: they are skipped entirely for `peer_access`, which writes directly to TP slots via NVLink. For the NCCL `naive` / `overlap` paths, the staging buffers still occupy ~1.16 GiB permanently.

**Improvement**: Instead of separate staging reservations, dynamically carve scratch space from the end of the KV cache region during the switch. This reclaims ~1.16 GiB for KV tokens during normal operation. The manager would need a `get_scratch(size_bytes)` method that returns a view into the KV tail.

### 3. TP→EP Reverse Switch (Implemented)

Reverse switching is implemented. TP→EP restores EP expert weights via reverse transfer, restores EP KV layout via per-layer scatter, and rebinds the attention / KV-pool state back to EP mode. See `parallelism_switch.md` for the control flow.

### 4. Support Shared Experts

Qwen models can have fused shared experts that run alongside routed experts. These are currently excluded from the manager (V1 scope).

**Improvement**: Reserve shared expert weights in the managed buffer. They don't participate in EP→TP redistribution (they're replicated on all ranks), so they just need a fixed reservation with no union layout.

### 5. Fused Cross-Rank Weight Transfer (Implemented)

The contiguous layout now powers fused NVLink peer-access kernels for both directions. The remaining future work is to collapse the NCCL fallback from a per-layer loop to a coarser-grained collective over the MoE region.

### 6. Profile-Guided Buffer Sizing

Currently, KV token counts are computed from `mem_fraction_static * total_gpu_memory - weight_bytes`. This doesn't account for PyTorch's CUDA allocator overhead, activation memory, or other runtime allocations.

**Improvement**: Run a short profiling step (like the existing `profile_max_num_token`) that actually allocates the buffer, runs a dummy forward pass, and measures remaining free memory. This would give a more accurate KV budget.

### 7. Remove Legacy ParaSWeightBuffer

The old `ParaSWeightBuffer` class in `paras/layers/utils.py` is a dynamic pool that's no longer used for MoE weight redistribution (replaced by static staging buffers). It may still be referenced elsewhere.

**Improvement**: Audit all usages and remove the class entirely once all callers use the manager's staging buffers.

### 8. FP8 Scale Transfer for Peer Access

The peer access kernels transfer weight data (w13, w2) but do not yet handle FP8 scale tensors (`w13_weight_scale`, `w2_weight_scale`). For FP8 quantized models, these scales must also be redistributed during EP→TP switching via the same NVLink peer access mechanism.

**Improvement**: Add scale transfer kernels or extend the existing v2 kernels to also copy the corresponding scale tensors alongside the weights.

### 9. FP8 KV Cache Support

The manager supports FP8 weight dtypes but the KV cache reservation currently uses BF16. FP8 KV cache would halve the KV memory, doubling the token capacity.

**Improvement**: Wire `kv_cache_dtype=fp8` through to `reserve_kv_cache()` and ensure the pool's `store_dtype` (uint8 for FP8) matches the manager's reservation dtype.

### 9. Multi-Model / Pipeline Parallelism Support

The current global manager pattern supports one model per process. Pipeline parallelism with different model shards per rank would need per-shard managers.

**Improvement**: Replace the global singleton with a registry keyed by model instance or pipeline stage. The `create_weights` intercept would look up the appropriate manager for the current module.

### 10. Memory Accounting and Monitoring

The manager tracks byte offsets but doesn't expose runtime memory accounting (how much of the KV region is actually in use, fragmentation metrics, etc.).

**Improvement**: Add a `status()` method that reports: total buffer bytes, weight bytes, KV bytes reserved, KV bytes in use (from pool allocator), staging bytes, and waste from alignment padding. This would help with capacity planning and debugging OOM issues.
