# SWA (Sliding Window Attention) in SGLang and ParaS

This document covers the sliding-window attention (SWA) mechanism in sglang — how it is implemented, how the K/V pool is bounded dynamically during decode, and how ParaS adapts and transports the SWA pool across EP↔TP switches. It also documents the redundancy still present in the current cache-transfer path and the cleanest follow-up design that would eliminate it.

## Sliding window semantics

Sliding-window attention bounds each token's receptive field to the most recent `W` positions:

- At decode position `p`, attention attends to keys/values at positions `[p - W + 1, p]` — `W` positions including the current one.
- Positions outside the window contribute zero (typically masked at compute time).
- Used by gpt-oss, Mistral, Gemma3, and other modern decoder models, interleaved with full-attention layers in hybrid configurations (e.g., gpt-oss-120b alternates full + sliding-window across its 36 layers).

For SGLang's purposes the salient consequence is: per request, only `min(W, P + decode_steps)` K/V slots in the SWA pool need to be live at any moment, regardless of how many tokens have been generated. Anything older than `W` can be freed.

Note that this is fundamentally different from **chunked attention** (Llama4-style), where the sequence is partitioned into fixed-size non-overlapping chunks and each token attends only within its own chunk. Both reduce K/V footprint but with different eviction granularity (per-token for sliding window vs per-chunk for chunked attention).

## SGLang's SWA implementation

For hybrid SWA models, sglang carries **two K/V pools** wired into a single allocator:

- **Full pool**: stores K/V for the full-attention layers (or for the full-attention-equivalent slots when `--disable-hybrid-swa-memory` is set). Size: `full_max_tokens` per layer.
- **SWA pool**: stores K/V for the sliding-window layers. Size: `swa_max_tokens` per layer, typically smaller than the full pool (e.g., `swa_full_tokens_ratio = 0.8`).

Both pools are managed by [`SWATokenToKVPoolAllocator`](file:///home/shaoyuw/sglang/python/sglang/srt/mem_cache/allocator.py), which performs **lockstep allocation** (`alloc(N)` returns N full slots AND N SWA slots together) and maintains a translation tensor `full_to_swa_index_mapping[full_slot] → swa_slot`. Reads and writes from SWA layers go through this mapping.

The tree cache for hybrid SWA models is one of:

- [`SWARadixCache`](file:///home/shaoyuw/sglang/python/sglang/srt/mem_cache/swa_radix_cache.py) — default, with prefix-matching and tombstone-aware nodes. Used when radix cache is enabled. See [`radix_cache.md`](file:///home/shaoyuw/sglang/docs/paras/radix_cache.md) for why ParaS doesn't use this.
- [`SWAChunkCache`](file:///home/shaoyuw/sglang/python/sglang/srt/mem_cache/chunk_cache.py) — used when `--disable-radix-cache` is set. No tree, no prefix matching; just per-request slot tracking through `req_to_token_pool`. **This is what ParaS uses.**

A request's `req_to_token_pool[req_pool_idx, p]` holds the full-pool slot index for position `p`. SWA reads translate to the SWA pool via `full_to_swa_index_mapping[full_slot]`. Mapping value `0` means "no SWA slot for this position" — attention reads padding from slot 0 (the reserved padding slot).

## Dynamic SWA eviction during decode (baseline)

Prior to PR #17220, sglang did NOT evict in-flight requests' SWA slots during decode. The SWA pool grew with each decoded token; only finished requests' SWA slots were ever reclaimed (via `cache_finished_req`). This meant the SWA pool's hybrid-design memory benefit was not realized for long-decode in-flight requests.

[PR #17220](https://github.com/sgl-project/sglang/pull/17220) added runtime SWA eviction during decode. The mechanism is per-Req:

- `Req.swa_evicted_seqlen: int` — counts how many leading SWA slots have been freed for this request.
- `ScheduleBatch.maybe_evict_swa()` — called at the top of `alloc_for_extend` and `alloc_for_decode` in [`mem_cache/common.py`](file:///home/shaoyuw/sglang/python/sglang/srt/mem_cache/common.py).
- For each in-flight req in decode mode:
  ```python
  new_swa_evicted = max(req.swa_evicted_seqlen, pre_len - sliding_window_size)
  if new_swa_evicted > req.swa_evicted_seqlen:
      free_slots = req_to_token[idx, req.swa_evicted_seqlen:new_swa_evicted]
      allocator.free_swa(free_slots)
      req.swa_evicted_seqlen = new_swa_evicted
  ```

Key invariants:

- **Monotonic max**: `req.swa_evicted_seqlen` only ever increases; slots `[0, current_swa_evicted_seqlen)` are never re-scanned.
- **Idempotent `free_swa`**: skips slots whose `full_to_swa_index_mapping` is already 0, so calling it on already-freed positions is a no-op.
- **Floor at request's tree-locked prefix length** in the radix-cache version (via `cache_protected_len`). For the chunk-cache version no floor is needed since there's no tree.

Steady-state SWA pool occupancy per request: `min(W, P + decode_steps) ≈ W` once `decode_steps > W`. For 32 concurrent reqs with `W = 128`, total SWA pool usage stabilizes at ~32 × 128 = 4096 tokens regardless of how long each request has been decoding.

ParaS adopts this mechanism by porting the per-Req field and the eviction methods (without the radix-cache tombstone machinery, since ParaS uses `SWAChunkCache`). See `swa_evicted_seqlen` and `ScheduleBatch.maybe_evict_swa` in [`schedule_batch.py`](file:///home/shaoyuw/sglang/python/sglang/srt/managers/schedule_batch.py).

## ParaS SWA cache transfer: current implementation

When ParaS switches between EP and TP, in-flight requests' K/V must be redistributed across ranks. Each layer's K/V is transferred independently via per-layer dispatch through `MHACacheTransfer` (full layers) or [`SWACacheTransfer`](file:///home/shaoyuw/sglang/python/sglang/srt/paras/cache_transfer/swa.py) (sliding-window layers).

The current strategy is **transfer-then-tighten**:

### Step 1: Pre-switch source state

After source-side `maybe_evict_swa` has been running every decode step:

| Per-req position | `req_to_token[idx, p]` | Source `full_to_swa_index_mapping[full_slot]` | SWA pool slot K/V |
|---|---|---|---|
| `p ∈ [0, swa_evicted_seqlen)` | live full slot | **0** (evicted) | (freed) |
| `p ∈ [swa_evicted_seqlen, S-1)` | live full slot | non-zero (alive) | real K/V |

The source's SWA pool holds only the in-window slice per request. The source's full pool holds the entire sequence (full attention layers need it).

### Step 2: Destination lockstep alloc

In `gather_manager.reorchestrate_cache` (or `scatter_manager`):

1. `paras_resize_and_clear` resets allocators to their post-switch capacities and zeros `full_to_swa_index_mapping`.
2. `alloc(num_global_tokens)` allocates N = sum of (S-1) per req **full** slots AND N **SWA** slots lockstep. After this, the destination's `full_to_swa_index_mapping` has non-zero entries for every alloc'd full slot — including the slots that correspond to OOW positions.
3. `req_to_token_pool` is filled with the new full slot indices for each req's positions `[0, S-1)`.

### Step 3: Cache transfer (per layer)

For SWA layers, `SWACacheTransfer.gather_one_layer` / `scatter_one_layer`:

- Reads source SWA slots via `_full_to_swa_source(local_indices)` — uses a snapshot of the source mapping (pre-resize) so OOW positions read padding from source SWA slot 0.
- Writes destination SWA slots via `_full_to_swa(global_indices)` — uses the live destination mapping. After lockstep alloc, ALL positions have non-zero mapping, so writes go to real destination SWA slots (including OOW positions, which receive padding data).

### Step 4: Destination tightening (`_tighten_swa_pool_to_in_window`)

After the alloc loop, [`gather_manager`](file:///home/shaoyuw/sglang/python/sglang/srt/paras/gather_manager.py) and [`scatter_manager`](file:///home/shaoyuw/sglang/python/sglang/srt/paras/scatter_manager.py) run a fixup:

```python
for req in reqs:
    in_window_start = max(req.swa_evicted_seqlen, seqlen - 1 - W)
    if in_window_start > 0:
        oow_full_slots = req_to_token[idx, 0:in_window_start]
        allocator.free_swa(oow_full_slots)  # zeros mapping, returns SWA slots
        req.swa_evicted_seqlen = in_window_start
```

This is structurally necessary because `maybe_evict_swa`'s monotonic-max invariant prevents the dynamic eviction from ever reclaiming slots in `[0, swa_evicted_seqlen_initial)` on the destination. The lockstep alloc unconditionally allocates SWA slots for every position, including positions whose source mapping was 0. Without `_tighten`, those slots are stuck in the allocated state until the request finishes.

### End state

After step 4, per migrated request on the destination:

- Full pool: `S-1` slots allocated, all populated with real K/V (full attention works).
- SWA pool: `min(W, S-1-X)` slots allocated, populated with real K/V; the rest have mapping = 0.
- `req.swa_evicted_seqlen = max(inherited_X, S-1-W)` — set so subsequent `maybe_evict_swa` on the destination is consistent.

Validated empirically (Phase D, 4×A100, gpt-oss-120b): post-switch SWA occupancy = `num_migrated_reqs × W` exactly. For 8 migrated reqs with W=128, total = 1024 SWA tokens.

## Known inefficiency: redundant transfer for OOW positions

The transfer-then-tighten approach has the right **destination memory footprint** but does redundant **transfer work**.

The flat `local_token_indices` tensor built in `gather_manager.__init__` concatenates **all** `[0, S-1)` positions per req — including the OOW positions `[0, X)` whose source SWA mapping is already 0. Both `MHACacheTransfer` and `SWACacheTransfer` receive this same flat tensor.

For SWA layers, the kernel iterates over every position. OOW positions read source SWA slot 0 (padding) via the source mapping and write to destination SWA slot 0 (padding) via the destination mapping (`_tighten` zeros the destination mapping before transfer fires for `SWACacheTransfer` only if `_tighten` ran first — in the current Phase C, `_tighten` runs in `reorchestrate_cache` before `gather_cache`, so this ordering does hold). The result is correct (padding is harmless), but the kernel still touches those positions:

- NCCL path: padding data is serialized across the `all_to_all_single` collective. Wire bandwidth wasted.
- Peer-access path: kernel issues NVLink writes for padding positions (all targeting slot 0). Compute time wasted.

Magnitude on gpt-oss-120b, 4×A100, typical in-flight switch (32 reqs at avg seqlen 2000, W=128 → X ≈ 1872):

- 18 SWA layers
- 2 KV heads/rank × 128 head_dim × bf16 = 512 bytes per (K or V) per token, ~1 KB for K+V combined
- Wasted data per switch: 32 reqs × 1872 OOW positions × 18 SWA layers × 1 KB ≈ **1.05 GB**
- At ~100 GB/s NVLink: **~10 ms wasted per switch**, ~1-3% of typical in-flight switch latency.

The fix is straightforward in concept and is the cleanest possible design (see next section), but it has not been implemented because the current latencies are not transfer-bound.

## Cleanest design: source-side filter at index-build time

The current approach is "transfer all source positions, then fix up destination after". The clean alternative is "compute the in-window subset once, transfer only that".

### What changes

In `gather_manager.__init__` (and the matching scatter path), additionally build SWA-filtered tensors:

```python
local_token_indices_swa_parts = []
for req in local_reqs:
    seqlen_no_last = req.seqlen - 1
    in_window_start = max(req.swa_evicted_seqlen, seqlen_no_last - W)
    local_token_indices_swa_parts.append(
        req_to_token_pool.req_to_token[req.req_pool_idx, in_window_start:seqlen_no_last]
    )
self.local_token_indices_swa = torch.cat(local_token_indices_swa_parts)
self.num_local_tokens_swa = self.local_token_indices_swa.shape[0]
```

After all-gathering `swa_evicted_seqlen` and `seqlen` per req across paras_tp ranks (already free via the existing pickle-based gather), compute the symmetric `global_token_indices_swa`, `num_global_tokens_swa`, and the per-rank `global_num_tokens_swa` list.

Pass the SWA-filtered tensors to `SWACacheTransfer`'s constructor (`local_token_indices_swa`, etc.) instead of the unfiltered ones. `MHACacheTransfer` continues to receive the unfiltered tensors.

### What that buys

| Property | Transfer-then-tighten (current) | Source-side filter (proposed) |
|---|---|---|
| Destination SWA memory at switch boundary | tightened to `W × num_reqs` via `_tighten` | tightened to `W × num_reqs` by allocating only what's transferred (via [asymmetric allocator API](#asymmetric-allocator-api-required)) |
| Source-side transfer iteration | iterates over all `[0, S-1)` positions per req | iterates over only `[in_window_start, S-1)` per req |
| Wire/NVLink data | includes padding for OOW positions | only in-window data |
| Estimated switch latency improvement | — | ~10 ms on gpt-oss-120b 32-req scenario (1-3% of switch) |
| Code complexity | `_tighten` (~25 LOC × 2 managers) | asymmetric alloc API + per-req metadata flow + filtered tensor construction (~80-150 LOC across `cache_transfer/base.py`, gather/scatter managers, allocator) |

### Asymmetric allocator API required

Without `_tighten`, the destination still needs the full pool sized at `num_global_tokens_full` (for the full attention layers) but the SWA pool sized at `num_global_tokens_swa < num_global_tokens_full`. The current lockstep `alloc(N)` cannot do this. New entry points needed on `SWATokenToKVPoolAllocator`:

```python
def alloc_full_only(self, need_size: int) -> torch.Tensor:
    """Alloc full slots without touching SWA allocator."""
def alloc_swa_only(self, need_size: int) -> torch.Tensor:
    """Alloc SWA slots only (caller is responsible for the corresponding full slots)."""
def set_full_to_swa(self, full_slots: torch.Tensor, swa_slots: torch.Tensor):
    """Establish the mapping for in-window positions only."""
```

After alloc:
- `alloc_full_only(num_global_tokens_full)` → all `[0, S-1)` per req get full slots.
- `alloc_swa_only(num_global_tokens_swa)` → only in-window positions get SWA slots.
- `set_full_to_swa(in_window_full_slots, swa_slots)` → maps only the in-window full slots to live SWA slots; OOW full slots have mapping = 0 by default (since `paras_resize_and_clear` zero-fills the mapping).

With this in place, `_tighten` becomes unnecessary: the destination has the right shape from the start.

### Status

Not implemented. The current `_tighten` solution is correct and validated, and the ~10 ms / 1-3% transfer-time gain isn't a forcing function yet. If profiling shows in-flight switch latency becoming transfer-bound (e.g., at higher context lengths or larger paras_tp_size), this is the design to land.

## References

- [`schedule_batch.py:maybe_evict_swa`](file:///home/shaoyuw/sglang/python/sglang/srt/managers/schedule_batch.py) — per-decode-step dynamic SWA eviction.
- [`mem_cache/allocator.py:SWATokenToKVPoolAllocator`](file:///home/shaoyuw/sglang/python/sglang/srt/mem_cache/allocator.py) — lockstep alloc and `free_swa`.
- [`mem_cache/chunk_cache.py:SWAChunkCache`](file:///home/shaoyuw/sglang/python/sglang/srt/mem_cache/chunk_cache.py) — tree-less cache used by ParaS.
- [`paras/gather_manager.py:_tighten_swa_pool_to_in_window`](file:///home/shaoyuw/sglang/python/sglang/srt/paras/gather_manager.py) — destination-side post-alloc fixup.
- [`paras/cache_transfer/swa.py:SWACacheTransfer`](file:///home/shaoyuw/sglang/python/sglang/srt/paras/cache_transfer/swa.py) — per-layer SWA transfer dispatch.
- [`docs/paras/radix_cache.md`](file:///home/shaoyuw/sglang/docs/paras/radix_cache.md) — why ParaS does not use `SWARadixCache`.
- [`docs/paras/parallelism_switch.md`](file:///home/shaoyuw/sglang/docs/paras/parallelism_switch.md) — overall EP↔TP switch design.
- PR #17220 (commit `ce8a6ac69`) on sglang main — the dynamic SWA eviction concept ParaS adopted.
