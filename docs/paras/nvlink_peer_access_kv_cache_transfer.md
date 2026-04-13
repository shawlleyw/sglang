# NVLink Peer Access KV Cache Transfer for ParaS

## Overview

This document describes the NVLink peer access KV cache transfer optimization for the ParaS EP→TP parallelism switch. During the switch, each GPU's local KV cache (containing all KV heads for local tokens) must be redistributed so that each GPU holds a subset of KV heads for all tokens across all ranks.

The peer access approach uses a custom CUDA kernel that reads from the local EP KV buffer and writes directly to peer GPUs' TP KV buffers via NVLink, eliminating the intermediate permutation buffers and NCCL collectives used by the fallback path.

### Performance Summary (Qwen3-30B-A3B dimensions, 3 layers, bf16)

| Config | Method | Latency (ms) | Speedup |
|--------|--------|-------------|---------|
| 4×A100, 30k tokens/rank | NCCL all_to_all | 2.54 | baseline |
| 4×A100, 30k tokens/rank | **peer_access** | **1.42** | **1.79×** |
| 8×A100, 30k tokens/rank (replicated heads) | NCCL all_to_all | 4.83 | baseline |
| 8×A100, 30k tokens/rank (replicated heads) | **peer_access** | **2.96** | **1.64×** |

## Background

### KV Cache Redistribution During EP→TP Switch

In EP mode, each rank stores KV cache for all `num_kv_heads` heads but only for its local tokens. In TP mode, each rank stores KV cache for `max(1, num_kv_heads // tp_size)` heads but for all tokens across all ranks.

The redistribution is a head-splitting operation: each rank sends specific KV head shards to the ranks that own those heads in TP mode.

### Head Replication

When `num_kv_heads < tp_size` (e.g., 4 KV heads with TP=8), there aren't enough heads to give each rank a unique head. In this case, heads are replicated: multiple ranks receive the same head data. The mapping uses the formula:

```
ep_head = peer * num_kv_heads // tp_size
heads_per_peer = max(1, num_kv_heads // tp_size)
```

For 4 heads / 8 GPUs: ranks 0,1 both get head 0; ranks 2,3 both get head 1; etc.

### Previous NCCL Path

The NCCL fallback uses:
1. `gather_kv_and_permute`: gather local tokens, permute to `[heads, tokens, KV, dim]`
2. `repeat_interleave` (if replicated): expand from `num_heads` to `tp_size` virtual heads
3. `all_to_all_single`: redistribute head shards across ranks
4. `permute_and_scatter_kv`: scatter into TP KV buffers

This involves **3 HBM operations + 1 NVLink transfer** per element. The peer access approach reduces this to **1 HBM read + 1 NVLink write**.

## Key Design Decisions

### 1. N+1 KV Slot Design

Identical to the N+1 MoE weight slot design (see `nvlink_peer_access_weight_transfer.md`). The memory manager reserves N+1 KV slots for N layers:

```
KV slots:  [ slot 0 | slot 1 | slot 2 | ... | slot N ]
              ↑                                   ↑
          TP layer 0                         EP layer N-1
                     EP layer 0
                     TP layer 1
```

- EP layer `i` KV lives in slot `i+1`
- TP layer `i` KV lives in slot `i`

This eliminates aliasing between source (EP) and destination (TP) without staging buffers. The overhead is one extra KV slot (~740 MB for Qwen3-30B-A3B with 4 heads, bf16).

EP/TP aliases are created via `create_paras_kv_aliases()` after `materialize()`:
- `model.layers.{i}.kv.ep.k/v` → slot `i+1`
- `model.layers.{i}.kv.tp.k/v` → slot `i`

### 2. Fused K+V CUDA Kernel

A single kernel (`peer_access_kv_transfer`) handles both K and V buffers in one pass. The kernel iterates over `(token, kv_idx, head, dim)` tuples, where `kv_idx ∈ {0, 1}` selects between K and V offsets.

**Token indexing**: Unlike weights (which have fixed offsets), KV cache tokens are at arbitrary positions in the EP buffer. The kernel receives a `local_token_indices` array for scattered source reads. Destination positions are contiguous (sequentially allocated from a freshly cleared pool).

**Head mapping**: The unified formula `ep_head = peer * num_kv_heads / tp_size` handles all cases — split (heads ≥ tp), 1:1, and replicated (heads < tp) — with no branching.

**NVLink optimizations** (same as weight kernels):
- Warp-level peer assignment: `peer = global_warp_id % tp_size`
- int4 vectorized stores (128-bit, 16 bytes per thread)
- 8-store unrolling (4 KB contiguous per warp per iteration)
- Self-write bypass when `peer == tp_rank`
- `__ldg()` read-only cache for source reads
- Fast uint32 integer division

### 3. Synchronization

Per-layer `dist.all_reduce(barrier_tensor)` ensures all ranks finish writing to slot `i` before the next layer reads from slot `i+1`. Sequential kernel launches on the same CUDA stream enforce intra-layer ordering.

### 4. CUDA IPC Without cudaDeviceEnablePeerAccess

The peer access infrastructure uses `cudaIpcOpenMemHandle` with `cudaIpcMemLazyEnablePeerAccess` to map peer buffers into the local address space. We intentionally do NOT call `cudaDeviceEnablePeerAccess()`, which would create full CUDA contexts (~416 MiB each) on every peer GPU.

The lazy IPC flag is sufficient for NVLink stores. This matches DeepEP's approach (see `DeepEP/csrc/deep_ep.cpp Buffer::sync()`). Savings: ~2.9 GB per GPU on 8-GPU systems.

## Data Flow

### Source: EP KV Buffer (scattered read)

```
EP KV slot[i+1]: (ep_max_tokens + page_size, num_kv_heads, head_dim) bf16

For each local token t (position = local_token_indices[t]):
  For each peer p:
    ep_head = p * num_kv_heads / tp_size
    Read: ep_k[local_token_indices[t], ep_head : ep_head + heads_per_peer, :]
```

### Destination: Peer TP KV Buffer (contiguous write via NVLink)

```
TP KV slot[i] on peer p: (tp_view_tokens, heads_per_peer, head_dim) bf16

dst_token = dst_token_start + t   (contiguous, dst_token_start = sum(global_num_tokens[:tp_rank]))
Write: peer_tp_k[dst_token, 0 : heads_per_peer, :]
```

Both K and V follow the same pattern with different base offsets.

## NCCL Fallback Path

### gather_kv_and_permute Dimension Ordering

The permutation outputs `[heads, tokens, KV, dim]` (NOT `[heads, KV, tokens, dim]`). This is critical: each head's chunk in the flat buffer is token-interleaved (`t0_K, t0_V, t1_K, t1_V, ...`). After `all_to_all` splits by head and concatenates received chunks, the result is `[total_tokens, KV, heads, dim]`, which `permute_and_scatter_kv` expects.

The previous ordering `[heads, KV, tokens, dim]` made each received chunk KV-grouped (`K_all_tokens, V_all_tokens`), causing K/V misalignment for N>1 tokens per sender.

### Head Replication via repeat_interleave (Option A)

When `num_kv_heads < tp_size`, the permuted buffer has fewer head chunks than `all_to_all` destinations. We use `repeat_interleave` to expand from `num_heads` to `tp_size` virtual heads:

```python
if replication_factor > 1:
    permuted = (
        permuted
        .view(num_heads, N * 2 * head_dim)
        .repeat_interleave(replication_factor, dim=0)
        .flatten()
    )
# Virtual heads 0,1 = copies of real head 0 → sent to ranks 0,1
```

**Alternative considered (Option B)**: Split `head_dim` into sub-heads, `all_to_all` the sub-heads, then `all_gather` within replication groups to reconstruct full heads. This avoids duplicating data before `all_to_all` and scales better for large replication factors (e.g., 2 heads / 8 GPUs). We chose Option A for simplicity since this is the NCCL fallback path and replication_factor > 2 is rare.

| Aspect | Option A (repeat_interleave) | Option B (sub-head + all_gather) |
|--------|------------------------------|----------------------------------|
| Collectives | 1 (all_to_all) | 2 (all_to_all + all_gather) |
| Extra HBM | repeat_interleave writes 2× buffer | None (but all_gather needs recv buffer) |
| Sub-groups | None | Need dist.new_group per replication group |
| Code complexity | +3 lines | +20 lines |
| Scales with replication_factor | NVLink grows linearly | NVLink constant for all_to_all |

## File Map

| File | Role |
|------|------|
| `paras/csrc/peer_access_transfer.cu` | CUDA kernel: `peer_access_kv_transfer` (fused K+V) |
| `paras/csrc/binding.cpp` | PyTorch C++ binding: `launch_peer_access_kv_transfer` |
| `paras/peer_access.py` | IPC handle exchange, `peer_access_kv_transfer()` Python wrapper |
| `paras/gather_manager.py` | `_gather_cache_peer_access()`, `_gather_cache_nccl()`, `gather_kv_and_permute()`, `permute_and_scatter_kv()` |
| `paras/paras_memory_manager.py` | N+1 KV slot reservation, `create_paras_kv_aliases()` |
| `paras/models/qwen3_moe.py` | Wires `create_paras_kv_aliases()` into model init |
| `paras/scheduler_paras_mixin.py` | Passes `peer_ctx` and `PARAS_KV_TRANSFER_METHOD` to gather manager |
| `mem_cache/memory_pool.py` | `paras_resize_cache()` N+1 TP alias path |
| `test/srt/test_paras_kv_peer_access.py` | Multi-GPU correctness + benchmark (4-GPU and 8-GPU) |

## Configuration

```bash
# Select KV cache transfer method (default: nccl)
export PARAS_KV_TRANSFER_METHOD=peer_access   # NVLink direct (fastest)
export PARAS_KV_TRANSFER_METHOD=nccl          # NCCL all_to_all fallback
```

## Future Work

1. **TP→EP reverse KV transfer**: Process layers in reverse order (N-1, ..., 0). Layer `i`'s read from slot `i` completes before layer `i-1`'s write to slot `i`.

2. **FP8 KV cache**: Wire `kv_cache_dtype=fp8` through to `reserve_kv_cache()` and kernel elem_size.

3. **NVSHMEM migration**: Replace CUDA IPC with NVSHMEM symmetric heap allocation (`nvshmem_align` + `nvshmemi_get_p2p_ptr`). Would eliminate IPC handle exchange entirely. Medium-large effort — requires coordinating with DeepEP's NVSHMEM init and wrapping NVSHMEM memory as torch tensors. Not needed now that `cudaDeviceEnablePeerAccess` overhead is eliminated.
