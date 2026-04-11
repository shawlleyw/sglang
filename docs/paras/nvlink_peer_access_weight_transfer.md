# NVLink Peer Access Weight Transfer for ParaS

## Overview

This document describes the NVLink peer access weight transfer optimization for the ParaS EP→TP parallelism switch. Instead of using NCCL `all_to_all` collectives for MoE weight redistribution, we use custom CUDA kernels that write directly to peer GPU memory via NVLink, achieving a **1.23× speedup** over NCCL on Qwen3-30B-A3B with 4×A100-80GB.

### Performance Summary

| Method | 8 layers | 48 layers (projected) | vs NCCL |
|--------|----------|----------------------|---------|
| NCCL `all_to_all` | 18.2ms | ~109ms | baseline |
| Peer access (NVLink direct) | 14.7ms | ~88ms | **1.23×** |

## Background

The ParaS Unified Memory Manager allocates all MoE weights in a single contiguous buffer with deterministic offsets (see `unified_memory_manager.md`). During EP→TP switching, each GPU must redistribute its local expert weights to all peers. The NCCL path uses:

1. Permute EP weights into a staging buffer (HBM write)
2. NCCL `all_to_all_single` (NVLink transfer)

This involves **2 HBM operations + 1 NVLink transfer** per element. The peer access approach eliminates step 1.

## Key Design Decisions

### 1. CUDA IPC for Cross-Process Peer Access

In sglang's multi-process architecture (`torchrun`), each GPU runs in a separate process with its own virtual address space. Raw `data_ptr()` values are meaningless across processes.

**Solution**: Use CUDA IPC (Inter-Process Communication) handles:
- Each rank calls `cudaIpcGetMemHandle()` on its managed buffer
- Handles are exchanged via `all_gather` (64 bytes per rank)
- Each rank calls `cudaIpcOpenMemHandle()` to map peer buffers into its address space
- The resulting pointers are valid for direct NVLink stores from CUDA kernels

This initialization takes ~6 seconds (NVLink connection setup) and is performed once during model loading, not during the switch.

### 2. N+1 Slot Design: Eliminating Staging Buffers

**The problem**: In the original memory manager, EP and TP modes share the same physical buffer for each layer via `get_view_as()`. During transfer, one GPU reads its EP data from slot `i` while another GPU writes TP data to the same slot `i` — a race condition.

The NCCL path solves this with staging buffers: EP data is first copied to a staging buffer, then the `all_to_all` reads from staging and writes to the EP/TP buffer. This requires an extra ~1.16 GiB of staging memory and adds an HBM write.

**Our solution**: Allocate N+1 layer slots instead of N:

```
Buffer slots:  [ slot 0 | slot 1 | slot 2 | ... | slot N ]
                  ↑                                   ↑
              TP layer 0                         EP layer N-1
                         EP layer 0
                         TP layer 1
```

- EP layer `i` lives in slot `i+1`
- TP layer `i` lives in slot `i`
- Source (slot `i+1`) and destination (slot `i`) are always different physical regions

This eliminates the aliasing race condition without staging buffers. The overhead is 1 extra layer slot ≈ 288 MB (0.4% of 65 GiB total).

**Virtual-to-physical slot mapping**: The memory manager's `alias()` method creates virtual entries that map to the same physical offset as a target entry. After `materialize()`, `create_tp_aliases()` registers two aliases per layer — an EP alias (`model.layers.{i}.mlp.experts.*`) pointing to slot `i+1`, and a TP alias (`model.layers.{i}.mlp.tp_experts.*`) pointing to slot `i`. Both views are created at model init time and never change, eliminating the need for `update_views()` after fused transfer. The kernel and MoeBlock code use uniform TP alias lookups with no `layer_id == 0` branching.

**Layer ordering constraint**: EP→TP must process layers in forward order (0, 1, 2, ..., N-1). Layer `i+1`'s write to slot `i+1` must not overlap with layer `i`'s read from slot `i+1`. Sequential kernel launches on the same CUDA stream enforce this ordering.

### 3. Process Streamline: Barrier Placement

**Initial design** (96 barriers, 4+ seconds): Two `dist.barrier()` calls per layer — before peer writes and after peer writes. With 48 layers, this produced 96 barriers at ~40ms each.

**Optimized design** (0 barriers): Since the N+1 slot design eliminates inter-layer aliasing and the scheduler ensures quiescence before the switch, no barriers are needed in the model-level method:

```python
for layer in layers:
    launch_kernel(layer)     # No barriers between layers
for layer in layers:
    configure_tp_attn(layer) # Reconfigure attention
    configure_tp(layer)      # Switch to TP mode (views already correct)
```

TP views point to the correct physical slots from init (via virtual-to-physical alias mapping), so no `update_views()` pass is needed. The per-layer convenience wrapper retains barriers for standalone test use.

### 4. Kernel Design: NVLink Store Optimization

The kernel follows the guidelines in `nvlink_peer_access_guielines.md`:

**Grid configuration**:
- `num_SMs × tp_size` blocks (432 on A100 with 4 GPUs)
- 256 threads (8 warps) per block
- Dynamically queries SM count via `cudaDeviceGetAttribute`

**Warp-level peer assignment**:
```cuda
int peer = global_warp_id % tp_size;
int warp_index = global_warp_id / tp_size;
```
This distributes NVLink traffic evenly across all peers from every SM, maximizing bandwidth utilization.

**Vectorized stores**: All reads and writes use `int4` (128-bit / 16 bytes per thread), producing 512-byte coalesced warp transactions — the optimal NVLink transaction size.

**8-store unrolling**:
```cuda
#pragma unroll 8
for (int u = 0; u < 8; u++) {
    // 8 × 32 lanes × 16B = 4KB contiguous per warp per iteration
}
```

**Self-write bypass**: When `peer == tp_rank`, the destination is on the same GPU. The kernel bypasses the IPC pointer and writes directly to the local buffer, avoiding UVA address resolution overhead:
```cuda
char* dst_buf = (peer == tp_rank) ? const_cast<char*>(local_buffer) : peer_buffers[peer];
```

**Read-only cache**: Source reads use `__ldg()` (texture cache path) for better L2 utilization on non-reused data.

**Fast integer division**: Index decomposition uses `uint32` arithmetic (hardware 32-bit divider, ~20 cycles) instead of `int64` software division (~100 cycles).

### 5. Kernel Tuning

We tuned grid size and thread count via environment variables (`V2_GRID_MULT`, `V2_THREADS`) and settled on:
- Grid: `108 × 4 = 432` blocks (4 blocks per SM)
- Threads: 256 (8 warps per block)

Higher block counts (864 = 8 per SM) showed no improvement, suggesting NVLink bandwidth is the bottleneck, not SM occupancy.

## Data Flow

### w13 (gate + up projection)

EP shape: `(E_local, 2, tp_size, I'×H)` — the TP dimension is embedded in the weight layout. For each `(expert, gate, peer)` combination, `I'×H` elements are **contiguous** in the source buffer.

```
Source: local EP buffer, slot[i+1]
  For block (peer=r, expert=e, gate=k):
    src = ep_offset + (e × 2 × tp_size + k × tp_size + r) × I'H × elem_size
    → I'H contiguous bytes (1.5 MB per chunk)

Destination: peer r's TP buffer, slot[i]
  dst = tp_offset + (tp_rank × E_local × 2 + e × 2 + k) × I'H × elem_size
  → I'H contiguous bytes
```

Both reads and writes are fully coalesced. This is the ideal case for NVLink stores.

### w2 (down projection)

EP shape: `(E_local, H, I_full)` — TP split on the last dimension. For peer `r`, columns `[r×I', (r+1)×I')` from each row.

Within each row, the `I'` elements **are contiguous**. Between rows, there's a stride of `I_full`. The kernel reads row-by-row with `int4` vectorization:

```
Source: row h of expert e, peer r's shard
  src = ep_offset + e × H × I_full_bytes + h × I_full_bytes + r × I'_bytes
  → I'_bytes contiguous (768 bytes per row)

Destination: row h of expert (tp_rank × E_local + e) on peer r
  dst = tp_offset + (tp_rank × E_local + e) × H × I'_bytes + h × I'_bytes
  → I'_bytes contiguous
```

Both reads and writes are coalesced within each row. The row stride causes 25% HBM cache utilization (read 768B from 3072B cache line), but HBM bandwidth (3.35 TB/s) is not the bottleneck — NVLink (150 GB/s) is.

## Theoretical Analysis

For Qwen3-30B-A3B, 4×A100-80GB:

```
Per layer: 288 MB total (192 MB w13 + 96 MB w2)
NVLink send per GPU per layer: 216 MB (3/4 to peers)
NVLink send per GPU, 48 layers: 10.4 GB

A100 NVLink bandwidth: ~150 GB/s unidirectional (achieved)
Theoretical minimum: 10.4 GB / 150 GB/s = 69 ms (48 layers)

Measured: ~88 ms (48 layers projected) = 78% of peak NVLink bandwidth
```

The 22% gap is primarily from integer division overhead in index computation and the w2 strided read pattern.

## File Map

| File | Role |
|------|------|
| `paras/csrc/peer_access_transfer.cu` | CUDA kernels: `peer_access_fused_transfer_w13_v2`, `peer_access_fused_transfer_w2_v2` |
| `paras/csrc/binding.cpp` | PyTorch C++ bindings exposing kernels to Python |
| `paras/csrc/setup.py` | Standalone CUDA extension build (`pip install -e`) |
| `paras/peer_access.py` | Peer access init (IPC handles), Python kernel wrappers |
| `paras/layers/paras_moe_block.py` | Per-layer kernel launch (`paras_configure_tp_fused_peer_access_kernel`) |
| `paras/layers/paras_model.py` | Model-level orchestration with 2-barrier design |
| `paras/models/qwen3_moe.py` | Pre-initializes peer access during model load |
| `paras/paras_memory_manager.py` | N+1 slot reservation (`paras.moe_slot0.*`), `alias()`, `create_tp_aliases()` |
| `test/srt/test_paras_peer_access.py` | 4-GPU correctness + benchmark test |

## Future Work

1. **TP→EP reverse switch**: The N+1 slot design supports this by processing layers in reverse order (N-1, N-2, ..., 0). Layer `i`'s read from slot `i` completes before layer `i-1`'s write to slot `i`.

2. **KV cache migration**: Currently uses NCCL. Could use the same peer access approach since KV buffers are also in the managed buffer.

3. **Larger TP groups (8 GPUs)**: More peers = more NVLink bandwidth available. The warp-level peer assignment scales naturally.

4. **Kernel fusion**: Fusing w13 and w2 into a single kernel launch per layer halves launch overhead. A combined kernel was prototyped but showed marginal improvement (~0.1ms) since NVLink bandwidth dominates.

5. **Eliminate index division**: Restructuring the kernel to iterate over (chunk, position) pairs instead of flat indices would remove the per-element integer division, potentially closing the remaining 22% gap to theoretical peak.

### Kernel Optimization Opportunities

1. **Eliminate index division**: The inner loop computes `chunk_id = idx / int4_per_chunk` and `pos = idx % int4_per_chunk` per element. Restructuring to iterate over (chunk, position) pairs would remove this (~20 cycles per division × millions of iterations).

2. **Two-phase even/odd launch**: Launch even layers (0,2,4,...) as one kernel, sync, then odd layers (1,3,5,...). Even layers' reads (odd slots) and writes (even slots) never overlap. This enables a mega-kernel launch for half the layers at once.

3. **Cooperative groups grid sync**: Use `cooperative_launch` with `__grid_sync()` to synchronize between even and odd phases within a single kernel launch, eliminating the inter-phase sync overhead.

4. **Warp specialization**: Dedicate specific warps to specific chunk sizes. w13 chunks (1.5MB) benefit from many warps; w2 rows (768B) might benefit from fewer warps with better cache locality.

5. **L2 cache prefetch**: Use `__prefetch_l2()` hints to pre-load the next chunk's source data while the current chunk's NVLink writes are in flight.

6. **Adaptive grid sizing**: Instead of fixed `num_SMs × tp_size`, dynamically compute grid size based on total data volume and per-SM NVLink bandwidth target.
