# NVLink Peer Access Weight Transfer for ParaS

## Overview

This document describes expert-weight redistribution kernels. Buffer ownership,
workspace reuse, attention reconstruction, and migration ordering are defined
in the [unified memory manager reference](unified_memory_manager.md).
The runtime requires peer-access weight switching; legacy NCCL model-level
`naive` and `overlap` strategies are no longer supported by the planner.

The current model calls the v2 expert kernels through `peer_access.py`.
That module also exposes optional v3 variants for explicit callers; their
presence does not change the default runtime selection.

### Historical Performance Summary (Qwen3-30B-A3B, 48 layers, 4×A100-80GB)

| Method | transfer_weights | configure TP total | vs naive |
|--------|-----------------|-------------------|----------|
| `naive` (NCCL sequential) | ~96 ms | ~117 ms | baseline |
| `overlap` (NCCL pipelined) | ~83 ms | ~100 ms | 1.17× |
| `peer_access` (NVLink direct) | **~61 ms** | **~97 ms** | **1.57×** |

These measurements predate the combined weights/KV/workspace layout and
are not current end-to-end switch measurements. In that run, peer access
kernel time was only **~9 ms** for all 48 layers. The remaining `transfer_weights` time is dominated by attention and TP reconfiguration overhead shared by all methods. The `configure TP` total includes cache migration, request gathering, and weight transfer.

## Background

The ParaS Unified Memory Manager allocates all MoE weights in a single contiguous buffer with deterministic offsets (see `unified_memory_manager.md`). During EP→TP switching, each GPU must redistribute its local expert weights to all peers. The earlier NCCL weight path used:

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

### 2. Planned Source and Destination Views

`plan_layout` assigns fixed per-mode offsets for
`model.layers.{i}.mlp.ep_experts.*` and `mlp.tp_experts.*`. Checkpoint-loading
names `mlp.experts.*` alias EP entries. The kernels consume these offsets;
they do not derive addresses from an N+1 slot index.

EP→TP sends intermediate-dimension slices of each local expert to its TP
owner. TP→EP sends each expert's shards back to its EP owner. Source and
destination for the current layer are disjoint, while subsequent destinations
can reuse earlier source bytes. Whole-model transfer order and fences are
therefore part of the correctness contract.

The model transfers attention QKV/O alongside experts before fencing each
layer. See [migration safety](unified_memory_manager.md#migration-safety)
for the local EP→TP slices and direct peer reads used to reconstruct EP
attention, including replicated KV heads.

### 3. Cross-Rank Synchronization

**The problem**: Fixed offsets protect the current layer's source from its
destination, but later transfers can overwrite source storage from earlier
layers. CUDA stream ordering is local to one device; a fast rank must not
reuse bytes while another rank is still reading them.

**Solution — per-layer NCCL all-reduce barrier**: After each layer's kernel, a lightweight `dist.all_reduce()` on a 1-element tensor provides GPU-side cross-rank synchronization with near-zero overhead:

```python
barrier_tensor = torch.zeros(1, device="cuda")
for layer in self.layers:
    layer.paras_configure_tp_mlp_fused_peer_access_kernel(...)
    transfer_attention(...)  # Same layer, before releasing its source bytes
    dist.all_reduce(barrier_tensor, group=paras_tp_group)
```

**Why this is correct**: The NCCL all-reduce is a collective that doesn't complete on any rank until all ranks participate. PyTorch synchronizes the current stream with the NCCL stream via `cudaStreamWaitEvent`, ensuring:
1. The kernel's NVLink writes complete before NCCL starts (CUDA memory model guarantees peer write visibility at kernel retirement)
2. All ranks finish the current layer before any rank starts the next
3. The next kernel launch waits for the all-reduce to complete

**Why this is fast**: All synchronization happens via GPU-side `cudaStreamWaitEvent` — no CPU-GPU round trips. Measured overhead is <0.5 ms for 48 layers.

**Why not `cuda.synchronize() + dist.barrier()`**: That approach forces two CPU-GPU round trips per layer (~100μs each), adding ~10 ms for 48 layers. The NCCL all-reduce stays entirely on the GPU.

After all layers complete, `ParaSModelMixin.paras_configure_helper()` calls `torch.cuda.synchronize()` (invoked automatically by the `@paras_func` decorator).

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

### GPT-OSS Interleaved w13

For Qwen, gate and up occupy separate contiguous halves. GPT-OSS stores
interleaved gate/up rows, so the wrapper sets `num_gates=1` and doubles the
per-peer chunk extent to `2 * I_per_tp * H`. This keeps each gate/up pair
together without changing the transport kernel. Biases and attention sinks
remain replicated outside the UMM and use per-mode views; they are not
transferred by these kernels. See [GPT-OSS support](gpt_oss_support.md).

## Historical Latency Measurements

The following tables are retained from the earlier slot-based implementation.
They compare kernel/collective behavior, not the current buffer footprint.
For present-day memory accounting, use the
[canonical reference](unified_memory_manager.md#runtime-budget-and-overhead-accounting).

### Latency Breakdown (E2E, `configure_tp`)

Measured via `torch.profiler` with `PARAS_CONFIGURE_METHOD` env var. Each method was tested with a fresh server launch, 1 EP warmup request, then `paras_configure_tp`.

| Phase | naive | overlap | peer_access |
|-------|------:|--------:|------------:|
| `gather_global_reqs` | 2.7 ms | 2.8 ms | 2.7 ms* |
| `reorchestrate_cache` | 4.9 ms | 5.2 ms | 4.8 ms |
| `gather_cache` | 7.4 ms | 7.3 ms | 7.4 ms |
| **`transfer_weights`** | **96 ms** | **83 ms** | **61 ms** |
| **`configure_tp` total** | **117 ms** | **100 ms** | **97 ms** |

*Jitter in `gather_global_reqs` is scheduling-dependent, not method-dependent.

### `transfer_weights` Decomposition

| Sub-phase | naive | overlap | peer_access |
|-----------|------:|--------:|------------:|
| EP→staging permute (48 layers) | ~15 ms | ~15 ms (pipelined) | — |
| NCCL all-to-all (48 layers) | ~55 ms | ~45 ms (pipelined) | — |
| Peer access v2 kernels (48 layers) | — | — | **9 ms** |
| NCCL all-reduce barriers (48×) | — | — | <0.5 ms |
| Attn + TP reconfiguration | ~26 ms | ~23 ms | ~22 ms |
| **Total** | **~96 ms** | **~83 ms** | **~61 ms** |

The overlap path saves ~13 ms vs naive by pipelining the permute of layer `i+1` with the NCCL all-to-all of layer `i`. However, NCCL collectives serialize on a single internal stream, limiting the overlap to permute↔NCCL only.

The peer access path eliminates both the permute and NCCL all-to-all, replacing them with direct NVLink stores that complete in 9 ms for all 48 layers.

### Theoretical NVLink Analysis

```
NVLink send per GPU per layer: 216 MB (3/4 of 288 MB sent to 3 peers)
NVLink send per GPU, 48 layers: 10.4 GB
A100 NVLink bandwidth: ~150 GB/s unidirectional (achieved)
Theoretical minimum: 10.4 GB / 150 GB/s = 69 ms

Measured v2 kernel time: ~9 ms (48 layers)
```

The v2 kernel time (9 ms) is well below the NVLink-bound theoretical minimum (69 ms). This apparent discrepancy is because the 69 ms estimate assumes serial unidirectional transfer, while in practice all 4 GPUs write simultaneously and NVLink is bidirectional. With 4 GPUs each writing 3/4 of their data, the effective aggregate bandwidth is 4×150 = 600 GB/s, giving a theoretical minimum of 10.4 GB / (600/4) = 10.4 / 150 ≈ 69 ms per GPU — but each GPU only needs to *initiate* stores for its 3/4 share, and the NVLink fabric handles the routing in parallel. The measured 9 ms kernel time reflects the GPU's ability to saturate NVLink write buffers faster than the data can physically traverse the fabric; the actual transfer may still be in flight when the kernel retires, with visibility guaranteed by the NCCL all-reduce barrier before the next layer.

## File Map

| File | Role |
|------|------|
| `paras/csrc/peer_access_transfer.cu` | CUDA kernels: `peer_access_fused_transfer_w13_v2`, `peer_access_fused_transfer_w2_v2` |
| `paras/csrc/binding.cpp` | PyTorch C++ bindings exposing kernels to Python |
| `paras/csrc/setup.py` | Standalone CUDA extension build (`pip install -e`) |
| `paras/peer_access.py` | Peer access init (IPC handles), Python kernel wrappers |
| `paras/layers/paras_moe_block.py` | Per-layer kernel launch (`paras_configure_tp_fused_peer_access_kernel`) |
| `paras/layers/paras_model.py` | Complete layer bundles with attention transfer and per-layer cross-rank fences |
| `paras/models/qwen3_moe.py` | Qwen3 ParaS model init, pre-initializes peer access |
| `paras/models/gpt_oss.py` | GPT-OSS ParaS model init, pre-initializes peer access |
| `paras/paras_memory_manager.py` | Per-mode tensor offsets from the unified plan |
| `test/srt/paras/test_paras_peer_access.py` | 4-GPU correctness + benchmark test |

## Future Work

1. **FP8 scale transfer**: The runtime rejects quantized weights. Supporting FP8 would require a weight/scale layout and transfer contract, not just accepting one-byte elements in the expert transport kernels.

2. **KV cache migration**: Peer-access kernels now exist for both EP→TP gather and TP→EP scatter. The remaining work is deeper kernel tuning and broader production coverage, not initial support.

3. **Eight-GPU tuning**: TP8 is already supported, including replicated attention KV heads. Larger groups change the peer traffic balance and warrant separate performance measurements.

4. **Kernel fusion**: Fusing w13 and w2 into a single kernel launch per layer halves launch overhead. A combined kernel was prototyped but showed marginal improvement (~0.1ms) since NVLink bandwidth dominates.

5. **Eliminate index division**: Restructuring the kernel to iterate over (chunk, position) pairs instead of flat indices would remove the per-element integer division, potentially closing the remaining 22% gap to theoretical peak.

### Kernel Optimization Opportunities

1. **Eliminate index division**: The inner loop computes `chunk_id = idx / int4_per_chunk` and `pos = idx % int4_per_chunk` per element. Restructuring to iterate over (chunk, position) pairs would remove this (~20 cycles per division × millions of iterations).

2. **Reduce per-layer barrier overhead**: Any grouping of layers must first
   prove that every grouped destination is disjoint from all unread source
   layers in the asymmetric layout. The current implementation fences every
   complete layer; increasing scratch alone does not remove that requirement.

3. **Warp specialization**: Dedicate specific warps to specific chunk sizes. w13 chunks (1.5MB) benefit from many warps; w2 rows (768B) might benefit from fewer warps with better cache locality.

4. **L2 cache prefetch**: Use `__prefetch_l2()` hints to pre-load the next chunk's source data while the current chunk's NVLink writes are in flight.

5. **Adaptive grid sizing**: Instead of fixed `num_SMs × tp_size`, dynamically compute grid size based on total data volume and per-SM NVLink bandwidth target.

## TP→EP Reverse Weight Transfer

The reverse kernels (`peer_access_fused_transfer_w13_ep`, `peer_access_fused_transfer_w2_ep`) are structural mirrors of the EP→TP v2 kernels with swapped source/destination:

| Aspect | EP→TP (v2) | TP→EP (ep) |
|--------|-----------|-----------|
| Source | Local planned EP view, strided layout | Local planned TP view, contiguous layout |
| Destination | Peer planned TP view, contiguous layout | Peer planned EP view, strided layout |
| Layer order | Forward (0→N-1) | **Reverse** (N-1→0) |

### Why Reverse Weight Transfer is Mandatory

The EP and TP interpretations overlap across layers. EP→TP overwrites EP
source bytes with TP weights, KV, and workspace contents. A pointer swap
back to EP would expose stale data. TP→EP first migrates KV, then transfers
weight bundles in reverse layer order to reconstruct experts and full
attention weights.

### Historical Performance

The reverse kernels use identical NVLink optimizations (warp-level peer assignment, int4 stores, 8-unrolling, `__ldg`, self-write bypass, uint32 fast division) and achieve comparable performance:

| Direction | Weight kernel time (48 layers, 4×A100) |
|-----------|---------------------------------------|
| EP→TP (v2) | ~78 ms |
| TP→EP (ep) | ~70 ms |
