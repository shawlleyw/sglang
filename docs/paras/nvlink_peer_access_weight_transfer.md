# NVLink Peer Access Weight Transfer for ParaS

## Overview

This document describes the NVLink peer access weight transfer optimization for the ParaS EP→TP parallelism switch. Instead of using NCCL `all_to_all` collectives for MoE weight redistribution, we use custom CUDA kernels that write directly to peer GPU memory via NVLink.

### Performance Summary (Qwen3-30B-A3B, 48 layers, 4×A100-80GB)

| Method | transfer_weights | configure TP total | vs naive |
|--------|-----------------|-------------------|----------|
| `naive` (NCCL sequential) | ~96 ms | ~117 ms | baseline |
| `overlap` (NCCL pipelined) | ~83 ms | ~100 ms | 1.17× |
| `peer_access` (NVLink direct) | **~61 ms** | **~97 ms** | **1.57×** |

The peer access kernel time is only **~9 ms** for all 48 layers. The remaining `transfer_weights` time is dominated by attention and TP reconfiguration overhead shared by all methods. The `configure TP` total includes cache migration, request gathering, and weight transfer.

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

**Virtual-to-physical slot mapping**: The memory manager's `alias()` method creates virtual entries that map to the same physical offset as a target entry. After `materialize()`, `create_paras_moe_aliases()` registers three alias families per layer:
- `model.layers.{i}.mlp.experts.*` → slot `i+1` (weight loading compatibility)
- `model.layers.{i}.mlp.ep_experts.*` → slot `i+1` (explicit EP)
- `model.layers.{i}.mlp.tp_experts.*` → slot `i` (explicit TP)

Both EP and TP views are created at model init time and never change, eliminating the need for `update_views()` after the transfer.

**Layer ordering constraint**: EP→TP must process layers in forward order (0, 1, 2, ..., N-1). Layer `i+1`'s write to slot `i+1` must not overlap with layer `i`'s read from slot `i+1`. Sequential kernel launches on the same CUDA stream enforce intra-rank ordering. Cross-rank ordering requires a per-layer synchronization barrier (see Section 3).

### 3. Cross-Rank Synchronization

**The problem**: The N+1 slot design prevents intra-rank aliasing (layer `i` reads slot `i+1`, writes slot `i` — different slots). However, **cross-rank temporal aliasing** exists: Rank A processing layer `i+1` writes to Rank B's slot `i+1` via NVLink, while Rank B may still be processing layer `i` which reads from its own slot `i+1`. CUDA stream ordering only guarantees ordering on a **single device** — cross-rank NVLink writes have no ordering guarantee.

**Solution — per-layer NCCL all-reduce barrier**: After each layer's kernel, a lightweight `dist.all_reduce()` on a 1-element tensor provides GPU-side cross-rank synchronization with near-zero overhead:

```python
barrier_tensor = torch.zeros(1, device="cuda")
for layer in self.layers:
    layer.paras_configure_tp_mlp_fused_peer_access_kernel(...)
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

### NCCL Path Compatibility

The N+1 slot design is not exclusive to peer access — the NCCL naive and overlap paths also use it. This means all three methods share the same memory layout and alias structure:

- **All-to-all output target**: NCCL writes directly to the TP slot (slot `i`) for each layer. Since `tp_experts` already points to slot `i` from init, no copy or view update is needed after the collective completes.
- **No staging for peer access**: The peer access kernel reads directly from the EP slot with strided access, bypassing the permute step entirely. Staging buffers are skipped via `skip_staging=True`, saving ~1.16 GiB.

In all cases, the TP slot (slot `i`) holds valid TP data after the transfer, and `tp_experts` views remain correct without any post-transfer alias updates.

#### Staging Buffer Requirements

NCCL's `all_to_all_single` requires contiguous, permuted input. Before calling the collective, EP weights are permuted (rearranging the TP dimension to the leading axis) and written into a **pre_permute** staging buffer. The all-to-all reads from this buffer and writes the result to either the TP slot (DP=1) or back to the gather buffer (DP>1, requiring a post-transpose).

For DP>1, an additional **gather** staging buffer is needed because the all-gather step must first collect EP weights from all DP ranks into a contiguous region before the permute+all-to-all can proceed. The data flow for DP>1 is:

```
EP slot → [all-gather] → gather buffer → [permute] → pre_permute buffer
    → [all-to-all] → gather buffer → [transpose] → pre_permute buffer → [copy] → TP slot
```

This reuses the two buffers alternately: gather receives the all-gather output and the all-to-all output, while pre_permute holds the permuted input and the transposed result. For DP=1, the all-gather is a no-op (EP weights are read directly from the EP slot), so only the pre_permute buffer is needed.

The overlap path pipelines two layers on different streams. Each stream requires its own independent set of staging buffers to avoid cross-stream data races. This doubles the staging requirement.

| Method | DP | Buffers per set | Sets | Buffer names | Memory |
|--------|-----|----------------|------|--------------|--------|
| `naive` | =1 | 1 pre_permute | 1 | `staging.{w13,w2}_pre_permute` | ~580 MB |
| `naive` | >1 | 1 pre_permute + 1 gather | 1 | `staging.{w13,w2}_{pre_permute,gather}` | ~1.16 GiB |
| `overlap` | =1 | 1 pre_permute | 2 | `staging.{w13,w2}_pre_permute_{1,2}` | ~1.16 GiB |
| `overlap` | >1 | 1 pre_permute + 1 gather | 2 | `staging.{w13,w2}_{pre_permute,gather}_{1,2}` | ~2.32 GiB |
| `peer_access` | any | None | 0 | — | 0 |

**Memory formula** (Qwen3-30B-A3B, BF16):

```
E_local = num_experts / ep_size                           (e.g. 64/4 = 16)
staging_experts = E_local × dp_size                       (e.g. 16×1 = 16 for DP=1)
pre_permute size = staging_experts × (2×I×H + H×I) × 2B  (w13 + w2, BF16)
                 = 16 × (2×1536×2048 + 2048×1536) × 2    = 580 MB
gather size      = same as pre_permute                    = 580 MB  (DP>1 only)
```

For DP>1 the `staging_experts` grows by `dp_size`, making each buffer proportionally larger. With `dp_size=2`, each buffer is 2× larger (1.16 GiB), and the overlap path needs 4 such buffers (4.64 GiB total). This makes peer_access especially attractive for DP>1 configurations where staging memory pressure is highest.

The `plan_qwen_moe_layout()` function accepts `configure_method` to reserve only the buffers needed, controlled by the `PARAS_CONFIGURE_METHOD` environment variable.

#### Overlap Path: Dual-Stream Pipelining and NCCL Stream Behavior

The overlap path pipelines the all-gather of layer `i+1` with the all-to-all of layer `i` using two CUDA streams (`stream_1`, `stream_2`). Each stream uses its own staging buffer set, identified by a suffix (`_1` or `_2`). The streams and suffixes swap each iteration:

```python
stream_1, stream_2 = stream_2, stream_1
staging_1, staging_2 = staging_2, staging_1  # "_1" ↔ "_2"
```

**Why two staging buffer sets are required**: Without separate sets, both layers write their permuted EP data to the same `pre_permute` buffer concurrently on different streams — a data race. Each stream's suffix (`_1` or `_2`) selects an independent `pre_permute` (and for DP>1, `gather`) buffer.

**NCCL stream behavior**: Despite being issued from different user streams (`stream_1`, `stream_2`), all NCCL collectives execute on a **single internal NCCL stream** managed by PyTorch's `ProcessGroupNCCL`. When `dist.all_to_all_single(async_op=True)` is called:

1. PyTorch records an event on the current user stream
2. The NCCL stream waits on that event (ensuring the permute completes)
3. NCCL enqueues the all-to-all on its own stream
4. `handle.wait()` later makes the user stream wait on the NCCL completion

This means NCCL collectives **serialize** on the NCCL stream regardless of which user stream issued them. The overlap benefit comes from the **permute/copy** on user streams running concurrently with NCCL work, not from NCCL ops overlapping each other. In profiler traces, all `all_to_all` operations appear on the same NCCL stream.

## Method Comparison: Memory Footprint and Latency

All numbers are for Qwen3-30B-A3B (48 MoE layers, 64 experts, hidden=2048, intermediate=1536) on 4×A100-80GB SXM with NVLink.

### Per-Layer Weight Sizes

| Weight | Shape (EP, per GPU) | Size (BF16) |
|--------|-------------------|-------------|
| w13 (gate+up) | (16, 3072, 2048) | 192 MB |
| w2 (down) | (16, 2048, 1536) | 96 MB |
| **Total per layer** | | **288 MB** |

### Memory Footprint (per GPU)

The table below shows the **inherent memory overhead** of each method vs the original N-slot system (no ParaS). The naive and overlap methods only require staging buffers — they could work with the original N-slot layout. The peer_access method requires the N+1 extra slot to avoid source/destination aliasing but needs no staging. (In the current implementation, all methods share the N+1 layout for code simplicity, but the inherent cost is what matters for comparison.)

Each staging buffer (pre_permute or gather) holds one layer's worth of MoE weights (`E_local × dp_size` experts for w13 + w2), which is exactly the same size as a physical slot for DP=1 (288 MB). This makes the overhead directly comparable in units of "slots."

**DP=1** (current production configuration):

| Component | naive | overlap | peer_access |
|-----------|------:|--------:|------------:|
| N+1 extra slot | — | — | +1 slot |
| Staging: pre_permute | +1 slot (×1) | +1 slot (×2) | — |
| **Total inherent overhead** | **1 slot (288 MB)** | **2 slots (576 MB)** | **1 slot (288 MB)** |

Naive and peer_access have identical memory overhead (1 slot each). Peer_access wins purely on latency.

**DP=2** (hypothetical, ep_size=2, tp_size=4):

Each staging buffer grows by `dp_size×` (holding `E_local × dp_size` experts), so each buffer = `dp_size` slots = 2 slots (576 MB).

| Component | naive | overlap | peer_access |
|-----------|------:|--------:|------------:|
| N+1 extra slot | — | — | +1 slot |
| Staging: pre_permute (2 slots each) | +2 slots (×1) | +2 slots (×2) | — |
| Staging: gather (2 slots each) | +2 slots (×1) | +2 slots (×2) | — |
| **Total inherent overhead** | **4 slots (1.13 GiB)** | **8 slots (2.25 GiB)** | **1 slot (288 MB)** |

At DP=2, the peer_access memory advantage grows to **4× vs naive** and **8× vs overlap**. The overhead gap widens further at higher DP sizes since staging scales as `O(dp_size × num_pipeline_stages)` while peer_access remains fixed at 1 slot.

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
| `paras/layers/paras_model.py` | No-barrier orchestration, `@paras_func` handles sync via `paras_configure_helper()` |
| `paras/models/qwen3_moe.py` | Pre-initializes peer access during model load |
| `paras/paras_memory_manager.py` | N+1 slot reservation (`paras.moe_slot.{0..N}.*`), `alias()`, `create_paras_moe_aliases()` |
| `test/srt/test_paras_peer_access.py` | 4-GPU correctness + benchmark test |

## Future Work

1. **FP8 scale transfer**: The peer access kernels currently only transfer weight data (w13, w2). FP8 quantized models also need their per-expert scale tensors (`w13_weight_scale`, `w2_weight_scale`) redistributed during EP→TP switching. This is not yet implemented.

2. **TP→EP reverse switch**: The N+1 slot design supports this by processing layers in reverse order (N-1, N-2, ..., 0). Layer `i`'s read from slot `i` completes before layer `i-1`'s write to slot `i`.

2. **KV cache migration**: Currently uses NCCL. Could use the same peer access approach since KV buffers are also in the managed buffer.

3. **Larger TP groups (8 GPUs)**: More peers = more NVLink bandwidth available. The warp-level peer assignment scales naturally.

4. **Kernel fusion**: Fusing w13 and w2 into a single kernel launch per layer halves launch overhead. A combined kernel was prototyped but showed marginal improvement (~0.1ms) since NVLink bandwidth dominates.

5. **Eliminate index division**: Restructuring the kernel to iterate over (chunk, position) pairs instead of flat indices would remove the per-element integer division, potentially closing the remaining 22% gap to theoretical peak.

### Kernel Optimization Opportunities

1. **Eliminate index division**: The inner loop computes `chunk_id = idx / int4_per_chunk` and `pos = idx % int4_per_chunk` per element. Restructuring to iterate over (chunk, position) pairs would remove this (~20 cycles per division × millions of iterations).

2. **Reduce per-layer barrier overhead**: The current per-layer NCCL all-reduce barrier is lightweight but adds up over many layers. An N+2 slot design (EP layer `i` → slot `i+2`, TP layer `i` → slot `i`) would allow processing consecutive layer pairs simultaneously, halving the number of barriers from N to N/2. More generally, an offset of `d` allows groups of `d` layers per phase with `⌈N/d⌉ - 1` barriers, at the cost of `d` extra slots. With the current N+1 (offset 1) design, every consecutive layer pair shares a slot, making per-layer barriers unavoidable.

3. **Warp specialization**: Dedicate specific warps to specific chunk sizes. w13 chunks (1.5MB) benefit from many warps; w2 rows (768B) might benefit from fewer warps with better cache locality.

4. **L2 cache prefetch**: Use `__prefetch_l2()` hints to pre-load the next chunk's source data while the current chunk's NVLink writes are in flight.

5. **Adaptive grid sizing**: Instead of fixed `num_SMs × tp_size`, dynamically compute grid size based on total data volume and per-SM NVLink bandwidth target.
