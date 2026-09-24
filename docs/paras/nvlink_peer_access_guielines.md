# NVLink Store-Based P2P All-to-All Kernel Guideline

## Target: A100 GPUs, NVSwitch, Custom CUDA Kernels

---

## Kernel Configuration

- **Block size**: 256 threads (8 warps per block)
- **Grid size**: `108 × (num_gpus - 1)` blocks (one block per SM per peer)
- **Occupancy target**: 8 blocks per SM (64 warps/SM, full occupancy)

## Memory Access Pattern

- **Vector type**: `float4` or `uint4` (16 bytes per thread per store)
- **Store width per warp**: 512 bytes per store instruction (32 threads × 16B)
- **Stores per thread**: 8 iterations (unrolled)
- **Contiguous segment per warp**: 4 KB (8 × 512B)
- **Contiguous segment per block**: 32 KB (8 warps × 4 KB)
- **All destination pointers must be 128-byte aligned.** Misaligned stores split into smaller NVLink transactions and degrade bandwidth.

## Peer Assignment Strategy

Assign peers at the warp level within each block:

- `peer = global_warp_id % num_peers`
- `warp_index = global_warp_id / num_peers`

This distributes NVLink traffic evenly across all peers from every SM.

## Minimum Data Thresholds

| Scope              | Minimum             | Recommended              |
|--------------------|---------------------|--------------------------|
| Per warp           | 512 B               | 2–4 KB                   |
| Per peer           | 2 MB                | 4+ MB                    |
| Per GPU (total)    | `2 MB × num_peers`  | `4+ MB × num_peers`      |

Below per-warp minimum, NVLink latency dominates and bandwidth is wasted.

## NVLink Transaction Rules

- NVLink minimum transaction: 32 bytes
- Optimal transaction: 128 bytes (one cache line)
- SM outstanding remote store slots: ~8–16 per SM
- NVLink store round-trip latency: ~1–2 µs
- Per-SM achievable bandwidth: ~1.4 GB/s → 108 SMs ≈ ~150 GB/s unidirectional

## Critical Constraints

1. Always use 128-byte aligned base pointers for remote destinations.
2. Always use widest vector type (`float4`/`uint4`) for stores.
3. Unroll the inner store loop (`#pragma unroll`).
4. Each warp must write at least 4 contiguous 128B cache lines (512B absolute minimum) to amortize NVLink latency.
5. Avoid partial-warp stores — all 32 lanes should participate.
6. Do not use `cudaMemcpyPeerAsync` or DMA engines; all transfers are direct kernel stores to remote-mapped pointers obtained via `cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle` or unified memory.