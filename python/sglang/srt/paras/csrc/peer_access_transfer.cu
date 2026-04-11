#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

// Maximum number of GPUs in TP group (NVSwitch supports up to 8)
#define MAX_PEERS 8

// Fused strided-read + peer-write kernel for EP→TP weight transfer.
// Reads EP weights with stride (no staging buffer needed).
// Each block handles one (dst_rank r, expert e, gate k) chunk of I'H elements.
//
// EP layout: (E_local, num_gates, tp_size, I'H) in memory
// TP layout: (tp_size * E_local, num_gates, I'H) = (num_global_experts, num_gates * I'H) in memory
//
// For block (r, e, k):
//   src = ep_base + (e * num_gates * tp_size + k * tp_size + r) * I'H * elem_size
//   dst = peer_r_tp_base + (tp_rank * E_local * num_gates + e * num_gates + k) * I'H * elem_size
__global__ void peer_access_fused_transfer_kernel(
    const char* __restrict__ local_buffer,  // Local managed buffer base
    char* const* peer_buffers,              // Peer buffer bases [MAX_PEERS]
    int64_t src_ep_offset,                  // Byte offset of EP layer in local buffer
    int64_t dst_tp_offset,                  // Byte offset of TP slot in peer buffer
    int tp_rank,
    int tp_size,
    int E_local,
    int64_t I_prime_H,                      // (intermediate/tp_size) * hidden, in elements
    int num_gates,                          // 2 for w13, 1 for w2
    int elem_size                           // 2 for bf16
) {
    // Decode block index → (r, e, k)
    int block_idx = blockIdx.x;
    int r = block_idx / (E_local * num_gates);
    int rem = block_idx % (E_local * num_gates);
    int e = rem / num_gates;
    int k = rem % num_gates;

    // Source: EP[e, k, r, :] — I'H contiguous elements
    int64_t src_chunk = src_ep_offset +
        (int64_t)(e * num_gates * tp_size + k * tp_size + r) * I_prime_H * elem_size;

    // Destination: peer r's TP buffer at tp_rank's slot
    int64_t dst_chunk = dst_tp_offset +
        (int64_t)(tp_rank * E_local * num_gates + e * num_gates + k) * I_prime_H * elem_size;

    const char* src = local_buffer + src_chunk;
    char* dst = peer_buffers[r] + dst_chunk;

    int64_t n_bytes = I_prime_H * elem_size;
    int64_t n_int4 = n_bytes / 16;

    const int4* src4 = reinterpret_cast<const int4*>(src);
    int4* dst4 = reinterpret_cast<int4*>(dst);

    for (int64_t i = threadIdx.x; i < n_int4; i += blockDim.x) {
        dst4[i] = src4[i];
    }

    if (threadIdx.x == 0) {
        for (int64_t i = n_int4 * 16; i < n_bytes; i++) {
            dst[i] = src[i];
        }
    }
}

// Host-side launch for fused kernel
void launch_peer_access_fused_transfer(
    int64_t local_buffer_ptr,
    int64_t* peer_buffer_ptrs,      // device array [MAX_PEERS]
    int64_t src_ep_offset,
    int64_t dst_tp_offset,
    int tp_rank,
    int tp_size,
    int E_local,
    int64_t I_prime_H,
    int num_gates,
    int elem_size,
    cudaStream_t stream
) {
    int blocks = tp_size * E_local * num_gates;
    int threads = 256;

    peer_access_fused_transfer_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const char*>(local_buffer_ptr),
        reinterpret_cast<char* const*>(peer_buffer_ptrs),
        src_ep_offset,
        dst_tp_offset,
        tp_rank,
        tp_size,
        E_local,
        I_prime_H,
        num_gates,
        elem_size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA fused kernel error: %s\n", cudaGetErrorString(err));
    }
}

// Fused kernel for w2: EP shape (E_local, H, I_full) with TP split on last dim.
// Each block handles one (dst_rank r, expert e) pair.
// Within each row h, the I_prime elements are contiguous — exploited with int4.
// Source: row h at offset (e*H + h)*I_full_bytes + r*I_prime_bytes (contiguous I_prime bytes)
// Dest:   row h at offset ((tp_rank*E_local+e)*H + h)*I_prime_bytes (contiguous I_prime bytes)
// Both reads and writes are coalesced within each row.
__global__ void peer_access_fused_transfer_w2_kernel(
    const char* __restrict__ local_buffer,
    char* const* peer_buffers,
    int64_t src_ep_offset,
    int64_t dst_tp_offset,
    int tp_rank,
    int tp_size,
    int E_local,
    int H,
    int I_full_bytes,   // I_full * elem_size  (divisible by 16)
    int I_prime_bytes   // I_prime * elem_size  (divisible by 16, = I_full_bytes / tp_size)
) {
    int r = blockIdx.x / E_local;
    int e = blockIdx.x % E_local;

    int n_int4_src = I_full_bytes / 16;   // int4 per source row
    int n_int4_dst = I_prime_bytes / 16;  // int4 per dest row (shard size)
    int r_int4 = r * n_int4_dst;          // int4 offset of shard r in source row

    const int4* src_base = reinterpret_cast<const int4*>(local_buffer + src_ep_offset)
                           + (int64_t)e * H * n_int4_src;
    int4* dst_base = reinterpret_cast<int4*>(peer_buffers[r] + dst_tp_offset)
                     + (int64_t)(tp_rank * E_local + e) * H * n_int4_dst;

    int64_t total_int4 = (int64_t)H * n_int4_dst;

    for (int64_t idx = threadIdx.x; idx < total_int4; idx += blockDim.x) {
        int64_t h  = idx / n_int4_dst;
        int64_t ii = idx % n_int4_dst;
        dst_base[h * n_int4_dst + ii] = src_base[h * n_int4_src + r_int4 + ii];
    }
}

void launch_peer_access_fused_transfer_w2(
    int64_t local_buffer_ptr,
    int64_t* peer_buffer_ptrs,
    int64_t src_ep_offset,
    int64_t dst_tp_offset,
    int tp_rank,
    int tp_size,
    int E_local,
    int H,
    int I_full_bytes,
    int I_prime_bytes,
    cudaStream_t stream
) {
    int blocks = tp_size * E_local;
    int threads = 256;

    peer_access_fused_transfer_w2_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const char*>(local_buffer_ptr),
        reinterpret_cast<char* const*>(peer_buffer_ptrs),
        src_ep_offset,
        dst_tp_offset,
        tp_rank,
        tp_size,
        E_local,
        H,
        I_full_bytes,
        I_prime_bytes
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA fused w2 kernel error: %s\n", cudaGetErrorString(err));
    }
}

// =============================================================================
// V2 kernels: NVLink-optimized with warp-level peer assignment
// Following NVLink store guidelines:
//   - Grid: num_sms × tp_size blocks (divisible by tp_size for balanced load)
//   - Block: 256 threads = 8 warps (occupancy target: 8 blocks/SM)
//   - Peer assigned at warp level: peer = global_warp_id % tp_size
//   - 8 unrolled int4 (16B) stores per thread (#pragma unroll)
//   - Per warp: 32 lanes × 16B = 512B contiguous per store, 4KB per 8 stores
// =============================================================================

// V2 w13 kernel: warp-level peer assignment with NVLink-optimal access pattern.
// EP layout: (E_local, num_gates, tp_size, I'H)
// TP layout: (tp_size * E_local, num_gates, I'H) — contiguous per (e, k) chunk.
__global__ void peer_access_fused_transfer_w13_v2(
    const char* __restrict__ local_buffer,
    char* const* __restrict__ peer_buffers,
    int64_t src_ep_offset,
    int64_t dst_tp_offset,
    int tp_rank,
    int tp_size,
    int E_local,
    int64_t I_prime_H,
    int num_gates,
    int elem_size
) {
    const int warps_in_block = blockDim.x >> 5;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int global_warp = blockIdx.x * warps_in_block + warp_in_block;
    const int total_warps = gridDim.x * warps_in_block;

    // Warp-level peer assignment (NVLink guideline)
    const int peer = global_warp % tp_size;
    const int warp_index = global_warp / tp_size;
    const int warps_per_peer = total_warps / tp_size;

    // Self-write bypass: avoid IPC pointer for local rank (no UVA overhead)
    char* const dst_buf = (peer == tp_rank) ? const_cast<char*>(local_buffer) : peer_buffers[peer];

    // Each chunk = one (expert, gate) pair: I'H * elem_size contiguous bytes
    const int64_t chunk_bytes = I_prime_H * elem_size;
    const int chunks_per_peer = E_local * num_gates;
    const int64_t int4_per_chunk = chunk_bytes >> 4;  // / 16
    const int64_t total_int4 = (int64_t)chunks_per_peer * int4_per_chunk;

    // Destination base: tp_rank's expert slot (contiguous across all chunks)
    const int64_t dst_expert_base = dst_tp_offset +
        (int64_t)tp_rank * chunks_per_peer * chunk_bytes;

    // 8 unrolled × 32 lanes = 256 int4 per warp per iteration = 4KB
    int64_t pos = (int64_t)warp_index * 256 + lane;
    const int64_t stride = (int64_t)warps_per_peer * 256;

    // Use uint32 for decomposition math — avoids expensive int64 software division
    // total_int4 fits in uint32 (e.g. 32 chunks × 98304 = 3,145,728 < 2^32)
    const unsigned int int4_per_chunk_u = (unsigned int)int4_per_chunk;

    while (pos < total_int4) {
        #pragma unroll 8
        for (int u = 0; u < 8; u++) {
            const int64_t idx = pos + (int64_t)u * 32;
            if (idx < total_int4) {
                // Fast uint32 decomposition (hardware divider, ~20 cycles vs ~100 for int64)
                const unsigned int idx_u = (unsigned int)idx;
                const unsigned int chunk_id = idx_u / int4_per_chunk_u;
                const unsigned int in_chunk = idx_u - chunk_id * int4_per_chunk_u;

                // num_gates is typically 2 — compiler optimizes to shift+mask
                const unsigned int e = chunk_id / (unsigned int)num_gates;
                const unsigned int k = chunk_id - e * (unsigned int)num_gates;

                // Source: EP[e, k, peer, :] — strided layout
                const int64_t src_off = src_ep_offset +
                    (int64_t)(e * num_gates * tp_size + k * tp_size + peer) * chunk_bytes +
                    (int64_t)in_chunk * 16;

                // Dest: TP[tp_rank, e, k, :] — contiguous
                const int64_t dst_off = dst_expert_base +
                    (int64_t)(e * num_gates + k) * chunk_bytes +
                    (int64_t)in_chunk * 16;

                *reinterpret_cast<int4*>(dst_buf + dst_off) =
                    __ldg(reinterpret_cast<const int4*>(local_buffer + src_off));
            }
        }
        pos += stride;
    }
}

void launch_peer_access_fused_transfer_w13_v2(
    int64_t local_buffer_ptr,
    int64_t* peer_buffer_ptrs,
    int64_t src_ep_offset,
    int64_t dst_tp_offset,
    int tp_rank,
    int tp_size,
    int E_local,
    int64_t I_prime_H,
    int num_gates,
    int elem_size,
    cudaStream_t stream
) {
    int device;
    cudaGetDevice(&device);
    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);

    // Grid divisible by tp_size for balanced warp-level peer assignment
    // Env vars for tuning sweep (V2_GRID_MULT × tp_size blocks, V2_THREADS threads)
    const char* grid_env = getenv("V2_GRID_MULT");
    int grid_mult = grid_env ? atoi(grid_env) : num_sms;
    int blocks = grid_mult * tp_size;
    const char* thr_env = getenv("V2_THREADS");
    int threads = thr_env ? atoi(thr_env) : 256;

    peer_access_fused_transfer_w13_v2<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const char*>(local_buffer_ptr),
        reinterpret_cast<char* const*>(peer_buffer_ptrs),
        src_ep_offset, dst_tp_offset,
        tp_rank, tp_size, E_local, I_prime_H, num_gates, elem_size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA w13_v2 kernel error: %s\n", cudaGetErrorString(err));
    }
}

// V2 w2 kernel: warp-level peer assignment for strided row copies.
// EP layout: (E_local, H, I_full) with TP split on last dim.
// For peer r, copies columns [r*I_prime, (r+1)*I_prime) — contiguous within each row.
__global__ void peer_access_fused_transfer_w2_v2(
    const char* __restrict__ local_buffer,
    char* const* __restrict__ peer_buffers,
    int64_t src_ep_offset,
    int64_t dst_tp_offset,
    int tp_rank,
    int tp_size,
    int E_local,
    int H,
    int I_full_bytes,
    int I_prime_bytes
) {
    const int warps_in_block = blockDim.x >> 5;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int global_warp = blockIdx.x * warps_in_block + warp_in_block;
    const int total_warps = gridDim.x * warps_in_block;

    const int peer = global_warp % tp_size;
    const int warp_index = global_warp / tp_size;
    const int warps_per_peer = total_warps / tp_size;

    // Self-write bypass: avoid IPC pointer for local rank (no UVA overhead)
    char* const dst_buf = (peer == tp_rank) ? const_cast<char*>(local_buffer) : peer_buffers[peer];

    const int n_int4_per_row_dst = I_prime_bytes >> 4;  // I_prime_bytes / 16
    const int peer_shard_byte_off = peer * I_prime_bytes;
    const int64_t total_int4 = (int64_t)E_local * H * n_int4_per_row_dst;

    // 8 unrolled × 32 lanes = 256 int4 per warp per iteration = 4KB
    int64_t pos = (int64_t)warp_index * 256 + lane;
    const int64_t stride = (int64_t)warps_per_peer * 256;

    // Use uint32 for decomposition math — avoids expensive int64 software division
    // total_int4 fits in uint32 (e.g. 16*2048*48 = 1,572,864 < 2^32)
    const unsigned int n_int4_per_row_dst_u = (unsigned int)n_int4_per_row_dst;
    const unsigned int H_u = (unsigned int)H;

    while (pos < total_int4) {
        #pragma unroll 8
        for (int u = 0; u < 8; u++) {
            const int64_t idx = pos + (int64_t)u * 32;
            if (idx < total_int4) {
                // Fast uint32 decomposition (hardware divider)
                const unsigned int idx_u = (unsigned int)idx;
                const unsigned int eh_idx = idx_u / n_int4_per_row_dst_u;
                const unsigned int col = idx_u - eh_idx * n_int4_per_row_dst_u;
                const unsigned int e = eh_idx / H_u;
                const unsigned int h = eh_idx - e * H_u;

                // Source: EP[e, h, peer*I_prime + col] — strided rows
                const int64_t src_off = src_ep_offset +
                    (int64_t)e * H * I_full_bytes +
                    (int64_t)h * I_full_bytes +
                    peer_shard_byte_off + col * 16;

                // Dest: TP[(tp_rank*E_local+e), h, col]
                const int64_t dst_off = dst_tp_offset +
                    (int64_t)(tp_rank * E_local + e) * H * I_prime_bytes +
                    (int64_t)h * I_prime_bytes + col * 16;

                *reinterpret_cast<int4*>(dst_buf + dst_off) =
                    __ldg(reinterpret_cast<const int4*>(local_buffer + src_off));
            }
        }
        pos += stride;
    }
}

void launch_peer_access_fused_transfer_w2_v2(
    int64_t local_buffer_ptr,
    int64_t* peer_buffer_ptrs,
    int64_t src_ep_offset,
    int64_t dst_tp_offset,
    int tp_rank,
    int tp_size,
    int E_local,
    int H,
    int I_full_bytes,
    int I_prime_bytes,
    cudaStream_t stream
) {
    int device;
    cudaGetDevice(&device);
    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);

    const char* grid_env = getenv("V2_GRID_MULT");
    int grid_mult = grid_env ? atoi(grid_env) : num_sms;
    int blocks = grid_mult * tp_size;
    const char* thr_env = getenv("V2_THREADS");
    int threads = thr_env ? atoi(thr_env) : 256;

    peer_access_fused_transfer_w2_v2<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const char*>(local_buffer_ptr),
        reinterpret_cast<char* const*>(peer_buffer_ptrs),
        src_ep_offset, dst_tp_offset,
        tp_rank, tp_size, E_local, H, I_full_bytes, I_prime_bytes
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA w2_v2 kernel error: %s\n", cudaGetErrorString(err));
    }
}

// =============================================================================
// Combined kernel: fuses w13 + w2 into a single launch per layer set
// =============================================================================

// Combined kernel for w13+w2: processes all layers in one launch.
// Grid: num_layers * (w13_blocks_per_layer + w2_blocks_per_layer) blocks.
// First w13_blocks_per_layer blocks per layer do w13, rest do w2.
__global__ void peer_access_fused_transfer_combined_kernel(
    const char* __restrict__ local_buffer,
    char* const* peer_buffers,
    const int64_t* __restrict__ w13_ep_offsets,    // [num_layers]
    const int64_t* __restrict__ w13_tp_offsets,    // [num_layers]
    const int64_t* __restrict__ w2_ep_offsets,     // [num_layers]
    const int64_t* __restrict__ w2_tp_offsets,     // [num_layers]
    int tp_rank,
    int tp_size,
    int E_local,
    int64_t I_prime_H,
    int num_gates,          // = 2
    int elem_size,          // = 2 for bf16
    int H,
    int I_full_bytes,
    int I_prime_bytes,
    int w13_blocks_per_layer,  // = tp_size * E_local * num_gates
    int w2_blocks_per_layer,   // = tp_size * E_local
    int num_layers
) {
    int blocks_per_layer = w13_blocks_per_layer + w2_blocks_per_layer;
    int layer = blockIdx.x / blocks_per_layer;
    int block_in_layer = blockIdx.x % blocks_per_layer;

    if (layer >= num_layers) return;

    if (block_in_layer < w13_blocks_per_layer) {
        // w13 path
        int r = block_in_layer / (E_local * num_gates);
        int rem = block_in_layer % (E_local * num_gates);
        int e = rem / num_gates;
        int k = rem % num_gates;

        int64_t src_ep_offset = w13_ep_offsets[layer];
        int64_t dst_tp_offset = w13_tp_offsets[layer];

        int64_t src_chunk = src_ep_offset +
            (int64_t)(e * num_gates * tp_size + k * tp_size + r) * I_prime_H * elem_size;
        int64_t dst_chunk = dst_tp_offset +
            (int64_t)(tp_rank * E_local * num_gates + e * num_gates + k) * I_prime_H * elem_size;

        const char* src = local_buffer + src_chunk;
        char* dst = peer_buffers[r] + dst_chunk;

        int64_t n_bytes = I_prime_H * elem_size;
        int64_t n_int4 = n_bytes / 16;

        const int4* src4 = reinterpret_cast<const int4*>(src);
        int4* dst4 = reinterpret_cast<int4*>(dst);

        for (int64_t i = threadIdx.x; i < n_int4; i += blockDim.x) {
            dst4[i] = src4[i];
        }

        if (threadIdx.x == 0) {
            for (int64_t i = n_int4 * 16; i < n_bytes; i++) {
                dst[i] = src[i];
            }
        }
    } else {
        // w2 path
        int blk = block_in_layer - w13_blocks_per_layer;
        int r = blk / E_local;
        int e = blk % E_local;

        int64_t src_ep_offset = w2_ep_offsets[layer];
        int64_t dst_tp_offset = w2_tp_offsets[layer];

        int n_int4_src = I_full_bytes / 16;
        int n_int4_dst = I_prime_bytes / 16;
        int r_int4 = r * n_int4_dst;

        const int4* src_base = reinterpret_cast<const int4*>(local_buffer + src_ep_offset)
                               + (int64_t)e * H * n_int4_src;
        int4* dst_base = reinterpret_cast<int4*>(peer_buffers[r] + dst_tp_offset)
                         + (int64_t)(tp_rank * E_local + e) * H * n_int4_dst;

        int64_t total_int4 = (int64_t)H * n_int4_dst;

        for (int64_t idx = threadIdx.x; idx < total_int4; idx += blockDim.x) {
            int64_t h  = idx / n_int4_dst;
            int64_t ii = idx % n_int4_dst;
            dst_base[h * n_int4_dst + ii] = src_base[h * n_int4_src + r_int4 + ii];
        }
    }
}

// Host-side launch for combined w13+w2 kernel
void launch_peer_access_fused_transfer_combined(
    int64_t local_buffer_ptr,
    int64_t* peer_buffer_ptrs,
    const int64_t* w13_ep_offsets,
    const int64_t* w13_tp_offsets,
    const int64_t* w2_ep_offsets,
    const int64_t* w2_tp_offsets,
    int tp_rank,
    int tp_size,
    int E_local,
    int64_t I_prime_H,
    int num_gates,
    int elem_size,
    int H,
    int I_full_bytes,
    int I_prime_bytes,
    int num_layers,
    int num_threads,
    cudaStream_t stream
) {
    int w13_blocks_per_layer = tp_size * E_local * num_gates;
    int w2_blocks_per_layer = tp_size * E_local;
    int blocks_per_layer = w13_blocks_per_layer + w2_blocks_per_layer;
    int total_blocks = num_layers * blocks_per_layer;
    int threads = num_threads > 0 ? num_threads : 256;

    peer_access_fused_transfer_combined_kernel<<<total_blocks, threads, 0, stream>>>(
        reinterpret_cast<const char*>(local_buffer_ptr),
        reinterpret_cast<char* const*>(peer_buffer_ptrs),
        w13_ep_offsets,
        w13_tp_offsets,
        w2_ep_offsets,
        w2_tp_offsets,
        tp_rank,
        tp_size,
        E_local,
        I_prime_H,
        num_gates,
        elem_size,
        H,
        I_full_bytes,
        I_prime_bytes,
        w13_blocks_per_layer,
        w2_blocks_per_layer,
        num_layers
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA combined w13+w2 kernel error: %s\n", cudaGetErrorString(err));
    }
}


