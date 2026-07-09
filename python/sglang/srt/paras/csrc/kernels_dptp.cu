// DPTP forward (EP → DP×TP) peer-access weight transfer kernels.
// Generalization of kernels_v3.cu's EP→TP forward kernels to a target grid
// world W = G*T (G = dp_size = replication factor, T = tp_size = shard count).
//
// Key insight (replicated scatter): for fixed shard t, all G replicas receive
// the SAME source bytes, written to G different GPUs. So READ ONCE per warp
// per int4 position, then BROADCAST to G peer destinations via a per-thread
// register unroll. The destination offset is replica-invariant (canonical
// slot e_global = R*E_local + e), so only the base pointer differs across d.
// The canonical reorder is therefore FREE (no post-permute).
//
// Layout (source rank R, tp_rank = R % T, dp_rank = R / T):
//   EP src w13: (E_local, num_gates, T, I'*H)  with I' = I/T
//   EP src w2:  (E_local, H, I_full = T*I')
// Destination rank (d, t) = d*T + t for ALL d in [0,G), t in [0,T):
//   TP dst w13: (E, num_gates, I'*H)  -- canonical (e_global, k) slots
//   TP dst w2:  (E, H, I')             -- canonical (e_global, h) slots
//
// Template parameters:
//   T              shard count (tp_size), in {2, 4, 8}
//   G              replication factor (dp_size), in {1, 2, 4}
//   NUM_GATES      w13 only; 1 or 2
//   ELEM_SIZE      bytes per element (2 for bf16)
//   BLOCKS_PER_CHUNK  8, 16, 32, 64
//   H              hidden_size (model config)
//   I              moe_intermediate_size (model config)
//
// Constraints:
//   WARPS_PER_BLOCK % T == 0 must hold (WARPS_PER_BLOCK = 8 → T ∈ {1,2,4,8} ✓).
//   G * T <= MAX_PEERS (8 here).
//
// Knobs (build-time -D):
//   V3_UNROLL              int4 stores per warp per outer iter (default 32).
//   V3_MIN_BLOCKS_PER_SM   __launch_bounds__ occupancy hint (default 4).

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#define MAX_PEERS 8

#ifndef WARPS_PER_BLOCK
#define WARPS_PER_BLOCK 16
#endif

#ifndef V3_UNROLL
#define V3_UNROLL 32
#endif

#ifndef V3_MIN_BLOCKS_PER_SM
#define V3_MIN_BLOCKS_PER_SM 2
#endif

#ifndef DPTP_PREFETCH
#define DPTP_PREFETCH 8
#endif

#define DPTP_THREADS_PER_BLOCK (WARPS_PER_BLOCK * 32)

// Match v3's pick_blocks_per_chunk semantics (translation unit local).
static inline int dptp_pick_blocks_per_chunk(int chunks_per_block, int sms_per_gpu) {
    const int target_blocks = sms_per_gpu * V3_MIN_BLOCKS_PER_SM * 2;
    int bpc = (target_blocks + chunks_per_block - 1) / chunks_per_block;
    if (bpc < 1) bpc = 1;
    int p = 1;
    while (p < bpc) p <<= 1;
    if (p > 64) p = 64;
    if (p < 8) p = 8;
    return p;
}

// =============================================================================
// w13 dptp forward kernel (EP → DP×TP)
// =============================================================================

template <int T, int G, int NUM_GATES, int ELEM_SIZE, int BLOCKS_PER_CHUNK, int H, int I>
__global__ __launch_bounds__(DPTP_THREADS_PER_BLOCK, V3_MIN_BLOCKS_PER_SM)
void peer_access_fused_transfer_w13_dptp_tpl(
    const char* __restrict__ local_buffer,
    char* const* __restrict__ peer_buffers,
    int64_t src_ep_offset,
    int64_t dst_tp_offset,
    int R,
    int E_local
) {
    constexpr int WARPS_PER_SHARD_IN_BLOCK = WARPS_PER_BLOCK / T;
    constexpr int WARPS_PER_CHUNK_SHARD = BLOCKS_PER_CHUNK * WARPS_PER_SHARD_IN_BLOCK;
    constexpr unsigned int STRIDE_U = (unsigned int)WARPS_PER_CHUNK_SHARD * DPTP_PREFETCH * 32u;
    constexpr int I_PRIME = I / T;
    constexpr int64_t CHUNK_BYTES = (int64_t)I_PRIME * H * ELEM_SIZE;
    constexpr unsigned int INT4_PER_CHUNK_U = (unsigned int)(CHUNK_BYTES >> 4);

    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int t = warp_in_block / WARPS_PER_SHARD_IN_BLOCK;

    const int chunk_id = blockIdx.x / BLOCKS_PER_CHUNK;
    const int chunks_per_block_dim = E_local * NUM_GATES;
    if (chunk_id >= chunks_per_block_dim) return;

    int e, k;
    if constexpr (NUM_GATES == 2) { e = chunk_id >> 1; k = chunk_id & 1; }
    else                          { e = chunk_id;      k = 0;            }

    const char* const src_chunk_ptr = local_buffer + src_ep_offset +
        (int64_t)(e * NUM_GATES * T + k * T + t) * CHUNK_BYTES;

    const int64_t dst_off = dst_tp_offset +
        (int64_t)((R * E_local + e) * NUM_GATES + k) * CHUNK_BYTES;

    char* dbase[G];
    #pragma unroll
    for (int d = 0; d < G; d++) {
        const int dr = d * T + t;
        dbase[d] = ((dr == R) ? const_cast<char*>(local_buffer) : peer_buffers[dr]) + dst_off;
    }

    const int sub_block_in_chunk = blockIdx.x - chunk_id * BLOCKS_PER_CHUNK;
    const int sub_warp_in_shard = warp_in_block - t * WARPS_PER_SHARD_IN_BLOCK;
    const int warp_id_in_chunk_shard = sub_block_in_chunk * WARPS_PER_SHARD_IN_BLOCK + sub_warp_in_shard;

    unsigned int pos = (unsigned int)warp_id_in_chunk_shard * DPTP_PREFETCH * 32u + (unsigned int)lane;

    while (pos < INT4_PER_CHUNK_U) {
        int4 vbuf[DPTP_PREFETCH];
        #pragma unroll
        for (int u = 0; u < DPTP_PREFETCH; u++) {
            const unsigned int idx = pos + (unsigned int)(u * 32);
            if (idx < INT4_PER_CHUNK_U) {
                vbuf[u] = *reinterpret_cast<const int4*>(src_chunk_ptr + (size_t)idx * 16);
            }
        }
        #pragma unroll
        for (int u = 0; u < DPTP_PREFETCH; u++) {
            const unsigned int idx = pos + (unsigned int)(u * 32);
            if (idx < INT4_PER_CHUNK_U) {
                #pragma unroll
                for (int d = 0; d < G; d++) {
                    *reinterpret_cast<int4*>(dbase[d] + (size_t)idx * 16) = vbuf[u];
                }
            }
        }
        pos += STRIDE_U;
    }
}

// =============================================================================
// w2 dptp forward kernel (EP → DP×TP); row-aligned tiling, zero divmod in inner loop.
// =============================================================================

template <int T, int G, int ELEM_SIZE, int BLOCKS_PER_CHUNK, int H, int I>
__global__ __launch_bounds__(DPTP_THREADS_PER_BLOCK, V3_MIN_BLOCKS_PER_SM)
void peer_access_fused_transfer_w2_dptp_tpl(
    const char* __restrict__ local_buffer,
    char* const* __restrict__ peer_buffers,
    int64_t src_ep_offset,
    int64_t dst_tp_offset,
    int R,
    int E_local
) {
    constexpr int WARPS_PER_SHARD_IN_BLOCK = WARPS_PER_BLOCK / T;
    constexpr int WARPS_PER_CHUNK_SHARD = BLOCKS_PER_CHUNK * WARPS_PER_SHARD_IN_BLOCK;
    constexpr int I_PRIME = I / T;
    constexpr int I_FULL_BYTES = I * ELEM_SIZE;
    constexpr int I_PRIME_BYTES = I_PRIME * ELEM_SIZE;
    constexpr unsigned int N_INT4_PER_ROW = (unsigned int)(I_PRIME_BYTES >> 4);
    constexpr unsigned int ROWS_PER_WARP = ((unsigned int)H + WARPS_PER_CHUNK_SHARD - 1u) / (unsigned int)WARPS_PER_CHUNK_SHARD;
    constexpr unsigned int COL_ITERS = (N_INT4_PER_ROW + 31u) >> 5;

    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int t = warp_in_block / WARPS_PER_SHARD_IN_BLOCK;

    const int chunk_id = blockIdx.x / BLOCKS_PER_CHUNK;
    if (chunk_id >= E_local) return;
    const int e = chunk_id;

    // Source: expert e, all rows, columns [t*I', (t+1)*I') from local EP region.
    const char* const src_base = local_buffer + src_ep_offset +
        (int64_t)e * (int64_t)H * I_FULL_BYTES +
        (int64_t)t * I_PRIME_BYTES;

    // Replica-invariant destination offset (canonical slot e_global = R*E_local + e).
    const int64_t dst_off = dst_tp_offset +
        (int64_t)(R * E_local + e) * (int64_t)H * I_PRIME_BYTES;

    // Hoist G destination base pointers (uniform per warp).
    char* dbase[G];
    #pragma unroll
    for (int d = 0; d < G; d++) {
        const int dr = d * T + t;
        dbase[d] = ((dr == R) ? const_cast<char*>(local_buffer) : peer_buffers[dr]) + dst_off;
    }

    const int sub_block_in_chunk = blockIdx.x - chunk_id * BLOCKS_PER_CHUNK;
    const int sub_warp_in_shard = warp_in_block - t * WARPS_PER_SHARD_IN_BLOCK;
    const int warp_id_in_chunk_shard = sub_block_in_chunk * WARPS_PER_SHARD_IN_BLOCK + sub_warp_in_shard;

    const unsigned int row_start = (unsigned int)warp_id_in_chunk_shard * ROWS_PER_WARP;
    const unsigned int row_end = (row_start + ROWS_PER_WARP <= (unsigned int)H) ? (row_start + ROWS_PER_WARP) : (unsigned int)H;
    const unsigned int lane_u = (unsigned int)lane;

    for (unsigned int h_base = row_start; h_base < row_end; h_base += DPTP_PREFETCH) {
        int4 vbuf[DPTP_PREFETCH][COL_ITERS];
        #pragma unroll
        for (int u = 0; u < DPTP_PREFETCH; u++) {
            const unsigned int h = h_base + (unsigned int)u;
            if (h < row_end) {
                const size_t row_src_off = (size_t)h * I_FULL_BYTES;
                #pragma unroll
                for (unsigned int c = 0; c < COL_ITERS; c++) {
                    const unsigned int col = (c << 5) + lane_u;
                    if (col < N_INT4_PER_ROW) {
                        const size_t off = (size_t)col * 16;
                        vbuf[u][c] = *reinterpret_cast<const int4*>(src_base + row_src_off + off);
                    }
                }
            }
        }
        #pragma unroll
        for (int u = 0; u < DPTP_PREFETCH; u++) {
            const unsigned int h = h_base + (unsigned int)u;
            if (h < row_end) {
                const size_t row_dst_off = (size_t)h * I_PRIME_BYTES;
                #pragma unroll
                for (unsigned int c = 0; c < COL_ITERS; c++) {
                    const unsigned int col = (c << 5) + lane_u;
                    if (col < N_INT4_PER_ROW) {
                        const size_t off = (size_t)col * 16;
                        #pragma unroll
                        for (int d = 0; d < G; d++) {
                            *reinterpret_cast<int4*>(dbase[d] + row_dst_off + off) = vbuf[u][c];
                        }
                    }
                }
            }
        }
    }
}

// =============================================================================
// Dispatchers. Supported model presets (H, I):
//   Qwen3-235B: (4096, 1536)
//   Qwen3-30B:  (2048, 768)
// (T, G) in {(8, 1), (4, 2), (2, 4), (2, 2)}, NUM_GATES in {1, 2}, ELEM_SIZE = 2,
// BPC in {8, 16, 32, 64}.
// =============================================================================

#define W13_DPTP_BPC_CASE(KERNEL, TPS, GPS, NG, ES, BPC, HH, II, ...) \
    case BPC: KERNEL<TPS, GPS, NG, ES, BPC, HH, II><<<blocks, threads, 0, stream>>>(__VA_ARGS__); return

#define DISPATCH_W13_DPTP_BPC_ALL(KERNEL, TPS, GPS, NG, ES, HH, II, ...)             \
    switch (bpc) {                                                                    \
        W13_DPTP_BPC_CASE(KERNEL, TPS, GPS, NG, ES, 8,  HH, II, __VA_ARGS__);         \
        W13_DPTP_BPC_CASE(KERNEL, TPS, GPS, NG, ES, 16, HH, II, __VA_ARGS__);         \
        W13_DPTP_BPC_CASE(KERNEL, TPS, GPS, NG, ES, 32, HH, II, __VA_ARGS__);         \
        W13_DPTP_BPC_CASE(KERNEL, TPS, GPS, NG, ES, 64, HH, II, __VA_ARGS__);         \
        default: break;                                                                \
    }

#define W2_DPTP_BPC_CASE(KERNEL, TPS, GPS, ES, BPC, HH, II, ...) \
    case BPC: KERNEL<TPS, GPS, ES, BPC, HH, II><<<blocks, threads, 0, stream>>>(__VA_ARGS__); return

#define DISPATCH_W2_DPTP_BPC_ALL(KERNEL, TPS, GPS, ES, HH, II, ...)             \
    switch (bpc) {                                                                \
        W2_DPTP_BPC_CASE(KERNEL, TPS, GPS, ES, 8,  HH, II, __VA_ARGS__);         \
        W2_DPTP_BPC_CASE(KERNEL, TPS, GPS, ES, 16, HH, II, __VA_ARGS__);         \
        W2_DPTP_BPC_CASE(KERNEL, TPS, GPS, ES, 32, HH, II, __VA_ARGS__);         \
        W2_DPTP_BPC_CASE(KERNEL, TPS, GPS, ES, 64, HH, II, __VA_ARGS__);         \
        default: break;                                                           \
    }

#define W13_DPTP_DISPATCH_PRESET(KERNEL, TPS, GPS, NG, ES, ...) \
    do {                                                                                              \
        if      (H == 4096 && I == 1536) { DISPATCH_W13_DPTP_BPC_ALL(KERNEL, TPS, GPS, NG, ES, 4096, 1536, __VA_ARGS__); } \
        else if (H == 2048 && I == 768)  { DISPATCH_W13_DPTP_BPC_ALL(KERNEL, TPS, GPS, NG, ES, 2048, 768,  __VA_ARGS__); } \
    } while (0)

#define W2_DPTP_DISPATCH_PRESET(KERNEL, TPS, GPS, ES, ...) \
    do {                                                                                            \
        if      (H == 4096 && I == 1536) { DISPATCH_W2_DPTP_BPC_ALL(KERNEL, TPS, GPS, ES, 4096, 1536, __VA_ARGS__); } \
        else if (H == 2048 && I == 768)  { DISPATCH_W2_DPTP_BPC_ALL(KERNEL, TPS, GPS, ES, 2048, 768,  __VA_ARGS__); } \
    } while (0)

void launch_peer_access_fused_transfer_w13_dptp(
    int64_t local_buffer_ptr,
    int64_t* peer_buffer_ptrs,
    int64_t src_ep_offset,
    int64_t dst_tp_offset,
    int rank,
    int T,
    int G,
    int E_local,
    int H,
    int I,
    int num_gates,
    int elem_size,
    cudaStream_t stream
) {
    int device, sms;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    const int chunks_per_block_dim = E_local * num_gates;
    const int bpc = dptp_pick_blocks_per_chunk(chunks_per_block_dim, sms);
    const int blocks = chunks_per_block_dim * bpc;
    const int threads = WARPS_PER_BLOCK * 32;

    const auto local_buf = reinterpret_cast<const char*>(local_buffer_ptr);
    const auto peer_bufs = reinterpret_cast<char* const*>(peer_buffer_ptrs);

#define ARGS local_buf, peer_bufs, src_ep_offset, dst_tp_offset, rank, E_local
    if (elem_size == 2) {
        if (num_gates == 2) {
            if      (T == 8 && G == 1) { W13_DPTP_DISPATCH_PRESET(peer_access_fused_transfer_w13_dptp_tpl, 8, 1, 2, 2, ARGS); }
            else if (T == 4 && G == 2) { W13_DPTP_DISPATCH_PRESET(peer_access_fused_transfer_w13_dptp_tpl, 4, 2, 2, 2, ARGS); }
            else if (T == 2 && G == 4) { W13_DPTP_DISPATCH_PRESET(peer_access_fused_transfer_w13_dptp_tpl, 2, 4, 2, 2, ARGS); }
            else if (T == 2 && G == 2) { W13_DPTP_DISPATCH_PRESET(peer_access_fused_transfer_w13_dptp_tpl, 2, 2, 2, 2, ARGS); }
        } else if (num_gates == 1) {
            if      (T == 8 && G == 1) { W13_DPTP_DISPATCH_PRESET(peer_access_fused_transfer_w13_dptp_tpl, 8, 1, 1, 2, ARGS); }
            else if (T == 4 && G == 2) { W13_DPTP_DISPATCH_PRESET(peer_access_fused_transfer_w13_dptp_tpl, 4, 2, 1, 2, ARGS); }
            else if (T == 2 && G == 4) { W13_DPTP_DISPATCH_PRESET(peer_access_fused_transfer_w13_dptp_tpl, 2, 4, 1, 2, ARGS); }
            else if (T == 2 && G == 2) { W13_DPTP_DISPATCH_PRESET(peer_access_fused_transfer_w13_dptp_tpl, 2, 2, 1, 2, ARGS); }
        }
    }
#undef ARGS
    printf("CUDA w13_dptp unsupported combo: T=%d G=%d gates=%d elem=%d H=%d I=%d bpc=%d\n",
           T, G, num_gates, elem_size, H, I, bpc);
}

void launch_peer_access_fused_transfer_w2_dptp(
    int64_t local_buffer_ptr,
    int64_t* peer_buffer_ptrs,
    int64_t src_ep_offset,
    int64_t dst_tp_offset,
    int rank,
    int T,
    int G,
    int E_local,
    int H,
    int I,
    int elem_size,
    cudaStream_t stream
) {
    int device, sms;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    const int bpc = dptp_pick_blocks_per_chunk(E_local, sms);
    const int blocks = E_local * bpc;
    const int threads = WARPS_PER_BLOCK * 32;

    const auto local_buf = reinterpret_cast<const char*>(local_buffer_ptr);
    const auto peer_bufs = reinterpret_cast<char* const*>(peer_buffer_ptrs);

#define ARGS local_buf, peer_bufs, src_ep_offset, dst_tp_offset, rank, E_local
    if (elem_size == 2) {
        if      (T == 8 && G == 1) { W2_DPTP_DISPATCH_PRESET(peer_access_fused_transfer_w2_dptp_tpl, 8, 1, 2, ARGS); }
        else if (T == 4 && G == 2) { W2_DPTP_DISPATCH_PRESET(peer_access_fused_transfer_w2_dptp_tpl, 4, 2, 2, ARGS); }
        else if (T == 2 && G == 4) { W2_DPTP_DISPATCH_PRESET(peer_access_fused_transfer_w2_dptp_tpl, 2, 4, 2, ARGS); }
        else if (T == 2 && G == 2) { W2_DPTP_DISPATCH_PRESET(peer_access_fused_transfer_w2_dptp_tpl, 2, 2, 2, ARGS); }
    }
#undef ARGS
    printf("CUDA w2_dptp unsupported combo: T=%d G=%d H=%d I=%d elem=%d bpc=%d\n",
           T, G, H, I, elem_size, bpc);
}

// =============================================================================
// w13 dptp REVERSE kernel (DP x TP -> EP); replica-local within T-group, no broadcast.
// =============================================================================

template <int T, int G, int NUM_GATES, int ELEM_SIZE, int BLOCKS_PER_CHUNK, int H, int I>
__global__ __launch_bounds__(DPTP_THREADS_PER_BLOCK, V3_MIN_BLOCKS_PER_SM)
void peer_access_fused_transfer_w13_ep_dptp_tpl(
    const char* __restrict__ local_buffer,
    char* const* __restrict__ peer_buffers,
    int64_t src_tp_offset,
    int64_t dst_ep_offset,
    int R,
    int E_local
) {
    constexpr int WARPS_PER_PEER_IN_BLOCK = WARPS_PER_BLOCK / T;
    constexpr int WARPS_PER_CHUNK_PEER = BLOCKS_PER_CHUNK * WARPS_PER_PEER_IN_BLOCK;
    constexpr unsigned int STRIDE_U = (unsigned int)WARPS_PER_CHUNK_PEER * DPTP_PREFETCH * 32u;
    constexpr int I_PRIME = I / T;
    constexpr int64_t CHUNK_BYTES = (int64_t)I_PRIME * H * ELEM_SIZE;
    constexpr unsigned int INT4_PER_CHUNK_U = (unsigned int)(CHUNK_BYTES >> 4);
    (void)G;

    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int peer = warp_in_block / WARPS_PER_PEER_IN_BLOCK;

    const int tp_rank = R % T;
    const int dp_rank = R / T;
    const int dest_rank = dp_rank * T + peer;

    const int chunk_id = blockIdx.x / BLOCKS_PER_CHUNK;
    const int chunks_per_block_dim = E_local * NUM_GATES;
    if (chunk_id >= chunks_per_block_dim) return;

    int e, k;
    if constexpr (NUM_GATES == 2) { e = chunk_id >> 1; k = chunk_id & 1; }
    else                          { e = chunk_id;      k = 0;            }

    const int e_global = dest_rank * E_local + e;

    char* const dst_buf = (dest_rank == R) ? const_cast<char*>(local_buffer) : peer_buffers[dest_rank];

    const char* const src_chunk_ptr = local_buffer + src_tp_offset +
        (int64_t)(e_global * NUM_GATES + k) * CHUNK_BYTES;
    char* const dst_chunk_ptr = dst_buf + dst_ep_offset +
        (int64_t)(e * NUM_GATES * T + k * T + tp_rank) * CHUNK_BYTES;

    const int sub_block_in_chunk = blockIdx.x - chunk_id * BLOCKS_PER_CHUNK;
    const int sub_warp_in_peer = warp_in_block - peer * WARPS_PER_PEER_IN_BLOCK;
    const int warp_id_in_chunk_peer = sub_block_in_chunk * WARPS_PER_PEER_IN_BLOCK + sub_warp_in_peer;

    unsigned int pos = (unsigned int)warp_id_in_chunk_peer * DPTP_PREFETCH * 32u + (unsigned int)lane;

    while (pos < INT4_PER_CHUNK_U) {
        int4 vbuf[DPTP_PREFETCH];
        #pragma unroll
        for (int u = 0; u < DPTP_PREFETCH; u++) {
            const unsigned int idx = pos + (unsigned int)(u * 32);
            if (idx < INT4_PER_CHUNK_U) {
                vbuf[u] = *reinterpret_cast<const int4*>(src_chunk_ptr + (size_t)idx * 16);
            }
        }
        #pragma unroll
        for (int u = 0; u < DPTP_PREFETCH; u++) {
            const unsigned int idx = pos + (unsigned int)(u * 32);
            if (idx < INT4_PER_CHUNK_U) {
                *reinterpret_cast<int4*>(dst_chunk_ptr + (size_t)idx * 16) = vbuf[u];
            }
        }
        pos += STRIDE_U;
    }
}

// =============================================================================
// w2 dptp REVERSE kernel (DP x TP -> EP); row-aligned, no broadcast.
// =============================================================================

template <int T, int G, int ELEM_SIZE, int BLOCKS_PER_CHUNK, int H, int I>
__global__ __launch_bounds__(DPTP_THREADS_PER_BLOCK, V3_MIN_BLOCKS_PER_SM)
void peer_access_fused_transfer_w2_ep_dptp_tpl(
    const char* __restrict__ local_buffer,
    char* const* __restrict__ peer_buffers,
    int64_t src_tp_offset,
    int64_t dst_ep_offset,
    int R,
    int E_local
) {
    constexpr int WARPS_PER_PEER_IN_BLOCK = WARPS_PER_BLOCK / T;
    constexpr int WARPS_PER_CHUNK_PEER = BLOCKS_PER_CHUNK * WARPS_PER_PEER_IN_BLOCK;
    constexpr int I_PRIME = I / T;
    constexpr int I_FULL_BYTES = I * ELEM_SIZE;
    constexpr int I_PRIME_BYTES = I_PRIME * ELEM_SIZE;
    constexpr unsigned int N_INT4_PER_ROW = (unsigned int)(I_PRIME_BYTES >> 4);
    constexpr unsigned int ROWS_PER_WARP = ((unsigned int)H + WARPS_PER_CHUNK_PEER - 1u) / (unsigned int)WARPS_PER_CHUNK_PEER;
    constexpr unsigned int COL_ITERS = (N_INT4_PER_ROW + 31u) >> 5;
    (void)G;

    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int peer = warp_in_block / WARPS_PER_PEER_IN_BLOCK;

    const int tp_rank = R % T;
    const int dp_rank = R / T;
    const int dest_rank = dp_rank * T + peer;

    const int chunk_id = blockIdx.x / BLOCKS_PER_CHUNK;
    if (chunk_id >= E_local) return;
    const int e = chunk_id;
    const int e_global = dest_rank * E_local + e;

    char* const dst_buf = (dest_rank == R) ? const_cast<char*>(local_buffer) : peer_buffers[dest_rank];

    const char* const src_chunk_ptr = local_buffer + src_tp_offset +
        (int64_t)e_global * (int64_t)H * I_PRIME_BYTES;
    char* const dst_chunk_ptr = dst_buf + dst_ep_offset +
        (int64_t)e * (int64_t)H * I_FULL_BYTES +
        (int64_t)tp_rank * I_PRIME_BYTES;

    const int sub_block_in_chunk = blockIdx.x - chunk_id * BLOCKS_PER_CHUNK;
    const int sub_warp_in_peer = warp_in_block - peer * WARPS_PER_PEER_IN_BLOCK;
    const int warp_id_in_chunk_peer = sub_block_in_chunk * WARPS_PER_PEER_IN_BLOCK + sub_warp_in_peer;

    const unsigned int row_start = (unsigned int)warp_id_in_chunk_peer * ROWS_PER_WARP;
    const unsigned int row_end = (row_start + ROWS_PER_WARP <= (unsigned int)H) ? (row_start + ROWS_PER_WARP) : (unsigned int)H;
    const unsigned int lane_u = (unsigned int)lane;

    for (unsigned int h_base = row_start; h_base < row_end; h_base += DPTP_PREFETCH) {
        int4 vbuf[DPTP_PREFETCH][COL_ITERS];
        #pragma unroll
        for (int u = 0; u < DPTP_PREFETCH; u++) {
            const unsigned int h = h_base + (unsigned int)u;
            if (h < row_end) {
                const size_t row_src_off = (size_t)h * I_PRIME_BYTES;
                #pragma unroll
                for (unsigned int c = 0; c < COL_ITERS; c++) {
                    const unsigned int col = (c << 5) + lane_u;
                    if (col < N_INT4_PER_ROW) {
                        const size_t off = (size_t)col * 16;
                        vbuf[u][c] = *reinterpret_cast<const int4*>(src_chunk_ptr + row_src_off + off);
                    }
                }
            }
        }
        #pragma unroll
        for (int u = 0; u < DPTP_PREFETCH; u++) {
            const unsigned int h = h_base + (unsigned int)u;
            if (h < row_end) {
                const size_t row_dst_off = (size_t)h * I_FULL_BYTES;
                #pragma unroll
                for (unsigned int c = 0; c < COL_ITERS; c++) {
                    const unsigned int col = (c << 5) + lane_u;
                    if (col < N_INT4_PER_ROW) {
                        const size_t off = (size_t)col * 16;
                        *reinterpret_cast<int4*>(dst_chunk_ptr + row_dst_off + off) = vbuf[u][c];
                    }
                }
            }
        }
    }
}

void launch_peer_access_fused_transfer_w13_ep_dptp(
    int64_t local_buffer_ptr,
    int64_t* peer_buffer_ptrs,
    int64_t src_tp_offset,
    int64_t dst_ep_offset,
    int rank,
    int T,
    int G,
    int E_local,
    int H,
    int I,
    int num_gates,
    int elem_size,
    cudaStream_t stream
) {
    int device, sms;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    const int chunks_per_block_dim = E_local * num_gates;
    const int bpc = dptp_pick_blocks_per_chunk(chunks_per_block_dim, sms);
    const int blocks = chunks_per_block_dim * bpc;
    const int threads = WARPS_PER_BLOCK * 32;

    const auto local_buf = reinterpret_cast<const char*>(local_buffer_ptr);
    const auto peer_bufs = reinterpret_cast<char* const*>(peer_buffer_ptrs);

#define ARGS local_buf, peer_bufs, src_tp_offset, dst_ep_offset, rank, E_local
    if (elem_size == 2) {
        if (num_gates == 2) {
            if      (T == 8 && G == 1) { W13_DPTP_DISPATCH_PRESET(peer_access_fused_transfer_w13_ep_dptp_tpl, 8, 1, 2, 2, ARGS); }
            else if (T == 4 && G == 2) { W13_DPTP_DISPATCH_PRESET(peer_access_fused_transfer_w13_ep_dptp_tpl, 4, 2, 2, 2, ARGS); }
            else if (T == 2 && G == 4) { W13_DPTP_DISPATCH_PRESET(peer_access_fused_transfer_w13_ep_dptp_tpl, 2, 4, 2, 2, ARGS); }
            else if (T == 2 && G == 2) { W13_DPTP_DISPATCH_PRESET(peer_access_fused_transfer_w13_ep_dptp_tpl, 2, 2, 2, 2, ARGS); }
        } else if (num_gates == 1) {
            if      (T == 8 && G == 1) { W13_DPTP_DISPATCH_PRESET(peer_access_fused_transfer_w13_ep_dptp_tpl, 8, 1, 1, 2, ARGS); }
            else if (T == 4 && G == 2) { W13_DPTP_DISPATCH_PRESET(peer_access_fused_transfer_w13_ep_dptp_tpl, 4, 2, 1, 2, ARGS); }
            else if (T == 2 && G == 4) { W13_DPTP_DISPATCH_PRESET(peer_access_fused_transfer_w13_ep_dptp_tpl, 2, 4, 1, 2, ARGS); }
            else if (T == 2 && G == 2) { W13_DPTP_DISPATCH_PRESET(peer_access_fused_transfer_w13_ep_dptp_tpl, 2, 2, 1, 2, ARGS); }
        }
    }
#undef ARGS
    printf("CUDA w13_ep_dptp unsupported combo: T=%d G=%d gates=%d elem=%d H=%d I=%d bpc=%d\n",
           T, G, num_gates, elem_size, H, I, bpc);
}

void launch_peer_access_fused_transfer_w2_ep_dptp(
    int64_t local_buffer_ptr,
    int64_t* peer_buffer_ptrs,
    int64_t src_tp_offset,
    int64_t dst_ep_offset,
    int rank,
    int T,
    int G,
    int E_local,
    int H,
    int I,
    int elem_size,
    cudaStream_t stream
) {
    int device, sms;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    const int bpc = dptp_pick_blocks_per_chunk(E_local, sms);
    const int blocks = E_local * bpc;
    const int threads = WARPS_PER_BLOCK * 32;

    const auto local_buf = reinterpret_cast<const char*>(local_buffer_ptr);
    const auto peer_bufs = reinterpret_cast<char* const*>(peer_buffer_ptrs);

#define ARGS local_buf, peer_bufs, src_tp_offset, dst_ep_offset, rank, E_local
    if (elem_size == 2) {
        if      (T == 8 && G == 1) { W2_DPTP_DISPATCH_PRESET(peer_access_fused_transfer_w2_ep_dptp_tpl, 8, 1, 2, ARGS); }
        else if (T == 4 && G == 2) { W2_DPTP_DISPATCH_PRESET(peer_access_fused_transfer_w2_ep_dptp_tpl, 4, 2, 2, ARGS); }
        else if (T == 2 && G == 4) { W2_DPTP_DISPATCH_PRESET(peer_access_fused_transfer_w2_ep_dptp_tpl, 2, 4, 2, ARGS); }
        else if (T == 2 && G == 2) { W2_DPTP_DISPATCH_PRESET(peer_access_fused_transfer_w2_ep_dptp_tpl, 2, 2, 2, ARGS); }
    }
#undef ARGS
    printf("CUDA w2_ep_dptp unsupported combo: T=%d G=%d H=%d I=%d elem=%d bpc=%d\n",
           T, G, H, I, elem_size, bpc);
}
