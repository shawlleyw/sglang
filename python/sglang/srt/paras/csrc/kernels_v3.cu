// V3 peer-access weight transfer kernels: contiguous-tile tiling, templated on
// raw model config (H, I) plus (TP_SIZE, NUM_GATES, ELEM_SIZE, BLOCKS_PER_CHUNK).
// All derived sizes (I_PRIME, CHUNK_BYTES, INT4_PER_CHUNK, ROWS_PER_WARP,
// COL_ITERS, etc.) are constexpr inside each kernel from those template args.
//
// Template parameters:
//   TP_SIZE           e.g. 4, 8
//   NUM_GATES         w13 only; 1 or 2
//   ELEM_SIZE         bytes per element (2 for bf16)
//   BLOCKS_PER_CHUNK  8, 16, 32, 64
//   H                 hidden_size (model config), e.g. 2048, 4096
//   I                 moe_intermediate_size (model config), e.g. 768, 1536
//
// Knobs (build-time -D):
//   V3_UNROLL              int4 stores per warp per outer iter (default 32).
//   V3_MIN_BLOCKS_PER_SM   __launch_bounds__ occupancy hint (default 4).

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#define MAX_PEERS 8
#define WARPS_PER_BLOCK 8

#ifndef V3_UNROLL
#define V3_UNROLL 32
#endif

#ifndef V3_MIN_BLOCKS_PER_SM
#define V3_MIN_BLOCKS_PER_SM 4
#endif

static inline int pick_blocks_per_chunk(int chunks_per_peer, int sms_per_gpu, int tp_size) {
    (void)tp_size;
    // Target ~2 waves at __launch_bounds__-implied occupancy.
    const int target_blocks = sms_per_gpu * V3_MIN_BLOCKS_PER_SM * 2;
    int bpc = (target_blocks + chunks_per_peer - 1) / chunks_per_peer;
    if (bpc < 1) bpc = 1;
    int p = 1;
    while (p < bpc) p <<= 1;
    if (p > 64) p = 64;
    if (p < 8) p = 8;
    return p;
}

// =============================================================================
// w13 kernels (templated)
// =============================================================================

template <int TP_SIZE, int NUM_GATES, int ELEM_SIZE, int BLOCKS_PER_CHUNK, int H, int I>
__global__ __launch_bounds__(256, V3_MIN_BLOCKS_PER_SM)
void peer_access_fused_transfer_w13_v3_tpl(
    const char* __restrict__ local_buffer,
    char* const* __restrict__ peer_buffers,
    int64_t src_ep_offset,
    int64_t dst_tp_offset,
    int tp_rank,
    int E_local
) {
    constexpr int WARPS_PER_PEER_IN_BLOCK = WARPS_PER_BLOCK / TP_SIZE;
    constexpr int WARPS_PER_CHUNK_PEER = BLOCKS_PER_CHUNK * WARPS_PER_PEER_IN_BLOCK;
    constexpr unsigned int STRIDE_U = (unsigned int)WARPS_PER_CHUNK_PEER * V3_UNROLL * 32u;
    constexpr int I_PRIME = I / TP_SIZE;
    constexpr int64_t CHUNK_BYTES = (int64_t)I_PRIME * H * ELEM_SIZE;
    constexpr unsigned int INT4_PER_CHUNK_U = (unsigned int)(CHUNK_BYTES >> 4);

    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int peer = warp_in_block / WARPS_PER_PEER_IN_BLOCK;

    const int chunk_id = blockIdx.x / BLOCKS_PER_CHUNK;
    const int chunks_per_peer = E_local * NUM_GATES;
    if (chunk_id >= chunks_per_peer) return;

    int e, k;
    if constexpr (NUM_GATES == 2) { e = chunk_id >> 1; k = chunk_id & 1; }
    else                          { e = chunk_id;      k = 0;            }

    char* const dst_buf = (peer == tp_rank) ? const_cast<char*>(local_buffer) : peer_buffers[peer];

    const char* const src_chunk_ptr = local_buffer + src_ep_offset +
        (int64_t)(e * NUM_GATES * TP_SIZE + k * TP_SIZE + peer) * CHUNK_BYTES;
    char* const dst_chunk_ptr = dst_buf + dst_tp_offset +
        (int64_t)tp_rank * (int64_t)chunks_per_peer * CHUNK_BYTES +
        (int64_t)(e * NUM_GATES + k) * CHUNK_BYTES;

    const int sub_block_in_chunk = blockIdx.x - chunk_id * BLOCKS_PER_CHUNK;
    const int sub_warp_in_peer = warp_in_block - peer * WARPS_PER_PEER_IN_BLOCK;
    const int warp_id_in_chunk_peer = sub_block_in_chunk * WARPS_PER_PEER_IN_BLOCK + sub_warp_in_peer;

    unsigned int pos = (unsigned int)warp_id_in_chunk_peer * V3_UNROLL * 32u + (unsigned int)lane;

    while (pos < INT4_PER_CHUNK_U) {
        #pragma unroll
        for (int u = 0; u < V3_UNROLL; u++) {
            const unsigned int idx = pos + (unsigned int)(u * 32);
            if (idx < INT4_PER_CHUNK_U) {
                *reinterpret_cast<int4*>(dst_chunk_ptr + (size_t)idx * 16) =
                    __ldg(reinterpret_cast<const int4*>(src_chunk_ptr + (size_t)idx * 16));
            }
        }
        pos += STRIDE_U;
    }
}

template <int TP_SIZE, int NUM_GATES, int ELEM_SIZE, int BLOCKS_PER_CHUNK, int H, int I>
__global__ __launch_bounds__(256, V3_MIN_BLOCKS_PER_SM)
void peer_access_fused_transfer_w13_v3_ep_tpl(
    const char* __restrict__ local_buffer,
    char* const* __restrict__ peer_buffers,
    int64_t src_tp_offset,
    int64_t dst_ep_offset,
    int tp_rank,
    int E_local
) {
    constexpr int WARPS_PER_PEER_IN_BLOCK = WARPS_PER_BLOCK / TP_SIZE;
    constexpr int WARPS_PER_CHUNK_PEER = BLOCKS_PER_CHUNK * WARPS_PER_PEER_IN_BLOCK;
    constexpr unsigned int STRIDE_U = (unsigned int)WARPS_PER_CHUNK_PEER * V3_UNROLL * 32u;
    constexpr int I_PRIME = I / TP_SIZE;
    constexpr int64_t CHUNK_BYTES = (int64_t)I_PRIME * H * ELEM_SIZE;
    constexpr unsigned int INT4_PER_CHUNK_U = (unsigned int)(CHUNK_BYTES >> 4);

    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int peer = warp_in_block / WARPS_PER_PEER_IN_BLOCK;

    const int chunk_id = blockIdx.x / BLOCKS_PER_CHUNK;
    const int chunks_per_peer = E_local * NUM_GATES;
    if (chunk_id >= chunks_per_peer) return;

    int e, k;
    if constexpr (NUM_GATES == 2) { e = chunk_id >> 1; k = chunk_id & 1; }
    else                          { e = chunk_id;      k = 0;            }

    char* const dst_buf = (peer == tp_rank) ? const_cast<char*>(local_buffer) : peer_buffers[peer];

    const char* const src_chunk_ptr = local_buffer + src_tp_offset +
        (int64_t)peer * (int64_t)chunks_per_peer * CHUNK_BYTES +
        (int64_t)(e * NUM_GATES + k) * CHUNK_BYTES;
    char* const dst_chunk_ptr = dst_buf + dst_ep_offset +
        (int64_t)(e * NUM_GATES * TP_SIZE + k * TP_SIZE + tp_rank) * CHUNK_BYTES;

    const int sub_block_in_chunk = blockIdx.x - chunk_id * BLOCKS_PER_CHUNK;
    const int sub_warp_in_peer = warp_in_block - peer * WARPS_PER_PEER_IN_BLOCK;
    const int warp_id_in_chunk_peer = sub_block_in_chunk * WARPS_PER_PEER_IN_BLOCK + sub_warp_in_peer;

    unsigned int pos = (unsigned int)warp_id_in_chunk_peer * V3_UNROLL * 32u + (unsigned int)lane;

    while (pos < INT4_PER_CHUNK_U) {
        #pragma unroll
        for (int u = 0; u < V3_UNROLL; u++) {
            const unsigned int idx = pos + (unsigned int)(u * 32);
            if (idx < INT4_PER_CHUNK_U) {
                *reinterpret_cast<int4*>(dst_chunk_ptr + (size_t)idx * 16) =
                    __ldg(reinterpret_cast<const int4*>(src_chunk_ptr + (size_t)idx * 16));
            }
        }
        pos += STRIDE_U;
    }
}

// =============================================================================
// w2 kernels (templated; row-aligned tiling, zero divmod in inner loop)
// =============================================================================

template <int TP_SIZE, int ELEM_SIZE, int BLOCKS_PER_CHUNK, int H, int I>
__global__ __launch_bounds__(256, V3_MIN_BLOCKS_PER_SM)
void peer_access_fused_transfer_w2_v3_tpl(
    const char* __restrict__ local_buffer,
    char* const* __restrict__ peer_buffers,
    int64_t src_ep_offset,
    int64_t dst_tp_offset,
    int tp_rank,
    int E_local
) {
    constexpr int WARPS_PER_PEER_IN_BLOCK = WARPS_PER_BLOCK / TP_SIZE;
    constexpr int WARPS_PER_CHUNK_PEER = BLOCKS_PER_CHUNK * WARPS_PER_PEER_IN_BLOCK;
    constexpr int I_PRIME = I / TP_SIZE;
    constexpr int I_FULL_BYTES = I * ELEM_SIZE;
    constexpr int I_PRIME_BYTES = I_PRIME * ELEM_SIZE;
    constexpr unsigned int N_INT4_PER_ROW = (unsigned int)(I_PRIME_BYTES >> 4);
    constexpr unsigned int ROWS_PER_WARP = ((unsigned int)H + WARPS_PER_CHUNK_PEER - 1u) / (unsigned int)WARPS_PER_CHUNK_PEER;
    constexpr unsigned int COL_ITERS = (N_INT4_PER_ROW + 31u) >> 5;

    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int peer = warp_in_block / WARPS_PER_PEER_IN_BLOCK;

    const int chunk_id = blockIdx.x / BLOCKS_PER_CHUNK;
    if (chunk_id >= E_local) return;
    const int e = chunk_id;

    char* const dst_buf = (peer == tp_rank) ? const_cast<char*>(local_buffer) : peer_buffers[peer];

    const char* const src_chunk_ptr = local_buffer + src_ep_offset +
        (int64_t)e * (int64_t)H * I_FULL_BYTES +
        (int64_t)peer * I_PRIME_BYTES;
    char* const dst_chunk_ptr = dst_buf + dst_tp_offset +
        (int64_t)(tp_rank * E_local + e) * (int64_t)H * I_PRIME_BYTES;

    const int sub_block_in_chunk = blockIdx.x - chunk_id * BLOCKS_PER_CHUNK;
    const int sub_warp_in_peer = warp_in_block - peer * WARPS_PER_PEER_IN_BLOCK;
    const int warp_id_in_chunk_peer = sub_block_in_chunk * WARPS_PER_PEER_IN_BLOCK + sub_warp_in_peer;

    const unsigned int row_start = (unsigned int)warp_id_in_chunk_peer * ROWS_PER_WARP;
    const unsigned int row_end = (row_start + ROWS_PER_WARP <= (unsigned int)H) ? (row_start + ROWS_PER_WARP) : (unsigned int)H;
    const unsigned int lane_u = (unsigned int)lane;

    for (unsigned int h_base = row_start; h_base < row_end; h_base += V3_UNROLL) {
        #pragma unroll
        for (int u = 0; u < V3_UNROLL; u++) {
            const unsigned int h = h_base + (unsigned int)u;
            if (h < row_end) {
                const size_t row_src_off = (size_t)h * I_FULL_BYTES;
                const size_t row_dst_off = (size_t)h * I_PRIME_BYTES;
                #pragma unroll
                for (unsigned int c = 0; c < COL_ITERS; c++) {
                    const unsigned int col = (c << 5) + lane_u;
                    if (col < N_INT4_PER_ROW) {
                        const size_t off = (size_t)col * 16;
                        *reinterpret_cast<int4*>(dst_chunk_ptr + row_dst_off + off) =
                            __ldg(reinterpret_cast<const int4*>(src_chunk_ptr + row_src_off + off));
                    }
                }
            }
        }
    }
}

template <int TP_SIZE, int ELEM_SIZE, int BLOCKS_PER_CHUNK, int H, int I>
__global__ __launch_bounds__(256, V3_MIN_BLOCKS_PER_SM)
void peer_access_fused_transfer_w2_v3_ep_tpl(
    const char* __restrict__ local_buffer,
    char* const* __restrict__ peer_buffers,
    int64_t src_tp_offset,
    int64_t dst_ep_offset,
    int tp_rank,
    int E_local
) {
    constexpr int WARPS_PER_PEER_IN_BLOCK = WARPS_PER_BLOCK / TP_SIZE;
    constexpr int WARPS_PER_CHUNK_PEER = BLOCKS_PER_CHUNK * WARPS_PER_PEER_IN_BLOCK;
    constexpr int I_PRIME = I / TP_SIZE;
    constexpr int I_FULL_BYTES = I * ELEM_SIZE;
    constexpr int I_PRIME_BYTES = I_PRIME * ELEM_SIZE;
    constexpr unsigned int N_INT4_PER_ROW = (unsigned int)(I_PRIME_BYTES >> 4);
    constexpr unsigned int ROWS_PER_WARP = ((unsigned int)H + WARPS_PER_CHUNK_PEER - 1u) / (unsigned int)WARPS_PER_CHUNK_PEER;
    constexpr unsigned int COL_ITERS = (N_INT4_PER_ROW + 31u) >> 5;

    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int peer = warp_in_block / WARPS_PER_PEER_IN_BLOCK;

    const int chunk_id = blockIdx.x / BLOCKS_PER_CHUNK;
    if (chunk_id >= E_local) return;
    const int e = chunk_id;

    char* const dst_buf = (peer == tp_rank) ? const_cast<char*>(local_buffer) : peer_buffers[peer];

    const char* const src_chunk_ptr = local_buffer + src_tp_offset +
        (int64_t)(peer * E_local + e) * (int64_t)H * I_PRIME_BYTES;
    char* const dst_chunk_ptr = dst_buf + dst_ep_offset +
        (int64_t)e * (int64_t)H * I_FULL_BYTES +
        (int64_t)tp_rank * I_PRIME_BYTES;

    const int sub_block_in_chunk = blockIdx.x - chunk_id * BLOCKS_PER_CHUNK;
    const int sub_warp_in_peer = warp_in_block - peer * WARPS_PER_PEER_IN_BLOCK;
    const int warp_id_in_chunk_peer = sub_block_in_chunk * WARPS_PER_PEER_IN_BLOCK + sub_warp_in_peer;

    const unsigned int row_start = (unsigned int)warp_id_in_chunk_peer * ROWS_PER_WARP;
    const unsigned int row_end = (row_start + ROWS_PER_WARP <= (unsigned int)H) ? (row_start + ROWS_PER_WARP) : (unsigned int)H;
    const unsigned int lane_u = (unsigned int)lane;

    for (unsigned int h_base = row_start; h_base < row_end; h_base += V3_UNROLL) {
        #pragma unroll
        for (int u = 0; u < V3_UNROLL; u++) {
            const unsigned int h = h_base + (unsigned int)u;
            if (h < row_end) {
                const size_t row_src_off = (size_t)h * I_PRIME_BYTES;
                const size_t row_dst_off = (size_t)h * I_FULL_BYTES;
                #pragma unroll
                for (unsigned int c = 0; c < COL_ITERS; c++) {
                    const unsigned int col = (c << 5) + lane_u;
                    if (col < N_INT4_PER_ROW) {
                        const size_t off = (size_t)col * 16;
                        *reinterpret_cast<int4*>(dst_chunk_ptr + row_dst_off + off) =
                            __ldg(reinterpret_cast<const int4*>(src_chunk_ptr + row_src_off + off));
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
// TP_SIZE in {4, 8}, NUM_GATES in {1, 2}, ELEM_SIZE = 2, BPC in {8, 16, 32, 64}.
// =============================================================================

#define W13_BPC_CASE(KERNEL, TPS, NG, ES, BPC, HH, II, ...) \
    case BPC: KERNEL<TPS, NG, ES, BPC, HH, II><<<blocks, threads, 0, stream>>>(__VA_ARGS__); return

#define DISPATCH_W13_BPC_ALL(KERNEL, TPS, NG, ES, HH, II, ...)             \
    switch (bpc) {                                                          \
        W13_BPC_CASE(KERNEL, TPS, NG, ES, 8,  HH, II, __VA_ARGS__);         \
        W13_BPC_CASE(KERNEL, TPS, NG, ES, 16, HH, II, __VA_ARGS__);         \
        W13_BPC_CASE(KERNEL, TPS, NG, ES, 32, HH, II, __VA_ARGS__);         \
        W13_BPC_CASE(KERNEL, TPS, NG, ES, 64, HH, II, __VA_ARGS__);         \
        default: break;                                                     \
    }

#define W2_BPC_CASE(KERNEL, TPS, ES, BPC, HH, II, ...) \
    case BPC: KERNEL<TPS, ES, BPC, HH, II><<<blocks, threads, 0, stream>>>(__VA_ARGS__); return

#define DISPATCH_W2_BPC_ALL(KERNEL, TPS, ES, HH, II, ...)             \
    switch (bpc) {                                                     \
        W2_BPC_CASE(KERNEL, TPS, ES, 8,  HH, II, __VA_ARGS__);         \
        W2_BPC_CASE(KERNEL, TPS, ES, 16, HH, II, __VA_ARGS__);         \
        W2_BPC_CASE(KERNEL, TPS, ES, 32, HH, II, __VA_ARGS__);         \
        W2_BPC_CASE(KERNEL, TPS, ES, 64, HH, II, __VA_ARGS__);         \
        default: break;                                                \
    }

#define W13_DISPATCH_PRESET(KERNEL, TPS, NG, ES, ...) \
    do {                                                                                \
        if      (H == 4096 && I == 1536) { DISPATCH_W13_BPC_ALL(KERNEL, TPS, NG, ES, 4096, 1536, __VA_ARGS__); } \
        else if (H == 2048 && I == 768)  { DISPATCH_W13_BPC_ALL(KERNEL, TPS, NG, ES, 2048, 768,  __VA_ARGS__); } \
    } while (0)

#define W2_DISPATCH_PRESET(KERNEL, TPS, ES, ...) \
    do {                                                                              \
        if      (H == 4096 && I == 1536) { DISPATCH_W2_BPC_ALL(KERNEL, TPS, ES, 4096, 1536, __VA_ARGS__); } \
        else if (H == 2048 && I == 768)  { DISPATCH_W2_BPC_ALL(KERNEL, TPS, ES, 2048, 768,  __VA_ARGS__); } \
    } while (0)

void launch_peer_access_fused_transfer_w13_v3(
    int64_t local_buffer_ptr,
    int64_t* peer_buffer_ptrs,
    int64_t src_ep_offset,
    int64_t dst_tp_offset,
    int tp_rank,
    int tp_size,
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
    const int chunks_per_peer = E_local * num_gates;
    const int bpc = pick_blocks_per_chunk(chunks_per_peer, sms, tp_size);
    const int blocks = chunks_per_peer * bpc;
    const int threads = WARPS_PER_BLOCK * 32;

    const auto local_buf = reinterpret_cast<const char*>(local_buffer_ptr);
    const auto peer_bufs = reinterpret_cast<char* const*>(peer_buffer_ptrs);

#define ARGS local_buf, peer_bufs, src_ep_offset, dst_tp_offset, tp_rank, E_local
    if (elem_size == 2) {
        if (num_gates == 2) {
            if      (tp_size == 8) { W13_DISPATCH_PRESET(peer_access_fused_transfer_w13_v3_tpl, 8, 2, 2, ARGS); }
            else if (tp_size == 4) { W13_DISPATCH_PRESET(peer_access_fused_transfer_w13_v3_tpl, 4, 2, 2, ARGS); }
        } else if (num_gates == 1) {
            if      (tp_size == 8) { W13_DISPATCH_PRESET(peer_access_fused_transfer_w13_v3_tpl, 8, 1, 2, ARGS); }
            else if (tp_size == 4) { W13_DISPATCH_PRESET(peer_access_fused_transfer_w13_v3_tpl, 4, 1, 2, ARGS); }
        }
    }
#undef ARGS
    printf("CUDA w13_v3 unsupported combo: tp=%d gates=%d elem=%d H=%d I=%d bpc=%d\n",
           tp_size, num_gates, elem_size, H, I, bpc);
}

void launch_peer_access_fused_transfer_w13_v3_ep(
    int64_t local_buffer_ptr,
    int64_t* peer_buffer_ptrs,
    int64_t src_tp_offset,
    int64_t dst_ep_offset,
    int tp_rank,
    int tp_size,
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
    const int chunks_per_peer = E_local * num_gates;
    const int bpc = pick_blocks_per_chunk(chunks_per_peer, sms, tp_size);
    const int blocks = chunks_per_peer * bpc;
    const int threads = WARPS_PER_BLOCK * 32;

    const auto local_buf = reinterpret_cast<const char*>(local_buffer_ptr);
    const auto peer_bufs = reinterpret_cast<char* const*>(peer_buffer_ptrs);

#define ARGS local_buf, peer_bufs, src_tp_offset, dst_ep_offset, tp_rank, E_local
    if (elem_size == 2) {
        if (num_gates == 2) {
            if      (tp_size == 8) { W13_DISPATCH_PRESET(peer_access_fused_transfer_w13_v3_ep_tpl, 8, 2, 2, ARGS); }
            else if (tp_size == 4) { W13_DISPATCH_PRESET(peer_access_fused_transfer_w13_v3_ep_tpl, 4, 2, 2, ARGS); }
        } else if (num_gates == 1) {
            if      (tp_size == 8) { W13_DISPATCH_PRESET(peer_access_fused_transfer_w13_v3_ep_tpl, 8, 1, 2, ARGS); }
            else if (tp_size == 4) { W13_DISPATCH_PRESET(peer_access_fused_transfer_w13_v3_ep_tpl, 4, 1, 2, ARGS); }
        }
    }
#undef ARGS
    printf("CUDA w13_v3_ep unsupported combo: tp=%d gates=%d elem=%d H=%d I=%d bpc=%d\n",
           tp_size, num_gates, elem_size, H, I, bpc);
}

void launch_peer_access_fused_transfer_w2_v3(
    int64_t local_buffer_ptr,
    int64_t* peer_buffer_ptrs,
    int64_t src_ep_offset,
    int64_t dst_tp_offset,
    int tp_rank,
    int tp_size,
    int E_local,
    int H,
    int I,
    int elem_size,
    cudaStream_t stream
) {
    int device, sms;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    const int bpc = pick_blocks_per_chunk(E_local, sms, tp_size);
    const int blocks = E_local * bpc;
    const int threads = WARPS_PER_BLOCK * 32;

    const auto local_buf = reinterpret_cast<const char*>(local_buffer_ptr);
    const auto peer_bufs = reinterpret_cast<char* const*>(peer_buffer_ptrs);

#define ARGS local_buf, peer_bufs, src_ep_offset, dst_tp_offset, tp_rank, E_local
    if (elem_size == 2) {
        if      (tp_size == 8) { W2_DISPATCH_PRESET(peer_access_fused_transfer_w2_v3_tpl, 8, 2, ARGS); }
        else if (tp_size == 4) { W2_DISPATCH_PRESET(peer_access_fused_transfer_w2_v3_tpl, 4, 2, ARGS); }
    }
#undef ARGS
    printf("CUDA w2_v3 unsupported combo: tp=%d H=%d I=%d elem=%d bpc=%d\n",
           tp_size, H, I, elem_size, bpc);
}

void launch_peer_access_fused_transfer_w2_v3_ep(
    int64_t local_buffer_ptr,
    int64_t* peer_buffer_ptrs,
    int64_t src_tp_offset,
    int64_t dst_ep_offset,
    int tp_rank,
    int tp_size,
    int E_local,
    int H,
    int I,
    int elem_size,
    cudaStream_t stream
) {
    int device, sms;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    const int bpc = pick_blocks_per_chunk(E_local, sms, tp_size);
    const int blocks = E_local * bpc;
    const int threads = WARPS_PER_BLOCK * 32;

    const auto local_buf = reinterpret_cast<const char*>(local_buffer_ptr);
    const auto peer_bufs = reinterpret_cast<char* const*>(peer_buffer_ptrs);

#define ARGS local_buf, peer_bufs, src_tp_offset, dst_ep_offset, tp_rank, E_local
    if (elem_size == 2) {
        if      (tp_size == 8) { W2_DISPATCH_PRESET(peer_access_fused_transfer_w2_v3_ep_tpl, 8, 2, ARGS); }
        else if (tp_size == 4) { W2_DISPATCH_PRESET(peer_access_fused_transfer_w2_v3_ep_tpl, 4, 2, ARGS); }
    }
#undef ARGS
    printf("CUDA w2_v3_ep unsupported combo: tp=%d H=%d I=%d elem=%d bpc=%d\n",
           tp_size, H, I, elem_size, bpc);
}
