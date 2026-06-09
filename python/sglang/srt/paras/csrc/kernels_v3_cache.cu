// V3 peer-access KV cache transfer / scatter kernels.
//
// Design guidelines (per user direction):
//   1) One warp owns one TOKEN SLOT per step (not one int4, not one peer).
//   2) For replicated heads (R = TP_SIZE/NUM_KV_HEADS > 1), the SAME warp
//      broadcasts to all R replica peers — reading source once, fanning out
//      R NVLink stores.  This eliminates duplicate HBM reads on the R=2 path
//      that the production kernel relies on L1 to dedup.
//   3) The SAME warp handles K and V for the same token via a half-warp split
//      (lanes 0..INT4_PER_HEAD-1 = K, lanes INT4_PER_HEAD..2*INT4_PER_HEAD-1 = V).
//      Token / head / in-head index compute is shared across K and V; only
//      the buffer-base pointer differs per half-warp.
//
// Tile structure (contiguous-tile, mirrors v3 weight kernel):
//   * gridDim.x = chunks_total * BLOCKS_PER_CHUNK
//   * Each chunk = a contiguous token range; BLOCKS_PER_CHUNK contiguous
//     blocks share one chunk and split work.
//   * 8 warps per block; for the EP->TP transfer kernel each warp owns one
//     head (WARPS_PER_HEAD = WARPS_PER_BLOCK / NUM_KV_HEADS); for the TP->EP
//     scatter kernel each warp processes a contiguous token range and
//     resolves the destination peer per-token.
//
// Template parameters:
//   TP_SIZE          {4, 8}
//   NUM_KV_HEADS     model config (e.g. 4 for Qwen3)
//   HEAD_DIM         model config (locked to 128 by static_assert)
//   ELEM_SIZE        bytes per element (2 = bf16)
//   BLOCKS_PER_CHUNK {8, 16, 32}
//
// Build-time knobs (-D):
//   V3_UNROLL              tokens per warp per outer iter (default 32)
//   V3_MIN_BLOCKS_PER_SM   __launch_bounds__ hint (default 4)

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#define WARPS_PER_BLOCK 8

#ifndef V3_UNROLL
#define V3_UNROLL 32
#endif

#ifndef V3_MIN_BLOCKS_PER_SM
#define V3_MIN_BLOCKS_PER_SM 4
#endif

static inline int pick_blocks_per_chunk_cache(int sms_per_gpu, int /*tp_size*/) {
    // Target ~2 waves at __launch_bounds__-implied occupancy.
    const int target_blocks = sms_per_gpu * V3_MIN_BLOCKS_PER_SM * 2;
    // For cache, we tile by token slabs.  More chunks => smaller chunks =>
    // better load balance at small N; fewer chunks => longer contiguous
    // NVLink TX per chunk.  Start at BPC=8 (= the smallest in our preset).
    (void)target_blocks;
    return 8;
}

// Compute (chunks_total, blocks) targeting ~2 waves with the chosen BPC.
static inline void compute_grid_cache(
    int num_local_tokens, int sms_per_gpu, int bpc,
    int& chunks_total, int& total_blocks
) {
    const int target_blocks = sms_per_gpu * V3_MIN_BLOCKS_PER_SM * 2;
    int chunks = (target_blocks + bpc - 1) / bpc;
    if (chunks < 1) chunks = 1;
    // Don't have more chunks than tokens (each chunk must have >= 1 token).
    if (chunks > num_local_tokens) chunks = num_local_tokens;
    if (chunks < 1) chunks = 1;
    chunks_total = chunks;
    total_blocks = chunks * bpc;
}

// =============================================================================
// EP -> TP: cache transfer kernel (broadcast: each EP token replicated to R
// TP peers per head).  Each warp owns one HEAD; the inner loop does
// 1 read + R NVLink stores per (warp, lane, token).
// =============================================================================

template <int TP_SIZE, int NUM_KV_HEADS, int HEAD_DIM, int ELEM_SIZE, int BLOCKS_PER_CHUNK>
__global__ __launch_bounds__(256, V3_MIN_BLOCKS_PER_SM)
void peer_access_kv_transfer_v3_tpl(
    const char* __restrict__ local_buffer,
    char* const* __restrict__ peer_buffers,
    const int* __restrict__ local_token_indices,
    int64_t src_k_offset, int64_t src_v_offset,
    int64_t dst_k_offset, int64_t dst_v_offset,
    int num_local_tokens,
    int dst_token_start,
    int tp_rank
) {
    static_assert(TP_SIZE >= NUM_KV_HEADS, "v3 cache: TP_SIZE must be >= NUM_KV_HEADS");
    static_assert(TP_SIZE % NUM_KV_HEADS == 0, "v3 cache: TP_SIZE must be multiple of NUM_KV_HEADS");
    static_assert(WARPS_PER_BLOCK % NUM_KV_HEADS == 0,
                  "v3 cache: NUM_KV_HEADS must divide WARPS_PER_BLOCK=8");
    static_assert(HEAD_DIM == 128, "v3 cache: HEAD_DIM locked to 128 (16 int4 per head)");
    static_assert(ELEM_SIZE == 2,  "v3 cache: ELEM_SIZE locked to 2 (bf16)");

    constexpr int R                = TP_SIZE / NUM_KV_HEADS;
    constexpr int WARPS_PER_HEAD   = WARPS_PER_BLOCK / NUM_KV_HEADS;
    constexpr int HEADS_PER_PEER   = (NUM_KV_HEADS / TP_SIZE > 0) ? (NUM_KV_HEADS / TP_SIZE) : 1;
    constexpr int INT4_PER_HEAD    = (HEAD_DIM * ELEM_SIZE) / 16;            // = 16
    constexpr int SRC_TOKEN_STRIDE = NUM_KV_HEADS * HEAD_DIM * ELEM_SIZE;    // EP src token stride
    constexpr int DST_TOKEN_STRIDE = HEADS_PER_PEER * HEAD_DIM * ELEM_SIZE;  // TP dst token stride
    constexpr int HEAD_STRIDE      = HEAD_DIM * ELEM_SIZE;

    const int warp_in_block = threadIdx.x >> 5;
    const int lane          = threadIdx.x & 31;

    // 1 warp = 1 head; WARPS_PER_HEAD warps split tokens for that head.
    const int head      = warp_in_block / WARPS_PER_HEAD;
    const int sub_warp  = warp_in_block - head * WARPS_PER_HEAD;
    const int ep_head_off = head * HEAD_STRIDE;

    // Chunking: contiguous token slabs.
    const int chunk_id           = blockIdx.x / BLOCKS_PER_CHUNK;
    const int sub_block_in_chunk = blockIdx.x - chunk_id * BLOCKS_PER_CHUNK;
    const int chunks_total       = gridDim.x  / BLOCKS_PER_CHUNK;

    const int tokens_per_chunk = num_local_tokens / chunks_total;
    const int rem              = num_local_tokens - tokens_per_chunk * chunks_total;
    const int chunk_t_start    = chunk_id * tokens_per_chunk
                                + ((chunk_id < rem) ? chunk_id : rem);
    const int chunk_t_count    = tokens_per_chunk + ((chunk_id < rem) ? 1 : 0);

    // Token range owned by this warp within the chunk.
    const int warps_per_chunk_head  = BLOCKS_PER_CHUNK * WARPS_PER_HEAD;
    const int warp_id_in_chunk_head = sub_block_in_chunk * WARPS_PER_HEAD + sub_warp;
    const int tokens_per_warp       = chunk_t_count / warps_per_chunk_head;
    const int wrem                  = chunk_t_count - tokens_per_warp * warps_per_chunk_head;
    const int warp_in_chunk_t_off   = warp_id_in_chunk_head * tokens_per_warp
                                    + ((warp_id_in_chunk_head < wrem) ? warp_id_in_chunk_head : wrem);
    const int warp_t_start = chunk_t_start + warp_in_chunk_t_off;
    const int warp_t_count = tokens_per_warp + ((warp_id_in_chunk_head < wrem) ? 1 : 0);
    const int warp_t_end   = warp_t_start + warp_t_count;

    // R replica destination buffers — resolved once per warp.
    char* dst_bufs[R];
    #pragma unroll
    for (int r = 0; r < R; r++) {
        const int peer = head * R + r;
        dst_bufs[r] = (peer == tp_rank)
                      ? const_cast<char*>(local_buffer)
                      : peer_buffers[peer];
    }

    // Half-warp K/V split: lane in [0, INT4_PER_HEAD) -> K; [INT4_PER_HEAD, 2*INT4_PER_HEAD) -> V.
    const bool is_k        = (lane < INT4_PER_HEAD);
    const int  in_head     = lane & (INT4_PER_HEAD - 1);
    const char* const src_kv_base = is_k ? (local_buffer + src_k_offset)
                                         : (local_buffer + src_v_offset);
    const int64_t dst_kv_off      = is_k ? dst_k_offset : dst_v_offset;
    const int lane_byte_off       = in_head * 16;

    // Inner loop: 1 token per warp-step, V3_UNROLL tokens per outer iter.
    int t = warp_t_start;
    while (t < warp_t_end) {
        #pragma unroll
        for (int u = 0; u < V3_UNROLL; u++) {
            const int t_now = t + u;
            if (t_now < warp_t_end) {
                const int src_token = local_token_indices[t_now];
                const int dst_token = dst_token_start + t_now;
                if (src_token != 0 && dst_token != 0) {
                    const size_t src_off = (size_t)src_token * SRC_TOKEN_STRIDE
                                         + ep_head_off
                                         + lane_byte_off;
                    const size_t dst_off = (size_t)dst_token * DST_TOKEN_STRIDE
                                         + lane_byte_off;
                    // ONE source read; index compute shared K+V across the warp.
                    const int4 data = __ldg(reinterpret_cast<const int4*>(src_kv_base + src_off));
                    // Broadcast to R replica peers.
                    #pragma unroll
                    for (int r = 0; r < R; r++) {
                        *reinterpret_cast<int4*>(dst_bufs[r] + dst_kv_off + dst_off) = data;
                    }
                }
            }
        }
        t += V3_UNROLL;
    }
}

// =============================================================================
// TP -> EP: cache scatter kernel.  Each TP rank holds HEADS_PER_RANK heads
// (= 1 for Qwen3 tp=8); each TP token is written to exactly ONE EP peer per
// token_to_rank[t].  No R-broadcast.  K+V still share index compute via the
// half-warp split.  Warps are assigned contiguous token RANGES (not peers);
// dst_buf is resolved per-token from peer_buffers[].
// =============================================================================

template <int TP_SIZE, int NUM_KV_HEADS, int HEAD_DIM, int ELEM_SIZE, int BLOCKS_PER_CHUNK>
__global__ __launch_bounds__(256, V3_MIN_BLOCKS_PER_SM)
void peer_access_kv_scatter_v3_tpl(
    const char* __restrict__ local_buffer,
    char* const* __restrict__ peer_buffers,
    const int* __restrict__ tp_token_positions,
    const int* __restrict__ token_to_rank,
    const int* __restrict__ ep_dst_positions,
    int64_t src_k_offset, int64_t src_v_offset,
    int64_t dst_k_offset, int64_t dst_v_offset,
    int num_local_tokens,
    int tp_rank
) {
    static_assert(TP_SIZE >= NUM_KV_HEADS, "v3 cache: TP_SIZE must be >= NUM_KV_HEADS");
    static_assert(TP_SIZE % NUM_KV_HEADS == 0, "v3 cache: TP_SIZE must be multiple of NUM_KV_HEADS");
    static_assert(HEAD_DIM == 128, "v3 cache: HEAD_DIM locked to 128");
    static_assert(ELEM_SIZE == 2,  "v3 cache: ELEM_SIZE locked to 2 (bf16)");

    constexpr int HEADS_PER_RANK   = (NUM_KV_HEADS / TP_SIZE > 0) ? (NUM_KV_HEADS / TP_SIZE) : 1;
    constexpr int INT4_PER_HEAD    = (HEAD_DIM * ELEM_SIZE) / 16;            // = 16
    constexpr int SRC_TOKEN_STRIDE = HEADS_PER_RANK * HEAD_DIM * ELEM_SIZE;  // TP src token stride
    constexpr int DST_TOKEN_STRIDE = NUM_KV_HEADS * HEAD_DIM * ELEM_SIZE;    // EP dst token stride
    constexpr int HEAD_STRIDE      = HEAD_DIM * ELEM_SIZE;

    const int warp_in_block = threadIdx.x >> 5;
    const int lane          = threadIdx.x & 31;

    // Chunking: same structure as transfer.  All 8 warps per block share the
    // (single) head this rank owns; each warp processes a contiguous token range
    // and resolves dst_peer per-token.
    const int chunk_id           = blockIdx.x / BLOCKS_PER_CHUNK;
    const int sub_block_in_chunk = blockIdx.x - chunk_id * BLOCKS_PER_CHUNK;
    const int chunks_total       = gridDim.x  / BLOCKS_PER_CHUNK;

    const int tokens_per_chunk = num_local_tokens / chunks_total;
    const int rem              = num_local_tokens - tokens_per_chunk * chunks_total;
    const int chunk_t_start    = chunk_id * tokens_per_chunk
                                + ((chunk_id < rem) ? chunk_id : rem);
    const int chunk_t_count    = tokens_per_chunk + ((chunk_id < rem) ? 1 : 0);

    const int warps_per_chunk  = BLOCKS_PER_CHUNK * WARPS_PER_BLOCK;
    const int warp_id_in_chunk = sub_block_in_chunk * WARPS_PER_BLOCK + warp_in_block;
    const int tokens_per_warp  = chunk_t_count / warps_per_chunk;
    const int wrem             = chunk_t_count - tokens_per_warp * warps_per_chunk;
    const int warp_in_chunk_t_off = warp_id_in_chunk * tokens_per_warp
                                  + ((warp_id_in_chunk < wrem) ? warp_id_in_chunk : wrem);
    const int warp_t_start = chunk_t_start + warp_in_chunk_t_off;
    const int warp_t_count = tokens_per_warp + ((warp_id_in_chunk < wrem) ? 1 : 0);
    const int warp_t_end   = warp_t_start + warp_t_count;

    // This rank's head index in the EP layout (per kernel convention).
    // For Qwen3 tp=8, num_kv_heads=4: dst_head_idx = tp_rank / 2.
    // For tp=4: dst_head_idx = tp_rank.
    const int dst_head_idx = tp_rank * NUM_KV_HEADS / TP_SIZE;
    const int dst_head_off = dst_head_idx * HEAD_STRIDE;

    // Half-warp K/V split: same as transfer kernel.
    const bool is_k        = (lane < INT4_PER_HEAD);
    const int  in_head     = lane & (INT4_PER_HEAD - 1);
    const char* const src_kv_base = is_k ? (local_buffer + src_k_offset)
                                         : (local_buffer + src_v_offset);
    const int64_t dst_kv_off      = is_k ? dst_k_offset : dst_v_offset;
    const int lane_byte_off       = in_head * 16;

    int t = warp_t_start;
    while (t < warp_t_end) {
        #pragma unroll
        for (int u = 0; u < V3_UNROLL; u++) {
            const int t_now = t + u;
            if (t_now < warp_t_end) {
                const int dst_peer  = token_to_rank[t_now];
                const int src_token = tp_token_positions[t_now];
                const int dst_token = ep_dst_positions[t_now];
                if (src_token != 0 && dst_token != 0) {
                    char* const dst_buf = (dst_peer == tp_rank)
                                          ? const_cast<char*>(local_buffer)
                                          : peer_buffers[dst_peer];
                    const size_t src_off = (size_t)src_token * SRC_TOKEN_STRIDE
                                         + lane_byte_off;
                    const size_t dst_off = (size_t)dst_token * DST_TOKEN_STRIDE
                                         + dst_head_off
                                         + lane_byte_off;
                    const int4 data = __ldg(reinterpret_cast<const int4*>(src_kv_base + src_off));
                    *reinterpret_cast<int4*>(dst_buf + dst_kv_off + dst_off) = data;
                }
            }
        }
        t += V3_UNROLL;
    }
}

// =============================================================================
// Dispatchers.  Supported model presets:
//   Qwen3-235B / Qwen3-30B (NUM_KV_HEADS=4, HEAD_DIM=128, ELEM_SIZE=2)
//   TP_SIZE in {4, 8}; BLOCKS_PER_CHUNK in {8, 16, 32}.
// =============================================================================

#define KV_BPC_CASE(KERNEL, TPS, NKH, HD, ES, BPC, ...) \
    case BPC: KERNEL<TPS, NKH, HD, ES, BPC><<<blocks, threads, 0, stream>>>(__VA_ARGS__); return

#define DISPATCH_KV_BPC_ALL(KERNEL, TPS, NKH, HD, ES, ...)                 \
    switch (bpc) {                                                          \
        KV_BPC_CASE(KERNEL, TPS, NKH, HD, ES, 8,  __VA_ARGS__);             \
        KV_BPC_CASE(KERNEL, TPS, NKH, HD, ES, 16, __VA_ARGS__);             \
        KV_BPC_CASE(KERNEL, TPS, NKH, HD, ES, 32, __VA_ARGS__);             \
        default: break;                                                     \
    }

#define KV_DISPATCH_PRESET(KERNEL, ...)                                                         \
    do {                                                                                        \
        if (num_kv_heads == 4 && head_dim == 128 && elem_size == 2) {                           \
            if      (tp_size == 8) { DISPATCH_KV_BPC_ALL(KERNEL, 8, 4, 128, 2, __VA_ARGS__); }  \
            else if (tp_size == 4) { DISPATCH_KV_BPC_ALL(KERNEL, 4, 4, 128, 2, __VA_ARGS__); }  \
        }                                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// EP -> TP transfer launcher.
// ---------------------------------------------------------------------------
void launch_peer_access_kv_transfer_v3(
    int64_t local_buffer_ptr,
    int64_t* peer_buffer_ptrs,
    int* local_token_indices,
    int64_t src_k_offset, int64_t src_v_offset,
    int64_t dst_k_offset, int64_t dst_v_offset,
    int num_local_tokens, int dst_token_start,
    int num_kv_heads, int tp_rank, int tp_size,
    int head_dim, int elem_size,
    cudaStream_t stream
) {
    if (num_local_tokens <= 0) return;

    int device, sms;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);

    const int bpc = pick_blocks_per_chunk_cache(sms, tp_size);
    int chunks_total = 0, blocks = 0;
    compute_grid_cache(num_local_tokens, sms, bpc, chunks_total, blocks);
    const int threads = WARPS_PER_BLOCK * 32;

    const auto local_buf = reinterpret_cast<const char*>(local_buffer_ptr);
    const auto peer_bufs = reinterpret_cast<char* const*>(peer_buffer_ptrs);

#define ARGS local_buf, peer_bufs, local_token_indices, \
             src_k_offset, src_v_offset, dst_k_offset, dst_v_offset, \
             num_local_tokens, dst_token_start, tp_rank
    KV_DISPATCH_PRESET(peer_access_kv_transfer_v3_tpl, ARGS);
#undef ARGS
    printf("CUDA kv_transfer_v3 unsupported combo: tp=%d num_kv_heads=%d head_dim=%d elem=%d bpc=%d\n",
           tp_size, num_kv_heads, head_dim, elem_size, bpc);
}

// ---------------------------------------------------------------------------
// TP -> EP scatter launcher.
// ---------------------------------------------------------------------------
void launch_peer_access_kv_scatter_v3(
    int64_t local_buffer_ptr,
    int64_t* peer_buffer_ptrs,
    int* tp_token_positions,
    int* token_to_rank,
    int* ep_dst_positions,
    int64_t src_k_offset, int64_t src_v_offset,
    int64_t dst_k_offset, int64_t dst_v_offset,
    int num_local_tokens,
    int num_kv_heads, int tp_rank, int tp_size,
    int head_dim, int elem_size,
    cudaStream_t stream
) {
    if (num_local_tokens <= 0) return;

    int device, sms;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);

    const int bpc = pick_blocks_per_chunk_cache(sms, tp_size);
    int chunks_total = 0, blocks = 0;
    compute_grid_cache(num_local_tokens, sms, bpc, chunks_total, blocks);
    const int threads = WARPS_PER_BLOCK * 32;

    const auto local_buf = reinterpret_cast<const char*>(local_buffer_ptr);
    const auto peer_bufs = reinterpret_cast<char* const*>(peer_buffer_ptrs);

#define ARGS local_buf, peer_bufs, tp_token_positions, token_to_rank, ep_dst_positions, \
             src_k_offset, src_v_offset, dst_k_offset, dst_v_offset, \
             num_local_tokens, tp_rank
    KV_DISPATCH_PRESET(peer_access_kv_scatter_v3_tpl, ARGS);
#undef ARGS
    printf("CUDA kv_scatter_v3 unsupported combo: tp=%d num_kv_heads=%d head_dim=%d elem=%d bpc=%d\n",
           tp_size, num_kv_heads, head_dim, elem_size, bpc);
}
