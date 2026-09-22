// BF16 attention weights: local EP->TP slicing and NVLink TP->EP restore.
// Source and destination are disjoint views in each rank's IPC arena. All ranks
// must finish their source writes before restore, then collectively fence after
// the complete layer before consuming/reusing either view. Launches use the
// current PyTorch CUDA stream; they allocate no buffers and perform no fence.
// Duplicate KV replicas may be invalid: restore reads only canonical owners.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>
#include <string>

namespace paras_attention {

// Each instruction transfers eight BF16 values. Cache-global loads avoid stale
// L1 data after another GPU writes the IPC buffer; caller supplies rank fences.
__device__ __forceinline__ uint4 load_vector(const uint4* p) {
  uint4 v;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}
__device__ __forceinline__ void store_vector(uint4* p, uint4 v) {
  asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(v.x),
               "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
}

template <bool QKV, bool PUSH, int HH = 0, int QQ = 0, int KK = 0, int DD = 0,
          int TT = 0, int THREADS = 0, int UNROLL = 0>
__global__ void restore(uint64_t local_base, const int64_t* __restrict__ peers,
                        int64_t src_offset, int64_t dst_offset, int h_arg,
                        int q_arg, int kv_arg, int head_arg, int t_arg,
                        int rank, int tile_vectors, int rotation) {
  const int H = HH ? HH : h_arg, Q = QQ ? QQ : q_arg;
  const int KV = KK ? KK : kv_arg, head = DD ? DD : head_arg,
            T = TT ? TT : t_arg;
  const int peer = (int(blockIdx.x) % T + rank * rotation) % T;
  const int source_rank = PUSH ? rank : peer;
  const int q = Q / T, ks = max(head, KV / T);
  const int replica = max(1, T / (KV / head));
  const int count = QKV ? (q + 2 * ks) * (H / 8) : H * (q / 8);
  const int begin = int(blockIdx.x / T) * tile_vectors;
  const uint64_t src_base = PUSH ? local_base : uint64_t(peers[peer]);
  const uint64_t dst_base = PUSH ? uint64_t(peers[peer]) : local_base;
  const uint4* src = reinterpret_cast<const uint4*>(src_base + src_offset);
  uint4* dst = reinterpret_cast<uint4*>(dst_base + dst_offset);
#pragma unroll
  for (int it = 0;
       it < (UNROLL ? UNROLL
                    : (tile_vectors + int(blockDim.x) - 1) / int(blockDim.x));
       ++it) {
    const int v = int(threadIdx.x) + it * (THREADS ? THREADS : int(blockDim.x));
    const int x = begin + v;
    if (v >= tile_vectors) continue;
    if (x >= count) break;
    int y;
    if constexpr (QKV) {
      const int row = x / (H / 8), col = x % (H / 8);
      if (row >= q && source_rank % replica != 0) continue;
      const int dstrow = row < q ? source_rank * q + row
                                 : Q + ((row - q) / ks) * KV +
                                       (source_rank / replica) * ks +
                                       (row - q) % ks;
      y = dstrow * (H / 8) + col;
    } else {
      y = (x / (q / 8)) * (Q / 8) + source_rank * (q / 8) + x % (q / 8);
    }
    store_vector(dst + y, load_vector(src + x));
  }
}

void launch_attention_restore(uint64_t local_buffer_ptr,
                              torch::Tensor peer_bases, int64_t src_offset,
                              int64_t dst_offset, int H, int Q, int KV,
                              int head_dim, int tp_size, int rank, bool is_qkv,
                              const std::string& method, int tile_bytes,
                              int threads, int rotation) {
  TORCH_CHECK(peer_bases.is_cuda() &&
                  peer_bases.scalar_type() == torch::kInt64 &&
                  peer_bases.is_contiguous() && peer_bases.numel() >= tp_size,
              "peer_bases must be contiguous CUDA int64 with tp_size entries");
  TORCH_CHECK(H > 0 && Q > 0 && KV > 0 && head_dim > 0 && tp_size > 0,
              "dimensions must be positive");
  TORCH_CHECK(Q % tp_size == 0 && H % 8 == 0 && (Q / tp_size) % 8 == 0 &&
                  KV % head_dim == 0 && head_dim % 8 == 0,
              "requires integral Q shards, heads, and 16-byte aligned rows");
  const int nh = KV / head_dim;
  TORCH_CHECK(nh >= tp_size ? nh % tp_size == 0 : tp_size % nh == 0,
              "KV head count and TP size must evenly divide one another");
  TORCH_CHECK(local_buffer_ptr % 16 == 0 && src_offset >= 0 &&
                  dst_offset >= 0 && src_offset % 16 == 0 &&
                  dst_offset % 16 == 0,
              "buffers and offsets must be 16-byte aligned");
  TORCH_CHECK(rank >= 0 && rank < tp_size && rotation >= 0,
              "invalid rank/rotation");
  TORCH_CHECK(tile_bytes > 0 && tile_bytes % 16 == 0 && threads >= 32 &&
                  threads <= 1024 && threads % 32 == 0,
              "invalid tile/threads");
  TORCH_CHECK(method == "pull" || method == "push",
              "method must be pull or push");
  c10::cuda::CUDAGuard guard(peer_bases.device());
  const int q = Q / tp_size, ks = std::max(head_dim, KV / tp_size);
  const int64_t vectors = is_qkv ? int64_t(q + 2 * ks) * (H / 8) : H * (q / 8);
  const int tile_vectors = tile_bytes / 16;
  const int64_t blocks =
      ((vectors + tile_vectors - 1) / tile_vectors) * tp_size;
  TORCH_CHECK(blocks <= 2147483647 && vectors <= 2147483647 &&
                  int64_t(Q + 2 * KV) * H / 8 <= 2147483647,
              "32-bit vector index limit exceeded");
  auto stream = at::cuda::getCurrentCUDAStream();
#define RUN(QKVB, PUSHB, HH, QQ, KK, DD, TT, NT, NU)                    \
  restore<QKVB, PUSHB, HH, QQ, KK, DD, TT, NT, NU>                      \
      <<<blocks, threads, 0, stream>>>(                                 \
          local_buffer_ptr, peer_bases.data_ptr<int64_t>(), src_offset, \
          dst_offset, H, Q, KV, head_dim, tp_size, rank, tile_vectors,  \
          rotation)
#define DIR(HH, QQ, KK, DD, TT, NT, NU)                \
  do {                                                 \
    if (is_qkv) {                                      \
      if (method == "push") {                          \
        RUN(true, true, HH, QQ, KK, DD, TT, NT, NU);   \
      } else {                                         \
        RUN(true, false, HH, QQ, KK, DD, TT, NT, NU);  \
      }                                                \
    } else {                                           \
      if (method == "push") {                          \
        RUN(false, true, HH, QQ, KK, DD, TT, NT, NU);  \
      } else {                                         \
        RUN(false, false, HH, QQ, KK, DD, TT, NT, NU); \
      }                                                \
    }                                                  \
  } while (0)
#define TILE(HH, QQ, KK, DD, TT, NT)     \
  do {                                   \
    if (tile_vectors == NT) {            \
      DIR(HH, QQ, KK, DD, TT, NT, 1);    \
    } else if (tile_vectors == NT * 2) { \
      DIR(HH, QQ, KK, DD, TT, NT, 2);    \
    } else if (tile_vectors == NT * 4) { \
      DIR(HH, QQ, KK, DD, TT, NT, 4);    \
    } else if (tile_vectors == NT * 8) { \
      DIR(HH, QQ, KK, DD, TT, NT, 8);    \
    } else {                             \
      DIR(HH, QQ, KK, DD, TT, NT, 0);    \
    }                                    \
  } while (0)
#define SHAPE(HH, QQ, KK, DD, TT)    \
  do {                               \
    if (threads == 256) {            \
      TILE(HH, QQ, KK, DD, TT, 256); \
    } else if (threads == 128) {     \
      TILE(HH, QQ, KK, DD, TT, 128); \
    } else {                         \
      DIR(HH, QQ, KK, DD, TT, 0, 0); \
    }                                \
  } while (0)
  if (H == 4096 && Q == 8192 && KV == 512 && head_dim == 128 && tp_size == 8) {
    SHAPE(4096, 8192, 512, 128, 8);
  } else if (H == 2048 && Q == 4096 && KV == 512 && head_dim == 128 &&
             tp_size == 8) {
    SHAPE(2048, 4096, 512, 128, 8);
  } else if (H == 2880 && Q == 4096 && KV == 512 && head_dim == 64 &&
             tp_size == 8) {
    SHAPE(2880, 4096, 512, 64, 8);
  } else {
    DIR(0, 0, 0, 0, 0, 0, 0);
  }
#undef SHAPE
#undef TILE
#undef DIR
#undef RUN
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int HH = 0, int QQ = 0, int KK = 0, int DD = 0, int TT = 0>
__global__ void slice_local(uint64_t base, int64_t src_offset,
                            int64_t dst_offset, int h_arg, int q_arg,
                            int kv_arg, int head_arg, int t_arg, int rank,
                            bool is_qkv, int tile_vectors) {
  const int H = HH ? HH : h_arg, Q = QQ ? QQ : q_arg, KV = KK ? KK : kv_arg,
            head = DD ? DD : head_arg, T = TT ? TT : t_arg;
  const int q = Q / T, ks = max(head, KV / T),
            replica = max(1, T / (KV / head));
  const int count = is_qkv ? (q + 2 * ks) * (H / 8) : H * (q / 8);
  const uint4* src = reinterpret_cast<const uint4*>(base + src_offset);
  uint4* dst = reinterpret_cast<uint4*>(base + dst_offset);
  for (int v = threadIdx.x; v < tile_vectors; v += blockDim.x) {
    const int x = int(blockIdx.x) * tile_vectors + v;
    if (x >= count) break;
    int y;
    if (is_qkv) {
      const int row = x / (H / 8), col = x % (H / 8);
      const int srcrow = row < q ? rank * q + row
                                 : Q + ((row - q) / ks) * KV +
                                       (rank / replica) * ks + (row - q) % ks;
      y = srcrow * (H / 8) + col;
    } else {
      y = (x / (q / 8)) * (Q / 8) + rank * (q / 8) + x % (q / 8);
    }
    store_vector(dst + x, load_vector(src + y));
  }
}
void launch_attention_slice(uint64_t local_buffer_ptr, int64_t src_offset,
                            int64_t dst_offset, int H, int Q, int KV,
                            int head_dim, int tp_size, int rank, bool is_qkv,
                            int tile_bytes, int threads) {
  TORCH_CHECK(H > 0 && Q > 0 && KV > 0 && head_dim > 0 && tp_size > 0,
              "positive dimensions required");
  TORCH_CHECK(Q % tp_size == 0 && H % 8 == 0 && (Q / tp_size) % 8 == 0 &&
                  KV % head_dim == 0 && head_dim % 8 == 0,
              "unaligned shape");
  const int heads = KV / head_dim;
  TORCH_CHECK(heads >= tp_size ? heads % tp_size == 0 : tp_size % heads == 0,
              "invalid KV replication");
  TORCH_CHECK(rank >= 0 && rank < tp_size && src_offset >= 0 &&
                  dst_offset >= 0 && local_buffer_ptr % 16 == 0 &&
                  src_offset % 16 == 0 && dst_offset % 16 == 0,
              "invalid pointers/offset/rank");
  TORCH_CHECK(tile_bytes > 0 && tile_bytes % 16 == 0 && threads >= 32 &&
                  threads <= 1024 && threads % 32 == 0,
              "invalid launch params");
  TORCH_CHECK(int64_t(Q + 2 * KV) * H / 8 <= 2147483647, "32bit index limit");
  const int q = Q / tp_size, ks = std::max(head_dim, KV / tp_size),
            tv = tile_bytes / 16;
  const int count = is_qkv ? (q + 2 * ks) * (H / 8) : H * (q / 8),
            blocks = (count + tv - 1) / tv;
  auto stream = at::cuda::getCurrentCUDAStream();
#define SLICE(HH, QQ, KK, DD, TT)                                            \
  slice_local<HH, QQ, KK, DD, TT><<<blocks, threads, 0, stream>>>(           \
      local_buffer_ptr, src_offset, dst_offset, H, Q, KV, head_dim, tp_size, \
      rank, is_qkv, tv)
  if (H == 4096 && Q == 8192 && KV == 512 && head_dim == 128 && tp_size == 8) {
    SLICE(4096, 8192, 512, 128, 8);
  } else if (H == 2048 && Q == 4096 && KV == 512 && head_dim == 128 &&
             tp_size == 8) {
    SLICE(2048, 4096, 512, 128, 8);
  } else if (H == 2880 && Q == 4096 && KV == 512 && head_dim == 64 &&
             tp_size == 8) {
    SLICE(2880, 4096, 512, 64, 8);
  } else {
    SLICE(0, 0, 0, 0, 0);
  }
#undef SLICE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace paras_attention
