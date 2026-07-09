#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

// Forward declarations from .cu (v2 kernels only)
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
);

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
);

// Forward declarations for TP→EP reverse kernels
void launch_peer_access_fused_transfer_w13_ep(
    int64_t local_buffer_ptr,
    int64_t* peer_buffer_ptrs,
    int64_t src_tp_offset,
    int64_t dst_ep_offset,
    int tp_rank,
    int tp_size,
    int E_local,
    int64_t I_prime_H,
    int num_gates,
    int elem_size,
    cudaStream_t stream
);

void launch_peer_access_fused_transfer_w2_ep(
    int64_t local_buffer_ptr,
    int64_t* peer_buffer_ptrs,
    int64_t src_tp_offset,
    int64_t dst_ep_offset,
    int tp_rank,
    int tp_size,
    int E_local,
    int H,
    int I_full_bytes,
    int I_prime_bytes,
    cudaStream_t stream
);

void launch_peer_access_kv_transfer(
    int64_t local_buffer_ptr,
    int64_t* peer_buffer_ptrs,
    int* local_token_indices,
    int64_t src_k_offset,
    int64_t src_v_offset,
    int64_t dst_k_offset,
    int64_t dst_v_offset,
    int num_local_tokens,
    int dst_token_start,
    int num_kv_heads,
    int tp_rank,
    int tp_size,
    int head_dim,
    int elem_size,
    cudaStream_t stream
);

void launch_peer_access_kv_scatter(
    int64_t local_buffer_ptr,
    int64_t* peer_buffer_ptrs,
    int* tp_token_positions,
    int* token_to_rank,
    int* ep_dst_positions,
    int64_t src_k_offset,
    int64_t src_v_offset,
    int64_t dst_k_offset,
    int64_t dst_v_offset,
    int num_local_tokens,
    int heads_per_rank,
    int num_kv_heads,
    int tp_rank,
    int tp_size,
    int head_dim,
    int elem_size,
    cudaStream_t stream
);

// v3 launchers (kernels_v3.cu, kernels_v3_cache.cu)
void launch_peer_access_fused_transfer_w13_v3(
    int64_t, int64_t*, int64_t, int64_t,
    int, int, int, int, int, int, int, cudaStream_t);

void launch_peer_access_fused_transfer_w13_v3_ep(
    int64_t, int64_t*, int64_t, int64_t,
    int, int, int, int, int, int, int, cudaStream_t);

void launch_peer_access_fused_transfer_w2_v3(
    int64_t, int64_t*, int64_t, int64_t,
    int, int, int, int, int, int, cudaStream_t);

void launch_peer_access_fused_transfer_w2_v3_ep(
    int64_t, int64_t*, int64_t, int64_t,
    int, int, int, int, int, int, cudaStream_t);

void launch_peer_access_kv_transfer_v3(
    int64_t, int64_t*, int*,
    int64_t, int64_t, int64_t, int64_t,
    int, int, int, int, int, int, int,
    cudaStream_t);

void launch_peer_access_kv_scatter_v3(
    int64_t, int64_t*, int*, int*, int*,
    int64_t, int64_t, int64_t, int64_t,
    int, int, int, int, int, int,
    cudaStream_t);

// Python-facing wrappers: accept torch tensors
void launch_peer_access_fused_transfer_w13_v2_py(
    int64_t local_buffer_ptr,
    torch::Tensor peer_buffer_ptrs,
    int64_t src_ep_offset,
    int64_t dst_tp_offset,
    int tp_rank,
    int tp_size,
    int E_local,
    int64_t I_prime_H,
    int num_gates,
    int elem_size,
    int64_t stream_ptr
) {
    TORCH_CHECK(peer_buffer_ptrs.is_cuda(), "peer_buffer_ptrs must be on GPU");
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_peer_access_fused_transfer_w13_v2(
        local_buffer_ptr,
        peer_buffer_ptrs.data_ptr<int64_t>(),
        src_ep_offset,
        dst_tp_offset,
        tp_rank,
        tp_size,
        E_local,
        I_prime_H,
        num_gates,
        elem_size,
        stream
    );
}

void launch_peer_access_fused_transfer_w2_v2_py(
    int64_t local_buffer_ptr,
    torch::Tensor peer_buffer_ptrs,
    int64_t src_ep_offset,
    int64_t dst_tp_offset,
    int tp_rank,
    int tp_size,
    int E_local,
    int H,
    int I_full_bytes,
    int I_prime_bytes,
    int64_t stream_ptr
) {
    TORCH_CHECK(peer_buffer_ptrs.is_cuda(), "peer_buffer_ptrs must be on GPU");
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_peer_access_fused_transfer_w2_v2(
        local_buffer_ptr,
        peer_buffer_ptrs.data_ptr<int64_t>(),
        src_ep_offset,
        dst_tp_offset,
        tp_rank,
        tp_size,
        E_local,
        H,
        I_full_bytes,
        I_prime_bytes,
        stream
    );
}

void launch_peer_access_fused_transfer_w13_ep_py(
    int64_t local_buffer_ptr,
    torch::Tensor peer_buffer_ptrs,
    int64_t src_tp_offset,
    int64_t dst_ep_offset,
    int tp_rank,
    int tp_size,
    int E_local,
    int64_t I_prime_H,
    int num_gates,
    int elem_size,
    int64_t stream_ptr
) {
    TORCH_CHECK(peer_buffer_ptrs.is_cuda(), "peer_buffer_ptrs must be on GPU");
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_peer_access_fused_transfer_w13_ep(
        local_buffer_ptr,
        peer_buffer_ptrs.data_ptr<int64_t>(),
        src_tp_offset,
        dst_ep_offset,
        tp_rank,
        tp_size,
        E_local,
        I_prime_H,
        num_gates,
        elem_size,
        stream
    );
}

void launch_peer_access_fused_transfer_w2_ep_py(
    int64_t local_buffer_ptr,
    torch::Tensor peer_buffer_ptrs,
    int64_t src_tp_offset,
    int64_t dst_ep_offset,
    int tp_rank,
    int tp_size,
    int E_local,
    int H,
    int I_full_bytes,
    int I_prime_bytes,
    int64_t stream_ptr
) {
    TORCH_CHECK(peer_buffer_ptrs.is_cuda(), "peer_buffer_ptrs must be on GPU");
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_peer_access_fused_transfer_w2_ep(
        local_buffer_ptr,
        peer_buffer_ptrs.data_ptr<int64_t>(),
        src_tp_offset,
        dst_ep_offset,
        tp_rank,
        tp_size,
        E_local,
        H,
        I_full_bytes,
        I_prime_bytes,
        stream
    );
}

void launch_peer_access_kv_transfer_py(
    int64_t local_buffer_ptr,
    torch::Tensor peer_buffer_ptrs,
    torch::Tensor local_token_indices,
    int64_t src_k_offset,
    int64_t src_v_offset,
    int64_t dst_k_offset,
    int64_t dst_v_offset,
    int num_local_tokens,
    int dst_token_start,
    int num_kv_heads,
    int tp_rank,
    int tp_size,
    int head_dim,
    int elem_size,
    int64_t stream_ptr
) {
    TORCH_CHECK(peer_buffer_ptrs.is_cuda(), "peer_buffer_ptrs must be on GPU");
    TORCH_CHECK(local_token_indices.is_cuda(), "local_token_indices must be on GPU");
    TORCH_CHECK(
        (head_dim * elem_size) % 16 == 0,
        "launch_peer_access_kv_transfer: head_dim * elem_size must be a "
        "multiple of 16 for the int4-vectorized inner loop "
        "(int4_per_head = (head_dim * elem_size) >> 4 in peer_access_transfer.cu). "
        "Got head_dim=", head_dim, ", elem_size=", elem_size,
        ", product=", head_dim * elem_size,
        ", trailing bytes=", (head_dim * elem_size) % 16,
        " would be silently truncated per head."
    );
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_peer_access_kv_transfer(
        local_buffer_ptr,
        peer_buffer_ptrs.data_ptr<int64_t>(),
        local_token_indices.data_ptr<int>(),
        src_k_offset,
        src_v_offset,
        dst_k_offset,
        dst_v_offset,
        num_local_tokens,
        dst_token_start,
        num_kv_heads,
        tp_rank,
        tp_size,
        head_dim,
        elem_size,
        stream
    );
}

void launch_peer_access_kv_scatter_py(
    int64_t local_buffer_ptr,
    torch::Tensor peer_buffer_ptrs,
    torch::Tensor tp_token_positions,
    torch::Tensor token_to_rank,
    torch::Tensor ep_dst_positions,
    int64_t src_k_offset,
    int64_t src_v_offset,
    int64_t dst_k_offset,
    int64_t dst_v_offset,
    int num_local_tokens,
    int heads_per_rank,
    int num_kv_heads,
    int tp_rank,
    int tp_size,
    int head_dim,
    int elem_size,
    int64_t stream_ptr
) {
    TORCH_CHECK(peer_buffer_ptrs.is_cuda(), "peer_buffer_ptrs must be on GPU");
    TORCH_CHECK(tp_token_positions.is_cuda(), "tp_token_positions must be on GPU");
    TORCH_CHECK(token_to_rank.is_cuda(), "token_to_rank must be on GPU");
    TORCH_CHECK(ep_dst_positions.is_cuda(), "ep_dst_positions must be on GPU");
    TORCH_CHECK(
        (head_dim * elem_size) % 16 == 0,
        "launch_peer_access_kv_scatter: head_dim * elem_size must be a "
        "multiple of 16 for the int4-vectorized inner loop "
        "(int4_per_head = (head_dim * elem_size) >> 4 in peer_access_transfer.cu). "
        "Got head_dim=", head_dim, ", elem_size=", elem_size,
        ", product=", head_dim * elem_size,
        ", trailing bytes=", (head_dim * elem_size) % 16,
        " would be silently truncated per head."
    );
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_peer_access_kv_scatter(
        local_buffer_ptr,
        peer_buffer_ptrs.data_ptr<int64_t>(),
        tp_token_positions.data_ptr<int>(),
        token_to_rank.data_ptr<int>(),
        ep_dst_positions.data_ptr<int>(),
        src_k_offset,
        src_v_offset,
        dst_k_offset,
        dst_v_offset,
        num_local_tokens,
        heads_per_rank,
        num_kv_heads,
        tp_rank,
        tp_size,
        head_dim,
        elem_size,
        stream
    );
}

static void check_v3_tp_size(int tp_size) {
    TORCH_CHECK(tp_size == 4 || tp_size == 8,
                "v3 kernels require tp_size in {4, 8}; got ", tp_size);
}

void launch_w13_v3_py(
    int64_t local_buffer_ptr, torch::Tensor peer_buffer_ptrs,
    int64_t src_ep_offset, int64_t dst_tp_offset,
    int tp_rank, int tp_size, int E_local,
    int H, int I, int num_gates, int elem_size,
    int64_t stream_ptr
) {
    TORCH_CHECK(peer_buffer_ptrs.is_cuda());
    check_v3_tp_size(tp_size);
    launch_peer_access_fused_transfer_w13_v3(
        local_buffer_ptr, peer_buffer_ptrs.data_ptr<int64_t>(),
        src_ep_offset, dst_tp_offset, tp_rank, tp_size,
        E_local, H, I, num_gates, elem_size,
        reinterpret_cast<cudaStream_t>(stream_ptr));
}

void launch_w13_v3_ep_py(
    int64_t local_buffer_ptr, torch::Tensor peer_buffer_ptrs,
    int64_t src_tp_offset, int64_t dst_ep_offset,
    int tp_rank, int tp_size, int E_local,
    int H, int I, int num_gates, int elem_size,
    int64_t stream_ptr
) {
    TORCH_CHECK(peer_buffer_ptrs.is_cuda());
    check_v3_tp_size(tp_size);
    launch_peer_access_fused_transfer_w13_v3_ep(
        local_buffer_ptr, peer_buffer_ptrs.data_ptr<int64_t>(),
        src_tp_offset, dst_ep_offset, tp_rank, tp_size,
        E_local, H, I, num_gates, elem_size,
        reinterpret_cast<cudaStream_t>(stream_ptr));
}

void launch_w2_v3_py(
    int64_t local_buffer_ptr, torch::Tensor peer_buffer_ptrs,
    int64_t src_ep_offset, int64_t dst_tp_offset,
    int tp_rank, int tp_size, int E_local,
    int H, int I, int elem_size,
    int64_t stream_ptr
) {
    TORCH_CHECK(peer_buffer_ptrs.is_cuda());
    check_v3_tp_size(tp_size);
    launch_peer_access_fused_transfer_w2_v3(
        local_buffer_ptr, peer_buffer_ptrs.data_ptr<int64_t>(),
        src_ep_offset, dst_tp_offset, tp_rank, tp_size,
        E_local, H, I, elem_size,
        reinterpret_cast<cudaStream_t>(stream_ptr));
}

void launch_w2_v3_ep_py(
    int64_t local_buffer_ptr, torch::Tensor peer_buffer_ptrs,
    int64_t src_tp_offset, int64_t dst_ep_offset,
    int tp_rank, int tp_size, int E_local,
    int H, int I, int elem_size,
    int64_t stream_ptr
) {
    TORCH_CHECK(peer_buffer_ptrs.is_cuda());
    check_v3_tp_size(tp_size);
    launch_peer_access_fused_transfer_w2_v3_ep(
        local_buffer_ptr, peer_buffer_ptrs.data_ptr<int64_t>(),
        src_tp_offset, dst_ep_offset, tp_rank, tp_size,
        E_local, H, I, elem_size,
        reinterpret_cast<cudaStream_t>(stream_ptr));
}

void launch_kv_transfer_v3_py(
    int64_t local_buffer_ptr, torch::Tensor peer_buffer_ptrs,
    torch::Tensor local_token_indices,
    int64_t src_k_offset, int64_t src_v_offset,
    int64_t dst_k_offset, int64_t dst_v_offset,
    int num_local_tokens, int dst_token_start,
    int num_kv_heads, int tp_rank, int tp_size,
    int head_dim, int elem_size,
    int64_t stream_ptr
) {
    TORCH_CHECK(peer_buffer_ptrs.is_cuda());
    TORCH_CHECK(local_token_indices.is_cuda());
    TORCH_CHECK(local_token_indices.scalar_type() == torch::kInt32);
    check_v3_tp_size(tp_size);
    launch_peer_access_kv_transfer_v3(
        local_buffer_ptr, peer_buffer_ptrs.data_ptr<int64_t>(),
        local_token_indices.data_ptr<int>(),
        src_k_offset, src_v_offset, dst_k_offset, dst_v_offset,
        num_local_tokens, dst_token_start,
        num_kv_heads, tp_rank, tp_size, head_dim, elem_size,
        reinterpret_cast<cudaStream_t>(stream_ptr));
}

void launch_kv_scatter_v3_py(
    int64_t local_buffer_ptr, torch::Tensor peer_buffer_ptrs,
    torch::Tensor tp_token_positions, torch::Tensor token_to_rank,
    torch::Tensor ep_dst_positions,
    int64_t src_k_offset, int64_t src_v_offset,
    int64_t dst_k_offset, int64_t dst_v_offset,
    int num_local_tokens,
    int num_kv_heads, int tp_rank, int tp_size,
    int head_dim, int elem_size,
    int64_t stream_ptr
) {
    TORCH_CHECK(peer_buffer_ptrs.is_cuda());
    TORCH_CHECK(tp_token_positions.is_cuda() && token_to_rank.is_cuda() && ep_dst_positions.is_cuda());
    TORCH_CHECK(tp_token_positions.scalar_type() == torch::kInt32);
    TORCH_CHECK(token_to_rank.scalar_type() == torch::kInt32);
    TORCH_CHECK(ep_dst_positions.scalar_type() == torch::kInt32);
    check_v3_tp_size(tp_size);
    launch_peer_access_kv_scatter_v3(
        local_buffer_ptr, peer_buffer_ptrs.data_ptr<int64_t>(),
        tp_token_positions.data_ptr<int>(),
        token_to_rank.data_ptr<int>(),
        ep_dst_positions.data_ptr<int>(),
        src_k_offset, src_v_offset, dst_k_offset, dst_v_offset,
        num_local_tokens,
        num_kv_heads, tp_rank, tp_size, head_dim, elem_size,
        reinterpret_cast<cudaStream_t>(stream_ptr));
}

PYBIND11_MODULE(paras_peer_access_cuda, m) {
    m.doc() = "ParaS CUDA peer access transfer kernels (v2 baseline + v3 contiguous-tile)";
    m.def("launch_peer_access_fused_transfer_w13_v2", &launch_peer_access_fused_transfer_w13_v2_py,
          "Launch NVLink-optimized v2 w13 peer access transfer kernel");
    m.def("launch_peer_access_fused_transfer_w2_v2", &launch_peer_access_fused_transfer_w2_v2_py,
          "Launch NVLink-optimized v2 w2 peer access transfer kernel");
    m.def("launch_peer_access_kv_transfer", &launch_peer_access_kv_transfer_py,
          "Launch NVLink-optimized KV cache peer access transfer kernel");
    m.def("launch_peer_access_kv_scatter", &launch_peer_access_kv_scatter_py,
          "Launch NVLink-optimized KV cache peer access scatter kernel (TP→EP)");
    m.def("launch_peer_access_fused_transfer_w13_ep", &launch_peer_access_fused_transfer_w13_ep_py,
          "Launch TP→EP reverse w13 peer access transfer kernel");
    m.def("launch_peer_access_fused_transfer_w2_ep", &launch_peer_access_fused_transfer_w2_ep_py,
          "Launch TP→EP reverse w2 peer access transfer kernel");
    m.def("launch_peer_access_fused_transfer_w13_v3", &launch_w13_v3_py,
          "v3 w13 EP->TP (contiguous-tile)");
    m.def("launch_peer_access_fused_transfer_w13_v3_ep", &launch_w13_v3_ep_py,
          "v3 w13 TP->EP (contiguous-tile)");
    m.def("launch_peer_access_fused_transfer_w2_v3", &launch_w2_v3_py,
          "v3 w2 EP->TP (contiguous-tile, row-aligned)");
    m.def("launch_peer_access_fused_transfer_w2_v3_ep", &launch_w2_v3_ep_py,
          "v3 w2 TP->EP (contiguous-tile, row-aligned)");
    m.def("launch_peer_access_kv_transfer_v3", &launch_kv_transfer_v3_py,
          "v3 KV cache EP->TP transfer (R-broadcast, half-warp K/V)");
    m.def("launch_peer_access_kv_scatter_v3", &launch_kv_scatter_v3_py,
          "v3 KV cache TP->EP scatter (contiguous-tile, half-warp K/V)");
}
