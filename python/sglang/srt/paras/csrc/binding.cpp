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

PYBIND11_MODULE(paras_peer_access_cuda, m) {
    m.doc() = "ParaS CUDA peer access transfer kernels (v2, NVLink-optimized)";
    m.def("launch_peer_access_fused_transfer_w13_v2", &launch_peer_access_fused_transfer_w13_v2_py,
          "Launch NVLink-optimized v2 w13 peer access transfer kernel");
    m.def("launch_peer_access_fused_transfer_w2_v2", &launch_peer_access_fused_transfer_w2_v2_py,
          "Launch NVLink-optimized v2 w2 peer access transfer kernel");
    m.def("launch_peer_access_kv_transfer", &launch_peer_access_kv_transfer_py,
          "Launch NVLink-optimized KV cache peer access transfer kernel");
}
