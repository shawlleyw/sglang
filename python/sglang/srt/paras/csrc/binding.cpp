#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

// Forward declarations from .cu
void launch_peer_access_transfer(
    int64_t src_base_ptr,
    int64_t* dst_base_ptrs,
    const int64_t* src_offsets,
    const int64_t* dst_offsets,
    const int64_t* sizes,
    const int32_t* dst_ranks,
    int num_entries,
    cudaStream_t stream
);

void launch_peer_access_fused_transfer(
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
);

void launch_peer_access_mega_transfer(
    int64_t local_buffer_ptr,
    int64_t* peer_buffer_ptrs,
    const int64_t* ep_offsets,
    const int64_t* tp_offsets,
    int tp_rank,
    int tp_size,
    int E_local,
    int64_t I_prime_H,
    int num_gates,
    int elem_size,
    int num_layers,
    int num_threads,
    cudaStream_t stream
);

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
);

void launch_peer_access_mega_transfer_w2(
    int64_t local_buffer_ptr,
    int64_t* peer_buffer_ptrs,
    const int64_t* ep_offsets,
    const int64_t* tp_offsets,
    int tp_rank,
    int tp_size,
    int E_local,
    int H,
    int I_full_bytes,
    int I_prime_bytes,
    int num_layers,
    int num_threads,
    cudaStream_t stream
);

// Python-facing wrapper: accepts torch tensors
void launch_peer_access_transfer_py(
    int64_t src_base_ptr,
    torch::Tensor dst_base_ptrs,   // int64 tensor [MAX_PEERS] on GPU
    torch::Tensor src_offsets,     // int64 tensor [num_entries] on GPU
    torch::Tensor dst_offsets,     // int64 tensor [num_entries] on GPU
    torch::Tensor sizes,           // int64 tensor [num_entries] on GPU
    torch::Tensor dst_ranks,       // int32 tensor [num_entries] on GPU
    int64_t stream_ptr             // cudaStream_t as int64 (0 = default stream)
) {
    TORCH_CHECK(dst_base_ptrs.is_cuda(), "dst_base_ptrs must be on GPU");
    TORCH_CHECK(src_offsets.is_cuda(), "src_offsets must be on GPU");
    TORCH_CHECK(dst_offsets.is_cuda(), "dst_offsets must be on GPU");
    TORCH_CHECK(sizes.is_cuda(), "sizes must be on GPU");
    TORCH_CHECK(dst_ranks.is_cuda(), "dst_ranks must be on GPU");

    int num_entries = static_cast<int>(src_offsets.numel());
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    launch_peer_access_transfer(
        src_base_ptr,
        dst_base_ptrs.data_ptr<int64_t>(),
        src_offsets.data_ptr<int64_t>(),
        dst_offsets.data_ptr<int64_t>(),
        sizes.data_ptr<int64_t>(),
        dst_ranks.data_ptr<int32_t>(),
        num_entries,
        stream
    );
}

void launch_peer_access_fused_transfer_py(
    int64_t local_buffer_ptr,
    torch::Tensor peer_buffer_ptrs,   // int64 tensor [MAX_PEERS] on GPU
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
    launch_peer_access_fused_transfer(
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

void launch_peer_access_fused_transfer_w2_py(
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
    launch_peer_access_fused_transfer_w2(
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

void launch_peer_access_mega_transfer_py(
    int64_t local_buffer_ptr,
    torch::Tensor peer_buffer_ptrs,
    torch::Tensor ep_offsets,
    torch::Tensor tp_offsets,
    int tp_rank,
    int tp_size,
    int E_local,
    int64_t I_prime_H,
    int num_gates,
    int elem_size,
    int num_layers,
    int num_threads,
    int64_t stream_ptr
) {
    TORCH_CHECK(peer_buffer_ptrs.is_cuda(), "peer_buffer_ptrs must be on GPU");
    TORCH_CHECK(ep_offsets.is_cuda(), "ep_offsets must be on GPU");
    TORCH_CHECK(tp_offsets.is_cuda(), "tp_offsets must be on GPU");
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_peer_access_mega_transfer(
        local_buffer_ptr,
        peer_buffer_ptrs.data_ptr<int64_t>(),
        ep_offsets.data_ptr<int64_t>(),
        tp_offsets.data_ptr<int64_t>(),
        tp_rank,
        tp_size,
        E_local,
        I_prime_H,
        num_gates,
        elem_size,
        num_layers,
        num_threads,
        stream
    );
}

void launch_peer_access_fused_transfer_combined_py(
    int64_t local_buffer_ptr,
    torch::Tensor peer_buffer_ptrs,
    torch::Tensor w13_ep_offsets,
    torch::Tensor w13_tp_offsets,
    torch::Tensor w2_ep_offsets,
    torch::Tensor w2_tp_offsets,
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
    int64_t stream_ptr
) {
    TORCH_CHECK(peer_buffer_ptrs.is_cuda(), "peer_buffer_ptrs must be on GPU");
    TORCH_CHECK(w13_ep_offsets.is_cuda(), "w13_ep_offsets must be on GPU");
    TORCH_CHECK(w13_tp_offsets.is_cuda(), "w13_tp_offsets must be on GPU");
    TORCH_CHECK(w2_ep_offsets.is_cuda(), "w2_ep_offsets must be on GPU");
    TORCH_CHECK(w2_tp_offsets.is_cuda(), "w2_tp_offsets must be on GPU");
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_peer_access_fused_transfer_combined(
        local_buffer_ptr,
        peer_buffer_ptrs.data_ptr<int64_t>(),
        w13_ep_offsets.data_ptr<int64_t>(),
        w13_tp_offsets.data_ptr<int64_t>(),
        w2_ep_offsets.data_ptr<int64_t>(),
        w2_tp_offsets.data_ptr<int64_t>(),
        tp_rank,
        tp_size,
        E_local,
        I_prime_H,
        num_gates,
        elem_size,
        H,
        I_full_bytes,
        I_prime_bytes,
        num_layers,
        num_threads,
        stream
    );
}

void launch_peer_access_mega_transfer_w2_py(
    int64_t local_buffer_ptr,
    torch::Tensor peer_buffer_ptrs,
    torch::Tensor ep_offsets,
    torch::Tensor tp_offsets,
    int tp_rank,
    int tp_size,
    int E_local,
    int H,
    int I_full_bytes,
    int I_prime_bytes,
    int num_layers,
    int num_threads,
    int64_t stream_ptr
) {
    TORCH_CHECK(peer_buffer_ptrs.is_cuda(), "peer_buffer_ptrs must be on GPU");
    TORCH_CHECK(ep_offsets.is_cuda(), "ep_offsets must be on GPU");
    TORCH_CHECK(tp_offsets.is_cuda(), "tp_offsets must be on GPU");
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_peer_access_mega_transfer_w2(
        local_buffer_ptr,
        peer_buffer_ptrs.data_ptr<int64_t>(),
        ep_offsets.data_ptr<int64_t>(),
        tp_offsets.data_ptr<int64_t>(),
        tp_rank,
        tp_size,
        E_local,
        H,
        I_full_bytes,
        I_prime_bytes,
        num_layers,
        num_threads,
        stream
    );
}

PYBIND11_MODULE(paras_peer_access_cuda, m) {
    m.doc() = "ParaS CUDA peer access transfer kernels";
    m.def("launch_peer_access_transfer", &launch_peer_access_transfer_py,
          "Launch peer access transfer kernel",
          py::arg("src_base_ptr"),
          py::arg("dst_base_ptrs"),
          py::arg("src_offsets"),
          py::arg("dst_offsets"),
          py::arg("sizes"),
          py::arg("dst_ranks"),
          py::arg("stream_ptr") = int64_t(0));
    m.def("launch_peer_access_fused_transfer", &launch_peer_access_fused_transfer_py,
          "Launch fused strided-read peer access transfer kernel");
    m.def("launch_peer_access_fused_transfer_w2", &launch_peer_access_fused_transfer_w2_py,
          "Launch fused strided peer access transfer kernel for w2");
    m.def("launch_peer_access_mega_transfer", &launch_peer_access_mega_transfer_py,
          "Launch mega w13 kernel for all layers in one launch");
    m.def("launch_peer_access_mega_transfer_w2", &launch_peer_access_mega_transfer_w2_py,
          "Launch mega w2 kernel for all layers in one launch");
    m.def("launch_peer_access_fused_transfer_combined", &launch_peer_access_fused_transfer_combined_py,
          "Launch combined w13+w2 kernel for all layers in one launch");
}
