#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

// Forward declaration from .cu
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
}
