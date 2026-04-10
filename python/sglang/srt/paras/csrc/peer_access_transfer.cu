#include <cuda_runtime.h>
#include <stdint.h>

// Stub: minimal kernel to verify build system works
__global__ void peer_access_stub_kernel(const char* src, char* dst, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = src[idx];
    }
}

// Host-side stub launch wrapper
void launch_peer_access_stub(const char* src, char* dst, size_t size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    peer_access_stub_kernel<<<blocks, threads>>>(src, dst, size);
    cudaDeviceSynchronize();
}
