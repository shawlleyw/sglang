#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

// Maximum number of GPUs in TP group (NVSwitch supports up to 8)
#define MAX_PEERS 8

// Transfer plan entry (struct-of-arrays fed from Python)
// The kernel is launched with one entry per destination rank.
// Each entry: copy 'size' bytes from local staging to peer GPU's EP buffer.

__global__ void peer_access_transfer_kernel(
    const char* __restrict__ src_base,   // Local staging buffer base address
    char* const* dst_bases,              // Array of peer buffer base addresses [MAX_PEERS]
    const int64_t* src_offsets,          // Offset from src_base for each entry
    const int64_t* dst_offsets,          // Offset from dst_bases[dst_rank] for each entry
    const int64_t* sizes,                // Bytes to copy for each entry
    const int32_t* dst_ranks,            // Which peer GPU to write to
    int num_entries
) {
    // Each block handles one entry; threads within the block copy bytes of that entry
    int entry_idx = blockIdx.x;
    if (entry_idx >= num_entries) return;

    const char* src = src_base + src_offsets[entry_idx];
    char* dst = dst_bases[dst_ranks[entry_idx]] + dst_offsets[entry_idx];
    int64_t n = sizes[entry_idx];

    // Use int4 (128-bit) vectorized copy for coalesced NVLink access
    int64_t n_int4 = n / 16;

    const int4* src4 = reinterpret_cast<const int4*>(src);
    int4* dst4 = reinterpret_cast<int4*>(dst);

    for (int64_t i = threadIdx.x; i < n_int4; i += blockDim.x) {
        dst4[i] = src4[i];
    }

    // Handle tail bytes
    if (threadIdx.x == 0) {
        for (int64_t i = n_int4 * 16; i < n; i++) {
            dst[i] = src[i];
        }
    }
}

// Host-side launch function
// dst_base_ptrs: array of MAX_PEERS pointers (int64_t cast to char*), already in device memory
void launch_peer_access_transfer(
    int64_t src_base_ptr,             // local staging base address
    int64_t* dst_base_ptrs,           // device array of peer buffer addresses [MAX_PEERS]
    const int64_t* src_offsets,       // device array
    const int64_t* dst_offsets,       // device array
    const int64_t* sizes,             // device array
    const int32_t* dst_ranks,         // device array
    int num_entries,
    cudaStream_t stream
) {
    if (num_entries == 0) return;

    // dst_bases as device pointer array
    char* const* dst_bases_typed = reinterpret_cast<char* const*>(dst_base_ptrs);

    int threads = 256;
    int blocks = num_entries;  // one block per entry

    peer_access_transfer_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const char*>(src_base_ptr),
        dst_bases_typed,
        src_offsets,
        dst_offsets,
        sizes,
        dst_ranks,
        num_entries
    );

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}
