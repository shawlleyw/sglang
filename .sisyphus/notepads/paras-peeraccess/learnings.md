# ParaS Peer Access — Learnings

## Architecture Discoveries

### paras_decoder_layer.py has MLP wrapper methods
- `paras_configure_tp_mlp_all_gather(stream, handles, async_op)` → calls `self.mlp.paras_configure_tp_all_gather()`
- `paras_configure_tp_mlp_all_to_all(stream, handles)` → calls `self.mlp.paras_configure_tp_all_to_all()`
- **IMPORTANT**: We need to add `paras_configure_tp_mlp_peer_access(stream, handles)` wrapper too

### paras_model.py method selection
- Current interface: `paras_configure_tp(paras_tp_size, paras_tp_rank, overlap: bool = False)`
- Need to change to: `paras_configure_tp(paras_tp_size, paras_tp_rank, method: str = "overlap")` or keep backward-compatible

### DP=1 data flow (critical for peer access design)
- `paras_configure_tp_all_gather` is NO-OP for DP=1: sets `self.w13_ep_gathered = w13_ep` (just a view, no copy)
- `paras_configure_tp_all_to_all` for DP=1:
  - Uses `staging.w13_a` for permuted copy (source of all_to_all)
  - `staging.w13_b` is UNUSED for DP=1 → free for double-buffering in peer access
  - w13: view as `(E, 2, tp_size, I'*H)` → permute(2,0,1,3) → copy to staging_a as `(tp_size, E, 2, I'*H)` → all_to_all_single writes back into the same EP buffer
  - Post-processing (DP=1): just `get_view_as()` to reinterpret EP buffer as TP shape, no actual copy

### EP/TP buffer alias (CRITICAL)
- TP weights at init: `get_view_as(ep_w13_name, tp_shape)` → same bytes as EP buffer
- After all_to_all: EP buffer bytes contain TP data → `get_view_as()` just reinterprets
- Peer access: must permute to staging FIRST (staging_a or staging_b), barrier across ranks, then peer write from staging to remote EP buffer
- Remote GPU's EP buffer becomes overwritten with TP data for that rank's partition

### paras_configure_tp() at block level (line 300)
- Called AFTER weight transfer is complete
- Just sets `self.parallelism_config = "tp"`, `self.tp_size = paras_tp_size`, `self.experts = self.tp_experts`
- Already done at init time: `tp_experts.w13_weight = get_view_as(ep_name, tp_shape)` so TP view is ready

## Build System
- sgl_paras conda env exists at /home/shaoyuw/miniconda3/envs/sgl_paras
- Use: `conda run -n sgl_paras ...` or activate first
- Standalone extension: `setup.py` with `CUDAExtension` in `python/sglang/srt/paras/csrc/`

## Branch
- Starting from: `paras_memmgr`
- New branch: `paras_peeraccess`

## Task 1: CUDA Extension Build Scaffolding (COMPLETED)

### Build System Setup
- Created `python/sglang/srt/paras/csrc/` directory
- **setup.py**: Uses `torch.utils.cpp_extension.CUDAExtension`
  - Deferred torch import to avoid build isolation issues
  - nvcc flags: `-O3 -arch=sm_90 --expt-relaxed-constexpr` (H100 support)
  - cxx flags: `-O3`
- **pyproject.toml**: Specifies torch as build requirement
  - Enables `--no-build-isolation` pip install

### Stub Implementation
- **peer_access_transfer.cu**: Minimal kernel for build verification
  - `peer_access_stub_kernel`: Simple copy kernel (idx < size)
  - `launch_peer_access_stub`: Host wrapper with cudaDeviceSynchronize
- **binding.cpp**: pybind11 stub module
  - `PYBIND11_MODULE(paras_peer_access_cuda, m)`
  - `stub_hello()` function for testing

### Build & Test Results
- Build command: `conda run -n sgl_paras python setup.py build_ext --inplace`
- Output: `paras_peer_access_cuda.cpython-312-x86_64-linux-gnu.so` (170KB)
- Import test: `import torch; import paras_peer_access_cuda` ✓
- **Note**: torch must be imported first to set up CUDA library paths (libc10.so, libtorch_cpu.so)

### Commit
- `89c2126c0 feat(paras): add CUDA extension build scaffolding for peer access`
- Files: setup.py, pyproject.toml, peer_access_transfer.cu, binding.cpp

### Next Steps (Task 2+)
- Add actual peer access kernel logic to peer_access_transfer.cu
- Implement pybind11 bindings for kernel launch functions
- Add DP>1 support (currently stub is DP=1 only)

## Task 2: Peer Access Module (COMPLETED)

### Module: `python/sglang/srt/paras/peer_access.py`
- `check_peer_access_available(device_ids)` — ctypes call to `cudaDeviceCanAccessPeer` for all pairs
- `enable_peer_access(device_ids)` — calls `cudaDeviceEnablePeerAccess` via ctypes, skips already-enabled
- `exchange_buffer_addresses(local_buffer_ptr, tp_group, world_size)` — `dist.all_gather_into_tensor` of int64 pointers
- `PeerAccessContext` dataclass — holds peer_addresses, peer_access_enabled, tp_group, tp_size
- `init_peer_access(manager, tp_group, tp_size)` — top-level convenience combining enable + exchange

### Key Decisions
- Uses `ctypes.CDLL("libcudart.so")` directly instead of torch CUDA API for peer access control
- `manager.materialized` property exists (line 370 of paras_memory_manager.py)
- `manager._buffer.data_ptr()` gives int address after materialize
- TP group's `.device_group` attribute used for dist operations (seen in paras_parallel_state.py line 107)
- Peer check on this machine: GPUs 0↔1 support peer access ✓

### Evidence
- Import test: `.sisyphus/evidence/task-2-import.txt` → OK
- Peer check: `.sisyphus/evidence/task-2-peer-check.txt` → `Peer 0->1: True`
- NOT committed (will commit with T3)

## Task 4: CUDA Peer Access Transfer Kernel (COMPLETED)

### Kernel Implementation (`peer_access_transfer.cu`)
- `peer_access_transfer_kernel`: 1 block per transfer entry, 256 threads
- int4 (128-bit) vectorized copy for coalesced NVLink access
- Tail bytes handled by thread 0
- `launch_peer_access_transfer`: host wrapper, no sync (caller manages stream)

### C++ Binding (`binding.cpp`)
- `launch_peer_access_transfer_py`: torch::Tensor → raw pointer conversion
- TORCH_CHECK on all GPU tensors
- stream_ptr default=0 (default CUDA stream)

### Python Wrapper (`peer_access.py`)
- `peer_access_transfer(src_base_ptr, dst_base_ptrs_tensor, plan, stream)` added
- Lazy import of `paras_peer_access_cuda` with clear error message
- `stream_ptr = stream.cuda_stream` for non-None stream

### Build System Fix
- Changed `-arch=sm_90` to `-gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_90,code=sm_90`
- A100 = sm_80, H100 = sm_90
- Added `#include <stdio.h>` for printf in .cu file

### Evidence
- `.sisyphus/evidence/task-4-kernel-launch.txt` → "2-GPU peer access kernel test PASSED!"
- Module exports: `['__doc__', '__file__', ..., 'launch_peer_access_transfer']`

### Commit
- `0bcd11a3a feat(paras): implement CUDA peer access transfer kernel`
- Files: peer_access_transfer.cu, binding.cpp, setup.py, peer_access.py

## Task 5+6: MoeBlock + Model Mixin Integration (COMPLETED)

### T5: MoeBlock & DecoderLayer peer_access methods
- `ParaSMoeBlockMixin.paras_configure_tp_peer_access(peer_ctx, transfer_plans, packed_plans, staging_suffix, stream)` — 5-phase: permute→barrier→peer write→barrier→reinterpret
- `ParaSDecoderLayerMixin.paras_configure_tp_mlp_peer_access(peer_ctx, transfer_plans, packed_plans, staging_suffix, stream, handles)` — wrapper that waits handles then delegates to mlp

### T6: Model-level peer_access integration
- `ParaSModelMixin.paras_configure_tp_peer_access(paras_tp_size, paras_tp_rank)` — sequential per-layer: attn + peer_access_mlp + tp_config
- `paras_configure_tp()` updated: `method` param ("naive"/"overlap"/"peer_access"), backward compat with `overlap: bool`
- Fixed `paras_func` decorator to use `@functools.wraps(func)` so `inspect.getsource` returns original source

### Commit
- `eaffa5b39 feat(paras): integrate peer_access method into MoeBlock and Model mixins`
- Files: paras_moe_block.py, paras_decoder_layer.py, paras_model.py, utils.py

## Task 7+8: 4-GPU Comparison Test & Benchmark (COMPLETED)

### Test: `test/srt/test_paras_peer_access.py`
- Runs with `torchrun --nproc_per_node=4`
- Verifies bitwise match between NCCL all_to_all and peer access paths for all layers/weights/ranks
- Includes latency benchmark with `--benchmark` flag

### Critical Discovery: CUDA IPC Required for Multi-Process Peer Access
- `exchange_buffer_addresses()` exchanges raw `data_ptr()` values which are process-local virtual addresses
- In multi-process setup (torchrun), raw addresses are NOT valid across processes
- Solution: Use CUDA IPC (`cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle`) to create cross-process mappable handles
- IPC handle = 64 bytes (`cudaIpcMemHandle_t`), exchanged via `dist.all_gather_into_tensor`
- Flag: `cudaIpcMemLazyEnablePeerAccess = 1`
- ctypes gotcha: use `c_ubyte * 64` not `c_char * 64` (c_char stops at null bytes)

### Parallel State Bootstrap (No Full sglang Init Needed)
- Import of `ParaSMoeBlockMixin` triggers `fused_moe_triton/layer.py` module-level code
- That code calls `get_moe_runner_backend()` → `log_info_on_rank0()` → `get_tensor_model_parallel_rank()`
- Must set `sglang.srt.distributed.parallel_state._TP` BEFORE importing the mixin
- Minimal mock needs: `.device_group`, `.world_size`, `.device`, `.rank_in_group`, `.rank`, `.local_rank`
- DP=1 all_gather is no-op: just set `mixin.w13_ep_gathered` = EP buffer view directly

### Benchmark Results (A100 4-GPU, 16 experts, hidden=512, inter=512)
- NCCL: avg=2.0ms, min=1.5ms
- Peer Access: avg=2.2ms, min=2.1ms
- Note: Small test dimensions; real-world Qwen3-30B should show different characteristics

### Evidence
- `.sisyphus/evidence/task-7-comparison-test.txt` → All 4 layers × 2 weights × 4 ranks bitwise match
- `.sisyphus/evidence/task-8-benchmark.txt` → Timing numbers
