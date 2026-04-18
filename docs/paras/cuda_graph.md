# ParaS Dual EP/TP CUDA Graph Capture

## Overview

ParaS maintains two sets of CUDA graphs — one for EP mode and one for TP mode — so that mode switches at runtime don't require re-capturing graphs. Each set owns an **isolated** CUDA graph memory pool, and they are swapped via `paras_swap_cuda_graphs`.

All ParaS-specific CUDA graph logic lives in `python/sglang/srt/paras/paras_cuda_graph.py`. The `CudaGraphRunner` and `ModelRunner` classes have no ParaS-specific code beyond the memory-breakdown log hook.

## Capture Flow

### Init-time sequence

1. `CudaGraphRunner.__init__` captures EP graphs into the current `global_graph_memory_pool` (standard sglang path). This pool becomes EP's private pool (typically `(0, 1)`).
2. `model_runner.init_device_graphs` calls `paras_init_dual_cuda_graphs(model_runner)`.
3. Inside `paras_init_dual_cuda_graphs`:
   - **Save EP state**: `paras_save_cuda_graph_state(gr, "ep")` — snapshots `graphs`, `output_buffers`, `deepep_mode`, FlashInfer metadata, mode-dependent settings, **and the graph memory pool handle**.
   - **Switch to TP mode**: modify `server_args` (disable dp_attention, set dp_size=1, ep_size=1, moe_a2a_backend="none"), call `paras_comm_configure_tp()`, reconfigure `token_to_kv_pool`, `attn_backend`, and model weights.
   - **Clear live graph dicts**: `gr.graphs.clear()` + `gr.output_buffers.clear()`. EP state is preserved via the dict copies stored in `_paras_saved["ep"]`.
   - **Reset global pool**: `set_global_graph_memory_pool(None)` so the TP capture allocates a fresh pool via `graph_pool_handle()` on the first batch size.
   - **Capture TP graphs**: `paras_refresh_cuda_graph_settings(gr)` then `gr.capture()` — uses the same `capture_bs` list as EP but a **separate** CUDA graph memory pool (typically `(0, 2)`).
   - **Save TP state**: `paras_save_cuda_graph_state(gr, "tp")` — captures TP's distinct pool handle.
   - **Switch back to EP mode**: restore `server_args`, call `paras_comm_configure_ep()`, reconfigure pools/backend/model.
   - **Load EP state**: `paras_load_cuda_graph_state(gr, "ep")` — restores EP's graphs, buffers, settings, FlashInfer metadata, **and republishes EP's pool handle** via `set_global_graph_memory_pool`. Server starts in EP mode.

### Saved state per mode

Each mode's saved state (`runner._paras_saved[mode]`) contains:

| Key | Description |
|-----|-------------|
| `graphs` | `dict[int, CUDAGraph]` — batch size → captured graph |
| `output_buffers` | `dict[int, tensor]` — batch size → output buffer |
| `deepep_mode` | DeepEP captured mode (low_latency / normal) |
| `graph_memory_pool` | CUDA graph pool handle tuple (e.g. `(0, 1)` for EP, `(0, 2)` for TP) |
| FlashInfer metadata | `decode_cuda_graph_metadata`, `prefill_cuda_graph_metadata`, `draft_extend_cuda_graph_metadata` |
| Settings | `require_gathered_buffer`, `require_mlp_tp_gather`, `require_mlp_sync`, `require_attn_tp_gather`, `attn_tp_size`, `attn_tp_rank`, `tp_size`, `dp_size` |

## Runtime Swap

When `model_runner.paras_configure_tp()` or `paras_configure_ep()` is called (triggered by `/paras_configure_tp` or `/paras_configure_ep` HTTP endpoints):

1. Model weights are transferred (peer access).
2. KV cache pool and attention backend are reconfigured.
3. `paras_swap_cuda_graphs(model_runner, "tp" or "ep")` swaps the active graph set by calling `paras_load_cuda_graph_state`, which also republishes the active mode's graph pool handle globally.

No re-capture or re-instantiation happens — it's a Python dict reference swap. Measured switch time: ~80–90 ms total (including weight transfer ~67 ms) on 4×A100-80GB.

**Correctness**: EP generation output is byte-for-byte identical before and after any EP↔TP round-trip (verified on Qwen3-30B with greedy sampling).

## Fixed bugs (commit `8313fb698`)

### 1. EP state leaking into TP capture

Previously, `gr.graphs` / `gr.output_buffers` were not cleared before the TP capture loop. `capture_one_batch_size` overwrites same-keyed entries, but any batch size present only in EP's capture list would silently leak into TP's saved state.

### 2. EP and TP sharing one graph memory pool

Previously, both captures reused the same module-level `global_graph_memory_pool`. CUDA's graph-memory-pool allocator aliases blocks across different graphs in the same pool. Shared aliasing was fine for the raw footprint number (TP incremental was ~0.28 GB) but broke the invariant that each mode owns disjoint physical pages — pausing or releasing one mode's pool could corrupt the other mode's state.

Now:

- Each mode captures into its own pool (verified live: `ep_pool=(0, 1)`, `tp_pool=(0, 2)`, `pools_differ=True`).
- The steady-state isolation cost is ~+0.17 GB/GPU vs the old shared-pool approach (TP captures pay the full "TP isolated" cost instead of the 0.28 GB incremental cost).

## Memory Breakdown Instrumentation

Commit `a598b0b87` adds a per-capture breakdown emitted by `cuda_graph_runner.capture()`:

```
ParaS[mem-breakdown:pre-capture]   driver_used  torch_reserved
                                    graph_pool default_pool
                                    non_torch (deepep other)
                                    pools={(0,0)=..., (0,1)=...}
ParaS[mem-breakdown:post-capture]  (same fields)
ParaS[mem-breakdown:capture-delta] driver_used=+X torch_reserved=+Y
                                    graph_pool=+Z default_pool=+W
                                    non_torch=+V (deepep=+A other=+B)
```

Field definitions:

- **`driver_used`**: `cudaMemGetInfo()` used = `total - free`.
- **`torch_reserved`**: PyTorch caching allocator reserved segments (`memory_stats().reserved_bytes.all.current`).
- **`graph_pool`**: sum of `memory_snapshot()` segments with `segment_pool_id == (0, N)` where `N > 0`. This is the CUDA-graph-private pool — allocated by PyTorch's native caching allocator during capture, and intercepted 100% by TMS's `cudaMalloc` LD_PRELOAD hook when the capture runs under `tms.cuda_graph(tag=...)`.
- **`default_pool`**: `segment_pool_id == (0, 0)`. Normal caching allocator growth outside the capture-scoped private pool.
- **`non_torch`** = `driver_used − torch_reserved`. Memory held by things that bypass PyTorch's caching allocator. Decomposed into:
  - **`deepep`**: `DeepEPBuffer._buffer.num_nvl_bytes + num_rdma_bytes` — DeepEP's NVL/RDMA scratch buffers.
  - **`other`**: NCCL internals, cuBLAS workspaces, etc.

## Memory Footprint

All measurements on Qwen3-30B-A3B-Instruct-2507, 4×A100-80GB, `--mem-fraction-static 0.6`, `--tp-size 4 --dp-size 4 --ep-size 4 --enable-dp-attention --moe-a2a-backend deepep --enable-paras-moe --paras-tp-size 4`, `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256`.

### `--cuda-graph-max-bs 64` (12 batch sizes: 1,2,4,8,12,…,64)

| Bucket | EP capture Δ | TP capture Δ |
|---|---:|---:|
| `driver_used` | +3.314 GB | +0.283 GB |
| `torch_reserved` | +0.660 GB | +0.162 GB |
| `graph_pool` | **+0.521 GB** | **+0.023 GB** |
| `default_pool` | +0.139 GB | +0.139 GB |
| `non_torch` | +2.654 GB | +0.121 GB |
| &nbsp;&nbsp;&nbsp;↳ `deepep` | +0.566 GB | +0.000 GB |
| &nbsp;&nbsp;&nbsp;↳ `other` | +2.088 GB | +0.121 GB |
| Capture time | 4.86 s | ~3 s |

### `--cuda-graph-max-bs 256` (36 batch sizes: 1,2,4,8,…,248,256)

| Bucket | EP capture Δ | TP capture Δ |
|---|---:|---:|
| `driver_used` | +4.070 GB | +0.834 GB |
| `torch_reserved` | +0.916 GB | +0.475 GB |
| `graph_pool` | **+0.543 GB** | **+0.102 GB** |
| `default_pool` | +0.373 GB | +0.373 GB |
| `non_torch` | +3.154 GB | +0.359 GB |
| &nbsp;&nbsp;&nbsp;↳ `deepep` | +0.566 GB | +0.000 GB |
| &nbsp;&nbsp;&nbsp;↳ `other` | +2.588 GB | +0.359 GB |
| Capture time | 10.14 s | ~7 s |

After both captures at bs=256: final `avail mem = 19.5 GB/GPU`, 36 EP graphs, 36 TP graphs.

### Isolated plain-mode comparison (no ParaS)

| Config | Graph capture `mem usage` (sglang log) | `graph_pool` Δ | TMS `release(cuda_graph)` freed |
|---|---:|---:|---:|
| Plain TP=4, no MoE (`--tp-size 4`) | 0.45 GB | — | 0.062 GB |
| EP=4, dp=4, ep=4, deepep (no ParaS) | 3.43 GB | 0.521 GB | 0.521 GB |

TMS releases exactly the `graph_pool` bucket, 100% of it. The remaining "capture mem usage" reported by sglang's own log (3.43 GB for EP) is DeepEP + NCCL + cuBLAS — not graph-pool memory.

### What the buckets mean for offload

- **`graph_pool`** is the only bucket that is both capture-scoped **and** reliably reclaimable. A dual-tag TMS pause of the inactive mode would recover the inactive mode's entire `graph_pool` column — 0.02–0.10 GB if EP is active (TP paused), or 0.52–0.54 GB if TP is active (EP paused).
- **`deepep`** (0.566 GB) is constant; it depends on `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK`, `hidden_size`, and `num_experts`, **not** on `cuda_graph_max_bs`. It's live for EP's forward pass and cannot be offloaded without tearing down the DeepEP Buffer.
- **`default_pool`** growth (~0.14–0.37 GB) is cuBLAS workspace / allocator slack outside the capture region — shared across modes.
- **`other` non_torch** (~2–2.6 GB) is NCCL communicator state (ncclCommInitAll scratch, etc.) — process-global and per-communicator; not reclaimable without teardown.

### Scaling observations

- `graph_pool` scales weakly with `cuda_graph_max_bs`: 3× more batch sizes (12 → 36) added only 22 MB on EP and 79 MB on TP. CUDA's graph memory pool aggressively aliases intermediate allocations across batch sizes within the same pool.
- `non_torch` scales noticeably with `cuda_graph_max_bs`: +500 MB from bs=64 → bs=256 in EP, mostly lazy cuBLAS workspaces triggered by new kernel shapes.
- DeepEP buffer stays at 566 MB independent of `cuda_graph_max_bs`.

## Why `reset()/instantiate()` Doesn't Work

An alternative approach would prune inactive graph execs via `graph.reset()` and re-instantiate them via `graph.instantiate()` during mode switches, potentially saving memory. However, in PyTorch 2.8:

- `CUDAGraph(keep_graph=True)` only controls whether `capture_end()` auto-calls `instantiate()`.
- `reset()` destroys both the `cudaGraphExec_t` **and** the `cudaGraph_t` definition.
- After `reset()`, both `instantiate()` and `replay()` fail with `RuntimeError: capture_end() must have been called`.

Since the bulk of "capture cost" is DeepEP/NCCL state (not `graph_pool` memory), this path offers small savings for high complexity.

## Key Constraints

- **`--cuda-graph-max-bs` must fit DeepEP buffer**: The max captured batch size must not exceed `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK`. Otherwise EP graph capture fails with `x.size(0) <= num_max_dispatch_tokens_per_rank` assertion.
- **Batch sizes are shared**: `capture_bs` is computed once in `CudaGraphRunner.__init__` (under EP mode settings) and reused for TP capture. Both modes capture the same set of batch sizes.
- **`mem-fraction-static=0.75` will OOM**: Weight redistribution during ParaS configure requires headroom. Use 0.6 on A100-80GB.

## PyTorch allocator / TMS facts worth knowing

All citations are from PyTorch `077121cb0e919b7c329397e40187d11066abeb3f` and torch_memory_saver `d64a6394d1e09c613fab90260054cecc2684586d`.

- PyTorch's **native** caching allocator (SGLang's default — no `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` override) allocates graph-private pool segments via plain `cudaMalloc`, not `cudaMallocAsync` (`c10/cuda/CUDACachingAllocator.cpp:1199-1220`, `cudaMallocMaybeCapturing`).
- TMS's LD_PRELOAD `.so` exports exactly two overriding symbols: `cudaMalloc` and `cudaFree` (`nm -D` + `csrc/entrypoint.cpp:43-56`). No `cudaMallocAsync`, no `cuMemAlloc`, no `cudaMallocFromPoolAsync`.
- Inside `tms.cuda_graph(tag=..., enable_cpu_backup=True)`, new-segment `cudaMalloc` calls are replaced with `cuMemCreate + cuMemAddressReserve + cuMemMap` so the physical pages are pauseable (`csrc/core.cpp:13-59`).
- TMS catches **100% of the graph-private pool** because the pool starts empty on each capture — every allocation issues a new `cudaMalloc`, which the hook intercepts. (Cached block reuse via `get_free_block()` at `CUDACachingAllocator.cpp:1601-1623` only matters for the default pool, not the capture-scoped private pool.)

## Files modified (branch `paras_cudagraph`)

- `python/sglang/srt/paras/paras_cuda_graph.py` — pool isolation, save/load of pool handle, memory-breakdown helper + DeepEP buffer introspection.
- `python/sglang/srt/model_executor/cuda_graph_runner.py` — pre/post/delta memory breakdown logs around the capture loop.

Commits (local, unpushed):

- `8313fb698` `fix(paras): isolate EP and TP CUDA graph memory pools`
- `97be7748a` `chore(paras): add memory analysis logging to dual CUDA graph capture`
- `a598b0b87` `chore(paras): add graph-pool / non-torch memory breakdown instrumentation`
