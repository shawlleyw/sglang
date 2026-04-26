# ParaS Dual EP/TP CUDA Graph Capture

## Overview

ParaS maintains two sets of CUDA graphs — one for EP mode and one for TP mode — so that mode switches at runtime don't require re-capturing graphs. Each set owns an **isolated** CUDA graph memory pool, and they are swapped via `paras_swap_cuda_graphs`.

All ParaS-specific CUDA graph logic lives in `python/sglang/srt/paras/paras_cuda_graph.py`. The `CudaGraphRunner` and `ModelRunner` classes have no ParaS-specific code beyond the memory-breakdown log hook.

Hybrid sliding-window-attention models now use the same dual-pool capture design. The earlier startup restriction that rejected SWA + ParaS under CUDA graphs was removed once the EP↔TP hot-switch path was verified to run outside the captured region.

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

Commits `a598b0b87`, `905091d5c`, and `6ff7c38d1` add a per-capture breakdown emitted by `cuda_graph_runner.capture()`:

```
ParaS[mem-breakdown:pre-capture]   driver_used  torch_reserved
                                    graph_pool default_pool
                                    non_torch (deepep_buf  deepep_ws
                                               nvshmem  nccl_est  other)
                                    pools={(0,0)=..., (0,1)=...}
ParaS[mem-breakdown:post-capture]  (same fields)
ParaS[mem-breakdown:capture-delta] driver_used=+X torch_reserved=+Y
                                    graph_pool=+Z default_pool=+W
                                    non_torch=+V (deepep_buf=+A
                                                  deepep_ws=+B
                                                  nvshmem=+C
                                                  nccl_est=+D
                                                  other=+E)
```

Field definitions:

- **`driver_used`**: `cudaMemGetInfo()` used = `total - free`.
- **`torch_reserved`**: PyTorch caching allocator reserved segments (`memory_stats().reserved_bytes.all.current`).
- **`graph_pool`**: sum of `memory_snapshot()` segments with `segment_pool_id == (0, N)` where `N > 0`. This is the CUDA-graph-private pool — allocated by PyTorch's native caching allocator during capture, and intercepted 100% by TMS's `cudaMalloc` LD_PRELOAD hook when the capture runs under `tms.cuda_graph(tag=...)`.
- **`default_pool`**: `segment_pool_id == (0, 0)`. Normal caching allocator growth outside the capture-scoped private pool.
- **`non_torch`** = `driver_used − torch_reserved`. Memory held by things that bypass PyTorch's caching allocator. Decomposed into:
  - **`deepep_buf`**: `DeepEPBuffer._buffer.num_nvl_bytes + num_rdma_bytes` — DeepEP's NVL + RDMA scratch buffers (measured directly from the live `deep_ep.Buffer` object).
  - **`deepep_ws`**: 32 MiB hardcoded workspace per live `DeepEPBuffer._buffer`. Source: `deep_ep/csrc/deep_ep.cpp:192` (`NUM_WORKSPACE_BYTES = 32 * 1024 * 1024`, plain `cudaMalloc`).
  - **`nvshmem`**: `NVSHMEM_SYMMETRIC_SIZE` (env, default 1 GiB per PE). Counted only when DeepEP triggered `internode::init()` because either `low_latency_mode=True` or `num_rdma_ranks > 1`. This is the *reservation* — physical commit happens in `NVSHMEM_CUMEM_GRANULARITY` chunks (DeepEP sets 512 MiB).
  - **`nccl_est`**: `len(live NCCL communicators) × 144 MiB`. An **upper-bound estimate** based on `≈9 MiB × 8 channels × 2 (send+recv)` per communicator (NCCL 2.27.7 defaults). Live comms counted: `_WORLD`, `_TP`, `_PP`, `_MOE_EP`, `_MOE_TP`, `_PDMUX_PREFILL_TP_GROUP`. Note: on our setup **both EP and TP configs create 5 NCCL comms** (because sglang initializes `_MOE_EP`/`_MOE_TP` as trivial 1-rank groups in the TP-only case).
  - **`other`**: residual — dominated by NCCL CUDA-graph-capture VMM scratch (NCCL 2.19+ with `NCCL_CUMEM_ENABLE=1` allocates per-graph `cuMemCreate`/`cuMemMap` buffers, see NCCL issue #1234), plus lazy cuBLAS workspaces, plus any unaccounted driver state.

## Memory Footprint

All measurements on Qwen3-30B-A3B-Instruct-2507, 4×A100-80GB, `--mem-fraction-static 0.6`, `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256`, `--cuda-graph-max-bs 256` (36 batch sizes), DP0 unless noted. Captured with the instrumentation in commit `6ff7c38d1`. Raw logs in `artifacts/{ep,tp,paras}_bs256_v2.log`.

### Capture delta (EP vs TP, no ParaS)

| Bucket | Baseline EP Δ | Baseline TP Δ | EP − TP |
|---|---:|---:|---:|
| `driver_used` | +4.000 GB | +0.924 GB | **+3.076 GB** |
| `torch_reserved` | +0.916 GB | +0.549 GB | +0.367 GB |
| &nbsp;&nbsp;↳ `graph_pool` | **+0.543 GB** | **+0.176 GB** | +0.367 GB |
| &nbsp;&nbsp;↳ `default_pool` | +0.373 GB | +0.373 GB | 0 |
| `non_torch` | +3.084 GB | +0.375 GB | **+2.709 GB** |
| &nbsp;&nbsp;↳ `deepep_buf` | +0.566 GB | 0 | +0.566 GB |
| &nbsp;&nbsp;↳ `deepep_ws` | +0.031 GB | 0 | +0.031 GB |
| &nbsp;&nbsp;↳ `nvshmem` | +1.000 GB | 0 | +1.000 GB |
| &nbsp;&nbsp;↳ `nccl_est` | 0 | 0 | 0 |
| &nbsp;&nbsp;↳ `other` | +1.486 GB | +0.375 GB | **+1.111 GB** |
| Capture time | 15.2 s | 7.8 s | +7.4 s |

Of the 2.709 GB EP-over-TP capture penalty, **1.597 GB is attributed** (`deepep_buf` + `deepep_ws` + `nvshmem`). The remaining **1.111 GB in `other`** is not yet decomposed and is the main open investigation item (see "Open questions" below).

### Steady-state footprint (server ready for serving)

| Component | Baseline TP | Baseline EP | **ParaS (EP mode)** |
|---|---:|---:|---:|
| `torch_reserved` | 50.87 GB | 48.82 GB | 52.14 GB |
| &nbsp;&nbsp;↳ default_pool (weights + KV + workspaces) | 50.69 GB | 48.27 GB | 51.50 GB |
| &nbsp;&nbsp;↳ graph_pool (captured cuda graphs) | 0.18 GB | 0.54 GB | 0.65 GB (EP 0.54 + TP 0.10) |
| `non_torch` | 1.80 GB | 4.51 GB | 7.30 GB |
| &nbsp;&nbsp;↳ deepep_buf | 0 | 0.57 GB | 0.57 GB |
| &nbsp;&nbsp;↳ deepep_ws | 0 | 0.03 GB | 0.03 GB |
| &nbsp;&nbsp;↳ nvshmem heap | 0 | 1.00 GB | 1.00 GB |
| &nbsp;&nbsp;↳ nccl_est | 0.70 GB | 0.70 GB | 0.70 GB |
| &nbsp;&nbsp;↳ other | 1.10 GB | 2.21 GB | ~5.00 GB |
| **driver_used total** | **52.67 GB** | **53.33 GB** | **59.45 GB** |

ParaS overhead vs baseline EP ≈ **+6.1 GB/GPU**, split as:

- **+0.10 GB** second CUDA graph pool `(0, 2)` for TP graphs — the intended cost of dual capture.
- **+3.23 GB** ParaS's own pre-capture state (peer-access IPC buffers, `ParaSMemoryManager`, fused peer-access init). Visible in pre-capture `non_torch`: ParaS starts at 3.86 GB vs baseline EP's 1.42 GB.
- **~+2.7 GB** extra `other` non_torch, most likely from running two forward-pass shapes through capture (more distinct kernel shapes → more NCCL CUDA-graph VMM scratch + cuBLAS workspaces).

### Isolated plain-mode comparison (sanity check)

| Config | sglang's `Capture cuda graph end mem usage` (log) | Our `graph_pool` Δ | TMS `release(cuda_graph)` freed |
|---|---:|---:|---:|
| Plain TP=4, no MoE (`--tp-size 4`) | 0.45 GB | 0.18 GB | 0.062 GB |
| EP=4, dp=4, ep=4, deepep (no ParaS) | 3.43 GB | 0.54 GB | 0.521 GB |

TMS releases exactly the `graph_pool` bucket. The "capture mem usage" reported by sglang's own log is the driver-level delta and contains DeepEP + NVSHMEM + NCCL + cuBLAS in addition to `graph_pool`.

### What the buckets mean for offload

- **`graph_pool`** is the only bucket that is both capture-scoped **and** reliably reclaimable. A dual-tag TMS pause of the inactive mode would recover the inactive mode's entire `graph_pool` column — 0.10 GB if EP is active (TP paused), or 0.54 GB if TP is active (EP paused).
- **`deepep_buf`** (0.57 GB) is constant per process. Depends on `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK`, `hidden_size`, and `num_experts`, not on `cuda_graph_max_bs`. Live for EP's forward pass; can only be reclaimed by destroying the DeepEP Buffer.
- **`deepep_ws`** (0.03 GB) is hardcoded 32 MiB per Buffer.
- **`nvshmem`** (1.00 GB reserved) is NVSHMEM's symmetric heap, reserved at `internode::init()`. Can be trimmed via `NVSHMEM_SYMMETRIC_SIZE=<bytes>` env override (minimum rounded up to `NVSHMEM_CUMEM_GRANULARITY` = 512 MiB).
- **`nccl_est`** (0.70 GB estimate) is a per-communicator upper bound; same on EP and TP in our configs.
- **`other`** (1–5 GB depending on config) is the main residual. Suspected dominant contributors:
  - NCCL CUDA-graph VMM scratch (NCCL 2.19+ with `NCCL_CUMEM_ENABLE=1` allocates `cuMemCreate`/`cuMemMap` per captured graph — see NCCL issue #1234; disabling via `NCCL_CUMEM_ENABLE=0` may reduce this substantially).
  - Lazy cuBLAS/cuBLASLt workspaces for kernel shapes encountered during capture.
  - Any ParaS peer-access IPC allocations not tracked by my instrumentation.

### Scaling observations

- `graph_pool` scales weakly with `cuda_graph_max_bs`: 3× more batch sizes (12 → 36) added only 22 MB on EP and 79 MB on TP. CUDA's graph memory pool aggressively aliases intermediate allocations across batch sizes within the same pool.
- `non_torch` scales noticeably with `cuda_graph_max_bs`: roughly +500 MB on EP going from bs=64 → bs=256, all inside `other` (deepep/nvshmem/nccl_est are fixed).
- `deepep_buf`, `deepep_ws`, `nvshmem`, `nccl_est` are all independent of `cuda_graph_max_bs`.

## Open questions

- **What exactly is in `other` non_torch?** At bs=256, baseline EP has +1.486 GB of `other` on the capture delta alone, ParaS has ~5 GB steady state. Suspected: NCCL graph-capture VMM scratch (testable via `NCCL_CUMEM_ENABLE=0`), cuBLAS workspaces (testable via `CUBLAS_WORKSPACE_CONFIG`), ParaS peer-access buffers (testable by disabling the `fused peer access pre-initialized` path). Needs targeted experiments.
- **Is the `nvshmem` 1 GiB really physically committed?** My accounting assumes the full `NVSHMEM_SYMMETRIC_SIZE` is physical. If NVSHMEM commits less (in 512 MiB chunks), the reported `other` is under-counted.
- **Are all 5 "live" NCCL comms actually allocating 144 MiB each?** The per-comm 144 MiB estimate is an upper bound; real usage depends on NCCL's channel selection and whether `send`/`recv` was exercised. The 0.70 GB `nccl_est` is the same for EP and TP in our logs, so it cancels out in the EP-vs-TP delta, but its absolute value is not verified.

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

- `python/sglang/srt/paras/paras_cuda_graph.py` — pool isolation, save/load of pool handle, memory-breakdown helper with DeepEP / NVSHMEM / NCCL decomposition.
- `python/sglang/srt/model_executor/cuda_graph_runner.py` — pre/post/delta memory breakdown logs around the capture loop.

## Raw logs

Reference logs saved under `artifacts/` on the `paras_cudagraph` branch:

| File | Config |
|---|---|
| `paras_bs256.log` | ParaS dual EP+TP capture, initial run (no NVSHMEM/NCCL decomposition) |
| `ep_bs256.log` | Baseline EP (no ParaS), initial run |
| `tp_bs256.log` | Baseline TP (no MoE), initial run |
| `paras_bs256_v2.log` | (not re-run after v2 decomposition; refer to `ep_bs256_v2.log` for bucket semantics) |
| `ep_bs256_v2.log` | Baseline EP with full decomposition (commit `6ff7c38d1` instrumentation) |
| `tp_bs256_v2.log` | Baseline TP with full decomposition |

## Commits (local, unpushed)

- `8313fb698` `fix(paras): isolate EP and TP CUDA graph memory pools`
- `97be7748a` `chore(paras): add memory analysis logging to dual CUDA graph capture`
- `a598b0b87` `chore(paras): add graph-pool / non-torch memory breakdown instrumentation`
- `905091d5c` `chore(paras): break out DeepEP buffer size from non_torch bucket`
- `83a2c6bc6` `docs(paras): rewrite cuda_graph.md with pool isolation + memory breakdown`
- `6ff7c38d1` `chore(paras): decompose non_torch into deepep/nvshmem/nccl buckets`
