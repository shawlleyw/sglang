# Exploration Notes: KV Cache Peer Access Development

These notes document discoveries, bugs found, and design decisions made during the development of NVLink peer access for KV cache transfer (April 2026).

## 1. gather_kv_and_permute Dimension Ordering Bug

**Discovery**: The original `gather_kv_and_permute` used `.permute(2, 0, 1, 3)` producing layout `[heads, KV, tokens, dim]`. After `all_to_all` splits by head, each received chunk was `[K_all_tokens, V_all_tokens]` (KV-grouped). The subsequent `permute_and_scatter_kv` did `.view(total_tokens, 2, heads, dim)` which assumed token-interleaved layout `[t0_K, t0_V, t1_K, t1_V, ...]`.

**Impact**: For any sender with N>1 tokens, K and V data were scrambled. Token 0's "V" was actually token 1's K. This affected ALL cases (replicated or not), not just the replication path.

**Fix**: Change to `.permute(2, 1, 0, 3)` → layout `[heads, tokens, KV, dim]`. Each head's chunk is now token-interleaved, and concatenating received chunks gives `[total_tokens, KV, heads, dim]`.

**Verification**: Unit test with explicit value tracing confirmed the mismatch. The bug was latent in the original code — possibly masked by specific deployment conditions (e.g., single-token-per-rank scenarios).

## 2. cudaDeviceEnablePeerAccess Creates ~2.9 GB Overhead

**Discovery**: nvidia-smi showed 64 process entries (8 procs × 8 GPUs) during our KV peer access test, with each peer context consuming ~416 MiB. This was ~2.9 GB of wasted GPU memory per rank.

**Root cause**: `peer_access.py:enable_peer_access()` called `cudaDeviceEnablePeerAccess(peer_device, 0)` for every peer GPU before the IPC handle exchange. This created full CUDA contexts on each peer.

**Key finding**: `cudaIpcOpenMemHandle` with `cudaIpcMemLazyEnablePeerAccess` flag is SUFFICIENT for NVLink stores — the explicit `EnablePeerAccess` call is redundant and harmful.

**Evidence**: DeepEP (`deep_ep.cpp Buffer::sync()`) uses `cudaIpcOpenMemHandle` with the lazy flag and never calls `cudaDeviceEnablePeerAccess`. nvidia-smi confirms 1 process per GPU, 0 cross-GPU contexts.

**Fix**: Remove the `enable_peer_access()` call from `init_peer_access()`. 1-line change, ~2.9 GB saved.

**Clarification on IPC APIs**:
- `cudaIpcGetMemHandle(handle, ptr)` — EXPORT: packages metadata about a local allocation into a serializable 64-byte handle. No cross-GPU overhead.
- `cudaIpcOpenMemHandle(&ptr, handle, flags)` — IMPORT: maps peer GPU memory into this process's virtual address space. With the lazy flag, no full CUDA context is created on the peer GPU.
- `cudaDeviceEnablePeerAccess(peer, flags)` — creates a full bidirectional peer access context on the peer GPU (~416 MiB). NOT needed when using IPC handles with the lazy flag.

## 3. CUDA_MODULE_LOADING=LAZY Has No Effect

**Tested**: Setting `CUDA_MODULE_LOADING=LAZY` did not reduce the ~416 MiB per-context overhead. The 416 MiB is the minimum CUDA context allocation on A100, not module loading overhead.

**Conclusion**: The only way to eliminate the context overhead is to not create the context in the first place (i.e., remove `cudaDeviceEnablePeerAccess`).

## 4. NCCL Head Replication: Option A vs Option B

When `num_kv_heads < tp_size`, the `all_to_all` needs `tp_size` chunks but only `num_heads` heads of data exist.

**Option A (chosen): `repeat_interleave` before `all_to_all`**
- Duplicate heads: `[4 heads] → [8 virtual heads]` via `repeat_interleave(2, dim=0)`
- `all_to_all` with 8 uniform chunks
- Pros: +3 lines, no sub-groups, single collective
- Cons: sends duplicate data (same head to replicated peers)

**Option B (documented, not implemented): sub-head `all_to_all` + intra-group `all_gather`**
- Split `head_dim` in half: `[4, N, 2, 128]` → `[8, N, 2, 64]` (8 sub-heads)
- `all_to_all` of sub-heads (no duplication)
- `all_gather` within replication group `{0,1}`, `{2,3}`, etc. to reconstruct full `head_dim`
- Pros: no redundant NVLink traffic, scales for large replication factors
- Cons: 2 collectives, needs `dist.new_group`, more complex reshaping

For replication_factor=2, total NVLink traffic is comparable. Option B becomes clearly better for replication_factor≥4 (e.g., 2 heads / 8 GPUs).

## 5. NVSHMEM as Future Alternative

**Finding**: DeepEP's low-latency path uses NVSHMEM symmetric heap (`nvshmem_align` + `nvshmemi_get_p2p_ptr`) for zero-context-overhead peer access. nvidia-smi confirmed exactly 8 process entries (1 per GPU) with 8.06 GB each, achieving 72.5 GB/s NVLink bandwidth.

**Migration assessment**: MEDIUM-LARGE effort (3-4 weeks). Key blockers:
- NVSHMEM symmetric heap requires all ranks to allocate same size
- `torch.empty()` must be replaced with `nvshmem_align()` + custom tensor wrapping
- Must coordinate with DeepEP's existing NVSHMEM init (can't init twice per process)
- NVSHMEM memory isn't managed by torch's garbage collector

**Current status**: NOT NEEDED. Removing `cudaDeviceEnablePeerAccess` already eliminated the ~2.9 GB overhead. NVSHMEM migration would be warranted if we find other reasons to adopt it (e.g., multi-node peer access).

## 6. DeepEP Normal Mode CUDA IPC Behavior

**Tested**: Ran `bench_one_batch` with DeepEP `auto` mode (normal for prefill, LL for decode) on 8×A100 with Qwen3-30B-A3B. nvidia-smi showed exactly 8 process entries — 1 per GPU, 0 cross-GPU contexts.

**Analysis**: DeepEP's `Buffer::sync()` calls `cudaIpcOpenMemHandle` for all NVL peers (confirmed at `deep_ep.cpp:341`). However, it never calls `cudaDeviceEnablePeerAccess`. The lazy IPC flag is sufficient for the normal dispatch kernel's sender-push NVLink writes.

**Implication**: When ParaS peer access is used alongside DeepEP in the same system, there is NO double context overhead — CUDA contexts are per-process-per-GPU, and neither system creates unnecessary contexts.
