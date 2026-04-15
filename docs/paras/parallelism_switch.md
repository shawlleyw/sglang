# ParaS: Runtime EP→TP Parallelism Switching

## Why Switch From EP to TP?

Mixture-of-Experts (MoE) models can be served with two parallelism strategies:

- **Expert Parallelism (EP)**: Each GPU owns a subset of experts. Tokens are routed across GPUs via all-to-all dispatch. Each GPU runs attention on its local batch only (DP attention).
- **Tensor Parallelism (TP)**: Each GPU owns a shard of every expert (and every attention head). All GPUs process all tokens together, synchronized by AllReduce.

Neither strategy is universally better. Their relative performance depends on batch size:

| Batch Size | Winner | Why |
|------------|--------|-----|
| Small (≤512 equiv) | **TP** | DeepEP dispatch has ~5ms fixed overhead; TP AllReduce is <1ms for small tensors. EP's extra kernel launches dominate. |
| Large (≥1024 equiv) | **EP** | TP AllReduce grows linearly with batch size. EP's per-rank data is batch/dp_size — 8× less attention, norm, and elementwise work. EP's full-width GEMM (N=intermediate_size) achieves better tensor core utilization than TP's narrow GEMM (N=intermediate_size/tp_size). |

Reference measurements (Qwen3-30B-A3B, 8×A100, CUDA graphs):

| Equiv Batch | EP / TP Throughput |
|-------------|-------------------|
| 8 | 0.42× (TP wins) |
| 64 | 0.49× |
| 512 | 0.76× |
| 1024 | ~1.0× (crossover) |
| 2048 | **1.50×** (EP wins) |

In a real serving system, batch sizes fluctuate. During off-peak hours, the batch is small and TP is faster. During peak traffic, the batch is large and EP is faster. **ParaS enables the system to start in EP mode (optimal for prefill and large batches) and switch to TP mode at runtime when the workload favors it** — without restarting the server, reloading weights, or dropping requests.

## What Needs to Happen During the Switch

The EP→TP switch must transform the entire model's runtime state from EP layout to TP layout:

### 1. MoE Weight Redistribution

In EP mode, each GPU holds `num_experts / ep_size` complete experts. In TP mode, each GPU holds ALL experts but only a shard of each expert's weight matrices (sliced along the intermediate dimension).

**Data movement**: Every GPU must send its local experts' weight slices to every other GPU, and receive slices from every other GPU. For Qwen3-30B-A3B with 48 layers: ~10.4 GB of NVLink traffic per GPU.

### 2. KV Cache Redistribution

In EP mode, each GPU stores KV cache for all `num_kv_heads` attention heads but only for its local tokens (the tokens it processed via DP attention). In TP mode, each GPU stores KV cache for `num_kv_heads / tp_size` heads but for ALL tokens across all GPUs.

**Data movement**: Each GPU splits its local KV heads across peers (head-splitting), while collecting other GPUs' head shards for their tokens. For in-flight requests, this is critical — the KV cache must be transferred correctly or the model produces garbage on the next decode step.

### 3. Attention Reconfiguration

The attention layer must switch from DP attention (local QKV projection with full heads) to TP attention (sharded QKV projection with split heads). This involves:
- Slicing the full QKV weight to the TP shard
- Updating the attention backend's metadata (num_heads, req_to_token mapping)
- Resizing the KV pool to the TP token capacity

### 4. Request State Migration

Active requests' metadata (sequence lengths, token indices, sampling state) must be globally coordinated. Each GPU's local request state becomes a global shared state.

## How ParaS Achieves It

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   ParaS EP→TP Switch                     │
│                                                         │
│  1. Scheduler pauses, drains active batch                │
│  2. Gather request metadata across all ranks             │
│                                                         │
│  ┌─────────────────────────────────────────────────────┐ │
│  │           KV Cache Transfer (~34ms)                 │ │
│  │  peer_access: fused K+V kernel → NVLink direct     │ │
│  │  OR nccl: gather_kv_and_permute → all_to_all       │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                         │
│  ┌─────────────────────────────────────────────────────┐ │
│  │          Weight Transfer (~61ms)                     │ │
│  │  peer_access: fused w13/w2 kernels → NVLink direct  │ │
│  │  OR nccl: naive/overlap all_to_all                  │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                         │
│  3. Attention reconfiguration (QKV slice, backend update)│
│  4. Resume scheduling in TP mode                         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### The Unified Memory Manager

The foundation of fast switching is the Unified Memory Manager (`ParaSMemoryManager`). It allocates ALL persistent GPU memory — expert weights, attention weights, KV cache — in a single contiguous buffer at model init time.

**Key insight**: EP and TP layouts use the **same total bytes** per layer. An expert with shape `(E_local, 2I, H)` in EP becomes `(E_total, 2I/tp, H)` in TP — same bytes, different interpretation. The switch overwrites the same physical memory with the TP layout, avoiding any allocation or deallocation.

The **N+1 slot design** prevents read/write races during the transfer: layer `i`'s EP data lives in slot `i+1`, while TP data is written to slot `i`. Source and destination never overlap.

See: `unified_memory_manager.md`

### NVLink Peer Access Transfers

Instead of NCCL collectives (which require staging buffers, permutations, and kernel launch overhead), ParaS uses custom CUDA kernels that write directly to peer GPU memory via NVLink:

**Weight transfer** (`peer_access_fused_transfer_w13_v2`, `peer_access_fused_transfer_w2_v2`):
- Reads EP weights from local buffer
- Writes TP weight slices directly to each peer's TP slot via NVLink stores
- Fused kernel — one launch per layer handles all peers
- 1.57× faster than NCCL sequential for weights

**KV cache transfer** (`peer_access_kv_transfer`):
- Reads EP KV cache from scattered token positions (via index array)
- Writes to each peer's TP KV slot at contiguous positions via NVLink
- Fused K+V in a single kernel
- Handles head replication when `num_kv_heads < tp_size` via `ep_head = peer * num_kv_heads / tp_size`
- 2.74× faster than NCCL all_to_all for KV cache at scale

Both kernels follow the NVLink optimization guidelines:
- Warp-level peer assignment for balanced NVLink utilization
- int4 vectorized stores (512 bytes per warp per store)
- 8-store unrolling (4 KB contiguous per warp per iteration)
- `__ldg()` read-only cache for source reads
- Self-write bypass for local rank

See: `nvlink_peer_access_weight_transfer.md`, `nvlink_peer_access_kv_cache_transfer.md`

### CUDA IPC Without Context Overhead

Cross-process NVLink stores require mapping peer GPU memory into the local process's address space. ParaS uses `cudaIpcOpenMemHandle` with the lazy peer access flag — this maps the memory without creating full CUDA contexts on peer GPUs.

**Critical finding**: The commonly used `cudaDeviceEnablePeerAccess()` creates ~416 MiB CUDA contexts per peer GPU (~2.9 GB total on 8 GPUs). The lazy IPC flag alone is sufficient for NVLink stores. DeepEP uses the same approach. Removing `cudaDeviceEnablePeerAccess` saved ~2.9 GB per GPU with zero performance impact.

See: `exploration_notes_kv_cache_peer_access.md` §2

### NCCL Fallback Path

The NCCL path provides a portable fallback when NVLink peer access is unavailable:

**Weight transfer**: `all_to_all_single` with optional pipelining (overlap method).

**KV cache transfer**: `gather_kv_and_permute` → `repeat_interleave` (for head replication) → `all_to_all_single` → `permute_and_scatter_kv`.

The permutation outputs `[heads, tokens, KV, dim]` so that each head chunk is token-interleaved. After all_to_all splits by head and concatenates chunks from all senders, the result is `[total_tokens, KV, heads, dim]` — directly compatible with the scatter function.

For head replication (`num_kv_heads < tp_size`), `repeat_interleave` expands heads to `tp_size` virtual heads before all_to_all. This was chosen over a sub-head split + intra-group all_gather approach for simplicity, since the NCCL path is a fallback and replication factors > 2 are rare in practice.

### Switch Timeline (Qwen3-30B-A3B, 4×A100)

| Phase | Time | Method |
|-------|------|--------|
| Request gathering + metadata exchange | ~5 ms | NCCL all_gather |
| KV cache transfer | ~34 ms | peer_access |
| Weight transfer | ~61 ms | peer_access |
| Attention reconfiguration | ~20 ms | Local (QKV slice, backend update) |
| **Total** | **~120 ms** | |

The switch is transparent to clients — in-flight requests continue generating after the switch with correctly transferred KV cache. No requests are dropped.

## Design Documents

| Document | Contents |
|----------|----------|
| `unified_memory_manager.md` | Contiguous buffer allocation, N+1 slot design, alias system, KV cache integration |
| `nvlink_peer_access_weight_transfer.md` | w13/w2 CUDA kernels, data flow, theoretical analysis |
| `nvlink_peer_access_kv_cache_transfer.md` | Fused K+V kernel, head replication, NCCL fallback, smem vs L1 analysis |
| `nvlink_peer_access_guielines.md` | NVLink store optimization guidelines (grid config, vectorization, alignment) |
| `exploration_notes_kv_cache_peer_access.md` | Development notes: bugs found, CUDA IPC analysis, design tradeoffs |

## TP→EP Reverse Switch

The reverse switch (TP→EP) is now implemented, enabling full round-trip switching (EP→TP→EP→TP...).

### Key Design Decisions

1. **Weight transfer is mandatory, not a pointer swap**: The N+1 slot design means EP→TP overwrites EP weight slots (slot[i+1] becomes layer i+1's TP destination). After EP→TP, the original EP weights are destroyed. TP→EP must perform an actual reverse weight transfer (NCCL all-to-all or peer_access kernels) in reverse layer order (N-1→0).

2. **KV cache scatter**: The inverse of EP→TP gather. Each TP rank sends its head's token data to the EP ranks that will own those tokens. With head replication (`num_kv_heads < tp_size`), subgroup members split the token load — each sends a disjoint 1/R slice, cutting NVLink traffic by R.

3. **Request distribution**: Requests are partitioned across EP ranks using a greedy algorithm (balanced by request count first, then total tokens). All ranks compute the identical partition deterministically.

### Trigger

```bash
curl http://localhost:30000/paras_configure_ep
```

### Switch Timeline (Qwen3-30B-A3B, 4×A100)

| Phase | Time (naive) | Time (peer_access) |
|-------|-------------|-------------------|
| Request partition + pool resize | ~5 ms | ~5 ms |
| KV cache scatter | ~30 ms | TBD |
| Weight transfer (reverse) | ~70 ms | ~500 ms |
| Attention reconfiguration | ~5 ms | ~5 ms |
| **Total** | **~103 ms** | **~545 ms** |

## Limitations and Future Work

1. **Head replication e2e**: When `num_kv_heads < tp_size` (e.g., 4 heads / 8 GPUs), the KV transfer works correctly (verified by unit test), but the attention layer's `paras_configure_tp()` asserts `tp_size <= num_kv_heads`. Extending the attention reconfiguration to support replicated heads is a separate effort.

2. **Dynamic switching**: The current switch is triggered manually via `/paras_configure_tp` and `/paras_configure_ep`. An automatic policy that monitors batch size and switches when the crossover point is reached would enable fully adaptive serving.

3. **FP8 support**: The kernel and memory manager support FP8 weights but FP8 KV cache is not yet wired through.

4. **Peer access reverse kernel optimization**: The TP→EP peer_access weight kernels are ~5× slower than NCCL naive (545ms vs 103ms). The kernels need profiling and optimization to match the EP→TP direction's performance.
