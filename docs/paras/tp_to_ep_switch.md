# ParaS: Runtime TP→EP Parallelism Switching

## Why Switch From TP to EP?

As described in `parallelism_switch.md`, neither Expert Parallelism (EP) nor Tensor Parallelism (TP) is universally optimal. Their relative performance depends on batch size: TP wins at small batches, EP wins at large batches. The EP→TP switch enables the system to adapt when traffic drops. The **TP→EP switch completes the round-trip**, allowing the system to switch back when traffic ramps up again.

| Scenario | Direction | Why |
|----------|-----------|-----|
| Peak traffic → off-peak | EP→TP | Small batch, TP's low-latency AllReduce wins |
| Off-peak → peak traffic | **TP→EP** | Large batch, EP's per-rank data reduction wins |
| Bursty workload | EP→TP→EP→... | Oscillate as batch size fluctuates |

With both directions implemented, ParaS enables **fully adaptive serving**: the system can switch between parallelism strategies at runtime without restarting, reloading weights, or dropping requests.

## What's Different From EP→TP

The TP→EP switch is the structural reverse of EP→TP, but the two directions are not symmetric in difficulty:

| Aspect | EP→TP (gather) | TP→EP (scatter) |
|--------|----------------|-----------------|
| Requests | Each rank has local subset → gather into global set (simple concat) | All ranks share identical global set → **partition into disjoint subsets** (load-balancing problem) |
| KV cache | All heads, local tokens → subset heads, all tokens (head-split) | Subset heads, all tokens → all heads, local tokens (**head-gather**) |
| MoE weights | Local experts → sharded experts (all-to-all redistribution) | **Reverse all-to-all** (EP weights in slot[i+1] are overwritten during EP→TP) |
| Layer order | Forward (0, 1, ..., N-1) | **Reverse** (N-1, ..., 0) to respect N+1 slot aliasing |

The new hard problem is **request partitioning**: deciding which requests go to which EP rank, balanced so no GPU is overloaded. This problem doesn't exist in EP→TP because gathering is a simple concatenation.

## The Switch Flow

### Overview

```
┌─────────────────────────────────────────────────────────┐
│                   ParaS TP→EP Switch                     │
│                                                         │
│  Phase 1: Scheduler prepares                             │
│    tree_cache.reset() → merge_last_batch()               │
│    Build global request list (identical on all ranks)     │
│                                                         │
│  Phase 2: Request partitioning + KV scatter              │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  partition_requests_for_ep()                        │ │
│  │    Greedy balanced assignment across EP ranks        │ │
│  └─────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  KV Cache Scatter (reverse of EP→TP gather)         │ │
│  │    peer_access: 1/R token routing → NVLink direct   │ │
│  │    OR nccl: unified all_to_all with replication      │ │
│  │    Layer order: N-1, N-2, ..., 0 (reverse)          │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                         │
│  Phase 3: Weight + attention restoration                 │
│    MoE: reverse weight transfer (NCCL or peer_access)    │
│    Attention: restore full QKV, head counts, backend     │
│                                                         │
│  Phase 4: Resume scheduling in EP mode                   │
│    Rebuild ScheduleBatch from local partition             │
│    Restore config flags, tokenizer/detokenizer sockets   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Phase 1: Prepare

The scheduler receives a `CONFIGURE_EP` request and prepares for the switch:

1. **`tree_cache.reset()`** — The prefix tree must be cleared because the request partition changes which tokens are local to each rank. Prefix sharing relationships from TP mode are no longer valid.

2. **`merge_last_batch()`** — Ensures all in-flight requests are in decode status (no pending prefills). The switch operates on decode-only batches.

3. **Build global request list** — In TP mode, all ranks process the same requests. The running batch's request list is the global set. No all-gather is needed (unlike EP→TP where each rank's local requests had to be collected).

### Phase 2: Request Partitioning

In TP mode, all ranks hold the identical request set. For EP mode, this set must be partitioned into disjoint subsets — one per EP rank. The partition must be:

- **Balanced**: roughly equal work per rank to avoid load imbalance
- **Deterministic**: all ranks compute the identical partition without communication
- **Total**: every request is assigned to exactly one rank, no duplicates or losses

**Algorithm** (greedy, extensible):

```python
def partition_requests_for_ep(global_reqs, num_ranks, strategy="greedy"):
    # Sort by (-seqlen, rid) for determinism
    sorted_reqs = sorted(global_reqs, key=lambda r: (-r.seqlen, r.rid))
    
    partitions = [[] for _ in range(num_ranks)]
    counts = [0] * num_ranks      # request count per rank
    tokens = [0] * num_ranks      # total tokens per rank
    
    for req in sorted_reqs:
        # Primary: fewest requests. Secondary: least tokens. Tertiary: lowest index.
        best = min(range(num_ranks), key=lambda i: (counts[i], tokens[i], i))
        partitions[best].append(req)
        counts[best] += 1
        tokens[best] += req.seqlen
    
    return partitions
```

The algorithm is wrapped in a strategy registry (`PARTITION_STRATEGIES`) so new heuristics can be added without modifying callers.

### Phase 3: KV Cache Scatter

The KV cache must be redistributed from TP layout to EP layout:

- **TP layout**: each rank stores `heads_per_rank = max(1, num_kv_heads // tp_size)` heads for **all** tokens
- **EP layout**: each rank stores **all** `num_kv_heads` heads for only its **local** tokens (determined by the partition)

This is the reverse of EP→TP's head-splitting gather. See the dedicated sections below for NCCL and peer access implementation details.

### Phase 4: Weight + Attention Restoration

**MoE weights — Reverse transfer is MANDATORY**: Although the N+1 slot design places EP weights in slot[i+1] and TP weights in slot[i], the EP→TP forward transfer **destroys EP slots**. When EP→TP processes layer `i+1`, it writes TP data to slot[i+1] — which IS layer `i`'s EP slot. After EP→TP completes, slots 1..N-1 contain TP data, not the original EP weights.

Therefore, TP→EP must perform an **actual reverse weight transfer** (inverse permute + all-to-all), not a pointer swap. The reverse reads from TP slots (which have correct TP data) and reconstructs EP data via NCCL all-to-all or peer_access kernels. Layers must be processed in **reverse order** (N-1→0) to avoid aliasing — layer `i`'s reverse writes to EP slot[i+1], which is also layer `(i+1)`'s TP source.

```python
# paras_model.py — naive path
def paras_configure_ep_naive(self):
    for layer in reversed(self.layers):
        layer.paras_configure_ep_mlp_naive()      # reverse all-to-all
    for layer in self.layers:
        layer.paras_configure_ep()                 # attn + communicator restore
```

**Attention**: The attention layer restores full (unsharded) QKV and output projection weights, resets head counts to EP values (`tp_size=1`), and re-initializes the FlashInfer backend with the EP-mode `req_to_token` mapping.

**KV pool**: `MHATokenToKVPool.paras_configure_ep()` restores the original head count and points `k_buffer[i]`/`v_buffer[i]` to the EP aliases from the memory manager (slot[i+1], not a reshape of the TP view in slot[i]).

**Scheduler**: Rebuilds the `ScheduleBatch` from only the local partition's requests, restores `enable_dp_attention=True`, resets `dp_size` and `tp_size` to EP values, and restores the tokenizer/detokenizer communication sockets so all ranks resume handling I/O.

## KV Cache Scatter: NCCL Path in Detail

The NCCL scatter path uses `all_to_all_single` to redistribute KV data. A single unified code path handles both the standard case (`num_kv_heads >= tp_size`) and the head replication case (`num_kv_heads < tp_size`).

### Send Side

Each rank has all tokens in its TP KV buffer but only `heads_per_rank` heads. It needs to send each destination's tokens (from the partition) to that destination. With head replication (factor R), each subgroup of R contiguous ranks holds **identical** KV data for the same head. Each subgroup member sends only its **1/R slice** of tokens, cutting NVLink traffic by R:

```python
replication_factor = group_size // num_kv_heads if num_kv_heads < group_size else 1
intra_rank = tp_rank % replication_factor  # 0 when R=1

for dest in range(group_size):
    full = len(token_partition[dest])
    my_start = full * intra_rank // replication_factor
    my_end   = full * (intra_rank + 1) // replication_factor
    # When R=1: my_start=0, my_end=full (sends all tokens)
    # When R=2: sends half the tokens
```

The tokens are gathered from the TP KV buffer using `gather_tp_kv_and_permute` and packed into a contiguous send buffer.

### Receive Side

Each rank receives token slices from all source ranks. Within each subgroup of R contiguous sources, the slices are for the **same head** but cover **disjoint token ranges**. A key mathematical property ensures correctness:

> The sum of integer-division slices over all subgroup members always equals the full count:
> `sum(full * (i+1) // R - full * i // R for i in 0..R-1) = full`

This means each subgroup's contributions in the receive buffer sum to exactly `recv_full_count` tokens. Since subgroups are contiguous in the `all_to_all` output, a simple reshape groups them correctly:

```python
recv_buf.view(num_kv_heads, recv_full_count, heads_per_rank, 2, head_dim)
```

No per-chunk parsing or manual concatenation is needed. The existing `permute_and_scatter_kv_to_ep` function handles the rest, called with `group_size=num_kv_heads` (instead of `group_size=group_size` in the non-replicated case):

```python
reassembly_groups = group_size if heads_per_rank > 1 else num_kv_heads
```

This is the **only conditional** for replication support. The entire NCCL scatter is a single code path.

### Reverse Layer Order

The scatter processes layers in **reverse** order (N-1, N-2, ..., 0). This is required by the N+1 slot design: layer `i`'s TP source is slot[i], and layer `i`'s EP destination is slot[i+1]. But slot[i+1] is also layer `(i+1)`'s TP source. Processing in reverse ensures layer `(i+1)`'s TP read completes before layer `i`'s EP write clobbers slot[i+1].

### Comparison with EP→TP Gather

| Aspect | EP→TP Gather (NCCL) | TP→EP Scatter (NCCL) |
|--------|---------------------|----------------------|
| Head replication handling | `repeat_interleave` before send (inflate) | 1/R token slicing before send (deflate) |
| Send split sizes | Uniform (every rank sends same amount) | Variable when R>1 (each intra_rank sends different slice size) |
| Recv split sizes | Variable (different ranks have different token counts) | Variable when R>1, uniform when R=1 |
| Reassembly | `permute_and_scatter_kv` | `permute_and_scatter_kv_to_ep` with `group_size=reassembly_groups` |
| Layer order | Forward (0→N-1) | Reverse (N-1→0) |

## KV Cache Scatter: Peer Access Path in Detail

The peer access path uses a custom CUDA kernel (`peer_access_kv_scatter`) that writes directly to peer GPU memory via NVLink. The kernel itself is unchanged for head replication — the optimization is in the **Python wrapper** that builds routing tensors.

### Replication-Aware Routing

Without replication, every rank builds routing tensors for all `num_global_tokens` tokens. With replication factor R, each subgroup member only routes its 1/R slice:

```python
intra_rank = tp_rank % replication_factor

for dest in range(group_size):
    full_tokens = token_partition[dest]
    my_start = len(full_tokens) * intra_rank // replication_factor
    my_end   = len(full_tokens) * (intra_rank + 1) // replication_factor
    my_slice = full_tokens[my_start:my_end]
    # Build token_to_rank and ep_dst_positions for my_slice only
```

The routing arrays (`tp_token_positions`, `token_to_rank`, `ep_dst_positions`) shrink by R, and `num_my_tokens` replaces `num_global_tokens` as the kernel launch parameter. The CUDA kernel processes fewer items and NVLink traffic drops by R — with zero kernel code changes.

### Layer Order

Same as NCCL: reverse order (N-1→0), with a per-layer `dist.all_reduce(barrier)` for cross-rank synchronization.

## Round-Trip Support (EP→TP→EP→TP...)

The N+1 slot design naturally supports unlimited round-trips without explicit state caching:

| Component | Why it works |
|-----------|-------------|
| **Weight aliases** | `ep_experts` → slot[i+1] and `tp_experts` → slot[i] are permanent. EP→TP overwrites TP slots from EP data; TP→EP reverse-transfers from TP slots back to EP slots. Both directions use the same aliases. |
| **KV aliases** | Same principle: `kv.ep` → slot[i+1], `kv.tp` → slot[i]. |
| **Communication groups** | Created at init in `paras_parallel_state.py` (PARAS_TP, PARAS_DP, PARAS_EP), never destroyed. |
| **Dual LayerCommunicator** | EP and TP communicator objects co-exist in each decoder layer. The switch swaps which one is active. |
| **QKV weights** | Full (EP) and sharded (TP) weight views are permanent. The attention layer swaps between them. |

The only requirement is that `paras_configure_ep()` and `paras_configure_tp()` are **re-entrant**: each correctly saves current state before switching, regardless of how many times it has been called.

## Control Plane (API Server)

The TP→EP switch is triggered via HTTP and propagated through the server process hierarchy:

```
HTTP GET /paras_configure_ep
  → TokenizerManager.paras_configure_ep()
    Sets comm._fan_out = server_args.dp_size (= tp_size for ParaS)
    Sends ParaSConfigureReqInput(type=CONFIGURE_EP)
  → DataParallelController
    Dispatches to TP workers (currently only rank 0)
    Then paras_ep_configure(): restores EP worker list
  → Scheduler rank 0 (receives via zmq)
    recv_requests() broadcasts control_reqs to all ranks via tp_cpu_group
  → ALL scheduler ranks execute paras_configure_ep()
    Phases 1-4 (with NCCL collectives — all ranks participate)
    Restore send_to_tokenizer/detokenizer sockets
    Return ParaSConfigureReqOutput (all ranks, via restored sockets)
  → TokenizerManager receives dp_size responses, fan_out satisfied
```

No changes to the tokenizer or detokenizer code are needed. The existing EP→TP implementation already set up the symmetric restore paths: the scheduler saves EP sockets before the TP switch and restores them during the EP switch.

## File Map

| File | Role in TP→EP Switch |
|------|---------------------|
| `paras/scatter_manager.py` | `ParaSReqScatterManager`, `partition_requests_for_ep`, `_scatter_cache_nccl`, `gather_tp_kv_and_permute`, `permute_and_scatter_kv_to_ep`, `_EPCacheView` |
| `paras/csrc/peer_access_transfer.cu` | `peer_access_kv_scatter` CUDA kernel, `peer_access_fused_transfer_w13_ep`, `peer_access_fused_transfer_w2_ep` |
| `paras/csrc/binding.cpp` | C++ bindings for reverse kernels |
| `paras/peer_access.py` | Python wrappers for peer access scatter and reverse weight transfer |
| `paras/scheduler_paras_mixin.py` | `paras_configure_ep()` — full 4-phase switch orchestration |
| `paras/layers/paras_model.py` | `paras_configure_ep_naive()`, `paras_configure_ep_peer_access()` — per-layer orchestration |
| `paras/layers/paras_decoder_layer.py` | Swap to EP communicator, call attention + MoE configure_ep |
| `paras/layers/paras_moe_block.py` | `paras_configure_ep()` — pointer swap; `paras_configure_ep_mlp_naive()` / `paras_configure_ep_fused_peer_access_kernel()` — reverse weight transfer |
| `paras/layers/paras_attention.py` | Restore full QKV weights, EP head counts |
| `mem_cache/memory_pool.py` | `paras_configure_ep()` — restore original head_num |
| `model_executor/model_runner.py` | `paras_configure_ep()` — call pool, attn_backend, model |
| `layers/radix_attention.py` | Restore EP head counts on RadixAttention |
| `layers/linear.py` | `QKVParallelLinear.paras_configure_ep()` — restore full weight |
| `layers/attention/flashinfer_backend.py` | `paras_configure_ep()` — restore head counts, req_to_token |

## Verified Performance (Qwen3-30B-A3B, 4×A100)

### Switch Timings

| Phase | EP→TP | TP→EP |
|-------|-------|-------|
| Request gather/partition | 16ms | <1ms |
| Cache reorchestrate | 2ms | 1ms |
| KV cache transfer | 46ms | 3ms (empty batch) |
| Weight transfer (peer_access) | 78ms | 70ms |
| Attention + config | 20ms | 11ms |
| **Total** | **163ms** | **88ms** |

### Coherence Verification (E2E on Qwen3-30B-A3B)

| Test | Result |
|------|--------|
| EP request → EP→TP → TP request (same prompt) | ✅ Identical output |
| TP request → TP→EP → EP request (same prompt) | ✅ Identical output |
| In-flight request during TP→EP switch (single) | ✅ Coherent completion |
| In-flight requests during TP→EP switch (2 concurrent) | ✅ Both coherent |
| Full round-trip EP→TP→EP: fresh request in restored EP | ✅ Identical to original EP |

### Unit Test Verification

| Test Suite | Tests | Result |
|------------|-------|--------|
| Request partition (CPU) | 11 | ✅ All pass |
| KV cache transfer (4-GPU, ground truth) | 6 | ✅ All pass |
| KV cache transfer (8-GPU, R=2) | 6 | ✅ All pass |
| Weight transfer (4-GPU, ground truth + round-trip) | 7 | ✅ All pass |

## Limitations and Future Work

1. **`dp_size > 1`**: Currently only `paras_dp_size == 1` is supported. With multiple DP groups, the request partition would need to account for inter-group distribution, and the reverse weight transfer would involve additional all-gather steps.

2. **Automatic switching policy**: Both EP→TP and TP→EP are triggered manually via HTTP endpoints. An automatic policy that monitors batch size and switches at the crossover point would enable fully adaptive serving without operator intervention.

3. **FP8 KV cache**: The scatter kernels and memory manager support FP8 weights but FP8 KV cache is not yet wired through.

4. **Peer access kernel for replication**: The CUDA kernel handles replication via Python-side routing tensor slicing. A kernel-level optimization (warp-aware subgroup assignment) could further reduce overhead for high replication factors.
