# ParaS: Runtime EP↔TP Parallelism Switching

## Why Switch Between EP and TP?

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

ParaS enables the system to switch between EP and TP **at runtime** — without restarting the server, reloading weights, or dropping requests.

Current ParaS model support covers both Qwen3-MoE (MHA-only) and GPT-OSS (hybrid full + sliding-window attention). Each model has a dedicated ParaS subclass registered in the ParaS model registry.

| Scenario | Direction | Why |
|----------|-----------|-----|
| Peak traffic → off-peak | EP→TP | Small batch, TP's low-latency AllReduce wins |
| Off-peak → peak traffic | TP→EP | Large batch, EP's per-rank data reduction wins |
| Bursty workload | EP→TP→EP→... | Oscillate as batch size fluctuates |

## What Needs to Happen During a Switch

Both directions transform the model's runtime state between EP and TP layouts:

| Component | EP Layout | TP Layout |
|-----------|-----------|-----------|
| MoE weights | `num_experts/ep_size` complete experts per GPU | ALL experts, sharded along intermediate dimension |
| KV cache | All `num_kv_heads` heads, local tokens only | `num_kv_heads/tp_size` heads, ALL tokens |
| Attention | Full QKV projection (DP attention) | Sharded QKV projection (TP attention) |
| Requests | Each GPU owns a disjoint subset | All GPUs share the identical set |

### Asymmetry Between Directions

| Aspect | EP→TP (gather) | TP→EP (scatter) |
|--------|----------------|-----------------|
| Requests | Gather local subsets into global set (simple concat) | **Partition** global set into disjoint subsets (load-balancing problem) |
| KV cache | Head-split: all heads → subset heads | **Head-gather**: subset heads → all heads |
| MoE weights | all-to-all redistribution | **Reverse all-to-all** (EP weights destroyed during EP→TP; see N+1 Slot section) |
| Layer order | Forward (0, 1, ..., N-1) | **Reverse** (N-1, ..., 0) |

## The Unified Memory Manager

The foundation of fast switching. Allocates ALL persistent GPU memory — expert weights, attention weights, KV cache — in a single contiguous buffer at model init time.

**Key insight**: EP and TP layouts use the **same total bytes** per layer. An expert with shape `(E_local, 2I, H)` in EP becomes `(E_total, 2I/tp, H)` in TP — same bytes, different interpretation. The switch overwrites the same physical memory with the new layout, avoiding any allocation or deallocation.

### N+1 Slot Design

Prevents read/write races during transfer. For N model layers, N+1 slots are allocated:

```
Slots:  [ 0 | 1 | 2 | ... | N ]
TP:       0   1   2         N-1      ← layer i TP in slot[i]
EP:           0   1         N-2  N-1 ← layer i EP in slot[i+1]
```

**EP→TP** reads from slot[i+1] (EP), writes to slot[i] (TP). Forward order is safe.

**TP→EP** reads from slot[i] (TP), writes to slot[i+1] (EP). **Reverse order is required** because slot[i+1] = layer (i+1)'s TP source.

**Critical implication**: EP→TP **destroys EP weight data** in slots 1..N-1 (each becomes the next layer's TP target). TP→EP cannot use a pointer swap — it must perform an actual reverse weight transfer to reconstruct EP data.

See: `unified_memory_manager.md`

## NVLink Peer Access Transfers

Custom CUDA kernels write directly to peer GPU memory via NVLink, avoiding NCCL's staging buffers and kernel launch overhead.

### EP→TP Direction

**Weight transfer** (`peer_access_fused_transfer_w13_v2`, `peer_access_fused_transfer_w2_v2`):
- Reads EP weights from local buffer, writes TP weight slices to each peer's TP slot via NVLink
- Fused kernel — one launch per layer handles all peers
- 1.57× faster than NCCL sequential

**KV cache transfer** (`peer_access_kv_transfer`):
- Reads EP KV cache from scattered token positions, writes to each peer's TP KV slot
- Fused K+V in a single kernel
- Handles head replication via `ep_head = peer * num_kv_heads / tp_size`
- 2.74× faster than NCCL at scale

### TP→EP Direction

**Weight transfer** (`peer_access_fused_transfer_w13_ep`, `peer_access_fused_transfer_w2_ep`):
- Structural mirror of EP→TP v2 kernels with swapped src/dst
- Reverse layer order (N-1→0) with per-layer barrier

**KV cache scatter** (`peer_access_kv_scatter`):
- Reads local TP KV, writes to peer EP buffers at correct head slot
- Uses `num_kv_heads` (not `heads_per_rank * tp_size`) for EP destination stride — critical for head replication correctness
- Replication-aware routing in Python wrapper: each subgroup member routes 1/R tokens

All kernels follow the same NVLink optimization guidelines:
- Warp-level peer assignment for balanced NVLink utilization
- int4 vectorized stores (512 bytes per warp per store)
- 8-store unrolling (4 KB contiguous per warp per iteration)
- `__ldg()` read-only cache for source reads
- Self-write bypass for local rank

See: `nvlink_peer_access_weight_transfer.md`, `nvlink_peer_access_kv_cache_transfer.md`

## CUDA IPC Without Context Overhead

Cross-process NVLink stores require mapping peer GPU memory into the local address space. ParaS uses `cudaIpcOpenMemHandle` with the lazy peer access flag — no full CUDA contexts on peer GPUs.

**Critical finding**: `cudaDeviceEnablePeerAccess()` creates ~416 MiB contexts per peer GPU (~2.9 GB total on 8 GPUs). The lazy IPC flag alone is sufficient. Savings: ~2.9 GB per GPU.

See: `exploration_notes_kv_cache_peer_access.md` §2

## NCCL Fallback Path

### EP→TP

**Weight transfer**: `all_to_all_single` with optional pipelining (overlap method).

**KV cache transfer**: `gather_kv_and_permute` → `repeat_interleave` (for head replication) → `all_to_all_single` → `permute_and_scatter_kv`.

### TP→EP

**Weight transfer**: Reverse all-to-all (inverse permute + `all_to_all_single`), layers in reverse order.

**KV cache scatter**: A single unified code path handles both R=1 and R>1. With head replication, each subgroup member sends a disjoint 1/R token slice. On the receive side, contiguous subgroup chunks naturally concatenate via reshape — the only conditional is:
```python
reassembly_groups = group_size if heads_per_rank > 1 else num_kv_heads
```

For head replication in EP→TP, `repeat_interleave` inflates before send. For TP→EP, 1/R slicing deflates before send. Both maintain uniform per-subgroup totals.

## Request Partitioning (TP→EP only)

In TP mode, all ranks hold the identical request set. For EP mode, this must be partitioned into disjoint subsets — balanced so no GPU is overloaded:

```python
def partition_requests_for_ep(global_reqs, num_ranks, strategy="greedy"):
    sorted_reqs = sorted(global_reqs, key=lambda r: (-r.seqlen, r.rid))
    partitions = [[] for _ in range(num_ranks)]
    counts = [0] * num_ranks
    tokens = [0] * num_ranks
    for req in sorted_reqs:
        best = min(range(num_ranks), key=lambda i: (counts[i], tokens[i], i))
        partitions[best].append(req)
        counts[best] += 1
        tokens[best] += req.seqlen
    return partitions
```

Primary balance: equal request count. Secondary: equal token count. Tertiary: lowest rank index. Wrapped in a strategy registry for extensibility.

## Control Plane

Both switches are triggered via HTTP:

```bash
curl http://localhost:30000/paras_configure_tp   # EP→TP
curl http://localhost:30000/paras_configure_ep   # TP→EP
```

The request propagates: HTTP → TokenizerManager (adjusts fan_out) → DataParallelController (switches worker list) → Scheduler rank 0 (broadcasts via `tp_cpu_group`) → ALL ranks execute the switch with NCCL collectives → responses sent back via restored sockets.

No changes to tokenizer or detokenizer code are needed.

An in-process policy can also drive the switch automatically without HTTP triggering. The policy decision and ZMQ signaling live in [`parallelism_switch_policy.md`](parallelism_switch_policy.md); the **control-plane safety properties** documented below apply uniformly to both the HTTP and the auto-switch paths because they converge on the same `TokenizerManager.paras_configure_tp/ep` entry point.

### Why Tokenizer-First (not Scheduler-First)

Three pieces of state must change in lockstep for a mode switch to be globally consistent:

| Component | What it owns |
|---|---|
| TokenizerManager | `comm._fan_out` for each of the 18 control-message communicators (how many `*ReqOutput` responses to expect). |
| DataParallelController | `self.workers` (the active subset of scheduler sockets) and `control_message_step` (stride for control fan-out). |
| Schedulers | `paras_parallelism_config`, `tp_size`, `tp_group`, `attn_tp_*`, KV pool, attention backend, weight layout, CUDA graph set. |

The control plane funnels through TokenizerManager because both HTTP and policy-triggered paths land in `paras_configure_tp/ep`, which adjusts `_fan_out`, then dispatches `ParaSConfigureReqInput` to DataParallelController, which routes it to all current workers and then swaps `self.workers` to the post-switch set. A scheduler-first design would orphan in-flight user requests in dead workers' ZMQ buffers (because DPC's `self.workers` still indexes the pre-switch set) and break `_fan_out` accounting on any concurrent control RPC. The tokenizer-first design is **architecturally required**, not merely convenient.

### Race-Safety Invariants

Three invariants together ensure correctness when user requests, control messages, and mode-switch requests interleave:

1. **TokenizerManager → DataParallelController ZMQ FIFO.** Both user requests (`TokenizedGenerateReqInput`, `BatchTokenizedGenerateReqInput`) and the configure request travel over the **same** PUSH socket from TM to DPC. DPC's `event_loop` is a single-threaded synchronous Python loop that pulls messages in FIFO order, dispatches each via `_request_dispatcher` to the current `self.workers`, then — and only after dispatch completes — swaps `self.workers` for `ParaSConfigureReqInput`. User requests sent before the configure go to OLD workers; user requests sent after go to NEW workers. No window where a request can be misrouted.

2. **Per-iteration sequencing in the scheduler.** `process_input_requests` iterates the drained ZMQ batch sequentially. `paras_configure_handle` is synchronous (no `await`, no task creation), so the switch executes to completion before the next request in the batch is processed. With DP attention enabled (always true for ParaS), `recv_requests` splits incoming messages into `work_reqs` (user request types) and `control_reqs` (everything else, including `ParaSConfigureReqInput`) and reassembles them work-first; mixed batches like `[TokReq_A, ParaSConfigureReqInput, TokReq_B]` are reordered to `[TokReq_A, TokReq_B, ParaSConfigureReqInput]`, so both user requests are queued in OLD mode and the configure migrates the entire `waiting_queue` (plus `running_batch`) via the gather/scatter managers. The migration captures all queued and running requests regardless of their original arrival order within the batch.

3. **Idempotent fan-out across duplicate signals.** In EP mode every DP rank's policy runs independently and may fire `ParaSAutoSwitchReq` for the same target in the same iteration; TokenizerManager spawns one `asyncio.create_task` per signal. `_Communicator.queueing_call` serializes them: the first task wins the communicator and runs the switch; subsequent tasks queue on `_ready_queue` and execute after the first completes, finding `paras_parallelism_config` already at the target and short-circuiting at the idempotency guard in `paras_configure_tp/ep`. Because all duplicate signals carry the same target, `_fan_out` is written with the same value across all tasks, so response accounting stays consistent.

### Forward-Pass Boundary

`recv_requests` and `process_input_requests` run at the top of `event_loop_normal`, before `run_batch`. `ParaSConfigureReqInput` arriving at the ZMQ socket during a forward pass sits in the kernel buffer until the next iteration; it cannot interrupt an in-flight forward. After the switch returns, `self.last_batch = None` invalidates the pre-switch batch reference (its `req_pool_idx` points into the destroyed pool layout) so `get_next_batch_to_run` starts the new mode with a clean slate.

### Signal Path and Latency

For the auto-switch path, the policy-emitted `ParaSAutoSwitchReq` is sent **directly** from the scheduler to TokenizerManager via `send_to_tokenizer` (PUSH bound to `tokenizer_ipc_name`, where TokenizerManager PULLs). It does **not** transit DetokenizerManager. The full round-trip is four ZMQ hops: scheduler → TM (signal), TM → DPC (configure dispatch), DPC → scheduler (configure delivery), scheduler → TM (configure response). At ~30-50 µs per local ZMQ hop, the entire control-plane overhead is on the order of 200 µs — negligible against the ≈88-163 ms gather/scatter cost of the switch itself.

### Concurrent Control RPC Guard

`paras_configure_tp/ep` overwrites `comm._fan_out` on every TokenizerManager communicator (all 18, not just the configure communicator). If any other control RPC — `flush_cache`, `update_weights_*`, `get_internal_state`, profile, etc. — were in flight at the moment of the mutation, its expected response count would change under it, leading to premature event firing with truncated results, or a hang followed by `AttributeError` when late responses arrive at a cleared `_result_values`.

The guard is bi-directional and enforced at runtime by two shared counters wired through `_Communicator`:

- `_control_rpc_in_flight_counter` — every non-paras communicator increments on `__call__` entry and decrements in `finally`. `paras_configure_tp/ep` checks this counter at entry: if non-zero, it raises `RuntimeError` because the global `_fan_out` mutation would corrupt the in-flight call's accounting.
- `_paras_switch_counter` — `paras_configure_tp/ep` increments on entry and decrements in `finally`. Every non-paras `_Communicator.__call__` checks this counter at entry: if non-zero, it raises `RuntimeError` because the switch is mid-mutation.

`paras_configure_communicator` is exempt from both counters — it is the communicator that legitimately fires during a switch. Duplicate `ParaSAutoSwitchReq` signals from concurrent DP ranks still serialize cleanly via the existing `_ready_queue` on this communicator, and each duplicate task sees the same target (so the additional increments of `_paras_switch_counter` are correctly bracketed by their own `finally`).

The counter design (rather than a boolean flag) is required because duplicate auto-switch tasks can nest: with a boolean, the first task's `finally` would clear the guard while the second task is still mid-switch. Counters increment/decrement symmetrically so the guard stays raised until all concurrent switches complete.

Both guards raise `RuntimeError` rather than `assert`, so they survive `-O` interpreter flags. The application is expected to serialize control RPCs around switches; the guard is defensive and should never fire in correctly-coordinated code.

### Other Known Limitations

- **Pathological policy configuration can wedge the communicator queue.** With `paras_auto_switch_cooldown_sec=0` and `paras_auto_switch_window=1`, a TP→EP fire can land immediately after an EP→TP switch completes and overwrite `_fan_out` while queued duplicate tasks from the prior signal are still waiting on their responses, hanging the communicator. The default settings (`cooldown_sec=60`, `window=32`) make this physically impossible because the minimum inter-fire interval exceeds the maximum switch latency by more than two orders of magnitude.

## Round-Trip Support (EP→TP→EP→TP...)

Unlimited round-trips are supported without explicit state caching:

| Component | Why it works |
|-----------|-------------|
| **Weight aliases** | `ep_experts` → slot[i+1] and `tp_experts` → slot[i] are permanent. Each direction reconstructs its target slots from the source. |
| **KV aliases** | Same principle: `kv.ep` → slot[i+1], `kv.tp` → slot[i]. |
| **Communication groups** | Created at init (PARAS_TP, PARAS_DP, PARAS_EP), never destroyed. |
| **Dual LayerCommunicator** | EP and TP communicator objects co-exist. The switch swaps which one is active. |
| **QKV weights** | Full (EP) and sharded (TP) weight views are permanent. |

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

### E2E Coherence (Qwen3-30B-A3B)

| Test | Result |
|------|--------|
| EP request → EP→TP → TP request (same prompt) | ✅ Identical output |
| TP request → TP→EP → EP request (same prompt) | ✅ Identical output |
| In-flight requests during TP→EP switch | ✅ Coherent completion |
| Full round-trip EP→TP→EP | ✅ Identical to original EP |

### Gated Test Coverage (91 tests)

| Suite | Tests | Verified Against |
|-------|-------|-----------------|
| Layer cache spec + KV budget (CPU) | 14 | Formula parity and heterogeneous layer-spec invariants |
| Unified memory manager heterogeneous layout (CPU) | 5 | Union-layout bookkeeping |
| SWA allocator (CPU) | 8 | Allocator invariants |
| SWA pool rebind (CPU) | 9 | Buffer rebinding invariants |
| Hybrid round-trip (CPU) | 26 | Per-layer K/V rebinding and SWA warning behavior |
| Request partition (CPU) | 11 | Deterministic algorithm properties |
| KV cache R=1 (NCCL + peer_access) | 5 | Pattern ground truth |
| KV cache R=2 (NCCL + peer_access) | 5 | Pattern ground truth |
| SWA KV cache R=2 (NCCL + peer_access) | 5 | Pattern ground truth with sliding-window cap |
| GPT-OSS CUDA graph smoke | 3 | Class chain, EP↔TP round-trip, capture/replay |

## Design Documents

| Document | Contents |
|----------|----------|
| `unified_memory_manager.md` | Contiguous buffer allocation, N+1 slot design, alias system, KV cache integration |
| `nvlink_peer_access_weight_transfer.md` | w13/w2 CUDA kernels (EP→TP + TP→EP reverse), data flow, performance comparison |
| `nvlink_peer_access_kv_cache_transfer.md` | Fused K+V kernel (EP→TP + TP→EP scatter), head replication, NCCL fallback |
| `nvlink_peer_access_guielines.md` | NVLink store optimization guidelines (grid config, vectorization, alignment) |
| `exploration_notes_kv_cache_peer_access.md` | Development notes: bugs found, CUDA IPC analysis, design tradeoffs |

## Unsupported Features (hard constraints)

ParaS migration interacts with several scheduler subsystems in ways that are not currently safe. The following features must be **disabled** when `--enable-paras-moe` is set; the relevant assertions live in [`server_args._check_paras_config`](file:///home/shaoyuw/sglang/python/sglang/srt/server_args.py) and [`scheduler_paras_mixin`](file:///home/shaoyuw/sglang/python/sglang/srt/paras/scheduler_paras_mixin.py).

| Feature | Required flag | Why |
|---|---|---|
| Radix cache | `--disable-radix-cache` | ParaS uses `ChunkCache` / `SWAChunkCache`. The radix cache's tree state (lock_refs, tombstones, LRU lists) would not survive `tree.reset()` at switch boundaries, and prefix sharing is not a project priority for ParaS. |
| Chunked prefill | `--chunked-prefill-size -1` | ParaS migration cannot preserve mid-chunked-prefill state: `chunked_req` is not part of the gather/scatter request set, and per-token `kv_indices` in `req.prefix_indices` reference the pre-resize slot layout that paras_resize_and_clear destroys. |
| Overlap scheduler | `--disable-overlap-schedule` | The overlap scheduler runs the next forward pass while the previous result is still being processed. Switching mode mid-overlap would require migrating an in-flight forward's intermediate state, which is not modeled by the gather/scatter contract. Asserted at runtime in `SchedulerParasMixin.paras_configure_*`. |

In addition, ParaS asserts these positive requirements at startup:

- `--enable-dp-attention` and `--enable-dp-lm-head` (DP attention is the EP-mode shape).
- `0 < --paras-tp-size <= 8`.
- `--tp-size == --dp-size` (i.e., `attn_tp_size == 1`).

Both launch scripts under [`scripts/paras/eval/`](file:///home/shaoyuw/sglang/scripts/paras/eval/) bake all of the above into `PARAS_FLAGS` automatically when `ENABLE_PARAS=1`.

## Limitations and Future Work

1. **Head replication under SWA remains rare**: MHA replication is fully tested. Hybrid SWA replication is validated at replication factor 2 by the dedicated SWA cache-transfer suite, but production GPT-OSS / Gemma configs usually satisfy `num_kv_heads >= paras_tp_size`, so `SWACacheTransfer` emits a warning when replication is active to encourage end-to-end validation of the rare configuration.

2. **`dp_size > 1`**: Currently only `paras_dp_size == 1` is supported.

3. **FP8 support**: Kernels and memory manager support FP8 weights but FP8 KV cache is not yet wired through.

4. **Cross-request prefix sharing**: not available for ParaS (we run with `--disable-radix-cache`). Re-enabling would require porting tombstone-aware insert from PR #17220 to the ParaS path. Not on the roadmap.
