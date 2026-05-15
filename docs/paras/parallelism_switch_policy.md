# ParaS Automatic Parallelism Switching

## 1. Overview

The base ParaS design (`parallelism_switch.md`) provides EP↔TP switching as a manually-driven primitive: an operator hits `/paras_configure_tp` or `/paras_configure_ep` and the runtime swaps modes in ~250–300 ms. That works for orchestrated deployments, but it leaves the decision of *when* to switch to an external system that has to track per-iteration batch sizes, apply hysteresis, and avoid flapping.

This document describes the in-process policy that automates the decision and triggers the switch through the same control plane the HTTP route uses. The policy is on by default whenever `--enable-paras-moe` is set; the HTTP endpoints continue to work and bypass the policy.

The implementation lives in:
- `python/sglang/srt/paras/scheduler_paras_mixin.py` — `ParasAutoSwitchPolicy` class, `paras_auto_observe`, `paras_auto_pick_signal`
- `python/sglang/srt/managers/io_struct.py` — `ParaSAutoSwitchReq` message type
- `python/sglang/srt/managers/scheduler.py` — observe + signal-emit hook in `event_loop_normal`
- `python/sglang/srt/managers/tokenizer_manager.py` — `_handle_paras_auto_switch_req` (dispatcher entry); calls existing `paras_configure_tp/ep`
- `python/sglang/srt/paras/gather_manager.py` and `scatter_manager.py` — extended to preserve `waiting_queue` across the switch

See [`parallelism_switch.md`](parallelism_switch.md) for the underlying EP↔TP switch primitive — including the unified memory manager, gather/scatter, weight transfer, control-plane wiring, and the **race-safety invariants** that govern interleaved user-request / configure dispatch (those invariants apply to both this auto-switch path and the HTTP-triggered path).

## 2. When to Switch — Policy Definition

The policy is a sliding-window controller with a single threshold and a wall-clock cooldown. Three policy variants observe different iteration types and ship with different default thresholds, windows, and cooldowns.

### 2.1 Policy variants

| Policy | Iteration filter | Metric | Default threshold | Default window | Default cooldown |
|---|---|---|---|---|---|
| `decode` *(default)* | `ForwardMode.DECODE` | global decode batch size (= request count) | `64 * world_size` | `32` | `60 s` |
| `prefill` | `ForwardMode.EXTEND` | global prefill token count | `1024 * world_size` | `8` | `10 s` |
| `rollout` | (every iteration) | global running + waiting request count | `8 * world_size` | `1` | `5 s` |
| `hybrid` | (mixed prefill+decode) | n/a | n/a | n/a | n/a |

The decode and prefill defaults follow from the EP/TP crossover band measured in `parallelism_switch.md`. Threshold scales with `world_size` (= `tp_size`) so the per-GPU work that justifies a switch is constant. Prefill iterations are larger and rarer than decode, so the prefill policy uses a shorter window and shorter cooldown to stay reactive.

`rollout` is tuned for synchronous-rollout-style batch inference (GRPO post-training), where a single client submits a fixed burst of N requests (typically 512–2048), waits for all to complete, and submits the next burst. The metric is `global_running_reqs + global_waiting_reqs` rather than the per-iteration token count, because rollout cares about the **total in-flight workload** (not the size of any single forward). Low threshold (`8/GPU`) + `window=1` + short cooldown (`5 s`) make the policy react to load swings within a few seconds: TP during the boot grace window when the system is empty, EP when the burst arrives, and EP→TP again when the long-tail of slow requests drains the running batch below threshold.

`hybrid` is **not implemented** — picking it raises `NotImplementedError` at startup. ParaS disables chunked prefill, so `ForwardMode.MIXED` should not occur in practice; if a mixed workload becomes relevant, this is where to add a multi-metric policy.

### 2.2 Inputs and tunables

| CLI flag | Field | Default | Meaning |
|---|---|---|---|
| `--paras-auto-switch` | `paras_auto_switch` | `True` (when `--enable-paras-moe`) | Master enable. |
| `--paras-auto-switch-policy` | `paras_auto_switch_policy` | `decode` | Policy variant: `decode`, `prefill`, or `hybrid`. |
| `--paras-auto-switch-threshold` | `paras_auto_switch_threshold` | per-policy magic × `world_size` | Single switch threshold (no hysteresis). Override the policy magic by passing this. |
| `--paras-auto-switch-window` | `paras_auto_switch_window` | per-policy default | Sliding-window size (iterations). |
| `--paras-auto-switch-cooldown-sec` | `paras_auto_switch_cooldown_sec` | per-policy default | Wall-clock seconds between successive switches. |

Defaults resolve at startup inside `ServerArgs._handle_paras_auto_switch`. Any CLI-provided value wins over the policy default. Validation requires `threshold > 0`, `window > 0`, `cooldown_sec >= 0`, and the flags only apply when `--enable-paras-moe` is set.

### 2.3 Per-iteration observation

After each forward iteration that results in `process_batch_result(batch, result)`, the scheduler calls `paras_auto_observe(batch)`, which calls `policy.observe(scheduler, batch, now)` on the active policy. The base class invokes `observation_for_batch(scheduler, batch)` and appends the returned value (if any) to the policy's sliding window. Passing the scheduler reference lets each policy compute its metric using whatever scheduler state it needs — not just the batch — which is what `RolloutAutoSwitchPolicy` exploits to do mode-aware source selection (see below).

- `DecodeAutoSwitchPolicy.observation_for_batch` — observes **every iteration** (no `forward_mode` filter). Metric is `sum(batch.global_num_tokens)` in EP mode (the all-gathered per-DP token count summed across all DP ranks; in steady-state decode each request contributes exactly one token per iteration, so the sum equals the global decode batch size) and `len(batch.reqs)` in TP mode. Dropping the `is_decode()` filter is intentional: rank 0 may run an idle batch when other DP ranks hold the work, but its `batch.global_num_tokens` still carries the true global state via the MLP all-gather, so the in-flight indicator is preserved across both decode and idle iterations on rank 0. The `scheduler` argument is ignored.
- `PrefillAutoSwitchPolicy.observation_for_batch` — returns `None` unless `batch.forward_mode == ForwardMode.EXTEND`. Metric is `sum(batch.global_num_tokens)` in EP mode and `sum(req.seqlen for req in batch.reqs)` in TP mode. The `scheduler` argument is ignored.
- `RolloutAutoSwitchPolicy.observation_for_batch` — observes **every iteration** and returns `running + waiting` requests globally with **mode-aware source selection**:
   - **EP mode**: rank 0 holds only its DP slice, so the per-rank running batch and waiting queue underrepresent the global state. The policy sums the MLP all-gather output (`batch.global_running_reqs` / `batch.global_waiting_reqs`) across DP ranks. If the all-gather hasn't populated yet (idle / boot), it returns `None` to skip the iteration.
   - **TP mode**: unified data plane — every rank holds the same `running_batch` after the EP→TP gather. Rank 0's local view IS the global view, so the policy reads `scheduler.running_batch.reqs` and `scheduler.waiting_queue` directly. The all-gather is unreliable in TP mode post-switch (the MLP-sync path does not repopulate the `batch.global_*` fields when DP attention is off), so the policy must **not** consult `batch.global_*` in TP mode or it would go blind and never fire back.
- `HybridAutoSwitchPolicy.__init__` raises before any observation can happen.

Non-positive metric values are silently skipped. Otherwise the value is appended to the policy's `deque` (capped at `window`).

### 2.4 Decision evaluation

After `paras_auto_observe`, the scheduler calls `paras_auto_pick_signal()`. Algorithm (single threshold, no hysteresis):

```
if now < cooldown_until: return None
if len(window) < window.maxlen: return None       # window not yet full
avg = sum(window) / len(window)
if mode == "EP" and avg < threshold:  target = "TP"
elif mode == "TP" and avg > threshold: target = "EP"
else: return None
cooldown_until = now + cooldown_sec
window.clear()
return target
```

The cooldown bounds switch frequency to once per `cooldown_sec` in the worst case, capping the cost of false positives at `≤ switch_latency / cooldown_sec` (e.g., for the decode policy: `300 ms / 60 s ≈ 0.5 %` overhead; for the prefill policy: `300 ms / 10 s ≈ 3 %` peak overhead — acceptable because prefill bursts are infrequent and the policy is sized to react before the workload pattern dissipates).

### 2.5 Where the policy lives — rank 0 sole observer

The auto-switch hook in `event_loop_normal` gates both observation and firing on `self.tp_rank == 0`:

```python
if batch:
    result = self.run_batch(batch)
    self.process_batch_result(batch, result)
    if self._paras_auto_policy is not None and self.tp_rank == 0:
        self.paras_auto_observe(batch)
        signal = self.paras_auto_pick_signal()
        if signal is not None:
            self.send_to_tokenizer.send_output(signal)
```

Only one scheduler process — TP rank 0 — appends to its policy window, evaluates `pick_target`, and emits `ParaSAutoSwitchReq`. This is correct because the policy's metric is **global, not local**: `sum(batch.global_num_tokens)` is identical on every rank in both modes. In EP + DP-attention mode `prepare_mlp_sync_batch_raw` performs an all-gather and writes the per-DP token-count list on every rank; in TP mode every rank already shares the same data-plane batch, so `batch.global_num_tokens = [num_tokens]` is identical too. Running the policy on every rank would only produce redundant duplicate signals.

**Why `self.tp_rank == 0` and not `self.attn_tp_rank == 0`:** in EP + DP-attention with `attn_tp_size = 1`, `compute_dp_attention_world_info` returns `attn_tp_rank = tp_rank % 1 = 0` for every scheduler. Only `tp_rank` is `0` on exactly one process across both modes.

**Why rank 0 reliably observes the workload in EP mode:** the `DataParallelController` dispatches incoming HTTP requests via `--load-balance-method=round_robin` (the default), with `round_robin_counter` initialized to `0`. The first request after server boot routes to `workers[0]` (DP rank 0); subsequent requests rotate through all DP ranks. Rank 0 receives approximately `1/dp_size` of incoming traffic, which is sufficient to fill its policy window with the same global observations every other rank would have computed. (If a deployment uses a different load-balance method that systematically starves rank 0 — e.g., a custom router with rank-affinity — the rank-0 gate should be revisited; the default round-robin path is the supported configuration.)

The `_paras_auto_policy` field is still constructed unconditionally on every rank in `init_paras_config`. Non-rank-0 policies receive no observations, but keeping the field present everywhere avoids per-rank construction branches and keeps `paras_configure_tp/ep` symmetric — the `_paras_auto_clear_window_on_switch` call in §5.2 is a no-op on empty windows.

**Edge case — rank 0 idle iteration:** under DP attention, `prepare_mlp_sync_batch_raw` synthesizes an idle batch on a locally-empty scheduler so it can still participate in the MLP all-gather. The idle batch's `forward_mode` is `IDLE`, but `DecodeAutoSwitchPolicy` does **not** filter on `forward_mode` (see §2.3) and instead observes via `batch.global_num_tokens`, which the MLP all-gather populates with the true global per-DP token counts. So even when round-robin routes a light-load request to a non-zero DP rank (e.g., DP1, because server warmup advances `round_robin_counter` past 0), rank 0 still observes `sum = global token count` and the policy fires correctly. `PrefillAutoSwitchPolicy` retains its `forward_mode == EXTEND` filter because the prefill metric (token count) is qualitatively different from the decode metric (request count); a future revision should generalize this if the prefill policy is ever exercised.

## 3. How to Trigger — Control-Plane Flow

The auto-switch reuses the existing HTTP path's control plane verbatim. Both flows converge at `TokenizerManager.paras_configure_tp/ep`, which is the single source of truth for adjusting `_fan_out`, the DataParallelController worker list, and the per-scheduler `paras_configure_tp/ep` execution. The race-safety invariants that govern this convergence (TM→DPC ZMQ FIFO, per-iteration sequencing in the scheduler, idempotent fan-out across duplicate signals) are documented in [`parallelism_switch.md` § Control Plane](parallelism_switch.md#control-plane); this section describes only the policy-specific additions on top of that primitive.

### 3.1 Side-by-side comparison

```
HTTP path                                Auto-switch path
──────────                                ────────────────
GET /paras_configure_tp                   Scheduler TP rank 0 decodes
                                          ↓
                                          paras_auto_observe(batch)
                                          ↓
                                          policy.pick_target() → "TP"
                                          ↓
                                          send_to_tokenizer.send_output(
                                            ParaSAutoSwitchReq(target=...))
                                          ↓  (direct PUSH to tokenizer_ipc_name)
        TokenizerManager.paras_configure_tp() ← both paths converge here
                                          ↓
                                          adjust comm._fan_out
                                          send ParaSConfigureReqInput
                                          ↓
                                          DataParallelController:
                                            • routes ParaSConfigureReqInput
                                              to all current workers
                                            • THEN switches
                                              workers = paras_tp_workers
                                          ↓
                                          All schedulers receive
                                          ParaSConfigureReqInput
                                          ↓
                                          torch.distributed broadcast in
                                          recv_requests() ensures every
                                          TP rank sees the same request
                                          ↓
                                          Each rank dispatches via
                                          TypeBasedDispatcher to
                                          paras_configure_handle →
                                          paras_configure_tp/ep
```

The `ParaSAutoSwitchReq` is the only new message type added by this policy. The scheduler emits it via `send_to_tokenizer`, which is a PUSH socket bound to `tokenizer_ipc_name` — the same socket TokenizerManager PULLs from for scheduler-feedback messages. The signal goes **directly** to TokenizerManager; there is no DetokenizerManager hop on this path. Its handler in `TokenizerManager._result_dispatcher` is a two-line dispatch:

```python
def _handle_paras_auto_switch_req(self, req):
    if req.target == ParaSConfigureReqType.CONFIGURE_TP:
        asyncio.create_task(self.paras_configure_tp())
    elif req.target == ParaSConfigureReqType.CONFIGURE_EP:
        asyncio.create_task(self.paras_configure_ep())
```

The auto path adds *one* asyncio task in front of the HTTP entrypoint, leaving everything downstream unchanged. See [`parallelism_switch.md` § Signal Path and Latency](parallelism_switch.md#signal-path-and-latency) for the full hop breakdown and ~200 µs control-plane overhead estimate.

### 3.2 Where the signal is emitted

The signal-emit site is in `event_loop_normal` immediately after `process_batch_result`, gated on `self.tp_rank == 0`:

```python
if batch:
    result = self.run_batch(batch)
    self.process_batch_result(batch, result)
    if self._paras_auto_policy is not None and self.tp_rank == 0:
        self.paras_auto_observe(batch)
        signal = self.paras_auto_pick_signal()
        if signal is not None:
            self.send_to_tokenizer.send_output(signal)
```

Position matters in two ways. (a) The observation feeds the policy with the just-completed iteration's batch size, and the signal is emitted **between** iterations — never mid-forward. This matches the existing HTTP-triggered switch boundary, which also fires only between iterations because `recv_requests` runs at the top of the loop. (b) The `self.tp_rank == 0` gate ensures exactly one scheduler process emits per decision — see §2.5 for the rationale.

## 4. In-Flight Request Handling

The base ParaS gather/scatter machinery already preserves the **running batch** — requests in mid-decode (and mid-prefill, where supported) — across the switch via KV cache transfer (`gather_cache` / `scatter_cache`). This document focuses on the additional handling for the **waiting queue**, which is what the auto-switch path required to be added.

### 4.1 Why the waiting queue needs explicit handling

Before the auto-switch work, `paras_check()` rejected the switch if any rank's `waiting_queue` was non-empty. That was acceptable for HTTP-orchestrated switches (operators triggered them during quiescent periods) but it would cause livelock under sustained load: the policy would want to switch, the waiting queue would never drain, and the switch would never fire.

The fix is to gather/scatter the waiting queue alongside the running batch. Waiting-queue requests have **no KV cache yet** — they are plain Python `Req` objects holding prompt token IDs — so the same all-gather machinery used for cross-rank metadata works directly, with no GPU memory transfer required.

`paras_check()` is now a no-op; the queue-empty precondition is no longer needed.

### 4.2 EP→TP gather

In EP mode each DP rank has its own `waiting_queue`. The gather brings them into a single global queue at TP rank 0:

```python
class ParaSReqGatherManager:
    def __init__(self, local_reqs, ..., local_waiting_reqs=None):
        self.local_waiting_reqs = local_waiting_reqs or []

    def gather_global_reqs(self):
        # Existing: gather running_batch reqs across DP ranks
        self.global_reqs, ... = paras_tp_group_all_gather_reqs(self.local_reqs, ...)

        # New: gather waiting_queue reqs across DP ranks. They have no KV
        # cache — pickle + all-gather is sufficient, no GPU transfer.
        gathered_waiting, _ = paras_tp_group_all_gather_reqs(
            self.local_waiting_reqs, self.gather_group)
        self.global_waiting_reqs = gathered_waiting or []

    def get_new_waiting_queue(self, paras_tp_rank: int):
        return list(self.global_waiting_reqs) if paras_tp_rank == 0 else []
```

After the gather, TP rank 0's waiting queue contains every queued prefill from every former DP rank, and the other ranks' queues are empty (they no longer receive requests in TP mode).

### 4.3 TP→EP scatter

In TP mode the unified data plane runs on every rank, and `scheduler.recv_requests` broadcasts each new request to every rank's `waiting_queue` via the attention TP group (which is full-rank in TP mode). So every rank's `waiting_queue` actually holds the same global set of pending requests — **not** disjoint slices. The scatter must explicitly elect rank 0 as the sole contributor; otherwise the all-gather inside `ScatterManager` would multiply the global set by `paras_tp_size`, and the subsequent partition would redistribute duplicates back as the full global set on every rank's local queue.

The rank-0 gate is enforced **at the call site** in `paras_configure_ep` (`scheduler_paras_mixin.py:472-474`), not by the scatter manager itself:

```python
# scheduler_paras_mixin.py — paras_configure_ep
global_reqs = list(self.running_batch.reqs) if self.running_batch else []
local_waiting_reqs = (
    list(self.waiting_queue) if self.paras_tp_rank == 0 else []
)
```

This mirrors the EP→TP path's existing convention in `GatherManager` (`gather_manager.py`), where `get_new_waiting_queue(paras_tp_rank)` returns the global queue only on rank 0 and empty lists on the others. The scatter then proceeds normally:

```python
class ParaSReqScatterManager:
    def __init__(self, global_reqs, ..., local_waiting_reqs=None):
        # Caller (paras_configure_ep) has already gated to rank-0-only.
        # Non-zero ranks pass [] here.
        local_waiting_reqs = local_waiting_reqs or []
        gathered_waiting, _ = paras_tp_group_all_gather_reqs(
            local_waiting_reqs, scatter_group)
        self.global_waiting_reqs = gathered_waiting or []

    def partition_requests(self):
        # Existing: greedy partition of running_batch
        partitions = partition_requests_for_ep(self.global_reqs, self.paras_tp_size)
        self.local_reqs = partitions[self.paras_tp_rank]

        # New: greedy partition of waiting_queue using the same algorithm
        waiting_partitions = partition_requests_for_ep(
            self.global_waiting_reqs, self.paras_tp_size)
        self.local_waiting_reqs_after_partition = waiting_partitions[self.paras_tp_rank]
```

Each EP rank's new `waiting_queue` is the slice it received from the partition. The greedy strategy balances on `(num_requests, total_tokens, rank_index)` to avoid hot-spotting one rank with all the queued work.

### 4.4 Implication for in-flight prefills

Because `waiting_queue` requests are now preserved verbatim across the switch:

- A prefill that has been admitted to `running_batch` is handled by the existing `gather_cache` / `scatter_cache` (KV cache transferred byte-for-byte).
- A prefill still in `waiting_queue` (admitted by the dispatcher but not yet started) is gathered/scattered as a Python object and re-admitted in the new mode without re-tokenization or any user-visible side effect.

The scheduler does not need to pause new admissions or wait for the queue to drain; the switch can fire under any load condition.

## 5. Anti-Flap and Idempotency

Two edge cases would otherwise cause unwanted behavior. Both are handled inside `paras_configure_tp/ep` so they apply uniformly to both the HTTP and auto paths.

### 5.1 Idempotent guard against duplicate signals

The auto-switch path itself no longer produces duplicates: only TP rank 0 emits `ParaSAutoSwitchReq` (see §2.5). But two other control-plane paths can still produce duplicate `paras_configure_tp/ep` invocations against `TokenizerManager`:

1. **HTTP path concurrency** — an operator issues two `/paras_configure_tp` requests in rapid succession, and both reach `TokenizerManager.paras_configure_tp` before either completes its `_fan_out` adjustment.
2. **Future control-plane additions** — any new mechanism that emits `ParaSAutoSwitchReq` (e.g., a sidecar monitor, a different policy variant, or relaxing the rank-0 gate later) could race with an in-flight switch.

Without an entry guard, the second concurrent task would re-execute gather/scatter on a system already in TP, corrupting state.

The guard is a single check at the entry of `paras_configure_tp/ep`:

```python
@paras_func
def paras_configure_tp(self):
    if self.paras_parallelism_config == "TP":
        logger.warning("paras_configure_tp called but already in TP mode; skipping")
        return
    ...
```

The corresponding guard in `paras_configure_ep` was already present (`if self.paras_parallelism_config != "TP"`).

### 5.2 Window-clear and cooldown extension on switch

`_paras_auto_clear_window_on_switch` runs at the entry of every `paras_configure_tp/ep` call. It performs two things:

1. **Window clear.** On the auto path, `pick_target` already cleared rank 0's window when the decision fired — but the HTTP path bypasses `pick_target` entirely. Without this clear, an HTTP-triggered EP→TP switch would leave rank 0's window holding pre-switch EP-mode observations; the first few TP-mode samples appended to those stale entries could trigger an immediate reverse switch.
2. **Cooldown extension.** `cooldown_until = max(cooldown_until, time.time() + cooldown_sec)` anchors the cooldown to the switch-start time rather than the decision time. The switch itself takes ~250–300 ms; without this extension, the auto-path cooldown would effectively be `cooldown_sec − switch_latency` because `pick_target` set `cooldown_until` at the decision moment, not at the switch moment.

```python
def _paras_auto_clear_window_on_switch(self) -> None:
    policy = self._paras_auto_policy
    policy.window.clear()
    policy.cooldown_until = max(
        policy.cooldown_until, time.time() + policy.cooldown_sec)
```

Running on every rank (not just rank 0) keeps the configure entry symmetric and free of rank conditionals. On non-rank-0 schedulers the window is always empty (see §2.5), so the clear is a no-op and the cooldown bump is unused — but they cost nothing.

Combined with the idempotent guard (§5.1), the system absorbs any concurrent control-plane invocations, and stale state from any prior mode never drives a reverse switch.

## 6. Robustness Fixes for the Rollout Path (May 2026)

The first end-to-end rollout-policy run (qwen3-30B-A3B and gpt-oss-120b-BF16 on 8×A100, dapo dataset, 512 reqs, ParaS auto-switch with the rollout policy) surfaced four classes of bugs that were latent in the original design and would have prevented production use. Each is documented inline below with the commit that fixed it and the verification signal that confirms the fix.

### 6.1 Per-mode `max_bs` sync on the cuda-graph runner (`3f64fa33c`)

`paras_init_dual_cuda_graphs` captures two graph sets — one with EP-mode batch sizes (`cuda_graph_bs`, default up to 256) and one with TP-mode batch sizes (`paras_tp_cuda_graph_bs`, default up to 64) — and stores both under `runner._paras_saved[mode]`. At runtime, `paras_load_cuda_graph_state(runner, mode)` swaps the graphs and `capture_bs` per mode via a `_SETTINGS_KEYS` tuple. But `runner.max_bs` is set once at `CudaGraphRunner.__init__` to `max(EP_max, TP_max)` (= 256 in the canonical config) and was **not** in `_SETTINGS_KEYS`, so the runtime `can_run()` check `cuda_graph_bs <= self.max_bs` used the stale 256 in TP mode. A TP-mode batch with `cuda_graph_bs > 64` passed `can_run()` (256 > 64 = False, but 100 ≤ 256 = True), then `replay_prepare` did `bisect_left(capture_bs=[1..64], 100) = 12` → `capture_bs[12]` → `IndexError: list index out of range`. The fix appends two lines to `paras_load_cuda_graph_state` that sync `runner.max_bs = max(runner.capture_bs)` and `runner.max_num_token` per mode.

### 6.2 Rank-0 sole contributor to the waiting queue on TP→EP (`217b542fe`)

In TP mode, `scheduler.recv_requests` broadcasts each new request to every rank's `waiting_queue` via the attention TP group (`attn_tp_size == paras_tp_size == 8` post-switch). So every rank's `waiting_queue` ends up holding the **same** global set of pending requests, not disjoint slices. The original `paras_configure_ep` passed `local_waiting_reqs=list(self.waiting_queue)` from **every** rank into `ScatterManager`'s all-gather, producing an 8× duplicated global set that the partition then redistributed as the entire global set onto every rank's local queue. EP-mode steady-state then promoted those duplicates into the running batch until hitting the `max_running_requests / dp_size` per-rank cap — so every rank ran the same requests in parallel, wasting 7/8 of the compute and spamming the tokenizer manager with `"state was deleted"` orphan-output errors (~10/sec across 8 ranks). The fix elects rank 0 as the sole contributor at the call site, mirroring `GatherManager.get_new_waiting_queue(paras_tp_rank)` on the EP→TP side. Verified on smoke v9: per-rank running drops from 256 (replicated) to ~57 (disjoint), per-rank token counts diverge by ~16k confirming truly independent slices, and `"state was deleted"` count drops to **0**.

### 6.3 Per-mode `max_running_requests` via `plan_req_capacities` (`bad15b290`)

`tp_worker.max_running_requests` is computed once at startup as `server_args.max_running_requests // (dp_size if enable_dp_attention else 1)` — correct for EP (8 disjoint per-rank schedulers, each capped at `CLI/8`) but wrong for TP, where ParaS collapses to one unified scheduler whose running batch is shared across all ranks. With a static value, TP-mode peak running was capped at `CLI/8 = 256` (with `CLI=2048` and `dp_size=8`) even though `tp_max_num_reqs` already sized the pool for the full 2048. The fix computes both `ep_max_running_requests` and `tp_max_running_requests` inside `paras_memory_manager.plan_req_capacities` (alongside the existing per-mode pool sizes), and `paras_configure_helper` overrides `scheduler.max_running_requests` with the active-mode value after every EP↔TP swap.

### 6.4 Sampler `tp_sync_group` re-pointing on every mode swap (`2460fa666`)

`Sampler.tp_sync_group` is cached at `Sampler.__init__` from `get_attention_tp_group().device_group`, which under DP attention is a single-rank group. The optional token-id all-reduce gated by `SYNC_TOKEN_IDS_ACROSS_TP` is therefore a single-rank no-op for the lifetime of the process — even after ParaS swaps `parallel_state._TP` and the attn-tp metadata on an EP→TP transition. SGLang relies on deterministic kernels (last all-reduce, lm-head matmul, sampling) to keep per-rank sampled token ids in agreement without an explicit sync; for qwen3 (flashinfer + non-SWA) the kernels are deterministic enough, but for gpt-oss-120b (triton attention + triton MoE + SWA) they are not. On the wind-down EP→TP transition with active in-flight requests, ranks sampled different tokens → different `req.finished()` → different `running_batch.batch_size` (DP0=20, DP3=14, DP1/5/7=13 in the watchdog dump) → next forward's NCCL collective deadlocked on shape mismatch. 100% GPU util, 300 s watchdog timeout, no recovery. The fix has two parts:

1. `sampler.py`: add a `force_sync_token_ids` instance flag (default `False`) and OR it into the existing all-reduce gate alongside `SYNC_TOKEN_IDS_ACROSS_TP` and `sampling_info.grammars`.
2. `scheduler_paras_mixin.py`: on `paras_configure_tp`, repoint `sampler.tp_sync_group` to `self.paras_tp_group.device_group` (the full 8-rank TP group) and set `sampler.force_sync_token_ids = True`. On `paras_configure_ep`, restore `sampler.tp_sync_group` to `self.paras_tp_attn_tp_group.device_group` (the EP-mode single-rank attn-tp group) and clear the force flag.

The all-reduce is now a real cross-rank `MIN` reduce of `batch_next_token_ids` in TP mode and a single-rank no-op in EP mode, matching the runtime parallelism contract at every moment.

### 6.5 Additional log-spam reduction (`94dc29146`)

Two unrelated noise sources surfaced under the 512-burst rollout load. `scheduler_output_processor_mixin._handle_decode_batch` force-emitted an output batch every `DEFAULT_FORCE_STREAM_INTERVAL` decode tokens even for non-streaming requests, creating an IPC-in-flight window in which post-finish duplicates land orphaned in `TokenizerManager.rid_to_state` (`"state was deleted"` log spam — ~600/min in pre-fix smoke). The fix sets `should_output = False` on the non-streaming non-multimodal path, deferring to `req.finished()` for the single terminal output. Separately, `scheduler.recv_requests` was emitting a per-request `logger.info("Processing request: ...")` line that, with 8× DP broadcast and a 2k-req burst, produced thousands of TokenizedGenerateReqInput-repr lines per second. The line is now commented out.

## 7. Verified Behavior

### 7.1 Decode policy — original autoswitch smoke

End-to-end test on 8×A100 with gpt-oss-120b-bf16, decode policy, test-scale tunables (`--paras-auto-switch-policy decode --paras-auto-switch-threshold 8 --paras-auto-switch-window 4 --paras-auto-switch-cooldown-sec 15`):

| Phase | Trigger | Switch fired | observations | avg | Rank that fired |
|---|---|---|---|---|---|
| pre-burst light prompt → decode tail (load drops below threshold) | avg ≤ threshold=8 | EP→TP | `[8, 8, 8, 4]` | 7.00 | TP rank 0 (`DP0 TP0 EP0`) |
| 32-burst arrives in TP mode | avg > threshold=8 | TP→EP | `[2, 2, 1, 32]` | 9.25 | TP rank 0 (`DP0 TP0 EP0`) |
| post-burst light prompt | (within 15 s cooldown) | — | — | — | — |

Per `autoswitch_test.sh` run: **exactly 2 verbose `policy fired` events** (one per direction), both from TP rank 0 (`DP0 TP0 EP0`). The 9 → 2 reduction (vs. the original all-rank-fire design that fired once per scheduler) reflects the rank-0 sole-observer gate from §2.5 combined with `DecodeAutoSwitchPolicy` dropping its `forward_mode` filter (§2.3) so rank 0 still observes the global in-flight count via `batch.global_num_tokens` on iterations where its local batch is `IDLE` (which is most iterations, since round-robin dispatch advances past 0 during server warmup and routes light-load prompts to DP1+).

The non-trivial values in the EP→TP window (`[8,8,8,4]` rather than the naive `[1,1,1,1]`) reflect server-warmup broadcast batches that immediately precede phase 1's light prompt — `prepare_mlp_sync_batch_raw` populates rank 0's idle-batch `global_num_tokens` with the same all-gathered counts every other rank sees. The window still satisfies `avg ≤ threshold` and produces a correct EP→TP transition. Every request — pre-burst light prompt, the 32 burst prompts, and the post-burst light prompt — returned coherent text; zero flapping; zero errors in the server log. Per-fire lines visible via `grep "ParaS \\[DecodeAutoSwitchPolicy\\] policy fired" $LOG_FILE`.

For production-scale validation, the policy defaults resolve at startup via `ServerArgs._handle_paras_auto_switch` (see §2.2):

- `decode` policy: `threshold = 64 * world_size`, `window = 32`, `cooldown = 60 s`
- `prefill` policy: `threshold = 1024 * world_size`, `window = 8`, `cooldown = 10 s`
- `rollout` policy: `threshold = 8 * world_size`, `window = 1`, `cooldown = 5 s`

The production defaults require sustained traffic at the corresponding global batch sizes; the same control-plane flow applies as in the test above.

### 7.2 Rollout policy — 2026-05-15 smoke validation

Per-model end-to-end smoke on the dapo dataset (DAPO-Math-17k sampled to 8k, no spec mode, 16k max-completion cap), 512 requests submitted via async-gather to `/v1/chat/completions`, 8×A100-80GB:

| Model | mfs | completed/failed | e2e_time | output_throughput | policy fires | state-deleted errors |
|---|---|---|---|---|---|---|
| qwen3-30B-A3B | 0.80 | **512 / 0** | 689.9 s | 6,360 tok/s | 4× (boot EP→TP, burst TP→EP, wind-down EP→TP, tail-drain TP→EP) | 0 |
| gpt-oss-120b-BF16 | 0.75 | **512 / 0** | 650.6 s | 2,641 tok/s | 3× (boot EP→TP, burst TP→EP, wind-down EP→TP) | 0 |

Both runs exercise the canonical rollout pattern: an empty system at boot (policy fires EP→TP because running+waiting < 8/GPU); a 512-request burst arrives (policy fires TP→EP at obs ≫ 64, e.g. 283 or 335); the system runs in EP for several minutes draining the burst; the long-tail of slow requests drops the running+waiting count below threshold and the policy fires EP→TP (`obs = 63` in both smokes — exactly the threshold boundary); the remaining ~50 requests finish in TP mode and the run completes. **The wind-down EP→TP transition is the one that hung gpt-oss before commit `2460fa666` (§6.4) — both smokes confirm the fix.**

Investigation records and per-row forensics for the 2026-05-15 work are filed under `docs/paras/runs/20260515_rollout_matrix/`.

## 8. Limitations and Future Work

1. **Decode-only window.** Prefill iterations are not counted toward the moving average. This isolates steady-state decode load, but means a prefill-heavy workload (many short prompts in a tight loop) won't accumulate evidence to fire — even though prefill cost is also affected by EP vs TP. A future refinement could add a separate prefill window or a unified token-budget metric.
2. **No automatic policy tuning.** Defaults are taken from the design-doc crossover band on a fixed 8×A100 reference. A model-specific or hardware-specific policy (e.g., probing the actual crossover at startup) would improve generality.
3. **Cooldown is a fixed wall-clock duration.** Adaptive cooldown (longer after spurious fires, shorter when load is clearly trending) would reduce missed switches under bursty load.
4. **No metric export.** Switch events are only visible in the scheduler log via `ParaS auto-switch policy fired: ... -> ...` and `Time taken to configure TP/EP: ... ms`. A Prometheus counter would help operators observe the policy's behavior.
5. **The policy assumes batched workload.** A single long-running streaming request that decodes one token at a time will fire EP→TP after `window` decode iterations — which is correct, but the decode policy's default `threshold = 64 * world_size` was chosen for production batches and effectively makes any single-stream workload force TP. Operators serving primarily latency-sensitive streaming should override `--paras-auto-switch-threshold` to a smaller value or disable auto-switch entirely.

## 9. References

- `parallelism_switch.md` — base EP↔TP switch design (gather/scatter, weight transfer, N+1 slot, control plane)
- `parallelism_configuration.md` — why DP/EP and TP/TP are the two practical configurations and where the crossover is
- `cuda_graph.md` — dual graph capture and per-mode state preservation
- `gpt_oss_support.md` — model-specific adaptations including the in-flight switch correctness chronicle
- `runs/20260515_rollout_matrix/` — investigation records for the four rollout-path bugs documented in §6 (cuda-graph max_bs, rank-0 waiting, sampler sync, log spam) plus the original 2026-05-14 background and opening prompt that bootstrapped the work
