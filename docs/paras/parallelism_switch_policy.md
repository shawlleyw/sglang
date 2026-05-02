# ParaS Automatic Parallelism Switching

## 1. Overview

The base ParaS design (`parallelism_switch.md`) provides EP↔TP switching as a manually-driven primitive: an operator hits `/paras_configure_tp` or `/paras_configure_ep` and the runtime swaps modes in ~250–300 ms. That works for orchestrated deployments, but it leaves the decision of *when* to switch to an external system that has to track per-iteration batch sizes, apply hysteresis, and avoid flapping.

This document describes the in-process policy that automates the decision and triggers the switch through the same control plane the HTTP route uses. The policy is on by default whenever `--enable-paras-moe` is set; the HTTP endpoints continue to work and bypass the policy.

The implementation lives in:
- `python/sglang/srt/paras/scheduler_paras_mixin.py` — `ParasAutoSwitchPolicy` class, `paras_auto_observe`, `paras_auto_pick_signal`
- `python/sglang/srt/managers/io_struct.py` — `ParaSAutoSwitchReq` message type
- `python/sglang/srt/managers/scheduler.py` — observe + signal-emit hook in `event_loop_normal`
- `python/sglang/srt/managers/detokenizer_manager.py` — pass-through forwarder
- `python/sglang/srt/managers/tokenizer_communicator_mixin.py` — handler that calls existing `paras_configure_tp/ep`
- `python/sglang/srt/paras/gather_manager.py` and `scatter_manager.py` — extended to preserve `waiting_queue` across the switch

## 2. When to Switch — Policy Definition

The policy is a sliding-window hysteresis controller with a wall-clock cooldown.

### 2.1 Inputs and tunables

| CLI flag | Field | Default | Meaning |
|---|---|---|---|
| `--paras-auto-switch` | `paras_auto_switch` | `True` (when `--enable-paras-moe`) | Master enable. |
| `--paras-auto-switch-low` | `paras_auto_switch_low` | `256` | Switch **EP→TP** when sliding-window avg global batch < this. |
| `--paras-auto-switch-high` | `paras_auto_switch_high` | `1024` | Switch **TP→EP** when sliding-window avg global batch > this. |
| `--paras-auto-switch-window` | `paras_auto_switch_window` | `32` | Sliding-window size (decode iterations). |
| `--paras-auto-switch-cooldown-sec` | `paras_auto_switch_cooldown_sec` | `60.0` | Wall-clock seconds between successive switches. |

The defaults match the crossover band documented in `parallelism_switch.md` (TP wins ≤ 512, EP wins ≥ 1024 on 8×A100). Validation requires `0 < low < high`, `window > 0`, `cooldown_sec >= 0`, and the flags only apply when `--enable-paras-moe` is set.

### 2.2 Per-iteration observation

After each forward iteration that results in `process_batch_result(batch, result)`, the scheduler calls `paras_auto_observe(batch)`. The observation logic:

1. Skip if no policy is initialized.
2. Skip if `batch.forward_mode` is not decode (prefill iterations are not counted).
3. Compute the **global batch size**:
   - If `batch.global_num_tokens` is set (EP mode under DP attention, populated by `prepare_mlp_sync_batch_raw`), use `sum(batch.global_num_tokens)`.
   - Otherwise (TP mode, where `dp_size=1` skips the all-gather), use `len(batch.reqs)`. In TP mode the local batch *is* the global batch since all ranks hold the same request set.
4. Skip if the global batch is non-positive.
5. Append to the policy's `deque` (capped at `window`).

### 2.3 Decision evaluation

After `paras_auto_observe`, the scheduler calls `paras_auto_pick_signal()`. Algorithm:

```
if now < cooldown_until: return None
if len(window) < window.maxlen: return None       # window not yet full
avg = sum(window) / len(window)
if mode == "EP" and avg < low:  target = "TP"
elif mode == "TP" and avg > high: target = "EP"
else: return None
cooldown_until = now + cooldown_sec
window.clear()
return target
```

Hysteresis (low ≠ high) guarantees that batches in `[low, high]` produce no switch — the controller is stable in the dead zone. The cooldown bounds switch frequency to once per `cooldown_sec` in the worst case, capping the cost of false positives at `≤ switch_latency / cooldown_sec` (e.g., `300 ms / 60 s = 0.5 %` overhead).

### 2.4 Where the policy lives — every rank

The policy is constructed on **every scheduler rank** (not just rank 0). This handles two structural asymmetries:

| Mode | Which rank sees decode iterations? |
|---|---|
| EP (DP attention) | Whichever DP rank received the request (each DP rank has its own queue). |
| TP (DP attention disabled) | All ranks process the same replicated batch in lockstep. |

If the policy were only on rank 0, then in EP mode any request load-balanced to rank 1/2/3 would never feed observations into rank 0's policy. With the policy on every rank, whichever rank is actually running iterations contributes the data.

Two side-effects of this design — duplicate signals from concurrent fires across ranks, and stale window state across mode changes — are addressed by the idempotency guard (§4) and the anti-flap reset (§5).

## 3. How to Trigger — Control-Plane Flow

The auto-switch reuses the existing HTTP path's control plane verbatim. Both flows converge at `TokenizerManager.paras_configure_tp/ep`, which is the single source of truth for adjusting `_fan_out`, the DataParallelController worker list, and the per-scheduler `paras_configure_tp/ep` execution.

### 3.1 Side-by-side comparison

```
HTTP path                                Auto-switch path
──────────                                ────────────────
GET /paras_configure_tp                   Scheduler rank N decodes
                                          ↓
                                          paras_auto_observe(batch)
                                          ↓
                                          policy.pick_target() → "TP"
                                          ↓
                                          send_to_tokenizer.send_output(
                                            ParaSAutoSwitchReq(target=...))
                                          ↓
                                          DetokenizerManager
                                            (pass-through forwarder)
                                          ↓
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

The new `ParaSAutoSwitchReq` is the only added message type. Its handler in `TokenizerManager._result_dispatcher` is a two-line dispatch:

```python
def _handle_paras_auto_switch_req(self, req):
    if req.target == ParaSConfigureReqType.CONFIGURE_TP:
        asyncio.create_task(self.paras_configure_tp())
    elif req.target == ParaSConfigureReqType.CONFIGURE_EP:
        asyncio.create_task(self.paras_configure_ep())
```

This is intentional: the auto path adds *one* hop in front of the HTTP entrypoint, leaving everything downstream unchanged.

### 3.2 Why route through the TokenizerManager

A naive auto-switch could shortcut by injecting `ParaSConfigureReqInput` directly into the scheduler-side broadcast. That implementation **breaks correctness**: TokenizerManager still has `_fan_out = dp_size` and DataParallelController still has the EP worker list. After the schedulers swap to TP, new requests would continue to be load-balanced to all `dp_size` ranks instead of being directed to TP rank 0. Sub-rank-0 ranks would queue requests they no longer process, and the TP forward pass would deadlock waiting for collective participation from ranks that have nothing to contribute.

Routing through TokenizerManager preserves the three coordinated state changes the HTTP path performs:

| Component | What it changes |
|---|---|
| TokenizerManager | `comm._fan_out` (1 in TP, `dp_size` in EP) — controls how many ranks receive future requests. |
| DataParallelController | `self.workers` slice (`paras_tp_workers` = every Nth, `paras_ep_workers` = all). |
| Schedulers | `paras_parallelism_config`, `tp_size`, `tp_group`, `attn_tp_*`, KV pool, attention backend, weight layout, CUDA graph set. |

### 3.3 Where the signal is emitted

The signal-emit site is in `event_loop_normal` immediately after `process_batch_result`:

```python
if batch:
    result = self.run_batch(batch)
    self.process_batch_result(batch, result)
    self.paras_auto_observe(batch)
    signal = self.paras_auto_pick_signal()
    if signal is not None:
        self.send_to_tokenizer.send_output(signal)
```

Position matters: the observation feeds the policy with the just-completed iteration's batch size, and the signal is emitted **between** iterations — never mid-forward. This matches the existing HTTP-triggered switch boundary, which also fires only between iterations because `recv_requests` runs at the top of the loop.

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

In TP mode only rank 0 has a non-empty `waiting_queue`. The scatter partitions it across the new EP ranks using the same greedy strategy already used for the running batch:

```python
class ParaSReqScatterManager:
    def __init__(self, global_reqs, ..., local_waiting_reqs=None):
        # Only rank 0 populates local_waiting_reqs in TP mode. Other ranks
        # send [] into the all-gather, but every rank receives the union,
        # so all ranks can deterministically run the same partition algorithm.
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

In EP mode, multiple DP ranks may decide to switch in the same iteration (e.g., all four ranks observe the same global batch via `global_num_tokens` and reach the same conclusion). All four signals reach TokenizerManager, which fires four `paras_configure_tp` async tasks. The first task adjusts `_fan_out`, dispatches to schedulers, and completes the switch. The next three would otherwise re-execute the gather/scatter on a system that has already swapped modes, corrupting state.

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

### 5.2 Window-clear-on-switch to suppress flapping

Without this, the following sequence flaps:

1. Rank 0 fires EP→TP. Its window clears, its cooldown is set.
2. The configure broadcasts to all ranks. Other ranks transition to TP, but their policy state is not touched — their windows still hold pre-switch EP-mode observations.
3. The first iteration in TP mode gathers in-flight requests into a single batch. Other ranks observe one or two TP-mode samples appended to their stale EP-mode windows.
4. If the stale + fresh samples exceed `high`, those ranks immediately fire TP→EP, undoing the switch.

The fix is symmetric to step 1: every `paras_configure_tp/ep` call clears its rank's policy window and extends the cooldown by `cooldown_sec`. This applies on **every** rank that runs the configure (not just the originating rank), so the broadcast naturally resets all peers:

```python
def _paras_auto_clear_window_on_switch(self) -> None:
    policy = getattr(self, "_paras_auto_policy", None)
    if policy is None:
        return
    policy.window.clear()
    policy.cooldown_until = max(
        policy.cooldown_until, time.time() + policy.cooldown_sec)
```

Combined with the idempotent guard, the system can absorb any number of duplicate signals from concurrent fires, and stale state from any prior mode never drives a reverse switch.

## 6. Verified Behavior

End-to-end test on 4×A100 with gpt-oss-120b-bf16, test-scale thresholds (`low=2 high=8 window=4 cooldown=15s`), three workload phases:

| Time | Phase | Trigger | Switch fired | Latency |
|---|---|---|---|---|
| `T+0` | 1 single small request (80 tokens) | avg=1 < low=2 | EP→TP | 270 ms |
| `T+20s` | 32 concurrent requests (300 tokens each) | avg > high=8 in TP | TP→EP | 263 ms |
| `T+35s` | 5 single trickle requests | avg=1 < low=2 | EP→TP | 267 ms |

All 38 requests (1 + 32 + 5) returned successful coherent text. Three switches, each respecting the 15 s cooldown. Zero flapping (no spurious reverse fires immediately after the primary direction). Zero errors in the server log.

For production-scale validation, the defaults (`low=256 high=1024 window=32 cooldown_sec=60`) require sustained traffic at the corresponding global batch sizes; the same control-plane flow applies.

## 7. Limitations and Future Work

1. **Decode-only window.** Prefill iterations are not counted toward the moving average. This isolates steady-state decode load, but means a prefill-heavy workload (many short prompts in a tight loop) won't accumulate evidence to fire — even though prefill cost is also affected by EP vs TP. A future refinement could add a separate prefill window or a unified token-budget metric.
2. **No automatic policy tuning.** Defaults are taken from the design-doc crossover band on a fixed 8×A100 reference. A model-specific or hardware-specific policy (e.g., probing the actual crossover at startup) would improve generality.
3. **Cooldown is a fixed wall-clock duration.** Adaptive cooldown (longer after spurious fires, shorter when load is clearly trending) would reduce missed switches under bursty load.
4. **No metric export.** Switch events are only visible in the scheduler log via `ParaS auto-switch policy fired: ... -> ...` and `Time taken to configure TP/EP: ... ms`. A Prometheus counter would help operators observe the policy's behavior.
5. **The policy assumes batched workload.** A single long-running streaming request that decodes one token at a time will fire EP→TP after `window` decode iterations — which is correct, but the `low` default of 256 was chosen for production batches and effectively makes any single-stream workload force TP. Operators serving primarily latency-sensitive streaming should set `low` very small or disable auto-switch entirely.

## 8. References

- `parallelism_switch.md` — base EP↔TP switch design (gather/scatter, weight transfer, N+1 slot, control plane)
- `parallelism_configuration.md` — why DP/EP and TP/TP are the two practical configurations and where the crossover is
- `cuda_graph.md` — dual graph capture and per-mode state preservation
- `gpt_oss_support.md` — model-specific adaptations including the in-flight switch correctness chronicle
