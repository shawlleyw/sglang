---
name: paras-test-auto-switch
description: Test ParaS load-driven autoswitch policy. Launch with autoswitch enabled and a tuned threshold (decode policy with threshold=8 for the canonical 32-burst load), then drive scripts/paras/eval/paras_cmd/autoswitch_test.sh: light pre-burst prompt (verify clean baseline), 18s cooldown, 32-burst diverse prompts (load triggers EP↔TP autoswitch mid-burst), light post-burst prompt (verify state coherent post-autoswitch), assert ≥1 'ParaS auto-switch policy fired' event in server log, scan errors. Use after touching paras/ scheduler or autoswitch policy code, complementary to paras-test-manual-switch which exercises the manual /paras_configure_* HTTP path.
---

# Test ParaS Load-Driven Autoswitch

Verifies the ParaS autoswitch policy fires correctly under load and the
resulting EP↔TP transitions preserve correctness for both in-flight burst
requests and subsequent fresh prompts.

## When to use

- After any change to the ParaS autoswitch policy
  (`python/sglang/srt/paras/scheduler_paras_mixin.py` autoswitch handler,
  policy thresholds, cooldown logic)
- After any change to paras/ scheduler request-counting / load-tracking code
- Complementary to [`paras-test-manual-switch`](file:///home/shaoyuw/sglang/.skills/paras-test-manual-switch/SKILL.md):
  manual exercises the HTTP `/paras_configure_*` path; autoswitch exercises
  the load-driven policy path. Bugs in only the policy decision (not the
  switch mechanism itself) are caught here, not by manual testing.

## Supported models

Same model matrix as `paras-test-manual-switch` — both qwen3 and gpt-oss are
supported. See that skill's "Supported models" table for the per-model
differences (launch script, `MODEL_NAME`, `LOG_FILE`, `TIMEOUT_TRIES`,
attention backend, cuda graph mode, model file path). Only the launch
invocation and a few env vars differ; the autoswitch_test.sh procedure itself
is model-agnostic.

## Autoswitch threshold tuning (CRITICAL)

The autoswitch policy fires when the sliding-window average crosses a single threshold. The test must pick a threshold that **reliably straddles** the test's light load (~1 decode req) and burst load (~32 decode reqs), otherwise the test is meaningless (autoswitch never triggers).

The canonical settings for the default `BURST_SIZE=32` test on the `decode` policy:

```
--paras-auto-switch-policy decode \
--paras-auto-switch-threshold 8 \
--paras-auto-switch-window 4 \
--paras-auto-switch-cooldown-sec 15
```

**Rationale**:
- `policy=decode`: observes pure-decode iterations only; metric is global decode batch size (= request count).
- `threshold=8`: light load avg ~1 is `< 8` (fires EP→TP); 32-burst avg ~32 is `> 8` (fires TP→EP). The threshold sits in the middle of the test's load range with a 4× safety margin on both sides.
- `window=4`: short load-sampling window so the policy reacts quickly to burst onset within the test's runtime budget.
- `cooldown-sec=15`: short enough that the post-burst light prompt can trigger another switch if the policy decides; long enough to avoid thrashing during the burst.

If you change `BURST_SIZE`, scale `--paras-auto-switch-threshold` so that `BURST_SIZE > threshold > light_load`. With `BURST_SIZE=32` and `light=1`, `threshold=8` gives 4× margin on each side.

For the `prefill` policy (not exercised in this test by default), the metric is global prefill tokens; the default threshold scales differently (1024 × world_size). To test prefill, use prompts long enough that summed token count exceeds your chosen threshold within the window.

## Prerequisites

- **Conda env**: `sgl_paras`
- **GPUs**: 4× A100-80GB (qwen3) or 4-8× A100-80GB (gpt-oss)
- **Working dir**: this repo

## Quick Start (Qwen3-30B-A3B)

```bash
conda activate sgl_paras
cd /home/shaoyuw/sglang
pip install -e python/ -q --no-deps

bash scripts/paras/eval/paras_cmd/kill.sh

ENABLE_PARAS=1 NUM_GPUS=4 MEM_FRACTION_STATIC=0.7 \
    bash scripts/paras/eval/a100/qwen/launch_server_dp_ep.sh \
    --paras-auto-switch-policy decode \
    --paras-auto-switch-threshold 8 \
    --paras-auto-switch-window 4 \
    --paras-auto-switch-cooldown-sec 15 \
    2>&1 | tee /tmp/sglang_paras_test.log &

bash scripts/paras/eval/paras_cmd/wait_ready.sh
bash scripts/paras/eval/paras_cmd/health.sh
bash scripts/paras/eval/paras_cmd/autoswitch_test.sh

bash scripts/paras/eval/paras_cmd/kill.sh
```

(Autoswitch is **enabled by default** when `--enable-paras-moe` is set; `--paras-auto-switch-policy decode` is also the server default. The threshold/window/cooldown flags above override the policy's larger defaults — for production decode policy uses `threshold=64*world_size`, `window=32`, `cooldown=60s`, which would not fire within this test's runtime budget.)

## Quick Start (gpt-oss-120b-bf16, 4×A100)

```bash
conda activate sgl_paras
cd /home/shaoyuw/sglang
pip install -e python/ -q --no-deps

export MODEL_NAME=gpt-oss-120b-BF16-unsloth
export LOG_FILE=/tmp/sglang_paras_gptoss.log
export TIMEOUT_TRIES=60
export SLEEP_BETWEEN=10

bash scripts/paras/eval/paras_cmd/kill.sh

# 4-GPU: MEM_FRACTION_STATIC=0.8 (per-rank weights ≈ 57 GiB at TP=4)
ENABLE_PARAS=1 NUM_GPUS=4 MEM_FRACTION_STATIC=0.8 \
    bash scripts/paras/eval/a100/gptoss/launch_server_dp_ep.sh \
    --paras-auto-switch-policy decode \
    --paras-auto-switch-threshold 8 \
    --paras-auto-switch-window 4 \
    --paras-auto-switch-cooldown-sec 15 \
    2>&1 | tee "$LOG_FILE" &

bash scripts/paras/eval/paras_cmd/wait_ready.sh
bash scripts/paras/eval/paras_cmd/health.sh
bash scripts/paras/eval/paras_cmd/autoswitch_test.sh

bash scripts/paras/eval/paras_cmd/kill.sh
```

## Quick Start (gpt-oss-120b-bf16, 8×A100 — canonical deployment)

Same as above except for the launch line:

```bash
ENABLE_PARAS=1 NUM_GPUS=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MEM_FRACTION_STATIC=0.7 \
    bash scripts/paras/eval/a100/gptoss/launch_server_dp_ep.sh \
    --paras-auto-switch-policy decode \
    --paras-auto-switch-threshold 8 \
    --paras-auto-switch-window 4 \
    --paras-auto-switch-cooldown-sec 15 \
    2>&1 | tee "$LOG_FILE" &
```

## Detailed Procedure

`autoswitch_test.sh` runs 7 internal phases. The shape is similar to the
manual `e2e_test.sh` but uses load to trigger switches instead of HTTP
configure calls. Caller is responsible for steps 1, 2, 4 (kill, launch,
final cleanup).

### 1. Kill any existing sglang processes

```bash
bash scripts/paras/eval/paras_cmd/kill.sh
```

### 2. Launch with autoswitch tuned

See Quick Start blocks above. The launch must pass
`--paras-auto-switch-policy decode --paras-auto-switch-threshold 8` (or
otherwise scaled to your `BURST_SIZE`) and tee its log to `$LOG_FILE` so
phase 6 can grep for autoswitch events.

### 3. Wait for server ready and verify health

```bash
bash scripts/paras/eval/paras_cmd/wait_ready.sh
bash scripts/paras/eval/paras_cmd/health.sh
```

For gpt-oss cuda-graph variant, also run `check_log.sh cuda_graph` to
confirm dual capture (see manual-switch skill step 4).

### 4. Run the autoswitch test driver

```bash
bash scripts/paras/eval/paras_cmd/autoswitch_test.sh
```

Internal phases:

| Phase | What it does | Pass criterion |
|---|---|---|
| 1 | Send a single short prompt ("List three primary colors and one example object for each.", `LIGHT_MAX_TOKENS=80`) and verify the response is non-empty + non-degenerate. | Coherent response (≥10 chars, no `(\b\w+\b)(\s+\1){5,}` attractor). Establishes a clean baseline before any autoswitch fires. |
| 2 | 18 s cooldown (configurable via `COOLDOWN_SEC`). | Lets any spurious post-phase-1 autoswitch settle. With `--paras-auto-switch-cooldown-sec 15`, the next switch is unblocked. |
| 3 | Fire `BURST_SIZE=32` parallel diverse prompts (max_tokens=200 each). Load=32 should exceed `--paras-auto-switch-threshold=8` and trigger a TP→EP switch (after the policy enters TP at phase 1). | All requests return; none time out. |
| 4 | Run `paras_cmd_burst_verify` on the burst responses. | 0 / 32 degenerate (no response matches the attractor regex). Catches mid-burst KV cache corruption from migration during autoswitch. |
| 5 | Sleep 3 s to let any post-burst autoswitch settle, then send another light prompt and verify coherence. | Coherent response. Catches state corruption that survives the autoswitch event but only manifests on subsequent fresh prompts (e.g., stale weights, broken graph metadata). |
| 6 | Grep `$LOG_FILE` for `"ParaS auto-switch policy fired"`. | At least 1 event found. If none, autoswitch never fired and the test is meaningless — phases 3-5 didn't actually exercise the switch path. Most common cause: thresholds set too high for the burst load. |
| 7 | Run `check_log.sh errors`. | No `error\|exception` lines (filtering known-benign warnings). Catches scheduler exceptions, NVLink IPC errors, KV pool memory leak detection (which can fire after a buggy migration). |

### 5. Cleanup

```bash
bash scripts/paras/eval/paras_cmd/kill.sh
```

## Pass / Fail Criteria

| Check | Pass | Fail |
|---|---|---|
| Pre-burst light prompt (phase 1) | Coherent ≥10 chars | empty / degenerate / timeout |
| Burst responses (phase 4) | 0 / `BURST_SIZE` degenerate | any degenerate response |
| Post-burst light prompt (phase 5) | Coherent ≥10 chars | empty / degenerate |
| Autoswitch fired (phase 6) | ≥ 1 `"ParaS auto-switch policy fired"` event | 0 events (test invalid; tune thresholds) |
| Server errors (phase 7) | None | any scheduler exception, NVLink, CUDA error, KV leak detection |

A run with 0 burst degeneration but 0 autoswitch events is a **failed test
configuration**, not a passing test — `autoswitch_test.sh` returns non-zero
to surface this. Tune `--paras-auto-switch-threshold` so it sits between
your expected light-load and burst-load values.

## Important Notes

- **`MEM_FRACTION_STATIC` depends on (model, GPU count)** — same matrix as
  paras-test-manual-switch. Override on every launch invocation:

  | Model | GPUs | `MEM_FRACTION_STATIC` |
  |---|---|---|
  | Qwen3-30B-A3B | 4 or 8 | 0.7 |
  | gpt-oss-120b-bf16 | 4 | 0.8 |
  | gpt-oss-120b-bf16 | 8 | 0.7 |

  At an undersized fraction, the server boots through weight load and then
  dies in `init_memory_pool` with `kv_budget=0.000GiB`.
- **`--disable-radix-cache`, `--chunked-prefill-size -1`, and
  `--disable-overlap-schedule` are all mandatory** (baked into the launch
  scripts under `ENABLE_PARAS=1` and enforced by ParaS init assertions in
  `server_args._check_paras_config` and runtime asserts in
  `scheduler_paras_mixin`). Same constraints as manual switch:
  - **Radix cache disabled**: ParaS uses `ChunkCache` / `SWAChunkCache`;
    radix tree state would not survive `tree.reset()` at switch.
  - **Chunked prefill disabled**: ParaS migration cannot preserve
    mid-chunked-prefill state.
  - **Overlap scheduler disabled**: switching mid-overlap would require
    migrating an in-flight forward's intermediate state.
- **Burst size and threshold tuning interact**: `BURST_SIZE` must be
  comfortably greater than `--paras-auto-switch-threshold` for the TP→EP
  direction to fire on the burst. With `BURST_SIZE=32` and `threshold=8`,
  there's a 4× margin — safe. If you reduce `BURST_SIZE` (e.g., for
  memory-constrained smoke tests), reduce `threshold` proportionally.
- **The pre-burst light prompt purpose**: catches boot-time correctness
  issues independently. If the model is broken at boot (before any switch),
  phase 1 fails and the test stops. This separates "boot is broken" from
  "autoswitch is broken".
- **The post-burst light prompt purpose**: catches state corruption that
  persists past the burst. If the autoswitch event left the server in a
  silently-broken state, fresh prompts after the burst would degenerate.
  Phase 5 is the dedicated probe for this.
- **Phase 6 is critical for test validity**, not just for failure detection.
  Without phase 6, a misconfigured threshold could let phases 3-5 pass
  trivially (no migration ever happened, so nothing could go wrong). The
  log scan ensures the policy actually exercised at least once during the
  burst.

## Known Failure Modes

1. **Phase 6 fails: "no 'ParaS auto-switch policy fired' events"**:
   - Most common: launch flags missing or threshold set too high. Confirm
     `--paras-auto-switch-policy decode --paras-auto-switch-threshold 8`
     are on the launch line and `BURST_SIZE >= 16`.
   - LOG_FILE path mismatch: the launch script's `tee` target must match
     `$LOG_FILE` env var the helpers see. Check with `cat $LOG_FILE | head`.
   - Autoswitch globally disabled: check launch did NOT pass
     `--no-paras-auto-switch`.

For per-fire verbose diagnostics (window observations, avg, threshold,
class name), grep the launch log for `"ParaS ["` — every policy fire emits
a line like `ParaS [DecodeAutoSwitchPolicy] policy fired: EP -> TP at t=... | observations=[1,1,1,1] avg=1.00 threshold=8 window_maxlen=4 cooldown_sec=15.0`. This is the source of truth for verifying the policy made the right decision.

2. **Phase 4 fails (burst degenerate) while phase 1 / phase 5 pass**:
   The autoswitch fired and corrupted in-flight req state. Likely culprits:
   stale `req_to_token` post-migration, SWA layer-specs misdispatch, or
   captured-graph kv_indptr staleness. Cross-reference with manual switch
   `inflight_switch.sh` results — if those also fail, the bug is in the
   migration mechanism (not the autoswitch trigger).

3. **Phase 5 fails (post-burst light degenerate) while phase 4 passes**:
   The autoswitch event left the server in a corrupted state that the
   in-flight reqs survived (because they had their own KV cache snapshots)
   but fresh prefills hit. Likely culprits: stale weights post-transfer,
   broken graph metadata not refreshed, Parameter view rebinding wrong.
   This is a rare and subtle bug — also test by running
   `paras-test-manual-switch` and seeing if step 8 (`send_prompts.sh TP`
   post-EP→TP-switch) also fails.

4. **Phase 7 reports `token_to_kv_pool_allocator memory leak detected`**:
   The migration didn't fully release KV cache slots for the migrated reqs.
   Cross-reference with the bug pattern in
   `python/sglang/srt/managers/scheduler_runtime_checker_mixin.py:check_memory`.
   Likely a `cache_finished_req` slot-range computation issue post-migration.

## Companion Skills

- [`paras-test-manual-switch`](file:///home/shaoyuw/sglang/.skills/paras-test-manual-switch/SKILL.md) —
  manual `/paras_configure_*` switch test. Run BEFORE this skill: if manual
  fails, autoswitch will fail too with less clear diagnosis. Always
  validate manual first to isolate "is the migration mechanism broken" from
  "is the autoswitch trigger broken".

## See Also

- [`scripts/paras/eval/paras_cmd/autoswitch_test.sh`](file:///home/shaoyuw/sglang/scripts/paras/eval/paras_cmd/autoswitch_test.sh) —
  the test driver. All 7 phases live there; it's the canonical autoswitch
  test orchestrator.
- [`scripts/paras/eval/paras_cmd/lib.sh`](file:///home/shaoyuw/sglang/scripts/paras/eval/paras_cmd/lib.sh) —
  shared helpers used by autoswitch_test.sh:
  `paras_cmd_load_prompts`, `paras_cmd_burst_send`, `paras_cmd_burst_verify`.
- [`scripts/paras/eval/paras_cmd/prompts_diverse.txt`](file:///home/shaoyuw/sglang/scripts/paras/eval/paras_cmd/prompts_diverse.txt) —
  32 distinct technical prompts. (Note: the previously documented "~37%
  deterministic degeneration on `Topic N`" was a `/v1/completions` endpoint
  artifact specific to gpt-oss — the model is OOD without the harmony chat
  template and never emits `<|return|>` EOS, producing chat-completion
  overrun loops the regex falsely flagged. `lib.sh` now uses
  `/v1/chat/completions` which auto-applies the harmony template; the
  attractor pattern is no longer reproducible. The diverse-prompt set is
  still the recommended default.)
- **Endpoint and response shape**: helpers in `lib.sh`
  (`paras_cmd_burst_send`, `paras_cmd_burst_verify`) now post to
  `/v1/chat/completions` with `messages=[{"role": "user", "content": ...}]`
  and read responses from
  `choices[0].message.reasoning_content + choices[0].message.content`
  (generation-order concatenation). The autoswitch_test.sh inherits this
  automatically; no per-test changes needed.
- `python/sglang/srt/paras/scheduler_paras_mixin.py` — autoswitch policy
  implementation (search for `paras_auto_switch`, `_paras_auto_switch_step`).
- `docs/paras/automatic_parallelism_switching.md` — autoswitch policy design
  doc.
