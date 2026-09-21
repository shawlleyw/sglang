# Mode-local graph state and optional VMM backing

ParaS keeps graph and attention state separate for EP and TP. Mode-local sizing
is always enabled; releasing inactive physical backing requires the opt-in
`--paras-vmm-runtime-states` option.
CUDA remapping and graph replay have also been validated on eight A100s with
dummy GPT-OSS weights, including live-KV EP/TP roundtrips. Saved local artifacts
record the tested configuration and memory results. The current comparison is
documented in [the evaluation methodology](memory_evaluation.md); it does not
establish latency or throughput performance for other settings.

## Mode-local state

Previously the graph runner allocated for `max(EP batches, TP batches)`, and
Triton inherited that maximum for both saved attention states. Reconfiguration
also allocated replacement attention buffers immediately before restoring the
saved ones. EP therefore retained two TP-sized KV-index arrays, two large
verification masks, and one TP-sized logits array.

Now each graph set owns its own inputs, CPU sequence lengths, logits, attention
state, and graph pool. EP uses its capture maximum and TP uses its own maximum.
The matching references are restored together on a switch; Triton reuses its
saved state without allocating temporary replacements. FlashInfer also saves
its graph backing buffers alongside its wrapper metadata. Graph-private pools
remain separate. Shared LoRA metadata retains its previous maximum capacity.

Triton attention scratch in the unified planner likewise uses each mode's graph
and runtime request limits. The large Triton custom mask is allocated only when
speculative verification is configured; this change also applies to native
static runs so that the comparison stays consistent.

## Optional physical backing suspension

Pass `--paras-vmm-runtime-states` with ParaS and explicit
`--attention-backend triton`. The option requires CUDA graphs, PP=1, and no
speculative decoding, two-batch overlap, PDMux, torch compile, or memory saver.
It defaults off. It can be used with `--disable-hybrid-swa-memory` for the
GPT-OSS memory comparison. That flag disables the separate SWA memory pool,
not the model's sliding-window attention semantics.

The managed allocations are exactly:

- Triton's main `cuda_graph_kv_indices` (`int64`).
- The graph runner's `next_token_logits_buffer` (`float32`).

They are allocated with CUDA VMM from the beginning; existing PyTorch allocator
blocks are never unmapped. `cuMemAddressReserve` owns a stable virtual range.
`cuMemCreate`, `cuMemMap`, and `cuMemSetAccess` establish device-local backing;
the allocation handle is then released so that unmapping actually frees pages.
PyTorch tensors alias those addresses through the CUDA array interface, whose
owner retains the allocation object.

At a switch, the scheduler first drains its overlap pipeline. The VMM manager
also synchronizes the device, unmaps the outgoing mode, maps and clears the
target mode's ranges, and synchronizes initialization before graph replay.
It does not hold both modes' physical backing simultaneously. Virtual
reservations, tensors, and graph executables survive. Indices are regenerated
by attention metadata preparation and logits by each forward pass. Old scratch
contents are intentionally discarded. A failed remap aborts activation and
blocks replay; the worker must restart rather than use an unmapped pointer.

This does not suspend weights, live KV, request-to-token mappings, numerical
scratch inside the UMM, DeepEP/NCCL buffers, or graph-private intermediates.
The optional SWA index array and small graph inputs also remain resident.
There is no CPU backup or migration of live state in this allocator.

## Accounting and limits

For GPT-OSS (`context=131072`, `vocab=201088`, EP capture 256, TP capture
2048), the measured VMM-backed regions are **0.443 GiB in EP** and
**3.535 GiB in TP**, including mapping-granularity rounding. Only the active
region is physically backed. VMM off retains both mode-local buffers. These
figures cover KV-index/logits buffers only, not total HBM consumption.

Releasing inactive VMM backing does not release cached PyTorch blocks. A switch
can temporarily retain the previous workload's allocator cache while mapping the
target mode's larger backing. Include activation snapshots when reporting the
largest observed residency; one-second sampling can miss that transient.

VMM-owned memory is outside PyTorch's caching allocator counters. Use device
used/free memory plus the logged `ParaS runtime VMM` resident/virtual byte
counts; a lower `torch.cuda.memory_allocated()` alone is not evidence of lower
HBM usage. GPU measurements must match graph sizes, runtime limits, context,
attention backend, SWA setting, and total memory budget across static EP,
static TP, and ParaS.

The KV planner is not resized by a VMM mode switch. Reclaiming inactive backing
provides headroom; it does not automatically create KV slots.

The shared request-to-token backing table is capped by the configured
`max_running_requests`, just like the native request pool. Scheduler admission
still divides that limit across EP ranks. The table is outside UMM and is not
managed by VMM.

TP MoE scratch remains inside UMM, but its reservation follows the configured
prefill/chunked-prefill, running-request, and TP graph sizes (including draft
multiplicity when configured), up to the kernel's 64K input-token ceiling.
With no explicit running-request cap, planning uses the native pool's 4096-row
automatic ceiling. This is a reservation target, not a new admission limit:
an unchunked first prefill may exceed `max_prefill_tokens`. The existing runner
falls back to runtime scratch allocations when the requested views do not fit,
without writing outside the UMM reservation or changing the kernel chunk loop.
Reducing UMM scratch can increase KV capacity, although transfer headroom may
still set the minimum front/tail size. VMM itself does not resize this layout.

## Separate EP and TP prefill budgets

For the matched A100 GPT-OSS memory evaluation, native EP uses
`--max-prefill-tokens 2048` per DP rank, and native TP uses
`--max-prefill-tokens 8192`. ParaS uses `--max-prefill-tokens 2048` plus
`--paras-tp-max-prefill-tokens 8192`. The TP override controls both scheduler
admission and TP MoE workspace sizing, including after EP/TP switches. Without
the override, both modes retain the shared `--max-prefill-tokens` behavior.

These are scheduling budgets, not strict per-forward limits: an unchunked
request larger than the budget may run alone and use dynamic workspace fallback.
ParaS currently requires unchunked prefill because migration does not preserve
mid-chunk request state.

## CPU validation

The standalone [reconfiguration benchmark](../../benchmark/paras/reconfigure/README.md#compare-vmm-off-and-on)
accepts `--vmm off|on|both` for comparisons of switch latency and boundary memory.
It covers all five switching methods, preserves their Weights/Graph/Others
breakdown, and records VMM resident/virtual bytes separately from PyTorch memory.
Native engine restart is one shared reference labeled VMM not applicable.
The full method follows production graph activation; recapture methods use a
benchmark-only scratch-reuse adapter, with no changes to normal serving code.
New recapture configurations need GPU validation before using their results.

Run with GPUs hidden and the clone first on `PYTHONPATH`:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/python" python -m pytest -q \
  test/srt/paras/test_runtime_memory.py \
  test/srt/paras/test_runtime_states.py \
  test/srt/paras/test_runtime_sizing.py \
  test/srt/paras/test_unified_attention_workspace.py \
  test/srt/paras/test_unified_workspace_layout.py \
  test/srt/paras/test_unified_workspace_views.py
```

The state tests execute production methods on CPU tensors, excluding GPU-heavy
module imports via AST extraction. Driver tests mock CUDA calls and check mapping
ownership, handle release, stable addresses, synchronization ordering, and failure
cleanup. These tests cannot validate real CUDA graph replay, pointer acceptance
by PyTorch, or multi-GPU correctness. The ctypes structure layout is separately
checked against the installed CUDA headers using a CPU C compiler.
