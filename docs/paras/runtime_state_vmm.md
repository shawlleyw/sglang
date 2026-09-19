# Mode-local graph state and optional VMM backing

This change has two parts. The sizing/isolation fix is always enabled for
ParaS; physical backing suspension requires `--paras-vmm-runtime-states`.
Only CPU tests have been run on this branch. CUDA capture/replay, actual HBM
savings, and switch latency still need GPU validation.

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
planned GPT-OSS comparison. That flag disables the separate SWA memory pool,
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

For the earlier example (`context=131072`, `vocab=201088`, EP capture 256,
TP capture 2048), the two VMM-managed buffers require about **0.442 GiB in EP**
or **3.534 GiB in TP**, before allocation-granularity rounding. Without VMM,
the corrected separately sized buffers retain about **3.976 GiB** together.
These are shape calculations per rank, not measured HBM results, and exclude
all other memory. The removed verification masks are additional savings.

VMM-owned memory is outside PyTorch's caching allocator counters. Use device
used/free memory plus the logged `ParaS runtime VMM` resident/virtual byte
counts; a lower `torch.cuda.memory_allocated()` alone is not evidence of lower
HBM usage. GPU measurements must match graph sizes, runtime limits, context,
attention backend, SWA setting, and total memory budget across static EP,
static TP, and ParaS.

The KV planner is not resized by a VMM mode switch. Reclaiming inactive backing
provides headroom; it does not automatically create KV slots. Nor does this
change port the earlier uncommitted TP MoE scratch sizing work from the other
checkout: this branch's TP MoE planner still reserves its 64K-token chunk bound.
Consequently, equal total overhead or equal KV capacity to static baselines is
not yet established.

## CPU validation

Run with GPUs hidden and the clone first on `PYTHONPATH`:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/python" python -m pytest -q \
  test/srt/paras/test_runtime_memory.py \
  test/srt/paras/test_runtime_states.py \
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
