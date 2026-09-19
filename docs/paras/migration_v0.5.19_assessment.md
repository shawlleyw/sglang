# ParaS migration assessment: v0.5.5 to upstream v0.5.19

Assessed on 2026-09-08. **High difficulty, but feasible as a staged port.**
The main work is adapting runtime contracts for switching, not resolving textual
conflicts. No fundamental impossibility was established by this source review.

## Revisions and method

- Local branch: `paras_epdptp`, HEAD `8177b7260`.
- Confirmed base: v0.5.5, `0c006b8809cd99e1f95926401a2823dd952641c8`.
- Upstream target: v0.5.19, `0bcd822377da7b5718e674eaf9c870d349424dd1`.
- Upstream checkout: `/home/shaoyuw/project/sglang-v0.5.19`.
- [Official release](https://github.com/sgl-project/sglang/releases/tag/v0.5.19).

The comparison covers committed changes. Existing untracked artifacts, scripts,
and FP8 documentation were preserved; they are not included in the Git counts.
The target was cloned shallowly at the release tag. Local history was fetched
into that clone as `paras-assessment`, and `git merge-tree` used the explicit,
verified v0.5.5 base. Neither checkout was merged, rebased, or switched.

Reproduce the merge simulation inside the upstream clone:

```bash
git -c merge.renameLimit=10000 merge-tree --write-tree --merge-base=v0.5.5 --name-only HEAD paras-assessment
```

Exit code 1 denotes the expected merge conflicts. This writes temporary Git
objects but does not change the index or checked-out files.

| Measurement | Result |
|---|---:|
| Local commits after v0.5.5 | 374 |
| Local changed files | 203: 150 additions, 53 modifications |
| Local total diff | +42,116 / -244 lines |
| Runtime diff under `python/sglang/srt/` | 80 files; +15,237 / -200 lines |
| New `srt/paras/` implementation | 27 files; 11,282 lines |
| Modified existing files also changed upstream | 53 of 53 |
| Conflicting paths in simulated merge | 51 |
| Conflict breakdown | 44 content; 7 modify/delete |

New ParaS files merge textually because upstream has no corresponding files.
That does not establish compatibility: those files call many changed APIs.

## Main blockers and risks

### 1. Process-group switching and runtime configuration — critical

Current [paras_parallel_state.py](../../python/sglang/srt/paras/paras_parallel_state.py:219)
switches `_TP`, `_MOE_EP`, `_MOE_TP`, and old attention globals. Upstream now
has a dedicated `_ATTN_TP` group, and `dp_attention.py` no longer supplies the
old `get_attention_tp_rank/size/group` interface.

Upstream [runtime_context.py](/home/shaoyuw/project/sglang-v0.5.19/python/sglang/srt/runtime_context.py:185)
also distinguishes live communicator handles, published configuration sizes,
and stamped derived widths. `_ConfigBag` snapshots settings at publication;
changing `ServerArgs` afterward does not update those snapshots.

ParaS graph capture currently mutates `model_runner.server_args.dp_size`,
`ep_size`, `enable_dp_attention`, and `moe_a2a_backend`. A port retaining only
these writes can leave attention, MoE, and graph capture using different modes.
Likely failure modes are incorrect reductions, shape mismatches, and deadlocks.

Required work: make mode activation update the canonical groups, published
configuration, derived widths, dispatch state, and any cached module state
consistently. Verify rank/group identities on every rank before forwarding.

### 2. Request migration, KV ownership, and overlap scheduling — critical

ParaS gather/scatter directly reads and writes fields such as
`req.req_pool_idx` and `req.swa_evicted_seqlen`, reconstructing a running batch
from `req.seqlen - 1`. Upstream moved KV ownership into
[ReqKvInfo](/home/shaoyuw/project/sglang-v0.5.19/python/sglang/srt/managers/schedule_batch.py:849),
accessed through `req.kv`. It tracks committed and allocated lengths,
cache-protected prefixes, and SWA eviction floors separately.

Renaming fields alone is insufficient: migration must preserve these ownership
and length invariants, rebuild pool rows, and clear obsolete references. Python
can accept assignments to obsolete attribute names while the new scheduler
continues reading `req.kv`, making a partial port particularly dangerous.

The existing overlap drain is valuable, but upstream changed result processing
and future-token handling. Its output processor moved into
`managers/scheduler_components/batch_result_processor.py`. Revalidate the drain,
last-prefill merge, request redistribution, and future-map reconstruction as one
switch transaction. Failures can lose requests or corrupt the next decode step.

### 3. Unified memory manager and hybrid SWA integration — critical

ParaS owns one backing buffer with overlapping EP/TP expert and KV views. Its
capacity calculation and ordered transfers depend on exact shapes, addresses,
and allocation ownership. These algorithms can be retained, but their upstream
attachment points changed:

- `mem_cache/allocator.py` became an `allocator/` package. The generic token
  allocator remains exported; the SWA allocator has a separate module.
- `SWAKVPool` moved from `memory_pool.py` to `swa_memory_pool.py`.
- Memory sizing/pool creation is coordinated by `kv_cache_configurator.py`;
  tree construction is in `kv_cache_builder.py` and `registry.py`.
- Upstream MHA pool construction lacks ParaS's external K/V buffer arguments.
  Rebinding and resize hooks must be carried into the current pool classes.
- Upstream adds KV index translation, layout choices, and optional unified or
  post-capture allocation paths. Each enabled path must agree with ParaS's
  addresses, ownership, and capacity accounting.

Do not assume upstream's similarly named unified-memory feature replaces
ParaS's expert/KV overlap design. Establish one owner for the underlying memory
and initially use only the validated KV layout.

The unified radix tree default is **not itself a fatal blocker**. Upstream
[registry.py](/home/shaoyuw/project/sglang-v0.5.19/python/sglang/srt/mem_cache/registry.py:91)
still has `ChunkCache` / `SWAChunkCache` branches, subject to its selection
conditions. Preserve ParaS's disabled-prefix-cache behavior and explicitly test
which cache is selected with chunked prefill disabled. Prefix sharing remains
unsupported by the current ParaS design.

### 4. Dual EP/TP CUDA graphs — high

`model_executor/cuda_graph_runner.py` was deleted. The target separates runners
under `model_executor/runner/` and execution backends under `runner_backend/`.
[ModelRunner.init_cuda_graphs](/home/shaoyuw/project/sglang-v0.5.19/python/sglang/srt/model_executor/model_runner.py:1061)
now stores separate prefill and decode runners.

ParaS saves and restores an old runner's `graphs`, buffers, capture sizes,
attention state, and graph memory-pool state. This needs adaptation to the new
ownership structure, not just an import change. The target also distinguishes
prefill and decode attention backends, so patching only `attn_backend` can miss
decode state.

Start with eager execution. Then validate captures and repeated replay in both
modes, including uneven per-rank batches, capacity limits, and graph memory
usage. Graph replay can retain stale addresses even when eager switching works.

### 5. GPT-OSS DeepEP support and MoE weight layouts — high

There is a concrete missing upstream path:
[GptOssSparseMoeBlock.forward](/home/shaoyuw/project/sglang-v0.5.19/python/sglang/srt/models/gpt_oss.py:260)
raises `NotImplementedError("forward_deepep branch not implemented yet")` for
DeepEP. Your branch implements that path. It must be explicitly ported; copying
only `srt/paras/models/gpt_oss.py` will not supply the missing base method.
The upstream `forward_normal` signature changed as well.

MoE Triton kernels moved into `moe_runner/triton_utils/` and
`moe_runner/triton_kernels.py`. Port the BF16/FP8 dispatch, expert bias,
activation, scale-stride, and weight-binding changes against these implementations.
GPT-OSS's interleaved gate/up layout and Qwen's concatenated layout must continue
to match both EP and TP consumers.

A concrete optional hazard is upstream
`UnquantizedFusedMoEMethod._maybe_interleave_w13_for_fused_swiglu`: it can permute
Qwen BF16 W13 into an interleaved layout for a TP-compatible execution path.
ParaS transfer geometry assumes the model's selected layout. Keep this
optimization off initially, or teach switching and both expert modules about
the actual physical layout and its metadata.

Upstream now offers `--deepep-dispatcher-output-dtype bf16`; the legacy BF16
environment variable is deprecated but still recognized. Reassess which local
DeepEP patches remain necessary instead of carrying every old patch unchanged.
The v0.5.19 DeepEP v2 addition does not establish support for ParaS switching.

### 6. Environment and native extensions — prerequisite, compatibility unproven

The existing `sgl_paras` environment is Python 3.12.8, PyTorch 2.8.0+cu128,
Triton 3.4.0, Transformers 4.57.1, FlashInfer 0.5.0, and
`sgl-kernel` 0.3.16.post5. The host reports eight A100-SXM4-80GB GPUs and
driver 580.105.08.

The target [pyproject.toml](/home/shaoyuw/project/sglang-v0.5.19/python/pyproject.toml:1)
pins PyTorch 2.13.0, Transformers 5.12.1, FlashInfer 0.6.18 with CUDA 13 extras,
`sglang-kernel` 0.4.6.post1, `sgl-deep-ep` 0.1.2, and `sgl-deep-gemm` 0.1.7.
Its build requirements also include PyTorch 2.13.0.

Create a separate target environment and rebuild the ParaS CUDA extension.
Its checked-in build recipe targets SM80 and SM90, matching A100/Hopper, but
that does not prove compatibility with the new PyTorch/CUDA toolchain. Do not
reuse the existing CPython-3.12 `.so` merely because the Python version matches.
Driver/toolkit/package and DeepEP/NVSHMEM compatibility still need a real
installation and runtime smoke test. No driver incompatibility was established
by this review. Blackwell would additionally require revisiting the explicit
architecture flags.

## Recommended migration and validation sequence

Use a fresh v0.5.19 integration branch and port the final ParaS implementation
in coherent layers. Replaying all 374 historical commits would repeatedly
encounter obsolete code and intermediate designs.

1. **Establish the environment.** Install upstream in isolation, rebuild peer
   access, and run static Qwen TP and DP/EP baselines on the intended GPUs.
2. **Port configuration, groups, model selection, and memory binding.** Keep
   optional graph/compile/cache/layout features restricted to known paths.
3. **Prove Qwen eager switching.** Validate weights and KV bitwise across
   EP→TP→EP, followed by serving with active and queued requests. Start with
   one TP instance; then test multiple TP instances and KV-head replication.
4. **Restore scheduler features.** Validate overlap draining, prefill completion
   at the switch boundary, streaming/abort routing, and automatic switching.
5. **Port GPT-OSS and hybrid SWA.** Include DeepEP forwarding, expert bias and
   activation parity, long sequences crossing the sliding window, and pool
   accounting after completion and retraction.
6. **Restore graphs and quantized paths.** Validate dual captures/replay, FP8
   weights/scales, memory budgets, and repeated switches under changing load.
7. **Validate deployment scope.** Run multi-node reverse transfers and long
   serving tests if those are required; compare throughput, latency, switching
   cost, and capacity with fresh v0.5.19 static baselines.

Reuse the existing tests under `test/srt/paras/`, especially layout/capacity,
weight/KV round trips, TP-instance replication, SWA metadata/pool rebinding,
overlap draining, GPT-OSS CUDA graphs, and multi-node reverse transfer. Update
fixtures to exercise the new request/cache contracts; old mock-based tests
passing would not demonstrate compatibility with the new scheduler.

Preserve existing limitations initially: radix caching and chunked prefill are
disabled; TP subgroups must fit within a node for the current multi-node path.
Supporting prefix sharing, speculative decoding, other model families, or new
upstream parallel modes should be estimated separately.

## Effort estimate and decision gates

These are planning estimates for one engineer already familiar with ParaS,
with suitable GPUs and checkpoints available; they are not measured durations.

| Deliverable | Estimated cumulative effort |
|---|---|
| Eager Qwen BF16 prototype, single node, manual switching | 2–3 engineer-weeks |
| Qwen + GPT-OSS/SWA, overlap/auto switching, dual graphs, required FP8 paths, regression validation | 6–10 engineer-weeks |
| Multi-node/performance hardening if not covered above | Allow 1–3 additional weeks |

The widest uncertainty is interaction among the new runtime context, graph
ownership, memory allocation, and hardware-specific MoE implementations.
An environment failure or an expanded feature matrix can push these estimates
higher. The existing standalone ParaS algorithms and tests reduce the amount
of design work, but they do not eliminate integration testing.

The first decision gate is upstream static DP/EP plus the rebuilt transfer
extension on A100. The second is Qwen eager EP→TP→EP with real in-flight
requests and correct ownership/capacity accounting. Once those pass, refine the
remaining estimate using observed changes rather than Git conflict counts.

This assessment used source inspection, installed-package metadata, GPU identity,
and an isolated merge simulation. It did not install target dependencies,
compile kernels, run inference, or establish performance/accuracy parity.
