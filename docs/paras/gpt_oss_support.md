# ParaS Support for GPT-OSS

## Reader

This document is written for a ParaS contributor who already understands the
core EP↔TP switch, the unified memory manager, and the N+1 slot design (see
`parallelism_switch.md` and `unified_memory_manager.md`), and now needs to
understand what GPT-OSS adds to that baseline. The last session's bug-probing
chronicle is preserved at the end as a worked example of how to localize
failures in this codebase.

## Why GPT-OSS Is Different from Qwen3-MoE

Qwen3-MoE is the ParaS reference model. GPT-OSS adds six complications that
the baseline design did not anticipate. Each one required targeted code.

| Feature | Qwen3-MoE | GPT-OSS |
|---|---|---|
| MoE biases | none | `w13_weight_bias`, `w2_weight_bias` both present |
| Attention sinks | none | per-head `sinks` parameter |
| Attention type | dense full attention | hybrid full + sliding-window per layer |
| KV budget | homogeneous | heterogeneous per-layer (SWA short, full long) |
| Checkpoint layout | one weight per projection | fused `gate_up_proj` / `down_proj` with MXFP4 option |
| w13 layout | concatenated `[gate..., up...]` | interleaved `[g0, u0, g1, u1, ...]` |

The ParaS GPT-OSS adaptation lives in five files:

- `python/sglang/srt/paras/models/gpt_oss.py` — ParaS subclasses
  (`GptOssSparseMoeBlockParaS`, `GptOssAttentionParaS`,
  `GptOssDecoderLayerParaS`, `GptOssModelParaS`, `GptOssForCausalLMParaS`)
- `python/sglang/srt/paras/layers/paras_moe_block.py` — MoE block mixin, w13
  interleaved weight transport, replicated-bias views and all-gather helper
- `python/sglang/srt/paras/layers/paras_attention.py` — attention mixin
- `python/sglang/srt/paras/layers/paras_decoder_layer.py` — per-layer
  communicator switch
- `python/sglang/srt/paras/paras_memory_manager.py` — layout planner for
  fused weights, heterogeneous KV reservation

## Five Adaptations in Detail

### Fused Checkpoint Format

GPT-OSS ships weights in two formats: HuggingFace BF16 and MXFP4. Both use
a fused layout: a single `gate_up_proj` tensor of shape `(num_experts, H, 2*I)`
that stores both gate and up projections, and a single `down_proj` tensor of
shape `(num_experts, I, H)`. The first dimension is global (128 experts for
gpt-oss-120b), so the BF16 path requires manual expert sharding.

`_slice_ep_expert_weights` in `paras/models/gpt_oss.py:453-498` walks the
weight stream during `load_weights` and narrows the first dimension for each
tensor whose name contains `mlp.experts.gate_up_proj` or
`mlp.experts.down_proj`, slicing the global expert range down to this EP
rank's local range. Without this pre-slicing, `FusedMoE.weight_loader_fused`
(which only shards the intermediate dimension) would fail on a shape
mismatch.

This pre-slicing is the hidden source of Bug 2 in the bug-probing chronicle
below. The sliced `ep_experts.w2_weight_bias` is the EP rank's local slice,
so any downstream code that reuses it as a global-indexed TP bias will read
out-of-bounds memory for expert ids outside the local range.

### Interleaved w13 Layout

GPT-OSS stores the fused gate-and-up projection with gate and up columns
interleaved: `[g0, u0, g1, u1, ..., g_{I-1}, u_{I-1}]` along the 2I axis.
The `swiglu_with_alpha_and_limit` activation reads `x[..., ::2]` as gate and
`x[..., 1::2]` as up. Qwen3-MoE uses the concatenated layout
`[g0, ..., g_{I-1}, u0, ..., u_{I-1}]` and the silu_and_mul activation does
a simple split.

When EP→TP switches the w13 tensor via all-to-all redistribution, the
interleaved pairs must stay together inside each rank's I/TP-size slab.
`paras_configure_tp_all_to_all` in `paras_moe_block.py:295-320` handles this
by viewing the EP tensor as `(num_local_experts, paras_tp_size, 2 *
I_per_tp_rank * H)` — one contiguous block per (local expert, destination
rank) pair — before the all-to-all. The `self._paras_interleaved_w13` flag
(set to `True` in `GptOssSparseMoeBlockParaS.__init__`) gates this path.

The interleaved w13 transport was added in commit `6efd00aad` with
bf16-exact unit tests at
`test/srt/paras/test_paras_gpt_oss_cuda_graph.py::test_gpt_oss_w13_layout_semantics_interleaved`.

### Replicated Read-Only Tensors: w13 Bias, w2 Bias, and Sinks

GPT-OSS has three small per-expert / per-head tensors that ParaS holds
replicated on every rank rather than transporting during EP↔TP switches:
`w13_weight_bias` shape `(num_experts, 2*I)`, `w2_weight_bias` shape
`(num_experts, H)`, and `sinks` shape `(num_heads,)`. For
gpt-oss-120b-bf16 the total replicated storage is roughly 160 MB per
rank, about 0.1 percent of the weight storage. Transport is more
expensive than replication.

All three follow the same pattern:

1. **One shared full-size tensor per rank.** `GptOssSparseMoeBlockParaS`
   owns `self._full_w13_bias`, `self._full_w2_bias`. `GptOssAttentionParaS`
   owns `self._full_sinks`. Each is allocated once at
   construction / first-switch time on every rank.
2. **EP and TP forward paths read through Parameter views into the shared
   tensor.** Switching mode rebinds the Parameter; the underlying storage
   never moves. `ep_experts.w{13,2}_weight_bias.data` points to
   `_full_*_bias[ep_start:ep_end]`. `tp_experts.w13_weight_bias` wraps
   `_full_w13_bias[:, 2*tp_start:2*tp_end]` (interleaved I-dim slice).
   `tp_experts.w2_weight_bias` wraps the full `_full_w2_bias` on
   `paras_tp_rank == 0`; non-rank-0 leaves the attribute unregistered.
3. **No cross-rank transfer is required at switch time** for biases or
   sinks. `sinks` are already full on every rank right out of the
   checkpoint loader. Biases are also loaded **full on every rank** at
   init time: `paras_init_moe` re-registers the FusedMoE-allocated
   bias parameters as zero-filled `(num_global_experts, D)` tensors
   (`_full_w13_bias`, `_full_w2_bias`) before `load_weights` runs, so
   the checkpoint loader writes the full bias directly into them with
   no transport. After load, `paras_finalize_moe_bias_views` rebinds
   `ep_experts.w{13,2}_weight_bias` to local-slice views and
   `tp_experts.w{13,2}_weight_bias` to TP slice views (rank-0 only for
   w2). Switching mode is a pure Parameter rebind; no kernel and no
   collective fires for biases on the switch path.

The `tp_experts.w2_weight_bias` rank-0-only registration is important: in
TP mode the Triton fused-MoE kernel adds `w2` bias inside each rank's
partial GEMM, then the paras-TP all-reduce sums the partial outputs. If
every rank added the bias locally the post-all-reduce sum would contain
the bias `paras_tp_size` times. Registering the bias only on rank 0 makes
`getattr(layer, "w2_weight_bias", None)` return None on non-rank-0 and
the kernel skips the bias add there (see
`python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py`
lines 565-570). This mirrors the vanilla-TP checkpoint-load mask at
`gpt_oss.py:1072-1073`.

The TP w13-bias view is non-contiguous along dim 1, but the fused-MoE
kernel reads biases with explicit `stride_bias_n`, so strided access is
correct.

Sinks are consumed only in decode attention. The prefill path goes through
the `torch.ops.sglang.unified_attention_with_output` custom op, whose
signature does not forward kwargs; decode goes through
`forward_batch.attn_backend.forward(..., **kwargs)` which does. This is a
pre-existing sglang design constraint, not a ParaS issue; vanilla TP
gpt-oss handles prefill without sinks correctly.

### Hybrid Attention and Heterogeneous KV Budget

GPT-OSS layer types alternate between `full_attention` and
`sliding_attention` according to `config.layer_types`. Sliding layers need
much less KV cache (window-sized) than full layers (sequence-sized). The
ParaS memory manager must reserve different KV budgets per layer type, and
the switch must preserve this distinction.

`plan_gpt_oss_moe_layout` in `paras/models/gpt_oss.py:260-278` plans the
expert layout. `plan_hybrid_kv_budget` (imported from
`paras_memory_manager`) splits the global KV budget across full and
sliding layer counts according to `swa_full_tokens_ratio`. The resulting
`_full_max_tokens` and `_swa_max_tokens` are then fed into
`classify_layers_from_config`, which produces per-layer
`LayerCacheSpec` entries that `manager.reserve_kv_cache` uses to materialize
the right KV pool layout.

The hybrid SWA+ParaS path is enabled by default. The runtime instantiates
`SWAKVPool` (a container of `full_kv_pool` + `swa_kv_pool` with per-layer
routing via `layers_mapping`) and `SWATokenToKVPoolAllocator` (dual sub-allocators
plus a `full_to_swa_index_mapping` tensor). The scheduler reserves heterogeneous
KV using `plan_hybrid_kv_budget` and `classify_layers_from_config`. Three
defects had to be fixed before the dual-pool path was end-to-end correct
on the gpt-oss-120b-bf16 + 4×A100 + Triton + cuda-graph configuration:
(1) `SWAKVPool.paras_configure_tp/ep` used `get_view` (which returns the
EP-shaped LayoutEntry) instead of `get_view_as` with an explicit TP shape,
so the inner sub-pools' k/v buffers kept the EP head count after the switch;
(2) `Scheduler.full_tokens_per_layer` and `swa_tokens_per_layer` were not
refreshed after the allocator resized, so the runtime memory-leak detector
falsely flagged a leak on the first decode; (3) `paras_resize_and_clear`
unconditionally allocated a fresh `full_to_swa_index_mapping` tensor, which
invalidated the `data_ptr` baked into captured cuda graphs at boot. Fix
(3) pre-grows the mapping at allocator construction time to the TP-mode
max size and zeros it in place on subsequent resizes, keeping `data_ptr`
stable across switches.

### DP-Attention in EP Mode

ParaS requires DP attention in EP mode. The server launches with
`--enable-dp-attention --dp-size ep_size`, and in EP mode each of the
`ep_size` GPUs is its own DP attention rank (one rank per GPU, each
processing its local batch through full QKV).

On EP→TP switch, DP attention is turned off: `paras_comm_configure_tp`
sets `_ATTN_DP_SIZE=1` and `_ATTN_TP_SIZE=paras_tp_size`, and
`SchedulerParasMixin.paras_configure_tp` sets
`server_args.enable_dp_attention = False`, `dp_size = 1`. The attention now
runs in TP mode across the `paras_tp_size` ranks.

The interaction between the DP-attention setting and the scheduler's
padding logic was the source of Bug 1 in the chronicle below. When
`server_args.moe_a2a_backend` was stored as the `MoeA2ABackend` Enum after
the switch rather than the string `"none"`, `require_attn_tp_gather` (which
compares against the string literal) incorrectly returned True, and
`prepare_mlp_sync_batch` padded every prefill to a multiple of
`attn_tp_size=4`.

## Class Hierarchy

The ParaS GPT-OSS classes use mixin-first MRO so that ParaS methods
override base methods cleanly:

```
GptOssForCausalLMParaS                          (paras/models/gpt_oss.py:206)
└── GptOssForCausalLM                           (models/gpt_oss.py:586)

GptOssModelParaS(ParaSModelMixin, GptOssModel)
├── ParaSModelMixin                             (paras/layers/paras_model.py)
│   └── paras_configure_tp_naive / _overlap / _peer_access
└── GptOssModel                                 (models/gpt_oss.py:549)

GptOssDecoderLayerParaS(ParaSDecoderLayerMixin, GptOssDecoderLayer)
├── ParaSDecoderLayerMixin                      (paras/layers/paras_decoder_layer.py)
│   ├── dual LayerCommunicator (EP + TP)
│   ├── paras_configure_tp_attn / tp_mlp
│   └── paras_configure_ep_attn / ep_mlp
└── GptOssDecoderLayer                          (models/gpt_oss.py:422)

GptOssSparseMoeBlockParaS(ParaSMoeBlockMixin, GptOssSparseMoeBlock)
├── ParaSMoeBlockMixin                          (paras/layers/paras_moe_block.py)
│   ├── paras_init_moe (with interleaved_w13=True)
│   ├── dual experts (ep_experts + tp_experts)
│   ├── paras_configure_tp_all_gather / all_to_all
│   └── paras_forward (dispatches to forward_normal vs forward_deepep)
└── GptOssSparseMoeBlock                        (models/gpt_oss.py:98)

GptOssAttentionParaS(ParaSAttentionMixin, GptOssAttention)
├── ParaSAttentionMixin                         (paras/layers/paras_attention.py)
│   └── paras_configure_tp/ep (qkv_proj, attn, o_proj)
└── GptOssAttention                             (models/gpt_oss.py:213)
    + self.sinks slicing in GptOssAttentionParaS.paras_configure_tp
```

The decoder layer constructor performs a class swap on the base attention
instance: `self.self_attn.__class__ = GptOssAttentionParaS`. This adds
ParaS methods to the already-constructed base instance without
reconstructing (the mixin adds no `__init__` state).

## Bug-Probing Chronicle

The bug chronicled here is "paras gpt-oss-120b-bf16 TP-mode decode emits
token id 0 after EP→TP switch". EP mode produces " Paris." correctly;
vanilla `--tp 4 --ep 1` also works. The diagnostic chain below took three
sessions to resolve. Each step narrowed the search space further. The fix
landed in commit `2d0ba2668`.

### Hypotheses Ruled Out

1. **w13 layout** (prior session). Unit tests in
   `test_paras_gpt_oss_cuda_graph.py` verified the interleaved w13 transport
   is bf16-exact across EP→TP→EP. Refuted by unit test.
2. **MoeRunner dead code** (prior session). Direct experiment on the
   fused-func code path showed it was not invoked in the failing config.
3. **w2_weight_bias None vs zero tensor on non-rank-0** (this session,
   probe #1). Register an explicit zero tensor on non-rank-0 instead of
   relying on `getattr(..., None)`. Refuted by end-to-end test; TP still
   emitted token 0.
4. **LayerScatterModes mlp_mode SCATTERED vs FULL** (this session).
   Temporarily swap `MOE_A2A_BACKEND` to `NONE` during TP
   `LayerCommunicator` construction so scatter modes compute as FULL.
   Refuted; the empirical dumps showed hidden_states were identical across
   ranks at MoE entry, contradicting the SCATTERED hypothesis in the first
   place.
5. **Attention sinks prefill path** (this session). The prefill path
   through `unified_attention_with_output` drops kwargs, so sinks are not
   forwarded. Vanilla TP has the same behavior and works, so sinks are
   not the cause.

### Evidence Collection

The key instrumentation was an env-var-gated dump inserted into two places:

1. `GptOssSparseMoeBlock.forward_normal` — snapshot `hidden_states_in`
   (before `self.experts`, because `fused_experts` runs in place),
   `router_logits`, top-k ids and weights, pre-reduce MoE output, and
   final post-all-reduce output. Enabled by `PARAS_DUMP_MOE_IO=<tag>`.
2. `GptOssAttention.forward_core` — snapshot `attn_output` (post
   attention kernel, pre o_proj) and `output` (post o_proj). Enabled by
   `PARAS_DUMP_ATTN_IO=<tag>`.

Both limit captures to `layer_id == 0` and `tp_size > 1` (or `num_heads <
total_num_heads`) so the dumps only fire for TP-mode runs. A counter caps
the number of captures per rank so warmup and EP baseline requests do not
saturate the dump window.

Three iterations of the MoE instrumentation revealed:

- Call 0 (prefill) had NaN hidden_states on all ranks at MoE entry. The
  initial dump placed the snapshot after `self.experts()`, which is an
  inplace kernel, so the captured "input" was actually the partial MoE
  output. Moving the snapshot before `self.experts()` confirmed the NaN
  was already present at MoE entry — so the corruption was upstream of
  MoE.
- Later calls had finite hidden_states, identical across ranks, confirming
  the MoE TP forward itself was numerically correct (identical inputs →
  proper per-rank partial outputs → identical post-all-reduce).

Attention instrumentation then revealed:

- Paras TP prefill had `num_tokens=8` (not 6) on all ranks. The 2 extra
  padding tokens had qkv output that overflowed bf16 attention softmax on
  rank 1 (values up to `5.8e+37`) and produced NaN on ranks 0, 2, 3.
- Vanilla TP prefill had `num_tokens=6`, no overflow, no NaN.

The difference was the prefill batch size. Vanilla TP's 6 tokens exactly
matched the prompt length; paras TP's 8 was the prompt padded to a multiple
of `attn_tp_size=4`.

### Bug 1: Enum/String Mismatch Causing Prefill Padding

Tracing the padding led to `forward_batch_info.py:696-698`:

```python
global_num_tokens[i] = (
    (global_num_tokens[i] - 1) // attn_tp_size + 1
) * attn_tp_size
```

This pads the prefill batch to a multiple of `attn_tp_size`. The pad fires
only when `require_mlp_sync(server_args)` returns True. That in turn calls
`require_attn_tp_gather` (`utils/common.py:2751`):

```python
if server_args.moe_a2a_backend != "none" or server_args.moe_dense_tp_size == 1:
    if server_args.enable_dp_attention:
        return server_args.dp_size < server_args.tp_size
    else:
        return True
else:
    return False
```

`server_args.moe_a2a_backend` is typed as a plain string. After the paras
switch, `SchedulerParasMixin.paras_configure_tp` sets it to
`MoeA2ABackend.NONE` (the Enum) rather than the string. Python's `Enum !=
str` is True across types, so `require_attn_tp_gather` enters the true
branch, returns True, triggers the padding.

Fix: store `MoeA2ABackend.NONE.value` (the string "none") and
`MoeA2ABackend.DEEPEP.value` (the string "deepep") at the two
assignment sites. The Enum is still stored in `moe_utils.MOE_A2A_BACKEND`,
which is what the `get_moe_a2a_backend()` callers expect.

After this fix, prefill returned to 6 tokens, attention output became
finite on all ranks, and the decode output went from `!!!...` to a mixed
`' the 1.\n\n'` — different tokens, still incoherent. A second bug was
lurking.

### Bug 2: w2_weight_bias OOB Reads on Rank 0

With attention clean, MoE instrumentation comparing paras TP against
vanilla TP revealed that rank 0's `pre_reduce` MoE output was ~6× larger
in paras than in vanilla, while ranks 1, 2, 3 matched closely:

```
VANILLA rank 0 pre_reduce abs_mean = 0.191
PARAS   rank 0 pre_reduce abs_mean = 0.933   (~5x larger)
VANILLA rank 1,2,3      abs_mean ≈ 0.16
PARAS   rank 1,2,3      abs_mean ≈ 0.15      (match)
```

The difference localized to rank 0, which is the rank that adds the w2
bias. Examining the registration in `paras_moe_block.py`:

```python
if ep_with_bias and hasattr(...) and get_paras_tp_rank() == 0:
    self.tp_experts.register_parameter(
        "w2_weight_bias", self.ep_experts.w2_weight_bias
    )
```

`ep_experts.w2_weight_bias` has shape `(num_local_experts, H) = (32, H)`
because `_slice_ep_expert_weights` narrows the fused checkpoint to the
local EP rank's expert range at load time. But `tp_experts` has
`num_experts = num_global_experts = 128`, and the Triton kernel
(`fused_moe_triton_kernels.py:458-461`) indexes the bias by global expert
id:

```python
bias_ptrs = bias_ptr + off_experts * stride_bias_e + offs_bn[None, :] * stride_bias_n
```

For `off_experts` in `[32, 127]`, this reads out-of-bounds GPU memory.
About 75% of expert indices fall in this range, so ~75% of bias reads were
arbitrary data.

The first fix (commit `2d0ba2668`) allocated a fresh full-size buffer on
rank 0 and populated it via `dist.all_gather_into_tensor`. It restored
correctness but kept two parallel bias storage systems — the UMM-staged
w13 path (all-to-all with permute) and the rank-0-only w2 path
(all-gather). A follow-up refactor (commits `5b94a0b97` through
`ac559d161`) consolidated both into the single replicated-bias design
documented in the earlier "Replicated Read-Only Tensors" section: one
`_full_*_bias` tensor per rank, EP and TP views into it, one idempotent
all-gather helper called from both the NCCL and peer-access paths. The
UMM no longer carries bias entries at all.

After the fixes, paras TP produced " Paris." bitwise-matching EP, and
the EP→TP→EP→TP round trip preserved output across multiple switches.

## Bug-Probing Chronicle 2: In-flight EP↔TP Cache Transfer

The original "Paris" chronicle exercised the round-trip on a settled
empty pool — switch, request, switch back, request — and produced
bitwise-identical output across modes. It did not exercise the
**in-flight switch**: a request actively decoding when the scheduler
receives `/paras_configure_tp` (or `/paras_configure_ep`) must
continue to produce coherent output across the boundary. Resolving
in-flight correctness uncovered four additional bugs (Bugs 3-6) and
one A100/NCCL incompatibility (Bug 7). All four bugs were latent in
shared code; the round-trip-on-empty-pool test never reached them
because gather/scatter exits early when there are no in-flight tokens
to redistribute.

### Symptom

Setup: server in EP mode. Send a 500-token generation in the
background, sleep 1 second so decode begins, fire
`/paras_configure_tp`, wait for completion. Expected: roughly 500
coherent tokens. Actual (at commit `a6c8b424b`): the first ~50
characters (the pre-switch tokens that EP decoded normally before the
switch fired) were coherent, then the post-switch tokens collapsed
into a repetitive loop like `` i.e., 0, and the `i.e., 0, and the `i.e., 0, ...``.

The symptom was deterministic at `temperature=0`, reproduced across
prompts, and survived `--disable-cuda-graph`. The same setup did
expose a separate CUDA-graph bug (Bug 6 below), but the in-flight
correctness defect was independent of cuda graph.

### Bug 3: `get_new_running_batch` seqlen-1 off-by-one

After `gather_cache` completes, `ParaSReqGatherManager.get_new_running_batch`
(and its TP→EP counterpart in `ParaSReqScatterManager`) constructs
the post-switch `ScheduleBatch` from the recovered `Req` objects.
The pre-fix code wrote:

```python
seq_lens_list = [req.seqlen for req in self.global_reqs]   # WRONG
batch.seq_lens = torch.tensor(seq_lens_list, ...)
```

SGLang's runtime convention is that `batch.seq_lens` equals the K/V
cache history length, which is `req.seqlen - 1`, not `req.seqlen`.
The off-by-one corresponds to `req.output_ids[-1]`: the most recently
sampled token exists in the request's output list, but its K/V has
not been written yet. `alloc_for_decode` writes K/V for it on the
*next* decode iteration, when it becomes the input. See
`mem_cache/common.py:440-460`: `locs = batch.seq_lens.clone()`
captures the new K/V slot's position BEFORE `seq_lens.add_(1)`, so
the kernel writes K/V at `seq_lens` and only positions
`[0, seq_lens-1]` are valid pre-iteration.

`gather_cache` already respected this convention. It transferred
`req.seqlen - 1` tokens of K/V (`gather_manager.py:138`, slicing
`req_to_token[req_pool_idx][:req.seqlen - 1]`), leaving
`req_to_token[req_pool_idx][seqlen - 1]` as zero in the new pool.
The bug was downstream: setting `batch.seq_lens = req.seqlen`
post-switch made the next decode iteration's attention read position
`seqlen - 1` — the empty slot — and attention then read K/V from
slot 0 (or whatever the new pool's slot 0 happened to hold). The
resulting same-K-and-V-everywhere attention produced the
characteristic repetition loop.

Fix (commit `b9e8c9321`): subtract one in three sites:
`gather_manager.get_new_running_batch:359`,
`gather_manager.update_running_batch_inplace`, and
`scatter_manager.get_new_running_batch`.

A diagnostic probe added to `gather_manager.__init__` confirmed the
empty-slot reading. For one in-flight request with `seqlen=25`, the
EP-side `req_to_token[req_pool_idx][:seqlen+2]` was
`[1, 2, 3, ..., 24, 0, 0, 0]` — 24 valid slots, the rest zero.
Position 24 is `seqlen - 1`. Reading from `req_to_token[24] = 0`
sends the kernel to KV slot 0.

The bug exists in shared cache-transfer code, so it should in
principle affect any ParaS model that exercises the in-flight switch.
Qwen3-MoE was empirically tested both with and without this fix on
4×A100 with `--cuda-graph-max-bs 8 --disable-cuda-graph` (the test
launch flags are recorded in
`.skills/paras-test-qwen3/SKILL.md`), and the result is
**asymmetric**:

| Direction | qwen3 WITHOUT fix | qwen3 WITH fix |
|---|---|---|
| EP-only (no switch) | coherent | coherent |
| In-flight EP→TP | coherent | coherent |
| In-flight TP→EP | **coherent** | **GIBBERISH + scheduler crash** |

The `AssertionError: This request holds the node from another tree`
in `radix_cache.dec_lock_ref` and the "running-req: 2 on DP0, 1 on
DP1, DP2, DP3" log pattern point to the request being replicated
across all DP ranks during scatter rather than partitioned to one
rank. Why this happens with the seqlen-1 fix on qwen3 but not on
gpt-oss is not yet understood. The empirical observation is that
qwen3 + the default attention backend (FlashInfer) needs
`seq_lens = req.seqlen` for both directions, while gpt-oss + Triton
needs `seq_lens = req.seqlen - 1` for both. This means a single
unconditional fix in shared code cannot be right for both models;
either the fix should be made backend-aware, or there is a
companion bug elsewhere that the original `req.seqlen` value
happened to compensate for, and the right fix is to align the
companion site rather than this one.

**Status**: the seqlen-1 fix is committed in `b9e8c9321` and is
correct for gpt-oss-120b-bf16 with the Triton attention backend (the
ParaS-deployed configuration). It introduces a regression on qwen3
in-flight TP→EP. The next session should investigate whether the
companion bug lives in `partition_requests`, in `alloc_for_decode`,
or in the FlashInfer attention backend's seq_lens convention, and
land a fix that satisfies both models.

### Bug 4: `gather_kv_and_permute` layout for `sharded_heads > 1`

The NCCL all-to-all gather expects the receiver to view its received
buffer as `[total_tokens, KV=2, sharded_heads, head_dim]`
(`permute_and_scatter_kv` in `cache_transfer/utils.py:50`). To make
this work cleanly across multiple source ranks, each per-destination
chunk on the sender must already be laid out as
`[tokens_for_dest, KV=2, sharded_heads, head_dim]`, so concatenation
across sources yields the receiver's expected layout directly.

The pre-fix implementation produced `[heads, tokens, KV, dim]` per
chunk:

```python
return local_kvcache.permute(2, 1, 0, 3).contiguous().flatten()
```

The two layouts coincide only when `sharded_heads == 1`. Qwen3-MoE
has 4 KV heads with `paras_tp_size=4`, so `sharded_heads=1`, the head
dimension collapses, and the byte sequence happens to match.
GPT-OSS-120b has 8 KV heads with `paras_tp_size=4`, so
`sharded_heads=2`. The receiver's
`view(num_tokens, 2, num_heads=2, head_dim)` reinterpreted each
head's contiguous block as half a token. The K/V written to the new
TP cache slots was scrambled across heads on every TP rank, and every
layer's attention read corrupted K/V.

This is the textbook "works for one model by coincidence, breaks
silently for another" defect. It was invisible until the gpt-oss head
configuration was the first to violate the `sharded_heads == 1`
assumption.

Fix (commit `b9e8c9321`): reshape and permute the sender's K/V to
produce per-chunk layout `[tokens, KV, sharded_heads, dim]` directly,
gated by `sharded_heads > 1` to preserve the original
`sharded_heads == 1` fast path:

```python
reshaped = local_kvcache.view(2, n_tokens, group_size, sharded_heads, head_dim)
return reshaped.permute(2, 1, 0, 3, 4).contiguous().flatten()
```

### Bug 5: `req_to_token` reallocation invalidates backend caches (corrected)

#### Symptom (commit `b9e8c9321`)

`ReqToTokenPool.paras_resize_and_clear` originally allocated a fresh
tensor on every call:

```python
self.req_to_token = torch.zeros(
    (new_size, self.max_context_len), dtype=torch.int32, device=self.device
)
```

After a paras EP↔TP switch with cuda graph enabled, decode produced
incorrect K/V indices and silent output corruption.

#### Original fix (commit `b9e8c9321`)

The `b9e8c9321` commit attributed the bug to CUDA graphs baking
`req_to_token.data_ptr()` into Triton attention kernel arguments via
`create_flashinfer_kv_indices_triton`. The fix preserved `data_ptr`
by pre-growing `req_to_token` to `max(EP, EP * paras_tp_size)` before
`init_attention_backend` runs, then zeroing in place on subsequent
resizes.

#### Mechanism correction

The original mechanism description was wrong on one detail: cuda
graphs do **not** capture `req_to_token.data_ptr()` into any kernel
argument. `create_flashinfer_kv_indices_triton` is called from
`init_forward_metadata_capture_cuda_graph` and
`init_forward_metadata_replay_cuda_graph`, both of which run **outside**
the captured graph region (see `cuda_graph_runner.py:686-743`). The
captured graph contains only `run_once()` (the model.forward call).
The Triton kernel that reads `req_to_token` writes its output into
`cuda_graph_kv_indices`, which **is** captured. So
`cuda_graph_kv_indices.data_ptr()` matters, but
`req_to_token.data_ptr()` does not.

The actual mechanism that the pre-grow fix accidentally repaired was
backend-side caching: each attention backend caches a reference to
`req_to_token` in its own attribute, and reallocation invalidated the
cache. Both backends already rebind on `paras_configure_tp/ep`, so
caching is fine **provided** the rebind happens. The pre-grow trick
worked by removing the rebind requirement (the buffer never moved),
not by satisfying any cuda-graph invariant.

#### Replacement design

The pre-grow has been removed. `ReqToTokenPool.paras_resize_and_clear`
now allocates a fresh tensor unconditionally. `ModelRunner.paras_configure_tp/ep`
run a post-switch `_paras_assert_req_to_token_rebound()` check that
verifies `attn_backend.req_to_token`,
`indices_updater_decode.req_to_token`, and
`indices_updater_prefill.req_to_token` all point at the live
`ReqToTokenPool.req_to_token`. A future backend that forgets to rebind
will fail this assertion immediately rather than reading freed memory
later.

The `(paras_tp_size - 1) * max_running_requests * max_context_len * 4`
bytes of permanent GPU memory the pre-grow consumed (≈3 GB at
gpt-oss defaults) is recovered. The `--max-running-requests 256`
workaround flagged in the prior P1 is no longer required for cuda
graph capture to fit on A100-80GB at `mem-fraction-static=0.6`.

### Bug 6: Triton CUDA graph attention buffers reallocated between EP and TP capture

#### Symptom (commit `da3cd54c7`)

With `--cuda-graph-max-bs 256` the server crashed at the first decode
replay (no switch needed) with
`CUDA error: an illegal memory access was encountered`. With
`--cuda-graph-max-bs 8` the server produced silent garbage output.
The crash sat in `self.graphs[bs].replay()` for `bs=1`.

#### Mechanism

Unlike Bug 5, this mechanism description was correct.
`CudaGraphRunner.__init__` captures EP graphs that bake
`self.kv_indptr.data_ptr() = A` into the Triton attention kernel
arguments inside the captured region (the model's actual attention
kernel reads `kv_indptr` from `forward_metadata`). The same applies
to `qo_indptr`, `mask_indptr`, `window_kv_indptr`, and the
`cuda_graph_*` buffers that the kernel reads inside the graph.

`paras_init_dual_cuda_graphs` switches to TP mode before capturing TP
graphs. The original `_paras_reset_buffers` reallocated the four
indptr buffers to address `B`. EP graphs now reference an orphaned
address `A`. After the closing switch back to EP, a third
reallocation puts the live state at address `C`. EP replay reads
from `A` (captured), but
`init_forward_metadata_replay_cuda_graph` writes to `C` (live), so
the EP graph reads stale bytes from `A`'s last EP-capture cumsum.
With `cuda_graph_max_bs=8` the stale bytes happen to be small enough
that indices land within `cuda_graph_kv_indices` and produce silent
garbage; with `cuda_graph_max_bs=256` they index out of bounds and
crash.

#### Original fix (commit `da3cd54c7`)

Pre-grow `kv_indptr` family to `req_to_token.shape[0] + 1` (i.e. the
TP-mode size) before any cuda graph capture runs. Then change
`_paras_reset_buffers` to zero in place when capacity is sufficient.
After the fix, `data_ptr` for these buffers is identical across
modes, so the EP graph's baked address remains valid across switches.

#### Replacement design

The pre-grow + zero-in-place approach made one set of buffers serve
both modes and required a hidden cross-mode size invariant. The new
design preserves per-mode state via two generic backend hooks:

```python
class AttentionBackend:
    def paras_save_cuda_graph_state(self) -> Dict[str, Any]: ...
    def paras_load_cuda_graph_state(self, state: Dict[str, Any]) -> None: ...
```

`TritonAttnBackend.paras_save_cuda_graph_state` snapshots
`kv_indptr`, `window_kv_indptr`, `qo_indptr`, `mask_indptr`, and the
`cuda_graph_*` buffers (12 attributes). `paras_load_cuda_graph_state`
reassigns each `self.X` from the saved dict.

`paras_init_dual_cuda_graphs` calls
`attn_backend.paras_save_cuda_graph_state()` after EP capture, lets
`paras_configure_tp` allocate fresh TP-mode buffers, captures TP,
saves TP state, then loads EP state via the hook before returning.
Each mode owns its own buffers; the previous mode's buffers stay
alive via Python refs in the saved state dict.

`paras_configure_tp/ep` on Triton allocates fresh kv_indptr family
+ cuda graph buffers via `_paras_alloc_fresh_buffers()`. This keeps
the no-cuda-graph runtime path correct (TP mode needs a kv_indptr
sized for the TP pool, which is larger than EP's). In the
cuda-graph runtime path, `paras_swap_cuda_graphs` immediately
overwrites these freshly allocated buffers via the load hook, which
restores the saved-state buffer refs. The fresh allocation is
discarded (orphaned via Python GC) but does no harm.

`FlashInferAttnBackend` implements the same hooks by snapshotting
its existing per-mode `decode_cuda_graph_metadata`,
`prefill_cuda_graph_metadata`, and `draft_extend_cuda_graph_metadata`
dicts. The previous flashinfer-specific helpers
`_save_flashinfer_metadata` and `_load_flashinfer_metadata` in
`paras_cuda_graph.py` are removed; the orchestrator now calls the
generic hook on the backend uniformly.

The same fix is no longer needed preemptively in
`flashinfer_backend.py` (the prior P3 item) because FlashInfer was
already storing per-mode metadata; the generic hook just exposes
that mechanism for the orchestrator to call uniformly.

### Bug 7: Peer-access pre-init crashes on A100 with NCCL transfer

`GptOssForCausalLMParaS.__init__` previously called
`init_peer_access(...)` unconditionally at boot, regardless of
whether `PARAS_KV_TRANSFER_METHOD` was `nccl` (default) or
`peer_access`. On A100 (sm80), this enabled NVLink peer mappings
that interacted badly with NCCL's internal use of peer-to-peer
memory. The first `/paras_configure_tp` then triggered
`CUDA error: Invalid access of peer GPU memory over nvlink`
inside an unrelated NCCL collective.

Root cause is not yet understood. The pre-init was originally added
to amortize the ~6 second cost of `cudaDeviceEnablePeerAccess` over
boot rather than paying it on the first switch. Disabling the
pre-init removes the cost saving but does not affect correctness for
the NCCL transfer path.

Workaround (this commit): skip the pre-init by default, gated on
`PARAS_DISABLE_PEER_ACCESS=1` (the new default). The peer-access
KV transfer path is selected by setting both
`PARAS_DISABLE_PEER_ACCESS=0` and `PARAS_KV_TRANSFER_METHOD=peer_access`.

Status update: end-to-end peer-access KV transfer is now verified on
4×A100 for both Qwen3-30B-A3B (heads_per_rank=1) and gpt-oss-120b-bf16
(heads_per_rank=2, the sharded-heads case). EP→TP `gather_cache` runs
in ~8–15 ms via the fused NVLink kernel; TP→EP `scatter_cache` runs in
~10–24 ms; in-flight switches in both directions produce coherent
continuations with no `Invalid access of peer GPU memory over nvlink`
errors. Bug 7's original concern about the pre-init step interacting
badly with NCCL applied only when NCCL was used for the same fabric;
selecting peer-access for both weights and KV avoids the conflict.

Open question:
- Whether the peer-access pre-init step is harmless in isolation and
  conflicts only when followed by NCCL all-to-all on the same NVLink
  fabric. The verification above suggests yes — the pre-init is fine
  as long as the same fabric is used by peer-access kernels rather
  than NCCL all-to-all.
- Whether the pre-init should be deferred to first switch (when it
  is actually needed) instead of paid at boot.

The current default sidesteps all three for the NCCL transfer path
that production deployments use.

## Recent Updates: ParaS Switched to ChunkCache (May 2026)

After Chronicle 2 stabilized in-flight switch correctness on
`SWARadixCache`, the ParaS branch was updated to require
`--disable-radix-cache` instead of forbidding it
(`server_args._check_paras_config` line 1512+, assertion flipped). ParaS
now runs exclusively with `ChunkCache` (MHA-only models like Qwen3-MoE)
or `SWAChunkCache` (hybrid SWA models like gpt-oss). Cross-request prefix
sharing is unavailable in this configuration; the trade-off is justified
by ParaS's lack of demand for prefix sharing weighed against the
complexity of migrating radix-tree state across EP↔TP switches.

Three additions to the SWA pool lifecycle were introduced for gpt-oss:

1. **Runtime SWA pool eviction**
   (`ScheduleBatch.maybe_evict_swa` / `_evict_swa`,
   `schedule_batch.py:1573` / `:1582`). Fires every decode step. For
   each Req, computes `new_swa_evicted_seqlen = max(swa_evicted_seqlen,
   pre_len - W)` and frees `req_to_token[idx,
   swa_evicted_seqlen:new_swa_evicted_seqlen]` from the SWA pool via
   `allocator.free_swa(...)`. Keeps SWA pool occupancy bounded at
   `min(W, P) + decode_steps_within_window` per request — empirically
   validated on gpt-oss-120b at ~3% of `swa_max_tokens` capacity during
   8-req heavy decode. Inspired by upstream PR sgl-project/sglang#17220
   but without tombstone tracking (no tree to coordinate with under
   `SWAChunkCache`).
2. **Per-Req `swa_evicted_seqlen` carrying through gather/scatter**.
   Pickle preserves the field automatically: `prune_request`
   (`gather_manager.py:30`) only nulls 4 specific tensor-bearing fields,
   not the int field. No code change required; validated implicitly by
   in-flight switch tests passing 0/32 degenerate post-switch.
3. **Destination-side SWA pool tightening at switch boundary**
   (`ParaSReqGatherManager._tighten_swa_pool_to_in_window`,
   `gather_manager.py:242`, with the scatter mirror in
   `scatter_manager.py`). After lockstep alloc + `req_to_token`
   write-back, computes `in_window_start = max(req.swa_evicted_seqlen,
   seqlen-1-W)` per migrated Req and frees the OOW destination SWA slots
   via `allocator.free_swa(...)`. Achieves immediate post-switch
   convergence: destination SWA pool footprint drops from `seqlen` to
   `min(W, seqlen-1-evicted)` per migrated Req at the switch boundary,
   eliminating the up-to-W-step convergence delay that would otherwise
   apply.

The radix-cache-specific Bug 3 symptom in Chronicle 2 (`AssertionError:
This request holds the node from another tree` in
`radix_cache.dec_lock_ref`) is no longer reachable from the ParaS code
path: with `--disable-radix-cache` enforced, neither `RadixCache` nor
`SWARadixCache` is constructed. Prefix sharing in ParaS would require
migrating the radix tree across switches; this is documented as future
work in [`future/radix_cache.md`](file:///home/shaoyuw/sglang/docs/paras/future/radix_cache.md).

References for this update:

- `server_args.py:1512+` — assertion flip (`assert self.disable_radix_cache`).
- `gather_manager.py:36-49` — `recover_request` with `tree_cache.disable` branch.
- `gather_manager.py:242+` — `_tighten_swa_pool_to_in_window` (Phase C).
- `schedule_batch.py:1573+` — `maybe_evict_swa` and `_evict_swa` (Phase A).
- `runs/2026-05-09-swa-window-only-transfer-design/DESIGN.md` — full design.
- `future/radix_cache.md` — what would be required to re-enable radix cache for ParaS.

## Probes That Worked

Two methodological patterns made this chronicle tractable and are worth
reusing.

**Env-var-gated instrumentation, conditional on TP-sharded state.** Adding
dumps behind `PARAS_DUMP_*_IO=<tag>` lets you compare vanilla and paras
runs side-by-side with a short diff. Gating on `self.tp_size > 1` (for MoE)
or `self.num_heads < self.total_num_heads` (for attention) skips the
warmup and EP baseline calls automatically, so the capture window only
holds real TP-mode data. A counter cap prevents the first TP call from
exhausting the window before decode fires.

**Layered probes from output to input.** Start dumps at the most
downstream location (MoE entry) and work upstream. If the downstream is
NaN, the bug is above it. If the downstream is numerically wrong but
finite, the bug is usually weight/layout, so cross-check partial outputs
against a known-good run (vanilla TP). If the downstream is correct, the
bug is in a later layer (LM head, logits, sampling).

Three probes that wasted effort are also worth noting:

- Oracle consultations with stale context (timed out or gave generic
  advice). Oracle is useful for architectural review, not for
  concrete numerical debugging.
- Guessing at scatter modes without running the code. The
  `mlp_mode=SCATTERED` hypothesis was ruled out by a one-line check against
  the empirical dump (identical across ranks ⇒ not scattered).
- Committing a fix before a diff. Both refuted probes were reverted
  cleanly because the diagnostic scaffolding was kept separate from the
  attempted fix.

## Limitations and Next Steps

Pre-existing limitations that do not affect current deployments but
should be addressed before those configurations are used:

- `ep_num_redundant_experts > 0` with bias-bearing models will fail the
  assertion `local_b.shape[0] == self.num_local_experts` in
  `paras_configure_tp_all_to_all`. The default is 0 and gpt-oss-120b uses
  0. A full fix would compute the expected local size from
  `num_local_experts + ep_num_redundant_experts` and adjust the
  `all_gather_into_tensor` shape math accordingly.
- `PARAS_CONFIGURE_METHOD=peer_access` is now supported for GPT-OSS. The
  v2 / `_ep` peer-access kernels in `paras/csrc/peer_access_transfer.cu`
  handle the interleaved w13 layout without `.cu` changes — the call
  sites in `paras_configure_tp_fused_peer_access_kernel` and
  `paras_configure_ep_fused_peer_access_kernel` branch on
  `_paras_interleaved_w13` and pass `num_gates=1` together with a
  doubled per-chunk extent (`2 * I' * H` instead of `I' * H`). Each
  rank's `2*I'*H` contiguous interleaved `(g_k, u_k)` slab is then
  transferred as a single chunk per (expert, peer), preserving
  interleaving end-to-end. Biases are not part of the kernel transfer
  contract; they are loaded full on every rank at init time and exposed
  to the EP and TP forward paths via Parameter views into
  `_full_w{13,2}_bias` (see `paras_finalize_moe_bias_views`). End-to-end
  validation: 4×A100, gpt-oss-120b-bf16, `PARAS_CONFIGURE_METHOD=peer_access`,
  switch latency ~270 ms (vs ~600–900 ms for the previous naive default),
  three-prompt EP/TP/EP-RT batches and in-flight EP↔TP both directions
  all coherent, zero server errors.

Sinks are not propagated through the prefill attention path
(`unified_attention_with_output` drops kwargs). This is a pre-existing
sglang constraint that does not affect decode quality; the model was not
trained to require sinks at prefill for the degree of quality seen in
benchmarks. A future change to plumb sinks through the custom op would
close the gap with flash_attention and triton decode paths.

New open items from Chronicle 2:

- **Bug 3 (qwen3-MoE TP→EP in-flight regression with the fix) is not
  reproducible**. The `2026-04-25-bug3-qwen3-tp-ep-non-reproduction.md`
  run exercised ~50 in-flight requests across ~28 mode switches at the
  exact code state (`b9e8c9321`) and configuration that previously
  produced gibberish + a radix-cache assertion crash. All 50 requests
  produced coherent output; partition probes showed deterministic
  cross-rank consistency; no foreign-tree node was ever observed in
  `dec_lock_ref`. Output now matches the previous WITHOUT-fix coherent
  baseline byte-for-byte on the prefix that matters. The fix is
  canonically correct per the `prepare_for_decode` / `alloc_for_decode`
  convention and does not regress qwen3 in extensive testing.

- **Bug 5 + Bug 6 redesign (this commit)**. The pre-grow approach for
  `req_to_token` (Bug 5) and Triton attention buffers (Bug 6) has been
  replaced with state-preservation hooks
  (`paras_save_cuda_graph_state` / `paras_load_cuda_graph_state` on
  the `AttentionBackend` base class). Each mode owns its own buffers;
  `paras_init_dual_cuda_graphs` snapshots them per mode and the
  runtime mode switch restores via the load hook. The
  `--max-running-requests 256` workaround flagged previously for
  cuda graph memory pressure is no longer required. See the Bug 5
  and Bug 6 sections above for the corrected mechanism description
  and replacement design.

- **Cuda graph round-trip determinism degraded**. Without cuda graph
  the EP→TP→EP round trip on a fixed prompt produces bitwise
  identical EP outputs on either side of the round trip. With cuda
  graph the outputs match for ~30 tokens then diverge into
  semantically-equivalent-but-different completions. Likely cause is
  Triton autotune picking different configs across captures or some
  per-capture state not being reset on the swap. Functional
  correctness is unaffected; bitwise round-trip determinism was a
  nice-to-have property of the eager path, not a hard requirement.

- **Peer-access pre-init root cause (Bug 7)**. The
  `PARAS_DISABLE_PEER_ACCESS=1` default sidesteps the
  `Invalid access of peer GPU memory over nvlink` crash on A100 + NCCL
  transfer, but the underlying interaction is not understood. The
  three open questions in Bug 7 above should be answered before the
  peer-access KV transfer path is enabled in production.

- **In-flight switch as part of the regular validation matrix**. The
  qwen3 routine in `.skills/paras-test-qwen3/SKILL.md` (steps 11-12)
  exercises in-flight EP↔TP, but the gpt-oss routine in
  `docs/paras/gpt_oss_test.md` (predecessor of this doc) did not.
  Adding in-flight EP↔TP coverage to gpt-oss CI would have caught
  Bugs 3, 4, 6 before they shipped.

## References

- `parallelism_switch.md` — overall ParaS EP↔TP switch design
- `unified_memory_manager.md` — memory layout and N+1 slot design
- `cuda_graph.md` — dual CUDA graph capture for EP and TP
- `runs/2026-04-23-paras-swa-bias-transport.md` — earlier SWA+bias session
- `runs/2026-04-24-paras-gptoss-tp-degenerate.md` — Chronicle 1 session 1
- `runs/2026-04-24-paras-gptoss-tp-nan-upstream.md` — Chronicle 1 session 2
- Commit `2d0ba2668` — Chronicle 1 original fix (w2 bias OOB, Enum-vs-string)
- Commit `5b94a0b97` — replicated-bias stage 1 (full tensor + EP views)
- Commit `c11459eff` — replicated-bias stage 2 (all-gather replaces transport)
- Commit `b035aa391` — replicated-bias stage 3 (UMM bias entries removed)
- Commit `ac559d161` — sinks naming parity (`_sinks_full` → `_full_sinks`)
- Commit `6efd00aad` — interleaved w13 transport
- Commit `85d59b642` — SWA bias transport
- Commit `2640e303d` — activation / gemm1_alpha / clamp_limit propagation
- Commit `b9e8c9321` — Chronicle 2 fixes for Bugs 3, 4, 5 (in-flight
  cache transfer correctness for gpt-oss) plus Bug 7 workaround
- Commit `da3cd54c7` — Bug 6 original fix
  (`_paras_reset_buffers` data_ptr stability via pre-grow + in-place zero)
- `runs/2026-04-25-bug3-qwen3-tp-ep-non-reproduction.md` —
  empirical evidence that the Bug 3 P0 regression does not reproduce
  at HEAD `7b072c461`
- Bug 5 + Bug 6 redesign (this commit) — replaces the pre-grow
  approach with `paras_save_cuda_graph_state` /
  `paras_load_cuda_graph_state` hooks on `AttentionBackend`
