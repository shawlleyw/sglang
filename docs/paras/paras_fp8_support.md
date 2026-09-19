# Proposal: ParaS Support for FP8 Weights

This is an unimplemented design proposal. The current runtime accepts
unquantized BF16 weights and rejects FP8 weight quantization. References to
the N+1-slot design below describe the older allocator; a future implementation
must use the current [unified memory layout](unified_memory_manager.md).

## Reader

This document is written for a ParaS contributor who already understands the
core EP↔TP switch, the unified memory manager, the N+1 slot design, and the
GPT-OSS adaptation that introduced replicated read-only tensors for biases
and sinks. See `parallelism_switch.md`, `unified_memory_manager.md`, and
`gpt_oss_support.md` first. This document proposes how ParaS could support FP8
quantized models (Qwen3-MoE FP8, DeepSeek-V3 native FP8, GPT-OSS FP8) by
extending the replicated-tensor pattern to per-block FP8 scales.

ParaS V1 supports BF16 for both Qwen3-MoE and GPT-OSS. The kernels and
memory manager already accept FP8 dtypes, but FP8 weight scales are not yet
wired through the EP↔TP switch path. This document specifies the design.

## Why FP8 Needs Special Treatment

FP8 inference stores weights at 1 byte per parameter and recovers numerical
range with a separate scale tensor. Block-wise FP8 (the production scheme
used by Qwen3-MoE FP8, DeepSeek-V3, and GPT-OSS FP8) tiles each weight
matrix into 128×128 blocks and stores one float32 scale per block. The
GEMM kernel reads the FP8 byte and multiplies by the matching scale to
recover the dequantized value.

Three properties of FP8 affect ParaS:

1. **The weight payload is half the bytes of BF16.** NVLink transfer time
   for the EP→TP switch should drop roughly in half. The existing v2
   peer-access kernels handle this transparently because they operate on
   raw bytes via `int4` vectorized stores; the `elem_size` parameter in
   `paras/csrc/peer_access_transfer.cu` already plumbs the dtype byte
   width end-to-end.

2. **Scales are a second tensor that the GEMM consumes alongside
   weights.** Without scales, the FP8 weight bytes mean nothing. ParaS must
   make the right scales available to the GEMM kernel in both EP and TP
   modes.

3. **Scales are tiny.** A 4096×4096 FP8 weight matrix has 16 MB of weight
   bytes and 4 KB of scales (32×32 float32). The size ratio is roughly
   4000:1.

The third property is the design lever. Scales are small enough that
replicating them on every rank is cheaper than transferring them at
switch time.

## Scale Tensor Sizing

For Qwen3-MoE FP8 with `weight_block_size=[128, 128]`, the canonical
production layout, scales attach to expert weights as follows:

| Tensor | Weight shape | Scale shape | Block size |
|---|---|---|---|
| `w13_weight` (gate_up_proj) | `(E, 2I, H)` FP8 | `(E, ceil(2I/128), ceil(H/128))` float32 | 128×128 |
| `w2_weight` (down_proj) | `(E, H, I)` FP8 | `(E, ceil(H/128), ceil(I/128))` float32 | 128×128 |

For Qwen3-30B-A3B (E=64, I=1536, H=2048):

| Component | Full shape (all experts) | Bytes per layer | Bytes for 48 layers |
|---|---|---|---|
| `_full_w13_scale` | `(64, 24, 16)` float32 | 96 KB | ~4.6 MB |
| `_full_w2_scale` | `(64, 16, 12)` float32 | 48 KB | ~2.3 MB |
| **Combined scales** | | **144 KB** | **~6.9 MB** |

For DeepSeek-V3 (E=256, I=2048, H=7168, 60 layers), full scales total
roughly 158 MB per rank. Both numbers are negligible against the multi-GB
weight footprint and the 80 GB GPU memory.

## Design: Replicated Full Scale, View-Based Dispatch

ParaS stores the full scale tensor for all global experts on every rank,
and exposes EP and TP modes through Parameter views into the same
underlying storage. The mode switch rebinds Parameters; the scale data
never moves.

This pattern matches the existing GPT-OSS treatment of `w13_weight_bias`,
`w2_weight_bias`, and `sinks` documented in `gpt_oss_support.md`. The
mechanism transfers directly:

```
Per-rank shared storage:
    self._full_w13_scale  shape (num_global_experts, ceil(2I/B), ceil(H/B))
    self._full_w2_scale   shape (num_global_experts, ceil(H/B),  ceil(I/B))

EP-mode Parameter views:
    ep_experts.w13_weight_scale = _full_w13_scale[ep_start:ep_end]
    ep_experts.w2_weight_scale  = _full_w2_scale [ep_start:ep_end]

TP-mode Parameter views:
    tp_experts.w13_weight_scale = _full_w13_scale[:, w13_block_slice, :]
    tp_experts.w2_weight_scale  = _full_w2_scale [:, :, w2_block_slice]
```

The `w13_block_slice` and `w2_block_slice` indexers depend on whether the
model uses Qwen3's concat layout or GPT-OSS's interleaved layout for
gate-and-up; the next section gives the exact expressions.

### Switch Behavior

Mode transitions for scales become trivial:

| Step | Action |
|---|---|
| Init | Allocate `_full_*_scale` zero tensors (one shared per layer per rank) |
| Load | Checkpoint loader writes full scales into `_full_*_scale` on every rank |
| Finalize | Build EP and TP Parameter views into `_full_*_scale` |
| EP→TP switch | Rebind `self.experts.w*_weight_scale` to TP view; **no transfer** |
| TP→EP switch | Rebind `self.experts.w*_weight_scale` to EP view; **no transfer** |

No NCCL collectives, no peer-access kernels, no UMM staging buffers, and
no layer-order constraints apply to scales. The scale path stays out of
the critical switch loop.

### TP Slice Layouts

The TP-mode scale view must align with the TP-mode weight view so the
GEMM reads matching block scales for each weight block. Two layouts apply.

**Qwen3 (concat layout)**: w13 weight stores `[gate(I) | up(I)]`. Under TP
with paras_tp_size=`P` and block size B, rank r owns weight columns
`[r*I/P, (r+1)*I/P)` from gate and the same range from up. The matching
scale slice in block coordinates is the concatenation of two block ranges:

```python
i_blocks_per_tp = (I // P) // B  # blocks per gate per peer
gate_start = r * i_blocks_per_tp
up_start   = (I // B) + r * i_blocks_per_tp
tp_w13_scale = torch.cat([
    full_w13_scale[:, gate_start:gate_start + i_blocks_per_tp, :],
    full_w13_scale[:, up_start  :up_start   + i_blocks_per_tp, :],
], dim=1)
```

**GPT-OSS (interleaved layout)**: w13 weight stores `[g0, u0, g1, u1, ...]`.
Under TP, rank r owns the contiguous slab `[2*r*I/P, 2*(r+1)*I/P)`. Each
gate-up pair stays adjacent. The matching scale slice is a single
contiguous block range:

```python
two_i_blocks_per_tp = (2 * I // P) // B  # gate+up block pairs per peer
tp_w13_scale = full_w13_scale[
    :,
    r * two_i_blocks_per_tp : (r + 1) * two_i_blocks_per_tp,
    :,
]
```

For w2 in both layouts, the TP cut is on the last dimension (I), and the
scale slice is uniformly:

```python
i_blocks_per_tp = (I // P) // B
tp_w2_scale = full_w2_scale[
    :,
    :,
    r * i_blocks_per_tp : (r + 1) * i_blocks_per_tp,
]
```

The `_paras_interleaved_w13` flag set in `paras_init_moe` (existing field
used for weight transport) selects the correct w13 slice expression.

### Block Alignment Constraints

The slice expressions assume `I/P` is a multiple of B and `2*I/P` is a
multiple of B for the interleaved case. For Qwen3-30B-A3B (I=1536, B=128)
this holds for any P that divides 12 (P ∈ {1, 2, 3, 4, 6, 12}). For
DeepSeek-V3 (I=2048, B=128) this holds for any P that divides 16.

When alignment fails, the TP shard would split a 128-element block. ParaS
must detect this case at init time and either reject the configuration or
fall back to a zero-padded full-scale slice; production deployments at
paras_tp_size ∈ {2, 4, 8} satisfy alignment for all known FP8 MoE
checkpoints. The init-time validator emits a diagnostic and aborts
otherwise.

## Why Not Transfer-Based

An alternative design treats scales as a small second weight: extend the
N+1 slot system in the unified memory manager, add scale staging buffers
for the NCCL paths, and fuse scale transfer into the peer-access kernels
as optional arguments. This is the symmetry argument; it preserves the
"every persistent tensor lives in UMM" invariant.

The replicated-view design wins on every metric we measured.

| Aspect | Transfer-based | Replicated-view |
|---|---|---|
| New CUDA kernel code | ~120 lines across 4 kernels | 0 |
| New NCCL transfer code | ~150 lines | 0 |
| New UMM reservations | N+1 scale slots + staging | 0 |
| Switch latency for scales | ~1-5 ms (NCCL) or ~0.1 ms (kernel) | 0 |
| Memory cost on each rank | EP-local scale (~1 MB Qwen3) | Full scale (~7 MB Qwen3) |
| Test scope | Bitwise transfer correctness across 4 paths × 2 directions | View slicing correctness |
| CUDA graph compatibility | Requires per-mode capture of scale data_ptr | Already handled by existing dual-pool capture for biases |

The 6 MB extra memory per rank (Qwen3) buys us the elimination of every
moving part. The pattern also extends naturally to per-tensor FP8 scales
and to MXFP4 scales should those ever be needed; in both cases the size
ratio favors replication even more strongly.

The current `plan_qwen_moe_layout` in `paras_memory_manager.py:737-757`
already reserves FP8 scales, but the reservation is broken for ParaS:
scales sit outside the N+1 slot system, so EP and TP scale views would
alias the same physical memory. Removing this reservation is a net
simplification, not just an avoidance.

## Implementation Plan

The change is small and localized. Five steps, each independently
verifiable.

### Step 1: Verify Loader Writes Full Scales

The pattern depends on `Fp8MoEMethod` writing full scales when given a
`(num_global_experts, ...)` shaped scale Parameter. The bias path proves
the mechanism works; this step confirms the FP8 scale loader follows the
same convention. A short unit test allocates a full-shape scale tensor
and runs the loader against a checkpoint with a known per-expert pattern.

### Step 2: Allocate Full Scales in `paras_init_moe`

In `paras/layers/paras_moe_block.py`, add a `_build_full_scale` helper
parallel to `_build_full_bias` (lines 64-80). After the existing bias
re-registration block (lines 181-190), detect FP8 from `quant_config`
and from the presence of `w13_weight_scale` on the FusedMoE-allocated
`ep_experts`. Allocate `self._full_w13_scale` and `self._full_w2_scale`
zero-filled tensors of the global-expert shape, then re-register the
ep_experts scale Parameters to wrap the full tensors.

The base FusedMoE module created scale Parameters of shape
`(num_local_experts, ...)`. Replacing them with `(num_global_experts, ...)`
parameters works because the checkpoint loader indexes by global expert
ID; only experts in `[ep_start, ep_end)` receive writes from this rank's
checkpoint shard, but the parameter has slots for every global expert
ID, so writes land at the correct offset. This is the same trick the
bias path uses.

### Step 3: Add `paras_finalize_moe_scale_views`

Mirror `paras_finalize_moe_bias_views` (lines 396-443). After
`load_weights` populates `_full_w*_scale`, build:

- `ep_experts.w13_weight_scale`, `ep_experts.w2_weight_scale` as local
  expert slices `_full_*_scale[ep_start:ep_end]`.
- `tp_experts.w13_weight_scale`, `tp_experts.w2_weight_scale` as TP
  block-aligned slices using the layout-conditional expressions from the
  TP Slice Layouts section.

Both views inherit dtype float32 and `requires_grad=False`. The
`_paras_interleaved_w13` field selects between the concat and interleaved
slicers for w13.

### Step 4: Wire Into Model Init

In `paras/models/qwen3_moe.py` and `paras/models/gpt_oss.py`, add a call
to `paras_finalize_moe_scale_views()` at the same point that already
calls `paras_finalize_moe_bias_views()`. No other model-level changes are
needed.

### Step 5: Remove the Broken UMM Scale Reservation

Delete the FP8 scale block in `paras_memory_manager.py` at lines 737-757.
Scales no longer live in UMM. The associated `experts.w*_weight_scale`
alias entries created at lines 727-732 also drop. The unified memory
manager keeps its existing weight slots, KV cache reservation, attention
weights, and staging buffers; only the scale-related code goes away.

## What Stays Untouched

The replicated-view design changes nothing in the following components:

- `paras/csrc/peer_access_transfer.cu`: no kernel changes. The existing
  v2 and ep kernels already handle FP8 weight bytes via the dtype-agnostic
  `int4` vectorized stores.
- `paras/csrc/binding.cpp` and `paras/peer_access.py`: no new wrapper
  parameters. The existing `elem_size=1` path covers FP8 weights.
- `paras_configure_tp_all_gather`, `paras_configure_tp_all_to_all`,
  `paras_configure_tp_fused_peer_access_kernel`: no scale handling. The
  weight-only logic continues unchanged.
- `paras_configure_ep_mlp_naive`, `paras_configure_ep_fused_peer_access_kernel`:
  no reverse scale handling.
- The unified memory manager, except for the deletion in step 5.
- The CUDA graph dual-capture path. The existing capture treats scale
  Parameters the same way it treats bias Parameters; both work via
  Parameter rebinding without kernel reissue.

## Testing Strategy

The test scope shrinks from "transfer correctness across 4 paths × 2
directions" to "view slicing correctness" because no transfer occurs.

The test file `test/srt/paras/test_weight_transfer_fp8.py` follows the
structure of `test_weight_transfer.py` but with these adaptations:

1. Build a minimal `_full_w13_scale` and `_full_w2_scale` populated with
   deterministic per-expert per-block float32 values seeded by rank.
2. Call `paras_finalize_moe_scale_views()` against a mock mixin that
   sets `_paras_interleaved_w13` to both False (Qwen3 concat) and True
   (GPT-OSS interleaved).
3. Assert via `torch.equal`:
   - `ep_experts.w13_weight_scale` is bitwise equal to
     `_full_w13_scale[ep_start:ep_end]`.
   - `tp_experts.w13_weight_scale` is bitwise equal to the expected
     concat-or-interleaved TP block slice.
   - The same equalities for w2.
4. Verify storage sharing: the `data_ptr()` of EP and TP views differs
   by the expected byte offset, and both fall inside the
   `_full_*_scale` buffer's address range.
5. Round-trip: after `paras_configure_tp` followed by
   `paras_configure_ep`, the underlying `_full_*_scale` content must be
   bitwise unchanged. No write path touches scales, so this assertion
   guards against accidental in-place modification.
6. Cross-rank consistency: every rank must hold an identical
   `_full_*_scale` after load. The EP slices on different ranks must
   differ (different `ep_start`); the TP slices must differ (different
   block ranges).

Tests run on 4×A100 via `torchrun --nproc_per_node=4`, matching the BF16
test invocation. A100 has no FP8 tensor cores, but the test path never
executes a FP8 GEMM. It only exercises view slicing and storage sharing,
both of which are pure PyTorch operations independent of compute
capability.

## Memory Cost Accounting

The replicated-view design adds full-scale storage on every rank
(roughly 7 MB for Qwen3-30B-A3B, 158 MB for DeepSeek-V3). The
transfer-based alternative would have stored EP-local scales (roughly
1 MB Qwen3, 20 MB DeepSeek) plus staging buffers for the NCCL path
(equal to one EP-local scale, doubled for the overlap path).

The net memory delta on each rank is small in absolute terms and
negligible relative to weight bytes. The accounting at
mem_fraction_static=0.6 on A100-80GB does not require any KV budget
adjustment.

## Limitations and Future Work

The V1 design covers block-wise FP8 with `weight_block_size=[128, 128]`,
the universal production scheme for FP8 MoE checkpoints sglang serves.
Three configurations remain as future work.

**Per-tensor FP8 scales.** Per-tensor scales attach as `(E, 2)` for w13
and `(E,)` for w2. Production checkpoints rarely use this scheme, but
hand-calibrated FP8 paths sometimes do. The replicated-view pattern
extends straight through: `_full_w13_scale_per_tensor` of shape
`(num_global_experts, 2)` with EP slice `[ep_start:ep_end, :]` and TP
slice equal to the full tensor on every rank. Adds roughly 30 lines if
ever needed.

**MXFP4 (GPT-OSS optional checkpoint format).** MXFP4 packs FP4 weights
with E8M0 scales over 32-element blocks. The current loader decompresses
MXFP4 to FP8 at load time, so MXFP4 scales never reach the ParaS buffer
in the steady state. If a future design preserves MXFP4 at runtime, the
same replicated-view pattern applies with a uint8 scale dtype and a
32-element block size; the slice expressions adjust accordingly.

**FP8 KV cache.** The unified memory manager supports FP8 weight dtypes
but reserves the KV cache as BF16. FP8 KV cache halves KV memory and
roughly doubles token capacity. The work is orthogonal to FP8 weight
support and follows the existing KV cache reservation path; see the
relevant entry in `unified_memory_manager.md` Future Work.

**Block alignment validation under unusual TP sizes.** The init-time
validator should surface block-alignment failures with a clear message
before the model attempts to run. The check is one line per layer per
weight, but its absence in the current code would manifest as silently
incorrect TP slicing on ill-formed configurations.

## References

- `parallelism_switch.md`: overall ParaS EP↔TP switch design
- `unified_memory_manager.md`: memory layout, N+1 slot design, current
  FP8 reservation that this design replaces
- `gpt_oss_support.md`: replicated-bias and replicated-sinks patterns
  that motivate the replicated-scale design
- `nvlink_peer_access_weight_transfer.md`: peer-access weight kernels
  that already accept FP8 dtype via `elem_size=1`
- `python/sglang/srt/layers/quantization/fp8.py`: the `Fp8MoEMethod`
  class that defines the canonical scale shapes consumed by the GEMM
- `python/sglang/srt/paras/layers/paras_moe_block.py:64-190`: the
  bias-replication implementation this design parallels
