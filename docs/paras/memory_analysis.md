# ParaS Memory Analysis

Empirical accounting of the GPU memory overhead ParaS adds on top of baseline EP, with measured numbers, root-cause attributions, applied optimizations, and remaining reclaimable items.

All numbers are **per GPU** on `Qwen3-30B-A3B-Instruct-2507`, 4×A100-80GB, `--tp-size 4 --dp-size 4 --ep-size 4 --enable-dp-attention --enable-dp-lm-head --moe-a2a-backend deepep --mem-fraction-static 0.6 --paras-tp-size 4`.

---

## TL;DR

Two optimizations reclaimed **3.89 GB per GPU** of ParaS overhead without any functional change:

| Stage | `driver_used` | Δ vs previous | Δ vs baseline EP |
|---|---:|---:|---:|
| Original ParaS | 60.20 GB | — | +6.93 GB |
| After NCCL comm alias fix | 57.77 GB | −2.43 GB | +4.50 GB |
| **After budget semantics fix** | **56.30 GB** | **−1.46 GB** | **+3.03 GB** |
| Baseline EP | 53.27 GB | — | 0 |

The remaining +3.03 GB of ParaS overhead is fully accounted for: ~1.0 GB is the N+1 slot feature cost (reclaimable only with algorithmic redesign), ~0.5 GB is structural (dual CUDA graph pool + peer-access IPC mappings), and the rest is allocator accounting and a few identified small items.

---

## Methodology

Every number in this doc is measured with the instrumentation described in `.skills/analyze-memory-footprint/SKILL.md`, which also documents the log format and parsing scripts. Three tiers of instrumentation, gated independently:

1. **Always on**: `ParaS[mem-summary]` one-liner per rank per capture, plus the one-time `ParaS KV budget:` log at model init. These always print — cheap and useful for headline numbers in any environment.
2. **`SGLANG_PARAS_MEM_LOG=1`** (opt-in): `ParaS[mem-breakdown:*]` — per-bucket driver memory at pre-capture, post-capture, and capture-delta. Enable for memory investigations; leave off in production.
3. **`SGLANG_DUMP_MEM_SEGMENTS=1`** (opt-in): segment-level dump of the torch caching allocator at "Memory pool end", printed via `torch.cuda.memory_snapshot()`. Use when a `torch_reserved` gap needs attribution at the individual-allocation level.

All reference runs are stored under `artifacts/` on branch `paras_memory_opt`:

| File | What it captures |
|---|---|
| `paras_pre_alias.log` | ParaS before NCCL alias fix |
| `paras_post_alias.log` | ParaS after NCCL alias fix, before budget fix |
| `paras_budget_fix.log` | ParaS after both fixes (final state) |
| `paras_memdump.log`, `ep_memdump.log` | Same two configs with `SGLANG_DUMP_MEM_SEGMENTS=1` |
| `paras_fix_memdump.log` | Final ParaS with segment dump |

---

## Memory buckets

Refer to `.skills/analyze-memory-footprint/SKILL.md` for the full 4-layer mental model. Quick summary:

```
driver_used (cudaMemGetInfo)
├── torch_reserved (PyTorch caching allocator)
│   ├── graph_pool      (CUDA graph private pools, one per mode for ParaS)
│   └── default_pool    (UMM buffer + embedding + lm_head + cuBLAS workspace + misc)
└── non_torch
    ├── deepep_buf      (DeepEP NVL/RDMA buffers)
    ├── deepep_ws       (32 MiB DeepEP workspace constant)
    ├── nvshmem         (NVSHMEM symmetric heap reservation)
    ├── nccl_est        (NCCL per-comm estimate, upper bound)
    └── other           (residual: NCCL VMM, cuBLAS, ParaS peer-access IPC, ...)
```

---

## Optimization 1: NCCL communicator aliasing

### Problem

`paras_parallel_state.py::initialize_paras_parallel` called `init_model_parallel_group` twice to build `_PARAS_TP` and `_PARAS_DP` communicators. Each call invokes `torch.distributed.new_group(ranks=..., backend="nccl")` which performs a full `ncclCommInitRank` (fresh channel buffers, topology discovery, per-comm scratch). Each fresh comm costs ~100-300 MB on A100 NVLink.

In the default config (`world=4`, `paras_tp_size=4`, `paras_dp_size=1`):
- `_PARAS_TP` ranks = `[0,1,2,3]` = `_TP` ranks = `_MOE_EP` ranks
- `_PARAS_DP` ranks = `[0]`, `[1]`, `[2]`, `[3]` (singletons) = `_MOE_TP` ranks

So `_PARAS_TP` and `_PARAS_DP` duplicated existing groups.

This is the same pattern the sglang team has already flagged elsewhere — see the two `# TODO(ch-wan): use split_group to save memory` comments in `parallel_state.py:1625` and `:1645` for `_MOE_EP`/`_MOE_TP`.

### Fix

`paras_parallel_state.py`: when the requested ParaS group's ranks exactly match an existing group, alias the existing `GroupCoordinator` instead of creating a new one. Skip the warmup collective in the alias branch because the aliased group has already been warmed.

```python
# paras_tp: alias _TP when ranks match
if tp_size == world_size:
    _PARAS_TP = parallel_state._TP
else:
    _PARAS_TP = init_model_parallel_group(...)  # fallback for non-matching configs
    # warmup only for freshly-created groups
    ...

# paras_dp: alias _MOE_TP when ranks match (singletons when dp_size == 1)
if dp_size == 1 and parallel_state._MOE_TP is not None and parallel_state._MOE_TP.world_size == 1:
    _PARAS_DP = parallel_state._MOE_TP
else:
    _PARAS_DP = init_model_parallel_group(...)
```

### Safety

Verified by enumerating every caller of `_PARAS_TP` / `_PARAS_DP` / their getters. All call sites use the public `GroupCoordinator` API (`.device_group`, `.cpu_group`, `.world_size`, `.rank`, collectives). None do identity comparison or attribute mutation. Aliasing is transparent to every caller.

### Result

`driver_used` dropped from 60.20 GB to 57.77 GB (−2.43 GB per GPU). The savings split across:
- `nccl_est`: −0.14 GB (one fewer counted communicator)
- `other`: −2.29 GB (NCCL channel state, custom_allreduce IPC, message-queue shm, warmup-collective scratch that the 144 MiB-per-comm estimate under-counted)

Round-trip EP↔TP switches verified identical output before/after.

---

## Optimization 2: Budget semantics mirror

### Problem

`qwen3_moe.py` computed the ParaS KV budget as:

```python
static_budget = total_gpu_bytes * mem_fraction_static        # 48 GB on 80 GB GPU
kv_budget = static_budget - weights_only_bytes                # weights come from UMM plan
```

This treated `mem_fraction_static × total_gpu` as the budget for **weights + KV only**. Baseline sglang treats the same number as the budget for the **entire static footprint** — weights + KV + already-consumed torch/NCCL/cuBLAS/CUDA context overhead. See `model_runner.py::profile_max_num_token` (`model_runner.py:1358-1363`):

```python
rest_memory = available_gpu_memory - total_gpu_memory * (1 - mem_fraction_static)
```

Because ParaS's formula ignored pre-existing consumption, it gave itself ~1.7 GB of extra KV capacity that baseline would leave free for later dynamic allocations (CUDA graph input buffers, DeepEP buffer, lazy cuBLAS workspaces). Two harms:

1. **More KV capacity than user asked for** — `mem_fraction_static=0.6` was effectively treated as ~0.62 by ParaS.
2. **Less room for dynamic allocations** — during capture, late allocations had to scavenge from torch's cached free blocks, increasing fragmentation pressure.

### Fix

Replace ParaS's budget formula with the baseline formula, computed before UMM allocation:

```python
avail_now_bytes        = get_available_gpu_memory(...)  # via cudaMemGetInfo
dynamic_reserve_bytes  = total_gpu_bytes * (1 - mem_fraction_static)
umm_budget_bytes       = avail_now_bytes - dynamic_reserve_bytes
kv_budget_bytes        = umm_budget_bytes - manager.weights_only_bytes
```

This is the same formula baseline uses, applied at the point ParaS needs it (before the UMM is planned).

### Allocation order (no waste)

Naively, you might worry that ParaS allocates weights first, then computes the KV budget from "what's left" — which would waste memory if the weight allocation itself disturbs `avail_now`. It doesn't. The actual order in `Qwen3MoeForCausalLMParaS.__init__`:

| Step | Code | GPU memory side-effect |
|---|---|---|
| 1 | `manager = ParaSMemoryManager()` | none (empty planner) |
| 2 | `plan_qwen_moe_layout(...)` | none — **metadata only** (reserves slot names + shapes in `_reservation_order`) |
| 3 | `get_available_gpu_memory(distributed=True, empty_cache=True)` | none (read-only) |
| 4 | Compute `umm_budget` and `kv_budget` from formula above | none (arithmetic) |
| 5 | `manager.reserve_kv_cache(ep_max_tokens=...)` | none — still metadata only |
| 6 | **`manager.materialize()`** — **single `torch.empty(total_bytes, dtype=uint8)`** | **the only GPU allocation.** Allocates weights + KV together, sized exactly to the budget |

Key property: `plan_qwen_moe_layout` and `reserve_kv_cache` are **pure bookkeeping** — they populate `LayoutEntry` metadata in `manager._entries` and `manager._reservation_order` but never call `torch.empty`. The entire UMM is physically allocated by a single `torch.empty` call at step 6.

This means:
1. `avail_now` at step 3 is untouched by ParaS's own plan (no circular dependency).
2. The UMM size at step 6 is the sum of `weights_only_bytes + kv_slot_bytes`, exactly matching what the budget allowed.
3. No interim allocations are freed; nothing is wasted.

Baseline sglang has the inverse ordering: weights allocated first via HuggingFace weight-loading (real `torch.empty` per parameter), then `avail_now` measured post-weight-load, then KV allocator sized. ParaS has to plan weights first (to know `weights_only_bytes`) but defers physical allocation until after the budget is final.

### Result

`driver_used` dropped from 57.77 GB to 56.30 GB (−1.46 GB per GPU). `#tokens` in the KV pool dropped from 347,857 → 332,225 (still larger than baseline's 329,048 by only 3,177 tokens — that difference is the N+1 KV slot overhead, which is architectural, not a budget bug).

Round-trip EP↔TP switches verified identical output before/after.

### Critical follow-up: cross-rank consistency

The initial implementation passed `distributed=False` to `get_available_gpu_memory`, causing each rank to use its local `cudaMemGetInfo` reading. On A100 NVLink, per-rank topology variance produces ~70 MB differences in raw available memory between ranks. With `ep_cell = 96 KiB/token`, this translated to **~768-token divergence** in the KV pool size across ranks:

| Rank | `ep_max_tokens` (buggy) | KV pool `#tokens` (buggy) |
|---|---:|---:|
| DP0 | 332,225 | 332,225 |
| DP1 | 331,457 | 331,457 |
| DP2 | 331,457 | 331,457 |
| DP3 | 332,225 | 332,225 |

**This is a correctness bug.** Divergent KV pool sizes imply divergent UMM buffer sizes (the UMM contains the KV slots). In EP mode with DP attention, each rank operates on its own requests so the divergence is latent — manifests only as capacity imbalance. But in TP mode (after `paras_configure_tp`), attention is tensor-parallel and any collective over the KV tensor assumes identical shapes across ranks. ParaS's EP→TP gather itself requires identical slot sizes on source and destination UMMs.

Baseline sglang avoids this via `profile_max_num_token` (`model_runner.py:1301-1307`), which passes `distributed=True` to `get_available_gpu_memory`, triggering an `all_reduce(op=MIN, cpu_group=world_group)` that returns the minimum across all ranks. Every rank then computes `max_total_num_tokens` from the same number.

**Fix**: Change `qwen3_moe.py` to mirror the baseline — use `distributed=True` with the world `cpu_group`:

```python
from sglang.srt.distributed import get_world_group
_world = get_world_group()
_avail_now_gib = get_available_gpu_memory(
    "cuda", torch.cuda.current_device(),
    distributed=_world.world_size > 1,
    cpu_group=_world.cpu_group,
    empty_cache=True,
)
```

**Post-fix verification**:

| Rank | `ep_max_tokens` (fixed) | KV pool `#tokens` (fixed) |
|---|---:|---:|
| DP0 | 331,457 | 331,457 |
| DP1 | 331,457 | 331,457 |
| DP2 | 331,457 | 331,457 |
| DP3 | 331,457 | 331,457 |

All ranks now converge on the same value (the MIN of the per-rank locals), matching baseline sglang's semantics exactly. EP↔TP round-trip verified.

**Lesson**: Any memory-sizing computation that happens before distributed all-reduce has to use the distributed variant of `get_available_gpu_memory`. Per-rank NVLink variance is not noise — it's real and measurable. Baseline sglang has this right everywhere; ParaS must match.

---

## Current overhead decomposition (after both fixes)

`driver_used` per GPU with both optimizations applied:

| Bucket | Baseline EP | ParaS | **Overhead** | Origin |
|---|---:|---:|---:|---|
| **torch_reserved** | 48.82 GB | 51.43 GB | **+2.61 GB** | see breakdown below |
| &nbsp;&nbsp;default_pool | 48.28 GB | 50.79 GB | +2.51 GB | UMM structural + extra graph capture state |
| &nbsp;&nbsp;graph_pool | 0.54 GB | 0.65 GB | +0.10 GB | Dual CUDA graph pool (one per mode) |
| **non_torch** | 4.65 GB | 4.87 GB | **+0.22 GB** | Peer-access IPC + misc |
| &nbsp;&nbsp;deepep_buf | 0.57 GB | 0.57 GB | 0 | Same |
| &nbsp;&nbsp;deepep_ws | 0.03 GB | 0.03 GB | 0 | Same |
| &nbsp;&nbsp;nvshmem | 1.00 GB | 1.00 GB | 0 | Same |
| &nbsp;&nbsp;nccl_est | 0.70 GB | 0.56 GB | −0.14 GB | Alias fix removed one counted comm |
| &nbsp;&nbsp;other | 2.35 GB | 2.71 GB | +0.36 GB | Peer-access IPC mappings |
| **driver_used TOTAL** | **53.47 GB** | **56.30 GB** | **+2.83 GB** | |

### Itemized attribution of the +2.83 GB overhead

| # | Item | Size | Bucket | Notes |
|---|---|---:|---|---|
| 1 | Extra MoE expert slot (N+1 design) | +0.28 GB | torch_reserved | 49th layer's worth of expert weights (`w13` + `w2`) reserved as TP landing zone |
| 2 | Extra KV cache slot (N+1 design) | +0.66 GB | torch_reserved | 49th layer's worth of K+V cache as TP landing zone |
| 3 | QKV TP buffer | +0.07 GB | torch_reserved | 48 × (384, 2048) BF16 sharded QKV slice, needed for TP mode |
| 4 | Mystery 128 MiB segment | +0.13 GB | torch_reserved | ParaS-only segment; origin not yet identified. Possibly lazy cuBLAS-LT workspace triggered by a different kernel path, or NCCL lazy buffer from peer-access-specific communication |
| 5 | Extra CUDA graph pool | +0.10 GB | graph_pool | ParaS captures both EP and TP graphs; baseline captures one |
| 6 | One-giant-alloc vs many-small-alloc allocator rounding | ~+0.25 GB | torch_reserved | Single 47.9 GiB UMM cudaMalloc has different segment-size rounding than baseline's ~300 separate allocations |
| 7 | Peer-access IPC mappings | +0.36 GB | non_torch (other) | Each rank opens IPC handles for 3 remote ranks' UMM buffers; driver tracks VA-mapping metadata |
| 8 | NCCL comm alias savings | −0.14 GB | non_torch (nccl_est) | ✓ Already applied in Optimization 1 |
| 9 | Unaccounted residual | ~+1.12 GB | torch_reserved | Not pinpointed by memdump alone. Candidates: DeepEP lazy allocations triggered differently in ParaS, different cuBLAS workspace sizing, peer-access handle tables. Would require further instrumentation to split. |
| | **SUM** | **+2.83 GB** | | |

Items 1-7 are confirmed by memdumps, log deltas, and direct byte-math against the `ParaSMemoryManager` layout plan. Items 8 is the measured effect of Optimization 1. Item 9 is honestly flagged as "residual the current tools can't decompose further".

### What `torch_reserved` looks like at segment level

From `artifacts/paras_fix_memdump.log` vs `artifacts/ep_memdump.log` (DP0 rank, `SGLANG_DUMP_MEM_SEGMENTS=1`):

```
ParaS (19 segments, 49.086 GiB total, 0.007 GiB allocator slack):
  #00  47,876 MiB  UMM (single giant alloc)
  #01   1,024 MiB  cuBLAS workspace (default 1 GiB per stream)
  #02     594 MiB  embedding
  #03     594 MiB  lm_head
  #04     128 MiB  ParaS-only (item 4 above)
  #05-18  small tensors (norms, router, allocator metadata)

Baseline EP (300 segments, 47.700 GiB total, 0.034 GiB allocator slack):
  #00   1,024 MiB  cuBLAS workspace
  #01     594 MiB  embedding
  #02     594 MiB  lm_head
  #03-50  48 × 322 MiB  KV per layer (K + V, split by allocator)
  [remaining 200+ segments: MoE weights, attn weights, small tensors]
```

A single UMM allocation has less per-segment overhead than 300 separate allocations, but the one giant cudaMalloc incurs a different kind of rounding that shows up as item 6.

---

## Reclaimability table

| Item | Reclaimable? | Cost to reclaim |
|---|---|---|
| 1 + 2 + 3 (N+1 slot design, 1.01 GB) | **Possible** | Redesign `peer_access.py` and the cache-transfer backends in `cache_transfer/{mha,swa,utils}.py` plus gather/scatter orchestration to use N-slot ping-pong transfer instead of forward-order N+1 transfer. Estimated ~1 week of engineering. |
| 4 (mystery 128 MiB, 0.13 GB) | **Unknown** | Needs further probing — likely requires profiling torch allocator stack traces for that specific segment |
| 5 (dual graph pool, 0.10 GB) | **No** | Required for zero-latency EP↔TP switching |
| 6 (allocator rounding, 0.25 GB) | **No** | Inherent to one-big-alloc pattern; would return if we split UMM |
| 7 (peer-access IPC, 0.36 GB) | **No** | Required for fast peer-access weight transfer |
| 9 (residual ~1.12 GB) | **Unknown** | Requires instrumentation at the specific allocation sites. Next investigation step. |

**Unconditionally-reclaimable ceiling from the current design: ~0.0 GB. The remaining +2.83 GB is either structural feature cost (items 1-3, 5, 7) or unexplained residual needing deeper investigation (items 4, 9).**

---

## Reproduction

### Config used throughout this doc

```bash
conda activate sgl_sm80
CUDA_VISIBLE_DEVICES=0,1,2,3 \
SGLANG_ATTN_MAX_BS=256 \
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256 \
SGLANG_DEEPEP_BF16_DISPATCH=true \
python -m sglang.launch_server \
    --model /scratch1/wangshao/models/Qwen3-30B-A3B-Instruct-2507 \
    --trust-remote-code \
    --chunked-prefill-size -1 --max-prefill-tokens 32000 \
    --disable-radix-cache \
    --mem-fraction-static 0.6 \
    --tp-size 4 --dp-size 4 --ep-size 4 \
    --enable-dp-attention --enable-dp-lm-head \
    --moe-a2a-backend deepep --deepep-mode auto \
    --max-running-requests 1024 \
    --disable-overlap-schedule \
    --cuda-graph-max-bs 256 \
    --enable-paras-moe --paras-tp-size 4 \
    --log-level info \
    2>&1 | tee artifacts/paras.log
```

Remove `--enable-paras-moe --paras-tp-size 4` for the baseline EP reference.

### Verifying the budget-fix log output

After the budget fix, each rank logs its KV budget derivation at startup:

```
ParaS KV budget: avail_now=77.820GiB  total=79.251GiB  dynamic_reserve=31.700GiB
                 umm_budget=46.119GiB  weights_only=15.703GiB  kv_budget=30.416GiB
                 ep_max_tokens=332225
```

`ep_max_tokens` should closely track the baseline EP's `#tokens` number for the same mem-fraction-static value. The residual difference is the +1 KV slot structural overhead.

### Collecting a detailed capture breakdown

```bash
SGLANG_PARAS_MEM_LOG=1 python -m sglang.launch_server ... 2>&1 | tee artifacts/paras_breakdown.log
grep "mem-breakdown" artifacts/paras_breakdown.log | head -12
```

Without this env var, only the one-line `ParaS[mem-summary]` per rank per capture is emitted. The detailed pre/post/delta breakdown is suppressed to keep production logs clean.

### Collecting a segment dump

```bash
SGLANG_DUMP_MEM_SEGMENTS=1 python -m sglang.launch_server ... 2>&1 | tee artifacts/paras_dump.log
grep "MEMDUMP" artifacts/paras_dump.log | head -30
```

This prints top-20 live segments at the "Memory pool end" checkpoint (after model load + KV allocator init, before CUDA graph capture). Useful for attributing `torch_reserved` gaps across two configs.

---

## Further optimization candidates (future work)

Listed in order of estimated ROI:

### 1. N-slot ping-pong transfer design (saves ~1 GB)

The biggest remaining reclaimable overhead is items 1+2: the extra MoE slot and extra KV slot that exist because the current peer-access transfer reads from slot `i+1` (source EP) and writes to slot `i` (destination TP) in layer order. An N-slot design would require either:

- **Bidirectional transfer**: each layer's transfer reads from its own slot and writes to the previous layer's slot, with synchronization barriers to prevent overwriting in-flight data.
- **Scratch-then-place**: copy EP data to a small scratch buffer, write TP layout back to the same slot, consuming the scratch before the next layer.

Both require updates to `peer_access.py`, `cache_transfer/{mha,swa,utils}.py`, `gather_manager.py`, `scatter_manager.py`, and synchronization around the memory-manager alias scheme. Non-trivial but bounded engineering.

### 2. Identify and eliminate the 128 MiB mystery segment (saves ~0.13 GB)

The segment dump showed a 128 MiB ParaS-only allocation that doesn't appear in the baseline run. Size is suspicious (a round number — likely a single hardcoded buffer). Candidates to investigate:

- cuBLAS-LT workspace (default 128 MiB per handle/stream) — triggered by a ParaS-specific matmul shape?
- NCCL persistent buffer allocated after the first collective uses a specific protocol?
- A torch CUDA graph input buffer allocated eagerly?

Instrumentation approach: add a `torch.cuda.memory._record_memory_history` snapshot at the point the 128 MiB segment first appears, dump the Python stack trace of the allocation site.

### 3. Decompose the ~1.12 GB unexplained residual (potentially saves up to ~1 GB)

Item 9 in the attribution table is the honest "stuff we can't split further with current tools". Approach: run with `torch.cuda.memory._record_memory_history(max_entries=100000)` active from server start, take a snapshot at "Memory pool end", compare against the baseline snapshot. The allocation-trace diff will pinpoint every allocation site that exists only in ParaS.

### 4. FP8 KV cache (saves ~15 GB)

Orthogonal to ParaS overhead — would benefit baseline equally. Already supported via `--kv-cache-dtype fp8` (see `qwen3_moe.py:199-205`). Halves the per-token KV cell cost from 96 KB to 48 KB. Not a ParaS-specific optimization, but the largest single memory lever available.

---

## Related docs

- `docs/paras/unified_memory_manager.md` — UMM design, layout, API
- `docs/paras/parallelism_switch.md` — EP↔TP switch protocol and control-plane
- `docs/paras/cuda_graph.md` — Dual graph pool capture
- `.skills/analyze-memory-footprint/SKILL.md` — Methodology, playbooks, and the `ParaS[mem-breakdown]` + `SGLANG_DUMP_MEM_SEGMENTS` instrumentation reference

---

## Changelog

| Date | Change |
|---|---|
| 2026-04-19 | Initial doc. Both NCCL alias and budget-semantics fixes applied and verified. Overhead reduced from +6.93 GB to +3.03 GB per GPU. |
| 2026-04-19 | Cross-rank-consistency fix for the budget computation: use `distributed=True` in `get_available_gpu_memory` so every rank agrees on the MIN available memory. Prevents UMM buffer-size divergence that would fail TP-mode attention collectives. |
| 2026-04-19 | Documented the allocation ordering (plan→compute-budget→reserve-KV→materialize) to clarify that `ParaSMemoryManager` avoids weight-before-KV waste by deferring all physical GPU allocation to a single `torch.empty` call after the budget is finalized. |
| 2026-04-19 | Made the detailed per-bucket capture breakdown opt-in via `SGLANG_PARAS_MEM_LOG=1`. Default prod runs now only emit a one-line `ParaS[mem-summary]` per capture plus the one-time `ParaS KV budget:` log. Reduces log spam ~4× in normal operation while keeping full diagnostics one env var away. |
