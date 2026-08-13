# ParaS Unified Memory Layout for EP↔DP×TP

**Status:** Implemented in [`paras_memory_manager.py`](../../python/sglang/srt/paras/paras_memory_manager.py) (`_place_paras_run`) and the switch orchestration (`paras_model.py`, `gather_manager.py`, `scatter_manager.py`, `peer_access.py`, `scheduler_paras_mixin.py`). Address math proven in [`benchmark/paras/paras_layout.py`](../../benchmark/paras/paras_layout.py) (`compute_layout` + `check_safe`, 40k-case fuzz); validated on 4 GPUs by `test_roundtrip.py` (EP↔TP↔EP round-trip, KV + weights + no-leak) and `test_weight_transfer.py` (NCCL and peer-access kernels, both directions, bitwise-exact).

**Scope:** This document extends the [ParaS Unified Memory Manager](unified_memory_manager.md) to the asymmetric EP↔(DP×TP) case and replaces the `N+1` identical MoE weight slots. It works for any `ep_size ≥ tp_size`, with uniform or hybrid (sliding-window + full) attention cache.

**Two refinements landed during implementation.** (1) The cache tail anchor is the general `ANCHOR = max_i(ct_i + Σ_{k>i}(ct_k − ce_k))`, which reduces to `max(ct)` when `ct_i ≤ ce_i` but also tolerates `ct_i > ce_i` (from GQA head-division flooring when `num_kv_heads % tp_size ≠ 0`), so no `ct ≤ ce` precondition is required. (2) The per-layer shapes match the ParaS forward exactly: EP holds `num_experts/ep_size` experts with the **full** intermediate; TP holds **all** `num_experts` with the intermediate sharded by `tp_size` (equal bytes at `G=1`, TP `G`× larger for `G>1`). The switch loop orders are flipped from the old slot design: EP→TP runs cache-then-weights in **reverse** layer order; TP→EP runs weights-then-cache in **forward** order.

## Overview

The [unified memory manager](unified_memory_manager.md) holds all persistent ParaS state in one contiguous `uint8` buffer and switches between Expert Parallelism (EP) and Tensor Parallelism (TP) by reinterpreting the same bytes. The current layout reserves `N+1` identical MoE weight slots and aliases EP and TP onto neighboring slots, which is correct only when EP and TP occupy the same bytes per layer (`SE == ST`).

EP↔(DP×TP) breaks that. On a `W = G·T` grid (`G = dp_size`, `T = tp_size`, `ep_size = W`), EP weights are small (experts sharded across all `W` ranks) while DP×TP weights are `G` times larger (sharded across only `T`); the KV cache runs the other way, EP large and TP small. This design packs both interpretations into one combined `[weights | cache]` run in which the large TP weights overlap the EP cache, so the buffer is the shared per-mode footprint `B` plus **one layer**.

## Background

Two properties frame the design. EP and TP are never live at the same time, so a switch overwrites the buffer with the other mode's interpretation and the layout only holds one mode at rest. The transfer is cross-GPU: each kernel reads its local source region and writes a peer's destination region, so the hazard is a same-GPU race where a peer writes this GPU's destination bytes for a layer while this GPU still reads its source bytes for the same layer. Keeping each layer's EP and TP regions disjoint on every GPU removes it. Related control flow and kernels are in [`parallelism_switch.md`](parallelism_switch.md), [`nvlink_peer_access_weight_transfer.md`](nvlink_peer_access_weight_transfer.md), and [`nvlink_peer_access_kv_cache_transfer.md`](nvlink_peer_access_kv_cache_transfer.md).

## Per-Layer Sizes and the One Invariant

For `N` layers, write per-layer, per-GPU byte sizes `we[i]`, `wt[i]` (weights) and `ce[i]`, `ct[i]` (cache). `ep_size ≥ tp_size` gives two structural facts:

- **weights:** `we[i] ≤ wt[i]` (`wt[i] = G · we[i]`),
- **cache:** `ce[i] ≥ ct[i]` (EP holds the larger cache per layer).

Correctness rests on the **single invariant `ce[i] ≥ ct[i]`**, and `compute_layout` asserts it, so a violation (only possible if `tp_size > ep_size`) fails at build time rather than corrupting memory. Nothing else is assumed: no uniformity across layers, no ordering, no particular `G`.

## The Layout: Four Anchors

Let `P` be the base offset, `Σx` a total, and all regions 256-byte aligned. Read per mode, each is `weights | pad | cache` in address order.

```
EP weights:  EW[i].start = P + cumsum(we)[i]                         # forward from P
EP cache:    EC[i].start = P + Σwe + PAD + cumsum(ce)[i]             # forward, after the seam
             EP_end      = EC[N-1].end
TP weights:  TW[i].start = (P + we[0]) + cumsum(wt)[i]               # first layer at the head
TP cache:    tc_end      = EP_end + max(ct)                          # anchor: last layer ends here
             TC[i].start = (tc_end - Σct) + cumsum(ct)[i]            # forward, same as the others

PAD = max(0, tp_w_end - (P+Σwe) - Σce + Σct - max(ct))               # tp_w_end = P + we[0] + Σwt
```

`EP` is packed tight as `[weights | PAD | cache]`. The head offset `we[0]` keeps TP weight layer 0 off EP weight layer 0. The large TP weights then extend forward into the EP-cache region — that overlap is what keeps the buffer near `B`. The TP cache is anchored so its **last layer ends at `EP_end + max(ct)`** and is laid forward from `tc_end − Σct`, identical in form to the other three blocks. `PAD` is the seam that keeps the forward TP weights off the TP cache.

```
P ── EW0 EW1 … EW(N-1) │ PAD │ EC0 EC1 … EC(N-1) ──────── EP_end ─── +max(ct)
     ␣we0␣ TW0 TW1 … TW(N-1) ……(over EP cache)…… TC0 TC1 … TC(N-1) ── tc_end
       head: EP-only (we0)                                    tail: TP-only (max ct)
```

## Reference Code

The exact address math ([`paras_layout.py`](../../benchmark/paras/paras_layout.py)):

```python
def compute_layout(we, wt, ce, ct, align=ALIGN, P=0):
    N = len(we)
    we = [_au(x) for x in we]; wt = [_au(x) for x in wt]
    ce = [_au(x) for x in ce]; ct = [_au(x) for x in ct]
    assert all(ct[i] <= ce[i] for i in range(N))            # the one invariant
    sum_we, sum_wt, sum_ce, sum_ct = sum(we), sum(wt), sum(ce), sum(ct)
    max_ct = max(ct)

    w_end    = P + sum_we
    tp_w_end = P + we[0] + sum_wt
    PAD      = _au(max(0, tp_w_end - w_end - sum_ce + sum_ct - max_ct))
    EP_end   = w_end + PAD + sum_ce
    tc_end   = EP_end + max_ct

    addr = {}
    off = P
    for i in range(N): addr[("ep", i, "w")] = (off, we[i]); off += we[i]
    off = P + we[0]
    for i in range(N): addr[("tp", i, "w")] = (off, wt[i]); off += wt[i]
    off = w_end + PAD
    for i in range(N): addr[("ep", i, "c")] = (off, ce[i]); off += ce[i]
    off = tc_end - sum_ct
    for i in range(N): addr[("tp", i, "c")] = (off, ct[i]); off += ct[i]

    return {"addr": addr, "PAD": PAD, "EP_end": EP_end, "buffer_bytes": tc_end - P, ...}
```

`check_safe` in the same file re-validates, at materialization, per-mode contiguity, the exact anchors, and the non-clobber condition for both switch directions (EP→TP transfers cache `N-1..0` then weights `N-1..0`; TP→EP the reverse).

## Why It Is Safe

The only cross-mode hazard is a layer's EP cache overlapping its TP cache. With EP cache forward from `A = P+Σwe+PAD` and TP cache anchored at `tc_end = EP_end + max(ct)`, "EP cache before TP cache at layer `i`" reduces to

```
ct[i] - Σ_{k>i}(ce[k] - ct[k])  ≤  max(ct)
```

The left side is `≤ ct[i] ≤ max(ct)` **purely because `ce[k] ≥ ct[k]`**. This holds for every layer, in any order, for any per-layer sizes — which is exactly why hybrid SWA + full attention needs no reordering and introduces no bug. The `max(ct)` anchor (biggest single TP-cache layer) is the whole trick. Weights overlap safely because the transfer moves all cache before any weights, freeing the EP-cache region before TP weights land on it.

## Memory Overhead

Let `SE = we[0]` (one EP-weight layer) and `MC = max(ct)` (biggest TP-cache layer). With the elastic budget making the EP and TP footprints equal (`B`), the buffer is

```
buffer = B + max(SE, MC)
```

— the shared budget plus **one layer**. The slack lands in exactly one seam (a seesaw), never both:

- `MC ≥ SE` → `PAD = 0`, the extra layer is the TP-cache tail overhang `MC`;
- `SE > MC` → `PAD = SE − MC`, the extra layer is the EP-weight head slab `SE`.

Rounding (token flooring + 256-byte alignment) adds a token-scale term far below one layer. Prototype confirmation:

| profile | `B` | overhead | `= max(SE, MC)` |
|---|---|---|---|
| qwen3-30B N48 G2 | 60000 MB | 650 MB | 650 |
| qwen3-235B N94 G2 | 126900 MB | 550 MB | 550 |
| adversarial `SE>CT` | 880 MB | 100 MB | 100 (PAD=80) |
| hybrid SWA+full | — | 30 MB | 30 (`= max ct`, every pattern) |

## Integration

The change surface is in [`paras_memory_manager.py`](../../python/sglang/srt/paras/paras_memory_manager.py). Consumers key off entry names (`experts`, `ep_experts`, `tp_experts`) and `.offset_bytes`, so the contract to preserve is the names, not the offsets. Note the four-anchor places **EP low / TP high** (the mirror of today's slots), so the transfer offsets are assigned accordingly.

| Function | Change |
|----------|--------|
| `plan_qwen_moe_layout` / `plan_gpt_oss_moe_layout` | Stop reserving `N+1` identical `paras.moe_slot.{i}` pairs; record per-layer sizes for deferred placement. |
| `create_paras_moe_aliases` | Replace slot aliasing with direct EP/TP entries at the four-anchor offsets; keep `experts → ep_experts`. |
| `alias` | Add a sibling that registers an entry at an explicit offset (keep the mirror path). |
| `materialize` | Place all four blocks per `compute_layout`, then assert the non-clobber condition. |
| `_compute_kv_budget_bytes` | Use per-mode elastic budgets so the EP and TP footprints match `B`. |
| `reserve_kv_cache` / `_create_kv_layout` | Fold the cache block into the combined run. |

## DP×TP (`dp_size > 1`) Status

The asymmetric `dp=2, tp=2, ep=4` case is implemented and GPU-validated at the transfer layer, with two distinct data movements:

- **MoE weight transport depends on physical topology.** If the EP group is node-local, the fused peer-access `dptp` kernels ([`kernels_dptp.cu`](../../python/sglang/srt/paras/csrc/kernels_dptp.cu)) read each EP shard once and broadcast it to all `G = dp_size` replicas. If EP spans nodes, each node-local TP group first uses the original peer-access kernels to write its experts directly into the canonical TP interval `[dp_rank·E/G, (dp_rank+1)·E/G)`. A strided `_PARAS_DP` group then performs an in-place all-gather from that interval into the full TP tensor. No staging allocation is required. TP→EP is replica-local in both cases.
- **KV cache and requests redistribute only within a TP subgroup.** For `ep=4, tp=2, dp=2` the two subgroups are `{0,1}` and `{2,3}`; each performs a self-contained `tp=2` KV switch over its own 2-rank NCCL group (`_PARAS_TP` scopes to the subgroup, `paras_tp_rank ∈ {0,1}`). The subgroups never exchange KV bytes. The existing `TP_SIZE`-templated KV kernels and NCCL fallback are topology-agnostic and need no `dptp` variant.

Validated on 4×A100 (`CUDA_VISIBLE_DEVICES=4,5,6,7`):

| Harness | Result |
|---------|--------|
| [`test_weight_transfer_dptp.py`](../../test/srt/paras/test_weight_transfer_dptp.py) | 3/3 — direct dptp by default; set `PARAS_TEST_LOGICAL_MULTINODE=1` for node-local IPC + in-place DP all-gather |
| [`test_kv_roundtrip_dptp.py`](../../test/srt/paras/test_kv_roundtrip_dptp.py) | 2/2 — intra-subgroup KV EP→TP→EP round-trip, cross-group isolation |

The layout `compute_layout` proof and CPU unit suite (33 checks) cover the `G>1` byte geometry.

### End-to-end serving (live server, dp>1)

The full scheduler switch was brought up on a live qwen3-30B server at `EP=4 / DP2 / TP2` (`--paras-tp-size 2`, 4×A100, FlashInfer, cuda-graph).

**Peer access is topology-scoped.** CUDA IPC is opened only among GPUs on the same node. A node-local EP group uses one EP-wide mapping for dptp weights, with KV addressing the current TP subgroup slice. A multi-node EP group uses one TP-group mapping shared by weights and KV; `_PARAS_DP` carries cross-node weight traffic through NCCL. This avoids opening an IPC handle for a remote process and still keeps only one mapping of each peer buffer.

**Sync scoping follows the physical transfer.** The single-node dptp path uses an EP-group barrier after each layer. The multi-node path uses a TP-group barrier after node-local peer writes, then launches the in-place DP all-gather on a second stream. The next layer's NVLink reshard overlaps that NIC collective at model level. KV cache redistribution remains scoped to the TP subgroup.

Manual-switch procedure (`.skills/paras-test-manual-switch`): dual capture `pools_differ=True` (#EP=52, #TP=68 graphs, cuda graph intact both modes); EP/TP/EP-RT prompts 0/32 degenerate; `configure_tp` 160 ms / `configure_ep` 58 ms; in-flight EP→TP 3/3 and TP→EP **20/20** clean; no server errors. The gate at `scheduler_paras_mixin.py:742` (`paras_dp_size == 1`) is relaxed to enable this.

## Future Work

- **FP8 weights and scales.** The FP8 weight path is byte-agnostic (`elem_size = 1`) but untested, with a tighter 16-byte alignment constraint on `w2` rows. FP8 scales are pre-materialized at init and likely need no transfer kernel; verify under the `(d, t)` topology.
- **EP=8 / DP2 / TP4 scale.** Validated at `EP=4/DP2/TP2` (the identical dp>1 code path). The `EP=8` scale needs 8 healthy GPUs; on the current box GPU 1 is driver-wedged (CUDA init hangs, no root to reset), so `EP=8` awaits a GPU reset. Flipping the harness to `EP=8` is a one-line change (`NUM_GPUS=8 --paras-tp-size 4`).

## Design Documents

| Document | Contents |
|----------|----------|
| [`unified_memory_manager.md`](unified_memory_manager.md) | Base allocator, lifecycle, and the slot layout this design replaces. |
| [`parallelism_switch.md`](parallelism_switch.md) | Runtime EP↔TP switch control flow, race-safety invariants, and verified performance. |
| [`nvlink_peer_access_weight_transfer.md`](nvlink_peer_access_weight_transfer.md) | NVLink weight-transfer kernel design, synchronization, and tuning. |
| [`nvlink_peer_access_kv_cache_transfer.md`](nvlink_peer_access_kv_cache_transfer.md) | KV-cache transfer kernels and NCCL fallback. |
