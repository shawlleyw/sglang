# ParaS Unified Memory Layout for EP↔TP With Multiple TP Instances

**Status:** Implemented in
[`paras_memory_manager.py`](../../python/sglang/srt/paras/paras_memory_manager.py)
and the switch orchestration. The CPU layout tests validate materialized
four-anchor offsets. `test_weight_transfer.py` validates both intra-node
transports for EP4 <-> TP4, and `test_weight_transfer_tp_instances.py`
validates both transports plus the DP all-gather for EP4 <-> two TP2
instances.

**Scope:** This document describes the layout that replaced the `N+1` identical MoE weight slots. It supports `ep_size ≥ tp_size` with uniform or hybrid (sliding-window + full) attention cache, subject to the production capacity invariant `ct[i] ≤ ce[i]` described below.

**Two refinements landed during implementation.** (1) The address formula retains the general cache tail anchor `ANCHOR = max_i(ct_i + Σ_{k>i}(ct_k − ce_k))`, but the production KV budget reserves one `max(ct)` tail layer and therefore enforces `ct[i] ≤ ce[i]`; under that invariant the formula reduces to `max(ct)`. (2) The per-layer shapes match the ParaS forward exactly: EP holds `num_experts/ep_size` experts with the **full** intermediate; TP holds **all** `num_experts` with the intermediate sharded by `tp_size` (equal bytes at `G=1`, TP `G`× larger for `G>1`). The switch loop orders are flipped from the old slot design: EP→TP runs cache-then-weights in **reverse** layer order; TP→EP runs weights-then-cache in **forward** order.

## Overview

The [unified memory manager](unified_memory_manager.md) holds all persistent ParaS state in one contiguous `uint8` buffer and switches between Expert Parallelism (EP) and Tensor Parallelism (TP) by reinterpreting the same bytes. The former layout reserved `N+1` identical MoE weight slots and aliased EP and TP onto neighboring slots, which was correct only when EP and TP occupied the same bytes per layer (`SE == ST`).

Multiple TP instances break that. On a `W = G·T` grid (`G = dp_size`, `T = tp_size`, `ep_size = W`), EP weights are small (experts sharded across all `W` ranks) while each TP instance uses a TP layout that is `G` times larger (experts sharded across only `T` ranks); the KV cache runs the other way, EP large and TP small. This design packs both interpretations into one combined `[weights | cache]` run in which the large TP weights overlap the EP cache, so the buffer is the shared per-mode footprint `B` plus **one layer**.

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
    sum_we, sum_wt, sum_ce, sum_ct = sum(we), sum(wt), sum(ce), sum(ct)

    # The address formula is general; production currently enforces ct <= ce
    # because reserve_kv_cache budgets only a max(ct) tail layer.
    assert all(ct[i] <= ce[i] for i in range(N))
    anchor, suffix = 0, 0
    for i in range(N - 1, -1, -1):
        anchor = max(anchor, ct[i] + suffix)
        suffix += ct[i] - ce[i]

    w_end    = P + sum_we
    tp_w_end = P + we[0] + sum_wt
    PAD      = _au(max(0, tp_w_end - w_end - sum_ce + sum_ct - anchor))
    EP_end   = w_end + PAD + sum_ce
    tc_end   = EP_end + anchor

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

The change surface is in [`paras_memory_manager.py`](../../python/sglang/srt/paras/paras_memory_manager.py). Consumers key off entry names (`experts`, `ep_experts`, `tp_experts`) and `.offset_bytes`, so the contract to preserve is the names, not the offsets. The four-anchor places **EP low / TP high**, the mirror of the former slot layout, so transfer offsets are assigned accordingly.

| Function | Change |
|----------|--------|
| `plan_qwen_moe_layout` / `plan_gpt_oss_moe_layout` | Stop reserving `N+1` identical `paras.moe_slot.{i}` pairs; record per-layer sizes for deferred placement. |
| `create_paras_moe_aliases` | Replace slot aliasing with direct EP/TP entries at the four-anchor offsets; keep `experts → ep_experts`. |
| `alias` | Add a sibling that registers an entry at an explicit offset (keep the mirror path). |
| `materialize` | Place all four blocks per `compute_layout`, then assert the non-clobber condition. |
| `_compute_umm_budget_bytes` / `_plan_balanced_kv_footprint` | Derive the static UMM limit, then select separate EP and TP KV capacities whose exact combined layout fits it. |
| `reserve_kv_cache` / `_create_kv_layout` | Fold the cache block into the combined run. |

## Multiple TP Instances (`dp_size > 1`)

Let `G = dp_size`, `T = tp_size`, `E` be the global expert count, and
`L = E/(G*T)` be the experts owned by one EP rank. Logical DP rank `d`
identifies the TP instance whose ranks are `{d*T, ..., d*T+T-1}`.

Every topology uses the same algorithm:

- **EP -> TP, intra-node phase.** TP instance `d` reshards the EP experts
  held by its `T` ranks into `[d*E/G, (d+1)*E/G)`. The configured
  `peer_access` or `nccl` method controls only this TP-local operation.
- **EP -> TP, inter-node phase.** When `G > 1`, the strided `_PARAS_DP`
  groups perform in-place NCCL all-gathers from those intervals into the full
  TP tensors. All TP instances then hold identical TP weights.
- **TP -> EP.** No DP collective is required. TP instance `d` reads only
  `[d*E/G, (d+1)*E/G)` and reconstructs the full intermediate dimension
  through its selected TP-local transport. Other TP expert intervals are
  ignored when EP views are activated.
- **KV cache and requests.** These redistribute only within a TP instance. For
  `ep=4, tp=2, dp=2`, `{0,1}` and `{2,3}` perform independent TP2 KV
  switches; DP groups do not exchange KV bytes.

The reverse operation is more than a pointer change: owned experts still need
TP-local reconstruction. "Dropping" means skipping replicated TP experts owned
by other DP ranks; no buffer is freed because EP and TP are stable views in
the unified allocation.

Validated on 4xA100:

| Harness | Coverage |
|---------|----------|
| [`test_weight_transfer.py`](../../test/srt/paras/test_weight_transfer.py) | EP4 <-> TP4 through peer_access and NCCL |
| [`test_weight_transfer_tp_instances.py`](../../test/srt/paras/test_weight_transfer_tp_instances.py) | EP4 <-> two TP2 instances through both local transports and w13 layouts, DP all-gather, replica equality, and manager-backed bitwise round trip |
| [`test_weight_transfer_multinode_reverse.py`](../../test/srt/paras/test_weight_transfer_multinode_reverse.py) | Both logical DP ranks select only their owned interval with v2 and v3 kernels |
| [`test_kv_roundtrip_tp_instances.py`](../../test/srt/paras/test_kv_roundtrip_tp_instances.py) | TP-instance KV round trip and cross-instance isolation |

**Intra-node communication is always TP-local.** The weight method selects
`peer_access` or `nccl` within `_PARAS_TP`; `_PARAS_DP` always uses
NCCL and carries only EP -> TP weight replication. CUDA IPC is never opened
for remote processes.

**Synchronization follows the selected local transport.** Peer writes use a
TP-group visibility fence; NCCL all-to-all completion is stream ordered. On
EP -> TP, the inter-node stream waits for the local layer and launches the DP
all-gather while the intra-node stream starts the next layer. TP -> EP remains
local to the TP group.

### End-to-end serving

The full scheduler switch was validated on qwen3-30B at `EP=4`, `dp_size=2`,
`tp_size=2` with FlashInfer and CUDA graphs. Both TP instances use the same
weights after the forward DP all-gather, and the reverse switch restores the
four EP shards without inter-node communication.

## Future Work

- **FP8 weights and scales.** The FP8 weight path is byte-agnostic (`elem_size = 1`) but untested, with a tighter 16-byte alignment constraint on `w2` rows. FP8 scales are pre-materialized at init and likely need no transfer kernel; verify under the `(d, t)` topology.
- **Physical EP=8 with two TP4 instances.** The reverse ownership mapping is validated by emulating both DP ranks on one TP4 group, but the full eight-rank, two-node switch still needs an end-to-end run.

## Design Documents

| Document | Contents |
|----------|----------|
| [`unified_memory_manager.md`](unified_memory_manager.md) | Current allocator lifecycle, four-anchor placement, views, and invariants. |
| [`parallelism_switch.md`](parallelism_switch.md) | Runtime EP↔TP switch control flow, race-safety invariants, and verified performance. |
| [`nvlink_peer_access_weight_transfer.md`](nvlink_peer_access_weight_transfer.md) | NVLink weight-transfer kernel design, synchronization, and tuning. |
| [`nvlink_peer_access_kv_cache_transfer.md`](nvlink_peer_access_kv_cache_transfer.md) | KV-cache transfer kernels and NCCL fallback. |
