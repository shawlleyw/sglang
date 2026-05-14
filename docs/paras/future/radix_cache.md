# Future Direction: Radix Cache Support in ParaS

## Status

**Not currently supported.** ParaS asserts `--disable-radix-cache` at startup
([`server_args.py:1512`](file:///home/shaoyuw/sglang/python/sglang/srt/server_args.py#L1512)). The active tree cache is
[`ChunkCache`](file:///home/shaoyuw/sglang/python/sglang/srt/mem_cache/chunk_cache.py)
(MHA-only models like Qwen3-MoE) or
[`SWAChunkCache`](file:///home/shaoyuw/sglang/python/sglang/srt/mem_cache/chunk_cache.py)
(hybrid SWA models like gpt-oss). Cross-request prefix sharing is therefore
unavailable.

This document explains *why* re-enabling radix cache is non-trivial, separately
for MHA-only and SWA hybrid configurations, and sketches future directions for
support.

## Section 1: Why Radix Cache Is Hard for MHA in ParaS

A naive enable — flipping the assertion to allow `--disable-radix-cache=False`
under `--enable-paras-moe` — would launch and serve correctly while no switch
fires. Each switch then exposes a layered set of problems.

### Issue 1: `tree.reset()` destroys all tree state at switch

[`SchedulerParasMixin.paras_configure_tp/ep`](file:///home/shaoyuw/sglang/python/sglang/srt/paras/scheduler_paras_mixin.py)
calls `self.tree_cache.reset()` (`scheduler_paras_mixin.py:152, 236`) before
the cache transfer fires. After reset, the tree is empty:

- Cached prefixes from finished requests (the cross-request prefix-share pool) are gone.
- Locked prefixes for in-flight migrating requests are gone.
- All `lock_ref` accounting is reset.

For migrated requests, the band-aid in
[`gather_manager.recover_request`](file:///home/shaoyuw/sglang/python/sglang/srt/paras/gather_manager.py#L36)
sets each Req's tree-pointing fields:

```python
if tree_cache.disable:
    req.last_host_node = None
    req.last_node = None
else:
    req.last_host_node = tree_cache.root_node
    req.last_node = tree_cache.root_node
req.prefix_indices = []
```

The `else` branch (radix cache active) sets `last_node = root_node` so a
subsequent `cache_finished_req` calls `dec_lock_ref(req.last_node, ...)`, which
walks from root and immediately exits the while loop ([`radix_cache.dec_lock_ref`](file:///home/shaoyuw/sglang/python/sglang/srt/mem_cache/radix_cache.py#L525)
at line 525, [`SWARadixCache.dec_lock_ref`](file:///home/shaoyuw/sglang/python/sglang/srt/mem_cache/swa_radix_cache.py#L713) at line 713). This avoids
the assertion crash but provides **no real migration of tree state** — every
in-flight request loses its cached prefix bookkeeping.

### Issue 2: Stale tree-pointer fields beyond `last_node`

`Req` carries other tree-related state that becomes meaningless post-reset:

| Field | Pre-switch meaning | Post-switch state |
|---|---|---|
| `req.prefix_indices` | Slot indices for the matched prefix | Stale (slots reallocated by `paras_resize_and_clear`); reset to `[]` in `recover_request` |
| `req.last_matched_prefix_len` | Length of cached match | Inconsistent with `prefix_indices = []` |
| `req.swa_uuid_for_lock` | UUID identifying the SWA lock node ancestor | Points to a now-deleted node |

The current `recover_request` discards these fields conservatively rather than
migrating any of them. This means subsequent `cache_finished_req → insert(...)`
re-inserts the request as if no prefix had ever been cached.

### Issue 3: Asymmetric in-flight TP→EP failure observed on qwen3 (historical)

The
[gpt-oss bug-probing chronicle](file:///home/shaoyuw/sglang/docs/paras/gpt_oss_support.md)
"Bug 3" entry documents an asymmetric crash observed pre-`paras_disable_radix`:

> The `AssertionError: This request holds the node from another tree` in
> `radix_cache.dec_lock_ref` ... point to the request being replicated across
> all DP ranks during scatter rather than partitioned to one rank.

Specific to qwen3-MoE under in-flight TP→EP with the Triton attention backend
plus the seqlen-1 fix. Root cause was never fully understood; the workaround
was to switch ParaS to `ChunkCache`, which sidesteps the entire radix-specific
code path. Re-enabling radix cache means re-investigating this assertion.

### Issue 4: Cross-request prefix sharing is bounded by switch frequency

Even if all tree-pointer issues were solved, the prefix-share benefit only
persists between switches:

```
Request A (long context) prefills → tree caches A's prefix.
Request B (same context) prefills → matches A's prefix → fast prefill ✓
[switch fires]
tree.reset() → A's cached prefix is lost.
Request C (same context) prefills → tree empty → no match → full prefill.
```

For workloads that switch frequently, the benefit degrades to "useful only
within an inter-switch interval".

### Summary: what would solve MHA radix cache in ParaS

Genuine support requires migrating across the switch:

1. The full radix tree state (nodes, edges, values, lock_refs) per rank.
2. Per-Req `last_node` / `last_host_node` references with translation from
   pre-switch slot indices to post-switch slot indices.
3. The LRU list state (`full_lru_list`, `host_lru_list` if HiCache).
4. Cross-rank consistency (TP mode → identical trees on every rank;
   EP mode → per-rank disjoint trees).

None of this is currently implemented.

## Section 2: SWA Hybrid Cache Adds Three More Layers

`SWARadixCache` extends `RadixCache` with sliding-window-aware tombstone
machinery. Each adds a new migration constraint. See
[`gpt_oss_support.md`](file:///home/shaoyuw/sglang/docs/paras/gpt_oss_support.md)
for the SWA pool semantics underlying these constraints.

### Complication 1: Tombstone state must be migrated

[`SWARadixCache.evict(swa_num_tokens=...)`](file:///home/shaoyuw/sglang/python/sglang/srt/mem_cache/swa_radix_cache.py#L589)
frees SWA pool slots independently from full pool slots. When a non-leaf
node's SWA slots are freed, the node is marked
[`swa_tombstone = True`](file:///home/shaoyuw/sglang/python/sglang/srt/mem_cache/swa_radix_cache.py#L1005)
via `_tombstone_internal_node`. Subsequent
[`_match_prefix_helper`](file:///home/shaoyuw/sglang/python/sglang/srt/mem_cache/swa_radix_cache.py#L797)
applies the W-distance rule: a match that would cross a tombstone is truncated
unless the post-tombstone non-tombstone region has ≥ W tokens.

For tree migration this means:

- Source-side tree may contain many tombstoned nodes.
- Each tombstone references full-slot indices that don't exist on the
  destination (slots were reallocated).
- Translation must distinguish "this slot was SWA-evicted (tombstoned)" from
  "this slot has live SWA K/V (non-tombstoned)" because they have different
  destination semantics post-translation.
- Without correct tombstone migration, post-switch matches would either
  over-match (returning slots whose SWA mapping is 0 → wrong K/V) or
  under-match (missing valid prefix-share opportunities).

### Complication 2: Per-Req `swa_evicted_seqlen` interacts with tombstones

ParaS's
[`ScheduleBatch.maybe_evict_swa` / `_evict_swa`](file:///home/shaoyuw/sglang/python/sglang/srt/managers/schedule_batch.py#L1573)
frees per-Req SWA pool slots based on
`max(swa_evicted_seqlen, pre_len - W)`. Under `SWAChunkCache` (current state),
this is a clean allocator-level op with no tree to coordinate with. Under
`SWARadixCache`, the same eviction path would need to also create tombstones
in the tree for the freed positions — otherwise cross-request prefix matches
would return slots whose SWA mapping has been zeroed.

Upstream PR sgl-project/sglang#17220 introduces a tombstone-aware
`_insert_helper` that handles this at `cache_finished_req` time. For ParaS,
the same logic must fire on the *switch* path too: migrating Reqs whose
`swa_evicted_seqlen` was advanced mid-decode need their re-insert to split
into tombstone + non-tombstone regions per the W-distance rule.

### Complication 3: Token-window eviction creates high tombstone churn

With sliding window W, `maybe_evict_swa` fires at every decode step. For each
in-flight Req that advances past `swa_evicted_seqlen + 1` decode tokens, a new
batch of slots becomes OOW. Over a long-running request the tree accumulates
tombstones at the rate of approximately `(decode_steps - W)` tombstones per
request.

For ParaS at switch time, this means:

- Tree migration must preserve a potentially large tombstone set per request.
- Or: switch must precede a tombstone-compaction pass (cost grows with tree
  density).

### What we have today (Phase A + C, May 2026)

The current ParaS-on-`SWAChunkCache` implementation handles SWA pool
correctness without any tree-side tombstone work:

- `ScheduleBatch.maybe_evict_swa` runs every decode step (Phase A,
  [`schedule_batch.py:1573+`](file:///home/shaoyuw/sglang/python/sglang/srt/managers/schedule_batch.py#L1573)).
  Frees OOW SWA slots; no tree to update.
- [`gather_manager._tighten_swa_pool_to_in_window`](file:///home/shaoyuw/sglang/python/sglang/srt/paras/gather_manager.py#L242)
  runs at switch boundary (Phase C). Frees OOW SWA slots on destination after
  lockstep alloc; updates `req.swa_evicted_seqlen` to the new floor.
- Per-Req `swa_evicted_seqlen` carries through pickle in `gather_manager`
  (no explicit code; `prune_request` doesn't null int fields).

Adding radix cache support would require all of this PLUS the tombstone
migration described above.

## Section 3: Future Directions

Three rough designs, in increasing complexity / capability.

### Design A: Lazy rebuild — accept lost prefix sharing during transients

Simplest. Keep `ChunkCache` / `SWAChunkCache` as the active tree cache during
ParaS-eligible operation, but make `--disable-radix-cache` non-fatal (revert
the assertion). When the user wants prefix sharing, they run without
`--enable-paras-moe`.

- **Pros**: zero new ParaS code. The existing `tree_cache.disable` branches in
  [`gather_manager.recover_request`](file:///home/shaoyuw/sglang/python/sglang/srt/paras/gather_manager.py#L36)
  already accommodate both modes.
- **Cons**: cannot have both ParaS and prefix sharing simultaneously.
  Effectively the current state minus the hard assertion.
- **When to choose**: production demand for ParaS + prefix sharing is unproven.

### Design B: Per-rank tree preservation across switches (MHA only)

Migrate the radix tree as part of the gather/scatter contract. Each rank's
tree is serialized pre-switch, broadcast/partitioned via the switch's existing
collectives, and reconstructed post-switch with translated slot indices.

For MHA only (no tombstones):

- Serialize each tree node: `(parent_id, key_tokens, value_slots, lock_ref)`.
- All-gather across ranks (TP mode → identical trees) or partition (EP mode →
  per-rank).
- Translate `value_slots` from pre-switch slot indices to post-switch indices
  using the global token-index translation the cache transfer already produces.
- Reconstruct tree.

Estimated cost: ~10–50 ms additional switch latency for typical workloads
(dominant term is the per-node serialization). Significant code surface
(~500–1000 LOC across `gather_manager.py`, `scatter_manager.py`, plus a new
helper module for tree (de)serialization).

- **Pros**: cross-request prefix sharing survives switches in MHA mode.
- **Cons**: substantial implementation cost, scatter direction is harder
  (per-rank disjoint trees with consistent partitioning), Bug 3-style
  asymmetric corner cases must be re-investigated.

### Design C: Tombstone-aware tree preservation (full SWA support)

Design B plus:

- Per-node serialization extends to `swa_tombstone` flag, `swa_lock_ref`, and
  (for non-tombstone nodes) the SWA-pool slot positions translated similarly.
- Per-Req `swa_evicted_seqlen` propagation already exists (Phase B); the
  destination side must reconcile inherited `swa_evicted_seqlen` against the
  reconstructed tombstone set.
- Post-reconstruction: invariants checked (tombstone density, lock_ref
  accounting, SWA-evictable size).

Estimated cost: Design B + 30–50% additional code for SWA-specific paths
(roughly 200–400 LOC). Most new code mirrors PR sgl-project/sglang#17220's
tombstone-aware logic but applied at the switch boundary.

- **Pros**: full prefix sharing for hybrid SWA models like gpt-oss across
  switches.
- **Cons**: highest complexity, hardest to validate (Bug-3-style asymmetric
  failures will need a fresh chronicle), real benefit depends on workload
  having same-prefix repeat-traffic across switches.

### Recommended sequence

1. **Validate Design A is acceptable**: confirm with production deployment
   teams whether ParaS + prefix sharing is needed simultaneously. If no
   concrete demand surfaces, revert the hard assertion to a warning and call
   it done.
2. **If Design B is needed**: implement MHA-only tree preservation first.
   Validate with the qwen3 test suite (`paras-test-manual-switch` +
   `paras-test-auto-switch`).
3. **If Design C is needed**: layer SWA support on Design B once stable.
   Validate with gpt-oss-120b under the existing
   [SKILL](file:///home/shaoyuw/sglang/.skills/paras-test-manual-switch/SKILL.md)
   plus a new long-context cross-request workload.

## References

- [`parallelism_switch.md`](file:///home/shaoyuw/sglang/docs/paras/parallelism_switch.md)
  — overall ParaS EP↔TP design (Unsupported Features section lists the radix-cache constraint).
- [`gpt_oss_support.md`](file:///home/shaoyuw/sglang/docs/paras/gpt_oss_support.md)
  — SWA pool semantics, hybrid attention, the historical Bug 3 (radix cache
  assertion crash on qwen3 in-flight TP→EP), and the post-`paras_disable_radix`
  Recent Updates section describing the current ChunkCache-based state.
- [`runs/2026-05-09-swa-window-only-transfer-design/DESIGN.md`](file:///home/shaoyuw/sglang/docs/paras/runs/2026-05-09-swa-window-only-transfer-design/DESIGN.md)
  — design doc for the current ChunkCache-based approach.
- [`mem_cache/swa_radix_cache.py`](file:///home/shaoyuw/sglang/python/sglang/srt/mem_cache/swa_radix_cache.py)
  — `SWARadixCache` reference for tombstone semantics (`evict` at line 589,
  `_tombstone_internal_node` at line 1005, `_match_prefix_helper` at line 797).
- [`mem_cache/radix_cache.py`](file:///home/shaoyuw/sglang/python/sglang/srt/mem_cache/radix_cache.py)
  — `RadixCache` reference for MHA tree (`reset` at line 241, `evict` at line 482, `dec_lock_ref` at line 525).
- PR sgl-project/sglang#17220 — runtime SWA pool eviction with tombstone-aware insert (the upstream foundation any SWA support would build on).
