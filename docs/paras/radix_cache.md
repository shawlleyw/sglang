# Radix Cache in SGLang and ParaS

## Status: Supported (as of paras-radix-cache plan T30)

ParaS migration now supports radix cache via tree-state migration across EP↔TP switches.

- Operators can run with `--disable-radix-cache=False` (the default) under ParaS.
- The 4 hard asserts (HiRadix `--enable-hierarchical-cache`, CPP radix `SGLANG_EXPERIMENTAL_CPP_RADIX_TREE=1`, EAGLE `--speculative-algorithm EAGLE`, `--page-size > 1`) remain in place — those features are NOT yet supported with tree migration. See "Known Pre-existing Issues" section for `page_size > 1` background.
- Configurable preservation:
  - `--paras-radix-preserve-unlocked` (default `false`): if true, EP→TP transfers unlocked tree nodes' K/V too (TP→EP still drops). Adds latency.
  - `--paras-radix-migration-strict={fail,fallback}` (default `fail`): on post-switch validator failure, either crash (supervisor restarts) or orphan all migrated reqs and continue.

The body of this document below describes the historical hazards and the migration design.

---

## Historical Context: Why ParaS Previously Disabled Radix Cache

This document explains what the radix cache does in sglang, why combining it with ParaS migration was unsafe before tree-migration support (T2-T29), and what would have gone wrong if the assertion had been bypassed.

## What the radix cache provides

The radix cache (`RadixCache`, `SWARadixCache`) sits between requests and the K/V pool. Its core responsibilities:

1. **Prefix caching** — when a new request shares a prompt prefix with a finished or in-flight request, the cached K/V slots are returned via `match_prefix`. The request's prefill skips the matched range.
2. **Reference counting** — each in-flight request holds a `full_lock_ref` (and `swa_lock_ref` for SWA models) on the leaf node of its matched prefix path. This protects those K/V slots from eviction while the request is using them.
3. **Eviction under memory pressure** — when the K/V pool gets full, the radix tree's LRU (`full_lru_list`, `swa_lru_list`) selects the least-recently-used unlocked node and frees its slots. For SWA models, the tree supports a "tombstone" state where a node's K/V slots are freed but the node's identity is retained so prefix matching can correctly route around it.

For typical chat-completion workloads with shared system prompts and conversation prefixes, prefix caching is a substantial throughput win — often 30-70% of decoded tokens are served from the cache.

## Why ParaS cannot use the radix cache today

The fundamental issue is that ParaS's switch operation **rebuilds the K/V pool from scratch** and the radix tree's understanding of "what's in the pool" cannot be migrated cleanly across that rebuild. Specifically:

### 1. `tree.reset()` removes prefill nodes at switch boundary

The ParaS switch flow ([`scheduler_paras_mixin.py:paras_configure_tp` / `paras_configure_ep`](file:///home/shaoyuw/sglang/python/sglang/srt/paras/scheduler_paras_mixin.py)) does:

```python
self.tree_cache.reset()         # wipe all tree nodes
self.token_to_kv_pool_allocator.paras_resize_and_clear(new_cache_size)  # reset allocator
... alloc new slots ...
... transfer K/V ...
```

`tree.reset()` clears the tree root and all child nodes. Every prefix that was indexed before the switch — including the **prefill prefix** of every in-flight request, inserted at end-of-prefill via `cache_unfinished_req` — is removed.

What survives across the reset:

- `req.last_node`, `req.last_host_node`, `req.swa_uuid_for_lock` — Python references to TreeNode objects that are no longer in the tree.
- `req.prefix_indices` — torch.Tensor of slot indices that used to be the cached prefix; the underlying slots are reallocated by `paras_resize_and_clear`, so these indices are stale.
- The radix tree's accounting (`full_evictable_size_`, `swa_evictable_size_`, `full_protected_size_`, `swa_protected_size_`) — reset to 0.
- The in-flight request's actual K/V data — transferred to the new pool layout and re-indexed via the new `req_to_token_pool`.

### 2. Stale `last_node` references on migrated requests

Without intervention, a migrated request's `req.last_node` points to a TreeNode that is no longer part of any tree. When the request finishes after the switch, `cache_finished_req` calls:

```python
self.dec_lock_ref(req.last_node, req.swa_uuid_for_lock)
```

`dec_lock_ref` walks `node → node.parent → ... → root_node`, decrementing reference counters and asserting they were positive. The stale `last_node`'s `parent` chain may still exist in memory (Python won't garbage-collect TreeNodes that are still referenced), so the walk runs to "some" root. But:

- The counters on those stale nodes were set against the **pre-switch** tree's bookkeeping, which has since been zeroed.
- Decrementing them produces unbalanced counts; subsequent asserts can fire.
- The tree's `full_evictable_size_` / `full_protected_size_` accounting becomes inconsistent with the actual pool state, breaking the scheduler's admission control (which queries `tree_cache.swa_evictable_size()` and similar).

ParaS's gather_manager currently mitigates this specific symptom by resetting `req.last_node = tree_cache.root_node` in `recover_request`, so the `dec_lock_ref` walk hits root immediately and short-circuits. This works for the `dec_lock_ref` path but doesn't address issue (3) below.

### 3. `cache_finished_req → insert` corrupts the post-switch tree (correctness bug)

After Phase A's runtime SWA eviction is in play, each in-flight request maintains `req.swa_evicted_seqlen = X`. The K/V at positions `[0, X)` has been freed from the SWA pool (`full_to_swa_index_mapping[full_slot] = 0` for those positions).

When the migrated request finishes post-switch, `SWARadixCache.cache_finished_req → self.insert(token_ids, kv_indices, prev_prefix_len, swa_evicted_seqlen)` runs. The `kv_indices` cover all positions `[0, S-1)`, but for `[0, X)` the SWA mapping is 0.

Without the **tombstone-aware insert** path (added in [PR #17220](https://github.com/sgl-project/sglang/pull/17220), commit `ce8a6ac69`), the standard `_insert_helper` creates a single non-tombstoned TreeNode containing the full kv_indices. The tree now thinks position `p ∈ [0, X)` has a valid SWA slot — but `mapping[that_full_slot] = 0`.

A subsequent new request that shares this prefix:

1. Calls `match_prefix(token_ids)`, which walks the tree and returns the matched slot list (including the OOW slots).
2. Prefill loads K/V from those slots. Full attention reads from the live full-pool slots → correct.
3. SWA attention reads via `full_to_swa_index_mapping[full_slot]` → mapping is 0 → reads from SWA slot 0 (the padding slot) → **gets zero K/V for in-window positions of the new request**.

The new request's SWA layers see padding where they should see real cached K/V. Output is corrupted (garbled tokens, degenerate loops, or subtly wrong responses depending on which positions overlap).

PR #17220's tombstone-aware insert fixes this by splitting the new tree node at the `swa_evicted_seqlen` boundary: positions `[0, swa_evicted_seqlen)` become a tombstoned node (marked freed for SWA), and positions `[swa_evicted_seqlen, seqlen)` become a non-tombstoned node. The `_match_prefix_helper`'s W-distance rule then correctly caps the match length so SWA attention never reads a freed slot.

### 4. Tree state vs. pool state divergence (scheduling)

The radix tree's `swa_evictable_size()` and `full_evictable_size()` are consumed by `Scheduler.check_memory` and `schedule_policy` to decide how many new requests to admit. Post-`tree.reset()`, these counters report 0 even though the underlying pool has substantial in-flight slot usage that the tree has "forgotten" about.

The scheduler can either:

- Over-admit (thinking 100% of the pool is available when migrated reqs are using a big chunk of it) → OOM during prefill of newly admitted requests.
- Under-admit (if the migrated reqs' slot usage is double-counted by another path) → degraded throughput.

Neither is catastrophic by itself, but combined with (3) above, the system is in a fragile state for which we have no integration tests covering all edge cases.

## What ParaS does instead: ChunkCache / SWAChunkCache

Under `--disable-radix-cache`, [`scheduler.py:687-697`](file:///home/shaoyuw/sglang/python/sglang/srt/managers/scheduler.py) selects:

- `SWAChunkCache` for hybrid SWA models (gpt-oss, Gemma3).
- `ChunkCache` for non-SWA MoE models (qwen3-MoE, etc.).

Both are tree-less:

- No `match_prefix` → no prefix sharing across requests.
- No `inc_lock_ref` / `dec_lock_ref` → no per-node reference counts (the methods exist but return 0 / no-op).
- `cache_finished_req` simply frees the request's K/V and releases the `req_pool_idx`. No tree insert, no tombstones.
- `reset()` is a no-op `pass`.

Trade-off: ParaS gives up cross-request prefix sharing entirely. For workloads where prefix sharing matters (shared system prompts, multi-turn conversations), this is a real throughput cost. For the gpt-oss / qwen3-MoE workloads where ParaS adds value (the EP↔TP regime), most production traffic is one-shot completions or has minimal prefix overlap, so the loss is acceptable.

## What would be needed to support radix cache under ParaS

Three pieces would have to land before the assertion in `server_args._check_paras_config` could be lifted:

1. **Tombstone-aware insert in `SWARadixCache._insert_helper`** — port the three-branch logic from PR #17220 (`swa_evicted_seqlen` parameter, `_add_new_node` helper, tombstone-aware re-insert when reinserting after eviction). ~50 LOC.

2. **`cache_protected_len` floor in `_evict_swa`** — track per-request `cache_protected_len` (the tree-locked prefix length) and use it as the floor in `max(req.swa_evicted_seqlen, cache_protected_len, ...)` so runtime eviction never frees slots that the tree's `lock_ref` claims are protected. PR #17220 adds this. ~10 LOC.

3. **`tree_orphaned` flag or equivalent for migrated requests** — defensive flag set during `gather_manager.recover_request` so `cache_finished_req` short-circuits to the ChunkCache-style simple-free path for migrated requests, bypassing the tree insert entirely. This avoids issues (3) and (4) above for migrated requests specifically while still allowing new post-switch requests to populate the tree normally. ~10 LOC across `radix_cache.py`, `swa_radix_cache.py`, `schedule_batch.py`, and the gather/scatter managers.

Even with all three, **migrated** requests would not contribute to cross-request prefix sharing post-switch (they're orphaned from the tree). Only **new** post-switch requests would build up cacheable prefixes. This is a structural limit: there is no way to migrate radix tree state across `tree.reset()` short of fully serializing the tree, transmitting it, and rebuilding on the destination — which would be a significantly larger undertaking.

Per the project priorities, none of these three pieces is on the roadmap. The `--disable-radix-cache` requirement is the chosen design.

## What happens if you bypass the assertion

If a user comments out the `assert self.disable_radix_cache` in `server_args._check_paras_config` and launches ParaS without `--disable-radix-cache`:

| Phase | Outcome |
|---|---|
| Server boot | OK. `SWARadixCache` is constructed. ParaS init does not interact with the tree at boot. |
| Pre-switch traffic | OK. Normal radix-cache-backed serving, prefix sharing works. |
| First switch (no in-flight reqs) | OK. `tree.reset()` is a clean wipe with no migrated state. |
| First switch (with in-flight reqs) | OK by accident. The migrated reqs' `last_node` is reset to root_node, so `dec_lock_ref` no-ops. Tree state is wiped; new post-switch requests build it up again. |
| Post-switch decode for migrated reqs | OK by accident. Phase A's `maybe_evict_swa` continues to evict SWA slots as the window slides. |
| Migrated req finishes post-switch | **Tree corruption.** `cache_finished_req → insert` adds a non-tombstoned tree node containing slots whose SWA mapping is 0 for the OOW range. The corruption is silent at this point — no error, no assertion. |
| New request post-switch shares prefix with finished migrated req | **Garbage SWA K/V.** Cross-request prefix matching returns the corrupted slots. New request's SWA layers read padding for in-window positions → decode output is garbled. |
| Memory pressure under load | Possibly OOM. Tree's `swa_evictable_size` accounting is inconsistent with pool state post-switch; scheduler may over-admit. |

The user might not see anything wrong for many requests, depending on how often prefixes are shared and how long migrated requests live. When they DO see something wrong, the symptom is bad decode output for new requests, which is easy to misdiagnose as a model bug or training issue.

The assertion is therefore both a correctness guard and an honesty signal: ParaS in its current form is not safe with radix cache, and silently allowing it would produce subtle, hard-to-trace failures.

## References

- [`server_args._check_paras_config`](file:///home/shaoyuw/sglang/python/sglang/srt/server_args.py) — the `--disable-radix-cache` assertion (line 1512).
- [`scheduler.py:687-697`](file:///home/shaoyuw/sglang/python/sglang/srt/managers/scheduler.py) — `SWAChunkCache` / `ChunkCache` auto-selection under `--disable-radix-cache`.
- [`mem_cache/swa_radix_cache.py:SWARadixCache`](file:///home/shaoyuw/sglang/python/sglang/srt/mem_cache/swa_radix_cache.py) — the cache type ParaS does not use; see `_match_prefix_helper` (W-distance rule), `_tombstone_internal_node`, `inc_lock_ref` / `dec_lock_ref`.
- [`mem_cache/chunk_cache.py:SWAChunkCache`](file:///home/shaoyuw/sglang/python/sglang/srt/mem_cache/chunk_cache.py) — the cache type ParaS uses.
- [`paras/scheduler_paras_mixin.py`](file:///home/shaoyuw/sglang/python/sglang/srt/paras/scheduler_paras_mixin.py) — the `tree_cache.reset()` call at line 152 / 236.
- [`paras/gather_manager.py:recover_request`](file:///home/shaoyuw/sglang/python/sglang/srt/paras/gather_manager.py) — the `last_node = root_node / None` reset that workarounds issue (2).
- PR #17220 (commit `ce8a6ac69`) — adds the tombstone-aware insert and `cache_protected_len` to sglang main; would be the basis for re-enabling radix cache under ParaS.
- [`docs/paras/swa.md`](file:///home/shaoyuw/sglang/docs/paras/swa.md) — companion doc on the SWA mechanism and ParaS's SWA cache transfer.
- [`docs/paras/parallelism_switch.md`](file:///home/shaoyuw/sglang/docs/paras/parallelism_switch.md) — overall ParaS switch design and other unsupported features.

## Known Pre-existing Issues (Defer-Fix)

### `paras_resize_and_clear` is not page-aware

**File**: `python/sglang/srt/mem_cache/allocator.py:TokenToKVPoolAllocator.paras_resize_and_clear` (~line 174-184).

**Symptom**: When ParaS is enabled with `page_size > 1`, the free-list reset uses `torch.arange(1, new_size + 1)` which assumes token-level (page_size=1) indices. `PagedTokenToKVPoolAllocator` does not override `paras_resize_and_clear`, so it inherits this token-level reset and silently corrupts its page-level free list.

**Status**: Independent of the radix-cache work. Pre-existing, latent. Surfaces only with `page_size > 1 + ParaS`, which is currently a hard assert (see `_check_paras_config` in `server_args.py`, added by the radix-cache plan's T1).

**Workaround (in place)**: T1 of the paras-radix-cache plan asserts `page_size == 1` at startup when `enable_paras_moe=True and not disable_radix_cache`. This prevents the bug from surfacing in production.

**Fix**: A separate PR should override `paras_resize_and_clear` on `PagedTokenToKVPoolAllocator` to reset the page-level free list correctly. Out of scope for the current radix-cache work.

**Reference**: Discovered during Oracle architectural review of the radix-cache plan; flagged in `.sisyphus/notepads/paras-radix-cache/`.
