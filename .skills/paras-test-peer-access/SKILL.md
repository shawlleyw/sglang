---
name: paras-test-peer-access
description: Run ParaS correctness and benchmark tests for KV cache transfer (EP→TP and TP→EP, with/without head replication), weight transfer, request partition, memory invariants, and full round-trip. Knows GPU requirements, conda env, torchrun commands, and how to interpret results.
metadata:
  short-description: Test ParaS KV cache + weight transfer + partition + round-trip
---

# ParaS Transfer Tests

Correctness and benchmark tests for ParaS parallelism switching: KV cache transfer (both directions), weight transfer, request partition, memory invariants, and full EP↔TP round-trip.

## Prerequisites

- Conda env: `sgl_paras`
- Python path: `/home/shaoyuw/miniconda3/envs/sgl_paras/bin/python`
- Torchrun: `/home/shaoyuw/miniconda3/envs/sgl_paras/bin/torchrun`
- CUDA extension compiled: `cd python/sglang/srt/paras/csrc && pip install -e .`
- Empty GPUs required (check before running)

## GPU Check (ALWAYS run first)

```bash
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
# All values should be < 100 MiB
```

## Test Suite Overview

All tests live in `test/srt/paras/`:

| File | Tests | GPU Req | What It Verifies |
|------|-------|---------|------------------|
| `test_request_partition.py` | 11 | **None (CPU)** | Partition algorithm, replication routing, strategy extensibility |
| `test_kv_cache_transfer.py` | 6 | 4 or 8 GPUs | KV cache EP→TP, TP→EP, round-trip (R=1 and R=2) — standalone ground truth |
| `test_weight_transfer.py` | 7 | 4 GPUs | Weight EP→TP peer_access vs NCCL, ground truth, pointer swap, round-trip, reverse |
| `test_memory.py` | 2 | 4 GPUs | head_num save/restore, GPU memory leak detection |
| `test_roundtrip.py` | 4 | 4 GPUs | Full batch-level EP→TP→EP with model components |

**Total: 30 tests**

---

## 1. Request Partition Tests (CPU only — no GPU)

Tests the deterministic request partition algorithm and replication-aware routing.

```bash
/home/shaoyuw/miniconda3/envs/sgl_paras/bin/python -m pytest test/srt/paras/test_request_partition.py -v
```

**11 tests:**
- `TestPartitionRequestsForEP` (5): balanced, fewer_than_ranks, zero, equal_seqlens_deterministic, imbalanced
- `TestPeerAccessReplicationRouting` (4): R=1, R=2, no_token_lost_or_duplicated, R=4
- `TestPartitionStrategy` (2): greedy_strategy works, unknown_strategy raises ValueError

---

## 2. KV Cache Transfer Tests (4 or 8 GPUs)

Standalone correctness tests for KV cache transfer in BOTH directions. Each direction is verified independently against pattern-based ground truth (not round-trip symmetry).

### Pattern-based verification
Data is filled with `make_pattern(rank, layer, head, num_tokens)` which encodes source identity into values. After transfer, expected values at any destination are computed from first principles — no reliance on symmetry.

### Run on 4 GPUs (R=1 and R=2)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
/home/shaoyuw/miniconda3/envs/sgl_paras/bin/torchrun --nproc_per_node=4 \
  /home/shaoyuw/miniconda3/envs/sgl_paras/bin/pytest test/srt/paras/test_kv_cache_transfer.py -v
```

### Run on 8 GPUs (R=1 with 8 heads, R=2 with 4 heads)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
/home/shaoyuw/miniconda3/envs/sgl_paras/bin/torchrun --nproc_per_node=8 \
  /home/shaoyuw/miniconda3/envs/sgl_paras/bin/pytest test/srt/paras/test_kv_cache_transfer.py -v
```

**6 tests (adaptive to world_size):**

| Test | Direction | Replication | Verification Method |
|------|-----------|-------------|---------------------|
| `test_ep_to_tp_no_replication` | EP→TP | R=1 (heads=world_size) | Pattern ground truth |
| `test_ep_to_tp_with_replication` | EP→TP | R=2 (heads=world_size//2) | Pattern ground truth |
| `test_tp_to_ep_no_replication` | TP→EP | R=1 | Pattern ground truth |
| `test_tp_to_ep_with_replication` | TP→EP | R=2 | Pattern ground truth |
| `test_roundtrip_no_replication` | EP→TP→EP | R=1 | Bitwise snapshot match |
| `test_roundtrip_with_replication` | EP→TP→EP | R=2 | Bitwise snapshot match |

### Key implementation details
- EP→TP gather uses `gather_kv_and_permute` + `repeat_interleave` (if R>1) + `all_to_all` + `permute_and_scatter_kv` — matches production `_gather_cache_nccl`
- TP→EP scatter uses `_scatter_cache_nccl` directly with `_EPCacheView` — the production code path
- Layer order: reverse (N-1→0) for TP→EP to respect N+1 slot aliasing
- EP buffers are NEVER zeroed before scatter (EP slot[i+1] shares memory with TP slot[i+1])
- Tests adapt to world_size: `num_kv_heads = world_size` (R=1) or `world_size // 2` (R=2)
- **CRITICAL**: EP→TP DESTROYS EP weight slots (N+1 aliasing). TP→EP weight transfer must be actual reverse all-to-all, NOT pointer swap. Reverse must process layers in reverse order (N-1→0).

---

## 3. Weight Transfer Tests (4 GPUs)

Tests MoE weight transfer between EP and TP layouts.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
/home/shaoyuw/miniconda3/envs/sgl_paras/bin/torchrun --nproc_per_node=4 \
  /home/shaoyuw/miniconda3/envs/sgl_paras/bin/pytest test/srt/paras/test_weight_transfer.py -v
```

**7 tests:**

| Test | What It Verifies |
|------|------------------|
| `test_w13_peer_access_vs_nccl` | EP→TP w13 weights: peer_access kernel bitwise matches NCCL naive |
| `test_w2_peer_access_vs_nccl` | EP→TP w2 weights: peer_access kernel bitwise matches NCCL naive |
| `test_moe_pointer_swap` | TP→EP: `experts` pointer toggles between ep_experts and tp_experts |
| `test_weight_roundtrip` | EP→TP→EP: weight data bitwise match (reverse layer order) |
| `test_w13_ground_truth` | EP→TP w13: verified against independently computed expected values |
| `test_w2_ground_truth` | EP→TP w2: verified against independently computed expected values |
| `test_reverse_naive_vs_original` | TP→EP reverse: restored EP matches original snapshot |

### Benchmark (optional)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
/home/shaoyuw/miniconda3/envs/sgl_paras/bin/torchrun --nproc_per_node=4 \
  test/srt/paras/test_weight_transfer.py --benchmark
```

---

## 4. Memory Invariant Tests (4 GPUs)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
/home/shaoyuw/miniconda3/envs/sgl_paras/bin/torchrun --nproc_per_node=4 \
  /home/shaoyuw/miniconda3/envs/sgl_paras/bin/pytest test/srt/paras/test_memory.py -v
```

**2 tests:**
- `test_head_num_restored_after_ep` — `MHATokenToKVPool.head_num` correctly shards on TP (÷tp_size) and restores on EP (original)
- `test_no_memory_leak` — 5 TP↔EP cycles, asserts <1% GPU memory growth

---

## 5. Full Round-Trip Integration Tests (4 GPUs)

Batch-level integration with model components (memory manager, req pools, scatter/gather managers).

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
/home/shaoyuw/miniconda3/envs/sgl_paras/bin/torchrun --nproc_per_node=4 \
  /home/shaoyuw/miniconda3/envs/sgl_paras/bin/pytest test/srt/paras/test_roundtrip.py -v
```

**4 tests:**
- `test_roundtrip_ep_tp_ep` — Full EP→TP→EP with KV, weights, batch reconstruction
- `test_partition_consistency` — All ranks compute identical partitions (cross-rank assertion)
- `test_single_request_roundtrip` — Edge case: 1 request lands on exactly 1 EP rank
- `test_empty_batch_roundtrip` — Edge case: 0 requests, no crash

---

## Quick Reference: Run Everything

```bash
# CPU tests (no GPU needed)
/home/shaoyuw/miniconda3/envs/sgl_paras/bin/python -m pytest test/srt/paras/test_request_partition.py -v

# All GPU tests on 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 \
/home/shaoyuw/miniconda3/envs/sgl_paras/bin/torchrun --nproc_per_node=4 \
  /home/shaoyuw/miniconda3/envs/sgl_paras/bin/pytest test/srt/paras/ -v --ignore=test/srt/paras/test_request_partition.py

# KV cache tests on 8 GPUs (tests replication with R=2)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
/home/shaoyuw/miniconda3/envs/sgl_paras/bin/torchrun --nproc_per_node=8 \
  /home/shaoyuw/miniconda3/envs/sgl_paras/bin/pytest test/srt/paras/test_kv_cache_transfer.py -v
```

## Environment Variables

| Variable | Purpose | Default |
|---|---|---|
| `PARAS_KV_TRANSFER_METHOD` | KV transfer method in production | `nccl` |
| `PARAS_CONFIGURE_METHOD` | Weight transfer method | `naive` |
| `CUDA_VISIBLE_DEVICES` | GPU selection | all |

## Interpreting Results

### Correctness
```
PASSED test_ep_to_tp_no_replication        ← EP→TP, 8h/8g, pattern verified
PASSED test_ep_to_tp_with_replication      ← EP→TP, 4h/8g R=2, pattern verified
PASSED test_tp_to_ep_no_replication        ← TP→EP, 8h/8g, pattern verified
PASSED test_tp_to_ep_with_replication      ← TP→EP, 4h/8g R=2, pattern verified
PASSED test_roundtrip_no_replication       ← EP→TP→EP bitwise match
PASSED test_roundtrip_with_replication     ← EP→TP→EP bitwise match, R=2
```

Any `FAILED` means data corruption. Common causes:
- CUDA extension not recompiled after kernel changes
- GPU memory contention from other processes
- Wrong `CUDA_VISIBLE_DEVICES`

## Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| `Need N empty GPUs, only M available` | Other processes using GPUs | Kill them or wait |
| `NCCL timeout` | Leftover state from crashed run | `pkill -f torchrun` and retry |
| `Import error: paras_peer_access_cuda` | CUDA extension not compiled | `cd python/sglang/srt/paras/csrc && pip install -e .` |
| `assert world_size in (4, 8)` | Wrong number of GPUs | Set `CUDA_VISIBLE_DEVICES` correctly |
| `assert False` in paras_moe_block | Old code before bug fix | Pull latest changes |
