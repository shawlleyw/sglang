---
name: paras-test-peer-access
description: Run ParaS correctness tests for KV cache transfer (EP→TP and TP→EP, with/without head replication), weight transfer, and request partition. Knows GPU requirements, conda env, torchrun commands, test structure, and how to interpret results.
metadata:
  short-description: Test ParaS KV cache + weight transfer + partition
---

# ParaS Transfer Tests

Correctness tests for ParaS parallelism switching: KV cache transfer (both directions), weight transfer, and request partition. Every transfer method (NCCL naive, peer_access NVLink kernel) is verified independently against computed ground truth — never against each other.

## Quick Run

```bash
bash scripts/paras/eval/run_paras_tests.sh        # 4 GPUs (default)
bash scripts/paras/eval/run_paras_tests.sh 8      # 8 GPUs

# Single group (partition | kv | kv-rep | weight | gpt-oss-cuda-graph):
ONLY=kv bash scripts/paras/eval/run_paras_tests.sh
```

The wrapper ([`scripts/paras/eval/run_paras_tests.sh`](file:///home/shaoyuw/sglang/scripts/paras/eval/run_paras_tests.sh)) sources [`scripts/paras/eval/lib.sh`](file:///home/shaoyuw/sglang/scripts/paras/eval/lib.sh), defaults `CUDA_VISIBLE_DEVICES=0,...,NUM_GPUS-1`, runs each group, and prints a PASS/FAIL summary. Replication tests run in their own process to avoid CUDA IPC stale-handle issues across mismatched buffer sizes (see "Why KV replication is a separate file" below). Each section below documents what each group covers; the per-section `torchrun` snippets remain available for running a single test file with custom pytest options.

## Prerequisites

- Conda env: `sgl_paras`
- CUDA extension: `cd python/sglang/srt/paras/csrc && python setup.py build_ext --inplace`
- Empty GPUs: `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits` (all < 100 MiB)

## Test Coverage

### What is tested

All tests verify against **independently computed ground truth**, not round-trip symmetry or method-vs-method comparison.

| Component | Direction | Methods | Replication | Ground Truth |
|-----------|-----------|---------|-------------|-------------|
| KV cache | EP→TP | NCCL, peer_access | R=1, R=2 | `make_pattern(rank, layer, head, token)` encodes source identity. Expected TP value at any position computed from pattern. |
| KV cache | TP→EP | NCCL, peer_access | R=1, R=2 | Same pattern. Expected EP value computed by tracing which TP rank sent which token slice to which head slot. |
| KV cache | EP→TP→EP | NCCL | R=1, R=2 | Bitwise snapshot: save EP before, compare after round-trip. |
| MoE weights | EP→TP | NCCL, peer_access | — | All-gather EP data from all ranks, compute expected TP shards: `gate_shard = full_w13[:, r*I_tp:(r+1)*I_tp, :]`, `up_shard = full_w13[:, I+r*I_tp:I+(r+1)*I_tp, :]`. |
| MoE weights | TP→EP | NCCL reverse, peer_access reverse | — | Original EP snapshot (since EP→TP ground truth is verified, reverse must recover original). |
| MoE weights | EP→TP→EP | NCCL | — | Bitwise snapshot match. |
| Request partition | — | Greedy algorithm | — | Deterministic: `sort(-seqlen, rid)`, assign to rank with fewest requests then least tokens. Verified: count balance, token balance, no duplicates, no losses, cross-input determinism. |

### How the ground truth is built

**KV cache pattern**: `make_pattern(rank, layer, head, num_tokens)` returns a `(num_tokens, HEAD_DIM)` bf16 tensor where each value encodes the source:
```python
base = rank * 1000.0 + layer * 100.0 + head * 10.0
value[t, d] = base + t + d * 0.001
```
After transfer, the expected value at any destination position is computed by knowing which source rank/head/token should end up there. A mismatch of ~1000 means wrong source rank; ~100 means wrong layer; ~10 means wrong head.

**Weight ground truth**: All ranks' EP weights are all-gathered to build a global view `(NUM_EXPERTS, 2*I, H)`. The expected TP shard for rank `r` is extracted via column slicing. This is computed once and compared against both NCCL and peer_access results independently.

**Replication (R>1)**: When `num_kv_heads < tp_size`, R contiguous ranks share the same head. Each subgroup member sends a disjoint 1/R token slice. The verification traces which intra_rank within the subgroup sent each token position and checks against that member's pattern.

### What is NOT tested (out of scope)

- Performance benchmarks (latency measurement) — planned, not yet implemented
- FP8 KV cache
- dp_size > 1
- Automatic switching policy
- FlashInfer attention backend integration (tested via E2E only)

## Test Files

```
test/srt/paras/
├── test_request_partition.py                  # 11 CPU tests
├── test_kv_cache_transfer.py                  # 5 GPU tests (R=1 only)
├── test_kv_cache_transfer_replication.py      # 5 GPU tests (R=2 only)
├── test_weight_transfer.py                    # 6 GPU tests
├── test_memory.py                             # 2 GPU tests
└── test_roundtrip.py                          # 4 GPU tests
```

### Why KV replication is a separate file

Tests with different `num_kv_heads` allocate managed buffers of different sizes. CUDA IPC handles (`cudaIpcOpenMemHandle`) mapped to the old buffer become stale when the buffer is reallocated. Running R=1 and R=2 tests in the same process causes address space corruption. Separating into two files ensures each runs in a fresh process.

---

## 1. Request Partition Tests (11 CPU tests)

```bash
python -m pytest test/srt/paras/test_request_partition.py -v
```

| Class | Tests | What it verifies |
|-------|-------|-----------------|
| `TestPartitionRequestsForEP` | 5 | Balanced assignment, fewer-than-ranks, zero requests, determinism with equal seqlens, imbalanced (count-first priority) |
| `TestPeerAccessReplicationRouting` | 4 | 1/R token slicing: R=1 routes all, R=2 subgroup partners cover 100%, exhaustive no-token-lost for many sizes, R=4 |
| `TestPartitionStrategy` | 2 | Strategy registry works, unknown strategy raises ValueError |

---

## 2. KV Cache Transfer — No Replication (5 GPU tests)

```bash
torchrun --nproc_per_node=4 -m pytest test/srt/paras/test_kv_cache_transfer.py -v
```

| Test | Direction | Method | Verification |
|------|-----------|--------|-------------|
| `test_ep_to_tp_no_replication` | EP→TP | NCCL | Pattern ground truth |
| `test_ep_to_tp_peer_access_no_replication` | EP→TP | peer_access | Pattern ground truth |
| `test_tp_to_ep_no_replication` | TP→EP | NCCL | Pattern ground truth |
| `test_tp_to_ep_peer_access_no_replication` | TP→EP | peer_access | Pattern ground truth |
| `test_roundtrip_no_replication` | EP→TP→EP | NCCL | Bitwise snapshot |

---

## 3. KV Cache Transfer — With Replication (5 GPU tests)

```bash
torchrun --nproc_per_node=4 -m pytest test/srt/paras/test_kv_cache_transfer_replication.py -v
```

| Test | Direction | Method | Verification |
|------|-----------|--------|-------------|
| `test_ep_to_tp_nccl` | EP→TP | NCCL | Pattern ground truth (R=2) |
| `test_ep_to_tp_peer_access` | EP→TP | peer_access | Pattern ground truth (R=2) |
| `test_tp_to_ep_nccl` | TP→EP | NCCL | Pattern ground truth (R=2) |
| `test_tp_to_ep_peer_access` | TP→EP | peer_access | Pattern ground truth (R=2) |
| `test_roundtrip` | EP→TP→EP | NCCL | Bitwise snapshot (R=2) |

---

## 4. Weight Transfer (6 GPU tests)

```bash
torchrun --nproc_per_node=4 test/srt/paras/test_weight_transfer.py
```

| Test | Direction | Method | Verification |
|------|-----------|--------|-------------|
| `test_nccl_vs_ground_truth` | EP→TP | NCCL | Computed gate/up shards from all-gathered EP |
| `test_peer_access_vs_ground_truth` | EP→TP | peer_access | Same ground truth |
| `test_nccl_reverse_vs_original` | TP→EP | NCCL reverse | Original EP snapshot |
| `test_peer_access_reverse_vs_original` | TP→EP | peer_access reverse | Same original EP snapshot |
| `test_moe_pointer_swap` | TP→EP | Module attribute | `self.experts is self.ep_experts` after configure_ep |
| `test_weight_roundtrip` | EP→TP→EP | NCCL | Bitwise snapshot (reversed layer order) |

---

## Key Design Decisions

### N+1 slot aliasing

EP layer `i` uses slot[i+1], TP layer `i` uses slot[i]. Slot[i+1] = TP slot for layer i+1. This means:
- **EP→TP forward order** (0→N-1): safe because layer i reads slot[i+1] before layer i+1 writes to slot[i+1]
- **TP→EP reverse order** (N-1→0): safe because layer i+1 reads slot[i+1] before layer i writes to slot[i+1]
- **EP→TP DESTROYS EP weight data** in slots 1..N-1. TP→EP must use actual reverse all-to-all transfer, NOT pointer swap.

### KV cache scatter with replication

When `num_kv_heads < tp_size` (replication factor R>1):
- Each subgroup of R contiguous ranks holds identical KV data
- Each member sends a disjoint 1/R token slice, cutting NVLink traffic by R
- The NCCL path is a single unified code path — the only conditional is `reassembly_groups = group_size if heads_per_rank > 1 else num_kv_heads`
- The peer_access kernel uses `num_kv_heads` (not `heads_per_rank * tp_size`) for the EP destination stride

## Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| GPU memory not free | Other processes | `nvidia-smi` to check, kill or wait |
| `NCCL timeout` | Leftover from crashed run | `pkill -f torchrun` and retry |
| `Import error: paras_peer_access_cuda` | CUDA extension not compiled | `cd python/sglang/srt/paras/csrc && python setup.py build_ext --inplace` |
| R=2 tests fail in full suite | CUDA IPC isolation | Run replication tests separately (already split into own file) |
| `assert False` in paras_moe_block | Old code | Pull latest changes |
