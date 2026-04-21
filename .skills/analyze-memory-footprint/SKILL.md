---
name: analyze-memory-footprint
description: Investigate GPU memory footprint of SGLang server configs (TP / EP / ParaS / MoE + DeepEP). Decompose driver-level memory into named buckets (PyTorch caching allocator, CUDA graph private pool, DeepEP Buffer + workspace, NVSHMEM symmetric heap, NCCL communicator scratch, residual "other"). Covers the ParaS[mem-breakdown] instrumentation, SGLANG_DUMP_MEM_SEGMENTS segment-level dump, log parsing, experiment methodology, and offload-reachability analysis.
metadata:
  short-description: Decompose GPU memory into measurable buckets and reason about what's reclaimable
---

# Analyzing GPU Memory Footprint

## Mental Model: 4 Layers of GPU Memory

```
driver_used (cudaMemGetInfo used = total - free)
├── torch_reserved (PyTorch caching allocator: reserved_bytes.all.current)
│   ├── graph_pool      (segment_pool_id == (0, N>0): CUDA graph private pool)
│   └── default_pool    (segment_pool_id == (0, 0): everything else PyTorch tracks)
└── non_torch = driver_used - torch_reserved
    ├── deepep_buf      (DeepEPBuffer._buffer.num_nvl_bytes + num_rdma_bytes)
    ├── deepep_ws       (32 MiB hardcoded workspace per live DeepEP Buffer)
    ├── nvshmem_heap    (NVSHMEM_SYMMETRIC_SIZE, default 1 GiB, per PE)
    ├── nccl_scratch    (per-communicator NCCL buffers — estimate)
    └── other           (residual: NCCL graph-capture VMM, cuBLAS, misc)
```

**Why this decomposition matters**: You cannot reason about what memory is reclaimable without knowing where it came from. TMS `release(cuda_graph)` pauses only `graph_pool`. TMS `release(weights)` pauses only weight-region PyTorch tensors. NCCL/NVSHMEM/DeepEP bypass PyTorch's caching allocator entirely, so `torch.cuda.empty_cache()` and TMS tags do nothing for them.

---

## Instrumentation (already wired up in this repo)

Commits `a598b0b87`, `905091d5c`, `6ff7c38d1` add breakdown logging to `cuda_graph_runner.capture()`. Two tiers of output:

### Tier 1: Terse summary — ALWAYS on

One line per rank per capture showing the headline numbers. Cheap, always printed:

```
ParaS[mem-summary] post-capture: driver_used=XX.XX GB  torch_reserved=XX.XX GB
                                 non_torch=XX.XX GB  (capture-delta: driver=±X.XX GB)
```

### Tier 2: Detailed breakdown — OPT-IN via `SGLANG_PARAS_MEM_LOG=1`

When the env var is set, three additional log lines per capture give the full per-bucket decomposition:

```
ParaS[mem-breakdown:pre-capture]   driver_used  torch_reserved  graph_pool
                                    default_pool  non_torch
                                    (deepep_buf  deepep_ws  nvshmem  nccl_est  other)
                                    pools={(0,0)=..., (0,1)=...}
ParaS[mem-breakdown:post-capture]  (same fields)
ParaS[mem-breakdown:capture-delta] (post - pre) for each field
```

**Usage:**
```bash
# Default prod run — only Tier 1 summary
python -m sglang.launch_server ...

# Debug / memory investigation — Tier 1 + Tier 2
SGLANG_PARAS_MEM_LOG=1 python -m sglang.launch_server ...
```

Source: `python/sglang/srt/paras/paras_cuda_graph.py::paras_memory_breakdown()`, `paras_log_memory_breakdown()`, and `paras_mem_log_enabled()`. Called from `python/sglang/srt/model_executor/cuda_graph_runner.py::CudaGraphRunner.capture()`.

### Tier 3: Segment-level dump — OPT-IN via `SGLANG_DUMP_MEM_SEGMENTS=1`

See the dedicated section below. Complementary to Tier 2 — fires at Memory pool end (before graph capture), and shows torch allocator segments rather than bucket totals.

### Note on config scope

The breakdown works on **any** sglang config, not just ParaS:
- Plain TP server: `graph_pool` = captured TP graphs, `deepep_*` = 0, `nvshmem` = 0.
- Baseline EP (no ParaS): all buckets populated.
- ParaS: two graph pools show up (one per mode), all other buckets match baseline EP.

The unconditional one-time `ParaS KV budget:` log line at ParaS model init (from `qwen3_moe.py`) is orthogonal to `SGLANG_PARAS_MEM_LOG` — it always prints so users can see the derivation of `ep_max_tokens`.

---

## Field Semantics (READ THIS before interpreting numbers)

| Field | How computed | What it measures | Gotchas |
|---|---|---|---|
| `driver_used` | `cudaMemGetInfo().used` | Physical GPU memory used by this process | Includes ALL CUDA state; never lies |
| `torch_reserved` | `memory_stats().reserved_bytes.all.current` | Total bytes held by PyTorch's caching allocator (sum of all segments) | Reserved ≠ allocated. Cached free blocks count here. |
| `graph_pool` | Sum of `memory_snapshot()` segments with `pool_id[1] > 0` | Physical memory in CUDA graph private pools | 100% reclaimable via TMS `cuda_graph` tag (synthetic probe confirms) |
| `default_pool` | Sum of segments with `pool_id == (0, 0)` | Normal caching allocator segments | Weights + KV cache + workspaces live here |
| `non_torch` | `driver_used - torch_reserved` | Everything the PyTorch allocator doesn't track | NCCL + NVSHMEM + DeepEP + cuBLAS + driver internals |
| `deepep_buf` | `DeepEPBuffer._buffer.num_nvl_bytes + num_rdma_bytes` | DeepEP's NVL/RDMA scratch buffers | Measured directly from the live Buffer object. Reliable. |
| `deepep_ws` | 32 MiB constant per live `DeepEPBuffer._buffer` | Hardcoded workspace in `deep_ep/csrc/deep_ep.cpp:192` | Constant; reliable |
| `nvshmem` | `NVSHMEM_SYMMETRIC_SIZE` env (default 1 GiB) when DeepEP's NVSHMEM init fired | Symmetric heap **reservation** (virtual) | Physical commit is lazy in 512 MiB chunks (`NVSHMEM_CUMEM_GRANULARITY`). Actual physical may be less than reported. |
| `nccl_est` | `len(live NCCL comms) × 144 MiB` | Upper-bound estimate | Real per-comm usage depends on NCCL channel choice (4–8) and whether send/recv was exercised |
| `other` | residual after all above | Unattributed non_torch | Most-likely contributors: NCCL graph-capture VMM scratch, lazy cuBLAS workspaces, ParaS peer-access IPC |

**Hard rule**: Only `driver_used`, `torch_reserved`, `graph_pool`, `default_pool`, and `deepep_buf`/`deepep_ws` are **measured**. The rest are either env-var-derived (`nvshmem`) or upper-bound estimates (`nccl_est`). Do not aggregate them as if they were all measured.

---

## Segment-Level Memory Dump (`SGLANG_DUMP_MEM_SEGMENTS=1`)

When the breakdown is insufficient to attribute a `torch_reserved` gap — e.g. you want to know "which specific allocations inside default_pool grew between config A and config B?" — use the opt-in segment dump.

### What it does

At the `Memory pool end` checkpoint (after model load + KV pool init, before CUDA graph capture), `model_runner.py` calls `torch.cuda.memory_snapshot()` and logs the top-20 largest live allocations with their reserved/allocated bytes, largest block size, and block count.

Source: `python/sglang/srt/model_executor/model_runner.py` (search for `SGLANG_DUMP_MEM_SEGMENTS`). Disabled by default (opt-in env var), so it never pollutes normal runs.

### How to enable

```bash
SGLANG_DUMP_MEM_SEGMENTS=1 python -m sglang.launch_server ... 2>&1 | tee artifacts/<name>_memdump.log
```

### Log format

Per rank, one summary line plus 20 per-segment lines:

```
[MEMDUMP:PARAS] segments=19 total_allocated=49.079GiB total_reserved=49.086GiB
[MEMDUMP:PARAS] #00  alloc=47876.00MiB reserved=47876.00MiB largest_block=47876.00MiB type=large stream=0 nblocks=1
[MEMDUMP:PARAS] #01  alloc= 1024.02MiB reserved= 1026.00MiB largest_block=1024.02MiB type=large stream=0 nblocks=2
[MEMDUMP:PARAS] #02  alloc=  594.00MiB reserved=  594.00MiB largest_block= 594.00MiB type=large stream=0 nblocks=1
...
```

The tag is `PARAS` or `BASE` depending on whether `enable_paras_moe` is set, so you can grep for the tag without checking which log you have open.

### Fields

| Field | Meaning |
|---|---|
| `alloc` | Bytes currently allocated (in-use) in the segment |
| `reserved` | Bytes the allocator has reserved (reserved ≥ alloc; diff is cached free blocks) |
| `largest_block` | Largest currently-active allocated block inside the segment (useful for identifying which tensor dominates) |
| `nblocks` | Number of distinct blocks in the segment (high count → segment is holding many small tensors) |
| `type` | `large` (>1 MiB default) or `small` (cache of small allocations) |
| `stream` | CUDA stream the segment is associated with |

### Example: diff two runs

```python
# scripts/diff_memdumps.py (or inline in a notebook)
import re
def parse(path, tag):
    pat = re.compile(rf"DP0 TP0 EP0\].*\[MEMDUMP:{tag}\] #(\d+) alloc=\s*([\d.]+)MiB reserved=\s*([\d.]+)MiB largest_block=\s*([\d.]+)MiB type=(\w+) stream=(\S+) nblocks=(\d+)")
    total_pat = re.compile(rf"DP0 TP0 EP0\].*\[MEMDUMP:{tag}\] segments=(\d+) total_allocated=([\d.]+)GiB total_reserved=([\d.]+)GiB")
    segs, total = [], None
    for line in open(path):
        if m := total_pat.search(line):
            total = (int(m.group(1)), float(m.group(2)), float(m.group(3)))
        if m := pat.search(line):
            segs.append({"idx": int(m.group(1)), "alloc": float(m.group(2)), "reserved": float(m.group(3)), "largest": float(m.group(4)), "type": m.group(5), "nblocks": int(m.group(7))})
    return total, sorted(segs, key=lambda s: s["idx"])

a_total, a_segs = parse("artifacts/ep_memdump.log", "BASE")
b_total, b_segs = parse("artifacts/paras_memdump.log", "PARAS")
print(f"BASE:  {a_total[0]:3d} segs, alloc={a_total[1]:.3f}GiB, reserved={a_total[2]:.3f}GiB")
print(f"PARAS: {b_total[0]:3d} segs, alloc={b_total[1]:.3f}GiB, reserved={b_total[2]:.3f}GiB")
print(f"Gap:   {b_total[2] - a_total[2]:+.3f} GiB reserved")
```

### When to use vs the `ParaS[mem-breakdown:*]` lines

| Question | Tool |
|---|---|
| "What's in `non_torch` (NCCL, NVSHMEM, DeepEP)?" | `ParaS[mem-breakdown]` — already decomposes non_torch |
| "How much does graph capture cost?" | `ParaS[mem-breakdown:capture-delta]` |
| "Why is `torch_reserved` 2 GB bigger in config B?" | `SGLANG_DUMP_MEM_SEGMENTS=1`, diff segments |
| "Is there a ParaS-only tensor baseline doesn't allocate?" | `SGLANG_DUMP_MEM_SEGMENTS=1`, diff the top segments |
| "How many NCCL comms are live?" | `grep "nccl_est" artifacts/<log>` — comm names are in the label |
| "What's the torch allocator fragmentation overhead?" | `SGLANG_DUMP_MEM_SEGMENTS=1`, compute `total_reserved - total_allocated` |

### Gotcha: segment != tensor

A single segment can contain many tensors (small-block pools especially). Conversely, one large tensor is usually one segment. Use `largest_block` and `nblocks` to distinguish: `nblocks=1` with `largest_block ≈ alloc` means one giant tensor; `nblocks=175` with `largest_block=0.5` means a small-pool holding many tiny allocations.

### Gotcha: checkpoint timing

The dump fires at "Memory pool end", which is:
- **After**: model weight load, KV cache allocator init
- **Before**: CUDA graph capture, first request, any DeepEP/NVSHMEM lazy allocation from warmup

If you need a dump at a different lifecycle point (post-capture, after first request), add another call site or change the env-var-gated block.

---

## Methodology: Investigating a Memory Question

### Step 1: Baseline capture, save logs

Always save to a persistent path. `/tmp` gets wiped. Use `/home1/wangshao/sglang/artifacts/<name>.log`.

```bash
mkdir -p /home1/wangshao/sglang/artifacts

tmux new-session -d -s sglrun "\
  source /home1/wangshao/miniconda3/etc/profile.d/conda.sh && \
  conda activate sgl_sm80 && \
  cd /home1/wangshao/sglang && \
  export CUDA_VISIBLE_DEVICES=0,1,2,3 && \
  python -m sglang.launch_server \
      --model /scratch1/wangshao/models/Qwen3-30B-A3B-Instruct-2507 \
      --trust-remote-code \
      --mem-fraction-static 0.6 \
      --tp-size 4 ... \
      --cuda-graph-max-bs 256 \
      --log-level info 2>&1 | tee /home1/wangshao/sglang/artifacts/<name>.log"

# Wait for "The server is fired up and ready to roll!"
# Then kill:
pgrep -f sglang.launch_server | xargs -r kill -9
pgrep -f 'sglang::' | xargs -r kill -9
tmux kill-session -t sglrun
```

If the instrumentation is missing from a log, verify:
```bash
grep -c "ParaS\[mem-breakdown" artifacts/<name>.log
# Should be 12 on 4-GPU runs (4 ranks × 3 lines: pre/post/delta)
```

### Step 2: Extract the breakdown numbers

```bash
# All breakdown lines:
grep "ParaS\[mem-breakdown" artifacts/<name>.log

# Just DP0 capture-delta:
grep -E "DP0 TP0 EP0.*mem-breakdown:capture-delta" artifacts/<name>.log
# or for plain TP (no DP):
grep -E "TP0\].*mem-breakdown:capture-delta" artifacts/<name>.log

# Side-by-side of two runs, DP0 only:
for f in baseline variant; do
  echo "=== $f ==="
  grep -E "(TP0\b|DP0 TP0 EP0)" artifacts/${f}.log | grep "capture-delta"
done
```

### Step 3: Sanity-check arithmetic

Every breakdown line should satisfy these invariants. If any fail, the instrumentation is broken (or you misread the log):

```
torch_reserved ≈ graph_pool + default_pool           (within 1 MB rounding)
driver_used    = torch_reserved + non_torch          (exact)
non_torch      = deepep_buf + deepep_ws + nvshmem
                 + nccl_est + other                   (exact, by construction)
```

Example DP0 line to verify:
```
driver_used=53.326  torch_reserved=48.816
  (graph_pool=0.543, default_pool=48.273)  → 0.543+48.273=48.816 ✓
non_torch=4.509
  (deepep_buf=0.566 + deepep_ws=0.031
   + nvshmem=1.000 + nccl_est=0.703 + other=2.208)  → total 4.508 ≈ 4.509 ✓
driver_used 53.326 = torch_reserved 48.816 + non_torch 4.509 + 0.001 rounding ✓
```

### Step 4: Comparative analysis

Drive every conclusion from a **delta between two runs** whose only difference is the hypothesis under test. Example questions:

- "How much does `--cuda-graph-max-bs` cost?" → run bs=64 and bs=256, compare `graph_pool Δ` and `other Δ`.
- "Does `NCCL_CUMEM_ENABLE=0` reduce graph-capture memory?" → run with and without, compare `other Δ`.
- "How much ParaS overhead is real?" → run plain EP and ParaS, compare total steady state.

### Step 5: Offload-reachability check

For each bucket, answer "can we free this at runtime?"

| Bucket | Reclaimable by... | Cost to reclaim |
|---|---|---|
| `graph_pool` | Drop `CUDAGraph` + tensors OR `tms.pause(tag)` on the capture region | 0 (pause) or ~re-capture time (drop) |
| `default_pool` | `torch.cuda.empty_cache()` reclaims unreferenced segments | Low |
| `deepep_buf` | Destroy `DeepEPBuffer._buffer` | ~seconds (requires torn-down DeepEP state) |
| `deepep_ws` | Destroyed with Buffer | Same as above |
| `nvshmem` | `nvshmem_finalize()` | Destroys all NVSHMEM state; likely requires reinitialization |
| `nccl_est` | Destroy NCCL communicators | Seconds of communicator init |
| `other` | Depends on what's in it | Unknown until decomposed |

---

## Playbook: Common Investigations

### Playbook A: "What's in the `other` bucket?"

`other` is the residual. To decompose it, toggle known contributors one at a time.

#### Suspect 1: NCCL graph-capture VMM scratch (most likely)

NCCL 2.19+ with `NCCL_CUMEM_ENABLE=1` (default) allocates per-captured-graph buffers via `cuMemCreate`/`cuMemMap`. See NCCL issue #1234. Per-graph cost scales with number of captured batch sizes.

```bash
# Run same config with NCCL_CUMEM_ENABLE=0 and compare 'other'
NCCL_CUMEM_ENABLE=0 python -m sglang.launch_server ... 2>&1 | tee artifacts/ep_bs256_nocumem.log
diff <(grep "capture-delta" artifacts/ep_bs256.log | head -4) \
     <(grep "capture-delta" artifacts/ep_bs256_nocumem.log | head -4)
```

If `other` drops substantially (→ hundreds of MB), NCCL VMM was the main contributor.

#### Suspect 2: Lazy cuBLAS workspaces

cuBLAS/cuBLASLt allocates workspaces on first use for each problem shape. Scales with kernel-shape diversity, not batch count.

```bash
# Set an explicit workspace and re-measure
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m sglang.launch_server ... 2>&1 | tee artifacts/ep_bs256_cublasws.log
```

If `other` grows with this setting and shrinks with `CUBLAS_WORKSPACE_CONFIG=:1024:1`, you've proven cuBLAS scale.

#### Suspect 3: Number of NCCL communicators

If `nccl_est` is off, the `other` residual may be absorbing the difference.

```bash
# Count live communicators in the log:
grep "nccl_est.*GB\[" artifacts/<name>.log | head -1
# Example output: nccl_est=0.703GB[5:_WORLD,_TP,_PP,_MOE_EP,_MOE_TP]
```

Dump NCCL's actual allocations with `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALLOC` and sum the reported sizes to get ground-truth vs. the 144 MiB-per-comm estimate.

#### Suspect 4: ParaS peer-access IPC buffers

ParaS reserves peer-access IPC memory during init. If `other` is much higher in ParaS than baseline EP at pre-capture, that's likely it.

```bash
grep "pre-capture" artifacts/ep_bs256.log | head -1   # baseline EP pre-capture 'other'
grep "pre-capture" artifacts/paras_bs256.log | head -1   # ParaS pre-capture 'other'
# Difference ≈ ParaS peer-access state
```

### Playbook B: "Is TMS `release(cuda_graph)` worth enabling?"

Only `graph_pool` is reachable. Per-mode in our measurements:

| Config | `graph_pool` |
|---|---:|
| Plain TP (`--cuda-graph-max-bs 256`) | 0.18 GB |
| Baseline EP (`--cuda-graph-max-bs 256`) | 0.54 GB |
| ParaS (dual EP+TP) | 0.54 + 0.10 = 0.64 GB |

Ceiling savings ≈ those numbers. Everything else (DeepEP buffer 0.57 GB, NVSHMEM 1.0 GB, NCCL 2 GB+, ParaS peer-access 3 GB) is process-scoped.

### Playbook C: "Why does EP use more memory than TP?"

Compare baseline EP and baseline TP capture-delta, component by component:

```bash
# From artifacts/ep_bs256_v2.log and tp_bs256_v2.log:
# EP: driver=+4.000  graph_pool=+0.543  deepep_buf=+0.566
#     deepep_ws=+0.031  nvshmem=+1.000  nccl_est=+0.000
#     other=+1.486
# TP: driver=+0.924  graph_pool=+0.176  deepep_buf=+0.000
#     deepep_ws=+0.000  nvshmem=+0.000  nccl_est=+0.000
#     other=+0.375
```

Subtract row by row. EP − TP = 2.709 GB in capture delta, of which:
- 0.566 GB DeepEP Buffer (attributed)
- 0.031 GB DeepEP workspace (attributed)
- 1.000 GB NVSHMEM reservation (attributed)
- 0.367 GB graph_pool (EP has more graph-captured work)
- 1.111 GB in `other` (unattributed — triggers Playbook A)

### Playbook D-seg: "Which specific tensor contributes to the torch_reserved gap?"

When `ParaS[mem-breakdown]` shows `default_pool` grew but doesn't decompose further, use the segment dump.

```bash
SGLANG_DUMP_MEM_SEGMENTS=1 python -m sglang.launch_server ... --enable-paras-moe ... 2>&1 | tee artifacts/paras_memdump.log
SGLANG_DUMP_MEM_SEGMENTS=1 python -m sglang.launch_server ...                       2>&1 | tee artifacts/base_memdump.log

# Side-by-side top-5
paste <(grep "DP0 TP0 EP0.*MEMDUMP:BASE"  artifacts/base_memdump.log  | head -5) \
      <(grep "DP0 TP0 EP0.*MEMDUMP:PARAS" artifacts/paras_memdump.log | head -5)
```

Look for:
- **New segments in PARAS that aren't in BASE** → ParaS-only tensors (IPC state, extra workspace, etc.)
- **Segments that got bigger** → structural overhead (N+1 slot, bigger KV pool, etc.)
- **`total_reserved - total_allocated` delta** → allocator fragmentation difference (usually small; baseline has ~35 MiB slack, single-UMM configs have ~7 MiB)

See the example diff in `docs/paras/memory_analysis.md` for a fully-worked case where this approach identified a 128 MiB ParaS-only segment, the one-giant-alloc rounding cost, and the N+1 slot structural overhead.

### Playbook E: "What changed between two runs?"

Always save logs to `artifacts/` with a descriptive name. Compare with a dedicated script:

```python
import re
from pathlib import Path

def parse_delta(log_path, rank="DP0 TP0 EP0"):
    text = Path(log_path).read_text()
    for line in text.splitlines():
        if rank in line and "capture-delta" in line:
            # Extract each GB number
            fields = {}
            for m in re.finditer(r"(\w+)=([+-]?[0-9.]+)GB", line):
                fields[m.group(1)] = float(m.group(2))
            return fields
    return None

a = parse_delta("artifacts/ep_bs256_v2.log")
b = parse_delta("artifacts/ep_bs256_nocumem.log")
for k in sorted(set(a) | set(b)):
    d = b.get(k, 0) - a.get(k, 0)
    if abs(d) > 0.01:
        print(f"{k}: {a.get(k,0):+.3f} -> {b.get(k,0):+.3f}  delta {d:+.3f} GB")
```

---

## Reference Numbers (Qwen3-30B-A3B, 4×A100-80GB, bs=256)

### Steady-state driver_used per GPU

| Config | driver_used |
|---|---:|
| Plain TP (`--tp-size 4`) | 52.7 GB |
| Baseline EP (`--tp-size 4 --dp-size 4 --ep-size 4 --enable-dp-attention --moe-a2a-backend deepep`) | 53.3 GB |
| ParaS (same + `--enable-paras-moe --paras-tp-size 4`) | 59.4 GB |

### Non-torch decomposition at steady state per GPU

| Bucket | TP | EP | ParaS |
|---|---:|---:|---:|
| deepep_buf | 0 | 0.57 | 0.57 |
| deepep_ws | 0 | 0.03 | 0.03 |
| nvshmem | 0 | 1.00 | 1.00 |
| nccl_est | 0.70 | 0.70 | 0.70 |
| other | 1.10 | 2.21 | ~5.00 |
| **Total non_torch** | **1.80** | **4.51** | **~7.30** |

### Capture-delta per GPU (EP vs TP)

| Bucket | EP Δ | TP Δ | EP − TP |
|---|---:|---:|---:|
| driver_used | +4.000 | +0.924 | +3.076 |
| graph_pool | +0.543 | +0.176 | +0.367 |
| default_pool | +0.373 | +0.373 | 0 |
| non_torch | +3.084 | +0.375 | +2.709 |
| &nbsp;↳ deepep_buf | +0.566 | 0 | +0.566 |
| &nbsp;↳ deepep_ws | +0.031 | 0 | +0.031 |
| &nbsp;↳ nvshmem | +1.000 | 0 | +1.000 |
| &nbsp;↳ nccl_est | 0 | 0 | 0 |
| &nbsp;↳ other | +1.486 | +0.375 | +1.111 |

---

## Useful Env Vars (knobs for experiments)

| Env var | What it does | Likely effect on memory |
|---|---|---|
| `NCCL_CUMEM_ENABLE=0` | Force NCCL to use `cudaMalloc` instead of `cuMemCreate`/`cuMemMap` for internal buffers | Reduces `other` (per NCCL #1234). May slightly hurt perf. |
| `NCCL_MAX_NCHANNELS=4` | Halve NCCL channel count | Roughly halves `nccl_est` + reduces NCCL graph scratch. Slight BW cost on A100 NVLink. |
| `NVSHMEM_SYMMETRIC_SIZE=536870912` | Set NVSHMEM heap to 512 MiB | Reduces `nvshmem` reservation. Must be ≥ DeepEP's `num_rdma_bytes` rounded to `NVSHMEM_CUMEM_GRANULARITY` (default 512 MiB). |
| `NVSHMEM_DEBUG=INFO NVSHMEM_DEBUG_SUBSYS=ALLOC` | Dump NVSHMEM allocations | Lets you verify physical-commit size vs our assumed reservation. |
| `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALLOC` | Dump NCCL allocations | Ground-truth for per-comm sizes. |
| `CUBLAS_WORKSPACE_CONFIG=:4096:8` | Set cuBLAS workspace to 8 × 4 KiB blocks | Increases reserved cuBLAS workspace (~24 MiB per handle). Feeds into `other`. |
| `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` | Switch PyTorch to cudaMallocAsync backend | Changes everything — `graph_pool` stops being visible to TMS's cudaMalloc hook. Do not set unless you want to break TMS. |

---

## Raw Logs for Reference

All in `/home1/wangshao/sglang/artifacts/`:

### On branch `paras_cudagraph` (original instrumentation runs)

| File | Config |
|---|---|
| `tp_bs256_v2.log` | Plain TP=4, `--cuda-graph-max-bs 256`, full decomposition |
| `ep_bs256_v2.log` | Baseline EP=4 + DeepEP, `--cuda-graph-max-bs 256`, full decomposition |
| `paras_bs256.log` | ParaS dual EP+TP (older: pre-decomposition instrumentation) |
| `ep_bs256.log`, `tp_bs256.log` | Original runs (no nvshmem/nccl_est subfields) |

### On branch `paras_memory_opt` (optimization runs)

| File | Config |
|---|---|
| `paras_pre_alias.log` | ParaS before NCCL alias fix (driver_used ≈ 60.2 GB) |
| `paras_post_alias.log` | ParaS after NCCL alias fix (driver_used ≈ 57.8 GB) |
| `paras_budget_fix.log` | ParaS after NCCL alias + budget-semantics fix (driver_used ≈ 56.3 GB) |
| `ep_memdump.log` | Baseline EP with `SGLANG_DUMP_MEM_SEGMENTS=1` |
| `paras_memdump.log` | ParaS with alias fix + segment dump |
| `paras_fix_memdump.log` | Final ParaS (alias + budget fix) with segment dump |

---

## Related Code & Docs

| Path | Purpose |
|---|---|
| `python/sglang/srt/paras/paras_cuda_graph.py::paras_memory_breakdown()` | Core decomposition (lines ~38–180) |
| `python/sglang/srt/paras/paras_cuda_graph.py::paras_log_memory_breakdown()` | One-line log formatter |
| `python/sglang/srt/model_executor/cuda_graph_runner.py::CudaGraphRunner.capture()` | Pre/post/delta call sites around capture loop |
| `python/sglang/srt/model_executor/model_runner.py` (`SGLANG_DUMP_MEM_SEGMENTS`) | Opt-in segment-level dump at "Memory pool end" |
| `python/sglang/srt/layers/moe/token_dispatcher/deepep.py::DeepEPBuffer` | Source of `_buffer.num_nvl_bytes + num_rdma_bytes` |
| `python/sglang/srt/distributed/parallel_state.py` | Source of NCCL comm singleton names |
| `docs/paras/cuda_graph.md` | Full ParaS CUDA graph doc with memory footprint tables |
| `docs/paras/memory_analysis.md` | ParaS overhead breakdown, applied optimizations, reclaimability analysis |

---

## Pitfalls

1. **Don't measure inside `/tmp`**. It gets wiped. Always save to `artifacts/` or `/home1`.
2. **Don't read `Capture cuda graph end mem usage` as graph pool size**. That log line reports sglang's own `driver_used` delta, which includes DeepEP / NVSHMEM / NCCL growth. Use `graph_pool` from my instrumentation.
3. **Don't sum `nccl_est` into attribution totals**. It's an upper-bound estimate; real usage may be half or less.
4. **`nvshmem` field is virtual-reservation, not physical commit**. If your question is "what will actually run out of GPU memory?", the physical NVSHMEM commit may be less than 1 GiB.
5. **Per-rank variance is real**. DP0/DP3 and DP1/DP2 typically differ by ~70 MB in `other` on 4-GPU NVLink. Don't treat this as noise — it's topology.
6. **Don't run two sglang servers simultaneously**. GPUs leak if you forget to kill one before starting the next. Check `nvidia-smi` between runs.
7. **Graph capture is sensitive to request history**. A server that has served a few requests will have different lazy-allocated cuBLAS workspaces than one that has been idle. Measure a clean state or measure after a fixed number of warmup requests.
