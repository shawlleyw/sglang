# ParaS Peer-Access Kernel Benchmarks

Microbench suite for the production ParaS KV cache and MoE weight transfer
kernels in [`peer_access_transfer.cu`](file:///home/shaoyuw/sglang/python/sglang/srt/paras/csrc/peer_access_transfer.cu).
Compares the **CUDA peer-access kernel** against the **NCCL all_to_all** and
**NCCL all_to_all with 2-stream overlap** baselines, on both EP↔TP directions,
for both cache and weights, across a configurable per-GPU volume.

## Coverage

| Surface | Direction | Kernel | Methods |
|---|---|---|---|
| KV cache | TP → EP (scatter) | `peer_access_kv_scatter` | peer_access \| nccl \| nccl_overlap |
| KV cache | EP → TP (gather) | `peer_access_kv_transfer` | peer_access \| nccl \| nccl_overlap |
| MoE w13 | EP → TP | `peer_access_fused_transfer_w13_v2` | peer_access \| nccl \| nccl_overlap |
| MoE w13 | TP → EP | `peer_access_fused_transfer_w13_ep` | peer_access \| nccl \| nccl_overlap |
| MoE w2 | EP → TP | `peer_access_fused_transfer_w2_v2` | peer_access \| nccl \| nccl_overlap |
| MoE w2 | TP → EP | `peer_access_fused_transfer_w2_ep` | peer_access \| nccl \| nccl_overlap |

## Methods

- **`peer_access`** — production fused CUDA kernels with per-layer `dist.all_reduce` barrier (matches `paras_configure_tp_peer_access` / `peer_access_kv_scatter` per-layer pattern).
- **`nccl`** — `dist.all_to_all_single` per layer, single stream (mirrors `paras_configure_tp_mlp_naive` / `do_scatter_one_layer_nccl`).
- **`nccl_overlap`** — same `dist.all_to_all_single` pattern, pipelined across **two CUDA streams** so layer N's all_to_all overlaps with layer N+1's launch overhead (mirrors `paras_configure_tp_overlap`).

## Volume control

**Cache:**

```
--cache-size-gb N   # per-GPU EP buffer capacity (GiB)
--load f            # resident fraction in (0, 1]
```

Resident KV data per GPU = `cache_size_gb × load`. For example, `--cache-size-gb 20 --load 0.5` gives ~10 GiB resident KV per GPU. Both TP→EP and EP→TP move the same `num_resident_tokens` per source rank, so the two directions are directly comparable.

**Weights:** volume is fixed by the model preset (`num_experts`, `hidden_size`, `moe_intermediate_size`).

**Multi-layer timing:** every method's timed iteration runs **`num_hidden_layers`** back-to-back transfers (= one full EP↔TP switch). This is necessary for `nccl_overlap` to expose any pipelining benefit. Output reports both total time per iteration and per-layer mean (total / num_layers).

## Default model presets

Values match upstream Hugging Face `config.json`. All bf16, `num_gates=2`.

| Preset | num_hidden_layers | num_kv_heads | head_dim | num_experts | hidden_size | moe_intermediate_size |
|---|---:|---:|---:|---:|---:|---:|
| `qwen3-30b` (Qwen3-30B-A3B) | 48 | 4 | 128 | 128 | 2048 | 768 |
| `qwen3-235b` (Qwen3-235B-A22B) | 94 | 4 | 128 | 128 | 4096 | 1536 |

Use `--model custom --num-kv-heads ... --head-dim ... --num-experts ... --hidden-size ... --moe-intermediate-size ... --num-hidden-layers ...` to override.

## Quick start

```bash
source /home/shaoyuw/miniconda3/etc/profile.d/conda.sh && conda activate sgl_paras
export LD_LIBRARY_PATH=/home/shaoyuw/miniconda3/envs/sgl_paras/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH
cd /home/shaoyuw/sglang/benchmark/paras

# Cache: both directions, 8 GPUs, 10 GiB cache @ 0.5 load, peer-access kernel
torchrun --nproc_per_node=8 bench_cache.py \
    --model qwen3-235b --tp-size 8 \
    --cache-size-gb 10 --load 0.5 \
    --direction both --method peer_access \
    --warmup 3 --iters 10

# Same workload via NCCL baseline
torchrun --nproc_per_node=8 bench_cache.py \
    --model qwen3-235b --tp-size 8 \
    --cache-size-gb 10 --load 0.5 \
    --direction both --method nccl \
    --warmup 3 --iters 10

# Same workload via NCCL 2-stream overlap baseline
torchrun --nproc_per_node=8 bench_cache.py \
    --model qwen3-235b --tp-size 8 \
    --cache-size-gb 10 --load 0.5 \
    --direction both --method nccl_overlap \
    --warmup 3 --iters 10

# Weights: all 4 kernels (w13/w2 × EP↔TP), default qwen3-235b
torchrun --nproc_per_node=8 bench_weights.py \
    --model qwen3-235b --tp-size 8 \
    --kernel both --direction both --method peer_access \
    --warmup 3 --iters 10

# Full sweep wrapper across both models, all 3 methods, several cache configs
NUM_GPUS=8 bash run_all.sh
```

## CLI reference

### `bench_cache.py`

| Flag | Default | Purpose |
|---|---|---|
| `--model` | `qwen3-235b` | `qwen3-30b` \| `qwen3-235b` \| `custom` |
| `--num-kv-heads`, `--head-dim`, `--num-hidden-layers` | — | Override preset (required if `custom`) |
| `--tp-size` | `8` | Must equal `torchrun --nproc_per_node` |
| `--cache-size-gb` | `10.0` | Per-GPU EP capacity (GiB) |
| `--load` | `0.5` | Resident fraction in `(0, 1]` |
| `--direction` | `both` | `tp_to_ep` (scatter) \| `ep_to_tp` (gather) \| `both` |
| `--method` | `peer_access` | `peer_access` \| `nccl` \| `nccl_overlap` |
| `--warmup` / `--iters` | `3` / `10` | Per-iteration timing (one iter = `num_hidden_layers` transfers) |
| `--out-csv` | — | Append rows to this CSV |

### `bench_weights.py`

| Flag | Default | Purpose |
|---|---|---|
| `--model` | `qwen3-235b` | Preset |
| `--num-experts`, `--hidden-size`, `--moe-intermediate-size`, `--num-hidden-layers` | — | Override preset |
| `--tp-size` | `8` | torchrun nproc |
| `--kernel` | `both` | `w13` \| `w2` \| `both` |
| `--direction` | `both` | `ep_to_tp` \| `tp_to_ep` \| `both` |
| `--method` | `peer_access` | `peer_access` \| `nccl` \| `nccl_overlap` |
| `--warmup` / `--iters` | `3` / `10` | |
| `--out-csv` | — | Append rows to this CSV |

## Slot-based cache init

To mirror production (active KV tokens sit at arbitrary slot positions in a fixed pool, not at slots `0..N`):

1. Allocate `ep_max_tokens = cache_size_gb × 1024³ / bytes_per_ep_slot` slots per GPU.
2. Each rank seeds `torch.Generator` with `seed + rank`, draws `num_resident = ep_max_tokens × load` distinct slot indices from `[1, ep_max_tokens)` (slot 0 reserved as kernel-side padding).
3. Fills those slots with a bf16-EXACT pattern `v = rank*16 + (slot % 16) + kv_offset` (K uses 0, V uses 128; values in `[0, 256)` are bf16-exact step-1).

The `peer_access` method additionally verifies bulk-GPU equality of the destination against ground truth before timing. `nccl` and `nccl_overlap` are timing-only (the production NCCL paths are upstream-verified).

## Output schema

Both bench scripts print one summary line per direction/kernel and (with `--out-csv`) append CSV rows with columns:

```
timestamp, model, num_layers, tp_size, ...,
direction, method,
total_mean_ms,        # mean across timed iterations (one iter = N layers)
total_p50_ms,
per_layer_mean_ms,    # total_mean_ms / num_layers
per_layer_p50_ms,
min_ms, max_ms, n
```

Use `per_layer_p50_ms` as the headline number; total time is reported so you can compare a full per-switch cost across methods.

## Files

```
benchmark/paras/
├── README.md
├── run_all.sh                       # full sweep wrapper
├── bench_cache.py                   # KV scatter + gather
├── bench_weights.py                 # w13/w2 × EP↔TP
├── common/
│   ├── model_configs.py             # qwen3-30b, qwen3-235b, custom
│   ├── ipc.py                       # IPC arena + CudaTimer
│   ├── layouts.py                   # KVLayout / WeightLayout derivation
│   └── slot_init.py                 # random-slot resident init
└── results/                         # CSV output (timestamped subdirs)
```

## Notes

- Default models: **Qwen3-30B-A3B** and **Qwen3-235B-A22B**.
- The benchmark depends on the production `paras_peer_access_cuda` extension; build it once via `cd python/sglang/srt/paras/csrc && python setup.py build_ext --inplace`.
- Outputs include `mean / p20 / p50 / p90 / min / max`. Report `p50` not `mean` when variance is bimodal; consider Mann-Whitney U for A/B significance.
- This benchmark is **not committed** by default — files sit ready in `benchmark/paras/`.
