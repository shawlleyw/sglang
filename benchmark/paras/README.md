# ParaS transfer microbenchmarks

Compare BF16 direct peer-access kernels with NCCL staging/collectives for
EP↔TP weight and KV redistribution. These measure transfer operations;
server startup, request migration, graph capture, and graph activation are
outside the measurements.

For the five real-runtime empty-state reconfiguration baselines, see
[reconfigure/README.md](reconfigure/README.md) and `bench_reconfigure.py`.

## Weight bundles with the current UMM

`bench_weights.py` uses the production `plan_unified_layout`,
`ParaSMemoryManager` views, expert CUDA kernels, and attention reconstruction
kernel. Each model layer has distinct planned EP/TP views in one allocation.
EP→TP visits layers in order; TP→EP visits them in reverse. Every method fences
after the complete layer bundle before overlapping source bytes can be reused.
The benchmark restores the source mode outside each timed interval.

The memory plan reserves **zero inference workspace and minimal unused KV**.
This preserves the UMM's asymmetric weight placement and overwrite rules,
without constructing attention/MoE backends or claiming a serving memory
footprint. The benchmark allocates all selected model layers, including their
attention weights. Reduce `--num-hidden-layers` for a smoke test; such a run is
not a full-model timing. KV contents are not included in weight measurements.

| Selection (`--kernel`) | Measured work |
|---|---|
| `bundle` (default) | Expert w13 + w2 + attention QKV/O, one fence per layer |
| `attention` | QKV/O transfer, one fence per layer |
| `w13`, `w2` | One expert component, one fence per layer |
| `both` | Separate w13 and w2 measurements (legacy CLI compatibility) |
| `all` | w13, w2, attention, and combined bundle measurements |

Component totals are not additive: the combined bundle shares its layer fence.

| Method | Expert redistribution | Attention redistribution |
|---|---|---|
| `peer_access` | Production v2 kernels, or optional v3 | Local slices EP→TP; production Triton peer reads TP→EP |
| `nccl` | EP→TP pack + all-to-all into target; TP→EP all-to-all + unpack | Same local slices EP→TP; rank-major all-gather + unpack TP→EP |
| `nccl_overlap` | Independent within-layer packing stream overlaps w2 packing with w13 collective | Same as NCCL |

`nccl_overlap` deliberately keeps the UMM layer fence. It does **not** reproduce
the old model-level cross-layer overlap path. Its overlap is useful for the
EP→TP bundle; a single expert has no second pack to overlap, and TP→EP currently
uses the sequential NCCL schedule. CSV `overlap_scope` records this distinction.
Each expert has its own staging, avoiding cross-stream buffer reuse races.

Qwen uses separate gate/up halves. GPT-OSS uses interleaved gate/up rows, so the
w13 direct launch uses one gate and a doubled chunk extent. Attention handles
replicated K/V heads: TP→EP reconstructs each K/V head from its representative
rank. NCCL all-gather also communicates duplicate K/V replicas, then discards
them when unpacking; that additional work is included in its measured time.
Biases and attention sinks remain replicated outside the production UMM and
are not transferred.

All methods check coordinate-sensitive samples of every destination layer and
component, then check the round trip before timing. Full small-tensor CPU tests
check the NCCL layout transformations, gate ordering, actual overlapping UMM
views, and reconstruction with poisoned duplicate attention K/V replicas.
GPT-OSS-120B full bundles were validated on eight A100 GPUs on 2026-09-19;
see `artifacts/20260919T070046Z_gptoss_120b_switch_eval` at the repository root.
Other model/hardware combinations still require their own GPU smoke test below.

## KV kernels

`bench_cache.py` is a separate **isolated homogeneous layer** harness. It repeats
one layer's buffers `num_hidden_layers` times to measure launch/transfer costs;
it does not allocate a model's UMM layout, migrate request metadata, or reproduce
GPT-OSS's mix of full-attention and sliding-window live pages.

- `--cache-size-gb` sets the EP K+V capacity **for the one isolated layer** on
  each GPU. `--load` sets its resident fraction.
- EP→TP gathers N tokens per EP source into W·N token positions at each TP rank,
  including required replicated heads.
- TP→EP routes W·N/R tokens per source rank, where R is the head replication
  factor. Replicas handle disjoint token subsets. N is rounded down to a
  multiple of R; the old reverse-volume cap has been removed.
- NCCL includes destination packing and unpacking using the same routes as the
  direct kernels. All methods verify the destination, before and after timing.
- NCCL overlap uses double staging buffers and explicit reuse events in this
  disjoint-buffer harness. It is not a demonstrated cross-layer UMM schedule.

For GPT-OSS, run distinct volumes representing full-attention and sliding-window
resident pages and label them as component measurements. A single cache run
must not be described as the model's complete KV migration cost.

## Presets

All weights are BF16; GPT-OSS checkpoint quantization is not benchmarked.

| Preset | Layers | Q heads | KV heads | Head dim | Experts | Hidden | MoE intermediate | w13 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `qwen3-30b` | 48 | 32 | 4 | 128 | 128 | 2048 | 768 | Separate gates |
| `qwen3-235b` | 94 | 64 | 4 | 128 | 128 | 4096 | 1536 | Separate gates |
| `gpt-oss-120b` | 36 | 64 | 8 | 64 | 128 | 2880 | 2880 | Interleaved gates |

Dimension flags override presets. For custom weight runs, supply
`--model custom --num-attention-heads ... --num-kv-heads ... --head-dim ...
--num-experts ... --hidden-size ... --moe-intermediate-size ...
--num-hidden-layers ...`, optionally `--interleaved-w13`.

## Running

Use an activated SGLang environment with the production `paras_peer_access_cuda`
extension installed, on one NVLink server. No model checkpoint is loaded.
Run a small GPU correctness/timing check first:

```bash
cd benchmark/paras
for method in peer_access nccl nccl_overlap; do
  torchrun --standalone --nproc_per_node=8 bench_weights.py \
    --model gpt-oss-120b --tp-size 8 --num-hidden-layers 2 \
    --kernel all --direction both --method "$method" --warmup 1 --iters 2
done
```

Then run the full weight comparison on each corresponding server:

```bash
# Qwen3-235B on H200
MODELS=qwen3-235b COMPONENT=weights NUM_GPUS=8 bash run_all.sh
# GPT-OSS-120B on A100
MODELS=gpt-oss-120b COMPONENT=weights NUM_GPUS=8 bash run_all.sh
```

For selected kernel components or KV volumes:

```bash
torchrun --standalone --nproc_per_node=8 bench_weights.py \
  --model qwen3-235b --tp-size 8 --kernel all --method nccl \
  --direction both --warmup 3 --iters 10 --out-csv results/weights-nccl.csv

torchrun --standalone --nproc_per_node=8 bench_cache.py \
  --model qwen3-235b --tp-size 8 --cache-size-gb 1 --load 0.5 \
  --direction both --method peer_access --variant v2 \
  --warmup 3 --iters 10 --out-csv results/cache.csv
```

Use `--variant v3` for explicit optional kernel ablations; v2 is the runtime
expert-kernel default. The sweep wrapper defaults to v2 and stops on failures.
The current v3 expert extension only specializes Qwen shapes
`(H,I)=(4096,1536)` and `(2048,768)` at TP=4/8 with separate gates. GPT-OSS
expert transfers must use v2; requesting v3 fails before GPU initialization.
`--help` documents the full CLI.

## Timing and reporting

CUDA events enclose transfers, staging copies, and the appropriate fences.
All ranks contribute; statistics use the **maximum rank time per iteration**,
then discard warmup and compute mean/median. Weight restoration is excluded.
The weight total measures the full selected set of distinct layers; the KV
total is a repeated-layer microbenchmark. Neither is end-to-end switch latency.

CSV rows record model, layer count, direction, method, variant, total and
per-layer timing. Weight rows also include UMM arena size, explicit NCCL staging
allocation, gate layout, and overlap scope. KV rows include scope, actual remote
bytes per rank per layer, and staging bytes. Staging sizes are **not peak GPU
memory**: CUDA/NCCL state and temporary PyTorch packing allocations are excluded.
Use a fresh output file if its schema differs from an earlier benchmark version.

CPU reference checks (no GPU needed):

```bash
python -m unittest discover -s benchmark/paras/tests -v
python -m pytest -q benchmark/paras/tests
```
