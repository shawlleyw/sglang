# ParaS transfer microbenchmarks

Compare BF16 direct peer-access kernels with NCCL staging/collectives for
EP↔TP weight and KV redistribution. These measure transfer operations;
server startup, request migration, graph capture, and graph activation are
outside the measurements.

For the real-runtime empty-state reconfiguration baselines, see
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

`bench_cache.py` supports a legacy isolated-layer mode and a resident mode
(`--resident-cache-gib`) with distinct storage for every uniform layer. Resident
mode defaults to overlapping EP/TP layouts; `--cache-layout separate` reproduces
the historical disjoint-buffer experiment. Neither mode migrates request metadata
or reproduces GPT-OSS's mix of full-attention and sliding-window live pages.

- `--cache-size-gb` sets the EP K+V capacity **for the one isolated layer** on
  each GPU. `--load` sets its resident fraction.
- EP→TP gathers N tokens per EP source into W·N token positions at each TP rank,
  including required replicated heads.
- TP→EP routes W·N/R tokens per source rank, where R is the head replication
  factor. Replicas handle disjoint token subsets. N is rounded down to a
  multiple of R; the old reverse-volume cap has been removed.
- Both directions read scattered live source slots and write fresh consecutive
  destination slots starting at 1 (slot 0 is padding). EP source mappings are
  rank-local; the TP source mapping is shared by all TP ranks. This matches the
  production switch's allocator reset and request remapping. TP→EP does not
  restore the old physical EP slots. Routing is built once, outside timing.
- NCCL includes destination packing and unpacking using the same routes as the
  direct kernels. After head reassembly, both directions use PyTorch `copy_`
  into compact destination slices. NCCL baselines use no custom pack/unpack
  kernels. All methods verify the destination, before and after timing.
- NCCL overlap uses double staging buffers and explicit reuse events. With
  overlapping storage, safe layer order protects unread sources, preparation
  reads the next source into independent staging, and destination commits remain
  ordered on the main stream. This is an isolated kernel benchmark.

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
After changing CUDA sources, rebuild the extension before running:

```bash
(cd python/sglang/srt/paras/csrc && python setup.py build_ext --inplace)
export PYTHONPATH="$PWD/python/sglang/srt/paras/csrc:$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
torchrun --standalone --nproc_per_node=8 benchmark/paras/check_kv_scatter.py
```

`check_kv_scatter.py` also accepts 2 or 4 processes. It checks complete source
and destination buffers against a CPU reference, including uneven/unsorted
routes, replicated and multiple local heads, empty inputs, padding skips, and
CUDA graph replay. The production v2 scatter kernel visits each routing entry
once, distributing disjoint token ranges across warps. Grouped destinations
improve peer concurrency, but sorted or equally sized routes are not required.

The production `MHACacheTransfer` backend and overlapping UMM layer views have
a separate regression test. Run each head count in fresh workers:

```bash
for heads in 8 4 16; do
  PARAS_TEST_KV_HEADS="$heads" torchrun --standalone --nproc_per_node=8 \
    -m pytest -q test/srt/paras/test_kv_scatter_single_pass.py
done
```

These cover one local head, replicated heads, and multiple local heads, with
uneven destination counts and scattered source slots. Live slots start at 1,
matching the production allocator; slot 0 remains padding.

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

## Expert-only and resident-cache kernel ablation

### Model and hardware compatibility

The default production `v2` kernels support both A100 (SM80) and H200 (SM90);
`python/sglang/srt/paras/csrc/setup.py` builds both architectures. Kernel launch
geometry uses the device's SM count. Rebuild the extension from the checkout on
each server; changing Python source does not replace an older installed binary.
`PARAS_KV_TRANSFER_METHOD=peer_access` selects the production path shared by MHA
and SWA. No serving import depends on the benchmark adapters.

| Model preset | BF16 KV shape | Expert gate layout | Reconfiguration config |
|---|---|---|---|
| `gpt-oss-120b` | 8 heads × 64 | Interleaved gate/up | `configs/gpt_oss_120b_a100.json` |
| `qwen3-235b` | 4 heads × 128 | Separate gate/up | `configs/qwen3_235b_h200.json` |
| `qwen3-30b` | 4 heads × 128 | Separate gate/up | Qwen config with `--model-path` pointing to 30B |

The JSON filenames describe the intended experiments, not a hardware dispatch
restriction. The reconfiguration worker reads actual checkpoint dimensions, and
its `--model-path` override can select either Qwen size. Use BF16 checkpoints.
Memory fractions and graph limits still need to fit each model/server pairing;
old GPT-OSS/A100 measurements do not validate H200 or Qwen end-to-end execution.
CPU reference tests cover both gate layouts, KV head dimensions and replication;
production CUDA compilation is checked for both SM80 and SM90. Hardware smoke
tests remain necessary before recording new results after the runtime rebase.

For the reference build, pass architecture `80` on A100 or `90` on H200:

```bash
bash benchmark/paras/build_nvbandwidth.sh /tmp/nvbandwidth-h200 90
python benchmark/paras/run_kernel_ablation.py --model qwen3-235b --dry-run \
  --output /tmp/qwen3-kernel-plan
```

### Model and GPU portability

The same kernel benchmark supports `gpt-oss-120b`, `qwen3-235b`, and
`qwen3-30b`. Dimensions, layer counts, GPT-OSS gate/up interleaving, and
TP KV-head replication are derived from the model preset. It uses patterned
BF16 tensors; no model checkpoint or DeepGEMM/UCCL serving backend is required.
The production peer-access extension builds SM80 and SM90 code for A100 and
H100/H200 respectively. Direct transfer requires working peer access between
all participating GPUs; use a single NVLink-connected node for comparable results.
Build the extension for the target environment before running.

Run this smoke command for each model on each target machine before collecting
results (use a fresh output directory each time):

```bash
python benchmark/paras/run_kernel_ablation.py --model gpt-oss-120b --smoke \
  --output /tmp/gptoss-kernel-smoke
python benchmark/paras/run_kernel_ablation.py --model qwen3-235b --smoke \
  --output /tmp/qwen3-kernel-smoke
```

The smoke run exercises both directions and all three transports with two
layers and 64 MiB of resident EP KV. CPU tests and `--dry-run` validate layout
and command construction; they do not establish execution on a GPU model.
Full Qwen3 measurements have been collected on H200; A100/H100 execution of
this revision still requires target-machine validation.

Choose cache volumes for the actual device capacity. In Qwen3 TP8, 30 GiB
resident EP KV needs about 60.32 GiB of arena before staging/runtime overhead;
40 GiB needs about 80.43 GiB and therefore cannot fit an 80-GiB GPU. Smaller
GPU variants need smaller volumes. GPT-OSS has a different replication factor
and footprint; its uniform-layer benchmark explicitly disables hybrid SWA.

The existing `bench_cache.py` and `run_kernel_ablation.py` entry points and
arguments remain supported; no replacement script is required. Legacy isolated-layer
mode is unchanged. For resident-cache runs, add `--cache-layout separate` to
reproduce the previous storage layout.

Resident mode defaults to `--cache-layout overlapping`. Each layer has disjoint
source/destination views, but the EP and TP layouts share storage across layers.
An EP-layer gap separates their starts. The TP stride is the larger of one EP
layer and one TP layer (including slot-zero padding). EP→TP visits layers in
reverse; TP→EP visits them forward, protecting every unread source. The default
Qwen TP8/R2 arena is approximately **2V + V/L**, where V is resident EP K+V
per GPU and L is the layer count, instead of 3V with separate buffers. Thus
Qwen3-235B's 40/50/60 GiB points need approximately 80.43/100.53/120.64 GiB
arenas, before NCCL staging and runtime overhead. These are cache-only runs;
no model weights are allocated. The arena is a compact benchmark layout,
not a claim about an entire serving process's memory footprint.

Patterned sources are regenerated in place before **every** warmup/measured
iteration, outside the timed region. No full-cache backup is retained.
The timed operation includes transfer, NCCL packing/unpacking and required
synchronization. All destination elements in every distinct layer are checked
against a separate Torch reference before and after timing. Remote bytes count
EP→TP replication (2V × 7/8 at Qwen TP8) and unique reverse traffic (V × 7/8).

`run_kernel_ablation.py` runs expert-only weights (w13+w2, excluding attention)
and distinct uniform KV layers. Use `--cache-gib 10 20 30 40 50 60` for the H200
Qwen sweep. `--cache-layout separate` remains available for historical comparison;
never mix the two storage layouts in one curve. GPT-OSS uses all 36 uniform
layers here, with SWA disabled; this is not a full model's hybrid-cache switch.
`--load` defaults to 1 in this mode; use a smaller fraction to model sparse EP
source slots. The TP pool holds W·N live tokens plus padding; its source slots
are a permutation of that resident span, without additional capacity slack.
Actual resident bytes are rounded down to whole tokens/head-replica groups and
recorded in the CSV. Attention weights are not transferred. Expert weights retain
the production UMM placement but allocate no serving KV footprint in their run.

```bash
# Serving environment activated; build the reference without installing anything system-wide.
bash benchmark/paras/build_nvbandwidth.sh /tmp/nvbandwidth-reference 80

# CPU-only command/provenance preparation:
python benchmark/paras/run_kernel_ablation.py --dry-run \
  --output /tmp/kernel-ablation-plan \
  --nvbandwidth /tmp/nvbandwidth-reference/build/nvbandwidth

# GPU correctness smoke (all three methods, both directions):
python benchmark/paras/run_kernel_ablation.py --smoke --warmup 1 --iters 2 \
  --output results/kernel-smoke

# Full sweep, with SM-only nvbandwidth references (no copy-engine tests):
python benchmark/paras/run_kernel_ablation.py \
  --output results/kernel-ablation \
  --nvbandwidth /tmp/nvbandwidth-reference/build/nvbandwidth
```

The reference sweeps 256/512/1024 MiB per peer, five samples using the mean,
for `device_to_device_memcpy_write_sm`, `one_to_all_write_sm`, and
`all_to_one_write_sm`. All ParaS expert and KV kernels tested here use remote
writes. The one-to-all and all-to-one results measure outbound and inbound
bandwidth limits separately; they do not reproduce simultaneous all-to-all
traffic and are not a proof of a mathematical optimum. Report efficiency against
the **measured SM-copy reference**, including the selected denominator and its
traffic-pattern limitation. Count remote payload once, excluding self transfers;
do not sum send and receive bytes or compare against a duplex bandwidth number.

`run_kernel_ablation.py` records each command, status, source hashes/archive, Git
revision/diff, and GPU topology. Data initialization and destination verification
are outside timing. CUDA events include staging, collectives, and layer fences;
statistics use the slowest rank each iteration. Checksum data includes layer
identity, so a transfer accidentally reusing the wrong layer can be detected.

Cache logs/CSV record the seed and `slot_policy`:
`scattered_source_compact_destination_v1`. Earlier results used compact TP
sources and randomly indexed EP destinations in TP→EP; they describe a different
workload. Use a fresh CSV for corrected runs and rerun all three cache methods
together. The analyzer labels old data `legacy_random_ep_destination` and
rejects mixed slot policies; existing measurements are not silently relabeled.

### Archived kernel benchmark

The [previous kernel benchmark](legacy/kernel_benchmark/README.md) is retained
with its own measurement scripts, helpers, reproduction commands, and provenance.
Use the current entry points above for new GPT-OSS/Qwen3 measurements. Current
run archives exclude `legacy/` to avoid duplicating historical sources.

## Single-GPU VMM activation

`bench_runtime_vmm.py` measures the production runtime VMM allocator with no
checkpoint or distributed process group. Expose exactly one idle GPU. Defaults
match the GPT-OSS-120B experiment's EP/TP logits and KV-index shapes (256/2048
tokens, context 131072, vocabulary 201088); `--ep-max-tokens`, `--tp-max-tokens`,
`--context-length` and `--vocab-size` allow other settings.

```bash
# CPU-only provenance/shape preparation; requires a fresh output directory.
python benchmark/paras/bench_runtime_vmm.py --dry-run --output /tmp/vmm-plan

# Select an idle GPU UUID first; only this device is exposed to the process.
CUDA_VISIBLE_DEVICES=GPU_UUID python benchmark/paras/bench_runtime_vmm.py \
  --output results/vmm-idle --rounds 3 --iterations 20

# Same scratch sizes under synthetic HBM pressure (not real serving KV).
CUDA_VISIBLE_DEVICES=GPU_UUID python benchmark/paras/bench_runtime_vmm.py \
  --output results/vmm-resident60 --resident-gib 60 --rounds 3 --iterations 20
```

Three variants isolate the changes: `zero_two_sync` recreates the original
activation policy, `no_zero_two_sync` removes clearing only, and
`no_zero_one_sync` uses the optimized production policy for disposable buffers.
All variants release inactive physical backing and preserve virtual addresses;
all wait for outgoing users before unmapping. Initial allocation always clears
storage. Each timed sample is one activation, and both directions are reported.
Variant order rotates across rounds. A separate diagnostic pass reports host
time in each driver call and synchronization, without adding internal fences;
these timings can include waiting, not just execution of the named operation.
Use `--interval-ms 100` to leave an idle interval before each activation, outside
timing. The default tight loop stresses repeated physical allocation; driver
allocation latency under this churn can differ substantially from occasional
switches. Report the interval alongside memory residency and sample statistics.

Before timing, the optimized path captures both modes once and checks repeated
remapping/replay with invalid old indices and NaN logits. The checks use the
production KV-index builder and Triton decode attention, compare valid output
rows with a PyTorch reference, and exercise changing lengths, padding, tails,
empty metadata and a different replay stream. They validate scratch consumers,
not full model inference or multi-GPU scheduler behavior.

Outputs include exact command/configuration, source snapshots/hashes/patch,
GPU identity, all raw samples, correctness records, API profiles and a Markdown
table. Initial allocation, correctness checks, resident ballast and diagnostic
profiling are outside headline timing. This ablation does not replace the
eight-GPU full-model switch results or establish their end-to-end speedup.
