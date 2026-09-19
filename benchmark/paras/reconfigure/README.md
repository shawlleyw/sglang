# Empty-state reconfiguration benchmarks

`../bench_reconfigure.py` runs five reusable methods against the real SGLang
runtime. **V1 is archived at `artifacts/paras_switch_v1` and is not an accepted
production-fidelity evaluation.** Its functional probes passed, but its sparse
graph coverage, forced disposal GC and other confounds invalidate using those
timings to claim production switching costs. See the
[production-fidelity audit](production_fidelity_audit.md).

The current v2 candidate removes those graph/GC overrides and several launch
and synchronization mismatches. It requires fresh GPU validation and has no v2
performance results. No GPU workload is launched by importing the driver or
using `--dry-run`.

## Methods and boundaries

| CLI method | Measured transition |
|---|---|
| `rebuild` | Shut down a source SGLang Engine, create a target Engine, load checkpoint weights and capture graphs |
| `host_reload` | Reload prepared target CPU weight snapshots into fresh GPU tensors, bind them and capture target graphs |
| `naive_nccl` | Allocate each target layer, pack/exchange/unpack experts and attention using NCCL, release the source layer, bind and capture graphs |
| `fixed_buffer_recapture` | Production UMM **and fused peer-access transfer**, with graph discard and target recapture; this is not a fixed-buffer NCCL baseline |
| `full` | Production Scheduler ParaS switch using its prepared UMM views and retained graphs |

Every trial has an empty running/waiting request set. There is no live KV cache,
request draining, or HTTP latency in these measurements. Methods other than
`rebuild` instantiate real Scheduler/ModelRunner workers directly; they exclude
TokenizerManager/DataParallelController messaging. `rebuild` recreates the
complete Python Engine, including its controller/tokenizer infrastructure, but
not an HTTP web server. CSV/JSON consumers must preserve these scope labels.

The driver records both `switch_ms` (transition ready, including required graph
capture) and `through_probe_ms` (through the first inference probe). The probe
includes prompt prefill and the configured number of graph-based decode steps;
it is not pure first-token latency. Non-rebuild workers compare full probe logits
against an untimed reference in the **same target mode**, with configured BF16
tolerances and finite-value checks. The reference is captured before preparing
the source mode. Rebuild runs repeat the target inference probe and require equal
text; its second validation probe is outside the recorded time.

Worker timing uses CPU monotonic clocks with CUDA synchronization. The reported
switch and through-probe totals are the maximum rank durations. Raw per-rank
phase measurements remain available; do not add independently maximized phase
values and call the sum the measured critical path. Source initialization,
reference generation, host snapshot preparation, and warmup are untimed. There
is one measured switch per fresh worker group. CUDA/NCCL JIT compilation and
startup still occur during preparation; they are included for a fresh rebuild
if the target initialization incurs them.

Checkpoint files are read normally. The benchmark does not clear OS file caches:
initializing the source can warm the target checkpoint reads. Label row 1 as
engine rebuild, not guaranteed cold-disk startup.

## Independent weights for host reload and naive NCCL

The benchmark-only `IndependentWeightMemoryManager` uses the production planner
for endpoint shapes and cache capacities, but **does not allocate the UMM weight
arena**. EP weights initially have independent allocations. Inactive weights are
zero-stride placeholders, replaced before execution. Each completed layer's old
Parameters and storage-manager entries are rebound to placeholders, so the source
allocation can actually be reclaimed. Ordinary PyTorch allocator caching remains
enabled; there is no per-layer `empty_cache()`.

Empty EP/TP KV views alias separate per-layer K/V buffers, and runtime scratch is
retained. This is an empty-state benchmark backend, not a live KV migration
implementation. Stable common runtime state is retained in both baselines.
Attention QKV/O and expert w13/w2 weights get fresh destination allocations.
Common auxiliary parameters (norms, embeddings, biases, sinks registered as
parameters, etc.) remain at stable addresses; host reload also copies their CPU
snapshots back in place. CPU target snapshots are prepared and pinned before the
measured transition by default. CPU snapshot bytes are reported per rank.

Naive NCCL relies on PyTorch stream/lifetime tracking, with the production-style
per-layer GPU fence and a final device synchronization. It adds no per-layer
host wait. Host reload needs no cross-rank fence for its H2D copies.
Both directions include attention transfers:
EP→TP takes local QKV/O slices; TP→EP gathers and reconstructs full projections,
discarding duplicate KV-head replicas. GPT-OSS gate/up interleaving is supported.
All runtime patches exist only in the fresh benchmark workers, with no production
server flags, global sitecustomize changes, or modifications to serving code.

## Configuration and dry runs

Use the same serving environment as the model launchers. The configs specify
BF16 weights, eight GPUs, Triton backends, disabled prefix caching and the
production default of enabled scheduling overlap. Graph maxima follow the
launcher's request-capacity policy: with 2048 requests and eight GPUs, EP's
maximum is 256 and TP's is 2048. `ServerArgs` generates the lists; production
runner initialization applies its filtering and prepares per-mode buffers.
Recapture uses those resolved per-mode lists rather than regenerating a sparse
list. Current non-speculative defaults generate 36 EP and 100 TP sizes before
any runtime capacity filtering. Exact lists are recorded in worker ready events
and target sizes in results. Explicit maximum overrides remain supported via
`server_args.cuda_graph_max_bs` and `server_args.paras_tp_cuda_graph_max_bs`;
handwritten size lists are rejected. Runtime graph disposal only drops
references and the pool handle. SGLang's own capture GC policy is unchanged.

Full/fixed methods use the launcher's peer-access KV path. Independent-storage
methods explicitly use NCCL for empty-cache bookkeeping because their backend
does not own a real UMM IPC arena; the worker records that difference. Static
TP rebuild enables token-ID synchronization and clears stale DeepEP variables,
matching the launcher. Remaining scope/configuration caveats are in the audit.
Supported scope is a single node, PP=1, ordinary Qwen3-MoE/GPT-OSS decoding without
quantization, speculation, LoRA, memory-saver, or torch.compile.

From the repository root, validation without GPU access:

```bash
python benchmark/paras/bench_reconfigure.py \
  --config benchmark/paras/configs/gpt_oss_120b_a100.json \
  --output /tmp/gptoss-reconfigure-plan --dry-run
```

The output directory must be new. A dry run validates configuration fields and
records provenance; it does not initialize SGLang or prove that the model fits.

When the GPUs are available, start with one trial of the production method:

```bash
python benchmark/paras/bench_reconfigure.py \
  --config benchmark/paras/configs/gpt_oss_120b_a100.json \
  --methods full --direction both --repetitions 1 \
  --output results/gptoss-full-smoke
```

Then exercise all methods once before collecting repeated measurements:

```bash
python benchmark/paras/bench_reconfigure.py \
  --config benchmark/paras/configs/gpt_oss_120b_a100.json \
  --repetitions 1 --output results/gptoss-all-smoke

python benchmark/paras/bench_reconfigure.py \
  --config benchmark/paras/configs/gpt_oss_120b_a100.json \
  --repetitions 5 --output results/gptoss-reconfigure
```

Qwen3 uses the same implementation:

```bash
python benchmark/paras/bench_reconfigure.py \
  --config benchmark/paras/configs/qwen3_235b_h200.json \
  --model-path /models/Qwen3-235B-A22B-Instruct-2507 \
  --repetitions 5 --output results/qwen3-reconfigure
```

For a cheaper graph-capture smoke test, make a config copy with smaller graph
maxima, still using SGLang's generated lists. Such a run must not be pooled with
the full production-coverage results.
The driver defaults to a 30-minute timeout per worker stage; `--timeout` overrides
it. Failures stop the sweep, retain logs, and create failed trial records. Cleanup
is restricted to process groups created by this driver.

## Output and tests

- `manifest.json`: resolved experiment, versions, repository revision/branch,
  working-tree status, source hashes, and GPU inventory/topology for actual runs.
- `benchmark_source.tar.gz` and `tracked_changes.patch`: exact benchmark sources
  including uncommitted/new files, plus all tracked working-tree changes against
  the recorded revision (including serving/runtime changes outside the benchmark).
- `resolved_config.json`: configuration passed to every worker group.
- Per-trial commands, normalized `server_args.json`, supervisor and rank logs.
- `trials.jsonl`: successful and failed trials, per-rank timings and peak PyTorch
  allocated/reserved memory. These peaks exclude non-PyTorch allocations.
- `summary.json`: median/range by method and direction; failed trials are excluded.

CPU reference and orchestration tests, explicitly hiding all GPUs:

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s benchmark/paras/tests -v
```

These tests cover complete-tensor simulated NCCL routing, Qwen/GPT gate layouts,
replicated KV heads, host snapshot independence, allocation/alias lifetime,
configuration parity, graph operation ordering, and worker error propagation.
