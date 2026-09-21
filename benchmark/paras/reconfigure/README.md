# Empty-state reconfiguration benchmarks

`../bench_reconfigure.py` runs six reusable methods against the real SGLang
runtime. **V1 is archived at `artifacts/paras_switch_v1` and is not an accepted
production-fidelity evaluation.** Its functional probes passed, but its sparse
graph coverage, forced disposal GC and other confounds invalidate using those
timings to claim production switching costs. See the
[production-fidelity audit](production_fidelity_audit.md).

The current v2 implementation removes those graph/GC overrides and several
launch and synchronization mismatches. On 2026-09-19, all four switching methods
passed both directions on eight A100-80GB GPUs with GPT-OSS-120B BF16, full graph
coverage and memory fraction 0.70. All 64 rank results matched reference logits
exactly. The initial 0.75 attempt ran out of memory during untimed preparation;
the reduced common memory budget is explicit in the saved configuration.
See the [v2 smoke report](../../../artifacts/20260919T090553Z_gptoss_120b_switch_v2/README.md).
This is one trial per method/direction; restart rows are copied v1 references,
and Qwen still requires GPU validation. No GPU workload is launched by importing
the driver or using `--dry-run`.

On 2026-09-20, `host_model_to` and a fresh `host_reload` control passed both
directions on the same eight A100s at memory fraction 0.70. All 32 rank results
matched reference logits exactly; both methods captured the full graph sets.
Their weight phases were nearly identical in these single-trial measurements.
See the [host-method comparison](../../../artifacts/20260920T033755Z_gptoss_host_model_to/README.md)
for the six-method table, origin labels, memory diagnostics and timing breakdown.

After rebasing onto `paras_a100`'s runtime-state isolation, recapture allocates
the target mode's graph inputs and attention metadata through the production
`init_graph_buffers()` / `init_cuda_graph_state()` methods. EP's smaller buffers
are no longer reused for TP capture. Disposal drops saved backend metadata as
well as graph references, without explicit GC. These changes have CPU coverage
using the production allocation methods; the historical GPU measurements above
predate this rebase and do not validate its memory fit or capture latency.

## Methods and boundaries

| CLI method | Measured transition |
|---|---|
| `rebuild` | Shut down a source SGLang Engine, create a target Engine, load checkpoint weights and capture graphs |
| `host_reload` | Reload prepared target CPU weight snapshots into fresh GPU tensors, bind them and capture target graphs |
| `host_model_to` | Release all source bulk weights, move the prepared target weight module with one `Module.to(device, non_blocking=True)` call, bind and capture graphs |
| `naive_nccl` | Allocate each target layer, pack/exchange/unpack experts and attention using NCCL, release the source layer, bind and capture graphs |
| `fixed_buffer_recapture` | Production UMM **and fused peer-access transfer**, with graph discard and target recapture; this is not a fixed-buffer NCCL baseline |
| `full` | Production Scheduler ParaS switch using its prepared UMM views and retained graphs |

Every trial has an empty running/waiting request set. **The full planned KV
storage is allocated**, including GPT-OSS full-attention/SWA pools. Empty means
no resident request tokens, not an absent or minimal GPU cache allocation.
There is no live KV cache migration,
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
All methods use `world_size` global requests: one per EP worker or a batch of
`world_size` in TP. Engine scheduling can distribute them differently; matching
request counts does not make the manual probe an Engine-level endpoint.

Worker timing uses CPU monotonic clocks with CUDA synchronization at the total
measurement boundaries. Phase timers introduce no internal CUDA synchronization:
they report host wall time, including production waits, not isolated GPU duration.
For example, an asynchronous H2D enqueue phase is not its completed copy time.
The reported
switch and through-probe totals are the maximum rank durations. Raw per-rank
phase measurements remain available; do not add independently maximized phase
values and call the sum the measured critical path. Source initialization,
reference generation, host snapshot preparation, and warmup are untimed. There
is one measured switch per fresh worker group. CUDA/NCCL JIT compilation and
startup still occur during preparation; they are included for a fresh rebuild
if the target initialization incurs them.

`warmup_switches` defaults to one untimed execution of the actual measured
transition, including host H2D or target recapture, followed by source restoration
and a probe. Set it to zero to omit this extra warmup, but the initial reference
and mode preparation still occur: zero does not mean a cold first-ever switch.
Host source restoration uses NCCL because only the target has a CPU snapshot.
Rebuild remains a fresh Engine initialization and does not use switch warmup.

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
snapshots back in place. Only active target auxiliaries are copied: saved EP/TP
representations are excluded and identical tensor views are deduplicated.
CPU target snapshots are prepared and pinned before the
measured transition by default. CPU snapshot bytes are reported per rank.

Each fresh trial keeps only its destination-mode CPU snapshot: TP for EP→TP,
or EP for TP→EP. Preparing that snapshot, including the D2H copy, is untimed;
the measured switch does not offload the old GPU weights. A persistent system
using prepared host reload in both directions would need both representations
or would have to prepare the destination on demand.

`host_reload` allocates and enqueues copies for one layer's destination weights
before releasing that layer's source storage. EP→TP visits layers in forward
order; TP→EP visits them in reverse. Source parameter storage and manager entries
are replaced with expanded one-element placeholders. Once references and stream
dependencies allow, the old storage becomes reusable by PyTorch's allocator;
it need not return to the CUDA driver or reduce `nvidia-smi` usage. KV storage
and runtime scratch remain allocated. Source-first release is a possible future
host-reload optimization, not the ordering used by `host_reload`.

The explicit `empty()` + asynchronous `copy_()` loop controls destination
selection and storage rebinding. `Module.to()` also visits parameters and buffers
individually; it does not combine model weights into one transfer or update our
custom memory-manager references. Replacing the loop with that API is therefore
not inherently a transfer optimization. The current implementation uses no
per-tensor host synchronization during reload.

`host_model_to` exercises PyTorch's actual `nn.Module.to` implementation once
on a temporary module containing every destination expert and attention
parameter. It releases all source bulk weights first, avoiding simultaneous
full source and destination copies. CPU snapshots remain intact for warmup
and measurement. Module construction, source release, conversion, runtime
rebinding and transfer completion are timed. The existing runtime Parameter
objects are preserved, along with their custom attributes and aliases.
Auxiliary parameters use the same in-place asynchronous H2D reload as
`host_reload`; KV, registered runtime buffers and scratch remain allocated.
This is a target-weight-module conversion, not a `.to()` call on the serving
model (already on CUDA, with both saved mode representations). It changes
both the API and source-release schedule, so its difference from `host_reload`
must not be attributed solely to Python call overhead. Both methods use pinned
snapshots by default and recapture the same production graph lists.

To compare both host methods in fresh workers with a matched configuration:

```bash
python benchmark/paras/bench_reconfigure.py \
  --config path/to/matched-config.json \
  --methods host_reload host_model_to --direction both --repetitions 1 \
  --output artifacts/new-host-comparison
```

Naive NCCL relies on PyTorch stream/lifetime tracking, with the production-style
per-layer GPU fence and a final device synchronization. It adds no per-layer
host wait. Host reload needs no cross-rank fence for its H2D copies.
Naive NCCL includes attention transfers in both directions:
EP→TP takes local QKV/O slices; TP→EP gathers and reconstructs full projections,
discarding duplicate KV-head replicas. GPT-OSS gate/up interleaving is supported.
All runtime patches exist only in the fresh benchmark workers, with no new
production flags, global sitecustomize changes, or modifications to serving code.

## Configuration and dry runs

The benchmark runs directly from a SGLang checkout; no artifact source restoration
or archived patch is required. Install the serving dependencies and build/install
the ParaS CUDA extension from `python/sglang/srt/paras/csrc` for the local
Python/Torch/CUDA environment. If using an in-place extension build, prepend both
`$PWD/python` and `$PWD/python/sglang/srt/paras/csrc` to `PYTHONPATH` from the
repository root. The six methods, including `host_model_to`, are implemented in
this directory; artifact launchers only select configurations and invoke the driver.

The GPT-OSS/A100 config uses the validated memory fraction **0.70**. The earlier
0.75 run failed preparation. New runs apply the same config to all six methods,
including restart; they do not reproduce the older sparse-graph restart reference.

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
An explicit `paras_tp_max_prefill_tokens` is preserved for ParaS and translated
to native TP's `max_prefill_tokens` for restart, while native EP keeps its own
prefill limit. This also keeps the configured MoE workspace reservation aligned.

Full/fixed methods use the launcher's peer-access KV path. Independent-storage
methods explicitly use NCCL for empty-cache bookkeeping because their backend
does not own a real UMM IPC arena; the worker records that difference. Static
TP rebuild enables token-ID synchronization and clears stale DeepEP variables,
matching the launcher. Remaining scope/configuration caveats are in the audit.
Direct workers also apply production's optional GPU CPU-affinity and NUMA policy
before model construction, and record their effective CPU affinity.
Supported scope is a single node, PP=1, ordinary Qwen3-MoE/GPT-OSS decoding without
quantization, speculation, LoRA, memory-saver, or torch.compile. The optional
`paras_vmm_runtime_states` mode is supported for the five switching methods with
explicit Triton attention, prefill and decode backends (the latter two may inherit
the main backend). Native restart is a shared reference without dual-mode VMM.

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

### Compare VMM off and on

`--vmm off`, `--vmm on`, and `--vmm both` override
`server_args.paras_vmm_runtime_states`. Omitting the option preserves the config's
value, which defaults to off. The same option works with the GPT-OSS/A100 and
Qwen3/H200 configs; no model dimensions or GPU architecture are hardcoded in the
adapter. The production CUDA driver checks device VMM support at initialization.

```bash
# CPU-only plan: 22 trials for six methods, both directions, one repetition.
python benchmark/paras/bench_reconfigure.py \
  --config benchmark/paras/configs/gpt_oss_120b_a100.json \
  --vmm both --repetitions 1 \
  --output /tmp/gptoss-vmm-plan --dry-run

# Run only when all eight GPUs are available; start with a full-method smoke run.
python benchmark/paras/bench_reconfigure.py \
  --config benchmark/paras/configs/gpt_oss_120b_a100.json \
  --vmm both --methods full --direction both --repetitions 1 \
  --output results/gptoss-vmm-full-smoke

# All switching methods plus one shared restart reference per direction/repetition.
python benchmark/paras/bench_reconfigure.py \
  --config benchmark/paras/configs/gpt_oss_120b_a100.json \
  --vmm both --direction both --repetitions 3 \
  --output results/gptoss-vmm-comparison
```

Both variants use identical graph limits, KV planning inputs, model and transport
configuration in separate fresh workers. Restart constructs native single-mode
engines, for which the ParaS VMM flag is invalid. It therefore runs once per
direction/repetition and is labeled `vmm: "not_applicable"`, including when only
`--vmm on` is requested. Its effective Engine arguments explicitly disable VMM.
Do not count that shared reference twice when pooling repetitions.

This compares the existing production
[runtime-state VMM mechanism](../../../docs/paras/runtime_state_vmm.md): only the
active mode's **logits and main Triton KV-index scratch** have physical backing.
Weights, KV contents, SWA indices and graph-private pools are outside that scope.
It is a memory/latency tradeoff, not an alternative weight-transfer kernel.

`full` uses production graph-state activation without an adapter. Recapture
methods retain only the two named scratch tensors per mode and reinitialize
them during buffer setup; they still discard graph executables and saved graph
metadata. A worker-local allocator subclass permits repeated requests for the
same scratch name, rejecting changed shapes/dtypes. Physical mapping, unmapping,
zero-on-map, synchronization and fail-closed behavior use production code.
After graph disposal, the benchmark explicitly activates target VMM inside
`runtime_switch_ms`, because the discarded saved graph no longer performs that
activation. Recapture's buffer initialization is inside `graph_capture_ms`.

VMM unmap/map, initialization during mapping and synchronization belong to
**Others**. Recapture initialization (including resetting reused scratch) belongs
to **Graph**, while Weights is unchanged. Boundary reports validate that the
active mode is backed and inactive modes have zero resident VMM bytes. They
record per-mode virtual/resident bytes before the switch, after the switch and
after the validation probe, outside the measured interval. Use these counts
alongside driver free memory: PyTorch allocator statistics exclude VMM pages.
With VMM off, these VMM-specific byte counts are `null`, not an estimate of zero
ordinary graph memory. The actual planned KV capacities are recorded separately.

The new comparison has CPU lifecycle/ordering tests and dry-run coverage for both
configs. GPU recapture/replay correctness, memory fit and timings still require
fresh runs; the older measurements above did not exercise this adapter.

## Output and tests

- `manifest.json`: resolved experiment, versions, repository revision/branch,
  working-tree status, source hashes, and GPU inventory/topology for actual runs.
- `benchmark_source.tar.gz` and `tracked_changes.patch`: exact benchmark sources
  including uncommitted/new files, plus all tracked working-tree changes against
  the recorded revision (including serving/runtime changes outside the benchmark).
- `resolved_config.json`: primary resolved configuration. With `--vmm both`,
  `resolved_config_vmm_off.json` and `resolved_config_vmm_on.json` are the actual
  variant configs; the manifest records their paths and every planned trial.
- Per-trial commands, normalized `server_args.json`, supervisor and rank logs.
  `ready-ranks.json` records each worker's prepared source graph sets, active
  replay limits, KV reservation, memory and CPU placement before measurement.
- `trials.jsonl`: successful and failed trials, per-rank timings and peak PyTorch
  allocated/reserved memory. Non-rebuild results separate switch-only and probe
  peaks and record driver free memory at each boundary, allocator counter deltas,
  allocator settings and EP/TP full/SWA capacities. PyTorch peaks exclude
  non-PyTorch allocations; driver free memory is a boundary observation, not a
  sampled peak. Legacy top-level peaks cover both switch and probe.
- `summary.json`: median/range by method, direction and VMM setting; failed trials
  are excluded and off/on measurements are never pooled together.

CPU reference and orchestration tests, explicitly hiding all GPUs:

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s benchmark/paras/tests -v

# Includes the VMM pytest fixtures and production allocator/state tests.
CUDA_VISIBLE_DEVICES='' python -m pytest -q benchmark/paras/tests \
  test/srt/paras/test_runtime_memory.py test/srt/paras/test_runtime_states.py
```

These tests cover complete-tensor simulated NCCL routing, Qwen/GPT gate layouts,
replicated KV heads, host snapshot independence, allocation/alias lifetime,
configuration parity, target-mode graph-buffer allocation, graph operation
ordering, and worker error propagation.

### Weights / Graph / Others breakdown

Export an additive breakdown from saved trials without running GPUs:

```bash
python benchmark/paras/reconfigure/breakdown.py \
  --trials results/gptoss-reconfigure/trials.jsonl \
  --output results/gptoss-reconfigure/time-breakdown
```

Optional `--reference-trials` includes separately labeled reference rows. Each
trial uses the rank with the largest total duration for every bucket. Weights
includes the weight-transfer phase and host auxiliary enqueue time; Graph
includes disposal and capture; Others is the remainder. The CSV also separates
Others into non-weight runtime reconfiguration and time outside named phases,
including final synchronization/barrier. Full's retained-graph activation is
inside runtime reconfiguration, so it remains in Others. Host asynchronous
copies can complete in later phases; the generated report documents this limit.
Restart rows with only a combined initialization timer remain total-only.

The exporter writes CSV, JSON and Markdown with exact accounting definitions.
Failed trials are excluded and recorded separately; missing phase measurements
are never filled in as zero. The output directory must be new.

## Memory pressure and standalone scope

Keep the real planned KV allocation when comparing system methods. Increase
`server_args.mem_fraction_static` in separate, explicitly labeled configurations
to explore reduced headroom; never silently shrink graphs/cache after an OOM.
Equal fractions do not guarantee equal KV capacity or physical footprints:
independent storage allocates max(EP,TP) K/V buffers plus separate workspaces,
whereas UMM overlaps regions. Report both endpoint capacities and observed
headroom. Logical KV extents across modes must not be added as physical bytes.

GPU allocation pressure is distinct from Python GC. The caching allocator can
reuse cached blocks, reclaim them or retry a device allocation; Python's cycle
collector is not driven directly by allocated GPU bytes. Do not force `gc` or
`empty_cache` in the benchmark to amplify a baseline penalty. Capture retains
SGLang's own GC policy. The new allocator counters expose retries and allocation
calls where supported; missing counters remain null. A headroom sweep can OOM
and does not guarantee more collections or a particular performance ordering.

The chosen implementation remains standalone under `benchmark/paras`. The five
switching methods construct real Scheduler/ModelRunner state with production
weights, KV pools, backends, workspaces and graphs; benchmark adapters select
the weight-storage, transfer and recapture behavior. Normal SGLang serving has
no new strategy flags or dependency on benchmark code. Engine-path integration
and serving selectable strawmen are deferred.

Untimed endpoint checks verify active graph coverage and replay limits, both
saved mode sets for `full`, and absence of saved graphs after recapture. The
recapture path updates `max_bs`/`max_num_token` just as production graph-state
restoration does, allocating inputs for the target mode's dimensions. Fixed-buffer
initialization may still own both graph sets before a zero-warmup measurement;
they must be discarded during the measured transition.

Independent-storage endpoint checks require materialized active weight entries
and only scalar placeholders for inactive entries. They report backing bytes by
mode; this verifies manager ownership, not all possible external references or
the allocator returning cached memory to CUDA. These checks run outside switch
timing. Live KV request/page migration remains outside this benchmark's scope.
