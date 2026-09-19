# Production-fidelity audit of switch experiments v1

V1 (`3eb70791a`) is preserved at `artifacts/paras_switch_v1/results`, with a
file-by-file checksum manifest in `artifacts/paras_switch_v1/sha256.json`.
It is a functional smoke experiment, not an accepted production-performance
evaluation. V2 subsequently passed all four switching methods in both directions
on eight A100-80GB GPUs with GPT-OSS-120B BF16 at memory fraction 0.70; the 0.75
attempt hit an untimed preparation OOM. See the
[v2 smoke report](../../../artifacts/20260919T090553Z_gptoss_120b_switch_v2/README.md)
for exact configuration, failed attempt, per-rank checks and timings. Restart
numbers are explicitly copied v1 references. This audit is against the serving
implementation and `scripts/paras/eval/launch_common.sh`, including the user's
uncommitted UMM workspace changes. No serving implementation was edited.

## 1. Graph coverage and lifecycle

V1 supplied `[1,2,4,8,16,32,64,128,256]` to both EP and TP. This both sparsified
EP coverage and removed production's larger TP range. It also overwrote
`gr.capture_bs` with that list on every recapture, bypassing resolved settings.

The production launcher derives EP's maximum from global request capacity / GPU
count; ParaS scales the default TP maximum by GPU count. At 2048 requests / 8
GPUs, maxima are EP=256, TP=2048. `ServerArgs._generate_cuda_graph_batch_sizes`
generates **36 EP sizes and 100 TP sizes**, rather than nine each. Generic A100
`ServerArgs` defaults without the launcher can choose a different EP maximum;
that must not be confused with this launcher's configuration.

The current candidate passes maxima only to `ServerArgs`, preserves the actual
EP list resolved by `get_batch_sizes_to_capture` and the production ParaS TP
list, and reuses those per-mode lists during recapture. This preserves real
request-pool/alignment filtering, per-mode maximum buffer allocation and TP
scaling. Worker logs record both resolved lists. No arbitrary list is inserted
by benchmark code. Larger graph coverage can change memory fit as well as
capture latency; v1 memory-fit results do not validate v2.

Recapture also restores active `max_bs` and `max_num_token`, matching production
graph-state loading. Previously the benchmark updated the list but could leave
the prior mode's replay limits in place. Untimed endpoint checks now verify the
active set/limits, both retained sets for full, and disposal for recapture paths.
Prepared source reports are persisted in `ready-ranks.json` with KV reservation
and memory observations. Independent weight entries are checked for active
materialization and inactive scalar placeholders at source and target endpoints.

V1 explicitly called `gc.collect()` after dropping graph references. The
candidate removes that call: graph and output dictionaries, saved mode states,
and the global pool handle are released. Production's own
`cuda_graph_runner.freeze_gc` capture context independently performs GC on
entry/exit. That policy remains unchanged to match the serving code; capture
phase timings can therefore include production GC. Discard timing no longer
includes a benchmark-forced collection.

## 2. The 0.142 s result was not NCCL

`fixed_buffer_recapture` calls the production UMM transfer implementation,
including fused peer-access expert kernels and the attention transfer wrapper.
It differs from `full` by dropping graphs and recapturing the target. It never
implemented a fixed-buffer NCCL runtime baseline.

The v1 EP→TP **nested runtime-switch phase** was 0.142 s on rank 0, including
0.130 s of fused weight transfer. The **whole switch**, including graph work,
was 2.237 s. These are different timing boundaries.

A separate preallocated-staging NCCL weight microbenchmark measured 0.188 s,
versus 0.130 s for direct v2. That component benchmark is not the fixed-buffer
runtime row. Current manifests/results explicitly record each method's weight
transport. The comparison naive_nccl → fixed_buffer_recapture changes both
allocation and transport; it is not a clean allocation-only ablation. A real
fixed-buffer NCCL runtime control is still missing. The agreed five-row design
keeps row 4 as fused UMM + recapture. Use the transport microbenchmarks for kernel
ablations; do not attribute the row 3→4 gap solely to buffer reuse.

## 3. Other differences found

| Area | V1 behavior and consequence | Current candidate / remaining work |
|---|---|---|
| Layer synchronization | Independent methods added `torch.cuda.synchronize()` after every layer; production has GPU layer fences and one final device sync. Host reload also had a redundant rank fence despite only H2D copies. | Removed per-layer host waits; PyTorch NCCL/current-stream lifetime tracking protects queued tensor uses. Final device sync remains. H2D reload has no rank fence. Requires GPU validation. |
| Serving overlap | Forced `disable_overlap_schedule=True`, unlike launcher's enabled default. An empty boundary does not justify silently changing runtime configuration. | Presets now inherit enabled overlap. Manual probe still bypasses scheduling overlap; see probe scope below. |
| KV transport | Forced NCCL even for production full/fixed methods. Even empty-cache NCCL can exchange placeholders; peer path has different empty-path work. | Full/fixed now use production launcher's peer-access setting. Independent allocation backend has only an IPC anchor, so it explicitly retains NCCL empty-cache bookkeeping and reports that choice. Not suitable for live KV. |
| Static TP environment | Did not apply `SYNC_TOKEN_IDS_ACROSS_TP=1` or clear DeepEP environment as the launcher does. | Static TP rebuild now follows those launcher steps. Presets also set launcher's NVSHMEM_QP_DEPTH=2048. |
| Warmup history | Reference generation executes the target before timing, then restores the source. Host H2D itself was not the setup transport. | Default one additional warmup exercises the actual measured method, including H2D/recapture, then restores source. `warmup_switches` is explicit. Mode initialization/allocator histories still differ; this is a warmed transition, not cold first use. |
| Host reload scope | Snapshots already target-sharded tensors outside timing and retains model objects, request/KV pools, backends, scratch, communicators and precomputed auxiliaries. | Accurate name is **prepared host-weight reload + recapture**. It is not runtime reconstruction from a host checkpoint. CPU preparation/resharding/offload are excluded, as specified by this optimistic baseline. |
| Auxiliary weights | Host snapshots all registered Parameters except bulk expert/QKV/O weights, including inactive-mode biases and overlapping sink views; reloads them in place. Naive NCCL retains precomputed biases/sinks. | Now selects active target module aliases, excludes saved mode representations and deduplicates identical views. Registered buffers remain retained, not reconstructed. |
| Storage/capacities | Independent storage retains max(EP,TP) per-layer KV and separate workspaces for both modes, while production UMM overlaps regions. Equal mem_fraction_static does not imply equal footprint. | Both already allocate real full planned KV. Results now record full/SWA capacities, logical mode extents, independent KV backing bytes, driver headroom, allocator configuration/counter deltas and separate peaks. Common memory fraction is the current policy; no equality claim about capacities/footprints. |
| Static target placement | Native static TP may shard embedding/vocabulary weights, while ParaS retains EP-built replicated placement. Native allocation differs from UMM. Static TP launcher defaults memory fraction .8 vs .75 used in this comparison. | Rebuild is a whole-system alternative, not identical-runtime ablation. The common .75 budget is an explicit experimental choice, not exact reproduction of every launcher default. |
| Process wrapper | Four methods construct Scheduler directly; rebuild uses Engine. Direct workers bypassed optional production CPU/NUMA affinity. | Workers now apply production affinity/NUMA policy before allocation and record affinity. Engine messaging/event loops remain excluded; the chosen scope stays standalone. |
| Probe workload | Direct EP probe creates a request on each rank, while Engine probe submitted one request. Direct probes bypass Scheduler's loop. Rebuild validates repeated text; others compare same-mode logits. | All now use world_size global requests and matching token limits. Through-probe timing remains diagnostic: scheduling and correctness endpoints still differ until Engine integration. |
| Phase instrumentation | Added CUDA synchronization around named phases and a final distributed barrier. | Removed benchmark-added internal CUDA waits. Phases are explicitly host wall time; async enqueue time is not device duration. Total boundary sync/barrier remains. |
| Peak memory | Counters reset before switching but were read after the validation probe. Non-PyTorch allocations are excluded. | Now records switch-only and separately reset probe peaks, driver boundary headroom and switch allocator deltas. Legacy peaks cover both intervals. Diagnostics between intervals are excluded from through-probe total. |
| Replication/statistics | One fresh-process runtime trial per method/direction; no randomized repeated comparison. | Functional smoke evidence only. Repeat matched trials before reporting distributions or assigning small timing differences to mechanisms. |

Sources: `reconfigure/runtime.py`, `storage.py`, `transfers.py`, `worker.py`;
`python/sglang/srt/paras/layers/paras_model.py`, `paras_cuda_graph.py`,
`scheduler_paras_mixin.py`, `cache_transfer/utils.py`; production `ServerArgs`,
`CudaGraphRunner`, `ModelRunner`, `run_scheduler_process`; GPT-OSS native and
ParaS models; `scripts/paras/eval/launch_common.sh` and A100 GPT-OSS launchers.

## 4. Transfer microbenchmark boundaries

The weight microbenchmark has no CUDA graph capture at all. The sparse graph
problem was in the runtime reconfiguration harness. Weight microbenchmarks use
36 distinct production-planned layer weight views, zero inference workspace,
and minimal unused KV reservation. They restore source weights outside timing
and preallocate NCCL staging. They measure transport/packing, not dynamic
allocation, graph retention or serving memory footprint. Overlapped NCCL only
overlaps within-layer EP→TP packing; reverse is sequential and labeled so.

The KV benchmark repeats one homogeneous layer 36 times. It does not reproduce
GPT-OSS's mixed full/SWA live cache or request migration. Keep these component
results separate from empty-state system switching. V3 expert kernels lack
GPT-OSS specializations; only verified v2 results are reported.

## 5. Validation status and remaining evaluation work

1. All fresh switching trials validated 36 EP / 100 TP graphs and memory fit
   at the explicitly reduced 0.70 budget. The original 0.75 attempt is preserved
   as a failure; graph coverage was unchanged. Planned KV capacities matched
   across all fresh methods and ranks.
2. CPU lifecycle checks and GPU same-target logits/storage checks passed after
   pointer-only disposal and asynchronous source release. Full retained both
   graph sets; non-retaining methods retained none after recapture.
3. Retain agreed row 4 (fused UMM recapture) with explicit transport labels;
   use microbenchmarks for transport ablations.
4. Corrected warmup and active auxiliary handling passed GPU validation. CPU
   placement, capacities and headroom are recorded. No measured switch had an
   allocator retry; extra weight allocations must not be described as observed
   pressure-triggered reclamation in these trials.
5. Collect repeated trials and traces separating allocation, staging, transfer,
   graph destruction and capture. Keep readiness/probe workload consistent.

The current code addresses graph/GC bugs, memory observability, warmup, active
auxiliary selection and selected launch/sync/probe differences. It does not
claim common Engine-level endpoints or live KV migration for independent storage.
