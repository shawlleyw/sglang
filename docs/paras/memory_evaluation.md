# Reproducing the ParaS memory comparison

This protocol compares static EP8, static TP8, ParaS with both modes resident,
and ParaS with inactive runtime backing released through VMM. The reference
workload is GPT-OSS 120B on eight A100-SXM4 80GB GPUs. It measures GPU memory
and KV capacity; it does not establish trained-model quality or serving throughput.

The implementation references are [UMM ownership and sizing](unified_memory_manager.md),
[mode-local runtime state and VMM](runtime_state_vmm.md), and
[parallelism switching](parallelism_switch.md). Historical results in
[memory_analysis.md](memory_analysis.md) describe a different allocator.

## Baselines and fairness

| Configuration | Attention / experts | Resident runtime states | Extra reservation policy |
|---|---|---|---|
| Static EP8 | DP attention, DeepEP EP8 | EP only | Preallocate and reuse active EP MoE and attention scratch before KV profiling |
| Static TP8 | TP8 attention and experts | TP only | Preallocate and reuse active TP scratch before KV profiling; replicate input embedding and LM head |
| ParaS, VMM off | Switches between the above | Separate EP and TP states remain resident | MoE and attention scratch stay in UMM; transfer headroom and both graph sets remain |
| ParaS, VMM on | Same switching modes | Only active KV-index/logits backing is resident | Same UMM layout and KV capacity as VMM off |

The static baselines use native single-mode execution with an **explicit matching
workspace reservation**, not untouched native memory allocation. The
[evaluation helper](../../scripts/paras/eval/matched_baseline_workspace.py)
allocates scratch after weights load and before native KV profiling, then binds
it for actual reuse. Instrumentation must verify both reservation order and
reuse; allocating an unused matching buffer would double-count workspace.
Larger shapes retain dynamic fallback.

Both input embedding and LM head are fully replicated in static TP to match
ParaS: set `SGLANG_GPTOSS_REPLICATED_EMBEDDING=true` and
`SGLANG_GPTOSS_REPLICATED_LM_HEAD=true`. This is a storage-layout match while
transformer attention and experts stay TP-sharded. It is not the DP LM-head
execution path, which requires DP attention.

Use the same `mem_fraction_static` across cases, but measure actual device
residency separately. The fraction controls planning; it is not a cap on total
physical residency after graphs, communication state, or allocator caches grow.
Compare ParaS EP with static EP and ParaS TP with static TP.

## Reference configuration

| Setting | Value |
|---|---|
| Hardware | 8 × A100-SXM4 80GB; measured CUDA-addressable capacity 79.25 GiB/GPU |
| Model | `~/models/gpt-oss-120b-BF16-unsloth`; GPT-OSS 120B architecture |
| Weight loading | `--load-format dummy --dtype bfloat16 --random-seed 42` |
| KV / context | BF16 KV (`--kv-cache-dtype auto` with BF16 model); context 131072; page size 1 |
| Fractions | 0.70, 0.75, 0.80 |
| Request limit | 2048 global; EP scheduler admission 256/rank; backing request tables capped at 2048 rows |
| EP prefill budget | `--max-prefill-tokens 2048`, per DP rank |
| TP prefill budget | Static TP: `--max-prefill-tokens 8192`; ParaS: `--paras-tp-max-prefill-tokens 8192` |
| Chunking | `--chunked-prefill-size -1`; budgets are scheduler targets, not hard per-forward limits |
| Attention / MoE | Triton / Triton; DeepEP mode `auto`; BF16 dispatch |
| EP graph batches | 1, 2, 4, 8, 16, 32, 64, 128, 256 |
| TP graph batches | EP list plus 512, 1024, 2048 |
| SWA storage | `--disable-hybrid-swa-memory --swa-full-tokens-ratio 1.0`; disables the separate pool, not sliding-window attention semantics |
| Scheduling | Overlap and radix cache disabled; automatic switching disabled; `PARAS_POST_SWITCH_RAMP_ITERS=0` |
| Transfer | `PARAS_CONFIGURE_METHOD=peer_access`, `PARAS_KV_TRANSFER_METHOD=peer_access` |
| DeepEP | `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256`, `NVSHMEM_QP_DEPTH=2048` |
| NVSHMEM | `NVSHMEM_DISABLE_NCCL=1`; installed heap granularity 512 MiB, unchanged |
| Instrumentation | Python phase snapshots plus approximately 1-second NVML samples; no CUPTI or LD_PRELOAD allocation interposer |

The prefill override controls both ParaS TP scheduling and TP MoE reservation.
Do not shrink its TP workspace to the EP 2K budget. The configured active-mode
MoE/attention reservations are 540/32.5 MiB in EP and approximately
405.785/32.5 MiB in TP. DeepEP low-latency dispatch capacity is a separate
setting from the normal EP prefill budget.

Dummy BF16 weights retain the model's parameter shapes and exercise execution,
but output comparisons only validate consistency, not trained-model accuracy.
Record dependency versions, CUDA/driver versions, the Git base, uncommitted
patch, and source hashes with each independent run.

## Launching and reproducing

The A100 GPT-OSS server launchers supply the topology and backend settings:

```bash
# EP defaults to a per-rank 2K budget. ParaS additionally defaults TP to 8K.
MAX_PREFILL_TOKENS=2048 PARAS_TP_MAX_PREFILL_TOKENS=8192 \
  ENABLE_PARAS=1 PARAS_AUTO_SWITCH=0 HYBRID_SWA=0 \
  DISABLE_OVERLAP=1 DISABLE_RADIX_CACHE=1 NVSHMEM_DISABLE_NCCL=1 \
  NUM_GPUS=8 MEM_FRACTION_STATIC=0.80 \
  bash scripts/paras/eval/a100/gptoss/launch_server_dp_ep.sh \
  --load-format dummy --dtype bfloat16 --random-seed 42 \
  --paras-vmm-runtime-states
```

This is a launch example, not the complete measurement harness. Omit VMM for
the VMM-off case. Use `ENABLE_PARAS=0` for static EP, and the TP launcher with
`MAX_PREFILL_TOKENS=8192` for static TP. **Static memory measurements additionally
require the matched-workspace helper**, either as the entrypoint or through its
instrumentation hooks. Plain launcher runs do not install those reservations.
Use the complete configuration table above and explicit graph batch lists for
an exact comparison.

The local reference bundle is
`artifacts/20260920T000955Z_ep2k_tp8k_memory/`. Artifacts are intentionally
excluded from Git: retain or transfer this bundle separately when reproducing
from a fresh clone. Its `plan.json`, per-case `launch.json`/`launch.sh`,
`entry_snapshot.py`, and `environment/source/` record the commands and measured
source. The base Git revision alone is insufficient because that run included
uncommitted changes.

The bundle contains a `drivers/reproduce.py` launcher that creates a **new**
artifact directory, copies the measurement hooks, resolves paths against the
current checkout, records the current source/environment, and runs cases
serially. From the checkout root, using the same Python/CUDA environment:

```bash
export CUDA_HOME=/path/to/cuda-12.8
export MODEL_PATH="$HOME/models/gpt-oss-120b-BF16-unsloth"
python artifacts/20260920T000955Z_ep2k_tp8k_memory/drivers/reproduce.py \
  --output artifacts/NEW_RUN_NAME --fractions 0.70 0.75 0.80
```

Run only when all eight selected GPUs are free. Preserve failures and logs;
do not treat missing or failed cases as completed. The helper refuses to
reuse an existing output directory. It launches measurements; analysis against
a new reference requires the saved-data audit/report scripts and the matching
reference bundle. The report generator in the original bundle can be rerun
without touching GPUs:

```bash
CUDA_VISIBLE_DEVICES='' python \
  artifacts/20260920T000955Z_ep2k_tp8k_memory/drivers/report.py
```

## Workload and snapshot sequence

Run each case in isolation on all eight GPUs and release its processes before
the next case. For ParaS, capture both graph sets, sample initial EP readiness,
switch to TP for its idle sample, and return to EP for its comparable idle sample.

1. Run batches of 8 and 128 requests, each with 128 input tokens and 32 generated
   tokens, in the active mode; ParaS covers EP, TP, then EP again.
2. Exercise live-KV EP→TP→EP switching with eight requests generating 192 tokens.
3. Run one 32K-input request generating eight tokens, in both ParaS modes and
   the corresponding static mode. This deliberately exceeds the scheduling
   budget and exercises unchunked dynamic fallback.
4. Run 2048 requests with 128 input tokens and 32 outputs in each applicable mode.

Use greedy sampling, ignore EOS, and record request/token completion counts.
Flush request caches between phases. Preserve phase order: the PyTorch caching
allocator can retain intermediates, so residency depends on previous workloads.

## Reported metrics

| Metric | Definition and interpretation |
|---|---|
| Idle residency | CUDA driver total minus free, maximum across eight ranks. Static: post-capture readiness. ParaS EP: idle after EP→TP→EP. ParaS TP: first TP idle sample. Includes caches and external runtime allocations. |
| Additional idle memory | ParaS maximum idle residency minus the corresponding static-mode maximum; report VMM state and fraction. |
| KV slots | Actual available token capacity. EP is per DP replica; TP is the head-sharded shared cache. Compare within the same mode, not raw EP versus TP slot counts. |
| KV GiB/GPU | Logical K/V storage bytes per GPU from the pool. UMM rounding/slack is accounted separately. |
| KV retained | ParaS slots divided by the matching static-mode slots. VMM changes backing residency, not KV capacity. |
| Post-workload residency | Driver-used bytes immediately after the same named workload; includes allocator cache and must not be labelled idle readiness or peak live memory. |
| Peak live Torch allocation | Per-phase PyTorch allocated-byte high-water counter. Excludes VMM backing and non-PyTorch allocations; reset at phase boundaries. |
| Largest observed residency | Maximum of all NVML samples **and** saved driver snapshots, including VMM activation. Report the winning phase and telemetry source; this remains a lower bound on the true instantaneous peak. |
| Switch transient | Driver residency at activation, kept separate from steady-state mode residency. Inactive VMM backing is released, but cached Torch blocks can remain while larger target backing is mapped. |

All memory units are GiB (`2^30` bytes) or explicitly labelled MiB (`2^20`).
A maximum over an entire run can come from different phases before and after
an optimization. To attribute an improvement, compare the **same phase and
mode**, then separately report what determines the new whole-run maximum.
The previous coarse VMM sample of 74.328 GiB missed a saved 77.420 GiB switch
snapshot; use combined telemetry instead of repeating that sampled value as
an absolute peak.

## Validation and presentation requirements

Check the effective environment and scheduler budgets on every rank, including
after mode switches. Verify matching embedding/LM-head shapes, request-table
size, pre-KV workspace reservation and reuse, active/inactive VMM residency,
zero OOM counters, live-switch completion, and completed requests. Reconcile
per-rank memory ledgers exactly to driver-used bytes. Compare saved output IDs
and sampled logits against the reference, with the dummy-weight limitation.

Present the final 2K EP / 8K TP matrix as the main result. Keep earlier sizing,
NCCL-enabled, and 8K-EP experiments in a historical appendix; do not combine
numbers from unmatched configurations. Each exported figure should identify
its memory metric, mode, fraction, VMM setting, and the dummy BF16 setup.
Use `report.md` for the full record and `presentation.md` plus `figures/` in the
local bundle for slide material. No new GPU execution is needed to regenerate
those summaries from saved results.
