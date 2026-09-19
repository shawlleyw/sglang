# ParaS documentation

Start with [runtime parallelism switching](parallelism_switch.md) for the
EP↔TP workflow and supported scope. Use the references below for details;
each owns one part of the design.

## Current design

| Topic | Reference |
| --- | --- |
| Buffer layout, capacities, workspace ownership, and migration safety | [Unified memory manager](unified_memory_manager.md) |
| Request migration and control-plane ordering | [Parallelism switching](parallelism_switch.md) |
| Automatic switching policy | [Switch policy](parallelism_switch_policy.md) |
| CUDA graph capture and per-mode state | [CUDA graphs](cuda_graph.md) |
| GPT-OSS biases, interleaved experts, and hybrid attention | [GPT-OSS support](gpt_oss_support.md) |
| SWA allocation, eviction, and migration | [Sliding-window attention](swa.md) |
| Why ParaS disables prefix caching | [Radix cache](radix_cache.md) |

The unified memory reference is the source for layout formulas, backend
scratch requirements, and memory-overhead examples. Other docs link to it
rather than maintaining independent versions. The former
[memory reuse design](memory_reuse_design.md) is now a compatibility link.

## Transfer kernels

- [Expert weights](nvlink_peer_access_weight_transfer.md): expert slicing,
  peer writes, and synchronization; attention transfer links to the memory reference.
- [KV cache](nvlink_peer_access_kv_cache_transfer.md): token/head routing,
  replication, and the separate NCCL KV transport path.
- [NVLink kernel guidelines](nvlink_peer_access_guielines.md): kernel tuning guidance.

## Measurements, history, and proposals

- [Parallelism configurations](parallelism_configuration.md) compares the
  EP/TP serving configurations and records benchmark observations.
- [Historical memory analysis](memory_analysis.md) measures the earlier
  N+1-slot allocator. Its totals are not the current UMM overhead.
- [KV peer-access development notes](exploration_notes_kv_cache_peer_access.md)
  record earlier bugs and experiments.
- The GPT-OSS chronicles and performance tables in the graph/transfer docs
  are historical evidence, separate from their current implementation sections.
- [Future radix-cache design](future/radix_cache.md) is a proposal.
- [FP8 weight support](paras_fp8_support.md) is an unimplemented proposal.
- [v0.5.19 migration assessment](migration_v0.5.19_assessment.md) records a
  source review at the revisions named in that report.
- Editable design diagrams: [system overview](figures/system_overview.drawio)
  and [request redistribution](figures/request_redistribution.drawio).

For runnable settings, see [launch_common.sh](../../scripts/paras/eval/launch_common.sh)
and the [H200](../../scripts/paras/eval/h200/) or
[A100](../../scripts/paras/eval/a100/) scripts.
