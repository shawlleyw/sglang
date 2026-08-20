# ParaS Unified Memory Manager

## Purpose

`ParaSMemoryManager` owns the persistent GPU storage whose shape changes when
ParaS switches between expert parallelism (EP) and tensor parallelism (TP).
It allocates one contiguous `uint8` buffer and exposes typed tensor views into
that buffer. A switch moves data between preplanned views; it does not allocate
or free model weights or KV-cache tensors.

The manager covers:

- EP and TP expert weights for every MoE layer.
- Attention weights that need stable EP and TP views.
- EP and TP KV-cache views, including hybrid full/sliding-window layouts.
- One optional pair of weight-permutation staging buffers for NCCL
  intra-node resharding.

Every entry is 256-byte aligned. Model parameters remain ordinary PyTorch
`Parameter` objects backed by views of the manager buffer.

## Planning And Materialization

Model construction has two phases.

1. `plan_qwen_moe_layout` or `plan_gpt_oss_moe_layout` records expert sizes,
   reserves attention entries, and optionally reserves NCCL staging.
2. `reserve_kv_cache` records per-layer EP and TP cache sizes.
3. `materialize` assigns offsets for ordinary reservations, places the combined
   expert/KV four-anchor run, and allocates the backing buffer.
4. `create_paras_moe_aliases` validates the materialized expert entries and
   preserves `experts.* -> ep_experts.*` names used by weight loading.

The global manager lets model weight constructors bind directly to a planned
entry:

```python
manager = ParaSMemoryManager(device="cuda")
plan_qwen_moe_layout(
    manager,
    ...,
    intra_node_weight_transfer_method="peer_access",
)
manager.reserve_kv_cache(...)
manager.materialize()
create_paras_moe_aliases(manager, num_layers)
set_global_paras_memory_manager(manager)
```

The main view APIs are:

- `get_view(name)` for the registered shape and dtype.
- `get_view_as(name, shape)` for a zero-copy reshape of the same bytes.
- `get_kv_views(num_layers, mode)` for the EP or TP K/V views.
- `is_managed(tensor)` to check whether a tensor belongs to the backing buffer.

## Capacity Planning

The capacity planner first subtracts the dynamic reserve and non-UMM static
weights from currently available GPU memory. It then binary-searches a common
per-mode base footprint `B`:

```text
EP cache budget = B - sum(EP expert weights)
TP cache budget = B - sum(TP expert weights)
```

For `dp_size > 1`, TP expert weights are larger, so TP cache capacity is
reduced by that growth. The search evaluates the exact aligned four-anchor
geometry rather than an additive weights-plus-cache estimate. The returned
plan reports both mode-specific weight and cache bytes, and `materialize`
refuses to allocate if the resulting UMM exceeds the planned limit.

## Four-Anchor Layout

The former N+1 equal-slot layout only represented `ep_size == tp_size`. The
current layout also supports a wide EP group switching to `G` replicas of a
`T`-rank TP group, where `ep_size = G*T`.

For layer `i`, define the aligned per-GPU byte sizes:

- `we[i]`: EP expert weights.
- `wt[i]`: TP expert weights.
- `ce[i]`: EP KV cache.
- `ct[i]`: TP KV cache.

The manager places four contiguous blocks in one run:

```text
EP weights: forward from P
EP cache:   forward after EP weights and PAD
TP weights: forward from P + we[0]
TP cache:   forward, with its tail anchored after the EP footprint
```

The two modes are never live simultaneously. TP weights can therefore overlap
EP cache storage, which is necessary when `wt[i] = G * we[i]`. Separate EP and
TP entries retain stable addresses even though the regions overlap across
modes.

Production currently requires `ct[i] <= ce[i]` for every layer. Under this
condition the cache tail anchor reduces to `max(ct)`. Capacity planning and
materialization share the same geometry calculation, including this anchor and
all alignment, so the planned and allocated byte counts cannot diverge.

The complete offset derivation and safety proof are in
[`unified_memory_ep_tp.md`](unified_memory_ep_tp.md).

## Transfer Order

The overlapping layout imposes one order per direction:

| Direction | Phase order | Layer order |
|-----------|-------------|-------------|
| EP -> TP | cache, then expert weights | reverse (`N-1` to `0`) |
| TP -> EP | expert weights, then cache | forward (`0` to `N-1`) |

Within a peer-access expert-weight phase, every layer is fenced across the
physical TP peer group before overlapping regions can be reused. NCCL local
resharding is ordered by its TP-group all-to-all. Attention and communicator
state are activated only after all expert weights finish moving.

## Weight Transfer Methods

`PARAS_INTRA_NODE_WEIGHT_TRANSFER_METHOD` accepts exactly two values:

- `peer_access` (default): TP-local CUDA IPC resharding.
- `nccl`: TP-local permutation plus `all_to_all_single`.

The setting controls only the intra-node TP reshard. Both values support
`dp_size > 1`. After EP -> TP local resharding, an in-place NCCL all-gather
over the DP group replicates the owned expert intervals across TP instances.
TP -> EP selects the local `dp_rank` interval and uses no DP collective.

Unknown values fail while planning the layout. Only `nccl` reserves
`staging.w13_pre_permute` and `staging.w2_pre_permute`. The removed
`direct` and overlap names have no compatibility aliases.

See
[`nvlink_peer_access_weight_transfer.md`](nvlink_peer_access_weight_transfer.md)
for the topology matrix and synchronization details.

## Verification

The memory manager is covered at both layout and transfer levels:

- `test_paras_layout_unit.py` plans, materializes, and checks the real
  four-anchor offsets, aliases, overlap, alignment, and switch-order safety.
- `test_weight_transfer_method.py` materializes NCCL staging with
  `ep_size=4`, `tp_size=2`, and `dp_size=2`.
- `test_weight_transfer.py` and `test_weight_transfer_tp_instances.py`
  perform GPU transfers through materialized manager EP/TP views and verify
  bitwise forward and round-trip results.

## KV Cache Integration

`MHATokenToKVPool` consumes one external K/V pair per layer. `SWAKVPool`
consumes separate full-attention and sliding-window view lists and routes each
layer using its `LayerCacheSpec`.

The request allocator, free lists, and eviction behavior do not change. During
a switch the cache pool rebinds from `kv.ep` views to `kv.tp` views, or back,
after the corresponding cache data movement finishes.

Before EP -> TP movement begins, the gather precheck compares the in-flight
global token count with this planned TP token capacity. A switch is rejected
without mutating the pools when the reduced TP cache cannot hold the workload.

KV cache and requests redistribute within each TP subgroup. Expert-weight
replication across DP groups is independent of KV movement.

## Invariants

The manager validates the conditions needed by the transfer kernels and the
overlapping layout:

- Expert and intermediate dimensions divide the configured parallel sizes.
- `ep_size = dp_size * tp_size` when multiple TP instances are configured.
- Per-layer TP cache bytes do not exceed EP cache bytes.
- EP and TP regions for the same layer are disjoint while that layer is in
  flight.
- TP weights end before the TP cache begins.
- All ranks plan the same entry names, shapes, dtypes, and offsets.

The CPU reference implementation and fuzz checks live in
[`benchmark/paras/paras_layout.py`](../../benchmark/paras/paras_layout.py).

## Relevant Files

| File | Responsibility |
|------|----------------|
| `paras_memory_manager.py` | Reservations, capacity planning, four-anchor placement, and typed views |
| `layers/paras_model.py` | Method selection, topology selection, ordering, and synchronization |
| `layers/paras_moe_block.py` | Expert-weight views and transfer primitives |
| `gather_manager.py` / `scatter_manager.py` | EP/TP request and KV-cache movement |
| `unified_memory_ep_tp.md` | Offset derivation, proof, and multiple-TP-instance ownership |
