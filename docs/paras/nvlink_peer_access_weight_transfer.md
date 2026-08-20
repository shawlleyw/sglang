# Expert Weight Transfer for ParaS

## Scope

ParaS switches expert weights between one wide expert-parallel layout and one
or more instances of the same tensor-parallel layout. The setting
`PARAS_INTRA_NODE_WEIGHT_TRANSFER_METHOD` selects only the transport used to
reshard weights inside each node-local TP group:

- `peer_access` (default): CUDA IPC kernels write the TP-local result directly
  into peer memory.
- `nccl`: a TP-group `all_to_all_single` uses one pre-permute staging buffer
  per weight.

Both methods support one or multiple TP instances. When multiple TP instances
exist, EP -> TP always follows the selected local reshard with a DP-group NCCL
all-gather. TP -> EP never requires an inter-node collective.

Unknown method names fail during memory planning. The removed `direct` name
has no compatibility alias.

## Topology Contract

Let `G = dp_size`, `T = tp_size`, `E` be the global expert count, and
`L = E/(G*T)`. EP uses one group of `G*T` ranks; each EP rank owns `L`
experts with the full intermediate dimension. Each TP rank owns all `E`
experts and one `I/T` intermediate shard. There are `G` independent TP
instances, selected by `dp_rank`.

In a multi-node deployment, every TP instance must be contained within one
node. A DP group connects ranks with the same `tp_rank` across TP instances.
CUDA IPC and the NCCL all-to-all are therefore scoped to the local TP group.
The DP group is used only for EP -> TP replication.

The overlapped memory layout registers stable EP and TP views for every layer.
EP -> TP processes layers in reverse order; TP -> EP processes them in forward
order. See [`unified_memory_ep_tp.md`](unified_memory_ep_tp.md) for the
address proof.

## EP -> TP

### Intra-node reshard

Every `dp_rank` reshards only the expert interval owned by its TP instance:

```text
experts_per_dp_rank = E / G
expert_start = dp_rank * experts_per_dp_rank
expert_end = expert_start + experts_per_dp_rank
```

The layer APIs make the transport explicit:

- `paras_reshard_ep_to_tp_intra_node_peer_access`
- `paras_reshard_ep_to_tp_intra_node_nccl`

The peer-access path launches:

- `peer_access_reshard_w13_ep_to_tp_intra_node`
- `peer_access_reshard_w2_ep_to_tp_intra_node`

The NCCL path permutes the local EP tensor into
`staging.w13_pre_permute` or `staging.w2_pre_permute`, then runs a TP-group
`all_to_all_single` directly into the same owned interval of the TP view.
The staging size is one EP rank's weights and does not grow with `dp_size`.

### Inter-node replication

When `dp_size > 1`, ranks with the same `tp_rank` run an in-place
`all_gather_into_tensor` over their DP group for w13 and w2. After the
collective, every TP instance holds the complete TP layout. This phase always
uses NCCL, independent of the selected intra-node method.

The model pipelines two CUDA streams across layers:

```text
intra-node stream: reshard N-1 | reshard N-2 | reshard N-3
inter-node stream:               gather N-1  | gather N-2
```

A CUDA event releases each DP all-gather after its local reshard. Peer-access
writes additionally use a TP-group collective as a remote-write visibility
fence. NCCL local resharding needs no extra fence because its all-to-all
completion is already ordered on the intra-node stream.

For `dp_size=1`, the local interval is the full expert range and the DP
all-gather is skipped.

## TP -> EP

TP weights are already replicated across DP ranks. Each `dp_rank` selects
only its owned interval, then reconstructs the full intermediate dimension
inside its local TP group through one of:

- `paras_reshard_tp_to_ep_intra_node_peer_access`
- `paras_reshard_tp_to_ep_intra_node_nccl`

The peer-access path launches:

- `peer_access_reshard_w13_tp_to_ep_intra_node`
- `peer_access_reshard_w2_tp_to_ep_intra_node`

Experts outside the owned interval are not copied. They remain in the inactive
TP view and are ignored after EP views are activated. This is a logical drop,
not a deallocation operation.

## Memory Manager Integration

Both transports read and write typed views backed by the same
`ParaSMemoryManager` allocation:

- EP source/destination views use `ep_experts.*` entries.
- TP source/destination views use `tp_experts.*` entries.
- Only the NCCL intra-node method reserves the two staging entries.
- The DP all-gather writes directly into the full TP views.

The GPU tests do not substitute standalone allocations. They call
`plan_qwen_moe_layout`, `materialize`, and `create_paras_moe_aliases`,
then validate the materialized TP views and the recovered EP views.

## Kernel Versions

Production peer-access switching selects v2 for TP-local reshards. The v3
wrappers remain available for isolated kernel benchmarking and reverse
ownership tests. No multi-TP-instance kernel variant is required because
cross-instance replication belongs to the DP all-gather.

## File Map

| File | Responsibility |
|------|----------------|
| `paras/layers/paras_model.py` | Intra-node method selection, inter-node orchestration, streams, events, and peer fences |
| `paras/layers/paras_decoder_layer.py` | Layer-level transport-specific transfer interface |
| `paras/layers/paras_moe_block.py` | Expert intervals, memory-manager views, local reshards, and DP all-gather |
| `paras/weight_transfer.py` | Intra-node method enum and environment resolution |
| `paras/peer_access.py` | CUDA IPC initialization and semantic kernel wrappers |
| `paras/csrc/peer_access_transfer.cu` | Baseline v2 forward/reverse kernels |
| `paras/paras_memory_manager.py` | Overlapped EP/TP views and optional NCCL staging |
| `test/srt/paras/test_weight_transfer.py` | EP4 <-> TP4, NCCL and peer_access |
| `test/srt/paras/test_weight_transfer_tp_instances.py` | EP4 <-> two TP2 instances, both local transports and w13 layouts plus DP all-gather |
