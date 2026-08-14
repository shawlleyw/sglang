# Direct Expert Weight Transfer for ParaS

## Scope

ParaS switches expert weights between a wide expert-parallel layout and either
one tensor-parallel group or multiple data-parallel replicas of a TP group. The
runtime exposes two weight-transfer methods through `PARAS_CONFIGURE_METHOD`:

- `direct` (default): topology-aware CUDA IPC kernels, with a cross-node DP
  all-gather where required.
- `nccl`: a sequential `all_to_all_single` fallback for `dp_size=1`.

Unknown method names fail during memory planning. The removed NCCL overlap
method has no compatibility alias or staging-buffer reservations.

## Tensor Layouts

Let `G = dp_size`, `T = tp_size`, `E` be the global expert count, and
`L = E/(G*T)`. Each wide-EP rank owns `L` experts with the full intermediate
dimension. Each DPxTP rank owns all `E` experts, but only its `I/T`
intermediate shard.

The four-anchor memory layout registers separate EP and TP views for every
layer. EP->TP processes layers in reverse order; TP->EP processes them in
forward order. Those orders prevent a destination layer from overwriting a
source layer that has not yet been consumed. See
[`unified_memory_epdptp.md`](unified_memory_epdptp.md) for the address proof.

## Direct Strategy Matrix

### EP -> TP, `G = 1`

The baseline v2 kernel reads each local EP expert, selects the intermediate
shard for every TP peer, and writes it directly into that peer's TP view.

- w13: `peer_access_fused_transfer_w13_v2`
- w2: `peer_access_fused_transfer_w2_v2`

### EP -> DPxTP, node-local EP group

The DPTP kernels address all `G*T` ranks through one node-local IPC mapping.
Each EP shard is read once and broadcast into the corresponding expert range
of every DP replica.

- w13: `peer_access_fused_transfer_w13_dptp`
- w2: `peer_access_fused_transfer_w2_dptp`

### EP -> DPxTP, multi-node EP group

Each node first performs a TP-local IPC reshard into its canonical expert
interval `[d*E/G, (d+1)*E/G)`. A strided DP group then runs an in-place
all-gather from that interval into the full TP tensor.

The model uses two CUDA streams:

```text
IPC stream:  reshard N-1 | reshard N-2 | reshard N-3
DP stream:                 gather N-1  | gather N-2
```

For each layer, a TP-group all-reduce fences all peer writes. A CUDA event then
makes the DP stream wait for that layer's node-local interval. The next layer's
IPC reshard can overlap the current layer's NIC collective because they use
disjoint layer regions and different links.

The DP all-gather input is already located at the calling DP rank's output
offset. This is NCCL's in-place all-gather layout and needs no staging buffer.

### TP -> EP, `G = 1`

The reverse v2 kernels assemble each rank's full EP experts from the
intermediate shards stored on its TP peers.

- w13: `peer_access_fused_transfer_w13_ep`
- w2: `peer_access_fused_transfer_w2_ep`

### DPxTP -> EP, node-local EP group

The reverse DPTP kernels reconstruct the EP shard owned by every rank in the
wide node-local group.

### DPxTP -> EP, multi-node EP group

TP weights are replicated across DP groups, so no cross-node collective is
needed. Node `d` ignores experts outside `[d*E/G, (d+1)*E/G)`, shifts the TP
source base to that interval, and runs the baseline reverse kernel across its
`T` local peers. Each EP rank receives exactly `L` full experts.

## Synchronization

Direct kernels write remote GPU memory, so CUDA stream ordering on one rank is
not enough. After every layer, the model issues an all-reduce on the physical
peer group:

- EP group for a node-local wide-EP transfer.
- TP group for a multi-node transfer.

The collective is ordered after the local kernel and does not complete until
every peer has submitted its writes. This fences the current layer before the
next layer can reuse overlapping four-anchor regions. The multi-node forward
path records a CUDA event after that fence for the DP all-gather stream.

## NCCL Fallback

The `nccl` method supports `dp_size=1` only. EP->TP permutes each local EP
weight into `staging.w13_pre_permute` or `staging.w2_pre_permute`, then runs
`all_to_all_single` directly into the TP view. TP->EP runs the inverse
all-to-all and permutation. There is one staging set and one sequential layer
walk; no overlap method or dual staging suffixes remain.

## Kernel Versions

The production Python path selects v2 for the baseline local reshard. The v3
reverse mapping can select a multi-node node-owned interval through its source
offset and expert count, but its compiled presets support only `T in {4, 8}`
and BF16 model presets. The v3 forward mapping cannot consume a DP-gathered
expert order without a `G`-aware mapping. See the DPxTP status section in
[`unified_memory_epdptp.md`](unified_memory_epdptp.md).

## File Map

| File | Responsibility |
|------|----------------|
| `paras/layers/paras_model.py` | Method selection, topology, layer order, streams, events, and barriers |
| `paras/layers/paras_decoder_layer.py` | Thin layer-level transfer interface |
| `paras/layers/paras_moe_block.py` | Tensor views and topology-specific transfer primitives |
| `paras/peer_access.py` | CUDA IPC initialization and Python kernel wrappers |
| `paras/csrc/peer_access_transfer.cu` | Baseline v2 forward/reverse kernels |
| `paras/csrc/kernels_dptp.cu` | Node-local DPxTP forward/reverse kernels |
| `paras/paras_memory_manager.py` | Four-anchor EP/TP views and optional NCCL staging |
| `test/srt/paras/test_weight_transfer.py` | EP4 <-> TP4 NCCL/direct correctness |
| `test/srt/paras/test_weight_transfer_dptp.py` | EP4 <-> DP2xTP2 node-local and logical multi-node correctness |
