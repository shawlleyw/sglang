"""Fresh-storage weight transfers for empty-state reconfiguration baselines.

These helpers never write into inactive UMM views. The caller owns source
lifetime, module rebinding, layer ordering, synchronization, and graph capture.
Host snapshots must already contain the target-mode weights and be prepared
outside the interval measuring a host reload.
"""

from collections.abc import Mapping

import torch
import torch.distributed as dist

from common.weight_bundle import (
    COMPONENTS,
    attention_restore,
    attention_slice,
    expert_ep_packed_view,
)

_DIRECTIONS = ("ep_to_tp", "tp_to_ep")


def _validate_dimensions(model, world, rank=None):
    if not isinstance(world, int) or isinstance(world, bool) or world <= 1:
        raise ValueError("world must be an integer greater than one")
    if rank is not None and (
        not isinstance(rank, int) or isinstance(rank, bool) or not 0 <= rank < world
    ):
        raise ValueError("rank must be an integer in [0, world)")
    for name in (
        "num_experts",
        "hidden_size",
        "moe_intermediate_size",
        "num_attention_heads",
        "num_kv_heads",
        "head_dim",
    ):
        value = getattr(model, name, None)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"model.{name} must be a positive integer")
    for name in ("num_experts", "moe_intermediate_size", "num_attention_heads"):
        if getattr(model, name) % world:
            raise ValueError(f"model.{name} must be divisible by world")
    if model.num_attention_heads % model.num_kv_heads:
        raise ValueError("Q heads must be divisible by KV heads")
    if (
        world % model.num_kv_heads
        if world >= model.num_kv_heads
        else model.num_kv_heads % world
    ):
        raise ValueError("KV heads must divide world or be divisible by world")
    if getattr(model, "num_gates", 2) != 2 or getattr(model, "elem_size", 2) != 2:
        raise ValueError("Reconfiguration baselines require two-gate BF16 weights")
    if not isinstance(getattr(model, "interleaved_w13", None), bool):
        raise ValueError("model.interleaved_w13 must explicitly select gate layout")


def _mode_shapes(model, world, mode):
    ep = mode == "ep"
    experts = model.num_experts // world if ep else model.num_experts
    intermediate = (
        model.moe_intermediate_size if ep else model.moe_intermediate_size // world
    )
    q = model.num_attention_heads * model.head_dim // (1 if ep else world)
    kv = (
        model.num_kv_heads * model.head_dim
        if ep
        else max(1, model.num_kv_heads // world) * model.head_dim
    )
    return {
        "w13": (experts, 2 * intermediate, model.hidden_size),
        "w2": (experts, model.hidden_size, intermediate),
        "qkv": (q + 2 * kv, model.hidden_size),
        "o": (model.hidden_size, q),
    }


def target_shapes(model, world, direction):
    """Shapes for one target layer; model follows common.model_configs.ModelConfig."""
    _validate_dimensions(model, world)
    if direction not in _DIRECTIONS:
        raise ValueError(f"direction must be one of {_DIRECTIONS}")
    return _mode_shapes(model, world, direction[-2:])


def _validate_layer(tensors, shapes, *, cpu_only=False):
    if not isinstance(tensors, Mapping) or set(tensors) != set(COMPONENTS):
        raise ValueError(f"A layer must contain exactly {COMPONENTS}")
    devices = set()
    for name, shape in shapes.items():
        tensor = tensors[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"{name}: expected shape {shape}, got {tuple(tensor.shape)}"
            )
        if tensor.dtype != torch.bfloat16:
            raise ValueError(f"{name}: expected BF16 weights, got {tensor.dtype}")
        if tensor.layout != torch.strided or not tensor.is_contiguous():
            raise ValueError(f"{name}: expected contiguous dense weights")
        if cpu_only and tensor.device.type != "cpu":
            raise ValueError(f"{name}: host reload snapshots must reside on CPU")
        devices.add(tensor.device)
    if len(devices) != 1:
        raise ValueError("All source weights in a layer must use the same device")


@torch.no_grad()
def snapshot_host_tensors(tensors, *, pin_memory=False):
    """Take independent CPU copies of target weights or arbitrary auxiliary tensors.

    This is setup work, not part of reload timing. Even CPU inputs are copied,
    so later rebinding or source mutation cannot invalidate the snapshot.
    """
    snapshot = {}
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
            raise TypeError(f"{name}: expected a dense tensor")
        host = torch.empty(
            tuple(tensor.shape),
            dtype=tensor.dtype,
            device="cpu",
            pin_memory=pin_memory,
        )
        # Blocking D2H guarantees the snapshot is ready before timing starts.
        host.copy_(tensor.detach(), non_blocking=False)
        snapshot[name] = host
    return snapshot


@torch.no_grad()
def reload_host_tensors(snapshot, *, device, non_blocking=False):
    """Allocate new destination storage and copy each host tensor into it.

    Includes both empty() allocation and copy_(). With non_blocking=True the
    caller must synchronize the destination stream before stopping its timer
    or releasing pinned snapshots. Dtypes/shapes of auxiliary tensors survive.
    """
    for name, host in snapshot.items():
        if not isinstance(host, torch.Tensor) or host.device.type != "cpu":
            raise ValueError(f"{name}: expected a CPU snapshot tensor")
        if host.layout != torch.strided:
            raise ValueError(f"{name}: expected a dense snapshot tensor")
    target = {}
    for name, host in snapshot.items():
        tensor = torch.empty(tuple(host.shape), dtype=host.dtype, device=device)
        tensor.copy_(host, non_blocking=non_blocking)
        target[name] = tensor
    return target


def reload_host_layer(
    snapshot, model, world, rank, direction, *, device, non_blocking=False
):
    """Validate a prepared target-mode host layer, then allocate/copy its weights."""
    _validate_dimensions(model, world, rank)
    shapes = target_shapes(model, world, direction)
    _validate_layer(snapshot, shapes, cpu_only=True)
    return reload_host_tensors(snapshot, device=device, non_blocking=non_blocking)


@torch.no_grad()
def reload_host_model(snapshot, model, world, rank, direction, *, device):
    """Move all prepared bulk weights with one real nn.Module.to() call.

    The temporary module contains only target-mode expert/attention parameters.
    Parameter wrappers share the pinned CPU snapshots until conversion; changing
    their storage does not change the snapshots. The caller releases all source
    GPU weights first and rebinds the result into the runtime's existing objects.
    CPU destinations are rejected: Module.to(cpu) would be a no-op, not reload.
    """
    device = torch.device(device)
    if device.type == "cpu":
        raise ValueError("Module.to reload requires a non-CPU destination")
    _validate_dimensions(model, world, rank)
    shapes = target_shapes(model, world, direction)
    target_model = torch.nn.ModuleDict()
    for index, layer in snapshot.items():
        _validate_layer(layer, shapes, cpu_only=True)
        parameters = torch.nn.ParameterDict()
        for name, tensor in layer.items():
            parameters[name] = torch.nn.Parameter(tensor, requires_grad=False)
        target_model[str(index)] = parameters
    target_model.to(device=device, non_blocking=True)
    return {index: dict(target_model[str(index)].items()) for index in snapshot}


@torch.no_grad()
def naive_nccl_transfer_layer(
    source, model, world, rank, direction, *, group=None, collectives=dist
):
    """Pack/exchange/unpack into fresh target tensors for one complete layer.

    `source` maps w13/w2/qkv/o to source-mode tensors. No source tensor or
    mapping is modified. Collectives use the provided rank-local group;
    `collectives` is injectable solely to test the same path with CPU oracles.
    Returns a new tensor mapping, without rebinding any runtime module.

    CUDA collectives/copies follow PyTorch current-stream semantics. The caller
    must finish the transfer before timing completion or freeing source storage.
    This intentionally sequential baseline allocates staging inside the call.
    """
    _validate_dimensions(model, world, rank)
    shapes = target_shapes(model, world, direction)
    _validate_layer(source, _mode_shapes(model, world, direction[:2]))
    device = source["w13"].device
    target = {}
    for component in COMPONENTS:
        output = torch.empty(shapes[component], dtype=torch.bfloat16, device=device)
        target[component] = output
        if component in ("w13", "w2"):
            if direction == "ep_to_tp":
                packed = expert_ep_packed_view(
                    source[component], model, world, component
                )
                send = torch.empty(
                    tuple(packed.shape), dtype=packed.dtype, device=device
                )
                send.copy_(packed)
                collectives.all_to_all_single(
                    output.view(-1), send.view(-1), group=group
                )
                del send, packed
            else:
                unpacked = expert_ep_packed_view(output, model, world, component)
                received = torch.empty(
                    tuple(unpacked.shape), dtype=output.dtype, device=device
                )
                collectives.all_to_all_single(
                    received.view(-1), source[component].view(-1), group=group
                )
                unpacked.copy_(received)
                del received, unpacked
        elif direction == "ep_to_tp":
            attention_slice(source[component], output, model, world, rank, component)
        else:
            shards = torch.empty(
                (world, *source[component].shape), dtype=output.dtype, device=device
            )
            collectives.all_gather_into_tensor(
                shards.view(-1), source[component].view(-1), group=group
            )
            attention_restore(shards, output, model, world, component)
            del shards
    return target
