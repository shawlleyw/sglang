"""UMM weight views and collective reference transformations for microbenchmarks.

The real address planner is used with zero inference workspace and a minimal,
unused KV reservation. This is transfer geometry, not a serving memory budget.
"""

import torch

from sglang.srt.paras.mode import ParaSMode
from sglang.srt.paras.paras_memory_manager import (
    ParaSMemoryManager,
    UnifiedLayoutSpec,
    UnifiedMemoryPlan,
    UnifiedModeSpec,
)
from sglang.srt.paras.unified_layout import align_up, plan_unified_layout
from sglang.srt.paras.workspace import ModeWorkspaces, WorkspaceRequirement

COMPONENTS = ("w13", "w2", "qkv", "o")


def verification_indices(numel, device, count=4096):
    """Sample endpoints/interiors without FP32 rounding beyond large tensors."""
    if numel <= 0 or count <= 0:
        raise ValueError("Tensor size and sample count must be positive")
    samples = min(count, numel)
    positions = torch.arange(samples, device=device, dtype=torch.int64)
    stratified = positions * (numel - 1) // max(samples - 1, 1)
    return torch.cat((stratified, positions))


def weight_name(layer, mode, component):
    prefix = f"model.layers.{layer}"
    if component in ("w13", "w2"):
        return f"{prefix}.mlp.{mode.value}_experts.{component}_weight"
    projection = "qkv_proj" if component == "qkv" else "o_proj"
    suffix = "weight" if mode == ParaSMode.EP else "tp_weight"
    return f"{prefix}.self_attn.{projection}.{suffix}"


def make_manager(model, world, device, *, materialize=True):
    if world <= 1 or model.num_experts % world or model.moe_intermediate_size % world:
        raise ValueError(
            "Requires matching EP/TP > 1 and divisible experts/intermediate size"
        )
    if model.num_attention_heads <= 0 or model.num_attention_heads % world:
        raise ValueError("--num-attention-heads must be positive and divisible by TP")
    if (
        world % model.num_kv_heads
        if world >= model.num_kv_heads
        else model.num_kv_heads % world
    ):
        raise ValueError("KV heads must divide TP or be divisible by TP")
    if model.num_hidden_layers <= 0:
        raise ValueError("Layer count must be positive")
    manager = ParaSMemoryManager(device=device)
    workspace = ModeWorkspaces(
        WorkspaceRequirement("microbench", 0), WorkspaceRequirement("microbench", 0)
    )
    spec = UnifiedLayoutSpec(
        model.num_hidden_layers,
        "model",
        world,
        model.num_attention_heads,
        model.num_kv_heads,
        model.head_dim,
        model.hidden_size,
        UnifiedModeSpec(workspace),
        UnifiedModeSpec(workspace),
    )
    for mode in (ParaSMode.EP, ParaSMode.TP):
        e = model.num_experts // world if mode == ParaSMode.EP else model.num_experts
        i = (
            model.moe_intermediate_size
            if mode == ParaSMode.EP
            else model.moe_intermediate_size // world
        )
        q = (
            model.num_attention_heads
            * model.head_dim
            // (1 if mode == ParaSMode.EP else world)
        )
        kv = (
            model.num_kv_heads * model.head_dim
            if mode == ParaSMode.EP
            else max(1, model.num_kv_heads // world) * model.head_dim
        )
        shapes = (
            (e, 2 * i, model.hidden_size),
            (e, model.hidden_size, i),
            (q + 2 * kv, model.hidden_size),
            (model.hidden_size, q),
        )
        for layer in range(model.num_hidden_layers):
            names = []
            for component, shape in zip(COMPONENTS, shapes):
                name = weight_name(layer, mode, component)
                manager.reserve(name, shape, torch.bfloat16)
                names.append(name)
            spec.for_mode(mode).weight_names.append(names)
        spec.for_mode(mode).weight_bytes = sum(
            align_up(manager._entries[n].size_bytes) for n in names
        )
    manager._unified_spec = spec
    ep_row = 2 * model.num_kv_heads * model.head_dim * model.elem_size
    tp_row = 2 * max(1, model.num_kv_heads // world) * model.head_dim * model.elem_size
    layout = plan_unified_layout(
        num_layers=model.num_hidden_layers,
        budget=model.num_hidden_layers * spec.ep.weight_bytes
        + spec.tp.weight_bytes
        + model.num_hidden_layers * align_up(4 * ep_row, 512),
        ep_weight_bytes=spec.ep.weight_bytes,
        tp_weight_bytes=spec.tp.weight_bytes,
        ep_workspace_bytes=0,
        tp_workspace_bytes=0,
        ep_kv_row_bytes=ep_row,
        tp_kv_row_bytes=tp_row,
    )
    plan = UnifiedMemoryPlan(
        layout,
        manager._plan_tensor_entries(layout, torch.bfloat16, 1),
        torch.bfloat16,
        [],
    )
    if materialize:
        manager.materialize(plan)
    return manager, plan


def expert_ep_packed_view(tensor, model, world, component):
    """View/permutation only; caller times the copy into contiguous NCCL staging."""
    e, h, i = (
        model.num_experts // world,
        model.hidden_size,
        model.moe_intermediate_size // world,
    )
    if component == "w13":
        gates = 1 if model.interleaved_w13 else 2
        chunk = (2 if model.interleaved_w13 else 1) * i * h
        return tensor.view(e, gates, world, chunk).permute(2, 0, 1, 3)
    return tensor.view(e, h, world, i).permute(2, 0, 1, 3)


def attention_slice(ep, tp, model, world, rank, component):
    q = model.num_attention_heads * model.head_dim
    kv = model.num_kv_heads * model.head_dim
    qs, ks = q // world, max(model.head_dim, kv // world)
    owner = rank // max(1, world // model.num_kv_heads)
    if component == "qkv":
        tp[:qs].copy_(ep[rank * qs : (rank + 1) * qs])
        tp[qs : qs + ks].copy_(ep[q + owner * ks : q + (owner + 1) * ks])
        tp[qs + ks :].copy_(ep[q + kv + owner * ks : q + kv + (owner + 1) * ks])
    else:
        tp.copy_(ep[:, rank * qs : (rank + 1) * qs])


def attention_restore(gathered, ep, model, world, component):
    """Unpack rank-major all-gather; skip duplicate K/V replicas."""
    q, kv = (
        model.num_attention_heads * model.head_dim,
        model.num_kv_heads * model.head_dim,
    )
    qs, ks = q // world, max(model.head_dim, kv // world)
    replica = max(1, world // model.num_kv_heads)
    for rank in range(world):
        if component == "o":
            ep[:, rank * qs : (rank + 1) * qs].copy_(gathered[rank])
        else:
            ep[rank * qs : (rank + 1) * qs].copy_(gathered[rank, :qs])
            if rank % replica == 0:
                owner = rank // replica
                ep[q + owner * ks : q + (owner + 1) * ks].copy_(
                    gathered[rank, qs : qs + ks]
                )
                ep[q + kv + owner * ks : q + kv + (owner + 1) * ks].copy_(
                    gathered[rank, qs + ks :]
                )


def reference_values(indices, model, world, rank, mode, component, layer):
    """Coordinate-sensitive, exactly representable BF16 oracle, without a full copy."""
    h, inter = model.hidden_size, model.moe_intermediate_size
    q, kv = (
        model.num_attention_heads * model.head_dim,
        model.num_kv_heads * model.head_dim,
    )
    ip, qs, ks = inter // world, q // world, max(model.head_dim, kv // world)
    x = indices
    if component == "w13":
        local_inter = inter if mode == ParaSMode.EP else ip
        expert, rest = x // (2 * local_inter * h), x % (2 * local_inter * h)
        row, col = rest // h, rest % h
        if mode == ParaSMode.EP:
            expert = expert + rank * (model.num_experts // world)
        elif model.interleaved_w13:
            row = row + rank * 2 * ip
        else:
            row = row // ip * inter + rank * ip + row % ip
        x = (expert * 2 * inter + row) * h + col
    elif component == "w2":
        local_inter = inter if mode == ParaSMode.EP else ip
        expert, rest = x // (h * local_inter), x % (h * local_inter)
        row, col = rest // local_inter, rest % local_inter
        if mode == ParaSMode.EP:
            expert = expert + rank * (model.num_experts // world)
        else:
            col = col + rank * ip
        x = (expert * h + row) * inter + col
    elif mode == ParaSMode.TP:
        if component == "o":
            x = x // qs * q + rank * qs + x % qs
        else:
            row, col = x // h, x % h
            owner = rank // max(1, world // model.num_kv_heads)
            row = torch.where(
                row < qs,
                row + rank * qs,
                q + (row - qs) // ks * kv + owner * ks + (row - qs) % ks,
            )
            x = row * h + col
    # Mix coarse and fine coordinates, including gate/expert boundaries.
    return (
        (
            x % 127
            + (x // 127) % 113
            + (x // 14351) % 97
            + layer * 7
            + COMPONENTS.index(component) * 11
        )
        % 251
    ).to(torch.bfloat16)
