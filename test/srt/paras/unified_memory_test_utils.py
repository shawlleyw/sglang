"""Explicit unified plans for isolated weight and cache transfer tests."""

import torch

from sglang.srt.paras.layers.utils import LayerCacheSpec
from sglang.srt.paras.mode import ParaSMode
from sglang.srt.paras.paras_memory_manager import (
    ParaSMemoryManager,
    UnifiedLayoutSpec,
    UnifiedMemoryPlan,
    UnifiedModeSpec,
)
from sglang.srt.paras.unified_layout import CacheCapacity, UnifiedLayout, align_up
from sglang.srt.paras.workspace import ModeWorkspaces, WorkspaceRequirement


def materialize_test_cache(
    mgr,
    *,
    num_layers,
    ep_max_tokens,
    tp_max_tokens,
    num_kv_heads,
    head_dim,
    kv_dtype,
    tp_size=1,
    page_size=1,
    prefix="model",
    layer_specs=None,
):
    """Use exact fixture capacities with the production unified view builder."""
    specs = layer_specs or [
        LayerCacheSpec(
            i, "full", ep_max_tokens, tp_max_tokens, num_kv_heads, head_dim, None
        )
        for i in range(num_layers)
    ]
    no_workspace = ModeWorkspaces(
        WorkspaceRequirement("test", 0), WorkspaceRequirement("test", 0)
    )
    if mgr._unified_spec is None:
        mgr._unified_spec = UnifiedLayoutSpec(
            num_layers,
            prefix,
            tp_size,
            num_kv_heads,
            num_kv_heads,
            head_dim,
            head_dim,
            UnifiedModeSpec(no_workspace, [[] for _ in specs]),
            UnifiedModeSpec(no_workspace, [[] for _ in specs]),
        )
    spec = mgr._unified_spec
    ep_tokens = tuple(s.tokens_cap_ep for s in specs)
    tp_tokens = tuple(s.tokens_cap_tp for s in specs)
    ep_bytes = tuple(
        2 * align_up((n + page_size) * num_kv_heads * head_dim * kv_dtype.itemsize)
        for n in ep_tokens
    )
    tp_bytes = tuple(
        max(
            e,
            2
            * align_up(
                (n + page_size)
                * max(1, num_kv_heads // tp_size)
                * head_dim
                * kv_dtype.itemsize
            ),
        )
        for n, e in zip(tp_tokens, ep_bytes)
    )
    front = max(spec.tp.weight_bytes, spec.ep.workspaces.size_bytes)
    tail = max(max(ep_bytes), spec.tp.workspaces.size_bytes)
    budget = front + num_layers * spec.ep.weight_bytes + sum(tp_bytes) + tail
    layout = UnifiedLayout(
        num_layers,
        budget,
        spec.ep.weight_bytes,
        spec.tp.weight_bytes,
        front,
        tail,
        CacheCapacity(ep_max_tokens, ep_tokens, ep_bytes),
        CacheCapacity(tp_max_tokens, tp_tokens, tp_bytes),
    )
    entries = mgr._plan_tensor_entries(layout, kv_dtype, page_size)
    plan = UnifiedMemoryPlan(layout, entries, kv_dtype, specs)
    mgr.materialize(plan)
    return plan


def build_weight_manager(
    rank, world_size, num_layers, num_experts, hidden, intermediate, *, with_bias=False
):
    """Declare per-mode expert views; keep transport staging outside live weights."""
    mgr = ParaSMemoryManager(device=f"cuda:{rank}")
    no_workspace = ModeWorkspaces(
        WorkspaceRequirement("test", 0), WorkspaceRequirement("test", 0)
    )
    spec = UnifiedLayoutSpec(
        num_layers,
        "model",
        world_size,
        1,
        1,
        1,
        hidden,
        UnifiedModeSpec(no_workspace),
        UnifiedModeSpec(no_workspace),
    )
    num_local = num_experts // world_size
    for mode, mode_spec in ((ParaSMode.EP, spec.ep), (ParaSMode.TP, spec.tp)):
        experts = num_local if mode == ParaSMode.EP else num_experts
        inter = intermediate if mode == ParaSMode.EP else intermediate // world_size
        for i in range(num_layers):
            names = []
            for suffix, shape, dtype in (
                ("w13_weight", (experts, 2 * inter, hidden), torch.bfloat16),
                ("w2_weight", (experts, hidden, inter), torch.bfloat16),
            ):
                name = f"model.layers.{i}.mlp.{mode.value}_experts.{suffix}"
                mgr.reserve(name, shape, dtype)
                names.append(name)
                if mode == ParaSMode.EP:
                    mgr._entries[f"model.layers.{i}.mlp.experts.{suffix}"] = (
                        mgr._entries[name]
                    )
            mode_spec.weight_names.append(names)
        mode_spec.weight_bytes = sum(
            align_up(mgr._entries[n].size_bytes) for n in mode_spec.weight_names[0]
        )
    mgr._unified_spec = spec
    staging = []
    for suffix in ("", "_1", "_2"):
        for name, shape in (
            ("w13", (num_local, 2 * intermediate, hidden)),
            ("w2", (num_local, hidden, intermediate)),
        ):
            staging.append(
                mgr.reserve(
                    f"staging.{name}_pre_permute{suffix}", shape, torch.bfloat16
                )
            )
    if with_bias:
        for i in range(num_layers):
            staging.append(
                mgr.reserve(
                    f"model.layers.{i}.mlp.experts.w13_weight_bias",
                    (num_local, 2 * intermediate),
                    torch.float32,
                )
            )
        for suffix in ("", "_1", "_2"):
            staging.append(
                mgr.reserve(
                    f"staging.w13_bias_pre_permute{suffix}",
                    (num_local, 2 * intermediate),
                    torch.float32,
                )
            )
    staging_size = sum(align_up(e.size_bytes) for e in staging)
    spec.tp.workspaces = ModeWorkspaces(
        WorkspaceRequirement("test", staging_size), no_workspace.attention
    )
    # Plan without allocating twice; patch only the fixture's staging offsets.
    ep = CacheCapacity(1, (1,) * num_layers, (512,) * num_layers)
    tail = max(512, staging_size)
    budget = (num_layers + 1) * spec.ep.weight_bytes + 2 * ep.total_bytes + tail
    layout = UnifiedLayout(
        num_layers,
        budget,
        spec.ep.weight_bytes,
        spec.tp.weight_bytes,
        spec.tp.weight_bytes,
        tail,
        ep,
        ep,
    )
    offset = budget - tail
    for entry in staging:
        entry.offset_bytes = offset
        offset += align_up(entry.size_bytes)
    entries = mgr._plan_tensor_entries(layout, torch.bfloat16, 1)
    mgr.materialize(UnifiedMemoryPlan(layout, entries, torch.bfloat16, []))
    return mgr, num_local
