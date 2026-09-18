"""Workspace requirements and non-owning tensor views for the unified memory planner.

Numerical attention and MoE scratch occupy disjoint regions. Backend planning
metadata and outputs that escape an operator are not part of these regions.
"""

import math
from dataclasses import dataclass

import torch

from sglang.srt.paras.mode import ParaSMode
from sglang.srt.paras.unified_layout import align_up


@dataclass(frozen=True)
class WorkspaceRequirement:
    backend: str
    # None means externally allocated, not a zero-workspace implementation.
    size_bytes: int | None

    def __post_init__(self):
        if self.size_bytes is not None and self.size_bytes < 0:
            raise ValueError("Workspace size cannot be negative")

    @property
    def reserved_bytes(self) -> int:
        return align_up(self.size_bytes) if self.size_bytes is not None else 0


@dataclass(frozen=True)
class ModeWorkspaces:
    moe: WorkspaceRequirement
    attention: WorkspaceRequirement

    @property
    def size_bytes(self) -> int:
        return self.moe.reserved_bytes + self.attention.reserved_bytes

    def region(self, kind: str) -> tuple[int, int]:
        if kind == "moe":
            return 0, self.moe.reserved_bytes
        if kind == "attention":
            return self.moe.reserved_bytes, self.attention.reserved_bytes
        raise ValueError(kind)


def flashinfer_workspace_size(architectures, configured_bytes, deterministic):
    if deterministic:
        return 2048 << 20
    if any(
        arch in architectures
        for arch in ("Qwen2ForCausalLM", "Qwen3ForCausalLM", "MiMoForCausalLM")
    ):
        return 512 << 20
    return configured_bytes


def triton_attention_workspace_size(tokens, heads, splits, head_dim):
    # These are separate FP32 partial-output and LSE tensors.
    return align_up(tokens * heads * splits * head_dim * 4) + align_up(
        tokens * heads * splits * 4
    )


def triton_attention_split_config(server_args, context_len):
    """Return (maximum splits, tile size) using the resolved model context."""
    from sglang.srt.utils import get_int_env_var

    tile = server_args.triton_attention_split_tile_size
    if server_args.enable_deterministic_inference:
        tile = get_int_env_var("SGLANG_TRITON_DECODE_SPLIT_TILE_SIZE", 256)
    splits = server_args.triton_attention_num_kv_splits
    if tile is not None:
        if context_len is None or context_len <= 0:
            raise ValueError(
                "Triton workspace sizing requires the resolved context length"
            )
        splits = (context_len + tile - 1) // tile
    return splits, tile


def attention_workspace_requirements(
    server_args, config, tp_size, head_dim, *, context_len
):
    """Resolve supported numerical scratch before constructing the backend.

    Preserve the existing graph runner's max(EP, TP) allocation capacity in both
    modes. Concurrent/composite backends remain external until they can declare
    independent workspace lifetimes. This does not change their allocation policy.
    """
    from sglang.srt.environ import envs

    backend = server_args.attention_backend
    external = WorkspaceRequirement(backend, None)
    if (
        server_args.enable_two_batch_overlap
        or server_args.enable_pdmux
        or server_args.speculative_algorithm is not None
        or server_args.prefill_attention_backend not in (None, backend)
        or server_args.decode_attention_backend not in (None, backend)
    ):
        return external, external
    if backend == "flashinfer":
        size = flashinfer_workspace_size(
            config.architectures,
            envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get(),
            server_args.enable_deterministic_inference,
        )
        return (WorkspaceRequirement(backend, size),) * 2
    if backend != "triton" or server_args.max_running_requests is None:
        return external, external

    splits, _ = triton_attention_split_config(server_args, context_len)
    graph_tokens = 0
    if not server_args.disable_cuda_graph:
        graph_tokens = max(
            server_args.cuda_graph_bs + (server_args.paras_tp_cuda_graph_bs or [])
        )
    ep_tokens = max(graph_tokens, server_args.max_running_requests // tp_size)
    tp_tokens = max(graph_tokens, server_args.max_running_requests)
    ep_heads = config.num_attention_heads
    tp_heads = ep_heads // tp_size
    ep = triton_attention_workspace_size(ep_tokens, ep_heads, splits, head_dim)
    tp = triton_attention_workspace_size(tp_tokens, tp_heads, splits, head_dim)
    return WorkspaceRequirement(backend, ep), WorkspaceRequirement(backend, tp)


def workspace_views(buffer, shapes, dtype, device, *, label="workspace"):
    """Build aligned typed views within a contiguous uint8 scratch region."""
    requested_device = torch.device(device)
    if requested_device.type == "cuda" and requested_device.index is None:
        requested_device = torch.device("cuda", torch.cuda.current_device())
    if requested_device != buffer.device:
        raise RuntimeError(f"{label} requested on a different device")
    cursor = 0
    result = []
    for shape in shapes:
        size = math.prod(shape) * dtype.itemsize
        cursor = align_up(cursor)
        if cursor + size > buffer.numel():
            raise RuntimeError(f"{label} overflow: {cursor + size} > {buffer.numel()}")
        result.append(buffer[cursor : cursor + size].view(dtype).view(shape))
        cursor += size
    return result


def moe_workspace_views(buffer, intermediate_shape, activation_shape, dtype, device):
    """Return (intermediate, activation) views, or (None, None) for dynamic allocation."""
    shapes = (intermediate_shape, activation_shape)
    size = sum(align_up(math.prod(shape) * dtype.itemsize) for shape in shapes)
    if buffer is None or size > buffer.numel():
        return None, None
    intermediate, activation = workspace_views(
        buffer, shapes, dtype, device, label="ParaS MoE workspace"
    )
    return intermediate, activation


def workspace_or_empty(view, shape, *, dtype, device):
    """Reuse a workspace view, allocating fallback storage only when called."""
    return view if view is not None else torch.empty(shape, dtype=dtype, device=device)


@dataclass(frozen=True)
class MoEWorkspace:
    """Fixed mode and scratch region belonging to one expert runner."""

    mode: ParaSMode
    buffer: torch.Tensor
