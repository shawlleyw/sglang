"""Non-owning workspace bindings and checked tensor views."""

import math
from dataclasses import dataclass

import torch

from sglang.srt.paras.mode import ParaSMode
from sglang.srt.paras.unified_layout import align_up, validate_triton_workspace_blocks


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


def moe_workspace_views(buffer, shapes, dtype, device, *, block_sizes=()):
    """None retains the backend's original allocation policy."""
    if buffer is None:
        return None
    validate_triton_workspace_blocks(*block_sizes)
    return workspace_views(buffer, shapes, dtype, device, label="ParaS MoE workspace")


@dataclass(frozen=True)
class MoEWorkspace:
    """Fixed mode and scratch region belonging to one expert runner."""

    mode: ParaSMode
    buffer: torch.Tensor
