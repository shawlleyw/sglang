"""
Utilities for ParaS layer mixins.
"""

import torch


class ParaSWeightBuffer:
    def __init__(self):
        self.buffer: dict[int, list[torch.Tensor]] = {}

    def has(self, numel: int):
        return numel in self.buffer and len(self.buffer[numel]) > 0

    def get_buffer_like(self, tensor: torch.Tensor):
        numel = tensor.numel()
        if numel in self.buffer and len(self.buffer[numel]) > 0:
            return self.buffer[numel].pop()
        else:
            return torch.empty_like(tensor)

    def get_buffer(self, shape, dtype, device):
        numel = 1
        for dim in shape:
            numel *= dim
        if numel in self.buffer and len(self.buffer[numel]) > 0:
            # NOTE(shaoyuw): this assumes that all tensors have the same dtype and device
            tensor = self.buffer[numel].pop()
            return tensor.view(shape)
        else:
            return torch.empty(shape, dtype=dtype, device=device)

    def put(self, tensor: torch.Tensor):
        numel = tensor.numel()
        if numel not in self.buffer:
            self.buffer[numel] = []
        self.buffer[numel].append(tensor)

    def release_all(self):
        self.buffer.clear()
        torch.cuda.empty_cache()


paras_weight_buffer = ParaSWeightBuffer()
