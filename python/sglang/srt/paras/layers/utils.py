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


def paras_load_tp_experts_weight(params_dict, name, loaded_weight, shard_id, expert_id):
    """
    Load expert weights into tp_experts during model weight loading.
    Call this inside load_weights() after loading into ep_experts.

    Replaces 'experts' with 'tp_experts' in the param name and loads
    the weight if the tp_experts param exists in params_dict.

    Usage example inside load_weights():
        weight_loader(param, loaded_weight, name, shard_id=shard_id, expert_id=expert_id)
        if get_global_server_args().enable_paras_moe:
            paras_load_tp_experts_weight(params_dict, name, loaded_weight, shard_id, expert_id)

    Integration pattern for CausalLM classes:
        The CausalLM class should define:
          - paras_configure_helper(): torch.cuda.synchronize(); paras_weight_buffer.release_all()
          - @paras_func paras_configure_tp(paras_tp_size, paras_tp_rank): self.model.paras_configure_tp(...)
          - @paras_func paras_configure_ep(): self.model.paras_configure_ep()
    """
    tp_experts_name = name.replace("experts", "tp_experts")
    if tp_experts_name in params_dict:
        param = params_dict[tp_experts_name]
        # Skip if param is backed by the memory manager — its buffer is shared
        # with EP experts and will be filled in-place during the EP→TP switch.
        from sglang.srt.paras.paras_memory_manager import get_global_paras_memory_manager
        mgr = get_global_paras_memory_manager()
        if mgr is not None and mgr.is_managed(param):
            return
        weight_loader = param.weight_loader
        weight_loader(param, loaded_weight, tp_experts_name, shard_id=shard_id, expert_id=expert_id)
