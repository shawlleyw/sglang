"""Reusable scratch for native GPT-OSS and Qwen3 MoE BF16 execution.

Reserve before KV profiling so cache capacity accounts for active-mode scratch.
ParaS owns its workspace in UMM; concurrent/unsupported paths allocate normally.
"""

import logging
import math
from dataclasses import dataclass

import torch

from sglang.srt.paras.mode import ParaSMode
from sglang.srt.paras.unified_layout import (
    align_up,
    bf16_moe_workspace_sizes,
    tp_moe_workspace_token_capacity,
)
from sglang.srt.paras.workspace import (
    MoEWorkspace,
    attention_workspace_requirements,
    workspace_views,
)

logger = logging.getLogger(__name__)


@dataclass
class StaticWorkspaces:
    mode: ParaSMode
    moe: torch.Tensor
    attention: torch.Tensor
    bound_layers: int
    attention_reuses: int = 0
    attention_fallbacks: int = 0

    def attention_views(self, shapes, dtype, device, *, zero=False):
        size = sum(align_up(math.prod(shape) * dtype.itemsize) for shape in shapes)
        if size > self.attention.numel():
            self.attention_fallbacks += 1
            return None
        views = workspace_views(
            self.attention, shapes, dtype, device, label="static attention workspace"
        )
        if zero:
            for view in views:
                view.zero_()
        self.attention_reuses += 1
        return views


def reserve_static_workspaces(runner):
    """Bind shared scratch once, or keep native allocation for other paths.

    Uses resolved context length for attention splits; aggregate prefill tokens
    need not fit in one request's context. EP scratch follows DeepEP's configured
    padded dispatch capacity, independently of the prefill scheduling budget.
    """
    existing = getattr(runner, "static_workspaces", None)
    if existing is not None:
        return existing
    args = runner.server_args
    if (
        args.enable_paras_moe
        or getattr(runner, "is_draft_worker", False)
        or runner.dtype != torch.bfloat16
        or args.quantization is not None
        or getattr(runner.model_config, "quantization", None) is not None
        or args.attention_backend not in ("triton", "flashinfer")
        or args.moe_runner_backend not in ("triton", "deep_gemm")
        or args.max_running_requests is None
        or not args.disable_overlap_schedule
        or args.enable_two_batch_overlap
        or args.enable_pdmux
        or args.speculative_algorithm is not None
        or args.pp_size != 1
        or getattr(args, "enable_torch_compile", False)
        or getattr(args, "enable_piecewise_cuda_graph", False)
        or getattr(args, "enable_memory_saver", False)
        or getattr(args, "torchao_config", "")
    ):
        return None
    config = runner.model_config.hf_config
    if config.architectures not in (
        ["GptOssForCausalLM"],
        ["Qwen3MoeForCausalLM"],
    ):
        return None
    ep = args.enable_dp_attention
    if ep:
        if args.ep_size != args.tp_size or args.dp_size != args.tp_size:
            return None
        if args.moe_a2a_backend != "deepep":
            return None
    elif args.ep_size != 1:
        return None
    mode = ParaSMode.EP if ep else ParaSMode.TP
    requirements = attention_workspace_requirements(
        args,
        config,
        args.tp_size,
        runner.model_config.head_dim,
        context_len=runner.model_config.context_len,
    )
    attention_bytes = requirements[0 if ep else 1].size_bytes
    if attention_bytes is None:
        return None
    configs = list(
        {
            id(module.moe_runner_config): module.moe_runner_config
            for module in runner.model.modules()
            if hasattr(module, "moe_runner_config")
        }.values()
    )
    if len(configs) != config.num_hidden_layers or any(
        c.paras_workspace is not None for c in configs
    ):
        return None
    tokens = tp_moe_workspace_token_capacity(
        max_prefill_tokens=args.max_prefill_tokens,
        max_running_requests=args.max_running_requests,
        chunked_prefill_size=args.chunked_prefill_size,
        cuda_graph_bs=() if args.disable_cuda_graph else args.cuda_graph_bs,
    )
    from sglang.srt.utils import get_int_env_var

    # Qwen3 MoE uses different HF field names for the same expert geometry.
    qwen = config.architectures == ["Qwen3MoeForCausalLM"]
    ep_bytes, tp_bytes = bf16_moe_workspace_sizes(
        hidden_size=config.hidden_size,
        intermediate_size=(
            config.moe_intermediate_size if qwen else config.intermediate_size
        ),
        num_experts=config.num_experts if qwen else config.num_local_experts,
        top_k=config.num_experts_per_tok,
        tp_size=args.tp_size,
        dispatch_capacity=get_int_env_var(
            "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK", 128
        ),
        tp_input_tokens=tokens,
    )
    state = StaticWorkspaces(
        mode,
        torch.empty(
            ep_bytes if ep else tp_bytes, dtype=torch.uint8, device=runner.device
        ),
        torch.empty(attention_bytes, dtype=torch.uint8, device=runner.device),
        len(configs),
    )
    binding = MoEWorkspace(mode, state.moe)
    for config in configs:
        config.paras_workspace = binding
    runner.static_workspaces = state
    logger.info(
        "Reserved static %s scratch before KV sizing: MoE %.3f MiB, attention %.3f MiB "
        "(runtime context %d, prefill budget %d)",
        mode.value,
        state.moe.numel() / 2**20,
        state.attention.numel() / 2**20,
        runner.model_config.context_len,
        args.max_prefill_tokens,
    )
    return state
