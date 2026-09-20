"""Opt-in evaluation helpers for native EP/TP with ParaS-sized static scratch.

Call reserve_baseline_workspaces after loading weights and before KV profiling,
then bind_baseline_attention after creating the attention backend, before graph
capture. Normal server launches are unchanged. This matches scratch ownership,
not ParaS's additional transfer headroom or inactive-mode/communication state.
"""

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


@dataclass
class BaselineWorkspaces:
    mode: ParaSMode
    moe: torch.Tensor
    attention: torch.Tensor
    bound_layers: int
    attention_reuses: int = 0
    attention_fallbacks: int = 0


def reserve_baseline_workspaces(runner):
    """Allocate once so native KV profiling sees the same scratch reservation."""
    args = runner.server_args
    if args.enable_paras_moe:
        raise ValueError("ParaS already reserves managed workspace inside UMM")
    if hasattr(runner, "matched_baseline_workspaces"):
        raise RuntimeError("Baseline workspace was already reserved")
    if (
        runner.dtype != torch.bfloat16
        or args.quantization is not None
        or args.attention_backend != "triton"
        or args.moe_runner_backend != "triton"
        or not args.disable_overlap_schedule
        or args.enable_two_batch_overlap
        or args.enable_pdmux
        or args.speculative_algorithm is not None
        or args.pp_size != 1
    ):
        raise ValueError(
            "Matched workspace evaluation requires serial BF16 Triton EP/TP"
        )
    config = runner.model_config.hf_config
    if config.architectures != ["GptOssForCausalLM"]:
        raise ValueError("This evaluation helper supports GPT-OSS")
    ep = args.enable_dp_attention
    if (ep and args.ep_size != args.tp_size) or (not ep and args.ep_size != 1):
        raise ValueError("Expected native EP with DP attention, or native TP")
    mode = ParaSMode.EP if ep else ParaSMode.TP
    graph_bs = () if args.disable_cuda_graph else args.cuda_graph_bs
    tokens = tp_moe_workspace_token_capacity(
        max_prefill_tokens=args.max_prefill_tokens,
        max_running_requests=args.max_running_requests,
        chunked_prefill_size=args.chunked_prefill_size,
        cuda_graph_bs=graph_bs,
    )
    from sglang.srt.utils import get_int_env_var

    ep_bytes, tp_bytes = bf16_moe_workspace_sizes(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_experts=config.num_local_experts,
        top_k=config.num_experts_per_tok,
        tp_size=args.tp_size,
        dispatch_capacity=get_int_env_var(
            "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK", 128
        ),
        tp_input_tokens=tokens,
    )
    requirements = attention_workspace_requirements(
        args,
        config,
        args.tp_size,
        config.head_dim,
        context_len=runner.model_config.context_len,
    )
    attention_bytes = requirements[0 if ep else 1].size_bytes
    if attention_bytes is None:
        raise ValueError("Attention workspace must have a concrete reservation")
    # The layer and its quantization method can expose the same config.
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
        raise ValueError(
            f"Expected {config.num_hidden_layers} unbound native MoE configs, "
            f"found {len(configs)}"
        )
    state = BaselineWorkspaces(
        mode,
        torch.zeros(
            ep_bytes if ep else tp_bytes, dtype=torch.uint8, device=runner.device
        ),
        torch.zeros(attention_bytes, dtype=torch.uint8, device=runner.device),
        len(configs),
    )
    binding = MoEWorkspace(mode, state.moe)
    for config in configs:
        config.paras_workspace = binding
    runner.matched_baseline_workspaces = state
    return state


def bind_baseline_attention(runner):
    """Reuse pre-profiled attention scratch for eager execution and capture."""
    state = runner.matched_baseline_workspaces
    backend = runner.attn_backend
    if getattr(backend, "_paras_memory_manager", None) is not None:
        raise ValueError("Native baseline unexpectedly owns a ParaS memory manager")
    original = backend._allocate_decode_workspace

    def allocate(tokens, *, zero=False):
        shapes = [
            (tokens, backend.num_head, backend.max_kv_splits, backend.v_head_dim),
            (tokens, backend.num_head, backend.max_kv_splits),
        ]
        import math

        size = sum(align_up(math.prod(shape) * 4) for shape in shapes)
        if size > state.attention.numel():
            state.attention_fallbacks += 1
            return original(tokens, zero=zero)
        state.attention_reuses += 1
        views = workspace_views(
            state.attention,
            shapes,
            torch.float32,
            backend.device,
            label="matched baseline attention",
        )
        if zero:
            for view in views:
                view.zero_()
        return views

    backend._allocate_decode_workspace = allocate


def install():
    """Install opt-in hooks before launching the server, including spawn workers."""
    import functools

    from sglang.srt.model_executor.model_runner import ModelRunner

    if getattr(ModelRunner, "_matched_baseline_hooks_installed", False):
        return
    original_load = ModelRunner.load_model
    original_attention = ModelRunner.init_attention_backend

    @functools.wraps(original_load)
    def load(self, *args, **kwargs):
        result = original_load(self, *args, **kwargs)
        reserve_baseline_workspaces(self)
        return result

    @functools.wraps(original_attention)
    def attention(self, *args, **kwargs):
        result = original_attention(self, *args, **kwargs)
        bind_baseline_attention(self)
        return result

    ModelRunner.load_model = load
    ModelRunner.init_attention_backend = attention
    ModelRunner._matched_baseline_hooks_installed = True


if __name__ in ("__main__", "__mp_main__"):
    install()

if __name__ == "__main__":
    import runpy

    runpy.run_module("sglang.launch_server", run_name="__main__")
