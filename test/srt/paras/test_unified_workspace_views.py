"""Bound workspace views preserve live weights even when weight addresses alias."""

import pytest
import torch

from sglang.srt.paras import paras_memory_manager as memory
from sglang.srt.paras import unified_layout
from sglang.srt.paras.mode import ParaSMode
from sglang.srt.paras.workspace import ModeWorkspaces, WorkspaceRequirement


@pytest.mark.parametrize("attention_bytes", [0, 1024, 400_000])
def test_workspace_ownership_when_mode_weight_addresses_coincide(
    monkeypatch, attention_bytes
):
    monkeypatch.setattr(
        unified_layout, "bf16_moe_workspace_sizes", lambda **_: (4096, 8192)
    )
    from sglang.srt.layers.moe import utils as moe_utils

    monkeypatch.setattr(moe_utils, "use_deep_gemm_bf16", lambda *args, **kwargs: False)
    mgr = memory.ParaSMemoryManager(device="cpu")
    memory.reserve_model_weights(
        mgr,
        dp_size=1,
        moe_tp_size=1,
        num_layers=4,
        num_experts=8,
        hidden_size=64,
        intermediate_size=128,
        num_heads=8,
        num_kv_heads=2,
        head_dim=128,
        ep_size=4,
        tp_size=4,
        top_k=2,
        prefix="model",
    )
    spec = mgr._unified_spec
    for mode in (ParaSMode.EP, ParaSMode.TP):
        spec.for_mode(mode).workspaces = ModeWorkspaces(
            spec.for_mode(mode).workspaces.moe,
            WorkspaceRequirement("test", attention_bytes),
        )
    layout = unified_layout.plan_unified_layout(
        num_layers=4,
        budget=5 << 20,
        ep_weight_bytes=spec.ep.weight_bytes,
        tp_weight_bytes=spec.tp.weight_bytes,
        ep_workspace_bytes=spec.ep.workspaces.size_bytes,
        tp_workspace_bytes=spec.tp.workspaces.size_bytes,
        ep_kv_row_bytes=1024,
        tp_kv_row_bytes=512,
    )
    plan = memory.UnifiedMemoryPlan(
        layout, mgr._plan_tensor_entries(layout, torch.bfloat16, 1), torch.bfloat16, []
    )
    mgr.materialize(plan)
    ep_weight = mgr.get_view("model.layers.0.mlp.ep_experts.w13_weight")
    tp_weight = mgr.get_view("model.layers.1.mlp.tp_experts.w13_weight")
    if attention_bytes < 400_000:
        assert ep_weight.data_ptr() == tp_weight.data_ptr()
    for mode in (ParaSMode.EP, ParaSMode.TP):
        mgr._buffer.fill_(23)
        shapes = [(3, 17), (7, 19)]
        from sglang.srt.paras.workspace import moe_workspace_views

        binding = mgr.bind_moe_workspace(mode)
        assert binding.mode is mode
        scratch = moe_workspace_views(binding.buffer, *shapes, torch.bfloat16, "cpu")
        assert (
            binding.buffer.data_ptr()
            == mgr._buffer.data_ptr() + layout.workspace(mode)[0]
        )
        for view in scratch:
            assert (view.data_ptr() - mgr._buffer.data_ptr()) % 256 == 0
            view.fill_(1)
        offset, size = layout.workspace(mode)
        assert torch.all(mgr._buffer[:offset] == 23)
        assert torch.all(mgr._buffer[offset + size :] == 23)
        assert moe_workspace_views(
            binding.buffer, (size,), (size,), torch.bfloat16, "cpu"
        ) == (None, None)
        attention = mgr.get_attention_workspace(
            "test", mode, [(attention_bytes,)], torch.uint8, "cpu"
        )[0]
        attention_buffer = mgr.get_attention_workspace_buffer("test", mode)
        assert attention_buffer.dtype == torch.uint8
        assert attention_buffer.shape == (attention_bytes,)
        assert attention_buffer.data_ptr() == attention.data_ptr()
        attention.fill_(9)
        assert all(torch.all(view == 1) for view in scratch)
        assert torch.all(mgr._buffer[:offset] == 23)
        assert torch.all(mgr._buffer[offset + size :] == 23)
        # Accessors must not silently consume padding or another operator's region.
        with pytest.raises(RuntimeError, match="overflow"):
            mgr.get_attention_workspace(
                "test", mode, [(attention_bytes + 256,)], torch.uint8, "cpu"
            )
        with pytest.raises(RuntimeError, match="planned for"):
            mgr.get_attention_workspace("wrong", mode, [(1,)], torch.uint8, "cpu")
        with pytest.raises(RuntimeError, match="planned for"):
            mgr.get_attention_workspace_buffer("wrong", mode)
        mgr.initialize_attention_workspace(mode)
        assert torch.all(attention == 0)
        assert all(torch.all(view == 1) for view in scratch)


@pytest.mark.parametrize(
    "quant_name,method,dp_size",
    [("fp8", "peer_access", 1), (None, "naive", 1), (None, "peer_access", 2)],
)
def test_unsupported_model_scope_rejected_before_reservation(
    quant_name, method, dp_size
):
    mgr = memory.ParaSMemoryManager(device="cpu")
    with pytest.raises(AssertionError, match="ParaS"):
        memory.reserve_model_weights(
            mgr,
            num_layers=4,
            num_experts=8,
            hidden_size=64,
            intermediate_size=128,
            num_heads=8,
            num_kv_heads=4,
            head_dim=128,
            ep_size=4,
            tp_size=4,
            dp_size=dp_size,
            moe_tp_size=1,
            quant_name=quant_name,
            configure_method=method,
        )
    assert mgr.num_entries == 0


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("disable_hybrid_swa_memory", [False, True])
def test_gpt_oss_hybrid_views_match_capacity_plan(
    monkeypatch, head_dim, disable_hybrid_swa_memory
):
    from types import SimpleNamespace
    from sglang.srt.layers.moe import utils as moe_utils

    monkeypatch.setattr(moe_utils, "use_deep_gemm_bf16", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        unified_layout, "bf16_moe_workspace_sizes", lambda **kwargs: (4096, 8192)
    )
    args = SimpleNamespace(
        kv_cache_dtype="auto",
        page_size=1,
        swa_full_tokens_ratio=0.8,
        disable_hybrid_swa_memory=disable_hybrid_swa_memory,
        attention_backend="fa3",
        moe_runner_backend="triton",
        enable_two_batch_overlap=False,
        enable_pdmux=False,
        speculative_algorithm=None,
        prefill_attention_backend=None,
        decode_attention_backend=None,
    )
    mgr = memory.ParaSMemoryManager(device="cpu", server_args=args)
    memory.reserve_model_weights(
        mgr,
        num_layers=4,
        num_experts=8,
        hidden_size=64,
        intermediate_size=128,
        num_heads=8,
        num_kv_heads=4,
        with_bias=True,
        head_dim=head_dim,
        ep_size=4,
        tp_size=4,
        dp_size=1,
        moe_tp_size=1,
    )
    config = SimpleNamespace(
        num_hidden_layers=4,
        num_key_value_heads=4,
        num_attention_heads=8,
        hidden_size=64,
        head_dim=head_dim,
        sliding_window=128,
        layer_types=["sliding_attention", "full_attention"] * 2,
    )
    budget = 16 << 20
    plan = mgr.plan_layout(config, budget=budget)
    mgr.materialize(plan)
    assert mgr.total_bytes == budget
    if disable_hybrid_swa_memory:
        assert mgr.ep_max_kv_tokens_swa == mgr.ep_max_kv_tokens
        assert mgr.tp_max_kv_tokens_swa == mgr.tp_max_kv_tokens
    else:
        assert mgr.ep_max_kv_tokens_swa < mgr.ep_max_kv_tokens
    for mode in (ParaSMode.EP, ParaSMode.TP):
        cache = getattr(mgr._unified_layout, f"{mode.value}_cache")
        for i in range(4):
            key = mgr.get_view(f"model.layers.{i}.kv.{mode.value}.k")
            value = mgr.get_view(f"model.layers.{i}.kv.{mode.value}.v")
            assert key.shape[0] == cache.layer_tokens[i] + 1
            if disable_hybrid_swa_memory:
                assert key.shape[0] == cache.full_tokens + 1
            assert (
                unified_layout.align_up(key.numel() * key.element_size()) * 2
                <= cache.layer_bytes[i]
            )
            assert value.data_ptr() == key.data_ptr() + cache.layer_bytes[i] // 2
