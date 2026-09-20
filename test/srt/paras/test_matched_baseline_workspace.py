"""The evaluation baseline must allocate once and actually reuse its reservation."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.paras.mode import ParaSMode

PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts/paras/eval/matched_baseline_workspace.py"
)
SPEC = importlib.util.spec_from_file_location("matched_baseline_workspace_test", PATH)
HELPER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HELPER
SPEC.loader.exec_module(HELPER)


def runner(ep):
    model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Linear(1, 1))
    for layer in model:
        layer.moe_runner_config = SimpleNamespace(paras_workspace=None)
        layer.quant_method = torch.nn.Module()
        layer.quant_method.moe_runner_config = layer.moe_runner_config
    config = SimpleNamespace(
        architectures=["GptOssForCausalLM"],
        hidden_size=64,
        intermediate_size=64,
        num_local_experts=8,
        num_experts_per_tok=2,
        num_hidden_layers=2,
        num_attention_heads=8,
        head_dim=8,
    )
    args = SimpleNamespace(
        enable_paras_moe=False,
        quantization=None,
        attention_backend="triton",
        moe_runner_backend="triton",
        disable_overlap_schedule=True,
        enable_two_batch_overlap=False,
        enable_pdmux=False,
        speculative_algorithm=None,
        pp_size=1,
        enable_dp_attention=ep,
        ep_size=4 if ep else 1,
        tp_size=4,
        disable_cuda_graph=False,
        cuda_graph_bs=[1, 4] if ep else [1, 16],
        paras_tp_cuda_graph_bs=None,
        max_prefill_tokens=32,
        max_running_requests=16,
        chunked_prefill_size=-1,
        prefill_attention_backend=None,
        decode_attention_backend=None,
        enable_deterministic_inference=False,
        triton_attention_split_tile_size=None,
        triton_attention_num_kv_splits=2,
    )
    return SimpleNamespace(
        server_args=args,
        dtype=torch.bfloat16,
        device="cpu",
        model=model,
        model_config=SimpleNamespace(hf_config=config, context_len=128),
    )


@pytest.mark.parametrize("ep", [True, False])
def test_reservation_is_shared_across_layers_and_attention_capture(ep):
    r = runner(ep)
    state = HELPER.reserve_baseline_workspaces(r)
    assert state.mode == (ParaSMode.EP if ep else ParaSMode.TP)
    assert state.bound_layers == 2
    for layer in r.model:
        assert layer.moe_runner_config.paras_workspace.buffer is state.moe
    fallback_calls = []
    r.attn_backend = SimpleNamespace(
        num_head=8 if ep else 2,
        max_kv_splits=2,
        v_head_dim=8,
        device="cpu",
        _allocate_decode_workspace=lambda tokens, **kw: fallback_calls.append(tokens),
    )
    HELPER.bind_baseline_attention(r)
    max_tokens = 4 if ep else 16
    first = r.attn_backend._allocate_decode_workspace(max_tokens, zero=True)
    again = r.attn_backend._allocate_decode_workspace(max_tokens)
    assert [t.data_ptr() for t in first] == [t.data_ptr() for t in again]
    assert all(
        t.untyped_storage().data_ptr() == state.attention.data_ptr() for t in first
    )
    assert state.attention_reuses == 2
    assert fallback_calls == []
    saved = state.attention.clone()
    r.attn_backend._allocate_decode_workspace(max_tokens * 100)
    assert fallback_calls == [max_tokens * 100]
    assert state.attention_fallbacks == 1
    assert torch.equal(saved, state.attention)
    with pytest.raises(RuntimeError, match="already reserved"):
        HELPER.reserve_baseline_workspaces(r)


def test_rejects_paras_and_concurrent_execution():
    r = runner(True)
    r.server_args.enable_paras_moe = True
    with pytest.raises(ValueError, match="already reserves"):
        HELPER.reserve_baseline_workspaces(r)
    r.server_args.enable_paras_moe = False
    r.server_args.disable_overlap_schedule = False
    with pytest.raises(ValueError, match="serial BF16"):
        HELPER.reserve_baseline_workspaces(r)


def test_install_orders_hooks_and_is_idempotent(monkeypatch):
    calls = []

    class ModelRunner:
        def load_model(self):
            calls.append("load")
            return "loaded"

        def init_attention_backend(self):
            calls.append("attention")
            return "initialized"

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.model_executor.model_runner",
        SimpleNamespace(ModelRunner=ModelRunner),
    )
    monkeypatch.setattr(
        HELPER, "reserve_baseline_workspaces", lambda _: calls.append("reserve")
    )
    monkeypatch.setattr(
        HELPER, "bind_baseline_attention", lambda _: calls.append("bind")
    )
    HELPER.install()
    installed = ModelRunner.load_model
    HELPER.install()
    assert ModelRunner.load_model is installed
    r = ModelRunner()
    assert r.load_model() == "loaded"
    assert r.init_attention_backend() == "initialized"
    assert calls == ["load", "reserve", "attention", "bind"]
