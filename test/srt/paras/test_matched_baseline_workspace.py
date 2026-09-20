"""CPU checks for production scratch sizing, ordering, reuse and fallback."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.model_executor.static_workspace import reserve_static_workspaces
from sglang.srt.paras.mode import ParaSMode
from sglang.srt.paras.unified_layout import bf16_moe_workspace_sizes
from sglang.srt.paras.workspace import triton_attention_workspace_size

ROOT = Path(__file__).resolve().parents[3] / "python/sglang/srt"


@pytest.fixture(autouse=True)
def forbid_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU test attempted CUDA initialization")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)


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
        dp_size=4 if ep else 1,
        moe_a2a_backend="deepep" if ep else "none",
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


def backend(state, ep):
    # Execute the production allocation method without importing GPU kernels.
    tree = ast.parse((ROOT / "layers/attention/triton_backend.py").read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "TritonAttnBackend"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_allocate_decode_workspace"
    )
    namespace = dict(torch=torch)
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(ROOT), "exec"),
        namespace,
    )
    result = SimpleNamespace(
        num_head=8 if ep else 2,
        max_kv_splits=2,
        v_head_dim=8,
        device="cpu",
        _paras_memory_manager=None,
        _static_workspaces=state,
    )
    result.allocate = lambda tokens, **kw: namespace[method.name](result, tokens, **kw)
    return result


@pytest.mark.parametrize("ep", [True, False])
def test_production_attention_and_moe_reuse_shared_reservation(ep):
    r = runner(ep)
    state = reserve_static_workspaces(r)
    assert state.mode == (ParaSMode.EP if ep else ParaSMode.TP)
    assert state.bound_layers == 2
    for layer in r.model:
        assert layer.moe_runner_config.paras_workspace.buffer is state.moe
    b = backend(state, ep)
    tokens = 4 if ep else 16
    first = b.allocate(tokens, zero=True)
    assert all(torch.count_nonzero(x) == 0 for x in first)
    again = b.allocate(tokens)
    assert [x.data_ptr() for x in first] == [x.data_ptr() for x in again]
    assert all(
        x.untyped_storage().data_ptr() == state.attention.data_ptr() for x in first
    )
    state.attention.fill_(23)
    overflow = b.allocate(tokens * 100)
    assert all(
        x.untyped_storage().data_ptr() != state.attention.data_ptr() for x in overflow
    )
    assert torch.all(state.attention == 23)
    assert state.attention_reuses == 2 and state.attention_fallbacks == 1
    assert reserve_static_workspaces(r) is state  # No double reservation.


@pytest.mark.parametrize(
    "prefill,chunk,requests,graphs,expected",
    [
        (512, -1, 16, [1, 16], 512),
        (4096, 128, 16, [1, 16], 128),
        (32, -1, 64, [1, 16], 64),
        (32, -1, 16, [1, 128], 128),
    ],
)
def test_tp_moe_uses_actual_runtime_token_limits(
    prefill, chunk, requests, graphs, expected
):
    r = runner(False)
    r.server_args.max_prefill_tokens = prefill
    r.server_args.chunked_prefill_size = chunk
    r.server_args.max_running_requests = requests
    r.server_args.cuda_graph_bs = graphs
    state = reserve_static_workspaces(r)
    _, size = bf16_moe_workspace_sizes(
        hidden_size=64,
        intermediate_size=64,
        num_experts=8,
        top_k=2,
        tp_size=4,
        dispatch_capacity=128,
        tp_input_tokens=expected,
    )
    assert state.moe.numel() == size


@pytest.mark.parametrize("ep", [True, False])
@pytest.mark.parametrize("context", [128, 320])
def test_attention_uses_resolved_context_not_architectural_maximum(ep, context):
    r = runner(ep)
    r.model_config.hf_config.max_position_embeddings = 131072
    r.model_config.context_len = context
    r.server_args.triton_attention_split_tile_size = 64
    state = reserve_static_workspaces(r)
    assert state.attention.numel() == triton_attention_workspace_size(
        4 if ep else 16, 8 if ep else 2, (context + 63) // 64, 8
    )


def test_ep_moe_follows_dispatch_capacity(monkeypatch):
    sizes = []
    for cap in [16, 64]:
        monkeypatch.setenv("SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK", str(cap))
        sizes.append(reserve_static_workspaces(runner(True)).moe.numel())
    assert sizes[1] == 4 * sizes[0]


@pytest.mark.parametrize(
    "overrides",
    [
        dict(enable_paras_moe=True),
        dict(disable_overlap_schedule=False),
        dict(enable_two_batch_overlap=True),
        dict(enable_pdmux=True),
        dict(speculative_algorithm="EAGLE"),
        dict(pp_size=2),
        dict(quantization="fp8"),
        dict(attention_backend="flashinfer"),
        dict(prefill_attention_backend="flashinfer"),
        dict(max_running_requests=None),
        dict(enable_torch_compile=True),
        dict(enable_memory_saver=True),
        dict(dp_size=2),
        dict(moe_a2a_backend="none"),
    ],
)
def test_unsupported_paths_do_not_reserve_or_bind(overrides, monkeypatch):
    r = runner(True)
    for key, value in overrides.items():
        setattr(r.server_args, key, value)

    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected scratch allocation")

    monkeypatch.setattr(torch, "empty", forbidden)
    assert reserve_static_workspaces(r) is None
    assert all(layer.moe_runner_config.paras_workspace is None for layer in r.model)


def test_production_initialization_reserves_before_kv_profiling():
    tree = ast.parse((ROOT / "model_executor/model_runner.py").read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModelRunner"
    )
    init = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "initialize"
    )
    reserve_index = next(
        i
        for i, n in enumerate(init.body)
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Attribute) and t.attr == "static_workspaces"
            for t in n.targets
        )
    )
    pool_index = next(
        i
        for i, n in enumerate(init.body)
        if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Attribute)
        and n.value.func.attr == "init_memory_pool"
    )
    assert reserve_index < pool_index
    r = runner(False)
    r.server_args.max_total_tokens = None
    called = []

    def pool(*args):
        assert r.static_workspaces.moe.numel() > 0
        assert r.model[0].moe_runner_config.paras_workspace is not None
        called.append("kv")

    r.init_memory_pool = pool
    body = [init.body[reserve_index], init.body[pool_index]]
    exec(
        compile(ast.Module(body=body, type_ignores=[]), str(ROOT), "exec"),
        dict(
            self=r,
            server_args=r.server_args,
            min_per_gpu_memory=1,
            reserve_static_workspaces=reserve_static_workspaces,
        ),
    )
    assert called == ["kv"]


def test_auto_detected_quantization_keeps_native_allocation():
    r = runner(False)
    r.model_config.quantization = "mxfp4"
    assert r.server_args.quantization is None
    assert reserve_static_workspaces(r) is None
