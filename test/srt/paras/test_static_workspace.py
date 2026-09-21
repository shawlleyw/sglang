"""CPU checks for production scratch sizing, ordering, reuse and fallback."""

import ast
import sys
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
        model_config=SimpleNamespace(hf_config=config, context_len=128, head_dim=8),
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
        dict(attention_backend="fa3"),
        dict(moe_runner_backend="triton_kernel"),
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


@pytest.mark.parametrize("ep", [True, False])
@pytest.mark.parametrize("attention", ["triton", "flashinfer"])
@pytest.mark.parametrize("moe", ["triton", "deep_gemm"])
def test_qwen_reserves_supported_backend_combinations(monkeypatch, ep, attention, moe):
    monkeypatch.setenv("SGLANG_FLASHINFER_WORKSPACE_SIZE", "4096")
    r = runner(ep)
    r.server_args.attention_backend = attention
    r.server_args.moe_runner_backend = moe
    config = r.model_config.hf_config
    config.architectures = ["Qwen3MoeForCausalLM"]
    config.moe_intermediate_size = config.intermediate_size
    config.num_experts = config.num_local_experts
    del config.intermediate_size, config.num_local_experts, config.head_dim
    state = reserve_static_workspaces(r)
    assert state is not None
    assert state.moe.numel() > 0
    if attention == "flashinfer":
        assert state.attention.numel() == 4096
    for layer in r.model:
        assert layer.moe_runner_config.paras_workspace.buffer is state.moe


@pytest.mark.parametrize("ep", [True, False])
def test_flashinfer_reuses_preallocated_buffer_without_global_allocation(
    monkeypatch, ep
):
    monkeypatch.setenv("SGLANG_FLASHINFER_WORKSPACE_SIZE", "4096")
    r = runner(ep)
    r.server_args.attention_backend = "flashinfer"
    state = reserve_static_workspaces(r)
    state.attention.fill_(23)
    tree = ast.parse((ROOT / "layers/attention/flashinfer_backend.py").read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "FlashInferAttnBackend"
    )
    init = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    # Run the real constructor's allocation block without importing CUDA wrappers.
    start = next(
        i
        for i, n in enumerate(init.body)
        if isinstance(n, ast.Assign)
        and ast.unparse(n.targets[0]) == "self._paras_workspace_mode"
    )
    end = next(
        i
        for i, n in enumerate(init.body)
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "max_bs"
    )
    b = SimpleNamespace(_paras_workspace=lambda: None)
    namespace = dict(
        self=b,
        model_runner=r,
        init_new_workspace=False,
        ParaSMode=ParaSMode,
        get_global_paras_memory_manager=lambda: None,
        global_workspace_buffer=None,
        torch=torch,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("FlashInfer allocated a second workspace")

    monkeypatch.setattr(torch, "empty", forbidden)
    exec(
        compile(
            ast.Module(body=init.body[start:end], type_ignores=[]), str(ROOT), "exec"
        ),
        namespace,
    )
    assert b.workspace_buffer is state.attention
    assert torch.count_nonzero(b.workspace_buffer) == 0
    assert namespace["global_workspace_buffer"] is None


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("fits", [False, True])
def test_deepgemm_production_reservation_reuse_and_overflow(monkeypatch, masked, fits):
    monkeypatch.setenv("SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK", "4")
    r = runner(True)
    r.server_args.moe_runner_backend = "deep_gemm"
    state = reserve_static_workspaces(r)
    state.moe.fill_(23)
    method_name = "_run_masked_gemm_bf16" if masked else "_run_contiguous_gemm_bf16"
    tree = ast.parse((ROOT / "layers/moe/moe_runner/deep_gemm.py").read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "DeepGemmRunnerCore"
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name
    )
    groups, rows, hidden, intermediate = 2, (16 if fits else 32), 64, 64
    observed = []

    def grouped(x, weight, out, *args):
        observed.append(out.untyped_storage().data_ptr())
        value = torch.bmm(x.reshape(groups, rows, -1), weight.transpose(1, 2))
        out.copy_(value.reshape(out.shape))

    def activate(x, out, *args):
        gate, up = x.chunk(2, dim=-1)
        out.copy_(torch.nn.functional.silu(gate) * up)

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.layers.moe.ep_moe.kernels",
        SimpleNamespace(silu_and_mul_masked_fwd=activate),
    )
    namespace = dict(
        torch=torch,
        DeepGemmRunnerInput=object,
        DeepGemmMoeQuantInfo=object,
        _is_npu=False,
        _is_hip=False,
        dispose_tensor=lambda x: None,
        silu_and_mul=activate,
        deep_gemm_wrapper=SimpleNamespace(
            grouped_gemm_nt_bf16bf16bf16_contig=grouped,
            grouped_gemm_nt_bf16bf16bf16_masked=grouped,
        ),
    )
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(ROOT), "exec"),
        namespace,
    )
    x = torch.randn(groups, rows, hidden, dtype=torch.bfloat16) * 0.1
    w13 = torch.randn(groups, 2 * intermediate, hidden, dtype=torch.bfloat16) * 0.1
    w2 = torch.randn(groups, hidden, intermediate, dtype=torch.bfloat16) * 0.1
    inputs = SimpleNamespace(
        hidden_states=x if masked else x.flatten(0, 1),
        hidden_states_scale=None,
        masked_m=None,
        expected_m=rows,
        m_indices=None,
    )
    quant = SimpleNamespace(w13_weight=w13, w2_weight=w2)
    runtime = dict(
        all_tokens=groups * rows,
        hidden_states_device="cpu",
        hidden_states_shape=(groups * rows, hidden),
    )
    method_runner = SimpleNamespace(config=r.model[0].moe_runner_config)
    actual = namespace[method_name](method_runner, inputs, quant, runtime)
    assert (observed[0] == state.moe.data_ptr()) == fits
    assert actual.untyped_storage().data_ptr() != state.moe.data_ptr()
    if not fits:
        assert torch.all(state.moe == 23)
    method_runner.config.paras_workspace = None
    expected = namespace[method_name](method_runner, inputs, quant, runtime)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
