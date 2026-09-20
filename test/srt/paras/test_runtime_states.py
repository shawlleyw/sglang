"""Run production buffer/state methods on CPU without importing CUDA backends.

AST extraction excludes only the modules' CUDA-dependent imports and unrelated
methods. It executes the original method bodies, using real CPU tensors.
"""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
import torch

from sglang.srt.paras.mode import ParaSMode

ROOT = Path(__file__).resolve().parents[3] / "python/sglang/srt"


def load_nodes(path, names, class_name=None, **extra):
    tree = ast.parse((ROOT / path).read_text())
    if class_name:
        cls = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        cls.body = [
            node
            for node in cls.body
            if getattr(node, "name", None) in names
            or isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) in names for t in node.targets)
        ]
        nodes = [cls]
    else:
        nodes = [
            node
            for node in tree.body
            if getattr(node, "name", None) in names
            or isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) in names for t in node.targets)
        ]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            )
        ]
        + nodes,
        type_ignores=[],
    )
    namespace = dict(
        torch=torch,
        ParaSMode=ParaSMode,
        Optional=Optional,
        AttentionBackend=object,
        **extra
    )
    exec(
        compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), namespace
    )
    return namespace[class_name] if class_name else namespace


@pytest.fixture(autouse=True)
def forbid_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU test attempted CUDA initialization")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)


def backend():
    cls = load_nodes(
        "layers/attention/triton_backend.py",
        {
            "_PARAS_CUDA_GRAPH_BUFFER_ATTRS",
            "paras_configure_tp",
            "paras_configure_ep",
            "paras_save_cuda_graph_state",
            "paras_load_cuda_graph_state",
            "_paras_restore_or_allocate_buffers",
            "_paras_alloc_fresh_buffers",
            "init_cuda_graph_state",
            "_allocate_decode_workspace",
        },
        "TritonAttnBackend",
    )
    b = cls()
    b.device = "cpu"
    b._paras_workspace_mode = ParaSMode.EP
    b._paras_memory_manager = None
    b.total_num_attention_heads = b.num_head = 8
    b.num_kv_head = 8
    b._get_num_kv_heads = lambda size: 8 // size
    b.max_kv_splits = b.v_head_dim = 2
    b.max_context_len = 32
    b.sliding_window_size = None
    b.window_kv_indptr = None
    b.skip_prefill = False
    b.num_draft_tokens = None
    b.req_to_token = torch.zeros((8, 32), dtype=torch.int32)
    b._paras_alloc_fresh_buffers()
    return b


def test_backend_ep_tp_ep_restores_original_storage_without_allocating(monkeypatch):
    b = backend()
    b.init_cuda_graph_state(2, 2)
    ep = b.paras_save_cuda_graph_state()
    assert ep["cuda_graph_kv_indices"].numel() == 64
    assert ep["cuda_graph_custom_mask"] is None
    b.paras_configure_tp(4, b.req_to_token)
    b.init_cuda_graph_state(8, 8)
    tp = b.paras_save_cuda_graph_state()
    assert tp["cuda_graph_kv_indices"].numel() == 256
    assert (
        tp["cuda_graph_kv_indices"].data_ptr() != ep["cuda_graph_kv_indices"].data_ptr()
    )

    def forbidden():
        raise AssertionError("Mode switch allocated fresh buffers")

    monkeypatch.setattr(b, "_paras_alloc_fresh_buffers", forbidden)
    for _ in range(3):
        b.paras_configure_ep(b.req_to_token)
        assert b.num_head == 8
        for key, value in ep.items():
            assert getattr(b, key) is value
        b.paras_configure_tp(4, b.req_to_token)
        assert b.num_head == 2
        for key, value in tp.items():
            assert getattr(b, key) is value


def test_verification_keeps_custom_mask():
    b = backend()
    b.num_draft_tokens = 4
    b.init_cuda_graph_state(2, 8)
    assert b.cuda_graph_custom_mask.shape == (256,)
    assert b.cuda_graph_custom_mask.dtype == torch.uint8


def test_backend_routes_only_owned_indices_to_vmm():
    b = backend()
    calls = []

    def zeros(mode, name, shape, dtype):
        calls.append((mode, name, shape))
        return torch.zeros(shape, dtype=dtype)

    b._paras_runtime_memory = SimpleNamespace(zeros=zeros)
    b.init_cuda_graph_state(2, 2)
    assert calls == [(ParaSMode.EP, "kv_indices", (64,))]


def test_graph_inputs_and_output_views_follow_the_saved_mode(monkeypatch):
    cls = load_nodes(
        "model_executor/cuda_graph_runner.py",
        {"init_graph_buffers", "_cache_loc_dtype"},
        "CudaGraphRunner",
        TboCudaGraphRunnerPlugin=object,
    )
    runner = cls()
    runner.device = "cpu"
    runner.seq_len_fill_value = 1
    runner.encoder_len_fill_value = 0
    runner.pp_size = 1
    runner.num_tokens_per_bs = 1
    runner.is_encoder_decoder = False
    runner._paras_runtime_memory = None
    attn = backend()
    runner.model_runner = SimpleNamespace(
        spec_algorithm=SimpleNamespace(is_eagle3=lambda: False),
        model_config=SimpleNamespace(vocab_size=16),
        attn_backend=attn,
    )
    runner.deepep_adapter = SimpleNamespace(_captured_deepep_mode=None)
    pool = [None]
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.model_executor.cuda_graph_runner",
        SimpleNamespace(
            get_global_graph_memory_pool=lambda: pool[0],
            set_global_graph_memory_pool=lambda value: pool.__setitem__(0, value),
        ),
    )
    funcs = load_nodes(
        "paras/paras_cuda_graph.py",
        {
            "_SETTINGS_KEYS",
            "_BUFFER_KEYS",
            "paras_save_cuda_graph_state",
            "paras_load_cuda_graph_state",
        },
    )
    states = {}
    for mode, batch, dp in [(ParaSMode.EP, 2, 4), (ParaSMode.TP, 8, 1)]:
        for key in funcs["_SETTINGS_KEYS"]:
            setattr(runner, key, False)
        runner.capture_bs, runner.compile_bs = [1, batch], []
        runner.dp_size = dp
        runner.require_gathered_buffer = runner.require_mlp_tp_gather = dp > 1
        runner.max_bs = runner.max_num_token = batch
        attn._paras_workspace_mode = mode
        attn.init_cuda_graph_state(batch, batch)
        runner.init_graph_buffers()
        pool[0] = object()
        runner.graphs = {batch: object()}
        runner.output_buffers = {batch: runner.next_token_logits_buffer[:batch]}
        runner.next_token_logits_buffer.fill_(dp)
        funcs["paras_save_cuda_graph_state"](runner, mode)
        states[mode] = runner._paras_saved[mode]
    for mode, batch in [(ParaSMode.EP, 2), (ParaSMode.TP, 8), (ParaSMode.EP, 2)]:
        funcs["paras_load_cuda_graph_state"](runner, mode)
        state = states[mode]
        assert runner.max_bs == batch
        assert runner.next_token_logits_buffer.shape == (batch, 16)
        assert (
            runner.output_buffers[batch].data_ptr()
            == runner.next_token_logits_buffer.data_ptr()
        )
        assert runner.output_buffers[batch][0, 0] == (4 if mode == ParaSMode.EP else 1)
        assert pool[0] is state["graph_memory_pool"]
        for key, value in state["buffers"].items():
            assert getattr(runner, key) is value
        assert runner.graphs is state["graphs"]


def test_flashinfer_restores_backing_buffers_with_wrapper_metadata():
    cls = load_nodes(
        "layers/attention/flashinfer_backend.py",
        {"paras_save_cuda_graph_state", "paras_load_cuda_graph_state"},
        "FlashInferAttnBackend",
    )
    b = cls()
    b.decode_cuda_graph_metadata = {2: object()}
    b.prefill_cuda_graph_metadata = {}
    b.draft_extend_cuda_graph_metadata = {}
    ep_indices = torch.zeros(64, dtype=torch.int32)
    b.cuda_graph_kv_indices = [ep_indices]
    ep = b.paras_save_cuda_graph_state()
    b.cuda_graph_kv_indices = [torch.zeros(256, dtype=torch.int32)]
    b.decode_cuda_graph_metadata = {8: object()}
    tp = b.paras_save_cuda_graph_state()
    b.paras_load_cuda_graph_state(ep)
    assert b.cuda_graph_kv_indices[0] is ep_indices
    assert set(b.decode_cuda_graph_metadata) == {2}
    b.paras_load_cuda_graph_state(tp)
    assert b.cuda_graph_kv_indices[0].numel() == 256
    assert set(b.decode_cuda_graph_metadata) == {8}


def test_tp_prefill_ignores_divergent_ep_ramp_counters():
    cls = load_nodes(
        "paras/scheduler_paras_mixin.py",
        {"paras_effective_max_prefill_tokens"},
        "SchedulerParasMixin",
    )
    schedulers = []
    for remaining in (0, 3, 4, 5, 10, 20):
        scheduler = cls()
        scheduler.paras_parallelism_config = ParaSMode.TP
        scheduler.max_prefill_tokens = 8192
        scheduler._paras_post_switch_initial_cap = 2048
        scheduler._paras_post_switch_ramp_iters = 20
        scheduler._paras_post_switch_iters_remaining = remaining
        schedulers.append(scheduler)
    assert {s.paras_effective_max_prefill_tokens() for s in schedulers} == {8192}
    assert all(s._paras_post_switch_iters_remaining == 0 for s in schedulers)


def test_ep_prefill_ramp_still_progresses():
    cls = load_nodes(
        "paras/scheduler_paras_mixin.py",
        {"paras_effective_max_prefill_tokens"},
        "SchedulerParasMixin",
    )
    scheduler = cls()
    scheduler.paras_parallelism_config = ParaSMode.EP
    scheduler.max_prefill_tokens = 8192
    scheduler._paras_post_switch_initial_cap = 2048
    scheduler._paras_post_switch_ramp_iters = 20
    scheduler._paras_post_switch_iters_remaining = 20
    caps = [scheduler.paras_effective_max_prefill_tokens() for _ in range(21)]
    assert caps[0] == 2355
    assert caps[-2:] == [8192, 8192]
    assert caps == sorted(caps)


@pytest.mark.parametrize("ramp_cap", [1024, 2048, 8192])
def test_prefill_budget_roundtrip_honors_per_mode_limits(ramp_cap):
    cls = load_nodes(
        "paras/scheduler_paras_mixin.py",
        {"paras_effective_max_prefill_tokens"},
        "SchedulerParasMixin",
    )
    scheduler = cls()
    scheduler.server_args = SimpleNamespace(paras_tp_max_prefill_tokens=8192)
    scheduler.max_prefill_tokens = 2048
    scheduler._paras_post_switch_initial_cap = ramp_cap
    scheduler._paras_post_switch_ramp_iters = 20
    for mode in [ParaSMode.EP, ParaSMode.TP, ParaSMode.EP]:
        scheduler.paras_parallelism_config = mode
        scheduler._paras_post_switch_iters_remaining = 20
        caps = [scheduler.paras_effective_max_prefill_tokens() for _ in range(21)]
        if mode == ParaSMode.TP:
            assert caps == [8192] * 21
            assert scheduler._paras_post_switch_iters_remaining == 0
        else:
            assert max(caps) == 2048
            assert caps == sorted(caps)
        assert scheduler.max_prefill_tokens == 2048
