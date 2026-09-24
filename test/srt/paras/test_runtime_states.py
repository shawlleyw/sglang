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

    def zeros(mode, name, shape, dtype, *, zero_on_resume=True):
        calls.append((mode, name, shape, zero_on_resume))
        return torch.zeros(shape, dtype=dtype)

    b._paras_runtime_memory = SimpleNamespace(zeros=zeros)
    b.init_cuda_graph_state(2, 2)
    assert calls == [(ParaSMode.EP, "kv_indices", (64,), False)]


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


@pytest.mark.parametrize("rank", range(4))
def test_ep_return_restores_independent_sampling_group(monkeypatch, rank):
    """Uneven EP batches must not sample collectively across DP ranks."""
    import logging
    import time
    from contextlib import nullcontext
    from unittest.mock import Mock

    noop = lambda *args, **kwargs: None
    monkeypatch.setattr(torch.cuda, "synchronize", noop)
    scatter = Mock()
    scatter.precheck_ep_capacity.return_value = (True, "", {})
    scatter.get_new_running_batch.return_value = None
    scatter.get_new_waiting_queue.return_value = []
    cls = load_nodes(
        "paras/scheduler_paras_mixin.py",
        {"paras_configure_ep"},
        "SchedulerParasMixin",
        paras_func=lambda f: f,
        ParaSReqScatterManager=lambda **kwargs: scatter,
        TimeReporter=lambda *args: nullcontext(),
        time=time,
        logger=logging.getLogger(__name__),
        MoeA2ABackend=SimpleNamespace(DEEPEP=SimpleNamespace(value="deepep")),
        moe_utils=SimpleNamespace(),
        compute_dp_attention_world_info=lambda *args: (0, 1, rank),
    )
    scheduler = cls()
    tp_group = SimpleNamespace(device_group=object())
    ep_attn_group = SimpleNamespace(device_group=object())
    sampler = SimpleNamespace(
        tp_sync_group=tp_group.device_group, force_sync_token_ids=True
    )
    scheduler.__dict__.update(
        paras_parallelism_config=ParaSMode.TP,
        paras_check=lambda: True,
        server_args=SimpleNamespace(
            enable_paras_moe=True, enable_custom_logit_processor=False
        ),
        paras_dp_size=1,
        paras_tp_size=4,
        paras_tp_rank=rank,
        paras_ep_size=4,
        paras_ep_rank=rank,
        dp_size=4,
        paras_tp_group=tp_group,
        paras_tp_attn_tp_group=tp_group,
        paras_ep_group=tp_group,
        paras_ep_cpu_group=object(),
        paras_ep_attn_tp_group=ep_attn_group,
        paras_ep_attn_tp_cpu_group=object(),
        _paras_drain_overlap_pipeline=noop,
        _paras_auto_policy=None,
        _paras_post_switch_ramp_iters=0,
        paras_start_profile=noop,
        paras_stop_profile=noop,
        tree_cache=SimpleNamespace(reset=noop),
        merge_last_batch=noop,
        running_batch=None,
        waiting_queue=[],
        req_to_token_pool=object(),
        token_to_kv_pool_allocator=object(),
        tp_worker=SimpleNamespace(
            model_runner=SimpleNamespace(model=object(), sampler=sampler),
            paras_configure_ep=noop,
        ),
        tokenizer=None,
        model_config=None,
        enable_overlap=False,
        spec_algorithm=None,
        ep_recv_from_tokenizer=object(),
        ep_send_to_tokenizer=object(),
        ep_send_to_detokenizer=object(),
        ep_recv_from_rpc=object(),
        last_batch=object(),
    )
    scheduler.paras_configure_ep()
    assert sampler.tp_sync_group is ep_attn_group.device_group
    assert sampler.tp_sync_group is scheduler.attn_tp_group.device_group
    assert not sampler.force_sync_token_ids
    assert scheduler.paras_parallelism_config == ParaSMode.EP


@pytest.mark.parametrize("num_wrappers", [1, 2])
def test_flashinfer_vmm_state_owns_all_large_buffers_and_restores_aliases(num_wrappers):
    cls = load_nodes(
        "layers/attention/flashinfer_backend.py",
        {"init_cuda_graph_state", "paras_save_cuda_graph_state", "paras_load_cuda_graph_state"},
        "FlashInferAttnBackend",
    )
    allocations = {}

    def zeros(mode, name, shape, dtype):
        assert (mode, name) not in allocations
        value = torch.zeros(shape, dtype=dtype)
        allocations[mode, name] = value
        return value

    b = cls()
    b._paras_runtime_memory = SimpleNamespace(zeros=zeros)
    b.max_context_len = 16
    b.num_wrappers = num_wrappers
    b.skip_prefill = False
    b.kv_indptr = [torch.zeros(9, dtype=torch.int32) for _ in range(num_wrappers)]
    b.decode_cuda_graph_metadata = {}
    b.prefill_cuda_graph_metadata = {}
    b.draft_extend_cuda_graph_metadata = {}
    states = {}
    for mode, bs in [(ParaSMode.EP, 2), (ParaSMode.TP, 8)]:
        b._paras_workspace_mode = mode
        b.init_cuda_graph_state(bs, bs)
        # Represent the captured wrappers' retained aliases without loading CUDA.
        b.decode_cuda_graph_metadata = {bs: list(b.cuda_graph_kv_indices)}
        states[mode] = b.paras_save_cuda_graph_state()
        assert b.cuda_graph_custom_mask.numel() == bs * 16
        assert b.cuda_graph_custom_mask.dtype == torch.uint8
        for i, tensor in enumerate(b.cuda_graph_kv_indices):
            assert tensor is allocations[mode, f"flashinfer_kv_indices_{i}"]
            assert tensor.numel() == bs * 16
            assert tensor.dtype == torch.int32
    assert len(allocations) == 2 * (num_wrappers + 1)
    for mode in [ParaSMode.EP, ParaSMode.TP, ParaSMode.EP]:
        b.paras_load_cuda_graph_state(states[mode])
        for i, tensor in enumerate(b.cuda_graph_kv_indices):
            assert tensor is allocations[mode, f"flashinfer_kv_indices_{i}"]
            assert next(iter(b.decode_cuda_graph_metadata.values()))[i] is tensor
        assert b.cuda_graph_custom_mask is allocations[mode, "flashinfer_custom_mask"]
    with pytest.raises(ValueError, match="owned FlashInfer indices"):
        b.init_cuda_graph_state(2, 2, kv_indices_buf=torch.zeros(32))
