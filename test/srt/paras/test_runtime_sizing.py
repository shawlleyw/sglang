"""Configured reservations must not consume KV or scratch for unreachable batches."""

from types import SimpleNamespace

import pytest

from sglang.srt.paras import paras_memory_manager as memory
from sglang.srt.paras.mode import ParaSMode
from sglang.srt.paras.unified_layout import (
    bf16_moe_workspace_sizes,
    tp_moe_workspace_token_capacity,
)


@pytest.mark.parametrize("limit", [64, 256, 2048, 8192, None])
def test_request_backing_capacity_honors_configured_limit(limit):
    manager = memory.ParaSMemoryManager(device="cpu")
    manager.ep_max_kv_tokens = 330_000
    manager.tp_max_kv_tokens = 3_000_000
    ep, tp = manager.plan_req_capacities(
        context_len=131072,
        ep_max_num_reqs=min(2048, limit) if limit else 2048,
        max_running_requests=limit,
        dp_size=8,
    )
    assert ep == min(2048, limit or 2048)
    assert tp == min(4096, limit or 4096)
    assert manager.ep_max_running_requests == (
        min(max(limit // 8, 1), ep) if limit else ep
    )
    assert manager.tp_max_running_requests == tp
    if limit == 2048:
        # This is the stable backing table shared by both graph sets, not KV.
        assert tp * 131076 * 4 == 1_073_774_592


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({}, 8192),
        ({"max_running_requests": 16384}, 16384),
        ({"cuda_graph_bs": [1, 32768]}, 32768),
        ({"chunked_prefill_size": 512}, 2048),
        ({"max_running_requests": None, "max_prefill_tokens": 1024}, 4096),
        ({"max_prefill_tokens": 131072}, 65536),
        ({"max_prefill_tokens": None}, 65536),
        ({"speculative_num_draft_tokens": 8}, 16384),
    ],
)
def test_tp_reservation_follows_prefill_decode_and_capture_limits(overrides, expected):
    args = dict(max_prefill_tokens=8192, max_running_requests=2048)
    args.update(overrides)
    assert tp_moe_workspace_token_capacity(**args) == expected


@pytest.mark.parametrize("ep_prefill,tp_prefill", [(8192, None), (2048, 8192)])
def test_model_reservation_uses_runtime_config(monkeypatch, ep_prefill, tp_prefill):
    from sglang.srt.layers.moe import utils as moe_utils

    monkeypatch.setattr(moe_utils, "use_deep_gemm_bf16", lambda *a, **k: False)
    args = SimpleNamespace(
        moe_runner_backend="triton",
        max_prefill_tokens=ep_prefill,
        paras_tp_max_prefill_tokens=tp_prefill,
        max_running_requests=2048,
        chunked_prefill_size=-1,
        disable_cuda_graph=False,
        cuda_graph_bs=[1, 256],
        paras_tp_cuda_graph_bs=[1, 2048],
    )
    manager = memory.ParaSMemoryManager(device="cpu", server_args=args)
    memory.reserve_model_weights(
        manager,
        dp_size=1,
        moe_tp_size=1,
        num_layers=36,
        num_experts=128,
        hidden_size=2880,
        intermediate_size=2880,
        num_heads=64,
        num_kv_heads=8,
        head_dim=64,
        ep_size=8,
        tp_size=8,
        top_k=4,
        with_bias=True,
    )
    size = manager._unified_spec.for_mode(ParaSMode.TP).workspaces.moe.size_bytes
    assert 405 * 2**20 < size < 407 * 2**20
    old_ep, old_tp = bf16_moe_workspace_sizes(
        hidden_size=2880,
        intermediate_size=2880,
        num_experts=128,
        top_k=4,
        tp_size=8,
        dispatch_capacity=256,
    )
    new_ep, new_tp = bf16_moe_workspace_sizes(
        hidden_size=2880,
        intermediate_size=2880,
        num_experts=128,
        top_k=4,
        tp_size=8,
        dispatch_capacity=256,
        tp_input_tokens=8192,
    )
    assert new_ep == old_ep
    assert new_tp == size
    assert old_tp - new_tp > 1.3 * 2**30
