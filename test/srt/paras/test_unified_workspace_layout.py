"""CPU checks for the asymmetric weight/KV/workspace transfer contract."""

import random

import pytest

from sglang.srt.paras.mode import ParaSMode
from sglang.srt.paras.unified_layout import (
    plan_unified_layout,
    bf16_moe_workspace_sizes,
)


def assert_safe(layout):
    n = layout.num_layers
    entries = {}
    for mode in (ParaSMode.EP, ParaSMode.TP):
        ws, length = layout.workspace(mode)
        for i in range(n):
            for kind, offset, size in (
                (
                    "w",
                    layout.weight_offset(mode, i),
                    getattr(layout, f"{mode.value}_weight_bytes"),
                ),
                (
                    "c",
                    layout.cache_offset(mode, i),
                    getattr(layout, f"{mode.value}_cache").layer_bytes[i],
                ),
            ):
                assert offset % 256 == 0
                assert 0 <= offset < offset + size <= layout.budget
                assert offset + size <= ws or ws + length <= offset
                entries[mode, kind, i] = (offset, offset + size)
        assert layout.weight_offset(mode, n - 1) + getattr(
            layout, f"{mode.value}_weight_bytes"
        ) <= layout.cache_offset(mode, 0)
    for source, target, kinds, ids in (
        (ParaSMode.EP, ParaSMode.TP, ("w", "c"), range(n)),
        (ParaSMode.TP, ParaSMode.EP, ("c", "w"), reversed(range(n))),
    ):
        ids = list(ids)
        order = [(kind, i) for kind in kinds for i in ids]
        unread = {(kind, i): entries[source, kind, i] for kind, i in order}
        for kind, i in order:
            start, end = entries[target, kind, i]
            assert all(end <= s or e <= start for s, e in unread.values())
            del unread[kind, i]


def test_qwen235_asymmetric_workspaces():
    mib = 1 << 20
    layout = plan_unified_layout(
        num_layers=94,
        budget=130 << 30,
        ep_weight_bytes=712 * mib,
        tp_weight_bytes=594 * mib,
        ep_workspace_bytes=512 * mib,
        tp_workspace_bytes=1280 * mib,
        ep_kv_row_bytes=2048,
        tp_kv_row_bytes=512,
    )
    assert layout.ep_front == 594 * mib
    assert layout.tp_tail == 1280 * mib
    assert (layout.ep_cache.full_tokens + 1) * 2048 * 94 == 68784345088
    assert (layout.tp_cache.full_tokens + 1) * 512 * 94 == 79695925248
    assert_safe(layout)


@pytest.mark.parametrize("page_size", [1, 16, 64])
def test_transfer_safety_with_rounding(page_size):
    rng = random.Random(231)
    for _ in range(100):
        layers = rng.randrange(1, 20)
        wt = rng.randrange(1, 256) * 256
        we = wt + rng.randrange(1, 256) * 256
        layout = plan_unified_layout(
            num_layers=layers,
            budget=32 << 20,
            ep_weight_bytes=we,
            tp_weight_bytes=wt,
            ep_workspace_bytes=1 << 20,
            tp_workspace_bytes=(1 << 20) + layers * (we - wt) // 2,
            ep_kv_row_bytes=2048,
            tp_kv_row_bytes=512,
            page_size=page_size,
        )
        assert (
            layout.ep_cache.full_tokens % page_size
            == layout.tp_cache.full_tokens % page_size
            == 0
        )
        assert_safe(layout)


def test_insufficient_budget_fails_before_allocation():
    with pytest.raises(ValueError, match="budget"):
        plan_unified_layout(
            num_layers=94,
            budget=1 << 30,
            ep_weight_bytes=712 << 20,
            tp_weight_bytes=594 << 20,
            ep_workspace_bytes=1 << 20,
            tp_workspace_bytes=2 << 20,
            ep_kv_row_bytes=2048,
            tp_kv_row_bytes=512,
        )


def test_qwen30_tp_workspace_exceeds_attention_saving():
    ep_ws, tp_ws = bf16_moe_workspace_sizes(
        hidden_size=2048,
        intermediate_size=768,
        num_experts=128,
        top_k=8,
        tp_size=4,
        dispatch_capacity=512,
    )
    # EP capacity follows DeepEP's receive shape, independently of TP chunking.
    small_ep, same_tp = bf16_moe_workspace_sizes(
        hidden_size=2048,
        intermediate_size=768,
        num_experts=128,
        top_k=8,
        tp_size=4,
        dispatch_capacity=128,
    )
    assert small_ep == ep_ws // 4
    assert same_tp == tp_ws
    layout = plan_unified_layout(
        num_layers=48,
        budget=56763796480,
        ep_weight_bytes=324 << 20,
        tp_weight_bytes=297 << 20,
        ep_workspace_bytes=ep_ws,
        tp_workspace_bytes=tp_ws,
        ep_kv_row_bytes=2048,
        tp_kv_row_bytes=512,
    )
    assert layout.ep_front >= tp_ws - 48 * (27 << 20)
    assert layout.ep_front > max(ep_ws, 297 << 20)
    assert max(layout.tp_cache.layer_bytes) >= max(layout.ep_cache.layer_bytes)
    assert_safe(layout)


@pytest.mark.parametrize("page_size", [1, 16])
@pytest.mark.parametrize("swa_ratio", [0.5, 0.8, 1.0])
@pytest.mark.parametrize("tp_row_bytes", [256, 512])
def test_gpt_oss_hybrid_cache_transfer_safety(page_size, swa_ratio, tp_row_bytes):
    layout = plan_unified_layout(
        num_layers=36,
        budget=130 << 30,
        ep_weight_bytes=900 << 20,
        tp_weight_bytes=820 << 20,
        ep_workspace_bytes=768 << 20,
        tp_workspace_bytes=2 << 30,
        ep_kv_row_bytes=2048,
        tp_kv_row_bytes=tp_row_bytes,
        page_size=page_size,
        layer_token_ratios=(swa_ratio, 1.0) * 18,
    )
    for cache in (layout.ep_cache, layout.tp_cache):
        assert (
            cache.layer_tokens[0]
            == int(cache.full_tokens * swa_ratio) // page_size * page_size
        )
        assert cache.layer_tokens[1] == cache.full_tokens
        assert cache.layer_bytes[0] <= cache.layer_bytes[1]
    assert_safe(layout)


def test_configured_workspace_recovers_kv_without_changing_transfer_safety():
    kwargs = dict(
        hidden_size=2880,
        intermediate_size=2880,
        num_experts=128,
        top_k=4,
        tp_size=8,
        dispatch_capacity=256,
    )
    ep, legacy_tp = bf16_moe_workspace_sizes(**kwargs)
    same_ep, configured_tp = bf16_moe_workspace_sizes(**kwargs, tp_input_tokens=8192)
    assert same_ep == ep
    assert configured_tp < legacy_tp / 4
    assert bf16_moe_workspace_sizes(**kwargs, tp_input_tokens=131072) == (ep, legacy_tp)
    layouts = [
        plan_unified_layout(
            num_layers=36,
            budget=57 << 30,
            ep_weight_bytes=849346560,
            tp_weight_bytes=802897920,
            ep_workspace_bytes=ep + (260 << 20),
            tp_workspace_bytes=tp + (int(32.5 * (1 << 20))),
            ep_kv_row_bytes=2048,
            tp_kv_row_bytes=256,
        )
        for tp in (legacy_tp, configured_tp)
    ]
    assert layouts[1].tp_cache.full_tokens > layouts[0].tp_cache.full_tokens
    assert layouts[1].ep_cache.full_tokens >= layouts[0].ep_cache.full_tokens
    for layout in layouts:
        assert_safe(layout)
