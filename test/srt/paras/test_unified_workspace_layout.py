"""CPU checks for the asymmetric weight/KV/workspace transfer contract."""

import random

import pytest

from sglang.srt.paras.mode import ParaSMode
from sglang.srt.paras.unified_layout import plan_unified_layout, triton_workspace_sizes


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
                    getattr(layout, f"{mode.value}_cache_bytes"),
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
    assert layout.ep_cache_bytes * 94 == 68784345088
    assert layout.tp_cache_bytes * 94 == 79695925248
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
        assert layout.ep_tokens % page_size == layout.tp_tokens % page_size == 0
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
    ep_ws, tp_ws = triton_workspace_sizes(
        hidden_size=2048,
        intermediate_size=768,
        num_experts=128,
        top_k=8,
        tp_size=4,
        dispatch_capacity=512,
    )
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
    assert layout.tp_cache_bytes >= layout.ep_cache_bytes
    assert_safe(layout)
