"""Exercise attention reconstruction after destroying the original full weights."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.paras.mode import ParaSMode


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tp_size,kv_heads", [(4, 8), (8, 4)])
def test_attention_roundtrip_without_full_backup(tp_size, kv_heads):
    from sglang.srt.paras.attention_transfer import transfer_attention

    h, heads, d = 64, 16, 128
    q, kv = heads * d, kv_heads * d
    q_tp, kv_tp = q // tp_size, max(d, kv // tp_size)
    shapes = {"qkv_proj": (q + 2 * kv, h), "o_proj": (h, q)}
    tp_shapes = {"qkv_proj": (q_tp + 2 * kv_tp, h), "o_proj": (h, q_tp)}
    reference = {
        p: torch.randn(s, device="cuda", dtype=torch.bfloat16)
        for p, s in shapes.items()
    }
    buffers, managers = [], []
    for _ in range(tp_size):
        offsets, views, cursor = {}, {}, 256
        for mode, mode_shapes in ((ParaSMode.EP, shapes), (ParaSMode.TP, tp_shapes)):
            for p, shape in mode_shapes.items():
                name = f"model.layers.0.self_attn.{p}.{'weight' if mode == ParaSMode.EP else 'tp_weight'}"
                offsets[name] = SimpleNamespace(offset_bytes=cursor)
                cursor += 2 * shape[0] * shape[1]
        buffer = torch.empty(cursor, device="cuda", dtype=torch.uint8)
        buffers.append(buffer)
        for mode, mode_shapes in ((ParaSMode.EP, shapes), (ParaSMode.TP, tp_shapes)):
            for p, shape in mode_shapes.items():
                name = f"model.layers.0.self_attn.{p}.{'weight' if mode == ParaSMode.EP else 'tp_weight'}"
                offset = offsets[name].offset_bytes
                views[name] = (
                    buffer[offset : offset + 2 * shape[0] * shape[1]]
                    .view(torch.bfloat16)
                    .view(shape)
                )
                if mode == ParaSMode.EP:
                    views[name].copy_(reference[p])
        managers.append(
            SimpleNamespace(
                _unified_spec=SimpleNamespace(
                    prefix="model",
                    hidden_size=h,
                    head_dim=d,
                    tp_size=tp_size,
                    num_heads=heads,
                    num_kv_heads=kv_heads,
                ),
                _entries=offsets,
                get_view=views.__getitem__,
            )
        )
    bases = torch.tensor(
        [b.data_ptr() for b in buffers], device="cuda", dtype=torch.int64
    )
    for rank, manager in enumerate(managers):
        transfer_attention(manager, 0, ParaSMode.TP, rank, bases)
    for manager in managers:
        for p in shapes:
            manager.get_view(f"model.layers.0.self_attn.{p}.weight").fill_(float("nan"))
    for rank, manager in enumerate(managers):
        transfer_attention(manager, 0, ParaSMode.EP, rank, bases)
        for p in shapes:
            torch.testing.assert_close(
                manager.get_view(f"model.layers.0.self_attn.{p}.weight"),
                reference[p],
                rtol=0,
                atol=0,
            )
