"""CPU regression: wider attention weights must retain local Torch slicing."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.paras.attention_transfer import transfer_attention
from sglang.srt.paras.mode import ParaSMode


@pytest.mark.parametrize("rank", [0, 1])
def test_float32_ep_to_tp_bypasses_cuda_vector_copy(rank):
    h, q, kv, head, tp_size = 16, 32, 8, 8, 2
    prefix = "model.layers.0.self_attn"
    qkv_ep = torch.arange((q + 2 * kv) * h, dtype=torch.float32).reshape(-1, h)
    o_ep = torch.arange(h * q, dtype=torch.float32).reshape(h, q)
    qkv_tp = torch.full((q // tp_size + 2 * head, h), float("nan"))
    o_tp = torch.full((h, q // tp_size), float("nan"))
    views = {
        f"{prefix}.qkv_proj.weight": qkv_ep,
        f"{prefix}.qkv_proj.tp_weight": qkv_tp,
        f"{prefix}.o_proj.weight": o_ep,
        f"{prefix}.o_proj.tp_weight": o_tp,
    }
    # All shape/address checks otherwise pass, isolating the element-size guard.
    assert all(t.data_ptr() % 16 == 0 for t in views.values())
    manager = SimpleNamespace(
        _unified_spec=SimpleNamespace(
            prefix="model", hidden_size=h, head_dim=head, tp_size=tp_size,
            num_heads=q // head, num_kv_heads=kv // head,
        ),
        _entries={name: SimpleNamespace(offset_bytes=i * 4096)
                  for i, name in enumerate(views)},
        get_view=views.__getitem__,
    )
    cuda = SimpleNamespace(
        launch_attention_slice=Mock(side_effect=AssertionError("FP32 reached CUDA copy")),
        launch_attention_restore=Mock(side_effect=AssertionError("FP32 reached CUDA copy")),
    )
    with patch.dict(sys.modules, {"paras_peer_access_cuda": cuda}):
        transfer_attention(manager, 0, ParaSMode.TP, rank, torch.zeros(tp_size, dtype=torch.int64))
    expected_qkv = torch.cat((qkv_ep[rank * 16:(rank + 1) * 16], qkv_ep[q:q + kv], qkv_ep[q + kv:]))
    torch.testing.assert_close(qkv_tp, expected_qkv, rtol=0, atol=0)
    torch.testing.assert_close(o_tp, o_ep[:, rank * 16:(rank + 1) * 16], rtol=0, atol=0)
    cuda.launch_attention_slice.assert_not_called()
    cuda.launch_attention_restore.assert_not_called()
