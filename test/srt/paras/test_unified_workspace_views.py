"""Physical workspace views must preserve live weights and disambiguate aliases."""

import pytest
import torch

from sglang.srt.paras import paras_memory_manager as memory
from sglang.srt.paras import unified_layout


def test_workspace_ownership_when_mode_weight_addresses_coincide(monkeypatch):
    monkeypatch.setattr(
        unified_layout, "triton_workspace_sizes", lambda **_: (4096, 8192)
    )
    mgr = memory.ParaSMemoryManager(device="cpu")
    mgr.plan_unified_qwen(
        num_layers=4,
        num_experts=8,
        hidden_size=64,
        intermediate_size=128,
        num_heads=8,
        num_kv_heads=2,
        head_dim=128,
        ep_size=4,
        tp_size=4,
        top_k=2,
        prefix="model",
    )
    spec = mgr._unified_spec
    layout = unified_layout.plan_unified_layout(
        num_layers=4,
        budget=5 << 20,
        ep_weight_bytes=spec["ep_weight_bytes"],
        tp_weight_bytes=spec["tp_weight_bytes"],
        ep_workspace_bytes=4096,
        tp_workspace_bytes=8192,
        ep_kv_row_bytes=1024,
        tp_kv_row_bytes=512,
    )
    mgr._unified_layout = layout
    mgr.reserve_kv_cache(
        num_layers=4,
        ep_max_tokens=layout.ep_tokens,
        tp_max_tokens=layout.tp_tokens,
        num_kv_heads=2,
        head_dim=128,
        kv_dtype=torch.bfloat16,
        tp_size=4,
    )
    mgr.materialize()
    monkeypatch.setattr(memory, "_global_paras_memory_manager", mgr)
    ep_weight = mgr.get_view("model.layers.0.mlp.ep_experts.w13_weight")
    tp_weight = mgr.get_view("model.layers.1.mlp.tp_experts.w13_weight")
    assert ep_weight.data_ptr() == tp_weight.data_ptr()
    assert memory.get_paras_workspace_mode(ep_weight) == "ep"
    assert memory.get_paras_workspace_mode(tp_weight) == "tp"
    for mode in ("ep", "tp"):
        mgr._buffer.fill_(23)
        shapes = [(3, 17), (7, 19)]
        scratch = mgr.get_moe_workspace(mode, shapes, torch.bfloat16, "cpu")
        for view in scratch:
            assert (view.data_ptr() - mgr._buffer.data_ptr()) % 256 == 0
            view.fill_(1)
        offset, size = layout.workspace(mode)
        assert torch.all(mgr._buffer[:offset] == 23)
        assert torch.all(mgr._buffer[offset + size :] == 23)
        with pytest.raises(RuntimeError, match="overflow"):
            mgr.get_moe_workspace(mode, [(size,)], torch.bfloat16, "cpu")
