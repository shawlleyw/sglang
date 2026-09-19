"""Backend sizing and combined scratch/transfer geometry, without CUDA."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.paras.mode import ParaSMode
from sglang.srt.paras.unified_layout import plan_unified_layout
from sglang.srt.paras.workspace import (
    ModeWorkspaces,
    WorkspaceRequirement,
    attention_workspace_requirements,
    triton_attention_split_config,
)


def requirements(backend, *, resolved_context_len=None, num_heads=64, **overrides):
    args = dict(
        attention_backend=backend,
        enable_two_batch_overlap=False,
        enable_pdmux=False,
        speculative_algorithm=None,
        prefill_attention_backend=None,
        decode_attention_backend=None,
        enable_deterministic_inference=False,
        triton_attention_num_kv_splits=8,
        triton_attention_split_tile_size=None,
        max_running_requests=2048,
        disable_cuda_graph=False,
        cuda_graph_bs=[1, 256],
        paras_tp_cuda_graph_bs=[1, 2048],
        context_length=None,
    )
    args.update(overrides)
    config = SimpleNamespace(
        architectures=["Qwen3MoeForCausalLM"],
        num_attention_heads=num_heads,
        max_position_embeddings=32768,
    )
    return attention_workspace_requirements(
        SimpleNamespace(**args),
        config,
        8,
        128,
        context_len=resolved_context_len or args["context_length"] or 32768,
    )


def test_flashinfer_configuration(monkeypatch):
    monkeypatch.setenv("SGLANG_FLASHINFER_WORKSPACE_SIZE", str(384 << 20))
    assert [r.size_bytes for r in requirements("flashinfer")] == [384 << 20] * 2
    monkeypatch.setenv("SGLANG_FLASHINFER_WORKSPACE_SIZE", str(768 << 20))
    assert requirements("flashinfer")[0].size_bytes == 768 << 20
    assert (
        requirements("flashinfer", enable_deterministic_inference=True)[0].size_bytes
        == 2 << 30
    )


def test_triton_preserves_existing_graph_capacity():
    ep, tp = requirements("triton")
    assert ep.size_bytes == 516 << 20
    assert tp.size_bytes == int(64.5 * (1 << 20))
    ep, tp = requirements("triton", disable_cuda_graph=True)
    assert ep.size_bytes == tp.size_bytes == int(64.5 * (1 << 20))
    ep, tp = requirements(
        "triton", context_length=65536, triton_attention_split_tile_size=4096
    )
    assert ep.size_bytes == 1032 << 20
    assert tp.size_bytes == 129 << 20


def test_triton_uses_resolved_rope_context():
    # The raw HF limit is 32768, but ModelConfig resolves 4x RoPE to 131072.
    ep, tp = requirements(
        "triton",
        resolved_context_len=131072,
        num_heads=32,
        triton_attention_split_tile_size=4096,
    )
    assert ep.size_bytes == 1032 << 20
    assert tp.size_bytes == 129 << 20


def test_triton_deterministic_split_config(monkeypatch):
    monkeypatch.setenv("SGLANG_TRITON_DECODE_SPLIT_TILE_SIZE", "4096")
    args = SimpleNamespace(
        enable_deterministic_inference=True,
        triton_attention_num_kv_splits=8,
        triton_attention_split_tile_size=None,
    )
    assert triton_attention_split_config(args, 131072) == (32, 4096)
    with pytest.raises(ValueError, match="resolved context"):
        triton_attention_split_config(args, None)


@pytest.mark.parametrize(
    "backend,overrides",
    [
        ("fa3", {}),
        ("triton", {"max_running_requests": None}),
        ("flashinfer", {"enable_two_batch_overlap": True}),
        ("flashinfer", {"enable_pdmux": True}),
        ("flashinfer", {"decode_attention_backend": "triton"}),
        ("flashinfer", {"speculative_algorithm": "EAGLE"}),
    ],
)
def test_external_workspace_is_not_reported_as_zero(backend, overrides):
    assert all(r.size_bytes is None for r in requirements(backend, **overrides))


@pytest.mark.parametrize("attention_mib", [0, 64.5, 384, 1024])
def test_combined_workspace_on_either_side_of_transfer_gap(attention_mib):
    mib = 1 << 20
    ep, tp = (
        ModeWorkspaces(
            WorkspaceRequirement("triton", moe * mib),
            WorkspaceRequirement("attention", int(attention_mib * mib)),
        )
        for moe in (288, 536)
    )
    layout = plan_unified_layout(
        num_layers=94,
        budget=130 << 30,
        ep_weight_bytes=712 * mib,
        tp_weight_bytes=594 * mib,
        ep_workspace_bytes=ep.size_bytes,
        tp_workspace_bytes=tp.size_bytes,
        ep_kv_row_bytes=2048,
        tp_kv_row_bytes=512,
    )
    assert layout.ep_front == max(594 * mib, ep.size_bytes)
    # Recompute the TP gap from the resulting EP KV capacity.
    assert layout.tp_tail == max(max(layout.ep_cache.layer_bytes), tp.size_bytes)
    assert max(layout.tp_cache.layer_bytes) >= max(layout.ep_cache.layer_bytes)


def test_suballocation_alignment_and_validation():
    workspaces = ModeWorkspaces(
        WorkspaceRequirement("moe", 257), WorkspaceRequirement("attn", 1)
    )
    assert workspaces.region("moe") == (0, 512)
    assert workspaces.region("attention") == (512, 256)
    assert workspaces.size_bytes == 768
    with pytest.raises(ValueError):
        WorkspaceRequirement("moe", -1)


@pytest.mark.parametrize("backend_name", ["flashinfer", "triton"])
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Backend imports require CUDA"
)
def test_rebinding_does_not_overwrite_source_weights(monkeypatch, backend_name):
    from sglang.srt.paras.paras_memory_manager import (
        ParaSMemoryManager,
        UnifiedLayoutSpec,
        UnifiedModeSpec,
    )

    mgr = ParaSMemoryManager(device="cpu")
    mgr._buffer = torch.full((16384,), 23, dtype=torch.uint8)
    mgr._unified_layout = SimpleNamespace(workspace=lambda mode: (8192, 8192))
    workspaces = ModeWorkspaces(
        WorkspaceRequirement("triton", 4096),
        WorkspaceRequirement(backend_name, 4096),
    )
    mgr._unified_spec = UnifiedLayoutSpec(
        num_layers=1,
        prefix="model",
        tp_size=8,
        num_heads=32,
        num_kv_heads=4,
        head_dim=128,
        hidden_size=2048,
        ep=UnifiedModeSpec(workspaces),
        tp=UnifiedModeSpec(workspaces),
    )
    if backend_name == "flashinfer":
        from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend

        # A managed buffer uses its planned capacity, not a later env value.
        monkeypatch.setenv("SGLANG_FLASHINFER_WORKSPACE_SIZE", "8192")
        backend = FlashInferAttnBackend.__new__(FlashInferAttnBackend)
        backend._paras_memory_manager = mgr
        plan = object()
        wrapper = SimpleNamespace(_int_workspace_buffer=plan)
        backend.prefill_wrapper_ragged = wrapper
        backend.prefill_wrappers_paged = []
        backend.prefill_wrappers_verify = []
        backend.decode_wrappers = []
        backend._paras_bind_workspace(ParaSMode.TP)
        assert wrapper._int_workspace_buffer is plan
        assert wrapper._float_workspace_buffer.numel() == 4096
        assert (
            wrapper._float_workspace_buffer.data_ptr() == mgr.buffer[12288:].data_ptr()
        )
    else:
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

        backend = TritonAttnBackend.__new__(TritonAttnBackend)
        backend._paras_memory_manager = mgr
        backend._paras_workspace_mode = ParaSMode.TP
        backend.num_head = backend.max_kv_splits = 2
        backend.v_head_dim = 8
        backend.device = "cpu"
        backend._allocate_decode_workspace(2, zero=True)

    # A configure hook runs before source weights have necessarily moved.
    assert torch.all(mgr.buffer == 23)
    backend.paras_initialize_workspace()
    assert torch.all(mgr.buffer[:12288] == 23)
    assert torch.all(mgr.buffer[12288:] == 0)
