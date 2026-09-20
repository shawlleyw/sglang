"""Workspace reuse preserves original MoE execution, including larger EP batches."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.paras.mode import ParaSMode


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("gpt_oss", [False, True])
@pytest.mark.parametrize(
    "mode,inplace,reserved_rows,core_runner",
    [
        (ParaSMode.EP, False, None, True),
        (ParaSMode.EP, False, 32, True),
        (ParaSMode.TP, False, None, False),
        (ParaSMode.TP, False, 32, False),
        (ParaSMode.TP, True, None, False),
        (ParaSMode.TP, False, 1, False),
        (ParaSMode.TP, True, 1, False),
        (ParaSMode.TP, False, None, True),
        (ParaSMode.TP, False, 32, True),
    ],
)
def test_managed_triton_matches_original_allocations(
    monkeypatch, mode, inplace, reserved_rows, core_runner, gpt_oss
):
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
        fused_experts,
        moe_align_block_size,
    )
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.triton import (
        TritonMoeQuantInfo,
        TritonRunnerCore,
        TritonRunnerInput,
    )
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.paras import paras_memory_manager as memory
    from sglang.srt.paras import unified_layout
    from sglang.srt.paras.workspace import ModeWorkspaces, WorkspaceRequirement
    from sglang.srt import server_args

    monkeypatch.setattr(
        server_args,
        "_global_server_args",
        SimpleNamespace(enable_deterministic_inference=False),
    )

    torch.manual_seed(7)
    m, e, h, inter = 79, 4, 64, 64
    x = torch.randn((m, h), device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn((e, 2 * inter, h), device="cuda", dtype=torch.bfloat16) * 0.1
    w2 = torch.randn((e, h, inter), device="cuda", dtype=torch.bfloat16) * 0.1
    k = 1 if mode == ParaSMode.EP else 2
    ids = torch.randint(e, (m, k), device="cuda", dtype=torch.int64)
    weights = torch.rand((m, k), device="cuda")
    monkeypatch.setattr(memory, "_global_paras_memory_manager", None)
    runner_config = MoeRunnerConfig(
        no_combine=mode == ParaSMode.EP,
        inplace=inplace,
        gemm1_alpha=1.702 if gpt_oss else None,
        gemm1_clamp_limit=7.0 if gpt_oss else None,
    )
    if core_runner:
        config = dict(
            BLOCK_SIZE_M=16,
            BLOCK_SIZE_N=32,
            BLOCK_SIZE_K=32,
            GROUP_SIZE_M=1,
            num_warps=4,
            num_stages=2,
        )
        aligned = moe_align_block_size(ids, config["BLOCK_SIZE_M"], e)
        runner = TritonRunnerCore(runner_config)
        inputs = TritonRunnerInput(x, weights, ids, *aligned)
        quant = TritonMoeQuantInfo(w1, w2)

        def run():
            return runner.run(inputs, quant, {"config": config}).hidden_states

    else:

        def run():
            return fused_experts(
                x.clone() if inplace else x,
                w1,
                w2,
                StandardTopKOutput(weights, ids, None),
                runner_config,
            )

    expected = run()

    # TP keeps its original chunk loop. Both modes fall back to runtime
    # allocations when a batch/chunk exceeds the configured reservation.
    monkeypatch.setattr(unified_layout, "triton_moe_chunk_size", 32)
    rows = m * k if reserved_rows is None else reserved_rows
    size = unified_layout.align_up(rows * 2 * inter * 2)
    size += unified_layout.align_up(rows * inter * 2)
    mgr = memory.ParaSMemoryManager(device="cuda")
    workspaces = ModeWorkspaces(
        WorkspaceRequirement("triton", size),
        WorkspaceRequirement("external", None),
    )
    mgr._unified_spec = memory.UnifiedLayoutSpec(
        num_layers=1,
        prefix="model",
        tp_size=1,
        num_heads=1,
        num_kv_heads=1,
        head_dim=128,
        hidden_size=h,
        ep=memory.UnifiedModeSpec(workspaces),
        tp=memory.UnifiedModeSpec(workspaces),
    )
    mgr._unified_layout = SimpleNamespace(workspace=lambda _: (256, size))
    mgr._buffer = torch.full((size + 512,), 23, device="cuda", dtype=torch.uint8)
    mgr._materialized = True
    # Binding works without a global manager or a weight-address registry.
    runner_config.paras_workspace = mgr.bind_moe_workspace(mode)
    actual = run()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if reserved_rows is not None:
        assert torch.all(mgr._buffer == 23)
    if mode == ParaSMode.TP and not core_runner:
        compiled = torch.compile(run, backend="eager", fullgraph=True)
        torch.testing.assert_close(compiled(), expected, rtol=0, atol=0)
    # Returned outputs must survive reuse by subsequent layers.
    mgr._buffer[256:-256].fill_(0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.all(mgr._buffer[:256] == 23)
    assert torch.all(mgr._buffer[-256:] == 23)
