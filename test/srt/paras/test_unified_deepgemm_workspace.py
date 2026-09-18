"""Check BF16 DeepGEMM scratch lifetimes; reference GEMMs also run on pre-Hopper GPUs."""

import pytest
import torch

from sglang.srt.paras.mode import ParaSMode
from sglang.srt.paras.workspace import MoEWorkspace


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("fits", [False, True])
def test_deepgemm_bf16_workspace_matches_dynamic_allocations(monkeypatch, masked, fits):
    from sglang.srt.layers import deep_gemm_wrapper
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.deep_gemm import (
        DeepGemmRunnerCore,
        DeepGemmRunnerInput,
        DeepGemmMoeQuantInfo,
    )
    from sglang.srt.paras.unified_layout import align_up

    groups, rows, h, inter = 2, 32, 128, 128
    x = torch.randn(groups, rows, h, device="cuda", dtype=torch.bfloat16) * 0.1
    w13 = torch.randn(groups, 2 * inter, h, device="cuda", dtype=torch.bfloat16) * 0.1
    w2 = torch.randn(groups, h, inter, device="cuda", dtype=torch.bfloat16) * 0.1
    observed_outputs = []

    def grouped(a, w, out, *args):
        observed_outputs.append(out.data_ptr())
        result = torch.bmm(a.reshape(groups, rows, -1), w.transpose(1, 2))
        out.copy_(result.reshape(out.shape))

    suffix = "masked" if masked else "contig"
    monkeypatch.setattr(
        deep_gemm_wrapper, f"grouped_gemm_nt_bf16bf16bf16_{suffix}", grouped
    )
    config = MoeRunnerConfig(activation="silu")
    runner = DeepGemmRunnerCore(config)
    quant = DeepGemmMoeQuantInfo(w13, w2, use_fp8=False)

    def run():
        inputs = DeepGemmRunnerInput(
            hidden_states=x.clone() if masked else x.reshape(-1, h).clone(),
            hidden_states_scale=None,
            use_masked_gemm=masked,
            masked_m=torch.full((groups,), rows, device="cuda", dtype=torch.int32),
            expected_m=rows,
            m_indices=torch.arange(
                groups, device="cuda", dtype=torch.int32
            ).repeat_interleave(rows),
        )
        state = {
            "all_tokens": groups * rows,
            "hidden_states_device": x.device,
            "hidden_states_shape": (groups * rows, h),
        }
        return runner.run(inputs, quant, state).hidden_states

    expected = run()
    required = align_up(groups * rows * 2 * inter * 2) + align_up(
        groups * rows * inter * 2
    )
    capacity = required if fits else 256
    storage = torch.full((capacity + 512,), 23, device="cuda", dtype=torch.uint8)
    config.paras_workspace = MoEWorkspace(ParaSMode.EP, storage[256:-256])
    observed_outputs.clear()
    actual = run()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    start, end = storage.data_ptr() + 256, storage.data_ptr() + 256 + capacity
    assert (start <= observed_outputs[0] < end) == fits
    assert not start <= actual.data_ptr() < end
    assert torch.all(storage[:256] == 23) and torch.all(storage[-256:] == 23)
    # dispose_tensor on a temporary view must not invalidate the binding.
    assert config.paras_workspace.buffer.numel() == capacity
    torch.testing.assert_close(run(), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "ep_size,with_bias,activation,expected",
    [
        (4, False, "silu", True),
        (1, False, "silu", False),
        (4, True, "silu", False),
        (4, False, "gelu", False),
    ],
)
def test_bf16_backend_selection(monkeypatch, ep_size, with_bias, activation, expected):
    from types import SimpleNamespace
    from sglang.srt.layers import deep_gemm_wrapper
    from sglang.srt.layers.moe import utils

    monkeypatch.setattr(deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", True)
    monkeypatch.setattr(
        utils, "get_moe_a2a_backend", lambda: SimpleNamespace(is_deepep=lambda: True)
    )
    assert (
        utils.use_deep_gemm_bf16(ep_size, with_bias=with_bias, activation=activation)
        == expected
    )
