import pytest

from sglang.srt.paras.paras_memory_manager import (
    ParaSMemoryManager,
    plan_qwen_moe_layout,
)
from sglang.srt.paras.weight_transfer import (
    WeightTransferMethod,
    resolve_weight_transfer_method,
)


def _plan(*, method="direct", ep_size=2, tp_size=2, dp_size=1):
    manager = ParaSMemoryManager(device="cpu")
    plan_qwen_moe_layout(
        manager,
        num_layers=2,
        num_experts=8,
        hidden_size=64,
        intermediate_size=32,
        num_heads=8,
        num_kv_heads=2,
        head_dim=8,
        ep_size=ep_size,
        tp_size=tp_size,
        dp_size=dp_size,
        moe_tp_size=tp_size,
        configure_method=method,
    )
    return manager


def test_supported_methods_are_explicit():
    assert resolve_weight_transfer_method("direct") is WeightTransferMethod.DIRECT
    assert resolve_weight_transfer_method("nccl") is WeightTransferMethod.NCCL


@pytest.mark.parametrize("legacy_method", ["peer_access", "overlap", "all_to_all"])
def test_removed_method_names_fail(legacy_method):
    with pytest.raises(ValueError, match="Unsupported ParaS weight transfer method"):
        resolve_weight_transfer_method(legacy_method)


def test_only_nccl_reserves_staging():
    direct = _plan(method="direct")
    nccl = _plan(method="nccl")

    assert not any(name.startswith("staging.") for name in direct._entries)
    assert {
        "staging.w13_pre_permute",
        "staging.w2_pre_permute",
    }.issubset(nccl._entries)


def test_nccl_rejects_dptp():
    with pytest.raises(ValueError, match="supports only dp_size=1"):
        _plan(method="nccl", ep_size=4, tp_size=2, dp_size=2)


def test_parallel_grid_must_match_ep_size():
    with pytest.raises(ValueError, match=r"ep_size == dp_size \* tp_size"):
        _plan(ep_size=4, tp_size=2, dp_size=1)
