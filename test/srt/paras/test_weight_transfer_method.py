import pytest

from sglang.srt.paras.paras_memory_manager import (
    ParaSMemoryManager,
    create_paras_moe_aliases,
    plan_qwen_moe_layout,
)
from sglang.srt.paras.weight_transfer import (
    IntraNodeWeightTransferMethod,
    resolve_intra_node_weight_transfer_method,
)


def _plan(*, method="peer_access", ep_size=2, tp_size=2, dp_size=1):
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
        intra_node_weight_transfer_method=method,
    )
    return manager


def test_supported_intra_node_methods_are_explicit():
    assert (
        resolve_intra_node_weight_transfer_method("peer_access")
        is IntraNodeWeightTransferMethod.PEER_ACCESS
    )
    assert (
        resolve_intra_node_weight_transfer_method("nccl")
        is IntraNodeWeightTransferMethod.NCCL
    )


def test_default_and_environment_select_intra_node_method(monkeypatch):
    monkeypatch.delenv("PARAS_INTRA_NODE_WEIGHT_TRANSFER_METHOD", raising=False)
    assert (
        resolve_intra_node_weight_transfer_method()
        is IntraNodeWeightTransferMethod.PEER_ACCESS
    )

    monkeypatch.setenv("PARAS_INTRA_NODE_WEIGHT_TRANSFER_METHOD", "nccl")
    assert (
        resolve_intra_node_weight_transfer_method()
        is IntraNodeWeightTransferMethod.NCCL
    )


@pytest.mark.parametrize("removed_method", ["direct", "overlap", "all_to_all"])
def test_removed_method_names_fail(removed_method):
    with pytest.raises(
        ValueError, match="Unsupported ParaS intra-node weight transfer method"
    ):
        resolve_intra_node_weight_transfer_method(removed_method)


def test_only_nccl_reserves_intra_node_staging():
    peer_access = _plan(method="peer_access")
    nccl = _plan(method="nccl")

    assert not any(name.startswith("staging.") for name in peer_access._entries)
    assert {
        "staging.w13_pre_permute",
        "staging.w2_pre_permute",
    }.issubset(nccl._entries)


def test_nccl_memory_layout_supports_multiple_tp_instances():
    manager = _plan(method="nccl", ep_size=4, tp_size=2, dp_size=2)
    manager.materialize()
    create_paras_moe_aliases(manager, num_layers=2)

    assert manager.materialized
    assert tuple(manager.get_view("staging.w13_pre_permute").shape) == (2, 64, 64)
    assert tuple(manager.get_view("staging.w2_pre_permute").shape) == (2, 64, 32)
    assert tuple(
        manager.get_view("model.layers.0.mlp.ep_experts.w13_weight").shape
    ) == (2, 64, 64)
    assert tuple(
        manager.get_view("model.layers.0.mlp.tp_experts.w13_weight").shape
    ) == (8, 32, 64)


def test_parallel_grid_must_match_ep_size():
    with pytest.raises(ValueError, match=r"ep_size == dp_size \* tp_size"):
        _plan(ep_size=4, tp_size=2, dp_size=1)
