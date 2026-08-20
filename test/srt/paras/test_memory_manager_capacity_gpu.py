#!/usr/bin/env python3
"""Four-GPU CUDA coverage for ParaS UMM capacity planning.

The test replaces only the hardware-derived outer budget with a small,
deterministic value. It still runs the production capacity solver, overlapped
layout placement, and real CUDA allocation on every rank.

Usage:
  CUDA_VISIBLE_DEVICES=1,2,4,5 torchrun --standalone --nproc_per_node=4 \
      test/srt/paras/test_memory_manager_capacity_gpu.py
"""

import gc
import os
import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.join(_TEST_DIR, "..", "..", "..")
sys.path.insert(0, os.path.join(_ROOT_DIR, "python"))

WORLD_SIZE = 4
UMM_BUDGET_BYTES = 64 << 20


def _build_manager(rank: int, *, tp_size: int):
    from sglang.srt.paras.paras_memory_manager import (
        ParaSMemoryManager,
        plan_qwen_moe_layout,
    )

    dp_size = WORLD_SIZE // tp_size
    server_args = SimpleNamespace(
        kv_cache_dtype="auto",
        mem_fraction_static=0.8,
        page_size=1,
    )
    config = SimpleNamespace(
        hidden_size=256,
        num_attention_heads=8,
        num_hidden_layers=4,
        num_key_value_heads=4,
        tie_word_embeddings=True,
        vocab_size=0,
    )
    manager = ParaSMemoryManager(
        device=f"cuda:{rank}",
        gpu_id=rank,
        server_args=server_args,
        world_size=WORLD_SIZE,
    )
    plan_qwen_moe_layout(
        manager,
        num_layers=config.num_hidden_layers,
        num_experts=16,
        hidden_size=config.hidden_size,
        intermediate_size=128,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        head_dim=32,
        ep_size=WORLD_SIZE,
        tp_size=tp_size,
        dp_size=dp_size,
        moe_tp_size=tp_size,
        prefix="model",
    )

    budget_gib = UMM_BUDGET_BYTES / (1 << 30)
    manager._compute_umm_budget_bytes = lambda _config: (
        UMM_BUDGET_BYTES,
        UMM_BUDGET_BYTES,
        0,
        UMM_BUDGET_BYTES,
        0,
        UMM_BUDGET_BYTES,
        budget_gib,
    )
    plan = manager.plan_mha_kv_capacity(
        config=config,
        tp_size=tp_size,
        head_dim=32,
    )
    manager.reserve_kv_cache(
        num_layers=config.num_hidden_layers,
        ep_max_tokens=plan.ep_max_tokens,
        tp_max_tokens=plan.tp_max_tokens,
        num_kv_heads=config.num_key_value_heads,
        head_dim=32,
        kv_dtype=plan.kv_dtype,
        tp_size=tp_size,
        page_size=server_args.page_size,
        prefix="model",
    )
    total_bytes = manager.materialize()
    return manager, plan, total_bytes


def _assert_same_plan_on_every_rank(plan, total_bytes: int, rank: int) -> None:
    local = torch.tensor(
        [
            total_bytes,
            plan.planned_umm_bytes,
            plan.fixed_umm_bytes,
            plan.ep_expert_bytes,
            plan.tp_expert_bytes,
            plan.ep_kv_bytes,
            plan.tp_kv_bytes,
            plan.ep_max_tokens,
            plan.tp_max_tokens,
        ],
        dtype=torch.int64,
        device=f"cuda:{rank}",
    )
    gathered = [torch.empty_like(local) for _ in range(WORLD_SIZE)]
    dist.all_gather(gathered, local)
    assert all(torch.equal(local, peer) for peer in gathered)


def _run_case(rank: int, *, tp_size: int) -> None:
    manager, plan, total_bytes = _build_manager(rank, tp_size=tp_size)
    assert manager._buffer is not None and manager._buffer.is_cuda
    assert manager._buffer.device.index == rank
    assert total_bytes == plan.planned_umm_bytes
    assert total_bytes <= plan.umm_budget_bytes == UMM_BUDGET_BYTES
    assert manager._umm_budget_bytes == plan.umm_budget_bytes

    if tp_size == WORLD_SIZE:
        assert plan.tp_expert_bytes == plan.ep_expert_bytes
        assert plan.tp_kv_bytes == plan.ep_kv_bytes
    else:
        assert tp_size == 2
        assert plan.tp_expert_bytes == 2 * plan.ep_expert_bytes
        assert plan.tp_kv_bytes < plan.ep_kv_bytes

    first_ep_weight = manager.get_view("model.layers.0.mlp.ep_experts.w13_weight")
    last_tp_kv = manager.get_view("model.layers.3.kv.tp.v")
    first_ep_weight.zero_()
    last_tp_kv.zero_()
    torch.cuda.synchronize()
    _assert_same_plan_on_every_rank(plan, total_bytes, rank)

    manager._buffer = None
    del manager
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    assert dist.get_world_size() == WORLD_SIZE
    torch.cuda.set_device(rank)
    try:
        _run_case(rank, tp_size=4)
        _run_case(rank, tp_size=2)
        if rank == 0:
            print(
                "PASS: CUDA UMM planning/materialization validated for "
                "EP4/TP4 and EP4/DP2/TP2",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
