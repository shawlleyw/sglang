#!/usr/bin/env python3
"""
Smoke test: GPT-OSS ParaS weight transfer + CUDA graph replay across EP↔TP↔EP.

Verifies:
  1. GptOssSparseMoeBlockParaS class chain is importable and well-formed
  2. EP→TP weight transfer via peer access produces correct TP layout
  3. TP→EP reverse transfer restores the original EP weights exactly
  4. CUDA graph captured in EP mode replays correctly after EP→TP→EP
  5. CUDA graph captured in TP mode replays correctly

Usage:
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m pytest \
    test/srt/paras/test_paras_gpt_oss_cuda_graph.py -v
"""

import os
import sys

import pytest
import torch
import torch.distributed as dist

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.join(_TEST_DIR, "..", "..", "..")
sys.path.insert(0, os.path.join(_ROOT_DIR, "python"))

NUM_LAYERS = 4
HIDDEN = 2048
INTERMEDIATE = 1536
NUM_EXPERTS = 64
SEED = 42
BATCH = 32


def setup_distributed():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 4, f"Requires exactly 4 GPUs, got {world_size}"
    torch.cuda.set_device(rank)
    return rank, world_size


def teardown_distributed():
    dist.destroy_process_group()


class _SimpleGroupCoordinator:
    def __init__(self, device_group, world_size, device, rank_in_group=0):
        self.device_group = device_group
        self.world_size = world_size
        self.device = torch.device(device)
        self.rank_in_group = rank_in_group
        self.rank = dist.get_rank()
        self.local_rank = dist.get_rank()


def setup_paras_state(rank, world_size):
    import sglang.srt.distributed.parallel_state as ps
    import sglang.srt.paras.paras_parallel_state as pps

    tp_group = dist.new_group(ranks=list(range(world_size)))
    tp_coord = _SimpleGroupCoordinator(
        tp_group, world_size, f"cuda:{rank}", rank_in_group=rank
    )
    ps._TP = tp_coord
    ps._MOE_EP = tp_coord
    ps._MOE_TP = tp_coord

    pps._PARAS_TP = tp_coord
    pps._PARAS_DP = _SimpleGroupCoordinator(
        None, 1, f"cuda:{rank}", rank_in_group=0
    )
    pps._PARAS_SELF = _SimpleGroupCoordinator(
        None, 1, f"cuda:{rank}", rank_in_group=0
    )
    pps._PARAS_TP_SIZE = world_size
    pps._PARAS_TP_RANK = rank
    pps._PARAS_DP_SIZE = 1
    pps._PARAS_DP_RANK = 0
    pps._PARAS_EP_SIZE = world_size
    pps._PARAS_EP_RANK = rank
    pps._PARAS_EP_GROUP_IS_NODE_LOCAL = True
    pps._PARAS_TP_GROUP_IS_NODE_LOCAL = True

    return tp_group


def build_manager(rank, world_size):
    # Attention geometry not exercised by MoE-only forward paths; num_heads=32
    # and head_dim=64 satisfy num_heads*head_dim==HIDDEN. num_kv_heads=8 mirrors
    # GPT-OSS GQA 4:1. Symmetric switch: moe_tp_size==tp_size==ep_size==world_size.
    from sglang.srt.paras.paras_memory_manager import (
        ParaSMemoryManager,
        create_paras_moe_aliases,
        plan_gpt_oss_moe_layout,
        set_global_paras_memory_manager,
    )

    ep_size = world_size
    num_local = NUM_EXPERTS // ep_size

    mgr = ParaSMemoryManager(device=f"cuda:{rank}")

    plan_gpt_oss_moe_layout(
        mgr,
        num_layers=NUM_LAYERS,
        num_experts=NUM_EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_heads=32,
        num_kv_heads=8,
        head_dim=64,
        ep_size=ep_size,
        tp_size=world_size,
        dp_size=1,
        moe_tp_size=world_size,
        quant_name=None,
        configure_method="direct",
        prefix="model",
    )

    mgr.materialize()
    create_paras_moe_aliases(mgr, NUM_LAYERS, prefix="model")
    set_global_paras_memory_manager(mgr)
    return mgr, num_local


def fill_ep_weights(mgr, rank):
    for layer_id in range(NUM_LAYERS):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(SEED + layer_id * 100 + rank)
        w13 = mgr.get_view(f"model.layers.{layer_id}.mlp.experts.w13_weight")
        w13.copy_(
            torch.randn(w13.shape, generator=gen, dtype=torch.float32).to(
                dtype=w13.dtype, device=w13.device
            )
        )
        gen2 = torch.Generator(device="cpu")
        gen2.manual_seed(SEED + layer_id * 100 + rank + 50)
        w2 = mgr.get_view(f"model.layers.{layer_id}.mlp.experts.w2_weight")
        w2.copy_(
            torch.randn(w2.shape, generator=gen2, dtype=torch.float32).to(
                dtype=w2.dtype, device=w2.device
            )
        )


def snapshot_weights(mgr):
    snap = {}
    for layer_id in range(NUM_LAYERS):
        snap[layer_id] = (
            mgr.get_view(f"model.layers.{layer_id}.mlp.experts.w13_weight").clone(),
            mgr.get_view(f"model.layers.{layer_id}.mlp.experts.w2_weight").clone(),
        )
    return snap


class _MockExperts:
    def __init__(self, w13_view, w2_view):
        self.w13_weight = torch.nn.Parameter(w13_view, requires_grad=False)
        self.w2_weight = torch.nn.Parameter(w2_view, requires_grad=False)


def _make_mixin(layer_id, num_local, mgr, interleaved_w13=False):
    from sglang.srt.paras.layers.paras_moe_block import ParaSMoeBlockMixin

    m = object.__new__(ParaSMoeBlockMixin)
    m._paras_layer_id = layer_id
    m._paras_interleaved_w13 = interleaved_w13
    m.num_local_experts = num_local
    m.num_global_experts = NUM_EXPERTS
    m.hidden_size = HIDDEN
    m.moe_intermediate_size = INTERMEDIATE

    w13 = mgr.get_view(f"model.layers.{layer_id}.mlp.experts.w13_weight")
    w2 = mgr.get_view(f"model.layers.{layer_id}.mlp.experts.w2_weight")
    m.ep_experts = _MockExperts(w13, w2)
    m.w13_ep_gathered = w13.view(num_local, 2 * INTERMEDIATE, HIDDEN)
    m.w2_ep_gathered = w2.view(num_local, HIDDEN, INTERMEDIATE)
    return m


def ep_matmul_on_buffer(mgr, x, layer_id=0):
    """Simulate EP-mode computation: x @ w13[layer].T → (batch, 2*intermediate)."""
    w13 = mgr.get_view(f"model.layers.{layer_id}.mlp.experts.w13_weight")
    w_flat = w13.view(-1, HIDDEN).to(torch.float32)
    return torch.mm(x.to(torch.float32), w_flat.T).to(torch.bfloat16)


def tp_matmul_on_buffer(mgr, x, layer_id=0):
    """Simulate TP-mode computation: x @ tp_w13[layer].T."""
    from sglang.srt.paras.paras_parallel_state import get_paras_tp_size

    tp_size = get_paras_tp_size()
    tp_inter = INTERMEDIATE // tp_size
    tp_w13 = mgr.get_view_as(
        f"model.layers.{layer_id}.mlp.tp_experts.w13_weight",
        (NUM_EXPERTS, 2 * tp_inter, HIDDEN),
    )
    w_flat = tp_w13.view(-1, HIDDEN).to(torch.float32)
    return torch.mm(x.to(torch.float32), w_flat.T).to(torch.bfloat16)


def test_gpt_oss_class_chain_imports():
    """Verify the ParaS GPT-OSS class chain is importable and MRO is correct."""
    from sglang.srt.paras.models.gpt_oss import (
        GptOssAttentionParaS,
        GptOssDecoderLayerParaS,
        GptOssForCausalLMParaS,
        GptOssModelParaS,
        GptOssSparseMoeBlockParaS,
    )
    from sglang.srt.paras.layers.paras_moe_block import ParaSMoeBlockMixin
    from sglang.srt.paras.layers.paras_attention import ParaSAttentionMixin
    from sglang.srt.paras.layers.paras_decoder_layer import ParaSDecoderLayerMixin
    from sglang.srt.paras.layers.paras_model import ParaSModelMixin
    from sglang.srt.models.gpt_oss import (
        GptOssAttention,
        GptOssDecoderLayer,
        GptOssForCausalLM,
        GptOssModel,
        GptOssSparseMoeBlock,
    )

    assert issubclass(GptOssSparseMoeBlockParaS, ParaSMoeBlockMixin)
    assert issubclass(GptOssSparseMoeBlockParaS, GptOssSparseMoeBlock)
    assert issubclass(GptOssAttentionParaS, ParaSAttentionMixin)
    assert issubclass(GptOssAttentionParaS, GptOssAttention)
    assert issubclass(GptOssDecoderLayerParaS, ParaSDecoderLayerMixin)
    assert issubclass(GptOssDecoderLayerParaS, GptOssDecoderLayer)
    assert issubclass(GptOssModelParaS, ParaSModelMixin)
    assert issubclass(GptOssModelParaS, GptOssModel)
    assert issubclass(GptOssForCausalLMParaS, GptOssForCausalLM)


def test_dispatch_table_registration():
    """Verify GptOssForCausalLM is registered in the ParaS dispatch table."""
    from sglang.srt.model_loader.utils import _PARAS_MODEL_REGISTRY, _get_paras_model_class
    from sglang.srt.models.gpt_oss import GptOssForCausalLM
    from sglang.srt.paras.models.gpt_oss import GptOssForCausalLMParaS

    _PARAS_MODEL_REGISTRY.clear()
    result = _get_paras_model_class(GptOssForCausalLM)
    assert result is GptOssForCausalLMParaS


def test_ep_tp_ep_weight_roundtrip_and_cuda_graph():
    """EP→TP→EP weight round-trip + CUDA graph replay on managed buffer."""
    rank, world_size = setup_distributed()
    tp_group = setup_paras_state(rank, world_size)

    try:
        mgr, num_local = build_manager(rank, world_size)
        fill_ep_weights(mgr, rank)
        ep_snap = snapshot_weights(mgr)

        x = torch.randn(BATCH, HIDDEN, dtype=torch.bfloat16, device=f"cuda:{rank}")

        # --- EP eager forward ---
        y_ep_eager = ep_matmul_on_buffer(mgr, x, layer_id=0)

        # --- EP CUDA graph capture ---
        x_graph = x.clone()
        y_graph = torch.empty_like(y_ep_eager)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                y_graph.copy_(ep_matmul_on_buffer(mgr, x_graph, layer_id=0))
        torch.cuda.current_stream().wait_stream(s)

        ep_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(ep_graph, stream=s):
            y_graph.copy_(ep_matmul_on_buffer(mgr, x_graph, layer_id=0))

        ep_graph.replay()
        torch.cuda.synchronize()
        assert torch.allclose(
            y_graph, y_ep_eager, atol=1e-3, rtol=1e-2
        ), "EP graph replay != EP eager"

        # --- EP→TP weight transfer (peer access) ---
        from sglang.srt.paras.peer_access import init_peer_access
        from sglang.srt.paras.paras_parallel_state import get_paras_tp_group, get_paras_tp_size

        peer_ctx = init_peer_access(mgr, get_paras_tp_group().device_group, get_paras_tp_size())
        dst_base_ptrs = torch.tensor(
            peer_ctx.peer_addresses, dtype=torch.int64, device="cuda"
        )

        paras_tp_group = get_paras_tp_group().device_group
        barrier_tensor = torch.zeros(1, device="cuda")
        dist.barrier(group=paras_tp_group)

        # EP->TP reverse: TP weight i overlaps EP weight i+1 (four-anchor).
        for layer_id in reversed(range(NUM_LAYERS)):
            mixin = _make_mixin(layer_id, num_local, mgr)
            mixin.paras_reshard_ep_to_tp_peer(dst_base_ptrs, None)
            dist.all_reduce(barrier_tensor, op=dist.ReduceOp.SUM, group=paras_tp_group)

        # --- TP eager forward ---
        y_tp_eager = tp_matmul_on_buffer(mgr, x, layer_id=0)

        # --- TP CUDA graph capture ---
        x_tp_graph = x.clone()
        y_tp_graph = torch.empty_like(y_tp_eager)
        s_tp = torch.cuda.Stream()
        s_tp.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s_tp):
            for _ in range(3):
                y_tp_graph.copy_(tp_matmul_on_buffer(mgr, x_tp_graph, layer_id=0))
        torch.cuda.current_stream().wait_stream(s_tp)

        tp_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(tp_graph, stream=s_tp):
            y_tp_graph.copy_(tp_matmul_on_buffer(mgr, x_tp_graph, layer_id=0))

        tp_graph.replay()
        torch.cuda.synchronize()
        assert torch.allclose(
            y_tp_graph, y_tp_eager, atol=1e-3, rtol=1e-2
        ), "TP graph replay != TP eager"

        # --- TP→EP reverse weight transfer (peer access) ---
        dist.barrier(group=paras_tp_group)
        barrier_tensor.zero_()
        # TP->EP forward: EP weight i+1 overlaps TP weight i (four-anchor).
        for layer_id in range(NUM_LAYERS):
            mixin = _make_mixin(layer_id, num_local, mgr)
            mixin.paras_reshard_tp_to_ep_peer(dst_base_ptrs, None)
            dist.all_reduce(barrier_tensor, op=dist.ReduceOp.SUM, group=paras_tp_group)

        # --- Verify EP weights restored ---
        for layer_id in range(NUM_LAYERS):
            w13_now = mgr.get_view(
                f"model.layers.{layer_id}.mlp.experts.w13_weight"
            )
            w2_now = mgr.get_view(
                f"model.layers.{layer_id}.mlp.experts.w2_weight"
            )
            assert torch.equal(
                w13_now, ep_snap[layer_id][0]
            ), f"Layer {layer_id} w13 mismatch after round-trip"
            assert torch.equal(
                w2_now, ep_snap[layer_id][1]
            ), f"Layer {layer_id} w2 mismatch after round-trip"

        # --- Replay EP graph after round-trip ---
        ep_graph.replay()
        torch.cuda.synchronize()
        assert torch.allclose(
            y_graph, y_ep_eager, atol=1e-3, rtol=1e-2
        ), "EP graph replay after EP→TP→EP != original EP eager"

        if rank == 0:
            print("PASS: all assertions passed")

    finally:
        teardown_distributed()


def build_manager_with_bias(rank, world_size):
    # all_to_all / naive path variant: plan_gpt_oss_moe_layout with
    # configure_method="nccl" reserves staging.{w13,w2}_pre_permute for
    # NCCL transport. Biases are no longer part of the four-anchor MoE layout;
    # the direct reserves below preserve the test's mock-experts pattern
    # (see _MockExpertsWithBias / fill_biases) via regular reserve() entries.
    from sglang.srt.paras.paras_memory_manager import (
        ParaSMemoryManager,
        create_paras_moe_aliases,
        plan_gpt_oss_moe_layout,
        set_global_paras_memory_manager,
    )

    ep_size = world_size
    num_local = NUM_EXPERTS // ep_size
    tp_inter = INTERMEDIATE // world_size

    mgr = ParaSMemoryManager(device=f"cuda:{rank}")

    for i in range(NUM_LAYERS):
        mgr.reserve(
            f"model.layers.{i}.mlp.experts.w13_weight_bias",
            (num_local, 2 * INTERMEDIATE),
            torch.float32,
        )
        mgr.reserve(
            f"model.layers.{i}.mlp.tp_experts.w13_weight_bias",
            (NUM_EXPERTS, 2 * tp_inter),
            torch.float32,
        )

    plan_gpt_oss_moe_layout(
        mgr,
        num_layers=NUM_LAYERS,
        num_experts=NUM_EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_heads=32,
        num_kv_heads=8,
        head_dim=64,
        ep_size=ep_size,
        tp_size=world_size,
        dp_size=1,
        moe_tp_size=world_size,
        quant_name=None,
        configure_method="nccl",
        prefix="model",
    )

    mgr.materialize()
    create_paras_moe_aliases(mgr, NUM_LAYERS, prefix="model")
    set_global_paras_memory_manager(mgr)
    return mgr, num_local


def fill_biases(mgr, rank):
    # Bias is materialized statically for BOTH EP and TP and is never moved by
    # the EP<->TP switch, so fill both static buffers at "load time".
    for layer_id in range(NUM_LAYERS):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(SEED + layer_id * 100 + rank + 77)
        for name in (
            f"model.layers.{layer_id}.mlp.experts.w13_weight_bias",
            f"model.layers.{layer_id}.mlp.tp_experts.w13_weight_bias",
        ):
            b = mgr.get_view(name)
            b.copy_(
                torch.randn(b.shape, generator=gen, dtype=torch.float32).to(
                    device=b.device
                )
            )


def snapshot_biases(mgr):
    snap = {}
    for layer_id in range(NUM_LAYERS):
        snap[layer_id] = {
            "ep": mgr.get_view(
                f"model.layers.{layer_id}.mlp.experts.w13_weight_bias"
            ).clone(),
            "tp": mgr.get_view(
                f"model.layers.{layer_id}.mlp.tp_experts.w13_weight_bias"
            ).clone(),
        }
    return snap


class _MockExpertsWithBias:
    def __init__(self, w13_view, w2_view, b13_view):
        self.w13_weight = torch.nn.Parameter(w13_view, requires_grad=False)
        self.w2_weight = torch.nn.Parameter(w2_view, requires_grad=False)
        self.w13_weight_bias = torch.nn.Parameter(b13_view, requires_grad=False)


def _make_mixin_with_bias(layer_id, num_local, mgr, interleaved_w13=False):
    from sglang.srt.paras.layers.paras_moe_block import ParaSMoeBlockMixin

    m = object.__new__(ParaSMoeBlockMixin)
    m._paras_layer_id = layer_id
    m._paras_interleaved_w13 = interleaved_w13
    m.num_local_experts = num_local
    m.num_global_experts = NUM_EXPERTS
    m.hidden_size = HIDDEN
    m.moe_intermediate_size = INTERMEDIATE
    m.tp_size = 1

    w13 = mgr.get_view(f"model.layers.{layer_id}.mlp.experts.w13_weight")
    w2 = mgr.get_view(f"model.layers.{layer_id}.mlp.experts.w2_weight")
    b13 = mgr.get_view(f"model.layers.{layer_id}.mlp.experts.w13_weight_bias")
    m.ep_experts = _MockExpertsWithBias(w13, w2, b13)
    m.w13_ep_gathered = w13.view(num_local, 2 * INTERMEDIATE, HIDDEN)
    m.w2_ep_gathered = w2.view(num_local, HIDDEN, INTERMEDIATE)
    return m


def _run_bias_roundtrip(interleaved_w13: bool):
    rank, world_size = setup_distributed()
    setup_paras_state(rank, world_size)

    try:
        mgr, num_local = build_manager_with_bias(rank, world_size)
        fill_ep_weights(mgr, rank)
        fill_biases(mgr, rank)
        ep_weight_snap = snapshot_weights(mgr)
        ep_bias_snap = snapshot_biases(mgr)

        # EP->TP reverse: TP weight i overlaps EP weight i+1 (four-anchor).
        for layer_id in reversed(range(NUM_LAYERS)):
            mixin = _make_mixin_with_bias(
                layer_id, num_local, mgr, interleaved_w13=interleaved_w13
            )
            mixin.paras_reshard_ep_to_tp_nccl()

        # Bias is static: the switch must NOT touch the TP bias buffer. It stays
        # equal to its statically-filled value (and is non-zero).
        for layer_id in range(NUM_LAYERS):
            tp_b13 = mgr.get_view(
                f"model.layers.{layer_id}.mlp.tp_experts.w13_weight_bias"
            )
            assert torch.equal(tp_b13, ep_bias_snap[layer_id]["tp"]), (
                f"Layer {layer_id} TP bias changed during EP->TP (must be static)"
            )
            assert tp_b13.abs().sum() > 0, f"Layer {layer_id} TP bias unexpectedly zero"

        # TP->EP forward: EP weight i+1 overlaps TP weight i (four-anchor).
        for layer_id in range(NUM_LAYERS):
            mixin = _make_mixin_with_bias(
                layer_id, num_local, mgr, interleaved_w13=interleaved_w13
            )
            mixin.paras_reshard_tp_to_ep_nccl()

        for layer_id in range(NUM_LAYERS):
            w13_now = mgr.get_view(
                f"model.layers.{layer_id}.mlp.experts.w13_weight"
            )
            w2_now = mgr.get_view(
                f"model.layers.{layer_id}.mlp.experts.w2_weight"
            )
            b13_now = mgr.get_view(
                f"model.layers.{layer_id}.mlp.experts.w13_weight_bias"
            )
            assert torch.equal(w13_now, ep_weight_snap[layer_id][0]), (
                f"Layer {layer_id} w13 mismatch after bias round-trip"
            )
            assert torch.equal(w2_now, ep_weight_snap[layer_id][1]), (
                f"Layer {layer_id} w2 mismatch after bias round-trip"
            )
            assert torch.equal(b13_now, ep_bias_snap[layer_id]["ep"]), (
                f"Layer {layer_id} EP w13_bias changed (must be static)"
            )
            tp_b13_now = mgr.get_view(
                f"model.layers.{layer_id}.mlp.tp_experts.w13_weight_bias"
            )
            assert torch.equal(tp_b13_now, ep_bias_snap[layer_id]["tp"]), (
                f"Layer {layer_id} TP w13_bias changed (must be static)"
            )

        if rank == 0:
            print(
                "PASS: gpt-oss bias NCCL round-trip (interleaved_w13="
                f"{interleaved_w13})"
            )

    finally:
        teardown_distributed()


def test_gpt_oss_bias_nccl_roundtrip_concat():
    """EP→TP→EP round-trip with concatenated [gate|up] w13 layout (Qwen3)."""
    _run_bias_roundtrip(interleaved_w13=False)


def test_gpt_oss_bias_nccl_roundtrip_interleaved():
    """EP→TP→EP round-trip with interleaved [g0,u0,g1,u1,…] w13 layout (GPT-OSS)."""
    _run_bias_roundtrip(interleaved_w13=True)


def _fill_tagged_w13(mgr, rank, interleaved_w13: bool, world_size: int):
    """Seed w13 with a bf16-exact gate/up + sender-rank tag per position.

    Value at every (expert_local, j, h) is:
        v = sender_rank + (0.5 if j is an up-slot else 0.0)
    Valid sender_rank <= 4 keeps v in {0.0, 0.5, 1.0, ..., 3.5}, all
    exactly representable in bf16 (no rounding loss through copy_ into
    the bf16 UMM buffer).  After EP->TP transport to rank r:
        tp[e_total, j, :] must equal sender + (0.5 if up-slot else 0.0)
    where sender = e_total // num_local, and the "up-slot" rule depends
    on the layout flag:
        interleaved : j-odd  == up
        concat      : j >= I'  == up
    """
    num_local = NUM_EXPERTS // world_size
    for layer_id in range(NUM_LAYERS):
        w13 = mgr.get_view(
            f"model.layers.{layer_id}.mlp.experts.w13_weight"
        )
        j_idx = torch.arange(2 * INTERMEDIATE, device=w13.device)
        if interleaved_w13:
            is_up = (j_idx % 2).to(w13.dtype)
        else:
            is_up = (j_idx // INTERMEDIATE).to(w13.dtype)
        value_row = float(rank) + 0.5 * is_up
        filled = value_row.view(1, 2 * INTERMEDIATE, 1).expand(
            num_local, 2 * INTERMEDIATE, HIDDEN
        ).contiguous()
        w13.copy_(filled)


def _assert_tp_w13_layout(mgr, rank, world_size, interleaved_w13: bool):
    tp_size = world_size
    tp_inter = INTERMEDIATE // tp_size
    num_local = NUM_EXPERTS // world_size
    for layer_id in range(NUM_LAYERS):
        tp_w13 = mgr.get_view_as(
            f"model.layers.{layer_id}.mlp.tp_experts.w13_weight",
            (NUM_EXPERTS, 2 * tp_inter, HIDDEN),
        )
        tp_f32 = tp_w13.to(torch.float32)
        sender = (
            torch.arange(NUM_EXPERTS, device=tp_f32.device) // num_local
        ).to(torch.float32)
        j_idx = torch.arange(2 * tp_inter, device=tp_f32.device)
        if interleaved_w13:
            is_up = (j_idx % 2).to(torch.float32)
        else:
            is_up = (j_idx // tp_inter).to(torch.float32)
        expected_row = sender.view(NUM_EXPERTS, 1) + 0.5 * is_up.view(
            1, 2 * tp_inter
        )
        expected_full = expected_row.view(NUM_EXPERTS, 2 * tp_inter, 1).expand(
            NUM_EXPERTS, 2 * tp_inter, HIDDEN
        )
        assert torch.equal(tp_f32, expected_full), (
            f"Layer {layer_id} rank {rank}: w13 layout mismatch for "
            f"interleaved_w13={interleaved_w13}. "
            f"First mismatch at indices {(tp_f32 != expected_full).nonzero()[:3].tolist()}"
        )


def _run_layout_semantics_test(interleaved_w13: bool):
    rank, world_size = setup_distributed()
    setup_paras_state(rank, world_size)

    try:
        mgr, num_local = build_manager_with_bias(rank, world_size)
        _fill_tagged_w13(mgr, rank, interleaved_w13, world_size)
        fill_biases(mgr, rank)

        # EP->TP reverse: TP weight i overlaps EP weight i+1 (four-anchor).
        for layer_id in reversed(range(NUM_LAYERS)):
            mixin = _make_mixin_with_bias(
                layer_id, num_local, mgr, interleaved_w13=interleaved_w13
            )
            mixin.paras_reshard_ep_to_tp_nccl()

        _assert_tp_w13_layout(mgr, rank, world_size, interleaved_w13)

        if rank == 0:
            print(
                "PASS: layout semantics (interleaved_w13="
                f"{interleaved_w13})"
            )

    finally:
        teardown_distributed()


def test_gpt_oss_w13_layout_semantics_concat():
    """Concat layout: TP rank r's 2*I' axis must be [gate(I') | up(I')]."""
    _run_layout_semantics_test(interleaved_w13=False)


def test_gpt_oss_w13_layout_semantics_interleaved():
    """Interleaved layout: TP rank r's 2*I' axis must be [g0,u0,g1,u1,…]."""
    _run_layout_semantics_test(interleaved_w13=True)


def _run_peer_access_layout_semantics_test(interleaved_w13: bool):
    """Same semantics check as _run_layout_semantics_test but via peer_access.

    Verifies that ``paras_reshard_ep_to_tp_peer`` produces the same TP layout
    as the NCCL ``paras_reshard_ep_to_tp_nccl`` path
    for both concat and interleaved w13 layouts.
    """
    rank, world_size = setup_distributed()
    tp_group = setup_paras_state(rank, world_size)

    try:
        mgr, num_local = build_manager(rank, world_size)
        _fill_tagged_w13(mgr, rank, interleaved_w13, world_size)
        ep_snap = snapshot_weights(mgr)

        from sglang.srt.paras.peer_access import init_peer_access
        from sglang.srt.paras.paras_parallel_state import (
            get_paras_tp_group,
            get_paras_tp_size,
        )

        peer_ctx = init_peer_access(
            mgr, get_paras_tp_group().device_group, get_paras_tp_size()
        )
        dst_base_ptrs = torch.tensor(
            peer_ctx.peer_addresses, dtype=torch.int64, device="cuda"
        )
        paras_tp_group = get_paras_tp_group().device_group
        barrier_tensor = torch.zeros(1, device="cuda")
        dist.barrier(group=paras_tp_group)

        # EP->TP reverse: TP weight i overlaps EP weight i+1 (four-anchor).
        for layer_id in reversed(range(NUM_LAYERS)):
            mixin = _make_mixin(
                layer_id, num_local, mgr, interleaved_w13=interleaved_w13
            )
            mixin.paras_reshard_ep_to_tp_peer(dst_base_ptrs, None)
            dist.all_reduce(
                barrier_tensor, op=dist.ReduceOp.SUM, group=paras_tp_group
            )

        _assert_tp_w13_layout(mgr, rank, world_size, interleaved_w13)

        dist.barrier(group=paras_tp_group)
        barrier_tensor.zero_()
        # TP->EP forward: EP weight i+1 overlaps TP weight i (four-anchor).
        for layer_id in range(NUM_LAYERS):
            mixin = _make_mixin(
                layer_id, num_local, mgr, interleaved_w13=interleaved_w13
            )
            mixin.paras_reshard_tp_to_ep_peer(dst_base_ptrs, None)
            dist.all_reduce(
                barrier_tensor, op=dist.ReduceOp.SUM, group=paras_tp_group
            )

        for layer_id in range(NUM_LAYERS):
            w13_now = mgr.get_view(
                f"model.layers.{layer_id}.mlp.experts.w13_weight"
            )
            w2_now = mgr.get_view(
                f"model.layers.{layer_id}.mlp.experts.w2_weight"
            )
            assert torch.equal(w13_now, ep_snap[layer_id][0]), (
                f"Layer {layer_id} w13 mismatch after peer_access "
                f"round-trip (interleaved_w13={interleaved_w13})"
            )
            assert torch.equal(w2_now, ep_snap[layer_id][1]), (
                f"Layer {layer_id} w2 mismatch after peer_access "
                f"round-trip (interleaved_w13={interleaved_w13})"
            )

        if rank == 0:
            print(
                "PASS: peer_access layout semantics + round-trip "
                f"(interleaved_w13={interleaved_w13})"
            )

    finally:
        teardown_distributed()


def test_gpt_oss_w13_peer_access_layout_concat():
    """peer_access: concat layout TP shape and EP round-trip restoration."""
    _run_peer_access_layout_semantics_test(interleaved_w13=False)


def test_gpt_oss_w13_peer_access_layout_interleaved():
    """peer_access: interleaved layout TP shape and EP round-trip restoration."""
    _run_peer_access_layout_semantics_test(interleaved_w13=True)


def _make_paras_server_args(**overrides):
    from sglang.srt.server_args import ServerArgs

    base = dict(
        model_path="/data/shaoyuw/models/gpt-oss-120b-BF16-unsloth",
        enable_paras_moe=True,
        paras_tp_size=4,
        enable_dp_attention=True,
        enable_dp_lm_head=True,
        tp_size=4,
        dp_size=4,
        cuda_graph_max_bs=8,
        mem_fraction_static=0.8,
    )
    base.update(overrides)
    return ServerArgs(**base)


def test_paras_tp_cuda_graph_bs_default_scaling():
    sa = _make_paras_server_args()
    assert sa.paras_tp_cuda_graph_max_bs == sa.cuda_graph_max_bs * sa.paras_tp_size
    assert sa.paras_tp_cuda_graph_bs is not None
    assert max(sa.paras_tp_cuda_graph_bs) == sa.paras_tp_cuda_graph_max_bs
    assert max(sa.paras_tp_cuda_graph_bs) > max(sa.cuda_graph_bs)


def test_paras_tp_cuda_graph_bs_explicit_override():
    sa = _make_paras_server_args(paras_tp_cuda_graph_max_bs=64)
    assert sa.paras_tp_cuda_graph_max_bs == 64
    assert max(sa.paras_tp_cuda_graph_bs) == 64
    assert sa.cuda_graph_max_bs == 8


def test_paras_tp_cuda_graph_bs_rejected_without_paras():
    from sglang.srt.server_args import ServerArgs

    with pytest.raises(
        AssertionError, match="--paras-tp-cuda-graph-max-bs requires --enable-paras-moe"
    ):
        ServerArgs(
            model_path="/data/shaoyuw/models/gpt-oss-120b-BF16-unsloth",
            enable_paras_moe=False,
            paras_tp_cuda_graph_max_bs=64,
        )


def test_paras_tp_cuda_graph_bs_unset_when_paras_off():
    from sglang.srt.server_args import ServerArgs

    sa = ServerArgs(
        model_path="/data/shaoyuw/models/gpt-oss-120b-BF16-unsloth",
        enable_paras_moe=False,
    )
    assert sa.paras_tp_cuda_graph_max_bs is None
    assert sa.paras_tp_cuda_graph_bs is None


def test_paras_auto_switch_defaults():
    sa = _make_paras_server_args()
    assert sa.paras_auto_switch is True
    assert sa.paras_auto_switch_low == 256
    assert sa.paras_auto_switch_high == 1024
    assert sa.paras_auto_switch_window == 32
    assert sa.paras_auto_switch_cooldown_sec == 60.0


def test_paras_auto_switch_validation_low_lt_high():
    with pytest.raises(AssertionError, match="paras-auto-switch-low.*paras-auto-switch-high"):
        _make_paras_server_args(paras_auto_switch_low=1024, paras_auto_switch_high=256)


def test_paras_auto_switch_validation_window_positive():
    with pytest.raises(AssertionError, match="paras-auto-switch-window"):
        _make_paras_server_args(paras_auto_switch_window=0)


def test_paras_auto_switch_validation_cooldown_nonneg():
    with pytest.raises(AssertionError, match="paras-auto-switch-cooldown-sec"):
        _make_paras_server_args(paras_auto_switch_cooldown_sec=-1.0)


def _make_policy(low=4, high=16, window=4, cooldown=0.0):
    from sglang.srt.paras.scheduler_paras_mixin import ParasAutoSwitchPolicy
    return ParasAutoSwitchPolicy(low=low, high=high, window=window, cooldown_sec=cooldown)


def test_policy_switches_ep_to_tp_when_avg_below_low():
    p = _make_policy()
    for v in [3, 3, 3, 3]:
        p.observe(v, now=0.0)
    assert p.pick_target("EP", now=0.0) == "TP"


def test_policy_switches_tp_to_ep_when_avg_above_high():
    p = _make_policy()
    for v in [20, 20, 20, 20]:
        p.observe(v, now=0.0)
    assert p.pick_target("TP", now=0.0) == "EP"


def test_policy_no_switch_in_dead_zone():
    p = _make_policy(low=4, high=16, window=4)
    for v in [10, 10, 10, 10]:
        p.observe(v, now=0.0)
    assert p.pick_target("EP", now=0.0) is None
    assert p.pick_target("TP", now=0.0) is None


def test_policy_no_switch_until_window_full():
    p = _make_policy(window=4)
    for v in [3, 3, 3]:
        p.observe(v, now=0.0)
    assert p.pick_target("EP", now=0.0) is None


def test_policy_cooldown_blocks_immediate_reswitch():
    p = _make_policy(cooldown=10.0)
    for v in [3, 3, 3, 3]:
        p.observe(v, now=0.0)
    assert p.pick_target("EP", now=0.0) == "TP"
    for v in [20, 20, 20, 20]:
        p.observe(v, now=1.0)
    assert p.pick_target("TP", now=1.0) is None
    assert p.pick_target("TP", now=9.9) is None
    assert p.pick_target("TP", now=10.1) == "EP"


def test_policy_zero_global_batch_ignored():
    p = _make_policy(window=4)
    for v in [0, 0, 0, 3, 3, 3, 3]:
        p.observe(v, now=0.0)
    assert p.pick_target("EP", now=0.0) == "TP"


def test_policy_window_clears_after_switch():
    p = _make_policy(cooldown=0.0)
    for v in [3, 3, 3, 3]:
        p.observe(v, now=0.0)
    p.pick_target("EP", now=0.0)
    for v in [3]:
        p.observe(v, now=0.0)
    assert p.pick_target("TP", now=0.0) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
