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

    return tp_group


def build_manager(rank, world_size):
    from sglang.srt.paras.paras_memory_manager import (
        ParaSMemoryManager,
        create_paras_moe_aliases,
        set_global_paras_memory_manager,
    )

    ep_size = world_size
    num_local = NUM_EXPERTS // ep_size

    mgr = ParaSMemoryManager(device=f"cuda:{rank}")

    for slot in range(NUM_LAYERS + 1):
        mgr.reserve(
            f"paras.moe_slot.{slot}.w13",
            (num_local, 2 * INTERMEDIATE, HIDDEN),
            torch.bfloat16,
        )
        mgr.reserve(
            f"paras.moe_slot.{slot}.w2",
            (num_local, HIDDEN, INTERMEDIATE),
            torch.bfloat16,
        )

    for i in range(NUM_LAYERS):
        mgr._entries[f"model.layers.{i}.mlp.experts.w13_weight"] = mgr._entries[
            f"paras.moe_slot.{i + 1}.w13"
        ]
        mgr._entries[f"model.layers.{i}.mlp.experts.w2_weight"] = mgr._entries[
            f"paras.moe_slot.{i + 1}.w2"
        ]

    staging_experts = num_local
    w13_staging_shape = (staging_experts, 2 * INTERMEDIATE, HIDDEN)
    w2_staging_shape = (staging_experts, HIDDEN, INTERMEDIATE)
    for sfx in ("", "_1", "_2"):
        mgr.reserve(
            f"staging.w13_pre_permute{sfx}", w13_staging_shape, torch.bfloat16
        )
        mgr.reserve(
            f"staging.w2_pre_permute{sfx}", w2_staging_shape, torch.bfloat16
        )

    mgr.materialize()
    create_paras_moe_aliases(mgr, NUM_LAYERS)
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

        for layer_id in range(NUM_LAYERS):
            mixin = _make_mixin(layer_id, num_local, mgr)
            mixin.paras_configure_tp_fused_peer_access_kernel(
                peer_ctx, dst_base_ptrs, None
            )
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
        for layer_id in reversed(range(NUM_LAYERS)):
            mixin = _make_mixin(layer_id, num_local, mgr)
            mixin.paras_configure_ep_fused_peer_access_kernel(
                peer_ctx, dst_base_ptrs, None
            )
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
    """Build manager with w13/w2 weight slots, w13_bias slots, and all staging buffers."""
    from sglang.srt.paras.paras_memory_manager import (
        ParaSMemoryManager,
        create_paras_moe_aliases,
        set_global_paras_memory_manager,
    )

    ep_size = world_size
    num_local = NUM_EXPERTS // ep_size

    mgr = ParaSMemoryManager(device=f"cuda:{rank}")

    for slot in range(NUM_LAYERS + 1):
        mgr.reserve(
            f"paras.moe_slot.{slot}.w13",
            (num_local, 2 * INTERMEDIATE, HIDDEN),
            torch.bfloat16,
        )
        mgr.reserve(
            f"paras.moe_slot.{slot}.w2",
            (num_local, HIDDEN, INTERMEDIATE),
            torch.bfloat16,
        )

    for slot in range(NUM_LAYERS + 1):
        mgr.reserve(
            f"paras.moe_slot.{slot}.w13_bias",
            (num_local, 2 * INTERMEDIATE),
            torch.float32,
        )

    for i in range(NUM_LAYERS):
        mgr._entries[f"model.layers.{i}.mlp.experts.w13_weight"] = mgr._entries[
            f"paras.moe_slot.{i + 1}.w13"
        ]
        mgr._entries[f"model.layers.{i}.mlp.experts.w2_weight"] = mgr._entries[
            f"paras.moe_slot.{i + 1}.w2"
        ]
        mgr._entries[f"model.layers.{i}.mlp.experts.w13_weight_bias"] = mgr._entries[
            f"paras.moe_slot.{i + 1}.w13_bias"
        ]

    staging_experts = num_local
    for sfx in ("", "_1", "_2"):
        mgr.reserve(
            f"staging.w13_pre_permute{sfx}",
            (staging_experts, 2 * INTERMEDIATE, HIDDEN),
            torch.bfloat16,
        )
        mgr.reserve(
            f"staging.w2_pre_permute{sfx}",
            (staging_experts, HIDDEN, INTERMEDIATE),
            torch.bfloat16,
        )
        mgr.reserve(
            f"staging.w13_bias_pre_permute{sfx}",
            (staging_experts, 2 * INTERMEDIATE),
            torch.float32,
        )

    mgr.materialize()
    create_paras_moe_aliases(mgr, NUM_LAYERS)
    set_global_paras_memory_manager(mgr)
    return mgr, num_local


def fill_biases(mgr, rank):
    for layer_id in range(NUM_LAYERS):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(SEED + layer_id * 100 + rank + 77)
        b13 = mgr.get_view(f"model.layers.{layer_id}.mlp.experts.w13_weight_bias")
        b13.copy_(
            torch.randn(b13.shape, generator=gen, dtype=torch.float32).to(
                device=b13.device
            )
        )


def snapshot_biases(mgr):
    snap = {}
    for layer_id in range(NUM_LAYERS):
        snap[layer_id] = mgr.get_view(
            f"model.layers.{layer_id}.mlp.experts.w13_weight_bias"
        ).clone()
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

        for layer_id in range(NUM_LAYERS):
            mixin = _make_mixin_with_bias(
                layer_id, num_local, mgr, interleaved_w13=interleaved_w13
            )
            mixin.paras_configure_tp_all_to_all()

        for layer_id in range(NUM_LAYERS):
            tp_b13 = mgr.get_view(
                f"model.layers.{layer_id}.mlp.tp_experts.w13_weight_bias"
            )
            assert tp_b13.abs().sum() > 0, (
                f"Layer {layer_id} TP bias is all zeros after EP->TP all-to-all"
            )

        for layer_id in reversed(range(NUM_LAYERS)):
            mixin = _make_mixin_with_bias(
                layer_id, num_local, mgr, interleaved_w13=interleaved_w13
            )
            mixin.paras_configure_ep_mlp_naive()

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
            assert torch.equal(b13_now, ep_bias_snap[layer_id]), (
                f"Layer {layer_id} w13_bias mismatch after round-trip"
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

        for layer_id in range(NUM_LAYERS):
            mixin = _make_mixin_with_bias(
                layer_id, num_local, mgr, interleaved_w13=interleaved_w13
            )
            mixin.paras_configure_tp_all_to_all()

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
