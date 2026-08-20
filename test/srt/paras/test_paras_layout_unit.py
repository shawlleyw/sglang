"""CPU unit tests for the ParaS four-anchor unified layout.

Validates, against the real offsets produced by ``plan_qwen_moe_layout`` +
``reserve_kv_cache`` + ``materialize``, that:
  * every layer's EP and TP regions stay disjoint under the production switch
    orders (EP->TP cache-reverse then weights-reverse; TP->EP weights-forward
    then cache-forward), so a cross-GPU transfer never clobbers unread source
    bytes on the same GPU;
  * the buffer is smaller than the naive weights+cache sum (overlap present);
  * EP and TP cache capacities jointly respect the exact UMM byte limit;
  * no ``paras.moe_slot.*`` entries remain (slot design dropped);
  * standalone KV reservation still produces a valid cache-only layout.

No GPU required.
"""

from types import SimpleNamespace

import torch

from sglang.srt.paras.paras_memory_manager import (
    ParaSMemoryManager,
    create_paras_moe_aliases,
    plan_qwen_moe_layout,
)

ALIGN = 256


def _overlap(a, b):
    return a[0] < b[0] + b[1] and b[0] < a[0] + a[1]


def _build(num_layers=6, num_experts=64, hidden=2048, inter=1536,
           num_heads=32, num_kv_heads=4, head_dim=128, ep=2, tp=2,
           ep_tokens=8000, tp_tokens=16000, with_kv=True):
    mgr = ParaSMemoryManager(device="cpu")
    dp = ep // tp
    plan_qwen_moe_layout(
        mgr, num_layers=num_layers, num_experts=num_experts, hidden_size=hidden,
        intermediate_size=inter, num_heads=num_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim, ep_size=ep, tp_size=tp, dp_size=dp, moe_tp_size=tp,
        prefix="model",
    )
    if with_kv:
        mgr.reserve_kv_cache(
            num_layers=num_layers, ep_max_tokens=ep_tokens, tp_max_tokens=tp_tokens,
            num_kv_heads=num_kv_heads, head_dim=head_dim, kv_dtype=torch.bfloat16,
            tp_size=tp, page_size=1, prefix="model",
        )
    total = mgr.materialize()
    create_paras_moe_aliases(mgr, num_layers, prefix="model")
    return mgr, total, num_layers


def _rng(mgr, name):
    e = mgr._entries[name]
    return (e.offset_bytes, e.size_bytes)


def _blocks(mgr, mode, i, phase):
    """Sub-slabs for one (mode, layer, phase) atomic transfer unit."""
    if phase == "w":
        return [
            _rng(mgr, f"model.layers.{i}.mlp.{mode}_experts.w13_weight"),
            _rng(mgr, f"model.layers.{i}.mlp.{mode}_experts.w2_weight"),
        ]
    return [
        _rng(mgr, f"model.layers.{i}.kv.{mode}.k"),
        _rng(mgr, f"model.layers.{i}.kv.{mode}.v"),
    ]


def _assert_no_clobber(mgr, num_layers, order, src_mode, dst_mode):
    """Per-layer-atomic clobber check: writing dst_mode[i] must not overlap any
    unread src_mode block (barrier only between layers)."""
    unread = {(ph, i): _blocks(mgr, src_mode, i, ph) for (ph, i) in order}
    for (ph, i) in order:
        for dst in _blocks(mgr, dst_mode, i, ph):
            for key, srcs in unread.items():
                for r in srcs:
                    assert not _overlap(dst, r), (
                        f"{src_mode}->{dst_mode} clobber at {ph}{i} "
                        f"dst={dst} vs unread {key}={r}"
                    )
        del unread[(ph, i)]


def test_ep_to_tp_switch_is_clobber_safe():
    mgr, _, N = _build()
    # EP->TP reads EP, writes TP: cache reverse, then weights reverse.
    order = [("c", i) for i in range(N - 1, -1, -1)] + [
        ("w", i) for i in range(N - 1, -1, -1)
    ]
    _assert_no_clobber(mgr, N, order, src_mode="ep", dst_mode="tp")


def test_tp_to_ep_switch_is_clobber_safe():
    mgr, _, N = _build()
    # TP->EP reads TP, writes EP: weights forward, then cache forward.
    order = [("w", i) for i in range(N)] + [("c", i) for i in range(N)]
    _assert_no_clobber(mgr, N, order, src_mode="tp", dst_mode="ep")


def test_all_entries_aligned():
    mgr, _, N = _build()
    for i in range(N):
        for mode in ("ep", "tp"):
            for ph in ("w", "c"):
                for off, _ in _blocks(mgr, mode, i, ph):
                    assert off % ALIGN == 0


def test_ep_low_tp_high_orientation():
    mgr, _, N = _build()
    ep_w13 = _rng(mgr, "model.layers.0.mlp.ep_experts.w13_weight")
    tp_w13 = _rng(mgr, "model.layers.0.mlp.tp_experts.w13_weight")
    assert tp_w13[0] > ep_w13[0], "orientation must be EP-low / TP-high"


def test_g1_ep_tp_weight_bytes_equal():
    mgr, _, _ = _build(ep=2, tp=2)  # G = ep/tp = 1
    ep_w13 = _rng(mgr, "model.layers.0.mlp.ep_experts.w13_weight")
    tp_w13 = _rng(mgr, "model.layers.0.mlp.tp_experts.w13_weight")
    assert ep_w13[1] == tp_w13[1], "EP and TP weight bytes must match at G=1"


def test_ep_tp_expert_shapes_match_forward():
    # EP: num_experts/ep_size experts, FULL intermediate.
    # TP: all num_experts, intermediate sharded by tp_size.
    NE, H, I, EP, TP = 64, 2048, 1536, 2, 2
    mgr, _, _ = _build(num_experts=NE, hidden=H, inter=I, ep=EP, tp=TP)
    ep_w13 = mgr.get_view("model.layers.0.mlp.ep_experts.w13_weight")
    tp_w13 = mgr.get_view("model.layers.0.mlp.tp_experts.w13_weight")
    assert tuple(ep_w13.shape) == (NE // EP, 2 * I, H), tuple(ep_w13.shape)
    assert tuple(tp_w13.shape) == (NE, 2 * (I // TP), H), tuple(tp_w13.shape)


def test_g_greater_than_1_layout_shapes_and_safety():
    # G = ep_size / tp_size = 2 (dp_size=2): TP weights are G x EP weights, and
    # the four-anchor layout stays clobber-safe in both switch directions.
    NE, H, I, EP, TP = 64, 2048, 1536, 4, 2
    mgr, _, N = _build(num_experts=NE, hidden=H, inter=I, ep=EP, tp=TP)
    ep_e = mgr._entries["model.layers.0.mlp.ep_experts.w13_weight"]
    tp_e = mgr._entries["model.layers.0.mlp.tp_experts.w13_weight"]
    assert tuple(ep_e.shape) == (NE // EP, 2 * I, H), tuple(ep_e.shape)
    assert tuple(tp_e.shape) == (NE, 2 * (I // TP), H), tuple(tp_e.shape)
    assert tp_e.size_bytes == (EP // TP) * ep_e.size_bytes  # TP = G x EP

    order_ep = [("c", i) for i in range(N - 1, -1, -1)] + [("w", i) for i in range(N - 1, -1, -1)]
    _assert_no_clobber(mgr, N, order_ep, "ep", "tp")
    order_tp = [("w", i) for i in range(N)] + [("c", i) for i in range(N)]
    _assert_no_clobber(mgr, N, order_tp, "tp", "ep")


def test_experts_alias_points_to_ep():
    mgr, _, N = _build()
    for i in range(N):
        assert (
            mgr._entries[f"model.layers.{i}.mlp.experts.w13_weight"].offset_bytes
            == mgr._entries[f"model.layers.{i}.mlp.ep_experts.w13_weight"].offset_bytes
        )
        assert (
            mgr._entries[f"model.layers.{i}.kv.k"].offset_bytes
            == mgr._entries[f"model.layers.{i}.kv.ep.k"].offset_bytes
        )


def test_no_slot_entries_remain():
    mgr, _, _ = _build()
    assert not any("moe_slot" in n for n in mgr._entries)


def test_buffer_smaller_than_naive_sum():
    mgr, total, N = _build()
    naive = 0
    for i in range(N):
        for mode in ("ep", "tp"):
            for ph in ("w", "c"):
                naive += sum(s for _, s in _blocks(mgr, mode, i, ph))
    assert total < naive, "combined run must overlap weights and cache"


def test_deferred_expert_weights_are_materialized():
    mgr, total, _ = _build()
    assert mgr._deferred_weight_bytes < total
    assert mgr._deferred_weight_bytes > 0


def test_standalone_kv_degeneration():
    mgr = ParaSMemoryManager(device="cpu")
    mgr.reserve_kv_cache(
        num_layers=4, ep_max_tokens=1024, tp_max_tokens=4096, num_kv_heads=8,
        head_dim=128, kv_dtype=torch.bfloat16, tp_size=4, page_size=1, prefix="model",
    )
    mgr.materialize()
    for i in range(4):
        ep = _rng(mgr, f"model.layers.{i}.kv.ep.k")
        tp = _rng(mgr, f"model.layers.{i}.kv.tp.k")
        assert ep[0] % ALIGN == 0 and tp[0] % ALIGN == 0
        # per-layer [k|v]: v starts at k end (aligned).
        ep_v = _rng(mgr, f"model.layers.{i}.kv.ep.v")
        assert ep_v[0] >= ep[0] + ep[1]


def test_anchor_reduces_to_max_ct_for_real_configs():
    # Real configs (num_kv_heads divisible by tp_size, page >= 1) always yield
    # ct <= ce, so the general tail anchor collapses to max(ct).
    mgr = ParaSMemoryManager(device="cpu")
    N = 4
    mgr.reserve_kv_cache(
        num_layers=N, ep_max_tokens=1024, tp_max_tokens=4096, num_kv_heads=8,
        head_dim=128, kv_dtype=torch.bfloat16, tp_size=4, page_size=1, prefix="model",
    )
    mgr.materialize()

    def au(x):
        return (x + ALIGN - 1) // ALIGN * ALIGN

    def slab(mode, i):
        k = mgr._entries[f"model.layers.{i}.kv.{mode}.k"].size_bytes
        v = mgr._entries[f"model.layers.{i}.kv.{mode}.v"].size_bytes
        return au(k) + au(v)

    ct = [slab("tp", i) for i in range(N)]
    ce = [slab("ep", i) for i in range(N)]
    assert all(ct[i] <= ce[i] for i in range(N)), (ct, ce)

    # General anchor collapses to max(ct) when ct <= ce.
    anchor, suffix = 0, 0
    for i in range(N - 1, -1, -1):
        anchor = max(anchor, ct[i] + suffix)
        suffix += ct[i] - ce[i]
    assert anchor == max(ct), (anchor, max(ct))

    _assert_no_clobber(mgr, N, [("c", i) for i in range(N - 1, -1, -1)], "ep", "tp")
    _assert_no_clobber(mgr, N, [("c", i) for i in range(N)], "tp", "ep")


def _build_budgeted_capacity_plan(*, ep_size, tp_size, hybrid=False, num_kv_heads=4):
    budget_bytes = 4 << 20
    server_args = SimpleNamespace(
        kv_cache_dtype="auto",
        page_size=1,
        mem_fraction_static=0.6,
        swa_full_tokens_ratio=0.25,
    )
    config = SimpleNamespace(
        num_hidden_layers=4,
        num_key_value_heads=num_kv_heads,
        num_attention_heads=8,
        hidden_size=64,
        vocab_size=0,
        tie_word_embeddings=True,
    )
    if hybrid:
        config.layer_types = [
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ]
        config.sliding_window = 128

    manager = ParaSMemoryManager(device="cpu", server_args=server_args)
    plan_qwen_moe_layout(
        manager,
        num_layers=config.num_hidden_layers,
        num_experts=8,
        hidden_size=config.hidden_size,
        intermediate_size=32,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        head_dim=8,
        ep_size=ep_size,
        tp_size=tp_size,
        dp_size=ep_size // tp_size,
        moe_tp_size=tp_size,
    )
    manager._compute_umm_budget_bytes = lambda config: (
        budget_bytes,
        budget_bytes,
        0,
        budget_bytes,
        budget_bytes,
        0,
        budget_bytes / (1 << 30),
    )

    if hybrid:
        plan = manager.plan_hybrid_swa_kv_capacity(
            config=config,
            tp_size=tp_size,
            head_dim=8,
        )
    else:
        plan = manager.plan_mha_kv_capacity(
            config=config,
            tp_size=tp_size,
            head_dim=8,
        )

    manager.reserve_kv_cache(
        num_layers=config.num_hidden_layers,
        ep_max_tokens=plan.ep_max_tokens,
        tp_max_tokens=plan.tp_max_tokens,
        num_kv_heads=config.num_key_value_heads,
        head_dim=8,
        kv_dtype=plan.kv_dtype,
        tp_size=tp_size,
        page_size=server_args.page_size,
        layer_specs=plan.layer_specs,
    )
    total_bytes = manager.materialize()
    return manager, plan, total_bytes


def test_mha_capacity_charges_larger_tp_weights_against_tp_cache():
    manager, plan, total_bytes = _build_budgeted_capacity_plan(
        ep_size=4,
        tp_size=2,
    )

    weight_growth = plan.tp_expert_weight_bytes - plan.ep_expert_weight_bytes
    assert plan.tp_expert_weight_bytes == 2 * plan.ep_expert_weight_bytes
    assert plan.ep_kv_bytes - plan.tp_kv_bytes >= weight_growth
    assert plan.tp_max_tokens < 2 * plan.ep_max_tokens
    assert total_bytes == plan.planned_umm_bytes
    assert total_bytes <= plan.manager_budget_bytes
    assert manager._planned_umm_bytes_limit == plan.manager_budget_bytes


def test_mha_capacity_charges_tp_weights_with_replicated_kv_heads():
    _, plan, total_bytes = _build_budgeted_capacity_plan(
        ep_size=4,
        tp_size=2,
        num_kv_heads=1,
    )

    weight_growth = plan.tp_expert_weight_bytes - plan.ep_expert_weight_bytes
    assert plan.ep_kv_heads == plan.tp_kv_heads == 1
    assert plan.ep_kv_bytes - plan.tp_kv_bytes >= weight_growth
    assert plan.tp_max_tokens < plan.ep_max_tokens
    assert total_bytes == plan.planned_umm_bytes
    assert total_bytes <= plan.manager_budget_bytes


def test_mha_capacity_preserves_equal_cache_bytes_when_weights_match():
    _, plan, total_bytes = _build_budgeted_capacity_plan(
        ep_size=2,
        tp_size=2,
    )

    assert plan.tp_expert_weight_bytes == plan.ep_expert_weight_bytes
    assert plan.tp_kv_bytes == plan.ep_kv_bytes
    assert total_bytes == plan.planned_umm_bytes
    assert total_bytes <= plan.manager_budget_bytes


def test_hybrid_capacity_charges_larger_tp_weights_against_tp_cache():
    _, plan, total_bytes = _build_budgeted_capacity_plan(
        ep_size=4,
        tp_size=2,
        hybrid=True,
    )

    weight_growth = plan.tp_expert_weight_bytes - plan.ep_expert_weight_bytes
    assert plan.ep_kv_bytes - plan.tp_kv_bytes >= weight_growth
    assert plan.tp_max_tokens < 2 * plan.ep_max_tokens
    assert plan.tp_max_tokens_swa < 2 * plan.ep_max_tokens_swa
    assert total_bytes == plan.planned_umm_bytes
    assert total_bytes <= plan.manager_budget_bytes
