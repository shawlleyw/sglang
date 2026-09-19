"""Uniform and hybrid KV views use the completed unified memory plan."""

import torch

from sglang.srt.paras.layers.utils import LayerCacheSpec
from sglang.srt.paras.mode import ParaSMode
from sglang.srt.paras.paras_memory_manager import ParaSMemoryManager
from test.srt.paras.unified_memory_test_utils import materialize_test_cache


def make_manager(kinds):
    specs = [
        LayerCacheSpec(
            i,
            kind,
            256 if kind == "swa" else 1024,
            1024 if kind == "swa" else 4096,
            8,
            128,
            2048 if kind == "swa" else None,
        )
        for i, kind in enumerate(kinds)
    ]
    mgr = ParaSMemoryManager(device="cpu")
    materialize_test_cache(
        mgr,
        num_layers=len(specs),
        ep_max_tokens=1024,
        tp_max_tokens=4096,
        num_kv_heads=8,
        head_dim=128,
        kv_dtype=torch.bfloat16,
        tp_size=4,
        layer_specs=specs,
    )
    return mgr, specs


def test_uniform_shapes():
    mgr, _ = make_manager(["full"] * 6)
    for i in range(6):
        for side in ("k", "v"):
            assert mgr.get_view(f"model.layers.{i}.kv.ep.{side}").shape == (
                1025,
                8,
                128,
            )
            assert mgr.get_view(f"model.layers.{i}.kv.tp.{side}").shape == (
                4097,
                2,
                128,
            )


def test_heterogeneous_shapes():
    mgr, specs = make_manager(["full"] * 2 + ["swa"] * 4)
    for spec in specs:
        for side in ("k", "v"):
            assert mgr.get_view(f"model.layers.{spec.layer_id}.kv.ep.{side}").shape == (
                spec.tokens_cap_ep + 1,
                8,
                128,
            )
            assert mgr.get_view(f"model.layers.{spec.layer_id}.kv.tp.{side}").shape == (
                spec.tokens_cap_tp + 1,
                2,
                128,
            )


def test_alias_views():
    mgr, specs = make_manager(["full"] * 2 + ["swa"] * 2)
    for spec in specs:
        for side in ("k", "v"):
            alias = mgr.get_view(f"model.layers.{spec.layer_id}.kv.{side}")
            ep = mgr.get_view(f"model.layers.{spec.layer_id}.kv.ep.{side}")
            assert alias.data_ptr() == ep.data_ptr()
            assert alias.shape == ep.shape


def test_alias_names():
    mgr, specs = make_manager(["full", "swa"])
    for spec in specs:
        for side in ("k", "v"):
            for mode in ("", "ep.", "tp."):
                assert f"model.layers.{spec.layer_id}.kv.{mode}{side}" in mgr._entries


def test_no_overlap_heterogeneous():
    mgr, specs = make_manager(["swa", "full", "full"])
    for mode in (ParaSMode.EP, ParaSMode.TP):
        regions = []
        for spec in specs:
            for side in ("k", "v"):
                entry = mgr._entries[
                    f"model.layers.{spec.layer_id}.kv.{mode.value}.{side}"
                ]
                regions.append(
                    (entry.offset_bytes, entry.offset_bytes + entry.size_bytes)
                )
        regions.sort()
        assert all(a[1] <= b[0] for a, b in zip(regions, regions[1:]))
    for spec in specs:
        for side in ("k", "v"):
            ep = mgr._entries[f"model.layers.{spec.layer_id}.kv.ep.{side}"]
            tp = mgr._entries[f"model.layers.{spec.layer_id}.kv.tp.{side}"]
            assert tp.offset_bytes + tp.size_bytes <= ep.offset_bytes
