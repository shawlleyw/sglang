"""CPU references: entire-tensor collectives through overlapping UMM views."""

import os
import sys
import unittest
from dataclasses import replace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../python"))
)

from common.model_configs import ModelConfig, PRESETS
from common.weight_bundle import (
    COMPONENTS,
    ParaSMode,
    attention_restore,
    attention_slice,
    expert_ep_packed_view,
    make_manager,
    reference_values,
    weight_name,
)


class WeightBundleTest(unittest.TestCase):
    def model(self, interleaved):
        return ModelConfig(
            "tiny",
            2,
            8,
            16,
            16,
            32,
            4,
            num_attention_heads=8,
            interleaved_w13=interleaved,
        )

    def run_roundtrip(self, interleaved, world):
        model = self.model(interleaved)
        managers = [make_manager(model, world, "cpu")[0] for _ in range(world)]

        def view(rank, layer, mode, component):
            return managers[rank].get_view(weight_name(layer, mode, component))

        def reference(tensor, rank, mode, component, layer):
            return reference_values(
                torch.arange(tensor.numel()), model, world, rank, mode, component, layer
            ).view_as(tensor)

        for rank in range(world):
            for layer in range(model.num_hidden_layers):
                for component in COMPONENTS:
                    ep = view(rank, layer, ParaSMode.EP, component)
                    ep.copy_(reference(ep, rank, ParaSMode.EP, component, layer))
        for layer in range(model.num_hidden_layers):
            for component in COMPONENTS:
                if component in ("w13", "w2"):
                    packed = [
                        expert_ep_packed_view(
                            view(rank, layer, ParaSMode.EP, component),
                            model,
                            world,
                            component,
                        ).contiguous()
                        for rank in range(world)
                    ]
                    for rank in range(world):
                        view(rank, layer, ParaSMode.TP, component).view(-1).copy_(
                            torch.cat([p[rank].flatten() for p in packed])
                        )
                else:
                    for rank in range(world):
                        attention_slice(
                            view(rank, layer, ParaSMode.EP, component),
                            view(rank, layer, ParaSMode.TP, component),
                            model,
                            world,
                            rank,
                            component,
                        )
        for rank in range(world):
            for layer in range(model.num_hidden_layers):
                for component in COMPONENTS:
                    tp = view(rank, layer, ParaSMode.TP, component)
                    torch.testing.assert_close(
                        tp,
                        reference(tp, rank, ParaSMode.TP, component, layer),
                        rtol=0,
                        atol=0,
                    )
        # Reconstruct with peer shards; every full EP reference is overwritten
        # by the actual asymmetric planner, so pointer swapping cannot pass.
        for layer in reversed(range(model.num_hidden_layers)):
            for component in COMPONENTS:
                sources = [
                    view(rank, layer, ParaSMode.TP, component).clone()
                    for rank in range(world)
                ]
                if component in ("w13", "w2"):
                    for rank in range(world):
                        ep = view(rank, layer, ParaSMode.EP, component)
                        packed = expert_ep_packed_view(ep, model, world, component)
                        received = torch.cat(
                            [p.view(world, -1)[rank] for p in sources]
                        ).view_as(packed)
                        packed.copy_(received)
                else:
                    gathered = torch.stack(sources)
                    if component == "qkv" and world > model.num_kv_heads:
                        qs = model.num_attention_heads * model.head_dim // world
                        for rank in range(world):
                            if rank % (world // model.num_kv_heads):
                                gathered[rank, qs:].fill_(float("nan"))
                    for rank in range(world):
                        attention_restore(
                            gathered,
                            view(rank, layer, ParaSMode.EP, component),
                            model,
                            world,
                            component,
                        )
        for rank in range(world):
            for layer in range(model.num_hidden_layers):
                for component in COMPONENTS:
                    ep = view(rank, layer, ParaSMode.EP, component)
                    torch.testing.assert_close(
                        ep,
                        reference(ep, rank, ParaSMode.EP, component, layer),
                        rtol=0,
                        atol=0,
                    )

    def test_qwen_replicated_attention_roundtrip(self):
        self.run_roundtrip(False, 8)

    def test_gptoss_interleaved_roundtrip(self):
        self.run_roundtrip(True, 2)

    def test_real_presets_plan_without_allocating_model(self):
        for preset in ("qwen3-235b", "gpt-oss-120b"):
            model = PRESETS[preset]
            manager, plan = make_manager(model, 8, "cpu", materialize=False)
            for mode in (ParaSMode.EP, ParaSMode.TP):
                for layer in range(model.num_hidden_layers):
                    names = manager._unified_spec.for_mode(mode).weight_names[layer]
                    for name in names:
                        entry = manager._entries[name]
                        self.assertEqual(entry.offset_bytes % 256, 0)
                        self.assertLessEqual(
                            entry.offset_bytes + entry.size_bytes, plan.layout.budget
                        )
            # First layer's source and destination bundles must be disjoint.
            self.assertGreaterEqual(plan.layout.ep_front, plan.layout.tp_weight_bytes)


if __name__ == "__main__":
    unittest.main()
