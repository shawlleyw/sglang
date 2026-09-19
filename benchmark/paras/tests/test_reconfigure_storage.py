"""Independent-storage aliases and lifecycle, without GPU allocations."""

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
from common.weight_bundle import ParaSMode, make_manager, weight_name
from reconfigure.storage import IndependentWeightMemoryManager
from sglang.srt.paras.workspace import ModeWorkspaces, WorkspaceRequirement


def independent_manager(model, *, world=4, device="cpu", force_span_collision=False):
    planned, plan = make_manager(model, world, device, materialize=False)
    entries = list(plan.entries)
    if force_span_collision:
        ep_name = weight_name(0, ParaSMode.EP, "w13")
        tp_name = weight_name(1, ParaSMode.TP, "w13")
        ep_entry = next(entry for entry in entries if entry.name == ep_name)
        entries = [
            (
                replace(entry, offset_bytes=ep_entry.offset_bytes)
                if entry.name == tp_name
                else entry
            )
            for entry in entries
        ]
    # reserve_model_weights creates these checkpoint aliases in production.
    entries.extend(
        replace(entry, name=entry.name.replace(".mlp.ep_experts.", ".mlp.experts."))
        for entry in tuple(entries)
        if ".mlp.ep_experts." in entry.name
    )
    for mode in (ParaSMode.EP, ParaSMode.TP):
        planned._unified_spec.for_mode(mode).workspaces = ModeWorkspaces(
            WorkspaceRequirement("microbench", 512),
            WorkspaceRequirement("microbench", 1024),
        )
    manager = IndependentWeightMemoryManager(device=device)
    manager._unified_spec = planned._unified_spec
    manager.materialize(replace(plan, entries=tuple(entries)))
    return manager


class IndependentStorageTest(unittest.TestCase):
    def model(self):
        return ModelConfig("tiny", 2, 8, 8, 16, 32, 3, num_attention_heads=8)

    def test_equal_umm_spans_do_not_alias_distinct_modes(self):
        manager = independent_manager(self.model(), force_span_collision=True)
        ep_name = weight_name(0, ParaSMode.EP, "w13")
        tp_name = weight_name(1, ParaSMode.TP, "w13")
        checkpoint_name = ep_name.replace(".mlp.ep_experts.", ".mlp.experts.")
        self.assertEqual(
            manager._entries[ep_name].offset_bytes,
            manager._entries[tp_name].offset_bytes,
        )
        self.assertEqual(
            manager._entries[ep_name].size_bytes, manager._entries[tp_name].size_bytes
        )
        ep, tp = manager.get_view(ep_name), manager.get_view(tp_name)
        self.assertIs(ep, manager.get_view(checkpoint_name))
        self.assertNotEqual(ep.data_ptr(), tp.data_ptr())
        self.assertEqual(tuple(ep.shape), manager._entries[ep_name].shape)
        self.assertEqual(tuple(tp.shape), manager._entries[tp_name].shape)
        self.assertTrue(ep.is_contiguous())
        self.assertEqual(tp.untyped_storage().nbytes(), 2)
        updated = torch.full_like(ep, 17)
        manager.replace_weight(checkpoint_name, updated)
        self.assertIs(updated, manager.get_view(ep_name))
        self.assertIs(tp, manager.get_view(tp_name))

    def test_replacement_and_source_release_update_owned_bytes(self):
        manager = independent_manager(self.model())
        ep_name = weight_name(0, ParaSMode.EP, "w2")
        tp_name = weight_name(0, ParaSMode.TP, "w2")
        before = manager.allocated_bytes()
        source_bytes = manager.get_view(ep_name).untyped_storage().nbytes()
        shape = manager._entries[tp_name].shape
        new_target = torch.empty(shape, dtype=torch.bfloat16)
        manager.replace_weight(tp_name, new_target)
        self.assertIs(manager.get_view(tp_name), new_target)
        self.assertEqual(manager.allocated_bytes(), before + source_bytes - 2)
        manager.replace_weight(ep_name, manager.placeholder(ep_name))
        self.assertEqual(manager.allocated_bytes(), before)
        self.assertEqual(manager.total_bytes, manager.allocated_bytes())
        self.assertEqual(manager.get_view(ep_name).untyped_storage().nbytes(), 2)
        self.assertTrue(manager.is_managed(new_target))
        self.assertFalse(manager.is_managed(torch.empty(8)))
        # Any weight reshape must share the new allocation, including aliases.
        self.assertEqual(
            manager.get_view_as(tp_name, (new_target.numel(),)).data_ptr(),
            new_target.data_ptr(),
        )

    def test_empty_cache_aliases_and_separate_workspaces(self):
        manager = independent_manager(self.model())
        prefix = "model.layers.0.kv"
        ep = manager.get_view(f"{prefix}.ep.k")
        tp = manager.get_view(f"{prefix}.tp.k")
        alias = manager.get_view(f"{prefix}.k")
        self.assertEqual(ep.data_ptr(), tp.data_ptr())
        self.assertEqual(ep.data_ptr(), alias.data_ptr())
        self.assertNotEqual(
            ep.data_ptr(), manager.get_view(f"{prefix}.ep.v").data_ptr()
        )
        self.assertNotEqual(
            ep.data_ptr(), manager.get_view("model.layers.1.kv.ep.k").data_ptr()
        )
        self.assertTrue(manager.is_managed(ep))
        workspace_pointers = []
        for mode in (ParaSMode.EP, ParaSMode.TP):
            for kind in ("moe", "attention"):
                (scratch,) = manager._get_workspace(
                    mode, kind, [(16,)], torch.float32, "cpu"
                )
                workspace_pointers.append(scratch.data_ptr())
                self.assertTrue(manager.is_managed(scratch))
        self.assertEqual(len(set(workspace_pointers)), 4)
        self.assertNotIn(ep.data_ptr(), workspace_pointers)
        self.assertEqual(manager.buffer.numel(), 256)

    def test_full_presets_on_meta_device_preserve_alias_names(self):
        # Meta tensors exercise shapes and allocation accounting for full model
        # plans without allocating the tens of GB required by actual weights.
        for name in ("qwen3-235b", "gpt-oss-120b"):
            model = PRESETS[name]
            manager = independent_manager(model, world=8, device="meta")
            self.assertEqual(manager.num_entries, len(manager._entries))
            for mode in (ParaSMode.EP, ParaSMode.TP):
                for layer in range(model.num_hidden_layers):
                    for component in ("w13", "w2", "qkv", "o"):
                        canonical = weight_name(layer, mode, component)
                        self.assertEqual(manager._aliases[canonical], canonical)
                        self.assertEqual(
                            tuple(manager.get_view(canonical).shape),
                            manager._entries[canonical].shape,
                        )
                        if mode == ParaSMode.TP:
                            self.assertEqual(
                                manager.get_view(canonical).untyped_storage().nbytes(),
                                2,
                            )
            self.assertEqual(manager.total_bytes, manager.allocated_bytes())

    def test_rejects_nonweight_replacement_and_bad_shapes(self):
        manager = independent_manager(self.model())
        with self.assertRaisesRegex(KeyError, "Not an independent weight"):
            manager.replace_weight("model.layers.0.kv.ep.k", torch.empty(1))
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            manager.replace_weight(weight_name(0, ParaSMode.EP, "w13"), torch.empty(1))


if __name__ == "__main__":
    unittest.main()
