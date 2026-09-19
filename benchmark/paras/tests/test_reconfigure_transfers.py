"""Fresh-storage baseline transfers, using independent CPU collective oracles."""

import os
import sys
import unittest
from dataclasses import replace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../python"))
)

from common.model_configs import ModelConfig
from reconfigure.transfers import (
    naive_nccl_transfer_layer,
    reload_host_layer,
    reload_host_tensors,
    snapshot_host_tensors,
    target_shapes,
)


def global_weights(model):
    shapes = {
        "w13": (model.num_experts, 2 * model.moe_intermediate_size, model.hidden_size),
        "w2": (model.num_experts, model.hidden_size, model.moe_intermediate_size),
        "qkv": (
            (model.num_attention_heads + 2 * model.num_kv_heads) * model.head_dim,
            model.hidden_size,
        ),
        "o": (model.hidden_size, model.num_attention_heads * model.head_dim),
    }
    result = {}
    for offset, (name, shape) in enumerate(shapes.items()):
        count = 1
        for extent in shape:
            count *= extent
        result[name] = (
            ((torch.arange(count) + offset * 29) % 251)
            .to(torch.bfloat16)
            .reshape(shape)
        )
    return result


def shard_expert(tensor, model, world, rank, component):
    width = model.moe_intermediate_size // world
    start, end = rank * width, (rank + 1) * width
    if component == "w2":
        return tensor[..., start:end].contiguous()
    if model.interleaved_w13:
        return tensor[:, 2 * start : 2 * end, :].contiguous()
    return torch.cat(
        (
            tensor[:, start:end],
            tensor[
                :,
                model.moe_intermediate_size + start : model.moe_intermediate_size + end,
            ],
        ),
        dim=1,
    )


def expected_modes(model, world):
    full = global_weights(model)
    ep, tp = [], []
    width = model.num_experts // world
    q = model.num_attention_heads * model.head_dim
    kv = model.num_kv_heads * model.head_dim
    qwidth, kvwidth = q // world, max(model.head_dim, kv // world)
    replica = max(1, world // model.num_kv_heads)
    for rank in range(world):
        ep.append(
            {
                name: (
                    tensor[rank * width : (rank + 1) * width].clone()
                    if name in ("w13", "w2")
                    else tensor.clone()
                )
                for name, tensor in full.items()
            }
        )
        shards = {
            name: shard_expert(full[name], model, world, rank, name)
            for name in ("w13", "w2")
        }
        kvstart = rank // replica * kvwidth
        shards["qkv"] = torch.cat(
            (
                full["qkv"][rank * qwidth : (rank + 1) * qwidth],
                full["qkv"][q + kvstart : q + kvstart + kvwidth],
                full["qkv"][q + kv + kvstart : q + kv + kvstart + kvwidth],
            )
        )
        shards["o"] = full["o"][:, rank * qwidth : (rank + 1) * qwidth].contiguous()
        tp.append(shards)
    return ep, tp


class SimulatedCollectives:
    def __init__(self, test, sources, expected, model, world, rank, direction):
        self.test, self.sources, self.expected = test, sources, expected
        self.model, self.world, self.rank, self.direction = (
            model,
            world,
            rank,
            direction,
        )
        self.expert_calls = self.attention_calls = 0

    def all_to_all_single(self, output, source, *, group):
        self.test.assertEqual(group, "test-group")
        component = ("w13", "w2")[self.expert_calls]
        self.expert_calls += 1
        if self.direction == "ep_to_tp":
            expected_input = torch.cat(
                [
                    shard_expert(
                        self.sources[self.rank][component],
                        self.model,
                        self.world,
                        destination,
                        component,
                    ).flatten()
                    for destination in range(self.world)
                ]
            )
            incoming = self.expected[self.rank][component].flatten()
        else:
            expected_input = self.sources[self.rank][component].flatten()
            local_experts = self.model.num_experts // self.world
            incoming = torch.cat(
                [
                    weights[component][
                        self.rank * local_experts : (self.rank + 1) * local_experts
                    ].flatten()
                    for weights in self.sources
                ]
            )
        torch.testing.assert_close(source, expected_input, rtol=0, atol=0)
        output.copy_(incoming)

    def all_gather_into_tensor(self, output, source, *, group):
        self.test.assertEqual(group, "test-group")
        component = ("qkv", "o")[self.attention_calls]
        self.attention_calls += 1
        torch.testing.assert_close(
            source,
            self.sources[self.rank][component].flatten(),
            rtol=0,
            atol=0,
            equal_nan=True,
        )
        output.copy_(
            torch.cat([weights[component].flatten() for weights in self.sources])
        )


class ReconfigurationTransfersTest(unittest.TestCase):
    def model(self, interleaved=False, kv_heads=1):
        return ModelConfig(
            "tiny",
            kv_heads,
            2,
            8,
            6,
            8,
            2,
            num_attention_heads=4,
            interleaved_w13=interleaved,
        )

    def test_naive_nccl_both_directions_and_gate_layouts(self):
        for interleaved in (False, True):
            for kv_heads in (1, 4):
                model, world = self.model(interleaved, kv_heads), 4
                ep, tp = expected_modes(model, world)
                for direction, sources, expected in (
                    ("ep_to_tp", ep, tp),
                    ("tp_to_ep", tp, ep),
                ):
                    originals = [
                        {key: tensor.clone() for key, tensor in layer.items()}
                        for layer in sources
                    ]
                    outputs = []
                    for rank in range(world):
                        collectives = SimulatedCollectives(
                            self, sources, expected, model, world, rank, direction
                        )
                        target = naive_nccl_transfer_layer(
                            sources[rank],
                            model,
                            world,
                            rank,
                            direction,
                            group="test-group",
                            collectives=collectives,
                        )
                        outputs.append(target)
                        self.assertEqual(collectives.expert_calls, 2)
                        self.assertEqual(
                            collectives.attention_calls,
                            2 if direction == "tp_to_ep" else 0,
                        )
                        for name, tensor in target.items():
                            torch.testing.assert_close(
                                tensor, expected[rank][name], rtol=0, atol=0
                            )
                            self.assertNotEqual(
                                tensor.data_ptr(), sources[rank][name].data_ptr()
                            )
                            self.assertFalse(tensor.requires_grad)
                    for original, current in zip(originals, sources):
                        for name in original:
                            torch.testing.assert_close(
                                current[name], original[name], rtol=0, atol=0
                            )

    def test_attention_reverse_ignores_duplicate_kv_replicas(self):
        model, world = self.model(), 4
        ep, tp = expected_modes(model, world)
        qwidth = model.num_attention_heads * model.head_dim // world
        for rank in (1, 2, 3):
            tp[rank]["qkv"][qwidth:].fill_(float("nan"))
        for rank in range(world):
            collectives = SimulatedCollectives(
                self, tp, ep, model, world, rank, "tp_to_ep"
            )
            target = naive_nccl_transfer_layer(
                tp[rank],
                model,
                world,
                rank,
                "tp_to_ep",
                group="test-group",
                collectives=collectives,
            )
            torch.testing.assert_close(target["qkv"], ep[rank]["qkv"], rtol=0, atol=0)

    def test_host_snapshots_and_reload_have_independent_storage(self):
        model, world = self.model(True), 4
        ep, tp = expected_modes(model, world)
        for direction, target_layers in (("ep_to_tp", tp), ("tp_to_ep", ep)):
            for rank, layer in enumerate(target_layers):
                snapshot = snapshot_host_tensors(layer)
                loaded = reload_host_layer(
                    snapshot, model, world, rank, direction, device="cpu"
                )
                again = reload_host_layer(
                    snapshot, model, world, rank, direction, device="cpu"
                )
                for name in layer:
                    torch.testing.assert_close(
                        loaded[name], layer[name], rtol=0, atol=0
                    )
                    self.assertEqual(
                        len(
                            {
                                layer[name].data_ptr(),
                                snapshot[name].data_ptr(),
                                loaded[name].data_ptr(),
                                again[name].data_ptr(),
                            }
                        ),
                        4,
                    )
                    loaded[name].zero_()
                    torch.testing.assert_close(
                        snapshot[name], layer[name], rtol=0, atol=0
                    )

    def test_auxiliary_snapshots_preserve_shape_and_dtype(self):
        tensors = {
            "bias": torch.arange(16, dtype=torch.float32),
            "indices": torch.arange(5),
            "scalar": torch.tensor(7, dtype=torch.int32),
        }
        snapshot = snapshot_host_tensors(tensors)
        loaded = reload_host_tensors(snapshot, device="cpu")
        for name in tensors:
            torch.testing.assert_close(loaded[name], tensors[name])
            self.assertNotEqual(loaded[name].data_ptr(), snapshot[name].data_ptr())
        self.assertEqual(reload_host_tensors({}, device="cpu"), {})

    def test_rejects_invalid_model_source_and_host_target(self):
        model, world = self.model(), 4
        ep, tp = expected_modes(model, world)
        with self.assertRaisesRegex(ValueError, "divisible"):
            target_shapes(replace(model, num_experts=7), world, "ep_to_tp")
        with self.assertRaisesRegex(ValueError, "positive"):
            target_shapes(replace(model, head_dim=0), world, "ep_to_tp")
        with self.assertRaisesRegex(ValueError, "direction"):
            target_shapes(model, world, "forward")
        with self.assertRaisesRegex(ValueError, "rank"):
            naive_nccl_transfer_layer(ep[0], model, world, world, "ep_to_tp")
        malformed = dict(ep[0], w13=ep[0]["w13"].float())
        with self.assertRaisesRegex(ValueError, "BF16"):
            naive_nccl_transfer_layer(malformed, model, world, 0, "ep_to_tp")
        malformed = dict(ep[0], qkv=ep[0]["qkv"].T.contiguous().T)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            naive_nccl_transfer_layer(malformed, model, world, 0, "ep_to_tp")
        with self.assertRaisesRegex(ValueError, "expected shape"):
            reload_host_layer(ep[0], model, world, 0, "ep_to_tp", device="cpu")
        malformed = dict(ep[0])
        del malformed["o"]
        with self.assertRaisesRegex(ValueError, "exactly"):
            naive_nccl_transfer_layer(malformed, model, world, 0, "ep_to_tp")


if __name__ == "__main__":
    unittest.main()
