"""Real eight-GPU IPC attention reconstruction with four replicated KV heads.

Run with torchrun --standalone --nproc-per-node=8 -m pytest -q <this file>.
"""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from sglang.srt.paras.mode import ParaSMode


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 8, reason="Requires torchrun with 8 ranks"
)
def test_tp8_attention_ipc_uses_four_distinct_kv_shards():
    from sglang.srt.paras.attention_transfer import transfer_attention
    from sglang.srt.paras.peer_access import exchange_buffer_addresses_ipc

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    try:
        # Actual Qwen3-235B attention dimensions, including replicated KV.
        h, heads, kv_heads, d, t = 4096, 64, 4, 128, 8
        q, kv = heads * d, kv_heads * d
        qs, ks = q // t, d
        shapes = {
            ParaSMode.EP: {"qkv_proj": (q + 2 * kv, h), "o_proj": (h, q)},
            ParaSMode.TP: {"qkv_proj": (qs + 2 * ks, h), "o_proj": (h, qs)},
        }
        entries, cursor = {}, 256
        for mode, projections in shapes.items():
            for proj, shape in projections.items():
                name = f"model.layers.0.self_attn.{proj}.{'weight' if mode == ParaSMode.EP else 'tp_weight'}"
                entries[name] = SimpleNamespace(offset_bytes=cursor, shape=shape)
                cursor += 2 * shape[0] * shape[1]
        buffer = torch.empty(cursor, dtype=torch.uint8, device="cuda")

        def view(name):
            entry = entries[name]
            size = 2 * entry.shape[0] * entry.shape[1]
            return (
                buffer[entry.offset_bytes : entry.offset_bytes + size]
                .view(torch.bfloat16)
                .view(entry.shape)
            )

        manager = SimpleNamespace(
            _unified_spec=SimpleNamespace(
                prefix="model",
                hidden_size=h,
                num_heads=heads,
                num_kv_heads=kv_heads,
                head_dim=d,
                tp_size=t,
            ),
            _entries=entries,
            get_view=view,
        )
        generator = torch.Generator().manual_seed(1789)
        reference = {
            proj: torch.randn(shape, generator=generator, dtype=torch.bfloat16).cuda()
            for proj, shape in shapes[ParaSMode.EP].items()
        }
        for proj, tensor in reference.items():
            view(f"model.layers.0.self_attn.{proj}.weight").copy_(tensor)
        addresses = exchange_buffer_addresses_ipc(
            buffer.data_ptr(), dist.group.WORLD, t, rank
        )
        bases = torch.tensor(addresses, device="cuda", dtype=torch.int64)

        transfer_attention(manager, 0, ParaSMode.TP, rank, bases)
        qkv = view("model.layers.0.self_attn.qkv_proj.tp_weight")
        torch.testing.assert_close(
            qkv[:qs], reference["qkv_proj"][rank * qs : (rank + 1) * qs], rtol=0, atol=0
        )
        head = rank // 2
        for local_start, full_start in ((qs, q), (qs + ks, q + kv)):
            torch.testing.assert_close(
                qkv[local_start : local_start + ks],
                reference["qkv_proj"][
                    full_start + head * ks : full_start + (head + 1) * ks
                ],
                rtol=0,
                atol=0,
            )
        torch.testing.assert_close(
            view("model.layers.0.self_attn.o_proj.tp_weight"),
            reference["o_proj"][:, rank * qs : (rank + 1) * qs],
            rtol=0,
            atol=0,
        )

        # Destroy every full copy and poison the duplicate KV replicas.
        # Reconstruction must read K/V only from ranks 0,2,4,6, yet still
        # read distinct Q/O shards from all eight ranks.
        if rank % 2:
            qkv[qs:].fill_(float("nan"))
        sources = {
            proj: view(f"model.layers.0.self_attn.{proj}.tp_weight")
            for proj in reference
        }

        def clear_destination():
            for proj in reference:
                view(f"model.layers.0.self_attn.{proj}.weight").fill_(float("nan"))

        def fence():
            torch.cuda.synchronize()
            dist.barrier()

        # Compile before CUDA graph capture. Fences remain outside the graph:
        # every rank must finish reading peer shards before source updates.
        before_warmup = {proj: tensor.view(torch.int16).clone() for proj, tensor in sources.items()}
        clear_destination()
        fence()
        transfer_attention(manager, 0, ParaSMode.EP, rank, bases)
        fence()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            transfer_attention(manager, 0, ParaSMode.EP, rank, bases)
        fence()

        for proj, tensor in sources.items():
            assert torch.equal(tensor.view(torch.int16), before_warmup[proj])

        for execution in ("eager", "cuda_graph", "cuda_graph_changed_source"):
            if execution == "cuda_graph_changed_source":
                for tensor in reference.values():
                    tensor.neg_()
                for tensor in sources.values():
                    tensor.neg_()
            snapshots = {proj: tensor.view(torch.int16).clone() for proj, tensor in sources.items()}
            clear_destination()
            fence()
            if execution == "eager":
                transfer_attention(manager, 0, ParaSMode.EP, rank, bases)
            else:
                graph.replay()
            fence()
            for proj, expected in reference.items():
                torch.testing.assert_close(
                    view(f"model.layers.0.self_attn.{proj}.weight"),
                    expected, rtol=0, atol=0,
                )
                # Exact bit equality also checks the poisoned replica NaNs.
                assert torch.equal(sources[proj].view(torch.int16), snapshots[proj])
            fence()
    finally:
        dist.destroy_process_group()
