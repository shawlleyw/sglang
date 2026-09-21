"""GPU correctness checks for the production TP->EP peer-access kernel.

Run with torchrun (2, 4, or 8 GPUs). Covers uneven/unsorted routes, replicated
heads, multiple local heads, empty input, slot-zero skips, vector tails, and
CUDA graph replay. Checks the entire destination, including untouched slots,
against a CPU reference and verifies that the source was not modified.
"""

import torch
import torch.distributed as dist

from common.ipc import init_torchrun, setup_ipc_arena


def routing(rank, world, heads, counts, shuffled):
    replication = max(world // heads, 1)
    replica = rank % replication
    generator = torch.Generator().manual_seed(91)
    source_slots = torch.randperm(sum(counts), generator=generator) + 1
    rows = []
    offset = 0
    for owner, count in enumerate(counts):
        slots = torch.randperm(count, generator=generator) + 1
        begin = count * replica // replication
        end = count * (replica + 1) // replication
        token = torch.arange(begin, end)
        src = source_slots[offset + token].clone()
        dst = slots[token].clone()
        # Exercise independent source/destination padding skips.
        src[token % 17 == 3] = 0
        dst[token % 19 == 5] = 0
        rows.append(torch.stack((src, torch.full_like(src, owner), dst), dim=1))
        offset += count
    rows = torch.cat(rows)
    if shuffled:
        rows = rows[torch.randperm(len(rows), generator=generator)]
    return rows.to(torch.int32).contiguous()


def source_values(rank, capacity, hpr, dim, dtype):
    # Each K/V element depends on rank, source slot, head and channel.
    slot = torch.arange(capacity)[:, None, None]
    head = torch.arange(hpr)[None, :, None]
    channel = torch.arange(dim)[None, None, :]
    k = ((slot * 7 + head * 11 + channel + rank * 13) % 127).to(dtype)
    v = ((slot * 3 + head * 17 + channel * 5 + rank * 19) % 127).to(dtype)
    return k, v


def main():
    import paras_peer_access_cuda as ppa

    rank, world = init_torchrun()
    torch.set_num_threads(1)
    cases = [
        ("gpt-oss", 8, 64, torch.bfloat16, 37, False),
        ("gpt-oss-unsorted", 8, 64, torch.bfloat16, 41, True),
        ("replicated-heads", max(world // 2, 1), 128, torch.bfloat16, 43, True),
        ("multiple-heads-tail", world * 2, 96, torch.bfloat16, 39, True),
        ("one-byte", world, 64, torch.uint8, 29, True),
        ("four-byte", world, 64, torch.float32, 31, True),
        ("tiny", world, 64, torch.bfloat16, 1, True),
        ("empty", world, 64, torch.bfloat16, 0, False),
        ("multi-iteration", world, 64, torch.bfloat16, 20003, False),
    ]
    # Reuse one IPC allocation across cases; every case has disjoint TP/EP views.
    ctx = setup_ipc_arena(256 * 1024**2)
    for name, heads, dim, dtype, count, shuffled in cases:
        counts = [count + owner * 2 if count > 1 else count for owner in range(world)]
        counts[1] = 0  # Include an EP peer with no incoming tokens.
        hpr = max(heads // world, 1)
        src_capacity, dst_capacity = sum(counts) + 3, max(counts) + 3
        elem = dtype.itemsize
        src_bytes = src_capacity * hpr * dim * elem
        dst_bytes = dst_capacity * heads * dim * elem
        offsets = [0, src_bytes, 2 * src_bytes, 2 * src_bytes + dst_bytes]
        assert 2 * (src_bytes + dst_bytes) <= ctx.buf.numel()
        views = []
        for offset, capacity, local_heads in zip(
            offsets, [src_capacity] * 2 + [dst_capacity] * 2, [hpr] * 2 + [heads] * 2
        ):
            size = capacity * local_heads * dim * elem
            views.append(
                ctx.buf[offset : offset + size]
                .view(dtype)
                .view(capacity, local_heads, dim)
            )
        src_k, src_v = source_values(rank, src_capacity, hpr, dim, dtype)
        views[0].copy_(src_k)
        views[1].copy_(src_v)
        expected = [
            torch.full((dst_capacity, heads, dim), 251, dtype=dtype) for _ in range(2)
        ]
        for sender in range(world):
            rows = routing(sender, world, heads, counts, shuffled).long()
            valid = (rows[:, 1] == rank) & (rows[:, 0] != 0) & (rows[:, 2] != 0)
            src, dst = rows[valid, 0], rows[valid, 2]
            first_head = sender * heads // world
            for target, values in zip(
                expected, source_values(sender, src_capacity, hpr, dim, dtype)
            ):
                target[dst, first_head : first_head + hpr] = values[src]
        rows = routing(rank, world, heads, counts, shuffled).to(ctx.device)
        src, owner, dst = [rows[:, col].contiguous() for col in range(3)]

        def launch():
            ppa.launch_peer_access_kv_scatter(
                ctx.local_buffer_ptr,
                ctx.peer_buffer_ptrs,
                src,
                owner,
                dst,
                *offsets,
                len(rows),
                hpr,
                heads,
                rank,
                world,
                dim,
                elem,
                torch.cuda.current_stream().cuda_stream,
            )

        for mode in ("eager", "graph"):
            views[2].fill_(251)
            views[3].fill_(251)
            ctx.barrier()
            if mode == "eager":
                launch()
            else:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    launch()
                graph.replay()
            ctx.barrier()
            ok = all(
                torch.equal(view.cpu(), ref)
                for view, ref in zip(views, [src_k, src_v, *expected])
            )
            passed = torch.tensor(int(ok), device=ctx.device)
            dist.all_reduce(passed, op=dist.ReduceOp.MIN)
            if not passed.item():
                raise AssertionError(f"{name}/{mode}: incorrect source or destination")
        if rank == 0:
            print(f"PASS {name}: eager and graph replay", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
