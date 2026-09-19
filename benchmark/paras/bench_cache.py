"""KV cache peer-access vs NCCL benchmark, both directions, multi-layer.

Compares three transport methods per direction:
    peer_access  - production CUDA peer-access kernel
                   (peer_access_kv_scatter / peer_access_kv_transfer).
    nccl         - gather, NCCL all_to_all_single, and destination scatter.
    nccl_overlap - same NCCL pattern, pipelined across 2 streams so a layer's
                   all_to_all overlaps with the next repetition's prep, using
                   separate staging slots and explicit reuse events.

Volume control:
    `--cache-size-gb` is the per-GPU EP cache capacity.
    `--load` is the resident fraction in (0, 1].
    The preset layer count controls repeated transfers of ONE isolated layer.
    This is not a full UMM migration or a heterogeneous GPT-OSS SWA trace.

Usage:
    torchrun --nproc_per_node=8 bench_cache.py \\
        --model qwen3-235b --tp-size 8 \\
        --cache-size-gb 10 --load 0.5 \\
        --direction both --method peer_access \\
        --warmup 3 --iters 10
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import csv
import os
import sys
import time

import torch
import torch.distributed as dist

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from common.ipc import CudaTimer, IPCContext, setup_ipc_arena
from common.layouts import KVLayout, add_volume_args, make_kv_layout
from common.model_configs import add_model_args, resolve_model
from common.slot_init import random_resident_slots
from common.cache_reference import pack_gather, unpack_gather, unpack_scatter

ppa = ppa3c = None  # Load the CUDA extension after CLI validation.


def offsets_in_arena(layout: KVLayout) -> dict:
    return {
        "tp_k": 0,
        "tp_v": layout.tp_buffer_bytes,
        "ep_k": 2 * layout.tp_buffer_bytes,
        "ep_v": 2 * layout.tp_buffer_bytes + layout.ep_buffer_bytes,
        "total": 2 * layout.tp_buffer_bytes + 2 * layout.ep_buffer_bytes,
    }


def build_scatter_routing(ctx: IPCContext, layout: KVLayout, seed: int):
    """Each replicated head sends a disjoint token chunk to every EP owner."""
    n, world, replication = (
        layout.num_resident_tokens,
        ctx.world_size,
        layout.replication_factor,
    )
    chunk = n // replication
    local_tokens = (
        torch.arange(chunk, device=ctx.device) + (ctx.rank % replication) * chunk
    )
    src_slots = (
        torch.arange(world, device=ctx.device)[:, None] * n + local_tokens + 1
    ).flatten()
    dst_ranks = torch.arange(world, device=ctx.device).repeat_interleave(chunk)
    ep_slots = torch.cat(
        [
            random_resident_slots(n, layout.ep_max_tokens, owner, seed).to(ctx.device)[
                local_tokens
            ]
            for owner in range(world)
        ]
    )
    return src_slots.int(), dst_ranks.int(), ep_slots.int()


def _peer_access_scatter_layer(
    ctx, layout, off, src_slots, dst_ranks, ep_slots, variant
):
    if variant == "v3":
        ppa3c.launch_peer_access_kv_scatter_v3(
            ctx.local_buffer_ptr,
            ctx.peer_buffer_ptrs,
            src_slots,
            dst_ranks,
            ep_slots,
            off["tp_k"],
            off["tp_v"],
            off["ep_k"],
            off["ep_v"],
            src_slots.numel(),
            layout.num_kv_heads,
            ctx.rank,
            ctx.world_size,
            layout.head_dim,
            layout.elem_size,
            0,
        )
    else:
        ppa.launch_peer_access_kv_scatter(
            ctx.local_buffer_ptr,
            ctx.peer_buffer_ptrs,
            src_slots,
            dst_ranks,
            ep_slots,
            off["tp_k"],
            off["tp_v"],
            off["ep_k"],
            off["ep_v"],
            src_slots.numel(),
            layout.heads_per_rank,
            layout.num_kv_heads,
            ctx.rank,
            ctx.world_size,
            layout.head_dim,
            layout.elem_size,
            0,
        )


def _peer_access_transfer_layer(ctx, layout, off, src_slots, dst_token_start, variant):
    if variant == "v3":
        ppa3c.launch_peer_access_kv_transfer_v3(
            ctx.local_buffer_ptr,
            ctx.peer_buffer_ptrs,
            src_slots,
            off["ep_k"],
            off["ep_v"],
            off["tp_k"],
            off["tp_v"],
            src_slots.numel(),
            dst_token_start,
            layout.num_kv_heads,
            ctx.rank,
            ctx.world_size,
            layout.head_dim,
            layout.elem_size,
            0,
        )
    else:
        ppa.launch_peer_access_kv_transfer(
            ctx.local_buffer_ptr,
            ctx.peer_buffer_ptrs,
            src_slots,
            off["ep_k"],
            off["ep_v"],
            off["tp_k"],
            off["tp_v"],
            src_slots.numel(),
            dst_token_start,
            layout.num_kv_heads,
            ctx.rank,
            ctx.world_size,
            layout.head_dim,
            layout.elem_size,
            0,
        )


def _views(ctx, layout, off):
    result = {}
    for mode in ("ep", "tp"):
        capacity = getattr(layout, f"{mode}_max_tokens")
        heads = layout.num_kv_heads if mode == "ep" else layout.heads_per_rank
        size = getattr(layout, f"{mode}_buffer_bytes")
        for kind in ("k", "v"):
            key = f"{mode}_{kind}"
            result[key] = (
                ctx.buf[off[key] : off[key] + size]
                .view(torch.bfloat16)
                .view(capacity, heads, layout.head_dim)
            )
    return result


def _pattern(tokens, heads, dim, value_offset=0):
    # Head and dimension terms catch errors that a rank-only constant misses.
    values = (
        tokens[:, None, None] * 17
        + heads[None, :, None] * 7
        + torch.arange(dim, device=tokens.device)[None, None, :] * 3
    ) % 127
    return (values + value_offset).to(torch.bfloat16)


def _initialize(ctx, layout, views, direction, seed):
    n = layout.num_resident_tokens
    batch = 4096
    if direction == "ep_to_tp":
        slots = (
            random_resident_slots(n, layout.ep_max_tokens, ctx.rank, seed)
            .to(ctx.device)
            .long()
        )
        heads = torch.arange(layout.num_kv_heads, device=ctx.device)
        for start in range(0, n, batch):
            tokens = (
                torch.arange(start, min(start + batch, n), device=ctx.device)
                + ctx.rank * n
            )
            dst = slots[start : start + batch]
            views["ep_k"][dst] = _pattern(tokens, heads, layout.head_dim)
            views["ep_v"][dst] = _pattern(tokens, heads, layout.head_dim, 128)
    else:
        heads = (
            torch.arange(layout.heads_per_rank, device=ctx.device)
            + ctx.rank * layout.num_kv_heads // ctx.world_size
        )
        for start in range(0, n * ctx.world_size, batch):
            tokens = torch.arange(
                start, min(start + batch, n * ctx.world_size), device=ctx.device
            )
            views["tp_k"][tokens + 1] = _pattern(tokens, heads, layout.head_dim)
            views["tp_v"][tokens + 1] = _pattern(tokens, heads, layout.head_dim, 128)


def _verify(ctx, layout, views, direction, seed):
    n = layout.num_resident_tokens
    if direction == "ep_to_tp":
        count = n * ctx.world_size
        heads = (
            torch.arange(layout.heads_per_rank, device=ctx.device)
            + ctx.rank * layout.num_kv_heads // ctx.world_size
        )
        positions = None
        target = "tp"
    else:
        count = n
        heads = torch.arange(layout.num_kv_heads, device=ctx.device)
        positions = (
            random_resident_slots(n, layout.ep_max_tokens, ctx.rank, seed)
            .to(ctx.device)
            .long()
        )
        target = "ep"
    ok = torch.ones((), dtype=torch.int32, device=ctx.device)
    for start in range(0, count, 4096):
        index = torch.arange(start, min(start + 4096, count), device=ctx.device)
        tokens = index if target == "tp" else index + ctx.rank * n
        dst = index + 1 if positions is None else positions[start : start + 4096]
        for kind, offset in (("k", 0), ("v", 128)):
            ok.mul_(
                (
                    views[f"{target}_{kind}"][dst]
                    == _pattern(tokens, heads, layout.head_dim, offset)
                )
                .all()
                .int()
            )
    dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=ctx.tp_group)
    if not ok.item():
        raise RuntimeError(f"{direction}: destination KV differs from reference")


def run_direction(
    ctx, layout, seed, method, direction, num_layers, warmup, iters, variant
):
    off = offsets_in_arena(layout)
    views = _views(ctx, layout, off)
    _initialize(ctx, layout, views, direction, seed)
    n, world, hpr, dim = (
        layout.num_resident_tokens,
        ctx.world_size,
        layout.heads_per_rank,
        layout.head_dim,
    )
    if direction == "ep_to_tp":
        slots = random_resident_slots(n, layout.ep_max_tokens, ctx.rank, seed).to(
            ctx.device
        )
        shape = (world, n, hpr, 2, dim)

        def direct():
            _peer_access_transfer_layer(
                ctx, layout, off, slots, ctx.rank * n + 1, variant
            )

        def pack(send):
            send.copy_(pack_gather(views["ep_k"], views["ep_v"], slots, world))

        def unpack(recv):
            values = unpack_gather(recv)
            views["tp_k"][1 : world * n + 1].copy_(values[:, :, 0, :])
            views["tp_v"][1 : world * n + 1].copy_(values[:, :, 1, :])

        remote_bytes = (
            n
            * layout.bytes_per_ep_slot
            * 2
            * layout.replication_factor
            * (world - 1)
            // world
        )
    else:
        slots, owners, dst_slots = build_scatter_routing(ctx, layout, seed)
        src_long = slots.long()
        shape = (world, n // layout.replication_factor, hpr, 2, dim)
        ep_positions = (
            random_resident_slots(n, layout.ep_max_tokens, ctx.rank, seed)
            .to(ctx.device)
            .long()
        )

        def direct():
            _peer_access_scatter_layer(
                ctx, layout, off, slots, owners, dst_slots, variant
            )

        def pack(send):
            send_view = send.view(-1, hpr, 2, dim)
            send_view[:, :, 0, :] = views["tp_k"][src_long]
            send_view[:, :, 1, :] = views["tp_v"][src_long]

        def unpack(recv):
            values = unpack_scatter(recv, layout.num_kv_heads)
            views["ep_k"][ep_positions] = values[:, :, 0, :]
            views["ep_v"][ep_positions] = values[:, :, 1, :]

        remote_bytes = n * layout.bytes_per_ep_slot * 2 * (world - 1) // world

    buffers = []
    if method != "peer_access":
        buffers = [
            (
                torch.empty(shape, device=ctx.device, dtype=torch.bfloat16),
                torch.empty(shape, device=ctx.device, dtype=torch.bfloat16),
            )
            for _ in range(2 if method == "nccl_overlap" else 1)
        ]
    prep_stream = torch.cuda.Stream() if method == "nccl_overlap" else None
    reusable = [torch.cuda.Event() for _ in buffers]
    ready = [torch.cuda.Event() for _ in buffers]

    def execute(layers):
        main = torch.cuda.current_stream()
        if prep_stream is not None:
            prep_stream.wait_stream(main)
        for layer in range(layers):
            if method == "peer_access":
                direct()
                ctx.barrier()
                continue
            index = layer % len(buffers)
            send, recv = buffers[index]
            if prep_stream is None:
                pack(send)
            else:
                with torch.cuda.stream(prep_stream):
                    if layer >= len(buffers):
                        prep_stream.wait_event(reusable[index])
                    pack(send)
                    ready[index].record()
                main.wait_event(ready[index])
            dist.all_to_all_single(recv.view(-1), send.view(-1), group=ctx.tp_group)
            unpack(recv)
            if prep_stream is not None:
                reusable[index].record()

    ctx.barrier()
    execute(1)
    ctx.barrier()
    _verify(ctx, layout, views, direction, seed)
    timer = CudaTimer(ctx.device, warmup=warmup, iters=iters)
    for _ in range(timer.total_iters):
        ctx.barrier()
        timer.tick()
        execute(num_layers)
        timer.tock()
        ctx.barrier()
    # Also check the pipelined path after it has reused both staging slots.
    _verify(ctx, layout, views, direction, seed)
    stats = timer.summary()
    stats["per_layer_mean_ms"] = stats["mean"] / num_layers
    stats["per_layer_p50_ms"] = stats["p50"] / num_layers
    stats["remote_bytes_per_rank_per_layer"] = remote_bytes
    stats["staging_bytes"] = sum(
        t.numel() * t.element_size() for pair in buffers for t in pair
    )
    return {"pass": True, **stats}


def main():
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    add_volume_args(parser)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument(
        "--direction",
        choices=("tp_to_ep", "ep_to_tp", "both"),
        default="both",
        help="tp_to_ep=scatter, ep_to_tp=gather (default: both)",
    )
    parser.add_argument(
        "--method",
        choices=("peer_access", "nccl", "nccl_overlap"),
        default="peer_access",
        help="Transport method (default: peer_access)",
    )
    parser.add_argument(
        "--variant",
        choices=("v2", "v3"),
        default="v2",
        help="peer_access kernel variant (only used when method=peer_access)",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0xCAFE)
    parser.add_argument("--out-csv", type=str, default=None)
    args = parser.parse_args()

    if args.warmup < 0 or args.iters <= 0:
        parser.error("--warmup must be nonnegative and --iters positive")
    model = resolve_model(args)
    if (
        min(args.tp_size, model.num_kv_heads, model.head_dim, model.num_hidden_layers)
        <= 0
    ):
        parser.error(
            "TP size, KV heads, head dimension, and layer count must be positive"
        )
    if model.num_kv_heads % args.tp_size and args.tp_size % model.num_kv_heads:
        parser.error("KV heads and TP size must divide one another")
    layout = make_kv_layout(model, args.tp_size, args.cache_size_gb, args.load)
    replication = layout.replication_factor
    n = layout.num_resident_tokens // replication * replication
    if n == 0:
        parser.error("cache must hold at least one token per head replica")
    layout = replace(layout, num_resident_tokens=n, tp_max_tokens=args.tp_size * n + 1)
    num_layers = model.num_hidden_layers
    global ppa, ppa3c
    if args.method == "peer_access":
        import paras_peer_access_cuda as ppa

        ppa3c = ppa

    arena = 2 * layout.tp_buffer_bytes + 2 * layout.ep_buffer_bytes
    ctx = setup_ipc_arena(arena, peer_access=args.method == "peer_access")
    if ctx.world_size != args.tp_size:
        raise SystemExit(
            f"torchrun world_size ({ctx.world_size}) != --tp-size ({args.tp_size})"
        )

    if ctx.rank == 0:
        resident_gib = (
            layout.num_resident_tokens * layout.bytes_per_ep_slot * 2 / (1024**3)
        )
        print(
            f"[rank 0] model={model.name} layers={num_layers} tp={args.tp_size} "
            f"R={layout.replication_factor} cache={args.cache_size_gb:.1f}GiB "
            f"load={args.load:.2f} resident={resident_gib:.2f}GiB "
            f"num_resident_tokens={layout.num_resident_tokens}"
        )
        print(f"[rank 0] arena={arena/(1024**3):.2f}GiB method={args.method}")
        print(
            "[rank 0] scope=isolated_homogeneous_layer_repeated; not a full UMM or hybrid-SWA switch"
        )

    results = []

    if args.direction in ("tp_to_ep", "both"):
        if ctx.rank == 0:
            print(f"[rank 0] RUN scatter ({args.method}) × {num_layers} layers")
        stats = run_direction(
            ctx,
            layout,
            args.seed,
            args.method,
            "tp_to_ep",
            num_layers,
            args.warmup,
            args.iters,
            args.variant,
        )
        results.append({"direction": "tp_to_ep", **stats})
        if ctx.rank == 0:
            print(
                f"[rank 0] scatter: total_mean={stats['mean']:.3f}ms "
                f"per_layer_mean={stats['per_layer_mean_ms']:.4f}ms "
                f"p50_per_layer={stats['per_layer_p50_ms']:.4f}ms"
            )

    if args.direction in ("ep_to_tp", "both"):
        if ctx.rank == 0:
            print(f"[rank 0] RUN transfer ({args.method}) × {num_layers} layers")
        stats = run_direction(
            ctx,
            layout,
            args.seed,
            args.method,
            "ep_to_tp",
            num_layers,
            args.warmup,
            args.iters,
            args.variant,
        )
        results.append({"direction": "ep_to_tp", **stats})
        if ctx.rank == 0:
            print(
                f"[rank 0] transfer: total_mean={stats['mean']:.3f}ms "
                f"per_layer_mean={stats['per_layer_mean_ms']:.4f}ms "
                f"p50_per_layer={stats['per_layer_p50_ms']:.4f}ms"
            )

    if ctx.rank == 0 and args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        new_file = (
            not os.path.exists(args.out_csv) or os.path.getsize(args.out_csv) == 0
        )
        with open(args.out_csv, "a", newline="") as f:
            fields = [
                "timestamp",
                "model",
                "num_layers",
                "tp_size",
                "replication",
                "cache_size_gb",
                "load",
                "num_resident_tokens",
                "direction",
                "method",
                "variant",
                "scope",
                "remote_bytes_per_rank_per_layer",
                "staging_bytes",
                "total_mean_ms",
                "total_p50_ms",
                "per_layer_mean_ms",
                "per_layer_p50_ms",
                "min_ms",
                "max_ms",
                "n",
            ]
            if not new_file:
                with open(args.out_csv, newline="") as existing:
                    if next(csv.reader(existing)) != fields:
                        raise ValueError(
                            "CSV schema differs; use a fresh --out-csv file"
                        )
            w = csv.DictWriter(f, fieldnames=fields)
            if new_file:
                w.writeheader()
            for r in results:
                w.writerow(
                    {
                        "timestamp": int(time.time()),
                        "model": model.name,
                        "num_layers": num_layers,
                        "tp_size": args.tp_size,
                        "replication": layout.replication_factor,
                        "cache_size_gb": args.cache_size_gb,
                        "load": args.load,
                        "num_resident_tokens": layout.num_resident_tokens,
                        "direction": r["direction"],
                        "method": args.method,
                        "variant": args.variant if args.method == "peer_access" else "",
                        "scope": "isolated_homogeneous_layer_repeated",
                        "remote_bytes_per_rank_per_layer": r[
                            "remote_bytes_per_rank_per_layer"
                        ],
                        "staging_bytes": r["staging_bytes"],
                        "total_mean_ms": r.get("mean", 0.0),
                        "total_p50_ms": r.get("p50", 0.0),
                        "per_layer_mean_ms": r.get("per_layer_mean_ms", 0.0),
                        "per_layer_p50_ms": r.get("per_layer_p50_ms", 0.0),
                        "min_ms": r.get("min", 0.0),
                        "max_ms": r.get("max", 0.0),
                        "n": r.get("n", 0),
                    }
                )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
