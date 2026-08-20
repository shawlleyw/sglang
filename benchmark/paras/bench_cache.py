"""KV cache peer-access vs NCCL benchmark, both directions, multi-layer.

Compares three transport methods per direction:
    peer_access  - production CUDA peer-access kernel
                   (peer_access_kv_scatter / peer_access_kv_transfer).
    nccl         - production NCCL all_to_all_single per layer.
    nccl_overlap - benchmark-only two-stream NCCL baseline.

Volume control:
    `--cache-size-gb` is the per-GPU EP cache capacity.
    `--load` is the resident fraction in (0, 1].
    `--num-layers` (from model preset) controls how many back-to-back
    transfers form ONE timed iteration (one iter ≈ one full EP↔TP switch).

Usage:
    torchrun --nproc_per_node=8 bench_cache.py \\
        --model qwen3-235b --tp-size 8 \\
        --cache-size-gb 10 --load 0.5 \\
        --direction both --method peer_access \\
        --warmup 3 --iters 10
"""

from __future__ import annotations

import argparse
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
from common.slot_init import fill_slots_bf16, random_resident_slots

import paras_peer_access_cuda as ppa
ppa3c = ppa


def offsets_in_arena(layout: KVLayout) -> dict:
    return {
        "tp_k": 0,
        "tp_v": layout.tp_buffer_bytes,
        "ep_k": 2 * layout.tp_buffer_bytes,
        "ep_v": 2 * layout.tp_buffer_bytes + layout.ep_buffer_bytes,
        "total": 2 * layout.tp_buffer_bytes + 2 * layout.ep_buffer_bytes,
    }


def build_scatter_routing(ctx: IPCContext, layout: KVLayout, seed: int):
    """TP->EP routing mirroring production scatter volume.

    After EP->TP each TP rank holds the FULL global cache (W*N tokens of its
    1 head, per the layout's tp_max_tokens=W*N+1).  Production scatter
    applies R-way head-replica dedup, so each TP rank sources
    M_target = W*N/R tokens (= (W/R) * num_resident_tokens).

    We cap M at ep_max_tokens-1 to keep per-source-rank slot ranges
    disjoint within each destination's EP buffer (n = M/W <= ep_region
    = ep_max_tokens/W).  This preserves v2/v3 bytewise correctness at the
    cost of under-shooting production volume when load > R/W
    (e.g. load=0.5, R=2 -> M = 2N instead of the 4N production value).
    """
    N = layout.num_resident_tokens
    W = ctx.world_size
    R = layout.replication_factor
    M = min((W * N) // R, layout.ep_max_tokens - 1)
    M = (M // W) * W
    tp_src_slots = random_resident_slots(M, layout.tp_max_tokens, ctx.rank, seed).to(ctx.device)
    dst_ranks = (torch.arange(M, dtype=torch.int32) % W).to(ctx.device)
    ep_region = layout.ep_max_tokens // W
    ep_slots = torch.zeros(M, dtype=torch.int32, device=ctx.device)
    for r in range(W):
        mask = dst_ranks == r
        idx = mask.nonzero(as_tuple=False).flatten()
        n = idx.numel()
        ep_slots[idx] = 1 + ctx.rank * ep_region + torch.arange(
            n, dtype=torch.int32, device=ctx.device)
    return tp_src_slots, dst_ranks, ep_slots


def _peer_access_scatter_layer(ctx, layout, off, src_slots, dst_ranks, ep_slots, variant):
    if variant == "v3":
        ppa3c.launch_peer_access_kv_scatter_v3(
            ctx.local_buffer_ptr, ctx.peer_buffer_ptrs,
            src_slots, dst_ranks, ep_slots,
            off["tp_k"], off["tp_v"], off["ep_k"], off["ep_v"],
            src_slots.numel(),
            layout.num_kv_heads, ctx.rank, ctx.world_size,
            layout.head_dim, layout.elem_size, 0,
        )
    else:
        ppa.launch_peer_access_kv_scatter(
            ctx.local_buffer_ptr, ctx.peer_buffer_ptrs,
            src_slots, dst_ranks, ep_slots,
            off["tp_k"], off["tp_v"], off["ep_k"], off["ep_v"],
            src_slots.numel(), layout.heads_per_rank, layout.num_kv_heads,
            ctx.rank, ctx.world_size, layout.head_dim, layout.elem_size, 0,
        )


def _peer_access_transfer_layer(ctx, layout, off, src_slots, dst_token_start, variant):
    if variant == "v3":
        ppa3c.launch_peer_access_kv_transfer_v3(
            ctx.local_buffer_ptr, ctx.peer_buffer_ptrs, src_slots,
            off["ep_k"], off["ep_v"], off["tp_k"], off["tp_v"],
            src_slots.numel(), dst_token_start,
            layout.num_kv_heads, ctx.rank, ctx.world_size,
            layout.head_dim, layout.elem_size, 0,
        )
    else:
        ppa.launch_peer_access_kv_transfer(
            ctx.local_buffer_ptr, ctx.peer_buffer_ptrs, src_slots,
            off["ep_k"], off["ep_v"], off["tp_k"], off["tp_v"],
            src_slots.numel(), dst_token_start, layout.num_kv_heads,
            ctx.rank, ctx.world_size, layout.head_dim, layout.elem_size, 0,
        )


def run_scatter(ctx: IPCContext, layout: KVLayout, seed: int, method: str,
                num_layers: int, warmup: int, iters: int, variant: str) -> dict:
    off = offsets_in_arena(layout)
    src_slots, dst_ranks, ep_slots = build_scatter_routing(ctx, layout, seed)

    tp_k = ctx.buf[off["tp_k"]:off["tp_k"] + layout.tp_buffer_bytes].view(torch.bfloat16).view(
        layout.tp_max_tokens, layout.heads_per_rank, layout.head_dim
    )
    tp_v = ctx.buf[off["tp_v"]:off["tp_v"] + layout.tp_buffer_bytes].view(torch.bfloat16).view(
        layout.tp_max_tokens, layout.heads_per_rank, layout.head_dim
    )
    ep_k = ctx.buf[off["ep_k"]:off["ep_k"] + layout.ep_buffer_bytes].view(torch.bfloat16).view(
        layout.ep_max_tokens, layout.num_kv_heads, layout.head_dim
    )
    ep_v = ctx.buf[off["ep_v"]:off["ep_v"] + layout.ep_buffer_bytes].view(torch.bfloat16).view(
        layout.ep_max_tokens, layout.num_kv_heads, layout.head_dim
    )
    ctx.buf[off["ep_k"]:off["ep_k"] + layout.ep_buffer_bytes].zero_()
    ctx.buf[off["ep_v"]:off["ep_v"] + layout.ep_buffer_bytes].zero_()
    fill_slots_bf16(tp_k, src_slots, ctx.rank, layout.heads_per_rank, layout.head_dim, 0.0)
    fill_slots_bf16(tp_v, src_slots, ctx.rank, layout.heads_per_rank, layout.head_dim, 128.0)

    M = src_slots.numel()
    W = ctx.world_size
    HPR = layout.heads_per_rank
    HD = layout.head_dim
    send_buf = recv_buf = None
    sorted_src = None
    ep_post_pos = None
    if method in ("nccl", "nccl_overlap"):
        per_token = 2 * HPR * HD
        order = torch.argsort(dst_ranks, stable=True)
        sorted_src = src_slots[order].to(torch.long)
        send_buf = torch.empty(M * per_token, dtype=torch.bfloat16, device=ctx.device)
        recv_buf = torch.empty_like(send_buf)
        ep_post_pos = (torch.arange(M // W, dtype=torch.long, device=ctx.device) + 1)

    def nccl_scatter_layer():
        send_view = send_buf.view(M, HPR, 2, HD)
        send_view[:, :, 0, :] = tp_k[sorted_src]
        send_view[:, :, 1, :] = tp_v[sorted_src]
        dist.all_to_all_single(recv_buf, send_buf, group=ctx.tp_group)
        recv_view = recv_buf.view(W, M // W, HPR, 2, HD)
        permuted = recv_view.permute(1, 0, 2, 3, 4).contiguous().view(
            M // W, W * HPR, 2, HD)
        ep_k[ep_post_pos] = permuted[:, :layout.num_kv_heads, 0, :]
        ep_v[ep_post_pos] = permuted[:, :layout.num_kv_heads, 1, :]

    timer = CudaTimer(ctx.device, warmup=warmup, iters=iters)
    for _ in range(timer.total_iters):
        ctx.barrier()
        timer.tick()
        if method == "peer_access":
            for _ in range(num_layers):
                _peer_access_scatter_layer(ctx, layout, off, src_slots, dst_ranks, ep_slots, variant)
                ctx.barrier()
        elif method == "nccl":
            for _ in range(num_layers):
                nccl_scatter_layer()
        elif method == "nccl_overlap":
            s1 = torch.cuda.Stream()
            s2 = torch.cuda.Stream()
            for i in range(num_layers):
                with torch.cuda.stream(s1 if i % 2 == 0 else s2):
                    nccl_scatter_layer()
            torch.cuda.current_stream().wait_stream(s1)
            torch.cuda.current_stream().wait_stream(s2)
        else:
            raise SystemExit(f"Unknown method: {method}")
        timer.tock()
        ctx.barrier()

    s = timer.summary()
    s["per_layer_mean_ms"] = s["mean"] / num_layers
    s["per_layer_p50_ms"] = s["p50"] / num_layers
    return {"pass": True, **s}


def run_transfer(ctx: IPCContext, layout: KVLayout, seed: int, method: str,
                 num_layers: int, warmup: int, iters: int, variant: str) -> dict:
    off = offsets_in_arena(layout)
    ep_k = ctx.buf[off["ep_k"]:off["ep_k"] + layout.ep_buffer_bytes].view(torch.bfloat16).view(
        layout.ep_max_tokens, layout.num_kv_heads, layout.head_dim
    )
    ep_v = ctx.buf[off["ep_v"]:off["ep_v"] + layout.ep_buffer_bytes].view(torch.bfloat16).view(
        layout.ep_max_tokens, layout.num_kv_heads, layout.head_dim
    )
    tp_k = ctx.buf[off["tp_k"]:off["tp_k"] + layout.tp_buffer_bytes].view(torch.bfloat16).view(
        layout.tp_max_tokens, layout.heads_per_rank, layout.head_dim
    )
    tp_v = ctx.buf[off["tp_v"]:off["tp_v"] + layout.tp_buffer_bytes].view(torch.bfloat16).view(
        layout.tp_max_tokens, layout.heads_per_rank, layout.head_dim
    )
    ctx.buf[off["tp_k"]:off["tp_k"] + layout.tp_buffer_bytes].zero_()
    ctx.buf[off["tp_v"]:off["tp_v"] + layout.tp_buffer_bytes].zero_()

    src_slots = random_resident_slots(layout.num_resident_tokens, layout.ep_max_tokens,
                                       ctx.rank, seed).to(ctx.device)
    fill_slots_bf16(ep_k, src_slots, ctx.rank, layout.num_kv_heads, layout.head_dim, 0.0)
    fill_slots_bf16(ep_v, src_slots, ctx.rank, layout.num_kv_heads, layout.head_dim, 128.0)

    N = layout.num_resident_tokens
    W = ctx.world_size
    R = layout.replication_factor
    HPR = layout.heads_per_rank
    HD = layout.head_dim
    NUM_HEADS_SRC = layout.num_kv_heads // HPR
    dst_token_start = int(ctx.rank * N + 1)

    send_buf = recv_buf = None
    src_long = None
    tp_post_pos = None
    if method in ("nccl", "nccl_overlap"):
        send_elems = W * N * HPR * 2 * HD
        send_buf = torch.empty(send_elems, dtype=torch.bfloat16, device=ctx.device)
        recv_buf = torch.empty_like(send_buf)
        src_long = src_slots.to(torch.long)
        tp_post_pos = (torch.arange(N, dtype=torch.long, device=ctx.device) + dst_token_start)
        tp_post_pos = tp_post_pos.clamp_max(layout.tp_max_tokens - 1)

    def nccl_transfer_layer():
        gathered_k = ep_k[src_long]
        gathered_v = ep_v[src_long]
        stacked = torch.stack([gathered_k, gathered_v], dim=2)
        permuted = stacked.view(N, NUM_HEADS_SRC, HPR, 2, HD).permute(1, 0, 2, 3, 4).contiguous()
        if R > 1:
            permuted = permuted.repeat_interleave(R, dim=0)
        send_buf.copy_(permuted.view(-1))
        dist.all_to_all_single(recv_buf, send_buf, group=ctx.tp_group)
        recv_view = recv_buf.view(W, N, HPR, 2, HD)
        post = recv_view.permute(1, 0, 2, 3, 4).contiguous().view(N, W * HPR, 2, HD)
        tp_k[tp_post_pos] = post[:, :HPR, 0, :]
        tp_v[tp_post_pos] = post[:, :HPR, 1, :]

    timer = CudaTimer(ctx.device, warmup=warmup, iters=iters)
    for _ in range(timer.total_iters):
        ctx.barrier()
        timer.tick()
        if method == "peer_access":
            for _ in range(num_layers):
                _peer_access_transfer_layer(ctx, layout, off, src_slots, dst_token_start, variant)
                ctx.barrier()
        elif method == "nccl":
            for _ in range(num_layers):
                nccl_transfer_layer()
        elif method == "nccl_overlap":
            s1 = torch.cuda.Stream()
            s2 = torch.cuda.Stream()
            for i in range(num_layers):
                with torch.cuda.stream(s1 if i % 2 == 0 else s2):
                    nccl_transfer_layer()
            torch.cuda.current_stream().wait_stream(s1)
            torch.cuda.current_stream().wait_stream(s2)
        else:
            raise SystemExit(f"Unknown method: {method}")
        timer.tock()
        ctx.barrier()

    s = timer.summary()
    s["per_layer_mean_ms"] = s["mean"] / num_layers
    s["per_layer_p50_ms"] = s["p50"] / num_layers
    return {"pass": True, **s}


def main():
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    add_volume_args(parser)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--direction", choices=("tp_to_ep", "ep_to_tp", "both"),
                        default="both",
                        help="tp_to_ep=scatter, ep_to_tp=gather (default: both)")
    parser.add_argument("--method", choices=("peer_access", "nccl", "nccl_overlap"),
                        default="peer_access",
                        help="Transport method (default: peer_access)")
    parser.add_argument("--variant", choices=("v2", "v3"), default="v2",
                        help="peer_access kernel variant (only used when method=peer_access)")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0xCAFE)
    parser.add_argument("--out-csv", type=str, default=None)
    args = parser.parse_args()

    model = resolve_model(args)
    layout = make_kv_layout(model, args.tp_size, args.cache_size_gb, args.load)
    num_layers = model.num_hidden_layers

    arena = 2 * layout.tp_buffer_bytes + 2 * layout.ep_buffer_bytes
    ctx = setup_ipc_arena(arena)
    if ctx.world_size != args.tp_size:
        raise SystemExit(f"torchrun world_size ({ctx.world_size}) != --tp-size ({args.tp_size})")

    if ctx.rank == 0:
        resident_gib = layout.num_resident_tokens * layout.bytes_per_ep_slot * 2 / (1024 ** 3)
        print(f"[rank 0] model={model.name} layers={num_layers} tp={args.tp_size} "
              f"R={layout.replication_factor} cache={args.cache_size_gb:.1f}GiB "
              f"load={args.load:.2f} resident={resident_gib:.2f}GiB "
              f"num_resident_tokens={layout.num_resident_tokens}")
        print(f"[rank 0] arena={arena/(1024**3):.2f}GiB method={args.method}")

    results = []

    if args.direction in ("tp_to_ep", "both"):
        if ctx.rank == 0:
            print(f"[rank 0] RUN scatter ({args.method}) × {num_layers} layers")
        stats = run_scatter(ctx, layout, args.seed, args.method, num_layers,
                            args.warmup, args.iters, args.variant)
        results.append({"direction": "tp_to_ep", **stats})
        if ctx.rank == 0:
            print(f"[rank 0] scatter: total_mean={stats['mean']:.3f}ms "
                  f"per_layer_mean={stats['per_layer_mean_ms']:.4f}ms "
                  f"p50_per_layer={stats['per_layer_p50_ms']:.4f}ms")

    if args.direction in ("ep_to_tp", "both"):
        if ctx.rank == 0:
            print(f"[rank 0] RUN transfer ({args.method}) × {num_layers} layers")
        stats = run_transfer(ctx, layout, args.seed, args.method, num_layers,
                             args.warmup, args.iters, args.variant)
        results.append({"direction": "ep_to_tp", **stats})
        if ctx.rank == 0:
            print(f"[rank 0] transfer: total_mean={stats['mean']:.3f}ms "
                  f"per_layer_mean={stats['per_layer_mean_ms']:.4f}ms "
                  f"p50_per_layer={stats['per_layer_p50_ms']:.4f}ms")

    if ctx.rank == 0 and args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        new_file = not os.path.exists(args.out_csv)
        with open(args.out_csv, "a", newline="") as f:
            fields = ["timestamp", "model", "num_layers", "tp_size", "replication",
                      "cache_size_gb", "load", "num_resident_tokens",
                      "direction", "method",
                      "total_mean_ms", "total_p50_ms",
                      "per_layer_mean_ms", "per_layer_p50_ms",
                      "min_ms", "max_ms", "n"]
            w = csv.DictWriter(f, fieldnames=fields)
            if new_file:
                w.writeheader()
            for r in results:
                w.writerow({
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
                    "total_mean_ms": r.get("mean", 0.0),
                    "total_p50_ms": r.get("p50", 0.0),
                    "per_layer_mean_ms": r.get("per_layer_mean_ms", 0.0),
                    "per_layer_p50_ms": r.get("per_layer_p50_ms", 0.0),
                    "min_ms": r.get("min", 0.0),
                    "max_ms": r.get("max", 0.0),
                    "n": r.get("n", 0),
                })

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
