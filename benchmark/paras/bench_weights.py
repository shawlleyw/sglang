"""MoE weight peer-access vs NCCL benchmark, both kernels, both directions, multi-layer.

Compares three transport methods per direction:
    peer_access  - production fused CUDA kernels
                   (peer_access_fused_transfer_w{13,2}_v2 for EP→TP,
                    peer_access_fused_transfer_w{13,2}_ep for TP→EP).
    nccl         - production NCCL all_to_all_single per layer
                   (mirrors paras_configure_tp_mlp_naive).
    nccl_overlap - same NCCL pattern, pipelined across 2 streams to overlap
                   layer N's all_to_all with layer N+1's pre-permute
                   (mirrors paras_configure_tp_overlap).

Layer count comes from the model preset (`num_hidden_layers`); one timed
iteration is one full per-direction switch across all layers.

Usage:
    torchrun --nproc_per_node=8 bench_weights.py \\
        --model qwen3-235b --tp-size 8 \\
        --kernel both --direction both --method peer_access \\
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
from common.layouts import WeightLayout, make_weight_layout
from common.model_configs import add_model_args, resolve_model

import paras_peer_access_cuda as ppa
ppa3 = ppa


def w13_offsets(layout: WeightLayout) -> dict:
    return {
        "ep": 0,
        "tp": layout.w13_ep_buffer_bytes,
        "total": layout.w13_ep_buffer_bytes + layout.w13_tp_buffer_bytes,
    }


def w2_offsets(layout: WeightLayout) -> dict:
    return {
        "ep": 0,
        "tp": layout.w2_ep_buffer_bytes,
        "total": layout.w2_ep_buffer_bytes + layout.w2_tp_buffer_bytes,
    }


def _peer_access_w13_ep_to_tp(ctx, layout, off, variant):
    if variant == "v3":
        ppa3.launch_peer_access_fused_transfer_w13_v3(
            ctx.local_buffer_ptr, ctx.peer_buffer_ptrs,
            off["ep"], off["tp"],
            ctx.rank, ctx.world_size,
            layout.E_local,
            layout.H, layout.I_full,
            layout.num_gates, layout.elem_size, 0,
        )
    else:
        ppa.launch_peer_access_fused_transfer_w13_v2(
            ctx.local_buffer_ptr, ctx.peer_buffer_ptrs,
            off["ep"], off["tp"],
            ctx.rank, ctx.world_size,
            layout.E_local, layout.I_prime_H, layout.num_gates,
            layout.elem_size, 0,
        )


def _peer_access_w13_tp_to_ep(ctx, layout, off, variant):
    if variant == "v3":
        ppa3.launch_peer_access_fused_transfer_w13_v3_ep(
            ctx.local_buffer_ptr, ctx.peer_buffer_ptrs,
            off["tp"], off["ep"],
            ctx.rank, ctx.world_size,
            layout.E_local,
            layout.H, layout.I_full,
            layout.num_gates, layout.elem_size, 0,
        )
    else:
        ppa.launch_peer_access_fused_transfer_w13_ep(
            ctx.local_buffer_ptr, ctx.peer_buffer_ptrs,
            off["tp"], off["ep"],
            ctx.rank, ctx.world_size,
            layout.E_local, layout.I_prime_H, layout.num_gates,
            layout.elem_size, 0,
        )


def _peer_access_w2_ep_to_tp(ctx, layout, off, variant):
    if variant == "v3":
        ppa3.launch_peer_access_fused_transfer_w2_v3(
            ctx.local_buffer_ptr, ctx.peer_buffer_ptrs,
            off["ep"], off["tp"],
            ctx.rank, ctx.world_size,
            layout.E_local,
            layout.H, layout.I_full, layout.elem_size, 0,
        )
    else:
        ppa.launch_peer_access_fused_transfer_w2_v2(
            ctx.local_buffer_ptr, ctx.peer_buffer_ptrs,
            off["ep"], off["tp"],
            ctx.rank, ctx.world_size,
            layout.E_local, layout.H,
            layout.I_full * layout.elem_size,
            layout.I_prime * layout.elem_size, 0,
        )


def _peer_access_w2_tp_to_ep(ctx, layout, off, variant):
    if variant == "v3":
        ppa3.launch_peer_access_fused_transfer_w2_v3_ep(
            ctx.local_buffer_ptr, ctx.peer_buffer_ptrs,
            off["tp"], off["ep"],
            ctx.rank, ctx.world_size,
            layout.E_local,
            layout.H, layout.I_full, layout.elem_size, 0,
        )
    else:
        ppa.launch_peer_access_fused_transfer_w2_ep(
            ctx.local_buffer_ptr, ctx.peer_buffer_ptrs,
            off["tp"], off["ep"],
            ctx.rank, ctx.world_size,
            layout.E_local, layout.H,
            layout.I_full * layout.elem_size,
            layout.I_prime * layout.elem_size, 0,
        )


def _nccl_w13_send_recv(layout: WeightLayout, device) -> tuple:
    """Pre-permuted (tp_size, E_local, num_gates, I'*H) buffer that mirrors
    the all_to_all input shape used in paras_configure_tp_mlp_naive."""
    elems = layout.tp_size * layout.E_local * layout.num_gates * layout.I_prime_H
    send = torch.zeros(elems, dtype=torch.bfloat16, device=device)
    recv = torch.empty_like(send)
    return send, recv


def _nccl_w2_send_recv(layout: WeightLayout, device) -> tuple:
    """Pre-permuted (tp_size, E_local, H, I') buffer that mirrors the w2
    all_to_all input shape."""
    elems = layout.tp_size * layout.E_local * layout.H * layout.I_prime
    send = torch.zeros(elems, dtype=torch.bfloat16, device=device)
    recv = torch.empty_like(send)
    return send, recv


def _nccl_layer(send_buf, recv_buf, group, stream=None):
    if stream is not None:
        with torch.cuda.stream(stream):
            dist.all_to_all_single(recv_buf, send_buf, group=group)
    else:
        dist.all_to_all_single(recv_buf, send_buf, group=group)


def run_kernel(ctx: IPCContext, layout: WeightLayout, kernel: str, direction: str,
               method: str, variant: str, num_layers: int, warmup: int, iters: int) -> dict:
    if kernel == "w13":
        off = w13_offsets(layout)
    else:
        off = w2_offsets(layout)

    if method == "peer_access":
        if kernel == "w13" and direction == "ep_to_tp":
            launch = lambda: _peer_access_w13_ep_to_tp(ctx, layout, off, variant)
        elif kernel == "w13" and direction == "tp_to_ep":
            launch = lambda: _peer_access_w13_tp_to_ep(ctx, layout, off, variant)
        elif kernel == "w2" and direction == "ep_to_tp":
            launch = lambda: _peer_access_w2_ep_to_tp(ctx, layout, off, variant)
        elif kernel == "w2" and direction == "tp_to_ep":
            launch = lambda: _peer_access_w2_tp_to_ep(ctx, layout, off, variant)
        else:
            raise SystemExit(f"bad kernel/direction: {kernel}/{direction}")
    else:
        if kernel == "w13":
            send_buf, recv_buf = _nccl_w13_send_recv(layout, ctx.device)
        else:
            send_buf, recv_buf = _nccl_w2_send_recv(layout, ctx.device)
        def launch_nccl(stream=None):
            _nccl_layer(send_buf, recv_buf, ctx.tp_group, stream)

    timer = CudaTimer(ctx.device, warmup=warmup, iters=iters)
    for _ in range(timer.total_iters):
        ctx.barrier()
        timer.tick()
        if method == "peer_access":
            for _ in range(num_layers):
                launch()
                ctx.barrier()
        elif method == "nccl":
            for _ in range(num_layers):
                launch_nccl()
        elif method == "nccl_overlap":
            s1 = torch.cuda.Stream()
            s2 = torch.cuda.Stream()
            for i in range(num_layers):
                launch_nccl(s1 if i % 2 == 0 else s2)
            torch.cuda.current_stream().wait_stream(s1)
            torch.cuda.current_stream().wait_stream(s2)
        else:
            raise SystemExit(f"Unknown method: {method}")
        timer.tock()
        ctx.barrier()

    s = timer.summary()
    s["per_layer_mean_ms"] = s["mean"] / num_layers
    s["per_layer_p50_ms"] = s["p50"] / num_layers
    return s


def main():
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--kernel", choices=("w13", "w2", "both"), default="both")
    parser.add_argument("--direction", choices=("ep_to_tp", "tp_to_ep", "both"),
                        default="both")
    parser.add_argument("--method", choices=("peer_access", "nccl", "nccl_overlap"),
                        default="peer_access")
    parser.add_argument("--variant", choices=("v2", "v3"), default="v2",
                        help="Peer-access kernel variant: v2=production, v3=contig-tile")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--out-csv", type=str, default=None)
    args = parser.parse_args()

    model = resolve_model(args)
    layout = make_weight_layout(model, args.tp_size)
    num_layers = model.num_hidden_layers

    needed_w13 = w13_offsets(layout)["total"] if args.kernel in ("w13", "both") else 0
    needed_w2 = w2_offsets(layout)["total"] if args.kernel in ("w2", "both") else 0
    needed_nccl_w13 = (2 * layout.tp_size * layout.E_local * layout.num_gates
                       * layout.I_prime_H * layout.elem_size
                       if args.kernel in ("w13", "both") and args.method != "peer_access" else 0)
    needed_nccl_w2 = (2 * layout.tp_size * layout.E_local * layout.H
                      * layout.I_prime * layout.elem_size
                      if args.kernel in ("w2", "both") and args.method != "peer_access" else 0)
    arena = max(needed_w13, needed_w2, 1)
    ctx = setup_ipc_arena(arena)
    if ctx.world_size != args.tp_size:
        raise SystemExit(f"torchrun world_size ({ctx.world_size}) != --tp-size ({args.tp_size})")

    method_str = args.method if args.method != "peer_access" else f"{args.method}({args.variant})"
    if ctx.rank == 0:
        print(f"[rank 0] model={model.name} layers={num_layers} tp={args.tp_size} "
              f"E_local={layout.E_local} H={layout.H} I'={layout.I_prime} "
              f"num_gates={layout.num_gates} method={method_str}")
        print(f"[rank 0] peer-access arena={arena/(1024**3):.2f}GiB")

    kernels = ("w13", "w2") if args.kernel == "both" else (args.kernel,)
    directions = ("ep_to_tp", "tp_to_ep") if args.direction == "both" else (args.direction,)

    results = []
    for k in kernels:
        for d in directions:
            if ctx.rank == 0:
                print(f"[rank 0] RUN {k} {d} ({method_str}) × {num_layers} layers")
            stats = run_kernel(ctx, layout, k, d, args.method, args.variant, num_layers,
                               args.warmup, args.iters)
            results.append({"kernel": k, "direction": d, **stats})
            if ctx.rank == 0:
                print(f"[rank 0] {k} {d}: total_mean={stats['mean']:.3f}ms "
                      f"per_layer_mean={stats['per_layer_mean_ms']:.4f}ms "
                      f"p50_per_layer={stats['per_layer_p50_ms']:.4f}ms")

    if ctx.rank == 0 and args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        new_file = not os.path.exists(args.out_csv)
        with open(args.out_csv, "a", newline="") as f:
            fields = ["timestamp", "model", "num_layers", "tp_size",
                      "E_local", "H", "I_prime", "num_gates",
                      "kernel", "direction", "method", "variant",
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
                    "E_local": layout.E_local,
                    "H": layout.H,
                    "I_prime": layout.I_prime,
                    "num_gates": layout.num_gates,
                    "kernel": r["kernel"],
                    "direction": r["direction"],
                    "method": args.method,
                    "variant": args.variant if args.method == "peer_access" else "",
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
