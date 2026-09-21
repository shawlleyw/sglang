"""UMM weight-bundle microbenchmark (BF16 experts + QKV/O, both directions).

Every iteration visits distinct planned layer views. Restoring the opposite
mode happens outside timing. This excludes serving/cache/graph orchestration.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import replace

import torch
import torch.distributed as dist

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from common.ipc import CudaTimer, init_torchrun, setup_ipc_arena
from common.layouts import make_weight_layout
from common.model_configs import add_model_args, resolve_model
from common.weight_bundle import (
    COMPONENTS,
    ParaSMode,
    attention_restore,
    attention_slice,
    expert_ep_packed_view,
    make_manager,
    reference_values,
    verification_indices,
    weight_name,
)


def _peer_access_w13_ep_to_tp(ctx, layout, off, variant):
    if variant == "v3":
        ppa3.launch_peer_access_fused_transfer_w13_v3(
            ctx.local_buffer_ptr,
            ctx.peer_buffer_ptrs,
            off["ep"],
            off["tp"],
            ctx.rank,
            ctx.world_size,
            layout.E_local,
            layout.H,
            layout.I_full,
            layout.num_gates,
            layout.elem_size,
            0,
        )
    else:
        ppa.launch_peer_access_fused_transfer_w13_v2(
            ctx.local_buffer_ptr,
            ctx.peer_buffer_ptrs,
            off["ep"],
            off["tp"],
            ctx.rank,
            ctx.world_size,
            layout.E_local,
            layout.I_prime_H,
            layout.num_gates,
            layout.elem_size,
            0,
        )


def _peer_access_w13_tp_to_ep(ctx, layout, off, variant):
    if variant == "v3":
        ppa3.launch_peer_access_fused_transfer_w13_v3_ep(
            ctx.local_buffer_ptr,
            ctx.peer_buffer_ptrs,
            off["tp"],
            off["ep"],
            ctx.rank,
            ctx.world_size,
            layout.E_local,
            layout.H,
            layout.I_full,
            layout.num_gates,
            layout.elem_size,
            0,
        )
    else:
        ppa.launch_peer_access_fused_transfer_w13_ep(
            ctx.local_buffer_ptr,
            ctx.peer_buffer_ptrs,
            off["tp"],
            off["ep"],
            ctx.rank,
            ctx.world_size,
            layout.E_local,
            layout.I_prime_H,
            layout.num_gates,
            layout.elem_size,
            0,
        )


def _peer_access_w2_ep_to_tp(ctx, layout, off, variant):
    if variant == "v3":
        ppa3.launch_peer_access_fused_transfer_w2_v3(
            ctx.local_buffer_ptr,
            ctx.peer_buffer_ptrs,
            off["ep"],
            off["tp"],
            ctx.rank,
            ctx.world_size,
            layout.E_local,
            layout.H,
            layout.I_full,
            layout.elem_size,
            0,
        )
    else:
        ppa.launch_peer_access_fused_transfer_w2_v2(
            ctx.local_buffer_ptr,
            ctx.peer_buffer_ptrs,
            off["ep"],
            off["tp"],
            ctx.rank,
            ctx.world_size,
            layout.E_local,
            layout.H,
            layout.I_full * layout.elem_size,
            layout.I_prime * layout.elem_size,
            0,
        )


def _peer_access_w2_tp_to_ep(ctx, layout, off, variant):
    if variant == "v3":
        ppa3.launch_peer_access_fused_transfer_w2_v3_ep(
            ctx.local_buffer_ptr,
            ctx.peer_buffer_ptrs,
            off["tp"],
            off["ep"],
            ctx.rank,
            ctx.world_size,
            layout.E_local,
            layout.H,
            layout.I_full,
            layout.elem_size,
            0,
        )
    else:
        ppa.launch_peer_access_fused_transfer_w2_ep(
            ctx.local_buffer_ptr,
            ctx.peer_buffer_ptrs,
            off["tp"],
            off["ep"],
            ctx.rank,
            ctx.world_size,
            layout.E_local,
            layout.H,
            layout.I_full * layout.elem_size,
            layout.I_prime * layout.elem_size,
            0,
        )


class WeightBench:
    def __init__(self, ctx, manager, model, components, method, variant):
        self.ctx, self.manager, self.model = ctx, manager, model
        self.components, self.method, self.variant = components, method, variant
        self.layout = make_weight_layout(model, ctx.world_size)
        self.buffers = {}
        self.pack_stream = torch.cuda.Stream() if method == "nccl_overlap" else None
        # Each component owns its staging: no concurrent writes to shared storage.
        if method != "peer_access":
            for component in components:
                ep, tp = self.views(0, component)
                if component in ("w13", "w2"):
                    packed = expert_ep_packed_view(ep, model, ctx.world_size, component)
                    self.buffers[component] = (
                        torch.empty_like(packed, memory_format=torch.contiguous_format),
                        torch.empty_like(packed, memory_format=torch.contiguous_format),
                    )
                else:
                    self.buffers[component] = (
                        None,
                        torch.empty(
                            (ctx.world_size, *tp.shape),
                            device=ctx.device,
                            dtype=tp.dtype,
                        ),
                    )

    def views(self, layer, component):
        return tuple(
            self.manager.get_view(weight_name(layer, mode, component))
            for mode in (ParaSMode.EP, ParaSMode.TP)
        )

    @property
    def scratch_bytes(self):
        return sum(
            t.numel() * t.element_size()
            for pair in self.buffers.values()
            for t in pair
            if t is not None
        )

    def direct(self, layer, component, direction):
        ep, tp = self.views(layer, component)
        if component in ("w13", "w2"):
            off = {
                "ep": ep.data_ptr() - self.ctx.local_buffer_ptr,
                "tp": tp.data_ptr() - self.ctx.local_buffer_ptr,
            }
            layout = self.layout
            if component == "w13" and self.model.interleaved_w13:
                layout = replace(layout, num_gates=1, I_prime=2 * layout.I_prime)
            globals()[f"_peer_access_{component}_{direction}"](
                self.ctx, layout, off, self.variant
            )
        elif component == "qkv":
            from sglang.srt.paras.attention_transfer import transfer_attention

            # Attention selections always contain QKV and O together. Invoke
            # the production wrapper once; the following O entry is a no-op.
            mode = ParaSMode.TP if direction == "ep_to_tp" else ParaSMode.EP
            transfer_attention(
                self.manager, layer, mode, self.ctx.rank, self.ctx.peer_buffer_ptrs
            )

    def pack(self, layer, component):
        ep, _ = self.views(layer, component)
        send, _ = self.buffers[component]
        send.copy_(
            expert_ep_packed_view(ep, self.model, self.ctx.world_size, component)
        )

    def collective(self, layer, component, direction):
        ep, tp = self.views(layer, component)
        send, recv = self.buffers[component]
        if component in ("w13", "w2"):
            if direction == "ep_to_tp":
                # Receive rank-major expert blocks directly in the target UMM view.
                dist.all_to_all_single(
                    tp.view(-1), send.view(-1), group=self.ctx.tp_group
                )
            else:
                dist.all_to_all_single(
                    recv.view(-1), tp.view(-1), group=self.ctx.tp_group
                )
                expert_ep_packed_view(
                    ep, self.model, self.ctx.world_size, component
                ).copy_(recv)
        elif direction == "ep_to_tp":
            attention_slice(
                ep, tp, self.model, self.ctx.world_size, self.ctx.rank, component
            )
        else:
            dist.all_gather_into_tensor(
                recv.view(-1), tp.view(-1), group=self.ctx.tp_group
            )
            attention_restore(recv, ep, self.model, self.ctx.world_size, component)

    def run(self, direction):
        layers = range(self.model.num_hidden_layers)
        if direction == "tp_to_ep":
            layers = reversed(layers)
        for layer in layers:
            if self.pack_stream is not None and direction == "ep_to_tp":
                # Only overlap independent components of this layer. The UMM may
                # alias next-layer destinations with unfinished source reads.
                self.pack_stream.wait_stream(torch.cuda.current_stream())
                ready = {}
                with torch.cuda.stream(self.pack_stream):
                    for component in self.components:
                        if component in ("w13", "w2"):
                            self.pack(layer, component)
                            ready[component] = torch.cuda.Event()
                            ready[component].record()
            for component in self.components:
                if self.method == "peer_access":
                    self.direct(layer, component, direction)
                else:
                    if direction == "ep_to_tp" and component in ("w13", "w2"):
                        if self.pack_stream is not None:
                            torch.cuda.current_stream().wait_event(ready[component])
                        else:
                            self.pack(layer, component)
                    self.collective(layer, component, direction)
            self.ctx.barrier()  # Complete expert+attention bundle before reusing bytes.

    def initialize(self, mode):
        chunk = 1 << 20
        for layer in range(self.model.num_hidden_layers):
            for component in self.components:
                view = self.manager.get_view(weight_name(layer, mode, component)).view(
                    -1
                )
                for start in range(0, view.numel(), chunk):
                    indices = torch.arange(
                        start, min(start + chunk, view.numel()), device=self.ctx.device
                    )
                    view[start : start + chunk].copy_(
                        reference_values(
                            indices,
                            self.model,
                            self.ctx.world_size,
                            self.ctx.rank,
                            mode,
                            component,
                            layer,
                        )
                    )
        self.ctx.barrier()

    def verify(self, mode):
        # Stratified + contiguous samples cover boundaries, gates, and interiors;
        # the CPU tests separately check entire small tensors and round trips.
        failure = torch.zeros(1, device=self.ctx.device, dtype=torch.int32)
        for layer in range(self.model.num_hidden_layers):
            for component in self.components:
                view = self.manager.get_view(weight_name(layer, mode, component)).view(
                    -1
                )
                indices = verification_indices(view.numel(), self.ctx.device)
                expected = reference_values(
                    indices,
                    self.model,
                    self.ctx.world_size,
                    self.ctx.rank,
                    mode,
                    component,
                    layer,
                )
                failure.copy_(
                    torch.maximum(
                        failure,
                        (view[indices] != expected).any().to(torch.int32).reshape(1),
                    )
                )
        dist.all_reduce(failure, op=dist.ReduceOp.MAX, group=self.ctx.tp_group)
        if failure.item():
            raise RuntimeError(
                f"Weight verification failed for {self.method}, {mode.value}"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_args(parser)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument(
        "--kernel",
        choices=("w13", "w2", "both", "experts", "attention", "bundle", "all"),
        default="bundle",
        help="both=separate experts; experts=w13+w2 together; bundle=experts+QKV/O; all=components and bundles",
    )
    parser.add_argument(
        "--direction", choices=("ep_to_tp", "tp_to_ep", "both"), default="both"
    )
    parser.add_argument(
        "--method",
        choices=("peer_access", "nccl", "nccl_overlap"),
        default="peer_access",
    )
    parser.add_argument("--variant", choices=("v2", "v3"), default="v2")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--out-csv")
    args = parser.parse_args()
    model = resolve_model(args)
    if (
        args.method == "peer_access"
        and args.variant == "v3"
        and args.kernel != "attention"
        and (
            args.tp_size not in (4, 8)
            or model.interleaved_w13
            or (model.hidden_size, model.moe_intermediate_size)
            not in ((4096, 1536), (2048, 768))
        )
    ):
        parser.error(
            "Current v3 expert kernels only specialize Qwen shapes "
            "(H,I)=(4096,1536)/(2048,768), separate gates, TP=4/8; "
            "use --variant v2 for GPT-OSS or other shapes"
        )
    if args.method == "peer_access":
        global ppa, ppa3
        import paras_peer_access_cuda as ppa

        ppa3 = ppa
    if args.iters < 1 or args.warmup < 0:
        parser.error("--iters must be positive and --warmup nonnegative")
    rank, world = init_torchrun()
    if world != args.tp_size:
        parser.error("torchrun world_size must equal --tp-size")
    manager, plan = make_manager(model, world, "cuda")
    ctx = setup_ipc_arena(
        plan.layout.budget,
        buffer=manager._buffer,
        peer_access=args.method == "peer_access",
    )
    selections = {
        "w13": ("w13",),
        "w2": ("w2",),
        "experts": ("w13", "w2"),
        "attention": ("qkv", "o"),
        "bundle": COMPONENTS,
    }
    names = (
        ("w13", "w2")
        if args.kernel == "both"
        else tuple(selections) if args.kernel == "all" else (args.kernel,)
    )
    directions = (
        ("ep_to_tp", "tp_to_ep") if args.direction == "both" else (args.direction,)
    )
    results = []
    for name in names:
        bench = WeightBench(
            ctx, manager, model, selections[name], args.method, args.variant
        )
        if rank == 0:
            print(
                f"RUN {name}: model={model.name} layers={model.num_hidden_layers} UMM={plan.layout.budget / 2**30:.3f}GiB staging={bench.scratch_bytes / 2**30:.3f}GiB",
                flush=True,
            )
        for direction in directions:
            source = ParaSMode.EP if direction == "ep_to_tp" else ParaSMode.TP
            target = ParaSMode.TP if source == ParaSMode.EP else ParaSMode.EP
            reverse = "tp_to_ep" if direction == "ep_to_tp" else "ep_to_tp"
            bench.initialize(source)
            bench.run(direction)
            bench.verify(target)
            bench.run(reverse)
            bench.verify(source)
            timer = CudaTimer(ctx.device, args.warmup, args.iters)
            for _ in range(timer.total_iters):
                ctx.barrier()
                timer.tick()
                bench.run(direction)
                timer.tock()
                # Source/destination overlap across layers: restore before reuse.
                bench.run(reverse)
            summary = timer.summary()
            bench.verify(source)
            stats = dict(
                total_mean_ms=summary["mean"],
                total_p50_ms=summary["p50"],
                min_ms=summary["min"],
                max_ms=summary["max"],
                n=summary["n"],
            )
            row = dict(
                timestamp=int(time.time()),
                model=model.name,
                num_layers=model.num_hidden_layers,
                tp_size=world,
                kernel=name,
                direction=direction,
                method=args.method,
                variant=args.variant if args.method == "peer_access" else "",
                interleaved_w13=model.interleaved_w13,
                arena_bytes=plan.layout.budget,
                staging_bytes=bench.scratch_bytes,
                expert_payload_bytes_per_rank=sum(
                    bench.views(0, component)[0].numel() * model.elem_size
                    for component in selections[name]
                    if component in ("w13", "w2")
                )
                * model.num_hidden_layers,
                expert_remote_bytes_per_rank=sum(
                    bench.views(0, component)[0].numel() * model.elem_size
                    for component in selections[name]
                    if component in ("w13", "w2")
                )
                * model.num_hidden_layers
                * (world - 1)
                // world,
                overlap_scope=(
                    "within_layer_pack_collective"
                    if args.method == "nccl_overlap" and direction == "ep_to_tp"
                    else "none"
                ),
                **stats,
            )
            row["per_layer_mean_ms"] = stats["total_mean_ms"] / model.num_hidden_layers
            row["per_layer_p50_ms"] = stats["total_p50_ms"] / model.num_hidden_layers
            results.append(row)
            if rank == 0:
                print(
                    f"{name} {direction}: total_mean={stats['total_mean_ms']:.3f}ms per_layer={row['per_layer_mean_ms']:.4f}ms verified",
                    flush=True,
                )
        del bench
    if rank == 0 and args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        fields = list(results[0])
        exists = os.path.exists(args.out_csv) and os.path.getsize(args.out_csv) > 0
        if exists:
            with open(args.out_csv, newline="") as f:
                if next(csv.reader(f)) != fields:
                    raise ValueError("CSV schema differs; use a fresh --out-csv file")
        with open(args.out_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerows(results)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
