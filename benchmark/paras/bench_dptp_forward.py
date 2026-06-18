"""Multi-layer benchmark: dptp peer-access kernel vs NCCL DP x TP baseline.

Amortizes one world all_reduce barrier (peer_access cross-replica sync) over
N_LAYERS back-to-back transfers per timed iteration to extract per-layer cost
and NVLink utilization without barrier contamination.

Config: Qwen3-235B preset (E=128, H=4096, I=1536, num_gates=2, bf16),
grid (T=4, G=2), world W = G*T = 8 ranks.

Subgroups built locally (every rank participates in every dist.new_group):
  TP subgroups (T=4 contiguous):  {0,1,2,3}, {4,5,6,7}
  DP subgroups (G=2, stride T=4): {0,4}, {1,5}, {2,6}, {3,7}

Per timed iter:
  peer_access: tick -> N_LAYERS * (launch_w13 + launch_w2) -> ONE world barrier -> tock
  NCCL:        tick -> N_LAYERS * (full DPxTP path) -> tock  (collectives self-sync)
Per-layer time = elapsed / N_LAYERS.

NVLink util: A100 SXM4 single-direction = 300 GB/s/GPU. Per-rank send factor
((W-1)/T) accounts for self-write staying local: for W=8 T=4 -> 1.75x EP source.

Correctness gate runs once on layer 0 before timing (both methods must equal
the canonical TP slice for tp_rank = rank % T, byte-exact via torch.equal).

Usage:
    torchrun --nproc_per_node=8 bench_dptp_forward.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from common.ipc import CudaTimer, IPCContext, setup_ipc_arena

import paras_peer_access_cuda as ppa


E = 128
H = 4096
I = 1536
NUM_GATES = 2
ELEM = 2
W = 8
T = 4
G = 2
N_LAYERS = 16
WARMUP = 10
ITERS = 50
NVLINK_BW_GB_S = 300.0


def _bf16_pattern(indices: torch.Tensor) -> torch.Tensor:
    return (indices & 255).to(torch.bfloat16)


def _fill_ep_w13(buf_u8: torch.Tensor, ep_off: int, rank: int, E_ep: int) -> None:
    I_prime = I // T
    I_prime_H = I_prime * H
    numel = E_ep * NUM_GATES * T * I_prime_H
    bytes_ = numel * ELEM
    start = rank * numel
    idx = torch.arange(numel, dtype=torch.int32, device=buf_u8.device) + start
    pattern = _bf16_pattern(idx).view(E_ep, NUM_GATES, T, I_prime_H)
    view = buf_u8[ep_off:ep_off + bytes_].view(torch.bfloat16).view(
        E_ep, NUM_GATES, T, I_prime_H
    )
    view.copy_(pattern)


def _fill_ep_w2(buf_u8: torch.Tensor, ep_off: int, rank: int, E_ep: int) -> None:
    numel = E_ep * H * I
    bytes_ = numel * ELEM
    start = rank * numel
    idx = torch.arange(numel, dtype=torch.int32, device=buf_u8.device) + start
    pattern = _bf16_pattern(idx).view(E_ep, H, I)
    view = buf_u8[ep_off:ep_off + bytes_].view(torch.bfloat16).view(E_ep, H, I)
    view.copy_(pattern)


def _expected_w13_tp(device: torch.device, tp_rank: int) -> torch.Tensor:
    I_prime = I // T
    I_prime_H = I_prime * H
    e_idx = torch.arange(E, dtype=torch.int32, device=device).view(E, 1, 1)
    k_idx = torch.arange(NUM_GATES, dtype=torch.int32, device=device).view(1, NUM_GATES, 1)
    i_idx = torch.arange(I_prime_H, dtype=torch.int32, device=device).view(1, 1, I_prime_H)
    linear = (
        e_idx * (NUM_GATES * T * I_prime_H)
        + k_idx * (T * I_prime_H)
        + tp_rank * I_prime_H
        + i_idx
    )
    return _bf16_pattern(linear)


def _expected_w2_tp(device: torch.device, tp_rank: int) -> torch.Tensor:
    I_prime = I // T
    e_idx = torch.arange(E, dtype=torch.int32, device=device).view(E, 1, 1)
    h_idx = torch.arange(H, dtype=torch.int32, device=device).view(1, H, 1)
    j_idx = torch.arange(I_prime, dtype=torch.int32, device=device).view(1, 1, I_prime)
    linear = e_idx * (H * I) + h_idx * I + tp_rank * I_prime + j_idx
    return _bf16_pattern(linear)


def _build_subgroups(rank: int) -> tuple[dist.ProcessGroup, dist.ProcessGroup]:
    tp_group: dist.ProcessGroup | None = None
    dp_group: dist.ProcessGroup | None = None
    for g in range(G):
        members = [g * T + t for t in range(T)]
        grp = dist.new_group(ranks=members)
        if rank in members:
            tp_group = grp
    for t in range(T):
        members = [d * T + t for d in range(G)]
        grp = dist.new_group(ranks=members)
        if rank in members:
            dp_group = grp
    assert tp_group is not None and dp_group is not None
    return tp_group, dp_group


class LayerOffsets:
    """Per-layer byte offsets into the IPC arena (16 distinct (EP, TP) slots)."""

    def __init__(self, E_ep: int):
        I_prime = I // T
        self.w13_ep_bytes = E_ep * NUM_GATES * T * I_prime * H * ELEM
        self.w13_tp_bytes = E * NUM_GATES * I_prime * H * ELEM
        self.w2_ep_bytes = E_ep * H * I * ELEM
        self.w2_tp_bytes = E * H * I_prime * ELEM
        self.per_layer_bytes = (
            self.w13_ep_bytes + self.w13_tp_bytes + self.w2_ep_bytes + self.w2_tp_bytes
        )
        self.w13_ep_off: list[int] = []
        self.w13_tp_off: list[int] = []
        self.w2_ep_off: list[int] = []
        self.w2_tp_off: list[int] = []
        for l in range(N_LAYERS):
            base = l * self.per_layer_bytes
            self.w13_ep_off.append(base)
            self.w13_tp_off.append(base + self.w13_ep_bytes)
            self.w2_ep_off.append(base + self.w13_ep_bytes + self.w13_tp_bytes)
            self.w2_tp_off.append(
                base + self.w13_ep_bytes + self.w13_tp_bytes + self.w2_ep_bytes
            )
        self.total_bytes = N_LAYERS * self.per_layer_bytes


def _peer_access_dptp_w13(
    ctx: IPCContext, ep_off: int, tp_off: int, E_ep: int
) -> None:
    stream_ptr = torch.cuda.current_stream(ctx.device).cuda_stream
    ppa.launch_peer_access_fused_transfer_w13_dptp(
        ctx.local_buffer_ptr, ctx.peer_buffer_ptrs,
        ep_off, tp_off,
        ctx.rank, T, G, E_ep,
        H, I, NUM_GATES, ELEM,
        stream_ptr,
    )


def _peer_access_dptp_w2(
    ctx: IPCContext, ep_off: int, tp_off: int, E_ep: int
) -> None:
    stream_ptr = torch.cuda.current_stream(ctx.device).cuda_stream
    ppa.launch_peer_access_fused_transfer_w2_dptp(
        ctx.local_buffer_ptr, ctx.peer_buffer_ptrs,
        ep_off, tp_off,
        ctx.rank, T, G, E_ep,
        H, I, ELEM,
        stream_ptr,
    )


def _peer_access_dptp_ep_w13(
    ctx: IPCContext, tp_off: int, ep_off: int, E_ep: int
) -> None:
    stream_ptr = torch.cuda.current_stream(ctx.device).cuda_stream
    ppa.launch_peer_access_fused_transfer_w13_ep_dptp(
        ctx.local_buffer_ptr, ctx.peer_buffer_ptrs,
        tp_off, ep_off,
        ctx.rank, T, G, E_ep,
        H, I, NUM_GATES, ELEM,
        stream_ptr,
    )


def _peer_access_dptp_ep_w2(
    ctx: IPCContext, tp_off: int, ep_off: int, E_ep: int
) -> None:
    stream_ptr = torch.cuda.current_stream(ctx.device).cuda_stream
    ppa.launch_peer_access_fused_transfer_w2_ep_dptp(
        ctx.local_buffer_ptr, ctx.peer_buffer_ptrs,
        tp_off, ep_off,
        ctx.rank, T, G, E_ep,
        H, I, ELEM,
        stream_ptr,
    )


class NcclState:
    """Pre-allocated NCCL DP x TP staging buffers + per-layer TP destinations + reverse EP scratch."""

    def __init__(self, device: torch.device, E_ep: int):
        I_prime = I // T
        E_local = E_ep * G
        self.E_ep = E_ep
        self.E_local = E_local
        self.I_prime = I_prime

        self.w13_gathered = torch.empty(E_local, NUM_GATES * I, H, dtype=torch.bfloat16, device=device)
        self.w13_pre_perm = torch.empty_like(self.w13_gathered)
        self.w2_gathered = torch.empty(E_local, H, I, dtype=torch.bfloat16, device=device)
        self.w2_pre_perm = torch.empty_like(self.w2_gathered)

        self.tp_w13_layers = [
            torch.empty(E, NUM_GATES * I_prime, H, dtype=torch.bfloat16, device=device)
            for _ in range(N_LAYERS)
        ]
        self.tp_w2_layers = [
            torch.empty(E, H, I_prime, dtype=torch.bfloat16, device=device)
            for _ in range(N_LAYERS)
        ]

        self.ep_w13_rev = torch.empty(E_ep, NUM_GATES, T, I_prime * H, dtype=torch.bfloat16, device=device)
        self.ep_w2_rev = torch.empty(E_ep, H, T * I_prime, dtype=torch.bfloat16, device=device)

    def staging_bytes(self) -> int:
        return (
            self.w13_gathered.numel() * 2
            + self.w13_pre_perm.numel() * 2
            + self.w2_gathered.numel() * 2
            + self.w2_pre_perm.numel() * 2
        )

    def tp_dest_bytes(self) -> int:
        return N_LAYERS * (
            self.tp_w13_layers[0].numel() * 2 + self.tp_w2_layers[0].numel() * 2
        )


def _nccl_dptp_w13_layer(
    w13_ep: torch.Tensor, st: NcclState, tp_w13: torch.Tensor,
    dp_group: dist.ProcessGroup, tp_group: dist.ProcessGroup,
) -> None:
    E_local = st.E_local
    I_prime = st.I_prime
    I_prime_H = I_prime * H

    h_g = dist.all_gather_into_tensor(
        st.w13_gathered, w13_ep, group=dp_group, async_op=True,
    )
    h_g.wait()
    w13_ep_view = st.w13_gathered.view(E_local, NUM_GATES, T, I_prime_H)
    w13_pre = st.w13_pre_perm.view(T, E_local, NUM_GATES, I_prime_H)
    w13_pre.copy_(w13_ep_view.permute(2, 0, 1, 3))
    h_a = dist.all_to_all_single(
        output=st.w13_gathered,
        input=st.w13_pre_perm.view(st.w13_gathered.shape),
        group=tp_group,
        async_op=True,
    )
    h_a.wait()
    w13_post = st.w13_pre_perm.view(G, T, -1)
    w13_post.copy_(st.w13_gathered.view(T, G, -1).transpose(0, 1))
    tp_w13.copy_(w13_post.view(E, NUM_GATES * I_prime, H))


def _nccl_dptp_w2_layer(
    w2_ep: torch.Tensor, st: NcclState, tp_w2: torch.Tensor,
    dp_group: dist.ProcessGroup, tp_group: dist.ProcessGroup,
) -> None:
    E_local = st.E_local
    I_prime = st.I_prime

    h_g = dist.all_gather_into_tensor(
        st.w2_gathered, w2_ep, group=dp_group, async_op=True,
    )
    h_g.wait()
    w2_ep_view = st.w2_gathered.view(E_local, H, T, I_prime)
    w2_pre = st.w2_pre_perm.view(T, E_local, H, I_prime)
    w2_pre.copy_(w2_ep_view.permute(2, 0, 1, 3))
    h_a = dist.all_to_all_single(
        output=st.w2_gathered,
        input=st.w2_pre_perm.view(st.w2_gathered.shape),
        group=tp_group,
        async_op=True,
    )
    h_a.wait()
    w2_post = st.w2_pre_perm.view(G, T, -1)
    w2_post.copy_(st.w2_gathered.view(T, G, -1).transpose(0, 1))
    tp_w2.copy_(w2_post.view(E, H, I_prime))


def _nccl_dptp_total_layer(
    w13_ep: torch.Tensor, w2_ep: torch.Tensor, st: NcclState,
    tp_w13: torch.Tensor, tp_w2: torch.Tensor,
    dp_group: dist.ProcessGroup, tp_group: dist.ProcessGroup,
) -> None:
    E_local = st.E_local
    I_prime = st.I_prime
    I_prime_H = I_prime * H

    h_g13 = dist.all_gather_into_tensor(
        st.w13_gathered, w13_ep, group=dp_group, async_op=True,
    )
    h_g2 = dist.all_gather_into_tensor(
        st.w2_gathered, w2_ep, group=dp_group, async_op=True,
    )

    h_g13.wait()
    w13_ep_view = st.w13_gathered.view(E_local, NUM_GATES, T, I_prime_H)
    w13_pre = st.w13_pre_perm.view(T, E_local, NUM_GATES, I_prime_H)
    w13_pre.copy_(w13_ep_view.permute(2, 0, 1, 3))
    h_a13 = dist.all_to_all_single(
        output=st.w13_gathered,
        input=st.w13_pre_perm.view(st.w13_gathered.shape),
        group=tp_group,
        async_op=True,
    )

    h_g2.wait()
    w2_ep_view = st.w2_gathered.view(E_local, H, T, I_prime)
    w2_pre = st.w2_pre_perm.view(T, E_local, H, I_prime)
    w2_pre.copy_(w2_ep_view.permute(2, 0, 1, 3))
    h_a2 = dist.all_to_all_single(
        output=st.w2_gathered,
        input=st.w2_pre_perm.view(st.w2_gathered.shape),
        group=tp_group,
        async_op=True,
    )

    h_a13.wait()
    w13_post = st.w13_pre_perm.view(G, T, -1)
    w13_post.copy_(st.w13_gathered.view(T, G, -1).transpose(0, 1))
    tp_w13.copy_(w13_post.view(E, NUM_GATES * I_prime, H))

    h_a2.wait()
    w2_post = st.w2_pre_perm.view(G, T, -1)
    w2_post.copy_(st.w2_gathered.view(T, G, -1).transpose(0, 1))
    tp_w2.copy_(w2_post.view(E, H, I_prime))


def _nccl_reverse_w13_layer(
    src_tp_w13: torch.Tensor, st: NcclState, ep_out: torch.Tensor,
    dp_rank: int, tp_group: dist.ProcessGroup,
) -> None:
    E_ep = st.E_ep
    I_prime = st.I_prime
    I_prime_H = I_prime * H
    a2a_numel = T * E_ep * NUM_GATES * I_prime_H
    a2a_scratch = st.w13_pre_perm.view(-1)[:a2a_numel].view(T, E_ep, NUM_GATES, I_prime_H)
    src_slice = src_tp_w13[dp_rank * T * E_ep:(dp_rank + 1) * T * E_ep].reshape(
        T, E_ep, NUM_GATES, I_prime_H
    )
    h = dist.all_to_all_single(
        output=a2a_scratch.reshape(-1),
        input=src_slice.contiguous().reshape(-1),
        group=tp_group,
        async_op=True,
    )
    h.wait()
    ep_out.view(E_ep, NUM_GATES, T, I_prime_H).copy_(a2a_scratch.permute(1, 2, 0, 3))


def _nccl_reverse_w2_layer(
    src_tp_w2: torch.Tensor, st: NcclState, ep_out: torch.Tensor,
    dp_rank: int, tp_group: dist.ProcessGroup,
) -> None:
    E_ep = st.E_ep
    I_prime = st.I_prime
    a2a_numel = T * E_ep * H * I_prime
    a2a_scratch = st.w2_pre_perm.view(-1)[:a2a_numel].view(T, E_ep, H, I_prime)
    src_slice = src_tp_w2[dp_rank * T * E_ep:(dp_rank + 1) * T * E_ep].reshape(
        T, E_ep, H, I_prime
    )
    h = dist.all_to_all_single(
        output=a2a_scratch.reshape(-1),
        input=src_slice.contiguous().reshape(-1),
        group=tp_group,
        async_op=True,
    )
    h.wait()
    ep_out.view(E_ep, H, T, I_prime).copy_(a2a_scratch.permute(1, 2, 0, 3))


def _nccl_reverse_total_layer(
    src_tp_w13: torch.Tensor, src_tp_w2: torch.Tensor, st: NcclState,
    ep_w13_out: torch.Tensor, ep_w2_out: torch.Tensor,
    dp_rank: int, tp_group: dist.ProcessGroup,
) -> None:
    _nccl_reverse_w13_layer(src_tp_w13, st, ep_w13_out, dp_rank, tp_group)
    _nccl_reverse_w2_layer(src_tp_w2, st, ep_w2_out, dp_rank, tp_group)


def _world_barrier(ctx: IPCContext) -> None:
    bar = torch.zeros(1, device=ctx.device)
    dist.all_reduce(bar)


def _fill_tp_w13(buf_u8: torch.Tensor, tp_off: int, tp_rank: int) -> None:
    I_prime = I // T
    I_prime_H = I_prime * H
    bytes_ = E * NUM_GATES * I_prime_H * ELEM
    e_idx = torch.arange(E, dtype=torch.int32, device=buf_u8.device).view(E, 1, 1)
    k_idx = torch.arange(NUM_GATES, dtype=torch.int32, device=buf_u8.device).view(1, NUM_GATES, 1)
    i_idx = torch.arange(I_prime_H, dtype=torch.int32, device=buf_u8.device).view(1, 1, I_prime_H)
    linear = (
        e_idx * (NUM_GATES * T * I_prime_H)
        + k_idx * (T * I_prime_H)
        + tp_rank * I_prime_H
        + i_idx
    )
    pattern = (linear & 255).to(torch.bfloat16)
    buf_u8[tp_off:tp_off + bytes_].view(torch.bfloat16).view(E, NUM_GATES, I_prime_H).copy_(pattern)


def _fill_tp_w2(buf_u8: torch.Tensor, tp_off: int, tp_rank: int) -> None:
    I_prime = I // T
    bytes_ = E * H * I_prime * ELEM
    e_idx = torch.arange(E, dtype=torch.int32, device=buf_u8.device).view(E, 1, 1)
    h_idx = torch.arange(H, dtype=torch.int32, device=buf_u8.device).view(1, H, 1)
    j_idx = torch.arange(I_prime, dtype=torch.int32, device=buf_u8.device).view(1, 1, I_prime)
    linear = e_idx * (H * I) + h_idx * I + tp_rank * I_prime + j_idx
    pattern = (linear & 255).to(torch.bfloat16)
    buf_u8[tp_off:tp_off + bytes_].view(torch.bfloat16).view(E, H, I_prime).copy_(pattern)


def _expected_ep_w13_rev(device: torch.device, rank: int, E_ep: int) -> torch.Tensor:
    I_prime = I // T
    I_prime_H = I_prime * H
    numel = E_ep * NUM_GATES * T * I_prime_H
    start = rank * numel
    idx = torch.arange(numel, dtype=torch.int32, device=device) + start
    return (idx & 255).to(torch.bfloat16).view(E_ep, NUM_GATES, T, I_prime_H)


def _expected_ep_w2_rev(device: torch.device, rank: int, E_ep: int) -> torch.Tensor:
    numel = E_ep * H * I
    start = rank * numel
    idx = torch.arange(numel, dtype=torch.int32, device=device) + start
    return (idx & 255).to(torch.bfloat16).view(E_ep, H, I)


def _reverse_correctness_gate(
    ctx: IPCContext, E_ep: int, lo: LayerOffsets, st: NcclState,
    tp_group: dist.ProcessGroup,
) -> None:
    rank = ctx.rank
    tp_rank = rank % T
    dp_rank = rank // T
    device = ctx.device
    I_prime = I // T

    ctx.buf[lo.w13_ep_off[0]:lo.w13_ep_off[0] + lo.w13_ep_bytes].zero_()
    ctx.buf[lo.w2_ep_off[0]:lo.w2_ep_off[0] + lo.w2_ep_bytes].zero_()
    _fill_tp_w13(ctx.buf, lo.w13_tp_off[0], tp_rank)
    _fill_tp_w2(ctx.buf, lo.w2_tp_off[0], tp_rank)
    _world_barrier(ctx)

    _peer_access_dptp_ep_w13(ctx, lo.w13_tp_off[0], lo.w13_ep_off[0], E_ep)
    _peer_access_dptp_ep_w2(ctx, lo.w2_tp_off[0], lo.w2_ep_off[0], E_ep)
    torch.cuda.synchronize(device)
    _world_barrier(ctx)

    dptp_w13 = ctx.buf[lo.w13_ep_off[0]:lo.w13_ep_off[0] + lo.w13_ep_bytes].view(
        torch.bfloat16
    ).view(E_ep, NUM_GATES, T, I_prime * H)
    dptp_w2 = ctx.buf[lo.w2_ep_off[0]:lo.w2_ep_off[0] + lo.w2_ep_bytes].view(
        torch.bfloat16
    ).view(E_ep, H, I)
    exp_w13 = _expected_ep_w13_rev(device, rank, E_ep)
    exp_w2 = _expected_ep_w2_rev(device, rank, E_ep)

    src_tp_w13 = ctx.buf[lo.w13_tp_off[0]:lo.w13_tp_off[0] + lo.w13_tp_bytes].view(
        torch.bfloat16
    ).view(E, NUM_GATES * I_prime, H)
    src_tp_w2 = ctx.buf[lo.w2_tp_off[0]:lo.w2_tp_off[0] + lo.w2_tp_bytes].view(
        torch.bfloat16
    ).view(E, H, I_prime)
    st.ep_w13_rev.zero_()
    st.ep_w2_rev.zero_()
    _world_barrier(ctx)
    _nccl_reverse_total_layer(src_tp_w13, src_tp_w2, st, st.ep_w13_rev, st.ep_w2_rev, dp_rank, tp_group)
    torch.cuda.synchronize(device)
    _world_barrier(ctx)

    failures = []
    if not torch.equal(dptp_w13, exp_w13):
        d = (dptp_w13.to(torch.int32) != exp_w13.to(torch.int32)).sum().item()
        failures.append(f"dptp_rev w13 != expected ({d})")
    if not torch.equal(st.ep_w13_rev, exp_w13):
        d = (st.ep_w13_rev.to(torch.int32) != exp_w13.to(torch.int32)).sum().item()
        failures.append(f"nccl_rev w13 != expected ({d})")
    if not torch.equal(dptp_w13, st.ep_w13_rev):
        d = (dptp_w13.to(torch.int32) != st.ep_w13_rev.to(torch.int32)).sum().item()
        failures.append(f"dptp_rev w13 != nccl_rev w13 ({d})")
    if not torch.equal(dptp_w2, exp_w2):
        d = (dptp_w2.to(torch.int32) != exp_w2.to(torch.int32)).sum().item()
        failures.append(f"dptp_rev w2 != expected ({d})")
    if not torch.equal(st.ep_w2_rev, exp_w2):
        d = (st.ep_w2_rev.to(torch.int32) != exp_w2.to(torch.int32)).sum().item()
        failures.append(f"nccl_rev w2 != expected ({d})")
    if not torch.equal(dptp_w2, st.ep_w2_rev):
        d = (dptp_w2.to(torch.int32) != st.ep_w2_rev.to(torch.int32)).sum().item()
        failures.append(f"dptp_rev w2 != nccl_rev w2 ({d})")

    local_fail = torch.tensor([len(failures)], device=device, dtype=torch.int32)
    dist.all_reduce(local_fail, op=dist.ReduceOp.MAX)
    if local_fail.item() > 0:
        if failures:
            print(f"[rank {rank}] REVERSE CORRECTNESS FAIL: {failures}")
        dist.barrier()
        if rank == 0:
            print("REVERSE CORRECTNESS GATE: FAIL  aborting reverse timing")
        dist.destroy_process_group()
        sys.exit(2)
    if rank == 0:
        print("REVERSE CORRECTNESS GATE: PASS  (dptp_rev == NCCL_rev == expected, byte-exact bf16)")


def _ep_view_w13(buf_u8: torch.Tensor, off: int, bytes_: int, E_ep: int) -> torch.Tensor:
    return buf_u8[off:off + bytes_].view(torch.bfloat16).view(E_ep, NUM_GATES * I, H)


def _ep_view_w2(buf_u8: torch.Tensor, off: int, bytes_: int, E_ep: int) -> torch.Tensor:
    return buf_u8[off:off + bytes_].view(torch.bfloat16).view(E_ep, H, I)


def _correctness_gate(
    ctx: IPCContext, E_ep: int, lo: LayerOffsets, st: NcclState,
    dp_group: dist.ProcessGroup, tp_group: dist.ProcessGroup,
) -> None:
    rank = ctx.rank
    tp_rank = rank % T
    device = ctx.device
    I_prime = I // T
    I_prime_H = I_prime * H

    ctx.buf[lo.w13_tp_off[0]:lo.w13_tp_off[0] + lo.w13_tp_bytes].zero_()
    ctx.buf[lo.w2_tp_off[0]:lo.w2_tp_off[0] + lo.w2_tp_bytes].zero_()
    ctx.barrier()
    _peer_access_dptp_w13(ctx, lo.w13_ep_off[0], lo.w13_tp_off[0], E_ep)
    _peer_access_dptp_w2(ctx, lo.w2_ep_off[0], lo.w2_tp_off[0], E_ep)
    torch.cuda.synchronize(device)
    _world_barrier(ctx)

    dptp_w13 = ctx.buf[lo.w13_tp_off[0]:lo.w13_tp_off[0] + lo.w13_tp_bytes].view(
        torch.bfloat16
    ).view(E, NUM_GATES, I_prime_H)
    dptp_w2 = ctx.buf[lo.w2_tp_off[0]:lo.w2_tp_off[0] + lo.w2_tp_bytes].view(
        torch.bfloat16
    ).view(E, H, I_prime)

    w13_ep_view = _ep_view_w13(ctx.buf, lo.w13_ep_off[0], lo.w13_ep_bytes, E_ep)
    w2_ep_view = _ep_view_w2(ctx.buf, lo.w2_ep_off[0], lo.w2_ep_bytes, E_ep)
    st.tp_w13_layers[0].zero_()
    st.tp_w2_layers[0].zero_()
    _world_barrier(ctx)
    _nccl_dptp_total_layer(
        w13_ep_view, w2_ep_view, st,
        st.tp_w13_layers[0], st.tp_w2_layers[0],
        dp_group, tp_group,
    )
    torch.cuda.synchronize(device)
    _world_barrier(ctx)

    expected_w13 = _expected_w13_tp(device, tp_rank)
    expected_w2 = _expected_w2_tp(device, tp_rank)
    nccl_w13 = st.tp_w13_layers[0].view(E, NUM_GATES, I_prime_H)
    nccl_w2 = st.tp_w2_layers[0]

    failures = []
    if not torch.equal(dptp_w13, expected_w13):
        d = (dptp_w13.to(torch.int32) != expected_w13.to(torch.int32)).sum().item()
        failures.append(f"dptp w13 != expected ({d} elements)")
    if not torch.equal(nccl_w13, expected_w13):
        d = (nccl_w13.to(torch.int32) != expected_w13.to(torch.int32)).sum().item()
        failures.append(f"nccl w13 != expected ({d} elements)")
    if not torch.equal(dptp_w13, nccl_w13):
        d = (dptp_w13.to(torch.int32) != nccl_w13.to(torch.int32)).sum().item()
        failures.append(f"dptp w13 != nccl w13 ({d} elements)")
    if not torch.equal(dptp_w2, expected_w2):
        d = (dptp_w2.to(torch.int32) != expected_w2.to(torch.int32)).sum().item()
        failures.append(f"dptp w2 != expected ({d} elements)")
    if not torch.equal(nccl_w2, expected_w2):
        d = (nccl_w2.to(torch.int32) != expected_w2.to(torch.int32)).sum().item()
        failures.append(f"nccl w2 != expected ({d} elements)")
    if not torch.equal(dptp_w2, nccl_w2):
        d = (dptp_w2.to(torch.int32) != nccl_w2.to(torch.int32)).sum().item()
        failures.append(f"dptp w2 != nccl w2 ({d} elements)")

    local_fail = torch.tensor([len(failures)], device=device, dtype=torch.int32)
    dist.all_reduce(local_fail, op=dist.ReduceOp.MAX)
    if local_fail.item() > 0:
        if failures:
            print(f"[rank {rank}] CORRECTNESS FAIL: {failures}")
        dist.barrier()
        if rank == 0:
            print("=" * 60)
            print("CORRECTNESS GATE: FAIL  aborting timing")
        dist.destroy_process_group()
        sys.exit(2)

    if rank == 0:
        print("CORRECTNESS GATE: PASS  (dptp == NCCL == expected, byte-exact bf16)")


def _time_multi_layer(ctx: IPCContext, op_seq, end_barrier: bool) -> dict:
    """op_seq() runs N_LAYERS layers back-to-back. Optional trailing world barrier."""
    timer = CudaTimer(ctx.device, warmup=WARMUP, iters=ITERS)
    for _ in range(timer.total_iters):
        ctx.barrier()
        timer.tick()
        op_seq()
        if end_barrier:
            _world_barrier(ctx)
        timer.tock()
        ctx.barrier()
    return timer.summary()


def main():
    E_ep = E // W
    I_prime = I // T

    lo = LayerOffsets(E_ep)
    arena_bytes = lo.total_bytes

    ctx = setup_ipc_arena(arena_bytes)
    if ctx.world_size != W:
        raise SystemExit(f"world_size {ctx.world_size} != expected {W}")

    rank = ctx.rank
    device = ctx.device

    tp_group, dp_group = _build_subgroups(rank)

    for l in range(N_LAYERS):
        _fill_ep_w13(ctx.buf, lo.w13_ep_off[l], rank, E_ep)
        _fill_ep_w2(ctx.buf, lo.w2_ep_off[l], rank, E_ep)
    ctx.barrier()

    st = NcclState(device, E_ep)

    ep_w13_per_rank_bytes = E_ep * NUM_GATES * I * H * ELEM
    ep_w2_per_rank_bytes = E_ep * H * I * ELEM
    nvlink_factor = (W - 1) / T
    w13_send_bytes = nvlink_factor * ep_w13_per_rank_bytes
    w2_send_bytes = nvlink_factor * ep_w2_per_rank_bytes
    total_send_bytes = w13_send_bytes + w2_send_bytes

    if rank == 0:
        print(f"[bench_dptp_forward] qwen3-235b E={E} H={H} I={I} num_gates={NUM_GATES} bf16")
        print(f"[bench_dptp_forward] grid T={T} G={G} W={W} E_ep={E_ep} I'={I_prime}  N_LAYERS={N_LAYERS}")
        print(f"[bench_dptp_forward] IPC arena       = {arena_bytes/(1024**3):.2f} GiB/rank")
        print(f"[bench_dptp_forward] NCCL TP dest    = {st.tp_dest_bytes()/(1024**3):.2f} GiB/rank")
        print(f"[bench_dptp_forward] NCCL staging    = {st.staging_bytes()/(1024**3):.2f} GiB/rank (reused)")
        print(f"[bench_dptp_forward] warmup={WARMUP} iters={ITERS}")
        print(f"[bench_dptp_forward] NVLink send factor (W-1)/T = {nvlink_factor:.3f}x EP")
        print(f"[bench_dptp_forward]   w13 send/GPU = {w13_send_bytes/(1024**2):.1f} MiB")
        print(f"[bench_dptp_forward]   w2  send/GPU = {w2_send_bytes/(1024**2):.1f} MiB")
        print(f"[bench_dptp_forward]   tot send/GPU = {total_send_bytes/(1024**2):.1f} MiB")
        print()

    _correctness_gate(ctx, E_ep, lo, st, dp_group, tp_group)
    dist.barrier()

    if rank == 0:
        print()
        print("=" * 70)
        print(f"PER-LAYER TIMING (ms; iter = {N_LAYERS} layers, warmup={WARMUP}, iters={ITERS})")
        print("=" * 70)

    def pa_w13_seq():
        for l in range(N_LAYERS):
            _peer_access_dptp_w13(ctx, lo.w13_ep_off[l], lo.w13_tp_off[l], E_ep)

    def pa_w2_seq():
        for l in range(N_LAYERS):
            _peer_access_dptp_w2(ctx, lo.w2_ep_off[l], lo.w2_tp_off[l], E_ep)

    def pa_total_seq():
        for l in range(N_LAYERS):
            _peer_access_dptp_w13(ctx, lo.w13_ep_off[l], lo.w13_tp_off[l], E_ep)
            _peer_access_dptp_w2(ctx, lo.w2_ep_off[l], lo.w2_tp_off[l], E_ep)

    def nccl_w13_seq():
        for l in range(N_LAYERS):
            w13_ep_view = _ep_view_w13(ctx.buf, lo.w13_ep_off[l], lo.w13_ep_bytes, E_ep)
            _nccl_dptp_w13_layer(w13_ep_view, st, st.tp_w13_layers[l], dp_group, tp_group)

    def nccl_w2_seq():
        for l in range(N_LAYERS):
            w2_ep_view = _ep_view_w2(ctx.buf, lo.w2_ep_off[l], lo.w2_ep_bytes, E_ep)
            _nccl_dptp_w2_layer(w2_ep_view, st, st.tp_w2_layers[l], dp_group, tp_group)

    def nccl_total_seq():
        for l in range(N_LAYERS):
            w13_ep_view = _ep_view_w13(ctx.buf, lo.w13_ep_off[l], lo.w13_ep_bytes, E_ep)
            w2_ep_view = _ep_view_w2(ctx.buf, lo.w2_ep_off[l], lo.w2_ep_bytes, E_ep)
            _nccl_dptp_total_layer(
                w13_ep_view, w2_ep_view, st,
                st.tp_w13_layers[l], st.tp_w2_layers[l],
                dp_group, tp_group,
            )

    pa_w13_s = _time_multi_layer(ctx, pa_w13_seq, end_barrier=True)
    pa_w2_s = _time_multi_layer(ctx, pa_w2_seq, end_barrier=True)
    pa_tot_s = _time_multi_layer(ctx, pa_total_seq, end_barrier=True)
    nccl_w13_s = _time_multi_layer(ctx, nccl_w13_seq, end_barrier=False)
    nccl_w2_s = _time_multi_layer(ctx, nccl_w2_seq, end_barrier=False)
    nccl_tot_s = _time_multi_layer(ctx, nccl_total_seq, end_barrier=False)

    barrier_s = _time_multi_layer(ctx, lambda: None, end_barrier=True)

    def per_layer(s: dict) -> dict:
        return {
            "mean": s["mean"] / N_LAYERS,
            "p50": s["p50"] / N_LAYERS,
            "p20": s["p20"] / N_LAYERS,
            "p90": s["p90"] / N_LAYERS,
        }

    pa_w13 = per_layer(pa_w13_s)
    pa_w2 = per_layer(pa_w2_s)
    pa_tot = per_layer(pa_tot_s)
    nccl_w13 = per_layer(nccl_w13_s)
    nccl_w2 = per_layer(nccl_w2_s)
    nccl_tot = per_layer(nccl_tot_s)

    if rank == 0:
        def row(name, pa, nc):
            sp_mean = nc["mean"] / pa["mean"] if pa["mean"] > 0 else float("nan")
            sp_p50 = nc["p50"] / pa["p50"] if pa["p50"] > 0 else float("nan")
            return (
                f"  {name:>5} | peer_access mean={pa['mean']:6.3f} p50={pa['p50']:6.3f}"
                f" | NCCL mean={nc['mean']:7.3f} p50={nc['p50']:7.3f}"
                f" | speedup mean={sp_mean:5.2f}x p50={sp_p50:5.2f}x"
            )
        print(row("w13", pa_w13, nccl_w13))
        print(row("w2", pa_w2, nccl_w2))
        print(row("total", pa_tot, nccl_tot))
        print()
        amortized_barrier_ms = barrier_s["p50"] / N_LAYERS
        print(f"[barrier amortized] world all_reduce p50/N_LAYERS = {amortized_barrier_ms:.4f} ms/layer "
              f"(measured idle-loop p50 = {barrier_s['p50']:.3f} ms over {N_LAYERS} layers)")
        print()
        print("=" * 70)
        print(f"NVLINK UTILIZATION (A100 SXM4, {NVLINK_BW_GB_S:.0f} GB/s/GPU single-direction)")
        print("=" * 70)

        def util_line(name, bytes_, t_ms):
            t_s = t_ms / 1000.0
            gb_s = (bytes_ / t_s) / 1e9
            pct = gb_s / NVLINK_BW_GB_S * 100.0
            return (f"  {name:>5} | send={bytes_/(1024**2):8.1f} MiB  "
                    f"p50={t_ms:6.3f} ms  -> {gb_s:6.1f} GB/s  ({pct:5.1f}% of {NVLINK_BW_GB_S:.0f})")

        print(util_line("w13", w13_send_bytes, pa_w13["p50"]))
        print(util_line("w2", w2_send_bytes, pa_w2["p50"]))
        print(util_line("total", total_send_bytes, pa_tot["p50"]))
        print()
        sp_total = nccl_tot["p50"] / pa_tot["p50"]
        util_total_pct = ((total_send_bytes / (pa_tot["p50"] / 1000.0)) / 1e9 / NVLINK_BW_GB_S * 100.0)
        print(f"[takeaway-forward] per-layer p50: dptp {pa_tot['p50']:.3f} ms vs NCCL "
              f"{nccl_tot['p50']:.3f} ms = {sp_total:.2f}x speedup; "
              f"dptp NVLink util = {util_total_pct:.1f}% of {NVLINK_BW_GB_S:.0f} GB/s")

    dp_rank_local = rank // T
    rev_w13_send = ((T - 1) / T) * E_ep * NUM_GATES * I * H * ELEM
    rev_w2_send = ((T - 1) / T) * E_ep * H * I * ELEM
    rev_total_send = rev_w13_send + rev_w2_send

    for l in range(N_LAYERS):
        _fill_tp_w13(ctx.buf, lo.w13_tp_off[l], rank % T)
        _fill_tp_w2(ctx.buf, lo.w2_tp_off[l], rank % T)
    ctx.barrier()
    _reverse_correctness_gate(ctx, E_ep, lo, st, tp_group)
    dist.barrier()

    if rank == 0:
        print()
        print("=" * 70)
        print(f"REVERSE PER-LAYER TIMING (ms; iter = {N_LAYERS} layers, warmup={WARMUP}, iters={ITERS})")
        print("=" * 70)

    def pa_rev_w13_seq():
        for l in range(N_LAYERS):
            _peer_access_dptp_ep_w13(ctx, lo.w13_tp_off[l], lo.w13_ep_off[l], E_ep)

    def pa_rev_w2_seq():
        for l in range(N_LAYERS):
            _peer_access_dptp_ep_w2(ctx, lo.w2_tp_off[l], lo.w2_ep_off[l], E_ep)

    def pa_rev_total_seq():
        for l in range(N_LAYERS):
            _peer_access_dptp_ep_w13(ctx, lo.w13_tp_off[l], lo.w13_ep_off[l], E_ep)
            _peer_access_dptp_ep_w2(ctx, lo.w2_tp_off[l], lo.w2_ep_off[l], E_ep)

    def nccl_rev_w13_seq():
        for l in range(N_LAYERS):
            _nccl_reverse_w13_layer(st.tp_w13_layers[l], st, st.ep_w13_rev, dp_rank_local, tp_group)

    def nccl_rev_w2_seq():
        for l in range(N_LAYERS):
            _nccl_reverse_w2_layer(st.tp_w2_layers[l], st, st.ep_w2_rev, dp_rank_local, tp_group)

    def nccl_rev_total_seq():
        for l in range(N_LAYERS):
            _nccl_reverse_total_layer(
                st.tp_w13_layers[l], st.tp_w2_layers[l], st,
                st.ep_w13_rev, st.ep_w2_rev, dp_rank_local, tp_group,
            )

    pa_rev_w13_s = _time_multi_layer(ctx, pa_rev_w13_seq, end_barrier=True)
    pa_rev_w2_s = _time_multi_layer(ctx, pa_rev_w2_seq, end_barrier=True)
    pa_rev_tot_s = _time_multi_layer(ctx, pa_rev_total_seq, end_barrier=True)
    nccl_rev_w13_s = _time_multi_layer(ctx, nccl_rev_w13_seq, end_barrier=False)
    nccl_rev_w2_s = _time_multi_layer(ctx, nccl_rev_w2_seq, end_barrier=False)
    nccl_rev_tot_s = _time_multi_layer(ctx, nccl_rev_total_seq, end_barrier=False)

    pa_rev_w13 = per_layer(pa_rev_w13_s)
    pa_rev_w2 = per_layer(pa_rev_w2_s)
    pa_rev_tot = per_layer(pa_rev_tot_s)
    nccl_rev_w13 = per_layer(nccl_rev_w13_s)
    nccl_rev_w2 = per_layer(nccl_rev_w2_s)
    nccl_rev_tot = per_layer(nccl_rev_tot_s)

    if rank == 0:
        def row(name, pa, nc):
            sp_mean = nc["mean"] / pa["mean"] if pa["mean"] > 0 else float("nan")
            sp_p50 = nc["p50"] / pa["p50"] if pa["p50"] > 0 else float("nan")
            return (
                f"  {name:>5} | peer_access mean={pa['mean']:6.3f} p50={pa['p50']:6.3f}"
                f" | NCCL mean={nc['mean']:7.3f} p50={nc['p50']:7.3f}"
                f" | speedup mean={sp_mean:5.2f}x p50={sp_p50:5.2f}x"
            )
        print(row("w13", pa_rev_w13, nccl_rev_w13))
        print(row("w2", pa_rev_w2, nccl_rev_w2))
        print(row("total", pa_rev_tot, nccl_rev_tot))
        print()
        print("=" * 70)
        print(f"REVERSE NVLINK UTILIZATION (A100 SXM4, {NVLINK_BW_GB_S:.0f} GB/s/GPU single-direction)")
        print(f"  send factor (T-1)/T = {(T-1)/T:.3f}x full EP per rank")
        print("=" * 70)

        def util_line(name, bytes_, t_ms):
            t_s = t_ms / 1000.0
            gb_s = (bytes_ / t_s) / 1e9
            pct = gb_s / NVLINK_BW_GB_S * 100.0
            return (f"  {name:>5} | send={bytes_/(1024**2):8.1f} MiB  "
                    f"p50={t_ms:6.3f} ms  -> {gb_s:6.1f} GB/s  ({pct:5.1f}% of {NVLINK_BW_GB_S:.0f})")

        print(util_line("w13", rev_w13_send, pa_rev_w13["p50"]))
        print(util_line("w2", rev_w2_send, pa_rev_w2["p50"]))
        print(util_line("total", rev_total_send, pa_rev_tot["p50"]))
        print()
        sp_rev = nccl_rev_tot["p50"] / pa_rev_tot["p50"]
        util_rev_pct = ((rev_total_send / (pa_rev_tot["p50"] / 1000.0)) / 1e9 / NVLINK_BW_GB_S * 100.0)
        print(f"[takeaway-reverse] per-layer p50: dptp_rev {pa_rev_tot['p50']:.3f} ms vs NCCL_rev "
              f"{nccl_rev_tot['p50']:.3f} ms = {sp_rev:.2f}x speedup; "
              f"dptp_rev NVLink util = {util_rev_pct:.1f}% of {NVLINK_BW_GB_S:.0f} GB/s")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
