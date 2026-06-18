"""Correctness test for dptp REVERSE (DPxTP -> EP) weight-transfer kernels.

For each (T, G) in {(8,1), (4,2), (2,4)} with world W = G*T = 8 and qwen3-30b
preset (E=128, H=2048, I=768, num_gates=2, bf16), this test:

  1. Fills each rank's canonical TP region with the bf16-exact slice
     W13[:, :, tp_rank, :] / W2[:, :, tp_rank*I':(tp_rank+1)*I'] of a
     deterministic global pattern (idx % 256), reusing the helpers from
     test_dptp_forward.py.
  2. Runs launch_peer_access_fused_transfer_w{13,2}_ep_dptp.
  3. After the world barrier, asserts each rank's EP region equals the
     original EP fill for that rank (each rank holds experts
     [R*E_ep, (R+1)*E_ep) full, layout (E_ep, NG, T, I'*H) for w13 and
     (E_ep, H, I) for w2).
  4. At (T=8, G=1) additionally runs the v3 *_ep kernel on the same TP input
     into a side EP region and asserts byte-identical to the dptp output.

Usage:
    torchrun --nproc_per_node=8 test_dptp_reverse.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from common.ipc import IPCContext, setup_ipc_arena

import paras_peer_access_cuda as ppa


E = 128
H = 2048
I = 768
NUM_GATES = 2
ELEM = 2
W = 8


def _bf16_pattern(indices: torch.Tensor) -> torch.Tensor:
    return (indices & 255).to(torch.bfloat16)


def _fill_tp_w13(buf_u8: torch.Tensor, tp_off: int, tp_rank: int, T: int) -> None:
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
    pattern = _bf16_pattern(linear)
    view = buf_u8[tp_off:tp_off + bytes_].view(torch.bfloat16).view(E, NUM_GATES, I_prime_H)
    view.copy_(pattern)


def _fill_tp_w2(buf_u8: torch.Tensor, tp_off: int, tp_rank: int, T: int) -> None:
    I_prime = I // T
    bytes_ = E * H * I_prime * ELEM
    e_idx = torch.arange(E, dtype=torch.int32, device=buf_u8.device).view(E, 1, 1)
    h_idx = torch.arange(H, dtype=torch.int32, device=buf_u8.device).view(1, H, 1)
    j_idx = torch.arange(I_prime, dtype=torch.int32, device=buf_u8.device).view(1, 1, I_prime)
    linear = e_idx * (H * I) + h_idx * I + tp_rank * I_prime + j_idx
    pattern = _bf16_pattern(linear)
    view = buf_u8[tp_off:tp_off + bytes_].view(torch.bfloat16).view(E, H, I_prime)
    view.copy_(pattern)


def _expected_ep_w13(device: torch.device, rank: int, E_ep: int, T: int) -> torch.Tensor:
    I_prime = I // T
    I_prime_H = I_prime * H
    numel = E_ep * NUM_GATES * T * I_prime_H
    start = rank * numel
    idx = torch.arange(numel, dtype=torch.int32, device=device) + start
    return _bf16_pattern(idx).view(E_ep, NUM_GATES, T, I_prime_H)


def _expected_ep_w2(device: torch.device, rank: int, E_ep: int) -> torch.Tensor:
    numel = E_ep * H * I
    start = rank * numel
    idx = torch.arange(numel, dtype=torch.int32, device=device) + start
    return _bf16_pattern(idx).view(E_ep, H, I)


def _w13_ep_bytes(E_ep: int, T: int) -> int:
    return E_ep * NUM_GATES * T * (I // T) * H * ELEM


def _w13_tp_bytes(T: int) -> int:
    return E * NUM_GATES * (I // T) * H * ELEM


def _w2_ep_bytes(E_ep: int) -> int:
    return E_ep * H * I * ELEM


def _w2_tp_bytes(T: int) -> int:
    return E * H * (I // T) * ELEM


def _check_w13(
    ctx: IPCContext, T: int, G: int, E_ep: int,
    tp_off: int, ep_off: int, ep_v3_off: int | None,
) -> tuple[bool, str]:
    rank = ctx.rank
    tp_rank = rank % T
    device = ctx.device
    I_prime = I // T
    I_prime_H = I_prime * H

    ctx.buf[ep_off:ep_off + _w13_ep_bytes(E_ep, T)].zero_()
    if ep_v3_off is not None:
        ctx.buf[ep_v3_off:ep_v3_off + _w13_ep_bytes(E_ep, T)].zero_()
    _fill_tp_w13(ctx.buf, tp_off, tp_rank, T)
    ctx.barrier()

    stream_ptr = torch.cuda.current_stream(device).cuda_stream
    ppa.launch_peer_access_fused_transfer_w13_ep_dptp(
        ctx.local_buffer_ptr, ctx.peer_buffer_ptrs,
        tp_off, ep_off,
        rank, T, G, E_ep,
        H, I, NUM_GATES, ELEM,
        stream_ptr,
    )
    torch.cuda.synchronize(device)
    bar = torch.zeros(1, device=device); dist.all_reduce(bar)

    got = ctx.buf[ep_off:ep_off + _w13_ep_bytes(E_ep, T)].view(torch.bfloat16).view(
        E_ep, NUM_GATES, T, I_prime_H
    )
    expected = _expected_ep_w13(device, rank, E_ep, T)
    if not torch.equal(got, expected):
        d = (got.to(torch.int32) != expected.to(torch.int32)).sum().item()
        return False, f"w13 reverse mismatch: {d} elements"

    if ep_v3_off is not None:
        ppa.launch_peer_access_fused_transfer_w13_v3_ep(
            ctx.local_buffer_ptr, ctx.peer_buffer_ptrs,
            tp_off, ep_v3_off,
            rank, T, E_ep,
            H, I, NUM_GATES, ELEM,
            stream_ptr,
        )
        torch.cuda.synchronize(device)
        bar2 = torch.zeros(1, device=device); dist.all_reduce(bar2)
        v3 = ctx.buf[ep_v3_off:ep_v3_off + _w13_ep_bytes(E_ep, T)].view(torch.bfloat16).view(
            E_ep, NUM_GATES, T, I_prime_H
        )
        if not torch.equal(got, v3):
            d = (got.to(torch.int32) != v3.to(torch.int32)).sum().item()
            return False, f"w13 dptp_ep != v3_ep regression: {d} elements"
    return True, "w13 PASS"


def _check_w2(
    ctx: IPCContext, T: int, G: int, E_ep: int,
    tp_off: int, ep_off: int, ep_v3_off: int | None,
) -> tuple[bool, str]:
    rank = ctx.rank
    tp_rank = rank % T
    device = ctx.device

    ctx.buf[ep_off:ep_off + _w2_ep_bytes(E_ep)].zero_()
    if ep_v3_off is not None:
        ctx.buf[ep_v3_off:ep_v3_off + _w2_ep_bytes(E_ep)].zero_()
    _fill_tp_w2(ctx.buf, tp_off, tp_rank, T)
    ctx.barrier()

    stream_ptr = torch.cuda.current_stream(device).cuda_stream
    ppa.launch_peer_access_fused_transfer_w2_ep_dptp(
        ctx.local_buffer_ptr, ctx.peer_buffer_ptrs,
        tp_off, ep_off,
        rank, T, G, E_ep,
        H, I, ELEM,
        stream_ptr,
    )
    torch.cuda.synchronize(device)
    bar = torch.zeros(1, device=device); dist.all_reduce(bar)

    got = ctx.buf[ep_off:ep_off + _w2_ep_bytes(E_ep)].view(torch.bfloat16).view(E_ep, H, I)
    expected = _expected_ep_w2(device, rank, E_ep)
    if not torch.equal(got, expected):
        d = (got.to(torch.int32) != expected.to(torch.int32)).sum().item()
        return False, f"w2 reverse mismatch: {d} elements"

    if ep_v3_off is not None:
        ppa.launch_peer_access_fused_transfer_w2_v3_ep(
            ctx.local_buffer_ptr, ctx.peer_buffer_ptrs,
            tp_off, ep_v3_off,
            rank, T, E_ep,
            H, I, ELEM,
            stream_ptr,
        )
        torch.cuda.synchronize(device)
        bar2 = torch.zeros(1, device=device); dist.all_reduce(bar2)
        v3 = ctx.buf[ep_v3_off:ep_v3_off + _w2_ep_bytes(E_ep)].view(torch.bfloat16).view(E_ep, H, I)
        if not torch.equal(got, v3):
            d = (got.to(torch.int32) != v3.to(torch.int32)).sum().item()
            return False, f"w2 dptp_ep != v3_ep regression: {d} elements"
    return True, "w2 PASS"


def _run_config(T: int, G: int) -> dict:
    assert T * G == W
    E_ep = E // W

    w13_ep = _w13_ep_bytes(E_ep, T)
    w13_tp = _w13_tp_bytes(T)
    w2_ep = _w2_ep_bytes(E_ep)
    w2_tp = _w2_tp_bytes(T)
    include_v3 = (T == 8 and G == 1)

    w13_tp_off = 0
    w13_ep_off = w13_tp_off + w13_tp
    if include_v3:
        w13_ep_v3_off: int | None = w13_ep_off + w13_ep
        w2_tp_off = w13_ep_v3_off + w13_ep
    else:
        w13_ep_v3_off = None
        w2_tp_off = w13_ep_off + w13_ep
    w2_ep_off = w2_tp_off + w2_tp
    if include_v3:
        w2_ep_v3_off: int | None = w2_ep_off + w2_ep
        arena = w2_ep_v3_off + w2_ep
    else:
        w2_ep_v3_off = None
        arena = w2_ep_off + w2_ep

    ctx = setup_ipc_arena(arena)
    if ctx.world_size != W:
        raise SystemExit(f"world_size {ctx.world_size} != {W}")

    if ctx.rank == 0:
        print(f"[T={T} G={G}] arena={arena/(1024**2):.1f} MiB E_ep={E_ep} I'={I//T} include_v3={include_v3}")

    w13_ok, w13_msg = _check_w13(ctx, T, G, E_ep, w13_tp_off, w13_ep_off, w13_ep_v3_off)
    w2_ok, w2_msg = _check_w2(ctx, T, G, E_ep, w2_tp_off, w2_ep_off, w2_ep_v3_off)
    return {"w13_ok": w13_ok, "w13_msg": w13_msg, "w2_ok": w2_ok, "w2_msg": w2_msg}


def main():
    configs = [(8, 1), (4, 2), (2, 4)]
    all_pass = True
    rank = int(os.environ.get("RANK", "0"))

    for T, G in configs:
        res = _run_config(T, G)
        ok_w13 = torch.tensor([1 if res["w13_ok"] else 0], device=f"cuda:{rank}")
        ok_w2 = torch.tensor([1 if res["w2_ok"] else 0], device=f"cuda:{rank}")
        dist.all_reduce(ok_w13, op=dist.ReduceOp.MIN)
        dist.all_reduce(ok_w2, op=dist.ReduceOp.MIN)
        gw13 = ok_w13.item() == 1
        gw2 = ok_w2.item() == 1
        cfg_pass = gw13 and gw2
        all_pass = all_pass and cfg_pass
        if rank == 0:
            tag = "PASS" if cfg_pass else "FAIL"
            print(f"[(T={T}, G={G})] {tag}: w13={'PASS' if gw13 else 'FAIL'} w2={'PASS' if gw2 else 'FAIL'}")
        if not res["w13_ok"]:
            print(f"[(T={T}, G={G})] rank {rank}: {res['w13_msg']}")
        if not res["w2_ok"]:
            print(f"[(T={T}, G={G})] rank {rank}: {res['w2_msg']}")
        dist.barrier()

    if rank == 0:
        print("=" * 60)
        print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
    dist.barrier()
    dist.destroy_process_group()
    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
