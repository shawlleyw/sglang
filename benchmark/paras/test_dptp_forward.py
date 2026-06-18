"""Correctness test for dptp (EP -> DP x TP) forward weight-transfer kernels.

For each (T, G) in {(8,1), (4,2), (2,4)} with world size W = G*T = 8 and model
preset qwen3-30b (E=128, H=2048, I=768, num_gates=2, bf16), this test:

  1. Allocates one IPC arena per (T, G) sized for EP + TP regions for both w13
     and w2.
  2. Initializes the EP region on each rank with a bf16-exact deterministic
     pattern (linear-index mod 256). The pattern is computed per-rank from
     the rank-global index range so all ranks agree on the same global tensor
     without materializing it.
  3. Launches launch_peer_access_fused_transfer_w{13,2}_dptp from
     paras_peer_access_cuda.
  4. After ctx.barrier() asserts the TP region equals the expected canonical
     slice for tp_rank = rank % T, byte-exact via torch.equal.
  5. At (T=8, G=1) additionally runs the trusted v3 kernel into a side TP region
     on the same EP input and asserts the dptp output is byte-identical to v3.

Exits with status 0 only if all configs PASS for both kernels.

Usage:
    torchrun --nproc_per_node=8 test_dptp_forward.py
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


def _bf16_pattern_from_indices(indices: torch.Tensor) -> torch.Tensor:
    """linear index (int32/int64) -> bf16 value `(idx & 255)` exactly."""
    return (indices & 255).to(torch.bfloat16)


def _fill_ep_w13(buf_u8: torch.Tensor, ep_off: int, rank: int, E_local: int, T: int) -> None:
    """Fill rank `rank`'s w13 EP region from global W13 = (idx % 256) bf16 tensor.

    Global W13 shape: (E, NUM_GATES, T, I'*H). Rank R owns global slice
    W13[R*E_local : (R+1)*E_local] laid out as (E_local, NUM_GATES, T, I'*H)
    in its EP region.
    """
    I_prime = I // T
    I_prime_H = I_prime * H
    slice_numel = E_local * NUM_GATES * T * I_prime_H
    slice_bytes = slice_numel * ELEM
    slice_start_global = rank * E_local * NUM_GATES * T * I_prime_H

    idx = torch.arange(slice_numel, dtype=torch.int32, device=buf_u8.device) + slice_start_global
    pattern = _bf16_pattern_from_indices(idx).view(E_local, NUM_GATES, T, I_prime_H)

    ep_view = buf_u8[ep_off:ep_off + slice_bytes].view(torch.bfloat16).view(
        E_local, NUM_GATES, T, I_prime_H
    )
    ep_view.copy_(pattern)


def _expected_w13_tp(device: torch.device, tp_rank: int, T: int) -> torch.Tensor:
    """Build expected canonical TP region for w13 at given tp_rank, shape (E, NG, I'*H)."""
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
    return _bf16_pattern_from_indices(linear)


def _fill_ep_w2(buf_u8: torch.Tensor, ep_off: int, rank: int, E_local: int) -> None:
    """Fill rank `rank`'s w2 EP region from global W2 = (idx % 256) bf16 tensor.

    Global W2 shape: (E, H, I). Rank R owns global slice
    W2[R*E_local : (R+1)*E_local] laid out as (E_local, H, I) in EP region.
    """
    slice_numel = E_local * H * I
    slice_bytes = slice_numel * ELEM
    slice_start_global = rank * E_local * H * I

    idx = torch.arange(slice_numel, dtype=torch.int32, device=buf_u8.device) + slice_start_global
    pattern = _bf16_pattern_from_indices(idx).view(E_local, H, I)

    ep_view = buf_u8[ep_off:ep_off + slice_bytes].view(torch.bfloat16).view(E_local, H, I)
    ep_view.copy_(pattern)


def _expected_w2_tp(device: torch.device, tp_rank: int, T: int) -> torch.Tensor:
    """Build expected canonical TP region for w2 at given tp_rank, shape (E, H, I')."""
    I_prime = I // T
    e_idx = torch.arange(E, dtype=torch.int32, device=device).view(E, 1, 1)
    h_idx = torch.arange(H, dtype=torch.int32, device=device).view(1, H, 1)
    j_idx = torch.arange(I_prime, dtype=torch.int32, device=device).view(1, 1, I_prime)
    linear = e_idx * (H * I) + h_idx * I + tp_rank * I_prime + j_idx
    return _bf16_pattern_from_indices(linear)


def _w13_region_bytes(E_local: int, T: int, is_ep: bool) -> int:
    I_prime = I // T
    if is_ep:
        return E_local * NUM_GATES * T * I_prime * H * ELEM
    return E * NUM_GATES * I_prime * H * ELEM


def _w2_region_bytes(E_local: int, T: int, is_ep: bool) -> int:
    I_prime = I // T
    if is_ep:
        return E_local * H * I * ELEM
    return E * H * I_prime * ELEM


def _check_w13(
    ctx: IPCContext, T: int, G: int, E_local: int,
    w13_ep_off: int, w13_tp_off: int, w13_tp_v3_off: int | None,
) -> tuple[bool, str]:
    """Run w13 dptp, validate, and optionally regress against v3 (T=8, G=1).

    Returns (passed, msg).
    """
    rank = ctx.rank
    tp_rank = rank % T
    device = ctx.device

    ctx.buf.zero_()
    _fill_ep_w13(ctx.buf, w13_ep_off, rank, E_local, T)
    ctx.barrier()

    stream_ptr = torch.cuda.current_stream(device).cuda_stream
    ppa.launch_peer_access_fused_transfer_w13_dptp(
        ctx.local_buffer_ptr, ctx.peer_buffer_ptrs,
        w13_ep_off, w13_tp_off,
        rank, T, G, E_local,
        H, I, NUM_GATES, ELEM,
        stream_ptr,
    )
    torch.cuda.synchronize(device)
    ctx.barrier()

    I_prime_H = (I // T) * H
    tp_bytes = _w13_region_bytes(E_local, T, is_ep=False)
    got = ctx.buf[w13_tp_off:w13_tp_off + tp_bytes].view(torch.bfloat16).view(
        E, NUM_GATES, I_prime_H
    )
    expected = _expected_w13_tp(device, tp_rank, T)
    if not torch.equal(got, expected):
        diff = (got.to(torch.int32) != expected.to(torch.int32)).sum().item()
        return False, f"w13 mismatch: {diff} bf16 elements differ (out of {got.numel()})"

    if w13_tp_v3_off is not None:
        ctx.buf[w13_tp_v3_off:w13_tp_v3_off + tp_bytes].zero_()
        ctx.barrier()
        ppa.launch_peer_access_fused_transfer_w13_v3(
            ctx.local_buffer_ptr, ctx.peer_buffer_ptrs,
            w13_ep_off, w13_tp_v3_off,
            rank, T, E_local,
            H, I, NUM_GATES, ELEM,
            stream_ptr,
        )
        torch.cuda.synchronize(device)
        ctx.barrier()
        v3_got = ctx.buf[w13_tp_v3_off:w13_tp_v3_off + tp_bytes].view(torch.bfloat16).view(
            E, NUM_GATES, I_prime_H
        )
        if not torch.equal(got, v3_got):
            diff = (got.to(torch.int32) != v3_got.to(torch.int32)).sum().item()
            return False, f"w13 dptp vs v3 regression mismatch: {diff} bf16 elements differ"

    return True, "w13 PASS"


def _check_w2(
    ctx: IPCContext, T: int, G: int, E_local: int,
    w2_ep_off: int, w2_tp_off: int, w2_tp_v3_off: int | None,
) -> tuple[bool, str]:
    rank = ctx.rank
    tp_rank = rank % T
    device = ctx.device

    ctx.buf.zero_()
    _fill_ep_w2(ctx.buf, w2_ep_off, rank, E_local)
    ctx.barrier()

    stream_ptr = torch.cuda.current_stream(device).cuda_stream
    ppa.launch_peer_access_fused_transfer_w2_dptp(
        ctx.local_buffer_ptr, ctx.peer_buffer_ptrs,
        w2_ep_off, w2_tp_off,
        rank, T, G, E_local,
        H, I, ELEM,
        stream_ptr,
    )
    torch.cuda.synchronize(device)
    ctx.barrier()

    I_prime = I // T
    tp_bytes = _w2_region_bytes(E_local, T, is_ep=False)
    got = ctx.buf[w2_tp_off:w2_tp_off + tp_bytes].view(torch.bfloat16).view(E, H, I_prime)
    expected = _expected_w2_tp(device, tp_rank, T)
    if not torch.equal(got, expected):
        diff = (got.to(torch.int32) != expected.to(torch.int32)).sum().item()
        return False, f"w2 mismatch: {diff} bf16 elements differ (out of {got.numel()})"

    if w2_tp_v3_off is not None:
        ctx.buf[w2_tp_v3_off:w2_tp_v3_off + tp_bytes].zero_()
        ctx.barrier()
        ppa.launch_peer_access_fused_transfer_w2_v3(
            ctx.local_buffer_ptr, ctx.peer_buffer_ptrs,
            w2_ep_off, w2_tp_v3_off,
            rank, T, E_local,
            H, I, ELEM,
            stream_ptr,
        )
        torch.cuda.synchronize(device)
        ctx.barrier()
        v3_got = ctx.buf[w2_tp_v3_off:w2_tp_v3_off + tp_bytes].view(torch.bfloat16).view(E, H, I_prime)
        if not torch.equal(got, v3_got):
            diff = (got.to(torch.int32) != v3_got.to(torch.int32)).sum().item()
            return False, f"w2 dptp vs v3 regression mismatch: {diff} bf16 elements differ"

    return True, "w2 PASS"


def _run_config(T: int, G: int) -> dict:
    """Returns {'w13_ok': bool, 'w13_msg': str, 'w2_ok': bool, 'w2_msg': str}."""
    assert T * G == W
    E_local = E // W

    w13_ep_bytes = _w13_region_bytes(E_local, T, is_ep=True)
    w13_tp_bytes = _w13_region_bytes(E_local, T, is_ep=False)
    w2_ep_bytes = _w2_region_bytes(E_local, T, is_ep=True)
    w2_tp_bytes = _w2_region_bytes(E_local, T, is_ep=False)

    include_v3 = (T == 8 and G == 1)

    w13_ep_off = 0
    w13_tp_off = w13_ep_off + w13_ep_bytes
    if include_v3:
        w13_tp_v3_off: int | None = w13_tp_off + w13_tp_bytes
        w2_ep_off = w13_tp_v3_off + w13_tp_bytes
    else:
        w13_tp_v3_off = None
        w2_ep_off = w13_tp_off + w13_tp_bytes
    w2_tp_off = w2_ep_off + w2_ep_bytes
    if include_v3:
        w2_tp_v3_off: int | None = w2_tp_off + w2_tp_bytes
        arena_bytes = w2_tp_v3_off + w2_tp_bytes
    else:
        w2_tp_v3_off = None
        arena_bytes = w2_tp_off + w2_tp_bytes

    ctx = setup_ipc_arena(arena_bytes)
    if ctx.world_size != W:
        raise SystemExit(f"world_size {ctx.world_size} != expected {W}")

    if ctx.rank == 0:
        print(f"[T={T} G={G}] arena={arena_bytes/(1024**2):.1f} MiB "
              f"E_local={E_local} I'={I // T} include_v3={include_v3}")

    w13_ok, w13_msg = _check_w13(ctx, T, G, E_local, w13_ep_off, w13_tp_off, w13_tp_v3_off)
    w2_ok, w2_msg = _check_w2(ctx, T, G, E_local, w2_ep_off, w2_tp_off, w2_tp_v3_off)

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
        global_w13_ok = ok_w13.item() == 1
        global_w2_ok = ok_w2.item() == 1
        config_pass = global_w13_ok and global_w2_ok
        all_pass = all_pass and config_pass

        if rank == 0:
            tag = "PASS" if config_pass else "FAIL"
            print(f"[(T={T}, G={G})] {tag}: w13={'PASS' if global_w13_ok else 'FAIL'} "
                  f"w2={'PASS' if global_w2_ok else 'FAIL'}")
            if not res["w13_ok"]:
                print(f"   [rank {rank}] {res['w13_msg']}")
            if not res["w2_ok"]:
                print(f"   [rank {rank}] {res['w2_msg']}")
        if not res["w13_ok"] and rank != 0:
            print(f"[(T={T}, G={G})] rank {rank}: {res['w13_msg']}")
        if not res["w2_ok"] and rank != 0:
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
