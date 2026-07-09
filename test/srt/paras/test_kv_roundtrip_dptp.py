#!/usr/bin/env python3
"""Real dp=2 tp=2 ep=4 KV-cache round-trip test (intra-TP-group redistribution).

Design (authoritative): the global EP group is all 4 GPUs and MoE weights
reshard across it, but the KV cache and requests redistribute ONLY within each
TP subgroup of 2 GPUs. With ep=4, tp=2, dp=2 there are two independent TP
subgroups: group 0 = {rank 0, 1}, group 1 = {rank 2, 3}. Each subgroup performs
a self-contained tp=2 KV switch (EP -> TP -> EP) over its own 2-rank NCCL group;
the two subgroups never exchange KV bytes.

This harness validates:
  1. Within each subgroup, EP -> TP -> EP recovers each rank's KV bitwise.
  2. Cross-group isolation: seeding KV by GLOBAL rank makes the two subgroups
     carry disjoint data, so a correct intra-group round trip reproduces only
     the owning group's inputs (any cross-group leak would corrupt the bytes).

It reuses the production topology-agnostic KV helpers from
sglang.srt.paras.cache_transfer.utils (the same functions the scatter/gather
managers call), driven over a group_size=2 subgroup with
heads_per_rank = num_kv_heads / tp_size.

Usage:
  CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
      test/srt/paras/test_kv_roundtrip_dptp.py
"""

import os
import sys

import torch
import torch.distributed as dist

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.join(_TEST_DIR, "..", "..", "..")
sys.path.insert(0, os.path.join(_ROOT_DIR, "python"))

NUM_LAYERS = 3
NUM_KV_HEADS = 4
HEAD_DIM = 128
KV_DTYPE = torch.bfloat16
SEED = 42

TP_SIZE = 2
DP_SIZE = 2
EP_SIZE = DP_SIZE * TP_SIZE

# Per-subgroup-local-rank token counts; the two subgroups use the same counts,
# but distinct KV data (seeded by global rank).
TOKENS_PER_LOCAL_RANK = [64, 48]


def setup_distributed():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == EP_SIZE, f"requires exactly {EP_SIZE} GPUs, got {world_size}"
    torch.cuda.set_device(rank)
    return rank, world_size


def make_tp_subgroup(rank):
    """Return this rank's 2-rank TP subgroup handle and its tp-local rank.

    dp_group_id = rank // TP_SIZE selects the subgroup; the subgroup ranks are
    contiguous [dp_group_id*TP_SIZE, (dp_group_id+1)*TP_SIZE). Every rank must
    create ALL subgroups (collective), then keep its own.
    """
    my_group = None
    my_tp_rank = rank % TP_SIZE
    for g in range(DP_SIZE):
        ranks = list(range(g * TP_SIZE, (g + 1) * TP_SIZE))
        grp = dist.new_group(ranks=ranks)
        if rank in ranks:
            my_group = grp
    return my_group, my_tp_rank


def fill_ep_kv(rank, num_local_tokens):
    """EP-mode KV buffers seeded by GLOBAL rank (distinct across subgroups).

    Layout per layer: (num_local_tokens, NUM_KV_HEADS, HEAD_DIM). Token t lives
    at slot t.
    """
    k_bufs, v_bufs = [], []
    for layer_id in range(NUM_LAYERS):
        gk = torch.Generator(device="cpu")
        gk.manual_seed(SEED + layer_id * 1000 + rank)
        k = torch.randn(
            (num_local_tokens, NUM_KV_HEADS, HEAD_DIM), generator=gk, dtype=torch.float32
        ).to(dtype=KV_DTYPE, device=f"cuda:{rank}")
        gv = torch.Generator(device="cpu")
        gv.manual_seed(SEED + layer_id * 1000 + rank + 500)
        v = torch.randn(
            (num_local_tokens, NUM_KV_HEADS, HEAD_DIM), generator=gv, dtype=torch.float32
        ).to(dtype=KV_DTYPE, device=f"cuda:{rank}")
        k_bufs.append(k)
        v_bufs.append(v)
    return k_bufs, v_bufs


def gather_ep_to_tp(k_bufs, v_bufs, tp_group, my_tp_rank, num_local_tokens, total_tokens):
    """EP -> TP gather within the 2-rank subgroup (mirrors MHACacheTransfer gather).

    Each rank holds num_local_tokens with all NUM_KV_HEADS; after the switch each
    rank holds ALL total_tokens tokens but only heads_per_rank = NUM_KV_HEADS/TP.
    """
    from sglang.srt.paras.cache_transfer.utils import (
        gather_kv_and_permute,
        permute_and_scatter_kv,
    )

    heads_per_rank = NUM_KV_HEADS // TP_SIZE
    splited = heads_per_rank * HEAD_DIM
    local_idx = torch.arange(num_local_tokens, dtype=torch.long, device="cuda")
    global_idx = torch.arange(total_tokens, dtype=torch.long, device="cuda")

    # This rank sends num_local_tokens to every peer; receives every peer's full
    # token count (all peers contribute to the gathered global sequence).
    input_split = [2 * splited * num_local_tokens] * TP_SIZE
    output_split = [2 * splited * n for n in TOKENS_PER_LOCAL_RANK]

    tp_k, tp_v = [], []
    for layer_id in range(NUM_LAYERS):
        permuted = gather_kv_and_permute(
            k_bufs[layer_id], v_bufs[layer_id], local_idx, heads_per_rank
        )
        gathered = torch.empty(
            2 * total_tokens * splited, dtype=KV_DTYPE, device="cuda"
        )
        dist.all_to_all_single(
            gathered, permuted, output_split, input_split, group=tp_group
        )
        k = torch.empty(total_tokens, heads_per_rank, HEAD_DIM, dtype=KV_DTYPE, device="cuda")
        v = torch.empty(total_tokens, heads_per_rank, HEAD_DIM, dtype=KV_DTYPE, device="cuda")
        permute_and_scatter_kv(
            gathered, k, v, global_idx, total_tokens, heads_per_rank, HEAD_DIM
        )
        tp_k.append(k)
        tp_v.append(v)
    return tp_k, tp_v


def scatter_tp_to_ep(tp_k, tp_v, tp_group, my_tp_rank, num_local_tokens, total_tokens):
    """TP -> EP scatter within the 2-rank subgroup (mirrors MHACacheTransfer scatter).

    EP partition is deterministic: tp-local rank r owns the contiguous global
    token range assigned to it, matching the ascending fill order so the round
    trip reproduces each rank's original tokens.
    """
    from sglang.srt.paras.cache_transfer.utils import (
        gather_tp_kv_and_permute,
        permute_and_scatter_kv_to_ep,
    )

    heads_per_rank = NUM_KV_HEADS // TP_SIZE
    per_token_elems = heads_per_rank * 2 * HEAD_DIM

    # Deterministic EP token partition: rank r gets the same contiguous range it
    # originally filled (ranges in ascending tp-rank order).
    starts = []
    acc = 0
    for n in TOKENS_PER_LOCAL_RANK:
        starts.append(acc)
        acc += n
    my_start = starts[my_tp_rank]
    my_count = TOKENS_PER_LOCAL_RANK[my_tp_rank]

    # Tokens this rank SENDS to EP rank e: e's contiguous global range.
    send_counts = list(TOKENS_PER_LOCAL_RANK)
    scatter_in = [c * per_token_elems for c in send_counts]
    scatter_out = [my_count * per_token_elems] * TP_SIZE

    sorted_parts = []
    for e in range(TP_SIZE):
        e_start = starts[e]
        e_cnt = TOKENS_PER_LOCAL_RANK[e]
        if e_cnt > 0:
            sorted_parts.append(
                torch.arange(e_start, e_start + e_cnt, dtype=torch.long, device="cuda")
            )
    sorted_idx = (
        torch.cat(sorted_parts)
        if sorted_parts
        else torch.empty(0, dtype=torch.long, device="cuda")
    )
    dst_positions = torch.arange(my_count, dtype=torch.long, device="cuda")

    ep_k, ep_v = [], []
    for layer_id in range(NUM_LAYERS):
        send_buf = gather_tp_kv_and_permute(
            tp_k[layer_id], tp_v[layer_id], sorted_idx,
            NUM_KV_HEADS, heads_per_rank, HEAD_DIM, TP_SIZE,
        )
        recv_buf = torch.empty(
            my_count * TP_SIZE * per_token_elems, dtype=KV_DTYPE, device="cuda"
        )
        dist.all_to_all_single(
            recv_buf, send_buf, scatter_out, scatter_in, group=tp_group
        )
        k = torch.zeros(my_count, NUM_KV_HEADS, HEAD_DIM, dtype=KV_DTYPE, device="cuda")
        v = torch.zeros(my_count, NUM_KV_HEADS, HEAD_DIM, dtype=KV_DTYPE, device="cuda")
        permute_and_scatter_kv_to_ep(
            recv_buf, k, v, dst_positions,
            my_count, NUM_KV_HEADS, heads_per_rank, HEAD_DIM, TP_SIZE,
        )
        ep_k.append(k)
        ep_v.append(v)
    return ep_k, ep_v, my_start, my_count


def main():
    rank, world_size = setup_distributed()
    passed = failed = 0
    try:
        tp_group, my_tp_rank = make_tp_subgroup(rank)
        num_local_tokens = TOKENS_PER_LOCAL_RANK[my_tp_rank]
        total_tokens = sum(TOKENS_PER_LOCAL_RANK)

        k_bufs, v_bufs = fill_ep_kv(rank, num_local_tokens)
        orig = [(k_bufs[l].clone(), v_bufs[l].clone()) for l in range(NUM_LAYERS)]

        tp_k, tp_v = gather_ep_to_tp(
            k_bufs, v_bufs, tp_group, my_tp_rank, num_local_tokens, total_tokens
        )
        ep_k, ep_v, my_start, my_count = scatter_tp_to_ep(
            tp_k, tp_v, tp_group, my_tp_rank, num_local_tokens, total_tokens
        )
        torch.cuda.synchronize()

        # Test 1: bitwise round trip. The recovered EP tokens for this rank are
        # this rank's original tokens (deterministic partition == fill order).
        if rank == 0:
            print("\n=== KV round trip EP -> TP -> EP (intra-subgroup) ===", flush=True)
        ok = True
        for layer_id in range(NUM_LAYERS):
            if not torch.equal(ep_k[layer_id], orig[layer_id][0]):
                ok = False
                print(f"  [FAIL rank {rank}] k mismatch layer {layer_id}", flush=True)
            if not torch.equal(ep_v[layer_id], orig[layer_id][1]):
                ok = False
                print(f"  [FAIL rank {rank}] v mismatch layer {layer_id}", flush=True)
        ok_t = torch.tensor([1 if ok else 0], device="cuda")
        dist.all_reduce(ok_t, op=dist.ReduceOp.MIN)
        if ok_t.item() == 1:
            if rank == 0:
                print("  [OK] all ranks recovered original EP KV bitwise", flush=True)
            passed += 1
        else:
            failed += 1

        # Test 2: cross-group isolation. Recompute what the round trip produced
        # and confirm it equals THIS rank's global-rank-seeded data, which is
        # distinct from the other subgroup's data. A cross-group leak would have
        # mixed the other group's bytes and failed Test 1 already; here we
        # additionally assert the two subgroups hold different inputs so the
        # isolation claim is meaningful (not a trivial all-equal pass).
        if rank == 0:
            print("\n=== Cross-group isolation ===", flush=True)
        partner = (rank + TP_SIZE) % world_size  # same tp-local rank, other subgroup
        mine = orig[0][0].contiguous()
        theirs = torch.empty_like(mine)
        if rank < partner:
            dist.send(mine, dst=partner)
            dist.recv(theirs, src=partner)
        else:
            dist.recv(theirs, src=partner)
            dist.send(mine, dst=partner)
        distinct = not torch.equal(mine, theirs)
        distinct_t = torch.tensor([1 if distinct else 0], device="cuda")
        dist.all_reduce(distinct_t, op=dist.ReduceOp.MIN)
        if distinct_t.item() == 1 and ok_t.item() == 1:
            if rank == 0:
                print(
                    "  [OK] subgroups carry distinct data AND each recovered only "
                    "its own -> no cross-group leakage",
                    flush=True,
                )
            passed += 1
        else:
            if rank == 0:
                print("  [FAIL] isolation check", flush=True)
            failed += 1

        dist.barrier()
        if rank == 0:
            total = passed + failed
            print(f"\n{'=' * 60}")
            print(f"RESULTS: {passed}/{total} passed, {failed}/{total} failed")
            print(
                "SUCCESS: dp=2 tp=2 ep=4 intra-group KV switch validated!"
                if failed == 0
                else "FAILED"
            )
            print(f"{'=' * 60}", flush=True)

        if failed > 0:
            dist.destroy_process_group()
            sys.exit(1)
    except Exception as e:
        print(f"[Rank {rank}] ERROR: {e}", flush=True)
        import traceback

        traceback.print_exc()
        try:
            dist.destroy_process_group()
        except Exception:
            pass
        sys.exit(1)

    dist.destroy_process_group()
    sys.exit(0)


if __name__ == "__main__":
    main()
