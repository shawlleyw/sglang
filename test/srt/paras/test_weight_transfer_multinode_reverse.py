#!/usr/bin/env python3
"""Validate node-local TP->EP resharding for a logical DP2 x TP4 layout.

Four GPUs represent one physical TP4 group at a time. The test runs both
logical DP ranks. Each group starts from a full, replicated TP tensor, selects
only its E/G expert interval, and reshards that interval into the four EP
destinations. Both the production v2 kernel and the v3 direct-access kernel
must reproduce the expected global EP shards bitwise.

Usage:
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
      test/srt/paras/test_weight_transfer_multinode_reverse.py
"""

import os
import sys

import torch
import torch.distributed as dist

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.join(_TEST_DIR, "..", "..", "..")
sys.path.insert(0, os.path.join(_ROOT_DIR, "python"))

from sglang.srt.paras.peer_access import (  # noqa: E402
    exchange_buffer_addresses_ipc,
    peer_access_fused_transfer_w13_ep,
    peer_access_fused_transfer_w2_ep,
)

TP_SIZE = 4
DP_SIZE = 2
EP_SIZE = TP_SIZE * DP_SIZE
NUM_EXPERTS = 64
EXPERTS_PER_EP_RANK = NUM_EXPERTS // EP_SIZE
HIDDEN = 2048
INTERMEDIATE = 768
INTERMEDIATE_PER_TP = INTERMEDIATE // TP_SIZE
NUM_GATES = 2
ELEM_SIZE = 2

W13_CHUNK_ELEMS = INTERMEDIATE_PER_TP * HIDDEN
W13_EXPERT_BYTES = NUM_GATES * W13_CHUNK_ELEMS * ELEM_SIZE
W13_TP_BYTES = NUM_EXPERTS * W13_EXPERT_BYTES
W13_EP_BYTES = EXPERTS_PER_EP_RANK * NUM_GATES * INTERMEDIATE * HIDDEN * ELEM_SIZE

W2_EXPERT_BYTES = HIDDEN * INTERMEDIATE_PER_TP * ELEM_SIZE
W2_TP_BYTES = NUM_EXPERTS * W2_EXPERT_BYTES
W2_EP_BYTES = EXPERTS_PER_EP_RANK * HIDDEN * INTERMEDIATE * ELEM_SIZE

W13_TP_OFFSET = 0
W13_V2_EP_OFFSET = W13_TP_OFFSET + W13_TP_BYTES
W13_V3_EP_OFFSET = W13_V2_EP_OFFSET + W13_EP_BYTES
W2_TP_OFFSET = W13_V3_EP_OFFSET + W13_EP_BYTES
W2_V2_EP_OFFSET = W2_TP_OFFSET + W2_TP_BYTES
W2_V3_EP_OFFSET = W2_V2_EP_OFFSET + W2_EP_BYTES
ARENA_BYTES = W2_V3_EP_OFFSET + W2_EP_BYTES


def _barrier(device):
    value = torch.zeros(1, device=device)
    dist.all_reduce(value)


def _pattern(indices):
    return ((indices * 17 + indices // 251) & 0x7FFF).to(torch.int16)


def _fill_tp_w13(buffer, tp_rank):
    expert = torch.arange(NUM_EXPERTS, dtype=torch.int32, device=buffer.device).view(
        NUM_EXPERTS, 1, 1
    )
    gate = torch.arange(NUM_GATES, dtype=torch.int32, device=buffer.device).view(
        1, NUM_GATES, 1
    )
    element = torch.arange(
        W13_CHUNK_ELEMS, dtype=torch.int32, device=buffer.device
    ).view(1, 1, W13_CHUNK_ELEMS)
    linear = (
        expert * (NUM_GATES * TP_SIZE * W13_CHUNK_ELEMS)
        + gate * (TP_SIZE * W13_CHUNK_ELEMS)
        + tp_rank * W13_CHUNK_ELEMS
        + element
    )
    view = (
        buffer[W13_TP_OFFSET : W13_TP_OFFSET + W13_TP_BYTES]
        .view(torch.int16)
        .view(NUM_EXPERTS, NUM_GATES, W13_CHUNK_ELEMS)
    )
    view.copy_(_pattern(linear))


def _fill_tp_w2(buffer, tp_rank):
    expert = torch.arange(NUM_EXPERTS, dtype=torch.int32, device=buffer.device).view(
        NUM_EXPERTS, 1, 1
    )
    row = torch.arange(HIDDEN, dtype=torch.int32, device=buffer.device).view(
        1, HIDDEN, 1
    )
    column = torch.arange(
        INTERMEDIATE_PER_TP, dtype=torch.int32, device=buffer.device
    ).view(1, 1, INTERMEDIATE_PER_TP)
    linear = (
        expert * (HIDDEN * INTERMEDIATE)
        + row * INTERMEDIATE
        + tp_rank * INTERMEDIATE_PER_TP
        + column
    )
    view = (
        buffer[W2_TP_OFFSET : W2_TP_OFFSET + W2_TP_BYTES]
        .view(torch.int16)
        .view(NUM_EXPERTS, HIDDEN, INTERMEDIATE_PER_TP)
    )
    view.copy_(_pattern(linear))


def _expected_w13(device, ep_rank):
    elements_per_ep_rank = EXPERTS_PER_EP_RANK * NUM_GATES * TP_SIZE * W13_CHUNK_ELEMS
    indices = torch.arange(elements_per_ep_rank, dtype=torch.int32, device=device)
    indices += ep_rank * elements_per_ep_rank
    return _pattern(indices).view(
        EXPERTS_PER_EP_RANK,
        NUM_GATES,
        TP_SIZE,
        W13_CHUNK_ELEMS,
    )


def _expected_w2(device, ep_rank):
    elements_per_ep_rank = EXPERTS_PER_EP_RANK * HIDDEN * INTERMEDIATE
    indices = torch.arange(elements_per_ep_rank, dtype=torch.int32, device=device)
    indices += ep_rank * elements_per_ep_rank
    return _pattern(indices).view(EXPERTS_PER_EP_RANK, HIDDEN, INTERMEDIATE)


def _run_group(buffer, peer_ptrs, tp_rank, dp_rank):
    node_expert_start = dp_rank * TP_SIZE * EXPERTS_PER_EP_RANK
    ep_rank = dp_rank * TP_SIZE + tp_rank
    stream = torch.cuda.current_stream()

    buffer[W13_V2_EP_OFFSET : W13_V3_EP_OFFSET + W13_EP_BYTES].fill_(0xA5)
    buffer[W2_V2_EP_OFFSET : W2_V3_EP_OFFSET + W2_EP_BYTES].fill_(0xA5)
    _barrier(buffer.device)

    for variant, w13_ep_offset, w2_ep_offset in (
        ("v2", W13_V2_EP_OFFSET, W2_V2_EP_OFFSET),
        ("v3", W13_V3_EP_OFFSET, W2_V3_EP_OFFSET),
    ):
        peer_access_fused_transfer_w13_ep(
            buffer.data_ptr(),
            peer_ptrs,
            W13_TP_OFFSET + node_expert_start * W13_EXPERT_BYTES,
            w13_ep_offset,
            tp_rank,
            TP_SIZE,
            EXPERTS_PER_EP_RANK,
            W13_CHUNK_ELEMS,
            num_gates=NUM_GATES,
            elem_size=ELEM_SIZE,
            stream=stream,
            variant=variant,
            hidden_size=HIDDEN,
        )
        peer_access_fused_transfer_w2_ep(
            buffer.data_ptr(),
            peer_ptrs,
            W2_TP_OFFSET + node_expert_start * W2_EXPERT_BYTES,
            w2_ep_offset,
            tp_rank,
            TP_SIZE,
            EXPERTS_PER_EP_RANK,
            hidden_size=HIDDEN,
            full_intermediate=INTERMEDIATE,
            tp_intermediate=INTERMEDIATE_PER_TP,
            elem_size=ELEM_SIZE,
            stream=stream,
            variant=variant,
        )

    torch.cuda.synchronize(buffer.device)
    _barrier(buffer.device)

    expected_w13 = _expected_w13(buffer.device, ep_rank)
    expected_w2 = _expected_w2(buffer.device, ep_rank)
    for variant, w13_ep_offset, w2_ep_offset in (
        ("v2", W13_V2_EP_OFFSET, W2_V2_EP_OFFSET),
        ("v3", W13_V3_EP_OFFSET, W2_V3_EP_OFFSET),
    ):
        actual_w13 = (
            buffer[w13_ep_offset : w13_ep_offset + W13_EP_BYTES]
            .view(torch.int16)
            .view_as(expected_w13)
        )
        actual_w2 = (
            buffer[w2_ep_offset : w2_ep_offset + W2_EP_BYTES]
            .view(torch.int16)
            .view_as(expected_w2)
        )
        assert torch.equal(actual_w13, expected_w13), (
            f"{variant} w13 mismatch for logical dp_rank={dp_rank}, "
            f"tp_rank={tp_rank}"
        )
        assert torch.equal(actual_w2, expected_w2), (
            f"{variant} w2 mismatch for logical dp_rank={dp_rank}, "
            f"tp_rank={tp_rank}"
        )


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    assert world_size == TP_SIZE, f"requires exactly {TP_SIZE} GPUs, got {world_size}"
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    buffer = torch.empty(ARENA_BYTES, dtype=torch.uint8, device=device)
    peer_addresses = exchange_buffer_addresses_ipc(
        buffer.data_ptr(), dist.group.WORLD, TP_SIZE, rank
    )
    peer_ptrs = torch.tensor(peer_addresses, dtype=torch.int64, device=device)

    _fill_tp_w13(buffer, rank)
    _fill_tp_w2(buffer, rank)
    _barrier(device)

    for dp_rank in range(DP_SIZE):
        _run_group(buffer, peer_ptrs, rank, dp_rank)
        if rank == 0:
            print(
                f"[OK] logical DP rank {dp_rank}: v2/v3 dropped "
                "non-owned experts and reconstructed EP bitwise",
                flush=True,
            )

    if rank == 0:
        print("SUCCESS: DP2 x TP4 multi-node TP->EP validated")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
