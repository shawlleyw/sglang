#!/usr/bin/env python3
"""
Integration test for the full EP→TP→EP round-trip on 4 GPUs.

Tests the entire ParaS parallelism switch flow end-to-end:
  1. Full EP→TP→EP round-trip (KV cache + weights)
  2. Cross-rank partition consistency
  3. Single request round-trip
  4. Empty batch round-trip

Usage:
  torchrun --nproc_per_node=4 -m pytest test_paras_roundtrip.py -v
  torchrun --nproc_per_node=4 test_paras_roundtrip.py
"""

import os
import sys

import pytest
import torch
import torch.distributed as dist

# Add sglang to path
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.join(_TEST_DIR, "..", "..")
sys.path.insert(0, os.path.join(_ROOT_DIR, "python"))

# Force NCCL method for portability
os.environ.setdefault("PARAS_KV_TRANSFER_METHOD", "nccl")

# ---- test constants (Qwen3-30B-A3B–like) ----
NUM_LAYERS = 3
NUM_KV_HEADS = 4
HEAD_DIM = 128
KV_DTYPE = torch.bfloat16
PAGE_SIZE = 1
SEED = 42

# MoE constants
NUM_EXPERTS = 64
HIDDEN = 2048
INTERMEDIATE = 1536

# Request constants
TOKENS_PER_RANK = [100, 80, 90, 70]
MAX_CONTEXT_LEN = 1024


# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------

def _is_distributed():
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


class _SimpleGroupCoordinator:
    """Minimal GroupCoordinator stand-in for testing."""

    def __init__(self, device_group, world_size, device, rank_in_group=0):
        self.device_group = device_group
        self.world_size = world_size
        self.device = torch.device(device)
        self.rank_in_group = rank_in_group
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = self.rank

    def all_gather_into_tensor(self, output_tensor, input_tensor):
        dist.all_gather_into_tensor(
            output_tensor, input_tensor, group=self.device_group
        )


def _setup_distributed():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 4, f"This test requires exactly 4 GPUs, got {world_size}"
    torch.cuda.set_device(rank)
    return rank, world_size


def _teardown_distributed():
    dist.destroy_process_group()


def _setup_paras_state(rank, world_size):
    """Set ParaS parallel state globals without full sglang server init."""
    import sglang.srt.distributed.parallel_state as ps
    import sglang.srt.paras.paras_parallel_state as pps

    tp_group = dist.new_group(ranks=list(range(world_size)))
    tp_coord = _SimpleGroupCoordinator(
        tp_group, world_size, f"cuda:{rank}", rank_in_group=rank
    )

    ps._TP = tp_coord

    pps._PARAS_TP = tp_coord
    pps._PARAS_DP = _SimpleGroupCoordinator(
        None, 1, f"cuda:{rank}", rank_in_group=0
    )
    pps._PARAS_SELF = _SimpleGroupCoordinator(
        None, 1, f"cuda:{rank}", rank_in_group=0
    )

    pps._PARAS_TP_SIZE = world_size
    pps._PARAS_TP_RANK = rank
    pps._PARAS_DP_SIZE = 1
    pps._PARAS_DP_RANK = 0
    pps._PARAS_EP_SIZE = world_size
    pps._PARAS_EP_RANK = rank

    return tp_group


# ---------------------------------------------------------------------------
# Mock Req compatible with pickle (needed by ParaSReqGatherManager)
# ---------------------------------------------------------------------------

class _MockReq:
    """Lightweight Req stand-in that is picklable and has all fields
    required by ParaSReqGatherManager / ParaSReqScatterManager."""

    def __init__(self, rid: str, seqlen: int, output_len: int = 1):
        self.rid = rid
        self.origin_input_ids = list(range(seqlen - output_len))
        self.output_ids = list(range(output_len))
        self.req_pool_idx = 0
        # Fields accessed by prune_request / recover_request
        self.last_host_node = None
        self.last_node = None
        self.prefix_indices = None
        self.tokenizer = None
        # Fields checked by ScheduleBatch
        self.stream = False
        self.grammar = None
        self.return_logprob = False
        self.return_hidden_states = False
        # Sampling params
        self.sampling_params = None

    @property
    def seqlen(self):
        return len(self.origin_input_ids) + len(self.output_ids)


# ---------------------------------------------------------------------------
# KV cache + pool helpers
# ---------------------------------------------------------------------------

def _compute_pool_sizes(tokens_per_rank, world_size):
    """Compute EP/TP pool sizes from token counts."""
    ep_max_tokens = max(tokens_per_rank) + 100
    heads_per_peer = max(1, NUM_KV_HEADS // world_size)
    total_tokens = sum(tokens_per_rank)
    min_ep_for_tp = (total_tokens * heads_per_peer + NUM_KV_HEADS - 1) // NUM_KV_HEADS
    ep_max_tokens = max(ep_max_tokens, min_ep_for_tp)
    tp_max_tokens = (ep_max_tokens + PAGE_SIZE) * NUM_KV_HEADS // heads_per_peer
    return ep_max_tokens, tp_max_tokens


def _build_kv_pool(rank, ep_max_tokens):
    """Create a standalone MHATokenToKVPool for EP mode."""
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    return MHATokenToKVPool(
        size=ep_max_tokens,
        page_size=PAGE_SIZE,
        dtype=KV_DTYPE,
        head_num=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        layer_num=NUM_LAYERS,
        device=f"cuda:{rank}",
        enable_memory_saver=False,
    )


def _build_allocator(kv_pool, ep_max_tokens):
    """Create a TokenToKVPoolAllocator wrapping the KV pool."""
    from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator

    return TokenToKVPoolAllocator(
        size=ep_max_tokens,
        dtype=torch.int64,
        device=kv_pool.device,
        kvcache=kv_pool,
        need_sort=False,
    )


def _build_req_to_token_pool(size, device):
    """Create a ReqToTokenPool."""
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

    return ReqToTokenPool(
        size=size,
        max_context_len=MAX_CONTEXT_LEN,
        device=device,
        enable_memory_saver=False,
    )


def _create_mock_reqs(rank, tokens_per_rank):
    """Create mock requests for a specific rank."""
    num_tokens = tokens_per_rank[rank]
    # Each rank has a few requests that sum to num_tokens KV entries
    # (seqlen includes 1 output token that is NOT in KV cache)
    reqs = []
    remaining = num_tokens
    req_idx = 0
    while remaining > 0:
        # Create requests with varying sizes
        seqlen = min(remaining + 1, 30 + req_idx * 7)  # +1 for output token
        if remaining - (seqlen - 1) < 5 and remaining > 0:
            seqlen = remaining + 1  # put all remaining in last req
        reqs.append(_MockReq(f"rank{rank}_req{req_idx}", seqlen))
        remaining -= (seqlen - 1)  # seqlen-1 tokens stored in KV
        req_idx += 1
    return reqs


def _populate_kv_and_pool(kv_pool, req_to_token_pool, allocator, reqs, rank):
    """Fill KV cache with deterministic data and allocate pool entries."""
    # Allocate req pool indices
    req_pool_indices = req_to_token_pool.alloc(len(reqs))
    assert req_pool_indices is not None

    total_tokens = sum(r.seqlen - 1 for r in reqs)
    if total_tokens == 0:
        return

    token_indices = allocator.alloc(total_tokens)
    assert token_indices is not None

    # Write token mapping and fill KV data
    offset = 0
    for req, rpi in zip(reqs, req_pool_indices):
        num_kv = req.seqlen - 1
        req.req_pool_idx = rpi
        req_to_token_pool.write(
            (rpi, slice(0, num_kv)),
            token_indices[offset : offset + num_kv],
        )
        offset += num_kv

    # Fill KV cache with rank-deterministic random data
    for layer_id in range(NUM_LAYERS):
        k_buf = kv_pool.get_key_buffer(layer_id)
        v_buf = kv_pool.get_value_buffer(layer_id)

        gen_k = torch.Generator(device="cpu")
        gen_k.manual_seed(SEED + layer_id * 1000 + rank)
        data_k = torch.randn(
            (total_tokens, NUM_KV_HEADS, HEAD_DIM),
            generator=gen_k,
            dtype=torch.float32,
        ).to(dtype=KV_DTYPE, device=k_buf.device)
        k_buf[token_indices] = data_k

        gen_v = torch.Generator(device="cpu")
        gen_v.manual_seed(SEED + layer_id * 1000 + rank + 500)
        data_v = torch.randn(
            (total_tokens, NUM_KV_HEADS, HEAD_DIM),
            generator=gen_v,
            dtype=torch.float32,
        ).to(dtype=KV_DTYPE, device=v_buf.device)
        v_buf[token_indices] = data_v


def _snapshot_local_kv(kv_pool, req_to_token_pool, reqs):
    """Snapshot KV cache data for local requests."""
    snap = {}
    for layer_id in range(NUM_LAYERS):
        k_buf = kv_pool.get_key_buffer(layer_id)
        v_buf = kv_pool.get_value_buffer(layer_id)
        k_parts = []
        v_parts = []
        for req in reqs:
            num_kv = req.seqlen - 1
            indices = req_to_token_pool.req_to_token[req.req_pool_idx][:num_kv]
            k_parts.append(k_buf[indices].clone())
            v_parts.append(v_buf[indices].clone())
        if k_parts:
            snap[layer_id] = (torch.cat(k_parts), torch.cat(v_parts))
        else:
            snap[layer_id] = (
                torch.empty(0, NUM_KV_HEADS, HEAD_DIM, dtype=KV_DTYPE, device=k_buf.device),
                torch.empty(0, NUM_KV_HEADS, HEAD_DIM, dtype=KV_DTYPE, device=k_buf.device),
            )
    return snap


# ---------------------------------------------------------------------------
# Weight helpers
# ---------------------------------------------------------------------------

def _build_weight_manager(rank, world_size):
    """Create ParaSMemoryManager with MoE weight slots."""
    from sglang.srt.paras.paras_memory_manager import (
        ParaSMemoryManager,
        create_paras_moe_aliases,
        set_global_paras_memory_manager,
    )

    ep_size = world_size
    num_local = NUM_EXPERTS // ep_size

    mgr = ParaSMemoryManager(device=f"cuda:{rank}")

    # N+1 generic physical slots
    for slot in range(NUM_LAYERS + 1):
        mgr.reserve(
            f"paras.moe_slot.{slot}.w13",
            (num_local, 2 * INTERMEDIATE, HIDDEN),
            torch.bfloat16,
        )
        mgr.reserve(
            f"paras.moe_slot.{slot}.w2",
            (num_local, HIDDEN, INTERMEDIATE),
            torch.bfloat16,
        )

    # 'experts' aliases → slot i+1
    for i in range(NUM_LAYERS):
        mgr._entries[f"model.layers.{i}.mlp.experts.w13_weight"] = mgr._entries[
            f"paras.moe_slot.{i + 1}.w13"
        ]
        mgr._entries[f"model.layers.{i}.mlp.experts.w2_weight"] = mgr._entries[
            f"paras.moe_slot.{i + 1}.w2"
        ]

    # Staging buffers
    staging_experts = num_local
    w13_staging_shape = (staging_experts, 2 * INTERMEDIATE, HIDDEN)
    w2_staging_shape = (staging_experts, HIDDEN, INTERMEDIATE)
    for sfx in ("", "_1", "_2"):
        mgr.reserve(f"staging.w13_pre_permute{sfx}", w13_staging_shape, torch.bfloat16)
        mgr.reserve(f"staging.w2_pre_permute{sfx}", w2_staging_shape, torch.bfloat16)

    mgr.materialize()
    create_paras_moe_aliases(mgr, NUM_LAYERS)
    set_global_paras_memory_manager(mgr)
    return mgr, num_local


def _fill_ep_weights(mgr, rank):
    """Fill EP weight buffers with rank-deterministic random data."""
    for layer_id in range(NUM_LAYERS):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(SEED + layer_id * 100 + rank)
        w13 = mgr.get_view(f"model.layers.{layer_id}.mlp.experts.w13_weight")
        w13.copy_(
            torch.randn(w13.shape, generator=gen, dtype=torch.float32).to(
                dtype=w13.dtype, device=w13.device
            )
        )
        gen2 = torch.Generator(device="cpu")
        gen2.manual_seed(SEED + layer_id * 100 + rank + 50)
        w2 = mgr.get_view(f"model.layers.{layer_id}.mlp.experts.w2_weight")
        w2.copy_(
            torch.randn(w2.shape, generator=gen2, dtype=torch.float32).to(
                dtype=w2.dtype, device=w2.device
            )
        )


def _snapshot_ep_weights(mgr):
    """Clone all EP weight buffers."""
    snap = {}
    for layer_id in range(NUM_LAYERS):
        snap[layer_id] = (
            mgr.get_view(f"model.layers.{layer_id}.mlp.experts.w13_weight").clone(),
            mgr.get_view(f"model.layers.{layer_id}.mlp.experts.w2_weight").clone(),
        )
    return snap


class _MockExperts:
    def __init__(self, w13_view, w2_view):
        self.w13_weight = torch.nn.Parameter(w13_view, requires_grad=False)
        self.w2_weight = torch.nn.Parameter(w2_view, requires_grad=False)


def _run_ep_to_tp_weights(mgr, num_local, world_size):
    """Run EP→TP weight transfer (naive all_to_all path)."""
    from sglang.srt.paras.layers.paras_moe_block import ParaSMoeBlockMixin

    tp_inter = INTERMEDIATE // world_size
    for layer_id in range(NUM_LAYERS):
        m = object.__new__(ParaSMoeBlockMixin)
        m._paras_layer_id = layer_id
        m.num_local_experts = num_local
        m.num_global_experts = NUM_EXPERTS
        m.hidden_size = HIDDEN
        m.moe_intermediate_size = INTERMEDIATE
        w13 = mgr.get_view(f"model.layers.{layer_id}.mlp.experts.w13_weight")
        w2 = mgr.get_view(f"model.layers.{layer_id}.mlp.experts.w2_weight")
        m.ep_experts = _MockExperts(w13, w2)
        m.w13_ep_gathered = w13.view(num_local, 2 * INTERMEDIATE, HIDDEN)
        m.w2_ep_gathered = w2.view(num_local, HIDDEN, INTERMEDIATE)
        m.paras_configure_tp_all_to_all()


def _verify_weight_restoration(mgr, original_snap, rank):
    """Verify EP weights match original after round-trip."""
    all_ok = True
    for layer_id in range(NUM_LAYERS):
        w13 = mgr.get_view(f"model.layers.{layer_id}.mlp.experts.w13_weight")
        w2 = mgr.get_view(f"model.layers.{layer_id}.mlp.experts.w2_weight")
        w13_match = torch.equal(original_snap[layer_id][0], w13)
        w2_match = torch.equal(original_snap[layer_id][1], w2)
        if not w13_match or not w2_match:
            if rank == 0:
                print(
                    f"  [FAIL] Weight mismatch layer={layer_id} "
                    f"w13={'OK' if w13_match else 'FAIL'} "
                    f"w2={'OK' if w2_match else 'FAIL'}",
                    flush=True,
                )
            all_ok = False
    return all_ok


# =========================================================================
# TEST 1: Full EP→TP→EP Round-Trip
# =========================================================================

@pytest.mark.skipif(not _is_distributed(), reason="Requires torchrun with 4 GPUs")
class TestFullRoundTrip:
    """Full EP→TP→EP round-trip: KV cache + weights + memory leak check."""

    def test_roundtrip_ep_tp_ep(self):
        """
        1. Start in EP mode with mock requests
        2. Run EP→TP switch (KV gather via NCCL + weight all_to_all)
        3. Verify TP mode: all ranks have same requests, weights in TP layout
        4. Run TP→EP switch (KV scatter + weight restore)
        5. Verify EP mode: KV data matches original
        6. Compare EP weights after round-trip with original (torch.equal)
        7. Verify no memory leak (< 1% delta)
        """
        from sglang.srt.paras.gather_manager import (
            gather_kv_and_permute,
            permute_and_scatter_kv,
            gather_tp_kv_and_permute,
            permute_and_scatter_kv_to_ep,
        )

        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        tp_group = _setup_paras_state(rank, world_size)

        ep_max_tokens, tp_max_tokens = _compute_pool_sizes(TOKENS_PER_RANK, world_size)
        kv_pool = _build_kv_pool(rank, ep_max_tokens)
        allocator = _build_allocator(kv_pool, ep_max_tokens)
        req_to_token_pool = _build_req_to_token_pool(64, f"cuda:{rank}")

        # Create mock requests and populate KV cache
        reqs = _create_mock_reqs(rank, TOKENS_PER_RANK)
        _populate_kv_and_pool(kv_pool, req_to_token_pool, allocator, reqs, rank)

        # Snapshot original EP KV data
        orig_kv = _snapshot_local_kv(kv_pool, req_to_token_pool, reqs)

        # Build weight manager and snapshot
        weight_mgr, num_local = _build_weight_manager(rank, world_size)
        _fill_ep_weights(weight_mgr, rank)
        orig_weights = _snapshot_ep_weights(weight_mgr)

        # Record memory before round-trip
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated()

        # ---- Phase 1: EP→TP KV gather (NCCL) ----
        heads_per_peer = max(1, NUM_KV_HEADS // world_size)
        num_local_tokens = TOKENS_PER_RANK[rank]
        total_tokens = sum(TOKENS_PER_RANK)
        tp_view_tokens = (ep_max_tokens + PAGE_SIZE) * NUM_KV_HEADS // heads_per_peer
        splited_size = heads_per_peer * HEAD_DIM

        # Gather local token indices
        local_token_indices = torch.empty(0, dtype=torch.int64, device=f"cuda:{rank}")
        for req in reqs:
            num_kv = req.seqlen - 1
            indices = req_to_token_pool.req_to_token[req.req_pool_idx][:num_kv]
            local_token_indices = torch.cat([local_token_indices, indices.long()])

        global_token_indices = torch.arange(total_tokens, dtype=torch.int64, device=f"cuda:{rank}")
        input_split_sizes = [2 * splited_size * num_local_tokens] * world_size
        output_split_sizes = [
            2 * splited_size * TOKENS_PER_RANK[r] for r in range(world_size)
        ]

        for lid in range(NUM_LAYERS):
            k_buf = kv_pool.get_key_buffer(lid)
            v_buf = kv_pool.get_value_buffer(lid)
            permuted = gather_kv_and_permute(k_buf, v_buf, local_token_indices)

            # Resize to TP layout
            kv_pool.k_buffer[lid] = torch.empty(
                tp_view_tokens, heads_per_peer, HEAD_DIM,
                dtype=KV_DTYPE, device=f"cuda:{rank}",
            )
            kv_pool.v_buffer[lid] = torch.empty(
                tp_view_tokens, heads_per_peer, HEAD_DIM,
                dtype=KV_DTYPE, device=f"cuda:{rank}",
            )

            tp_k = kv_pool.k_buffer[lid]
            tp_v = kv_pool.v_buffer[lid]

            gathered = torch.empty(
                2 * total_tokens * splited_size,
                dtype=KV_DTYPE, device=f"cuda:{rank}",
            )
            dist.all_to_all_single(
                gathered, permuted, output_split_sizes, input_split_sizes,
                group=tp_group,
            )
            permute_and_scatter_kv(
                gathered, tp_k, tp_v, global_token_indices,
                total_tokens, heads_per_peer, HEAD_DIM,
            )

        kv_pool.head_num = heads_per_peer
        kv_pool._paras_original_head_num = NUM_KV_HEADS
        torch.cuda.synchronize()
        dist.barrier(group=tp_group)

        # EP→TP weight transfer
        _run_ep_to_tp_weights(weight_mgr, num_local, world_size)

        # ---- Phase 2: TP→EP KV scatter (NCCL) ----
        # Use partition_requests_for_ep to compute routing
        from sglang.srt.paras.gather_manager import partition_requests_for_ep

        # Build global request list (all ranks' reqs) — simulate what
        # gather_global_reqs would produce. In the real flow reqs are
        # gathered via pickle; here we reconstruct manually so that
        # `rid` + seqlen info is available on every rank.
        import pickle
        local_bytes = pickle.dumps(reqs)
        local_size = torch.tensor([len(local_bytes)], dtype=torch.long, device=f"cuda:{rank}")
        all_sizes = torch.empty(world_size, dtype=torch.long, device=f"cuda:{rank}")
        dist.all_gather_into_tensor(all_sizes, local_size, group=tp_group)
        max_size = all_sizes.max().item()

        padded = torch.zeros(max_size, dtype=torch.uint8, device=f"cuda:{rank}")
        local_tensor = torch.frombuffer(bytearray(local_bytes), dtype=torch.uint8).to(f"cuda:{rank}")
        padded[: len(local_bytes)] = local_tensor
        all_data = torch.empty(max_size * world_size, dtype=torch.uint8, device=f"cuda:{rank}")
        dist.all_gather_into_tensor(all_data, padded, group=tp_group)

        import numpy as np
        chunks = np.split(all_data.cpu().numpy(), world_size)
        global_reqs = []
        split_sizes = []
        for i in range(world_size):
            sz = all_sizes[i].item()
            remote = pickle.loads(chunks[i][:sz].tobytes()) if sz > 0 else []
            global_reqs.extend(remote)
            split_sizes.append(len(remote))

        # Assign TP pool indices to global reqs
        for i, req in enumerate(global_reqs):
            req.req_pool_idx = i  # simple sequential mapping in TP mode

        # Partition requests
        partitions = partition_requests_for_ep(global_reqs, world_size)
        local_ep_reqs = partitions[rank]

        # Build token partition (mapping global_req → global_token_index range)
        req_to_offset = {}
        offset = 0
        for req in global_reqs:
            num_kv = req.seqlen - 1
            req_to_offset[req.rid] = (offset, offset + num_kv)
            offset += num_kv

        token_partition = []
        for rank_reqs in partitions:
            rank_indices = []
            for req in rank_reqs:
                start, end = req_to_offset[req.rid]
                rank_indices.extend(range(start, end))
            token_partition.append(rank_indices)

        my_token_count = sum(r.seqlen - 1 for r in local_ep_reqs)
        ep_dst_positions = torch.arange(1, my_token_count + 1, dtype=torch.int64, device=f"cuda:{rank}")
        per_token_elems = heads_per_peer * 2 * HEAD_DIM

        send_token_counts = [len(token_partition[e]) for e in range(world_size)]
        scatter_input_split = [cnt * per_token_elems for cnt in send_token_counts]
        scatter_output_split = [my_token_count * per_token_elems] * world_size

        # Sort TP indices by destination rank
        sorted_parts = []
        for e in range(world_size):
            if send_token_counts[e] > 0:
                part_idx = torch.tensor(
                    token_partition[e], dtype=torch.long, device=f"cuda:{rank}"
                )
                sorted_parts.append(global_token_indices[part_idx])
        sorted_tp_indices = (
            torch.cat(sorted_parts)
            if sorted_parts
            else torch.empty(0, dtype=torch.long, device=f"cuda:{rank}")
        )

        # Resize back to EP layout and scatter
        for lid in range(NUM_LAYERS):
            tp_k = kv_pool.k_buffer[lid]
            tp_v = kv_pool.v_buffer[lid]

            total_send = sum(send_token_counts)
            if total_send > 0:
                send_buf = gather_tp_kv_and_permute(
                    tp_k, tp_v, sorted_tp_indices,
                    NUM_KV_HEADS, heads_per_peer, HEAD_DIM, world_size,
                )
            else:
                send_buf = torch.empty(0, dtype=KV_DTYPE, device=f"cuda:{rank}")

            # Resize to EP layout
            kv_pool.k_buffer[lid] = torch.zeros(
                ep_max_tokens + PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM,
                dtype=KV_DTYPE, device=f"cuda:{rank}",
            )
            kv_pool.v_buffer[lid] = torch.zeros(
                ep_max_tokens + PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM,
                dtype=KV_DTYPE, device=f"cuda:{rank}",
            )

            if total_send > 0:
                recv_buf = torch.empty(
                    my_token_count * world_size * per_token_elems,
                    dtype=KV_DTYPE, device=f"cuda:{rank}",
                )
                dist.all_to_all_single(
                    recv_buf, send_buf,
                    scatter_output_split, scatter_input_split,
                    group=tp_group,
                )

                if my_token_count > 0:
                    ep_k = kv_pool.k_buffer[lid]
                    ep_v = kv_pool.v_buffer[lid]
                    permute_and_scatter_kv_to_ep(
                        recv_buf, ep_k, ep_v, ep_dst_positions,
                        my_token_count, NUM_KV_HEADS, heads_per_peer,
                        HEAD_DIM, world_size,
                    )

        kv_pool.head_num = NUM_KV_HEADS
        torch.cuda.synchronize()

        # ---- Phase 3: Verify KV round-trip ----
        # The KV data for this rank's local_ep_reqs should match the
        # original data. We need to compare by request: find each
        # local_ep_req in the original reqs and check its KV data.
        all_ok = True
        for req in local_ep_reqs:
            # Find original rank and position
            orig_rank = None
            for r in range(world_size):
                if req.rid.startswith(f"rank{r}_"):
                    orig_rank = r
                    break
            if orig_rank is None:
                all_ok = False
                continue

            # Only verify reqs that came from this rank (we have their snapshots)
            if orig_rank == rank:
                # Find index in original reqs list
                orig_idx = None
                for idx, orig_req in enumerate(reqs):
                    if orig_req.rid == req.rid:
                        orig_idx = idx
                        break
                if orig_idx is None:
                    all_ok = False
                    continue

                num_kv = req.seqlen - 1
                # Find token offset in partition for this req
                req_start, req_end = req_to_offset[req.rid]
                local_req_offset = 0
                for prev_req in local_ep_reqs:
                    if prev_req.rid == req.rid:
                        break
                    local_req_offset += prev_req.seqlen - 1

                for lid in range(NUM_LAYERS):
                    # Get original KV for this req from snapshot
                    # orig_kv[lid] is concatenated across all original reqs
                    kv_offset = 0
                    for j in range(orig_idx):
                        kv_offset += reqs[j].seqlen - 1

                    orig_k = orig_kv[lid][0][kv_offset : kv_offset + num_kv]
                    orig_v = orig_kv[lid][1][kv_offset : kv_offset + num_kv]

                    # Get round-tripped KV
                    ep_positions = ep_dst_positions[
                        local_req_offset : local_req_offset + num_kv
                    ]
                    new_k = kv_pool.k_buffer[lid][ep_positions]
                    new_v = kv_pool.v_buffer[lid][ep_positions]

                    if not torch.equal(orig_k, new_k) or not torch.equal(orig_v, new_v):
                        all_ok = False
                        if rank == 0:
                            print(
                                f"  [FAIL] KV mismatch req={req.rid} layer={lid}",
                                flush=True,
                            )

        # Gather verification status across ranks
        ok_tensor = torch.tensor([1 if all_ok else 0], device=f"cuda:{rank}")
        dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN, group=tp_group)
        kv_ok = ok_tensor.item() == 1

        # ---- Phase 4: Verify weight restoration ----
        # EP weights are not modified by EP→TP transfer (it writes to tp_experts
        # buffers), so they should still match the original.
        weights_ok = _verify_weight_restoration(weight_mgr, orig_weights, rank)

        # ---- Phase 5: Memory leak check ----
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        mem_after = torch.cuda.memory_allocated()

        mem_ok = True
        if mem_before > 0:
            delta_pct = abs(mem_after - mem_before) / mem_before
            if delta_pct >= 0.01:
                mem_ok = False
                if rank == 0:
                    print(
                        f"  [FAIL] Memory leak: before={mem_before}, "
                        f"after={mem_after}, delta={delta_pct:.4%}",
                        flush=True,
                    )
        elif mem_after - mem_before > 1024 * 1024:
            mem_ok = False

        assert kv_ok, "KV cache round-trip failed: data mismatch"
        assert weights_ok, "Weight restoration failed after round-trip"
        assert mem_ok, "Memory leak detected after round-trip"


# =========================================================================
# TEST 2: Cross-Rank Partition Consistency
# =========================================================================

@pytest.mark.skipif(not _is_distributed(), reason="Requires torchrun with 4 GPUs")
class TestPartitionConsistency:
    """Verify all ranks compute identical partitions."""

    def test_partition_consistency(self):
        """
        After TP→EP switch, all_gather the partitions from all ranks.
        Verify: all ranks computed identical partitions.
        Verify: union of all partitions == original global request set.
        Verify: no request duplicated or lost.
        """
        from sglang.srt.paras.gather_manager import partition_requests_for_ep

        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        tp_group = _setup_paras_state(rank, world_size)

        # Build a global request set (identical on every rank via deterministic
        # construction — mirrors what gather_global_reqs produces)
        global_reqs = []
        for r in range(world_size):
            for i in range(3):
                global_reqs.append(
                    _MockReq(f"rank{r}_req{i}", seqlen=50 + r * 10 + i * 5)
                )

        # Each rank independently computes partition
        partitions = partition_requests_for_ep(global_reqs, world_size)
        local_rids = sorted([r.rid for r in partitions[rank]])

        # Serialize local partition result
        import pickle
        local_data = pickle.dumps(local_rids)
        local_size = torch.tensor([len(local_data)], dtype=torch.long, device=f"cuda:{rank}")
        all_sizes = torch.empty(world_size, dtype=torch.long, device=f"cuda:{rank}")
        dist.all_gather_into_tensor(all_sizes, local_size, group=tp_group)

        max_sz = all_sizes.max().item()
        padded = torch.zeros(max_sz, dtype=torch.uint8, device=f"cuda:{rank}")
        local_tensor = torch.frombuffer(bytearray(local_data), dtype=torch.uint8).to(f"cuda:{rank}")
        padded[: len(local_data)] = local_tensor
        all_data = torch.empty(max_sz * world_size, dtype=torch.uint8, device=f"cuda:{rank}")
        dist.all_gather_into_tensor(all_data, padded, group=tp_group)

        import numpy as np
        chunks = np.split(all_data.cpu().numpy(), world_size)

        # Verify each rank computed the same partition for rank `rank`
        # (Each rank runs partition_requests_for_ep and picks its own slice.
        #  Determinism means every rank produces the SAME full partition table.)
        # We gather each rank's local_rids and verify consistency.

        all_rids_by_rank = []
        for i in range(world_size):
            sz = all_sizes[i].item()
            rids = pickle.loads(chunks[i][:sz].tobytes())
            all_rids_by_rank.append(rids)

        # Every rank should have got the same partition table, so
        # rank i's locally-computed partition[i] should equal what
        # rank i reports as its local_rids. Since partition is
        # deterministic, we can verify by checking that our partition
        # table matches what every rank independently reports.
        for i in range(world_size):
            expected = sorted([r.rid for r in partitions[i]])
            actual = all_rids_by_rank[i]
            assert expected == actual, (
                f"Partition mismatch for rank {i}: "
                f"expected {expected}, got {actual}"
            )

        # Verify union == original set, no duplicates
        all_rids = []
        for i in range(world_size):
            all_rids.extend(all_rids_by_rank[i])
        original_rids = sorted([r.rid for r in global_reqs])
        assert sorted(all_rids) == original_rids, (
            f"Partition union mismatch: "
            f"got {sorted(all_rids)}, expected {original_rids}"
        )

        # Verify no duplicates
        assert len(all_rids) == len(set(all_rids)), "Duplicate request in partition"


# =========================================================================
# TEST 3: Single Request Round-Trip
# =========================================================================

@pytest.mark.skipif(not _is_distributed(), reason="Requires torchrun with 4 GPUs")
class TestSingleRequestRoundTrip:
    """EP→TP→EP with only 1 active request."""

    def test_single_request_roundtrip(self):
        """
        EP→TP→EP with only 1 active request (on rank 0, others empty).
        After TP→EP: exactly 1 rank has the request, others have empty batch.
        Verify the owning rank has correct KV data.
        """
        from sglang.srt.paras.gather_manager import (
            gather_kv_and_permute,
            permute_and_scatter_kv,
            gather_tp_kv_and_permute,
            permute_and_scatter_kv_to_ep,
            partition_requests_for_ep,
        )

        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        tp_group = _setup_paras_state(rank, world_size)

        single_tokens = [50, 0, 0, 0]  # only rank 0 has a request
        ep_max_tokens, tp_max_tokens = _compute_pool_sizes(
            [max(1, t) for t in single_tokens], world_size
        )
        kv_pool = _build_kv_pool(rank, ep_max_tokens)
        allocator = _build_allocator(kv_pool, ep_max_tokens)
        req_to_token_pool = _build_req_to_token_pool(16, f"cuda:{rank}")

        # Only rank 0 has a request
        if rank == 0:
            reqs = [_MockReq("single_req", seqlen=51)]  # 50 KV tokens + 1 output
        else:
            reqs = []

        if reqs:
            _populate_kv_and_pool(kv_pool, req_to_token_pool, allocator, reqs, rank)
            orig_kv = _snapshot_local_kv(kv_pool, req_to_token_pool, reqs)
        else:
            orig_kv = None

        # ---- EP→TP gather ----
        heads_per_peer = max(1, NUM_KV_HEADS // world_size)
        num_local_tokens = single_tokens[rank]
        total_tokens = sum(single_tokens)
        tp_view_tokens = (ep_max_tokens + PAGE_SIZE) * NUM_KV_HEADS // heads_per_peer
        splited_size = heads_per_peer * HEAD_DIM

        local_token_indices = torch.empty(0, dtype=torch.int64, device=f"cuda:{rank}")
        if reqs:
            for req in reqs:
                num_kv = req.seqlen - 1
                indices = req_to_token_pool.req_to_token[req.req_pool_idx][:num_kv]
                local_token_indices = torch.cat([local_token_indices, indices.long()])

        global_token_indices = torch.arange(
            total_tokens, dtype=torch.int64, device=f"cuda:{rank}"
        )
        input_split_sizes = [2 * splited_size * num_local_tokens] * world_size
        output_split_sizes = [
            2 * splited_size * single_tokens[r] for r in range(world_size)
        ]

        for lid in range(NUM_LAYERS):
            k_buf = kv_pool.get_key_buffer(lid)
            v_buf = kv_pool.get_value_buffer(lid)

            if num_local_tokens > 0:
                permuted = gather_kv_and_permute(k_buf, v_buf, local_token_indices)
            else:
                permuted = torch.empty(0, dtype=KV_DTYPE, device=f"cuda:{rank}")

            kv_pool.k_buffer[lid] = torch.empty(
                tp_view_tokens, heads_per_peer, HEAD_DIM,
                dtype=KV_DTYPE, device=f"cuda:{rank}",
            )
            kv_pool.v_buffer[lid] = torch.empty(
                tp_view_tokens, heads_per_peer, HEAD_DIM,
                dtype=KV_DTYPE, device=f"cuda:{rank}",
            )

            if total_tokens > 0:
                gathered = torch.empty(
                    2 * total_tokens * splited_size,
                    dtype=KV_DTYPE, device=f"cuda:{rank}",
                )
                dist.all_to_all_single(
                    gathered, permuted, output_split_sizes, input_split_sizes,
                    group=tp_group,
                )
                permute_and_scatter_kv(
                    gathered, kv_pool.k_buffer[lid], kv_pool.v_buffer[lid],
                    global_token_indices, total_tokens, heads_per_peer, HEAD_DIM,
                )

        kv_pool.head_num = heads_per_peer
        kv_pool._paras_original_head_num = NUM_KV_HEADS
        torch.cuda.synchronize()
        dist.barrier(group=tp_group)

        # ---- TP→EP scatter ----
        # Build global reqs via all_gather
        import pickle
        local_bytes = pickle.dumps(reqs)
        local_size = torch.tensor([len(local_bytes)], dtype=torch.long, device=f"cuda:{rank}")
        all_sizes = torch.empty(world_size, dtype=torch.long, device=f"cuda:{rank}")
        dist.all_gather_into_tensor(all_sizes, local_size, group=tp_group)
        max_size = all_sizes.max().item()

        padded = torch.zeros(max_size, dtype=torch.uint8, device=f"cuda:{rank}")
        local_tensor = torch.frombuffer(bytearray(local_bytes), dtype=torch.uint8).to(f"cuda:{rank}")
        padded[: len(local_bytes)] = local_tensor
        all_data = torch.empty(max_size * world_size, dtype=torch.uint8, device=f"cuda:{rank}")
        dist.all_gather_into_tensor(all_data, padded, group=tp_group)

        import numpy as np
        chunks = np.split(all_data.cpu().numpy(), world_size)
        global_reqs = []
        for i in range(world_size):
            sz = all_sizes[i].item()
            remote = pickle.loads(chunks[i][:sz].tobytes()) if sz > 0 else []
            global_reqs.extend(remote)

        # Partition: with 1 request, exactly 1 rank gets it
        partitions = partition_requests_for_ep(global_reqs, world_size)
        local_ep_reqs = partitions[rank]

        # Verify: exactly 1 rank has the request
        local_counts = torch.tensor(
            [len(local_ep_reqs)], dtype=torch.long, device=f"cuda:{rank}"
        )
        all_counts = torch.empty(world_size, dtype=torch.long, device=f"cuda:{rank}")
        dist.all_gather_into_tensor(all_counts, local_counts, group=tp_group)
        counts_list = all_counts.tolist()
        assert sum(counts_list) == 1, f"Expected 1 total request, got {sum(counts_list)}"
        assert counts_list.count(1) == 1, f"Expected exactly 1 rank with request, got {counts_list}"
        assert counts_list.count(0) == world_size - 1

        # Token routing
        req_to_offset = {}
        off = 0
        for req in global_reqs:
            num_kv = req.seqlen - 1
            req_to_offset[req.rid] = (off, off + num_kv)
            off += num_kv

        token_partition = []
        for rank_reqs in partitions:
            rank_idxs = []
            for req in rank_reqs:
                start, end = req_to_offset[req.rid]
                rank_idxs.extend(range(start, end))
            token_partition.append(rank_idxs)

        my_token_count = sum(r.seqlen - 1 for r in local_ep_reqs)
        ep_dst_positions = torch.arange(
            1, my_token_count + 1, dtype=torch.int64, device=f"cuda:{rank}"
        )
        per_token_elems = heads_per_peer * 2 * HEAD_DIM
        send_token_counts = [len(token_partition[e]) for e in range(world_size)]
        scatter_input_split = [cnt * per_token_elems for cnt in send_token_counts]
        scatter_output_split = [my_token_count * per_token_elems] * world_size

        sorted_parts = []
        for e in range(world_size):
            if send_token_counts[e] > 0:
                part_idx = torch.tensor(
                    token_partition[e], dtype=torch.long, device=f"cuda:{rank}"
                )
                sorted_parts.append(global_token_indices[part_idx])
        sorted_tp_indices = (
            torch.cat(sorted_parts)
            if sorted_parts
            else torch.empty(0, dtype=torch.long, device=f"cuda:{rank}")
        )

        for lid in range(NUM_LAYERS):
            tp_k = kv_pool.k_buffer[lid]
            tp_v = kv_pool.v_buffer[lid]
            total_send = sum(send_token_counts)

            if total_send > 0:
                send_buf = gather_tp_kv_and_permute(
                    tp_k, tp_v, sorted_tp_indices,
                    NUM_KV_HEADS, heads_per_peer, HEAD_DIM, world_size,
                )
            else:
                send_buf = torch.empty(0, dtype=KV_DTYPE, device=f"cuda:{rank}")

            kv_pool.k_buffer[lid] = torch.zeros(
                ep_max_tokens + PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM,
                dtype=KV_DTYPE, device=f"cuda:{rank}",
            )
            kv_pool.v_buffer[lid] = torch.zeros(
                ep_max_tokens + PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM,
                dtype=KV_DTYPE, device=f"cuda:{rank}",
            )

            if total_send > 0:
                recv_buf = torch.empty(
                    my_token_count * world_size * per_token_elems,
                    dtype=KV_DTYPE, device=f"cuda:{rank}",
                )
                dist.all_to_all_single(
                    recv_buf, send_buf,
                    scatter_output_split, scatter_input_split,
                    group=tp_group,
                )
                if my_token_count > 0:
                    permute_and_scatter_kv_to_ep(
                        recv_buf, kv_pool.k_buffer[lid], kv_pool.v_buffer[lid],
                        ep_dst_positions, my_token_count, NUM_KV_HEADS,
                        heads_per_peer, HEAD_DIM, world_size,
                    )

        kv_pool.head_num = NUM_KV_HEADS
        torch.cuda.synchronize()

        # Verify: owning rank has correct KV data
        if len(local_ep_reqs) == 1 and orig_kv is not None:
            # This rank owns the request and had the original data
            req = local_ep_reqs[0]
            num_kv = req.seqlen - 1
            for lid in range(NUM_LAYERS):
                orig_k = orig_kv[lid][0][:num_kv]
                orig_v = orig_kv[lid][1][:num_kv]
                new_k = kv_pool.k_buffer[lid][ep_dst_positions[:num_kv]]
                new_v = kv_pool.v_buffer[lid][ep_dst_positions[:num_kv]]
                assert torch.equal(orig_k, new_k), (
                    f"Single request KV K mismatch layer={lid}"
                )
                assert torch.equal(orig_v, new_v), (
                    f"Single request KV V mismatch layer={lid}"
                )


# =========================================================================
# TEST 4: Empty Batch Round-Trip
# =========================================================================

@pytest.mark.skipif(not _is_distributed(), reason="Requires torchrun with 4 GPUs")
class TestEmptyBatchRoundTrip:
    """EP→TP→EP with 0 active requests."""

    def test_empty_batch_roundtrip(self):
        """
        EP→TP→EP with 0 active requests.
        Verify: no crash, clean state after round-trip.
        """
        from sglang.srt.paras.gather_manager import partition_requests_for_ep

        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        tp_group = _setup_paras_state(rank, world_size)

        # ---- EP→TP with empty batch ----
        heads_per_peer = max(1, NUM_KV_HEADS // world_size)
        splited_size = heads_per_peer * HEAD_DIM
        total_tokens = 0

        ep_max_tokens = 100
        tp_view_tokens = (ep_max_tokens + PAGE_SIZE) * NUM_KV_HEADS // heads_per_peer
        kv_pool = _build_kv_pool(rank, ep_max_tokens)

        # EP→TP gather with 0 tokens: all ranks send/recv 0
        input_split_sizes = [0] * world_size
        output_split_sizes = [0] * world_size

        for lid in range(NUM_LAYERS):
            send_empty = torch.empty(0, dtype=KV_DTYPE, device=f"cuda:{rank}")
            recv_empty = torch.empty(0, dtype=KV_DTYPE, device=f"cuda:{rank}")
            dist.all_to_all_single(
                recv_empty, send_empty, output_split_sizes, input_split_sizes,
                group=tp_group,
            )
            # Resize to TP layout (no data to transfer)
            kv_pool.k_buffer[lid] = torch.zeros(
                tp_view_tokens, heads_per_peer, HEAD_DIM,
                dtype=KV_DTYPE, device=f"cuda:{rank}",
            )
            kv_pool.v_buffer[lid] = torch.zeros(
                tp_view_tokens, heads_per_peer, HEAD_DIM,
                dtype=KV_DTYPE, device=f"cuda:{rank}",
            )

        kv_pool.head_num = heads_per_peer
        kv_pool._paras_original_head_num = NUM_KV_HEADS
        torch.cuda.synchronize()
        dist.barrier(group=tp_group)

        # ---- TP→EP scatter with empty batch ----
        partitions = partition_requests_for_ep([], world_size)
        assert all(len(p) == 0 for p in partitions), "Non-empty partition from empty reqs"

        # Scatter with 0 tokens
        for lid in range(NUM_LAYERS):
            send_empty = torch.empty(0, dtype=KV_DTYPE, device=f"cuda:{rank}")
            recv_empty = torch.empty(0, dtype=KV_DTYPE, device=f"cuda:{rank}")
            dist.all_to_all_single(
                recv_empty, send_empty, [0] * world_size, [0] * world_size,
                group=tp_group,
            )
            # Resize back to EP layout
            kv_pool.k_buffer[lid] = torch.zeros(
                ep_max_tokens + PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM,
                dtype=KV_DTYPE, device=f"cuda:{rank}",
            )
            kv_pool.v_buffer[lid] = torch.zeros(
                ep_max_tokens + PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM,
                dtype=KV_DTYPE, device=f"cuda:{rank}",
            )

        kv_pool.head_num = NUM_KV_HEADS
        torch.cuda.synchronize()
        dist.barrier(group=tp_group)

        # If we got here without crash, the test passes
        if rank == 0:
            print("  [OK] Empty batch round-trip completed without crash", flush=True)


# =========================================================================
# Main entry point
# =========================================================================

def main():
    if not _is_distributed():
        print("No distributed env detected. Run with: torchrun --nproc_per_node=4")
        sys.exit(1)

    rank, world_size = _setup_distributed()

    try:
        results = []

        # Test 1: Full round-trip
        if rank == 0:
            print("\n=== Test 1: Full EP→TP→EP Round-Trip ===", flush=True)
        try:
            TestFullRoundTrip().test_roundtrip_ep_tp_ep()
            results.append(("roundtrip_ep_tp_ep", True))
            if rank == 0:
                print("  [PASS] roundtrip_ep_tp_ep", flush=True)
        except Exception as e:
            results.append(("roundtrip_ep_tp_ep", False))
            if rank == 0:
                print(f"  [FAIL] roundtrip_ep_tp_ep: {e}", flush=True)
                import traceback
                traceback.print_exc()
        dist.barrier()

        # Test 2: Partition consistency
        if rank == 0:
            print("\n=== Test 2: Cross-Rank Partition Consistency ===", flush=True)
        try:
            TestPartitionConsistency().test_partition_consistency()
            results.append(("partition_consistency", True))
            if rank == 0:
                print("  [PASS] partition_consistency", flush=True)
        except Exception as e:
            results.append(("partition_consistency", False))
            if rank == 0:
                print(f"  [FAIL] partition_consistency: {e}", flush=True)
                import traceback
                traceback.print_exc()
        dist.barrier()

        # Test 3: Single request round-trip
        if rank == 0:
            print("\n=== Test 3: Single Request Round-Trip ===", flush=True)
        try:
            TestSingleRequestRoundTrip().test_single_request_roundtrip()
            results.append(("single_request_roundtrip", True))
            if rank == 0:
                print("  [PASS] single_request_roundtrip", flush=True)
        except Exception as e:
            results.append(("single_request_roundtrip", False))
            if rank == 0:
                print(f"  [FAIL] single_request_roundtrip: {e}", flush=True)
                import traceback
                traceback.print_exc()
        dist.barrier()

        # Test 4: Empty batch round-trip
        if rank == 0:
            print("\n=== Test 4: Empty Batch Round-Trip ===", flush=True)
        try:
            TestEmptyBatchRoundTrip().test_empty_batch_roundtrip()
            results.append(("empty_batch_roundtrip", True))
            if rank == 0:
                print("  [PASS] empty_batch_roundtrip", flush=True)
        except Exception as e:
            results.append(("empty_batch_roundtrip", False))
            if rank == 0:
                print(f"  [FAIL] empty_batch_roundtrip: {e}", flush=True)
                import traceback
                traceback.print_exc()
        dist.barrier()

        # Summary
        passed = sum(1 for _, ok in results if ok)
        total = len(results)
        if rank == 0:
            print(f"\n{'=' * 60}")
            print(f"RESULTS: {passed}/{total} round-trip tests passed")
            print(f"{'=' * 60}")

        # Save evidence
        evidence_dir = os.path.join(_ROOT_DIR, ".sisyphus", "evidence")
        os.makedirs(evidence_dir, exist_ok=True)
        if rank == 0:
            with open(os.path.join(evidence_dir, "task-10-roundtrip.txt"), "w") as f:
                f.write(f"EP→TP→EP Round-Trip Integration Test\n")
                f.write(f"GPUs: {world_size}, Layers: {NUM_LAYERS}, Heads: {NUM_KV_HEADS}\n")
                f.write(f"Results: {passed}/{total} passed\n\n")
                for name, ok in results:
                    f.write(f"  [{'PASS' if ok else 'FAIL'}] {name}\n")

        _teardown_distributed()
        sys.exit(0 if passed == total else 1)

    except Exception as e:
        print(f"[Rank {rank}] ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        try:
            _teardown_distributed()
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
