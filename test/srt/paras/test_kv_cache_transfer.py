#!/usr/bin/env python3
"""
Standalone correctness tests for KV cache transfer in both EP→TP and TP→EP directions.
Each direction verified independently against pattern-based ground truth.

Usage:
  torchrun --nproc_per_node=4 test/srt/paras/test_kv_cache_transfer.py
  torchrun --nproc_per_node=4 test/srt/paras/test_kv_cache_transfer.py --num-kv-heads 2  # replication
  torchrun --nproc_per_node=4 test/srt/paras/test_kv_cache_transfer.py --num-kv-heads 8  # sharded heads (gpt-oss)
"""

import ctypes
import os
import sys
from typing import List

import pytest
import torch
import torch.distributed as dist

try:
    import paras_peer_access_cuda

    _HAS_PEER_ACCESS = True
except ImportError:
    _HAS_PEER_ACCESS = False

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.join(_TEST_DIR, "..", "..", "..")
sys.path.insert(0, os.path.join(_ROOT_DIR, "python"))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_LAYERS = 3
HEAD_DIM = 128
DTYPE = torch.bfloat16
TOKENS_PER_RANK_4GPU = [50, 40, 45, 35]
TOKENS_PER_RANK_8GPU = [50, 40, 45, 35, 55, 30, 60, 25]
PAGE_SIZE = 1


def _tokens_for_world(world_size):
    if world_size <= 4:
        return TOKENS_PER_RANK_4GPU[:world_size]
    return TOKENS_PER_RANK_8GPU[:world_size]


# ---------------------------------------------------------------------------
# Pattern generator
# ---------------------------------------------------------------------------

def make_pattern(rank, layer, head, num_tokens):
    """Deterministic test data: each (rank, layer, head, token, dim) is unique.

    Returns: Tensor of shape (num_tokens, HEAD_DIM) in bfloat16.
    """
    base = rank * 1000.0 + layer * 100.0 + head * 10.0
    t = torch.arange(num_tokens, dtype=torch.float32).unsqueeze(1)
    d = torch.arange(HEAD_DIM, dtype=torch.float32).unsqueeze(0) * 0.001
    return (base + t + d).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Distributed helpers
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


def _ensure_distributed():
    """Idempotent distributed init. Returns (rank, world_size)."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size in (4, 8), f"Requires 4 or 8 GPUs, got {world_size}"
    torch.cuda.set_device(rank)
    return rank, world_size


_CACHED_TP_GROUP = None


def _setup_paras_state(rank, world_size):
    """Set ParaS parallel state globals without full sglang server init."""
    import sglang.srt.distributed.parallel_state as ps
    import sglang.srt.paras.paras_parallel_state as pps

    global _CACHED_TP_GROUP
    if _CACHED_TP_GROUP is None:
        _CACHED_TP_GROUP = dist.new_group(ranks=list(range(world_size)))
    tp_group = _CACHED_TP_GROUP
    tp_coord = _SimpleGroupCoordinator(
        tp_group, world_size, f"cuda:{rank}", rank_in_group=rank
    )

    ps._TP = tp_coord
    pps._PARAS_TP = tp_coord
    pps._PARAS_DP = _SimpleGroupCoordinator(None, 1, f"cuda:{rank}", 0)
    pps._PARAS_SELF = _SimpleGroupCoordinator(None, 1, f"cuda:{rank}", 0)
    pps._PARAS_TP_SIZE = world_size
    pps._PARAS_TP_RANK = rank
    pps._PARAS_DP_SIZE = 1
    pps._PARAS_DP_RANK = 0
    pps._PARAS_EP_SIZE = world_size
    pps._PARAS_EP_RANK = rank
    pps._PARAS_EP_GROUP_IS_NODE_LOCAL = True
    pps._PARAS_TP_GROUP_IS_NODE_LOCAL = True

    return tp_group


_OPEN_IPC_PTRS: list = []


def _close_old_ipc_handles():
    _cudart = ctypes.CDLL("libcudart.so")
    _cudart.cudaIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
    _cudart.cudaIpcCloseMemHandle.restype = ctypes.c_int
    for ptr in _OPEN_IPC_PTRS:
        _cudart.cudaIpcCloseMemHandle(ctypes.c_void_p(ptr))
    _OPEN_IPC_PTRS.clear()


def setup_peer_ctx(mgr, rank, world_size, tp_group):
    """Enable peer access and exchange CUDA IPC handles."""
    from sglang.srt.paras.peer_access import PeerAccessContext

    _close_old_ipc_handles()

    _cudart = ctypes.CDLL("libcudart.so")

    # --- ctypes types for CUDA IPC ---
    IPC_HANDLE_SIZE = 64

    class CudaIpcMemHandle(ctypes.Structure):
        _fields_ = [("reserved", ctypes.c_ubyte * IPC_HANDLE_SIZE)]

    _cudart.cudaIpcGetMemHandle.argtypes = [
        ctypes.POINTER(CudaIpcMemHandle),
        ctypes.c_void_p,
    ]
    _cudart.cudaIpcGetMemHandle.restype = ctypes.c_int
    _cudart.cudaIpcOpenMemHandle.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        CudaIpcMemHandle,
        ctypes.c_uint,
    ]
    _cudart.cudaIpcOpenMemHandle.restype = ctypes.c_int

    # 1. Get IPC handle for local buffer
    local_handle = CudaIpcMemHandle()
    ret = _cudart.cudaIpcGetMemHandle(
        ctypes.byref(local_handle), ctypes.c_void_p(mgr._buffer.data_ptr())
    )
    assert ret == 0, f"cudaIpcGetMemHandle failed (cuda error {ret})"

    # 2. Exchange handles via all_gather (64 bytes per rank)
    handle_tensor = torch.tensor(
        list(local_handle.reserved), dtype=torch.uint8, device=f"cuda:{rank}"
    )
    assert handle_tensor.numel() == IPC_HANDLE_SIZE
    all_handles = torch.zeros(
        world_size * IPC_HANDLE_SIZE, dtype=torch.uint8, device=f"cuda:{rank}"
    )
    dist.all_gather_into_tensor(all_handles, handle_tensor, group=tp_group)

    # 3. Open remote handles to get local-address mappings
    peer_addresses = []
    for r in range(world_size):
        if r == rank:
            peer_addresses.append(mgr._buffer.data_ptr())
        else:
            raw_list = (
                all_handles[r * IPC_HANDLE_SIZE : (r + 1) * IPC_HANDLE_SIZE]
                .cpu()
                .tolist()
            )
            remote_handle = CudaIpcMemHandle()
            for idx, val in enumerate(raw_list):
                remote_handle.reserved[idx] = val
            remote_ptr = ctypes.c_void_p()
            ret = _cudart.cudaIpcOpenMemHandle(
                ctypes.byref(remote_ptr),
                remote_handle,
                1,  # cudaIpcMemLazyEnablePeerAccess
            )
            assert ret == 0, (
                f"cudaIpcOpenMemHandle for rank {r} failed (cuda error {ret})"
            )
            peer_addresses.append(remote_ptr.value)
            _OPEN_IPC_PTRS.append(remote_ptr.value)

    return PeerAccessContext(
        peer_addresses=peer_addresses,
        peer_access_enabled=True,
        tp_group=tp_group,
        tp_size=world_size,
    )


# ---------------------------------------------------------------------------
# Memory manager setup
# ---------------------------------------------------------------------------

def cleanup_global_manager():
    from sglang.srt.paras.paras_memory_manager import set_global_paras_memory_manager
    set_global_paras_memory_manager(None)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def setup_memory_manager(rank, world_size, num_kv_heads, tokens_per_rank):
    """Create a manager with equal-byte EP and TP KV capacities.

    Returns (mgr, ep_max_tokens, tp_max_tokens).
    """
    from sglang.srt.paras.paras_memory_manager import (
        ParaSMemoryManager,
        set_global_paras_memory_manager,
    )
    set_global_paras_memory_manager(None)

    ep_max_tokens = max(tokens_per_rank) + 100
    heads_per_rank = max(1, num_kv_heads // world_size)
    total_tokens = sum(tokens_per_rank)
    # Ensure EP buffer is large enough for TP-mode token count
    min_ep_for_tp = (
        (total_tokens * heads_per_rank + num_kv_heads - 1) // num_kv_heads
    )
    ep_max_tokens = max(ep_max_tokens, min_ep_for_tp)
    tp_max_tokens = (
        (ep_max_tokens + PAGE_SIZE) * num_kv_heads // heads_per_rank
        - PAGE_SIZE
    )

    mgr = ParaSMemoryManager(device=f"cuda:{rank}")
    mgr.reserve_kv_cache(
        num_layers=NUM_LAYERS,
        ep_max_tokens=ep_max_tokens,
        tp_max_tokens=tp_max_tokens,
        num_kv_heads=num_kv_heads,
        head_dim=HEAD_DIM,
        kv_dtype=DTYPE,
        tp_size=world_size,
        page_size=PAGE_SIZE,
    )
    mgr.materialize()
    set_global_paras_memory_manager(mgr)
    return mgr, ep_max_tokens, tp_max_tokens


# ---------------------------------------------------------------------------
# Mock KV cache
# ---------------------------------------------------------------------------

class _MockKVCache:
    """Minimal KV cache proxy backed by ParaSMemoryManager.

    Tracks per-layer prefix/head_num/view_tokens to mirror MHATokenToKVPool's
    per-layer k_buffer/v_buffer arrays.
    """

    def __init__(self, mgr, num_heads, head_dim, num_layers, dtype, device,
                 prefix, view_tokens=None):
        self.head_num = num_heads
        self.head_dim = head_dim
        self.layer_num = num_layers
        self.store_dtype = dtype
        self.device = device
        self._mgr = mgr
        self._layer_prefix = {i: prefix for i in range(num_layers)}
        self._layer_heads = {i: num_heads for i in range(num_layers)}
        self._layer_view_tokens = {i: view_tokens for i in range(num_layers)}

    def get_key_buffer(self, layer_id):
        prefix = self._layer_prefix[layer_id]
        heads = self._layer_heads[layer_id]
        vt = self._layer_view_tokens[layer_id]
        name = f"model.layers.{layer_id}.kv.{prefix}.k"
        if vt is not None:
            return self._mgr.get_view_as(name, (vt, heads, self.head_dim))
        return self._mgr.get_view(name)

    def get_value_buffer(self, layer_id):
        prefix = self._layer_prefix[layer_id]
        heads = self._layer_heads[layer_id]
        vt = self._layer_view_tokens[layer_id]
        name = f"model.layers.{layer_id}.kv.{prefix}.v"
        if vt is not None:
            return self._mgr.get_view_as(name, (vt, heads, self.head_dim))
        return self._mgr.get_view(name)

    def paras_configure_tp(self, paras_tp_size):
        sharded = self.head_num // paras_tp_size
        self.full_head_num = self.head_num
        self.head_num = sharded
        for lid in range(self.layer_num):
            self._layer_prefix[lid] = "tp"
            self._layer_heads[lid] = sharded
            total_elems = self._mgr._entries[f"model.layers.{lid}.kv.tp.k"].numel
            self._layer_view_tokens[lid] = total_elems // (sharded * self.head_dim)

    def paras_configure_ep(self):
        self.head_num = self.full_head_num
        for lid in range(self.layer_num):
            self._layer_prefix[lid] = "ep"
            self._layer_heads[lid] = self.head_num
            total_elems = self._mgr._entries[f"model.layers.{lid}.kv.ep.k"].numel
            self._layer_view_tokens[lid] = total_elems // (self.head_num * self.head_dim)


# ---------------------------------------------------------------------------
# Fill helpers
# ---------------------------------------------------------------------------

def fill_ep_kv(mgr, rank, num_kv_heads, tokens_per_rank):
    """Fill EP KV buffers with deterministic patterns.

    K uses make_pattern(rank, layer, head, n).
    V uses make_pattern(rank, layer, head + 50, n) for distinct values.
    """
    num_tokens = tokens_per_rank[rank]
    device = f"cuda:{rank}"
    for lid in range(NUM_LAYERS):
        ep_k = mgr.get_view(f"model.layers.{lid}.kv.ep.k")
        ep_v = mgr.get_view(f"model.layers.{lid}.kv.ep.v")
        ep_k.zero_()
        ep_v.zero_()
        for h in range(num_kv_heads):
            ep_k[:num_tokens, h, :] = make_pattern(
                rank, lid, h, num_tokens
            ).to(device)
            ep_v[:num_tokens, h, :] = make_pattern(
                rank, lid, h + 50, num_tokens
            ).to(device)


def fill_tp_kv(mgr, rank, world_size, num_kv_heads, total_tokens,
               tp_view_tokens):
    """Fill TP KV buffers with deterministic patterns.

    K uses make_pattern(rank, layer, local_head, total_tokens).
    V uses make_pattern(rank, layer, local_head + 50, total_tokens).
    """
    heads_per_rank = max(1, num_kv_heads // world_size)
    device = f"cuda:{rank}"
    for lid in range(NUM_LAYERS):
        tp_k = mgr.get_view_as(
            f"model.layers.{lid}.kv.tp.k",
            (tp_view_tokens, heads_per_rank, HEAD_DIM),
        )
        tp_v = mgr.get_view_as(
            f"model.layers.{lid}.kv.tp.v",
            (tp_view_tokens, heads_per_rank, HEAD_DIM),
        )
        tp_k.zero_()
        tp_v.zero_()
        for lh in range(heads_per_rank):
            tp_k[:total_tokens, lh, :] = make_pattern(
                rank, lid, lh, total_tokens
            ).to(device)
            tp_v[:total_tokens, lh, :] = make_pattern(
                rank, lid, lh + 50, total_tokens
            ).to(device)


# ---------------------------------------------------------------------------
# EP→TP gather (manual, mirrors _gather_cache_nccl)
# ---------------------------------------------------------------------------

def do_ep_to_tp_gather(mgr, rank, world_size, num_kv_heads, tokens_per_rank,
                       ep_max_tokens, tp_group):
    """Execute EP→TP gather via NCCL all_to_all.

    Mirrors the logic in ParaSReqGatherManager._gather_cache_nccl, including
    repeat_interleave for head replication when num_kv_heads < world_size.

    Returns tp_view_tokens (the token dimension of the TP-shaped view).
    """
    from sglang.srt.paras.gather_manager import (
        gather_kv_and_permute,
        permute_and_scatter_kv,
    )

    heads_per_rank = max(1, num_kv_heads // world_size)
    replication_factor = (
        max(1, world_size // num_kv_heads)
        if num_kv_heads < world_size
        else 1
    )
    total_tokens = sum(tokens_per_rank)
    num_local = tokens_per_rank[rank]
    splited_size = heads_per_rank * HEAD_DIM
    tp_view_tokens = (
        (ep_max_tokens + PAGE_SIZE) * num_kv_heads // heads_per_rank
    )

    local_token_indices = torch.arange(
        num_local, dtype=torch.int64, device="cuda"
    )
    global_token_indices = torch.arange(
        total_tokens, dtype=torch.int64, device="cuda"
    )

    input_split_sizes = [2 * splited_size * num_local] * world_size
    output_split_sizes = [
        2 * splited_size * tokens_per_rank[r] for r in range(world_size)
    ]

    for lid in range(NUM_LAYERS - 1, -1, -1):
        ep_k = mgr.get_view(f"model.layers.{lid}.kv.ep.k")
        ep_v = mgr.get_view(f"model.layers.{lid}.kv.ep.v")

        permuted = gather_kv_and_permute(ep_k, ep_v, local_token_indices, heads_per_rank)

        # Replicate heads for all_to_all when num_kv_heads < world_size
        if replication_factor > 1:
            permuted = (
                permuted
                .view(num_kv_heads, num_local * 2 * HEAD_DIM)
                .repeat_interleave(replication_factor, dim=0)
                .flatten()
            )

        tp_k = mgr.get_view_as(
            f"model.layers.{lid}.kv.tp.k",
            (tp_view_tokens, heads_per_rank, HEAD_DIM),
        )
        tp_v = mgr.get_view_as(
            f"model.layers.{lid}.kv.tp.v",
            (tp_view_tokens, heads_per_rank, HEAD_DIM),
        )

        gathered = torch.empty(
            2 * total_tokens * splited_size, dtype=DTYPE, device="cuda"
        )
        dist.all_to_all_single(
            gathered, permuted,
            output_split_sizes, input_split_sizes,
            group=tp_group,
        )
        permute_and_scatter_kv(
            gathered, tp_k, tp_v, global_token_indices,
            total_tokens, heads_per_rank, HEAD_DIM,
        )

    torch.cuda.synchronize()
    return tp_view_tokens


# ---------------------------------------------------------------------------
# TP→EP scatter (MHACacheTransfer backend)
# ---------------------------------------------------------------------------

def do_tp_to_ep_scatter(mgr, rank, world_size, num_kv_heads, tokens_per_rank,
                        tp_view_tokens, tp_group):
    """Execute TP→EP scatter by driving ``MHACacheTransfer`` directly.

    Returns (token_partition, ep_dst_positions).
    """
    from sglang.srt.paras.cache_transfer.base import LayerCacheSpec
    from sglang.srt.paras.cache_transfer.mha import MHACacheTransfer

    heads_per_rank = max(1, num_kv_heads // world_size)
    total_tokens = sum(tokens_per_rank)
    global_token_indices = torch.arange(
        total_tokens, dtype=torch.int64, device="cuda"
    )

    token_partition = []
    offset = 0
    for r in range(world_size):
        n = tokens_per_rank[r]
        token_partition.append(list(range(offset, offset + n)))
        offset += n

    ep_dst_positions = torch.arange(
        tokens_per_rank[rank], dtype=torch.int64, device="cuda"
    )

    tp_cache = _MockKVCache(
        mgr, heads_per_rank, HEAD_DIM, NUM_LAYERS, DTYPE,
        f"cuda:{rank}", "tp", view_tokens=tp_view_tokens,
    )
    group_coord = _SimpleGroupCoordinator(
        tp_group, world_size, f"cuda:{rank}", rank
    )

    backend = MHACacheTransfer(
        method="nccl",
        direction="scatter",
        kv_cache=tp_cache,
        mgr=mgr,
        group=group_coord,
        global_token_indices=global_token_indices,
        ep_head_num=num_kv_heads,
        token_partition=token_partition,
        ep_dst_positions=ep_dst_positions,
    )

    for layer_id in range(NUM_LAYERS):
        spec = LayerCacheSpec(
            layer_id=layer_id,
            kind="full",
            tokens_cap_ep=0,
            tokens_cap_tp=0,
            num_kv_heads=num_kv_heads,
            head_dim=HEAD_DIM,
            sliding_window_size=None,
        )
        backend.scatter_one_layer(spec)

    torch.cuda.synchronize()
    return token_partition, ep_dst_positions


# ---------------------------------------------------------------------------
# EP→TP gather (peer_access, mirrors _gather_cache_peer_access)
# ---------------------------------------------------------------------------

def do_ep_to_tp_gather_peer_access(mgr, rank, world_size, num_kv_heads,
                                   tokens_per_rank, ep_max_tokens, tp_group,
                                   peer_ctx):
    """Execute EP→TP gather via peer_access kernel.

    Mirrors the logic in ParaSReqGatherManager._gather_cache_peer_access.
    Each rank reads its local EP tokens and writes them into all TP ranks'
    buffers via NVLink using CUDA IPC.

    Returns tp_view_tokens (the token dimension of the TP-shaped view).
    """
    from sglang.srt.paras.peer_access import peer_access_kv_transfer

    heads_per_rank = max(1, num_kv_heads // world_size)
    num_local = tokens_per_rank[rank]
    tp_view_tokens = (
        (ep_max_tokens + PAGE_SIZE) * num_kv_heads // heads_per_rank
    )

    local_buffer_ptr = mgr._buffer.data_ptr()
    dst_base_ptrs = torch.tensor(
        peer_ctx.peer_addresses, dtype=torch.int64, device="cuda"
    )
    # Production reserves slot 0 in both pools. Stage this fixture's compact
    # source data into slots [1, num_local] before invoking the real kernel.
    for layer_id in range(NUM_LAYERS):
        ep_k = mgr.get_view(f"model.layers.{layer_id}.kv.ep.k")
        ep_v = mgr.get_view(f"model.layers.{layer_id}.kv.ep.v")
        ep_k[1 : num_local + 1].copy_(ep_k[:num_local].clone())
        ep_v[1 : num_local + 1].copy_(ep_v[:num_local].clone())

    local_token_indices = torch.arange(
        1, num_local + 1, dtype=torch.int32, device="cuda"
    )
    dst_token_start = sum(tokens_per_rank[:rank]) + 1
    elem_size = 2  # bfloat16

    barrier_tensor = torch.zeros(1, device="cuda")

    for layer_id in range(NUM_LAYERS - 1, -1, -1):
        src_k_offset = mgr._entries[
            f"model.layers.{layer_id}.kv.ep.k"
        ].offset_bytes
        src_v_offset = mgr._entries[
            f"model.layers.{layer_id}.kv.ep.v"
        ].offset_bytes
        dst_k_offset = mgr._entries[
            f"model.layers.{layer_id}.kv.tp.k"
        ].offset_bytes
        dst_v_offset = mgr._entries[
            f"model.layers.{layer_id}.kv.tp.v"
        ].offset_bytes

        if num_local > 0:
            peer_access_kv_transfer(
                local_buffer_ptr, dst_base_ptrs,
                local_token_indices,
                src_k_offset, src_v_offset,
                dst_k_offset, dst_v_offset,
                num_local, dst_token_start,
                num_kv_heads, rank, world_size, HEAD_DIM,
                elem_size,
            )

        # The reverse layer order and barrier preserve unread EP sources in
        # the overlapped layout.
        dist.all_reduce(barrier_tensor, group=tp_group)

    torch.cuda.synchronize()
    return tp_view_tokens


# ---------------------------------------------------------------------------
# TP→EP scatter (peer_access, mirrors _scatter_cache_peer_access)
# ---------------------------------------------------------------------------

def do_tp_to_ep_scatter_peer_access(mgr, rank, world_size, num_kv_heads,
                                    tokens_per_rank, tp_view_tokens,
                                    tp_group, peer_ctx):
    """Execute TP→EP scatter via peer_access kernel.

    Mirrors the logic in ParaSReqScatterManager._scatter_cache_peer_access,
    including replication-aware 1/R token slicing and reverse-order per-layer
    barriers.

    Returns (token_partition, ep_dst_positions).
    """
    heads_per_rank = max(1, num_kv_heads // world_size)
    total_tokens = sum(tokens_per_rank)

    # Contiguous token partition (same as NCCL variant)
    token_partition: List[List[int]] = []
    offset = 0
    for r in range(world_size):
        n = tokens_per_rank[r]
        token_partition.append(list(range(offset, offset + n)))
        offset += n

    ep_dst_positions = torch.arange(
        tokens_per_rank[rank], dtype=torch.int64, device="cuda"
    )

    # Global token indices (identity mapping in this test)
    # Production reserves slot 0 in both pools. Stage compact TP fixture data
    # into slots [1, total_tokens] and route to EP destinations starting at 1.
    for layer_id in range(NUM_LAYERS):
        tp_k = mgr.get_view_as(
            f"model.layers.{layer_id}.kv.tp.k",
            (tp_view_tokens, heads_per_rank, HEAD_DIM),
        )
        tp_v = mgr.get_view_as(
            f"model.layers.{layer_id}.kv.tp.v",
            (tp_view_tokens, heads_per_rank, HEAD_DIM),
        )
        tp_k[1 : total_tokens + 1].copy_(tp_k[:total_tokens].clone())
        tp_v[1 : total_tokens + 1].copy_(tp_v[:total_tokens].clone())

    global_token_indices = torch.arange(
        1, total_tokens + 1, dtype=torch.int64, device="cuda"
    )

    # -- Replication-aware token selection (from scatter_manager) ----------
    replication_factor = (
        world_size // num_kv_heads
        if num_kv_heads < world_size
        else 1
    )
    intra_rank = rank % replication_factor

    my_global_indices: List[int] = []
    my_dst_ranks: List[int] = []
    my_ep_dst_pos: List[int] = []

    for e in range(world_size):
        full_tokens = token_partition[e]
        full = len(full_tokens)
        my_start = full * intra_rank // replication_factor
        my_end = full * (intra_rank + 1) // replication_factor
        my_slice = full_tokens[my_start:my_end]
        for local_idx, global_idx in enumerate(my_slice):
            my_global_indices.append(global_idx)
            my_dst_ranks.append(e)
            my_ep_dst_pos.append(my_start + local_idx + 1)

    num_my_tokens = len(my_global_indices)

    if num_my_tokens > 0:
        gi_tensor = torch.tensor(
            my_global_indices, dtype=torch.long, device="cuda"
        )
        tp_token_positions = global_token_indices[gi_tensor].to(torch.int32)
        token_to_rank = torch.tensor(
            my_dst_ranks, dtype=torch.int32, device="cuda"
        )
        ep_dst_pos_all = torch.tensor(
            my_ep_dst_pos, dtype=torch.int32, device="cuda"
        )
    else:
        tp_token_positions = torch.empty(0, dtype=torch.int32, device="cuda")
        token_to_rank = torch.empty(0, dtype=torch.int32, device="cuda")
        ep_dst_pos_all = torch.empty(0, dtype=torch.int32, device="cuda")

    # Build per-layer byte offsets (source=TP, dest=EP)
    src_k_offsets: List[int] = []
    src_v_offsets: List[int] = []
    dst_k_offsets: List[int] = []
    dst_v_offsets: List[int] = []
    for layer_id in range(NUM_LAYERS):
        src_k_offsets.append(
            mgr._entries[f"model.layers.{layer_id}.kv.tp.k"].offset_bytes
        )
        src_v_offsets.append(
            mgr._entries[f"model.layers.{layer_id}.kv.tp.v"].offset_bytes
        )
        dst_k_offsets.append(
            mgr._entries[f"model.layers.{layer_id}.kv.ep.k"].offset_bytes
        )
        dst_v_offsets.append(
            mgr._entries[f"model.layers.{layer_id}.kv.ep.v"].offset_bytes
        )

    local_buffer_ptr = mgr._buffer.data_ptr()
    peer_buffer_ptrs = torch.tensor(
        peer_ctx.peer_addresses, dtype=torch.int64, device="cuda"
    )
    elem_size = 2  # bfloat16

    # Forward layer order and a per-layer barrier preserve unread TP sources
    # in the overlapped layout.
    import paras_peer_access_cuda as _pa_cuda

    barrier_tensor = torch.zeros(1, device="cuda")
    for layer_idx in range(NUM_LAYERS):
        if num_my_tokens > 0:
            _pa_cuda.launch_peer_access_kv_scatter(
                local_buffer_ptr,
                peer_buffer_ptrs,
                tp_token_positions,
                token_to_rank,
                ep_dst_pos_all,
                src_k_offsets[layer_idx],
                src_v_offsets[layer_idx],
                dst_k_offsets[layer_idx],
                dst_v_offsets[layer_idx],
                num_my_tokens,
                heads_per_rank,
                num_kv_heads,
                rank,
                world_size,
                HEAD_DIM,
                elem_size,
                0,  # default stream
            )
        # Per-layer barrier: ALL ranks participate
        dist.all_reduce(barrier_tensor, group=tp_group)

    torch.cuda.synchronize()
    return token_partition, ep_dst_positions


# ---------------------------------------------------------------------------
# Evidence saving
# ---------------------------------------------------------------------------

def _save_evidence(test_name, passed, rank):
    """Append test result to evidence file (rank 0 only)."""
    if rank != 0:
        return
    evidence_dir = os.path.join(_ROOT_DIR, ".sisyphus", "evidence")
    os.makedirs(evidence_dir, exist_ok=True)
    path = os.path.join(evidence_dir, "kv-cache-transfer-test.txt")
    with open(path, "a") as f:
        f.write(f"{test_name}: {'PASS' if passed else 'FAIL'}\n")


# ---------------------------------------------------------------------------
# Shared verification helpers
# ---------------------------------------------------------------------------

def verify_ep_to_tp(mgr, rank, world_size, num_kv_heads, tokens_per_rank,
                    tp_view_tokens, token_offset=0):
    """Verify EP→TP gather result against ground truth patterns.

    Works for both R=1 (no replication) and R>1 (head replication).
    Returns True if all checks pass.
    """
    replication_factor = (
        max(1, world_size // num_kv_heads)
        if num_kv_heads < world_size
        else 1
    )
    real_head = rank // replication_factor
    device = f"cuda:{rank}"

    all_ok = True
    for lid in range(NUM_LAYERS):
        tp_k = mgr.get_view_as(
            f"model.layers.{lid}.kv.tp.k",
            (tp_view_tokens, 1, HEAD_DIM),
        )
        tp_v = mgr.get_view_as(
            f"model.layers.{lid}.kv.tp.v",
            (tp_view_tokens, 1, HEAD_DIM),
        )

        offset = 0
        for src in range(world_size):
            n = tokens_per_rank[src]
            expected_k = make_pattern(src, lid, real_head, n).to(device)
            expected_v = make_pattern(src, lid, real_head + 50, n).to(device)
            start = token_offset + offset
            actual_k = tp_k[start:start + n, 0, :]
            actual_v = tp_v[start:start + n, 0, :]
            if not torch.equal(actual_k, expected_k):
                all_ok = False
                if rank == 0:
                    diff = (actual_k.float() - expected_k.float()).abs().max()
                    print(f"  K mismatch L{lid} src={src}: max_diff={diff}")
            if not torch.equal(actual_v, expected_v):
                all_ok = False
                if rank == 0:
                    diff = (actual_v.float() - expected_v.float()).abs().max()
                    print(f"  V mismatch L{lid} src={src}: max_diff={diff}")
            offset += n

    return all_ok


def verify_tp_to_ep(mgr, rank, world_size, num_kv_heads, tokens_per_rank,
                    token_partition, token_offset=0):
    """Verify TP→EP scatter result against ground truth patterns.

    Works for both R=1 (no replication) and R>1 (head replication).
    Returns True if all checks pass.
    """
    total_tokens = sum(tokens_per_rank)
    R = (
        max(1, world_size // num_kv_heads)
        if num_kv_heads < world_size
        else 1
    )
    my_tokens = token_partition[rank]
    count = len(my_tokens)
    device = f"cuda:{rank}"

    ep_dst = torch.arange(
        token_offset, token_offset + count, dtype=torch.int64, device=device
    )

    all_ok = True
    for lid in range(NUM_LAYERS):
        ep_k = mgr.get_view(f"model.layers.{lid}.kv.ep.k")
        ep_v = mgr.get_view(f"model.layers.{lid}.kv.ep.v")

        for h in range(num_kv_heads):
            for ir in range(R):
                start = count * ir // R
                end = count * (ir + 1) // R
                if start == end:
                    continue
                src_tp_rank = h * R + ir

                full_k = make_pattern(
                    src_tp_rank, lid, 0, total_tokens
                ).to(device)
                full_v = make_pattern(
                    src_tp_rank, lid, 50, total_tokens
                ).to(device)

                global_indices = torch.tensor(
                    my_tokens[start:end], dtype=torch.long, device=device
                )
                expected_k = full_k[global_indices]
                expected_v = full_v[global_indices]
                actual_k = ep_k[ep_dst[start:end], h, :]
                actual_v = ep_v[ep_dst[start:end], h, :]

                if not torch.equal(actual_k, expected_k):
                    all_ok = False
                    if rank == 0:
                        diff = (
                            actual_k.float() - expected_k.float()
                        ).abs().max()
                        print(
                            f"  K mismatch L{lid} h={h} ir={ir}: "
                            f"max_diff={diff}"
                        )
                if not torch.equal(actual_v, expected_v):
                    all_ok = False
                    if rank == 0:
                        diff = (
                            actual_v.float() - expected_v.float()
                        ).abs().max()
                        print(
                            f"  V mismatch L{lid} h={h} ir={ir}: "
                            f"max_diff={diff}"
                        )

    return all_ok


# =========================================================================
# EP→TP Tests
# =========================================================================

@pytest.mark.skipif(not _is_distributed(), reason="Requires torchrun with 4 GPUs")
class TestEPtoTPStandalone:
    """EP→TP gather verified against first-principles pattern computation."""

    def test_ep_to_tp_no_replication(self):
        """num_kv_heads == world_size: each TP rank gets 1 unique head."""
        rank, world_size = _ensure_distributed()
        num_kv_heads = world_size
        tokens_per_rank = _tokens_for_world(world_size)

        tp_group = _setup_paras_state(rank, world_size)
        mgr, ep_max, _ = setup_memory_manager(
            rank, world_size, num_kv_heads, tokens_per_rank
        )

        fill_ep_kv(mgr, rank, num_kv_heads, tokens_per_rank)
        tp_view_tokens = do_ep_to_tp_gather(
            mgr, rank, world_size, num_kv_heads, tokens_per_rank,
            ep_max, tp_group,
        )
        dist.barrier(group=tp_group)

        # Verify: TP rank r stores head r (heads_per_rank=1)
        head_idx = rank  # global head index for this TP rank
        device = f"cuda:{rank}"

        all_ok = True
        for lid in range(NUM_LAYERS):
            tp_k = mgr.get_view_as(
                f"model.layers.{lid}.kv.tp.k",
                (tp_view_tokens, 1, HEAD_DIM),
            )
            tp_v = mgr.get_view_as(
                f"model.layers.{lid}.kv.tp.v",
                (tp_view_tokens, 1, HEAD_DIM),
            )

            offset = 0
            for src in range(world_size):
                n = tokens_per_rank[src]
                expected_k = make_pattern(src, lid, head_idx, n).to(device)
                expected_v = make_pattern(src, lid, head_idx + 50, n).to(device)
                actual_k = tp_k[offset:offset + n, 0, :]
                actual_v = tp_v[offset:offset + n, 0, :]
                if not torch.equal(actual_k, expected_k):
                    all_ok = False
                    if rank == 0:
                        diff = (actual_k.float() - expected_k.float()).abs().max()
                        print(f"  K mismatch L{lid} src={src}: max_diff={diff}")
                if not torch.equal(actual_v, expected_v):
                    all_ok = False
                    if rank == 0:
                        diff = (actual_v.float() - expected_v.float()).abs().max()
                        print(f"  V mismatch L{lid} src={src}: max_diff={diff}")
                offset += n

        _save_evidence("ep_to_tp_no_replication", all_ok, rank)
        assert all_ok, f"EP→TP (no replication) failed on rank {rank}"

    # -- Peer-access variants ------------------------------------------

    @pytest.mark.skipif(
        not _HAS_PEER_ACCESS,
        reason="paras_peer_access_cuda not available",
    )
    def test_ep_to_tp_peer_access_no_replication(self):
        """num_kv_heads == world_size: EP→TP via peer_access kernel."""
        rank, world_size = _ensure_distributed()
        num_kv_heads = world_size
        tokens_per_rank = _tokens_for_world(world_size)

        tp_group = _setup_paras_state(rank, world_size)
        mgr, ep_max, _ = setup_memory_manager(
            rank, world_size, num_kv_heads, tokens_per_rank
        )
        peer_ctx = setup_peer_ctx(mgr, rank, world_size, tp_group)

        fill_ep_kv(mgr, rank, num_kv_heads, tokens_per_rank)
        tp_view_tokens = do_ep_to_tp_gather_peer_access(
            mgr, rank, world_size, num_kv_heads, tokens_per_rank,
            ep_max, tp_group, peer_ctx,
        )
        dist.barrier(group=tp_group)

        all_ok = verify_ep_to_tp(
            mgr, rank, world_size, num_kv_heads, tokens_per_rank,
            tp_view_tokens, token_offset=1,
        )
        _save_evidence("ep_to_tp_peer_access_no_replication", all_ok, rank)
        assert all_ok, f"EP→TP peer_access (no replication) failed on rank {rank}"



# =========================================================================
# TP→EP Tests
# =========================================================================
# TP→EP Tests
# =========================================================================

@pytest.mark.skipif(not _is_distributed(), reason="Requires torchrun with 4 GPUs")
class TestTPtoEPStandalone:
    """TP→EP scatter verified against first-principles pattern computation."""

    def test_tp_to_ep_no_replication(self):
        """num_kv_heads == world_size: standard scatter, no replication."""
        rank, world_size = _ensure_distributed()
        num_kv_heads = world_size
        tokens_per_rank = _tokens_for_world(world_size)
        total_tokens = sum(tokens_per_rank)

        tp_group = _setup_paras_state(rank, world_size)
        mgr, ep_max, _ = setup_memory_manager(
            rank, world_size, num_kv_heads, tokens_per_rank
        )

        heads_per_rank = num_kv_heads // world_size  # 1
        tp_view_tokens = (
            (ep_max + PAGE_SIZE) * num_kv_heads // heads_per_rank
        )

        fill_tp_kv(
            mgr, rank, world_size, num_kv_heads, total_tokens, tp_view_tokens
        )

        token_partition, ep_dst = do_tp_to_ep_scatter(
            mgr, rank, world_size, num_kv_heads, tokens_per_rank,
            tp_view_tokens, tp_group,
        )

        # Verify: EP rank e, head h → data from TP rank h, local head 0
        my_tokens = token_partition[rank]
        num_local = len(my_tokens)
        device = f"cuda:{rank}"

        all_ok = True
        for lid in range(NUM_LAYERS):
            ep_k = mgr.get_view(f"model.layers.{lid}.kv.ep.k")
            ep_v = mgr.get_view(f"model.layers.{lid}.kv.ep.v")

            for h in range(num_kv_heads):
                src_tp_rank = h
                # Full pattern for the source TP rank
                full_k = make_pattern(src_tp_rank, lid, 0, total_tokens).to(device)
                full_v = make_pattern(src_tp_rank, lid, 50, total_tokens).to(device)

                global_indices = torch.tensor(
                    my_tokens, dtype=torch.long, device=device
                )
                expected_k = full_k[global_indices]
                expected_v = full_v[global_indices]
                actual_k = ep_k[ep_dst[:num_local], h, :]
                actual_v = ep_v[ep_dst[:num_local], h, :]

                if not torch.equal(actual_k, expected_k):
                    all_ok = False
                    if rank == 0:
                        diff = (actual_k.float() - expected_k.float()).abs().max()
                        print(f"  K mismatch L{lid} h={h}: max_diff={diff}")
                if not torch.equal(actual_v, expected_v):
                    all_ok = False
                    if rank == 0:
                        diff = (actual_v.float() - expected_v.float()).abs().max()
                        print(f"  V mismatch L{lid} h={h}: max_diff={diff}")

        _save_evidence("tp_to_ep_no_replication", all_ok, rank)
        assert all_ok, f"TP→EP (no replication) failed on rank {rank}"

    # -- Peer-access variants ------------------------------------------

    @pytest.mark.skipif(
        not _HAS_PEER_ACCESS,
        reason="paras_peer_access_cuda not available",
    )
    def test_tp_to_ep_peer_access_no_replication(self):
        """num_kv_heads == world_size: TP→EP via peer_access kernel."""
        rank, world_size = _ensure_distributed()
        num_kv_heads = world_size
        tokens_per_rank = _tokens_for_world(world_size)
        total_tokens = sum(tokens_per_rank)

        tp_group = _setup_paras_state(rank, world_size)
        mgr, ep_max, _ = setup_memory_manager(
            rank, world_size, num_kv_heads, tokens_per_rank
        )
        peer_ctx = setup_peer_ctx(mgr, rank, world_size, tp_group)

        heads_per_rank = num_kv_heads // world_size  # 1
        tp_view_tokens = (
            (ep_max + PAGE_SIZE) * num_kv_heads // heads_per_rank
        )

        fill_tp_kv(
            mgr, rank, world_size, num_kv_heads, total_tokens, tp_view_tokens
        )

        token_partition, _ = do_tp_to_ep_scatter_peer_access(
            mgr, rank, world_size, num_kv_heads, tokens_per_rank,
            tp_view_tokens, tp_group, peer_ctx,
        )

        all_ok = verify_tp_to_ep(
            mgr, rank, world_size, num_kv_heads, tokens_per_rank,
            token_partition, token_offset=1,
        )
        _save_evidence(
            "tp_to_ep_peer_access_no_replication", all_ok, rank
        )
        assert all_ok, (
            f"TP→EP peer_access (no replication) failed on rank {rank}"
        )



# =========================================================================
# Sharded-heads Tests (num_kv_heads > tp_size — gpt-oss-style config)
# =========================================================================
#
# These tests exercise the case where each TP rank owns MULTIPLE KV heads
# (heads_per_rank > 1). For 4 GPUs with num_kv_heads=8, heads_per_rank=2.
# This is the configuration that the original NCCL Bug 4 fix in commit
# b9e8c9321 enabled (silently corrupted gpt-oss-style configs because
# gather_kv_and_permute used a heads-first layout that only matched scatter
# for sharded_heads == 1).
#
# The peer_access kernel does not use all_to_all, so it cannot have the
# same Bug 4 layout mismatch. These tests verify the peer_access path is
# correct end-to-end for sharded_heads > 1, which had zero coverage prior
# to this addition.
# =========================================================================


def verify_ep_to_tp_sharded(mgr, rank, world_size, num_kv_heads,
                            tokens_per_rank, tp_view_tokens, token_offset=0):
    """Verify EP→TP gather for heads_per_rank > 1.

    TP rank r owns heads ``[r*heads_per_rank, (r+1)*heads_per_rank)``
    (global indices).  For each (source_rank, local_token) we expect
    ``make_pattern(src, layer, global_head, n)`` for K and
    ``make_pattern(src, layer, global_head + 50, n)`` for V.
    """
    heads_per_rank = num_kv_heads // world_size
    assert heads_per_rank > 1, (
        "verify_ep_to_tp_sharded requires heads_per_rank > 1; "
        "use verify_ep_to_tp for heads_per_rank == 1"
    )
    device = f"cuda:{rank}"

    all_ok = True
    for lid in range(NUM_LAYERS):
        tp_k = mgr.get_view_as(
            f"model.layers.{lid}.kv.tp.k",
            (tp_view_tokens, heads_per_rank, HEAD_DIM),
        )
        tp_v = mgr.get_view_as(
            f"model.layers.{lid}.kv.tp.v",
            (tp_view_tokens, heads_per_rank, HEAD_DIM),
        )

        for lh in range(heads_per_rank):
            global_head = rank * heads_per_rank + lh
            offset = 0
            for src in range(world_size):
                n = tokens_per_rank[src]
                expected_k = make_pattern(src, lid, global_head, n).to(device)
                expected_v = make_pattern(src, lid, global_head + 50, n).to(device)
                start = token_offset + offset
                actual_k = tp_k[start:start + n, lh, :]
                actual_v = tp_v[start:start + n, lh, :]
                if not torch.equal(actual_k, expected_k):
                    all_ok = False
                    if rank == 0:
                        diff = (actual_k.float() - expected_k.float()).abs().max()
                        print(
                            f"  K mismatch L{lid} src={src} lh={lh} "
                            f"(global_head={global_head}): max_diff={diff}"
                        )
                if not torch.equal(actual_v, expected_v):
                    all_ok = False
                    if rank == 0:
                        diff = (actual_v.float() - expected_v.float()).abs().max()
                        print(
                            f"  V mismatch L{lid} src={src} lh={lh} "
                            f"(global_head={global_head}): max_diff={diff}"
                        )
                offset += n

    return all_ok


def verify_tp_to_ep_sharded(mgr, rank, world_size, num_kv_heads,
                            tokens_per_rank, token_partition, token_offset=0):
    """Verify TP→EP scatter for heads_per_rank > 1.

    EP rank ``e`` receives ``token_partition[e]`` tokens.  For each global
    EP head ``h``:
      ``src_tp_rank = h // heads_per_rank``
      ``src_local_head = h % heads_per_rank``
    Expected K = ``make_pattern(src_tp_rank, lid, src_local_head, total)``
    indexed by ``token_partition[e]`` (and analogously for V with +50).
    """
    heads_per_rank = num_kv_heads // world_size
    assert heads_per_rank > 1, (
        "verify_tp_to_ep_sharded requires heads_per_rank > 1; "
        "use verify_tp_to_ep for heads_per_rank == 1"
    )
    total_tokens = sum(tokens_per_rank)
    my_tokens = token_partition[rank]
    count = len(my_tokens)
    device = f"cuda:{rank}"

    ep_dst = torch.arange(
        token_offset, token_offset + count, dtype=torch.int64, device=device
    )

    all_ok = True
    for lid in range(NUM_LAYERS):
        ep_k = mgr.get_view(f"model.layers.{lid}.kv.ep.k")
        ep_v = mgr.get_view(f"model.layers.{lid}.kv.ep.v")

        for h in range(num_kv_heads):
            src_tp_rank = h // heads_per_rank
            src_local_head = h % heads_per_rank

            full_k = make_pattern(
                src_tp_rank, lid, src_local_head, total_tokens
            ).to(device)
            full_v = make_pattern(
                src_tp_rank, lid, src_local_head + 50, total_tokens
            ).to(device)

            global_indices = torch.tensor(
                my_tokens, dtype=torch.long, device=device
            )
            expected_k = full_k[global_indices]
            expected_v = full_v[global_indices]
            actual_k = ep_k[ep_dst, h, :]
            actual_v = ep_v[ep_dst, h, :]

            if not torch.equal(actual_k, expected_k):
                all_ok = False
                if rank == 0:
                    diff = (actual_k.float() - expected_k.float()).abs().max()
                    print(
                        f"  K mismatch L{lid} h={h} "
                        f"(src_rank={src_tp_rank} src_lh={src_local_head}): "
                        f"max_diff={diff}"
                    )
            if not torch.equal(actual_v, expected_v):
                all_ok = False
                if rank == 0:
                    diff = (actual_v.float() - expected_v.float()).abs().max()
                    print(
                        f"  V mismatch L{lid} h={h} "
                        f"(src_rank={src_tp_rank} src_lh={src_local_head}): "
                        f"max_diff={diff}"
                    )

    return all_ok


@pytest.mark.skipif(not _is_distributed(), reason="Requires torchrun with 4 GPUs")
class TestShardedHeads:
    """num_kv_heads > tp_size: each TP rank owns multiple heads (gpt-oss case).

    For 4 GPUs with num_kv_heads=8, heads_per_rank=2.
    """

    @pytest.mark.skipif(
        not _HAS_PEER_ACCESS,
        reason="paras_peer_access_cuda not available",
    )
    def test_ep_to_tp_peer_access_sharded_heads(self):
        """EP→TP via peer_access kernel with heads_per_rank > 1."""
        rank, world_size = _ensure_distributed()
        # 2x world_size gives heads_per_rank == 2 on any supported world size.
        num_kv_heads = world_size * 2
        tokens_per_rank = _tokens_for_world(world_size)

        tp_group = _setup_paras_state(rank, world_size)
        mgr, ep_max, _ = setup_memory_manager(
            rank, world_size, num_kv_heads, tokens_per_rank
        )
        peer_ctx = setup_peer_ctx(mgr, rank, world_size, tp_group)

        fill_ep_kv(mgr, rank, num_kv_heads, tokens_per_rank)
        tp_view_tokens = do_ep_to_tp_gather_peer_access(
            mgr, rank, world_size, num_kv_heads, tokens_per_rank,
            ep_max, tp_group, peer_ctx,
        )
        dist.barrier(group=tp_group)

        all_ok = verify_ep_to_tp_sharded(
            mgr, rank, world_size, num_kv_heads, tokens_per_rank,
            tp_view_tokens, token_offset=1,
        )
        _save_evidence("ep_to_tp_peer_access_sharded_heads", all_ok, rank)
        assert all_ok, (
            f"EP→TP peer_access (sharded_heads) failed on rank {rank}"
        )

    @pytest.mark.skipif(
        not _HAS_PEER_ACCESS,
        reason="paras_peer_access_cuda not available",
    )
    def test_tp_to_ep_peer_access_sharded_heads(self):
        """TP→EP via peer_access kernel with heads_per_rank > 1."""
        rank, world_size = _ensure_distributed()
        num_kv_heads = world_size * 2
        tokens_per_rank = _tokens_for_world(world_size)
        total_tokens = sum(tokens_per_rank)

        tp_group = _setup_paras_state(rank, world_size)
        mgr, ep_max, _ = setup_memory_manager(
            rank, world_size, num_kv_heads, tokens_per_rank
        )
        peer_ctx = setup_peer_ctx(mgr, rank, world_size, tp_group)

        heads_per_rank = num_kv_heads // world_size
        tp_view_tokens = (
            (ep_max + PAGE_SIZE) * num_kv_heads // heads_per_rank
        )

        fill_tp_kv(
            mgr, rank, world_size, num_kv_heads, total_tokens, tp_view_tokens
        )

        token_partition, _ = do_tp_to_ep_scatter_peer_access(
            mgr, rank, world_size, num_kv_heads, tokens_per_rank,
            tp_view_tokens, tp_group, peer_ctx,
        )

        all_ok = verify_tp_to_ep_sharded(
            mgr, rank, world_size, num_kv_heads, tokens_per_rank,
            token_partition, token_offset=1,
        )
        _save_evidence(
            "tp_to_ep_peer_access_sharded_heads", all_ok, rank
        )
        assert all_ok, (
            f"TP→EP peer_access (sharded_heads) failed on rank {rank}"
        )


# =========================================================================
# Round-trip Tests
# =========================================================================

@pytest.mark.skipif(not _is_distributed(), reason="Requires torchrun with 4 GPUs")
class TestKVRoundTrip:
    """EP→TP→EP round-trip: verify bitwise match after full cycle."""

    def _run_roundtrip(self, num_kv_heads, test_name):
        """Shared round-trip logic for both replication modes."""
        rank, world_size = _ensure_distributed()
        tokens_per_rank = _tokens_for_world(world_size)
        num_local = tokens_per_rank[rank]

        tp_group = _setup_paras_state(rank, world_size)
        mgr, ep_max, _ = setup_memory_manager(
            rank, world_size, num_kv_heads, tokens_per_rank
        )

        # Fill EP with patterns and snapshot
        fill_ep_kv(mgr, rank, num_kv_heads, tokens_per_rank)
        orig_ep = {}
        for lid in range(NUM_LAYERS):
            ep_k = mgr.get_view(f"model.layers.{lid}.kv.ep.k")
            ep_v = mgr.get_view(f"model.layers.{lid}.kv.ep.v")
            orig_ep[lid] = (
                ep_k[:num_local].clone(),
                ep_v[:num_local].clone(),
            )

        # Step 1: EP→TP gather
        tp_view_tokens = do_ep_to_tp_gather(
            mgr, rank, world_size, num_kv_heads, tokens_per_rank,
            ep_max, tp_group,
        )
        dist.barrier(group=tp_group)

        # Step 2: TP→EP scatter
        # NOTE: Do NOT zero EP buffers here — they share physical memory
        # with TP buffers via the N+1 slot design.  MHACacheTransfer
        # processes layers in reverse order to avoid corrupting TP source.
        token_partition, ep_dst = do_tp_to_ep_scatter(
            mgr, rank, world_size, num_kv_heads, tokens_per_rank,
            tp_view_tokens, tp_group,
        )

        # Step 3: Verify bitwise match
        all_ok = True
        for lid in range(NUM_LAYERS):
            ep_k = mgr.get_view(f"model.layers.{lid}.kv.ep.k")
            ep_v = mgr.get_view(f"model.layers.{lid}.kv.ep.v")
            k_match = torch.equal(orig_ep[lid][0], ep_k[:num_local])
            v_match = torch.equal(orig_ep[lid][1], ep_v[:num_local])
            if not k_match or not v_match:
                all_ok = False
                if rank == 0:
                    if not k_match:
                        diff = (
                            orig_ep[lid][0].float()
                            - ep_k[:num_local].float()
                        ).abs().max()
                        print(f"  K round-trip mismatch L{lid}: max_diff={diff}")
                    if not v_match:
                        diff = (
                            orig_ep[lid][1].float()
                            - ep_v[:num_local].float()
                        ).abs().max()
                        print(f"  V round-trip mismatch L{lid}: max_diff={diff}")

        _save_evidence(test_name, all_ok, rank)
        return all_ok

    def test_roundtrip_no_replication(self):
        """EP→TP→EP with num_kv_heads == world_size (R=1)."""
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        ok = self._run_roundtrip(num_kv_heads=world_size, test_name="roundtrip_no_replication")
        assert ok, f"Round-trip (no replication) failed on rank {rank}"




# =========================================================================
# Main entry point (for torchrun execution)
# =========================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="KV cache transfer correctness tests"
    )
    parser.add_argument(
        "--num-kv-heads", type=int, default=None,
        help="Run only tests matching this head count (2, 4, or 8)",
    )
    args = parser.parse_args()

    if not _is_distributed():
        print("ERROR: Must run under torchrun with 4 GPUs.")
        print("  torchrun --nproc_per_node=4 test/srt/paras/test_kv_cache_transfer.py")
        sys.exit(1)

    rank, world_size = _ensure_distributed()

    # Clear evidence file
    if rank == 0:
        evidence_dir = os.path.join(_ROOT_DIR, ".sisyphus", "evidence")
        os.makedirs(evidence_dir, exist_ok=True)
        path = os.path.join(evidence_dir, "kv-cache-transfer-test.txt")
        with open(path, "w") as f:
            f.write(f"KV Cache Transfer Tests — {world_size} GPUs\n")
            f.write(f"Tokens per rank: {TOKENS_PER_RANK_4GPU}\n")
            f.write(f"Layers: {NUM_LAYERS}, HEAD_DIM: {HEAD_DIM}\n\n")

    # Build test list
    tests = []

    if args.num_kv_heads is None or args.num_kv_heads == 4:
        tests.append(
            ("ep_to_tp_no_replication",
             TestEPtoTPStandalone().test_ep_to_tp_no_replication)
        )
        tests.append(
            ("tp_to_ep_no_replication",
             TestTPtoEPStandalone().test_tp_to_ep_no_replication)
        )
        tests.append(
            ("roundtrip_no_replication",
             TestKVRoundTrip().test_roundtrip_no_replication)
        )
        if _HAS_PEER_ACCESS:
            tests.append(
                ("ep_to_tp_peer_access_no_replication",
                 TestEPtoTPStandalone().test_ep_to_tp_peer_access_no_replication)
            )
            tests.append(
                ("tp_to_ep_peer_access_no_replication",
                 TestTPtoEPStandalone().test_tp_to_ep_peer_access_no_replication)
            )

    if args.num_kv_heads is None or args.num_kv_heads == world_size * 2:
        if _HAS_PEER_ACCESS:
            tests.append(
                ("ep_to_tp_peer_access_sharded_heads",
                 TestShardedHeads().test_ep_to_tp_peer_access_sharded_heads)
            )
            tests.append(
                ("tp_to_ep_peer_access_sharded_heads",
                 TestShardedHeads().test_tp_to_ep_peer_access_sharded_heads)
            )

    # Replication tests (R=2) are in test_kv_cache_transfer_replication.py
    # to avoid CUDA IPC handle isolation issues when mixing different
    # num_kv_heads in the same process.

    results = []
    for name, fn in tests:
        if rank == 0:
            print(f"\n[Test] {name}", flush=True)
        try:
            fn()
            results.append((name, True))
            if rank == 0:
                print(f"  [PASS] {name}", flush=True)
        except Exception as e:
            results.append((name, False))
            if rank == 0:
                print(f"  [FAIL] {name}: {e}", flush=True)
                import traceback
                traceback.print_exc()

    dist.barrier()

    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    if rank == 0:
        print(f"\n{'=' * 60}")
        print(f"RESULTS: {passed}/{total} tests passed")
        print(f"{'=' * 60}")

    dist.destroy_process_group()
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
