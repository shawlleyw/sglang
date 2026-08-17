"""Peer access initialization and buffer address exchange for ParaS NVLink transfers.

Exchanges managed-buffer base addresses via CUDA IPC so that kernels on any
rank can directly write to peer GPU memory via NVLink stores.

NOTE: We intentionally do NOT call cudaDeviceEnablePeerAccess(). The
cudaIpcOpenMemHandle() with cudaIpcMemLazyEnablePeerAccess flag is sufficient
for NVLink stores and avoids creating ~416 MiB CUDA contexts on peer GPUs.
DeepEP uses the same IPC-only approach.
"""

import ctypes
import logging
from dataclasses import dataclass
from typing import List

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# Load CUDA runtime via ctypes
_cudart = ctypes.CDLL("libcudart.so")

_IPC_HANDLE_SIZE = 64


class _CudaIpcMemHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_ubyte * _IPC_HANDLE_SIZE)]


def _setup_ipc_argtypes() -> None:
    """Set ctypes argtypes/restype for CUDA IPC functions (idempotent)."""
    if getattr(_setup_ipc_argtypes, "_done", False):
        return
    _cudart.cudaIpcGetMemHandle.argtypes = [
        ctypes.POINTER(_CudaIpcMemHandle),
        ctypes.c_void_p,
    ]
    _cudart.cudaIpcGetMemHandle.restype = ctypes.c_int
    _cudart.cudaIpcOpenMemHandle.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        _CudaIpcMemHandle,
        ctypes.c_uint,
    ]
    _cudart.cudaIpcOpenMemHandle.restype = ctypes.c_int
    _setup_ipc_argtypes._done = True


def exchange_buffer_addresses_ipc(
    local_buffer_ptr: int, tp_group, world_size: int, rank: int
) -> List[int]:
    """Exchange managed-buffer addresses using CUDA IPC handles.

    Unlike ``exchange_buffer_addresses`` (raw ``data_ptr()``), this works
    across separate OS processes (torchrun / sglang server) because each
    rank opens a cross-process IPC mapping for every remote buffer.
    """
    _setup_ipc_argtypes()

    local_handle = _CudaIpcMemHandle()
    ret = _cudart.cudaIpcGetMemHandle(
        ctypes.byref(local_handle), ctypes.c_void_p(local_buffer_ptr)
    )
    assert ret == 0, f"cudaIpcGetMemHandle failed (cuda error {ret})"

    handle_tensor = torch.tensor(
        list(local_handle.reserved), dtype=torch.uint8, device="cuda"
    )
    all_handles = torch.zeros(
        world_size * _IPC_HANDLE_SIZE, dtype=torch.uint8, device="cuda"
    )
    dist.all_gather_into_tensor(all_handles, handle_tensor, group=tp_group)

    peer_addresses: List[int] = []
    for r in range(world_size):
        if r == rank:
            peer_addresses.append(local_buffer_ptr)
        else:
            raw_bytes = (
                all_handles[r * _IPC_HANDLE_SIZE : (r + 1) * _IPC_HANDLE_SIZE]
                .cpu()
                .tolist()
            )
            remote_handle = _CudaIpcMemHandle()
            for idx, val in enumerate(raw_bytes):
                remote_handle.reserved[idx] = val
            remote_ptr = ctypes.c_void_p()
            ret = _cudart.cudaIpcOpenMemHandle(
                ctypes.byref(remote_ptr),
                remote_handle,
                1,  # cudaIpcMemLazyEnablePeerAccess
            )
            assert (
                ret == 0
            ), f"cudaIpcOpenMemHandle for rank {r} failed (cuda error {ret})"
            peer_addresses.append(remote_ptr.value)

    assert all(
        a != 0 for a in peer_addresses
    ), f"Some IPC buffer addresses are null: {peer_addresses}"
    return peer_addresses


@dataclass
class PeerAccessContext:
    """Holds all state needed for peer-access weight transfers."""

    peer_addresses: List[int]  # Buffer base addresses for each rank
    peer_access_enabled: bool  # Whether peer access was successfully initialized
    tp_group: object  # TP process group handle
    tp_size: int  # TP world size


def init_peer_access(manager, tp_group, tp_size: int) -> PeerAccessContext:
    """Initialize peer access for ParaS weight transfers.

    Call once after ``manager.materialize()`` and before the first EP→TP switch.

    Args:
        manager: :class:`ParaSMemoryManager` (must be materialized).
        tp_group: TP process group (``device_group``).
        tp_size: Size of TP group.

    Returns:
        :class:`PeerAccessContext` with exchanged buffer addresses.
    """
    assert manager.materialized, "manager must be materialized before init_peer_access"

    local_ptr = manager._buffer.data_ptr()
    rank = dist.get_rank(group=tp_group)
    peer_addresses = exchange_buffer_addresses_ipc(local_ptr, tp_group, tp_size, rank)

    logger.info(
        "ParaS peer access initialized (IPC). Buffer addresses: %s",
        [hex(a) for a in peer_addresses],
    )

    return PeerAccessContext(
        peer_addresses=peer_addresses,
        peer_access_enabled=True,
        tp_group=tp_group,
        tp_size=tp_size,
    )


def peer_access_reshard_w13_ep_to_tp_intra_node(
    local_buffer_ptr: int,
    dst_base_ptrs_tensor: torch.Tensor,
    src_ep_offset: int,
    dst_tp_offset: int,
    tp_rank: int,
    tp_size: int,
    num_local_experts: int,
    intermediate_per_tp_times_hidden: int,
    num_gates: int,
    elem_size: int = 2,
    stream=None,
    variant: str = "v2",
    hidden_size: int = None,
) -> None:
    import paras_peer_access_cuda

    stream_ptr = stream.cuda_stream if stream is not None else 0
    if variant == "v3":
        assert hidden_size is not None, "w13 v3 requires hidden_size"
        I = (intermediate_per_tp_times_hidden // hidden_size) * tp_size
        paras_peer_access_cuda.launch_peer_access_fused_transfer_w13_v3(
            local_buffer_ptr,
            dst_base_ptrs_tensor,
            src_ep_offset,
            dst_tp_offset,
            tp_rank,
            tp_size,
            num_local_experts,
            hidden_size,
            I,
            num_gates,
            elem_size,
            stream_ptr,
        )
        return
    paras_peer_access_cuda.launch_peer_access_fused_transfer_w13_v2(
        local_buffer_ptr,
        dst_base_ptrs_tensor,
        src_ep_offset,
        dst_tp_offset,
        tp_rank,
        tp_size,
        num_local_experts,
        intermediate_per_tp_times_hidden,
        num_gates,
        elem_size,
        stream_ptr,
    )


def peer_access_reshard_w2_ep_to_tp_intra_node(
    local_buffer_ptr: int,
    dst_base_ptrs_tensor: torch.Tensor,
    src_ep_offset: int,
    dst_tp_offset: int,
    tp_rank: int,
    tp_size: int,
    num_local_experts: int,
    hidden_size: int,
    full_intermediate: int,
    tp_intermediate: int,
    elem_size: int = 2,
    stream=None,
    variant: str = "v2",
) -> None:
    import paras_peer_access_cuda

    stream_ptr = stream.cuda_stream if stream is not None else 0
    if variant == "v3":
        paras_peer_access_cuda.launch_peer_access_fused_transfer_w2_v3(
            local_buffer_ptr,
            dst_base_ptrs_tensor,
            src_ep_offset,
            dst_tp_offset,
            tp_rank,
            tp_size,
            num_local_experts,
            hidden_size,
            full_intermediate,
            elem_size,
            stream_ptr,
        )
        return
    paras_peer_access_cuda.launch_peer_access_fused_transfer_w2_v2(
        local_buffer_ptr,
        dst_base_ptrs_tensor,
        src_ep_offset,
        dst_tp_offset,
        tp_rank,
        tp_size,
        num_local_experts,
        hidden_size,
        full_intermediate * elem_size,
        tp_intermediate * elem_size,
        stream_ptr,
    )


def peer_access_reshard_w13_tp_to_ep_intra_node(
    local_buffer_ptr: int,
    peer_base_ptrs_tensor: torch.Tensor,
    src_tp_offset: int,
    dst_ep_offset: int,
    tp_rank: int,
    tp_size: int,
    num_local_experts: int,
    intermediate_per_tp_times_hidden: int,
    num_gates: int,
    elem_size: int = 2,
    stream=None,
    variant: str = "v2",
    hidden_size: int = None,
) -> None:
    import paras_peer_access_cuda

    stream_ptr = stream.cuda_stream if stream is not None else 0
    if variant == "v3":
        assert hidden_size is not None, "w13_ep v3 requires hidden_size"
        I = (intermediate_per_tp_times_hidden // hidden_size) * tp_size
        paras_peer_access_cuda.launch_peer_access_fused_transfer_w13_v3_ep(
            local_buffer_ptr,
            peer_base_ptrs_tensor,
            src_tp_offset,
            dst_ep_offset,
            tp_rank,
            tp_size,
            num_local_experts,
            hidden_size,
            I,
            num_gates,
            elem_size,
            stream_ptr,
        )
        return
    paras_peer_access_cuda.launch_peer_access_fused_transfer_w13_ep(
        local_buffer_ptr,
        peer_base_ptrs_tensor,
        src_tp_offset,
        dst_ep_offset,
        tp_rank,
        tp_size,
        num_local_experts,
        intermediate_per_tp_times_hidden,
        num_gates,
        elem_size,
        stream_ptr,
    )


def peer_access_reshard_w2_tp_to_ep_intra_node(
    local_buffer_ptr: int,
    peer_base_ptrs_tensor: torch.Tensor,
    src_tp_offset: int,
    dst_ep_offset: int,
    tp_rank: int,
    tp_size: int,
    num_local_experts: int,
    hidden_size: int,
    full_intermediate: int,
    tp_intermediate: int,
    elem_size: int = 2,
    stream=None,
    variant: str = "v2",
) -> None:
    import paras_peer_access_cuda

    stream_ptr = stream.cuda_stream if stream is not None else 0
    if variant == "v3":
        paras_peer_access_cuda.launch_peer_access_fused_transfer_w2_v3_ep(
            local_buffer_ptr,
            peer_base_ptrs_tensor,
            src_tp_offset,
            dst_ep_offset,
            tp_rank,
            tp_size,
            num_local_experts,
            hidden_size,
            full_intermediate,
            elem_size,
            stream_ptr,
        )
        return
    paras_peer_access_cuda.launch_peer_access_fused_transfer_w2_ep(
        local_buffer_ptr,
        peer_base_ptrs_tensor,
        src_tp_offset,
        dst_ep_offset,
        tp_rank,
        tp_size,
        num_local_experts,
        hidden_size,
        full_intermediate * elem_size,
        tp_intermediate * elem_size,
        stream_ptr,
    )


def peer_access_kv_transfer(
    local_buffer_ptr: int,
    dst_base_ptrs_tensor: torch.Tensor,
    local_token_indices: torch.Tensor,
    src_k_offset: int,
    src_v_offset: int,
    dst_k_offset: int,
    dst_v_offset: int,
    num_local_tokens: int,
    dst_token_start: int,
    num_kv_heads: int,
    tp_rank: int,
    tp_size: int,
    head_dim: int,
    elem_size: int = 2,
    stream=None,
    variant: str = "v2",
) -> None:
    import paras_peer_access_cuda

    stream_ptr = stream.cuda_stream if stream is not None else 0
    launcher = (
        paras_peer_access_cuda.launch_peer_access_kv_transfer_v3
        if variant == "v3"
        else paras_peer_access_cuda.launch_peer_access_kv_transfer
    )
    launcher(
        local_buffer_ptr,
        dst_base_ptrs_tensor,
        local_token_indices,
        src_k_offset,
        src_v_offset,
        dst_k_offset,
        dst_v_offset,
        num_local_tokens,
        dst_token_start,
        num_kv_heads,
        tp_rank,
        tp_size,
        head_dim,
        elem_size,
        stream_ptr,
    )


def peer_access_kv_scatter(
    local_buffer_ptr: int,
    peer_buffer_ptrs_tensor: torch.Tensor,
    tp_token_positions: torch.Tensor,
    token_to_rank: torch.Tensor,
    ep_dst_positions: torch.Tensor,
    src_k_offsets: List[int],
    src_v_offsets: List[int],
    dst_k_offsets: List[int],
    dst_v_offsets: List[int],
    num_local_tokens: int,
    heads_per_rank: int,
    num_kv_heads: int,
    tp_rank: int,
    tp_size: int,
    head_dim: int,
    tp_group,
    num_layers: int,
    elem_size: int = 2,
    stream=None,
    variant: str = "v2",
) -> None:
    """Scatter local TP KV cache to peer EP KV buffers via NVLink for all layers.

    This is the reverse of peer_access_kv_transfer (EP→TP). Each local TP token
    is sent to exactly one EP rank determined by token_to_rank[t].

    Layers are processed in FORWARD order (0 to N-1) with a dist.all_reduce
    barrier after each layer. In the four-anchor layout, writing EP cache
    layer i+1 overlaps TP cache layer i, so layer i must be read before layer
    i+1's EP is written.

    Data flow per token t:
      Source:  local TP buffer at tp_token_positions[t], heads [0, heads_per_rank)
      Dest:   rank token_to_rank[t]'s EP buffer at ep_dst_positions[t],
              heads [tp_rank*heads_per_rank, (tp_rank+1)*heads_per_rank)

    Args:
        local_buffer_ptr: Base address of local managed buffer.
        peer_buffer_ptrs_tensor: GPU tensor of int64 base addresses for each
            rank's managed buffer (from IPC exchange).
        tp_token_positions: GPU int32 tensor [num_local_tokens] — TP pool
            indices for each local token (source positions).
        token_to_rank: GPU int32 tensor [num_local_tokens] — destination EP
            rank for each local token.
        ep_dst_positions: GPU int32 tensor [num_local_tokens] — EP pool
            position on the destination rank for each token.
        src_k_offsets: Per-layer byte offsets to TP K within local buffer.
        src_v_offsets: Per-layer byte offsets to TP V within local buffer.
        dst_k_offsets: Per-layer byte offsets to EP K within peer buffers.
        dst_v_offsets: Per-layer byte offsets to EP V within peer buffers.
        num_local_tokens: Number of tokens on this rank.
        heads_per_rank: Number of KV heads per TP rank (num_kv_heads // tp_size).
        tp_rank: This rank's TP index.
        tp_size: Total number of TP ranks.
        head_dim: Dimension per KV head.
        tp_group: TP process group for barrier synchronization.
        num_layers: Number of model layers to scatter.
        elem_size: Bytes per element (default 2 for bf16/fp16).
        stream: Optional CUDA stream.
    """
    import paras_peer_access_cuda

    stream_ptr = stream.cuda_stream if stream is not None else 0
    barrier = torch.zeros(1, device="cuda")

    for layer_idx in range(num_layers):
        if variant == "v3":
            paras_peer_access_cuda.launch_peer_access_kv_scatter_v3(
                local_buffer_ptr,
                peer_buffer_ptrs_tensor,
                tp_token_positions,
                token_to_rank,
                ep_dst_positions,
                src_k_offsets[layer_idx],
                src_v_offsets[layer_idx],
                dst_k_offsets[layer_idx],
                dst_v_offsets[layer_idx],
                num_local_tokens,
                num_kv_heads,
                tp_rank,
                tp_size,
                head_dim,
                elem_size,
                stream_ptr,
            )
        else:
            paras_peer_access_cuda.launch_peer_access_kv_scatter(
                local_buffer_ptr,
                peer_buffer_ptrs_tensor,
                tp_token_positions,
                token_to_rank,
                ep_dst_positions,
                src_k_offsets[layer_idx],
                src_v_offsets[layer_idx],
                dst_k_offsets[layer_idx],
                dst_v_offsets[layer_idx],
                num_local_tokens,
                heads_per_rank,
                num_kv_heads,
                tp_rank,
                tp_size,
                head_dim,
                elem_size,
                stream_ptr,
            )
        dist.all_reduce(barrier, group=tp_group)
