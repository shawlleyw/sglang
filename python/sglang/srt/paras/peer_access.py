"""Peer access initialization and buffer address exchange for ParaS NVLink weight transfers.

Enables CUDA peer access between TP-group GPUs and exchanges managed-buffer
base addresses so that kernels on any rank can directly read peer memory
via NVLink / NVSwitch (CUDA Unified Virtual Addressing).
"""

import ctypes
import logging
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# Load CUDA runtime via ctypes
_cudart = ctypes.CDLL("libcudart.so")


def check_peer_access_available(device_ids: List[int]) -> bool:
    """Check if all GPU pairs in *device_ids* support peer access."""
    can_access = ctypes.c_int(0)
    for i in device_ids:
        for j in device_ids:
            if i != j:
                _cudart.cudaDeviceCanAccessPeer(ctypes.byref(can_access), i, j)
                if can_access.value == 0:
                    return False
    return True


def enable_peer_access(device_ids: List[int]) -> None:
    """Enable CUDA peer access between all GPU pairs.

    Must be called from each rank for its own device.
    Raises ``RuntimeError`` if peer access is not available.
    """
    if not check_peer_access_available(device_ids):
        failed = []
        can_access = ctypes.c_int(0)
        for i in device_ids:
            for j in device_ids:
                if i != j:
                    _cudart.cudaDeviceCanAccessPeer(ctypes.byref(can_access), i, j)
                    if can_access.value == 0:
                        failed.append((i, j))
        raise RuntimeError(
            f"CUDA peer access not available for GPU pairs: {failed}. "
            "NVLink/NVSwitch connectivity required for paras_peer_access method."
        )

    # Enable peer access from this rank's device to every other device
    current_device = torch.cuda.current_device()
    for peer_device in device_ids:
        if peer_device != current_device:
            try:
                _cudart.cudaDeviceEnablePeerAccess(peer_device, 0)
                logger.debug(
                    "Enabled peer access from GPU %d to GPU %d",
                    current_device,
                    peer_device,
                )
            except Exception:
                # cudaErrorPeerAccessAlreadyEnabled (704) is OK
                pass


def exchange_buffer_addresses(
    local_buffer_ptr: int, tp_group, world_size: int
) -> List[int]:
    """Exchange managed-buffer base addresses across all TP ranks via all_gather.

    Returns a list of length *world_size* where index ``r`` contains rank ``r``'s
    buffer address.  With CUDA UVA + peer access enabled these pointers can be
    used directly in kernels.
    """
    # Pack local pointer as int64 tensor
    local_addr = torch.tensor([local_buffer_ptr], dtype=torch.int64, device="cuda")

    # Gather all addresses
    all_addrs = torch.zeros(world_size, dtype=torch.int64, device="cuda")
    dist.all_gather_into_tensor(all_addrs, local_addr, group=tp_group)

    addresses = all_addrs.tolist()

    # Validate: all addresses must be non-zero and distinct
    assert all(a != 0 for a in addresses), f"Some buffer addresses are null: {addresses}"
    assert len(set(addresses)) == len(addresses), (
        f"Buffer addresses are not unique: {addresses}"
    )

    return addresses


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
            assert ret == 0, (
                f"cudaIpcOpenMemHandle for rank {r} failed (cuda error {ret})"
            )
            peer_addresses.append(remote_ptr.value)

    assert all(a != 0 for a in peer_addresses), (
        f"Some IPC buffer addresses are null: {peer_addresses}"
    )
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

    device_ids = list(range(tp_size))
    enable_peer_access(device_ids)

    local_ptr = manager._buffer.data_ptr()
    rank = dist.get_rank(group=tp_group)
    peer_addresses = exchange_buffer_addresses_ipc(
        local_ptr, tp_group, tp_size, rank
    )

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


def peer_access_transfer(
    src_base_ptr: int,
    dst_base_ptrs_tensor: torch.Tensor,
    plan: dict,
    stream: Optional[torch.cuda.Stream] = None,
) -> None:
    """Execute peer access transfer: copy from local staging to peer GPU EP buffers.

    Args:
        src_base_ptr: Local managed buffer base address (int from data_ptr())
        dst_base_ptrs_tensor: int64 GPU tensor of shape [MAX_PEERS] with each rank's buffer address
        plan: dict from pack_transfer_plan() with src_offsets, dst_offsets, sizes, dst_ranks
        stream: CUDA stream to launch on (None = current stream)
    """
    try:
        import paras_peer_access_cuda
    except ImportError:
        raise ImportError(
            "paras_peer_access_cuda extension not found. "
            "Build it first: pip install -e python/sglang/srt/paras/csrc/"
        )

    stream_ptr = 0
    if stream is not None:
        stream_ptr = stream.cuda_stream

    paras_peer_access_cuda.launch_peer_access_transfer(
        src_base_ptr,
        dst_base_ptrs_tensor,
        plan["src_offsets"],
        plan["dst_offsets"],
        plan["sizes"],
        plan["dst_ranks"],
        stream_ptr,
    )


def peer_access_fused_transfer(
    local_buffer_ptr: int,
    dst_base_ptrs_tensor: torch.Tensor,
    src_ep_offset: int,
    dst_tp_offset: int,
    tp_rank: int,
    tp_size: int,
    E_local: int,
    I_prime_H: int,
    num_gates: int,
    elem_size: int = 2,
    stream=None,
) -> None:
    """Fused contiguous-read + peer-write for w13 (tp split on 2nd dim)."""
    import paras_peer_access_cuda
    stream_ptr = stream.cuda_stream if stream is not None else 0
    paras_peer_access_cuda.launch_peer_access_fused_transfer(
        local_buffer_ptr,
        dst_base_ptrs_tensor,
        src_ep_offset,
        dst_tp_offset,
        tp_rank,
        tp_size,
        E_local,
        I_prime_H,
        num_gates,
        elem_size,
        stream_ptr,
    )


def peer_access_fused_transfer_w2(
    local_buffer_ptr: int,
    dst_base_ptrs_tensor: torch.Tensor,
    src_ep_offset: int,
    dst_tp_offset: int,
    tp_rank: int,
    tp_size: int,
    E_local: int,
    H: int,
    I_full: int,
    I_prime: int,
    elem_size: int = 2,
    stream=None,
) -> None:
    """Fused strided-read + peer-write for w2 (tp split on last dim)."""
    import paras_peer_access_cuda
    stream_ptr = stream.cuda_stream if stream is not None else 0
    paras_peer_access_cuda.launch_peer_access_fused_transfer_w2(
        local_buffer_ptr,
        dst_base_ptrs_tensor,
        src_ep_offset,
        dst_tp_offset,
        tp_rank,
        tp_size,
        E_local,
        H,
        I_full,
        I_prime,
        elem_size,
        stream_ptr,
    )
