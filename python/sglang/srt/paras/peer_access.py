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

    # GPU IDs in the TP group — assume GPU 0..tp_size-1 for now
    device_ids = list(range(tp_size))

    # Enable peer access
    enable_peer_access(device_ids)

    # Exchange buffer addresses
    local_ptr = manager._buffer.data_ptr()
    peer_addresses = exchange_buffer_addresses(local_ptr, tp_group, tp_size)

    logger.info(
        "ParaS peer access initialized. Buffer addresses: %s",
        [hex(a) for a in peer_addresses],
    )

    return PeerAccessContext(
        peer_addresses=peer_addresses,
        peer_access_enabled=True,
        tp_group=tp_group,
        tp_size=tp_size,
    )
