"""Transfer plan builder for ParaS peer access weight transfers.

Precomputes per-layer src/dst byte offsets so the CUDA kernel can
transfer the correct data from local staging buffers to remote
EP/TP buffer regions on peer GPUs via NVLink.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass
class TransferEntry:
    """One peer transfer: copy `size` bytes from local staging to remote EP buffer."""
    src_offset: int   # Byte offset from managed buffer base (local staging buffer region)
    dst_offset: int   # Byte offset from managed buffer base (remote EP/TP buffer region)
    size: int         # Bytes to transfer
    dst_rank: int     # Target GPU rank


def build_transfer_plan(
    manager,
    layer_id: int,
    weight_type: str,      # "w13" or "w2"
    tp_size: int,
    tp_rank: int,
) -> List[TransferEntry]:
    """Build transfer plan for one layer's one weight type (DP=1 only).

    Returns tp_size entries, one per destination rank r.
    This rank (tp_rank) writes its slice of local staging_a to each rank r's EP buffer.

    After the all-to-all equivalent completes, rank r's EP buffer will contain
    contributions from all ranks concatenated along the expert dimension,
    which is the TP weight layout (num_total_experts, 2*I', H).
    """
    assert weight_type in ("w13", "w2"), f"Unknown weight_type: {weight_type}"

    ep_weight_name = f"model.layers.{layer_id}.mlp.experts.{weight_type}_weight"
    staging_name = f"staging.{weight_type}_a"

    ep_entry = manager._entries[ep_weight_name]
    staging_entry = manager._entries[staging_name]

    # Each destination rank receives ep_entry.size_bytes / tp_size bytes
    assert ep_entry.size_bytes % tp_size == 0, (
        f"ep_entry size {ep_entry.size_bytes} not divisible by tp_size {tp_size}"
    )
    slice_size = ep_entry.size_bytes // tp_size

    entries = []
    for r in range(tp_size):
        # Source: slice r of this rank's staging buffer (what goes to rank r)
        src_offset = staging_entry.offset_bytes + r * slice_size
        # Dest: slot tp_rank in rank r's EP buffer (where rank tp_rank's contribution goes)
        dst_offset = ep_entry.offset_bytes + tp_rank * slice_size
        entries.append(TransferEntry(
            src_offset=src_offset,
            dst_offset=dst_offset,
            size=slice_size,
            dst_rank=r,
        ))

    return entries


def build_all_transfer_plans(
    manager,
    num_layers: int,
    tp_size: int,
    tp_rank: int,
) -> Dict[Tuple[int, str], List[TransferEntry]]:
    """Build and cache transfer plans for all layers and both weight types."""
    plans = {}
    for layer_id in range(num_layers):
        for weight_type in ("w13", "w2"):
            key = (layer_id, weight_type)
            plans[key] = build_transfer_plan(
                manager, layer_id, weight_type, tp_size, tp_rank
            )
    return plans


def pack_transfer_plan(entries: List[TransferEntry]) -> dict:
    """Pack transfer entries into GPU tensors for CUDA kernel consumption.

    Returns struct-of-arrays layout for coalesced memory access patterns.
    """
    src_offsets = torch.tensor([e.src_offset for e in entries], dtype=torch.int64).cuda()
    dst_offsets = torch.tensor([e.dst_offset for e in entries], dtype=torch.int64).cuda()
    sizes = torch.tensor([e.size for e in entries], dtype=torch.int64).cuda()
    dst_ranks = torch.tensor([e.dst_rank for e in entries], dtype=torch.int32).cuda()
    return {
        "src_offsets": src_offsets,
        "dst_offsets": dst_offsets,
        "sizes": sizes,
        "dst_ranks": dst_ranks,
        "num_entries": len(entries),
    }
