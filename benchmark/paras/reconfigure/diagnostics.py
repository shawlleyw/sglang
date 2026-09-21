"""Memory accounting without changing allocator or garbage-collection policy."""

import os

import torch

ALLOCATOR_COUNTERS = (
    "num_alloc_retries",
    "num_ooms",
    "num_sync_all_streams",
    "num_device_alloc",
    "num_device_free",
)


def memory_snapshot():
    stats = torch.cuda.memory_stats()
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "driver_free_bytes": free,
        "driver_total_bytes": total,
        "inactive_split_bytes": stats.get("inactive_split_bytes.all.current"),
        "allocator_counters": {key: stats.get(key) for key in ALLOCATOR_COUNTERS},
    }


def allocator_delta(before, after):
    return {
        key: (
            after[key] - before[key]
            if before[key] is not None and after[key] is not None
            else None
        )
        for key in ALLOCATOR_COUNTERS
    }


def kv_reservation(manager):
    # Logical view extents cannot be summed across modes: UMM deliberately
    # overlaps KV, weights and scratch. Report each endpoint separately.
    modes = {}
    for mode in ("ep", "tp"):
        modes[mode] = {
            "full_tokens_per_rank": getattr(manager, f"{mode}_max_kv_tokens"),
            "swa_tokens_per_rank": getattr(manager, f"{mode}_max_kv_tokens_swa", 0),
            "logical_kv_bytes": sum(
                entry.size_bytes
                for name, entry in manager._entries.items()
                if f".kv.{mode}." in name
            ),
        }
    buffers = getattr(manager, "_cache_buffers", None)
    return {
        "modes": modes,
        "independent_kv_backing_bytes": (
            sum(t.untyped_storage().nbytes() for t in buffers)
            if buffers is not None
            else None
        ),
        "storage_policy": (
            "separate_layer_buffers_max_of_ep_tp"
            if buffers is not None
            else "production_overlapping_umm"
        ),
        "live_requests_at_switch": 0,
    }


def allocator_configuration():
    return {
        "backend": torch.cuda.memory.get_allocator_backend(),
        "PYTORCH_ALLOC_CONF": os.environ.get("PYTORCH_ALLOC_CONF"),
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    }
