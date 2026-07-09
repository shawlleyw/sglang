"""IPC arena + CudaTimer for the ParaS peer-access microbenches.

torchrun-only: every worker owns one GPU and allocates a single uint8 arena
which is IPC-exchanged so every rank knows where every peer's arena lives.
Mirrors the production ParaSMemoryManager carving without depending on it.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.distributed as dist

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "python"))

from sglang.srt.paras.peer_access import exchange_buffer_addresses_ipc  # noqa: E402


@dataclass
class IPCContext:
    rank: int
    world_size: int
    tp_group: dist.ProcessGroup
    device: torch.device
    buf: torch.Tensor
    local_buffer_ptr: int
    peer_buffer_ptrs: torch.Tensor

    def barrier(self) -> None:
        bar = torch.zeros(1, device=self.device)
        dist.all_reduce(bar, group=self.tp_group)


def init_torchrun() -> Tuple[int, int]:
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def setup_ipc_arena(buf_bytes: int) -> IPCContext:
    rank, world_size = init_torchrun()
    device = torch.device(f"cuda:{rank}")
    buf = torch.zeros(buf_bytes, dtype=torch.uint8, device=device)
    local_ptr = buf.data_ptr()
    tp_group = dist.group.WORLD
    peer_addrs = exchange_buffer_addresses_ipc(local_ptr, tp_group, world_size, rank)
    peer_ptrs = torch.tensor(peer_addrs, dtype=torch.int64, device=device)
    return IPCContext(
        rank=rank,
        world_size=world_size,
        tp_group=tp_group,
        device=device,
        buf=buf,
        local_buffer_ptr=local_ptr,
        peer_buffer_ptrs=peer_ptrs,
    )


class CudaTimer:
    """Paired-CUDA-event timer. Usage:

        timer = CudaTimer(device, warmup=5, iters=20)
        for _ in range(timer.total_iters):
            timer.tick(); <launch>; timer.tock()
        timer.summary()  # {mean, p50, p90, min, max, n}
    """

    def __init__(self, device: torch.device, warmup: int = 5, iters: int = 20):
        self.device = device
        self.warmup = warmup
        self.iters = iters
        self.total_iters = warmup + iters
        self._starts: List[torch.cuda.Event] = []
        self._ends: List[torch.cuda.Event] = []

    def tick(self) -> None:
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self._starts.append(e)

    def tock(self) -> None:
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self._ends.append(e)

    def summary(self) -> dict:
        torch.cuda.synchronize(self.device)
        all_times = [s.elapsed_time(e) for s, e in zip(self._starts, self._ends)]
        kept = all_times[self.warmup:]
        if not kept:
            return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p20": 0.0, "n": 0}
        kept_sorted = sorted(kept)
        n = len(kept_sorted)
        return {
            "mean": sum(kept) / n,
            "p20": kept_sorted[max(0, int(n * 0.2) - 1)],
            "p50": kept_sorted[n // 2],
            "p90": kept_sorted[min(n - 1, int(n * 0.9))],
            "min": kept_sorted[0],
            "max": kept_sorted[-1],
            "n": n,
        }
