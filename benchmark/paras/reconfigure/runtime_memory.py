"""Benchmark-only scratch reuse for repeated capture with production CUDA VMM.

Production captures each mode once, so its allocator rejects duplicate names.
Recapture baselines keep those same scratch tensor addresses (not graph objects)
and reinitialize them when production buffer setup requests the name again.
Physical map/unmap, synchronization and failure handling remain production code.
"""

import torch

from sglang.srt.paras.runtime_memory import CudaModeRuntimeMemory


class RecaptureRuntimeMemory(CudaModeRuntimeMemory):
    def __init__(self, device):
        super().__init__(device)
        self._scratch_tensors = {}

    def zeros(self, mode, name, shape, dtype):
        self.check_ready()
        if mode != self.active:
            raise RuntimeError("Cannot initialize scratch for an inactive mode")
        key = (mode, name)
        tensor = self._scratch_tensors.get(key)
        if tensor is None:
            tensor = super().zeros(mode, name, shape, dtype)
            self._scratch_tensors[key] = tensor
        else:
            if tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype:
                raise RuntimeError(
                    f"Recaptured scratch changed shape/dtype: {mode}/{name}"
                )
            # init_graph_buffers / init_cuda_graph_state expect zeros. This is
            # graph-capture initialization, charged to the Graph bucket.
            with torch.cuda.device(self.device):
                tensor.zero_()
        return tensor


def install_recapture_vmm():
    """Call before Scheduler construction, only in a recapture worker process."""
    from sglang.srt.paras import runtime_memory

    runtime_memory.CudaModeRuntimeMemory = RecaptureRuntimeMemory


def runtime_memory_report(graph_runner, mode, *, enabled):
    """Check physical residency outside timing; PyTorch stats omit VMM pages."""
    memory = getattr(graph_runner, "_paras_runtime_memory", None)
    if (memory is not None) != enabled:
        raise RuntimeError("Requested and effective runtime VMM settings differ")
    if memory is None:
        return {"enabled": False, "active_mode": mode.value, "modes": None}
    memory.check_ready()
    if memory.active != mode:
        raise RuntimeError("Runtime VMM mode differs from scheduler mode")
    stats = memory.stats()
    for name, sizes in stats.items():
        expected = sizes["virtual_bytes"] if name == mode.value else 0
        if sizes["resident_bytes"] != expected:
            raise RuntimeError(f"Unexpected runtime VMM residency in {name}")
    return {"enabled": True, "active_mode": mode.value, "modes": stats}
