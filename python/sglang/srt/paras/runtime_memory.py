"""Reclaimable, mode-local CUDA graph scratch with stable virtual addresses.

Only disposable scratch belongs here. Weights, KV contents, request mappings,
collective buffers and graph-private allocator pools are deliberately excluded.
No CUDA library is loaded until the opt-in CUDA allocator is constructed.
"""

import ctypes as C
import logging
import math

import torch

from sglang.srt.paras.mode import ParaSMode

logger = logging.getLogger(__name__)


class _Location(C.Structure):
    _fields_ = [("type", C.c_int), ("id", C.c_int)]


class _AllocationFlags(C.Structure):
    _fields_ = [
        ("compressionType", C.c_ubyte),
        ("gpuDirectRDMACapable", C.c_ubyte),
        ("usage", C.c_ushort),
        ("reserved", C.c_ubyte * 4),
    ]


class _AllocationProp(C.Structure):
    _fields_ = [
        ("type", C.c_int),
        ("requestedHandleTypes", C.c_int),
        ("location", _Location),
        ("win32HandleMetaData", C.c_void_p),
        ("allocFlags", _AllocationFlags),
    ]


class _AccessDesc(C.Structure):
    _fields_ = [("location", _Location), ("flags", C.c_int)]


class CudaVmmDriver:
    """Small CUDA Driver API adapter; PyTorch must establish the context first."""

    def __init__(self, device):
        self.lib = C.CDLL("libcuda.so.1")
        u64, size, ptr = C.c_uint64, C.c_size_t, C.c_void_p
        signatures = {
            "cuDeviceGetAttribute": [ptr, C.c_int, C.c_int],
            "cuMemGetAllocationGranularity": [ptr, ptr, C.c_int],
            "cuMemAddressReserve": [ptr, size, size, u64, u64],
            "cuMemAddressFree": [u64, size],
            "cuMemCreate": [ptr, size, ptr, u64],
            "cuMemMap": [u64, size, size, u64, u64],
            "cuMemSetAccess": [u64, size, ptr, size],
            "cuMemUnmap": [u64, size],
            "cuMemRelease": [u64],
            "cuMemsetD8_v2": [u64, C.c_ubyte, size],
        }
        for name, args in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, C.c_int
        supported = C.c_int()
        self._call("cuDeviceGetAttribute", C.byref(supported), 102, device)
        if not supported.value:
            raise RuntimeError("ParaS runtime VMM is unsupported on this CUDA device")
        self.prop = _AllocationProp(type=1, location=_Location(1, device))
        self.access = _AccessDesc(_Location(1, device), 3)
        granularity = size()
        self._call(
            "cuMemGetAllocationGranularity", C.byref(granularity), C.byref(self.prop), 0
        )
        self.granularity = granularity.value

    def _call(self, name, *args):
        result = getattr(self.lib, name)(*args)
        if result:
            raise RuntimeError(f"ParaS VMM {name} failed with CUDA error {result}")

    def reserve(self, size):
        address = C.c_uint64()
        self._call("cuMemAddressReserve", C.byref(address), size, 0, 0, 0)
        return address.value

    def free_address(self, address, size):
        self._call("cuMemAddressFree", address, size)

    def map(self, address, size):
        handle = C.c_uint64()
        self._call("cuMemCreate", C.byref(handle), size, C.byref(self.prop), 0)
        mapped = False
        try:
            self._call("cuMemMap", address, size, 0, handle, 0)
            mapped = True
            self._call("cuMemSetAccess", address, size, C.byref(self.access), 1)
            # Scratch contents are disposable. Reset them before reusing a graph.
            self._call("cuMemsetD8_v2", address, 0, size)
        except BaseException:
            if mapped:
                self.unmap(address, size)
            raise
        finally:
            # The mapping owns a reference. Once unmapped there are no handles
            # keeping its physical pages alive (unlike retaining a cuMemCreate handle).
            self._call("cuMemRelease", handle)

    def unmap(self, address, size):
        self._call("cuMemUnmap", address, size)


class VmmAllocation:
    def __init__(self, driver, size):
        self.driver = driver
        self.size = (
            (size + driver.granularity - 1) // driver.granularity
        ) * driver.granularity
        if self.size <= 0:
            raise ValueError("VMM allocations must be nonempty")
        self.address = driver.reserve(self.size)
        self.mapped = False

    def resume(self):
        if not self.mapped:
            self.driver.map(self.address, self.size)
            self.mapped = True

    def suspend(self):
        if self.mapped:
            self.driver.unmap(self.address, self.size)
            self.mapped = False

    def close(self):
        if self.address:
            self.suspend()
            self.driver.free_address(self.address, self.size)
            self.address = 0

    def __del__(self):
        # Tensor storage retains its CUDA-array-interface owner. In normal use
        # saved graph states retain the tensors until the worker exits.
        if getattr(self, "address", 0):
            try:
                self.close()
            except Exception:
                logger.exception("Could not release ParaS runtime VMM allocation")


class _CudaArray:
    def __init__(self, allocation, shape, dtype):
        self.allocation = allocation
        self.__cuda_array_interface__ = {
            "shape": tuple(shape),
            "strides": None,
            "typestr": {torch.int64: "<i8", torch.float32: "<f4"}[dtype],
            "data": (allocation.address, False),
            "version": 3,
        }


class ModeRuntimeMemory:
    """Own scratch mappings for both modes; only the active mode is backed.

    Call activate only at a drained mode-switch boundary. synchronize must wait
    for all streams on this device, including outstanding users of output logits.
    An allocation failure propagates; no graph may launch until activate succeeds.
    """

    def __init__(self, driver, synchronize):
        self.driver = driver
        self.synchronize = synchronize
        self.active = ParaSMode.EP
        self.allocations = {mode: {} for mode in (ParaSMode.EP, ParaSMode.TP)}
        self.failed = False

    def allocate(self, mode, name, size):
        if self.failed or mode != self.active:
            raise RuntimeError("Cannot allocate scratch for an inactive/failed mode")
        if name in self.allocations[mode]:
            raise RuntimeError(f"Duplicate ParaS scratch allocation: {mode}/{name}")
        allocation = VmmAllocation(self.driver, size)
        try:
            allocation.resume()
        except BaseException:
            allocation.close()
            raise
        self.allocations[mode][name] = allocation
        return allocation

    def activate(self, mode):
        if self.failed:
            raise RuntimeError("ParaS VMM activation previously failed; restart worker")
        if mode == self.active:
            return
        self.synchronize()
        # Fail closed: a partial unmap/map cannot leave a replayable mode. The
        # caller must abort the switch/worker instead of launching stale pointers.
        try:
            for allocation in self.allocations[self.active].values():
                allocation.suspend()
            for allocation in self.allocations[mode].values():
                allocation.resume()
            # Complete scratch initialization before replay on another stream.
            self.synchronize()
        except BaseException:
            self.failed = True
            raise
        self.active = mode

    def check_ready(self):
        if self.failed:
            raise RuntimeError("ParaS VMM activation previously failed; restart worker")

    def stats(self):
        return {
            mode.value: {
                "virtual_bytes": sum(a.size for a in allocations.values()),
                "resident_bytes": sum(a.size for a in allocations.values() if a.mapped),
            }
            for mode, allocations in self.allocations.items()
        }


class CudaModeRuntimeMemory(ModeRuntimeMemory):
    def __init__(self, device):
        self.device = torch.device(device)
        with torch.cuda.device(self.device):
            torch.cuda.init()
            self.device = torch.device("cuda", torch.cuda.current_device())
            driver = CudaVmmDriver(self.device.index)
        super().__init__(driver, lambda: torch.cuda.synchronize(self.device))

    def zeros(self, mode, name, shape, dtype):
        with torch.cuda.device(self.device):
            allocation = self.allocate(
                mode,
                name,
                math.prod(shape)
                * torch.empty((), dtype=dtype, device="cpu").element_size(),
            )
            self.synchronize()
            owner = _CudaArray(allocation, shape, dtype)
            tensor = torch.as_tensor(owner, device=self.device)
            if tensor.data_ptr() != allocation.address:
                raise RuntimeError(
                    "PyTorch copied ParaS VMM storage instead of aliasing it"
                )
            return tensor

    def activate(self, mode):
        with torch.cuda.device(self.device):
            super().activate(mode)
        logger.info("ParaS runtime VMM: active=%s bytes=%s", mode, self.stats())


def validate_runtime_vmm_config(args):
    """Reject unsupported opt-in combinations before constructing a worker."""
    required = (
        (args.enable_paras_moe, "--enable-paras-moe"),
        (not args.disable_cuda_graph, "CUDA graphs"),
        (args.attention_backend == "triton", "--attention-backend triton"),
        (args.device in (None, "cuda"), "CUDA"),
        (args.speculative_algorithm is None, "no speculative decoding"),
        (args.pp_size == 1, "pipeline parallel size 1"),
        (not args.enable_two_batch_overlap, "two-batch overlap disabled"),
        (not args.enable_pdmux, "pdmux disabled"),
        (not args.enable_torch_compile, "torch compile disabled"),
        (not args.enable_memory_saver, "memory saver disabled"),
        (args.prefill_attention_backend in (None, "triton"), "Triton prefill"),
        (args.decode_attention_backend in (None, "triton"), "Triton decode"),
    )
    for supported, requirement in required:
        if not supported:
            raise ValueError(f"--paras-vmm-runtime-states requires {requirement}")
