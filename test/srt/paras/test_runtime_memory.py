"""CPU-only tests of VMM ownership and mode-switch failure handling."""

from types import SimpleNamespace

import pytest

from sglang.srt.paras.mode import ParaSMode
from sglang.srt.paras.runtime_memory import (
    ModeRuntimeMemory,
    VmmAllocation,
    validate_runtime_vmm_config,
)

EP, TP = ParaSMode.EP, ParaSMode.TP


class FakeDriver:
    granularity = 64

    def __init__(self):
        self.next_address = 1 << 40
        self.reserved = {}
        self.resident = {}
        self.events = []
        self.fail_map = False
        self.fail_unmap = False
        self.peak = 0

    def reserve(self, size):
        address = self.next_address
        self.next_address += size
        self.reserved[address] = size
        return address

    def free_address(self, address, size):
        assert address not in self.resident
        assert self.reserved.pop(address) == size

    def map(self, address, size):
        self.events.append(("map", address))
        if self.fail_map:
            raise RuntimeError("injected allocation failure")
        assert address not in self.resident
        assert self.reserved[address] == size
        self.resident[address] = size
        self.peak = max(self.peak, sum(self.resident.values()))

    def unmap(self, address, size):
        self.events.append(("unmap", address))
        if self.fail_unmap:
            raise RuntimeError("injected unmap failure")
        assert self.resident.pop(address) == size


def manager():
    driver = FakeDriver()
    return ModeRuntimeMemory(driver, lambda: driver.events.append(("sync",))), driver


def test_switch_releases_physical_memory_but_preserves_virtual_addresses():
    memory, driver = manager()
    ep = memory.allocate(EP, "indices", 65)
    ep_address = ep.address
    assert ep.size == 128
    memory.activate(TP)
    assert not driver.resident
    tp = memory.allocate(TP, "indices", 1024)
    tp_address = tp.address
    for _ in range(3):
        memory.activate(EP)
        assert driver.resident == {ep_address: 128}
        memory.activate(TP)
        assert driver.resident == {tp_address: 1024}
    assert ep.address == ep_address and tp.address == tp_address
    assert len(driver.reserved) == 2
    assert driver.peak == 1024  # Never pay EP + TP backing simultaneously.
    assert memory.stats() == {
        "ep": {"virtual_bytes": 128, "resident_bytes": 0},
        "tp": {"virtual_bytes": 1024, "resident_bytes": 1024},
    }
    for index, event in enumerate(driver.events):
        if event[0] == "unmap":
            assert driver.events[index - 1] == ("sync",)


def test_switch_to_active_mode_is_a_noop():
    memory, driver = manager()
    memory.allocate(EP, "indices", 64)
    before = list(driver.events)
    memory.activate(EP)
    assert driver.events == before


def test_multiple_buffers_are_all_suspended():
    memory, driver = manager()
    memory.allocate(EP, "indices", 128)
    memory.allocate(EP, "logits", 256)
    memory.activate(TP)
    assert not driver.resident
    memory.activate(EP)
    assert sum(driver.resident.values()) == 384


@pytest.mark.parametrize("failure", ["fail_map", "fail_unmap"])
def test_failed_switch_blocks_replay_and_later_activation(failure):
    memory, driver = manager()
    memory.allocate(EP, "indices", 64)
    memory.activate(TP)
    memory.allocate(TP, "indices", 512)
    setattr(driver, failure, True)
    with pytest.raises(RuntimeError, match="injected"):
        memory.activate(EP)
    for operation in (memory.check_ready, lambda: memory.activate(TP)):
        with pytest.raises(RuntimeError, match="restart worker"):
            operation()
    setattr(driver, failure, False)  # Permit deterministic cleanup.


def test_initial_allocation_failure_frees_reserved_address():
    memory, driver = manager()
    driver.fail_map = True
    with pytest.raises(RuntimeError, match="allocation failure"):
        memory.allocate(EP, "indices", 64)
    assert not driver.reserved and not driver.resident
    assert not memory.allocations[EP]


def test_allocation_ownership_and_close_are_idempotent():
    driver = FakeDriver()
    allocation = VmmAllocation(driver, 1)
    allocation.resume()
    allocation.resume()
    allocation.suspend()
    allocation.suspend()
    allocation.close()
    allocation.close()
    assert not driver.reserved and not driver.resident


def test_duplicate_and_inactive_allocations_are_rejected():
    memory, driver = manager()
    memory.allocate(EP, "indices", 64)
    with pytest.raises(RuntimeError, match="Duplicate"):
        memory.allocate(EP, "indices", 64)
    with pytest.raises(RuntimeError, match="inactive"):
        memory.allocate(TP, "indices", 64)
    assert len(driver.reserved) == 1


def valid_config():
    return SimpleNamespace(
        enable_paras_moe=True,
        disable_cuda_graph=False,
        attention_backend="triton",
        device="cuda",
        speculative_algorithm=None,
        pp_size=1,
        enable_two_batch_overlap=False,
        enable_pdmux=False,
        enable_torch_compile=False,
        enable_memory_saver=False,
        prefill_attention_backend=None,
        decode_attention_backend=None,
    )


def test_supported_configuration():
    validate_runtime_vmm_config(valid_config())


@pytest.mark.parametrize(
    "key,value",
    [
        ("enable_paras_moe", False),
        ("disable_cuda_graph", True),
        ("attention_backend", "fa3"),
        ("device", "cpu"),
        ("speculative_algorithm", "EAGLE"),
        ("pp_size", 2),
        ("enable_two_batch_overlap", True),
        ("enable_pdmux", True),
        ("enable_torch_compile", True),
        ("enable_memory_saver", True),
        ("prefill_attention_backend", "fa3"),
        ("decode_attention_backend", "fa3"),
    ],
)
def test_unsupported_configurations_fail_before_cuda_initialization(key, value):
    args = valid_config()
    setattr(args, key, value)
    with pytest.raises(ValueError, match="--paras-vmm-runtime-states requires"):
        validate_runtime_vmm_config(args)


@pytest.mark.parametrize(
    "failure", [None, "cuMemMap", "cuMemSetAccess", "cuMemsetD8_v2"]
)
def test_driver_releases_allocation_handles_and_cleans_failed_mappings(
    monkeypatch, failure
):
    """Exercise the actual ctypes adapter against a fake CUDA shared library."""
    import ctypes as C
    from sglang.srt.paras import runtime_memory as rm

    events = []
    live_handles, mappings = set(), set()

    class Function:
        def __init__(self, name):
            self.name = name

        def __call__(self, *args):
            name = self.name
            events.append(name)
            if name == failure:
                return 2
            if name == "cuDeviceGetAttribute":
                C.cast(args[0], C.POINTER(C.c_int))[0] = 1
            elif name == "cuMemGetAllocationGranularity":
                C.cast(args[0], C.POINTER(C.c_size_t))[0] = 64
            elif name == "cuMemAddressReserve":
                C.cast(args[0], C.POINTER(C.c_uint64))[0] = 1 << 40
            elif name == "cuMemCreate":
                C.cast(args[0], C.POINTER(C.c_uint64))[0] = 7
                live_handles.add(7)
            elif name == "cuMemMap":
                mappings.add(args[0])
            elif name == "cuMemRelease":
                live_handles.remove(args[0].value)
            elif name == "cuMemUnmap":
                mappings.remove(args[0])
            return 0

    class Library:
        def __getattr__(self, name):
            fn = Function(name)
            setattr(self, name, fn)
            return fn

    monkeypatch.setattr(rm.C, "CDLL", lambda name: Library())
    driver = rm.CudaVmmDriver(0)
    allocation = VmmAllocation(driver, 128)
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            allocation.resume()
        assert not mappings
    else:
        allocation.resume()
        assert mappings == {allocation.address}
    # The mapped allocation must not retain a handle pinning physical pages.
    assert not live_handles
    allocation.close()
    assert not mappings
    assert events[-1] == "cuMemAddressFree"


def test_cuda_manager_pins_the_device_ordinal_without_using_cuda(monkeypatch):
    from contextlib import nullcontext
    from sglang.srt.paras import runtime_memory as rm

    requested_devices = []
    monkeypatch.setattr(rm.torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(rm.torch.cuda, "init", lambda: None)
    monkeypatch.setattr(rm.torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(
        rm,
        "CudaVmmDriver",
        lambda device: requested_devices.append(device) or FakeDriver(),
    )
    memory = rm.CudaModeRuntimeMemory("cuda")
    assert memory.device == rm.torch.device("cuda:3")
    assert requested_devices == [3]


@pytest.mark.parametrize("backend", ["triton", "flashinfer"])
def test_matching_prefill_decode_backends_supported(backend):
    args = valid_config()
    args.attention_backend = backend
    args.prefill_attention_backend = backend
    args.decode_attention_backend = backend
    validate_runtime_vmm_config(args)


def test_mixed_attention_backends_rejected():
    args = valid_config()
    args.attention_backend = "flashinfer"
    args.decode_attention_backend = "triton"
    with pytest.raises(ValueError, match="matching"):
        validate_runtime_vmm_config(args)


@pytest.mark.parametrize("dtype,typestr", [("int32", "<i4"), ("uint8", "|u1")])
def test_flashinfer_cuda_array_interface_dtypes(dtype, typestr):
    from sglang.srt.paras import runtime_memory as rm

    allocation = VmmAllocation(FakeDriver(), 128)
    owner = rm._CudaArray(allocation, (32,), getattr(rm.torch, dtype))
    interface = owner.__cuda_array_interface__
    assert interface["typestr"] == typestr
    assert interface["data"] == (allocation.address, False)
    assert owner.allocation is allocation
