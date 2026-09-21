"""CPU VMM lifecycle checks, including production graph-buffer initialization."""

from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1]),
    str(Path(__file__).resolve().parents[3] / "python"),
]

from reconfigure.runtime import Runtime
from reconfigure.runtime_memory import (
    RecaptureRuntimeMemory,
    install_recapture_vmm,
    runtime_memory_report,
)
from sglang.srt.paras.mode import ParaSMode
from sglang.srt.paras.runtime_memory import CudaModeRuntimeMemory, ModeRuntimeMemory
from test.srt.paras.test_runtime_memory import FakeDriver
from test.srt.paras.test_runtime_states import backend, load_nodes

EP, TP = ParaSMode.EP, ParaSMode.TP


@pytest.fixture
def memory(monkeypatch):
    monkeypatch.setattr(
        torch.cuda, "_lazy_init", lambda: pytest.fail("CPU test used CUDA")
    )
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())

    def cpu_zeros(self, mode, name, shape, dtype):
        tensor = torch.zeros(shape, dtype=dtype, device="cpu")
        self.allocate(mode, name, tensor.numel() * tensor.element_size())
        return tensor

    monkeypatch.setattr(CudaModeRuntimeMemory, "zeros", cpu_zeros)
    memory = RecaptureRuntimeMemory.__new__(RecaptureRuntimeMemory)
    driver = FakeDriver()
    ModeRuntimeMemory.__init__(memory, driver, lambda: driver.events.append(("sync",)))
    memory.device = "cpu"
    memory._scratch_tensors = {}
    return memory


def test_recapture_reuses_tensor_addresses_without_retaining_physical_backing(memory):
    tensors, pointers = {}, {}
    for mode, size in ((EP, 8), (TP, 64), (EP, 8), (TP, 64), (EP, 8)):
        memory.activate(mode)
        tensor = memory.zeros(mode, "logits", (size,), torch.float32)
        assert not tensor.count_nonzero()
        if mode in tensors:
            assert tensor is tensors[mode]
            assert tensor.data_ptr() == pointers[mode]
        tensors[mode], pointers[mode] = tensor, tensor.data_ptr()
        tensor.fill_(123)
        report = runtime_memory_report(
            SimpleNamespace(_paras_runtime_memory=memory), mode, enabled=True
        )
        other = TP if mode == EP else EP
        assert report["modes"][other.value]["resident_bytes"] == 0
    assert len(memory.driver.reserved) == 2
    assert memory.driver.peak == 256


@pytest.mark.parametrize("shape,dtype", [((16,), torch.float32), ((8,), torch.int64)])
def test_recapture_rejects_shape_or_dtype_drift(memory, shape, dtype):
    memory.zeros(EP, "logits", (8,), torch.float32)
    with pytest.raises(RuntimeError, match="shape/dtype"):
        memory.zeros(EP, "logits", shape, dtype)
    assert len(memory.driver.reserved) == 1


def test_recapture_cannot_touch_unmapped_or_failed_scratch(memory):
    memory.zeros(EP, "logits", (8,), torch.float32)
    memory.activate(TP)
    with pytest.raises(RuntimeError, match="inactive"):
        memory.zeros(EP, "logits", (8,), torch.float32)
    memory.driver.fail_map = True
    with pytest.raises(RuntimeError, match="injected"):
        memory.activate(EP)
    with pytest.raises(RuntimeError, match="restart worker"):
        memory.zeros(EP, "logits", (8,), torch.float32)
    memory.driver.fail_map = False


def test_vmm_report_rejects_setting_mode_or_residency_mismatch(memory):
    gr = SimpleNamespace(_paras_runtime_memory=memory)
    memory.zeros(EP, "logits", (8,), torch.float32)
    with pytest.raises(RuntimeError, match="settings differ"):
        runtime_memory_report(gr, EP, enabled=False)
    with pytest.raises(RuntimeError, match="scheduler mode"):
        runtime_memory_report(gr, TP, enabled=True)
    memory.allocations[EP]["logits"].suspend()
    with pytest.raises(RuntimeError, match="residency"):
        runtime_memory_report(gr, EP, enabled=True)
    assert runtime_memory_report(SimpleNamespace(), EP, enabled=False)["modes"] is None


def test_installer_only_replaces_allocator_in_benchmark_worker(monkeypatch):
    from sglang.srt.paras import runtime_memory as production

    with monkeypatch.context() as scoped:
        scoped.setattr(production, "CudaModeRuntimeMemory", CudaModeRuntimeMemory)
        install_recapture_vmm()
        assert production.CudaModeRuntimeMemory is RecaptureRuntimeMemory
    assert production.CudaModeRuntimeMemory is CudaModeRuntimeMemory


def test_repeated_capture_uses_production_graph_buffers_and_vmm_names(
    memory, monkeypatch
):
    """EP→TP→EP recaptures must not allocate duplicate production scratch names."""
    cls = load_nodes(
        "model_executor/cuda_graph_runner.py",
        {"init_graph_buffers", "_cache_loc_dtype"},
        "CudaGraphRunner",
        TboCudaGraphRunnerPlugin=object,
    )
    attn = backend()
    attn._paras_runtime_memory = memory
    runner = SimpleNamespace(
        attn_backend=attn,
        model_config=SimpleNamespace(vocab_size=16),
        spec_algorithm=SimpleNamespace(is_eagle3=lambda: False),
    )
    gr = SimpleNamespace(
        device="cpu",
        seq_len_fill_value=1,
        pp_size=1,
        is_encoder_decoder=False,
        require_gathered_buffer=False,
        _paras_runtime_memory=memory,
        model_runner=runner,
        num_tokens_per_bs=1,
        graphs={},
        output_buffers={},
        _paras_saved={},
    )
    gr.init_graph_buffers = cls.init_graph_buffers.__get__(gr)
    gr._cache_loc_dtype = cls._cache_loc_dtype.__get__(gr)
    runner.graph_runner = gr
    runtime = Runtime.__new__(Runtime)
    runtime.runner = runner
    runtime.scheduler = SimpleNamespace(paras_parallelism_config=EP)
    runtime.graph_batches = {EP: [1, 2], TP: [1, 2, 4, 8]}
    gr.capture = lambda: gr.graphs.update(dict.fromkeys(gr.capture_bs))
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.model_executor.cuda_graph_runner",
        SimpleNamespace(
            model_capture_mode=nullcontext,
            set_global_graph_memory_pool=lambda pool: None,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.paras.paras_cuda_graph",
        SimpleNamespace(
            paras_refresh_cuda_graph_settings=lambda gr: None,
        ),
    )
    monkeypatch.setattr("gc.collect", lambda: pytest.fail("explicit disposal GC"))
    pointers = {}
    for mode in (EP, TP, EP, TP, EP):
        runtime.discard_graphs()
        memory.activate(mode)
        runtime.scheduler.paras_parallelism_config = mode
        attn._paras_workspace_mode = mode
        runtime.capture()
        shape = (max(runtime.graph_batches[mode]), 16)
        assert gr.next_token_logits_buffer.shape == shape
        assert attn.cuda_graph_kv_indices.numel() == shape[0] * 32
        current = (
            gr.next_token_logits_buffer.data_ptr(),
            attn.cuda_graph_kv_indices.data_ptr(),
        )
        if mode in pointers:
            assert pointers[mode] == current
        pointers[mode] = current
        runtime_memory_report(gr, mode, enabled=True)
    assert set(memory.allocations[EP]) == {"logits", "kv_indices"}
    assert set(memory.allocations[TP]) == {"logits", "kv_indices"}
    assert len(memory.driver.reserved) == 4
    assert memory.driver.peak == sum(a.size for a in memory.allocations[TP].values())
