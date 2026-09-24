"""CPU observations and spawn-safe restart hooks do not alter graph operations."""
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reconfigure import restart_audit
from reconfigure.runtime import Runtime


def test_snapshot_report_counts_real_bytes_and_unpinned_cpu_tensors(monkeypatch):
    runtime = Runtime.__new__(Runtime)
    runtime.manager = SimpleNamespace(host_snapshots={0: {"weight": torch.empty(32, dtype=torch.bfloat16)}})
    runtime.auxiliary_snapshots = [(None, torch.empty(8, dtype=torch.float32))]
    value = runtime.host_snapshot_report()
    assert value["bytes"] == 96 and value["tensor_count"] == 2
    assert value["pinned_bytes"] == 0 and value["all_pinned"] is False
    assert value["host_rss_peak_bytes"] > 0


def test_restart_hooks_are_spawn_picklable():
    for function in (restart_audit.audited_scheduler_process, restart_audit.audited_controller_process):
        assert pickle.loads(pickle.dumps(function)) is function


def test_restart_hook_saves_actual_capture_and_first_replay(tmp_path, monkeypatch):
    calls = []
    weight = SimpleNamespace(shape=(151936, 4096))
    graph = SimpleNamespace(capture_bs=[1, 2, 4], graphs={1: object(), 2: object(), 4: object()})
    def replay(value):
        calls.append(value)
        return value + 1
    graph.replay = replay
    runner = SimpleNamespace(graph_runner=graph,
        server_args=SimpleNamespace(attention_backend="flashinfer", moe_runner_backend="triton"),
        model=SimpleNamespace(model=SimpleNamespace(embed_tokens=SimpleNamespace(weight=weight)),
                              lm_head=SimpleNamespace(weight=weight)))
    class Scheduler:
        def __init__(self):
            self.tp_worker = SimpleNamespace(model_runner=runner)
            self.tp_rank = 3
            self.max_prefill_tokens = 8192
    def run_scheduler_process():
        Scheduler()
        path = tmp_path / "rank-3.json"
        assert json.loads(path.read_text())["replay_count_lower_bound"] == 0
        assert graph.replay(10) == 11
        assert graph.replay(20) == 21
        return 99
    module = SimpleNamespace(Scheduler=Scheduler, run_scheduler_process=run_scheduler_process)
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.scheduler", module)
    monkeypatch.setenv("PARAS_SWITCH_ENGINE_AUDIT_DIR", str(tmp_path))
    assert restart_audit.audited_scheduler_process() == 99
    assert calls == [10, 20]
    result = json.loads((tmp_path / "rank-3.json").read_text())
    assert result["capture_batch_sizes"] == result["captured_graph_keys"] == [1, 2, 4]
    assert result["replay_count_lower_bound"] == 1
    assert result["rank"] == 3
