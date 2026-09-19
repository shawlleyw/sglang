"""EP/TP decisions use enums while emitted metric labels stay compatible."""

import csv
import io
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.managers.paras_metrics_sampler import ParasMetricsSampler
from sglang.srt.managers.paras_per_step_metrics_sampler import (
    ParasPerStepMetricsSampler,
)
from sglang.srt.paras.mode import ParaSMode


@pytest.mark.parametrize("mode", [ParaSMode.EP, ParaSMode.TP])
@pytest.mark.parametrize("static", [False, True])
def test_metrics_keep_mode_labels_and_request_counts(mode, static):
    batch = SimpleNamespace(
        reqs=[object()],
        global_running_reqs=[1, 2],
        global_waiting_reqs=[2, 3],
        global_total_decode_tokens=[10, 20],
        global_total_prefill_tokens=[100, 200],
    )
    scheduler = SimpleNamespace(
        paras_parallelism_config=None if static else mode,
        server_args=SimpleNamespace(enable_dp_attention=mode is ParaSMode.EP),
        last_batch=batch,
        running_batch=batch,
        waiting_queue=[object()],
    )
    steps = ParasPerStepMetricsSampler.__new__(ParasPerStepMetricsSampler)
    steps.scheduler = scheduler
    steps._t0 = None
    snapshot = steps._snapshot(batch)
    assert snapshot["mode"] == mode.name
    assert snapshot["batch_size_global"] == (3 if mode is ParaSMode.EP else 1)

    metrics = ParasMetricsSampler.__new__(ParasMetricsSampler)
    metrics.scheduler = scheduler
    metrics._t0 = 0
    metrics._prev_mode = None
    metrics.interval_sec = 1
    output = io.StringIO()
    metrics._writer = csv.writer(output)
    metrics._sample_once()
    row = next(csv.reader(io.StringIO(output.getvalue())))
    assert row[2] == mode.name
    assert int(row[3]) == (3 if mode is ParaSMode.EP else 1)
    assert int(row[4]) == (5 if mode is ParaSMode.EP else 1)
    assert metrics._prev_mode is mode


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Scheduler imports require CUDA"
)
@pytest.mark.parametrize("hybrid", [False, True])
def test_auto_switch_policy_returns_enum_and_preserves_cooldown(hybrid):
    from sglang.srt.paras.scheduler_paras_mixin import (
        DecodeAutoSwitchPolicy,
        HybridAutoSwitchPolicy,
    )

    cls = HybridAutoSwitchPolicy if hybrid else DecodeAutoSwitchPolicy
    kwargs = {"low_ratio": 0.5} if hybrid else {}
    policy = cls(threshold=8, window=2, cooldown_sec=5, **kwargs)
    policy.window.extend([2, 2])
    assert policy.pick_target(ParaSMode.EP, now=0) is ParaSMode.TP
    policy.window.extend([10, 10])
    assert policy.pick_target(ParaSMode.TP, now=1) is None
    assert policy.pick_target(ParaSMode.TP, now=5) is ParaSMode.EP
