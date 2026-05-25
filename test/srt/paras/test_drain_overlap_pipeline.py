#!/usr/bin/env python3
"""Regression test for the paras overlap-drain merge-batch crash.

Bug history:
  Commit 6b2f1b437 ("fix(paras): merge in-flight prefill into running_batch
  in overlap drain") added inside _paras_drain_overlap_pipeline's while loop:

      self.last_batch = tmp_batch
      self.merge_last_batch()

  where ``tmp_batch`` is popped from ``self.result_queue`` and originates from
  ``event_loop_overlap``'s::

      self.result_queue.append((batch.copy(), batch_result))

  ScheduleBatch.copy() (schedule_batch.py) builds a thin batch that does NOT
  carry sampling_info — its dataclass default (None) is preserved. The merge
  path then crashes inside SamplingBatchInfo.merge_batch at
  ``self.penalizer_orchestrator.merge(other.penalizer_orchestrator)`` with
  ``AttributeError: 'NoneType' object has no attribute 'penalizer_orchestrator'``
  on the NoneType ``other`` (sampling_info of the copied batch).

  Smoke repro: qwen3-235b paras-t64 + rollout autoswitch + 512 concurrent
  bench requests; the first EP->TP autoswitch fires within ~5 s and the
  scheduler crashes at sampling_batch_info.py:315.

This test verifies three layers of the fix:

1. ``ScheduleBatch.copy()`` does not carry ``sampling_info`` — the structural
   invariant the bug depends on.
2. ``ScheduleBatch.merge_batch(other)`` with ``other.sampling_info is None``
   raises the exact production crash signature.
3. ``_paras_drain_overlap_pipeline`` calls ``merge_last_batch`` exactly once,
   AFTER the queue is drained, with ``self.last_batch`` pointing at the
   original batch (the one event_loop_overlap stashed at the end of the
   previous iter), NOT at the popped ``tmp_batch``. A regression that
   reintroduces ``self.last_batch = tmp_batch`` inside the loop fails this
   test.

CPU only. No GPU or distributed required.

Usage::

    python -m pytest test/srt/paras/test_drain_overlap_pipeline.py -v
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


# ---------------------------------------------------------------------------
# Real ScheduleBatch / SamplingBatchInfo are importable on CPU
# (only the deeper paras / quantization chain needs stubbing). Use them
# directly for tests 1 and 2.
# ---------------------------------------------------------------------------


def test_schedule_batch_copy_drops_sampling_info():
    from sglang.srt.managers.schedule_batch import ScheduleBatch

    src = ScheduleBatch(reqs=[])
    src.sampling_info = object()
    assert src.sampling_info is not None

    cp = src.copy()
    assert cp.sampling_info is None, (
        "ScheduleBatch.copy() must omit sampling_info; the paras drain "
        "pipeline relies on this invariant being known so it never feeds "
        "a copied batch to merge_batch."
    )


def test_merge_batch_with_copy_other_raises_on_penalizer_orchestrator():
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

    class _StubPenalizerOrchestrator:
        is_required = False

        def merge(self, other):
            pass

        def filter(self, *_a, **_k):
            pass

    running = ScheduleBatch(reqs=[])
    running.sampling_info = SamplingBatchInfo(
        temperatures=torch.tensor([[1.0]]),
        top_ps=torch.tensor([1.0]),
        top_ks=torch.tensor([1], dtype=torch.int32),
        min_ps=torch.tensor([0.0]),
        is_all_greedy=True,
        need_top_p_sampling=False,
        need_top_k_sampling=False,
        need_min_p_sampling=False,
        vocab_size=10,
        penalizer_orchestrator=_StubPenalizerOrchestrator(),
        device="cpu",
    )

    other_from_copy = ScheduleBatch(reqs=[])
    assert other_from_copy.sampling_info is None

    with pytest.raises(AttributeError, match="penalizer_orchestrator"):
        running.merge_batch(other_from_copy)


# ---------------------------------------------------------------------------
# Test 3: drain pipeline must merge with the original last_batch.
# Needs scheduler_paras_mixin, whose transitive imports pull sgl_kernel
# (CUDA extension). Mirror test_request_partition.py: stub the heavy
# modules, then load the target file by spec.
# ---------------------------------------------------------------------------


def _load_paras_mixin_with_stubs():
    mod_name = "sglang.srt.paras.scheduler_paras_mixin"
    if mod_name in sys.modules:
        return sys.modules[mod_name]

    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    python_root = os.path.join(repo_root, "python")
    mixin_path = os.path.join(
        python_root, "sglang", "srt", "paras", "scheduler_paras_mixin.py"
    )

    heavy_modules = [
        "sglang.srt.managers.io_struct",
        "sglang.srt.managers.schedule_batch",
        "sglang.srt.layers.dp_attention",
        "sglang.srt.mem_cache.memory_pool",
        "sglang.srt.mem_cache.allocator",
        "sglang.srt.model_executor.forward_batch_info",
        "sglang.srt.server_args",
        "sglang.srt.paras.utils",
        "sglang.srt.paras.gather_manager",
        "sglang.srt.paras.scatter_manager",
        "sglang.srt.layers.moe",
        "sglang.srt.layers.moe.utils",
        "sglang.srt.managers.utils",
        "sglang.srt.utils.common",
    ]

    saved = {}
    for name in heavy_modules:
        if name in sys.modules:
            saved[name] = sys.modules[name]
        sys.modules[name] = MagicMock()

    sys.modules["sglang.srt.paras.utils"].__dict__["paras_func"] = lambda fn: fn
    sys.modules["sglang.srt.paras.utils"].__dict__["paras_profile_func"] = (
        lambda fn: fn
    )
    sys.modules["sglang.srt.managers.schedule_batch"].__dict__["Req"] = type(
        "Req", (), {}
    )
    sys.modules["sglang.srt.managers.schedule_batch"].__dict__["ScheduleBatch"] = type(
        "ScheduleBatch", (), {}
    )

    for pkg_path in (
        "sglang",
        "sglang.srt",
        "sglang.srt.paras",
    ):
        if pkg_path not in sys.modules:
            pkg = types.ModuleType(pkg_path)
            pkg.__path__ = [os.path.join(python_root, *pkg_path.split("."))]
            sys.modules[pkg_path] = pkg

    spec = importlib.util.spec_from_file_location(mod_name, mixin_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(mod_name, None)
        for name, orig in saved.items():
            sys.modules[name] = orig
        raise
    return mod


def test_drain_pipeline_merges_with_original_last_batch():
    mixin_mod = _load_paras_mixin_with_stubs()
    drain = mixin_mod.SchedulerParasMixin._paras_drain_overlap_pipeline

    original_batch = SimpleNamespace(_tag="original_last_batch_from_prev_iter")
    tmp_batch = SimpleNamespace(_tag="tmp_batch_from_result_queue_copy")
    tmp_result = SimpleNamespace(_tag="tmp_result")

    seen = {}

    def fake_process_batch_result(batch, result):
        seen["process_called_with"] = (batch, result)

    def fake_merge_last_batch():
        seen.setdefault("merge_calls", []).append(stub.last_batch)

    stub = SimpleNamespace(
        enable_overlap=True,
        result_queue=deque([(tmp_batch, tmp_result)]),
        last_batch=original_batch,
        cur_batch=SimpleNamespace(_tag="cur"),
        process_batch_result=fake_process_batch_result,
        merge_last_batch=fake_merge_last_batch,
    )

    drain(stub)

    assert seen["process_called_with"] == (tmp_batch, tmp_result)
    assert seen["merge_calls"] == [original_batch], (
        "merge_last_batch must be invoked with self.last_batch still pointing "
        "at the original batch from the previous iter (intact sampling_info). "
        "If this lists tmp_batch, a regression has re-introduced "
        "`self.last_batch = tmp_batch` inside the drain loop and merge_batch "
        "will crash in production on a NoneType penalizer_orchestrator."
    )
    assert stub.last_batch is None
    assert stub.cur_batch is None


def test_drain_pipeline_noop_when_overlap_disabled():
    mixin_mod = _load_paras_mixin_with_stubs()
    drain = mixin_mod.SchedulerParasMixin._paras_drain_overlap_pipeline

    sentinel_last = SimpleNamespace(_tag="last")
    sentinel_cur = SimpleNamespace(_tag="cur")
    stub = SimpleNamespace(
        enable_overlap=False,
        result_queue=deque([(object(), object())]),
        last_batch=sentinel_last,
        cur_batch=sentinel_cur,
        process_batch_result=lambda *a, **k: pytest.fail("must not be called"),
        merge_last_batch=lambda: pytest.fail("must not be called"),
    )

    drain(stub)

    assert stub.last_batch is sentinel_last
    assert stub.cur_batch is sentinel_cur


def test_drain_pipeline_noop_when_result_queue_missing():
    mixin_mod = _load_paras_mixin_with_stubs()
    drain = mixin_mod.SchedulerParasMixin._paras_drain_overlap_pipeline

    sentinel_last = SimpleNamespace(_tag="last")
    sentinel_cur = SimpleNamespace(_tag="cur")
    stub = SimpleNamespace(
        enable_overlap=True,
        last_batch=sentinel_last,
        cur_batch=sentinel_cur,
        process_batch_result=lambda *a, **k: pytest.fail("must not be called"),
        merge_last_batch=lambda: pytest.fail("must not be called"),
    )

    drain(stub)

    assert stub.last_batch is sentinel_last
    assert stub.cur_batch is sentinel_cur
