"""CPU-only ordering checks for the real runtime adapter using scheduler mocks."""

import os
import sys
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../python"))
)

from common.weight_bundle import ParaSMode
from reconfigure.runtime import Runtime


class RuntimeOrderingTest(unittest.TestCase):
    def runtime(self, source, method, independent=False, reject=False):
        events = []
        manager = SimpleNamespace()
        runner = SimpleNamespace(
            model=SimpleNamespace(
                model=SimpleNamespace(), paras_memory_manager=manager
            ),
            req_to_token_pool=SimpleNamespace(
                clear=lambda: events.append("clear_requests")
            ),
            token_to_kv_pool_allocator=SimpleNamespace(
                clear=lambda: events.append("clear_pages")
            ),
        )
        scheduler = SimpleNamespace(
            tp_worker=SimpleNamespace(model_runner=runner),
            paras_parallelism_config=source,
            running_batch=None,
            waiting_queue=[],
            tree_cache=SimpleNamespace(reset=lambda: events.append("reset_tree")),
        )

        def transition(target):
            events.append(f"switch_{target.value}")
            if not reject:
                scheduler.paras_parallelism_config = target

        scheduler.paras_configure_tp = lambda: transition(ParaSMode.TP)
        scheduler.paras_configure_ep = lambda: transition(ParaSMode.EP)
        runtime = Runtime(scheduler, {"graph_batch_sizes": [1]}, method)
        runtime.independent = independent
        runtime.discard_graphs = lambda: events.append("discard")
        runtime.capture = lambda: events.append("capture")

        @contextmanager
        def phase(name):
            events.append(f"begin_{name}")
            yield
            events.append(f"end_{name}")

        runtime.phase = phase
        return runtime, events

    def test_independent_discards_before_transfer_and_captures_after(self):
        for method in ("host_reload", "naive_nccl"):
            for source, target in (
                (ParaSMode.EP, ParaSMode.TP),
                (ParaSMode.TP, ParaSMode.EP),
            ):
                runtime, events = self.runtime(source, method, independent=True)
                runtime.switch(target)
                self.assertEqual(
                    events,
                    [
                        "begin_discard_graphs_ms",
                        "discard",
                        "end_discard_graphs_ms",
                        "begin_runtime_switch_ms",
                        f"switch_{target.value}",
                        "end_runtime_switch_ms",
                        "begin_graph_capture_ms",
                        "capture",
                        "end_graph_capture_ms",
                    ],
                )
                self.assertEqual(runtime.mode, target)

    def test_retained_graphs_avoid_recapture(self):
        runtime, events = self.runtime(ParaSMode.EP, "full")
        runtime.switch(ParaSMode.TP, measured=True)
        self.assertEqual(
            events, ["begin_runtime_switch_ms", "switch_tp", "end_runtime_switch_ms"]
        )

    def test_fixed_buffer_recaptures_only_measured_transition(self):
        runtime, events = self.runtime(ParaSMode.EP, "fixed_buffer_recapture")
        runtime.switch(ParaSMode.TP)
        self.assertNotIn("discard", events)
        self.assertNotIn("capture", events)
        events.clear()
        runtime.switch(ParaSMode.EP, measured=True)
        self.assertLess(events.index("discard"), events.index("switch_ep"))
        self.assertLess(events.index("switch_ep"), events.index("capture"))

    def test_rejected_switch_never_captures_wrong_mode(self):
        runtime, events = self.runtime(
            ParaSMode.EP, "naive_nccl", independent=True, reject=True
        )
        with self.assertRaisesRegex(RuntimeError, "rejected"):
            runtime.switch(ParaSMode.TP)
        self.assertIn("discard", events)
        self.assertNotIn("capture", events)
        self.assertEqual(runtime.mode, ParaSMode.EP)

    def test_same_mode_does_nothing_and_invalid_mode_fails(self):
        runtime, events = self.runtime(ParaSMode.EP, "full")
        runtime.switch(ParaSMode.EP, measured=True)
        self.assertEqual(events, [])
        with self.assertRaisesRegex(ValueError, "Unsupported target"):
            runtime.switch("invalid")
        self.assertEqual(events, [])

    def test_empty_scheduler_reset_accepts_absent_running_batch(self):
        runtime, events = self.runtime(ParaSMode.EP, "full")
        runtime.reset_requests()
        self.assertEqual(events, ["clear_requests", "clear_pages", "reset_tree"])

    def test_nonempty_scheduler_rejected_before_clearing_any_pool(self):
        for name, value in (
            ("running_batch", SimpleNamespace(reqs=[object()])),
            ("last_batch", SimpleNamespace(reqs=[object()])),
            ("cur_batch", SimpleNamespace(reqs=[object()])),
            ("waiting_queue", [object()]),
            ("chunked_req", object()),
            ("result_queue", [object()]),
        ):
            runtime, events = self.runtime(ParaSMode.EP, "full")
            setattr(runtime.scheduler, name, value)
            with self.assertRaisesRegex(RuntimeError, "empty scheduler"):
                runtime.reset_requests()
            self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
