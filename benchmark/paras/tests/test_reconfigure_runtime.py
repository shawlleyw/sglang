"""CPU-only ordering checks for the real runtime adapter using scheduler mocks."""

import os
import sys
import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../python"))
)

from common.weight_bundle import ParaSMode
from reconfigure.runtime import Runtime, active_auxiliary_parameters
from reconfigure.diagnostics import ALLOCATOR_COUNTERS, allocator_delta, kv_reservation


class AccountingTest(unittest.TestCase):
    def test_active_auxiliaries_exclude_saved_modes_and_deduplicate_views(self):
        model = torch.nn.Module()
        model.ep_experts = torch.nn.Linear(4, 4)
        model.tp_experts = torch.nn.Linear(4, 2)
        # Register the saved modes first, just as a real model can.
        model.experts = model.tp_experts
        model.ep_sinks = torch.nn.Parameter(torch.ones(4))
        model.tp_sinks = torch.nn.Parameter(model.ep_sinks[1:3])
        model.sinks = model.tp_sinks
        model.tied_sinks = torch.nn.Parameter(model.sinks.data)
        model.ep_qkv_bias = torch.nn.Parameter(torch.ones(4))
        model.tp_qkv_bias = torch.nn.Parameter(torch.ones(2))
        model.qkv_proj = torch.nn.Module()
        model.qkv_proj.bias = model.tp_qkv_bias
        bulk = {id(model.ep_experts.weight), id(model.tp_experts.weight)}
        actual = list(active_auxiliary_parameters(model, bulk))
        self.assertEqual(
            {id(p) for p in actual},
            {id(model.experts.bias), id(model.sinks), id(model.qkv_proj.bias)},
        )
        self.assertEqual(len(actual), 3)

    def test_kv_report_excludes_aliases_and_does_not_add_mode_extents(self):
        manager = SimpleNamespace(
            ep_max_kv_tokens=64,
            tp_max_kv_tokens=256,
            ep_max_kv_tokens_swa=32,
            tp_max_kv_tokens_swa=128,
            _entries={
                "layer.kv.ep.k": SimpleNamespace(size_bytes=128),
                "layer.kv.tp.k": SimpleNamespace(size_bytes=256),
                "layer.kv.k": SimpleNamespace(size_bytes=128),
            },
            _cache_buffers=[torch.empty(256, dtype=torch.uint8)],
        )
        result = kv_reservation(manager)
        self.assertEqual(result["modes"]["ep"]["logical_kv_bytes"], 128)
        self.assertEqual(result["modes"]["tp"]["logical_kv_bytes"], 256)
        self.assertEqual(result["independent_kv_backing_bytes"], 256)
        del manager._cache_buffers
        self.assertIsNone(kv_reservation(manager)["independent_kv_backing_bytes"])

    def test_allocator_missing_counters_are_not_reported_as_zero(self):
        before = dict.fromkeys(ALLOCATOR_COUNTERS, None)
        after = dict(before)
        before["num_alloc_retries"], after["num_alloc_retries"] = 4, 7
        delta = allocator_delta(before, after)
        self.assertEqual(delta["num_alloc_retries"], 3)
        self.assertIsNone(delta["num_device_alloc"])


class RuntimeOrderingTest(unittest.TestCase):
    def test_warmup_exercises_measured_path_then_restores_source(self):
        runtime, events = self.runtime(ParaSMode.EP, "full")
        reference = object()
        runtime.probe = Mock(return_value=reference)
        runtime.synchronize = Mock()

        def execute(target, expected):
            self.assertIs(expected, reference)
            events.append("measured_path")
            runtime.switch(target, measured=True)

        runtime.execute = execute
        self.assertIs(runtime.prepare(ParaSMode.EP, ParaSMode.TP), reference)
        self.assertEqual(runtime.mode, ParaSMode.EP)
        self.assertEqual(events.count("measured_path"), 1)
        self.assertEqual(events.count("switch_ep"), 2)
        self.assertEqual(events.count("switch_tp"), 2)
        self.assertEqual(runtime.phases, {})

    def test_phase_measurement_does_not_insert_device_synchronization(self):
        runtime, _ = self.runtime(ParaSMode.EP, "full")
        with patch("torch.cuda.synchronize", side_effect=AssertionError("extra sync")):
            with Runtime.phase(runtime, "example_ms"):
                pass
        self.assertGreaterEqual(runtime.phases["example_ms"], 0)

    def test_disposal_drops_graph_references_without_explicit_gc(self):
        import weakref

        class Graph:
            pass

        runtime, _ = self.runtime(ParaSMode.EP, "fixed_buffer_recapture")
        graph = Graph()
        ref = weakref.ref(graph)
        gr = runtime.runner.graph_runner
        gr.graphs = {1: graph}
        gr.output_buffers = {1: graph}
        gr._paras_saved = {ParaSMode.EP: {"graphs": {1: graph}}}
        runtime.runner.attn_backend._paras_graph_states = {
            ParaSMode.EP: {"metadata": graph}
        }
        runtime.runner.attn_backend.decode_cuda_graph_metadata = {1: graph}
        del graph
        pool_reset = Mock()
        module = SimpleNamespace(set_global_graph_memory_pool=pool_reset)
        with patch.dict(
            sys.modules, {"sglang.srt.model_executor.cuda_graph_runner": module}
        ), patch("gc.collect", side_effect=AssertionError("explicit GC")):
            Runtime.discard_graphs(runtime)
        self.assertIsNone(ref())
        pool_reset.assert_called_once_with(None)

    def test_recapture_uses_resolved_mode_specific_lists(self):
        from test.srt.paras.test_runtime_states import backend, load_nodes

        runtime, _ = self.runtime(ParaSMode.EP, "fixed_buffer_recapture")
        gr = runtime.runner.graph_runner
        seen = []
        # Execute the production allocation methods with CPU tensors. This catches
        # API/shape changes that a capture-order-only mock cannot detect.
        cls = load_nodes(
            "model_executor/cuda_graph_runner.py",
            {"init_graph_buffers", "_cache_loc_dtype"},
            "CudaGraphRunner",
            TboCudaGraphRunnerPlugin=object,
        )
        gr.init_graph_buffers = cls.init_graph_buffers.__get__(gr)
        gr._cache_loc_dtype = cls._cache_loc_dtype.__get__(gr)
        gr.device = "cpu"
        gr.seq_len_fill_value = 1
        gr.pp_size = 1
        gr.is_encoder_decoder = False
        gr._paras_runtime_memory = None
        gr.require_gathered_buffer = False
        gr.model_runner = runtime.runner
        runtime.runner.model_config = SimpleNamespace(vocab_size=16)
        runtime.runner.spec_algorithm = SimpleNamespace(is_eagle3=lambda: False)
        runtime.runner.attn_backend = backend()

        def capture():
            self.assertEqual(gr.input_ids.shape, (gr.max_num_token,))
            self.assertEqual(gr.req_pool_indices.shape, (gr.max_bs,))
            self.assertEqual(gr.next_token_logits_buffer.shape, (gr.max_num_token, 16))
            self.assertEqual(
                runtime.runner.attn_backend.cuda_graph_kv_indices.numel(),
                gr.max_num_token * 32,
            )
            seen.append((list(gr.capture_bs), gr.max_bs, gr.max_num_token))
            gr.graphs = dict.fromkeys(gr.capture_bs)

        gr.capture = capture
        modules = {
            "sglang.srt.model_executor.cuda_graph_runner": SimpleNamespace(
                model_capture_mode=nullcontext
            ),
            "sglang.srt.paras.paras_cuda_graph": SimpleNamespace(
                paras_refresh_cuda_graph_settings=lambda gr: None
            ),
        }
        with patch.dict(sys.modules, modules), patch(
            "torch.cuda._lazy_init", side_effect=AssertionError("CPU test used CUDA")
        ):
            Runtime.capture(runtime)
            runtime.scheduler.paras_parallelism_config = ParaSMode.TP
            Runtime.capture(runtime)
            runtime.scheduler.paras_parallelism_config = ParaSMode.EP
            Runtime.capture(runtime)
        self.assertEqual(seen, [([1, 2], 2, 2), ([1, 2, 4, 8], 8, 8), ([1, 2], 2, 2)])

    def test_flashinfer_recapture_recreates_input_and_backend_capacity_both_directions(self):
        runtime, _ = self.runtime(ParaSMode.EP, "fixed_buffer_recapture")
        runtime.graph_batches = {
            ParaSMode.EP: [1, 2, 4, 8, 256],
            ParaSMode.TP: [1, 2, 4, 8, 256, 2048],
        }
        gr = runtime.runner.graph_runner
        backend = runtime.runner.attn_backend
        gr.seq_lens = [0] * 256  # Source EP buffers reproduce the smoke failure.
        events = []
        def init_backend(max_bs, max_tokens):
            backend.capacity = max_tokens
            events.append(("backend", max_bs, max_tokens))
        def init_inputs():
            gr.seq_lens = [0] * gr.max_bs
            gr.input_ids = [0] * gr.max_num_token
            events.append(("inputs", gr.max_bs))
        def capture():
            for bs in reversed(gr.capture_bs):
                # FlashInfer wrappers fix their batch at construction; updater
                # derives runtime batch from sliced seq_lens. These must match.
                self.assertEqual(len(gr.seq_lens[:bs]), bs)
                self.assertEqual(len(gr.input_ids[:bs]), bs)
                self.assertGreaterEqual(backend.capacity, bs)
            events.append(("capture", gr.max_bs))
            gr.graphs = dict.fromkeys(gr.capture_bs)
        backend.init_cuda_graph_state = init_backend
        gr.init_graph_buffers = init_inputs
        gr.capture = capture
        modules = {
            "sglang.srt.model_executor.cuda_graph_runner": SimpleNamespace(model_capture_mode=nullcontext),
            "sglang.srt.paras.paras_cuda_graph": SimpleNamespace(paras_refresh_cuda_graph_settings=lambda gr: None),
        }
        with patch.dict(sys.modules, modules):
            for mode in (ParaSMode.TP, ParaSMode.EP):
                runtime.scheduler.paras_parallelism_config = mode
                Runtime.capture(runtime)
        self.assertEqual(events, [
            ("backend", 2048, 2048), ("inputs", 2048), ("capture", 2048),
            ("backend", 256, 256), ("inputs", 256), ("capture", 256),
        ])

    def test_discard_releases_flashinfer_wrappers_and_graph_inputs_without_gc(self):
        import weakref
        class Owned:
            pass
        runtime, _ = self.runtime(ParaSMode.EP, "fixed_buffer_recapture")
        gr, backend = runtime.runner.graph_runner, runtime.runner.attn_backend
        wrapper, buffer = Owned(), Owned()
        wrapper_ref, buffer_ref = weakref.ref(wrapper), weakref.ref(buffer)
        gr.graphs, gr.output_buffers = {}, {}
        gr._paras_saved = {ParaSMode.EP: {"buffer": buffer, "wrapper": wrapper}}
        gr.seq_lens = buffer
        backend.decode_cuda_graph_metadata = {256: [wrapper]}
        backend.prefill_cuda_graph_metadata = {256: [wrapper]}
        backend.draft_extend_cuda_graph_metadata = {256: [wrapper]}
        backend.forward_metadata = SimpleNamespace(decode_wrappers=[wrapper])
        backend.cuda_graph_kv_indices = [buffer]
        backend.cuda_graph_custom_mask = buffer
        backend.cuda_graph_qk_indptr = [buffer]
        backend.cuda_graph_qo_indptr = [buffer]
        workspace = Owned()
        backend.workspace_buffer = workspace
        del wrapper, buffer
        modules = {
            "sglang.srt.model_executor.cuda_graph_runner": SimpleNamespace(set_global_graph_memory_pool=Mock()),
            "sglang.srt.paras.paras_cuda_graph": SimpleNamespace(_BUFFER_KEYS=("seq_lens",)),
        }
        with patch.dict(sys.modules, modules), patch("gc.collect", side_effect=AssertionError("explicit GC")):
            Runtime.discard_graphs(runtime)
        self.assertIsNone(wrapper_ref())
        self.assertIsNone(buffer_ref())
        self.assertIsNone(backend.forward_metadata)
        self.assertFalse(hasattr(gr, "seq_lens"))
        self.assertFalse(hasattr(backend, "cuda_graph_kv_indices"))
        self.assertIs(backend.workspace_buffer, workspace)

    def test_full_graph_report_requires_both_sets_and_correct_replay_limits(self):
        runtime, _ = self.runtime(ParaSMode.EP, "full")
        gr = runtime.runner.graph_runner
        gr.max_bs, gr.max_num_token = 2, 2
        gr.graphs = dict.fromkeys([1, 2])
        gr._paras_saved = {
            mode: {"graphs": dict.fromkeys(sizes)}
            for mode, sizes in runtime.graph_batches.items()
        }
        report = runtime.graph_state_report()
        self.assertEqual(report["retained_batch_sizes_by_mode"]["tp"], [1, 2, 4, 8])
        # A stale TP replay maximum must not silently admit missing EP graphs.
        gr.max_bs = 8
        with self.assertRaisesRegex(RuntimeError, "replay limits"):
            runtime.graph_state_report()
        gr.max_bs = 2
        del gr._paras_saved[ParaSMode.TP]["graphs"][8]
        with self.assertRaisesRegex(RuntimeError, "retain both"):
            runtime.graph_state_report()

    def test_recapture_report_rejects_saved_graphs_after_switch(self):
        runtime, _ = self.runtime(ParaSMode.EP, "fixed_buffer_recapture")
        gr = runtime.runner.graph_runner
        gr.max_bs, gr.max_num_token = 2, 2
        gr.graphs = dict.fromkeys([1, 2])
        gr._paras_saved = {ParaSMode.EP: {"graphs": dict(gr.graphs)}}
        # With zero switch warmups, fixed-buffer initialization can still own
        # the dual graphs. The measured recapture must discard them all.
        runtime.graph_state_report()
        with self.assertRaisesRegex(RuntimeError, "still owns"):
            runtime.graph_state_report(require_discarded=True)
        gr._paras_saved.clear()
        self.assertFalse(runtime.graph_state_report()["retained_batch_sizes_by_mode"])
        del gr.graphs[2]
        with self.assertRaisesRegex(RuntimeError, "production set"):
            runtime.graph_state_report()

    def runtime(self, source, method, independent=False, reject=False):
        events = []
        manager = SimpleNamespace()
        runner = SimpleNamespace(
            attn_backend=SimpleNamespace(),
            graph_runner=SimpleNamespace(
                capture_bs=[1, 2],
                _paras_tp_capture_bs=[1, 2, 4, 8],
                num_tokens_per_bs=1,
            ),
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

    def test_recapture_activates_vmm_between_runtime_switch_and_capture(self):
        for method in (
            "host_reload",
            "host_model_to",
            "naive_nccl",
            "fixed_buffer_recapture",
        ):
            for source, target in (
                (ParaSMode.EP, ParaSMode.TP),
                (ParaSMode.TP, ParaSMode.EP),
            ):
                runtime, events = self.runtime(
                    source, method, independent=method != "fixed_buffer_recapture"
                )
                runtime.runner.graph_runner._paras_runtime_memory = SimpleNamespace(
                    activate=lambda mode: events.append(f"activate_{mode.value}")
                )
                runtime.switch(target, measured=True)
                self.assertLess(
                    events.index(f"switch_{target.value}"),
                    events.index(f"activate_{target.value}"),
                )
                self.assertLess(
                    events.index(f"activate_{target.value}"),
                    events.index("end_runtime_switch_ms"),
                )
                self.assertLess(
                    events.index("end_runtime_switch_ms"), events.index("capture")
                )

    def test_full_vmm_activation_stays_in_production_graph_state_load(self):
        runtime, _ = self.runtime(ParaSMode.EP, "full")
        memory = SimpleNamespace(activate=Mock())
        runtime.runner.graph_runner._paras_runtime_memory = memory
        runtime.switch(ParaSMode.TP, measured=True)
        memory.activate.assert_not_called()  # Adapter adds no second activation.

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
