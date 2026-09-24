"""CPU-only configuration, process protocol, and reporting tests."""

import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

from bench_reconfigure import Worker, main, summarize
from reconfigure.configuration import (
    METHODS,
    DIRECTIONS,
    server_arguments,
    trial_plan,
    validate_config,
    vmm_configurations,
)


class ReconfigureDriverTest(unittest.TestCase):
    def test_preset_arguments_are_real_server_fields(self):
        source = BENCH.parents[1] / "python/sglang/srt/server_args.py"
        tree = ast.parse(source.read_text())
        cls = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ServerArgs"
        )
        fields = {
            node.target.id for node in cls.body if isinstance(node, ast.AnnAssign)
        }
        for path in (BENCH / "configs").glob("*.json"):
            config = validate_config(json.loads(path.read_text()))
            for mode in ("ep", "tp"):
                for paras in (True, False):
                    args = server_arguments(config, mode, paras)
                    self.assertFalse(set(args) - fields, set(args) - fields)
                    self.assertNotIn("cuda_graph_bs", args)
                    self.assertNotIn("paras_tp_cuda_graph_bs", args)
                    self.assertEqual(
                        args["cuda_graph_max_bs"], 256 if mode == "ep" else 2048
                    )
                    self.assertEqual(
                        args["paras_tp_cuda_graph_max_bs"], 2048 if paras else None
                    )
                    self.assertEqual(
                        args["dp_size"], config["world_size"] if mode == "ep" else 1
                    )
                    self.assertFalse(args["paras_auto_switch"])

    def test_rejects_graph_and_layout_overrides(self):
        config = json.loads((BENCH / "configs/gpt_oss_120b_a100.json").read_text())
        config["server_args"]["disable_cuda_graph"] = True
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_v1_sparse_lists_are_rejected(self):
        config = json.loads((BENCH / "configs/gpt_oss_120b_a100.json").read_text())
        config["graph_batch_sizes"] = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        with self.assertRaisesRegex(ValueError, "v1 override"):
            validate_config(config)

    def test_vmm_preserved_for_switching_but_disabled_for_native_restart(self):
        config = json.loads((BENCH / "configs/gpt_oss_120b_a100.json").read_text())
        config["server_args"]["paras_vmm_runtime_states"] = True
        resolved = validate_config(config)
        for mode in ("ep", "tp"):
            self.assertTrue(
                server_arguments(resolved, mode)["paras_vmm_runtime_states"]
            )
            self.assertFalse(
                server_arguments(resolved, mode, paras=False)[
                    "paras_vmm_runtime_states"
                ]
            )
        self.assertTrue(config["server_args"]["paras_vmm_runtime_states"])

    def test_vmm_backend_restrictions_apply_before_launch(self):
        config = json.loads((BENCH / "configs/gpt_oss_120b_a100.json").read_text())
        for field in (
            "attention_backend",
            "prefill_attention_backend",
            "decode_attention_backend",
        ):
            for invalid in (("fa3",) if field == "attention_backend" else ("flashinfer", "fa3")):
                with self.subTest(field=field, invalid=invalid):
                    variant = json.loads(json.dumps(config))
                    variant["server_args"][field] = invalid
                    with self.assertRaisesRegex(ValueError, "requires matching Triton/FlashInfer"):
                        vmm_configurations(variant, "on")
        config["server_args"]["paras_vmm_runtime_states"] = "false"
        with self.assertRaisesRegex(ValueError, "boolean"):
            vmm_configurations(config)

    def test_vmm_variants_are_independent_and_cli_overrides_config(self):
        config = json.loads((BENCH / "configs/gpt_oss_120b_a100.json").read_text())
        variants = vmm_configurations(config, "both")
        self.assertEqual(list(variants), ["off", "on"])
        variants["on"]["server_args"]["max_running_requests"] = 512
        self.assertEqual(variants["off"]["server_args"]["max_running_requests"], 2048)
        self.assertNotIn("paras_vmm_runtime_states", config["server_args"])
        config["server_args"]["paras_vmm_runtime_states"] = True
        self.assertEqual(list(vmm_configurations(config)), ["on"])
        self.assertEqual(list(vmm_configurations(config, "off")), ["off"])

    def test_both_variants_measure_restart_once_per_direction_and_repetition(self):
        plan = trial_plan(METHODS, DIRECTIONS, 2, ("off", "on"))
        self.assertEqual(len(plan), 44)
        restart = [x for x in plan if x["method"] == "rebuild"]
        self.assertEqual(len(restart), 4)
        self.assertEqual({x["vmm"] for x in restart}, {"not_applicable"})
        for method in METHODS[1:]:
            cells = [x for x in plan if x["method"] == method]
            self.assertEqual(len(cells), 8)
            self.assertEqual({x["vmm"] for x in cells}, {"off", "on"})

    def test_vmm_dry_runs_record_exact_plan_without_importing_runtime(self):
        import subprocess

        # A fresh interpreter rejects even importing torch; the real CLI must
        # still generate both model plans and their separately resolved configs.
        for preset in ("gpt_oss_120b_a100", "qwen3_235b_h200"):
            with tempfile.TemporaryDirectory() as temp:
                script = (
                    "import runpy,sys\n"
                    "class NoRuntime:\n"
                    " def find_spec(self, fullname, *args):\n"
                    "  if fullname.split('.')[0] in ('torch','sglang'):\n"
                    "   raise AssertionError('dry run imported runtime: '+fullname)\n"
                    "sys.meta_path.insert(0, NoRuntime())\n"
                    f"sys.path.insert(0, {str(BENCH)!r})\n"
                    f"runpy.run_path({str(BENCH / 'bench_reconfigure.py')!r}, run_name='__main__')\n"
                )
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        script,
                        "--config",
                        str(BENCH / f"configs/{preset}.json"),
                        "--vmm",
                        "both",
                        "--repetitions",
                        "1",
                        "--output",
                        temp,
                        "--dry-run",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                manifest = json.loads((Path(temp) / "manifest.json").read_text())
                self.assertEqual(manifest["vmm_settings"], ["off", "on"])
                self.assertEqual(len(manifest["trial_plan"]), 22)
                self.assertNotIn("gpu_inventory", manifest)
                for setting, name in manifest["resolved_configs"].items():
                    variant = json.loads((Path(temp) / name).read_text())
                    self.assertEqual(
                        variant["server_args"]["paras_vmm_runtime_states"],
                        setting == "on",
                    )

    def test_vmm_driver_passes_each_resolved_config_to_fresh_trial(self):
        config = BENCH / "configs/gpt_oss_120b_a100.json"
        seen = []

        def run_trial(path, directory, method, direction, timeout):
            resolved = json.loads(path.read_text())
            args = server_arguments(resolved, paras=method != "rebuild")
            setting = (
                "not_applicable"
                if method == "rebuild"
                else ("on" if args["paras_vmm_runtime_states"] else "off")
            )
            seen.append((method, direction, setting, directory))
            return dict(vmm=setting, switch_ms=1, through_probe_ms=2)

        with tempfile.TemporaryDirectory() as temp, patch(
            "sys.argv",
            [
                "bench_reconfigure.py",
                "--config",
                str(config),
                "--vmm",
                "both",
                "--methods",
                "rebuild",
                "full",
                "--repetitions",
                "1",
                "--output",
                temp,
            ],
        ), patch("bench_reconfigure.trial", side_effect=run_trial), patch(
            "bench_reconfigure.command_output",
            return_value={"stdout": "", "returncode": 0},
        ):
            main()
            rows = [
                json.loads(x)
                for x in (Path(temp) / "trials.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(rows), 6)
            self.assertEqual(len({x[3] for x in seen}), 6)
            self.assertEqual(len([x for x in seen if x[2] == "on"]), 2)
            self.assertEqual(
                len(json.loads((Path(temp) / "summary.json").read_text())), 6
            )

    def test_static_restart_translates_mode_specific_prefill_limits(self):
        config = json.loads((BENCH / "configs/gpt_oss_120b_a100.json").read_text())
        config["server_args"].update(
            max_prefill_tokens=2048, paras_tp_max_prefill_tokens=8192
        )
        for mode, expected in (("ep", 2048), ("tp", 8192)):
            native = server_arguments(config, mode, paras=False)
            self.assertEqual(native["max_prefill_tokens"], expected)
            self.assertNotIn("paras_tp_max_prefill_tokens", native)
        paras = server_arguments(config, "ep", paras=True)
        self.assertEqual(paras["max_prefill_tokens"], 2048)
        self.assertEqual(paras["paras_tp_max_prefill_tokens"], 8192)
        config["server_args"]["paras_tp_max_prefill_tokens"] = 0
        with self.assertRaisesRegex(ValueError, "paras_tp_max_prefill_tokens"):
            validate_config(config)

    def test_production_graph_generator_and_explicit_maxima(self):
        # Execute the actual ServerArgs method without importing CUDA/runtime.
        from types import SimpleNamespace
        from typing import List, Optional

        tree = ast.parse(
            (BENCH.parents[1] / "python/sglang/srt/server_args.py").read_text()
        )
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "ServerArgs"
        )
        method = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "_generate_cuda_graph_batch_sizes"
        )
        namespace = {"Optional": Optional, "List": List}
        exec(
            compile(
                ast.Module(body=[method], type_ignores=[]),
                "production_graph_generator",
                "exec",
            ),
            namespace,
        )
        generate = namespace[method.name]
        config = json.loads((BENCH / "configs/gpt_oss_120b_a100.json").read_text())
        for mode, expected_count, maximum in (("ep", 36, 256), ("tp", 100, 2048)):
            args = server_arguments(config, mode, paras=False)
            state = SimpleNamespace(
                **args, disable_cuda_graph_padding=False, speculative_algorithm=None
            )
            sizes = generate(state)
            self.assertEqual(len(sizes), expected_count)
            self.assertEqual(max(sizes), maximum)
            self.assertIn(12, sizes)
            self.assertIn(248, sizes)
        config["server_args"].update(
            cuda_graph_max_bs=128, paras_tp_cuda_graph_max_bs=1024
        )
        self.assertEqual(server_arguments(config, "ep")["cuda_graph_max_bs"], 128)
        self.assertEqual(
            server_arguments(config, "ep")["paras_tp_cuda_graph_max_bs"], 1024
        )
        self.assertEqual(
            server_arguments(config, "tp", paras=False)["cuda_graph_max_bs"], 1024
        )

    def test_summary_does_not_count_failed_trials(self):
        rows = [
            dict(
                method="full",
                direction="ep_to_tp",
                status="passed",
                switch_ms=x,
                through_probe_ms=x + 3,
            )
            for x in (2, 5, 9)
        ]
        rows.append(
            dict(method="full", direction="ep_to_tp", status="failed", switch_ms=10000)
        )
        summary = summarize(rows)
        self.assertEqual(summary[0]["n"], 3)
        self.assertEqual(summary[0]["switch_median_ms"], 5)

    def test_summary_never_pools_vmm_on_and_off(self):
        rows = [
            dict(
                method="full",
                direction="ep_to_tp",
                status="passed",
                vmm=vmm,
                switch_ms=value,
                through_probe_ms=value + 3,
            )
            for vmm, value in (("off", 2), ("off", 4), ("on", 40), ("on", 60))
        ]
        summary = {x["vmm"]: x for x in summarize(rows)}
        self.assertEqual(summary["off"]["switch_median_ms"], 3)
        self.assertEqual(summary["on"]["switch_median_ms"], 50)

    def test_worker_protocol_and_owned_process_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            script = directory / "fake.py"
            script.write_text(
                'import json,sys\nprint("ordinary startup log",flush=True)\nprint(json.dumps({"event":"ready"}),flush=True)\nassert input()=="run"\nprint(json.dumps({"event":"result","result":{"value":7}}),flush=True)\n'
            )
            worker = Worker([sys.executable, str(script)], directory)
            try:
                worker.wait("ready", 5)
                worker.send("run")
                self.assertEqual(worker.wait("result", 5)["result"]["value"], 7)
                worker.finish(5)
            finally:
                worker.close()
            self.assertIn(
                "ordinary startup log", (directory / "supervisor.log").read_text()
            )

    def test_worker_failure_is_not_a_successful_measurement(self):
        with tempfile.TemporaryDirectory() as temp:
            worker = Worker(
                [
                    sys.executable,
                    "-c",
                    'print(\'{"event":"error","error":"test failure"}\',flush=True)',
                ],
                Path(temp),
            )
            try:
                with self.assertRaisesRegex(RuntimeError, "test failure"):
                    worker.wait("ready", 5)
            finally:
                worker.close()


if __name__ == "__main__":
    unittest.main()
