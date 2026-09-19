"""CPU-only configuration, process protocol, and reporting tests."""

import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

from bench_reconfigure import Worker, summarize
from reconfigure.configuration import server_arguments, validate_config


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
                    self.assertEqual(args["cuda_graph_bs"], config["graph_batch_sizes"])
                    self.assertEqual(
                        args["dp_size"], config["world_size"] if mode == "ep" else 1
                    )
                    self.assertFalse(args["paras_auto_switch"])

    def test_rejects_graph_and_layout_overrides(self):
        config = json.loads((BENCH / "configs/gpt_oss_120b_a100.json").read_text())
        config["server_args"]["disable_cuda_graph"] = True
        with self.assertRaises(ValueError):
            validate_config(config)

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
