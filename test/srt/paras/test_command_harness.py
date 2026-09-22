"""CPU-only regression checks for false passes in the manual-switch harness."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


HELPERS = Path(__file__).resolve().parents[3] / "scripts/paras/eval/paras_cmd"


class CommandHarnessTests(unittest.TestCase):
    def shell(self, command, *args, **environment):
        return subprocess.run(
            ["bash", "-c", 'source "$1/lib.sh"; shift; ' + command, "test", str(HELPERS), *map(str, args)],
            env={**os.environ, **environment},
            capture_output=True,
            text=True,
        )

    def test_response_validation(self):
        cases = [
            ("not json", False),
            (json.dumps({"error": "out of memory"}), False),
            (json.dumps({"choices": [{"message": {"content": "short"}}]}), False),
            (json.dumps({"choices": [{"message": {"content": "loop " * 8}}]}), False),
            (json.dumps({"choices": [{"message": {"content": "A valid answer about GPU memory."}}]}), True),
        ]
        with tempfile.TemporaryDirectory() as directory:
            for response, succeeds in cases:
                with self.subTest(response=response):
                    Path(directory, "burst_1.json").write_text(response)
                    result = self.shell('paras_cmd_burst_verify "$1" test 1', directory)
                    self.assertEqual(result.returncode == 0, succeeds, result.stdout + result.stderr)
            # One good response must not hide the absence of another request.
            result = self.shell('paras_cmd_burst_verify "$1" test 2', directory)
            self.assertNotEqual(result.returncode, 0)

    def test_switch_body_and_latency(self):
        for body, elapsed, succeeds in [
            ("ParaS TP parallelism configured.", "100", True),
            ("ParaS EP parallelism configured.", "100", False),
            ("Internal error", "100", False),
            ("ParaS TP parallelism configured.", "3000", False),
        ]:
            with self.subTest(body=body, elapsed=elapsed):
                result = self.shell('paras_cmd_verify_switch tp "$1" "$2"', body, elapsed, CONFIGURE_MAX_MS="2500")
                self.assertEqual(result.returncode == 0, succeeds, result.stderr)

    def test_wait_checks_every_request(self):
        result = self.shell('(exit 22) & first=$!; (sleep 0.05; exit 0) & last=$!; paras_cmd_burst_wait "$first" "$last"')
        self.assertNotEqual(result.returncode, 0)

    def test_switch_requires_fresh_scheduler_success(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory, "server.log")
            old = "Time taken to configure TP: 20 ms\n"
            for recent, succeeds in [
                ("", False),
                ("ParaS EP->TP switch rejected by precheck\n", False),
                ("Time taken to configure EP: 20 ms\n", False),
                ("Time taken to configure TP: 30 ms\n", True),
            ]:
                with self.subTest(recent=recent):
                    log.write_text(old + recent)
                    result = self.shell(
                        'paras_cmd_verify_switch tp "ParaS TP parallelism configured." 100 "$1"',
                        len(old.encode()), LOG_FILE=str(log), CONFIGURE_MAX_MS="2500",
                    )
                    self.assertEqual(result.returncode == 0, succeeds, result.stderr)

    def test_health_requires_expected_model_type(self):
        with tempfile.TemporaryDirectory() as directory:
            curl = Path(directory, "curl")
            curl.write_text('#!/bin/sh\nprintf 200\n')
            curl.chmod(0o755)
            log = Path(directory, "server.log")
            for model, paras, succeeds in [
                ("", "1", False),
                ("Qwen3MoeForCausalLM", "1", False),
                ("Qwen3MoeForCausalLM", "0", True),
                ("Qwen3MoeForCausalLMParaS", "1", True),
            ]:
                with self.subTest(model=model, paras=paras):
                    log.write_text(f"Load weight end. type={model}, dtype=torch.bfloat16\n")
                    result = self.shell('bash "$1/health.sh"', HELPERS, LOG_FILE=str(log), ENABLE_PARAS=paras, PATH=directory + os.pathsep + os.environ["PATH"])
                    self.assertEqual(result.returncode == 0, succeeds, result.stdout)

    def test_timing_and_missing_log(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory, "server.log")
            result = self.shell('bash "$1/check_log.sh" errors', HELPERS, LOG_FILE=str(log))
            self.assertNotEqual(result.returncode, 0)
            for content, succeeds in [
                ("transfer_weights: 5 ms\n", False),
                ("Time taken to configure TP: 20.5 ms\n", True),
                ("Time taken to configure TP: 3000 ms\n", False),
            ]:
                log.write_text(content)
                result = self.shell('bash "$1/check_log.sh" timing', HELPERS, LOG_FILE=str(log), CONFIGURE_MAX_MS="2500")
                self.assertEqual(result.returncode == 0, succeeds, result.stdout + result.stderr)

    def test_ready_waits_for_model_health(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory, "server.log")
            log.write_text("Application startup complete.\n")
            curl = Path(directory, "curl")
            curl.write_text('#!/bin/sh\nif [ -f "$READY_MARKER" ]; then printf 200; else touch "$READY_MARKER"; printf 503; fi\n')
            curl.chmod(0o755)
            result = self.shell(
                'bash "$1/wait_ready.sh"', HELPERS,
                PATH=directory + os.pathsep + os.environ["PATH"],
                LOG_FILE=str(log), READY_MARKER=str(Path(directory, "marker")),
                TIMEOUT_TRIES="2", SLEEP_BETWEEN="0.01",
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("READY after 2x", result.stdout)

    def test_inflight_rejects_already_completed_burst(self):
        with tempfile.TemporaryDirectory() as directory:
            curl = Path(directory, "curl")
            curl.write_text('#!/bin/sh\nprintf \'{"choices":[{"message":{"content":"Already completed answer."}}]}\'\n')
            curl.chmod(0o755)
            prompts = Path(directory, "prompts.txt")
            prompts.write_text("Describe tensor parallelism.\n")
            result = self.shell(
                'bash "$1/inflight_switch.sh" tp', HELPERS,
                PATH=directory + os.pathsep + os.environ["PATH"],
                PROMPTS_FILE=str(prompts), BURST_SIZE="1", INFLIGHT_DELAY="0.1",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("all requests finished before", result.stderr)


if __name__ == "__main__":
    unittest.main()
