"""Run five real-runtime empty-state reconfiguration methods in fresh processes.

Example (in the serving environment, when GPUs are available):
  python benchmark/paras/bench_reconfigure.py \
    --config benchmark/paras/configs/gpt_oss_120b_a100.json \
    --methods full fixed_buffer_recapture --direction both --repetitions 3 \
    --output results/gptoss-reconfigure

--dry-run validates and writes the manifest without importing torch/SGLang or
querying/using GPUs. The benchmark has no automatic remote execution.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import queue
import signal
import statistics
import subprocess
import sys
import threading
import tarfile
import time

from reconfigure.configuration import DIRECTIONS, METHODS, validate_config

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def command_output(command):
    try:
        result = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, timeout=15
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def manifest(config, args):
    versions = {}
    for package in ("torch", "triton", "sglang", "transformers", "nvidia-nccl-cu12"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    files = sorted(
        {
            *HERE.glob("*.py"),
            *HERE.glob("*.sh"),
            *HERE.glob("*.md"),
            *(
                p
                for folder in ("common", "reconfigure", "configs", "tests")
                for p in (HERE / folder).rglob("*")
                if p.suffix in (".py", ".json", ".md", ".sh")
            ),
        }
    )
    return {
        "schema_version": 1,
        "created_at": time.time(),
        "config": config,
        "methods": args.methods,
        "directions": (
            list(DIRECTIONS) if args.direction == "both" else [args.direction]
        ),
        "repetitions": args.repetitions,
        "python": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "versions": versions,
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_branch": command_output(["git", "branch", "--show-current"]),
        "git_status": command_output(["git", "status", "--short"]),
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "checkpoint_cache_policy": "OS cache unchanged; source initialization may warm checkpoint files",
        "timing_policy": "fresh workers per trial; snapshot and warmup excluded; raw per-rank times retained",
        "dry_run": args.dry_run,
    }


class Worker:
    """Own exactly one supervisor process group; never kill unrelated servers."""

    def __init__(self, command, directory):
        self.events = queue.Queue()
        self.log = open(directory / "supervisor.log", "w", buffering=1)
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
            cwd=ROOT,
        )
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        for line in self.process.stdout:
            self.log.write(line)
            try:
                message = json.loads(line)
                if isinstance(message, dict) and "event" in message:
                    self.events.put(message)
            except json.JSONDecodeError:
                pass
        self.events.put({"event": "eof"})

    def wait(self, event, timeout):
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Worker timed out waiting for {event}; see supervisor.log"
                )
            try:
                message = self.events.get(timeout=remaining)
            except queue.Empty:
                raise TimeoutError(
                    f"Worker timed out waiting for {event}; see supervisor.log"
                ) from None
            if message["event"] == event:
                return message
            if message["event"] in ("error", "eof"):
                raise RuntimeError(f"Worker failed before {event}: {message}")

    def send(self, command):
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def finish(self, timeout=60):
        status = self.process.wait(timeout=timeout)
        if status:
            raise RuntimeError(
                f"Worker exited with status {status}; see supervisor.log"
            )

    def close(self):
        if self.process.poll() is None:
            try:
                self.send("stop")
                self.process.wait(timeout=10)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait()
        self.reader.join(timeout=5)
        if self.reader.is_alive():
            # A descendant still holding stdout is a failed cleanup, not a
            # successful trial. Kill only the process group we created.
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.reader.join(timeout=5)
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        for pipe in (self.process.stdin, self.process.stdout):
            try:
                pipe.close()
            except BrokenPipeError:
                pass
        self.log.close()


def trial(config_path, directory, method, direction, timeout):
    def start(subdir, mode=None):
        subdir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(HERE / "reconfigure/worker.py"),
            "--config",
            str(config_path),
            "--directory",
            str(subdir),
            "--method",
            method,
            "--direction",
            direction,
            "--timeout",
            str(timeout),
        ]
        if mode:
            command.extend(["--engine-mode", mode])
        (subdir / "command.json").write_text(json.dumps(command, indent=2))
        return Worker(command, subdir)

    if method != "rebuild":
        worker = start(directory)
        try:
            worker.wait("ready", timeout)
            worker.send("run")
            result = worker.wait("result", timeout)["result"]
            worker.finish()
            return result
        finally:
            worker.close()

    source, target = DIRECTIONS[direction]
    old = start(directory / "source", source)
    try:
        old.wait("ready", timeout)
        begin = time.perf_counter()
        old.send("stop")
        old.finish()
        stopped = time.perf_counter()
    finally:
        old.close()
    new = start(directory / "target", target)
    try:
        ready = new.wait("ready", timeout)
        result = {
            "switch_ms": (ready["engine_initialized_at"] - begin) * 1000,
            "through_probe_ms": (ready["first_probe_done_at"] - begin) * 1000,
            "phases": {
                "engine_shutdown_ms": (stopped - begin) * 1000,
                "launch_and_initialize_ms": (ready["engine_initialized_at"] - stopped)
                * 1000,
            },
            "validation": ready["validation"],
            "probe_text": ready["probe_text"],
            "scope": "engine_rebuild_empty_requests",
        }
        new.send("stop")
        new.finish()
        return result
    finally:
        new.close()


def summarize(rows):
    result = []
    for method in METHODS:
        for direction in DIRECTIONS:
            matching = [
                r
                for r in rows
                if r["method"] == method
                and r["direction"] == direction
                and r["status"] == "passed"
            ]
            if matching:
                result.append(
                    {
                        "method": method,
                        "direction": direction,
                        "n": len(matching),
                        "switch_median_ms": statistics.median(
                            r["switch_ms"] for r in matching
                        ),
                        "through_probe_median_ms": statistics.median(
                            r["through_probe_ms"] for r in matching
                        ),
                        "switch_min_ms": min(r["switch_ms"] for r in matching),
                        "switch_max_ms": max(r["switch_ms"] for r in matching),
                    }
                )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model-path")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--direction", choices=(*DIRECTIONS, "both"), default="both")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.repetitions <= 0 or args.timeout <= 0:
        parser.error("repetitions and timeout must be positive")
    config = json.loads(args.config.read_text())
    if args.model_path:
        config["model_path"] = args.model_path
    config = validate_config(config)
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "manifest.json").exists():
        parser.error("Output already contains a run; choose a new directory")
    resolved = directory / "resolved_config.json"
    resolved.write_text(json.dumps(config, indent=2) + "\n")
    info = manifest(config, args)
    if not args.dry_run:
        info["gpu_inventory"] = command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv",
            ]
        )
        info["gpu_topology"] = command_output(["nvidia-smi", "topo", "-m"])
    (directory / "manifest.json").write_text(json.dumps(info, indent=2) + "\n")
    with tarfile.open(directory / "benchmark_source.tar.gz", "w:gz") as snapshot:
        for relative in info["source_sha256"]:
            snapshot.add(ROOT / relative, arcname=relative)

    (directory / "tracked_changes.patch").write_text(
        # Runtime changes outside the benchmark affect layouts and measured
        # behavior too. Include staged and unstaged tracked changes against HEAD.
        command_output(["git", "diff", "--binary", "HEAD"])["stdout"]
    )
    if args.dry_run:
        print(
            f"Validated {config['name']}; manifest written to {directory}; no GPU access"
        )
        return
    rows = []
    directions = list(DIRECTIONS) if args.direction == "both" else [args.direction]
    for repetition in range(args.repetitions):
        for method in args.methods:
            for direction in directions:
                label = f"{method}-{direction}-rep{repetition + 1}"
                print(f"Starting {label}", flush=True)
                row = {
                    "method": method,
                    "direction": direction,
                    "repetition": repetition + 1,
                }
                try:
                    row.update(
                        trial(
                            resolved, directory / label, method, direction, args.timeout
                        )
                    )
                    row["status"] = "passed"
                except BaseException as error:
                    row.update(status="failed", error=str(error))
                    with open(directory / "trials.jsonl", "a") as output:
                        output.write(json.dumps(row) + "\n")
                    raise
                rows.append(row)
                with open(directory / "trials.jsonl", "a") as output:
                    output.write(json.dumps(row) + "\n")
                (directory / "summary.json").write_text(
                    json.dumps(summarize(rows), indent=2) + "\n"
                )
                print(
                    f"{label}: switch={row['switch_ms']:.3f}ms through_probe={row['through_probe_ms']:.3f}ms",
                    flush=True,
                )


if __name__ == "__main__":
    main()
