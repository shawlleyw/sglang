"""Fresh-process worker supervisor for bench_reconfigure.py.

Protocol: emit JSON `ready`, read `run` or `stop` from stdin, emit `result`.
Only this entry point imports SGLang's serving runtime. --dry-run in the driver
never starts it. Child logs and effective ServerArgs are retained per trial.
"""

import argparse
import dataclasses
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback

BENCH = Path(__file__).resolve().parents[1]
REPO = BENCH.parents[1]
sys.path[:0] = [str(BENCH), str(REPO / "python")]

from reconfigure.configuration import DIRECTIONS, METHODS, server_arguments


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}), flush=True)


def _rank_worker(
    rank, args_dict, ports, config, method, direction, connection, log_dir
):
    # Each worker has its own log; avoid interleaved output or lost tracebacks.
    with open(Path(log_dir) / f"rank-{rank}.log", "a", buffering=1) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        try:
            import torch
            import torch.distributed as dist
            from sglang.srt.managers.scheduler import Scheduler
            from sglang.srt.server_args import ServerArgs, PortArgs
            from sglang.srt.utils import configure_logger
            from reconfigure.runtime import Runtime, install_independent_storage
            from sglang.srt.paras.mode import ParaSMode

            if method in ("host_reload", "naive_nccl"):
                install_independent_storage()
            args = ServerArgs(**args_dict)
            configure_logger(args, prefix=f" benchmark-rank-{rank}")
            scheduler = Scheduler(args, PortArgs(**ports), rank, rank, rank, 0, None)
            runtime = Runtime(scheduler, config, method)
            source, target = (ParaSMode(value) for value in DIRECTIONS[direction])
            reference = runtime.prepare(source, target)
            connection.send(
                {
                    "stage": "ready",
                    "rank": rank,
                    "graph_batch_sizes": list(runtime.runner.graph_runner.capture_bs),
                }
            )
            command = connection.recv()
            if command != "run":
                return
            result = runtime.execute(target, reference)
            result["rank"] = rank
            connection.send({"stage": "result", "result": result})
            # Let the supervisor receive results before distributed teardown.
            dist.barrier()
            dist.destroy_process_group()
        except BaseException:
            error = traceback.format_exc()
            print(error, flush=True)
            connection.send({"stage": "error", "rank": rank, "error": error})
            raise
        finally:
            connection.close()


def collect(connections, processes, stage, timeout):
    deadline = time.monotonic() + timeout
    pending = set(range(len(connections)))
    messages = []
    while pending:
        for index in list(pending):
            if connections[index].poll(0.05):
                message = connections[index].recv()
                if message.get("stage") != stage:
                    raise RuntimeError(f"Worker failed: {message}")
                messages.append(message)
                pending.remove(index)
            elif processes[index].exitcode is not None:
                raise RuntimeError(
                    f"Rank {index} exited before {stage}: {processes[index].exitcode}"
                )
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Timed out waiting for ranks {sorted(pending)} at {stage}"
            )
    return messages


def supervise(config, method, direction, directory, timeout):
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.server_args import ServerArgs, PortArgs

    raw = server_arguments(config, "ep", paras=True)
    args = ServerArgs(**raw)
    args.check_server_args()
    _set_envs_and_config(args)
    (directory / "server_args.json").write_text(
        json.dumps(dataclasses.asdict(args), indent=2, default=str)
    )
    ports = dataclasses.asdict(PortArgs.init_new(args))
    context = mp.get_context("spawn")
    processes, connections = [], []
    try:
        for rank in range(config["world_size"]):
            parent, child = context.Pipe()
            process = context.Process(
                target=_rank_worker,
                args=(
                    rank,
                    raw,
                    ports,
                    config,
                    method,
                    direction,
                    child,
                    str(directory),
                ),
            )
            process.start()
            child.close()
            processes.append(process)
            connections.append(parent)
        ready = collect(connections, processes, "ready", timeout)
        emit("ready", ranks=ready)
        command = sys.stdin.readline().strip()
        for connection in connections:
            connection.send(command)
        if command != "run":
            return
        messages = collect(connections, processes, "result", timeout)
        results = sorted((x["result"] for x in messages), key=lambda x: x["rank"])
        # Wall-clock critical path; do not sum independently maximized phases.
        result = {
            "switch_ms": max(x["switch_ms"] for x in results),
            "through_probe_ms": max(x["through_probe_ms"] for x in results),
            "rank_results": results,
            "validation": "passed",
            "scope": "scheduler_worker_reconfiguration_empty_requests",
        }
        emit("result", result=result)
        for process in processes:
            process.join(timeout=30)
            if process.exitcode not in (0, None):
                raise RuntimeError(f"Rank exited with {process.exitcode}")
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join()
        for connection in connections:
            connection.close()


def engine_worker(config, mode, directory):
    """Actual SGLang Engine initialization/release for the rebuild baseline."""
    from sglang.srt.entrypoints.engine import Engine

    args = server_arguments(config, mode, paras=False)
    started = time.perf_counter()
    engine = Engine(**args)
    initialized = time.perf_counter()
    (directory / "server_args.json").write_text(
        json.dumps(dataclasses.asdict(engine.server_args), indent=2, default=str)
    )
    try:
        prompt = config.get("probe_prompt", "The capital of France is")
        sampling = {
            "temperature": 0,
            "max_new_tokens": config.get("probe_decode_steps", 2) + 1,
            "ignore_eos": True,
        }
        output = engine.generate(prompt, sampling)
        first_probe_done = time.perf_counter()
        again = engine.generate(prompt, sampling)
        if output["text"] != again["text"]:
            raise RuntimeError(
                "Repeated target-mode decode probe produced different output"
            )
        emit(
            "ready",
            engine_initialized_at=initialized,
            first_probe_done_at=first_probe_done,
            worker_start_at=started,
            validation="repeat_decode_passed",
            probe_text=output["text"],
        )
        sys.stdin.readline()
    finally:
        engine.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--direction", choices=DIRECTIONS, required=True)
    parser.add_argument("--engine-mode", choices=("ep", "tp"))
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    directory = Path(args.directory)
    directory.mkdir(parents=True, exist_ok=True)
    os.environ.update(config.get("environment", {}))
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    try:
        if args.engine_mode:
            engine_worker(config, args.engine_mode, directory)
        else:
            supervise(config, args.method, args.direction, directory, args.timeout)
    except BaseException:
        emit("error", error=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
