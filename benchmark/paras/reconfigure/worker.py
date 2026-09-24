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

from reconfigure.configuration import (
    DIRECTIONS,
    HOST_METHODS,
    METHODS,
    METHOD_TRANSPORT,
    server_arguments,
    configure_vocabulary_environment,
    verify_ep_provider,
)


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
            from sglang.srt.utils import (
                get_bool_env_var,
                numa_bind_to_node,
                set_gpu_proc_affinity,
            )
            from reconfigure.runtime import Runtime, install_independent_storage
            from sglang.srt.paras.mode import ParaSMode

            if args_dict.get("paras_vmm_runtime_states", False) and method != "full":
                from reconfigure.runtime_memory import install_recapture_vmm

                install_recapture_vmm()
            if method in (*HOST_METHODS, "naive_nccl"):
                # Independent storage has no production IPC weight/KV arena.
                # Empty-state baseline only: no live cache payload is moved.
                os.environ["PARAS_KV_TRANSFER_METHOD"] = "nccl"
                install_independent_storage()
            args = ServerArgs(**args_dict)
            configure_logger(args, prefix=f" benchmark-rank-{rank}")
            # Follow run_scheduler_process before any model/host allocation.
            if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
                set_gpu_proc_affinity(args.pp_size, args.tp_size, args.nnodes, rank)
            if args.numa_node is not None:
                numa_bind_to_node(args.numa_node[rank])
            scheduler = Scheduler(args, PortArgs(**ports), rank, rank, rank, 0, None)
            runtime = Runtime(scheduler, config, method)
            source, target = (ParaSMode(value) for value in DIRECTIONS[direction])
            reference = runtime.prepare(source, target)
            from reconfigure.diagnostics import kv_reservation, memory_snapshot

            connection.send(
                {
                    "stage": "ready",
                    "runtime_layout": runtime.layout_report(),
                    "host_snapshot_report": runtime.host_snapshot_report(),
                    "rank": rank,
                    "graph_batch_sizes": list(runtime.runner.graph_runner.capture_bs),
                    "graph_batches_by_mode": {
                        mode.value: sizes
                        for mode, sizes in runtime.graph_batches.items()
                    },
                    "kv_transfer_method": os.environ.get("PARAS_KV_TRANSFER_METHOD"),
                    "graph_state": runtime.graph_state_report(),
                    "kv_reservation": kv_reservation(runtime.manager),
                    "memory": memory_snapshot(),
                    "runtime_vmm": runtime.vmm_report(),
                    "independent_weight_storage": (
                        runtime.manager.weight_storage_report(runtime.mode)
                        if runtime.independent
                        else None
                    ),
                    "cpu_affinity": sorted(os.sched_getaffinity(0)),
                    "numa_node": args.numa_node[rank] if args.numa_node else None,
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

    configure_vocabulary_environment("ep", paras=True, environ=os.environ)
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
        (directory / "ready-ranks.json").write_text(
            json.dumps(sorted(ready, key=lambda item: item["rank"]), indent=2)
        )
        emit("ready", ranks=ready)
        command = sys.stdin.readline().strip()
        for connection in connections:
            connection.send(command)
        if command != "run":
            return
        messages = collect(connections, processes, "result", timeout)
        results = sorted((x["result"] for x in messages), key=lambda x: x["rank"])
        vmm = "on" if args.paras_vmm_runtime_states else "off"
        if any(x["vmm"] != vmm for x in results):
            raise RuntimeError("Worker VMM setting differs from supervisor")
        # Wall-clock critical path; do not sum independently maximized phases.
        result = {
            "switch_ms": max(x["switch_ms"] for x in results),
            "through_probe_ms": max(x["through_probe_ms"] for x in results),
            "rank_results": results,
            "vmm": vmm,
            "validation": "passed",
            "weight_transport": METHOD_TRANSPORT[method],
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

    configure_vocabulary_environment(mode, paras=False, environ=os.environ)
    from reconfigure.restart_audit import install_engine_audit
    install_engine_audit(directory / "graph-audit")
    args = server_arguments(config, mode, paras=False)
    if mode == "tp":
        # Match launch_common.sh's static TP sampler safety and environment.
        os.environ["SYNC_TOKEN_IDS_ACROSS_TP"] = "1"
        for key in (
            "SGLANG_DEEPEP_BF16_DISPATCH",
            "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK",
            "NVSHMEM_QP_DEPTH",
        ):
            os.environ.pop(key, None)
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
        prompts = [prompt] * config["world_size"]
        output = engine.generate(prompts, sampling)
        first_probe_done = time.perf_counter()
        again = engine.generate(prompts, sampling)
        texts = [item["text"] for item in output]
        if texts != [item["text"] for item in again]:
            raise RuntimeError(
                "Repeated target-mode decode probe produced different output"
            )
        emit(
            "ready",
            engine_initialized_at=initialized,
            first_probe_done_at=first_probe_done,
            worker_start_at=started,
            validation="repeat_decode_passed",
            probe_text=texts,
            probe_global_requests=config["world_size"],
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
        provider = verify_ep_provider(config)
        (directory / "ep_provider.json").write_text(json.dumps(provider, indent=2))
        if args.engine_mode:
            engine_worker(config, args.engine_mode, directory)
        else:
            supervise(config, args.method, args.direction, directory, args.timeout)
    except BaseException:
        emit("error", error=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
