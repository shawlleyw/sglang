"""Artifact-only Engine child hooks: observe actual captured graphs and replay.

Hooks do not alter model operations. One JSON write at scheduler readiness is
inside Engine initialization; another after its first graph replay is in the
untimed correctness probe. Spawn targets are module-level for picklability.
"""
import json
import os
from pathlib import Path
import resource


def host_rss_peak_bytes():
    # Linux ru_maxrss is KiB and is a lifetime high-water mark, not current RSS.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def captured_state(scheduler, replay_count):
    runner = scheduler.tp_worker.model_runner
    graph = runner.graph_runner
    return {
        "rank": scheduler.tp_rank,
        "capture_batch_sizes": list(graph.capture_bs),
        "captured_graph_keys": sorted(graph.graphs),
        "replay_count_lower_bound": replay_count,
        "attention_backend": runner.server_args.attention_backend,
        "moe_runner_backend": runner.server_args.moe_runner_backend,
        "max_prefill_tokens": scheduler.max_prefill_tokens,
        "embedding_shape": list(runner.model.model.embed_tokens.weight.shape),
        "lm_head_shape": list(runner.model.lm_head.weight.shape),
        "host_rss_peak_bytes": host_rss_peak_bytes(),
    }


def audited_scheduler_process(*args, **kwargs):
    from sglang.srt.managers.scheduler import Scheduler, run_scheduler_process

    original = Scheduler.__init__
    def initialize(scheduler, *init_args, **init_kwargs):
        original(scheduler, *init_args, **init_kwargs)
        directory = Path(os.environ["PARAS_SWITCH_ENGINE_AUDIT_DIR"])
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"rank-{scheduler.tp_rank}.json"
        def save(replays):
            temporary = destination.with_suffix(".tmp")
            temporary.write_text(json.dumps(captured_state(scheduler, replays), indent=2))
            temporary.replace(destination)
        save(0)
        graph = scheduler.tp_worker.model_runner.graph_runner
        replay = graph.replay
        first = True
        def audited_replay(*replay_args, **replay_kwargs):
            nonlocal first
            result = replay(*replay_args, **replay_kwargs)
            if first:
                save(1)
                first = False
            return result
        graph.replay = audited_replay
    Scheduler.__init__ = initialize
    return run_scheduler_process(*args, **kwargs)


def audited_controller_process(*args, **kwargs):
    import sglang.srt.managers.data_parallel_controller as controller
    controller.run_scheduler_process = audited_scheduler_process
    return controller.run_data_parallel_controller_process(*args, **kwargs)


def install_engine_audit(directory):
    import sglang.srt.entrypoints.engine as engine
    os.environ["PARAS_SWITCH_ENGINE_AUDIT_DIR"] = str(Path(directory).resolve())
    engine.run_scheduler_process = audited_scheduler_process
    engine.run_data_parallel_controller_process = audited_controller_process
