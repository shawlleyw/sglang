"""Compatibility entrypoint for archived memory-evaluation drivers.

Supported native GPT-OSS runs now reserve/reuse scratch in production. New
launches should use sglang.launch_server directly; no hooks are installed here.
"""

from sglang.srt.model_executor.static_workspace import reserve_static_workspaces


def reserve_baseline_workspaces(runner):
    state = reserve_static_workspaces(runner)
    if state is None:
        raise ValueError("This configuration does not use production static workspace")
    # Retain the measurement field name for archived snapshot readers.
    runner.matched_baseline_workspaces = state
    return state


def bind_baseline_attention(runner):
    # The backend consumes runner.static_workspaces in its constructor.
    if (
        getattr(runner.attn_backend, "_static_workspaces", None)
        is not runner.static_workspaces
    ):
        raise RuntimeError("Attention backend did not bind production static workspace")


def install():
    """Retained for old launchers; production initialization needs no hooks."""


if __name__ == "__main__":
    import runpy

    runpy.run_module("sglang.launch_server", run_name="__main__")
