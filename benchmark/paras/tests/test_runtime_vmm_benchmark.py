"""CPU checks of the single-GPU VMM experiment's configuration/provenance."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "bench_runtime_vmm.py"
SPEC = importlib.util.spec_from_file_location("bench_runtime_vmm", SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_default_shapes_match_gptoss_runtime_regions(tmp_path):
    args = benchmark.parse_args(["--output", str(tmp_path / "run")])
    shapes = benchmark.buffer_shapes(args)
    assert shapes["ep"] == {"kv_indices": (256 * 131072,), "logits": (256, 201088)}
    assert shapes["tp"] == {"kv_indices": (2048 * 131072,), "logits": (2048, 201088)}


@pytest.mark.parametrize(
    "options",
    [
        ["--resident-gib", "nan"],
        ["--iterations", "0"],
        ["--rounds", "-1"],
        ["--warmup", "-1"],
        ["--context-length", "512"],
        ["--interval-ms", "nan"],
    ],
)
def test_invalid_experiments_fail_before_gpu_import(tmp_path, options):
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--output", str(tmp_path / "run"), *options])


def test_dry_run_needs_no_torch_and_snapshots_untracked_benchmark(tmp_path):
    output = tmp_path / "plan"
    # -S excludes site-packages: even importing torch would fail.
    subprocess.run(
        [sys.executable, "-S", str(SCRIPT), "--dry-run", "--output", str(output)],
        check=True,
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["variants"] == list(benchmark.VARIANTS)
    relative = SCRIPT.relative_to(benchmark.ROOT)
    assert (output / "source" / relative).read_bytes() == SCRIPT.read_bytes()
    assert "gpu_inventory" not in manifest


def test_statistics_preserve_high_latency_samples():
    result = benchmark.summarize([1.0] * 19 + [101.0])
    assert result["n"] == 20 and result["mean_ms"] == 6
    assert result["median_ms"] == 1 and result["max_ms"] == 101
