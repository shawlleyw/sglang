# Legacy kernel benchmark

This is the previous benchmark from SGLang commit `f2900de32aab6bef96e1e81b6c86e7b4bbb736fb`, before the
overlapping EP/TP cache update. New measurements should use the maintained
[generic benchmark](../../README.md) and its existing entry points.

The legacy resident-cache benchmark keeps separate EP and TP buffers. Its
isolated-layer mode also remains available. This snapshot preserves the old
measurement scripts and their shared Python helpers; it is not a frozen
CUDA/runtime environment. It imports the active checkout's built extensions.
`PROVENANCE.json` records original and archived hashes. Only source-root lookup
was changed in two files to make the relocated scripts runnable.

## Reproduce

Activate the compatible CUDA/PyTorch environment and build the active checkout's
peer-access extension as documented in the maintained benchmark README. From
the SGLang repository root, first check command generation without GPU access:

```bash
python benchmark/paras/legacy/kernel_benchmark/run_kernel_ablation.py \
  --model gpt-oss-120b --dry-run --output /tmp/legacy-gptoss-plan
```

Run a small eight-GPU validation with a new output directory:

```bash
python benchmark/paras/legacy/kernel_benchmark/run_kernel_ablation.py \
  --model qwen3-235b --smoke --output /tmp/legacy-qwen-smoke
```

Omit `--smoke` and choose `--cache-gib 10 20 30` for full-layer measurements,
subject to available memory. The separate Qwen3 TP8 layout requires roughly
three times the requested EP KV volume per GPU, plus staging/runtime overhead.
Use `analyze_kernel_ablation.py --help` for analysis options. Do not combine
legacy separate-layout measurements with current overlapping-layout measurements
in one curve. A100/H100 execution of the current revision remains unverified;
SM80/SM90 build targets alone are not a hardware validation result.
