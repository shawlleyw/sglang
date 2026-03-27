# SGLang Multi-Baseline Evaluation — NCSA Delta

## File layout

```
eval/
  ep16_eval.sh      # main: experiment matrix + orchestration loop
  config.sh         # all fixed variables (sourced by main)
  evallib/
    cluster.sh      # discover_nodes(), kill_all() — SSH+tmux cluster management
    server.sh       # launch_server(), wait_for_server(), kill_server(), is_oom()
    benchmark.sh    # run_benchmark() — sglang.bench_serving wrapper
  README.md
```

`evallib/` scripts only define functions; they are sourced, not executed.
To change any single concern, edit only that one file.

---

## What the main script does

Evaluates **3 server profiles × 4 gate profiles = 12 experiments**.
For each experiment, up to `MAX_RETRIES=3` times:

1. **`kill_server`** — kills any existing sglang processes + tmux sessions
   across all nodes. Required between runs to release GPUs.
2. **`launch_server`** — builds the full `sglang.launch_server` command based
   on the current `SERVER_PROFILE` (ep16, ep16_limited, or pp4tp4) with
   `--enable-fake-prefill` and `--profile-driven-gate-path`. Launches head
   (rank 0) + 3 workers via SSH+tmux. Saves commands to `server_cmd.sh`.
3. **`wait_for_server`** — polls `http://<head>:30000/health` every 10s,
   up to `SERVER_READY_TIMEOUT=1800s` (multi-node init on Delta can be slow).
   - If timeout, calls **`is_oom`** on the server logs. On OOM,
     `MEM_FRAC` is decreased by `MEM_FRAC_STEP=0.02` before the next attempt.
4. **`run_benchmark`** — runs `sglang.bench_serving` with fixed parameters
   matching the AsyncMoE evaluation; saves output to `bench_result.json`.
5. On success, copies the result to `result.json` and moves to the next experiment.

Final cleanup: `kill_server`.

---

## Server profiles

| Profile | Parallelism | Key flags |
|---|---|---|
| `ep16` | tp=16, dp=16, ep=16 | `--enable-dp-attention --enable-dp-lm-head --moe-a2a-backend mooncake-nccl` |
| `ep16_limited` | tp=16, dp=16, ep=16 | Same as ep16 + `--max-running-requests 256` |
| `pp4tp4` | tp=4, pp=4 | No EP/DP-attention (pure pipeline + tensor parallel) |

All profiles share: `--enable-fake-prefill`, `--profile-driven-gate-path`,
`--disable-radix-cache`, `--chunked-prefill-size -1`, `--disable-custom-all-reduce`,
`--moe-runner-backend triton`.

---

## Fixed config (edit `config.sh`)

| Parameter | Default | Env override |
|---|---|---|
| Model | `lmsys/gpt-oss-120b-bf16` (36 layers, 128 experts, dummy weights) | — |
| Cluster | 4 nodes × 4 A100-SXM4-40GB = 16 GPUs | — |
| Network | HPE Slingshot `hsn0`, NCCL+Gloo | — |
| Conda env | `sglang` | `CONDA_ENV` |
| Initial memory fraction | 0.80 | — |
| OOM step | −0.02 per retry | — |
| Benchmark rate | 2000 rps | `BENCH_REQUEST_RATE` |
| Benchmark prompts | 10k | `BENCH_NUM_PROMPTS` |
| Benchmark input/output | 256–512 uniform | `BENCH_RANDOM_INPUT_LEN`, `BENCH_RANDOM_OUTPUT_LEN`, `BENCH_RANDOM_RANGE_RATIO` |
| Server ready timeout | 1800s (30 min) | `SERVER_READY_TIMEOUT` |
| Benchmark timeout | 600s (10 min) | `BENCH_TIMEOUT` |

---

## Experiment matrix

The full matrix is `SERVER_PROFILES × GATE_PROFILES` (3 × 4 = 12 experiments):

**3 server profiles**: `ep16`, `ep16_limited`, `pp4tp4`

**4 workloads** (`{sharegpt, legal-court} × {regular, balanced}`):

| Workload | Gate profile | Description |
|---|---|---|
| `sharegpt_regular` | `gating_gptoss120b_sharegpt_200.parquet` | ShareGPT trace, real expert routing |
| `sharegpt_balanced` | `balanced_gptoss120b_sharegpt_200.parquet` | ShareGPT trace, balanced expert routing |
| `legal_court_regular` | `gating_legal_court_opinions_200.parquet` | Legal court opinions trace, real expert routing |
| `legal_court_balanced` | `balanced_legal_court_opinions_200.parquet` | Legal court opinions trace, balanced expert routing |

Regular profiles are captured from real inference traces.
Balanced profiles are pre-generated and placed in `gating_profiles/balanced_output/`.

Benchmark parameters are fixed in `config.sh` to match the AsyncMoE evaluation:
2000 rps, 10k requests, input/output 256–512 uniform.

---

## How to run

```bash
# 1. Ensure you have a SLURM allocation (4 nodes × 4 A100)
squeue -u $USER

# 2. Run from login node (uses SSH+tmux to reach compute nodes)
cd ~/sglang
bash experiments/delta/eval/ep16_eval.sh /path/to/my_results \
    |& tee experiments/eval.log

# Or override node discovery:
HEAD=gpua002 WORKERS="gpua007 gpua047 gpua076" \
    bash experiments/delta/eval/ep16_eval.sh /path/to/my_results

# Override benchmark parameters without editing config.sh:
BENCH_REQUEST_RATE=500 BENCH_NUM_PROMPTS=1000 \
    bash experiments/delta/eval/ep16_eval.sh /path/to/my_results

# Use a different conda environment:
CONDA_ENV=my_env bash experiments/delta/eval/ep16_eval.sh /path/to/my_results
```

---

## Output layout

Run directories are named `<system>_<server_profile>-<dataset_label>` under `RESULTS_DIR`.

```
<RESULTS_DIR>/
  sglang_ep16-sharegpt_regular/
    server_cmd.sh                        # exact server launch commands (replayable)
    logs/
      server_head.log                    # head node stdout/stderr
      server_w1.log                      # worker 1 log
      server_w2.log                      # worker 2 log
      server_w3.log                      # worker 3 log
    bench_cmd.sh                         # exact benchmark command (replayable)
    bench_result.json                    # raw benchmark output
    bench_result.log                     # benchmark stdout
    result.json                          # copy of the successful result
  sglang_ep16-sharegpt_balanced/         ...
  sglang_ep16_limited-sharegpt_regular/  ...
  sglang_pp4tp4-sharegpt_regular/        ...
  ...  (12 directories total)
```

Failed attempt artifacts are archived to `attempt<N>/` subdirectories before
each retry, preserving logs for post-mortem debugging.

---

## Server command construction

The eval harness builds the full `sglang.launch_server` command directly in
`evallib/server.sh`, dispatching on `SERVER_PROFILE` via a `case` statement.
This keeps the eval self-contained and allows each experiment to use a different
parallelism strategy and gate profile without modifying shared launch scripts.

Compatible with DisagMoE gate profile format (Parquet).
