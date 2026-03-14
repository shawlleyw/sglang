# SGPU 4/6/7/8 Cluster Rules for Agents

Rules and constraints for running SGLang on the sgpu4/6/7/8 L40S cluster. **Read this before doing anything.**

---

## 1. Cluster Topology

| Node Rank | Hostname | RoCE IP | GPUs | Role |
|-----------|----------|---------|------|------|
| 0 | sgpu4 | 10.0.0.1 | 2× L40S (46 GB each) | **Head node** — this is where you run commands, API server runs here |
| 1 | sgpu6 | 10.0.0.2 | 2× L40S (46 GB each) | Worker — accessible via `ssh sgpu6` |
| 2 | sgpu7 | 10.0.0.3 | 2× L40S (46 GB each) | Worker — accessible via `ssh sgpu7` |
| 3 | sgpu8 | 10.0.0.4 | 2× L40S (46 GB each) | Worker — accessible via `ssh sgpu8` |

**Total: 4 nodes × 2 GPUs = 8× NVIDIA L40S GPUs**

- OS: Ubuntu 24.04 LTS, kernel 6.8.0
- Inter-node: RoCE v2 via `ens1f1np1` (Mellanox ConnectX, `mlx5_1`)
- Intra-node: PCIe (no NVLink — L40S does not have NVLink)

---

## 2. Filesystem

**Filesystem is NOT shared across nodes.** Each node has its own local disk.

| Path | Scope | Notes |
|------|-------|-------|
| `/home/yizhuoliang/` | Per-node, **not shared** | Code repos, configs, conda base |
| `/home/yizhuoliang/sglang-fake-prefill/` | Must exist on **all 4 nodes** | Main repo |
| `/home/yizhuoliang/miniconda3/` | Must exist on **all 4 nodes** | Conda installation |

### Hard Rules

- **NEVER** assume files written on sgpu4 are visible on sgpu6/7/8.
- **ALWAYS rsync** after modifying code, configs, or gating profiles:
  ```bash
  rsync -avz --exclude '.git' --exclude '__pycache__' \
    /home/yizhuoliang/sglang-fake-prefill/ sgpu6:/home/yizhuoliang/sglang-fake-prefill/
  rsync -avz --exclude '.git' --exclude '__pycache__' \
    /home/yizhuoliang/sglang-fake-prefill/ sgpu7:/home/yizhuoliang/sglang-fake-prefill/
  rsync -avz --exclude '.git' --exclude '__pycache__' \
    /home/yizhuoliang/sglang-fake-prefill/ sgpu8:/home/yizhuoliang/sglang-fake-prefill/
  ```
- The `sglang-fp` conda env is installed locally on each node. If you `pip install` something, you must repeat on all nodes or rsync the env.

---

## 3. SSH Access

Passwordless SSH is configured from sgpu4 to the workers:

```bash
ssh sgpu6 '<command>'
ssh sgpu7 '<command>'
ssh sgpu8 '<command>'
```

You can use the short hostnames `sgpu6`, `sgpu7`, `sgpu8` directly (resolved via `/etc/hosts` or DNS).

---

## 4. Environment Setup

**Copy-paste this block at the start of any shell session or script:**

```bash
eval "$(/home/yizhuoliang/miniconda3/bin/conda shell.bash hook)"
conda activate sglang-fp
```

### What This Gives You

| Component | Detail |
|-----------|--------|
| Python | 3.12.12 |
| SGLang | 0.5.5 (installed from local `fake_prefill_coul` branch) |
| Conda env name | `sglang-fp` |
| Conda env path | `/home/yizhuoliang/miniconda3/envs/sglang-fp/` |

### No Module System

Unlike the CarcAI HPC node, this cluster does **not** use `module load`. CUDA is bundled with PyTorch inside the conda env. Do not look for or try to load environment modules.

### If You Need New Python Packages

```bash
# Install on head node
eval "$(/home/yizhuoliang/miniconda3/bin/conda shell.bash hook)" && conda activate sglang-fp
pip install <package>

# Then install on workers too
ssh sgpu6 'eval "$(/home/yizhuoliang/miniconda3/bin/conda shell.bash hook)" && conda activate sglang-fp && pip install <package>'
ssh sgpu7 'eval "$(/home/yizhuoliang/miniconda3/bin/conda shell.bash hook)" && conda activate sglang-fp && pip install <package>'
ssh sgpu8 'eval "$(/home/yizhuoliang/miniconda3/bin/conda shell.bash hook)" && conda activate sglang-fp && pip install <package>'
```

Do **not** create new conda envs unless specifically asked.

---

## 5. NCCL / Network Configuration

All multi-node SGLang launches require these environment variables:

```bash
export NCCL_SOCKET_IFNAME=ens1f1np1
export NCCL_IB_HCA=mlx5_1
export GLOO_SOCKET_IFNAME=ens1f1np1
export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=WARN
```

| Variable | Why |
|----------|-----|
| `NCCL_SOCKET_IFNAME=ens1f1np1` | Use the RoCE network interface (10.0.0.x subnet) |
| `NCCL_IB_HCA=mlx5_1` | Mellanox HCA device for RDMA |
| `GLOO_SOCKET_IFNAME=ens1f1np1` | Gloo (PyTorch distributed) uses the same interface |
| `NCCL_IB_GID_INDEX=3` | GID index for RoCE v2 |
| `NCCL_DEBUG=WARN` | Suppress verbose NCCL logs; set to `INFO` for debugging |

The launch scripts (`launch_head_pp.sh`, `launch_worker_pp.sh`, etc.) already set all of these. If you launch manually, you **must** export them yourself.

---

## 6. Running SGLang Server

### Repo and Branch

```
Repo: /home/yizhuoliang/sglang-fake-prefill/
Branch: fake_prefill_coul
```

### Available Launch Scripts

| Script | Parallelism | Arguments |
|--------|-------------|-----------|
| `launch_head_pp.sh` | TP2×PP4 | `<gating_profile_path> <log_file>` |
| `launch_worker_pp.sh` | TP2×PP4 | `<node_rank> <gating_profile_path> <log_file>` |
| `launch_head.sh` | TP8 | (hardcoded, no args) |
| `launch_worker.sh` | TP8 | `<node_rank> <log_suffix>` |

### Standard Launch: TP2×PP4 (Recommended)

Run everything from sgpu4. Each node gets its own tmux session:

```bash
tmux new-session -d -s sglang-head \
  "bash /home/yizhuoliang/sglang-fake-prefill/launch_head_pp.sh \
   ./gating_profiles/gating_gptoss120b_200.parquet server_head.log"

tmux new-session -d -s sglang-w1 \
  "ssh sgpu6 'bash /home/yizhuoliang/sglang-fake-prefill/launch_worker_pp.sh 1 \
   ./gating_profiles/gating_gptoss120b_200.parquet server_w1.log'"

tmux new-session -d -s sglang-w2 \
  "ssh sgpu7 'bash /home/yizhuoliang/sglang-fake-prefill/launch_worker_pp.sh 2 \
   ./gating_profiles/gating_gptoss120b_200.parquet server_w2.log'"

tmux new-session -d -s sglang-w3 \
  "ssh sgpu8 'bash /home/yizhuoliang/sglang-fake-prefill/launch_worker_pp.sh 3 \
   ./gating_profiles/gating_gptoss120b_200.parquet server_w3.log'"
```

### Manual Launch (Without Scripts)

On **every** node, set env and NCCL vars (see Section 4 and 5), then:

```bash
python -m sglang.launch_server \
    --model-path lmsys/gpt-oss-120b-bf16 \
    --load-format dummy \
    --tp-size 2 \
    --pp-size 4 \
    --nnodes 4 \
    --node-rank <0|1|2|3> \
    --dist-init-addr 10.0.0.1:25000 \
    --enable-fake-prefill \
    --profile-driven-gate-path ./gating_profiles/<profile>.parquet \
    --disable-radix-cache \
    --chunked-prefill-size -1 \
    --mem-fraction-static 0.85 \
    --trust-remote-code \
    --log-level-http warning \
    --moe-runner-backend triton \
    --log-level warning
```

### Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `--model-path` | `lmsys/gpt-oss-120b-bf16` | GPToss 120B model |
| `--load-format dummy` | — | Random weights, skip download. Always use for benchmarking. |
| `--tp-size 2` | — | 2-way tensor parallelism within each node |
| `--pp-size 4` | — | 4-way pipeline parallelism across nodes |
| `--dist-init-addr` | `10.0.0.1:25000` | Head node RoCE IP. Port 25000 for torch distributed init. |
| `--mem-fraction-static 0.85` | — | **Max safe value.** 0.90+ OOMs during CUDA graph capture on PP2/PP3. |
| `--moe-runner-backend triton` | — | Must be triton on L40S (no DeepEP, no flashinfer MoE) |
| `--enable-fake-prefill` | — | Skip real prefill for decode-only benchmarking |
| `--profile-driven-gate-path` | `./gating_profiles/<file>` | Pre-profiled expert routing |
| `--disable-radix-cache` | — | Required with profile-driven gating |
| `--chunked-prefill-size -1` | — | Required with profile-driven gating |
| `--trust-remote-code` | — | Required for GPToss model |

### Server Port

- **TP2×PP4**: Port `30000` (default)
- **TP8**: Port `30000` (default)

### Server Startup Time

- With `--load-format dummy`: ~3-5 minutes (CUDA graph capture dominates)
- Health check: `curl -v http://localhost:30000/health` — returns HTTP 200 with **empty body** when ready

### Checking Server Status

```bash
# Health check (empty body = healthy)
curl -v http://localhost:30000/health

# Watch head node logs
tmux attach -t sglang-head
# Server is ready when you see "Throughput: X tokens/s" lines
```

---

## 7. Killing the Server

SGLang spawns many processes (launch_server, scheduler, detokenizer, torch compile workers). Kill aggressively on **all nodes**:

```bash
pkill -9 -f "sglang" 2>/dev/null
pkill -9 -f "torch._inductor.compile_worker" 2>/dev/null
ssh sgpu6 'pkill -9 -f "sglang" 2>/dev/null; pkill -9 -f "torch._inductor.compile_worker" 2>/dev/null'
ssh sgpu7 'pkill -9 -f "sglang" 2>/dev/null; pkill -9 -f "torch._inductor.compile_worker" 2>/dev/null'
ssh sgpu8 'pkill -9 -f "sglang" 2>/dev/null; pkill -9 -f "torch._inductor.compile_worker" 2>/dev/null'
tmux kill-session -t sglang-head 2>/dev/null
tmux kill-session -t sglang-w1 2>/dev/null
tmux kill-session -t sglang-w2 2>/dev/null
tmux kill-session -t sglang-w3 2>/dev/null
```

**Always kill and wait 5s before starting a new server** — leftover processes hold GPU memory.

Verify all clean:
```bash
pgrep -af "sglang.launch_server" || echo "head clean"
ssh sgpu6 'pgrep -af "sglang.launch_server" || echo "sgpu6 clean"'
ssh sgpu7 'pgrep -af "sglang.launch_server" || echo "sgpu7 clean"'
ssh sgpu8 'pgrep -af "sglang.launch_server" || echo "sgpu8 clean"'
```

---

## 8. Running Benchmarks

Run from sgpu4 (head node) after the server is healthy:

```bash
eval "$(/home/yizhuoliang/miniconda3/bin/conda shell.bash hook)"
conda activate sglang-fp

python -m sglang.bench_serving \
    --backend sglang \
    --host 127.0.0.1 --port 30000 \
    --model lmsys/gpt-oss-120b-bf16 \
    --dataset-name random \
    --random-input-len 256 --random-output-len 1024 \
    --random-range-ratio 0.5 \
    --num-prompts <N> --request-rate <R> \
    2>&1 | tee logs/<descriptive_name>.log
```

### Standard Benchmark Configurations

| Workload | `--num-prompts` | `--request-rate` |
|----------|----------------|-----------------|
| Low | 1000 | 250 |
| High | 2000 | 500 |

### Log Naming Convention

```
logs/bench_pp_p1_r250.log   # PP4, profile 1 (general), rate=250
logs/bench_pp_p2_r500.log   # PP4, profile 2 (math), rate=500
logs/bench_run1.log          # TP8, run 1
```

---

## 9. Available Gating Profiles

```
gating_profiles/gating_gptoss120b_200.parquet   # General-purpose (ShareGPT-like)
gating_profiles/gating_math_gsm8k_200.parquet   # Math topic (GSM8K)
now there are more! chinese, legal, math
```

Used with `--profile-driven-gate-path`. **Always** pair with `--disable-radix-cache --chunked-prefill-size -1`.

---

## 10. Known Bugs and Fixes

### PP Bug: `triton_backend.py` Hardcoded Layer 0

**File**: `python/sglang/srt/layers/attention/triton_backend.py`, lines 102-104

**Problem**: `get_value_buffer(0)` hardcodes layer 0. On PP stages 2+ where `start_layer > 0`, this causes `IndexError: list index out of range`.

**Fix applied**: Changed to `get_value_buffer(model_runner.token_to_kv_pool.start_layer)`.

**Same bug exists** in `aiter_backend.py:85`, `intel_amx_backend.py:27`, `wave_backend.py:153` — only `triton_backend.py` was fixed (only backend in use on L40S).

**If you reinstall sglang or reset the repo, this fix must be reapplied and rsynced to all nodes.**

---

## 11. Common Pitfalls

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| Editing code but not rsyncing to workers | Workers run stale code, mysterious failures | Always rsync after edits |
| `--mem-fraction-static` > 0.85 with PP4 | OOM during CUDA graph capture on PP2/PP3 | Use 0.85 |
| Forgetting NCCL env vars on manual launch | Hangs at distributed init or uses wrong interface | Use the launch scripts, or export all 5 vars |
| Not killing old server before new one | Port conflict or GPU memory exhaustion | Full kill command (Section 7) + verify clean |
| Using `--moe-runner-backend` other than `triton` | Other backends not installed/working on L40S | Always `triton` |
| Health check expecting response body | SGLang returns HTTP 200 with **empty body** | Check HTTP status code, not body content |
| Starting workers before head node | Workers can't connect to `dist-init-addr` | Start head first, workers within ~30s after |

---

## 12. Existing Documentation

Other docs in `coulson_docs/` that may be relevant:

| File | Description |
|------|-------------|
| `running-gptoss-pp-tp.md` | Step-by-step benchmark runbook for this cluster (TP8 and TP2×PP4) |
| `running-gptoss-ep.md` | Running GPT-OSS with Expert Parallelism on A100 (different cluster) |
| `carcai-rules-for-agents.md` | Agent rules for the CarcAI A100 node (different cluster, different rules) |
| `plot_styling.md` | Matplotlib plot styling conventions |
| `deepep-a100-plan.md` | DeepEP integration plan (A100, historical) |
| `deepep-a100-changes.md` | DeepEP code changes (A100, historical) |

---

## 13. Quick Reference

```bash
# Full env setup (run on sgpu4)
eval "$(/home/yizhuoliang/miniconda3/bin/conda shell.bash hook)" && conda activate sglang-fp

# Check GPU status on all nodes
nvidia-smi; ssh sgpu6 nvidia-smi; ssh sgpu7 nvidia-smi; ssh sgpu8 nvidia-smi

# Check if server is running
curl -v http://localhost:30000/health

# Kill all SGLang on all nodes
pkill -9 -f "sglang" 2>/dev/null; pkill -9 -f "torch._inductor" 2>/dev/null; ssh sgpu6 'pkill -9 -f "sglang"; pkill -9 -f "torch._inductor"' 2>/dev/null; ssh sgpu7 'pkill -9 -f "sglang"; pkill -9 -f "torch._inductor"' 2>/dev/null; ssh sgpu8 'pkill -9 -f "sglang"; pkill -9 -f "torch._inductor"' 2>/dev/null; sleep 5

# Rsync code to all workers
rsync -avz --exclude '.git' --exclude '__pycache__' /home/yizhuoliang/sglang-fake-prefill/ sgpu6:/home/yizhuoliang/sglang-fake-prefill/ && rsync -avz --exclude '.git' --exclude '__pycache__' /home/yizhuoliang/sglang-fake-prefill/ sgpu7:/home/yizhuoliang/sglang-fake-prefill/ && rsync -avz --exclude '.git' --exclude '__pycache__' /home/yizhuoliang/sglang-fake-prefill/ sgpu8:/home/yizhuoliang/sglang-fake-prefill/

# Repo root
cd /home/yizhuoliang/sglang-fake-prefill
```
