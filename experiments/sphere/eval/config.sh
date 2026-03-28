#!/usr/bin/bash
# config.sh — Shared cluster / network / runtime / benchmark config for Sphere eval
# Source this file; do not execute directly.
# Model-specific settings live in config_gptoss.sh / config_glm4air.sh.
#
# Usage: source experiments/sphere/eval/config.sh

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="$REPO_DIR/experiments/sphere"
GATING_DIR="$REPO_DIR/gating_profiles"
MINICONDA="$HOME/miniconda3"
CONDA_ENV="sglang-fp"

# ── System identity ───────────────────────────────────────────────────────────
SYSTEM_NAME="sglang"

# ── Cluster — 8-node × 2-GPU L40S (Sphere) ───────────────────────────────────
N_NODE=8
N_GPU_PER_NODE=2
WORLD_SIZE=16

# ── Default node names (override via HEAD / WORKERS env vars) ─────────────────
DEFAULT_WORKERS="sgpu2 sgpu3 sgpu4 sgpu6 sgpu7 sgpu8 sgpu9"

# ── Common runtime ────────────────────────────────────────────────────────────
MEM_FRAC=0.77
SERVER_PORT=30000
DIST_INIT_PORT=25000
DIST_TIMEOUT=1800

# ── Network — InfiniBand (Sphere) ────────────────────────────────────────────
HOST_IFNAME="ens1f1np1"
NCCL_IB_HCA="mlx5_1"
NCCL_IB_GID_INDEX=3

# ── Server profiles ──────────────────────────────────────────────────────────
# Each profile defines parallelism strategy and optional server flags.
# _build_server_cmd in evallib/server.sh dispatches on SERVER_PROFILE.
#
#   ep16          — tp=16, dp=16, ep=16, DP-attention, mooncake-nccl a2a
#   ep16_limited  — same as ep16 + --max-running-requests cap
#   pp8tp2        — pp=8, tp=2, no EP/DP-attention (pure pipeline+tensor parallel)

EP16_LIMITED_MAX_RUNNING_REQS=256

# ── Benchmark — common ────────────────────────────────────────────────────────
BENCH_BACKEND="sglang"
BENCH_DATASET=${BENCH_DATASET:-"sharegpt"}
BENCH_NUM_PROMPTS=${BENCH_NUM_PROMPTS:-10000}
BENCH_REQUEST_RATE=${BENCH_REQUEST_RATE:-2000}
BENCH_TIMEOUT=1200

# ── Benchmark — random dataset ───────────────────────────────────────────────
BENCH_RANDOM_INPUT_LEN=512
BENCH_RANDOM_OUTPUT_LEN=512
BENCH_RANDOM_RANGE_RATIO=0.5

# ── Benchmark — sharegpt dataset ─────────────────────────────────────────────
BENCH_SHAREGPT_CONTEXT_LEN=${BENCH_SHAREGPT_CONTEXT_LEN:-2048}
# BENCH_SHAREGPT_OUTPUT_LEN=       # unset → use natural completion length

# ── Benchmark — gsm8k dataset ────────────────────────────────────────────────
BENCH_GSM8K_CONTEXT_LEN=${BENCH_GSM8K_CONTEXT_LEN:-2048}
# BENCH_GSM8K_OUTPUT_LEN=          # unset → use natural answer length

# ── Server startup timeout ────────────────────────────────────────────────────
SERVER_READY_TIMEOUT=1800
