#!/usr/bin/bash
# config.sh — Shared cluster / network / runtime / benchmark config for Sphere eval
# Source this file; do not execute directly.
# Model-specific settings live in config_gptoss.sh / config_glm4air.sh.
#
# Usage: source experiments/sphere/eval/config.sh

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
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
MEM_FRAC=0.80
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

# ── Benchmark — 10 000 requests, 2000 rps, lengths 256-512 uniform ────────────
BENCH_BACKEND="sglang"
BENCH_DATASET="random"
BENCH_NUM_PROMPTS=10000
BENCH_REQUEST_RATE=2000
BENCH_RANDOM_INPUT_LEN=512
BENCH_RANDOM_OUTPUT_LEN=512
BENCH_RANDOM_RANGE_RATIO=0.5
BENCH_TIMEOUT=600

# ── Server startup timeout ────────────────────────────────────────────────────
SERVER_READY_TIMEOUT=1800
