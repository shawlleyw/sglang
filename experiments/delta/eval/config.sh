#!/usr/bin/bash
# config.sh — Fixed cluster / model / runtime / benchmark config for SGLang eval
# Source this file; do not execute directly.
#
# Usage: source experiments/delta/eval/config.sh

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="$REPO_DIR/experiments/delta"
GATING_DIR="$REPO_DIR/gating_profiles"
MODEL_PATH="/projects/bgro/spark36/models/gpt-oss-120b-bf16"
MINICONDA="$HOME/miniconda3"

# ── System identity ───────────────────────────────────────────────────────────
SYSTEM_NAME="sglang"

# ── Cluster — 4-node × 4-GPU A100-SXM4-40GB (Delta gpuA100x4) ───────────────
N_NODE=4
N_GPU_PER_NODE=4
WORLD_SIZE=16

# ── Model — gpt-oss-120b-bf16 (36 layers, 128 experts, dummy weights) ────────
MODEL_NAME="lmsys/gpt-oss-120b-bf16"
LOAD_FORMAT="dummy"

# ── Common runtime ────────────────────────────────────────────────────────────
MEM_FRAC=0.80
SERVER_PORT=30000
DIST_INIT_PORT=25000
DIST_TIMEOUT=1800

# ── Network — HPE Slingshot (Delta) ───────────────────────────────────────────
HOST_IFNAME="hsn0"

# ── Server profiles ──────────────────────────────────────────────────────────
# Each profile defines parallelism strategy and optional server flags.
#   _build_server_cmd in evallib/server.sh dispatches on SERVER_PROFILE.
#
#   ep16          — tp=16, dp=16, ep=16, DP-attention, mooncake-nccl a2a
#   ep16_limited  — same as ep16 + --max-running-requests cap
#   pp4tp4        — pp=4, tp=4, no EP/DP-attention (pure pipeline+tensor parallel)

EP16_LIMITED_MAX_RUNNING_REQS=256

# ── Conda environment ─────────────────────────────────────────────────────────
CONDA_ENV="${CONDA_ENV:-sglang}"

# ── Benchmark — common ────────────────────────────────────────────────────────
BENCH_BACKEND="${BENCH_BACKEND:-sglang}"
BENCH_DATASET="${BENCH_DATASET:-sharegpt}"
BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-10000}"
BENCH_REQUEST_RATE="${BENCH_REQUEST_RATE:-2000}"
BENCH_TIMEOUT="${BENCH_TIMEOUT:-600}"

# ── Benchmark — random dataset ───────────────────────────────────────────────
BENCH_RANDOM_INPUT_LEN="${BENCH_RANDOM_INPUT_LEN:-512}"
BENCH_RANDOM_OUTPUT_LEN="${BENCH_RANDOM_OUTPUT_LEN:-512}"
BENCH_RANDOM_RANGE_RATIO="${BENCH_RANDOM_RANGE_RATIO:-0.5}"

# ── Benchmark — sharegpt dataset ─────────────────────────────────────────────
BENCH_SHAREGPT_CONTEXT_LEN="${BENCH_SHAREGPT_CONTEXT_LEN:-2048}"
# BENCH_SHAREGPT_OUTPUT_LEN=       # unset → use natural completion length

# ── Benchmark — gsm8k dataset ────────────────────────────────────────────────
BENCH_GSM8K_CONTEXT_LEN="${BENCH_GSM8K_CONTEXT_LEN:-2048}"
# BENCH_GSM8K_OUTPUT_LEN=          # unset → use natural answer length

# ── Server startup timeout ────────────────────────────────────────────────────
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-1800}"
