#!/bin/bash
# PP4×TP2 head (rank 0) — no recorder
# Usage: ./launch_head_pptp.sh <gating_profile> <log_file> [mem_frac]
set -euo pipefail

GATING_PROFILE=$1
LOG_FILE=$2
MEM_FRAC=${3:-0.80}

eval "$(/home/yizhuoliang/miniconda3/bin/conda shell.bash hook)"
conda activate sglang-fp
cd /home/yizhuoliang/sglang-fake-prefill

mkdir -p "$(dirname "$LOG_FILE")"

export NCCL_SOCKET_IFNAME=ens1f1np1
export NCCL_IB_HCA=mlx5_1
export GLOO_SOCKET_IFNAME=ens1f1np1
export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=WARN

python -m sglang.launch_server \
    --model-path lmsys/gpt-oss-120b-bf16 \
    --load-format dummy \
    --tp-size 2 \
    --pp-size 4 \
    --nnodes 4 \
    --node-rank 0 \
    --dist-init-addr 10.0.0.1:25000 \
    --enable-fake-prefill \
    --profile-driven-gate-path "$GATING_PROFILE" \
    --disable-radix-cache \
    --chunked-prefill-size -1 \
    --mem-fraction-static "$MEM_FRAC" \
    --trust-remote-code \
    --log-level-http warning \
    --moe-runner-backend triton \
    --dist-timeout 1800 \
    --log-level warning 2>&1 | tee "$LOG_FILE"
