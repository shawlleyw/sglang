#!/bin/bash
# DP-attention EP8 head (rank 0) — mooncake-nccl, WITH recorder
# Usage: ./launch_head_ep_record.sh <gating_profile> <log_file> <record_dir> [mem_frac]
set -euo pipefail

GATING_PROFILE=$1
LOG_FILE=$2
RECORD_DIR=$3
MEM_FRAC=${4:-0.80}

eval "$(/home/yizhuoliang/miniconda3/bin/conda shell.bash hook)"
conda activate sglang-fp
cd /home/yizhuoliang/sglang-fake-prefill

mkdir -p "$(dirname "$LOG_FILE")" "$RECORD_DIR"

export NCCL_SOCKET_IFNAME=ens1f1np1
export NCCL_IB_HCA=mlx5_1
export GLOO_SOCKET_IFNAME=ens1f1np1
export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=WARN
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR="$RECORD_DIR"

python -m sglang.launch_server \
    --model-path lmsys/gpt-oss-120b-bf16 \
    --load-format dummy \
    --tp-size 8 \
    --moe-a2a-backend mooncake-nccl \
    --enable-dp-attention \
    --dp-size 8 \
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
    --expert-distribution-recorder-mode stat \
    --dist-timeout 1800 \
    --log-level warning 2>&1 | tee "$LOG_FILE"
