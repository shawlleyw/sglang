#!/bin/bash
# PP4×TP2 worker — WITH recorder
# Usage: ./launch_worker_pptp_record.sh <node_rank> <gating_profile> <log_file> <record_dir> [mem_frac]
set -euo pipefail

NODE_RANK=$1
GATING_PROFILE=$2
LOG_FILE=$3
RECORD_DIR=$4
MEM_FRAC=${5:-0.80}

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
    --tp-size 2 \
    --pp-size 4 \
    --nnodes 4 \
    --node-rank "$NODE_RANK" \
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
