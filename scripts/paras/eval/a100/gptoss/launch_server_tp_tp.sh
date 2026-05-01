#!/bin/bash
# launch_server — gpt-oss-120b-bf16, TP/TP (TP attention + TP experts), A100.
# Bench against this with `python -m sglang.bench_serving --backend sglang --host $HOST --port $PORT --dataset-name sharegpt ...`
#
# Common overrides (env vars):
#   MODEL_PATH HOST PORT NUM_GPUS CUDA_VISIBLE_DEVICES
#   MEM_FRACTION_STATIC MAX_RUNNING_REQUESTS
#
# Toggles:
#   ENABLE_CUDA_GRAPH=0  Pass --disable-cuda-graph (default 1).
#   CUDA_GRAPH_MAX_BS=N  Pass --cuda-graph-max-bs N (only honored when ENABLE_CUDA_GRAPH=1).

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "$SCRIPT_DIR/../../lib.sh"

MODEL_PATH=${MODEL_PATH:-/data/shaoyuw/models/gpt-oss-120b-BF16-unsloth}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-30000}
NUM_GPUS=${NUM_GPUS:-8}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.7}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-256}
ENABLE_CUDA_GRAPH=${ENABLE_CUDA_GRAPH:-1}

paras_default_cvd

unset SGLANG_DEEPEP_BF16_DISPATCH SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK NVSHMEM_QP_DEPTH

paras_init_cuda_graph

python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --trust-remote-code \
    --host "$HOST" --port "$PORT" \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --attention-backend triton \
    --moe-runner-backend triton \
    --disable-hybrid-swa-memory \
    --tp-size "$NUM_GPUS" \
    --max-running-requests "$MAX_RUNNING_REQUESTS" \
    "${CUDA_GRAPH_FLAGS[@]}"
