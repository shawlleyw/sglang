#!/bin/bash
# launch_server — Qwen3-30B-A3B, TP/TP (TP attention + TP experts), A100.
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

MODEL_PATH=${MODEL_PATH:-/data/shaoyuw/models/Qwen3-30B-A3B}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-30000}
NUM_GPUS=${NUM_GPUS:-8}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.85}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-256}
ENABLE_CUDA_GRAPH=${ENABLE_CUDA_GRAPH:-1}

paras_default_cvd

# TP/TP: ensure DeepEP env is unset so we don't accidentally pick up a stale config.
unset SGLANG_DEEPEP_BF16_DISPATCH SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK NVSHMEM_QP_DEPTH

# Force a MIN all-reduce on sampled token ids across TP ranks. Paras forces
# Sampler.force_sync_token_ids=True after every EP->TP swap (see
# paras/scheduler_paras_mixin.py paras_configure_tp) to absorb non-deterministic
# attention / MoE / sampler kernels that would otherwise diverge across ranks
# and deadlock at the next collective. tp-static must run the same sync to be a
# structurally fair comparison and to inherit the same safety net.
export SYNC_TOKEN_IDS_ACROSS_TP=1

paras_init_cuda_graph

python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --trust-remote-code \
    --host "$HOST" --port "$PORT" \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --tp-size "$NUM_GPUS" \
    --max-running-requests "$MAX_RUNNING_REQUESTS" \
    --chunked-prefill-size -1 \
    "${CUDA_GRAPH_FLAGS[@]}"
