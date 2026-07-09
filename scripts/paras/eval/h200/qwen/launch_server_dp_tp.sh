#!/bin/bash
# launch_server — Qwen3-235B-A22B-Instruct-2507, DP/TP (DP attention + TP experts), H200.
# DP attention (--enable-dp-attention) with TP-sharded experts via AllReduce (no DeepEP).
# Bench against this with `python -m sglang.bench_serving --backend sglang --host $HOST --port $PORT --dataset-name sharegpt ...`
#
# Common overrides (env vars):
#   MODEL_PATH HOST PORT NUM_GPUS CUDA_VISIBLE_DEVICES
#   MEM_FRACTION_STATIC MAX_RUNNING_REQUESTS
#
# Toggles (see ../../launch_common.sh for full semantics):
#   ENABLE_CUDA_GRAPH=0  Disable cuda graphs (default 1).
#   CUDA_GRAPH_MAX_BS=N  Override cuda-graph max bs. Default = MAX_RUNNING_REQUESTS/NUM_GPUS
#                        (per-rank attn batch under DP attention).
#   DISABLE_OVERLAP=0|1  1 adds --disable-overlap-schedule (default 0).
#   DISABLE_RADIX_CACHE=0|1
#                        1 (default) adds --disable-radix-cache (paras parity).

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "$SCRIPT_DIR/../../launch_common.sh"

MODEL_PATH=${MODEL_PATH:-/models/Qwen3-235B-A22B-Instruct-2507}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.85}

paras_launch_setup_dp_tp

python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --trust-remote-code \
    --host "$HOST" --port "$PORT" \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --tp-size "$NUM_GPUS" --dp-size "$NUM_GPUS" \
    --enable-dp-attention --enable-dp-lm-head \
    --max-running-requests "$MAX_RUNNING_REQUESTS" \
    --max-prefill-tokens "$MAX_PREFILL_TOKENS" \
    --chunked-prefill-size -1 \
    "${OVERLAP_FLAGS[@]}" \
    "${RADIX_FLAGS[@]}" \
    "${CUDA_GRAPH_FLAGS[@]}" \
    "$@"
