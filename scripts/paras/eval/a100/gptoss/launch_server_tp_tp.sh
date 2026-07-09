#!/bin/bash
# launch_server — gpt-oss-120b-bf16, TP/TP (TP attention + TP experts), A100.
# Bench against this with `python -m sglang.bench_serving --backend sglang --host $HOST --port $PORT --dataset-name sharegpt ...`
#
# Common overrides (env vars):
#   MODEL_PATH HOST PORT NUM_GPUS CUDA_VISIBLE_DEVICES
#   MEM_FRACTION_STATIC MAX_RUNNING_REQUESTS
#
# Toggles (see ../../launch_common.sh for full semantics):
#   ENABLE_CUDA_GRAPH=0  Disable cuda graphs (default 1).
#   CUDA_GRAPH_MAX_BS=N  Override cuda-graph max bs. Default = MAX_RUNNING_REQUESTS
#                        (TP global batch size).
#   HYBRID_SWA=0|1       0 adds --disable-hybrid-swa-memory; 1 (default) keeps
#                        hybrid full + SWA memory pools. Used by run_smoke.sh to
#                        walk SWA on/off × overlap on/off matrix.
#   DISABLE_OVERLAP=0|1  1 adds --disable-overlap-schedule (default 0).
#   DISABLE_RADIX_CACHE=0|1
#                        1 (default) adds --disable-radix-cache. Matches paras's
#                        required radix-off (avoids SWARadixCache's SWA-accounting
#                        drift when SWA is enabled).

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "$SCRIPT_DIR/../../launch_common.sh"

MODEL_PATH=${MODEL_PATH:-/data/shaoyuw/models/gpt-oss-120b-BF16-unsloth}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.8}
HYBRID_SWA=${HYBRID_SWA:-1}

paras_launch_setup_tp_tp

python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --trust-remote-code \
    --host "$HOST" --port "$PORT" \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --attention-backend triton \
    --moe-runner-backend triton \
    --tp-size "$NUM_GPUS" \
    --max-running-requests "$MAX_RUNNING_REQUESTS" \
    --max-prefill-tokens "$MAX_PREFILL_TOKENS" \
    --chunked-prefill-size -1 \
    "${HYBRID_SWA_FLAGS[@]}" \
    "${OVERLAP_FLAGS[@]}" \
    "${RADIX_FLAGS[@]}" \
    "${CUDA_GRAPH_FLAGS[@]}" \
    "$@"
