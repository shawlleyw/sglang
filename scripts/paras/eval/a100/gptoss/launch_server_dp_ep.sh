#!/bin/bash
# launch_server — gpt-oss-120b-bf16, DP/EP (DP attention + DeepEP experts), A100.
# Bench against this with `python -m sglang.bench_serving --backend sglang --host $HOST --port $PORT --dataset-name sharegpt ...`
#
# Common overrides (env vars):
#   MODEL_PATH HOST PORT NUM_GPUS CUDA_VISIBLE_DEVICES
#   MEM_FRACTION_STATIC MAX_RUNNING_REQUESTS
#   SGLANG_DEEPEP_BF16_DISPATCH SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK NVSHMEM_QP_DEPTH
#
# Toggles (see ../../launch_common.sh for full semantics):
#   ENABLE_PARAS=1       Bake in ParaS EP↔TP switching (--enable-paras-moe + canonical
#                        defaults; overlap stays enabled via SchedulerParasMixin drain).
#   PARAS_AUTO_SWITCH=0  Under ENABLE_PARAS=1, disable load-driven autoswitch (use
#                        /paras_configure_{tp,ep} HTTP endpoints for manual switches).
#   ENABLE_CUDA_GRAPH=0  Disable cuda graphs (default 1).
#   CUDA_GRAPH_MAX_BS=N  Override cuda-graph max bs. Default = per-rank attn batch
#                        = MAX_RUNNING_REQUESTS / NUM_GPUS.
#   HYBRID_SWA=auto|0|1  auto (default) follows ENABLE_PARAS (paras=on, static=off).
#                        Used by run_smoke.sh to walk SWA on/off × overlap on/off.
#   DISABLE_OVERLAP=0|1  1 adds --disable-overlap-schedule (default 0).
#   DISABLE_RADIX_CACHE=0|1
#                        1 (default) adds --disable-radix-cache. Required for paras
#                        (UMM init) and forces SWAChunkCache on static (avoids
#                        SWARadixCache's SWA-accounting drift / check_memory leak
#                        at idle when SWA is also enabled).

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "$SCRIPT_DIR/../../launch_common.sh"

MODEL_PATH=${MODEL_PATH:-/data/shaoyuw/models/gpt-oss-120b-BF16-unsloth}
ENABLE_PARAS=${ENABLE_PARAS:-0}

MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.75}
HYBRID_SWA=${HYBRID_SWA:-auto}

paras_launch_setup_dp_ep

python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --trust-remote-code \
    --host "$HOST" --port "$PORT" \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --attention-backend triton \
    --moe-runner-backend triton \
    --tp-size "$NUM_GPUS" --dp-size "$NUM_GPUS" --ep-size "$NUM_GPUS" \
    --enable-dp-attention --enable-dp-lm-head \
    --moe-a2a-backend deepep --deepep-mode auto \
    --max-running-requests "$MAX_RUNNING_REQUESTS" \
    --max-prefill-tokens "$MAX_PREFILL_TOKENS" \
    --chunked-prefill-size -1 \
    "${HYBRID_SWA_FLAGS[@]}" \
    "${OVERLAP_FLAGS[@]}" \
    "${RADIX_FLAGS[@]}" \
    "${CUDA_GRAPH_FLAGS[@]}" \
    "${PARAS_FLAGS[@]}" \
    "$@"
