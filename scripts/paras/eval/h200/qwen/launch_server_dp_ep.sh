#!/bin/bash
# launch_server — Qwen3-235B-A22B-Instruct-2507, DP/EP (DP attention + UCCL EP experts), H200.
# Bench against this with `python -m sglang.bench_serving --backend sglang --host $HOST --port $PORT --dataset-name sharegpt ...`
#
# Common overrides (env vars):
#   MODEL_PATH HOST PORT NUM_GPUS CUDA_VISIBLE_DEVICES
#   MEM_FRACTION_STATIC MAX_RUNNING_REQUESTS
#   SGLANG_DEEPEP_BF16_DISPATCH SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK NVSHMEM_QP_DEPTH
#
# Toggles (see ../../launch_common.sh for full semantics):
#   ENABLE_PARAS=1       Bake in ParaS EP↔TP switching (--enable-paras-moe + canonical
#                        defaults). Overlap stays enabled (drain-on-switch in
#                        SchedulerParasMixin).
#   ENABLE_CUDA_GRAPH=0  Disable cuda graphs (default 1).
#   CUDA_GRAPH_MAX_BS=N  Override cuda-graph max bs. Default = MAX_RUNNING_REQUESTS/NUM_GPUS.
#   DISABLE_OVERLAP=0|1  1 adds --disable-overlap-schedule (default 0).
#   DISABLE_RADIX_CACHE=0|1
#                        1 (default) adds --disable-radix-cache.

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "$SCRIPT_DIR/../../launch_common.sh"

source "$SCRIPT_DIR/backend_common.sh"
paras_h200_check_ep_provider || exit 1

MODEL_PATH=${MODEL_PATH:-$HOME/models/Qwen3-235B-A22B-Instruct-2507}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-flashinfer}
MOE_RUNNER_BACKEND=${MOE_RUNNER_BACKEND:-deep_gemm}
ENABLE_PARAS=${ENABLE_PARAS:-0}

MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.85}
# EP admits prefill tokens per DP rank; TP processes a shared token batch.
MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-2048}
PARAS_TP_MAX_PREFILL_TOKENS=${PARAS_TP_MAX_PREFILL_TOKENS:-8192}
export NVSHMEM_DISABLE_NCCL=${NVSHMEM_DISABLE_NCCL:-1}

paras_launch_setup_dp_ep
if [[ "$ENABLE_PARAS" == "1" ]]; then
    PARAS_FLAGS+=(--paras-tp-max-prefill-tokens "$PARAS_TP_MAX_PREFILL_TOKENS")
    if [[ "${PARAS_VMM_RUNTIME_STATES:-0}" == "1" ]]; then
        PARAS_FLAGS+=(--paras-vmm-runtime-states)
    fi
fi

python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --trust-remote-code \
    --attention-backend "$ATTENTION_BACKEND" \
    --moe-runner-backend "$MOE_RUNNER_BACKEND" \
    --host "$HOST" --port "$PORT" \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --tp-size "$NUM_GPUS" --dp-size "$NUM_GPUS" --ep-size "$NUM_GPUS" \
    --enable-dp-attention --enable-dp-lm-head \
    --moe-a2a-backend deepep --deepep-mode "${DEEPEP_MODE:-auto}" \
    --max-running-requests "$MAX_RUNNING_REQUESTS" \
    --max-prefill-tokens "$MAX_PREFILL_TOKENS" \
    --chunked-prefill-size -1 \
    "${OVERLAP_FLAGS[@]}" \
    "${RADIX_FLAGS[@]}" \
    "${CUDA_GRAPH_FLAGS[@]}" \
    "${PARAS_FLAGS[@]}" \
    "$@"
