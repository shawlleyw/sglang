#!/bin/bash
# launch_server — Qwen3-30B-A3B, DP/EP (DP attention + DeepEP experts), A100.
# Bench against this with `python -m sglang.bench_serving --backend sglang --host $HOST --port $PORT --dataset-name sharegpt ...`
#
# Common overrides (env vars):
#   MODEL_PATH HOST PORT NUM_GPUS CUDA_VISIBLE_DEVICES
#   MEM_FRACTION_STATIC MAX_RUNNING_REQUESTS
#   SGLANG_DEEPEP_BF16_DISPATCH SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK NVSHMEM_QP_DEPTH
#
# Toggles:
#   ENABLE_PARAS=1       Bake in ParaS EP↔TP switching (--enable-paras-moe + canonical defaults).
#                        Shifts: MEM_FRACTION_STATIC→0.6, MAX_RUNNING_REQUESTS→1024,
#                        SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK→256,
#                        ENABLE_CUDA_GRAPH→0 (qwen ParaS canonical = eager).
#                        Adds --chunked-prefill-size -1 --max-prefill-tokens 32000
#                        plus PARAS_* env vars. Overlap scheduling stays enabled
#                        (drain-on-switch is handled by SchedulerParasMixin).
#   ENABLE_CUDA_GRAPH=0  Pass --disable-cuda-graph (default 1; auto-flipped to 0 under ENABLE_PARAS).
#   CUDA_GRAPH_MAX_BS=N  Pass --cuda-graph-max-bs N (only honored when ENABLE_CUDA_GRAPH=1).
#   DISABLE_OVERLAP=0|1  0 (default) keeps the scheduler overlap; 1 adds
#                        --disable-overlap-schedule.
#   DISABLE_RADIX_CACHE=0|1
#                        1 (default) adds --disable-radix-cache. Required for paras
#                        (correct UMM init) and also forces SWAChunkCache on static
#                        baselines to avoid SWARadixCache's SWA-accounting drift.

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "$SCRIPT_DIR/../../lib.sh"

MODEL_PATH=${MODEL_PATH:-/data/shaoyuw/models/Qwen3-30B-A3B}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-30000}
NUM_GPUS=${NUM_GPUS:-8}
ENABLE_PARAS=${ENABLE_PARAS:-0}

# Defaults shift under ENABLE_PARAS=1; set them BEFORE the unconditional defaults
# so user overrides still take precedence over both.
if [ "$ENABLE_PARAS" = "1" ]; then
    : "${MEM_FRACTION_STATIC:=0.6}"
    : "${MAX_RUNNING_REQUESTS:=1024}"
    : "${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:=256}"
    : "${SGLANG_ATTN_MAX_BS:=256}"
    : "${PARAS_CONFIGURE_METHOD:=peer_access}"
    : "${PARAS_KV_TRANSFER_METHOD:=peer_access}"
    : "${ENABLE_CUDA_GRAPH:=0}"
fi

MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.85}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-256}
ENABLE_CUDA_GRAPH=${ENABLE_CUDA_GRAPH:-1}

paras_default_cvd

export SGLANG_DEEPEP_BF16_DISPATCH=${SGLANG_DEEPEP_BF16_DISPATCH:-true}
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-512}
export NVSHMEM_QP_DEPTH=${NVSHMEM_QP_DEPTH:-2048}

PARAS_FLAGS=()
DISABLE_OVERLAP=${DISABLE_OVERLAP:-0}
DISABLE_RADIX_CACHE=${DISABLE_RADIX_CACHE:-1}
if [ "$ENABLE_PARAS" = "1" ]; then
    export SGLANG_ATTN_MAX_BS
    export PARAS_CONFIGURE_METHOD
    export PARAS_KV_TRANSFER_METHOD
    PARAS_FLAGS=(
        --enable-paras-moe
        --paras-tp-size "$NUM_GPUS"
        --max-prefill-tokens 32000
        --enable-nan-detection
    )
fi
OVERLAP_FLAGS=()
if [ "$DISABLE_OVERLAP" = "1" ]; then
    OVERLAP_FLAGS=(--disable-overlap-schedule)
fi
RADIX_FLAGS=()
if [ "$DISABLE_RADIX_CACHE" = "1" ]; then
    RADIX_FLAGS=(--disable-radix-cache)
fi

paras_init_cuda_graph

python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --trust-remote-code \
    --host "$HOST" --port "$PORT" \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --tp-size "$NUM_GPUS" --dp-size "$NUM_GPUS" --ep-size "$NUM_GPUS" \
    --enable-dp-attention --enable-dp-lm-head \
    --moe-a2a-backend deepep --deepep-mode auto \
    --max-running-requests "$MAX_RUNNING_REQUESTS" \
    --chunked-prefill-size -1 \
    "${OVERLAP_FLAGS[@]}" \
    "${RADIX_FLAGS[@]}" \
    "${CUDA_GRAPH_FLAGS[@]}" \
    "${PARAS_FLAGS[@]}" \
    "$@"
