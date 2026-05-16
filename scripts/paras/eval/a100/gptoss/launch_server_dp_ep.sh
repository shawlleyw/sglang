#!/bin/bash
# launch_server — gpt-oss-120b-bf16, DP/EP (DP attention + DeepEP experts), A100.
# Bench against this with `python -m sglang.bench_serving --backend sglang --host $HOST --port $PORT --dataset-name sharegpt ...`
#
# Common overrides (env vars):
#   MODEL_PATH HOST PORT NUM_GPUS CUDA_VISIBLE_DEVICES
#   MEM_FRACTION_STATIC MAX_RUNNING_REQUESTS
#   SGLANG_DEEPEP_BF16_DISPATCH SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK NVSHMEM_QP_DEPTH
#
# Toggles:
#   ENABLE_PARAS=1       Bake in ParaS EP↔TP switching (--enable-paras-moe + canonical defaults).
#                        Shifts: MEM_FRACTION_STATIC→0.8, MAX_RUNNING_REQUESTS→1024,
#                        SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK→256,
#                        CUDA_GRAPH_MAX_BS→8 (gpt-oss ParaS canonical = cuda-graph dual capture).
#                        Drops --disable-hybrid-swa-memory (gpt-oss ParaS uses dual full+SWA memory pools).
#                        Adds PARAS_* env vars. Overlap scheduling stays enabled
#                        (drain-on-switch is handled by SchedulerParasMixin).
#   PARAS_AUTO_SWITCH=0  When ENABLE_PARAS=1, disable load-driven EP↔TP autoswitch by passing
#                        --no-paras-auto-switch. Use this when driving manual switches via
#                        /paras_configure_tp and /paras_configure_ep HTTP endpoints. Default 1
#                        keeps the canonical autoswitch behavior.
#   ENABLE_CUDA_GRAPH=0  Pass --disable-cuda-graph (default 1).
#   CUDA_GRAPH_MAX_BS=N  Pass --cuda-graph-max-bs N (only honored when ENABLE_CUDA_GRAPH=1).
#   HYBRID_SWA=auto|0|1  auto (default) follows ENABLE_PARAS (paras=on, static=off).
#                        Force 0 to add --disable-hybrid-swa-memory; force 1 to omit it
#                        regardless of ENABLE_PARAS. Used by run_smoke.sh to walk the
#                        SWA on/off x overlap on/off matrix on static servers.
#   DISABLE_OVERLAP=0|1  0 (default) keeps the scheduler overlap; 1 adds
#                        --disable-overlap-schedule. Works for both paras and static.
#   DISABLE_RADIX_CACHE=0|1
#                        1 (default) adds --disable-radix-cache. Required for paras
#                        (correct UMM init), and also forces SWAChunkCache on static
#                        baselines to avoid SWARadixCache's known SWA-accounting drift
#                        that triggers the check_memory leak detector at idle. Set 0
#                        explicitly to re-enable radix cache.

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "$SCRIPT_DIR/../../lib.sh"

MODEL_PATH=${MODEL_PATH:-/data/shaoyuw/models/gpt-oss-120b-BF16-unsloth}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-30000}
NUM_GPUS=${NUM_GPUS:-8}
ENABLE_PARAS=${ENABLE_PARAS:-0}

if [ "$ENABLE_PARAS" = "1" ]; then
    : "${MEM_FRACTION_STATIC:=0.8}"
    : "${MAX_RUNNING_REQUESTS:=1024}"
    : "${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:=256}"
    : "${SGLANG_ATTN_MAX_BS:=256}"
    : "${PARAS_CONFIGURE_METHOD:=peer_access}"
    : "${PARAS_KV_TRANSFER_METHOD:=peer_access}"
    : "${PARAS_DISABLE_PEER_ACCESS:=0}"
    : "${CUDA_GRAPH_MAX_BS:=8}"
fi

MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.7}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-256}
ENABLE_CUDA_GRAPH=${ENABLE_CUDA_GRAPH:-1}

paras_default_cvd

export SGLANG_DEEPEP_BF16_DISPATCH=${SGLANG_DEEPEP_BF16_DISPATCH:-true}
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-512}
export NVSHMEM_QP_DEPTH=${NVSHMEM_QP_DEPTH:-2048}

PARAS_FLAGS=()
HYBRID_SWA=${HYBRID_SWA:-auto}
DISABLE_OVERLAP=${DISABLE_OVERLAP:-0}
DISABLE_RADIX_CACHE=${DISABLE_RADIX_CACHE:-1}
if [ "$ENABLE_PARAS" = "1" ]; then
    export SGLANG_ATTN_MAX_BS
    export PARAS_CONFIGURE_METHOD
    export PARAS_KV_TRANSFER_METHOD
    export PARAS_DISABLE_PEER_ACCESS
    PARAS_FLAGS=(
        --enable-paras-moe
        --paras-tp-size "$NUM_GPUS"
        --enable-nan-detection
    )
    if [ "${PARAS_AUTO_SWITCH:-1}" = "0" ]; then
        PARAS_FLAGS+=(--no-paras-auto-switch)
    fi
fi

# Resolve HYBRID_SWA=auto from ENABLE_PARAS (paras default = on, static default = off).
if [ "$HYBRID_SWA" = "auto" ]; then
    if [ "$ENABLE_PARAS" = "1" ]; then
        HYBRID_SWA=1
    else
        HYBRID_SWA=0
    fi
fi
HYBRID_SWA_FLAGS=()
if [ "$HYBRID_SWA" = "0" ]; then
    HYBRID_SWA_FLAGS=(--disable-hybrid-swa-memory)
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
    --attention-backend triton \
    --moe-runner-backend triton \
    --tp-size "$NUM_GPUS" --dp-size "$NUM_GPUS" --ep-size "$NUM_GPUS" \
    --enable-dp-attention --enable-dp-lm-head \
    --moe-a2a-backend deepep --deepep-mode auto \
    --max-running-requests "$MAX_RUNNING_REQUESTS" \
    --chunked-prefill-size -1 \
    "${HYBRID_SWA_FLAGS[@]}" \
    "${OVERLAP_FLAGS[@]}" \
    "${RADIX_FLAGS[@]}" \
    "${CUDA_GRAPH_FLAGS[@]}" \
    "${PARAS_FLAGS[@]}" \
    "$@"
