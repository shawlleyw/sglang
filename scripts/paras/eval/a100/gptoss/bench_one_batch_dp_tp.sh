#!/bin/bash
# bench_one_batch — gpt-oss-120b-bf16, DP/TP (DP attention + TP experts), A100.
# Note: --batch-size is per-DP-rank. Equivalent global batch = batch_size * dp_size (= NUM_GPUS).
# DP-attention reduces per-rank attention/KV work; experts stay TP-sharded with AllReduce.
#
# Common overrides (env vars):
#   MODEL_PATH NUM_GPUS CUDA_VISIBLE_DEVICES
#   BATCH_SIZE CUDA_GRAPH_BS INPUT_LEN OUTPUT_LEN MEM_FRACTION_STATIC
#   DATASET_NAME DATASET_PATH RESULT_FILE RUN_NAME
#
# Profile toggles (default disabled):
#   ENABLE_NSYS=1          Wrap with `nsys profile --cuda-graph-trace=node -t cuda`. Output prefix: NSYS_OUTPUT.
#   ENABLE_TORCH_PROFILE=1 Pass --profile and --disable-cuda-graph. Output dir: SGLANG_TORCH_PROFILER_DIR.

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "$SCRIPT_DIR/../../lib.sh"

MODEL_PATH=${MODEL_PATH:-/data/shaoyuw/models/gpt-oss-120b-BF16-unsloth}
NUM_GPUS=${NUM_GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-"1 8 64 256 1 8 64 256"}
CUDA_GRAPH_BS=${CUDA_GRAPH_BS:-"1 8 64 256"}
INPUT_LEN=${INPUT_LEN:-10}
OUTPUT_LEN=${OUTPUT_LEN:-10}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.7}
DATASET_NAME=${DATASET_NAME:-sharegpt}
DATASET_PATH=${DATASET_PATH:-/data/huggingface/hub/datasets--anon8231489123--ShareGPT_Vicuna_unfiltered/snapshots/192ab2185289094fc556ec8ce5ce1e8e587154ca/ShareGPT_V3_unfiltered_cleaned_split.json}
RESULT_FILE=${RESULT_FILE:-/tmp/gptoss_dp_tp.jsonl}
RUN_NAME=${RUN_NAME:-dp_tp}

paras_default_cvd

unset SGLANG_DEEPEP_BF16_DISPATCH SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK NVSHMEM_QP_DEPTH

paras_init_profile

"${LAUNCHER[@]}" python -m sglang.bench_one_batch \
    --model-path "$MODEL_PATH" \
    --trust-remote-code \
    --disable-overlap-schedule \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --attention-backend triton \
    --moe-runner-backend triton \
    --disable-hybrid-swa-memory \
    --tp-size "$NUM_GPUS" --dp-size "$NUM_GPUS" \
    --enable-dp-attention --enable-dp-lm-head \
    --batch-size $BATCH_SIZE \
    --cuda-graph-bs $CUDA_GRAPH_BS \
    --input-len "$INPUT_LEN" \
    --output-len "$OUTPUT_LEN" \
    --dataset-name "$DATASET_NAME" \
    --dataset-path "$DATASET_PATH" \
    --result-filename "$RESULT_FILE" \
    --run-name "$RUN_NAME" \
    "${PROFILE_FLAGS[@]}" \
    "${LOAD_FORMAT_FLAGS[@]}"
