#!/bin/bash
# bench_one_batch — Qwen3-235B-A22B-Instruct-2507, DP/EP (DP attention + DeepEP experts), H200.
# Note: --batch-size is per-DP-rank. Equivalent global batch = batch_size * dp_size (= NUM_GPUS).
#
# Common overrides (env vars):
#   MODEL_PATH NUM_GPUS CUDA_VISIBLE_DEVICES
#   BATCH_SIZE CUDA_GRAPH_BS INPUT_LEN OUTPUT_LEN MEM_FRACTION_STATIC
#   DATASET_NAME DATASET_PATH RESULT_FILE RUN_NAME
#   SGLANG_DEEPEP_BF16_DISPATCH SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK NVSHMEM_QP_DEPTH
#
# Profile toggles (default disabled):
#   ENABLE_NSYS=1          Wrap with `nsys profile --cuda-graph-trace=node -t cuda`. Output prefix: NSYS_OUTPUT.
#   ENABLE_TORCH_PROFILE=1 Pass --profile and --disable-cuda-graph. Output dir: SGLANG_TORCH_PROFILER_DIR.

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "$SCRIPT_DIR/../../lib.sh"

MODEL_PATH=${MODEL_PATH:-/models/Qwen3-235B-A22B-Instruct-2507}
NUM_GPUS=${NUM_GPUS:-8}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-2048}
MAX_REQ_PER_RANK=$((MAX_RUNNING_REQUESTS / NUM_GPUS))
BATCH_SIZE=${BATCH_SIZE:-"1 8 64 256 1 8 64 256"}
CUDA_GRAPH_BS=${CUDA_GRAPH_BS:-"1 8 64 256"}
INPUT_LEN=${INPUT_LEN:-10}
OUTPUT_LEN=${OUTPUT_LEN:-10}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.85}
DATASET_NAME=${DATASET_NAME:-sharegpt}
DATASET_PATH=${DATASET_PATH:-/data/huggingface/hub/datasets--anon8231489123--ShareGPT_Vicuna_unfiltered/snapshots/192ab2185289094fc556ec8ce5ce1e8e587154ca/ShareGPT_V3_unfiltered_cleaned_split.json}
RESULT_FILE=${RESULT_FILE:-/tmp/qwen235b_dp_ep.jsonl}
RUN_NAME=${RUN_NAME:-dp_ep}

paras_default_cvd

# DeepEP dispatch buffer per rank — match the tuned launch config:
# MAX_RUNNING_REQUESTS / NUM_GPUS (= 256 at 2048/8). 256 is verified sufficient.
export SGLANG_DEEPEP_BF16_DISPATCH=${SGLANG_DEEPEP_BF16_DISPATCH:-true}
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-$MAX_REQ_PER_RANK}
export NVSHMEM_QP_DEPTH=${NVSHMEM_QP_DEPTH:-2048}

paras_init_profile

"${LAUNCHER[@]}" python -m sglang.bench_one_batch \
    --model-path "$MODEL_PATH" \
    --trust-remote-code \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --tp-size "$NUM_GPUS" --dp-size "$NUM_GPUS" --ep-size "$NUM_GPUS" \
    --enable-dp-attention --enable-dp-lm-head \
    --moe-a2a-backend deepep --deepep-mode auto \
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
