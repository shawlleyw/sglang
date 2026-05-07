#!/bin/bash
# bench_one_batch — Qwen3-235B-A22B-Instruct-2507, TP/EP (TP attention + EP-sharded experts via AllReduce), H200.
# Experts are partitioned across ranks (--ep-size N) with no DeepEP — each rank
# runs its local experts on the full token batch with masking for non-routed
# tokens; results combined via full-hidden AllReduce. Anti-pattern (uniformly
# worse than TP/TP), kept for 4-way analysis completeness.
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

MODEL_PATH=${MODEL_PATH:-/models/Qwen3-235B-A22B-Instruct-2507}
NUM_GPUS=${NUM_GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-"8 64 512 2048 8 64 512 2048"}
CUDA_GRAPH_BS=${CUDA_GRAPH_BS:-"8 64 512 2048"}
INPUT_LEN=${INPUT_LEN:-10}
OUTPUT_LEN=${OUTPUT_LEN:-10}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.85}
DATASET_NAME=${DATASET_NAME:-sharegpt}
DATASET_PATH=${DATASET_PATH:-/data/huggingface/hub/datasets--anon8231489123--ShareGPT_Vicuna_unfiltered/snapshots/192ab2185289094fc556ec8ce5ce1e8e587154ca/ShareGPT_V3_unfiltered_cleaned_split.json}
RESULT_FILE=${RESULT_FILE:-/tmp/qwen235b_tp_ep.jsonl}
RUN_NAME=${RUN_NAME:-tp_ep}

paras_default_cvd

unset SGLANG_DEEPEP_BF16_DISPATCH SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK NVSHMEM_QP_DEPTH

paras_init_profile

"${LAUNCHER[@]}" python -m sglang.bench_one_batch \
    --model-path "$MODEL_PATH" \
    --trust-remote-code \
    --disable-overlap-schedule \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --tp-size "$NUM_GPUS" --ep-size "$NUM_GPUS" \
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
