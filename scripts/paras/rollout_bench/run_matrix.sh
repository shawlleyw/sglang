#!/bin/bash
# run_matrix.sh — sweep 2 models × 3 datasets × 3 server-modes = 18 runs.
#
# Per (model, dataset, mode): invokes run_one.sh, parses summary.json, and
# appends a row to $MATRIX_ROOT/matrix.csv. Retries once with
# MEM_FRACTION_STATIC=0.8 if server.log contains an OOM signature
# (matched as: "out of memory", "OutOfMemoryError", or "OOM").
#
# Env: MATRIX_ROOT (~/paras-bench-results/<UTC>_matrix),
#      NUM_REQUESTS (8000), GROUP_SIZE (1), SPEC_MODE (1 — use spec_8k
#      JSONL with deterministic output_len), MEM_FRACTION_STATIC (0.85),
#      DRY_RUN (0).
#
# Dataset resolution per model and SPEC_MODE:
#   SPEC_MODE=1: ~/paras-workload/<ds>/spec_8k_<model>.jsonl
#   SPEC_MODE=0: ~/paras-workload/<ds>/sampled_8k.jsonl
# Rows for which the JSONL is absent are skipped (logged, no row written).

set -euo pipefail

MATRIX_ROOT=${MATRIX_ROOT:-$HOME/paras-bench-results/$(date -u +%Y%m%dT%H%M%SZ)_matrix}
NUM_REQUESTS=${NUM_REQUESTS:-8000}
GROUP_SIZE=${GROUP_SIZE:-1}
SPEC_MODE=${SPEC_MODE:-1}
DRY_RUN=${DRY_RUN:-0}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.85}

WORKLOAD_DIR="$HOME/paras-workload"
MODELS=(qwen3-30b gpt-oss-120b)
DATASETS=(dapo acecode eurus2)
SERVER_MODES=(ep-static tp-static paras)

mkdir -p "$MATRIX_ROOT"
SUMMARY_CSV="$MATRIX_ROOT/matrix.csv"
echo "model,dataset,server_mode,spec_mode,num_requests,group_size,exit_code,completed,failed,e2e_time,input_throughput,output_throughput,retried_oom,output_dir" > "$SUMMARY_CSV"

# Invoke run_one.sh with $1 as the override MEM_FRACTION_STATIC; all other
# parameters are taken from the enclosing loop's MODEL_SLUG/MODE/DS state.
run_with_mem_frac() {
    local mem_frac="$1"
    MODEL_SLUG="$MODEL_SLUG" \
    MODE="$MODE" \
    DATASET="$DATASET" \
    OUTPUT_DIR="$OUTPUT_DIR" \
    NUM_REQUESTS="$NUM_REQUESTS" \
    GROUP_SIZE="$GROUP_SIZE" \
    SPEC_MODE="$SPEC_MODE" \
    MEM_FRACTION_STATIC="$mem_frac" \
    DRY_RUN="$DRY_RUN" \
        bash scripts/paras/rollout_bench/run_one.sh
}

for MODEL_SLUG in "${MODELS[@]}"; do
    for DS in "${DATASETS[@]}"; do
        if [[ "$SPEC_MODE" == "1" ]]; then
            DATASET="$WORKLOAD_DIR/$DS/spec_8k_${MODEL_SLUG}.jsonl"
        else
            DATASET="$WORKLOAD_DIR/$DS/sampled_8k.jsonl"
        fi

        if [[ ! -f "$DATASET" ]]; then
            echo "SKIP: $DATASET not found"
            continue
        fi

        for MODE in "${SERVER_MODES[@]}"; do
            OUTPUT_DIR="$MATRIX_ROOT/${MODEL_SLUG}__${DS}__${MODE}"
            mkdir -p "$OUTPUT_DIR"

            echo ""
            echo "================================================================"
            echo "RUN: $MODEL_SLUG / $DS / $MODE"
            echo "OUTPUT_DIR: $OUTPUT_DIR"
            echo "================================================================"

            RETRY_OOM=0
            EXIT_CODE=0
            run_with_mem_frac "$MEM_FRACTION_STATIC" || EXIT_CODE=$?

            # OOM retry: matches the three flavors of OOM message we have seen
            # in practice (CUDA runtime, PyTorch wrapper, sglang error string).
            if [[ $EXIT_CODE -ne 0 && -f "$OUTPUT_DIR/server.log" ]]; then
                if grep -qE "out of memory|OutOfMemoryError|OOM" "$OUTPUT_DIR/server.log"; then
                    echo "Detected OOM. Retrying with MEM_FRACTION_STATIC=0.8"
                    RETRY_OOM=1
                    EXIT_CODE=0
                    run_with_mem_frac "0.8" || EXIT_CODE=$?
                fi
            fi

            COMPLETED="-"
            FAILED="-"
            E2E_TIME="-"
            INPUT_TP="-"
            OUTPUT_TP="-"
            if [[ -f "$OUTPUT_DIR/summary.json" ]]; then
                COMPLETED=$(python -c "import json; print(json.load(open('$OUTPUT_DIR/summary.json'))['completed'])" 2>/dev/null || echo "-")
                FAILED=$(python -c "import json; print(json.load(open('$OUTPUT_DIR/summary.json'))['failed'])" 2>/dev/null || echo "-")
                E2E_TIME=$(python -c "import json; print(round(json.load(open('$OUTPUT_DIR/summary.json'))['e2e_time'], 3))" 2>/dev/null || echo "-")
                INPUT_TP=$(python -c "import json; print(round(json.load(open('$OUTPUT_DIR/summary.json'))['input_throughput'], 2))" 2>/dev/null || echo "-")
                OUTPUT_TP=$(python -c "import json; print(round(json.load(open('$OUTPUT_DIR/summary.json'))['output_throughput'], 2))" 2>/dev/null || echo "-")
            fi

            echo "$MODEL_SLUG,$DS,$MODE,$SPEC_MODE,$NUM_REQUESTS,$GROUP_SIZE,$EXIT_CODE,$COMPLETED,$FAILED,$E2E_TIME,$INPUT_TP,$OUTPUT_TP,$RETRY_OOM,$OUTPUT_DIR" >> "$SUMMARY_CSV"
        done
    done
done

echo ""
echo "================================================================"
echo "Matrix complete. Summary: $SUMMARY_CSV"
echo "================================================================"
cat "$SUMMARY_CSV"
