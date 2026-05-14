#!/bin/bash
# run_one.sh — drive one (model, server-mode, dataset) ParaS rollout bench run.
#
# Kills any leftover sglang server, launches the mode-specific server via
# scripts/paras/eval/a100/<model>/launch_server_*.sh, waits for readiness,
# invokes `python -m sglang.bench_paras`, snapshots nvidia-smi, tears down.
#
# Required env: MODEL_SLUG (qwen3-30b|gpt-oss-120b), MODE (ep-static|tp-static|
# paras), DATASET (JSONL path), OUTPUT_DIR.
#
# Optional env: NUM_REQUESTS (8000), GROUP_SIZE (1), SPEC_MODE (0),
# HOST (127.0.0.1), PORT (30000), MEM_FRACTION_STATIC (0.85), NUM_GPUS (8),
# DRY_RUN (0).
#
# Outputs in OUTPUT_DIR: summary.json, per_request.jsonl, run_config.json,
# server.log, nvidia_smi.txt; paras mode also writes metrics_timeseries.csv.

set -euo pipefail

: "${MODEL_SLUG:?MODEL_SLUG must be set (qwen3-30b or gpt-oss-120b)}"
: "${MODE:?MODE must be set (ep-static, tp-static, or paras)}"
: "${DATASET:?DATASET must be set (path to JSONL)}"
: "${OUTPUT_DIR:?OUTPUT_DIR must be set}"

NUM_REQUESTS=${NUM_REQUESTS:-2000}
GROUP_SIZE=${GROUP_SIZE:-1}
SPEC_MODE=${SPEC_MODE:-0}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-30000}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.85}
NUM_GPUS=${NUM_GPUS:-8}
ENABLE_CUDA_GRAPH=${ENABLE_CUDA_GRAPH:-1}
CUDA_GRAPH_MAX_BS=${CUDA_GRAPH_MAX_BS:-512}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-512}
# DeepEP per-rank dispatch cap must be >= the running-batch size or
# DeepEP asserts during graph capture under ENABLE_PARAS=1 (it ships a
# canonical 256 default, which collides with our cuda-graph max bs).
# Keep them in lockstep by deriving the cap from MAX_RUNNING_REQUESTS.
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-$MAX_RUNNING_REQUESTS}
DRY_RUN=${DRY_RUN:-0}

case "$MODEL_SLUG" in
    qwen3-30b)
        MODEL_NAME="Qwen3-30B-A3B"
        MODEL_DIR="scripts/paras/eval/a100/qwen"
        ;;
    gpt-oss-120b)
        MODEL_NAME="gpt-oss-120b-BF16-unsloth"
        MODEL_DIR="scripts/paras/eval/a100/gptoss"
        ;;
    *)
        echo "Unknown MODEL_SLUG: $MODEL_SLUG" >&2
        exit 2
        ;;
esac

EXTRA_ARGS=()
case "$MODE" in
    ep-static)
        LAUNCH_SCRIPT="$MODEL_DIR/launch_server_dp_ep.sh"
        ENABLE_PARAS=0
        ;;
    tp-static)
        LAUNCH_SCRIPT="$MODEL_DIR/launch_server_tp_tp.sh"
        ENABLE_PARAS=0
        ;;
    paras)
        LAUNCH_SCRIPT="$MODEL_DIR/launch_server_dp_ep.sh"
        ENABLE_PARAS=1
        EXTRA_ARGS=(
            --paras-auto-switch-policy rollout
            --paras-metrics-file "$OUTPUT_DIR/metrics_timeseries.csv"
        )
        ;;
    *)
        echo "Unknown MODE: $MODE" >&2
        exit 2
        ;;
esac

# COMMON_ARGS are passed via "$@" to every launch script. tp_tp also sets
# --chunked-prefill-size itself; sglang's argparse keeps the last occurrence
# so passing it twice is a safe no-op.
COMMON_ARGS=(
    --chunked-prefill-size -1
    --max-prefill-tokens 16384
)

mkdir -p "$OUTPUT_DIR"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY RUN: would launch server with:"
    echo "  ENABLE_PARAS=$ENABLE_PARAS NUM_GPUS=$NUM_GPUS ENABLE_CUDA_GRAPH=$ENABLE_CUDA_GRAPH CUDA_GRAPH_MAX_BS=$CUDA_GRAPH_MAX_BS MAX_RUNNING_REQUESTS=$MAX_RUNNING_REQUESTS MEM_FRACTION_STATIC=$MEM_FRACTION_STATIC \\"
    echo "  SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=$SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK \\"
    echo "    bash $LAUNCH_SCRIPT \\"
    echo "    ${COMMON_ARGS[*]} \\"
    echo "    ${EXTRA_ARGS[*]}"
    echo ""
    echo "DRY RUN: would run bench:"
    BENCH_ARGS=(
        --model "$MODEL_NAME"
        --dataset-jsonl "$DATASET"
        --mode-label "$MODE"
        --num-requests "$NUM_REQUESTS"
        --group-size "$GROUP_SIZE"
        --output-dir "$OUTPUT_DIR"
        --host "$HOST"
        --port "$PORT"
    )
    if [[ "$SPEC_MODE" == "1" ]]; then
        BENCH_ARGS+=(--spec-mode)
    fi
    echo "  python -m sglang.bench_paras ${BENCH_ARGS[*]}"
    exit 0
fi

bash scripts/paras/eval/paras_cmd/kill.sh || true
sleep 3

SERVER_LOG="$OUTPUT_DIR/server.log"
ENABLE_PARAS=$ENABLE_PARAS \
NUM_GPUS=$NUM_GPUS \
ENABLE_CUDA_GRAPH=$ENABLE_CUDA_GRAPH \
CUDA_GRAPH_MAX_BS=$CUDA_GRAPH_MAX_BS \
MAX_RUNNING_REQUESTS=$MAX_RUNNING_REQUESTS \
MEM_FRACTION_STATIC=$MEM_FRACTION_STATIC \
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=$SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK \
    bash "$LAUNCH_SCRIPT" "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server launched (pid=$SERVER_PID). Log: $SERVER_LOG"

# Worst-case wait_ready budget = TIMEOUT_TRIES * SLEEP_BETWEEN.
# qwen3-30b: 60 * 5s = 5min. gpt-oss-120b: 120 * 10s = 20min (BF16 load).
case "$MODEL_SLUG" in
    qwen3-30b)
        TIMEOUT_TRIES=60
        SLEEP_BETWEEN=5
        ;;
    gpt-oss-120b)
        TIMEOUT_TRIES=120
        SLEEP_BETWEEN=10
        ;;
esac
LOG_FILE="$SERVER_LOG" TIMEOUT_TRIES=$TIMEOUT_TRIES SLEEP_BETWEEN=$SLEEP_BETWEEN \
    bash scripts/paras/eval/paras_cmd/wait_ready.sh

BENCH_ARGS=(
    --model "$MODEL_NAME"
    --dataset-jsonl "$DATASET"
    --mode-label "$MODE"
    --num-requests "$NUM_REQUESTS"
    --group-size "$GROUP_SIZE"
    --output-dir "$OUTPUT_DIR"
    --host "$HOST"
    --port "$PORT"
)
if [[ "$SPEC_MODE" == "1" ]]; then
    BENCH_ARGS+=(--spec-mode)
fi
echo "Running bench: python -m sglang.bench_paras ${BENCH_ARGS[*]}"
BENCH_EXIT=0
python -m sglang.bench_paras "${BENCH_ARGS[@]}" || BENCH_EXIT=$?

nvidia-smi > "$OUTPUT_DIR/nvidia_smi.txt" 2>&1 || true
bash scripts/paras/eval/paras_cmd/kill.sh || true

if [[ $BENCH_EXIT -ne 0 ]]; then
    tail -n 200 "$SERVER_LOG" > "$OUTPUT_DIR/server.tail.log" || true
fi

exit $BENCH_EXIT
