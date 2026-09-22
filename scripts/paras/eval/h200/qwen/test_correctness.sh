#!/usr/bin/env bash
# Run static TP, static EP, or the canonical ParaS manual-switch test.
# Usage: bash test_correctness.sh [tp|ep|paras] [extra server arguments...]
# Activate the installed environment first. Logs are retained in RUN_DIR.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
HELPERS="$SCRIPT_DIR/../../paras_cmd"
MODE=${1:-paras}
if (($#)); then shift; fi
case "$MODE" in
    tp) LAUNCH=launch_server_tp_tp.sh; export ENABLE_PARAS=0 ;;
    ep) LAUNCH=launch_server_dp_ep.sh; export ENABLE_PARAS=0 ;;
    paras) LAUNCH=launch_server_dp_ep.sh; export ENABLE_PARAS=1 ;;
    *) echo "Usage: $0 [tp|ep|paras] [server arguments...]" >&2; exit 2 ;;
esac
export MODEL_PATH=${MODEL_PATH:-$HOME/models/Qwen3-235B-A22B-Instruct-2507}
export MODEL_NAME=${MODEL_NAME:-Qwen3-235B-A22B-Instruct-2507}
export HOST=${HOST:-127.0.0.1} PORT=${PORT:-30000}
export NUM_GPUS=${NUM_GPUS:-8} PARAS_AUTO_SWITCH=0
# Compile exercised shapes on demand; exhaustive warmup is unnecessary here.
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=${SGLANG_JIT_DEEPGEMM_PRECOMPILE:-false}
export MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.8}
export MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-256}
export CUDA_GRAPH_MAX_BS=${CUDA_GRAPH_MAX_BS:-32}
export TIMEOUT_TRIES=${TIMEOUT_TRIES:-180} SLEEP_BETWEEN=${SLEEP_BETWEEN:-10}
RUN_DIR=${RUN_DIR:-/tmp/sglang-h200-${MODE}-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$RUN_DIR"
export LOG_FILE="$RUN_DIR/server.log"
# Own and clean up only the server process group started by this invocation.
setsid bash "$SCRIPT_DIR/$LAUNCH" --served-model-name "$MODEL_NAME" "$@" >"$LOG_FILE" 2>&1 &
export SERVER_PID=$!
cleanup() {
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
    for _ in {1..10}; do
        kill -0 -- "-$SERVER_PID" 2>/dev/null || break
        sleep 1
    done
    kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT
printf 'Testing %s; server pid=%s; logs=%s\n' "$MODE" "$SERVER_PID" "$RUN_DIR"
if [ "$MODE" = paras ]; then
    bash "$HELPERS/e2e_test.sh" | tee "$RUN_DIR/test.log"
    grep -q 'type=Qwen3MoeForCausalLMParaS' "$LOG_FILE"
else
    bash "$HELPERS/wait_ready.sh" | tee "$RUN_DIR/test.log"
    bash "$HELPERS/health.sh" | tee -a "$RUN_DIR/test.log"
    bash "$HELPERS/send_prompts.sh" "static-$MODE" | tee -a "$RUN_DIR/test.log"
    bash "$HELPERS/check_log.sh" errors | tee -a "$RUN_DIR/test.log"
fi
printf 'PASS: %s\n' "$MODE" | tee -a "$RUN_DIR/test.log"
