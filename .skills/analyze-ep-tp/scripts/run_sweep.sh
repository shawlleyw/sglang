#!/bin/bash
# Driver: 4-way attention/expert parallelism sweep using bench_one_batch_*.sh
# helpers under scripts/paras/eval/a100/<model>/.
#
# Required env:
#   BENCH_DIR    Path to the bench_one_batch_{tp_tp,tp_ep,dp_tp,dp_ep}.sh scripts.
#                Default: $SGLANG_REPO/scripts/paras/eval/a100/gptoss
#   OUTDIR       Output directory for results.jsonl + logs/. Default: $PWD.
#
# Optional env (forwarded to the bench scripts via paras lib.sh):
#   MODEL_PATH NUM_GPUS MEM_FRACTION_STATIC INPUT_LEN OUTPUT_LEN
#   DATASET_NAME DATASET_PATH LOAD_FORMAT CUDA_VISIBLE_DEVICES
#
# Sweep grid (override by exporting):
#   TP_BS_UNIQ   "8 16 32 64 128 256 512 1024 2048"  (TP attention; equiv = bs)
#   DP_BS_UNIQ   "1 2 4 8 16 32 64 128 256"          (DP attention; equiv = bs * dp_size)
#
# Usage:
#   bash run_sweep.sh tp_tp     # only TP/TP
#   bash run_sweep.sh tp_ep     # only TP/EP (AllReduce-EP, no DeepEP)
#   bash run_sweep.sh dp_tp     # only DP/TP
#   bash run_sweep.sh dp_ep     # only DP/EP (DeepEP)
#   bash run_sweep.sh all       # all 4 sequentially (truncates results.jsonl)
#   bash run_sweep.sh missing   # tp_ep + dp_tp only (appends)
#
# Each batch is duplicated in BATCH_SIZE so bench_one_batch records cold+warm
# passes; pair with analyze.py / plot.py (per-metric best-of-N).

set -uo pipefail

SGLANG_REPO=${SGLANG_REPO:-/home/shaoyuw/sglang}
BENCH_DIR=${BENCH_DIR:-$SGLANG_REPO/scripts/paras/eval/a100/gptoss}
OUTDIR=${OUTDIR:-$PWD}
RESULT_FILE="$OUTDIR/results.jsonl"
LOGDIR="$OUTDIR/logs"
NUM_GPUS=${NUM_GPUS:-8}

mkdir -p "$LOGDIR"

if [ ! -d "$BENCH_DIR" ]; then
    echo "BENCH_DIR not found: $BENCH_DIR" >&2
    exit 1
fi
for cfg in tp_tp tp_ep dp_tp dp_ep; do
    if [ ! -f "$BENCH_DIR/bench_one_batch_${cfg}.sh" ]; then
        echo "Missing bench script: $BENCH_DIR/bench_one_batch_${cfg}.sh" >&2
        exit 1
    fi
done

TP_BS_UNIQ=${TP_BS_UNIQ:-"8 16 32 64 128 256 512 1024 2048"}
DP_BS_UNIQ=${DP_BS_UNIQ:-"1 2 4 8 16 32 64 128 256"}
TP_BS_DUP="$TP_BS_UNIQ $TP_BS_UNIQ"
DP_BS_DUP="$DP_BS_UNIQ $DP_BS_UNIQ"

cleanup() {
    echo "[$(date '+%H:%M:%S')] cleanup..."
    pkill -9 -f "sglang.bench_one_batch" 2>/dev/null || true
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
        | tr -d ',' | xargs -r kill -9 2>/dev/null || true
    sleep 8
}

run_one() {
    local cfg=$1 batch_dup=$2 batch_uniq=$3
    echo "============================================================"
    echo "[$(date '+%H:%M:%S')] starting: $cfg"
    echo "============================================================"
    BATCH_SIZE="$batch_dup" \
    CUDA_GRAPH_BS="$batch_uniq" \
    RESULT_FILE="$RESULT_FILE" \
    RUN_NAME=$cfg \
    NUM_GPUS=$NUM_GPUS \
    bash "$BENCH_DIR/bench_one_batch_${cfg}.sh" \
        2>&1 | tee "$LOGDIR/${cfg}.log"
    echo "[$(date '+%H:%M:%S')] $cfg done"
    cleanup
}

run_tp_tp() { run_one tp_tp "$TP_BS_DUP" "$TP_BS_UNIQ"; }
run_tp_ep() { run_one tp_ep "$TP_BS_DUP" "$TP_BS_UNIQ"; }
run_dp_tp() { run_one dp_tp "$DP_BS_DUP" "$DP_BS_UNIQ"; }
run_dp_ep() { run_one dp_ep "$DP_BS_DUP" "$DP_BS_UNIQ"; }

MODE=${1:-all}
case "$MODE" in
    tp_tp) run_tp_tp ;;
    tp_ep) run_tp_ep ;;
    dp_tp) run_dp_tp ;;
    dp_ep) run_dp_ep ;;
    all)
        : > "$RESULT_FILE"
        run_tp_tp
        run_tp_ep
        run_dp_tp
        run_dp_ep
        ;;
    missing) run_tp_ep; run_dp_tp ;;
    *)
        echo "Usage: $0 {tp_tp|tp_ep|dp_tp|dp_ep|missing|all}" >&2
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "[$(date '+%H:%M:%S')] DONE: $MODE"
echo "============================================================"
wc -l "$RESULT_FILE"
