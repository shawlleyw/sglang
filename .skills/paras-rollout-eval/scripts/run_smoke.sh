#!/bin/bash
# run_smoke.sh — server-per-cell driver template.
#
# Use when cells differ on launch-time flags (SWA, overlap, paras threshold,
# paras-on/off). Each cell launches its own server, runs one bench, kills.
#
# Output per cell:
#   $ARTIFACTS_DIR/<model>__<cell_label>/
#     server_launch.cmd.txt
#     server.log, nvidia_smi.txt, metrics_timeseries.csv (whole CSV = this cell)
#     n<N>_cap<CAP>/{summary.json, per_request.jsonl, outputs.jsonl,
#                    run_config.json, metrics_timeseries.csv (cp), bench.log}
#   matrix.csv (one row per cell), PROGRESS.md (event log)

set -uo pipefail

ARTIFACTS_DIR=${ARTIFACTS_DIR:-/home/shaoyuw/sglang/artifacts/$(date -u +%Y%m%dT%H%M%SZ)_smoke}
SAMPLES_DIR=${SAMPLES_DIR:-$ARTIFACTS_DIR/samples}
MATRIX_CSV=$ARTIFACTS_DIR/matrix.csv
PROGRESS_MD=$ARTIFACTS_DIR/PROGRESS.md
SKILL_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

MODEL_SLUG=${MODEL_SLUG:-gpt-oss-120b}
MODEL_NAME=${MODEL_NAME:-gpt-oss-120b-BF16-unsloth}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-30000}
NUM_GPUS=${NUM_GPUS:-8}

EP_MFS=${EP_MFS:-0.75};   EP_MAX_RUNNING=${EP_MAX_RUNNING:-2048};   EP_CG_MAX_BS=${EP_CG_MAX_BS:-256}
TP_MFS=${TP_MFS:-0.80};   TP_MAX_RUNNING=${TP_MAX_RUNNING:-2048};   TP_CG_MAX_BS=${TP_CG_MAX_BS:-256}
MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-8192}
TIMEOUT_TRIES=${TIMEOUT_TRIES:-120}; SLEEP_BETWEEN=${SLEEP_BETWEEN:-10}

mkdir -p "$ARTIFACTS_DIR"
[[ -f "$MATRIX_CSV" ]] || echo "cell,server,swa,overlap,N,cap,exit_code,completed,failed,e2e_time,input_throughput,output_throughput,p50_e2e,p99_e2e,row_dir,timestamp_utc,notes" > "$MATRIX_CSV"
[[ -f "$PROGRESS_MD" ]] || printf "# Smoke — %s\n\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$PROGRESS_MD"

log() { echo "- $(date -u +%Y-%m-%dT%H:%M:%SZ): $1" >> "$PROGRESS_MD"; }
field() { python3 -c "import json
try: print(json.load(open('$1')).get('$2','-'))
except Exception: print('-')" 2>/dev/null; }
alive() { curl -fsS --max-time 5 "http://$HOST:$PORT/health" >/dev/null 2>&1; }
alive_retry() { for ((i=1;i<=15;i++)); do alive && return 0; sleep 2; done; return 1; }

# Cell format: <cell_label>|<server>|<swa>|<overlap>|<N>|<CAP>|<sample_jsonl>
# server: tp-static | ep-static | paras-t<threshold>
# swa, overlap: on | off
CELLS=(
    "tp-static__swa-on__overlap-off|tp-static|on|off|32|2048|$SAMPLES_DIR/${MODEL_SLUG}__dapo__n1024_seed42.jsonl"
    "ep-static__swa-on__overlap-off|ep-static|on|off|2000|2048|$SAMPLES_DIR/${MODEL_SLUG}__dapo__n2048_seed43.jsonl"
    "paras-t64__swa-on__overlap-on|paras-t64|on|on|256|8192|$SAMPLES_DIR/${MODEL_SLUG}__dapo__n1024_seed42.jsonl"
)

cd /home/shaoyuw/sglang
log "=== SMOKE START === cells=${#CELLS[@]}"

for spec in "${CELLS[@]}"; do
    IFS='|' read -r CELL SERVER SWA OVERLAP N CAP SAMPLE <<< "$spec"
    SERVER_DIR="$ARTIFACTS_DIR/${MODEL_SLUG}__${CELL}"
    ROW_DIR="$SERVER_DIR/n${N}_cap${CAP}"
    SERVER_LOG="$SERVER_DIR/server.log"
    SERVER_METRICS="$SERVER_DIR/metrics_timeseries.csv"
    mkdir -p "$ROW_DIR"; rm -f "$SERVER_METRICS"

    ENABLE_PARAS=0; EXTRA=()
    case "$SERVER" in
        tp-static)
            LAUNCH=scripts/paras/eval/a100/gptoss/launch_server_tp_tp.sh
            MFS=$TP_MFS; MAX_RUN=$TP_MAX_RUNNING; CG=$TP_CG_MAX_BS ;;
        ep-static)
            LAUNCH=scripts/paras/eval/a100/gptoss/launch_server_dp_ep.sh
            MFS=$EP_MFS; MAX_RUN=$EP_MAX_RUNNING; CG=$EP_CG_MAX_BS ;;
        paras-t*)
            LAUNCH=scripts/paras/eval/a100/gptoss/launch_server_dp_ep.sh
            MFS=$EP_MFS; MAX_RUN=$EP_MAX_RUNNING; CG=$EP_CG_MAX_BS
            ENABLE_PARAS=1
            THRESHOLD=${SERVER##paras-t}
            EXTRA+=(--paras-auto-switch-policy rollout --paras-auto-switch-threshold "$THRESHOLD" --paras-tp-cuda-graph-max-bs 128) ;;
        *) log "ERROR unknown server: $SERVER"; continue ;;
    esac

    HSWA=$([[ "$SWA" == "on" ]] && echo 1 || echo 0)
    DOVL=$([[ "$OVERLAP" == "off" ]] && echo 1 || echo 0)

    log "=== CELL: $CELL ===  server=$SERVER swa=$SWA overlap=$OVERLAP N=$N cap=$CAP"

    cat > "$SERVER_DIR/server_launch.cmd.txt" <<EOF
ENABLE_PARAS=$ENABLE_PARAS \\
HYBRID_SWA=$HSWA \\
DISABLE_OVERLAP=$DOVL \\
DISABLE_RADIX_CACHE=1 \\
NUM_GPUS=$NUM_GPUS \\
ENABLE_CUDA_GRAPH=1 \\
CUDA_GRAPH_MAX_BS=$CG \\
MAX_RUNNING_REQUESTS=$MAX_RUN \\
MEM_FRACTION_STATIC=$MFS \\
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256 \\
    bash $LAUNCH \\
    --max-prefill-tokens $MAX_PREFILL_TOKENS \\
    --paras-metrics-file $SERVER_METRICS \\
    ${EXTRA[*]}
EOF

    bash scripts/paras/eval/paras_cmd/kill.sh >> "$PROGRESS_MD" 2>&1 || true
    sleep 5

    ENABLE_PARAS=$ENABLE_PARAS HYBRID_SWA=$HSWA DISABLE_OVERLAP=$DOVL DISABLE_RADIX_CACHE=1 \
    NUM_GPUS=$NUM_GPUS ENABLE_CUDA_GRAPH=1 CUDA_GRAPH_MAX_BS=$CG \
    MAX_RUNNING_REQUESTS=$MAX_RUN MEM_FRACTION_STATIC=$MFS \
    SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256 \
        bash "$LAUNCH" --max-prefill-tokens "$MAX_PREFILL_TOKENS" \
        --paras-metrics-file "$SERVER_METRICS" "${EXTRA[@]}" > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    log "  pid=$SERVER_PID"

    READY=0
    LOG_FILE="$SERVER_LOG" TIMEOUT_TRIES="$TIMEOUT_TRIES" SLEEP_BETWEEN="$SLEEP_BETWEEN" \
        bash scripts/paras/eval/paras_cmd/wait_ready.sh >> "$PROGRESS_MD" 2>&1 || READY=$?
    if [[ "$READY" -ne 0 ]] || ! alive_retry; then
        log "  FAILED to start (ready=$READY)"
        echo "$CELL,$SERVER,$SWA,$OVERLAP,$N,$CAP,-,-,-,-,-,-,-,-,$ROW_DIR,$(date -u +%Y-%m-%dT%H:%M:%SZ),\"server_failed\"" >> "$MATRIX_CSV"
        bash scripts/paras/eval/paras_cmd/kill.sh >> "$PROGRESS_MD" 2>&1 || true
        sleep 5; continue
    fi

    log "  bench start"
    BENCH_EXIT=0
    python -m sglang.bench_paras --model "$MODEL_NAME" --dataset-jsonl "$SAMPLE" \
        --mode-label "$CELL" --num-requests "$N" --group-size 1 --spec-mode \
        --max-completion-tokens-cap "$CAP" --output-dir "$ROW_DIR" \
        --host "$HOST" --port "$PORT" --seed 42 --dump-outputs \
        > "$ROW_DIR/bench.log" 2>&1 || BENCH_EXIT=$?

    sleep 10
    [[ -f "$SERVER_METRICS" ]] && cp "$SERVER_METRICS" "$ROW_DIR/metrics_timeseries.csv"

    COMPLETED=$(field "$ROW_DIR/summary.json" completed)
    FAILED=$(field "$ROW_DIR/summary.json" failed)
    E2E=$(field "$ROW_DIR/summary.json" e2e_time)
    IN_TP=$(field "$ROW_DIR/summary.json" input_throughput)
    OUT_TP=$(field "$ROW_DIR/summary.json" output_throughput)
    P50=$(field "$ROW_DIR/summary.json" p50_e2e_latency_s)
    P99=$(field "$ROW_DIR/summary.json" p99_e2e_latency_s)
    NOTES=""; [[ "$BENCH_EXIT" -ne 0 ]] && NOTES="bench_exit=$BENCH_EXIT"
    echo "$CELL,$SERVER,$SWA,$OVERLAP,$N,$CAP,$BENCH_EXIT,$COMPLETED,$FAILED,$E2E,$IN_TP,$OUT_TP,$P50,$P99,$ROW_DIR,$(date -u +%Y-%m-%dT%H:%M:%SZ),\"$NOTES\"" >> "$MATRIX_CSV"
    log "  bench end exit=$BENCH_EXIT completed=$COMPLETED out_tput=$OUT_TP"

    nvidia-smi > "$SERVER_DIR/nvidia_smi.txt" 2>&1 || true
    bash scripts/paras/eval/paras_cmd/kill.sh >> "$PROGRESS_MD" 2>&1 || true
    sleep 8
done

log "=== SMOKE COMPLETE ==="
echo "Matrix: $MATRIX_CSV"
cat "$MATRIX_CSV"
