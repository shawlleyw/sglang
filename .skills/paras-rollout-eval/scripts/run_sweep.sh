#!/bin/bash
# run_sweep.sh — server-reuse driver template.
#
# Use when cells differ only on CLIENT args (N, cap, seed, sample). One server
# amortizes multiple benches. Outer loop = system (server-per), inner loop =
# (N, cap) (server-reuse).
#
# Output per system:
#   $ARTIFACTS_DIR/<model>__<system>/
#     server_launch.cmd.txt
#     server.log, nvidia_smi.txt
#     metrics_timeseries.csv (full server lifetime)
#     n<N>_cap<CAP>/{summary.json, per_request.jsonl, outputs.jsonl,
#                    run_config.json, metrics_timeseries.csv (sliced), bench.log}
#   matrix.csv (one row per bench), PROGRESS.md, run_sweep.log

set -uo pipefail

ARTIFACTS_DIR=${ARTIFACTS_DIR:-/home/shaoyuw/sglang/artifacts/$(date -u +%Y%m%dT%H%M%SZ)_sweep}
SAMPLES_DIR=${SAMPLES_DIR:-$ARTIFACTS_DIR/samples}
MATRIX_CSV=$ARTIFACTS_DIR/matrix.csv
PROGRESS_MD=$ARTIFACTS_DIR/PROGRESS.md
SKILL_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

MODEL_SLUG=${MODEL_SLUG:-gpt-oss-120b}
MODEL_NAME=${MODEL_NAME:-gpt-oss-120b-BF16-unsloth}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-30000}
NUM_GPUS=${NUM_GPUS:-8}

EP_MFS=${EP_MFS:-0.75};  EP_MAX_RUNNING=${EP_MAX_RUNNING:-2048}; EP_CG=${EP_CG:-256}
TP_MFS=${TP_MFS:-0.80};  TP_MAX_RUNNING=${TP_MAX_RUNNING:-2048}; TP_CG=${TP_CG:-256}
MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-8192}
TIMEOUT_TRIES=${TIMEOUT_TRIES:-120}; SLEEP_BETWEEN=${SLEEP_BETWEEN:-10}

mkdir -p "$ARTIFACTS_DIR"
[[ -f "$MATRIX_CSV" ]] || echo "system,N,cap,exit_code,completed,failed,e2e_time,input_throughput,output_throughput,p50_e2e,p99_e2e,row_dir,timestamp_utc,notes" > "$MATRIX_CSV"
[[ -f "$PROGRESS_MD" ]] || printf "# Sweep — %s\n\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$PROGRESS_MD"

log() { echo "- $(date -u +%Y-%m-%dT%H:%M:%SZ): $1" >> "$PROGRESS_MD"; }
field() { python3 -c "import json
try: print(json.load(open('$1')).get('$2','-'))
except Exception: print('-')" 2>/dev/null; }
alive() { curl -fsS --max-time 5 "http://$HOST:$PORT/health" >/dev/null 2>&1; }
alive_retry() { for ((i=1;i<=15;i++)); do alive && return 0; sleep 2; done; return 1; }

append_row() {
    echo "$1,$2,$3,$4,$5,$6,$7,$8,$9,${10},${11},${12},$(date -u +%Y-%m-%dT%H:%M:%SZ),\"${13}\"" >> "$MATRIX_CSV"
}

# Per-system parity defaults (always emitted; override individually as needed).
# `system` values: tp-static | ep-static | paras-t<threshold>
SYSTEMS=(tp-static ep-static paras-t64)

# Bench grid; reused across all systems.
declare -a BENCHES_N=(1024 2048)
declare -a BENCHES_CAP=(16384 32768)

cd /home/shaoyuw/sglang
log "=== SWEEP START === systems=${SYSTEMS[*]}"

for SYSTEM in "${SYSTEMS[@]}"; do
    SERVER_DIR="$ARTIFACTS_DIR/${MODEL_SLUG}__${SYSTEM}"
    SERVER_LOG="$SERVER_DIR/server.log"
    SERVER_METRICS="$SERVER_DIR/metrics_timeseries.csv"
    mkdir -p "$SERVER_DIR"; rm -f "$SERVER_METRICS"

    # Parity contract: HYBRID_SWA=1 + DISABLE_OVERLAP=0 + DISABLE_RADIX_CACHE=1
    # for every system. SYNC_TOKEN_IDS_ACROSS_TP is baked into launch_server_tp_tp.sh.
    # Per-system overrides: launch script, mfs, max_running, cuda_graph_max_bs, paras flags.
    ENABLE_PARAS=0; EXTRA=()
    case "$SYSTEM" in
        tp-static)
            LAUNCH=scripts/paras/eval/a100/gptoss/launch_server_tp_tp.sh
            MFS=$TP_MFS; MAX_RUN=$TP_MAX_RUNNING; CG=$TP_CG ;;
        ep-static)
            LAUNCH=scripts/paras/eval/a100/gptoss/launch_server_dp_ep.sh
            MFS=$EP_MFS; MAX_RUN=$EP_MAX_RUNNING; CG=$EP_CG ;;
        paras-t*)
            LAUNCH=scripts/paras/eval/a100/gptoss/launch_server_dp_ep.sh
            MFS=$EP_MFS; MAX_RUN=$EP_MAX_RUNNING; CG=$EP_CG
            ENABLE_PARAS=1
            THRESHOLD=${SYSTEM##paras-t}
            EXTRA+=(--paras-auto-switch-policy rollout --paras-auto-switch-threshold "$THRESHOLD" --paras-tp-cuda-graph-max-bs 128) ;;
        *) log "ERROR unknown system: $SYSTEM"; continue ;;
    esac

    log "=== SYSTEM: $SYSTEM === launch=$LAUNCH mfs=$MFS max_running=$MAX_RUN"

    cat > "$SERVER_DIR/server_launch.cmd.txt" <<EOF
ENABLE_PARAS=$ENABLE_PARAS \\
HYBRID_SWA=1 \\
DISABLE_OVERLAP=0 \\
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

    ENABLE_PARAS=$ENABLE_PARAS HYBRID_SWA=1 DISABLE_OVERLAP=0 DISABLE_RADIX_CACHE=1 \
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
        log "  FAILED to start; marking all benches skipped_server_down"
        for N in "${BENCHES_N[@]}"; do for CAP in "${BENCHES_CAP[@]}"; do
            append_row "$SYSTEM" "$N" "$CAP" - - - - - - - - "-" "skipped_server_down"
        done; done
        bash scripts/paras/eval/paras_cmd/kill.sh >> "$PROGRESS_MD" 2>&1 || true
        sleep 5; continue
    fi
    log "  SERVER READY"

    SERVER_DIED=0
    for N in "${BENCHES_N[@]}"; do
        for CAP in "${BENCHES_CAP[@]}"; do
            ROW_DIR="$SERVER_DIR/n${N}_cap${CAP}"
            mkdir -p "$ROW_DIR"

            if (( SERVER_DIED )); then
                append_row "$SYSTEM" "$N" "$CAP" - - - - - - - - "$ROW_DIR" "skipped_server_died"
                continue
            fi
            if ! alive; then
                log "  server died before $N/$CAP"; SERVER_DIED=1
                append_row "$SYSTEM" "$N" "$CAP" - - - - - - - - "$ROW_DIR" "skipped_server_died"
                continue
            fi

            SAMPLE="$SAMPLES_DIR/${MODEL_SLUG}__dapo__n${N}_seed$([[ $N == 1024 ]] && echo 42 || echo 43).jsonl"
            LINES_BEFORE=$(wc -l < "$SERVER_METRICS" 2>/dev/null || echo 0)
            log "BENCH start: $SYSTEM N=$N cap=$CAP lines_before=$LINES_BEFORE"

            BENCH_EXIT=0
            python -m sglang.bench_paras --model "$MODEL_NAME" --dataset-jsonl "$SAMPLE" \
                --mode-label "$SYSTEM" --num-requests "$N" --group-size 1 --spec-mode \
                --max-completion-tokens-cap "$CAP" --output-dir "$ROW_DIR" \
                --host "$HOST" --port "$PORT" --seed 42 --dump-outputs \
                > "$ROW_DIR/bench.log" 2>&1 || BENCH_EXIT=$?

            sleep 10
            LINES_AFTER=$(wc -l < "$SERVER_METRICS" 2>/dev/null || echo 0)
            python3 "$SKILL_DIR/slice_metrics.py" --src "$SERVER_METRICS" \
                --dst "$ROW_DIR/metrics_timeseries.csv" \
                --lines-before "$LINES_BEFORE" --lines-after "$LINES_AFTER" \
                >> "$PROGRESS_MD" 2>&1 || true

            COMPLETED=$(field "$ROW_DIR/summary.json" completed)
            FAILED=$(field "$ROW_DIR/summary.json" failed)
            E2E=$(field "$ROW_DIR/summary.json" e2e_time)
            IN_TP=$(field "$ROW_DIR/summary.json" input_throughput)
            OUT_TP=$(field "$ROW_DIR/summary.json" output_throughput)
            P50=$(field "$ROW_DIR/summary.json" p50_e2e_latency_s)
            P99=$(field "$ROW_DIR/summary.json" p99_e2e_latency_s)
            NOTES=""; [[ "$BENCH_EXIT" -ne 0 ]] && NOTES="bench_exit=$BENCH_EXIT"
            append_row "$SYSTEM" "$N" "$CAP" "$BENCH_EXIT" "$COMPLETED" "$FAILED" "$E2E" "$IN_TP" "$OUT_TP" "$P50" "$P99" "$ROW_DIR" "$NOTES"
            log "BENCH end: $SYSTEM N=$N cap=$CAP exit=$BENCH_EXIT completed=$COMPLETED out=$OUT_TP"

            sleep 10  # cooldown between benches; policy state may carry over for paras
        done
    done

    nvidia-smi > "$SERVER_DIR/nvidia_smi.txt" 2>&1 || true
    bash scripts/paras/eval/paras_cmd/kill.sh >> "$PROGRESS_MD" 2>&1 || true
    sleep 8
done

log "=== SWEEP COMPLETE ==="
echo "Matrix: $MATRIX_CSV"
cat "$MATRIX_CSV"
