#!/bin/bash
# run_sweep.sh — 4-server, 16-bench sweep for the 2026-05-15 ParaS rollout matrix.
#
# Model:     gpt-oss-120b-BF16-unsloth
# Dataset:   dapo (pre-sampled snapshots in artifacts/.../samples/)
# Systems:   tp-static (mfs=0.80), ep-static (mfs=0.75), paras-t64 (mfs=0.75), paras-t128 (mfs=0.75)
# Per-server: N in {1024,2048} x cap in {16384,32768} = 4 benchmarks
# Total: 16 benchmarks across 4 server launches (4 systems x 4 benches each).
#
# Per-bench metrics: line-offset slicing from the continuous server-side CSV
# (--paras-metrics-file points at $SERVER_DIR/metrics_timeseries.csv; the
# sampler now reports the correct mode + global counters in static modes
# thanks to the ParasMetricsSampler patch).
#
# Failures: recorded in matrix.csv with a notes field; the sweep does NOT
# halt on failure. Retries are handled in a separate manual pass.
#
# This driver is intentionally uncommitted — single-use for the sweep.

set -uo pipefail   # NOT -e: we want to capture failures, not exit on them

ARTIFACTS_DIR=/home/shaoyuw/sglang/artifacts/20260515_sweep_gptoss_dapo
SAMPLES_DIR=$ARTIFACTS_DIR/samples
SAMPLE_N1024=$SAMPLES_DIR/gpt-oss-120b__dapo__n1024_seed42.jsonl
SAMPLE_N2048=$SAMPLES_DIR/gpt-oss-120b__dapo__n2048_seed43.jsonl
MATRIX_CSV=$ARTIFACTS_DIR/matrix.csv
PROGRESS_MD=$ARTIFACTS_DIR/PROGRESS.md

MODEL_SLUG=gpt-oss-120b
MODEL_NAME=gpt-oss-120b-BF16-unsloth
HOST=127.0.0.1
PORT=30000
NUM_GPUS=8

MAX_RUNNING_REQUESTS=2048
DEEPEP_TOKENS_PER_RANK=256
CUDA_GRAPH_MAX_BS=256
MAX_PREFILL_TOKENS=8192
PARAS_TP_CUDA_GRAPH_MAX_BS=128   # raised from 64 to cover paras-t128 threshold

# wait_ready budget for gpt-oss BF16 load: 120 tries * 10s = 20min
TIMEOUT_TRIES=120
SLEEP_BETWEEN=10

mkdir -p "$ARTIFACTS_DIR"
if [[ ! -f "$MATRIX_CSV" ]]; then
    echo "system,N,cap,exit_code,completed,failed,e2e_time,input_throughput,output_throughput,row_dir,timestamp_utc,notes" > "$MATRIX_CSV"
fi
if [[ ! -f "$PROGRESS_MD" ]]; then
    {
        echo "# ParaS sweep PROGRESS — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo ""
        echo "Live append-only event log written by run_sweep.sh. Aggregated into REPORT.md after the matrix completes."
        echo ""
    } > "$PROGRESS_MD"
fi

log_progress() {
    local msg="$1"
    local ts
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "- $ts: $msg" >> "$PROGRESS_MD"
}

get_summary_field() {
    local summary_json="$1"
    local field="$2"
    python3 -c "import json,sys
try:
    print(json.load(open('$summary_json'))['$field'])
except Exception:
    print('-')" 2>/dev/null
}

# Slice metrics CSV [lines_before+1 .. lines_after] into row_dir, prepending header.
slice_metrics() {
    local server_csv="$1"
    local row_csv="$2"
    local lines_before="$3"
    local lines_after="$4"

    if [[ ! -f "$server_csv" ]]; then
        return 1
    fi
    if [[ "$lines_after" -le "$lines_before" ]]; then
        head -n 1 "$server_csv" > "$row_csv" 2>/dev/null || return 1
        return 0
    fi
    {
        head -n 1 "$server_csv"
        sed -n "$((lines_before + 1)),${lines_after}p" "$server_csv"
    } > "$row_csv"
}

server_alive() {
    curl -fsS --max-time 5 "http://$HOST:$PORT/health" >/dev/null 2>&1
}

# Probe with retries: sglang's /health returns 503 during the post-startup
# /generate warmup (a ~1-2s window between "Application startup complete" and
# "fired up and ready to roll"). A single server_alive() call can hit that
# window and falsely report dead. Use this for the first check after wait_ready.
server_alive_with_retries() {
    local attempts=15
    local sleep_s=2
    for ((i = 1; i <= attempts; i++)); do
        if server_alive; then
            return 0
        fi
        sleep "$sleep_s"
    done
    return 1
}

append_matrix_row() {
    local system="$1" n="$2" cap="$3" exit_code="$4"
    local completed="$5" failed_count="$6" e2e="$7" in_tp="$8" out_tp="$9"
    local row_dir="${10}" notes="${11}"
    local ts
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "$system,$n,$cap,$exit_code,$completed,$failed_count,$e2e,$in_tp,$out_tp,$row_dir,$ts,\"$notes\"" >> "$MATRIX_CSV"
}

run_one_bench() {
    local system="$1" n="$2" cap="$3"
    local sample_path="$4" row_dir="$5" server_metrics="$6"

    mkdir -p "$row_dir"
    local lines_before
    lines_before=$(wc -l < "$server_metrics" 2>/dev/null || echo 0)

    log_progress "BENCH start: system=$system N=$n cap=$cap (server_metrics_lines_before=$lines_before)"

    local bench_exit=0
    python -m sglang.bench_paras \
        --model "$MODEL_NAME" \
        --dataset-jsonl "$sample_path" \
        --mode-label "$system" \
        --num-requests "$n" \
        --group-size 1 \
        --spec-mode \
        --max-completion-tokens-cap "$cap" \
        --output-dir "$row_dir" \
        --host "$HOST" --port "$PORT" \
        --seed 42 \
        > "$row_dir/bench.log" 2>&1 \
        || bench_exit=$?

    sleep 10  # tail drain for final samples + final scheduler ticks

    local lines_after
    lines_after=$(wc -l < "$server_metrics" 2>/dev/null || echo 0)
    if [[ -f "$server_metrics" ]]; then
        slice_metrics "$server_metrics" "$row_dir/metrics_timeseries.csv" \
            "$lines_before" "$lines_after" \
            || log_progress "WARN: slice_metrics failed for $row_dir"
    fi

    local completed="-" failed_count="-" e2e="-" in_tp="-" out_tp="-"
    if [[ -f "$row_dir/summary.json" ]]; then
        completed=$(get_summary_field "$row_dir/summary.json" completed)
        failed_count=$(get_summary_field "$row_dir/summary.json" failed)
        e2e=$(get_summary_field "$row_dir/summary.json" e2e_time)
        in_tp=$(get_summary_field "$row_dir/summary.json" input_throughput)
        out_tp=$(get_summary_field "$row_dir/summary.json" output_throughput)
    fi

    local notes=""
    if [[ "$bench_exit" -ne 0 ]]; then
        notes="bench_exit=$bench_exit"
    fi

    append_matrix_row "$system" "$n" "$cap" "$bench_exit" \
        "$completed" "$failed_count" "$e2e" "$in_tp" "$out_tp" \
        "$row_dir" "$notes"

    log_progress "BENCH end: system=$system N=$n cap=$cap exit=$bench_exit completed=$completed failed=$failed_count e2e=$e2e in_tput=$in_tp out_tput=$out_tp lines_after=$lines_after"
}

launch_server() {
    local system="$1" server_dir="$2" mfs="$3" extra_paras_args_str="$4"

    bash scripts/paras/eval/paras_cmd/kill.sh >> "$PROGRESS_MD" 2>&1 || true
    sleep 5

    local server_log="$server_dir/server.log"
    local server_metrics="$server_dir/metrics_timeseries.csv"
    mkdir -p "$server_dir"
    rm -f "$server_metrics"   # fresh CSV for this server

    local launch_script enable_paras
    case "$system" in
        tp-static)
            launch_script="scripts/paras/eval/a100/gptoss/launch_server_tp_tp.sh"
            enable_paras=0
            ;;
        ep-static)
            launch_script="scripts/paras/eval/a100/gptoss/launch_server_dp_ep.sh"
            enable_paras=0
            ;;
        paras-t64|paras-t128)
            launch_script="scripts/paras/eval/a100/gptoss/launch_server_dp_ep.sh"
            enable_paras=1
            ;;
        *)
            log_progress "ERROR: unknown system $system"
            return 2
            ;;
    esac

    local extra_args=(
        --chunked-prefill-size -1
        --max-prefill-tokens "$MAX_PREFILL_TOKENS"
        --paras-metrics-file "$server_metrics"
    )
    if [[ -n "$extra_paras_args_str" ]]; then
        # shellcheck disable=SC2206
        local paras_args_arr=($extra_paras_args_str)
        extra_args+=("${paras_args_arr[@]}")
    fi

    log_progress "SERVER launch: system=$system mfs=$mfs script=$launch_script extra=${extra_args[*]}"

    ENABLE_PARAS="$enable_paras" \
    NUM_GPUS="$NUM_GPUS" \
    ENABLE_CUDA_GRAPH=1 \
    CUDA_GRAPH_MAX_BS="$CUDA_GRAPH_MAX_BS" \
    MAX_RUNNING_REQUESTS="$MAX_RUNNING_REQUESTS" \
    MEM_FRACTION_STATIC="$mfs" \
    SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="$DEEPEP_TOKENS_PER_RANK" \
        bash "$launch_script" "${extra_args[@]}" \
        > "$server_log" 2>&1 &
    local server_pid=$!
    log_progress "SERVER pid=$server_pid log=$server_log"

    local ready_exit=0
    LOG_FILE="$server_log" TIMEOUT_TRIES="$TIMEOUT_TRIES" SLEEP_BETWEEN="$SLEEP_BETWEEN" \
        bash scripts/paras/eval/paras_cmd/wait_ready.sh >> "$PROGRESS_MD" 2>&1 \
        || ready_exit=$?

    if [[ "$ready_exit" -ne 0 ]]; then
        log_progress "SERVER FAILED to start: system=$system ready_exit=$ready_exit (see $server_log)"
        return 1
    fi

    if ! server_alive_with_retries; then
        log_progress "SERVER FAILED post-wait_ready /health probe: system=$system (see $server_log)"
        return 1
    fi

    log_progress "SERVER READY: system=$system"
    return 0
}

kill_server() {
    local system="$1" server_dir="$2"
    nvidia-smi > "$server_dir/nvidia_smi.txt" 2>&1 || true
    bash scripts/paras/eval/paras_cmd/kill.sh >> "$PROGRESS_MD" 2>&1 || true
    sleep 8
    log_progress "SERVER killed: system=$system"
}

cd /home/shaoyuw/sglang || { echo "ERROR: cd /home/shaoyuw/sglang failed" >&2; exit 2; }

for s in "$SAMPLE_N1024" "$SAMPLE_N2048"; do
    if [[ ! -f "$s" ]]; then
        echo "ERROR: snapshot missing: $s" >&2
        exit 2
    fi
done

declare -A SYSTEM_MFS=( [tp-static]=0.80 [ep-static]=0.75 [paras-t64]=0.75 [paras-t128]=0.75 )
declare -A SYSTEM_EXTRA=(
    [tp-static]=""
    [ep-static]=""
    [paras-t64]="--paras-auto-switch-policy rollout --paras-auto-switch-threshold 64 --paras-tp-cuda-graph-max-bs $PARAS_TP_CUDA_GRAPH_MAX_BS"
    [paras-t128]="--paras-auto-switch-policy rollout --paras-auto-switch-threshold 128 --paras-tp-cuda-graph-max-bs $PARAS_TP_CUDA_GRAPH_MAX_BS"
)
SYSTEMS=(ep-static paras-t128)
BENCHES=("1024 16384" "1024 32768" "2048 16384" "2048 32768")

log_progress "=== SWEEP START === artifacts=$ARTIFACTS_DIR systems=${SYSTEMS[*]}"

for system in "${SYSTEMS[@]}"; do
    SERVER_DIR="$ARTIFACTS_DIR/${MODEL_SLUG}__${system}"
    SERVER_METRICS="$SERVER_DIR/metrics_timeseries.csv"

    log_progress "=== Starting system: $system ==="

    mfs="${SYSTEM_MFS[$system]}"
    extra="${SYSTEM_EXTRA[$system]}"

    launch_exit=0
    launch_server "$system" "$SERVER_DIR" "$mfs" "$extra" || launch_exit=$?
    if [[ "$launch_exit" -ne 0 ]]; then
        log_progress "FAILED to launch $system; marking 4 benches as skipped_server_down"
        for bspec in "${BENCHES[@]}"; do
            read -r n cap <<< "$bspec"
            append_matrix_row "$system" "$n" "$cap" "-" "-" "-" "-" "-" "-" "-" "skipped_server_down"
        done
        # Belt-and-suspenders kill in case partial launch left a zombie
        bash scripts/paras/eval/paras_cmd/kill.sh >> "$PROGRESS_MD" 2>&1 || true
        sleep 5
        continue
    fi

    # Inner loop over (N, cap)
    server_died=0
    for i in "${!BENCHES[@]}"; do
        bspec="${BENCHES[$i]}"
        read -r n cap <<< "$bspec"

        if (( server_died )); then
            append_matrix_row "$system" "$n" "$cap" "-" "-" "-" "-" "-" "-" "-" "skipped_server_died"
            continue
        fi

        if ! server_alive; then
            log_progress "SERVER DIED before bench: system=$system N=$n cap=$cap"
            server_died=1
            append_matrix_row "$system" "$n" "$cap" "-" "-" "-" "-" "-" "-" "-" "skipped_server_died"
            continue
        fi

        sample_path="$SAMPLE_N1024"
        if [[ "$n" == "2048" ]]; then
            sample_path="$SAMPLE_N2048"
        fi
        row_dir="$SERVER_DIR/n${n}_cap${cap}"

        run_one_bench "$system" "$n" "$cap" "$sample_path" "$row_dir" "$SERVER_METRICS"

        # 10s cooldown between benches (policy state carries over for paras)
        sleep 10
    done

    kill_server "$system" "$SERVER_DIR"
    sleep 5
done

log_progress "=== SWEEP COMPLETE ==="
echo ""
echo "=== SWEEP COMPLETE ==="
echo "Matrix CSV: $MATRIX_CSV"
echo ""
cat "$MATRIX_CSV"
