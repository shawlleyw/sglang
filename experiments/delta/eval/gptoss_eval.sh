#!/usr/bin/bash
# eval.sh — SGLang multi-baseline evaluation, NCSA Delta
#
# Usage:
#   bash experiments/delta/eval/ep16_eval.sh [RESULTS_DIR]
#
#   RESULTS_DIR  required; a parent directory that holds one sub-dir per run.
#                Example: /scratch/myrun/results
#
# Run directory naming: <RESULTS_DIR>/<system>_<server_profile>-<dataset_label>/
#   e.g.  sglang_ep16-sharegpt_regular/
#         sglang_pp4tp4-legal_court_balanced/
#
# Prerequisites:
#   - SLURM allocation active (4 nodes × 4 A100-SXM4-40GB = 16 GPUs)
#   - conda env 'sglang' installed on all nodes
#   - SSH access to compute nodes + tmux available
#   - Model config/tokenizer at $MODEL_PATH (weights not needed for dummy)
#   - Gate profile parquets in place (see EXPERIMENT MATRIX below)

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Load fixed config and function libraries ──────────────────────────────────
source "$EVAL_DIR/config.sh"
source "$EVAL_DIR/evallib/cluster.sh"
source "$EVAL_DIR/evallib/server.sh"
source "$EVAL_DIR/evallib/benchmark.sh"

# ── Results directory (required as $1) ────────────────────────────────────────
RESULTS_DIR="${1:?ERROR: RESULTS_DIR is required as the first argument (e.g. /path/to/results)}"

# ── Gate profiles ─────────────────────────────────────────────────────────────
GATE_PROFILES=(
    "${GATING_DIR}/gating_gptoss120b_sharegpt_200.parquet:sharegpt_regular"
    "${GATING_DIR}/balanced_output/balanced_gptoss120b_sharegpt_200.parquet:sharegpt_balanced"
    "${GATING_DIR}/gating_legal_court_opinions_200.parquet:legal_court_regular"
    "${GATING_DIR}/balanced_output/balanced_legal_court_opinions_200.parquet:legal_court_balanced"
)

# ── Server profiles to evaluate ──────────────────────────────────────────────
SERVER_PROFILES=( ep16 ep16_limited pp4tp4 )

# ── Full experiment matrix: SERVER_PROFILES × GATE_PROFILES ──────────────────
EXPERIMENTS=()
for _sp in "${SERVER_PROFILES[@]}"; do
    for _gp in "${GATE_PROFILES[@]}"; do
        EXPERIMENTS+=("${_sp}:${_gp}")
    done
done

MAX_RETRIES=3
MEM_FRAC_STEP=0.02   # how much to reduce MEM_FRAC on each OOM retry

# ─────────────────────────────────────────────────────────────────────────────
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') [main] $*"; }

mkdir -p "$RESULTS_DIR"

# ── Discover nodes ────────────────────────────────────────────────────────────
log "Discovering cluster nodes..."
discover_nodes || exit 1

log "Evaluation starting"
log "  System      : $SYSTEM_NAME"
log "  Results dir : $RESULTS_DIR"
log "  Head        : $HEAD ($HEAD_IP)"
log "  Workers     : ${WORKERS[*]}"
log "  Server profiles : ${SERVER_PROFILES[*]}"
log "  Gate profiles   : ${#GATE_PROFILES[@]}"
log "  Experiments     : ${#EXPERIMENTS[@]} (${#SERVER_PROFILES[@]} × ${#GATE_PROFILES[@]}), up to $MAX_RETRIES retries each"
log "  Initial MEM_FRAC: $MEM_FRAC"

EXP_NUM=0
TOTAL=${#EXPERIMENTS[@]}

for exp_entry in "${EXPERIMENTS[@]}"; do
    IFS=: read -r server_profile gate_profile dataset <<< "$exp_entry"
    EXP_NUM=$((EXP_NUM + 1))

    run_name="${SYSTEM_NAME}_${server_profile}-${dataset}"
    run_dir="$RESULTS_DIR/$run_name"
    mkdir -p "$run_dir"

    log "================================================================"
    log "[$EXP_NUM/$TOTAL] $run_name  (server=$server_profile)"
    log "================================================================"

    if [ ! -f "$gate_profile" ]; then
        log "SKIP: profile not found: $gate_profile"
        printf '{"error":"profile_not_found","path":"%s"}\n' "$gate_profile" \
            > "$run_dir/result.json"
        continue
    fi

    SUCCESS=0
    server_log_dir="$run_dir/logs"
    server_cmd="$run_dir/server_cmd.sh"
    bench_result="$run_dir/bench_result.json"
    bench_cmd="$run_dir/bench_cmd.sh"

    for attempt in $(seq 1 "$MAX_RETRIES"); do
        log "Attempt $attempt/$MAX_RETRIES (MEM_FRAC=$MEM_FRAC)..."

        kill_server
        sleep 5

        rm -rf "$server_log_dir"
        launch_server "$server_profile" "$gate_profile" "$server_log_dir" "$server_cmd"

        if wait_for_server; then
            if run_benchmark "$bench_result" "$bench_cmd"; then
                cp "$bench_result" "$run_dir/result.json"
                SUCCESS=1
                break
            else
                log "Benchmark failed on attempt $attempt."
            fi
        else
            if is_oom "$server_log_dir"; then
                new_frac=$(awk "BEGIN {printf \"%.2f\", $MEM_FRAC - $MEM_FRAC_STEP}")
                log "OOM detected — reducing MEM_FRAC: $MEM_FRAC -> $new_frac"
                MEM_FRAC="$new_frac"
            else
                log "Server failed (non-OOM). See logs in: $server_log_dir/"
            fi
        fi

        kill_server
        sleep 10
    done

    if [ "$SUCCESS" -eq 0 ]; then
        log "FAILED: $run_name — all $MAX_RETRIES attempts unsuccessful."
    else
        log "SUCCESS: $run_name"
        log "Result: $(tr -d '\n' < "$run_dir/result.json")"
    fi
done

# ── Cleanup ───────────────────────────────────────────────────────────────────
kill_server

log "================================================================"
log "All $TOTAL experiments done. Results in: $RESULTS_DIR"
log "================================================================"
