#!/usr/bin/bash
# evallib/benchmark.sh — SGLang benchmark runner
# Source this file; do not execute directly.
#
# Requires (from config.sh):
#   SERVER_PORT, BENCH_TIMEOUT, MODEL_NAME, BENCH_BACKEND, BENCH_DATASET,
#   BENCH_NUM_PROMPTS, BENCH_REQUEST_RATE,
#   BENCH_RANDOM_INPUT_LEN, BENCH_RANDOM_OUTPUT_LEN, BENCH_RANDOM_RANGE_RATIO
# Requires (from cluster.sh):
#   HEAD

log_bench() { echo "$(date '+%Y-%m-%d %H:%M:%S') [bench] $*"; }

# run_benchmark <result_json_path> <cmd_file_path> [extra_args...]
run_benchmark() {
    local result_file="$1"
    local cmd_file="$2"
    shift 2
    local extra_args=("$@")

    local cmd=(
        python -m sglang.bench_serving
        --backend "$BENCH_BACKEND"
        --host "$HEAD"
        --port "$SERVER_PORT"
        --model "$MODEL_NAME"
        --dataset-name "$BENCH_DATASET"
        --num-prompts "$BENCH_NUM_PROMPTS"
        --request-rate "$BENCH_REQUEST_RATE"
        --random-input-len "$BENCH_RANDOM_INPUT_LEN"
        --random-output-len "$BENCH_RANDOM_OUTPUT_LEN"
        --random-range-ratio "$BENCH_RANDOM_RANGE_RATIO"
        --output-file "$result_file"
        "${extra_args[@]}"
    )

    log_bench "Running benchmark:"
    log_bench "  host=${HEAD}:${SERVER_PORT}"
    log_bench "  ${BENCH_NUM_PROMPTS} prompts, ${BENCH_REQUEST_RATE} rps"
    log_bench "  input=[${BENCH_RANDOM_INPUT_LEN}*${BENCH_RANDOM_RANGE_RATIO}, ${BENCH_RANDOM_INPUT_LEN}]"
    log_bench "  output=[${BENCH_RANDOM_OUTPUT_LEN}*${BENCH_RANDOM_RANGE_RATIO}, ${BENCH_RANDOM_OUTPUT_LEN}]"

    {
        printf '# Benchmark command\n'
        printf '# Generated: %s\n\n' "$(date)"
        printf '%s' "${cmd[0]}"
        for arg in "${cmd[@]:1}"; do printf ' \\\n    %s' "$arg"; done
        printf '\n'
    } > "$cmd_file"

    if timeout "$BENCH_TIMEOUT" "${MINICONDA}/envs/${CONDA_ENV}/bin/python" "${cmd[@]:1}" 2>&1 | tee "${result_file%.json}.log"; then
        if [ -f "$result_file" ]; then
            log_bench "Benchmark complete. Result: $result_file"
            return 0
        else
            log_bench "ERROR: Benchmark ran but no output file produced."
            printf '{"error":"no_output_file"}\n' > "$result_file"
            return 1
        fi
    else
        local exit_code=$?
        if [ "$exit_code" -eq 124 ]; then
            log_bench "ERROR: Benchmark timed out after ${BENCH_TIMEOUT}s."
            printf '{"error":"timeout"}\n' > "$result_file"
        else
            log_bench "ERROR: Benchmark failed with exit code $exit_code."
            printf '{"error":"exit_%d"}\n' "$exit_code" > "$result_file"
        fi
        return 1
    fi
}
