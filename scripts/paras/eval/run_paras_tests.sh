#!/bin/bash
# run_paras_tests — End-to-end correctness tests for ParaS (request partition,
# KV cache transfer, weight transfer). Runs CPU-only tests in-process and
# GPU tests via torchrun. Replication tests run in their own process so CUDA
# IPC handles do not get reused across mismatched buffer sizes.
#
# Usage:
#   bash scripts/paras/eval/run_paras_tests.sh           # NUM_GPUS=4 (default)
#   bash scripts/paras/eval/run_paras_tests.sh 8         # NUM_GPUS=8
#
# Common overrides (env vars):
#   NUM_GPUS                     GPU count (also accepted as positional arg). Default 4.
#   CUDA_VISIBLE_DEVICES         GPUs to use. Default 0,1,...,NUM_GPUS-1.
#   PYTEST_OPTS                  Extra pytest flags. Default "-v".
#   SGLANG_ROOT                  Repo root. Default: directory containing this script's parent's parent.
#   ONLY                         Run only one test group:
#                                  partition | kv | kv-rep | weight | gpt-oss-cuda-graph | all (default)
#
# Exits non-zero if any group fails. Each group's pass/fail is summarized at the end.

set -uo pipefail

NUM_GPUS=${1:-${NUM_GPUS:-4}}
PYTEST_OPTS=${PYTEST_OPTS:--v}
ONLY=${ONLY:-all}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "$SCRIPT_DIR/lib.sh"
SGLANG_ROOT=${SGLANG_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." &> /dev/null && pwd)}

paras_default_cvd

cd "$SGLANG_ROOT"

declare -A RESULTS

run_group() {
    local name=$1
    shift
    if [ "$ONLY" != "all" ] && [ "$ONLY" != "$name" ]; then
        return
    fi
    echo
    echo "================================================================"
    echo "[$name] $*"
    echo "================================================================"
    if "$@"; then
        RESULTS[$name]=PASS
    else
        RESULTS[$name]=FAIL
    fi
}

run_group partition \
    python -m pytest test/srt/paras/test_request_partition.py $PYTEST_OPTS

run_group kv \
    torchrun --nproc_per_node="$NUM_GPUS" \
        -m pytest test/srt/paras/test_kv_cache_transfer.py $PYTEST_OPTS

run_group kv-rep \
    torchrun --nproc_per_node="$NUM_GPUS" \
        -m pytest test/srt/paras/test_kv_cache_transfer_replication.py $PYTEST_OPTS

run_group weight \
    torchrun --nproc_per_node="$NUM_GPUS" \
        test/srt/paras/test_weight_transfer.py

run_group gpt-oss-cuda-graph \
    torchrun --nproc_per_node="$NUM_GPUS" \
        -m pytest test/srt/paras/test_paras_gpt_oss_cuda_graph.py $PYTEST_OPTS

echo
echo "================================================================"
echo "Summary (NUM_GPUS=$NUM_GPUS, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "================================================================"
exit_code=0
for name in partition kv kv-rep weight gpt-oss-cuda-graph; do
    status=${RESULTS[$name]:-SKIP}
    printf "  %-22s %s\n" "$name" "$status"
    [ "$status" = "FAIL" ] && exit_code=1
done
exit $exit_code
