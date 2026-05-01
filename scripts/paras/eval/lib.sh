#!/bin/bash
# lib.sh — Shared helpers for paras eval scripts. Source from sibling scripts:
#   SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
#   source "$SCRIPT_DIR/../../lib.sh"   # e.g. from a100/qwen/foo.sh
#
# Provides:
#   paras_default_cvd       Set CUDA_VISIBLE_DEVICES=0,1,...,NUM_GPUS-1 if not already set.
#   paras_init_profile      For bench_one_batch wrappers: build LAUNCHER, PROFILE_FLAGS,
#                           LOAD_FORMAT_FLAGS arrays from ENABLE_NSYS, ENABLE_TORCH_PROFILE,
#                           LOAD_FORMAT. Also reads RUN_NAME, NSYS_OUTPUT,
#                           SGLANG_TORCH_PROFILER_DIR, TMPDIR.
#   paras_init_cuda_graph   For launch_server wrappers: build CUDA_GRAPH_FLAGS array
#                           from ENABLE_CUDA_GRAPH, CUDA_GRAPH_MAX_BS.

paras_default_cvd() {
    local default_cvd
    default_cvd=$(seq -s, 0 $((NUM_GPUS - 1)))
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$default_cvd}
}

paras_init_profile() {
    LAUNCHER=()
    PROFILE_FLAGS=()
    LOAD_FORMAT_FLAGS=()

    if [ "${ENABLE_NSYS:-0}" = "1" ]; then
        NSYS_OUTPUT=${NSYS_OUTPUT:-/tmp/nsys_${RUN_NAME}}
        mkdir -p "$(dirname "$NSYS_OUTPUT")"
        : "${TMPDIR:=$(dirname "$NSYS_OUTPUT")/nsys_tmp}"
        mkdir -p "$TMPDIR"
        export TMPDIR
        LAUNCHER=(nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node -t cuda -f true -o "$NSYS_OUTPUT")
    fi

    if [ "${ENABLE_TORCH_PROFILE:-0}" = "1" ]; then
        SGLANG_TORCH_PROFILER_DIR=${SGLANG_TORCH_PROFILER_DIR:-/tmp/torch_profile_${RUN_NAME}}
        mkdir -p "$SGLANG_TORCH_PROFILER_DIR"
        export SGLANG_TORCH_PROFILER_DIR
        PROFILE_FLAGS=(--profile --profile-filename-prefix "$RUN_NAME" --disable-cuda-graph)
    fi

    if [ -n "${LOAD_FORMAT:-}" ]; then
        LOAD_FORMAT_FLAGS=(--load-format "$LOAD_FORMAT")
    fi
}

paras_init_cuda_graph() {
    CUDA_GRAPH_FLAGS=()
    if [ "${ENABLE_CUDA_GRAPH:-1}" = "0" ]; then
        CUDA_GRAPH_FLAGS=(--disable-cuda-graph)
    elif [ -n "${CUDA_GRAPH_MAX_BS:-}" ]; then
        CUDA_GRAPH_FLAGS=(--cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS")
    fi
}
