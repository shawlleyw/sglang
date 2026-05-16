#!/bin/bash
# launch_common.sh — Shared setup for `launch_server_*.sh` scripts under
# scripts/paras/eval/. Sourced by per-hardware launchers. Provides two
# topology-specific setup functions that resolve the parity contract from
# .skills/paras-rollout-eval/SKILL.md:
#
#   paras_launch_setup_dp_ep
#     For DP attention + EP/DeepEP experts. Handles ENABLE_PARAS toggle,
#     DeepEP env exports, PARAS_FLAGS, HYBRID_SWA_FLAGS (auto -> 1 under
#     paras, 0 under static), OVERLAP/RADIX flags, and CUDA_GRAPH_FLAGS
#     with per-rank cuda-graph max bs sizing.
#
#   paras_launch_setup_tp_tp
#     For TP attention + TP experts. Unsets DeepEP env (avoid stale
#     config), exports SYNC_TOKEN_IDS_ACROSS_TP=1 (parity with paras's
#     post-EP->TP MIN all-reduce), builds HYBRID_SWA/OVERLAP/RADIX/
#     CUDA_GRAPH flag arrays with global cuda-graph max bs sizing.
#
# CUDA_GRAPH_MAX_BS sizing rule (parity contract):
#   DP/EP: per-rank attn batch under DP attention
#          default = MAX_RUNNING_REQUESTS / NUM_GPUS
#   TP/TP: global batch dispatched across TP ranks
#          default = MAX_RUNNING_REQUESTS
# Applied uniformly to paras and static. User can override via env
# (`CUDA_GRAPH_MAX_BS=N bash launch_server_*.sh`) and that wins.
#
# After calling one of the setup functions, these bash arrays are populated
# and ready to splice into the python invocation:
#   HYBRID_SWA_FLAGS, OVERLAP_FLAGS, RADIX_FLAGS, CUDA_GRAPH_FLAGS, PARAS_FLAGS
# And these scalars are resolved:
#   HOST, PORT, NUM_GPUS, CUDA_VISIBLE_DEVICES,
#   MEM_FRACTION_STATIC, MAX_RUNNING_REQUESTS, CUDA_GRAPH_MAX_BS.
#
# Per-model launcher responsibilities (set BEFORE calling setup):
#   - MODEL_PATH default
#   - MEM_FRACTION_STATIC default (and ENABLE_PARAS branch if it differs)
#   - MAX_RUNNING_REQUESTS default (and ENABLE_PARAS branch if it differs)
#   - HYBRID_SWA default for gpt-oss (qwen has no hybrid SWA: leave unset)
#   - Any model-specific SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK
#     override (gpt-oss pins 256 in static mode; qwen lets common default
#     to 512 in static, 256 under paras)

_LAUNCH_COMMON_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "$_LAUNCH_COMMON_DIR/lib.sh"

# ---- internal helpers -------------------------------------------------------

_paras_launch_common_defaults() {
    HOST=${HOST:-0.0.0.0}
    PORT=${PORT:-30000}
    NUM_GPUS=${NUM_GPUS:-8}
    ENABLE_CUDA_GRAPH=${ENABLE_CUDA_GRAPH:-1}
    MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-8192}
    paras_default_cvd
}

_paras_launch_overlap_radix_flags() {
    DISABLE_OVERLAP=${DISABLE_OVERLAP:-0}
    DISABLE_RADIX_CACHE=${DISABLE_RADIX_CACHE:-1}

    OVERLAP_FLAGS=()
    [ "$DISABLE_OVERLAP" = "1" ] && OVERLAP_FLAGS=(--disable-overlap-schedule)

    RADIX_FLAGS=()
    [ "$DISABLE_RADIX_CACHE" = "1" ] && RADIX_FLAGS=(--disable-radix-cache)
}

# Build HYBRID_SWA_FLAGS from HYBRID_SWA env var. Values:
#   0     -> add --disable-hybrid-swa-memory
#   1     -> no flag (SWA on)
#   auto  -> resolves to 1 under paras, 0 under static
#   unset -> no flag (qwen models that have no hybrid SWA)
_paras_launch_hybrid_swa_flags() {
    local mode=${HYBRID_SWA:-}
    if [ "$mode" = "auto" ]; then
        if [ "${ENABLE_PARAS:-0}" = "1" ]; then mode=1; else mode=0; fi
    fi
    HYBRID_SWA_FLAGS=()
    [ "$mode" = "0" ] && HYBRID_SWA_FLAGS=(--disable-hybrid-swa-memory)
}

# ---- public API -------------------------------------------------------------

paras_launch_setup_dp_ep() {
    _paras_launch_common_defaults
    ENABLE_PARAS=${ENABLE_PARAS:-0}
    MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-2048}

    # Per-rank request capacity under DP attention. Single source of truth for
    # SGLANG_ATTN_MAX_BS, SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK, and
    # the cuda-graph max bs — all three size to the per-rank attn batch.
    MAX_REQ_PER_RANK=$((MAX_RUNNING_REQUESTS / NUM_GPUS))

    export SGLANG_DEEPEP_BF16_DISPATCH=${SGLANG_DEEPEP_BF16_DISPATCH:-true}
    export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-$MAX_REQ_PER_RANK}
    export NVSHMEM_QP_DEPTH=${NVSHMEM_QP_DEPTH:-2048}

    PARAS_FLAGS=()
    if [ "$ENABLE_PARAS" = "1" ]; then
        : "${SGLANG_ATTN_MAX_BS:=$MAX_REQ_PER_RANK}"
        : "${PARAS_CONFIGURE_METHOD:=peer_access}"
        : "${PARAS_KV_TRANSFER_METHOD:=peer_access}"
        : "${PARAS_DISABLE_PEER_ACCESS:=0}"
        export SGLANG_ATTN_MAX_BS PARAS_CONFIGURE_METHOD PARAS_KV_TRANSFER_METHOD PARAS_DISABLE_PEER_ACCESS
        PARAS_FLAGS=(
            --enable-paras-moe
            --paras-tp-size "$NUM_GPUS"
            --enable-nan-detection
        )
        if [ "${PARAS_AUTO_SWITCH:-1}" = "0" ]; then
            PARAS_FLAGS+=(--no-paras-auto-switch)
        fi
    fi

    : "${CUDA_GRAPH_MAX_BS:=$MAX_REQ_PER_RANK}"

    _paras_launch_hybrid_swa_flags
    _paras_launch_overlap_radix_flags
    paras_init_cuda_graph
}

paras_launch_setup_tp_tp() {
    _paras_launch_common_defaults
    MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-2048}

    # Avoid stale DeepEP config bleeding into a TP-only run.
    unset SGLANG_DEEPEP_BF16_DISPATCH SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK NVSHMEM_QP_DEPTH

    # Force MIN all-reduce on sampled token ids across TP ranks. Paras forces
    # Sampler.force_sync_token_ids=True after every EP->TP swap (see
    # paras/scheduler_paras_mixin.py paras_configure_tp) to absorb
    # non-deterministic attention / MoE / sampler kernels that would otherwise
    # diverge across ranks and deadlock at the next collective. tp-static must
    # run the same sync to be a structurally fair comparison and inherit the
    # same safety net.
    export SYNC_TOKEN_IDS_ACROSS_TP=1

    # TP/TP: single global batch dispatched across TP ranks.
    # Auto-size cuda-graph max bs to MAX_RUNNING_REQUESTS.
    : "${CUDA_GRAPH_MAX_BS:=$MAX_RUNNING_REQUESTS}"

    _paras_launch_hybrid_swa_flags
    _paras_launch_overlap_radix_flags
    paras_init_cuda_graph
}
