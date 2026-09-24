#!/bin/bash
# Use the already activated CUDA/PyTorch environment. Fail on any failed run.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$SCRIPT_DIR"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT_DIR="${OUT_DIR:-results/${STAMP}}"
mkdir -p "$OUT_DIR"
NUM_GPUS=${NUM_GPUS:-8}
WARMUP=${WARMUP:-3}
ITERS=${ITERS:-10}
TIMEOUT_SEC=${TIMEOUT_SEC:-1800}
TORCHRUN=${TORCHRUN:-torchrun}
# Select the model appropriate to each server, e.g. MODELS=gpt-oss-120b on A100.
read -r -a MODEL_LIST <<< "${MODELS:-qwen3-235b}"
read -r -a METHOD_LIST <<< "${METHODS:-peer_access nccl nccl_overlap}"
read -r -a CACHE_CONFIG_LIST <<< "${CACHE_CONFIGS:-1.0:0.25 1.0:0.5 1.0:1.0 4.0:0.5}"
read -r -a PEER_VARIANTS <<< "${PEER_ACCESS_VARIANTS:-v2}"
COMPONENT=${COMPONENT:-both}
case "$COMPONENT" in
    weights|cache|both) ;;
    *) echo "COMPONENT must be weights, cache, or both" >&2; exit 2 ;;
esac

CACHE_CSV="$OUT_DIR/cache.csv"
WEIGHTS_CSV="$OUT_DIR/weights.csv"
LOG="$OUT_DIR/run.log"
echo "sweep -> $OUT_DIR (NUM_GPUS=$NUM_GPUS WARMUP=$WARMUP ITERS=$ITERS)" | tee "$LOG"

for model in "${MODEL_LIST[@]}"; do
    for method in "${METHOD_LIST[@]}"; do
        if [ "$method" = peer_access ]; then
            variants=("${PEER_VARIANTS[@]}")
        else
            variants=("")
        fi
        for variant in "${variants[@]}"; do
            variant_args=()
            if [ -n "$variant" ]; then variant_args=(--variant "$variant"); fi
            if [ "$COMPONENT" != cache ]; then
                echo "[weights] model=$model method=$method variant=$variant" | tee -a "$LOG"
                timeout "$TIMEOUT_SEC" "$TORCHRUN" --nproc_per_node="$NUM_GPUS" bench_weights.py \
                    --model "$model" --tp-size "$NUM_GPUS" \
                    --kernel bundle --direction both --method "$method" "${variant_args[@]}" \
                    --warmup "$WARMUP" --iters "$ITERS" --out-csv "$WEIGHTS_CSV" \
                    2>&1 | tee -a "$LOG"
            fi
            if [ "$COMPONENT" != weights ]; then
                for config in "${CACHE_CONFIG_LIST[@]}"; do
                    IFS=: read -r cache_size load <<< "$config"
                    echo "[cache] model=$model cache=$cache_size load=$load method=$method variant=$variant" | tee -a "$LOG"
                    timeout "$TIMEOUT_SEC" "$TORCHRUN" --nproc_per_node="$NUM_GPUS" bench_cache.py \
                        --model "$model" --tp-size "$NUM_GPUS" \
                        --cache-size-gb "$cache_size" --load "$load" \
                        --direction both --method "$method" "${variant_args[@]}" \
                        --warmup "$WARMUP" --iters "$ITERS" --out-csv "$CACHE_CSV" \
                        2>&1 | tee -a "$LOG"
                done
            fi
        done
    done
done

echo "Done. Results: $OUT_DIR" | tee -a "$LOG"
