#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$SCRIPT_DIR"

source /home/shaoyuw/miniconda3/etc/profile.d/conda.sh
conda activate sgl_paras
export LD_LIBRARY_PATH=/home/shaoyuw/miniconda3/envs/sgl_paras/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT_DIR="${OUT_DIR:-results/${STAMP}}"
mkdir -p "$OUT_DIR"

NUM_GPUS=${NUM_GPUS:-8}
WARMUP=${WARMUP:-3}
ITERS=${ITERS:-10}

CACHE_CSV="$OUT_DIR/cache.csv"
WEIGHTS_CSV="$OUT_DIR/weights.csv"
LOG="$OUT_DIR/run.log"

case "$NUM_GPUS" in
    4|8) MODELS=("qwen3-30b" "qwen3-235b") ;;
    *) echo "NUM_GPUS must be 4 or 8 (got $NUM_GPUS)"; exit 1 ;;
esac

echo "sweep -> $OUT_DIR (NUM_GPUS=$NUM_GPUS WARMUP=$WARMUP ITERS=$ITERS)" | tee "$LOG"

CACHE_CONFIGS=("1.0:0.25" "1.0:0.5" "1.0:1.0" "4.0:0.5")
METHODS=("peer_access" "nccl" "nccl_overlap")
PEER_ACCESS_VARIANTS=("v2" "v3")

for model in "${MODELS[@]}"; do
    for cfg in "${CACHE_CONFIGS[@]}"; do
        IFS=':' read -r CSZ LD <<< "$cfg"
        for METHOD in "${METHODS[@]}"; do
            if [ "$METHOD" = "peer_access" ]; then
                VARIANTS_TO_RUN=("${PEER_ACCESS_VARIANTS[@]}")
            else
                VARIANTS_TO_RUN=("")
            fi
            for VARIANT in "${VARIANTS_TO_RUN[@]}"; do
                if [ -n "$VARIANT" ]; then
                    VARIANT_ARGS=(--variant "$VARIANT")
                    TAG="${METHOD}(${VARIANT})"
                else
                    VARIANT_ARGS=()
                    TAG="$METHOD"
                fi
                echo | tee -a "$LOG"
                echo "[cache] model=$model cache=$CSZ load=$LD method=$TAG" | tee -a "$LOG"
                set +e
                timeout 600 torchrun --nproc_per_node="$NUM_GPUS" bench_cache.py \
                    --model "$model" --tp-size "$NUM_GPUS" \
                    --cache-size-gb "$CSZ" --load "$LD" \
                    --direction both --method "$METHOD" "${VARIANT_ARGS[@]}" \
                    --warmup "$WARMUP" --iters "$ITERS" \
                    --out-csv "$CACHE_CSV" 2>&1 | tee -a "$LOG" | grep -E "RUN|scatter:|transfer:|error"
                set -e
            done
        done
    done

    for METHOD in "${METHODS[@]}"; do
        if [ "$METHOD" = "peer_access" ]; then
            VARIANTS_TO_RUN=("${PEER_ACCESS_VARIANTS[@]}")
        else
            VARIANTS_TO_RUN=("")
        fi
        for VARIANT in "${VARIANTS_TO_RUN[@]}"; do
            if [ -n "$VARIANT" ]; then
                VARIANT_ARGS=(--variant "$VARIANT")
                TAG="${METHOD}(${VARIANT})"
            else
                VARIANT_ARGS=()
                TAG="$METHOD"
            fi
            echo | tee -a "$LOG"
            echo "[weights] model=$model method=$TAG" | tee -a "$LOG"
            set +e
            timeout 600 torchrun --nproc_per_node="$NUM_GPUS" bench_weights.py \
                --model "$model" --tp-size "$NUM_GPUS" \
                --kernel both --direction both --method "$METHOD" "${VARIANT_ARGS[@]}" \
                --warmup "$WARMUP" --iters "$ITERS" \
                --out-csv "$WEIGHTS_CSV" 2>&1 | tee -a "$LOG" | grep -E "RUN|: total|error"
            set -e
        done
    done
done

echo | tee -a "$LOG"
echo "Done. Cache CSV: $CACHE_CSV  Weights CSV: $WEIGHTS_CSV" | tee -a "$LOG"
