#!/usr/bin/bash


METRICS_DIR=gqa_top2
TOPK=2
ATTN=gqa
PREFIX="$ATTN"_top"$TOPK"

for rate in 100 200 300 400 500; do
    echo "Running with request rate: $rate"
    target_dir="$PREFIX"_short/rate"$rate"
    mkdir -p $target_dir
    num_reqs=$(($rate * 100))
    python -m sglang.bench_serving --dataset-name random --random-input-len "(30, 70)" --random-output-len "(70, 130)" --num-prompts $num_reqs --metrics --disable-stream --request-rate=$rate
    mv $METRICS_DIR/* $target_dir
done

for rate in 50 100 150 200 250; do
    echo "Running with request rate: $rate"
    target_dir="$PREFIX"_medium/rate"$rate"
    mkdir -p $target_dir
    num_reqs=$(($rate * 100))
    python -m sglang.bench_serving --dataset-name random --random-input-len "(50, 150)" --random-output-len "(50, 250)" --num-prompts $num_reqs --metrics --disable-stream --request-rate=$rate
    mv $METRICS_DIR/* $target_dir
done

for rate in 25 50 75 100; do
    echo "Running with request rate: $rate"
    target_dir="$PREFIX"_reasonable/rate"$rate"
    mkdir -p $target_dir
    num_reqs=$(($rate * 100))
    python -m sglang.bench_serving --dataset-name random --random-input-len "(100, 300)" --random-output-len "(100, 500)" --num-prompts $num_reqs --metrics --disable-stream --request-rate=$rate
    mv $METRICS_DIR/* $target_dir
done

