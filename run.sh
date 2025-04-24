#!/usr/bin/bash

TOPK=2

for rate in 100 200 300 400 500; do
    echo "Running with request rate: $rate"
    target_dir=short_top"$TOPK"_mqa/rate"$rate"
    mkdir -p $target_dir
    num_reqs=$(($rate * 100))
    python -m sglang.bench_serving --dataset-name random --random-input-len "(30, 70)" --random-output-len "(70, 130)" --num-prompts $num_reqs --metrics --disable-stream --request-rate=$rate
    mv metrics_dir/* $target_dir
done

for rate in 40 60 80 120 160 200; do
    echo "Running with request rate: $rate"
    target_dir=resonable_top"$TOPK"_mqa/rate"$rate"
    mkdir -p $target_dir
    num_reqs=$(($rate * 100))
    python -m sglang.bench_serving --dataset-name random --random-input-len "(100, 300)" --random-output-len "(100, 500)" --num-prompts $num_reqs --metrics --disable-stream --request-rate=$rate
    mv metrics_dir/* $target_dir
done

for rate in 50 100 200 300 400; do
    echo "Running with request rate: $rate"
    target_dir=resonable_v2_top"$TOPK"_mqa/rate"$rate"
    mkdir -p $target_dir
    num_reqs=$(($rate * 100))
    python -m sglang.bench_serving --dataset-name random --random-input-len "(50, 150)" --random-output-len "(50, 250)" --num-prompts $num_reqs --metrics --disable-stream --request-rate=$rate
    mv metrics_dir/* $target_dir
done