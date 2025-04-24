#!/usr/bin/bash

for rate in 2000; do
    echo "Running with request rate: $rate"
    target_dir=short_req_top1_mqa/short_req_rate$rate
    mkdir -p $target_dir
    num_reqs=$(($rate * 100))
    python -m sglang.bench_serving --dataset-name random --random-input-len 10 --random-output-len 50 --num-prompts $num_reqs --metrics --disable-stream --request-rate=$rate
    mv short_req/* $target_dir
done