#!/usr/bin/bash

TOPK=1
METRICS_TARGET=/mnt/efs/baseline/scale_metrics


# QA=gqa

# for rate in 40 60 80 120 160 200 240; do
#     echo "Running with request rate: $rate"
#     target_dir=$METRICS_TARGET/top"$TOPK"_$QA/reasonable/rate"$rate"
#     mkdir -p $target_dir
#     num_reqs=$(($rate * 100))
#     python -m sglang.bench_serving --dataset-name random --random-input-len "(100, 300)" --random-output-len "(100, 500)" --num-prompts $num_reqs --metrics --disable-stream --request-rate=$rate
#     mv metrics_dir/* $target_dir
# done

# for rate in 50; do
#     echo "Running with request rate: $rate"
#     target_dir=$METRICS_TARGET/top"$TOPK"_$QA/reasonable_v2/rate"$rate"
#     mkdir -p $target_dir
#     num_reqs=$(($rate * 100))
#     python -m sglang.bench_serving --dataset-name random --random-input-len "(50, 150)" --random-output-len "(50, 250)" --num-prompts $num_reqs --metrics --disable-stream --request-rate=$rate
#     mv metrics_dir/* $target_dir
# done


QA=mqa

for rate in 120 200 240 280; do
    echo "Running with request rate: $rate"
    target_dir=$METRICS_TARGET/resonable_top"$TOPK"_$QA/rate"$rate"
    mkdir -p $target_dir
    num_reqs=$(($rate * 100))
    python -m sglang.bench_serving --dataset-name random --random-input-len "(100, 300)" --random-output-len "(100, 500)" --num-prompts $num_reqs --metrics --disable-stream --request-rate=$rate
    mv metrics_dir/* $target_dir
done

for rate in 120 200 240 280; do
    echo "Running with request rate: $rate"
    target_dir=$METRICS_TARGET/resonable_v2_top"$TOPK"_$QA/rate"$rate"
    mkdir -p $target_dir
    num_reqs=$(($rate * 100))
    python -m sglang.bench_serving --dataset-name random --random-input-len "(50, 150)" --random-output-len "(50, 250)" --num-prompts $num_reqs --metrics --disable-stream --request-rate=$rate
    mv metrics_dir/* $target_dir
done