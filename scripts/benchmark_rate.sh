#!/bin/bash

# make sure the server is running

INPUT_LEN=300
OUTPUT_LEN=500
NUM_PROMPTS=8000

for rate in 10 20 40 80 160 320 640
do
    echo "Running benchmark with request rate: $rate"
    # run benchmark script
    python -m sglang.bench_serving \
        --dataset-name random \
        --random-input-len $INPUT_LEN \
        --random-output-len $OUTPUT_LEN \
        --num-prompts $NUM_PROMPTS \
        --metrics --disable-stream \
        --request-rate $rate

    # plot throughput figures
    python scripts/throughput_analyze.py \
        torch_profile/sglang_metrics_detokenizer_throughput.pickle \
        profiles/plots/time-series-throughput_rate-$rate.png
done
