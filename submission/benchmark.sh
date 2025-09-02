#!/usr/bin/bash

run_once() {
    # launch server in terminal backend 
    python -m sglang.launch_server --model-path /fsx/checkpoints/Meta-Llama-3-8B --disable-overlap-schedule --chunked-prefill-size -1 --max-prefill-tokens 4096 &

    # wait for server to be ready
    sleep 60 

    python -m sglang.bench_serving --dataset-name sharegpt --num-prompts 200  --request-rate 5 --sharegpt-context-len 1024 --output-details

    mkdir -p $1

    mv scheduler_stats.pkl $1/

    python draw.py -d $1
}

run_once cont_batch

# hack sglang/python/sglang/srt/managers/scheduler.py, modify disable_continuous_batching to be True
sed -i 's/disable_continuous_batching = False/disable_continuous_batching = True/' sglang/python/sglang/srt/managers/scheduler.py

run_once non_cont_batch