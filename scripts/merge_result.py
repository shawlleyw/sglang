import pandas as pd
import os
import sys
from typing import Dict, List, Tuple
import pickle

input_dir = sys.argv[1]
output_file = sys.argv[2]

# Read all ITL & throughput pickles in the input directory
results = []
for dirs in os.listdir(input_dir):
    if dirs.endswith(".csv"):
        continue
    rate = int(dirs[4:])
    file_dir = os.path.join(input_dir, dirs)
    
    input_file = f"{file_dir}/sglang_metrics_detokenizer_itl.pickle"

    # analyze latency
    req_itls: Dict[int, List[float]] = None

    with open(input_file, "rb") as f:
        req_itls = pickle.load(f)

    itls = []
    for rank, itl in req_itls.items():
        itls.extend(itl)
    df = pd.DataFrame(itls, columns=["itls"])
    p99 = df['itls'].quantile(0.99)

    # analyze tput
    
    input_file = f"{file_dir}/sglang_metrics_detokenizer_throughput.pickle"
    with open(input_file, "rb") as f:
        reqs: List[Tuple[float, int]] = pickle.load(f)

    # Calculate throughput
    gap = 10
    timestamps, tokens = zip(*reqs)
    start_time = int(timestamps[0])
    end_time = int(timestamps[-1])
    time_bins = range(start_time, end_time + gap, gap)  # +gap to include the last second


    throughput = []
    for t in time_bins[:-1]:
        tokens_in_bin = [tokens[i] for i in range(len(timestamps)) if t <= timestamps[i] < t + gap]
        throughput.append(sum(tokens_in_bin) / gap)

    n = len(time_bins)
    num_tokens = sum(throughput[n//2 - 3 : n//2 + 3]) / 6

    result = {
        "rate": rate,
        "mean_latency (ms)": int(df['itls'].mean() * 1000),
        "median_latency (ms)": int(df['itls'].median() * 1000),
        "P99_latency (ms)": int(p99 * 1000),
        "token_tput": int(num_tokens),
    }

    results.append(result)

df = pd.DataFrame(results)
df.sort_values(by="rate", inplace=True)

df.to_csv(output_file, index=False, sep=',')