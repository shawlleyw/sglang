import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from typing import Dict, List, Tuple

input_dir = sys.argv[1]

input_file = f"{input_dir}/sglang_metrics_detokenizer_throughput.pickle"
output_file = f"{input_dir}/throughput.png"

# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

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
num_tokens = sum(throughput[n//2 - 3 : n//2 + 3])
print(f"peak throughput {num_tokens / 6:.1f} tokens/sec")
# Plot throughput
plt.figure(figsize=(10, 6))
plt.plot([t - time_bins[0] for t in time_bins[:-1]], throughput, marker='', label="Throughput (tokens/sec)")
plt.xlabel("Time (seconds)")
plt.ylabel("Tokens per second")
plt.title("Throughput vs Time")
plt.legend()
plt.grid(True)
plt.savefig(output_file)