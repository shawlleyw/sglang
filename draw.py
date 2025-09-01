import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-d", "--dir", type=str, default=".")
args = parser.parse_args()

font_size = 22

with open(os.path.join(args.dir, "scheduler_stats.pkl"), 'rb') as f:
    iteration_bsz, req_start_iteration, req_end_iteration, req_timing_tracker = pickle.load(f)

niters = len(iteration_bsz)

req_to_id = {}

req_cnt = 0 
for req in req_start_iteration.keys():
    req_cnt += 1
    req_to_id[req] = req_cnt
    
req_id_start = {}
for req, start_iter in req_start_iteration.items():
    req_id_start[req_to_id[req]] = start_iter
    
req_id_end = {}
for req, end_iter in req_end_iteration.items():
    req_id_end[req_to_id[req]] = end_iter
    
req_id_tracker = {}
for req, timing in req_timing_tracker.items():
    req_id_tracker[req_to_id[req]] = timing

print("Total iterations:", niters)
print("len(req_id_start):", len(req_id_start))
print("len(req_id_end):", len(req_id_end))
print("len(req_id_tracker):", len(req_id_tracker))

# Plot
plt.figure(figsize=(12, 5))

# set all font size to 11pt
plt.rcParams.update({
    'font.size': font_size,
    'axes.titlesize': font_size,
    'axes.labelsize': font_size,
    'xtick.labelsize': font_size,
    'ytick.labelsize': font_size,
    'legend.fontsize': font_size
})

# 1. Batch size over iterations
plt.subplot(1, 2, 1)
plt.plot(range(niters), iteration_bsz, linestyle='-')
plt.title("Batch Size", fontsize=font_size)
plt.xlabel("Iteration", fontsize=font_size)
plt.ylabel("Batch Size", fontsize=font_size)
plt.yticks(range(0, max(iteration_bsz)+1, max(1, max(iteration_bsz)//5)))
plt.grid(True)

# 2. Request lifespan over iterations
plt.subplot(1, 2, 2)
for req_id in req_id_start:
    if req_id not in req_id_end or req_id not in req_id_start:
        # print(f"Warning: req_id {req_id} missing start or end iteration.")
        continue
    start = req_id_start[req_id]
    end = req_id_end[req_id]
    plt.hlines(y=req_id, xmin=start, xmax=end, linewidth=1)

plt.title("Request Lifespan", fontsize=font_size)
plt.xlabel("Iteration", fontsize=font_size)
plt.ylabel("Req ID", fontsize=font_size)

plt.tight_layout()
plt.savefig(os.path.join(args.dir, "scheduler_stats.png"), dpi=300)

queue_delay = []
ttft = []
req_elapse = []

for req_id, (admitted, batching, finish) in req_id_tracker.items():
    queue_delay.append(batching - admitted)
    ttft.append(finish - admitted)
    req_elapse.append(finish - batching)

def plot_cdf(data, ax, title):
    sorted_data = np.sort(data)
    yvals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax.plot(sorted_data, yvals, marker=".", linestyle="-")
    ax.set_title(title, fontsize=font_size)
    ax.set_xlabel("time (s)", fontsize=font_size)
    ax.set_ylabel("CDF", fontsize=font_size)
    ax.grid(True)

fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

plot_cdf(queue_delay, axs[0], "Queueing Delay")
plot_cdf(ttft, axs[1], "Time to First Token")
plot_cdf(req_elapse, axs[2], "Request Generation Time")

plt.tight_layout()
plt.savefig(os.path.join(args.dir, "req_stats_cdf.png"), dpi=300)