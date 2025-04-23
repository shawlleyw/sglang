from sglang.srt.managers.utils import StepMetrics 
import os
import pickle
import sys
from typing import List
import matplotlib.pyplot as plt
import numpy as np

bin_size = 100

def make_plot(bin, worker_type, save_path):
    plt.figure(figsize=(10, 8))
    plt.bar(bin.keys(), bin.values(), width=1, align='center')
    plt.xlabel(f'Batch Size (by {bin_size})')
    plt.ylabel('Frequency')
    plt.title('Batch Size Distribution')
    
    max_ticks = max(bin.keys())
    xticks = range(0, max_ticks + 1, 5)
    xtick_labels = [x * bin_size for x in xticks]
    plt.xticks(xticks, xtick_labels, rotation=90)
    plt.grid(axis='y')
    plt.tight_layout()
    # Save the plot
    plt.savefig(f"{save_path}/{worker_type}_batch_size_distribution.png")    
    
def analyze(nparray, title):
    print(f"{title}: {np.min(nparray, axis=1)}, {np.median(nparray, axis=1)}, {np.max(nparray, axis=1)}, {np.std(nparray, axis=1)}")
    
def main():
    directory = sys.argv[1]
    metrics_all_ranks: List[List[StepMetrics]] = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if not filename.endswith(".pickle") or not "rank" in filename:
            continue
        if os.path.isfile(file_path):
            fn = filename.split(".")
            if "rank" in fn[0]:
                with open(file_path, 'rb') as file:
                    metrics_all_ranks.append(pickle.load(file))
                
    nranks = len(metrics_all_ranks)
    
    if nranks == 0:
        print("No data found")
        return
                
    nsteps = len(metrics_all_ranks[0])
    
    if nsteps == 0:
        print("No steps found")
        return
    
    nlayers_per_step = len(metrics_all_ranks[0][0].attention_elapse)
    nexperts_per_rank = len(metrics_all_ranks[0][0].moe_num_tokens_per_local_expert[0])
    nexperts = nexperts_per_rank * nranks
    
    n = nlayers_per_step * nsteps
    
    batch_sizes = [] # n * nranks
    ntokens_per_expert = [] # n * nexperts
    attn_elapse = []
    moe_elapse = []
    attn_all_gather_elapse = []
    all_gather_elapse = []
    
    for i in range(nsteps):
        
        batch_sizes.append([metrics_all_ranks[k][i].batch_size for k in range(nranks)])
        
        for j in range(nlayers_per_step):
            ntokens = []
            attn = []
            moe = []
            all_gather = []
            attn_all_gather = []
            for k in range(nranks):
                metric = metrics_all_ranks[k][i]
                ntokens.extend(metric.moe_num_tokens_per_local_expert[j])
                attn.append(metric.attention_elapse[j])
                moe.append(metric.moe_elapse[j])
                attn_all_gather.append(metric.attention_elapse[j] + metric.all_gather_elapse[j])
                all_gather.append(metric.all_gather_elapse[j])
                
                
            ntokens_per_expert.append(ntokens)
            attn_elapse.append(attn)
            moe_elapse.append(moe)
            all_gather_elapse.append(all_gather)
            attn_all_gather_elapse.append(attn_all_gather)
    
    bins = {}
    
    def put_to_bins(data, bin):
        id = data // bin_size
        if id not in bin:
            bin[id] = 0
        bin[id] += 1
        
    for i in range(nsteps):
        for k in range(nranks):
            for j in range(nlayers_per_step):
                batch_size = batch_sizes[i][k]
                put_to_bins(batch_size, bins)
                
    make_plot(bins, "attn", directory)
    
    bins = {}
    for i in range(nsteps * nlayers_per_step):
        for j in range(nexperts):
            batch_size = ntokens_per_expert[i][j]
            put_to_bins(batch_size, bins)
    make_plot(bins, "expert", directory)
        
    
if __name__ == '__main__':
    main()