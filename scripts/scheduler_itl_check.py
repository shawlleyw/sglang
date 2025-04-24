import sys
import os
import pandas as pd
import pickle

from typing import Dict, List

input_dir = sys.argv[1]
n_rank = int(sys.argv[2])

df_all = pd.DataFrame()

for rank in range(n_rank):
    fn = f"{input_dir}/sglang_metrics_rank{rank}_scheduler_itl.pickle"

    req_itls: Dict[int, List[float]] = None

    with open(fn, "rb") as f:
        req_itls = pickle.load(f)

    itls = []

    for rank, itl in req_itls.items():
        itls.extend(itl)

    df = pd.DataFrame(itls, columns=["itls"])
    
    df_all = pd.concat([df_all, df], ignore_index=True)

    print("======= rank", rank, "=======")
    print("mean", df['itls'].mean(), "median", df['itls'].median())

    p99 = df['itls'].quantile(0.99)
    print(f"P99: {p99}")
    
print("======= all ranks =======")
print("mean", df_all['itls'].mean(), "median", df_all['itls'].median())
p99 = df_all['itls'].quantile(0.99)
print(f"P99: {p99}")