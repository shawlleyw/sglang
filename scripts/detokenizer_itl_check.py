import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from typing import Dict, List

input_dir = sys.argv[1]
# output_dir = sys.argv[2]

# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

req_itls: Dict[int, List[float]] = None

with open(input_dir, "rb") as f:
    req_itls = pickle.load(f)

itls = []

for rank, itl in req_itls.items():
    itls.extend(itl)

df = pd.DataFrame(itls, columns=["itls"])

print("mean", df['itls'].mean(), "median", df['itls'].median())

p99 = df['itls'].quantile(0.99)
print(f"P99: {p99}")