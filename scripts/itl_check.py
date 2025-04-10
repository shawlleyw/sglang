import sys
import os
import pickle
import numpy as np
import pandas as pd

from dataclasses import asdict

from sglang.srt.managers.utils import TokenStepMetrics

directory = sys.argv[1]
output_dir = sys.argv[2]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

tokenizer_metrics = None
detokenizer_metrics = None
for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    if os.path.isfile(file_path):
        fn = filename.split(".")
        print(fn)
        if "_tokenizer" in fn[0]:
            with open(file_path, 'rb') as file:
                tokenizer_metrics = pickle.load(file)
                print(type(tokenizer_metrics))
        elif "_detokenizer" in fn[0]:
            with open(file_path, 'rb') as file:
                detokenizer_metrics = pickle.load(file)
                print(type(detokenizer_metrics))
    
print(detokenizer_metrics[0], len(detokenizer_metrics))
print(tokenizer_metrics[0], len(tokenizer_metrics))

token_df = pd.DataFrame([
    asdict(metric) for metric in tokenizer_metrics
])
detok_df = pd.DataFrame([
    asdict(metric) for metric in detokenizer_metrics
])

print(token_df['batch_size'].mean(), token_df['t_elapse'].mean(), token_df['t_elapse'].median(), token_df['t_elapse'].max(), token_df['t_wait'].mean())
print(detok_df['batch_size'].mean(), detok_df['t_elapse'].mean(), detok_df['t_elapse'].median(), detok_df['t_elapse'].max(), detok_df['t_wait'].mean(), detok_df['t_wait'].median(), detok_df['t_wait'].max())

import matplotlib.pyplot as plt

df_names = ["tokenizer", "detokenizer"]
for i, df in enumerate([token_df, detok_df]):
    for column in ['t_elapse', 't_wait']:
        col = df[column].rolling(window=100).mean()
        plt.figure(figsize=(12, 6))
        plt.plot(col, label=column, color='blue' if column == 't_elapse' else 'orange')
        plt.title(f'{df_names[i]} {column}')
        plt.xlabel('Index')
        plt.ylabel('Time (s)')
        plt.legend()
        plt.savefig(f'{output_dir}/{df_names[i]}_{column}.png')
        plt.close()
