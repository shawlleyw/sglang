from matplotlib import pyplot as plt
import pandas as pd
import os
import sys

input_file = "profiles/sglang.csv"

df = pd.read_csv(input_file)

print(df)

output_dir = "profiles/plots/"
os.makedirs(output_dir, exist_ok=True)

columns_to_plot = ["Token Throughput", "ITL mean (ms)", "ITL median (ms)", "ITL P99 (ms)"]
x_axis = "Rate"
plt.figure()
for column in ["ITL mean (ms)", "ITL median (ms)", "ITL P99 (ms)"]:
    plt.plot(df[x_axis], df[column], marker='o', label=column)
plt.text(400, plt.ylim()[0] - 60, "(inf)", fontsize=12, color='black', ha='center')
plt.title("ITL Metrics vs Rate")
plt.xlabel(x_axis)
plt.ylabel("ITL (ms)")
plt.legend()
plt.grid(True)
output_path = os.path.join(output_dir, "ITL_metrics_vs_Rate.png")
plt.savefig(output_path)
plt.close()

for column in columns_to_plot:
    plt.figure()
    plt.plot(df[x_axis], df[column], marker='o')
    plt.title(f"{column} vs {x_axis}")
    plt.text(400, plt.ylim()[0] - 60, "(inf)", fontsize=12, color='black', ha='center')
    plt.xlabel(x_axis)
    plt.ylabel(column)
    plt.grid(True)
    output_path = os.path.join(output_dir, f"{column.replace(' ', '_')}_vs_{x_axis}.png")
    plt.savefig(output_path)
    plt.close()