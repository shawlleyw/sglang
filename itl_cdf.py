from argparse import ArgumentParser
import json
import numpy as np
import os
import matplotlib.pyplot as plt
parser = ArgumentParser()
parser.add_argument("-i", "--input_file", type=str, required=True)
parser.add_argument("-d", "--output_dir", type=str, default=".")
args = parser.parse_args()

font_size = 15

with open(args.input_file, 'r') as f:
    lines = f.readlines()

non_cont_itls = json.loads(lines[0])["itls"]
cont_itls = json.loads(lines[-1])["itls"]

flat_cont_itls = []
flat_non_cont_itls = []
for itl in cont_itls:
    flat_cont_itls.extend(itl)

for itl in non_cont_itls:
    flat_non_cont_itls.extend(itl)

sorted_cont_itls = sorted(flat_cont_itls)
sorted_non_cont_itls = sorted(flat_non_cont_itls)

sorted_cont_itls = np.array(sorted_cont_itls) * 1000
sorted_non_cont_itls = np.array(sorted_non_cont_itls) * 1000
# generate cdf for sorted_itls
cont_cdf = np.arange(1, len(sorted_cont_itls) + 1) / len(sorted_cont_itls)
non_cont_cdf = np.arange(1, len(sorted_non_cont_itls) + 1) / len(sorted_non_cont_itls)

plt.rcParams.update({
    'font.size': font_size,
    'axes.titlesize': font_size,
    'axes.labelsize': font_size,
    'xtick.labelsize': font_size,
    'ytick.labelsize': font_size,
    'legend.fontsize': font_size
})

plt.plot(sorted_cont_itls, cont_cdf, label="Cont")
plt.plot(sorted_non_cont_itls, non_cont_cdf, label="Non-Cont")
plt.legend(fontsize=font_size)
plt.xlabel("ITL (ms)", fontsize=font_size)
plt.ylabel("CDF", fontsize=font_size)
plt.title("ITL CDF", fontsize=font_size)

plt.savefig(os.path.join(args.output_dir, "itl_cdf.png"), dpi=300)