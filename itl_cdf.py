from argparse import ArgumentParser
import json
import numpy as np
import os
import matplotlib.pyplot as plt
parser = ArgumentParser()
parser.add_argument("-i", "--input_file", type=str, required=True)
parser.add_argument("-d", "--output_dir", type=str, default=".")
args = parser.parse_args()

with open(args.input_file, 'r') as f:
    lines = f.readlines()
    
itls = json.loads(lines[-1])["itls"]

flat_itls = []
for itl in itls:
    flat_itls.extend(itl)

sorted_itls = sorted(flat_itls)
# generate cdf for sorted_itls
cdf = np.arange(1, len(sorted_itls) + 1) / len(sorted_itls)

plt.plot(sorted_itls, cdf)
plt.savefig(os.path.join(args.output_dir, "itl_cdf.png"))