import sys
import os
import pandas as pd
import numpy as np
from scipy.stats import expon
from scipy.interpolate import interp1d

def extend_token_distribution_with_exponential_fit(original_token_dist, num_experts):
    """
    Fit an exponential distribution to the original token distribution,
    generate a new distribution with num_experts using interpolated CDF spacing,
    and recover the original ordering pattern.
    
    Args:
        original_token_dist (list or np.array): Token counts per expert.
        num_experts (int): Desired number of experts in the extended distribution.

    Returns:
        extended_token_dist (np.array): New token distribution with `num_experts` experts.
    """
    original_token_dist = np.array(original_token_dist)
    n_orig = len(original_token_dist)

    # Fit exponential (with location fixed at 0)
    loc, scale = expon.fit(original_token_dist, floc=0)

    # Compute empirical CDF values of the sorted original tokens
    sorted_orig = np.sort(original_token_dist)
    cdf_vals = expon.cdf(sorted_orig, loc=loc, scale=scale)

    # Interpolate these CDF values to the desired number of experts
    interp_fn = interp1d(np.linspace(0, 1, n_orig), cdf_vals, kind='linear')
    new_cdf_vals = interp_fn(np.linspace(0, 1, num_experts))

    # Inverse-transform to get new token values
    extended_tokens = expon.ppf(new_cdf_vals, loc=loc, scale=scale).astype(int)

    # Recover rank pattern from original
    rank_order = np.argsort(-original_token_dist)  # descending
    repeated_order = np.resize(rank_order, num_experts)

    # Reorder the extended tokens based on repeated rank pattern
    sorted_extended = np.zeros_like(extended_tokens)
    sorted_extended[repeated_order.argsort()] = np.sort(extended_tokens)[::-1]

    return sorted_extended

# ---- Example usage and plotting ----

# # Given 8-expert distribution
# original_token_dist = [17214, 15855, 4810, 3401, 2928, 19882, 17007, 7097]
# num_experts = 16

# # Generate extended token distribution
# extended_token_dist = extend_token_distribution_with_exponential_fit(original_token_dist, num_experts)

# # Plot CDFs
# cdf_orig = np.arange(1, len(original_token_dist) + 1) / len(original_token_dist)
# cdf_extended = np.arange(1, num_experts + 1) / num_experts

# plt.figure(figsize=(8, 5))
# plt.step(np.sort(original_token_dist), cdf_orig, where='post', label='Original (8 Experts)', linewidth=2)
# plt.step(np.sort(extended_token_dist), cdf_extended, where='post', label='Extended (16 Experts)', linestyle='--', linewidth=2)
# plt.title("Empirical CDF Comparison")
# plt.xlabel("Token Count")
# plt.ylabel("Cumulative Probability")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # Print output
# print("Original token distribution:\n", original_token_dist)
# print("\nExtended token distribution (16 experts):\n", extended_token_dist)


def main():
    # /home/hogura/sglang/weights/dolly_decoding_expert_count.csv
    input_dir = sys.argv[1]
    
    # /home/hogura/sglang/weights/dolly_decoding_expert_count_16.csv
    output_dir = sys.argv[2]
    num_experts = 16
    
    df = pd.read_csv(input_dir)

    # append 8 extra columns to df
    for i in range(8, 16):
        df[f"expert_{i}"] = 0
    
    for i in range(len(df)):
        original_token_dist = df.iloc[i, -16: -8].values.astype("int32")
        extended_token_dist = extend_token_distribution_with_exponential_fit(original_token_dist, num_experts)
        df.iloc[i, -16:] = extended_token_dist
        
    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_dir, index=False)
    print(df)

if __name__ == "__main__":
    main()