#!/usr/bin/env python3
"""
Synthetic hybrid model fixture for T21 round-trip tests.

Provides a 6-layer config with 2 full-attention + 4 sliding-window-attention layers,
plus deterministic pattern-based KV cache generation for reproducible testing.
"""

from types import SimpleNamespace
import torch


# =========================================================================
# Constants
# =========================================================================

FULL_LAYER_IDS = [0, 1]
SWA_LAYER_IDS = [2, 3, 4, 5]
NUM_LAYERS = 6
NUM_KV_HEADS = 4
HEAD_DIM = 64
SLIDING_WINDOW = 1023  # config.sliding_window - 1


# =========================================================================
# Config factory
# =========================================================================

def make_synthetic_config():
    """Create a synthetic hybrid model config.
    
    Returns:
        SimpleNamespace with:
        - num_hidden_layers=6
        - layer_types=['full_attention','full_attention','sliding_attention','sliding_attention','sliding_attention','sliding_attention']
        - sliding_window=1024
        - num_key_value_heads=4
        - num_attention_heads=8
        - hidden_size=512
    """
    return SimpleNamespace(
        num_hidden_layers=6,
        layer_types=[
            'full_attention',
            'full_attention',
            'sliding_attention',
            'sliding_attention',
            'sliding_attention',
            'sliding_attention',
        ],
        sliding_window=1024,
        num_key_value_heads=4,
        num_attention_heads=8,
        hidden_size=512,
    )


# =========================================================================
# Pattern generator
# =========================================================================

def make_pattern(rank, layer, head, num_tokens):
    """Deterministic test data: each (rank, layer, head, token, dim) is unique.
    
    Mirrors the pattern-generator style from test_kv_cache_transfer.py.
    
    Args:
        rank: Rank index (0-3 for 4-GPU setup)
        layer: Layer index (0-5 for 6-layer model)
        head: Head index (0-3 for 4 KV heads)
        num_tokens: Number of tokens to generate
    
    Returns:
        Tensor of shape (num_tokens, HEAD_DIM) in bfloat16.
    """
    base = rank * 1000.0 + layer * 100.0 + head * 10.0
    t = torch.arange(num_tokens, dtype=torch.float32).unsqueeze(1)
    d = torch.arange(HEAD_DIM, dtype=torch.float32).unsqueeze(0) * 0.001
    return (base + t + d).to(torch.bfloat16)
