"""NCCL pack/unpack operations shared by the KV harness and CPU checks."""

import torch


def pack_gather(k, v, slots, world_size):
    """Return [destination rank, token, local head, K/V, dim]."""
    n = slots.numel()
    heads, dim = k.shape[1:]
    local_heads = max(1, heads // world_size)
    groups = heads // local_heads
    replication = world_size // groups
    values = torch.stack((k[slots.long()], v[slots.long()]), dim=2)
    packed = values.reshape(n, groups, local_heads, 2, dim).permute(1, 0, 2, 3, 4)
    return packed.repeat_interleave(replication, dim=0).contiguous()


def unpack_gather(received):
    """Source-rank-major receive chunks concatenate into the global token set."""
    return received.reshape(-1, *received.shape[2:])


def scatter_token_positions(rank, world_size, replication, tokens_per_owner):
    """Global TP token positions sent by one head replica, grouped by owner."""
    if tokens_per_owner % replication:
        raise ValueError("tokens_per_owner must be divisible by replication")
    chunk = tokens_per_owner // replication
    start = (rank % replication) * chunk
    return [
        owner * tokens_per_owner + token + 1
        for owner in range(world_size)
        for token in range(start, start + chunk)
    ]


def unpack_scatter(received, num_heads):
    """Join head groups and disjoint replica token chunks into [token, head, KV, dim]."""
    world_size, chunk, local_heads, _, dim = received.shape
    groups = num_heads // local_heads
    replication = world_size // groups
    return (
        received.reshape(groups, replication, chunk, local_heads, 2, dim)
        .permute(1, 2, 0, 3, 4, 5)
        .contiguous()
        .reshape(replication * chunk, num_heads, 2, dim)
    )
