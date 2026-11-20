"""Plan migration of live radix-owned KV without transferring cache-only slots."""

from dataclasses import dataclass

import torch


@dataclass
class LiveKVRows:
    """Request rows indexed into a compact transfer vector."""

    inverse: torch.Tensor
    lengths: list[int]
    prefixes: list[int]

    @classmethod
    def compact(cls, reqs, flat_slots):
        unique, inverse = torch.unique(flat_slots, return_inverse=True)
        rows = cls(
            inverse,
            [req.seqlen - 1 for req in reqs],
            [len(req.prefix_indices) for req in reqs],
        )
        return unique, rows

    def install(self, reqs, req_indices, destination, req_pool):
        flat = destination[self.inverse]
        for req, req_index, row in zip(reqs, req_indices, flat.split(self.lengths)):
            req.req_pool_idx = req_index
            req_pool.write((req_index, slice(0, row.numel())), row)

    def restore_prefixes(self, reqs, req_pool, cache):
        active = [(req, size) for req, size in zip(reqs, self.prefixes) if size]
        if not active:
            return
        slots = [
            req_pool.req_to_token[req.req_pool_idx, :size].to(torch.int64)
            for req, size in active
        ]
        keys = torch.cat(slots).tolist()
        offset = 0
        for (req, size), values in zip(active, slots):
            cache.adopt_live_prefix(req, values, keys[offset : offset + size])
            offset += size


def gather_live_rows(local_rows, local_count, request_counts, token_counts, group):
    """All-gather per-source inverse maps without conflating slots across ranks."""
    if group.world_size == 1:
        return [local_count], local_rows.prefixes, local_rows.inverse

    device = local_rows.inverse.device
    width = max(1 + r + n for r, n in zip(request_counts, token_counts))
    sizes = torch.tensor(
        [local_count, *local_rows.prefixes], dtype=torch.int32, device=device
    )
    packet = torch.zeros(width, dtype=torch.int32, device=device)
    packet[: sizes.numel()] = sizes
    packet[sizes.numel() : sizes.numel() + local_rows.inverse.numel()] = (
        local_rows.inverse
    )
    gathered = torch.empty(group.world_size * width, dtype=packet.dtype, device=device)
    group.all_gather_into_tensor(gathered, packet)
    packets = gathered.view(group.world_size, width).unbind()
    metadata = torch.cat([p[: 1 + n] for p, n in zip(packets, request_counts)]).tolist()
    counts, prefixes, maps = [], [], []
    offset = cursor = 0
    for packet, req_count, token_count in zip(packets, request_counts, token_counts):
        end = cursor + req_count + 1
        count = metadata[cursor]
        counts.append(count)
        prefixes.extend(metadata[cursor + 1 : end])
        maps.append(
            packet[req_count + 1 : req_count + 1 + token_count].to(torch.int64) + offset
        )
        offset += count
        cursor = end
    return counts, prefixes, torch.cat(maps)
