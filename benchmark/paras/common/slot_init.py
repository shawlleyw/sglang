"""Random-slot KV cache initialization.

Mirrors production: KV tokens live at arbitrary slot positions inside a
fixed-size pool (not at slots 0..N). We pick `num_resident_tokens` random
indices from `[1, max_tokens)` (slot 0 reserved for padding) and fill them
with a bf16-EXACT pattern keyed on (rank, slot mod 16) so a single value
read by the verifier uniquely identifies its source.

Pattern v = (rank * 16) + (slot_idx % 16) + kv_offset
  - K uses kv_offset = 0    -> values in [0, 128)
  - V uses kv_offset = 128  -> values in [128, 256)
Range [0, 256) is bf16 step-1 exact.
"""

from __future__ import annotations

import torch


def random_resident_slots(num_resident: int, max_tokens: int, rank: int,
                          seed: int) -> torch.Tensor:
    """Deterministic per-rank choice of `num_resident` distinct slot indices
    drawn from `[1, max_tokens)` (slot 0 reserved as padding)."""
    if num_resident >= max_tokens:
        raise SystemExit(
            f"num_resident ({num_resident}) >= max_tokens ({max_tokens}) - "
            "reduce --load or increase --cache-size-gb."
        )
    g = torch.Generator(device="cpu").manual_seed(seed + rank)
    slots = (torch.randperm(max_tokens - 1, generator=g) + 1)[:num_resident]
    return slots.to(torch.int32)


def fill_slots_bf16(buf: torch.Tensor, slots: torch.Tensor, rank: int,
                    heads: int, head_dim: int, kv_offset: float) -> None:
    """Write `(rank*16 + slot%16 + kv_offset)` to every (head, dim) cell at
    each of the listed slot positions."""
    values = ((slots.to(torch.float32) % 16) + float(rank * 16) + kv_offset)
    values = values.to(buf.device).to(torch.bfloat16)
    fill = values.view(-1, 1, 1).expand(-1, heads, head_dim).contiguous()
    buf.index_copy_(0, slots.to(torch.long).to(buf.device), fill)


def expected_value(src_rank: int, src_slot: int, kv_offset: float) -> float:
    return float(src_rank * 16) + float(src_slot % 16) + kv_offset
