"""Restore token penalties from the output IDs already carried by ParaS."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from sglang.srt.sampling.penaltylib import (
    BatchedFrequencyPenalizer,
    BatchedPresencePenalizer,
)

_packing_worker = ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="paras-penalties"
)


def _pack_history(outputs, lengths):
    ids = np.empty(sum(lengths), dtype=np.int32)
    offset = 0
    for output, length in zip(outputs, lengths):
        if length:
            ids[offset : offset + length] = output[:length]
            offset += length
    return torch.from_numpy(ids), lengths


def prepare_penalty_history(reqs):
    # These lists are stable while the scheduler is switching layouts.
    outputs = [
        req.output_ids
        if req.sampling_params.frequency_penalty or req.sampling_params.presence_penalty
        else ()
        for req in reqs
    ]
    # prepare_for_decode counts the newest output token on its next iteration.
    lengths = [max(0, len(output) - 1) for output in outputs]
    if any(lengths):
        return _packing_worker.submit(_pack_history, outputs, lengths)
    return None


def restore_penalty_history(batch, history):
    if history is None:
        return

    ids, lengths = history.result()
    orchestrator = batch.sampling_info.penalizer_orchestrator
    device = orchestrator.device
    rows = torch.repeat_interleave(
        torch.arange(len(lengths), device=device),
        torch.tensor(lengths, device=device),
        output_size=ids.numel(),
    )
    positions = rows * orchestrator.vocab_size + ids.to(device)

    frequency = orchestrator.penalizers[BatchedFrequencyPenalizer]
    presence = orchestrator.penalizers[BatchedPresencePenalizer]
    if frequency.is_prepared():
        frequency.restore_output_history(positions, rows)
    if presence.is_prepared():
        presence.restore_output_history(positions, rows)
