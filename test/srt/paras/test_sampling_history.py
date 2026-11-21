from types import SimpleNamespace

import pytest
import torch
from sglang.srt.paras.sampling_history import (
    prepare_penalty_history,
    restore_penalty_history,
)
from sglang.srt.sampling.penaltylib import (
    BatchedFrequencyPenalizer,
    BatchedPenalizerOrchestrator,
    BatchedPresencePenalizer,
)
from sglang.srt.sampling.sampling_params import SamplingParams


class Batch:
    def __init__(self, outputs, penalties, device):
        self.device = device
        self.reqs = [
            SimpleNamespace(
                output_ids=ids,
                sampling_params=SamplingParams(
                    frequency_penalty=frequency, presence_penalty=presence
                ),
            )
            for ids, (frequency, presence) in zip(outputs, penalties)
        ]
        self.sampling_info = SimpleNamespace(
            penalizer_orchestrator=BatchedPenalizerOrchestrator(
                vocab_size=8,
                batch=self,
                penalizers={BatchedFrequencyPenalizer, BatchedPresencePenalizer},
            )
        )

    def cumulate(self, tokens):
        self.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
            torch.tensor(tokens, device=self.device)
        )

    def scores(self):
        logits = torch.zeros(len(self.reqs), 8, device=self.device)
        self.sampling_info.penalizer_orchestrator.apply(logits)
        return logits


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "penalties",
    [
        [(0.25, 0), (-0.5, 0), (0, 0)],
        [(0, 0.25), (0, -0.5), (0, 0)],
        [(0.25, 0.5), (-0.5, -0.25), (0, 0)],
    ],
)
def test_restore_matches_uninterrupted_penalties(device, penalties):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    outputs = [[2, 2, 3, 4, 2], [5, 5, 2, 5, 1], [2, 4, 6, 6, 6]]
    uninterrupted = Batch(outputs, penalties, device)
    restored = Batch(outputs, penalties, device)
    for step in zip(*(ids[:-1] for ids in outputs)):
        uninterrupted.cumulate(step)

    restore_penalty_history(restored, prepare_penalty_history(restored.reqs))
    torch.testing.assert_close(
        restored.scores(), uninterrupted.scores(), rtol=0, atol=0
    )

    newest = [ids[-1] for ids in outputs]
    uninterrupted.cumulate(newest)
    restored.cumulate(newest)
    torch.testing.assert_close(
        restored.scores(), uninterrupted.scores(), rtol=0, atol=0
    )


def test_pack_skips_disabled_requests_and_handles_different_lengths():
    batch = Batch([[2, 2, 4], [1, 1, 2, 2], [5]], [(0.5, 0), (0, 0), (0, 0.5)], "cpu")
    history = prepare_penalty_history(batch.reqs)
    ids, lengths = history.result()
    assert ids.tolist() == [2, 2]
    assert lengths == [2, 0, 0]
    assert prepare_penalty_history(batch.reqs[1:]) is None
    assert prepare_penalty_history([]) is None
