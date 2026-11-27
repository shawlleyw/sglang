from types import SimpleNamespace

import torch
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.paras.cache_transfer.prefix import (
    LiveKVRows,
    gather_live_rows,
)
from sglang.srt.paras.prefix_cache import ParaSPrefixCache


def test_transfer_keeps_only_live_slots_and_frees_shared_prefix_after_last_owner():
    # Slot 90 is an evictable continuation on the source and has no live owner.
    source = ParaSPrefixCache(None, None, page_size=1)
    source.insert(RadixKey([7, 8, 9]), torch.tensor([40, 60, 90]))
    reqs = [
        SimpleNamespace(
            seqlen=4,
            prefix_indices=torch.tensor(prefix),
            origin_input_ids=tokens,
            output_ids=[100],
        )
        for prefix, tokens in [([40, 60], [7, 8, 11]), ([40], [7, 12, 13])]
    ]
    source_slots, rows = LiveKVRows.compact(
        reqs, torch.tensor([40, 60, 12, 40, 15, 16])
    )
    assert source_slots.tolist() == [12, 15, 16, 40, 60]
    assert source.total_size() == 3

    pool = ReqToTokenPool(2, 4, "cpu", enable_memory_saver=False)
    allocator = TokenToKVPoolAllocator(8, torch.bfloat16, "cpu", None, False)
    cache = ParaSPrefixCache(pool, allocator, page_size=1)
    rows.install(reqs, pool.alloc(2), allocator.alloc(len(source_slots)), pool)
    rows.restore_prefixes(reqs, pool, cache)

    assert pool.req_to_token[:, :3].tolist() == [[4, 5, 1], [4, 2, 3]]
    assert cache.protected_size() == 2
    assert cache.match_prefix(RadixKey([7, 8, 9])).device_indices.numel() == 0

    cache.cache_finished_req(reqs[0])
    assert cache.protected_size() == 1
    assert allocator.available_size() == 5
    cache.cache_finished_req(reqs[1])
    assert cache.total_size() == 0
    assert allocator.free_pages.sort().values.tolist() == list(range(1, 9))
    assert pool.available_size() == 2


def test_gather_offsets_slot_numbers_from_separate_ep_ranks():
    req = SimpleNamespace(seqlen=4, prefix_indices=torch.tensor([5, 7]))
    local_slots, local_rows = LiveKVRows.compact(
        [req, req], torch.tensor([5, 7, 13, 5, 7, 8])
    )

    class TwoEPWorkers:
        world_size = 2

        def all_gather_into_tensor(self, output, packet):
            gathered = output.view(2, -1)
            gathered[0].copy_(packet)
            # Rank 1 has one request using the same physical slot numbers.
            gathered[1].zero_()
            gathered[1, :5] = torch.tensor([3, 2, 0, 1, 2])

    counts, prefixes, inverse = gather_live_rows(
        local_rows, local_slots.numel(), [2, 1], [6, 3], TwoEPWorkers()
    )
    assert counts == [4, 3]
    assert prefixes == [2, 2, 2]
    assert inverse.tolist() == [0, 1, 3, 0, 1, 2, 4, 5, 6]
