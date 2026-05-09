import torch

from sglang.srt.paras.cache_transfer.base import LayerCacheSpec
from sglang.srt.paras.cache_transfer.swa import SWACacheTransfer


class _Entry:
    offset_bytes = 0


class _Mgr:
    _entries = {
        "model.layers.0.kv.tp.k": _Entry(),
        "model.layers.0.kv.tp.v": _Entry(),
        "model.layers.0.kv.ep.k": _Entry(),
        "model.layers.0.kv.ep.v": _Entry(),
    }


class _SwaPool:
    k_buffer = [torch.empty(1)]
    v_buffer = [torch.empty(1)]


class _KVCache:
    layers_mapping = {0: (0, True)}
    swa_kv_pool = _SwaPool()
    head_num = 4
    head_dim = 8
    store_dtype = torch.float16
    device = "cpu"


def _spec(cap: int) -> LayerCacheSpec:
    return LayerCacheSpec(
        layer_id=0,
        kind="swa",
        tokens_cap_ep=cap,
        tokens_cap_tp=0,
        num_kv_heads=4,
        head_dim=8,
        sliding_window_size=16,
    )


def test_gather_nccl_uses_source_mapping_for_reads_and_live_mapping_for_writes(
    monkeypatch,
):
    import sglang.srt.paras.cache_transfer.swa as swa_mod

    source_mapping = torch.zeros(64, dtype=torch.int64)
    live_mapping = torch.zeros(64, dtype=torch.int64)

    source_mapping[5] = 105
    source_mapping[6] = 106
    source_mapping[20] = 920
    source_mapping[21] = 921
    source_mapping[30] = 930
    source_mapping[31] = 931

    live_mapping[5] = 805
    live_mapping[6] = 806
    live_mapping[20] = 1
    live_mapping[21] = 2
    live_mapping[30] = 4
    live_mapping[31] = 5

    backend = SWACacheTransfer.__new__(SWACacheTransfer)
    backend.method = "nccl"
    backend.kv_cache = _KVCache()
    backend.mgr = _Mgr()
    backend.group = object()
    backend.group_size = 2
    backend.num_local_tokens = 3
    backend.local_token_indices = torch.tensor([5, 6, 7], dtype=torch.int64)
    backend.global_num_tokens = [3, 4]
    backend.global_token_indices = torch.tensor(
        [20, 21, 22, 30, 31, 32, 33], dtype=torch.int64
    )
    backend._full_to_swa_mapping = live_mapping
    backend.source_full_to_swa_mapping = source_mapping

    captured = {}

    def fake_gather_nccl(
        k_buffer,
        v_buffer,
        num_local_tokens,
        num_global_tokens,
        local_token_indices,
        global_token_indices,
        global_num_tokens,
        *args,
    ):
        captured["num_local_tokens"] = num_local_tokens
        captured["num_global_tokens"] = num_global_tokens
        captured["local_token_indices"] = local_token_indices
        captured["global_token_indices"] = global_token_indices
        captured["global_num_tokens"] = global_num_tokens

    monkeypatch.setattr(swa_mod, "do_gather_one_layer_nccl", fake_gather_nccl)

    backend.gather_one_layer(_spec(cap=2))

    assert captured["num_local_tokens"] == 2
    assert captured["num_global_tokens"] == 4
    assert captured["global_num_tokens"] == [2, 2]
    assert captured["local_token_indices"].tolist() == [105, 106]
    assert captured["global_token_indices"].tolist() == [1, 2, 4, 5]


def test_scatter_nccl_uses_source_mapping_for_reads_and_live_mapping_for_writes(
    monkeypatch,
):
    import sglang.srt.paras.cache_transfer.swa as swa_mod

    source_mapping = torch.zeros(64, dtype=torch.int64)
    live_mapping = torch.zeros(64, dtype=torch.int64)

    for full_idx in [5, 6, 7, 8]:
        source_mapping[full_idx] = full_idx + 100
        live_mapping[full_idx] = full_idx + 800

    live_mapping[20] = 1
    live_mapping[21] = 2
    source_mapping[20] = 920
    source_mapping[21] = 921

    backend = SWACacheTransfer.__new__(SWACacheTransfer)
    backend.method = "nccl"
    backend.kv_cache = _KVCache()
    backend.mgr = _Mgr()
    backend.group = object()
    backend.group_size = 2
    backend.token_partition = [[0, 1], [2, 3]]
    backend.global_token_indices = torch.tensor([5, 6, 7, 8], dtype=torch.int64)
    backend.ep_dst_positions = torch.tensor([20, 21], dtype=torch.int64)
    backend._intra_rank = 0
    backend._replication_factor = 1
    backend._per_token_elems = 16
    backend._recv_full_count = 2
    backend._num_kv_heads = 4
    backend._heads_per_rank = 1
    backend._head_dim = 8
    backend._total_global_tokens = 4
    backend._reassembly_groups = 2
    backend.ep_head_num = 4
    backend._full_to_swa_mapping = live_mapping
    backend.source_full_to_swa_mapping = source_mapping

    captured = {}

    def fake_scatter_nccl(*args, **kwargs):
        captured["ep_dst_positions"] = args[9]
        captured["sorted_tp_indices"] = args[10]
        captured["recv_full_count"] = args[13]

    monkeypatch.setattr(swa_mod, "do_scatter_one_layer_nccl", fake_scatter_nccl)

    backend.scatter_one_layer(_spec(cap=10))

    assert captured["recv_full_count"] == 2
    assert captured["sorted_tp_indices"].tolist() == [105, 106, 107, 108]
    assert captured["ep_dst_positions"].tolist() == [1, 2]


def test_peer_scatter_does_not_translate_remote_destinations_with_local_mapping(
    monkeypatch,
):
    import sglang.srt.paras.cache_transfer.swa as swa_mod

    source_mapping = torch.zeros(64, dtype=torch.int64)
    for full_idx in [10, 11, 12, 13]:
        source_mapping[full_idx] = full_idx + 100

    # Simulate rank 0 after TP->EP reorchestration: only its own destination
    # slots were allocated in the local live mapping.  Remote rank slots 2 and
    # 3 would incorrectly translate to padding slot 0 if peer scatter used this
    # local mapping for remote writes.
    live_mapping = torch.zeros(64, dtype=torch.int64)
    live_mapping[1] = 1

    backend = SWACacheTransfer.__new__(SWACacheTransfer)
    backend.method = "peer_access"
    backend.kv_cache = _KVCache()
    backend.mgr = _Mgr()
    backend.group_size = 2
    backend.token_partition = [[0], [1, 2, 3]]
    backend.global_token_indices = torch.tensor(
        [10, 11, 12, 13], dtype=torch.int64
    )
    backend._num_kv_heads = 4
    backend._heads_per_rank = 1
    backend._head_dim = 8
    backend._elem_size = 2
    backend._local_buffer_ptr = 0
    backend._peer_buffer_ptrs = torch.zeros(2, dtype=torch.int64)
    backend.paras_tp_rank = 0
    backend.paras_tp_size = 2
    backend._full_to_swa_mapping = live_mapping
    backend.source_full_to_swa_mapping = source_mapping

    captured = {}

    def fake_scatter_peer_access(
        local_buffer_ptr,
        peer_buffer_ptrs,
        tp_token_positions,
        token_to_rank,
        ep_dst_pos_all,
        *args,
    ):
        captured["tp_token_positions"] = tp_token_positions
        captured["token_to_rank"] = token_to_rank
        captured["ep_dst_pos_all"] = ep_dst_pos_all

    monkeypatch.setattr(
        swa_mod, "do_scatter_one_layer_peer_access", fake_scatter_peer_access
    )

    backend.scatter_one_layer(_spec(cap=10))

    assert captured["tp_token_positions"].tolist() == [110, 111, 112, 113]
    assert captured["token_to_rank"].tolist() == [0, 1, 1, 1]
    assert captured["ep_dst_pos_all"].tolist() == [1, 1, 2, 3]


def test_source_mapping_falls_back_to_live_mapping_when_snapshot_is_absent():
    live_mapping = torch.arange(32, dtype=torch.int64) + 10

    backend = SWACacheTransfer.__new__(SWACacheTransfer)
    backend._full_to_swa_mapping = live_mapping

    result = backend._full_to_swa_source(torch.tensor([1, 4], dtype=torch.int32))

    assert result.dtype == torch.int32
    assert result.tolist() == [11, 14]
