# TP8 / EP8 validation with four KV heads

2026-09-14: Qwen3-30B-A3B BF16 on eight A100-SXM4-80GB GPUs, Triton MoE in
both modes, FlashInfer attention, CUDA graphs, static memory fraction 0.7,
2,048 total running requests. Automatic switching disabled.

## Results

The canonical manual-switch procedure passed without production-code changes.
All five bursts of 32 requests passed: EP8, TP8, EP8 after round trip,
in-flight EP8→TP8, and in-flight TP8→EP8. The unchanged error-log check passed.
CUDA graph capture completed on all eight ranks: EP batches through 256,
TP batches through 2,048.

| Configure request | HTTP latency |
| --- | ---: |
| EP8→TP8, idle | 76.8 ms |
| TP8→EP8, idle | 74.2 ms |
| EP8→TP8, in flight | 95.0 ms |
| TP8→EP8, in flight | 107.8 ms |

In-flight KV gather took at most 3.75 ms per rank; scatter at most 17.79 ms.
These are measured test latencies, not a general performance guarantee.

The new `test_unified_attention_transfer_tp8.py` also passed on all eight
ranks using real CUDA IPC and the model's attention dimensions. It checks
each TP Q/K/V/O slice exactly, destroys all full EP weight copies, and
poisons the redundant K/V shards on odd ranks with NaNs. Full Q/K/V/O
reconstruction remains bit-exact on every GPU. This verifies that K/V use
only ranks 0,2,4,6 while Q/O still use all eight ranks.

## Replication and transfer direction

| TP ranks | KV head |
| --- | ---: |
| 0, 1 | 0 |
| 2, 3 | 1 |
| 4, 5 | 2 |
| 6, 7 | 3 |

EP→TP attention **weights** need only local slicing: EP attention weights
are replicated on every GPU. TP→EP reconstructs full K/V weights from four
representative ranks and Q/O weights from eight distinct shards. No
four-rank NCCL subgroup is required by the direct peer-read implementation.

Live **cache** differs: EP has different requests on each of eight ranks.
EP→TP gathers their token rows and replicates each head to its TP rank pair;
TP→EP redistributes token rows and reassembles the four heads at the target
EP rank. Both directions were exercised with in-flight requests.

## Per-GPU allocation

```
EP8: [front 0.771992][weights 8.437500][KV 43.653809]
TP8: [weights 6.984375][KV 43.653831][tail 2.225117]
     <------------ total 52.863330 GiB ------------>
```

Front/tail regions are shared MoE scratch and transfer headroom. Rounding
seams are omitted. Usable EP/TP token capacities: 476,815 / 1,907,264.
Each TP GPU holds one KV head, so TP cache bytes per token are one quarter
of EP's, not one eighth. Other allocations remain outside this buffer.

## Reproduction

```bash
source /data/shaoyuw/env/activate-sgl-paras.sh
export PYTHONPATH=/data/shaoyuw/sglang/python TMPDIR=/tmp
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_ENABLE_JIT_DEEPGEMM=0 NVSHMEM_IBGDA_NIC_HANDLER=cpu_host_memory
export ENABLE_PARAS=1 PARAS_AUTO_SWITCH=0 NUM_GPUS=8 MEM_FRACTION_STATIC=0.7

bash scripts/paras/eval/a100/qwen/launch_server_dp_ep.sh \
  --moe-runner-backend triton --attention-backend flashinfer \
  > /tmp/sglang_paras_tp8_test.log 2>&1 &

MODEL_NAME=Qwen3-30B-A3B TIMEOUT_TRIES=60 LOG_FILE=/tmp/sglang_paras_tp8_test.log \
  bash scripts/paras/eval/paras_cmd/e2e_test.sh

# Preserve logs before cleanup.
LOG_FILE=/tmp/sglang_paras_tp8_test.log bash scripts/paras/eval/paras_cmd/kill.sh

torchrun --standalone --nproc-per-node=8 -m pytest -q \
  test/srt/paras/test_unified_attention_transfer_tp8.py
```

Evidence: `server.log`, `manual_switch.log`, `attention_ipc.log`.
