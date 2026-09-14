# Unified memory validation

Validated on 2026-09-14 using Qwen3-30B-A3B BF16, four A100-SXM4-80GB GPUs,
and the `sgl_paras` conda environment. Model downloaded to
`/data/shaoyuw/models/Qwen3-30B-A3B`; existing DeepEP installation was usable.

## Results

- Canonical `.skills/paras-test-manual-switch` procedure: **PASS**.
- CUDA graphs: EP through batch 512, TP through batch 2048; both captures completed.
- Five bursts of 32 requests: EP, TP, EP round trip, in-flight EP→TP,
  and in-flight TP→EP. All 160 responses passed the degeneration checks.
- HTTP configure times: TP 91.0 ms, EP 93.3 ms; in-flight TP 136.2 ms,
  in-flight EP 119.5 ms (all below the 2,500 ms criterion).
- In-flight cache transfer: gather at most 4.46 ms, scatter at most 15.19 ms.
- Final unchanged error-log checker: no unexpected errors.
- Focused unit tests: **11 passed**, including exact attention reconstruction
  after destroying full-weight copies, replicated KV heads, workspace/weight
  address collisions, page rounding and transfer geometry, workspace guard
  regions, and numerical agreement with original Triton allocations. EP
  numerical test crosses three chunks, including a short final chunk.
- Compilation and `git diff --check` passed.

The broader existing `test_umm_heterogeneous.py` and
`test_drain_overlap_pipeline.py` tests produced six failures and four passes.
The same six failures were reproduced with those modules loaded from
unchanged HEAD: five tests assume an older KV layout; one scheduler fixture
omits `running_batch`. See `baseline_tests.log`. They are independent of this
implementation and were not modified to change their outcomes.

## Reproduction

```bash
source /data/shaoyuw/env/activate-sgl-paras.sh
export PYTHONPATH=/data/shaoyuw/sglang/python
export TMPDIR=/tmp CUDA_VISIBLE_DEVICES=0,1,2,3
export SGLANG_ENABLE_JIT_DEEPGEMM=0
export NVSHMEM_IBGDA_NIC_HANDLER=cpu_host_memory
export ENABLE_PARAS=1 PARAS_AUTO_SWITCH=0 NUM_GPUS=4 MEM_FRACTION_STATIC=0.7

bash scripts/paras/eval/a100/qwen/launch_server_dp_ep.sh \
  --moe-runner-backend triton --attention-backend flashinfer \
  > /tmp/sglang_paras_test.log 2>&1 &

MODEL_NAME=Qwen3-30B-A3B TIMEOUT_TRIES=60 \
  bash scripts/paras/eval/paras_cmd/e2e_test.sh

bash scripts/paras/eval/paras_cmd/kill.sh
```

The explicit NIC handler avoids this host's GPU NIC-mapping fallback warnings.
The installed NVSHMEM version supports `cpu_host_memory`; `cpu` requires
GDRCopy, which is unavailable here. This setting follows the installed
library's diagnostic and NVIDIA's [implementation](https://github.com/NVIDIA/nvshmem/blob/devel/src/modules/transport/ibgda/ibgda.cpp).
The test's error filtering was unchanged.

```bash
CUDA_VISIBLE_DEVICES=4 pytest -q \
  test/srt/paras/test_unified_workspace_layout.py \
  test/srt/paras/test_unified_workspace_views.py \
  test/srt/paras/test_unified_attention_transfer.py \
  test/srt/paras/test_unified_triton_workspace.py
```

## Measured allocation plan

52.865405 GiB combined region per GPU. Other allocations remain outside it.

| Region | EP (GiB) | TP (GiB) |
| --- | ---: | ---: |
| Front workspace / transfer gap | 1.059124 | 0 |
| Expert + attention weights | 15.187500 | 13.921875 |
| KV capacity, including sentinel rows | 36.618713 | 36.618759 |
| Tail workspace / transfer gap | 0 | 2.324749 |

EP and TP usable token capacities are 399,973 and 1,599,897 respectively.
Differences between totals and rounded table entries are alignment seams.
DeepEP transport, permutation metadata, and returned outputs are external;
the managed workspace covers internal Triton GEMM/activation intermediates.
H200/235B and DeepGEMM runtime validation were not performed on these A100s.

Logs: `server.log`, `manual_switch.log`, `unit_tests.log`, `baseline_tests.log`.
