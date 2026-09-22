# Qwen3-235B BF16 on 8 H200 GPUs

The Qwen launchers default to `$HOME/models/Qwen3-235B-A22B-Instruct-2507`,
FlashInfer attention, DeepGEMM MoE for DP/EP, and Triton MoE for TP/TP.
ParaS uses DeepGEMM in EP mode and Triton in TP mode. Override `MODEL_PATH`,
`ATTENTION_BACKEND=triton|flashinfer`, and `MOE_RUNNER_BACKEND=auto|triton|deep_gemm`.

Before these defaults were made explicit, the scripts left attention and MoE
selection to SGLang: Qwen3 MoE on H200 normally selected **FA3** attention;
BF16 DP/EP selected **DeepGEMM** when installed and JIT enabled; TP/TP selected
**Triton** MoE. FlashInfer and Triton support ParaS attention state switching.

| Topology | Default BF16 MoE | Communication |
| --- | --- | --- |
| `tp_tp` | Triton | NCCL TP reductions |
| `dp_ep` static | DeepGEMM (`deep_gemm`) | UCCL EP |
| `dp_ep`, `ENABLE_PARAS=1` | EP: DeepGEMM; TP: Triton | UCCL EP in EP mode; NCCL in TP mode |
| `dp_tp` / `tp_ep` | Triton | NCCL reductions |

BF16 DeepGEMM in this branch requires EP > 1, the DeepEP-compatible dispatcher,
SiLU activation without expert bias, and `SGLANG_ENABLE_JIT_DEEPGEMM=true`.
`MOE_RUNNER_BACKEND=triton` explicitly disables BF16 DeepGEMM selection.
Selecting `deep_gemm` does not enable a BF16 TP DeepGEMM path: TP remains Triton.

## UCCL EP provider

Build the latest [UCCL EP](https://github.com/uccl-project/uccl/tree/main/ep),
then install its compatibility wrapper from the same checkout:

```bash
python -m pip install --no-deps /path/to/uccl/ep/deep_ep_wrapper
```

`dp_ep` launchers default to `EP_PROVIDER=uccl` and verify that imported
`deep_ep.Config` is `uccl.ep.Config` before loading model weights. They retain
`--moe-a2a-backend deepep` because this branch uses that selector for the
compatible API; the implementation is UCCL. `DEEPEP_MODE=auto` selects normal
dispatch for prefill and low latency dispatch for decode. UCCL low latency
requires usable RDMA infrastructure even on one NVLink node. `DEEPEP_MODE=normal`
uses normal dispatch and disables CUDA graphs in this branch. An explicit
`EP_PROVIDER=deepep` permits the original DeepEP provider for comparisons.

## Manual switch validation

The correctness runner launches and cleans up its own server process group,
retaining server and test logs under `/tmp/sglang-h200-*` (override `RUN_DIR`):

```bash
source .venv/bin/activate
bash scripts/paras/eval/h200/qwen/test_correctness.sh tp
bash scripts/paras/eval/h200/qwen/test_correctness.sh ep
bash scripts/paras/eval/h200/qwen/test_correctness.sh paras
# Alternate supported kernels:
ATTENTION_BACKEND=triton MOE_RUNNER_BACKEND=triton \
    bash scripts/paras/eval/h200/qwen/test_correctness.sh paras
```

The runner defaults to FlashInfer attention and automatic MoE selection.
`CONFIGURE_MAX_MS=2500` is the manual-switch latency threshold; override it
explicitly when evaluating a model-specific threshold. Each switch must return
the expected response and add a fresh successful scheduler timing record;
capacity-precheck rejection fails the test. In-flight tests also require at least
one completion request still outstanding at switch time; tune `INFLIGHT_DELAY`
if all requests finish before the switch.

Use `.skills/paras-test-manual-switch/SKILL.md` and the shared `paras_cmd` helpers.
Static TP and EP use `send_prompts.sh` and log checks; only ParaS exposes the
configure endpoints needed by `e2e_test.sh`.

```bash
export MODEL_NAME=Qwen3-235B-A22B-Instruct-2507
export LOG_FILE=/tmp/sglang_paras_qwen235b.log
export TIMEOUT_TRIES=180 SLEEP_BETWEEN=10
ENABLE_PARAS=1 PARAS_AUTO_SWITCH=0 NUM_GPUS=8 MEM_FRACTION_STATIC=0.85 \
    bash scripts/paras/eval/h200/qwen/launch_server_dp_ep.sh >"$LOG_FILE" 2>&1 &
bash scripts/paras/eval/paras_cmd/e2e_test.sh
```

The full procedure checks EP, TP, round-trip EP, and both in-flight switching
directions with 32 diverse requests per phase, then checks server errors.

## Installed environment and JIT cache

On this server, run `source .local/activate-h200.sh` before the commands above.
The persistent DeepGEMM cache is `.local/cache/deep_gemm`; activation sets both
`SGLANG_DG_CACHE_DIR` and `DG_JIT_CACHE_DIR` to that directory. Override
`SGLANG_DG_CACHE_DIR` before sourcing activation to use another location.
Exact installed versions and build instructions are in `.local/INSTALLATION.md`
and `.local/requirements-installed.txt`.

## Matched memory measurements

The launch defaults now match the GPT-OSS memory comparison's prefill budgets:
DP/EP uses `MAX_PREFILL_TOKENS=2048` per rank; static TP uses 8192;
ParaS uses `PARAS_TP_MAX_PREFILL_TOKENS=8192` while in TP mode.
`PARAS_VMM_RUNTIME_STATES=1` adds `--paras-vmm-runtime-states` when
`ENABLE_PARAS=1`; its default is off. `NVSHMEM_DISABLE_NCCL=1` matches the
reference communication configuration (UCCL remains the EP provider).

Static TP defaults to full replicated vocabulary embedding and LM head, as
ParaS retains during TP execution. These are `[151936, 4096]` BF16 matrices
for this checkpoint; attention and experts still use TP8. Set
`SGLANG_QWEN3_REPLICATED_EMBEDDING=false` and
`SGLANG_QWEN3_REPLICATED_LM_HEAD=false` to restore native sharded vocabulary
storage. The model implementation keeps its original sharded default outside
this launcher. Full LM-head replication skips the vocabulary all-gather and
requires unquantized TP attention.

For memory comparisons, explicitly set `DISABLE_OVERLAP=1`,
`DISABLE_RADIX_CACHE=1`, `PARAS_AUTO_SWITCH=0`, and
`PARAS_POST_SWITCH_RAMP_ITERS=0`. Native static workspace reservation requires
overlap to be disabled. Use matched 2048 global requests, an explicit EP graph
list through 256, and a TP list through 2048. Save effective launches and
per-rank workspace, vocabulary, KV-capacity and physical-memory observations;
launcher defaults alone do not prove measurement parity.
