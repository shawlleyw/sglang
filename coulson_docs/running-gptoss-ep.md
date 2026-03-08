# Running GPT-OSS with Expert Parallelism, Profile-Driven Gating, and Fake-Prefill

Hardware: 4×A100-SXM4-80GB (SM80)
Model: `openai/gpt-oss-20b` (32 experts, top-4, hidden_size=2880, mxfp4)

---

## Prerequisites

### Environment

```bash
eval "$(conda shell.bash hook)"
conda activate /scratch1/yizhuoli/conda-envs/sglang-fp
module load cuda/12.6.3 ucx/1.16.0 gdrcopy/2.5.1-cuda
export HF_HOME=/scratch1/yizhuoli/hf_cache
export TMPDIR=/scratch1/yizhuoli/tmp
```

### Gating Profile

Profile-driven gating requires a pre-profiled parquet file. For GPT-OSS 20B:

```
gating_profiles/gating_gptoss_sharegptv3_200.parquet
```

This profile contains 201 requests, 36 layers, ~104k tokens/layer, 128 profiled experts
(projected 4:1 to match the system's 32 experts).

Required parquet columns: `rid`, `token_index`, `layer`, `expert_logical_k0..k3`.

---

## Launch Configurations

All commands are run from the repo root (`sglang-fake-prefill/`).

### 1. EP Only (baseline)

Starts the server with DeepEP expert parallelism, no fake-prefill, no profile gating.
Output is coherent (real weights, real routing).

```bash
python -m sglang.launch_server \
  --model openai/gpt-oss-20b \
  --tp 4 --ep-size 4 \
  --moe-a2a-backend deepep --deepep-mode normal \
  --host 0.0.0.0 --port 30005 \
  --mem-fraction-static 0.8
```

Notes:
- `--disable-cuda-graph` is not needed; `deepep_mode=normal` auto-disables CUDA graphs.
- Startup takes ~5–7 minutes (weight loading + warmup).

### 2. EP + Fake-Prefill

Enables fake-prefill (decode-only mode, prefill is skipped). Output will be garbled
since decode starts from uninitialized KV cache.

```bash
python -m sglang.launch_server \
  --model openai/gpt-oss-20b \
  --tp 4 --ep-size 4 \
  --moe-a2a-backend deepep --deepep-mode normal \
  --host 0.0.0.0 --port 30005 \
  --mem-fraction-static 0.8 \
  --enable-fake-prefill
```

### 3. EP + Profile-Driven Gating

Uses pre-profiled expert routing decisions instead of the model's gate network.
Requires `--disable-radix-cache` and `--chunked-prefill-size -1`.

```bash
python -m sglang.launch_server \
  --model openai/gpt-oss-20b \
  --tp 4 --ep-size 4 \
  --moe-a2a-backend deepep --deepep-mode normal \
  --host 0.0.0.0 --port 30005 \
  --mem-fraction-static 0.8 \
  --profile-driven-gate-path ./gating_profiles/gating_gptoss_sharegptv3_200.parquet \
  --disable-radix-cache \
  --chunked-prefill-size -1
```

### 4. EP + Profile-Driven Gating + Fake-Prefill

The full combination: expert parallelism with profiled routing and fake-prefill.

```bash
python -m sglang.launch_server \
  --model openai/gpt-oss-20b \
  --tp 4 --ep-size 4 \
  --moe-a2a-backend deepep --deepep-mode normal \
  --host 0.0.0.0 --port 30005 \
  --mem-fraction-static 0.8 \
  --enable-fake-prefill \
  --profile-driven-gate-path ./gating_profiles/gating_gptoss_sharegptv3_200.parquet \
  --disable-radix-cache \
  --chunked-prefill-size -1
```

### 5. EP + Profile-Driven Gating + Fake-Prefill + Dummy Weights

Skips downloading/loading real weights (uses random bf16 + zeroed mxfp4 uint8).
Fastest startup; useful for testing the serving pipeline without model access.

```bash
python -m sglang.launch_server \
  --model openai/gpt-oss-20b \
  --tp 4 --ep-size 4 \
  --moe-a2a-backend deepep --deepep-mode normal \
  --host 0.0.0.0 --port 30005 \
  --mem-fraction-static 0.8 \
  --enable-fake-prefill \
  --profile-driven-gate-path ./gating_profiles/gating_gptoss_sharegptv3_200.parquet \
  --disable-radix-cache \
  --chunked-prefill-size -1 \
  --load-format dummy
```

---

## Weight Dequantization Behavior

GPT-OSS uses mxfp4 quantized expert weights (uint8 packed). On A100 with EP, the
weights are **automatically dequantized to bf16 at model load time** — no code
changes or special flags are needed.

### How it works

In `Mxfp4FusedMoEMethod.process_weights_after_loading()` (in `mxfp4.py`), when the
MoE runner backend is `auto` (the default), both `use_flashinfer` and
`use_triton_kernels` are `False`. This triggers the `else` branch which calls
`upcast_from_mxfp()` to convert mxfp4 uint8 weights to bf16, then deletes the
scale tensors.

After load-time dequant:
- Expert weights are stored as bf16 tensors (no packed uint8).
- At inference time, the Triton MoE kernel runs pure bf16 GEMM — **no per-token
  dequantization overhead**.
- The MoE kernel config shows `E=8,N=2880` (full bf16 intermediate size), not
  `N=1440` (which would indicate packed uint8).

This applies to all launch configurations above (with or without `--load-format dummy`).

---

## Sending Requests

```bash
curl -s http://localhost:30005/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-20b",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 32,
    "temperature": 0
  }'
```

Expected output quality:
- **EP only**: Coherent responses (real weights + real routing).
- **EP + fake-prefill**: Garbled (uninitialized KV cache).
- **EP + profile gating**: Garbled (routing decisions from profile may not match input).
- **EP + profile gating + fake-prefill**: Garbled (both effects combined).
- **+ dummy weights**: Garbled (random/zero weights).

---

## Required Flags Explained

| Flag | Why |
|------|-----|
| `--tp 4 --ep-size 4` | EP requires ep-size == tp-size on single-node |
| `--moe-a2a-backend deepep` | Use DeepEP for all-to-all communication |
| `--deepep-mode normal` | High-throughput NVLink mode (auto-disables CUDA graphs) |
| `--mem-fraction-static 0.8` | Reserve 80% GPU memory for KV cache |
| `--disable-radix-cache` | Required with profile-driven gating (avoids cache interference) |
| `--chunked-prefill-size -1` | Disables chunked prefill (required with profile-driven gating) |
| `--enable-fake-prefill` | Skip real prefill; start decode from empty KV cache |
| `--profile-driven-gate-path` | Path to parquet file with pre-profiled expert routing |
| `--load-format dummy` | Skip real weight loading; use random/zero weights |

---

## Troubleshooting

### DeepEP assertion: `is_token_in_rank.size(0) == x.size(0)`

This was caused by `forward_batch.positions` being padded beyond `hidden_states.shape[0]`
during warmup. Fixed by aligning batch metadata to `num_tokens` in `gpt_oss.py`
`forward_deepep()` and `forward_normal()`.

### `Only use 20 SMs for DeepEP communication`

Performance warning. Can be tuned via `--deepep-config` with a JSON config file
specifying `num_sms`. Safe to ignore for functional testing.

### `Using default MoE kernel config`

Missing Triton autotuning config for the specific expert/hidden-size combo.
Can generate with the benchmarking script referenced in the warning. Safe to ignore.

### NCCL timeout during startup

Increase timeout with `--dist-timeout 600` (default 300s). Common when loading
large models over slow storage.
