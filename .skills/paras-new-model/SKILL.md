---
name: paras-new-model
description: Add ParaS EP↔TP switching support to a new MoE model. Use when extending ParaS to models beyond Qwen3.
---

# Add ParaS Support to a New Model

## Overview

ParaS (Parallelism Switch) enables dynamic switching between EP (Expert Parallel) and TP (Tensor Parallel) at runtime. Adding ParaS to a new model requires creating a per-model ParaS file under `python/sglang/srt/paras/models/` — the base model file stays untouched.

## Architecture

```
paras/
├── layers/                          # Generic building blocks (DO NOT MODIFY)
│   ├── paras_moe_block.py          # ParaSMoeBlockMixin — weight redistribution
│   ├── paras_attention.py          # ParaSAttentionMixin — attention DP↔TP (optional)
│   ├── paras_decoder_layer.py      # ParaSDecoderLayerMixin — dual communicators
│   ├── paras_model.py              # ParaSModelMixin — layer-level conversion
│   └── utils.py                    # ParaSWeightBuffer, paras_load_tp_experts_weight
├── models/                          # Per-model ParaS wrappers
│   ├── qwen3_moe.py               # Reference implementation — READ THIS FIRST
│   └── <your_model>.py            # Create this
└── utils.py                        # paras_func decorator
```

## Step-by-Step

### 1. Read the reference implementation

Read `python/sglang/srt/paras/models/qwen3_moe.py` end-to-end. It has 5 classes:

| Class | Base | Mixin | Purpose |
|-------|------|-------|---------|
| `Qwen3MoeSparseMoeBlockParaS` | `Qwen3MoeSparseMoeBlock` | `ParaSMoeBlockMixin` | Dual experts (EP+TP), weight redistribution |
| `Qwen3MoeAttentionParaS` | `Qwen3MoeAttention` | `ParaSAttentionMixin` | Attention DP↔TP switching (optional) |
| `Qwen3MoeDecoderLayerParaS` | `Qwen3MoeDecoderLayer` | `ParaSDecoderLayerMixin` | Factory override + dual communicators |
| `Qwen3MoeModelParaS` | `Qwen3MoeModel` | `ParaSModelMixin` | Passes ParaS decoder layer type |
| `Qwen3MoeForCausalLMParaS` | `Qwen3MoeForCausalLM` | (none) | Entry point, load_weights override, configure methods |

### 2. Understand the base model

Before writing ParaS code, study the target model file (e.g., `models/deepseek_v2.py`):

- **MoE block class**: What's the sparse MoE class? Does it use `FusedMoE` with `w13_weight`/`w2_weight`? (Required for `ParaSMoeBlockMixin`)
- **Attention class**: Standard QKV+O (works with `ParaSAttentionMixin`) or MLA/custom (skip attention mixin, handle manually)?
- **Decoder layer**: Does it have `is_layer_sparse`, `layer_communicator`, `layer_scatter_modes`?
- **Model**: Does it accept `decoder_layer_type` parameter? (If not, you need a different injection strategy)
- **CausalLM**: What does `__init__` look like? What does `load_weights` do?

### 3. Create the per-model ParaS file

Create `python/sglang/srt/paras/models/<model_name>.py`. Follow the Qwen3 reference:

#### a. MoE Block ParaS

```python
class YourMoeBlockParaS(ParaSMoeBlockMixin, YourMoeBlock):
    def __init__(self, layer_id, config, quant_config=None, prefix=""):
        super().__init__(layer_id, config, quant_config, prefix)
        self.paras_init_moe(config, quant_config, prefix, layer_id)

    def forward(self, hidden_states, forward_batch=None, **kwargs):
        return self.paras_forward(hidden_states, forward_batch, **kwargs)
```

`paras_init_moe` expects `config` to have: `num_experts` (or `n_routed_experts`), `hidden_size`, `moe_intermediate_size`, `num_experts_per_tok`. If your model uses different attribute names, override `paras_init_moe` and map them.

#### b. Decoder Layer ParaS

```python
class YourDecoderLayerParaS(ParaSDecoderLayerMixin, YourDecoderLayer):
    def _create_sparse_moe_block(self, config, layer_id, quant_config, prefix):
        return YourMoeBlockParaS(...)

    def __init__(self, ...):
        super().__init__(...)
        # Swap attention class if using ParaSAttentionMixin
        self.self_attn.__class__ = YourAttentionParaS
        # Init dual communicators
        self.paras_init_layer(config, layer_id, self.is_layer_sparse, is_previous_layer_sparse=...)
```

**Important**: The base decoder layer MUST have a `_create_sparse_moe_block` factory method. If the base model doesn't have one, add it (minimal change — just extract the MoE block creation into a method). This avoids double-creating the MoE block.

#### c. CausalLM ParaS

Override `__init__` (skip parent to avoid double model creation) and `load_weights` (add `paras_load_tp_experts_weight` call in the expert loading loop).

### 4. Register the model class swap

In `python/sglang/srt/model_loader/utils.py`, add your model to `_get_paras_model_class`:

```python
def _get_paras_model_class(base_cls):
    if not _PARAS_MODEL_REGISTRY:
        # Existing entries...
        from sglang.srt.paras.models.qwen3_moe import Qwen3MoeForCausalLMParaS
        from sglang.srt.models.qwen3_moe import Qwen3MoeForCausalLM
        _PARAS_MODEL_REGISTRY[Qwen3MoeForCausalLM] = Qwen3MoeForCausalLMParaS

        # ADD YOUR MODEL HERE:
        from sglang.srt.paras.models.your_model import YourForCausalLMParaS
        from sglang.srt.models.your_model import YourForCausalLM
        _PARAS_MODEL_REGISTRY[YourForCausalLM] = YourForCausalLMParaS
```

### 5. Verify

```bash
# Compile check
python -m py_compile python/sglang/srt/paras/models/<your_model>.py

# Launch and test (adjust model path, GPU count)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server \
    --model <model_path> --trust-remote-code \
    --mem-fraction-static 0.6 \
    --tp-size 4 --dp-size 4 --ep-size 4 \
    --enable-dp-attention --enable-dp-lm-head \
    --moe-a2a-backend deepep --deepep-mode auto \
    --disable-cuda-graph --disable-overlap-schedule \
    --disable-radix-cache --chunked-prefill-size -1 \
    --enable-paras-moe --paras-tp-size 4

# After server is up:
curl http://localhost:30000/paras_configure_tp   # Should return 200 in <1s
```

## Key Gotchas

1. **Double creation**: If the base decoder layer doesn't have a `_create_sparse_moe_block` factory method, the ParaS subclass will create the MoE block twice (base creates one, then ParaS replaces it). For large models this wastes GPU memory during init. Always add the factory method.

2. **Attention diversity**: DeepSeek uses MLA (not QKV+O). `ParaSAttentionMixin` won't work. Either skip attention ParaS or write a custom attention mixin.

3. **Config attribute names**: Different models use different names (`num_experts` vs `n_routed_experts`, `moe_intermediate_size` vs `moe_intermediate_size`). Map them in your MoE block's `paras_init_moe` or adjust before calling.

4. **Shared experts**: Some models (DeepSeek) have shared experts fused into the MoE block. The weight redistribution logic in `ParaSMoeBlockMixin` handles `w13_weight` and `w2_weight` which include fused shared experts. Verify this works with your model's expert layout.

5. **load_weights duplication**: The CausalLM `load_weights` is duplicated because the `paras_load_tp_experts_weight` call is deeply nested in the weight iteration loop. Copy from the base model's `load_weights` and add the ParaS hook.

6. **`is_previous_layer_sparse`**: This is model-specific. Qwen3-MoE has all layers sparse (always `True`). Other models (DeepSeek) have mixed dense/sparse layers — compute this correctly.
