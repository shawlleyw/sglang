"""Model presets for the ParaS peer-access kernel microbenches.

Each preset exposes the shapes the kernels actually consume: KV (num_kv_heads,
head_dim) and MoE (num_experts, hidden_size, moe_intermediate_size, num_gates).
Values match the upstream Hugging Face config.json for each model. Use
`--model custom` plus the explicit dim flags to override.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ModelConfig:
    name: str
    num_kv_heads: int
    head_dim: int
    num_experts: int
    hidden_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int = 0
    interleaved_w13: bool = False
    num_gates: int = 2
    elem_size: int = 2


PRESETS: dict[str, ModelConfig] = {
    "qwen3-30b": ModelConfig(
        name="qwen3-30b-a3b",
        num_kv_heads=4,
        head_dim=128,
        num_experts=128,
        hidden_size=2048,
        moe_intermediate_size=768,
        num_hidden_layers=48,
        num_attention_heads=32,
        num_gates=2,
        elem_size=2,
    ),
    "qwen3-235b": ModelConfig(
        name="qwen3-235b-a22b",
        num_kv_heads=4,
        head_dim=128,
        num_experts=128,
        hidden_size=4096,
        moe_intermediate_size=1536,
        num_hidden_layers=94,
        num_attention_heads=64,
        num_gates=2,
        elem_size=2,
    ),
    "gpt-oss-120b": ModelConfig(
        name="gpt-oss-120b-bf16",
        num_kv_heads=8,
        head_dim=64,
        num_experts=128,
        hidden_size=2880,
        moe_intermediate_size=2880,
        num_hidden_layers=36,
        num_attention_heads=64,
        interleaved_w13=True,
    ),
}


def resolve_model(args) -> ModelConfig:
    """Build a `ModelConfig` from argparse `args` honouring `--model custom` overrides."""
    if args.model in PRESETS:
        base = PRESETS[args.model]
        overrides = {
            field: getattr(args, field, None)
            for field in (
                "num_kv_heads",
                "head_dim",
                "num_experts",
                "hidden_size",
                "moe_intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "interleaved_w13",
            )
            if getattr(args, field, None) is not None
        }
        if overrides:
            return replace(base, name=f"{base.name}-overridden", **overrides)
        return base
    if args.model == "custom":
        required = [
            ("num_kv_heads", args.num_kv_heads),
            ("head_dim", args.head_dim),
            ("num_experts", args.num_experts),
            ("hidden_size", args.hidden_size),
            ("moe_intermediate_size", args.moe_intermediate_size),
            ("num_hidden_layers", args.num_hidden_layers),
        ]
        missing = [n for n, v in required if v is None]
        if missing:
            raise SystemExit(
                f"--model custom requires: {', '.join('--' + m.replace('_', '-') for m in missing)}"
            )
        return ModelConfig(
            name="custom",
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            num_experts=args.num_experts,
            hidden_size=args.hidden_size,
            moe_intermediate_size=args.moe_intermediate_size,
            num_hidden_layers=args.num_hidden_layers,
            num_attention_heads=args.num_attention_heads or 0,
            interleaved_w13=bool(args.interleaved_w13),
            num_gates=2,
            elem_size=2,
        )
    raise SystemExit(
        f"Unknown model {args.model!r}. Choose from: {list(PRESETS) + ['custom']}"
    )


def add_model_args(parser) -> None:
    parser.add_argument(
        "--num-attention-heads",
        type=int,
        default=None,
        help="Q head count; required for custom weight bundles",
    )
    parser.add_argument(
        "--interleaved-w13",
        action="store_true",
        default=None,
        help="Use GPT-OSS interleaved gate/up weight rows",
    )
    parser.add_argument(
        "--model",
        default="qwen3-235b",
        help=f"Model preset: {' | '.join(PRESETS)} | custom (default: qwen3-235b)",
    )
    parser.add_argument(
        "--num-kv-heads",
        type=int,
        default=None,
        help="Override num_kv_heads (required if --model custom)",
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        default=None,
        help="Override head_dim (required if --model custom)",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=None,
        help="Override total num_experts (required if --model custom)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=None,
        help="Override hidden_size (required if --model custom)",
    )
    parser.add_argument(
        "--moe-intermediate-size",
        type=int,
        default=None,
        help="Override moe_intermediate_size (required if --model custom)",
    )
    parser.add_argument(
        "--num-hidden-layers",
        type=int,
        default=None,
        help="Override num_hidden_layers (default from model)",
    )
