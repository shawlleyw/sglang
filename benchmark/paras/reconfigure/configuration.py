"""Configuration validation shared by the driver, workers, and CPU tests."""

from copy import deepcopy

METHODS = ("rebuild", "host_reload", "naive_nccl", "fixed_buffer_recapture", "full")
DIRECTIONS = {"ep_to_tp": ("ep", "tp"), "tp_to_ep": ("tp", "ep")}


def validate_config(config):
    result = deepcopy(config)
    if result.get("world_size", 0) < 2:
        raise ValueError("world_size must be at least 2")
    if not result.get("model_path"):
        raise ValueError("model_path is required")
    if not result.get("graph_batch_sizes") or any(
        x <= 0 for x in result["graph_batch_sizes"]
    ):
        raise ValueError("graph_batch_sizes must contain positive batch sizes")
    if len(set(result["graph_batch_sizes"])) != len(result["graph_batch_sizes"]):
        raise ValueError("graph_batch_sizes must be unique")
    forbidden = {
        "enable_paras_moe",
        "disable_cuda_graph",
        "tp_size",
        "dp_size",
        "ep_size",
        "enable_dp_attention",
        "paras_auto_switch",
    }
    if forbidden.intersection(result.get("server_args", {})):
        raise ValueError("layout and graph enablement are controlled by the benchmark")
    options = result.get("server_args", {})
    for field, expected in (("pp_size", 1), ("nnodes", 1), ("device", "cuda")):
        if options.get(field, expected) != expected:
            raise ValueError(f"This single-node benchmark requires {field}={expected}")
    for field in (
        "enable_torch_compile",
        "skip_tokenizer_init",
        "enable_lora",
        "enable_memory_saver",
        "enable_two_batch_overlap",
        "enable_pdmux",
    ):
        if options.get(field, False):
            raise ValueError(f"{field} is not supported by this benchmark")
    for field in ("speculative_algorithm", "quantization"):
        if options.get(field) is not None:
            raise ValueError(f"{field} is not supported by this BF16 decoder benchmark")
    if options.get("dtype", "bfloat16") not in ("bfloat16", "bf16"):
        raise ValueError("Benchmark weights must use BF16")
    if result.get("probe_decode_steps", 2) < 1:
        raise ValueError("probe_decode_steps must be positive")
    return result


def server_arguments(config, mode="ep", paras=True):
    """Identical backend/graph coverage across methods; layout is explicit."""
    args = deepcopy(config.get("server_args", {}))
    world = config["world_size"]
    ep = mode == "ep"
    args.update(
        model_path=config["model_path"],
        device="cuda",
        tp_size=world,
        dp_size=world if ep else 1,
        ep_size=world if ep else 1,
        enable_dp_attention=ep,
        enable_dp_lm_head=ep,
        moe_a2a_backend="deepep" if ep else "none",
        deepep_mode="auto",
        enable_paras_moe=paras,
        paras_tp_size=world,
        paras_auto_switch=False,
        disable_cuda_graph=False,
        cuda_graph_bs=list(config["graph_batch_sizes"]),
        cuda_graph_max_bs=max(config["graph_batch_sizes"]),
        paras_tp_cuda_graph_bs=list(config["graph_batch_sizes"]) if paras else None,
        disable_radix_cache=True,
        random_seed=config.get("seed", 42),
    )
    return args
