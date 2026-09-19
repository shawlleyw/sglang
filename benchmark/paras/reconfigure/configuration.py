"""Configuration validation shared by the driver, workers, and CPU tests."""

from copy import deepcopy

METHODS = ("rebuild", "host_reload", "naive_nccl", "fixed_buffer_recapture", "full")
DIRECTIONS = {"ep_to_tp": ("ep", "tp"), "tp_to_ep": ("tp", "ep")}
METHOD_TRANSPORT = {
    "rebuild": "checkpoint_load",
    "host_reload": "pinned_host_reload",
    "naive_nccl": "nccl_fresh_layer_allocations",
    "fixed_buffer_recapture": "fused_peer_access_umm",
    "full": "fused_peer_access_umm",
}


def validate_config(config):
    result = deepcopy(config)
    if result.get("world_size", 0) < 2:
        raise ValueError("world_size must be at least 2")
    if not result.get("model_path"):
        raise ValueError("model_path is required")
    if "graph_batch_sizes" in result:
        raise ValueError(
            "graph_batch_sizes is a v1 override; use production-generated graph "
            "lists via cuda_graph_max_bs/paras_tp_cuda_graph_max_bs. "
            "Reproduce v1 using its archived source."
        )
    forbidden = {
        "enable_paras_moe",
        "disable_cuda_graph",
        "tp_size",
        "dp_size",
        "ep_size",
        "enable_dp_attention",
        "paras_auto_switch",
        "cuda_graph_bs",
        "paras_tp_cuda_graph_bs",
    }
    if forbidden.intersection(result.get("server_args", {})):
        raise ValueError("layout and graph enablement are controlled by the benchmark")
    options = result.get("server_args", {})
    for field in ("cuda_graph_max_bs", "paras_tp_cuda_graph_max_bs"):
        if options.get(field) is not None and options[field] <= 0:
            raise ValueError(f"{field} must be positive")
    if options.get("max_running_requests", 2048) < result["world_size"]:
        raise ValueError(
            "max_running_requests must cover at least one request per rank"
        )
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
    """Use launcher maxima; let ServerArgs generate the actual graph lists.

    launch_common.sh caps EP by per-rank requests. ParaS defaults TP's maximum
    to EP maximum * world_size. Static TP must use that same target coverage.
    CudaGraphRunner applies production request-pool/alignment filtering at init.
    """
    args = deepcopy(config.get("server_args", {}))
    world = config["world_size"]
    ep = mode == "ep"
    ep_max = args.pop("cuda_graph_max_bs", None) or (
        args.get("max_running_requests", 2048) // world
    )
    tp_max = args.pop("paras_tp_cuda_graph_max_bs", None) or ep_max * world
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
        cuda_graph_max_bs=ep_max if ep else tp_max,
        paras_tp_cuda_graph_max_bs=tp_max if paras else None,
        disable_radix_cache=True,
        random_seed=config.get("seed", 42),
    )
    return args
