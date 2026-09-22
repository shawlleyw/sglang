"""Configuration validation shared by the driver, workers, and CPU tests."""

from copy import deepcopy
from pathlib import Path

HOST_METHODS = ("host_reload", "host_model_to")
METHODS = ("rebuild", *HOST_METHODS, "naive_nccl", "fixed_buffer_recapture", "full")
DIRECTIONS = {"ep_to_tp": ("ep", "tp"), "tp_to_ep": ("tp", "ep")}
METHOD_TRANSPORT = {
    "rebuild": "checkpoint_load",
    "host_reload": "pinned_host_reload",
    "host_model_to": "pinned_host_module_to_source_first",
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
    result["model_path"] = str(Path(result["model_path"]).expanduser())
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
    for field in (
        "cuda_graph_max_bs",
        "paras_tp_cuda_graph_max_bs",
        "paras_tp_max_prefill_tokens",
    ):
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
    vmm = options.get("paras_vmm_runtime_states", False)
    if type(vmm) is not bool:
        raise ValueError("paras_vmm_runtime_states must be a boolean")
    if vmm:
        # Match production's opt-in restrictions without importing CUDA/SGLang
        # in the dry-run driver. Other restrictions are checked above.
        for field in (
            "attention_backend",
            "prefill_attention_backend",
            "decode_attention_backend",
        ):
            backend = options.get("attention_backend")
            allowed = ("triton", "flashinfer") if field == "attention_backend" else (None, backend)
            if options.get(field) not in allowed:
                raise ValueError(f"paras_vmm_runtime_states requires matching Triton/FlashInfer {field}")
    if result.get("probe_decode_steps", 2) < 1:
        raise ValueError("probe_decode_steps must be positive")
    warmups = result.setdefault("warmup_switches", 1)
    if type(warmups) is not int or warmups < 0:
        raise ValueError("warmup_switches must be a nonnegative integer")
    return result


def vmm_configurations(config, selection=None):
    """Resolve CLI overrides before validation; no flag preserves config policy."""
    if selection not in (None, "off", "on", "both"):
        raise ValueError(f"Unsupported VMM selection: {selection}")
    if selection is None:
        resolved = validate_config(config)
        setting = (
            "on"
            if resolved.get("server_args", {}).get("paras_vmm_runtime_states", False)
            else "off"
        )
        return {setting: resolved}
    settings = ("off", "on") if selection == "both" else (selection,)
    variants = {}
    for setting in settings:
        variant = deepcopy(config)
        variant.setdefault("server_args", {})["paras_vmm_runtime_states"] = (
            setting == "on"
        )
        variants[setting] = validate_config(variant)
    return variants


def trial_plan(methods, directions, repetitions, settings):
    """Each VMM variant gets fresh workers; restart is one shared native reference."""
    return [
        {
            "method": method,
            "direction": direction,
            "repetition": repetition,
            "vmm": setting,
        }
        for repetition in range(1, repetitions + 1)
        for method in methods
        for direction in directions
        for setting in (("not_applicable",) if method == "rebuild" else settings)
    ]


def server_arguments(config, mode="ep", paras=True):
    """Use launcher maxima; let ServerArgs generate the actual graph lists.

    launch_common.sh caps EP by per-rank requests. ParaS defaults TP's maximum
    to EP maximum * world_size. Static TP must use that same target coverage.
    CudaGraphRunner applies production request-pool/alignment filtering at init.
    """
    args = deepcopy(config.get("server_args", {}))
    world = config["world_size"]
    ep = mode == "ep"
    if not paras:
        # Native restart has no dual-mode graph scratch. Record it as N/A in
        # the trial plan and never pass an invalid ParaS-only opt-in to Engine.
        args["paras_vmm_runtime_states"] = False
        # The native restart target has no ParaS settings. Translate its TP
        # prefill limit so scheduling and workspace reservation remain matched.
        tp_prefill = args.pop("paras_tp_max_prefill_tokens", None)
        if not ep and tp_prefill is not None:
            args["max_prefill_tokens"] = tp_prefill
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
    if not ep:
        args["moe_runner_backend"] = config.get(
            "tp_moe_runner_backend", args.get("moe_runner_backend", "triton")
        )
    return args


def configure_vocabulary_environment(mode, *, paras, environ):
    """Scope the static TP parity opt-in; do not leak it into DP constructors."""
    replicated = mode == "tp" and not paras
    for field in ("EMBEDDING", "LM_HEAD"):
        environ[f"SGLANG_QWEN3_REPLICATED_{field}"] = str(replicated).lower()
    if mode == "tp":
        environ["SYNC_TOKEN_IDS_ACROSS_TP"] = "1"


def verify_ep_provider(config, importer=None):
    """Check the real EP implementation before allocating/loading model weights."""
    if config.get("ep_provider") != "uccl":
        return None
    if importer is None:
        from importlib import import_module
        importer = import_module
    importer("torch")  # UCCL extension requires libtorch loaded first.
    deep_ep = importer("deep_ep")
    uccl_ep = importer("uccl.ep")
    if deep_ep.Config is not uccl_ep.Config:
        raise RuntimeError("This experiment requires UCCL's deep_ep compatibility wrapper")
    return {"provider": "uccl", "deep_ep_path": deep_ep.__file__, "uccl_ep_path": uccl_ep.__file__}
