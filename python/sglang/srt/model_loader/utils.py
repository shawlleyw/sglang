# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/model_loader/utils.py

"""Utilities for selecting and loading models."""
import contextlib
import logging
from typing import Tuple, Type

import torch
import transformers
from torch import nn
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from sglang.srt.configs.model_config import ModelConfig, ModelImpl

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


def resolve_transformers_arch(model_config: ModelConfig, architectures: list[str]):
    for i, arch in enumerate(architectures):
        if arch == "TransformersForCausalLM":
            continue
        auto_map: dict[str, str] = (
            getattr(model_config.hf_config, "auto_map", None) or dict()
        )
        # Make sure that config class is always initialized before model class,
        # otherwise the model class won't be able to access the config class,
        # the expected auto_map should have correct order like:
        # "auto_map": {
        #     "AutoConfig": "<your-repo-name>--<config-name>",
        #     "AutoModel": "<your-repo-name>--<config-name>",
        #     "AutoModelFor<Task>": "<your-repo-name>--<config-name>",
        # },
        auto_modules = {
            name: get_class_from_dynamic_module(
                module, model_config.model_path, revision=model_config.revision
            )
            for name, module in sorted(auto_map.items(), key=lambda x: x[0])
        }
        model_module = getattr(transformers, arch, None)
        if model_module is None:
            if "AutoModel" not in auto_map:
                raise ValueError(
                    f"Cannot find model module. '{arch}' is not a registered "
                    "model in the Transformers library (only relevant if the "
                    "model is meant to be in Transformers) and 'AutoModel' is "
                    "not present in the model config's 'auto_map' (relevant "
                    "if the model is custom)."
                )
            model_module = auto_modules["AutoModel"]
        if model_config.model_impl == ModelImpl.TRANSFORMERS:
            if not model_module.is_backend_compatible():
                raise ValueError(
                    f"The Transformers implementation of {arch} is not "
                    "compatible with SGLang."
                )
            architectures[i] = "TransformersForCausalLM"
        if model_config.model_impl == ModelImpl.AUTO:
            if not model_module.is_backend_compatible():
                raise ValueError(
                    f"{arch} has no SGlang implementation and the Transformers "
                    "implementation is not compatible with SGLang."
                )
            logger.warning(
                "%s has no SGLang implementation, falling back to Transformers "
                "implementation. Some features may not be supported and "
                "performance may not be optimal.",
                arch,
            )
            architectures[i] = "TransformersForCausalLM"
    return architectures


def get_model_architecture(model_config: ModelConfig) -> Tuple[Type[nn.Module], str]:
    from sglang.srt.models.registry import ModelRegistry

    architectures = getattr(model_config.hf_config, "architectures", [])
    # Special handling for quantized Mixtral.
    # FIXME(woosuk): This is a temporary hack.
    mixtral_supported = ["fp8", "compressed-tensors", "gptq_marlin", "awq_marlin"]

    if (
        model_config.quantization is not None
        and model_config.quantization not in mixtral_supported
        and "MixtralForCausalLM" in architectures
    ):
        architectures = ["QuantMixtralForCausalLM"]

    supported_archs = ModelRegistry.get_supported_archs()
    is_native_supported = any(arch in supported_archs for arch in architectures)

    if not is_native_supported or model_config.model_impl == ModelImpl.TRANSFORMERS:
        architectures = resolve_transformers_arch(model_config, architectures)
    model_cls, arch_name = ModelRegistry.resolve_model_cls(architectures)

    # ParaS class swap: if ParaS is enabled, use the ParaS variant
    from sglang.srt.server_args import get_global_server_args

    try:
        server_args = get_global_server_args()
        if server_args.enable_paras_moe:
            model_cls = _get_paras_model_class(model_cls)
    except ValueError:
        pass  # Server args not set yet

    return model_cls, arch_name


# ParaS model class registry
_PARAS_MODEL_REGISTRY: dict = {}


def _get_paras_model_class(base_cls: Type[nn.Module]) -> Type[nn.Module]:
    """Look up the ParaS variant of a model class."""
    if not _PARAS_MODEL_REGISTRY:
        try:
            from sglang.srt.paras.models.qwen3_moe import Qwen3MoeForCausalLMParaS
            from sglang.srt.models.qwen3_moe import Qwen3MoeForCausalLM

            _PARAS_MODEL_REGISTRY[Qwen3MoeForCausalLM] = Qwen3MoeForCausalLMParaS
        except ImportError:
            pass
        try:
            from sglang.srt.paras.models.gpt_oss import GptOssForCausalLMParaS
            from sglang.srt.models.gpt_oss import GptOssForCausalLM

            _PARAS_MODEL_REGISTRY[GptOssForCausalLM] = GptOssForCausalLMParaS
        except ImportError:
            pass

    if base_cls in _PARAS_MODEL_REGISTRY:
        return _PARAS_MODEL_REGISTRY[base_cls]

    raise ValueError(
        f"ParaS is enabled but no ParaS variant found for {base_cls.__name__}. "
        f"Create a ParaS model class in sglang.srt.paras.models."
    )


def get_architecture_class_name(model_config: ModelConfig) -> str:
    return get_model_architecture(model_config)[1]


def post_load_weights(model: nn.Module, model_config: ModelConfig):
    # Model weight loading consists of two stages:
    # 1. Initial weight loading.
    # 2. Post-processing of weights, including assigning specific member variables.
    # For `dummy_init`, only the second stage is required.
    if hasattr(model, "post_load_weights"):
        if model_config.hf_config.architectures[0] == "DeepseekV3ForCausalLMNextN":
            model.post_load_weights(is_nextn=True)
        else:
            model.post_load_weights()
