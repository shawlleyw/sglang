"""CPU checks for mode policy and provider identity before model allocation."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
from reconfigure.configuration import (
    configure_vocabulary_environment,
    server_arguments,
    validate_config,
    verify_ep_provider,
)


def config():
    return json.loads((BENCH / "configs/qwen3_235b_h200.json").read_text())


def test_qwen_backend_prefill_graph_contract():
    value = validate_config(config())
    for mode, paras in (("ep", False), ("tp", False), ("ep", True)):
        args = server_arguments(value, mode, paras)
        assert args["attention_backend"] == "flashinfer"
        assert args["moe_runner_backend"] == ("deep_gemm" if mode == "ep" else "triton")
        assert args["max_prefill_tokens"] == (2048 if mode == "ep" else 8192)
        assert args.get("paras_tp_max_prefill_tokens") == (8192 if paras else None)
        assert args["cuda_graph_max_bs"] == (256 if mode == "ep" else 2048)
        assert args["disable_overlap_schedule"]
        assert args["disable_radix_cache"]
        assert not args["paras_auto_switch"]
        assert not args["paras_vmm_runtime_states"]
        assert "cuda_graph_bs" not in args  # Production resolves full coverage.
        assert "paras_tp_cuda_graph_bs" not in args


def test_replication_env_does_not_leak_between_modes():
    env = {}
    configure_vocabulary_environment("tp", paras=False, environ=env)
    assert env["SGLANG_QWEN3_REPLICATED_LM_HEAD"] == "true"
    assert env["SGLANG_QWEN3_REPLICATED_EMBEDDING"] == "true"
    assert env["SYNC_TOKEN_IDS_ACROSS_TP"] == "1"
    for paras in (False, True):
        configure_vocabulary_environment("ep", paras=paras, environ=env)
        assert env["SGLANG_QWEN3_REPLICATED_LM_HEAD"] == "false"
        assert env["SGLANG_QWEN3_REPLICATED_EMBEDDING"] == "false"


def test_provider_identity_and_import_order():
    identity = object()
    modules = {
        "torch": object(),
        "deep_ep": SimpleNamespace(Config=identity, __file__="wrapper.py"),
        "uccl.ep": SimpleNamespace(Config=identity, __file__="uccl.py"),
    }
    imports = []
    def importer(name):
        imports.append(name)
        return modules[name]
    assert verify_ep_provider(config(), importer)["provider"] == "uccl"
    assert imports == ["torch", "deep_ep", "uccl.ep"]
    modules["deep_ep"].Config = object()
    with pytest.raises(RuntimeError, match="UCCL"):
        verify_ep_provider(config(), importer)


def test_flashinfer_vmm_is_preserved_on_current_runtime():
    value = config()
    value["server_args"]["paras_vmm_runtime_states"] = True
    assert validate_config(value)["server_args"]["paras_vmm_runtime_states"]
    assert server_arguments(value, "ep", True)["paras_vmm_runtime_states"]
    assert not server_arguments(value, "tp", False)["paras_vmm_runtime_states"]
