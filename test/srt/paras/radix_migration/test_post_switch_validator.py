"""T25: post-switch 4-invariant validator unit smoke tests.

Source-level AST checks so the test runs in CI environments without the
full SGLang dependency stack (safetensors, triton native, etc.).
"""
import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
MIXIN_PATH = REPO_ROOT / "python" / "sglang" / "srt" / "paras" / "scheduler_paras_mixin.py"
SERVER_ARGS_PATH = REPO_ROOT / "python" / "sglang" / "srt" / "server_args.py"


def _load_class_node(path: Path, class_name: str):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    return None


def test_validator_method_defs_present():
    """SchedulerParasMixin defines _validate_post_migration and _handle_validator_failure."""
    assert MIXIN_PATH.exists(), f"mixin file not found at {MIXIN_PATH}"
    cls = _load_class_node(MIXIN_PATH, "SchedulerParasMixin")
    assert cls is not None, "SchedulerParasMixin class not found"
    method_names = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
    assert "_validate_post_migration" in method_names, "missing _validate_post_migration"
    assert "_handle_validator_failure" in method_names, "missing _handle_validator_failure"


def test_validator_wired_in_both_switch_paths():
    """Both paras_configure_tp and paras_configure_ep must call the validator."""
    cls = _load_class_node(MIXIN_PATH, "SchedulerParasMixin")
    assert cls is not None

    def _calls_validator(fn_node):
        for sub in ast.walk(fn_node):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "_validate_post_migration"
            ):
                return True
        return False

    tp_fn = next(
        (n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "paras_configure_tp"),
        None,
    )
    ep_fn = next(
        (n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "paras_configure_ep"),
        None,
    )
    assert tp_fn is not None and ep_fn is not None
    assert _calls_validator(tp_fn), "paras_configure_tp does not call _validate_post_migration"
    assert _calls_validator(ep_fn), "paras_configure_ep does not call _validate_post_migration"


def test_strict_default_fail_in_server_args_source():
    """server_args declares paras_radix_migration_strict with default 'fail'."""
    src = SERVER_ARGS_PATH.read_text()
    assert 'paras_radix_migration_strict: str = "fail"' in src, (
        "default for paras_radix_migration_strict is not 'fail' in server_args.py"
    )


def test_failure_handler_branches_on_strict_mode():
    """_handle_validator_failure body covers raise/reset/orphan/metric for both modes."""
    cls = _load_class_node(MIXIN_PATH, "SchedulerParasMixin")
    assert cls is not None
    handler = next(
        (n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_handle_validator_failure"),
        None,
    )
    assert handler is not None
    body_src = ast.unparse(handler)
    assert "RuntimeError" in body_src, "fail branch must raise RuntimeError"
    assert "reset" in body_src, "fallback branch must call tree_cache.reset()"
    assert "tree_orphaned" in body_src, "fallback must orphan reqs"
    assert "fallbacks_total" in body_src, "fallback must bump metric"
    assert "failures_total" in body_src, "fail must bump metric"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
