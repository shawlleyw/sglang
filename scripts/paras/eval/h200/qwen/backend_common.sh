#!/bin/bash
# UCCL deliberately implements the existing deep_ep Python/CLI interface.
paras_h200_check_ep_provider() {
    case "${EP_PROVIDER:-uccl}" in
        uccl)
            python - <<'PY'
import torch  # Load libtorch before importing the UCCL extension.
import deep_ep
import uccl.ep

if deep_ep.Config is not uccl.ep.Config:
    raise RuntimeError(
        "EP_PROVIDER=uccl requires UCCL's ep/deep_ep_wrapper package; "
        f"the imported deep_ep at {deep_ep.__file__} is not the UCCL wrapper."
    )
print(f"UCCL EP provider verified: {deep_ep.__file__}", flush=True)
PY
            ;;
        deepep) ;;
        *) echo "Unknown EP_PROVIDER=${EP_PROVIDER}; expected uccl or deepep" >&2; return 1 ;;
    esac
}
