#!/bin/bash
# Inspect the server log for ParaS-related events.
# Usage: bash check_log.sh <timing|errors|cuda_graph|all>
#   timing      Grep transfer_weights / configure TP timing lines (skill step 7)
#   errors      Grep error/exception lines, excluding known benign warnings (skill step 13)
#   cuda_graph  Grep dual-capture lines (CUDA graph variant)
#   all         Run all three checks
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "$SCRIPT_DIR/lib.sh"

WHAT=${1:-all}
if [ ! -r "$LOG_FILE" ]; then
    echo "FAIL: server log is not readable: $LOG_FILE" >&2
    exit 1
fi

check_timing() {
    echo "=== timing ==="
    grep -E "Time taken to configure (TP|EP)|transfer_weights" "$LOG_FILE" || return 1
    python3 - "$LOG_FILE" "${CONFIGURE_MAX_MS:-2500}" <<'PY'
import math, re, sys
with open(sys.argv[1]) as stream:
    matches = re.findall(r'Time taken to configure (?:TP|EP): ([\d.eE+-]+) ms', stream.read())
if not matches:
    sys.exit('FAIL: no completed ParaS switch timing found')
limit = float(sys.argv[2])
if any(not math.isfinite(float(ms)) or not 0 <= float(ms) < limit for ms in matches):
    sys.exit(f'FAIL: configure timing exceeds {limit} ms')
PY
}

check_errors() {
    echo "=== errors ==="
    out=$(grep -iE "error|exception" "$LOG_FILE" \
        | grep -v "input_text=" \
        | grep -viE "import error|Config file|opentelemetry|WARNING|warn_only|warnings\.warn|UserWarning" || true)
    if [ -z "$out" ]; then
        echo "OK: no unexpected errors"
    else
        echo "$out"
        return 1
    fi
}

check_cuda_graph() {
    echo "=== cuda_graph dual capture ==="
    grep -E "ParaS: dual capture complete|saving EP graphs|capturing TP graphs|TP capture done|pools_differ" "$LOG_FILE" \
        || echo "(no dual-capture lines — eager mode?)"
}

case "$WHAT" in
    timing)     check_timing ;;
    errors)     check_errors ;;
    cuda_graph) check_cuda_graph ;;
    all)
        check_timing || exit $?
        echo
        check_cuda_graph
        echo
        check_errors
        ;;
    *) echo "WHAT must be timing|errors|cuda_graph|all, got '$WHAT'"; exit 2 ;;
esac
