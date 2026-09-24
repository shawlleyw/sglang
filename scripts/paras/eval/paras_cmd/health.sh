#!/bin/bash
# Check server /health and grep the loaded model type from the log.
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "$SCRIPT_DIR/lib.sh"

http_code=$(curl -s --max-time 5 -o /dev/null -w "%{http_code}" "http://${HOST}:${PORT}/health")
echo "health: HTTP ${http_code}"
if [ "$http_code" != "200" ]; then
    echo "FAIL: server not healthy"
    exit 1
fi

model_type=$(grep "Load weight end" "$LOG_FILE" 2>/dev/null | head -1 | grep -oE '\btype=[A-Za-z0-9]+' || true)
if [ -z "$model_type" ]; then
    echo "FAIL: could not parse model type from $LOG_FILE"
    exit 1
else
    echo "$model_type"
fi
if [ "${ENABLE_PARAS:-0}" = 1 ] && [[ "$model_type" != *ParaS ]]; then
    echo "FAIL: expected a ParaS model, found $model_type" >&2
    exit 1
fi
