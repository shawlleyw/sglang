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

model_type=$(grep "Load weight end" "$LOG_FILE" 2>/dev/null | head -1 | grep -oE 'type=[A-Za-z0-9]+' || true)
if [ -z "$model_type" ]; then
    echo "WARN: could not parse model type from $LOG_FILE"
else
    echo "$model_type"
fi
