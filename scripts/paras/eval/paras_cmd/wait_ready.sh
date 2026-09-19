#!/bin/bash
# Poll $LOG_FILE for "Application startup complete". Exits 0 when ready, 1 on timeout.
#
# Override env vars:
#   LOG_FILE        Server log to poll. Default: /tmp/sglang_paras_test.log
#   TIMEOUT_TRIES   Max tries before giving up. Default: 24 (24 * 5s = 120s)
#   SLEEP_BETWEEN   Sleep seconds between tries. Default: 5
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "$SCRIPT_DIR/lib.sh"

TIMEOUT_TRIES=${TIMEOUT_TRIES:-24}
SLEEP_BETWEEN=${SLEEP_BETWEEN:-5}

for i in $(seq 1 "$TIMEOUT_TRIES"); do
    sleep "$SLEEP_BETWEEN"
    if grep -q "Application startup complete" "$LOG_FILE" 2>/dev/null; then
        echo "READY after ${i}x${SLEEP_BETWEEN}s"
        exit 0
    fi
    echo "Waiting ${i}/${TIMEOUT_TRIES}: $(tail -1 "$LOG_FILE" 2>/dev/null | cut -c1-80)"
done
echo "TIMEOUT: server not ready after ${TIMEOUT_TRIES}x${SLEEP_BETWEEN}s"
exit 1
