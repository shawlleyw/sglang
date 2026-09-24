#!/bin/bash
# Wait for application startup and a healthy model worker. Exits 1 on timeout.
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
    if [ -n "${SERVER_PID:-}" ] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "FAIL: server process $SERVER_PID exited before readiness" >&2
        tail -30 "$LOG_FILE" >&2
        exit 1
    fi
    if grep -q "Scheduler hit an exception" "$LOG_FILE" 2>/dev/null; then
        echo "FAIL: scheduler failed during startup; see $LOG_FILE" >&2
        exit 1
    fi
    sleep "$SLEEP_BETWEEN"
    if grep -q "Application startup complete" "$LOG_FILE" 2>/dev/null \
        && [ "$(curl -s --max-time 5 -o /dev/null -w "%{http_code}" "http://${HOST}:${PORT}/health")" = "200" ]; then
        echo "READY after ${i}x${SLEEP_BETWEEN}s"
        exit 0
    fi
    echo "Waiting ${i}/${TIMEOUT_TRIES}: $(tail -1 "$LOG_FILE" 2>/dev/null | cut -c1-80)"
done
echo "TIMEOUT: server not ready after ${TIMEOUT_TRIES}x${SLEEP_BETWEEN}s"
exit 1
