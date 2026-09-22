#!/bin/bash
# Trigger ParaS configure_tp or configure_ep on the server.
# Usage: bash configure.sh <tp|ep>
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "$SCRIPT_DIR/lib.sh"

MODE=${1:?usage: configure.sh <tp|ep>}
case "$MODE" in
    tp|ep) ;;
    *) echo "MODE must be 'tp' or 'ep', got '$MODE'"; exit 2 ;;
esac

log_offset=$(wc -c < "$LOG_FILE") || exit $?
t0=$(date +%s.%N)
resp=$(curl --fail-with-body -sS --max-time 60 "http://${HOST}:${PORT}/paras_configure_${MODE}") || exit $?
t1=$(date +%s.%N)
elapsed_ms=$(python3 -c "print(round(($t1 - $t0) * 1000, 1))")
echo "configure_${MODE}: ${elapsed_ms}ms"
echo "$resp"
paras_cmd_verify_switch "$MODE" "$resp" "$elapsed_ms" "$log_offset"
