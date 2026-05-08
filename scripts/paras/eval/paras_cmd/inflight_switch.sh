#!/bin/bash
# In-flight ParaS switch test: fires BURST_SIZE concurrent long completions
# (default 32 reqs at INFLIGHT_MAX_TOKENS=500), waits INFLIGHT_DELAY (default
# 4s) so all are decoding, triggers /paras_configure_<TARGET>, then waits for
# completions and verifies coherence. Stresses the gather/scatter cache
# transfer and per-Req metadata propagation across the migration.
#
# Usage: bash inflight_switch.sh <tp|ep>
#   tp: server is in EP, switch to TP   (canonical step 11)
#   ep: server is in TP, switch to EP   (canonical step 12)
#
# Exit code: 0 if all responses are clean, 1 if any degenerate.
#
# Env overrides:
#   INFLIGHT_DELAY      seconds to wait before triggering switch (default 4)
#   INFLIGHT_MAX_TOKENS max_tokens per request (default 500; long enough
#                       that decode hasn't finished by INFLIGHT_DELAY,
#                       so the switch genuinely fires mid-decode)
#   BURST_SIZE          parallel concurrency (default 32, see lib.sh)
#   PROMPTS_FILE        one-prompt-per-line file (default
#                       $SCRIPT_DIR/prompts_diverse.txt)

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "$SCRIPT_DIR/lib.sh"

TARGET=${1:?usage: inflight_switch.sh <tp|ep>}
INFLIGHT_DELAY=${INFLIGHT_DELAY:-4}
INFLIGHT_MAX_TOKENS=${INFLIGHT_MAX_TOKENS:-500}

case "$TARGET" in
    tp) TAG="EP->TP" ;;
    ep) TAG="TP->EP" ;;
    *) echo "TARGET must be 'tp' or 'ep', got '$TARGET'" >&2; exit 2 ;;
esac

paras_cmd_load_prompts prompts || exit $?

TMPDIR=$(mktemp -d /tmp/paras_inflight_XXXXXX)
trap 'rm -rf "$TMPDIR"' EXIT

echo "[${TAG}] spawning ${#prompts[@]} concurrent completions (max_tokens=${INFLIGHT_MAX_TOKENS}); switch fires after ${INFLIGHT_DELAY}s"
paras_cmd_burst_send prompts pids "$TMPDIR" "$INFLIGHT_MAX_TOKENS"

sleep "$INFLIGHT_DELAY"

t0=$(date +%s.%N)
switch_resp=$(curl -s --max-time 60 "http://${HOST}:${PORT}/paras_configure_${TARGET}")
t1=$(date +%s.%N)
elapsed_ms=$(python3 -c "print(round(($t1 - $t0) * 1000, 1))")
echo "[${TAG}] configure_${TARGET}: ${elapsed_ms}ms; resp: ${switch_resp}"

wait "${pids[@]}"

if ! paras_cmd_burst_verify "$TMPDIR" "$TAG"; then
    paras_cmd_print_completion_file "${TAG} P1 (sample)" "$TMPDIR/burst_1.json"
    exit 1
fi

paras_cmd_print_completion_file "${TAG} P1 (sample)"                  "$TMPDIR/burst_1.json"
paras_cmd_print_completion_file "${TAG} P${#prompts[@]} (sample)"     "$TMPDIR/burst_${#prompts[@]}.json"
