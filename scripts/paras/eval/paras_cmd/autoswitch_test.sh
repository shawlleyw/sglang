#!/bin/bash
# Drives the autoswitch test phases (analog of e2e_test.sh, but for the
# load-driven autoswitch policy instead of manual /paras_configure_* calls).
# Caller is responsible for kill / launch (with autoswitch enabled and
# appropriate thresholds) / final cleanup.
#
# Procedure:
#   Phase 1: pre-burst light prompt           verifies clean baseline coherence
#   Phase 2: 18 s cooldown                    let any spurious autoswitch settle
#   Phase 3: BURST_SIZE-burst (diverse)       triggers load-driven autoswitch
#   Phase 4: verify burst responses           no degenerate-attractor regex
#   Phase 5: post-burst light prompt          verify post-autoswitch state coherent
#   Phase 6: verify >=1 autoswitch event      else autoswitch never fired (test invalid)
#   Phase 7: scan log errors                  no scheduler exceptions / nvlink / etc
#
# Usage: bash autoswitch_test.sh
#
# Required at launch time (caller's responsibility):
#   --paras-auto-switch-low 2 --paras-auto-switch-high 8
#   --paras-auto-switch-window 4 --paras-auto-switch-cooldown-sec 15
# These thresholds reliably fire BOTH directions with BURST_SIZE=32:
#   light prompt (load=1) < low=2  -> low-side policy fires (EP->TP)
#   32-burst   (load=32) > high=8  -> high-side policy fires (TP->EP)
#
# Env overrides (see lib.sh):
#   BURST_SIZE          parallel concurrency for phase 3 (default 32)
#   PROMPTS_FILE        prompts file (default $SCRIPT_DIR/prompts_diverse.txt)
#   MAX_TOKENS          burst max_tokens (default 200)
#   LIGHT_MAX_TOKENS    light prompt max_tokens (default 80)
#   COOLDOWN_SEC        cooldown between phase 1 and phase 3 (default 18)

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "$SCRIPT_DIR/lib.sh"

LIGHT_MAX_TOKENS=${LIGHT_MAX_TOKENS:-80}
COOLDOWN_SEC=${COOLDOWN_SEC:-18}
LIGHT_PROMPT="List three primary colors and one example object for each."

TMP_LIGHT1=$(mktemp /tmp/autoswitch_light1.XXXXXX.json)
TMP_LIGHT2=$(mktemp /tmp/autoswitch_light2.XXXXXX.json)
TMPDIR=$(mktemp -d /tmp/autoswitch_burst_XXXXXX)
trap 'rm -f "$TMP_LIGHT1" "$TMP_LIGHT2"; rm -rf "$TMPDIR"' EXIT

send_light() {
    local out=$1
    local payload
    payload=$(paras_cmd_build_payload "$LIGHT_PROMPT" "$LIGHT_MAX_TOKENS")
    curl -s --max-time 60 "http://${HOST}:${PORT}/v1/completions" \
        -H "Content-Type: application/json" -d "$payload" > "$out"
}

verify_light() {
    local file=$1
    local label=$2
    LIGHT_FILE="$file" LABEL="$label" python3 -c '
import json, os, re, sys
label = os.environ["LABEL"]
text = json.load(open(os.environ["LIGHT_FILE"]))["choices"][0]["text"]
if re.search(r"(\b\w+\b)(\s+\1){5,}", text):
    print(f"  [{label}] DEGENERATE")
    sys.exit(1)
if len(text.strip()) < 10:
    print(f"  [{label}] EMPTY/short response (chars={len(text)})")
    sys.exit(1)
'
}

echo
echo "================ phase 1: pre-burst light prompt ================"
send_light "$TMP_LIGHT1"
paras_cmd_print_completion_file "pre-burst" "$TMP_LIGHT1"
if ! verify_light "$TMP_LIGHT1" "pre-burst"; then
    echo "FAIL: pre-burst light prompt is degenerate. Server boot may be broken."
    exit 1
fi

echo
echo "================ phase 2: ${COOLDOWN_SEC}s cooldown ================"
sleep "$COOLDOWN_SEC"

echo
echo "================ phase 3: ${BURST_SIZE}-burst (triggers autoswitch) ================"
paras_cmd_load_prompts prompts || exit $?
paras_cmd_burst_send prompts pids "$TMPDIR" "$MAX_TOKENS"
echo "  spawned ${#pids[@]} concurrent completions; load=${#pids[@]} should exceed --paras-auto-switch-high"
wait "${pids[@]}"

echo
echo "================ phase 4: verify burst responses ================"
if ! paras_cmd_burst_verify "$TMPDIR" "burst"; then
    echo "FAIL: burst responses contain degeneration."
    echo "preserving response files at ${TMPDIR} for inspection"
    trap '' EXIT
    exit 1
fi

echo
echo "================ phase 5: post-burst light prompt ================"
sleep 3
send_light "$TMP_LIGHT2"
paras_cmd_print_completion_file "post-burst" "$TMP_LIGHT2"
if ! verify_light "$TMP_LIGHT2" "post-burst"; then
    echo "FAIL: post-burst light prompt degenerate. Autoswitch corrupted post-state."
    exit 1
fi

echo
echo "================ phase 6: verify autoswitch fired ================"
if [ -f "$LOG_FILE" ] && grep -q "ParaS auto-switch policy fired" "$LOG_FILE"; then
    n_events=$(grep -c "ParaS auto-switch policy fired" "$LOG_FILE")
    echo "  PASS: ${n_events} 'ParaS auto-switch policy fired' events found in ${LOG_FILE}"
    echo "  events:"
    grep "ParaS auto-switch policy fired" "$LOG_FILE" | head -10 | sed 's/^/    /'
else
    echo "  FAIL: no 'ParaS auto-switch policy fired' events in ${LOG_FILE}"
    echo "  Autoswitch never fired. Check: (a) launch passes --paras-auto-switch-low/high,"
    echo "    (b) BURST_SIZE > --paras-auto-switch-high (default 8), (c) LOG_FILE path matches"
    echo "    where the launch script is teeing the server log."
    exit 1
fi

echo
echo "================ phase 7: scan log errors ================"
"$SCRIPT_DIR/check_log.sh" errors

echo
echo "================ autoswitch_test PASS ================"
