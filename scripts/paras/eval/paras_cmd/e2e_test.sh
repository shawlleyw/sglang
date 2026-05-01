#!/bin/bash
# End-to-end ParaS server test: drives steps 3-13 of paras-test-qwen3 / paras-test-gpt-oss.
# The CALLER is responsible for steps 1 (pkill old server), 2 (launch via
# launch_server_dp_ep.sh ENABLE_PARAS=1 ...), and 14 (final cleanup).
#
# Usage: bash e2e_test.sh
#
# Required env: MODEL_NAME (so request bodies route correctly).
#   Qwen3-30B-A3B → bash e2e_test.sh
#   gpt-oss-120b BF16 → MODEL_NAME=gpt-oss-120b-BF16-unsloth bash e2e_test.sh
#
# Common overrides: HOST, PORT, LOG_FILE, INFLIGHT_DELAY.
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "$SCRIPT_DIR/lib.sh"

run_step() {
    local n=$1; shift
    local desc=$1; shift
    echo
    echo "================ step ${n}: ${desc} ================"
    "$@"
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "step ${n} FAILED (rc=$rc)"
        return $rc
    fi
}

run_step 3  "wait for server ready"            bash "$SCRIPT_DIR/wait_ready.sh"
run_step 4  "verify health + model type"       bash "$SCRIPT_DIR/health.sh"
run_step 5  "EP requests (pre-switch)"         bash "$SCRIPT_DIR/send_prompts.sh" "EP"
run_step 6  "trigger EP→TP switch"             bash "$SCRIPT_DIR/configure.sh"   "tp"
run_step 7  "check timing"                     bash "$SCRIPT_DIR/check_log.sh"   "timing"
run_step 8  "TP requests (after switch)"       bash "$SCRIPT_DIR/send_prompts.sh" "TP"
run_step 9  "trigger TP→EP switch"             bash "$SCRIPT_DIR/configure.sh"   "ep"
run_step 10 "EP requests (round-trip)"         bash "$SCRIPT_DIR/send_prompts.sh" "EP-RT"
run_step 11 "in-flight EP→TP switch"           bash "$SCRIPT_DIR/inflight_switch.sh" "tp"
run_step 12 "in-flight TP→EP switch"           bash "$SCRIPT_DIR/inflight_switch.sh" "ep"
run_step 13 "verify no errors"                 bash "$SCRIPT_DIR/check_log.sh"   "errors"

echo
echo "================ e2e PASS ================"
