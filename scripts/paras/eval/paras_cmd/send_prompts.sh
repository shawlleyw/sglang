#!/bin/bash
# Stress-coverage variant of "send N prompts in the current mode": fires
# BURST_SIZE diverse prompts (default 32) in parallel and verifies each
# response is free of the canonical degenerate-attractor pattern. Used by
# canonical e2e_test.sh steps 5/8/10 to test that fresh prefill+decode
# produces clean output before, between, and after switches.
#
# Usage: bash send_prompts.sh <LABEL>
#   e.g. bash send_prompts.sh EP
#        bash send_prompts.sh TP
#        bash send_prompts.sh EP-RT
#
# Exit code: 0 if all responses are clean, 1 if any degenerate.
#
# Env overrides (see lib.sh):
#   BURST_SIZE   parallel concurrency (default 32; lower for low-mem envs)
#   PROMPTS_FILE one-prompt-per-line file (default $SCRIPT_DIR/prompts_diverse.txt)
#   MAX_TOKENS   max_tokens per request (default 200)
#   PRINT_CHARS  chars to preview from first/last response (default 300)

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "$SCRIPT_DIR/lib.sh"

LABEL=${1:?usage: send_prompts.sh <LABEL>}

paras_cmd_load_prompts prompts || exit $?

TMPDIR=$(mktemp -d /tmp/paras_send_XXXXXX)
trap 'rm -rf "$TMPDIR"' EXIT

echo "[${LABEL}] sending ${#prompts[@]} parallel completions (max_tokens=${MAX_TOKENS})..."
paras_cmd_burst_send prompts pids "$TMPDIR" "$MAX_TOKENS"
wait "${pids[@]}"

if ! paras_cmd_burst_verify "$TMPDIR" "$LABEL"; then
    paras_cmd_print_completion_file "${LABEL} P1 (sample)" "$TMPDIR/burst_1.json"
    echo "[${LABEL}] preserving response files at ${TMPDIR} for inspection"
    trap '' EXIT
    exit 1
fi

paras_cmd_print_completion_file "${LABEL} P1 (sample)"                        "$TMPDIR/burst_1.json"
paras_cmd_print_completion_file "${LABEL} P${#prompts[@]} (sample)"           "$TMPDIR/burst_${#prompts[@]}.json"
