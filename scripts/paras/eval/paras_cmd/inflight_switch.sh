#!/bin/bash
# In-flight ParaS switch test: kicks off two background completion requests
# in the CURRENT mode, waits INFLIGHT_DELAY (default 4s) so they enter decode,
# triggers /paras_configure_<TARGET>, then waits for the requests and prints
# their outputs. Both responses must be coherent — degeneration after the
# switch indicates KV-cache transfer / attention-backend state corruption.
#
# Usage: bash inflight_switch.sh <tp|ep>
#   tp: server is in EP, prompts hash-table / photosynthesis (skill step 11)
#   ep: server is in TP, prompts prime-numbers / fibonacci      (skill step 12)
#
# Override INFLIGHT_DELAY if you need to start the switch later (e.g., 8s).
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "$SCRIPT_DIR/lib.sh"

TARGET=${1:?usage: inflight_switch.sh <tp|ep>}
INFLIGHT_DELAY=${INFLIGHT_DELAY:-4}
INFLIGHT_MAX_TOKENS=${INFLIGHT_MAX_TOKENS:-500}

case "$TARGET" in
    tp)
        TAG="EP→TP"
        PROMPT_A="Explain how a hash table works, including collision resolution strategies like chaining and open addressing."
        PROMPT_B="Describe the process of photosynthesis in plants, including the light-dependent and light-independent reactions."
        ;;
    ep)
        TAG="TP→EP"
        PROMPT_A="List the first 10 prime numbers and explain why each is prime."
        PROMPT_B="Write a recursive Fibonacci function in Python with memoization."
        ;;
    *) echo "TARGET must be 'tp' or 'ep', got '$TARGET'"; exit 2 ;;
esac

OUT_A=$(mktemp /tmp/inflight_a.XXXXXX.json)
OUT_B=$(mktemp /tmp/inflight_b.XXXXXX.json)
trap 'rm -f "$OUT_A" "$OUT_B"' EXIT

PAYLOAD_A=$(paras_cmd_build_payload "$PROMPT_A" "$INFLIGHT_MAX_TOKENS")
PAYLOAD_B=$(paras_cmd_build_payload "$PROMPT_B" "$INFLIGHT_MAX_TOKENS")

curl -s --max-time 120 "http://${HOST}:${PORT}/v1/completions" \
    -H "Content-Type: application/json" -d "$PAYLOAD_A" > "$OUT_A" &
PID_A=$!
curl -s --max-time 120 "http://${HOST}:${PORT}/v1/completions" \
    -H "Content-Type: application/json" -d "$PAYLOAD_B" > "$OUT_B" &
PID_B=$!
echo "spawned PID_A=$PID_A PID_B=$PID_B; sleeping ${INFLIGHT_DELAY}s before switch"

sleep "$INFLIGHT_DELAY"

echo "switching to $TARGET..."
t0=$(date +%s.%N)
switch_resp=$(curl -s --max-time 60 "http://${HOST}:${PORT}/paras_configure_${TARGET}")
t1=$(date +%s.%N)
elapsed_ms=$(python3 -c "print(round(($t1 - $t0) * 1000, 1))")
echo "configure_${TARGET}: ${elapsed_ms}ms"
echo "$switch_resp"

wait "$PID_A" "$PID_B"

paras_cmd_print_completion_file "$TAG R1" "$OUT_A"
paras_cmd_print_completion_file "$TAG R2" "$OUT_B"
