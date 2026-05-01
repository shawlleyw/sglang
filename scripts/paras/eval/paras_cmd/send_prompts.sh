#!/bin/bash
# Send the 3 canonical test prompts (binary search / train math / TCP-vs-UDP)
# sequentially; print each response with a [LABEL P<n>] prefix.
# These prompts stress-test multi-step reasoning, chain-of-thought, and long
# coherent decode (200 tokens) — short responses can mask partially corrupted
# weights post-switch.
#
# Usage: bash send_prompts.sh <LABEL>
#   e.g. bash send_prompts.sh EP
#        bash send_prompts.sh TP
#        bash send_prompts.sh EP-RT
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "$SCRIPT_DIR/lib.sh"

LABEL=${1:?usage: send_prompts.sh <LABEL>}

PROMPT_1="Write a Python function that implements binary search on a sorted list, with docstring and edge case handling."
PROMPT_2="A train leaves city A at 60mph. Another train leaves city B (300 miles away) at 80mph heading toward A. When do they meet? Show your work step by step."
PROMPT_3="Explain the difference between TCP and UDP protocols. Include: connection setup, reliability, use cases."

paras_cmd_send_completion "$LABEL P1" "$PROMPT_1" 200
paras_cmd_send_completion "$LABEL P2" "$PROMPT_2" 150
paras_cmd_send_completion "$LABEL P3" "$PROMPT_3" 200
