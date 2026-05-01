#!/bin/bash
# Shared helpers for paras_cmd scripts. Source from sibling scripts:
#   SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
#   source "$SCRIPT_DIR/lib.sh"
#
# Common env-var overrides:
#   HOST       Server host. Default: 127.0.0.1
#   PORT       Server port. Default: 30000
#   MODEL_NAME Model name in OpenAI request body. Default: Qwen3-30B-A3B
#   LOG_FILE   Path to server log (tee'd from launch_server). Default: /tmp/sglang_paras_test.log
#   PRINT_CHARS Number of response chars to print per prompt. Default: 300

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-30000}
MODEL_NAME=${MODEL_NAME:-Qwen3-30B-A3B}
LOG_FILE=${LOG_FILE:-/tmp/sglang_paras_test.log}
PRINT_CHARS=${PRINT_CHARS:-300}

# Send a /v1/completions request and print "[label] <text>" with the response.
# Usage: paras_cmd_send_completion <label> <prompt> [max_tokens=200] [timeout=60] [save_path=]
# If save_path is set, also writes the raw JSON response to that file.
paras_cmd_send_completion() {
    local label=$1
    local prompt=$2
    local max_tokens=${3:-200}
    local timeout=${4:-60}
    local save_path=${5:-}

    local payload
    payload=$(MODEL_NAME="$MODEL_NAME" PROMPT="$prompt" MAX_TOKENS="$max_tokens" python3 -c '
import os, json
print(json.dumps({
    "model": os.environ["MODEL_NAME"],
    "prompt": os.environ["PROMPT"],
    "max_tokens": int(os.environ["MAX_TOKENS"]),
    "temperature": 0,
}))
')

    local resp
    resp=$(curl -s --max-time "$timeout" "http://${HOST}:${PORT}/v1/completions" \
        -H "Content-Type: application/json" \
        -d "$payload")
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "[$label] CURL_ERROR rc=$rc"
        return $rc
    fi

    if [ -n "$save_path" ]; then
        printf '%s' "$resp" > "$save_path"
    fi

    LABEL="$label" RESP="$resp" PRINT_CHARS="$PRINT_CHARS" python3 -c '
import os, json, sys
label = os.environ["LABEL"]
n = int(os.environ["PRINT_CHARS"])
try:
    d = json.loads(os.environ["RESP"])
    print(f"[{label}]", d["choices"][0]["text"][:n])
except Exception as e:
    print(f"[{label}] PARSE_ERROR:", e, file=sys.stderr)
    print(os.environ["RESP"][:500], file=sys.stderr)
    sys.exit(2)
'
}

paras_cmd_build_payload() {
    local prompt=$1
    local max_tokens=${2:-200}
    MODEL_NAME="$MODEL_NAME" PROMPT="$prompt" MAX_TOKENS="$max_tokens" python3 -c '
import os, json
print(json.dumps({
    "model": os.environ["MODEL_NAME"],
    "prompt": os.environ["PROMPT"],
    "max_tokens": int(os.environ["MAX_TOKENS"]),
    "temperature": 0,
}))
'
}

# Print the [label] text for a JSON file. Use after a `wait` on a backgrounded
# curl that wrote its response to <json_path>.
# Usage: paras_cmd_print_completion_file <label> <json_path>
paras_cmd_print_completion_file() {
    local label=$1
    local path=$2
    LABEL="$label" PATH_=$path PRINT_CHARS="$PRINT_CHARS" python3 -c '
import os, json, sys
label = os.environ["LABEL"]
n = int(os.environ["PRINT_CHARS"])
try:
    d = json.load(open(os.environ["PATH_"]))
    print(f"[{label}]", d["choices"][0]["text"][:n])
except Exception as e:
    print(f"[{label}] PARSE_ERROR:", e, file=sys.stderr)
    sys.exit(2)
'
}
