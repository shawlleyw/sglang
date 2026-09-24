#!/bin/bash
# Shared helpers for paras_cmd scripts. Source from sibling scripts:
#   SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
#   source "$SCRIPT_DIR/lib.sh"
#
# Common env-var overrides:
#   HOST          Server host. Default: 127.0.0.1
#   PORT          Server port. Default: 30000
#   MODEL_NAME    Model name in OpenAI request body. Default: Qwen3-30B-A3B
#   LOG_FILE      Path to server log (tee'd from launch_server). Default: /tmp/sglang_paras_test.log
#   PRINT_CHARS   Response chars to print per prompt. Default: 300
#   BURST_SIZE    Concurrency for send_prompts.sh / inflight_switch.sh. Default: 32
#   PROMPTS_FILE  Path to one-prompt-per-line file. Default: $SCRIPT_DIR/prompts_diverse.txt
#   MAX_TOKENS    max_tokens for send_prompts.sh requests. Default: 200

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-30000}
MODEL_NAME=${MODEL_NAME:-Qwen3-30B-A3B}
LOG_FILE=${LOG_FILE:-/tmp/sglang_paras_test.log}
PRINT_CHARS=${PRINT_CHARS:-300}
BURST_SIZE=${BURST_SIZE:-32}
MAX_TOKENS=${MAX_TOKENS:-200}

# Require the endpoint's exact success message, not merely HTTP 200.
paras_cmd_verify_switch() {
    local mode=$1 response=$2 elapsed_ms=$3
    if [ "$response" != "ParaS ${mode^^} parallelism configured." ]; then
        echo "FAIL: unexpected configure_${mode} response: $response" >&2
        return 1
    fi
    python3 - "$elapsed_ms" "${CONFIGURE_MAX_MS:-2500}" "$LOG_FILE" "${4:-}" "$mode" <<'PY'
import math, re, sys
elapsed, limit = map(float, sys.argv[1:3])
if not math.isfinite(elapsed) or not 0 <= elapsed < limit:
    sys.exit(f'FAIL: configure took {elapsed} ms; threshold is {limit} ms')
if sys.argv[4]:
    with open(sys.argv[3], 'rb') as stream:
        stream.seek(int(sys.argv[4]))
        recent = stream.read().decode(errors='replace')
    if 'switch rejected by precheck' in recent:
        sys.exit('FAIL: scheduler rejected the requested switch')
    timings = re.findall(r'Time taken to configure ' + sys.argv[5].upper() + r': ([\d.eE+-]+) ms', recent)
    if not timings:
        sys.exit('FAIL: no fresh successful switch in server log')
    if any(not math.isfinite(float(ms)) or not 0 <= float(ms) < limit for ms in timings):
        sys.exit(f'FAIL: scheduler switch timing exceeds {limit} ms')
PY
}

paras_cmd_burst_wait() {
    local pid status=0
    for pid in "$@"; do
        wait "$pid" || status=1
    done
    return "$status"
}

# Send a /v1/chat/completions request and print "[label] <text>" with the response.
# gpt-oss requires the harmony chat template (auto-applied by /v1/chat/completions);
# raw /v1/completions skips it and the model never emits <|return|> EOS, so use
# the chat endpoint exclusively. Response field is choices[0].message.content
# (which contains the raw harmony output without a reasoning-parser configured).
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
    "messages": [{"role": "user", "content": os.environ["PROMPT"]}],
    "max_tokens": int(os.environ["MAX_TOKENS"]),
    "temperature": 0,
}))
')

    local resp
    resp=$(curl -s --max-time "$timeout" "http://${HOST}:${PORT}/v1/chat/completions" \
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
    msg = d["choices"][0].get("message", {})
    text = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
    print(f"[{label}]", text[:n])
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
    "messages": [{"role": "user", "content": os.environ["PROMPT"]}],
    "max_tokens": int(os.environ["MAX_TOKENS"]),
    "temperature": 0,
}))
'
}

# Print the [label] text for a JSON file. Use after a `wait` on a backgrounded
# curl that wrote its response to <json_path>. Reads chat-completions response
# shape: choices[0].message.content (+ reasoning_content if parser is enabled).
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
    msg = d["choices"][0].get("message", {})
    text = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
    print(f"[{label}]", text[:n])
except Exception as e:
    print(f"[{label}] PARSE_ERROR:", e, file=sys.stderr)
    sys.exit(2)
'
}

# Read the first $BURST_SIZE lines (or fewer) from $PROMPTS_FILE into a bash
# array. Caller-allocates the array name. Default PROMPTS_FILE is
# prompts_diverse.txt next to lib.sh.
# Usage: paras_cmd_load_prompts <array-name>
paras_cmd_load_prompts() {
    local _arr_name=$1
    local _script_dir
    _script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
    local _prompts_file=${PROMPTS_FILE:-$_script_dir/prompts_diverse.txt}
    if [ ! -f "$_prompts_file" ]; then
        echo "ERROR: prompts file not found: $_prompts_file" >&2
        return 2
    fi
    mapfile -t "$_arr_name" < <(head -n "$BURST_SIZE" "$_prompts_file")
}

# Fire $BURST_SIZE parallel /v1/completions requests with prompts from the
# given array, each saved as $tmpdir/burst_<N>.json. Stores PIDs in the
# nameref array $2. Caller MUST `wait "${pids[@]}"` and rm -rf the tmpdir.
# Usage: paras_cmd_burst_send <prompts-array-name> <pids-array-name> <tmpdir> <max_tokens> [timeout=240]
paras_cmd_burst_send() {
    local _prompts_name=$1
    local _pids_name=$2
    local _tmpdir=$3
    local _max_tokens=$4
    local _timeout=${5:-240}

    local -n _prompts_ref="$_prompts_name"
    local -n _pids_ref="$_pids_name"
    _pids_ref=()

    local _i _payload _n
    for _i in "${!_prompts_ref[@]}"; do
        _payload=$(paras_cmd_build_payload "${_prompts_ref[$_i]}" "$_max_tokens")
        _n=$((_i + 1))
        curl --fail-with-body -sS --max-time "$_timeout" "http://${HOST}:${PORT}/v1/chat/completions" \
            -H "Content-Type: application/json" -d "$_payload" \
            > "$_tmpdir/burst_${_n}.json" &
        _pids_ref+=("$!")
    done
}

# Scan $tmpdir/burst_*.json for the canonical "(\b\w+\b)(\s+\1){5,}" attractor
# (5+ consecutive identical word repetitions). Print degenerate file indices
# inline. Returns 0 if all clean, 1 if any degenerate.
# Usage: paras_cmd_burst_verify <tmpdir> <label> [expected-count]
paras_cmd_burst_verify() {
    local _tmpdir=$1
    local _label=$2
    local _total=0 _degen=0 _f _idx
    for _f in "$_tmpdir"/burst_*.json; do
        _total=$((_total + 1))
        if ! python3 - "$_f" <<'PYVERIFY'
import json, re, sys
try:
    with open(sys.argv[1]) as f:
        response = json.load(f)
    message = response['choices'][0]['message']
    text = (message.get('reasoning_content') or '') + (message.get('content') or '')
    if len(text.strip()) < 10:
        raise ValueError('response shorter than 10 characters')
    if re.search(r'(\b\w+\b)(\s+\1){5,}', text):
        raise ValueError('degenerate repeated words')
except Exception as error:
    print(f'{sys.argv[1]}: {error}', file=sys.stderr)
    sys.exit(1)
PYVERIFY
        then
            _degen=$((_degen + 1))
            _idx=$(basename "$_f" .json | sed 's/burst_//')
            echo "  [${_label}] burst_${_idx}: INVALID OR DEGENERATE"
        fi
    done
    if [ -n "${3:-}" ] && [ "$_total" -ne "$3" ]; then
        echo "[${_label}] FAIL: expected $3 responses, found $_total"
        return 1
    fi
    if [ "$_degen" -gt 0 ]; then
        echo "[${_label}] FAIL: ${_degen} / ${_total} invalid or degenerate"
        return 1
    fi
    echo "[${_label}] PASS: 0 / ${_total} degenerate"
}
