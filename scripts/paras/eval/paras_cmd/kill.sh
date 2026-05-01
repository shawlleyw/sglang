#!/bin/bash
# Kill any running sglang processes and clear the test log file.
# Override LOG_FILE env to clean a different log path.
set -uo pipefail

LOG_FILE=${LOG_FILE:-/tmp/sglang_paras_test.log}

pkill -9 -f "sglang" 2>/dev/null
sleep 3
rm -f "$LOG_FILE"
echo "killed sglang processes; removed $LOG_FILE"
