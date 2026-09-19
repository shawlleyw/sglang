#!/bin/bash
# Kill any running sglang processes and clear the test log file.
# Override LOG_FILE env to clean a different log path.
#
# The pkill pattern is anchored to actual sglang runtime process names
# (python -m sglang.launch_server, sglang_router, sglang::worker labels).
# A naive `pkill -f sglang` would also kill any bash subprocess whose argv
# contains the substring "sglang" (e.g. when run via `bash -c "cd
# /home/.../sglang && ..."`), self-terminating the caller.
set -uo pipefail

LOG_FILE=${LOG_FILE:-/tmp/sglang_paras_test.log}

pkill -9 -u "$USER" -f "python.*sglang\.launch_server|python.*sglang_router|sglang::" 2>/dev/null || true
sleep 3
rm -f "$LOG_FILE"
echo "killed sglang processes; removed $LOG_FILE"
