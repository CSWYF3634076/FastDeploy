#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${SCRIPT_DIR}/stop_1e1pd.sh" ]]; then
  bash "${SCRIPT_DIR}/stop_1e1pd.sh" || true
fi
if [[ -f "${SCRIPT_DIR}/stop_epd_qwen25vl.sh" ]]; then
  bash "${SCRIPT_DIR}/stop_epd_qwen25vl.sh" || true
fi

# Legacy broad cleanup for splitwise examples
pkill -9 -f "fastdeploy.entrypoints.openai.api_server" >/dev/null 2>&1 || true
pkill -9 -f "fastdeploy.router.launch" >/dev/null 2>&1 || true
pkill -9 -f "gunicorn" >/dev/null 2>&1 || true
# Kill redis-server if you need.
# pkill -9 -f redis-server >/dev/null 2>&1 || true

sleep 1
echo "splitwise services stopped."
