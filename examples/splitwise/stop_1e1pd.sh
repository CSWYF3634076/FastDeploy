#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/log_1e1pd}"
PID_FILE="${PID_FILE:-${LOG_DIR}/1e1pd.pid}"
WAIT_SEC="${WAIT_SEC:-20}"

PORT="${PORT:-8801}"
METRICS_PORT="${METRICS_PORT:-8802}"
ENGINE_WORKER_QUEUE_PORT="${ENGINE_WORKER_QUEUE_PORT:-8902}"
CACHE_QUEUE_PORT="${CACHE_QUEUE_PORT:-8903}"

kill_port_owner() {
  local port="$1"
  local pids=""

  if command -v lsof >/dev/null 2>&1; then
    pids="$(lsof -ti TCP:"${port}" -sTCP:LISTEN 2>/dev/null | tr '\n' ' ' || true)"
  else
    pids="$(ss -ltnp "sport = :${port}" 2>/dev/null | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | sort -u | tr '\n' ' ' || true)"
  fi

  if [[ -n "${pids// }" ]]; then
    echo "[1e1pd] force cleaning port ${port}, pid(s): ${pids}"
    kill -9 ${pids} >/dev/null 2>&1 || true
  fi
}

if [[ -f "${PID_FILE}" ]]; then
  pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
    echo "[1e1pd] stopping pid=${pid} ..."
    kill "${pid}" >/dev/null 2>&1 || true
    for ((i=0; i<WAIT_SEC; i++)); do
      if ! kill -0 "${pid}" >/dev/null 2>&1; then
        break
      fi
      sleep 1
    done
    if kill -0 "${pid}" >/dev/null 2>&1; then
      echo "[1e1pd] graceful stop timeout, force kill pid=${pid}"
      kill -9 "${pid}" >/dev/null 2>&1 || true
    fi
  fi
  rm -f "${PID_FILE}"
fi

for p in "${PORT}" "${METRICS_PORT}" "${ENGINE_WORKER_QUEUE_PORT}" "${CACHE_QUEUE_PORT}"; do
  kill_port_owner "${p}"
done

echo "[1e1pd] stopped."
