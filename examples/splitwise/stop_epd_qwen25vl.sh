#!/bin/bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

WAIT_SEC="${WAIT_SEC:-20}"
LATEST_PID_FILE="${LATEST_PID_FILE:-${SCRIPT_DIR}/log/epd_qwen25vl.latest}"
PID_FILE_INPUT="${PID_FILE:-}"

ROUTER_PORT="${ROUTER_PORT:-52700}"
E_PORT="${E_PORT:-52400}"
E_METRICS_PORT="${E_METRICS_PORT:-52401}"
E_ENGINE_WORKER_QUEUE_PORT="${E_ENGINE_WORKER_QUEUE_PORT:-52402}"
E_CACHE_QUEUE_PORT="${E_CACHE_QUEUE_PORT:-52403}"
PD_PORT="${PD_PORT:-52500}"
PD_METRICS_PORT="${PD_METRICS_PORT:-52501}"
PD_ENGINE_WORKER_QUEUE_PORT="${PD_ENGINE_WORKER_QUEUE_PORT:-52502}"
PD_CACHE_QUEUE_PORT="${PD_CACHE_QUEUE_PORT:-52503}"

kill_pid_gracefully() {
  local pid="$1"
  local name="$2"
  if [[ -z "${pid}" ]]; then
    return 0
  fi
  if ! kill -0 "${pid}" >/dev/null 2>&1; then
    return 0
  fi

  echo "[EPD] stopping ${name}, pid=${pid}"
  kill "${pid}" >/dev/null 2>&1 || true
  for ((i = 0; i < WAIT_SEC; i++)); do
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  if kill -0 "${pid}" >/dev/null 2>&1; then
    echo "[EPD] graceful stop timeout, force kill ${name}, pid=${pid}"
    kill -9 "${pid}" >/dev/null 2>&1 || true
  fi
}

kill_port_owner() {
  local port="$1"
  local pids=""
  if command -v lsof >/dev/null 2>&1; then
    pids="$(lsof -ti TCP:"${port}" -sTCP:LISTEN 2>/dev/null | tr '\n' ' ' || true)"
  else
    pids="$(ss -ltnp "sport = :${port}" 2>/dev/null | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | sort -u | tr '\n' ' ' || true)"
  fi
  if [[ -n "${pids// }" ]]; then
    echo "[EPD] force cleaning port ${port}, pid(s): ${pids}"
    kill -9 ${pids} >/dev/null 2>&1 || true
  fi
}

resolve_pid_file() {
  if [[ -n "${PID_FILE_INPUT}" ]]; then
    echo "${PID_FILE_INPUT}"
    return 0
  fi
  if [[ -f "${LATEST_PID_FILE}" ]]; then
    cat "${LATEST_PID_FILE}"
    return 0
  fi
  echo ""
}

read_kv_from_pid_file() {
  local pid_file="$1"
  local key="$2"
  if [[ ! -f "${pid_file}" ]]; then
    echo ""
    return 0
  fi
  local line
  line="$(grep -E "^${key}=" "${pid_file}" | tail -n 1 || true)"
  if [[ -z "${line}" ]]; then
    echo ""
  else
    echo "${line#*=}"
  fi
}

PID_FILE_REAL="$(resolve_pid_file)"
if [[ -n "${PID_FILE_REAL}" ]]; then
  echo "[EPD] using pid file: ${PID_FILE_REAL}"
fi

if [[ -n "${PID_FILE_REAL}" && -f "${PID_FILE_REAL}" ]]; then
  ROUTER_PID="$(read_kv_from_pid_file "${PID_FILE_REAL}" "ROUTER_PID")"
  E_PID="$(read_kv_from_pid_file "${PID_FILE_REAL}" "E_PID")"
  PD_PID="$(read_kv_from_pid_file "${PID_FILE_REAL}" "PD_PID")"

  ROUTER_PORT="$(read_kv_from_pid_file "${PID_FILE_REAL}" "ROUTER_PORT" || true)"
  E_PORT="$(read_kv_from_pid_file "${PID_FILE_REAL}" "E_PORT" || true)"
  E_METRICS_PORT="$(read_kv_from_pid_file "${PID_FILE_REAL}" "E_METRICS_PORT" || true)"
  E_ENGINE_WORKER_QUEUE_PORT="$(read_kv_from_pid_file "${PID_FILE_REAL}" "E_ENGINE_WORKER_QUEUE_PORT" || true)"
  E_CACHE_QUEUE_PORT="$(read_kv_from_pid_file "${PID_FILE_REAL}" "E_CACHE_QUEUE_PORT" || true)"
  PD_PORT="$(read_kv_from_pid_file "${PID_FILE_REAL}" "PD_PORT" || true)"
  PD_METRICS_PORT="$(read_kv_from_pid_file "${PID_FILE_REAL}" "PD_METRICS_PORT" || true)"
  PD_ENGINE_WORKER_QUEUE_PORT="$(read_kv_from_pid_file "${PID_FILE_REAL}" "PD_ENGINE_WORKER_QUEUE_PORT" || true)"
  PD_CACHE_QUEUE_PORT="$(read_kv_from_pid_file "${PID_FILE_REAL}" "PD_CACHE_QUEUE_PORT" || true)"

  kill_pid_gracefully "${PD_PID:-}" "pd(decode)"
  kill_pid_gracefully "${E_PID:-}" "encoder"
  kill_pid_gracefully "${ROUTER_PID:-}" "router"

  rm -f "${PID_FILE_REAL}"
fi

ports=(
  "${ROUTER_PORT:-52700}"
  "${E_PORT:-52400}" "${E_METRICS_PORT:-52401}" "${E_ENGINE_WORKER_QUEUE_PORT:-52402}" "${E_CACHE_QUEUE_PORT:-52403}"
  "${PD_PORT:-52500}" "${PD_METRICS_PORT:-52501}" "${PD_ENGINE_WORKER_QUEUE_PORT:-52502}" "${PD_CACHE_QUEUE_PORT:-52503}"
)
for p in "${ports[@]}"; do
  kill_port_owner "${p}"
done

pkill -9 -f "fastdeploy.router.launch" >/dev/null 2>&1 || true
pkill -9 -f "fastdeploy.entrypoints.openai.api_server.*--epd-enable" >/dev/null 2>&1 || true

if [[ -f "${LATEST_PID_FILE}" ]]; then
  rm -f "${LATEST_PID_FILE}"
fi

ps -auxww | grep wangyafeng | grep fastdeploy | grep -v grep | awk '{print $2}' | xargs -r kill -15

ps -auxww | grep router | grep fastdeploy | grep -v grep | awk '{print $2}' | xargs -r kill -15

echo "[EPD] stopped."
