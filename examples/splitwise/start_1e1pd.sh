#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-}"
if [[ -z "${MODEL_PATH}" && $# -ge 1 ]]; then
  MODEL_PATH="$1"
  shift
fi

if [[ -z "${MODEL_PATH}" ]]; then
  echo "[1e1pd] Missing model path."
  echo "Usage: MODEL_PATH=/path/to/model ${BASH_SOURCE[0]} [extra args]"
  echo "   or: ${BASH_SOURCE[0]} /path/to/model [extra args]"
  exit 1
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8801}"
METRICS_PORT="${METRICS_PORT:-8802}"
ENGINE_WORKER_QUEUE_PORT="${ENGINE_WORKER_QUEUE_PORT:-8902}"
CACHE_QUEUE_PORT="${CACHE_QUEUE_PORT:-8903}"
WORKERS="${WORKERS:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/log_1e1pd}"
PID_FILE="${PID_FILE:-${LOG_DIR}/1e1pd.pid}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/1e1pd.log}"
mkdir -p "${LOG_DIR}"

kill_port_owner() {
  local port="$1"
  local pids=""

  if command -v lsof >/dev/null 2>&1; then
    pids="$(lsof -ti TCP:"${port}" -sTCP:LISTEN 2>/dev/null | tr '\n' ' ' || true)"
  else
    pids="$(ss -ltnp "sport = :${port}" 2>/dev/null | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | sort -u | tr '\n' ' ' || true)"
  fi

  if [[ -n "${pids// }" ]]; then
    echo "[1e1pd] port ${port} is occupied, killing pid(s): ${pids}"
    kill -9 ${pids} >/dev/null 2>&1 || true
    sleep 1
  fi
}

for p in "${PORT}" "${METRICS_PORT}" "${ENGINE_WORKER_QUEUE_PORT}" "${CACHE_QUEUE_PORT}"; do
  kill_port_owner "${p}"
done

if [[ -f "${PID_FILE}" ]]; then
  old_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" >/dev/null 2>&1; then
    echo "[1e1pd] stopping stale pid from pidfile: ${old_pid}"
    kill -9 "${old_pid}" >/dev/null 2>&1 || true
  fi
  rm -f "${PID_FILE}"
fi

export ENABLE_V1_KVCACHE_SCHEDULER="${ENABLE_V1_KVCACHE_SCHEDULER:-1}"
export FD_ENABLE_E2W_TENSOR_CONVERT="${FD_ENABLE_E2W_TENSOR_CONVERT:-1}"
export FD_LOG_DIR="${FD_LOG_DIR:-${LOG_DIR}}"

CMD=(
  python -m fastdeploy.entrypoints.openai.api_server
  --model "${MODEL_PATH}"
  --host "${HOST}"
  --port "${PORT}"
  --metrics-port "${METRICS_PORT}"
  --engine-worker-queue-port "${ENGINE_WORKER_QUEUE_PORT}"
  --cache-queue-port "${CACHE_QUEUE_PORT}"
  --workers "${WORKERS}"
  --splitwise-role mixed
  --no-epd-enable
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
)

if [[ $# -gt 0 ]]; then
  CMD+=("$@")
fi

echo "[1e1pd] starting server..."
echo "[1e1pd] command: ${CMD[*]}"
echo "[1e1pd] log_file=${LOG_FILE}"
nohup "${CMD[@]}" >"${LOG_FILE}" 2>&1 &
new_pid=$!
echo "${new_pid}" >"${PID_FILE}"
echo "[1e1pd] started, pid=${new_pid}"
