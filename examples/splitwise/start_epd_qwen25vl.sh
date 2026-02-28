#!/bin/bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source "${SCRIPT_DIR}/utils.sh"

unset http_proxy && unset https_proxy

MODEL_PATH="${MODEL_PATH:-/root/paddlejob/workspace/env_run/output/wangyafeng/models/Qwen2.5-VL-7B-Instruct}"
if [[ $# -ge 1 ]]; then
  MODEL_PATH="$1"
  shift
fi

ROUTER_PORT="${ROUTER_PORT:-52700}"

E_PORT="${E_PORT:-52400}"
E_METRICS_PORT="${E_METRICS_PORT:-52401}"
E_ENGINE_WORKER_QUEUE_PORT="${E_ENGINE_WORKER_QUEUE_PORT:-52402}"
E_CACHE_QUEUE_PORT="${E_CACHE_QUEUE_PORT:-52403}"

PD_PORT="${PD_PORT:-52500}"
PD_METRICS_PORT="${PD_METRICS_PORT:-52501}"
PD_ENGINE_WORKER_QUEUE_PORT="${PD_ENGINE_WORKER_QUEUE_PORT:-52502}"
PD_CACHE_QUEUE_PORT="${PD_CACHE_QUEUE_PORT:-52503}"

CUDA_VISIBLE_DEVICES_E="${CUDA_VISIBLE_DEVICES_E:-0}"
CUDA_VISIBLE_DEVICES_PD="${CUDA_VISIBLE_DEVICES_PD:-1}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
CACHE_TRANSFER_PROTOCOL="${CACHE_TRANSFER_PROTOCOL:-ipc}"

LOG_DATE="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${LOG_ROOT:-${SCRIPT_DIR}/log/${LOG_DATE}_epd_qwen25vl}"
PID_FILE="${PID_FILE:-${LOG_ROOT}/epd_qwen25vl.pid}"
LATEST_PID_FILE="${LATEST_PID_FILE:-${SCRIPT_DIR}/log/epd_qwen25vl.latest}"
HEALTH_TIMEOUT_SEC="${HEALTH_TIMEOUT_SEC:-600}"
mkdir -p "${LOG_ROOT}"
mkdir -p "$(dirname "${LATEST_PID_FILE}")"

kill_port_owner() {
  local port="$1"
  local pids=""
  if command -v lsof >/dev/null 2>&1; then
    pids="$(lsof -ti TCP:"${port}" -sTCP:LISTEN 2>/dev/null | tr '\n' ' ' || true)"
  else
    pids="$(ss -ltnp "sport = :${port}" 2>/dev/null | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | sort -u | tr '\n' ' ' || true)"
  fi
  if [[ -n "${pids// }" ]]; then
    echo "[EPD] port ${port} occupied, killing pid(s): ${pids}"
    kill -9 ${pids} >/dev/null 2>&1 || true
    sleep 1
  fi
}

ports=(
  "${ROUTER_PORT}"
  "${E_PORT}" "${E_METRICS_PORT}" "${E_ENGINE_WORKER_QUEUE_PORT}" "${E_CACHE_QUEUE_PORT}"
  "${PD_PORT}" "${PD_METRICS_PORT}" "${PD_ENGINE_WORKER_QUEUE_PORT}" "${PD_CACHE_QUEUE_PORT}"
)

for p in "${ports[@]}"; do
  kill_port_owner "${p}"
done

if [[ -f "${PID_FILE}" ]]; then
  echo "[EPD] remove stale pid file: ${PID_FILE}"
  rm -f "${PID_FILE}"
fi

export ENABLE_V1_KVCACHE_SCHEDULER="${ENABLE_V1_KVCACHE_SCHEDULER:-1}"
export FD_ENABLE_E2W_TENSOR_CONVERT="${FD_ENABLE_E2W_TENSOR_CONVERT:-1}"
export FD_DEBUG="${FD_DEBUG:-1}"

echo "[EPD] model path: ${MODEL_PATH}"
echo "[EPD] log root : ${LOG_ROOT}"
echo "[EPD] health timeout: ${HEALTH_TIMEOUT_SEC}s"
echo "[EPD] cache-transfer-protocol: ${CACHE_TRANSFER_PROTOCOL}"

if [[ "${CACHE_TRANSFER_PROTOCOL}" == *"rdma"* ]]; then
  if ! python -c "import importlib.util,sys;sys.exit(0 if importlib.util.find_spec('rdma_comm') else 1)" >/dev/null 2>&1; then
    echo "[EPD][ERR] CACHE_TRANSFER_PROTOCOL includes 'rdma' but module 'rdma_comm' is not installed."
    echo "[EPD][ERR] Use CACHE_TRANSFER_PROTOCOL=ipc or install rdma_comm first."
    exit 1
  fi
fi

STARTED_PIDS=()
on_exit_cleanup() {
  local code=$?
  if [[ ${code} -eq 0 ]]; then
    return 0
  fi
  echo "[EPD][ERR] startup failed, cleaning started processes..."
  for pid in "${STARTED_PIDS[@]}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
      kill -9 "${pid}" >/dev/null 2>&1 || true
    fi
  done
}
trap on_exit_cleanup EXIT

wait_health_with_timeout() {
  local port="$1"
  local name="$2"
  local timeout="$3"
  local watch_pid="${4:-}"
  local start_ts
  start_ts=$(date +%s)
  while true; do
    local status_code
    status_code="$(curl -s --max-time 1 -o /dev/null -w "%{http_code}" "http://0.0.0.0:${port}/health" 2>/dev/null || true)"
    if [[ -z "${status_code}" ]]; then
      status_code="000"
    fi
    local now_ts elapsed
    now_ts=$(date +%s)
    elapsed=$((now_ts - start_ts))
    if [[ -n "${watch_pid}" ]] && ! kill -0 "${watch_pid}" >/dev/null 2>&1; then
      echo "[EPD][ERR] ${name} process exited before health ready, pid=${watch_pid}, elapsed=${elapsed}s"
      return 1
    fi
    if [[ "${status_code}" == "200" ]]; then
      echo "[EPD] ${name} health check passed, port=${port}, elapsed=${elapsed}s"
      return 0
    fi
    if (( elapsed >= timeout )); then
      echo "[EPD][ERR] ${name} health timeout, port=${port}, last_status=${status_code}, elapsed=${elapsed}s"
      return 1
    fi
    if (( elapsed % 5 == 0 )); then
      echo "[EPD] waiting ${name} health, port=${port}, status=${status_code}, elapsed=${elapsed}s"
    fi
    sleep 1
  done
}

# 1) Router
mkdir -p "${LOG_ROOT}/router"
ROUTER_CMD=(
  python -m fastdeploy.router.launch
  --port "${ROUTER_PORT}"
  --splitwise
)
echo "[EPD] launch router cmd: ${ROUTER_CMD[*]}"
nohup "${ROUTER_CMD[@]}" >"${LOG_ROOT}/router/nohup.log" 2>&1 &
ROUTER_PID=$!
STARTED_PIDS+=("${ROUTER_PID}")
echo "[EPD] router started, pid=${ROUTER_PID}, port=${ROUTER_PORT}"

# 2) E node (encoder role)
mkdir -p "${LOG_ROOT}/encoder"
E_CMD=(
  python -m fastdeploy.entrypoints.openai.api_server
  --model "${MODEL_PATH}"
  --port "${E_PORT}"
  --metrics-port "${E_METRICS_PORT}"
  --engine-worker-queue-port "${E_ENGINE_WORKER_QUEUE_PORT}"
  --cache-queue-port "${E_CACHE_QUEUE_PORT}"
  --limit-mm-per-prompt '{"image": 100, "video": 100}'
  --splitwise-role encoder
  --epd-enable
  --epd-encoder-model qwen2.5vl
  --cache-transfer-protocol "${CACHE_TRANSFER_PROTOCOL}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --router "0.0.0.0:${ROUTER_PORT}"
)
if [[ $# -gt 0 ]]; then
  E_CMD+=("$@")
fi
echo "[EPD] launch encoder cmd: FD_LOG_DIR=${LOG_ROOT}/encoder CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_E} ${E_CMD[*]}"
FD_LOG_DIR="${LOG_ROOT}/encoder" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_E}" \
nohup "${E_CMD[@]}" >"${LOG_ROOT}/encoder/nohup.log" 2>&1 &
E_PID=$!
STARTED_PIDS+=("${E_PID}")
echo "[EPD] encoder started, pid=${E_PID}, port=${E_PORT}, gpu=${CUDA_VISIBLE_DEVICES_E}"

if ! wait_health_with_timeout "${E_PORT}" "encoder" "${HEALTH_TIMEOUT_SEC}" "${E_PID}"; then
  echo "[EPD][ERR] encoder nohup log tail:"
  tail -n 120 "${LOG_ROOT}/encoder/nohup.log" || true
  exit 1
fi

# 3) PD node (decode role)
mkdir -p "${LOG_ROOT}/pd"
PD_CMD=(
  python -m fastdeploy.entrypoints.openai.api_server
  --model "${MODEL_PATH}"
  --port "${PD_PORT}"
  --metrics-port "${PD_METRICS_PORT}"
  --engine-worker-queue-port "${PD_ENGINE_WORKER_QUEUE_PORT}"
  --cache-queue-port "${PD_CACHE_QUEUE_PORT}"
  --limit-mm-per-prompt '{"image": 100, "video": 100}'
  --splitwise-role decode
  --epd-enable
  --epd-encoder-model qwen2.5vl
  --cache-transfer-protocol "${CACHE_TRANSFER_PROTOCOL}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --router "0.0.0.0:${ROUTER_PORT}"
)
if [[ $# -gt 0 ]]; then
  PD_CMD+=("$@")
fi
echo "[EPD] launch pd cmd: FD_LOG_DIR=${LOG_ROOT}/pd CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_PD} ${PD_CMD[*]}"
FD_LOG_DIR="${LOG_ROOT}/pd" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_PD}" \
nohup "${PD_CMD[@]}" >"${LOG_ROOT}/pd/nohup.log" 2>&1 &
PD_PID=$!
STARTED_PIDS+=("${PD_PID}")
echo "[EPD] pd(decode) started, pid=${PD_PID}, port=${PD_PORT}, gpu=${CUDA_VISIBLE_DEVICES_PD}"

if ! wait_health_with_timeout "${PD_PORT}" "pd(decode)" "${HEALTH_TIMEOUT_SEC}" "${PD_PID}"; then
  echo "[EPD][ERR] pd nohup log tail:"
  tail -n 120 "${LOG_ROOT}/pd/nohup.log" || true
  exit 1
fi

cat >"${PID_FILE}" <<EOF
ROUTER_PID=${ROUTER_PID}
E_PID=${E_PID}
PD_PID=${PD_PID}
ROUTER_PORT=${ROUTER_PORT}
E_PORT=${E_PORT}
E_METRICS_PORT=${E_METRICS_PORT}
E_ENGINE_WORKER_QUEUE_PORT=${E_ENGINE_WORKER_QUEUE_PORT}
E_CACHE_QUEUE_PORT=${E_CACHE_QUEUE_PORT}
PD_PORT=${PD_PORT}
PD_METRICS_PORT=${PD_METRICS_PORT}
PD_ENGINE_WORKER_QUEUE_PORT=${PD_ENGINE_WORKER_QUEUE_PORT}
PD_CACHE_QUEUE_PORT=${PD_CACHE_QUEUE_PORT}
EOF
echo "${PID_FILE}" > "${LATEST_PID_FILE}"

echo "[EPD] all services are up."
echo "[EPD] pid file: ${PID_FILE}"
echo "[EPD] latest pid file pointer: ${LATEST_PID_FILE}"
echo "[EPD] test endpoint: http://0.0.0.0:${ROUTER_PORT}/v1/chat/completions"
trap - EXIT
