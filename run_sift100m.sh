#!/usr/bin/env bash

# Run the DISPQ SIFT100M evaluation only after both LIRE and OnlinePQ finish.
#
# Task:
#   binary:  build_release/run_eval_large_streaming
#   config:  configs/sift100m_large.json
#   dataset: ~/data/sift100M/origin
#
# Recommended race-free launch:
#   /path/to/run_lire.sh &
#   lire_pid=$!
#   /path/to/run_onlinepq.sh &
#   onlinepq_pid=$!
#   LIRE_PID="${lire_pid}" ONLINEPQ_PID="${onlinepq_pid}" \
#     ./run_dispq_sift100m_after_lire_onlinepq.sh
#
# Optional controls:
#   NODE=0                       manually override automatic node selection
#   NUMA_NODES=0,1               candidate NUMA nodes
#   NODE_POLL_SECONDS=30          free-node polling interval
#   THREADS_PER_NODE=36          thread budget
#   LIRE_PID=<pid>               known LIRE driver PID
#   ONLINEPQ_PID=<pid>           known OnlinePQ driver PID
#   LIRE_DONE_FILE=<path>        LIRE completion marker
#   ONLINEPQ_DONE_FILE=<path>    OnlinePQ completion marker
#   UPSTREAM_POLL_SECONDS=30     polling interval
#   WAIT_FOR_LIRE=0              bypass the LIRE gate
#   WAIT_FOR_ONLINEPQ=0          bypass the OnlinePQ gate

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

contains_runner() {
    local directory="$1"
    [[ -x "${directory}/build_release/run_eval_large_streaming" ||
       -x "${directory}/run_eval_large_streaming" ]]
}

if [[ -n "${PROJECT_ROOT_OVERRIDE:-}" ]]; then
    PROJECT_ROOT="${PROJECT_ROOT_OVERRIDE}"
elif contains_runner "${PWD}" && [[ -d "${PWD}/configs" ]]; then
    PROJECT_ROOT="${PWD}"
elif contains_runner "${SCRIPT_DIR}" && [[ -d "${SCRIPT_DIR}/configs" ]]; then
    PROJECT_ROOT="${SCRIPT_DIR}"
elif contains_runner "${SCRIPT_DIR}/.." && [[ -d "${SCRIPT_DIR}/../configs" ]]; then
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
    echo "[FATAL] Cannot locate the DISPQ project root."
    echo "Expected build_release/run_eval_large_streaming (or run_eval_large_streaming)"
    echo "and configs/. Set PROJECT_ROOT_OVERRIDE to the project root if needed."
    exit 1
fi

PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"

if [[ -n "${RUN_EVAL_LARGE_STREAMING:-}" ]]; then
    BINARY="${RUN_EVAL_LARGE_STREAMING}"
elif [[ -x "${PROJECT_ROOT}/build_release/run_eval_large_streaming" ]]; then
    BINARY="${PROJECT_ROOT}/build_release/run_eval_large_streaming"
else
    BINARY="${PROJECT_ROOT}/run_eval_large_streaming"
fi

CONFIG_REL="configs/sift100m_large.json"
CONFIG_PATH="${PROJECT_ROOT}/${CONFIG_REL}"
DATASET="${SIFT100M_DATASET:-${HOME}/data/sift100M/origin}"

NODE="${NODE:-}"
NUMA_NODES="${NUMA_NODES:-0,1}"
NODE_POLL_SECONDS="${NODE_POLL_SECONDS:-30}"
THREADS_PER_NODE="${THREADS_PER_NODE:-36}"

WAIT_FOR_LIRE="${WAIT_FOR_LIRE:-1}"
WAIT_FOR_ONLINEPQ="${WAIT_FOR_ONLINEPQ:-1}"
LIRE_PID="${LIRE_PID:-}"
ONLINEPQ_PID="${ONLINEPQ_PID:-}"
LIRE_DONE_FILE="${LIRE_DONE_FILE:-}"
ONLINEPQ_DONE_FILE="${ONLINEPQ_DONE_FILE:-}"
UPSTREAM_POLL_SECONDS="${UPSTREAM_POLL_SECONDS:-30}"

LIRE_PROCESS_PATTERN="${LIRE_PROCESS_PATTERN:-[r]un_(ivfpq|ivfopq|ivfflat|faiss).*\.sh|[f]aiss_(ivfpq|ivfopq|ivfflat)(_lire)?_run_eval(_large|_ms)?}"
ONLINEPQ_PROCESS_PATTERN="${ONLINEPQ_PROCESS_PATTERN:-[r]un_onlinepq.*\.sh|[o]nlinepq_eval(_large_streaming|_ms)?}"
BUSY_PROCESS_PATTERN="${BUSY_PROCESS_PATTERN:-[f]aiss_(ivfpq|ivfopq|ivfflat)(_lire)?_run_eval(_large|_ms)?|[o]nlinepq_eval(_large_streaming|_ms)?|(^|/)[r]un_eval(_large(_streaming)?|_ms)?}"

PROC_ROOT="${PROC_ROOT:-/proc}"
SYS_NODE_ROOT="${SYS_NODE_ROOT:-/sys/devices/system/node}"

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/dispq_sift100m_${TIMESTAMP}}"
DRIVER_LOG="${LOG_DIR}/driver.log"
TASK_LOG="${LOG_DIR}/sift100m.${TIMESTAMP}.log"
STATUS_FILE="${LOG_DIR}/sift100m.${TIMESTAMP}.status"

mkdir -p "${LOG_DIR}"

fatal() {
    echo "[FATAL] $*" | tee -a "${DRIVER_LOG}" >&2
    exit 1
}

processes_running() {
    local pattern="$1"
    pgrep -f "${pattern}" >/dev/null 2>&1
    local pgrep_rc=$?

    case "${pgrep_rc}" in
        0)
            return 0
            ;;
        1)
            return 1
            ;;
        *)
            fatal "process detection failed: pgrep rc=${pgrep_rc}, pattern=${pattern}"
            ;;
    esac
}

cpusets_intersect() {
    local set_a="${1// /}"
    local set_b="${2// /}"
    local part_a
    local part_b
    local start_a
    local end_a
    local start_b
    local end_b
    local -a parts_a
    local -a parts_b

    IFS=',' read -r -a parts_a <<< "${set_a}"
    IFS=',' read -r -a parts_b <<< "${set_b}"

    for part_a in "${parts_a[@]}"; do
        [[ -n "${part_a}" ]] || continue
        if [[ "${part_a}" == *-* ]]; then
            start_a="${part_a%%-*}"
            end_a="${part_a##*-}"
        else
            start_a="${part_a}"
            end_a="${part_a}"
        fi

        for part_b in "${parts_b[@]}"; do
            [[ -n "${part_b}" ]] || continue
            if [[ "${part_b}" == *-* ]]; then
                start_b="${part_b%%-*}"
                end_b="${part_b##*-}"
            else
                start_b="${part_b}"
                end_b="${part_b}"
            fi

            if (( start_a <= end_b && start_b <= end_a )); then
                return 0
            fi
        done
    done

    return 1
}

pid_uses_node() {
    local pid="$1"
    local node_cpuset="$2"
    local process_dir="${PROC_ROOT}/${pid}"
    local allowed_cpus

    [[ -d "${process_dir}" ]] || return 1

    allowed_cpus="$(awk '/^Cpus_allowed_list:/ { print $2; exit }' \
        "${process_dir}/status" 2>/dev/null)"

    # If affinity cannot be inspected, treat the process as occupying the node.
    [[ -n "${allowed_cpus}" ]] || return 0
    cpusets_intersect "${allowed_cpus}" "${node_cpuset}"
}

busy_processes_on_node() {
    local node="$1"
    local node_cpuset
    local pids
    local pgrep_rc
    local pid
    local comm

    [[ -r "${SYS_NODE_ROOT}/node${node}/cpulist" ]] || return 3
    node_cpuset="$(<"${SYS_NODE_ROOT}/node${node}/cpulist")"
    [[ -n "${node_cpuset}" ]] || return 3

    pids="$(pgrep -f "${BUSY_PROCESS_PATTERN}" 2>/dev/null)"
    pgrep_rc=$?

    case "${pgrep_rc}" in
        0)
            ;;
        1)
            return 0
            ;;
        *)
            return 2
            ;;
    esac

    for pid in ${pids}; do
        if pid_uses_node "${pid}" "${node_cpuset}"; then
            comm="unknown"
            if [[ -r "${PROC_ROOT}/${pid}/comm" ]]; then
                comm="$(<"${PROC_ROOT}/${pid}/comm")"
            fi
            printf '%s:%s\n' "${pid}" "${comm}"
        fi
    done
}

select_free_node() {
    local node
    local occupants
    local detect_rc
    local normalized_nodes="${NUMA_NODES//,/ }"

    if [[ -n "${NODE}" ]]; then
        [[ "${NODE}" =~ ^[0-9]+$ ]] || fatal "NODE must be numeric: ${NODE}"
        [[ -r "${SYS_NODE_ROOT}/node${NODE}/cpulist" ]] \
            || fatal "NUMA node does not exist: node${NODE}"
        echo "[NODE] Using manually selected node${NODE}." \
            | tee -a "${DRIVER_LOG}"
        return 0
    fi

    while true; do
        for node in ${normalized_nodes}; do
            [[ "${node}" =~ ^[0-9]+$ ]] \
                || fatal "invalid NUMA node in NUMA_NODES: ${node}"

            occupants="$(busy_processes_on_node "${node}")"
            detect_rc=$?

            case "${detect_rc}" in
                0)
                    ;;
                2)
                    fatal "busy-process detection failed for node${node}"
                    ;;
                3)
                    fatal "cannot read CPU topology for node${node}"
                    ;;
                *)
                    fatal "unexpected node detection error for node${node}: rc=${detect_rc}"
                    ;;
            esac

            if [[ -z "${occupants}" ]]; then
                NODE="${node}"
                echo "[NODE] Selected free node${NODE}." \
                    | tee -a "${DRIVER_LOG}"
                return 0
            fi

            echo "[WAIT] node${node} is busy (${occupants//$'\n'/, })." \
                | tee -a "${DRIVER_LOG}"
        done

        echo "[WAIT] No candidate NUMA node is free; retrying in ${NODE_POLL_SECONDS}s." \
            | tee -a "${DRIVER_LOG}"
        sleep "${NODE_POLL_SECONDS}"
    done
}

wait_for_group() {
    local label="$1"
    local enabled="$2"
    local known_pid="$3"
    local done_file="$4"
    local pattern="$5"

    if [[ "${enabled}" == "0" ]]; then
        echo "[WARN] ${label} gate disabled." | tee -a "${DRIVER_LOG}"
        return 0
    fi

    if [[ -n "${done_file}" ]]; then
        echo "[WAIT] Waiting for ${label} marker: ${done_file}" \
            | tee -a "${DRIVER_LOG}"
        while [[ ! -e "${done_file}" ]]; do
            sleep "${UPSTREAM_POLL_SECONDS}"
        done
        echo "[WAIT] ${label} marker detected." | tee -a "${DRIVER_LOG}"
        return 0
    fi

    if [[ -n "${known_pid}" ]]; then
        [[ "${known_pid}" =~ ^[0-9]+$ ]] \
            || fatal "${label} PID must be numeric: ${known_pid}"
        echo "[WAIT] Waiting for ${label} driver PID ${known_pid}." \
            | tee -a "${DRIVER_LOG}"
        while kill -0 "${known_pid}" 2>/dev/null; do
            sleep "${UPSTREAM_POLL_SECONDS}"
        done
        echo "[WAIT] ${label} driver PID ${known_pid} has exited." \
            | tee -a "${DRIVER_LOG}"
        return 0
    fi

    if processes_running "${pattern}"; then
        echo "[WAIT] ${label} is running. Matched processes:" \
            | tee -a "${DRIVER_LOG}"
        pgrep -af "${pattern}" | tee -a "${DRIVER_LOG}"

        while processes_running "${pattern}"; do
            sleep "${UPSTREAM_POLL_SECONDS}"
        done

        echo "[WAIT] No ${label} driver/evaluator process remains." \
            | tee -a "${DRIVER_LOG}"
    else
        echo "[WAIT] No active ${label} process detected." \
            | tee -a "${DRIVER_LOG}"
    fi
}

[[ -x "${BINARY}" ]] \
    || fatal "run_eval_large_streaming not found or not executable: ${BINARY}"
[[ -f "${CONFIG_PATH}" ]] \
    || fatal "config not found: ${CONFIG_PATH}"
[[ -d "${DATASET}" ]] \
    || fatal "dataset not found: ${DATASET}"

for required_command in numactl; do
    command -v "${required_command}" >/dev/null 2>&1 \
        || fatal "required command not found: ${required_command}"
done

if [[ -z "${NODE}" ||
      ("${WAIT_FOR_LIRE}" != "0" &&
       -z "${LIRE_PID}" &&
       -z "${LIRE_DONE_FILE}") ||
      ("${WAIT_FOR_ONLINEPQ}" != "0" &&
       -z "${ONLINEPQ_PID}" &&
       -z "${ONLINEPQ_DONE_FILE}") ]]; then
    command -v pgrep >/dev/null 2>&1 \
        || fatal "pgrep is required for automatic upstream process detection"
fi

export OMP_NUM_THREADS="${THREADS_PER_NODE}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export MKL_NUM_THREADS="${THREADS_PER_NODE}"
export OPENBLAS_NUM_THREADS="${THREADS_PER_NODE}"
export BLIS_NUM_THREADS="${THREADS_PER_NODE}"
export NUMEXPR_NUM_THREADS="${THREADS_PER_NODE}"

cat <<INFO | tee -a "${DRIVER_LOG}"
============================================================
DISPQ SIFT100M task prepared at $(date '+%F %T')

PROJECT_ROOT=${PROJECT_ROOT}
BINARY=${BINARY}
CONFIG=${CONFIG_REL}
DATASET=${DATASET}
NODE=${NODE:-auto}
NUMA_NODES=${NUMA_NODES}
NODE_POLL_SECONDS=${NODE_POLL_SECONDS}
THREADS_PER_NODE=${THREADS_PER_NODE}

WAIT_FOR_LIRE=${WAIT_FOR_LIRE}
LIRE_PID=${LIRE_PID:-auto}
LIRE_DONE_FILE=${LIRE_DONE_FILE:-auto}

WAIT_FOR_ONLINEPQ=${WAIT_FOR_ONLINEPQ}
ONLINEPQ_PID=${ONLINEPQ_PID:-auto}
ONLINEPQ_DONE_FILE=${ONLINEPQ_DONE_FILE:-auto}

LOG=${TASK_LOG}
STATUS=${STATUS_FILE}
============================================================
INFO

wait_for_group \
    "LIRE" \
    "${WAIT_FOR_LIRE}" \
    "${LIRE_PID}" \
    "${LIRE_DONE_FILE}" \
    "${LIRE_PROCESS_PATTERN}"

wait_for_group \
    "OnlinePQ" \
    "${WAIT_FOR_ONLINEPQ}" \
    "${ONLINEPQ_PID}" \
    "${ONLINEPQ_DONE_FILE}" \
    "${ONLINEPQ_PROCESS_PATTERN}"

select_free_node

echo "[INFO] LIRE and OnlinePQ gates passed; starting DISPQ SIFT100M on node${NODE}." \
    | tee -a "${DRIVER_LOG}"

START_SECONDS="$(date +%s)"

{
    echo "============================================================"
    echo "[START]   $(date '+%F %T')"
    echo "[NODE]    ${NODE}"
    echo "[BINARY]  ${BINARY}"
    echo "[CONFIG]  ${CONFIG_REL}"
    echo "[DATASET] ${DATASET}"
    echo "[COMMAND] numactl --cpunodebind=${NODE} --membind=${NODE} ${BINARY} ${CONFIG_REL} ${DATASET}"
    echo "============================================================"
} | tee "${TASK_LOG}" | tee -a "${DRIVER_LOG}"

(
    cd "${PROJECT_ROOT}" || exit 125
    numactl \
        --cpunodebind="${NODE}" \
        --membind="${NODE}" \
        env \
            OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
            OMP_PROC_BIND="${OMP_PROC_BIND}" \
            OMP_PLACES="${OMP_PLACES}" \
            MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
            OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" \
            BLIS_NUM_THREADS="${BLIS_NUM_THREADS}" \
            NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS}" \
            "${BINARY}" \
            "${CONFIG_REL}" \
            "${DATASET}"
) >> "${TASK_LOG}" 2>&1

RC=$?
END_SECONDS="$(date +%s)"
ELAPSED=$((END_SECONDS - START_SECONDS))

if (( RC == 0 )); then
    STATUS="OK"
else
    STATUS="FAILED"
fi

{
    echo "============================================================"
    echo "[END]     $(date '+%F %T')"
    echo "[SECONDS] ${ELAPSED}"
    echo "[RC]      ${RC}"
    echo "[STATUS]  ${STATUS}"
    echo "============================================================"
} | tee -a "${TASK_LOG}" "${DRIVER_LOG}"

printf 'status=%s\nexit_code=%s\nelapsed_sec=%s\nnode=%s\nbinary=%s\nconfig=%s\ndataset=%s\nlog=%s\n' \
    "${STATUS}" \
    "${RC}" \
    "${ELAPSED}" \
    "${NODE}" \
    "${BINARY}" \
    "${CONFIG_REL}" \
    "${DATASET}" \
    "${TASK_LOG}" \
    > "${STATUS_FILE}"

exit "${RC}"