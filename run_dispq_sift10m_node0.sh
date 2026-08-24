#!/usr/bin/env bash

set -uo pipefail

PROJECT_ROOT="/home/ydy/DISPQ_large"
BINARY="${PROJECT_ROOT}/build_release/run_eval_large_streaming"
CONFIG="${PROJECT_ROOT}/configs/sift10m.json"
DATASET="/home/ydy/data/sift10M/origin"
NODE=0

# Fast process RSS sampling.
MONITOR_INTERVAL_SECONDS="${MONITOR_INTERVAL_SECONDS:-0.2}"

# Collect expensive NUMA/system information once every N fast samples.
# 10 * 0.2s = about 2 seconds.
FULL_MONITOR_EVERY="${FULL_MONITOR_EVERY:-10}"

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_DIR="${PROJECT_ROOT}/logs/sift10M_${TIMESTAMP}"
RUN_LOG="${LOG_DIR}/run.log"
MONITOR_LOG="${LOG_DIR}/memory_monitor.log"
RESOURCE_LOG="${LOG_DIR}/resource_usage.log"
DIAG_LOG="${LOG_DIR}/termination_diagnostics.log"
STATUS_FILE="${LOG_DIR}/status.txt"

mkdir -p "${LOG_DIR}"

fatal() {
    echo "[FATAL] $*" | tee -a "${RUN_LOG}" >&2
    exit 1
}

[[ -x "${BINARY}" ]] || fatal "binary not found or not executable: ${BINARY}"
[[ -f "${CONFIG}" ]] || fatal "config not found: ${CONFIG}"
[[ -d "${DATASET}" ]] || fatal "dataset not found: ${DATASET}"
[[ -r "/sys/devices/system/node/node${NODE}/cpulist" ]] \
    || fatal "cannot read CPU list for NUMA node${NODE}"

for command_name in numactl pgrep ps setsid stdbuf /usr/bin/time; do
    command -v "${command_name}" >/dev/null 2>&1 \
        || fatal "required command not found: ${command_name}"
done

count_cpus() {
    local cpu_list="$1"
    local item start end
    local count=0
    local -a items

    IFS=',' read -r -a items <<< "${cpu_list}"
    for item in "${items[@]}"; do
        if [[ "${item}" == *-* ]]; then
            start="${item%%-*}"
            end="${item##*-}"
            count=$((count + end - start + 1))
        else
            count=$((count + 1))
        fi
    done
    printf '%s\n' "${count}"
}

NODE_CPU_LIST="$(<"/sys/devices/system/node/node${NODE}/cpulist")"
THREADS="$(count_cpus "${NODE_CPU_LIST}")"

export OMP_NUM_THREADS="${THREADS}"
export OMP_PROC_BIND=close
export OMP_PLACES=threads
export MKL_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export BLIS_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"

CGROUP_RELATIVE="$(awk -F: '$1 == "0" {print $3; exit}' /proc/self/cgroup 2>/dev/null)"
CGROUP_DIR=""
if [[ -n "${CGROUP_RELATIVE}" && -d "/sys/fs/cgroup${CGROUP_RELATIVE}" ]]; then
    CGROUP_DIR="/sys/fs/cgroup${CGROUP_RELATIVE}"
fi

read_memory_event() {
    local event_name="$1"

    if [[ -n "${CGROUP_DIR}" && -r "${CGROUP_DIR}/memory.events" ]]; then
        awk -v key="${event_name}" \
            '$1 == key {print $2; found=1}
             END {if (!found) print 0}' \
            "${CGROUP_DIR}/memory.events"
    else
        printf 'unknown\n'
    fi
}

OOM_BEFORE="$(read_memory_event oom)"
OOM_KILL_BEFORE="$(read_memory_event oom_kill)"
START_EPOCH="$(date +%s)"
START_TIME="$(date '+%F %T')"

{
    echo "============================================================"
    echo "[START]         ${START_TIME}"
    echo "[PROJECT_ROOT]  ${PROJECT_ROOT}"
    echo "[BINARY]        ${BINARY}"
    echo "[CONFIG]        ${CONFIG}"
    echo "[DATASET]       ${DATASET}"
    echo "[NUMA_NODE]     ${NODE}"
    echo "[NODE_CPUS]     ${NODE_CPU_LIST}"
    echo "[THREADS]       ${THREADS}"
    echo "[MONITOR_FAST]  ${MONITOR_INTERVAL_SECONDS}s"
    echo "[MONITOR_FULL]  every ${FULL_MONITOR_EVERY} fast samples"
    echo "[CGROUP]        ${CGROUP_DIR:-unavailable}"
    echo "[OOM_BEFORE]    ${OOM_BEFORE}"
    echo "[OOMKILL_BEFORE] ${OOM_KILL_BEFORE}"
    echo "============================================================"
} | tee -a "${RUN_LOG}"

resolve_target_pid() {
    local wrapper_pid="$1"
    local child
    local attempt

    # /usr/bin/time normally has the benchmark as its direct child after
    # numactl/env/stdbuf exec their commands.
    for ((attempt = 0; attempt < 40; ++attempt)); do
        child="$(pgrep -P "${wrapper_pid}" 2>/dev/null | head -n 1 || true)"
        if [[ -n "${child}" ]]; then
            printf '%s\n' "${child}"
            return
        fi
        sleep 0.05
    done

    printf '%s\n' "${wrapper_pid}"
}

record_memory_snapshot() {
    local wrapper_pid="$1"
    local target_pid="$2"
    local full_snapshot="$3"

    {
        echo "===== $(date '+%F %T.%3N') wrapper_pid=${wrapper_pid} target_pid=${target_pid} full=${full_snapshot} ====="

        if [[ -r "/proc/${target_pid}/status" ]]; then
            grep -E \
                '^(Name|State|Threads|VmPeak|VmSize|VmHWM|VmRSS|RssAnon|RssFile|RssShmem):' \
                "/proc/${target_pid}/status" || true
        else
            echo "process_status=unavailable"
        fi

        if (( full_snapshot )); then
            if command -v numastat >/dev/null 2>&1; then
                numastat -p "${target_pid}" || true
            fi

            grep -E \
                '^(MemTotal|MemFree|MemAvailable|Buffers|Cached|SwapTotal|SwapFree):' \
                /proc/meminfo || true

            if [[ -r "/sys/devices/system/node/node${NODE}/meminfo" ]]; then
                grep -E \
                    'MemTotal|MemFree|MemUsed|FilePages|AnonPages|Slab' \
                    "/sys/devices/system/node/node${NODE}/meminfo" || true
            fi

            if [[ -r /proc/pressure/memory ]]; then
                sed 's/^/memory_pressure: /' /proc/pressure/memory
            fi

            if [[ -n "${CGROUP_DIR}" ]]; then
                [[ -r "${CGROUP_DIR}/memory.current" ]] \
                    && echo "cgroup_memory_current=$(<"${CGROUP_DIR}/memory.current")"

                [[ -r "${CGROUP_DIR}/memory.max" ]] \
                    && echo "cgroup_memory_max=$(<"${CGROUP_DIR}/memory.max")"

                if [[ -r "${CGROUP_DIR}/memory.events" ]]; then
                    sed 's/^/cgroup_memory_event: /' \
                        "${CGROUP_DIR}/memory.events"
                fi
            fi
        fi
    } >> "${MONITOR_LOG}" 2>&1
}

monitor_process() {
    local wrapper_pid="$1"
    local target_pid
    local sample=0
    local full_snapshot

    target_pid="$(resolve_target_pid "${wrapper_pid}")"

    echo "[MONITOR_TARGET] wrapper_pid=${wrapper_pid} target_pid=${target_pid}" \
        >> "${MONITOR_LOG}"

    while kill -0 "${wrapper_pid}" 2>/dev/null; do
        if (( sample % FULL_MONITOR_EVERY == 0 )); then
            full_snapshot=1
        else
            full_snapshot=0
        fi

        record_memory_snapshot \
            "${wrapper_pid}" \
            "${target_pid}" \
            "${full_snapshot}"

        sample=$((sample + 1))
        sleep "${MONITOR_INTERVAL_SECONDS}"
    done

    record_memory_snapshot \
        "${wrapper_pid}" \
        "${target_pid}" \
        1
}

REQUESTED_SIGNAL=""
RUN_PID=""
RUN_PGID=""

forward_signal() {
    local signal_name="$1"

    REQUESTED_SIGNAL="${signal_name}"

    echo "[SIGNAL] Supervisor received ${signal_name} at $(date '+%F %T.%3N')" \
        | tee -a "${RUN_LOG}" >&2

    if [[ -n "${RUN_PGID}" ]]; then
        kill -"${signal_name}" -- "-${RUN_PGID}" 2>/dev/null || true
    elif [[ -n "${RUN_PID}" ]]; then
        kill -"${signal_name}" "${RUN_PID}" 2>/dev/null || true
    fi
}

# Closing a tmux client or terminal must not terminate this long-running job.
trap '' HUP
trap 'forward_signal TERM' TERM
trap 'forward_signal INT' INT

echo "[COMMAND] numactl --cpunodebind=${NODE} --membind=${NODE} ${BINARY} ${CONFIG} ${DATASET}" \
    | tee -a "${RUN_LOG}"

setsid /usr/bin/time -v -o "${RESOURCE_LOG}" \
    numactl --cpunodebind="${NODE}" --membind="${NODE}" \
    env \
        OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
        OMP_PROC_BIND="${OMP_PROC_BIND}" \
        OMP_PLACES="${OMP_PLACES}" \
        MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
        OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" \
        BLIS_NUM_THREADS="${BLIS_NUM_THREADS}" \
        NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS}" \
        stdbuf -oL -eL \
        "${BINARY}" "${CONFIG}" "${DATASET}" \
    >> "${RUN_LOG}" 2>&1 &

RUN_PID=$!

RUN_PGID="$(ps -o pgid= -p "${RUN_PID}" 2>/dev/null | tr -d '[:space:]')"
if [[ -z "${RUN_PGID}" ]]; then
    RUN_PGID="${RUN_PID}"
fi

echo "[PID] supervisor=${RUN_PID} pgid=${RUN_PGID}" \
    | tee -a "${RUN_LOG}"

monitor_process "${RUN_PID}" &
MONITOR_PID=$!

# wait(1) may return early when this shell receives SIGINT/SIGTERM.
# Continue waiting while the benchmark process is still alive so that
# Ctrl-C cannot leave the detached setsid workload running unnoticed.
RC=0

while true; do
    wait "${RUN_PID}"
    RC=$?

    if ! kill -0 "${RUN_PID}" 2>/dev/null; then
        break
    fi
done

kill "${MONITOR_PID}" 2>/dev/null || true
wait "${MONITOR_PID}" 2>/dev/null || true

END_EPOCH="$(date +%s)"
ELAPSED=$((END_EPOCH - START_EPOCH))

OOM_AFTER="$(read_memory_event oom)"
OOM_KILL_AFTER="$(read_memory_event oom_kill)"

OOM_DELTA="unknown"
OOM_KILL_DELTA="unknown"

if [[ "${OOM_BEFORE}" =~ ^[0-9]+$ &&
      "${OOM_AFTER}" =~ ^[0-9]+$ ]]; then
    OOM_DELTA=$((OOM_AFTER - OOM_BEFORE))
fi

if [[ "${OOM_KILL_BEFORE}" =~ ^[0-9]+$ &&
      "${OOM_KILL_AFTER}" =~ ^[0-9]+$ ]]; then
    OOM_KILL_DELTA=$((OOM_KILL_AFTER - OOM_KILL_BEFORE))
fi

if (( RC == 0 )); then
    STATUS="OK"
elif [[ "${OOM_KILL_DELTA}" =~ ^[0-9]+$ ]] &&
     (( OOM_KILL_DELTA > 0 )); then
    STATUS="FAILED_CGROUP_OOM_KILL"
elif (( RC == 137 )); then
    STATUS="FAILED_SIGKILL_POSSIBLE_OOM"
elif (( RC == 130 )); then
    STATUS="INTERRUPTED_SIGINT"
elif (( RC == 143 )); then
    STATUS="FAILED_SIGTERM"
else
    STATUS="FAILED"
fi

{
    echo "===== termination diagnostics: $(date '+%F %T') ====="
    echo "exit_code=${RC}"
    echo "requested_signal=${REQUESTED_SIGNAL:-none}"
    echo "cgroup_oom_before=${OOM_BEFORE}"
    echo "cgroup_oom_after=${OOM_AFTER}"
    echo "cgroup_oom_delta=${OOM_DELTA}"
    echo "cgroup_oom_kill_before=${OOM_KILL_BEFORE}"
    echo "cgroup_oom_kill_after=${OOM_KILL_AFTER}"
    echo "cgroup_oom_kill_delta=${OOM_KILL_DELTA}"
    echo
    echo "===== kernel OOM messages since start (if permitted) ====="

    if command -v journalctl >/dev/null 2>&1; then
        journalctl -k \
            --since "@${START_EPOCH}" \
            --no-pager 2>&1 \
            | grep -Ei \
                'out of memory|oom-kill|killed process|memory cgroup' \
            || true
    else
        echo "journalctl unavailable"
    fi
} >> "${DIAG_LOG}" 2>&1

{
    echo "============================================================"
    echo "[END]       $(date '+%F %T')"
    echo "[STATUS]    ${STATUS}"
    echo "[EXIT_CODE] ${RC}"
    echo "[ELAPSED_S] ${ELAPSED}"
    echo "[OOM_DELTA] ${OOM_DELTA}"
    echo "[OOM_KILL_DELTA] ${OOM_KILL_DELTA}"
    echo "[RUN_LOG]   ${RUN_LOG}"
    echo "[MONITOR]   ${MONITOR_LOG}"
    echo "[RESOURCE]  ${RESOURCE_LOG}"
    echo "[DIAG]      ${DIAG_LOG}"
    echo "============================================================"
} | tee -a "${RUN_LOG}"

printf \
    'status=%s\nexit_code=%s\nelapsed_sec=%s\nnode=%s\nthreads=%s\noom_delta=%s\noom_kill_delta=%s\nrun_log=%s\nmonitor_log=%s\nresource_log=%s\ndiagnostic_log=%s\n' \
    "${STATUS}" \
    "${RC}" \
    "${ELAPSED}" \
    "${NODE}" \
    "${THREADS}" \
    "${OOM_DELTA}" \
    "${OOM_KILL_DELTA}" \
    "${RUN_LOG}" \
    "${MONITOR_LOG}" \
    "${RESOURCE_LOG}" \
    "${DIAG_LOG}" \
    > "${STATUS_FILE}"

exit "${RC}"