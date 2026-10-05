#!/usr/bin/env bash

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/ydy/DISPQ-profiling}"
BINARY="${BINARY:-${PROJECT_ROOT}/build_release/run_concurrent_eval}"
DATASET="${DATASET:-/home/ydy/data/sift10M/origin}"
QUERY="${QUERY:-${DATASET}}"
NODE="${NODE:-0}"
PREP_CONFIG="${PREP_CONFIG:-${PROJECT_ROOT}/configs/sift10M_concurrent_prepare.json}"
QUERY_CONFIG="${QUERY_CONFIG:-${PROJECT_ROOT}/configs/sift10M_concurrent_query.json}"

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/sift10M_concurrent_${TIMESTAMP}}"
GT_PATH="${GT_PATH:-${PROJECT_ROOT}/artifacts/sift10M_concurrent.gt}"

fatal() {
    echo "[FATAL] $*" >&2
    exit 1
}

[[ -x "${BINARY}" ]] || fatal "binary not found or not executable: ${BINARY}"
[[ -f "${PREP_CONFIG}" ]] || fatal "prepare config not found: ${PREP_CONFIG}"
[[ -f "${QUERY_CONFIG}" ]] || fatal "query config not found: ${QUERY_CONFIG}"
[[ -d "${DATASET}" ]] || fatal "dataset directory not found: ${DATASET}"
[[ -r "/sys/devices/system/node/node${NODE}/cpulist" ]] \
    || fatal "cannot read CPU list for NUMA node${NODE}"

for command_name in numactl stdbuf; do
    command -v "${command_name}" >/dev/null 2>&1 \
        || fatal "required command not found: ${command_name}"
done
[[ -x /usr/bin/time ]] || fatal "required command not found: /usr/bin/time"

count_cpus() {
    local cpu_list="$1" item start end count=0
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
(( THREADS > 0 )) || fatal "NUMA node${NODE} has no CPUs"

export OMP_NUM_THREADS="${THREADS}"
export OMP_PROC_BIND=close
export OMP_PLACES=threads
export MKL_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export BLIS_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"

mkdir -p "${LOG_DIR}" "${PROJECT_ROOT}/artifacts"
cd "${PROJECT_ROOT}"

run_stage() {
    local stage="$1" config="$2" summary="$3"
    local log="${LOG_DIR}/${stage}.log"
    local resource="${LOG_DIR}/${stage}_resource.log"
    local rc=0

    echo "[START] stage=${stage} node=${NODE} cpus=${NODE_CPU_LIST} threads=${THREADS}"
    echo "[CONFIG] ${config}"
    echo "[DATASET] ${DATASET}"
    echo "[GT] ${GT_PATH}"

    /usr/bin/time -v -o "${resource}" \
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
            "${BINARY}" "${config}" "${DATASET}" "${QUERY}" "${summary}" \
        >"${log}" 2>&1 || rc=$?

    echo "[END] stage=${stage} exit_code=${rc} log=${log} resource=${resource}"
    if (( rc != 0 )); then
        tail -n 80 "${log}" >&2 || true
        return "${rc}"
    fi
}

run_stage prepare "${PREP_CONFIG}" "${LOG_DIR}/prepare_summary.json" \
    || fatal "GT prepare failed; query stage was not started"
run_stage query "${QUERY_CONFIG}" "${LOG_DIR}/query_summary.json" \
    || fatal "concurrent query workload failed"

echo "[DONE] prepare and query completed; summaries are in ${LOG_DIR}"
