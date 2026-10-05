#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/ydy/DISPQ_con}"
DATASET="${DATASET:-/home/ydy/data/sift/drift_fast}"
QUERY="${QUERY:-${DATASET}}"
NODE="${NODE:-0}"
PREP_CONFIG="${PREP_CONFIG:-${PROJECT_ROOT}/configs/sift_drift_concurrent_prepare.json}"
QUERY_CONFIG="${QUERY_CONFIG:-${PROJECT_ROOT}/configs/sift_drift_concurrent_query.json}"
GT_PATH="${GT_PATH:-${PROJECT_ROOT}/artifacts/sift_drift_concurrent.gt}"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/sift_drift_concurrent_${TIMESTAMP}}"

export PROJECT_ROOT DATASET QUERY NODE PREP_CONFIG QUERY_CONFIG GT_PATH LOG_DIR
exec "${PROJECT_ROOT}/run_concurrent_sift10m_node0.sh"
