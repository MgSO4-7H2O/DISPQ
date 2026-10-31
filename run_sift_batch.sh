#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODE=0
CONFIG="$ROOT/configs/sift_batch.json"
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <dataset_dir>" >&2
  exit 2
fi
if [[ ! -d "$1" ]]; then
  echo "[ERROR] Dataset directory not found: $1" >&2
  exit 1
fi
DATASET="$(cd "$1" && pwd)"
BINARY="$ROOT/build_release/run_eval"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/logs/sift_batch_${TS}"
RUN_LOG="$LOG_DIR/run.log"

fail() {
  echo "[ERROR] $*" >&2
  exit 1
}

[[ -x "$BINARY" ]] || fail "run_eval binary not found or not executable: $BINARY"
[[ -f "$CONFIG" ]] || fail "config not found: $CONFIG"
[[ -d "$DATASET" ]] || fail "dataset directory not found: $DATASET"
command -v numactl >/dev/null 2>&1 || fail "numactl not found"

mkdir -p "$LOG_DIR"
echo "[START] SIFT batch run on NUMA node $NODE"
echo "[LOG]   $RUN_LOG"

if numactl --cpunodebind="$NODE" --membind="$NODE" \
  "$BINARY" "$CONFIG" "$DATASET" >"$RUN_LOG" 2>&1; then
  echo "[DONE]  SIFT batch run completed"
  echo "[LOG]   $RUN_LOG"
else
  status=$?
  echo "[FAIL]  SIFT batch run exited with status $status"
  echo "[LOG]   $RUN_LOG"
  echo "[ERROR] Last 30 log lines:"
  tail -n 30 "$RUN_LOG" >&2
  exit "$status"
fi
