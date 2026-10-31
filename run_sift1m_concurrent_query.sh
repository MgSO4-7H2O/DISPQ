#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
NODE=0
DATASET="/home/ydy/data/sift/origin"
QUERY_CONFIG="$ROOT/configs/sift_concurrent_query.json"
BINARY="$ROOT/build_release/run_concurrent_eval"
PLOT_SCRIPT="$ROOT/scripts/plot_concurrent_timeline.py"
GT_FILE="$ROOT/artifacts/sift_concurrent.gt"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/logs/sift1m_concurrent_${TS}"
SUMMARY="$RUN_DIR/query_summary.json"
TIMELINE="$RUN_DIR/query_summary_timeline.png"
PYTHON="${PYTHON:-python3}"

if [[ -x "$ROOT/.venv/bin/python3" ]]; then
  PYTHON="$ROOT/.venv/bin/python3"
elif [[ -x "/home/ydy/DISPQ_con/.venv/bin/python3" ]]; then
  PYTHON="/home/ydy/DISPQ_con/.venv/bin/python3"
fi

cd "$ROOT"

if [[ ! -f "$GT_FILE" ]]; then
  echo "[ERROR] GT file not found: $GT_FILE" >&2
  echo "Run run_sift1m_concurrent_gt_prepare.sh first." >&2
  exit 1
fi
for path in "$QUERY_CONFIG" "$BINARY" "$PLOT_SCRIPT"; do
  if [[ ! -f "$path" ]]; then
    echo "[ERROR] Required file not found: $path" >&2
    exit 1
  fi
done
if [[ ! -x "$BINARY" ]]; then
  echo "[ERROR] Binary is not executable: $BINARY" >&2
  exit 1
fi
if [[ ! -d "$DATASET" ]]; then
  echo "[ERROR] Dataset directory not found: $DATASET" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"

echo "[INFO] dataset: $DATASET"
echo "[INFO] NUMA node: $NODE"
echo "[INFO] config: $QUERY_CONFIG"
echo "[INFO] GT: $GT_FILE"
echo "[INFO] output: $RUN_DIR"

numactl --cpunodebind="$NODE" --membind="$NODE" \
  "$BINARY" \
  "$QUERY_CONFIG" \
  "$DATASET" \
  "$DATASET" \
  "$SUMMARY" \
  2>&1 | tee "$RUN_DIR/query.log"

echo "[2/2] Plotting concurrent timeline..."
numactl --cpunodebind="$NODE" --membind="$NODE" \
  "$PYTHON" "$PLOT_SCRIPT" \
  "$SUMMARY" \
  --output "$TIMELINE" \
  2>&1 | tee "$RUN_DIR/plot.log"

echo "[DONE] Summary: $SUMMARY"
echo "Timeline: $TIMELINE"
echo "Logs: $RUN_DIR"
