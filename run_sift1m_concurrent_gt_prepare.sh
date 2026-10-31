#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
NODE=0
DATASET="/home/ydy/data/sift/origin"
CONFIG="$ROOT/configs/sift_concurrent_prepare.json"
BINARY="$ROOT/build_release/run_concurrent_eval"
GT_FILE="$ROOT/artifacts/sift_concurrent.gt"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/logs/sift1m_concurrent_gt_prepare_${TS}"

cd "$ROOT"

for path in "$CONFIG" "$BINARY"; do
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

mkdir -p "$RUN_DIR" "$ROOT/artifacts"

echo "[INFO] dataset: $DATASET"
echo "[INFO] NUMA node: $NODE"
echo "[INFO] config: $CONFIG"
echo "[INFO] GT output: $GT_FILE"
echo "[INFO] log: $RUN_DIR/prepare.log"

numactl --cpunodebind="$NODE" --membind="$NODE" \
  "$BINARY" "$CONFIG" "$DATASET" \
  2>&1 | tee "$RUN_DIR/prepare.log"

if [[ ! -f "$GT_FILE" ]]; then
  echo "[ERROR] GT artifact was not created: $GT_FILE" >&2
  exit 1
fi

echo "[DONE] GT artifact: $GT_FILE"
echo "Logs: $RUN_DIR"
