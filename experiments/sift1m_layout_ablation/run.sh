#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
experiment_dir="$repo_dir/experiments/sift1m_layout_ablation"

usage() {
  cat >&2 <<'EOF'
Usage:
  experiments/sift1m_layout_ablation/run.sh DATASET_DIR [RESULT_ROOT]

Environment overrides:
  OMP_NUM_THREADS=36   OpenMP worker count (default: 36)
  NUMA_NODE=0          NUMA node for CPU and memory binding (default: 0)
  CPU_LIST=...         Explicit physical CPU list; by default the script picks
                       the first OMP_NUM_THREADS CPUs available on NUMA_NODE.
  RESULT_ROOT=...      Output directory (default: a timestamped directory under
                       result/sift1M/drifted/)
  BUILD_DIR=...        CMake build directory (default: build_release)
  RUN_EVAL_BIN=...     run_eval binary (default: BUILD_DIR/run_eval)
  BUILD_IF_MISSING=1   Build run_eval if it does not exist
  BUILD_JOBS=...       Parallel build jobs (default: nproc)
EOF
}

dataset_dir="${1:-${DATASET_DIR:-}}"
if [[ -z "$dataset_dir" ]]; then
  usage
  exit 2
fi
if [[ $# -gt 2 ]]; then
  usage
  exit 2
fi
dataset_dir="$(cd -- "$dataset_dir" 2>/dev/null && pwd)" || {
  echo "Dataset directory does not exist: ${1:-${DATASET_DIR:-}}" >&2
  exit 2
}

threads="${OMP_NUM_THREADS:-36}"
numa_node="${NUMA_NODE:-0}"
if [[ ! "$threads" =~ ^[1-9][0-9]*$ ]]; then
  echo "OMP_NUM_THREADS must be a positive integer, got: $threads" >&2
  exit 2
fi
if [[ ! "$numa_node" =~ ^[0-9]+$ ]]; then
  echo "NUMA_NODE must be a non-negative integer, got: $numa_node" >&2
  exit 2
fi

for required_file in sift1M_drifted_base.fvecs sift1M_drifted_query.fvecs; do
  if [[ ! -f "$dataset_dir/$required_file" ]]; then
    echo "Missing dataset file: $dataset_dir/$required_file" >&2
    exit 2
  fi
done

if ! command -v numactl >/dev/null 2>&1; then
  echo "numactl is required for CPU and NUMA memory binding." >&2
  exit 2
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required to select CPUs local to the requested NUMA node." >&2
  exit 2
fi

# Select exactly one distinct logical CPU per OpenMP worker. Respect both the
# node topology and the cpuset/affinity inherited by this process (e.g. cgroups).
cpu_list="$(NUMA_NODE="$numa_node" OMP_NUM_THREADS="$threads" \
  CPU_LIST="${CPU_LIST:-}" python3 - <<'PY'
import os
import sys
from pathlib import Path


def parse_cpu_list(value):
    cpus = set()
    for item in value.strip().split(","):
        if not item:
            continue
        if "-" in item:
            first, last = (int(part) for part in item.split("-", 1))
            if first > last:
                raise ValueError(f"invalid CPU range: {item}")
            cpus.update(range(first, last + 1))
        else:
            cpus.add(int(item))
    return cpus


def format_cpu_list(cpus):
    ordered = sorted(cpus)
    ranges = []
    first = previous = ordered[0]
    for cpu in ordered[1:]:
        if cpu == previous + 1:
            previous = cpu
            continue
        ranges.append(str(first) if first == previous else f"{first}-{previous}")
        first = previous = cpu
    ranges.append(str(first) if first == previous else f"{first}-{previous}")
    return ",".join(ranges)


node = os.environ["NUMA_NODE"]
threads = int(os.environ["OMP_NUM_THREADS"])
node_file = Path(f"/sys/devices/system/node/node{node}/cpulist")
if not node_file.is_file():
    sys.exit(f"NUMA node {node} is unavailable ({node_file} not found)")

node_cpus = parse_cpu_list(node_file.read_text())
allowed_cpus = set(os.sched_getaffinity(0))
explicit = os.environ.get("CPU_LIST", "").strip()

try:
    if explicit:
        selected = parse_cpu_list(explicit)
        if len(selected) != threads:
            sys.exit(f"CPU_LIST must contain exactly {threads} distinct CPUs; got {len(selected)}")
        outside_node = selected - node_cpus
        outside_affinity = selected - allowed_cpus
        if outside_node:
            sys.exit(f"CPU_LIST contains CPUs outside NUMA node {node}: {format_cpu_list(outside_node)}")
        if outside_affinity:
            sys.exit(f"CPU_LIST contains CPUs unavailable to this process: {format_cpu_list(outside_affinity)}")
    else:
        available = sorted(node_cpus & allowed_cpus)
        if len(available) < threads:
            sys.exit(
                f"NUMA node {node} has only {len(available)} CPUs available to this process; "
                f"{threads} requested. Set CPU_LIST explicitly or lower OMP_NUM_THREADS."
            )
        selected = set(available[:threads])
except ValueError as error:
    sys.exit(str(error))

print(format_cpu_list(selected))
PY
)"

build_dir="${BUILD_DIR:-$repo_dir/build_release}"
if [[ "$build_dir" != /* ]]; then
  build_dir="$repo_dir/$build_dir"
fi
binary="${RUN_EVAL_BIN:-$build_dir/run_eval}"
if [[ "$binary" != /* ]]; then
  binary="$repo_dir/$binary"
fi
if [[ ! -x "$binary" && "${BUILD_IF_MISSING:-0}" == "1" ]]; then
  build_jobs="${BUILD_JOBS:-$(nproc)}"
  cmake -S "$repo_dir" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo -DANN_ENABLE_PROFILING=ON
  cmake --build "$build_dir" --target run_eval --parallel "$build_jobs"
fi
if [[ ! -x "$binary" ]]; then
  cat >&2 <<EOF
run_eval binary not found or not executable: $binary
Build it on this machine with:
  cmake -S "$repo_dir" -B "$repo_dir/build_release" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DANN_ENABLE_PROFILING=ON
  cmake --build "$repo_dir/build_release" --target run_eval --parallel
Or rerun with BUILD_IF_MISSING=1.
EOF
  exit 2
fi

run_id="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
result_root="${2:-${RESULT_ROOT:-$repo_dir/result/sift1M/drifted/layout_ablation_$run_id}}"
mkdir -p "$result_root"
result_root="$(cd -- "$result_root" && pwd)"

run_names=(
  sift1m_layout_aos_record
  sift1m_layout_dim_record
  sift1m_layout_aos_subq
  sift1m_layout_dim_subq
)

# Refuse to overwrite any existing run directory so repeated invocations keep
# their raw measurements separate.
for run_name in "${run_names[@]}"; do
  result_dir="$result_root/$run_name"
  if [[ -d "$result_dir" ]] && find "$result_dir" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    echo "Refusing to overwrite non-empty result directory: $result_dir" >&2
    echo "Choose a new RESULT_ROOT or RUN_ID." >&2
    exit 2
  fi
done

cd "$repo_dir"
for run_name in "${run_names[@]}"; do
  config_path="$experiment_dir/configs/${run_name}.json"
  result_dir="$result_root/$run_name"
  mkdir -p "$result_dir"
  cp "$config_path" "$result_dir/config.json"
  {
    printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
    printf 'hostname=%s\n' "$(hostname)"
    printf 'dataset_dir=%s\n' "$dataset_dir"
    printf 'result_root=%s\n' "$result_root"
    printf 'run_name=%s\n' "$run_name"
    printf 'binary=%s\n' "$binary"
    printf 'omp_num_threads=%s\n' "$threads"
    printf 'numa_node=%s\n' "$numa_node"
    printf 'cpu_list=%s\n' "$cpu_list"
    printf 'git_revision=%s\n' "$(git -C "$repo_dir" rev-parse HEAD 2>/dev/null || echo unknown)"
    printf 'git_dirty=%s\n' "$(if [[ -n "$(git -C "$repo_dir" status --porcelain 2>/dev/null)" ]]; then echo yes; else echo no; fi)"
  } >"$result_dir/run_meta.txt"

  printf 'Running %s | threads=%s | CPUs=%s | NUMA node=%s\n' \
    "$run_name" "$threads" "$cpu_list" "$numa_node"
  env OMP_NUM_THREADS="$threads" \
      OMP_DYNAMIC=FALSE \
      OMP_PLACES=threads \
      OMP_PROC_BIND=TRUE \
      OMP_DISPLAY_ENV=VERBOSE \
      numactl --physcpubind="$cpu_list" --membind="$numa_node" \
      "$binary" "$config_path" "$dataset_dir" --fresh-index \
      >"$result_dir/run.log" 2>&1
  printf 'Completed %s | output=%s\n' "$run_name" "$result_dir"
done

printf '\nAll four runs completed. Result root: %s\n' "$result_root"
printf 'Summarize with:\n  python3 "%s/summarize.py" "%s"\n' "$experiment_dir" "$result_root"
