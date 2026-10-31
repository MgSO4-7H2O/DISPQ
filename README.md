# DISPQ

DISPQ is a C++17 research prototype for approximate nearest-neighbor (ANN) search over vector data that changes over time. It combines an IVF/PQ index with an online update path, so new vectors can be ingested and searched without rebuilding the full index after every batch.

## What it does

- Builds an IVF/PQ main index, with optional whitening and cosine-distance support.
- Accepts streaming inserts into an active delta index and adapts its product quantizer (OnlinePQ).
- Searches the main index and available delta shards together, with optional exact reranking of candidates.
- Freezes and merges delta shards into the main index as configured. Balanced merge locally reassigns vectors to help repair routing and list-size imbalance; an optional global rebuild is also available.
- Provides evaluation executables that report recall and performance metrics, as well as a concurrent query/insert workload runner.

The project is intended for experimenting with streaming ANN indexing, online PQ adaptation, merge policies, and query/update performance. It is an experimental index and evaluation codebase, not a standalone vector database service.

## Requirements

- CMake 3.16 or newer
- A C++17 compiler
- OpenMP
- Eigen 3 (the repository can use `third_party/eigen` as a submodule when populated, or a system Eigen installation)

## Executables

| Executable | Purpose |
| --- | --- |
| `run_eval` | Evaluate BA workload ANN search, optionally with streaming updates and prebuilt indexes. |
| `run_eval_ms` | Run the msturing-like evaluation workflow. |
| `run_concurrent_eval` | Measure concurrent queries and inserts against a streaming index. |

## Build and run

From the repository root, configure and build the available executables:

```bash
cmake -S . -B build_release -DCMAKE_BUILD_TYPE=Release
cmake --build build_release -j"$(nproc)"
```

### Quick SIFT1M evaluation

The repository includes shell scripts for the SIFT1M **Batch** and **Concurrency** workloads. Each script takes the dataset directory as its only argument. Store the vectors as `<path_to_dataset>/<dataset_name>_base.fvecs` and `<path_to_dataset>/<dataset_name>_query.fvecs`.

```bash
# Batch workload
./run_sift_batch.sh <path_to_dataset>

# Concurrency workload: prepare its ground-truth artifact once, then run queries
./run_sift1m_concurrent_gt_prepare.sh <path_to_dataset>
./run_sift1m_concurrent_query.sh <path_to_dataset>
```

For example, if the files are `/data/sift1m/sift1m_base.fvecs` and `/data/sift1m/sift1m_query.fvecs`, pass `/data/sift1m` to each script. The scripts use the checked-in SIFT configs and write outputs under the repository's `logs/` directory; the concurrency preparation script writes `artifacts/sift_concurrent.gt`.

### Run executables directly

After building, invoke an executable with its config and dataset directory. The loader resolves the `_base.fvecs` and `_query.fvecs` files from that directory. Dataset files use the `fvecs` format, and the config's `dim` must match their vector dimension.

```bash
# Batch workload
./build_release/run_eval configs/sift_batch.json <path_to_dataset>

# Concurrent workload: prepare GT first, then run the query configuration
./build_release/run_concurrent_eval configs/sift_concurrent_prepare.json <path_to_dataset>
./build_release/run_concurrent_eval configs/sift_concurrent_query.json <path_to_dataset>
```

The concurrent query config loads `artifacts/sift_concurrent.gt`; run the preparation command first if that artifact is not present. Concurrent datasets must also provide the workload JSON expected by the selected config. Replace the example config or dataset directory when evaluating other data.

### Paper evaluation environment

The paper's experiments use NUMA binding on node 0 (`numactl --cpunodebind=0 --membind=0`) and use at most 36 threads. The supplied SIFT1M shell scripts apply this NUMA binding; direct executable invocations should use the same binding and thread limit when reproducing the paper's evaluation environment.
