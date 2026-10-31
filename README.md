# DISPQ
DiSPQ is a dual-index IVF-PQ system for streaming vector search. It combines online quantization adaptation, hierarchical index maintenance, and efficient query processing to support evolving vector collections without frequent full-index reconstruction.

## Features and Core Modules

DiSPQ is a dual-index IVF-PQ system for streaming vector search. It combines online quantization adaptation, hierarchical index maintenance, and efficient query processing to support evolving vector collections without frequent full-index reconstruction.

### 1. Dual-Index Architecture and Query Processing

DiSPQ maintains a consolidated **Main Index** and a mutable **Delta Index** for recent updates. The Delta Index can transition between Active and Sealed states, allowing insertion and background merging to proceed concurrently.

Queries search the Main and available Delta shards, combine candidates, and optionally perform exact reranking using full-precision vectors.

**Implementation:** `src/index/ivf.cpp`, `apps/run_concurrent_eval.cpp`

### 2. Online PQ Adaptation

The Active Delta supports streaming insertions and incrementally adapts its PQ codebooks to changing data distributions. This reduces quantization mismatch without requiring frequent reconstruction of the Main Index.

**Implementation:** `src/index/ivf.cpp`

### 3. Hierarchical Index Maintenance

DiSPQ organizes maintenance into three complementary levels:

| Level | Mechanism | Purpose |
| --- | --- | --- |
| L1 | **Delta PQ Adaptation** | Adapt quantization to recent data. |
| L2 | **Balanced Merge** | Consolidate Delta vectors and repair affected Main partitions. |
| L3 | **Global Rebuild** | Reconstruct the index under substantial distribution changes. |

Maintenance can be triggered by index size, quantization quality, distribution drift, and partition imbalance.

**Implementation:** `src/index/ivf.cpp`, `src/index/merge.cpp`, `apps/run_concurrent_eval.cpp`

### 4. Balanced Merge

Balanced Merge consolidates a Sealed Delta into the Main Index without rebuilding the entire index. It uses distance- and load-aware assignment, identifies affected partition neighborhoods, and performs local reassignment to improve routing quality and list balance.

The resulting partitions are materialized as **complete replacement inverted lists**, while the Main quantizers remain unchanged.

**Implementation:** `src/index/merge.cpp`, `src/index/ivf.cpp`

### 5. Concurrent Search and Merge

DiSPQ performs expensive merge preparation in the background and coordinates only the final replacement stage.

Queries accessing affected Main partitions are blocked during the commit, while queries accessing unaffected partitions can continue. The replacement of affected partitions and retirement of the corresponding Sealed Delta are coordinated to provide a consistent query-visible transition.

**Implementation:** `src/index/ivf.cpp`, `apps/run_concurrent_eval.cpp`

### 6. Efficient IVF-PQ Computation

DiSPQ incorporates two optimizations to reduce search and maintenance overhead:

- **Computation-Aligned PQ Layouts:** Organize PQ codebooks and encoded vectors for efficient memory access and vectorized distance computation.
- **Reuse-Aware Merge Materialization:** Reuse valid existing PQ codes during Balanced Merge and re-encode only records whose encoding context changes.

These optimizations reduce redundant computation in PQ-based search and incremental maintenance.

**Implementation:** `src/index/ivf.cpp`, `src/index/merge.cpp`, `src/index/soa_kernels.cpp`

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
