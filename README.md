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
- Eigen 3 (the repository can use `third_party/eigen` when populated, or a system Eigen installation)

## Build and run

From the repository root, configure and build the available executables:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
```

Run an evaluation with a JSON config and a dataset directory or base `.fvecs` file. The loader resolves the dataset's `_base.fvecs` and `_query.fvecs` files when given a directory. You can also pass a query file or directory explicitly:

```bash
./build/run_eval configs/sift_concurrent_query.json /path/to/sift
# Or specify both vector files:
./build/run_eval configs/sift_concurrent_query.json \
  /path/to/sift/sift_base.fvecs /path/to/sift/sift_query.fvecs
```

Replace the example config and dataset paths with files available in your environment. Dataset files use the `fvecs` format; the config's `dim` must match the vector dimension. The checked-in configs are examples for 128-dimensional SIFT data.

Build a reusable index, then evaluate using it:

```bash
./build/build_index configs/sift_concurrent_query.json /path/to/sift
./build/run_eval configs/sift_concurrent_query.json /path/to/sift --use-prebuilt-index
```

By default, `build_index` writes to `index/<dataset>/<config>/`. Supply a third argument to choose another output directory; pass that directory to `run_eval` with `--prebuilt-index DIR`.

For concurrent query and streaming-insert evaluation, the dataset must also include a workload JSON file ending in `_workload.json` (or pass its path explicitly). The query example config loads ground truth from `artifacts/sift_concurrent.gt`; prepare it first if needed, then run the query workload:

```bash
./build/run_concurrent_eval configs/sift_concurrent_prepare.json /path/to/sift
./build/run_concurrent_eval configs/sift_concurrent_query.json /path/to/sift
```

For a short query run, limit the number of streamed vectors:

```bash
./build/run_concurrent_eval configs/sift_concurrent_query.json /path/to/sift \
  --max-insert-vectors 100000
```

## Executables

| Executable | Purpose |
| --- | --- |
| `run_eval` | Evaluate ANN search, optionally with streaming updates and prebuilt indexes. |
| `build_index` | Build and save a reusable main index and associated metadata. |
| `run_eval_ms` | Run the multi-snapshot evaluation workflow. |
| `run_concurrent_eval` | Measure concurrent queries and inserts against a streaming index. |

Most index and evaluation parameters are set in JSON config files. See `include/common/config.h` for the supported options and `configs/` for examples.
