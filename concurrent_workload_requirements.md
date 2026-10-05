# Concurrent Streaming Workload Requirements

## 1. Objective

Add a new concurrent streaming workload for DISPQ with minimal code changes and maximum reuse of existing interfaces.

Requirements:

- Reuse existing IVF / OnlinePQ / merge / whitening / rerank / dataset-loading logic.
- Preserve current batch-insert and merge-trigger semantics.
- Run query, insert, and maintenance concurrently.
- Decouple GT computation from the online workload.
- Keep implementation simple and low-risk.
- Do not change IVF/PQ data layout, memory order, serialization format, or merge patch format.
- Do not generate test files.
- Codex does **not** need to compile or run the code in the generation environment.

Recommended new executable:

```text
apps/run_concurrent_eval.cpp
```

Reuse existing helpers from `run_eval.cpp` / `run_eval_ms.cpp` instead of creating a new framework.

---

## 2. Dataset input and path resolution

Follow the existing workload evaluator behavior.

CLI:

```text
run_concurrent_eval <config.json> <dataset_dir_or_base_file> [query_or_dir] [summary.json]
```

Reuse existing logic for:

```cpp
ResolveFvecsPath(..., "_base.fvecs")
ResolveFvecsPath(..., "_query.fvecs")
ann::eval_memory::MakeFvecsBaseSource(...)
LoadFvecs(...)
```

Rules:

- base vectors are consumed in deterministic row order;
- respect `config.max_queries`;
- if query path is omitted, use the same parent-directory lookup behavior as current workload code;
- do not add a new dataset format or parser.

---

## 3. Runtime workflow

Use three logical streams:

```text
Query:
q0 q1 q2 ... qN q0 q1 ...                  --->

Insert:
      B0        B1        B2        B3       --->
      |         |         |         |
      commit    commit    commit    commit
      check     check     check     check

Maintenance:
                         frozen A -> main    --->
```

Global rebuild is a separate stop-the-world operation.

### Query stream

Use the fixed query set as a reusable query pool.

Each query is an independent request:

```cpp
qid = next_query.fetch_add(1) % query_count;
```

Use multiple closed-loop query workers.

Each query searches current routes in existing evaluator order:

```text
main
frozen, if present
active, if present
```

Reuse existing route-result merge, deduplication, and rerank behavior.

Do not modify returned candidates for evaluation purposes.

### Insert stream

Keep existing batch insert semantics.

Each batch:

```text
transform/build records
    ->
AddWithOnlinePQ(...) or existing sliding-window variant
    ->
successful commit
    ->
advance logical version
    ->
update active metadata
    ->
check merge trigger once
    ->
check global rebuild trigger once
```

Do **not** check maintenance per vector.

Keep using existing `stream_batch_size`.

Optional pacing:

```text
batch_period = batch_size / target_insert_vecps
```

If `target_insert_vecps == 0`, run unpaced.

If insertion falls behind the target schedule, do not drop data; record lag/backlog.

### Freeze / active lifecycle

Insert worker owns active-delta lifecycle.

On trigger:

```text
active A -> frozen A
active = empty/pending
```

Reuse existing delta-training / activation logic to create active B.

First implementation supports only:

```text
main + one frozen + one active
```

If frozen already exists and active reaches trigger again:

- do not create a second frozen shard;
- keep active growing;
- record maintenance backpressure / rows-over-trigger.

### Maintenance stream

Use one logical maintenance worker.

When frozen exists, call existing:

```cpp
merge_frozen_delta_into_main(...)
```

Do not duplicate merge logic.

After successful merge:

- publish updated main metadata;
- clear only the same frozen shard that was processed.

### Global rebuild

Global rebuild may block query, insert, and maintenance.

A workload-level `std::shared_mutex` is sufficient:

- query / insert / merge: shared access;
- global rebuild: unique access.

Do not implement COW / RCU for global rebuild in this task.

---

## 4. Existing interfaces to reuse

Prefer existing public APIs:

```cpp
IVFIndex::Build(...)
IVFIndex::Search(...)
IVFIndex::AddWithOnlinePQ(...)
IVFIndex::AddWithOnlinePQSlidingWindow(...)
IVFIndex::AddWithOnlinePQSlidingWindowRecords(...)
merge_frozen_delta_into_main(...)
```

Reuse current code for:

- main build;
- delta build / activation;
- OnlinePQ options;
- merge trigger evaluation;
- merge options;
- global rebuild;
- whitening;
- exact rerank;
- route result merging;
- output/counters where practical.

Do not introduce new index APIs unless strictly necessary.

---

## 5. Query parallelism

The old workload uses `SearchBatch()` and parallelizes across query vectors.

The new workload uses individual query requests, so add intra-query parallelism across IVF probed lists.

Keep:

```cpp
IVFIndex::Search(...)
```

unchanged at the public API level.

Minimal internal change in `SearchSingleQuery()`:

```text
select nprobe lists

parallel over probed lists:
    thread-local LUT / scan buffer
    thread-local top-k

reduce thread-local top-k
```

Requirements:

- no index-layout changes;
- no PQ-code reorder;
- no per-candidate locking;
- thread-local heaps/buffers;
- query-level routing data shared read-only;
- prefer dynamic scheduling across probed lists;
- route-level parallelism is not required initially;
- avoid nested OpenMP oversubscription.

Expose:

```text
query_workers
query_threads_per_request
```

so a fixed search CPU budget can later evaluate:

```text
16 x 1
8 x 2
4 x 4
2 x 8
1 x 16
```

---

## 6. Logical version and GT

### Logical version

Advance one logical version after every successfully committed insert batch.

Example:

```text
V0 = initial searchable state
V1 = after batch 1 commit
V2 = after batch 2 commit
...
```

Only insert/delete logical commits change the version.

Merge and global rebuild do **not** change it.

For append-only workloads, each version maps to one committed row frontier.

### GT preparation

GT must be computed offline and reusable.

Conceptual layout:

```text
GT[logical_version][query_id] -> exact top-K doc IDs
```

Do not compute GT in the concurrent query path.

Support a simple GT mode:

```text
off
prepare
load
```

`prepare`:

- replay logical committed states;
- compute exact top-K for every query at each logical version;
- write one reusable GT artifact;
- exit.

`load`:

- validate and load the GT artifact;
- use it only after/during lightweight lookup for recall aggregation;
- never perform exact search during the concurrent workload.

GT artifact metadata should include at least:

```text
dataset identity
query identity
query count
topk
initial committed rows
stream batch size
logical version count
committed row frontier per version
```

The same GT must be reusable across different:

- query worker counts;
- query intra-request thread counts;
- insert / maintenance thread allocations;
- merge durations;
- future locking / COW / RCU implementations.

---

## 7. Recall semantics

Use one primary Recall definition:

\[
Recall@K(q) =
\frac{|R_K(q) \cap GT_K(q, V_{return})|}{K}
\]

Where:

- `R_K(q)` = actual top-K returned by the system, unchanged;
- `V_return` = logical committed version observed when the query finishes;
- `GT_K(q, V_return)` = exact top-K over the logical dataset committed at `V_return`.

Important:

- do not filter returned candidates by request-start rows;
- if an insert batch commits while the query is running, it belongs to the GT if it committed before query completion;
- if the query did not observe the newly committed vectors, the resulting recall loss is part of the measured freshness/search quality;
- merge does not affect GT version.

---

## 8. Latency semantics

For every query:

\[
Latency = t_{return} - t_{dispatch}
\]

Latency includes:

- global-rebuild barrier wait;
- IVF lock wait;
- main/frozen/active route search;
- probe/list scan;
- route-result merge / deduplication;
- configured exact rerank;
- blocking caused by OnlinePQ or merge commit.

Latency excludes:

- GT computation;
- GT preparation;
- summary serialization;
- debug/log flush.

Report at least:

```text
mean
p50
p95
p99
p99.9
max
```

---

## 9. QPS semantics

Primary QPS:

\[
QPS =
\frac{\text{completed query requests}}
     {\text{concurrent measurement wall-clock time}}
\]

Rules:

- use real wall time;
- maintenance and global-rebuild blocking remain in the denominator;
- do not use summed query CPU time;
- do not include initial index construction;
- do not include GT preparation.

Measurement interval:

```text
start: immediately before concurrent workers begin
stop: after insert stream ends and required maintenance drains
```

Query workers stop at the end of this interval.

---

## 10. Output

Write one summary JSON, reusing existing output style/helpers where practical.

Required workload/config fields:

```text
query_workers
query_threads_per_request
insert_threads
maintenance_threads
target_insert_vecps
stream_batch_size
query_count
topk
nprobe
initial_rows
final_committed_rows
measurement_wall_ms
```

Required query metrics:

```text
completed_queries
qps
recall_at_k
latency_mean_ms
latency_p50_ms
latency_p95_ms
latency_p99_ms
latency_p999_ms
latency_max_ms
main_route_queries
frozen_route_queries
active_route_queries
```

Required insert metrics:

```text
inserted_vectors
committed_batches
actual_insert_vecps
insert_batch_mean_ms
insert_batch_p99_ms
max_insert_lag_ms
```

Required maintenance metrics:

```text
merge_count
merge_total_ms
merge_compute_ms
merge commit/profile fields already available
maintenance_busy_ratio
active_rows_over_trigger
global_rebuild_count
global_rebuild_total_ms
```

A compact per-query record may be retained until workload completion:

```cpp
struct QueryRecord {
    uint32_t query_id;
    uint64_t logical_version_at_return;
    double latency_ms;
    std::vector<DocId> returned_topk;
};
```

Use these records for recall aggregation after concurrent execution.

---

## 11. Config fields

Reuse existing IVF/PQ/streaming/merge/global-rebuild/rerank fields.

Add only the concurrent-workload-specific fields:

```cpp
bool concurrent_workload_enable{false};

uint32_t concurrent_query_workers{1};
uint32_t concurrent_query_threads_per_request{1};

uint32_t concurrent_insert_threads{1};
uint32_t concurrent_maintenance_threads{1};

// 0 = unpaced
double concurrent_target_insert_vecps{0.0};

// off / prepare / load
std::string concurrent_gt_mode{"off"};
std::string concurrent_gt_path;
```

Do not add duplicate config for behavior already controlled by existing fields such as:

```text
stream_batch_size
merge_trigger_*
merge_score_*
merge_assignment_*
online_pq_*
global_rebuild_*
exact_rerank_*
```

Nested OpenMP should be disabled. The workload should explicitly apply the intended thread budget for each role.

---

## 12. Runtime state ownership

Use a small workload-state mutex only for pointer/metadata snapshot and publication.

Conceptually:

```cpp
struct RuntimeState {
    std::shared_ptr<IVFIndex> main;
    VersionSet main_versions;

    std::optional<DeltaShard> frozen;
    std::optional<DeltaShard> active;

    uint64_t logical_version;
    uint32_t committed_rows;
};
```

Rules:

- never hold the state mutex across `Search`, OnlinePQ insertion, or full merge work;
- query copies route `shared_ptr`s before searching;
- insert worker owns active lifecycle;
- maintenance worker owns the current frozen shard;
- global rebuild publishes replacement state under stop-the-world protection.

Do not add evaluation-side locking that hides the current index-lock behavior.

---

## 13. Non-goals

Do not implement in this task:

- index-layout redesign;
- PQ-layout redesign;
- partition-level locks;
- COW;
- RCU;
- epoch reclamation;
- multiple frozen shards;
- per-vector maintenance checks;
- online GT computation;
- route-level parallel search unless strictly necessary;
- general-purpose scheduler/runtime;
- test files.

The workload/API boundary should remain stable so future concurrency work can replace the bottom-level locking strategy while reusing the same:

```text
workload
GT artifact
Recall definition
QPS definition
Latency definition
```

---

## 14. Expected lifecycle

```text
1. Parse config.
2. Resolve/load base/query using existing workload path logic.
3. Build/bootstrap main and initial delta using existing code.
4. If GT mode == prepare:
      generate reusable GT artifact;
      exit.
5. If GT mode == load:
      validate/load GT artifact.
6. Start measurement clock.
7. Start query workers, insert worker, maintenance worker.
8. Query workers continuously cycle through fixed query pool.
9. Insert worker commits fixed-size batches and checks maintenance once per batch.
10. Freeze active when triggered.
11. Maintenance merges frozen in background.
12. Global rebuild, when triggered, runs stop-the-world.
13. When input stream ends:
      drain insert work;
      finish required maintenance;
      stop query workers.
14. Stop measurement clock.
15. Compute recall from stored query results + precomputed GT.
16. Write summary JSON.
```

Favor direct reuse and clear ownership over abstraction-heavy redesign.
