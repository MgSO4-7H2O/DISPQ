# DiSPQ — Codex implementation brief: atomic, localized Balanced Merge replacement

**Status:** implementation specification / static-review only. **Target:** `MgSO4-7H2O/DISPQ`, branch `main`. **Baseline (read-only):** `MgSO4-7H2O/DISPQ_backup`, branch `origin`. **Paper:** `SIGMOD27_DiSPQ.pdf` (§4.2.3, Algorithm 1; §4.3, Fig. 4). Language of this specification: English for unambiguous code-agent execution.

> **First-priority instruction:** Make the smallest safe change to the *final Balanced Merge replacement/locking/publication path*, matching the paper's affected-list exclusive access and atomic multi-list replacement. Do not change merge computation, algorithm behavior, metrics, workloads, or configuration. Performance must remain broadly comparable to the baseline; a QPS improvement is **not** required.

## 0. Non-negotiable operating restrictions

1. **Do not build, compile, execute, benchmark, run tests, invoke sanitizers, or use CMake.** The human will compile and test later. You may statically inspect files and use Git read-only commands. Do not create any test source/module, test fixture, benchmark, CI task, script, or synthetic data.
2. **Do not push, open a PR, merge branches, alter remotes, or modify the backup repository.** Work only in the working tree of the official `DISPQ` repository. Leave the final diff ready for human inspection. Do not independently commit unless explicitly requested by the user.
3. **Preserve the exact existing** GT preparation/loading format, logical version advancement, recall formula/version selection, query completion/latency timing, QPS measurement, reranking, candidate budgets, query/search order, batch ingestion, merge triggering, maintenance scheduling, and output fields. No “fairness corrections” to recall, no changing `V_return`, and no dropping/ignoring queries that wait for a lock.
4. Do **not** change Balanced Merge assignment, local neighborhood repair, load scoring, PQ training/encoding, code reuse, `PartitionPatch` semantics or record identity. No changes to PQ memory layout, serialization format, or unrelated code paths. Maintain existing C++17 public API compatibility wherever feasible.
5. **No new external library/dependency, COW/RCU framework, persistent tree, general transaction engine, or whole-Main data copies.** Keep the implementation surgical. Any strictly necessary small internal synchronization helper belongs beside its existing owning code, not in a new standalone module.
6. Never use a global *long-held* index shared/exclusive lock to protect Main search versus Balanced Merge replacement as a disguised fallback. Never claim “per-cluster locking” if a global lock is held across Main scanning or list replacement. A **short coordination lock** protecting only admission state or publication metadata is permitted.
7. **Do not silently compromise correctness to satisfy minimal diff.** If a precise atomic Main+Frozen transition cannot be implemented safely within the constraints, stop, describe the blocker and proposed minimal extension; do not produce a partial unsafe implementation.

## 1. Source-grounded facts (inspect the current checkout again)

The last checked official and backup trees have identical C++ sources, headers, `CMakeLists.txt`, and shared configs (Git blobs identical); only `README.md` and `.gitignore` differ among shared files, while the backup contains additional experimental configs/docs.

Code pointers (approximate line numbers; verify against your checkout):

- `src/index/ivf.cpp`, `IndexData` (~399 onward): `lists`, `doc_ids_by_list`, `pq_codes_by_list`, `pq_codes_soa_by_list`, `doc_to_list`, `ntotal`, `mutation_generation`, quantizers and derived caches.
- `src/index/ivf.cpp`, `SearchSingleQuery` (~1686): computes centroid distances and selects `nprobe` lists; OpenMP loops over probed lists; reads both posting lists and compact/SoA code caches.
- `src/index/ivf.cpp`, `KMeansIVFIndex::Search` (~3275) and `SearchBatch` (~3296): each currently holds `std::shared_lock(mu_)` **for the entire scan**.
- `src/index/ivf.cpp`, `CommitPartitionPatch` (~3615–3940): already prepares/validates records, doc map and caches under a shared lock **before** the final critical section. It then obtains `std::unique_lock(mu_)` and swaps `doc_to_list`, `ntotal` and complete affected-list representations; sets `mutation_generation` and profiling. **Preserve this two-phase structure and every replacement invariant.**
- `src/index/merge.cpp`, `merge_frozen_delta_into_main` (~1600–1700): finishes assignment, rebalancing, patch preparation and PQ work before calling `commit_partition_patch`.
- `apps/run_concurrent_eval.cpp` (~1200–1305): merge worker calls `merge_frozen_delta_into_main` (which commits the Main patch), then **separately** takes `state_mutex`, increments `main_rows`, and clears `state.frozen`.
- `apps/run_concurrent_eval.cpp` (~1380–1475): query snapshots `main`, `frozen`, `active` under `state_mutex`, then searches Main → Frozen → Active; records `current_version` under `version_mutex` near completion.
- `apps/run_concurrent_eval.cpp` (~963): `workload_mutex` is the existing global-rebuild barrier; keep global rebuild semantics unchanged.
- `apps/run_concurrent_eval.cpp`, `ComputeIncrementalPrefixGroundTruth` (~146), `LoadGroundTruth` (~310), `SaveGroundTruth` (~343), recall aggregation (~1923): offline exact top-K for successive insertion-prefix frontiers, serialized as `DISPGT01` and checked on load. This code is **out of scope**.

**Critical mismatch to fix:** A main-index patch becomes visible on Main before `state.frozen.reset()`. A query that already snapshotted the old routes can otherwise search a mixture. A per-partition lock implementation alone cannot provide atomic Main+Frozen visibility.

## 2. Exact paper/semantic invariants

Read the supplied paper, particularly §4.2.3 Algorithm 1 and §4.3 Figure 4, *before editing*.

**Balanced Merge replacement is a complete replacement, not append:** for the affected set `A`, each replacement list contains its final full roster of preserved/relocated resident Main vectors plus incoming sealed-Delta vectors, with the correct target centroid, PQ code, compact list and SoA cache. Unaffected partitions remain byte/logically unchanged. Doc ID ownership and `doc_to_list`, `ntotal` must describe the same published data. Preserve the existing prepared-PQ generation checks and collision validation.

**Atomic multi-partition transaction:** all of `A` become visible together; no query may combine a newly replaced C2 with the pre-replacement C3 in one logical Main view. Do not acquire a few locks, swap those lists, and unblock readers before completing the remaining lists.

**Route-level atomicity:** one query sees a consistent `(Main_old, SealedDelta_present)` or `(Main_new, SealedDelta_retired)` for the merged sealed shard. Active Delta continues following its existing insertion-commit rules; do not make its ordinary insert path globally blocking or accidentally erase a newly published active shard. Keep lifetime of any pinned old routes safe until their readers finish.

**Cluster accessibility matching Fig. 4:** During **final replacement**, queries touching affected clusters wait for the exclusive commit; queries reading **only unaffected clusters** can continue. During background preparation, all published clusters remain searchable. The figure explicitly depicts queries blocked on C2'/C3' while they are locked; do not replace this with a fully nonblocking-old-version COW search design. A query spanning affected and unaffected clusters must be admitted as **one indivisible read transaction** or otherwise provably observe a consistent generation.

**GT/recall stay identical to code:** `GT[v][qid]` is exact top-K over the already implemented insertion-prefix frontier. Merge does **not** advance `current_version` and creates no GT frontier. `QueryRecord.version` is recorded near completion under `version_mutex`; aggregation uses current code's version lookup (including its bounds handling) and `hits/topk`. A query beginning before a newer insert and completing after it is still evaluated against the existing return-time logical version, without changing returned candidates or filtering. Preserve the existing timing boundaries so any lock wait counts toward latency.

## 3. Workload realities and performance goals

- Real Balanced Merge patches in prior experiments touch approximately **99%** of Main partitions. With `nlist=4096`, that is ~4055 partitions. Assume *dense* affected sets are normal, not pathological.
- The original whole-Main lock currently causes only approximately millisecond-scale commit blocking on SIFT100M and has not shown a severe p99 spike; **do not spend complexity to chase a negligible QPS gain**.
- Avoid thousands of contended `shared_mutex::lock()` calls on each query/merge, long sequential lock hold-before-wait patterns, and unnecessary work in the commit critical section.
- Target equivalent correctness and broadly baseline-comparable QPS/p99 latency/merge time under the **unchanged** existing configs. Human will perform measurements. No automatic threshold tuning or configuration edits.

## 4. Recommended minimal architecture: batch-granted read sets + atomic write batch

**Use a single small per-Main admission coordinator, not 4096 independent heavy OS locks**, backed by an affected-partition bitmap plus active-reader accounting (or a demonstrably equivalent efficient internal mechanism). This is *logical per-partition locking*: the coordinator lock is held for bookkeeping only, not during PQ scans, merge data swaps, or condition-variable waits.

### 4.1 Reader admission

1. Resolve the Main route and its probe-set partition IDs with *exactly the existing* centroid/nprobe/routing logic. If an internal helper is needed to share probe selection between admission and `SearchSingleQuery`, use a small local factoring refactor, preserving ties, ordering, and results. **Do not compute a different probe set as a substitute.** The query's acquired set must cover every list it will read.
2. Register all probed partition IDs as **one read batch** under the coordinator. If a pending writer conflicts with any probe, wait on a CV without holding the coordinator mutex; on wake, **retry route snapshot/admission and generation validation**, rather than blindly continuing with an obsolete frozen route.
3. Once admitted, the query holds its read ticket for the Main scan (or the entire cross-route consistent query transaction as required by the route-publication protocol). Use RAII for release on errors/returns; notify the writer when conflicting readers drain.
4. If the query probes no affected partition, it can register and scan while a merge is waiting/committing, provided its chosen route/view remains consistent. Do not allow a reader on unaffected partitions to race with mutating global metadata; audit every field accessed by its scan.
5. Never put a condition-variable wait while holding `state_mutex`, `version_mutex`, `workload_mutex` writer lock, or the coordinator lock. Avoid nested mutex acquisition in inconsistent order. Do not introduce locks inside per-candidate ADC loops.

### 4.2 Writer admission and commit

1. Compute affected IDs from the already prepared `PartitionPatch` and validate IDs/duplicates, preferably once. Represent the set as an efficient bitmap; batch lock acquisition is **one admission transaction**, not thousands of independent blocking mutex operations.
2. Mark all affected IDs **write-pending together**, blocking *new* overlapping read batches (writer preference for finite completion); allow unrelated readers to enter. Wait until pre-existing overlapping read tickets drain; the wait releases the coordinator mutex.
3. Only once the entire affected batch is exclusively granted, validate the prepared patch and execute the existing full-record/full-cache/docmap swaps. No reader may enter any affected list in this entire interval. Publish **all affected lists as one transaction**, not one cluster at a time.
4. Coordinate the same publication point with retirement of exactly the merged frozen shard. The write batch **must remain held** until the Main+Frozen publication protocol has completed. All error exits must leave the old routing and Main data unmodified or follow a clearly defined recoverable state; no partially committed Main after an error.
5. Release pending/exclusive status for the complete affected batch and wake waiting readers. Writer and readers must not starve one another indefinitely.

### 4.3 Route-snapshot/commit linearization: explicit design proof required

This is the hardest part. An index-only coordinator cannot atomically commit `state.frozen.reset()` in the workload. You must choose and implement one narrowly scoped, deadlock-free bridge between the Main patch transaction and the workload's route metadata.

**Preferred approach to evaluate:** a tiny internal commit hook/transaction boundary around the existing `CommitPartitionPatch` publication, used by `run_concurrent_eval.cpp` only, so the maintenance worker can retire the exact sealed shard at the same publish point. Existing other callers must continue working. The hook **must not be invoked while holding a lock that it needs to reacquire**, and must not extend the writer lock across merge preparation or expensive vector/PQ work. If any new internal API is unavoidable, keep it minimal and justify why no narrower implementation exists.

Alternatives are acceptable **only with a concrete proof** of a single query-side admission/publication boundary and no mixed Main/Frozen route, while leaving unrelated workloads intact. In particular:

- A query cannot snapshot `frozen` and then be allowed to inspect `Main_new` after publication without either a valid pinned old Main or a forced route revalidation/retry.
- A reader of unaffected clusters may continue on its pinned old route **only if** it cannot touch newly replaced lists and all relevant references remain live; show why its result corresponds to the pre-merge route.
- Insert batches can concurrently advance the logical version and mutate Active; keep their `version_mutex` and state publication semantics unchanged. Treat `main_rows` and frozen ownership updates consistently with the Main transition.
- Preserve the existing global rebuild barrier, including correct lock ordering against workload and partition coordinator locks.

**Do not perform a destructive Main swap and later retrofit the route state.** If the bridge cannot be proven with the existing function boundaries, stop and report the smallest required API change before implementing.

### 4.4 Global metadata and other index operations

The existing global `mu_` protects many unrelated `IndexData` fields and non-Main operations. **Do not remove it indiscriminately.** Separate the protection of mutable global map/quantizer/bookkeeping from the Main posting-list read tickets. In particular:

- Main `Search`/`SearchBatch` must not hold the existing global shared lock throughout scanning if that forces all clusters to wait for a single whole-Main exclusive commit.
- `doc_to_list`, `ntotal`, `mutation_generation`, `last_patch_profiling`, and PQ compact/SoA caches must never be read concurrently with writes without proper synchronization. Search may not need the doc map, but snapshot/inspection/serialization/helpers do.
- Review `GetPartitionSizes`, `GetPartitionDocIds`, `GetPartitionRecords`, `SnapshotDocIds`, `SnapshotRecords`, `PreparePartitionPQCodes`, `Serialize`, `Deserialize`, `Build`, `Add*`, `CommitPreparedOnlinePQBatch`, `ExportModel`, and version map lifetime. Preserve safety for **other workloads** (`run_eval`, `run_eval_ms`, `build_index`). Prefer leaving their behavior/locking untouched except where required by the shared Main structure.
- Assume resident Main quantizers do not change during Balanced Merge; rebuilding is governed by its existing stop-the-world barrier. If another main-mutating writer is possible, explicitly serialize it with the commit or retain existing safe behavior.
- Do not add a per-cluster mutex to the serialized format or make lock/mutex objects copy/move by accident.

## 5. Required algorithm sketch (not copy-paste code)

```text
Query:
    enter existing rebuild barrier, with unchanged latency start
    obtain a route snapshot and exactly the Main probe IDs
    atomically admit one read ticket for the complete probe set
      (or wait, then resnapshot/revalidate)
    scan Main using the admitted generation
    search Frozen and Active in the original order with pinned routes
    perform existing merge/dedup/rerank and record return-time GT version
    release ticket(s) safely

Merge:
    keep existing freeze + assignment + neighborhood + PQ + patch preparation
    enumerate entire affected partition set A from prepared patch
    atomically mark A write-pending; wait until all overlapping readers finish
    validate patch context, commit complete replacements and associated metadata
    publish new Main and retire exactly this Frozen as one logical operation
    release A, notify readers; keep profiling timestamps semantically comparable
```

**Warning:** This sketch is not proof that the ticket can be released after only Main scan: the appropriate ticket lifetime depends on the Main+Frozen route agreement. Audit and prove it before coding. In a dense (99%) merge, batching and short metadata scope matter more than naive fine-grained lock counts.

## 6. Required static analysis before edits

Write a concise internal analysis in your final response (or a short section in the implementation summary), covering:

1. Lock-order graph: `workload_mutex`, `workload_entry_mutex`, `state_mutex`, `version_mutex`, main `mu_`, coordinator, and any new synchronization. Identify CV predicates and prove no wait cycle.
2. A state machine with `idle → writer_pending(A) → writer_granted(A) → published → idle`, including incoming queries with intersecting and nonintersecting probes and route-snapshot revalidation.
3. Complete list of IndexData fields read by Main search, and whether any writer can mutate them outside Balanced Merge.
4. The **linearization point** of atomic multi-partition Replace + Sealed Delta retirement; describe old vs new query state and how active inserts are handled.
5. The precise scope of edits and why each changed file is necessary. Baseline compatibility for other entry points.

If the source audit contradicts an assumption in this document, prioritize the actual code + paper, explicitly explain the mismatch, and choose the smallest safe equivalent design. Do not implement an unproven or subtly weaker variant.

## 7. Static-review acceptance checklist (do not run anything)

- [ ] All `PartitionPatch.partition_ids` replaced as complete lists; unchanged lists remain unmodified; prepared PQ codes, cache alignments, metadata, doc map and `ntotal` are consistent.
- [ ] No query observes a partially published multi-list patch or mismatched Main+Frozen transition.
- [ ] During actual replacement, overlapping readers wait; disjoint readers are admitted. No multi-list query admits only part of its read set.
- [ ] The design remains correct at ~99% affected partitions and doesn't take thousands of separate blocking locks per query/commit.
- [ ] No global exclusive lock covers main scans or extensive patch preparation; coordinator mutex protects only short admission/publication bookkeeping.
- [ ] No deadlock across background Merge, queries, Active inserts, and Global Rebuild; old route/object lifetimes safe.
- [ ] No alterations to GT preparation/loading, `current_version` and return-time recall, QPS/latency denominators, config, ANN ranking, query budgets, or other workload outputs.
- [ ] No changes to `run_eval` / `run_eval_ms` observable behavior, PQ format, serialization, or unrelated code.
- [ ] No new tests, scripts, builds or runtime execution; describe what the human should test instead.

## 8. Deliverable from Codex

1. Minimal source modifications on official `DISPQ` only; no changes to backup.
2. A short change summary by file and explanation of the correctness protocol, linearization point, and known limitations.
3. Exact Git diff / `git status` summary (read-only Git commands permitted).
4. **Explicit statement: not compiled, not executed, not benchmarked, not tested.**
5. A human-only validation plan (no test files created): compile existing targets, compare same-SIFT1M config/GT and QPS/Recall/p99/merge-profile vs untouched backup, and inspect concurrent commits. The user, not Codex, will run these steps.

## 9. Important priorities when objectives conflict

**Correct paper semantics and no evaluation changes > no deadlocks/data races > minimal scope and compatibility > baseline-comparable performance > theoretical maximization of concurrency.** A proven short localized commit is better than a broad, speculative rewrite. Acknowledge if dense merges effectively block most queries because they actually touch affected partitions; that is consistent with the observed 99% workload and is not a reason to fake smaller patches.
