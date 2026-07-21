# Merge 优化执行计划

状态：主计划的生产路径收尾已完成（`pq_residual=true`）。阶段一 profiling、阶段 1A 流式/merge assignment 分离、阶段二 assignment 优化、阶段三 repartition 搬运精简和阶段四 PQ code reuse 第一阶段均已实现并通过 SIFT1M 验证；Deep10M 的 `dim=96, nlist=8192` 实际 build 也已通过。`pq_residual=false` 仅保留兼容路径，不再作为生产优化目标。

收尾决策：neighborhood cache、routing norm-dot、bounded-overflow repartition 和 routing SoA 不在本轮继续扩展。它们会增加状态或改变分配语义；当前 profile 已经把主要收益集中到 assignment、repartition 搬运和 PQ re-encode 上，后续另立 microbenchmark/优化任务。

目标：在保持当前 `balanced_append` 语义的前提下，降低 merge 的实际耗时，并确保优化适用于不同的 inverted-list 数量、向量维度和 PQ 参数。

## 1. 范围冻结

当前主计划只处理以下问题：

1. merge 内部耗时拆分和可观测性；
2. centroid neighborhood 的重复计算；
3. repartition 中的高维向量复制、临时分配和排序；
4. PQ code 的不必要重新编码；
5. 在不同 `nlist` 和数据集上的验证。

以下方向暂时不加入当前主计划：

- routing centroid SoA；
- routing KMeans 的新 SIMD kernel；
- 改变 `balanced_append` 决策规则；
- 固定 `nlist=4096`、`M=16` 或 `nbits=8` 的专用实现。

routing centroid SoA 只作为文档末尾的后续候选方向保存，等当前主计划完整实现并验证后再单独立项。

## 2. 当前 profile 证据

SIFT1M 当前配置：

```text
dim       = 128
nlist     = 4096
PQ M      = 16
PQ nbits  = 8
top_r     = 8
main rows = 200000
threads   = 48
```

9 次 merge 的汇总结果：

```text
merge_compute_ms       41732.18 ms   70.3%
PQ re-encode           16441.45 ms   27.7%
其它 commit 开销         1211.11 ms    2.0%
merge total            59384.74 ms
```

当前日志中：

```text
patched_partitions ≈ 4082--4096
append_partitions  ≈ 2868--3093
```

`patched_partitions` 表示被 `PartitionPatch` 替换的 partition/list 数量，不表示记录数量。由于当前 `balanced_append` 会对较大的 neighborhood 重新分区，patched list 数量接近 `nlist` 并不等价于所有记录都发生了移动。

因此后续必须重点记录：

```text
pooled_records
main_records_loaded
moved_existing_records
records_repartitioned
pq_codes_reused
pq_codes_reencoded
```

## 3. 泛化约束

所有实现必须使用实际的 `effective_nlist`，不能依赖常量 4096。

需要覆盖：

```text
nlist = 4096
nlist = 8192
更大的 nlist（资源允许时）
```

同时保持 PQ 参数运行时可配置：

```text
M
dsub = dim / M
Ks = 1 << nbits
```

当前基准仍采用：

```text
pq_m = 16
pq_nbits = 8
```

这是基准配置，不是实现限制。

注意：如果 `Build()` 对请求的 `nlist` 做了 `min(requested_nlist, rows)`，日志和指标必须记录实际生效的 `effective_nlist`。

## 4. 阶段一：建立 merge 分层 profiling

### 目标

在不改变算法和数据结构语义的情况下，把当前笼统的 `merge_compute_ms` 拆成可定位的阶段。

### 需要增加的时间项

```text
merge_delta_to_main_assignment_ms
TopRNeighborPartitions_ms
fetch_main_records_ms
repartition_distance_ms
repartition_candidate_selection_ms
repartition_sort_ms
patch_construction_ms
pq_code_assignment_ms
pq_code_copy_or_reuse_ms
pq_list_flatten_ms
```

当前日志中的 `codebook_rebuild_ms` 实际来自 `GetLastPatchPQReencodeMs()`，名称不准确。应拆分为 PQ code assignment、code reuse/copy 和 list flatten，避免把不同工作混为 codebook rebuild。

### 需要增加的计数器

```text
requested_nlist
effective_nlist
frozen_records
seed_partitions
neighborhoods
pooled_records
main_records_loaded
delta_records
patched_partitions
append_partitions
moved_existing_records
records_repartitioned
pq_codes_reused
pq_codes_reencoded
```

### 归一化指标

```text
patched_partitions / effective_nlist
pooled_records / main_records_loaded
moved_existing_records / main_records_loaded
pq_codes_reencoded / patch_records
```

### 主要代码位置

- `src/index/merge.cpp:89`：frozen delta 到 main centroid 的 merge assignment；
- `src/index/merge.cpp:283`：`TopRNeighborPartitions`；
- `src/index/merge.cpp:310`：`RepartitionNeighborhood`；
- `src/index/ivf.cpp:2101`：`CommitPartitionPatch`。

### 阶段验收

- 优化前后 merge 结果一致；
- 每个阶段的耗时之和与 merge wall time 基本对应；
- 能区分距离计算、记录搬运、排序和 PQ 编码耗时；
- 不修改现有 `balanced_append` 决策。

### 阶段 1A：分离流式写入与 merge assignment（已完成）

为避免把两个不同语义的 assignment 混为一个指标，当前实现已经分别输出：

```text
[STREAM_PROFILE] delta_ingest_assignment_us
[STREAM_PROFILE_TOTAL] delta_ingest_assignment_us
[MERGE_PROFILE] merge_delta_to_main_assignment_us
```

其中：

- `delta_ingest_assignment_us`：新记录进入 active delta 时，分配到 delta IVF 自己 list 的时间；
- `merge_delta_to_main_assignment_us`：delta 冻结后，分配到 main IVF partition 的时间；
- `GetLastIngestProfiling()`：提供普通 `AddBatch()` 路径的最近一次插入计时；
- `OnlinePQUpdateStats::insert_assignment_us`：提供 streaming OnlinePQ 路径的每批计时。

SIFT1M 真实运行结果（当前 profiling build）：

```text
stream delta assignment = 194,927 us / 790,000 records
merge delta-to-main assignment = 37,344,540 us / 9 merges
```

这确认流式写入 assignment 不是当前 merge 的主要瓶颈；两者必须保持独立统计。

注意：为了得到不含 encode/commit 的 assignment wall-clock 计时，插入路径被拆成 assignment、encode、commit 三个阶段并增加了 OpenMP barrier。因此该 profiling build 的端到端插入时间不能直接作为未插桩版本的性能基线；它用于阶段归因和后续优化前后同配置比较。

## 5. 阶段二：优化 merge delta-to-main assignment 和重复计算

### 当前执行优先级

基于阶段 1A 的真实计数，阶段二不再从 neighborhood cache 开始，顺序调整为：

1. 拆分 merge assignment 内部的距离计算、top-r 选择和 balanced candidate 选择；
2. 复用 merge assignment 的 per-thread scratch buffer；
3. 在不引入 routing SoA 的前提下优化 norm-dot 距离组织；
4. 最后实现 `TopRNeighborPartitions()` cache。

原因是 SIFT1M 中 merge assignment 占 merge compute 的主要部分，而 9 次 merge 的 Top-r neighbor search 只有约 0.08 秒量级。

### 5.1 缓存 centroid neighborhood

`TopRNeighborPartitions()` 会针对每个 seed partition 重新计算所有 routing centroid 的距离。

普通 merge 中 routing centroid 不会改变，因此缓存：

```text
neighbor_ids[centroid_id][0:top_r]
```

缓存 key 至少包含：

```text
routing_centroid_version
effective_nlist
top_r
```

只有 routing centroid 更新、global rebuild 或相关版本变化时才失效。

该缓存必须对以下规模统一工作：

```text
4096 × top_r
8192 × top_r
更大的 effective_nlist × top_r
```

缓存只保存邻居 ID，不保存与具体 merge 状态相关的 active/claimed 状态。

收尾时暂不实现持久化 cache：当前每个 merge 的 seed/neighborhood 重复查询有限，9 次 SIFT1M merge 的累计 `top_r_neighbor_us` 约为 `0.70 s`，低于 assignment、repartition 和 PQ 路径；保持当前无额外状态的实现更精炼。

### 5.2 复用 assignment scratch buffer

当前每条记录都可能创建：

```cpp
std::vector<std::pair<float, uint32_t>> dists;
std::vector<CandidateState> legal_candidates;
std::vector<CandidateState> healthier_candidates;
```

改为：

- 每个线程复用 scratch buffer；
- 根据运行时 `nlist` 和 `top_r` 预留容量；
- 不在每条记录内重复申请和释放内存；
- 保持原有 `(distance, partition_id)` 的确定性排序规则。

### 5.3 距离计算

在不引入 routing SoA 的前提下，优先使用预计算 norm 的形式：

```text
||x-c||² = ||x||² + ||c||² - 2 x·c
```

其中：

- centroid norm 在 index build/load 时计算；
- 一个输入向量的 norm 只计算一次；
- assignment、neighbor selection 和 merge repartition 使用统一的数值规则。

这一步只优化计算组织，不改变 top-r、gamma、hard cap、lambda 的策略。

收尾时暂不替换为 norm-dot：该路径尚未成为本轮剩余主瓶颈，且会引入另一套数值一致性验证；保留给后续 routing kernel microbenchmark。

### 5.4 已完成：assignment scratch、exact top-r 和分块并行

当前实现已经完成阶段二的第一批低风险改动：

- 预先检查所有 frozen record 的维度，避免在并行循环中返回错误；
- 按运行时 `effective_nlist` 和 `top_r` 计算 chunk 大小，不依赖 `4096`、`128` 或 `8`；
- 对每个 chunk 并行计算所有 record 到 centroid 的距离；
- 使用每个 OpenMP worker 的 scratch 保留 exact top-r，替代每条 record 的 `partial_sort`；
- `balanced_append` 仍按原始 record 顺序串行 replay，保持 `projected_size` 和 assignment 结果；
- candidate 选择直接维护 legal/healthier 的最优状态，不再构造两个临时 candidate vector；
- 新增 `distance/top-r/balance/materialize` 四个微秒级计时项。

该实现使用受限的 chunk working set（distance buffer 加 top-r scratch），避免一次物化完整的
`frozen_records × effective_nlist` 距离矩阵。它没有引入 routing SoA，也没有改变 balanced_append 的规则。

### 阶段验收

- `TopRNeighborPartitions_ms` 在第二次及后续 merge 中显著下降或消失；
- assignment 临时分配次数下降；
- `nlist=4096` 和 `nlist=8192` 均可运行；
- assignment 结果差异为零，或差异被明确统计且 recall 不下降。

阶段二第一批的 SIFT1M 验收结果记录在 `merge_profile_phase1_sift1m.md`：

```text
merge_delta_to_main_assignment_us  37.344540 s -> 1.847543 s
merge_compute_ms                   41.212 s   -> 5.672457 s
Recall@100                         0.981059  -> 0.981059
```

阶段二第一批之后的基线为 `pq_codes_reused=0`；PQ reuse 第一阶段已在后续实现并单独完成 profiling。

## 6. 阶段三：降低 repartition 数据搬运成本（已完成）

当前 `RepartitionNeighborhood()` 的数据路径大致是：

```text
main records
    -> pooled_records
    -> repartitioned buckets
    -> replacement_records
    -> CommitPartitionPatch
```

其中原实现存在多次 `Eigen::VectorXf` 深拷贝和每条 record 的临时 candidate vector。

### 优化内容

1. 对已有记录尽量使用 move，而不是复制高维向量；
2. 移除类似 `VectorRecord out = rec` 的第二次向量复制；
3. 使用 record reference/view 表示：

   ```text
   doc_id
   vector pointer 或 vector owner
   source partition
   version
   ```

4. 只有真正进入 replacement list 的记录才物化新的 `VectorRecord`；
5. 根据主记录数和 delta 数量预先 `reserve`；
6. 每条记录不再创建多个临时候选 vector；
7. 只有必要时才进行最终 `doc_id` 排序。

本次收尾实现：

- `RepartitionNeighborhood()` 接收并 move 主 partition records，避免把已有高维向量先复制到 `pooled_records`；
- record 进入最终 bucket 时直接 move，避免 `VectorRecord out = rec` 的第二次向量复制；
- 用两个最优 `CandidateState` 直接完成 legal/healthier 选择，移除每条 record 的两个 candidate vector。

### 语义要求

本阶段不改变：

```text
nearest partition
top-r 候选
gamma distance cap
hard cap
lambda penalty
balanced_append 的选择顺序
```

只改变对象所有权、内存分配和复制次数。

### 阶段验收

- `pooled_records` 数量不变；
- `moved_existing_records` 数量不变；
- 每个 doc 仍然只存在于一个 list；
- list size 分布和原始实现一致；
- Deep 的 768/1536 维数据没有额外的高峰内存增长；
- `fetch_main_records_ms` 和 `repartition` 相关时间下降。

SIFT1M 收尾 profiling（9 次 merge）：

```text
repartition_distance_us  424,237.5 -> 320,136.3
patch_prepare_us       3,813,543.0 -> 2,845,777.0
merge_compute_ms          5,641.15 -> 4,947.613
merge_ms                 10,132.488 -> 9,346.274
Recall@100                 0.981059 -> 0.981059
```

## 7. 阶段四：PQ code reuse 和必要编码

当前 `CommitPartitionPatch()` 会对 patched list 中的记录重新执行 `M` 次 `NearestCodeword`，即使部分旧记录没有变化。

### 7.1 复用条件

生产路径固定使用 `pq_residual=true`。在 PQ codebook 和 routing centroid 没有变化时：

```text
pq_residual = true:
    记录仍在原 routing list 才能复用旧 code

记录被移动到其它 routing list:
    必须重新编码，因为 residual centroid 改变

delta 新记录:
    没有旧 code，必须编码
```

如果 codebook 版本发生变化，则相关旧 code 全部失效，必须重新编码。

### 7.2 数据结构方向

现有 `ListEntry` 已保存：

```text
doc_id
vector
pq_code
```

需要在构造 patch 或 commit 时建立旧 code 的可查找关系，避免因为 `VectorRecord` 本身没有 `pq_code` 而丢失复用机会。

建议至少支持：

```text
doc_id -> old partition
doc_id -> old pq code
```

查询范围优先限制在 patched partitions，避免为全 index 建立额外的大 map。

### 7.3 计时拆分

```text
pq_code_assignment_ms
pq_codes_reused
pq_codes_reencoded
pq_list_flatten_ms
```

`RebuildListPQCodes()` 仍然需要单独计时，因为它是线性拷贝，不应与 codeword distance 混为一谈。

### 7.4 泛化要求

PQ re-encode 不得写死：

```text
M=16
nbits=8
Ks=256
dsub=8
```

所有循环使用运行时的：

```text
M
dsub
Ks
```

当前基准为 `M=16, nbits=8`，但 Deep 和其它数据集必须继续支持已有维度组合。

### 7.5 已实现：PQ code reuse 第一阶段

当前实现位于 `src/index/ivf.cpp:2198` 的 `CommitPartitionPatch()`：

- 对每个 patched partition 暂存旧 `ListEntry`；
- 建立局部 `doc_id -> old entry` 索引；
- 同一 partition、旧 code 长度等于运行时 `M` 且 codebook validity 未失效时复用；
- delta 新记录、移动记录和无效旧 code 执行实际 `NearestCodeword`；
- OnlinePQ 修改 codebook 后将 `pq_code_reuse_safe` 置为 false，保守避免复用旧 code；
- `pq_code_assignment_us` 只统计真实 codeword assignment，reuse bookkeeping 单独记录。

SIFT1M 结果（9 次 merge）：

```text
pq_codes_reused       3,893,412
pq_codes_reencoded      764,975
pq_code_assignment_us  2,738,538 us
merge wall sum        10,132.488 ms
Recall@100                  0.981059
```

当前第一版对跨 partition record 必须重新编码；`pq_residual=false` 只作为兼容路径保留。SIFT1M 已完成 `nlist=4096` 的真实 merge profiling；Deep10M 已完成 `dim=96, nlist=8192` 的真实 build 验证。当前数据目录没有 Deep 768 维数据，因此不将该项写成已完成。

### 阶段验收

- `pq_codes_reused` 和 `pq_codes_reencoded` 计数正确；
- 对未移动、codebook 未变化的记录不重复编码；
- PQ code 长度仍为 `M`；
- query recall、PQ distance 和 list 内容正确；
- `PQ re-encode` 时间按必要编码量下降。

## 8. 阶段五：可选的 bounded overflow repartition

这一阶段不是第一版默认改动，因为它会改变当前 `balanced_append` 的行为。

只有阶段一到阶段四完成并测量后，才决定是否启用：

```text
delta 先追加到 nearest partition
只有超过 hard cap 的记录才迁移到邻居 partition
```

`hard cap` 必须按实际的 `effective_nlist` 动态计算：

```text
average_list_size = total_records / effective_nlist
hard_cap = average_list_size × hard_cap_ratio
```

不能使用固定的 list 数量或固定记录阈值。

启用前必须确认：

- list imbalance 在可接受范围；
- assignment distance ratio 没有明显恶化；
- recall 不下降；
- merge compute 和 PQ re-encode 的收益足够抵消策略变化风险。

## 9. 基准和验证矩阵

固定运行环境：

```bash
numactl --cpunodebind=0 --membind=0 \
env OMP_NUM_THREADS=48 \
OMP_DYNAMIC=FALSE \
OMP_PROC_BIND=close \
OMP_PLACES=cores \
./build_release/run_eval \
configs/sift/sift.json \
/home/ydy/data/sift/origin \
--prebuilt-index index/origin/sift
```

### SIFT1M

```text
dim=128
nlist=4096
nlist=8192
pq_m=16
pq_nbits=8
```

### Deep

当前仓库可直接使用的 Deep10M 配置为：

```text
dim=96
nlist=8192
pq_residual=true
```

该配置已完成真实 build 验证。当前数据目录没有 Deep 768 维数据；`DBpedia_openai1536` 是独立的 1536 维 cosine 数据集，不能冒充 Deep 768/1536 验收样本。

每个实验至少重复三次，报告中位数，并记录：

```text
build/index load time
merge wall time
merge_compute_ms
PQ re-encode time
total update maintenance time
recall
list imbalance
peak memory
```

## 10. 正确性检查

每个优化阶段都要对比优化前版本：

1. doc 到 list 的映射；
2. 每个 list 的记录数；
3. `doc_id` 唯一性；
4. PQ code 的数量和长度；
5. `pq_codes_by_list` 与 list 内容一致；
6. merge report 中 assignment distance 和 imbalance；
7. query recall 和 latency；
8. 多次运行的确定性。

如果 norm-dot 或其它浮点重排导致 assignment ID 变化，必须记录：

```text
assignment mismatch count
assignment mismatch ratio
recall difference
imbalance difference
```

不能只依据总耗时宣布优化成功。

## 11. 完成标准

主计划完成需要同时满足：

- merge 内部瓶颈已经可以按阶段解释；
- `nlist=4096` 和 `nlist=8192` 都没有特殊分支；
- SIFT1M merge 和 Deep10M（96 维）build 能运行；
- merge 总耗时下降；
- `merge_compute_ms` 或 PQ re-encode 至少一个主要瓶颈得到实质改善；
- 没有 recall 或 list consistency 回归；
- 没有为了性能引入固定维度或固定 PQ 参数限制；
- 所有重要收益都有对应的计数器和 profile 证据。

## 12. 后续候选方向：routing centroid SoA（暂不执行）

该方向单独保存，不属于当前主计划。

当前 routing centroid 是：

```text
C[nlist][dim]          # row-major
```

后续可以增加运行时派生 cache：

```text
C_soa[dim][nlist]
```

它可以让 AVX-512 一次计算 16 个 centroid 的距离。适合：

- 单向量 `NearestCentroid`；
- merge 中扫描全部 `nlist` 并保留 top-r 的 assignment。

但不能直接复用当前只返回单个 block minimum 的 PQ SoA 接口，因为 merge 需要保留每个 centroid 的 top-r 候选。

另外，`RepartitionNeighborhood()` 通常只有约 8 个局部候选，单记录 SoA 的收益未必高；该路径仍应优先优化数据复制和临时分配。

该方向的启动条件是：当前主计划完成后，单独建立 routing row-major 与 routing SoA 的 microbenchmark，并覆盖不同 `nlist`、`dim` 和 top-r。
