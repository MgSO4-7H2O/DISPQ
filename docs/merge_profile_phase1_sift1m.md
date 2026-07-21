# Merge Phase 1 Profiling：SIFT1M

状态：阶段一 profiling 已完成。

本报告对应带有微秒级 merge profiling 的 `profiling` 分支实现。此次没有改变 assignment、balanced repartition 或 PQ 编码决策；阶段 1A 仅将 streaming 插入路径按 assignment、encode、commit 拆段，以便获得独立计时。

## 运行命令

```bash
numactl --cpunodebind=0 --membind=0 \
env OMP_NUM_THREADS=48 \
OMP_DYNAMIC=FALSE \
OMP_PROC_BIND=close \
OMP_PLACES=cores \
./build_release/run_eval \
configs/sift/sift.json \
/home/ydy/data/sift/origin \
--prebuilt-index index/origin/sift \
> /tmp/sa_wrq_merge_profile_phase1_sift1m_v3.log 2>&1
```

数据集和配置：

```text
dataset       SIFT1M
dim           128
effective_nlist 4096
pq_m          16
pq_nbits      8
top_r         8
OMP threads   48
NUMA          node0 CPU + memory
merge count   9
```

## 汇总结果

```text
merge_compute_ms          43379.540 ms
PQ re-encode              16338.102 ms
merge wall sum            60924.250 ms
```

新增的内部计时字段使用微秒（`_us`）。聚合结果如下：

| 阶段 | 总耗时 |
|---|---:|
| merge delta-to-main assignment | 39,513.590 ms |
| partition stats | 2.980 ms |
| partition scoring | 0.411 ms |
| Top-r neighbor search | 719.581 ms |
| fetch main records | 606.797 ms |
| repartition pool construction | 453.706 ms |
| repartition distance | 482.118 ms |
| repartition candidate selection | 234.848 ms |
| repartition sorting | 285.438 ms |
| patch preparation | 3,859.783 ms |
| commit | 17,038.772 ms |
| PQ code assignment | 16,338.102 ms |
| PQ list flatten | 35.181 ms |

## 主要计数器

```text
frozen_records          740000
seed_partitions         36683
neighborhoods            10377
main_records_loaded     3918387
pooled_records          4658387
repartitioned_records   4658387
patch_records           4658387
pq_codes_reused         0
pq_codes_reencoded      4658387
```

其中：

```text
pooled_records = main_records_loaded + frozen_records
repartitioned_records = patch_records
```

这确认当前 `balanced_append` 会把大量已有 main records 纳入 neighborhood pool，并对整个 pool 重新生成 replacement lists。

## 结论

### 1. 当前第一瓶颈是 delta assignment

```text
39.514 s / 43.380 s = 91.1%
```

这里的 `merge_delta_to_main_assignment_us` 包含：

- 每个 delta record 扫描全部 4096 个 centroid；
- 距离计算；
- `partial_sort` 保留 top-r；
- balanced append 的 candidate 选择。

因此它不是单纯的 SIMD 距离循环，但已经可以确认 merge compute 的主要耗时在这一条路径，而不是 stats 或 scoring。

### 2. PQ re-encode 是第二瓶颈

```text
16.412 s
```

本次：

```text
pq_codes_reused       0
pq_codes_reencoded    4658387
```

说明当前 commit 对 patched records 全量重新编码。`PQ list flatten` 只有约 35.6 ms，不是主要问题；主要成本在 `NearestCodeword`。

### 3. neighborhood 和 repartition 是中等开销

9 次 merge 中：

```text
Top-r neighbor search          0.723 s
repartition distance           0.481 s
candidate selection             0.234 s
pool construction               0.452 s
sorting                         0.285 s
```

这些内部计时合计约 2.78 s，而 patch preparation 总计 3.86 s，剩余部分是 neighborhood 调度、vector container 操作和其它 patch 构造开销。

因此 centroid neighborhood cache 有收益空间，但不是当前 SIFT1M merge compute 的第一瓶颈。

## 当前结论对应的下一阶段

阶段一没有提前实现任何优化。基于这次观测，后续仍按既定计划执行，但优先级可以明确为：

```text
1. 优化 delta assignment 的全量 centroid 距离/top-r 路径
2. 优化 PQ code reuse，减少 465 万条记录的重复编码
3. 优化 repartition pool 的数据搬运和临时对象
4. 缓存 Top-r centroid neighborhood
```

上述排序只用于阶段二开始后的实现顺序，不代表本次已经修改了这些算法。

## 阶段 1A：streaming 与 merge assignment 分离

第一项执行后，当前 profiling build 额外输出两条独立路径：

```text
[STREAM_PROFILE] delta_ingest_assignment_us
[MERGE_PROFILE] merge_delta_to_main_assignment_us
```

使用同一套 SIFT1M streaming 命令重新运行，结果为：

```text
stream delta assignment       194,927 us / 790,000 records
merge delta-to-main assignment 37,344,540 us / 9 merges
final Recall@100               0.981059
```

流式写入 assignment 平均约为 `2,468 us / 10,000 records`；相比之下，merge assignment 达到约 `37.3 s`。因此当前优化重点应继续放在 frozen delta 到 main 的 assignment，而不是 active delta 的写入 assignment。

本次为获得 assignment 的独立 wall-clock 计时，将插入路径拆成 assignment、encode、commit 三个阶段，增加了 OpenMP barrier。该日志适合做阶段归因，不应直接与未插桩版本比较端到端插入时间。

## 阶段二第一批：assignment scratch、exact top-r 和并行 distance/top-r

在保持 `balanced_append` 顺序和输出结果不变的前提下，完成了第一批 assignment 优化。新的运行命令与阶段一相同，日志为：

```text
/tmp/sa_wrq_merge_profile_phase2_assignment.log
```

累计 9 次 merge 的结果：

| 阶段 | 阶段一 | 阶段二第一批 |
|---|---:|---:|
| merge delta-to-main assignment | 37,344.540 ms | 1,847.543 ms |
| merge compute | 41,212.000 ms | 5,672.457 ms |
| PQ code assignment | 16,353.251 ms | 16,248.493 ms |
| merge wall sum | 59,384.740 ms | 23,153.750 ms |
| Recall@100 | 0.981059 | 0.981059 |

阶段二第一批 assignment 子计时：

```text
distance       1,038.468 ms
exact top-r      372.167 ms
balanced choice  53.922 ms
materialize     142.214 ms
```

本次各 merge 的 assignment 结果与阶段一保持一致：`moved_delta_ratio`、`avg_assignment_dist_ratio`、`max_assignment_dist_ratio`、imbalance 和 patched/append/recluster partition 计数均一致。PQ 仍然是全量重新编码路径：

```text
pq_codes_reused       0
pq_codes_reencoded    4658387
```

PQ reuse 和 repartition 搬运收尾已经在后续阶段完成；routing SoA 不属于本轮范围。

## PQ code reuse 第一阶段

在阶段二 assignment 优化的基础上，对 `CommitPartitionPatch()` 增加保守的 PQ code reuse：旧 record 仍位于同一 partition、旧 code 长度有效且 codebook 没有被 OnlinePQ 修改时直接复用旧 code。delta 新 record、跨 partition 移动 record 和 validity 不满足的旧 code 仍重新编码。

运行日志：

```text
/tmp/sa_wrq_merge_profile_phase3_pq_reuse.log
```

累计 9 次 merge 的结果：

| 指标 | 阶段二第一批 | 阶段三 PQ reuse |
|---|---:|---:|
| PQ code assignment | 16.248493 s | 2.738538 s |
| PQ code copy/reuse bookkeeping | 未拆分 | 0.271136 s |
| `pq_codes_reused` | 0 | 3,893,412 |
| `pq_codes_reencoded` | 4,658,387 | 764,975 |
| merge wall sum | 23.153750 s | 10.132488 s |
| Recall@100 | 0.981059 | 0.981059 |

计数满足：

```text
pq_codes_reused + pq_codes_reencoded = patch_records
```

阶段三没有改变 routing assignment、partition 决策或 recall。生产路径使用 `pq_residual=true` 时，跨 partition 移动 record 不能直接复用旧 residual code；`pq_residual=false` 仅保留为兼容路径。Deep10M 的 `dim=96, nlist=8192` 实际 build 已通过；当前数据目录没有 Deep 768 维数据，因此不将 768 维 merge 写成已完成。

## 阶段四：repartition 搬运精简收尾

在不改变 `balanced_append` 结果的前提下，主 partition records 进入 pooled buffer 时改为 move，进入最终 bucket 时再次直接 move；同时移除 `RepartitionNeighborhood()` 中每条 record 的 legal/healthier candidate vector。

日志：

```text
/tmp/sa_wrq_merge_profile_phase4_cleanup.log
```

累计 9 次 merge：

| 指标 | 阶段三 PQ reuse | 阶段四 cleanup |
|---|---:|---:|
| repartition distance | 424.238 ms | 320.136 ms |
| patch prepare | 3,813.543 ms | 2,845.777 ms |
| merge compute | 5,641.150 ms | 4,947.613 ms |
| merge wall sum | 10,132.488 ms | 9,346.274 ms |
| PQ code assignment | 2,738.538 ms | 2,726.174 ms |
| Recall@100 | 0.981059 | 0.981059 |

`pq_codes_reused=3,893,412`、`pq_codes_reencoded=764,975` 与上一阶段完全一致。

另外，使用仓库现有 Deep10M 数据完成了泛化 build 验证：

```text
config          configs/deep10m/deep10m.json
dim             96
effective_nlist 8192
pq_m            16
pq_nbits        8
pq_residual     true
total build     166,680 ms
```

输出 index 位于 `/tmp/sa_wrq_deep10m_index_cleanup`。该验证覆盖了运行时 `nlist=8192` 和非 SIFT 维度，但没有执行完整 Deep10M streaming merge。

## 验证状态

```text
cmake --build build_release --parallel 8       passed
ctest --test-dir build_release --output-on-failure
4/4 tests passed
```

原始完整日志：

```text
/tmp/sa_wrq_merge_profile_phase1_sift1m_v3.log
```

流式/merge 分离后的日志：

```text
/tmp/sa_wrq_stream_merge_profile_phase1_final.log
```
