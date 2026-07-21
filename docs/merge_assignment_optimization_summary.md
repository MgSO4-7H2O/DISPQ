# Merge Assignment 优化阶段总结

## 本轮目标

本轮针对 SIFT1M profiling 中确认的 merge assignment 瓶颈进行优化。目标是降低 frozen delta 到 main IVF partition 的分配耗时，同时不改变当前 `balanced_append` 的决策顺序、top-r 规则和 merge 结果。

基准配置为：

```text
dataset         SIFT1M
dim             128
effective_nlist 4096
pq_m            16
pq_nbits        8
top_r           8
OMP threads     48
NUMA            node0 CPU + memory
```

## 优化前的主要问题

`assign_delta_to_main_centroids_for_merge()` 对每个 frozen record 扫描全部 routing centroid，并且原实现对每条 record：

1. 创建一个保存全部 centroid 距离的动态 vector；
2. 计算 `nlist × dim` 的距离；
3. 使用 `partial_sort` 从 4096 个候选中保留 top-r；
4. 创建 legal/healthier 两个 candidate vector；
5. 复制 `VectorRecord` 并插入 partition bucket。

此外，`balanced_append` 依赖逐条更新的 `projected_size`，因此整个循环不能直接使用无序的 OpenMP `parallel for`。

## 本轮实现

实现位置：

- [src/index/merge.cpp](/home/ydy/SA-WRQ/src/index/merge.cpp:96)
- [include/index/merge.h](/home/ydy/SA-WRQ/include/index/merge.h:67)
- [apps/run_eval.cpp](/home/ydy/SA-WRQ/apps/run_eval.cpp:3304)
- [apps/run_eval_ms.cpp](/home/ydy/SA-WRQ/apps/run_eval_ms.cpp:1621)
- [tests/test_merge.cpp](/home/ydy/SA-WRQ/tests/test_merge.cpp:338)

### 1. 分块并行 distance/top-r

frozen records 被分成运行时计算的 chunk。代码按 distance 数组和 top-r scratch 的总工作集计算 chunk 大小：

```text
chunk working set (distance + top-r scratch) <= 64 MiB
chunk record count <= 16384

bytes_per_record = nlist * sizeof(float) + top_r * sizeof(AssignmentTopCandidate)
chunk_records = max(1, min(16384, 64 MiB / bytes_per_record))
```

每个 chunk 内：

```text
并行计算每条 record 到全部 centroid 的距离
        ↓
并行使用 per-thread scratch 保留 exact top-r
        ↓
按原始 record 顺序串行执行 balanced_append
```

这避免了一次物化完整的 `frozen_records × nlist` 距离矩阵，同时允许最重的距离计算和 top-r 选择使用 OpenMP。

### 2. 移除逐条动态分配

以下逐条分配被移除：

```cpp
std::vector<std::pair<float, uint32_t>> dists;
std::vector<CandidateState> legal_candidates;
std::vector<CandidateState> healthier_candidates;
```

top-r 使用固定容量的 per-thread scratch；balanced candidate 选择直接维护 legal 和 healthier 的当前最优状态。

### 3. 保持 balanced_append 语义

distance/top-r 阶段只产生每条 record 的候选结果；最终的：

```text
projected_size 更新
candidate penalty 计算
partition bucket 插入
```

仍按原始输入顺序执行。因此没有改变 `gamma`、`hard_cap`、`lambda` 或 `projected_size` 的语义。

### 4. 增加微秒级分项计时

新增计时项：

```text
merge_assignment_distance_us
merge_assignment_top_r_us
merge_assignment_balance_us
merge_assignment_materialize_us
```

它们与原有的 `merge_delta_to_main_assignment_us` 一起输出，用于区分距离、top-r、平衡决策和 record 物化的开销。

## SIFT1M 验证结果

运行日志：

```text
/tmp/sa_wrq_merge_profile_phase2_assignment.log
```

9 次 merge 累计结果：

| 指标 | 优化前 | 本轮优化后 |
|---|---:|---:|
| merge delta-to-main assignment | 37.344540 s | 1.847543 s |
| merge compute | 约 41.212 s | 5.672457 s |
| PQ code assignment | 约 16.353 s | 16.248493 s |
| Recall@100 | 0.981059 | 0.981059 |

assignment 子阶段累计耗时：

```text
distance        1.038468 s
top-r           0.372167 s
balanced choice 0.053922 s
materialize     0.142214 s
```

9 次 merge 的以下输出保持一致：

```text
moved_delta_ratio
avg_assignment_dist_ratio
max_assignment_dist_ratio
imbalance_before / imbalance_after
patched_partitions
append_partitions
recluster_partitions
```

验证命令：

```bash
rtk cmake --build build_release --parallel 8
rtk ctest --test-dir build_release --output-on-failure
rtk git diff --check
```

结果：构建成功，4/4 测试成功，diff 检查成功。

## PQ code reuse 第一阶段

在本轮 assignment 优化之后，继续实现了第一版 PQ code reuse。实现位置为 [src/index/ivf.cpp](/home/ydy/SA-WRQ/src/index/ivf.cpp:2198)。

`CommitPartitionPatch()` 对每个目标 partition 临时保存旧 list，并建立该 list 内的 `doc_id -> old entry` 索引：

```text
旧 record 仍在原 partition，旧 code 有效，codebook 未被 OnlinePQ 修改
    -> 复用旧 PQ code

delta 新 record、record 从其它 partition 移入、旧 code 无效
    -> 执行 M 次 NearestCodeword
```

由于当前 SIFT 配置使用 `pq_residual=true`，只复用仍位于原 partition 的旧 record。移动 record 即使 codebook 没有变化，也会重新编码，因为 residual centroid 已经改变。OnlinePQ 修改 codebook 后会保守关闭 reuse，避免使用过期 code。

新增计数和计时：

```text
pq_codes_reused
pq_codes_reencoded
pq_code_copy_or_reuse_us
pq_code_assignment_us  # 只统计真正的 NearestCodeword
```

## PQ reuse 的 SIFT1M 结果

日志：

```text
/tmp/sa_wrq_merge_profile_phase3_pq_reuse.log
```

与 assignment 优化后的上一阶段相比：

| 指标 | assignment 阶段 | PQ reuse 阶段 |
|---|---:|---:|
| PQ code assignment | 16.248493 s | 2.738538 s |
| PQ code copy/reuse bookkeeping | 未拆分 | 0.271136 s |
| pq_codes_reused | 0 | 3,893,412 |
| pq_codes_reencoded | 4,658,387 | 764,975 |
| merge wall sum | 23.153750 s | 10.132488 s |
| Recall@100 | 0.981059 | 0.981059 |

计数满足：

```text
pq_codes_reused + pq_codes_reencoded = patch_records
```

本阶段仍没有固定 `nlist`、`dim`、`M` 或 `nbits`。生产路径使用 `pq_residual=true` 时，跨 partition record 的 residual centroid 已改变，不能直接复用旧 code；`pq_residual=false` 仅保留为兼容路径。

## 上一轮计划收尾：repartition 搬运精简

为了减少 merge 中已有高维 record 的重复复制，`RepartitionNeighborhood()` 现在：

- move 主 partition record 到 pooled buffer；
- 将 pooled record 直接 move 到最终 bucket；
- 用两个最优 candidate state 替代每条 record 的 legal/healthier 临时 vector。

SIFT1M 收尾 profiling：

```text
repartition_distance_us  424,237.5 -> 320,136.3
patch_prepare_us       3,813,543.0 -> 2,845,777.0
merge_compute_ms          5,641.15 -> 4,947.613
merge_ms                 10,132.488 -> 9,346.274
Recall@100                 0.981059 -> 0.981059
```

这部分没有改变 `balanced_append`、top-r、gamma、hard cap 或 lambda 的语义。

## 当前剩余瓶颈

收尾后的 merge 主要耗时集中在 patch preparation、commit 和仍然必须执行的 PQ code assignment。对 `pq_residual=true` 而言，跨 partition record 的重新编码是数学上必要的，不再把它误判为可直接 reuse 的代码缺陷。

基线实现的计数为：

```text
pq_codes_reused       = 0
pq_codes_reencoded    = 4,658,387
```

第一版 reuse 后变为：

```text
pq_codes_reused       = 3,893,412
pq_codes_reencoded    = 764,975
```

第一版 reuse 后，当前 merge 只对未复用的 records 执行 `M` 次 `NearestCodeword`：

```text
已有 record 未改变 partition 且 codebook 未改变
    -> 已在第一阶段复用已有 PQ code

delta 新 record、record 移动 partition 或 codebook validity 不满足
    -> 当前仍重新编码
```

本轮没有引入 routing centroid SoA，也没有固定 `nlist=4096`、`dim=128`、`M=16` 或 `nbits=8` 的专用分支。减少 record 移动和优化剩余 generic kernel 属于下一轮独立工作。
