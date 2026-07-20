## SA-WRQ: Streaming ANN with OnlinePQ Delta and Balanced Merge

SA-WRQ 当前实现的是一套面向流式向量检索的双层索引框架：

- `main index` 负责稳定的大规模检索
- `delta index` 负责接收最新流数据
- `OnlinePQ` 负责 delta 内的轻量在线适配
- `balanced merge` 负责把 delta 中已经稳定的数据并回 main，同时做局部 IVF 结构修复

这份 `README` 以当前代码为准。  
如果你之前看过“三层更新架构”的旧图，请以本文描述为准。

## 当前版本和原始设想的区别

最初的设想是三层更新：

1. 轻微失配：更新 PQ
2. 中等漂移：单独更新 delta 的 IVF coarse
3. 严重失配：merge / rebuild

现在代码已经收敛成更简洁的版本：

- `delta` 在线阶段只做 `OnlinePQ`
- 不再单独维护一套“delta coarse 在线刷新”机制
- 原来关于 IVF 路由修复、局部重分桶的思想，被吸收进了 `balanced merge`

也就是说，当前系统不是“PQ 更新 + delta coarse 更新 + merge”三套并行机制，而是：

- 在线写入时保持简单，只更新 PQ
- 到 merge 时再集中处理 coarse routing mismatch 和 list imbalance

这样做的原因很直接：

- 单独维护 delta coarse 更新太冗余
- 在线路径会变复杂，状态更多
- 很多 coarse 层面的修复，本质上更适合在 merge 时和 main 一起做

## 当前架构

当前运行时最多会出现三个检索路由：

- `main`
- `active_delta`
- `frozen_delta`

其中：

- `main` 是稳定主索引
- `active_delta` 是当前接收新写入的增量索引
- `frozen_delta` 是已经停止写入、等待并入 main 的上一段 delta

因此，系统不是永远只有“1 main + 1 delta”。  
在 merge 交接窗口内，实际会短暂变成：

- `1 main + 1 frozen_delta + 1 active_delta`

这是为了保证：

- 旧 delta 可以先冻结等待 merge
- 新数据仍能继续进入新的 active delta
- 查询不中断

## 整体流程

### 1. 初始化

初始化阶段会完成三件事：

1. 用初始主数据训练 whitening
2. 用 whitening 后的数据构建 `main IVF/PQ`
3. 从主数据后面的一个保留窗口中训练第一代 `active_delta`

这里的 `delta_train_window` 不是在线 coarse 更新，而是：

- 用一小段预留数据训练一个初始 delta shard
- 让 active delta 从一开始就有自己的路由结构

之后的在线阶段，delta 不再持续刷新 coarse centroids，只做 PQ 相关更新。

### 2. 在线写入

每批流数据进入系统后的路径是：

1. 使用当前 whitening 变换
2. 写入 `active_delta`
3. 在 `active_delta` 内执行 `OnlinePQ` 更新

当前实现里，delta 的在线更新只有两种模式：

- `minibatch`
- `sliding_window`

无论哪种模式，本质都只是在做：

- 码本统计更新
- 必要时触发 PQ codebook update
- 可选的 batch re-encode

**不会在在线写入路径里单独刷新 delta 的 coarse centroids。**

### 3. 查询

查询阶段的路径是：

1. 对 query 做 whitening
2. 并行搜索 `main`
3. 如果存在 `frozen_delta`，同时搜索 `frozen_delta`
4. 如果存在 `active_delta`，同时搜索 `active_delta`
5. 合并多路结果
6. 可选 exact rerank
7. 返回最终 top-k

因此，当前系统的在线适应依赖两部分：

- `active_delta` 负责收纳最新数据
- `merge` 负责把已经稳定的数据并回 main

## 触发逻辑

`active_delta` 不会无限增长。当前代码会根据配置触发 freeze / merge，常见信号包括：

- `rows`：active delta 达到容量上限
- `qe_ratio`：OnlinePQ 质量指标恶化
- `drift`：codebook drift 升高
- `delta_main_ratio`：delta / main 比例过大
- `imbalance`：active delta 内部 list imbalance 过高

触发后流程是：

1. 当前 `active_delta` 冻结为 `frozen_delta`
2. 新流数据不再写入它
3. 系统开始从后续窗口训练新的 `active_delta`
4. 当新 active delta 准备好，或流结束时，提交一次 merge

这意味着当前实现实际上是“轮换 delta shard”，而不是对同一个 delta 做长期 coarse 在线自适应。

## Balanced Merge 才是当前 IVF 更新的核心

这一点是当前流程里最重要的部分。

我们虽然删除了“单独更新 delta coarse”的机制，但并没有放弃 IVF 层面的更新思想。  
相反，这部分思想被合并进了 `balanced merge`。

### 它解决什么问题

单纯把 frozen delta 直接 append 回 main 会有几个问题：

- 最近分区可能已经很满
- 新分布可能让大量向量挤进同一批 main list
- main 的局部 list imbalance 会越来越严重
- coarse routing mismatch 会累积

所以 merge 不能只是“把 delta 追加回 main”，而必须承担一部分 IVF 结构修复工作。

### 它怎么做

当前 `merge_assignment_mode="balanced_append"` 时，merge 分两步完成。

#### 第一步：受约束的主分区分配

对于 `frozen_delta` 中的每个向量：

1. 找到它在 main centroids 上最近的 `top-r` 个分区
2. 不一定直接选最近那个
3. 如果最近分区已经偏满，就允许在一个局部邻域里改投更健康的分区

这一步受几个参数控制：

- `merge_assignment_top_r`
- `merge_assignment_gamma`
- `merge_assignment_hard_cap_ratio`
- `merge_assignment_lambda`

通俗理解就是：

- 距离不能偏离最近中心太多
- 但也不能继续把向量全塞进已经过载的桶

因此这里已经体现出“IVF 路由更新”的思想：  
**不是死板沿用旧最近邻分桶，而是做带平衡约束的局部重路由。**

#### 第二步：邻域联合重分桶

这一步才是 `balanced merge` 的核心。

对于每个收到 delta 的 seed 分区：

1. 找到它在 main centroid 空间中的 `top-r` 个邻居分区
2. 取出这些 main 分区原有的记录
3. 再把分配到这些分区的 delta 记录一起放进一个池子
4. 在这个局部邻域内重新分配全部记录
5. 生成 `partition patch`
6. 一次性 patch 回 main

这意味着 merge 不是：

- “把 delta 某条向量插进某个现成 list”

而是：

- “把一小片相邻 main lists 拿出来，和 delta 一起重新摆一遍”

这正是原来 IVF 更新思想在当前系统中的落点。

### 为什么说这是 IVF 更新，而不是普通 append

虽然当前 merge **不会直接重训全局 coarse centroids**，但它确实在做 IVF 结构层面的更新：

- 它重新决定哪些向量应该归属哪些 main partition
- 它修复局部路由错误
- 它在邻域内重新平衡 list 大小
- 它通过 partition patch 改写 main 的分桶内容

所以更准确地说，当前版本的 IVF 更新不是：

- `delta` 在线单独刷新 coarse centers

而是：

- `merge` 阶段围绕现有 main centroids 做局部重路由与重分桶

这是一个更集中、更工程化的实现方式。

## 当前流程可以怎样理解

可以把当前系统概括成一句话：

**OnlinePQ 负责让 delta 先活下来，balanced merge 负责把 coarse 层面的结构问题带回 main 一起修。**

更展开一点：

- 在线阶段尽量轻，只做 PQ 自适应
- coarse mismatch 不在写入路径里单独修
- 等一段 delta 累积后，再通过 merge 做一次局部 IVF 结构整理

这样既保留了在线更新能力，也避免了三套更新机制同时存在带来的冗余。

## Global Rebuild 的角色

除了 merge，代码里还保留了可选的 `global rebuild`。

它的作用不是替代 balanced merge，而是兜底：

- 当 main 的全局结构已经明显失衡
- 或 main 从上次重建后增长太多

这时可以触发一次更重的全局重建，重新训练 whitening、重建 main，并重新初始化 delta。

因此层次关系是：

- `OnlinePQ`：最轻量的在线适配
- `balanced merge`：局部 IVF 结构修复 + 并回 main
- `global rebuild`：全局重建兜底

## 当前实现的真实能力边界

当前代码已经支持：

- 初始 whitening / main / delta 构建
- streaming 写入到 active delta
- delta 内 `OnlinePQ` 更新
- `main + frozen_delta + active_delta` 多路查询
- 基于多种信号的 freeze / merge 触发
- `balanced merge` 局部重分桶并 patch main
- 可选的 global rebuild

当前代码**没有**做的是：

- 在 steady state 下单独在线刷新 delta coarse centroids
- 在不经过 merge 的情况下直接改写 main coarse centroids
- 完整的异步 watermark 双版本切换框架

## 推荐理解方式

如果要向别人解释当前 SA-WRQ，最简洁的说法不是“三层在线更新”，而是：

1. `main` 保持稳定
2. `delta` 在线只做 OnlinePQ
3. `balanced merge` 承担 IVF 结构更新的职责
4. 必要时再做 global rebuild

这比旧版表述更准确，也更贴近当前代码。

## 运行方式

```bash
cmake -S . -B build
cmake --build build -j
./build/run_eval configs/sift/sift.json data/sift
```

```bash
cmake -S . -B build_release \
-DCMAKE_BUILD_TYPE=RelWithDebInfo \
-DANN_ENABLE_PROFILING=ON
```

在支持 AVX-512 的目标机器上，可以启用本机 ISA 优化；该构建不可直接移植到不支持对应指令集的机器：

```bash
cmake -S . -B build_native \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DANN_ENABLE_PROFILING=ON \
  -DANN_ENABLE_NATIVE=ON
cmake --build build_native -j
```

预先构建 main 索引和码本：

```bash
./build/build_index configs/sift/sift.json data/sift
./build/run_eval configs/sift/sift.json data/sift --use-prebuilt-index
```

`build_index` 默认写入 `index/<dataset>/<config>/`。也可以显式指定输出/读取目录：

```bash
./build/build_index configs/sift/sift.json data/sift index/sift/custom
./build/run_eval configs/sift/sift.json data/sift --prebuilt-index index/sift/custom
```

不传预构建索引参数时，`run_eval` 保持原有现场构建流程。

如果不开 streaming，可直接跑离线主索引评估；  
开启 streaming 后，`run_eval` 会输出 merge、OnlinePQ、latency、imbalance 等过程指标。
