## 首先要明确的点

### 白化

是否应该按照输入的数据分段白化？

理论上来说**不可行**。白化定义的是整个检索空间的度量结构；一旦不同分段使用不同白化，查询变换、coarse 分桶和距离比较就不再共享同一几何空间，导致检索路径复杂化、跨索引结果不可直接比较，并使系统稳定性和可合并性显著下降。

实际结果上来看确实**不可行**。基于windows的累计更新方法中，若采用数据分段白化方式，recall骤然降低，且qps、latency没有显著提升。

应该怎么做？

白化应作为随 **epoch** 慢速更新的全局空间参数，在同一 epoch 内 main 与 delta **共享** `W_main`、由 delta 通过更新 `Q_delta` 适配局部分布漂移；只有当监控表明二阶统计和 coarse 结构也显著失配时，才通过后台重建新的 `W/C/Q` 进入下一代主索引，而不应让每个在线 delta 独立学习自己的白化。

## 0. 核心目标

构建一个支持以下能力的在线流式 ANN 系统：

- **主索引 Main Index**：稳定、压缩、只读、高吞吐
- **增量索引 Delta Index**：在线写入、快速适配分布漂移
- **Drift-based Trigger**：不再按累计条数切版本，而是按漂移信号驱动
- **并行检索 + 结果合并**：查询同时搜索 main 与 delta
- **条件式 merge / compaction**：防止 delta 无限制堆积
- **后台 rebuild main**：当漂移已经影响 coarse 结构时，异步重建 main

## 1. 总体架构

#### A. Main Index（主索引）

保存当前 epoch 的稳定大库索引：

- `W_main`：白化变换
- `C_main`：IVF coarse centroids
- `Q_main`：主 PQ 码本
- `MainPostingLists`

特点：

- 冻结
- 查询吞吐高
- 不参与频繁在线更新

------

#### B. Active Delta Index（活动增量索引）

所有新来的向量先写入这里。

建议：

- `W_delta = W_main`
- `C_delta = C_main`
- `Q_delta` 可随 drift 更新

也就是说，**delta 共享主索引的几何空间，只适配自己的 PQ 量化器**。

这样有巨大好处：

- query transform 只做一次
- coarse probing 只做一次
- main 和 delta 可以共用同一组 probed lists
- 实现复杂度大幅下降

------

#### C. Closed Delta Index（封存增量索引）

当 active delta 被 seal 之后，变成 closed delta：

- 不再接收写入
- 继续参与查询
- 等待 merge 或淘汰或 main rebuild 吸收

------

#### D. Shadow Delta（影子候选索引）

当 drift monitor 认为可能需要切新 delta 时，不要直接 promote。
 先构造一个 shadow delta：

- 用最近窗口训练 `Q_shadow`
- 少量流量试运行或 compare-only
- 如果收益明显，再 promote 为新的 active delta

### 2. Epoch 语义

一个 epoch 表示“主空间语义一致的一代索引”：

- 一个 epoch 内，`W_main / C_main` 固定
- active delta / closed delta 都共享这套 `W/C`
- 只允许 `Q_delta` 演化

当 drift 已经严重到 `W/C` 也失配时：

- 开启后台 `epoch + 1` 的 rebuild
- 重训新 `W/C/Q_main`
- 切换到新 epoch

### 3. 在线写入流程

投影到主空间 -> coarse assignment -> residual 计算 -> 用 active delta 的 `Q_delta` 编码 -> 写入 Active Delta -> 更新在线 drift 统计

### 4. drift指标

#### 4.1 Residual Energy（RE）

$NRE_t = \mathbb{E}\left[\frac{||z-c_1||^2}{||z||^2+\epsilon}\right]$

- 反映 coarse centroid 对当前数据拟合是否变差
- 如果这个量持续上升，说明 `C_main` 开始不贴数据了

#### 4.2 PQ Distortion（PD）

$NPD_t = \mathbb{E}\left[\frac{||r-\hat r||^2}{||r||^2+\epsilon}\right]$

- 最直接衡量当前 PQ 是否开始失配
- 如果 NPD 升了，而 NRE 没太变，说明问题主要在 PQ，不在 coarse

#### 4.3 Centroid Margin（CM）

定义：

- 最近中心距离 `d1`
- 第二近中心距离 `d2`

$CM_t = \mathbb{E}\left[\frac{d2-d1}{d1+\epsilon}\right]$

#### 4.4 List Distribution Shift（LDS）

统计当前窗口向量分配到各 `list_id` 的概率分布 `P_t(list)`，与 baseline `P_0(list)` 比较：

$LDS_t = JS(P_t || P_0)$

- 宏观上看热点 coarse 区域是否在迁移
- 这个量对“全局数据流向变了”很敏感

#### 4.5 Code Usage Entropy（CUE）

统计每个 subquantizer 码字使用分布熵：

$H_m = -\sum_k p_{m,k}\log p_{m,k}$

- 某些码字突然不用，或大量塌缩到少数码字，说明量化器健康度变差

### 5. Trigger

#### 5.1 Soft Drift 触发

- 共享 `W/C` 还基本合理
- 主要是 `Q_delta` 老化

#### 5.2 Soft Drift 触发

- 不只是 PQ 失配
- coarse/main 几何结构开始不靠谱
- 该准备重建 main 了

#### 5.3 容忍机制

##### 5.3.1 连续命中 + 多窗口确认

不要一超阈值就切，必须：

- 连续 `K` 个窗口超阈
- 或过去 `M` 个窗口中超过 `p%`

##### 5.3.2 delta 最小生存期

Active delta 刚建出来别立刻又切。

#### 5.4 乐观估计机制

在真正创建新 delta 前，先用最近窗口训练候选 `Q'`，做一次收益估计。

##### 5.4.1 流程

1. 取最近窗口 residual 样本
2. 从当前 `Q_delta` 派生 `Q_candidate`
3. 计算：
   - `NPD_old`
   - `NPD_new`
4. 计算收益：

$Gain = \frac{NPD_{old} - NPD_{new}}{NPD_{old} + \epsilon}$

##### 5.4.2 决策

只有当：

- `Gain >= 5% ~ 10%`
- 且 `CM` 没有明显恶化
- 且预计查询代价还能接受

才允许创建 shadow / promote 新 delta。

#### 5.5 merge

##### 5.5.1 主空间兼容性仍然好

old delta 上的：

- `NRE_delta / μ_NRE < 1.2`
- `CM_delta` 未明显恶化
- list assignment 模式与 main epoch 一致

说明 `W_main / C_main` 仍适用于这批数据。

##### 5.5.2 用 `Q_main` 重编码损失可接受

对 old delta 采样一批 residual：

比较：

- 用 `Q_delta_old` 编码的 `NPD_delta`
- 用 `Q_main` 重编码的 `NPD_main_reencode`

##### 5.5.3 合并收益大于成本

比如：

- 关闭一个 closed delta 后，查询少扫一个分支
- scanned candidates 显著下降
- merge 后内存占用更规整

如果收益不明显，就不急着 merge。

##### 5.5.4 实现

###### 同 Epoch Merge

适用于：

- `W/C` 还稳定
- 只是 old delta 需要并入 main

###### 跨 Epoch Rebuild Merge

适用于：

- hard drift 已经发生
- `W/C` 也开始过时
- 不应再把旧 delta 硬塞回旧 main

#### 5.6 main rebuild

满足以下任一建议触发：

- `NRE / μ_NRE > 1.5`
- `z_CM > 3`
- `LDS > 0.2`
- `closed_delta_count >= 3`
- `closed_delta_docs / total_docs > 15%`
- 查询平均扫描量超过基线 `1.3x`
- active + closed delta 总延迟占比明显升高

后台异步rebuild，不阻塞在线服务。

- 训练线程构建新 main
- 前台继续：
  - 写 active delta
  - 查 old main + deltas
- 新 main ready 后原子切换

### 6. search

#### 6.1 Search on Main

对 probed lists：

- 计算 PQ distance table
- 扫描 main posting lists
- 得到 main 候选集

#### 6.2 Search on Active Delta

在同一组 probed lists 上：

- 用 `Q_delta_active` 解码距离
- 扫描 active delta 候选

#### 6.3 Search on Closed Deltas

对最近 `K` 个 closed deltas：

- 复用同一组 list_id
- 各自用自己的 `Q_delta_i` 做 approx dist
- 扫描候选

对多个索引源返回的候选：

- 按 `doc_id` 去重
- 对候选进行 rerank
- 最终输出 top-k