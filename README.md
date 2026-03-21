## SA-WRQ: Streaming ANN with Main + Delta

SA-WRQ 面向流式向量检索场景，核心目标是：

- `main` 保持稳定，负责大规模检索
- `delta` 持续接收 streaming 数据，负责快速适应新分布
- `drift` 不控制 delta 更新，而是判断是否需要后台异步 `rebuild main`

系统在线稳定态始终只维护：

- `1 main`
- `1 delta`

## 核心思想

### 1. Main 保持稳定

`main` 是主索引，在一个 serving epoch 内不做在线更新。

它负责：

- whitening
- coarse centroids
- PQ / residual PQ
- 大规模、稳定的 ANN 检索

`main` 不直接接收 streaming 写入。  
当分布明显变化时，不是在线改 `main`，而是后台重建新的 `main`。

### 2. Delta 持续在线更新

所有 streaming 数据只写入 `delta`。

`delta` 负责：

- 接收最新数据
- 用独立的 `C_delta` 适应流式分布
- 补偿 `main` 对新分布的滞后

为了适应 streaming 数据，`delta` 的 centroids 可以按 batch 做带遗忘的 EMA / online k-means 更新，例如：

```text
N_j <- rho * N_j + n_j
S_j <- rho * S_j + sum_j
c_j <- S_j / N_j
```

重点不是逐条更新，而是按小批量稳定更新。

### 3. Drift 只负责 Rebuild Main

本方案里，`delta` 默认一直更新，所以 drift 的职责很单一：

- 判断 `main` 是否已经落后于当前数据分布

drift 更适合看全局信号，例如：

- residual energy 是否持续升高
- centroid margin 是否恶化
- list distribution 是否偏移
- 有多少比例的向量都明显漂移了
- `delta` 是否持续变大

当这些信号持续恶化时，触发后台 `rebuild main`。

## 简要流程

### 在线写入流程

1. streaming 向量到来
2. 用当前 whitening 变换
3. 写入 `delta`
4. 按 micro-batch 更新 `delta` 的 centroids
5. 查询时并行搜索 `main + delta`，再合并结果

### 异步 Rebuild Main 流程

当 drift 判定需要重建时：

1. 打一个 `watermark = t0`
2. 后台用 `main + delta[:t0]` 构建新的 `main`
3. 前台继续服务，`t0` 之后的新数据继续写当前 `delta`
4. 后台再用 `t0` 之后的数据构建新的 `delta`
5. 原子切换到新的 `main + delta`

这样做的好处是：

- 不阻塞前台服务
- 新 `main` 能吸收已经稳定下来的新分布
- 新 `delta` 只保留最新流数据

## 方案总结

可以把这套方案概括成一句话：

`main` 负责稳定，`delta` 负责适应，`drift` 负责判断何时把已经稳定的新分布吸收到新的 `main` 中。

也就是：

- 平时只更新 `delta`
- 必要时后台重建 `main`
- 重建时通过 `watermark` 完成平滑切换

## 当前仓库状态

当前仓库已经支持：

- 初始 `main` 构建
- 单 `delta` 流式写入
- `main + delta` 并行检索
- drift 指标监控与输出

当前仍未完整实现：

- `delta` 独立 centroids 的在线自适应
- `watermark` 驱动的异步 `rebuild main`
- 新旧 `main + delta` 的原子切换

## 运行方式

```bash
cmake -S . -B build
cmake --build build -j
./build/run_eval configs/cifar.json data/cifar
```
