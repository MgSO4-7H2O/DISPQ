# 配置说明

本文档介绍 `run_eval` 会读取的全部 JSON 配置项。  
所有字段都可以省略；未显式填写时，将使用
[`Config`](/mnt/d/research/SA-WRQ/include/common/config.h) 中定义的默认值。

## 使用方式

运行评测：

```bash
./build/run_eval configs/sift.json
```

命令行格式为：

```text
./build/run_eval <config.json> [dataset_spec] [query_spec]
```

- `dataset_spec` 可以是数据集目录，也可以是 `_base.fvecs` 文件路径。
- `query_spec` 可以是数据集目录，也可以是 `_query.fvecs` 文件路径。
- 如果不提供数据路径，则程序会自动生成合成数据。

## 示例

```json
{
  "ivf_nlist": 8192,
  "topk": 100,
  "nprobe": 64,
  "use_whitening": true,
  "enable_dual_route": true,
  "dim": 128,
  "seed": 1337,
  "pq_enable": true,
  "pq_m": 64,
  "pq_nbits": 8,
  "pq_residual": true,
  "max_queries": 10000,
  "snapshot_interval": 10000,
  "enable_miss_diag": false,
  "enable_rerank_source_diag": false,
  "enable_latency_debug": false,
  "exact_rerank_enable": true,
  "exact_rerank_candidates_per_route": 100,
  "main_exact_rerank_candidates": 300,
  "active_exact_rerank_candidates": 200,
  "frozen_exact_rerank_candidates": 100,
  "enable_streaming": true,
  "main_index_rows": 200000,
  "delta_train_window": 1,
  "delta_ivf_nlist": 1024,
  "merge_score_alpha": 1.0,
  "merge_score_beta": 1e-7,
  "merge_score_threshold": 10.0,
  "merge_trigger_mode": "state",
  "merge_trigger_rows": 50000,
  "merge_trigger_qe_ratio": 1.12,
  "merge_trigger_drift": 0.0,
  "merge_trigger_delta_main_ratio": 0.2,
  "merge_trigger_imbalance_ratio": 10.0,
  "merge_assignment_mode": "balanced_append",
  "merge_assignment_top_r": 8,
  "merge_assignment_gamma": 1.10,
  "merge_assignment_hard_cap_ratio": 1.5,
  "merge_assignment_lambda": 0.2,
  "enable_global_rebuild": false,
  "global_rebuild_max_count": 100,
  "global_rebuild_main_imbalance_ratio": 20.0,
  "global_rebuild_force_main_rows": 0,
  "global_rebuild_cooldown_rows": 50000,
  "streaming_mode": "batch",
  "stream_batch_size": 10000,
  "streaming_use_stream_batch_size": true,
  "online_pq_enable": true,
  "online_pq_update_scheme": "minibatch",
  "online_pq_sliding_window_size": 0,
  "online_pq_sliding_window_use_batches": true,
  "online_pq_force_update_interval": 1,
  "online_pq_partial_top_alpha": false,
  "online_pq_alpha": 1.0,
  "online_pq_partial_top_lambda": false,
  "online_pq_lambda": 1.0,
  "online_pq_reencode_batch": false
}

```

## 参数

### 基础检索

| 参数 | 类型 | 默认值 | 含义 | 用法 |
| --- | --- | --- | --- | --- |
| `ivf_nlist` | uint | `1024` | 主索引的 IVF coarse 分区数量。 | 值越大，单个 list 越短，但更依赖训练质量。 |
| `topk` | uint | `10` | 最终返回的近邻个数。 | 常见取值为 `10` 到 `100`。 |
| `nprobe` | uint | `8` | 查询时探测的 IVF list 数量。 | 值越大通常召回更高，但 QPS 更低。 |
| `use_whitening` | bool | `true` | 是否拟合并使用 whitening 变换。 | 通常建议开启。 |
| `enable_dual_route` | bool | `true` | 在共享搜索参数中保留双路检索支持。 | 流式实验一般保持开启。 |
| `main_query_only` | bool | `false` | 只查询主索引，忽略 delta 路由。 | 关闭，以前调试用的，**忽略**。 |
| `dim` | uint | `128` | 合成数据的维度，或用于校验真实数据维度。 | 必须与真实数据维度一致。 |
| `seed` | uint | `42` | 合成数据使用的随机种子。 | 仅对合成数据生效，**忽略**。 |
| `max_queries` | uint | `0` | 查询数上限；`0` 表示使用全部查询。 | 批量调参时可用来加速。 |
| `snapshot_interval` | uint | `0` | 流式阶段每隔多少条插入记录一个 snapshot。 | 用于流式过程诊断。 |

### PQ / 索引构建

| 参数 | 类型 | 默认值 | 含义 | 常见用法 |
| --- | --- | --- | --- | --- |
| `pq_enable` | bool | `false` | 是否在 IVF 中启用 PQ 编码。 | ANN 实验通常开启。 |
| `pq_m` | uint | `0` | PQ 子空间数量。 | 常取 `dim / 2` 或 `dim / 4`。 |
| `pq_nbits` | uint | `8` | 每个子空间的码本位数。 | `8` 是最常见设置。 |
| `pq_residual` | bool | `true` | 是否使用 residual PQ。 | 一般建议保持开启。 |

### 诊断输出（debug用，一般关闭）

| 参数 | 类型 | 默认值 | 含义 | 常见用法 |
| --- | --- | --- | --- | --- |
| `enable_miss_diag` | bool | `true` | 输出 miss/probe 诊断以及 worst-query 摘要。 | 想要更干净输出或更低开销时可关闭。 |
| `enable_rerank_source_diag` | bool | `false` | 统计精排后 top-k 来自主索引还是 delta。 | 需要开启 exact rerank。 |
| `enable_latency_debug` | bool | `false` | 输出 route 级延迟和 partition 级调试信息。 | 仅在分析性能瓶颈时开启。 |

### Exact Rerank（精排，一般开启）

| 参数 | 类型 | 默认值 | 含义 | 常见用法 |
| --- | --- | --- | --- | --- |
| `exact_rerank_enable` | bool | `false` | 是否在 ANN 候选生成后执行精确精排。 | 召回优先时开启。 |
| `exact_rerank_candidates_per_route` | uint | `0` | 每一路用于精排的默认候选数上限；若开启 rerank，必须 `> 0` 且 `>= topk`。 | 作为全局兜底配置。 |
| `main_exact_rerank_candidates` | uint | `0` | 主索引 route 的精排候选数上限；`0` 表示回退到 `exact_rerank_candidates_per_route`。 | 一般设置得最大。 |
| `active_exact_rerank_candidates` | uint | `0` | `active_delta` route 的精排候选数上限。 | 若 delta 是补充路由，通常可设小一些。 |
| `frozen_exact_rerank_candidates` | uint | `0` | `frozen_delta` route 的精排候选数上限。 | 通常可以设得最小。 |

说明：

- 开启 exact rerank 后，每一路搜索仍至少返回 `topk` 个候选。
- 如果某一路的 route-specific rerank cap 小于 `topk`，可能会降低最终召回。

### 流式参数

| 参数 | 类型 | 默认值 | 含义 | 常见用法 |
| --- | --- | --- | --- | --- |
| `enable_streaming` | bool | `false` | 是否启用 main + delta 的流式流程。 | 在线实验必须开启。 |
| `main_index_rows` | uint | `0` | 初始化时放入主索引的样本数；`0` 表示自动推断。 | 控制主索引 warmup 规模。 |
| `delta_train_window` | uint | `1` | 用于训练初始 delta shard 的窗口大小，可以按“行”或“批”解释。 | batch 模式下常设为 `1`。 |
| `delta_ivf_nlist` | uint | `0` | delta IVF 的分区数；`0` 表示回退到 `ivf_nlist`。 | 轻量 delta 时通常可小于主索引。 |
| `streaming_mode` | string | `"streaming"` | 流式模式，可选 `"streaming"` 或 `"batch"`。 | 大多数实验使用 `"batch"`。 |
| `stream_batch_size` | uint | `100` | batch 模式下每次插入的批大小。 | 控制更新粒度。 |
| `streaming_use_stream_batch_size` | bool | `false` | 是否将 `delta_train_window` 解释为“批数”而不是“行数”。 | batch 模式下一般可设为 `true`。 |

### Merge 打分（这三个值很重要，影响merge方式）

这些参数控制 frozen delta 合并到 main 时，某个 partition 是倾向于
`append/patch`，还是直接进入 `recluster`。

打分公式为：

```text
score = merge_score_alpha * growth_ratio
      + merge_score_beta  * avg_residual_dist
```

若：

```text
score >= merge_score_threshold
```

则该 partition 会被判定为 `recluster`；否则优先走 `append/patch`。

| 参数 | 类型 | 默认值 | 含义 | 常见用法 |
| --- | --- | --- | --- | --- |
| `merge_score_alpha` | double | `1.0` | partition 增长比例项的权重。 | 调大后会更关注 list 尺寸膨胀。 |
| `merge_score_beta` | double | `1e-7` | 平均 residual 距离项的权重。 | 调大后会更倾向于在 coarse 拟合变差时触发 recluster。这一项需要**注意**：因为开启白化矩阵，需要根据实际跑出来的调试结果设置，不然一直会触发recluster，带来不必要开销。 |
| `merge_score_threshold` | double | `10.0` | 合并打分的 recluster 阈值。 | 值越小，merge 越激进。 |

### Merge Trigger

| 参数 | 类型 | 默认值 | 含义 | 常见用法 |
| --- | --- | --- | --- | --- |
| `merge_trigger_mode` | string | `"rows"` | merge 触发模式，可选 `rows`、`qe_ratio`、`drift`、`imbalance`、`delta_main_ratio`、`state`、`hybrid`。 | 常见选择是 `state` （阈值触发）或 `hybrid`（阈值+强制触发）。 |
| `merge_trigger_rows` | uint | `0` | active delta 的行数阈值；`0` 表示回退到解析后的 `delta_train_rows`。 | 控制 delta 上限。 |
| `merge_trigger_qe_ratio` | double | `0.0` | 当 OnlinePQ 的 QE ratio 超过该值时触发 merge。 | 与 `state` / `hybrid` 搭配使用。 |
| `merge_trigger_drift` | double | `0.0` | 当 OnlinePQ 的 codebook drift 超过该值时触发 merge。 | 与 `state` / `hybrid` 搭配使用。 |
| `merge_trigger_delta_main_ratio` | double | `0.0` | 当 `active_delta_rows / main_rows` 超过该值时触发 merge。 | 用于控制 delta 相对主索引的规模。 |
| `merge_trigger_imbalance_ratio` | double | `0.0` | 当 active delta 的 list imbalance 超过该值时触发 merge。 | 用于结构性触发。 |

触发模式说明：

- `rows`：仅使用 `merge_trigger_rows`
- `qe_ratio`：仅使用 `merge_trigger_qe_ratio`
- `drift`：仅使用 `merge_trigger_drift`
- `imbalance`：仅使用 `merge_trigger_imbalance_ratio`
- `delta_main_ratio`：仅使用 `merge_trigger_delta_main_ratio`
- `state`：启用任意一个状态型触发器即可
- `hybrid`：`rows` 或任意状态型触发器满足即触发

### Merge 分配 / Balanced Merge

| 参数 | 类型 | 默认值 | 含义 | 常见用法 |
| --- | --- | --- | --- | --- |
| `merge_assignment_mode` | string | `"balanced_append"` | delta 向 main 分配时的策略，可选 `nearest` 或 `balanced_append`。 | `balanced_append` 是当前主方法。 |
| `merge_assignment_top_r` | uint | `4` | 约束分配时考虑的 top-r 邻近 main centroid 数。 | 值更大时局部调整空间更大，能减少全局重建次数，但会影响性能。 |
| `merge_assignment_gamma` | double | `1.05` | 候选分区必须满足 `dist <= gamma * nearest_dist`。 | 值越小越强调局部性。 |
| `merge_assignment_hard_cap_ratio` | double | `1.5` | 分配时 list 负载的硬上限比例。 | 值越小，负载均衡约束越强。 |
| `merge_assignment_lambda` | double | `0.2` | 约束分配中 balance penalty 的权重。 | 值越大越偏向均衡而非最近路由。 |

### 全局重建

| 参数 | 类型 | 默认值 | 含义 | 常见用法 |
| --- | --- | --- | --- | --- |
| `enable_global_rebuild` | bool | `false` | 是否启用全局重建作为兜底机制。 | 长时间运行且漂移严重时可开启。 |
| `global_rebuild_max_count` | uint | `0` | 单次运行中允许的最大重建次数；若开启全局重建则必须 `> 0`。 | 作为安全上限。 |
| `global_rebuild_main_imbalance_ratio` | double | `0.0` | 当 main imbalance 超过该值时触发全局重建。 | 结构性兜底触发器。 |
| `global_rebuild_force_main_rows` | uint | `0` | 当 main 自上次重建后增长达到该值时触发重建。 | 基于规模/时间的兜底策略。 |
| `global_rebuild_cooldown_rows` | uint | `0` | 两次全局重建之间至少间隔多少行。 | 避免重建过于频繁。 |

### OnlinePQ（阈值触发逻辑删掉了，设置参数强制触发）

| 参数 | 类型 | 默认值 | 含义 | 常见用法 |
| --- | --- | --- | --- | --- |
| `online_pq_enable` | bool | `true` | 是否在 active delta 上启用 OnlinePQ 更新。 | 在线自适应的主开关。 |
| `online_pq_update_scheme` | string | `"minibatch"` | OnlinePQ 更新模式，可选 `minibatch` 或 `sliding_window`（滑动窗口逻辑已经删了）。 | 默认推荐 `minibatch`。 |
| `online_pq_sliding_window_size` | uint | `0` | `sliding_window` 模式下的窗口大小。 | 仅在滑窗模式下使用。 |
| `online_pq_sliding_window_use_batches` | bool | `true` | 是否将滑窗大小按“批”解释。 | **忽略**。 |
| `online_pq_qe_ratio_threshold` | double | `1.05` | 触发 OnlinePQ 更新的 QE ratio 阈值。 | 值越大，更新频率越低。 |
| `online_pq_ema_alpha` | double | `0.1` | OnlinePQ 统计量的 EMA 平滑系数。 | 必须在 `(0, 1]` 内。 |
| `online_pq_eps` | double | `1e-6` | OnlinePQ 统计中的数值稳定项。 | 通常保持默认。 |
| `online_pq_warmup_enable` | bool | `false` | 是否在开始阶段设置 warmup。 | 有助于早期统计更稳定，**忽略**。 |
| `online_pq_warmup_batches` | uint | `0` | warmup 批次数；若开启 warmup 则必须设置。 | 常见取值如 `5`。 |
| `online_pq_force_update_interval` | uint | `0` | 每隔多少批强制触发一次 OnlinePQ 更新。 | 用于控制最大陈旧度，设置为1，**强制每批触发**。 |
| `online_pq_partial_top_alpha` | bool | `false` | 是否只更新 top 影响的 alpha 分量。 | 实验性选项。 |
| `online_pq_alpha` | double | `1.0` | alpha 更新强度，必须在 `(0, 1]` 内。 | 值越小越保守。 |
| `online_pq_partial_top_lambda` | bool | `false` | 是否只更新 top 影响的 lambda 分量。 | 实验性选项。 |
| `online_pq_lambda` | double | `1.0` | lambda 更新强度，必须在 `(0, 1]` 内。 | 值越小越保守。 |
| `online_pq_reencode_batch` | bool | `true` | codebook 更新后是否重编码刚插入的 batch。 | 提高一致性，但会增加开销。 |

## 常见配置模板

### 离线主索引基线

```json
{
  "enable_streaming": false,
  "main_query_only": true,
  "exact_rerank_enable": false
}
```

### 启用 Balanced Merge 的流式主索引 + Delta

```json
{
  "enable_streaming": true,
  "merge_assignment_mode": "balanced_append",
  "merge_trigger_mode": "state",
  "merge_trigger_rows": 50000,
  "merge_trigger_qe_ratio": 1.12,
  "merge_trigger_delta_main_ratio": 0.2,
  "merge_trigger_imbalance_ratio": 10.0
}
```

### 更激进的 Merge Recluster

```json
{
  "merge_score_alpha": 1.0,
  "merge_score_beta": 1e-6,
  "merge_score_threshold": 5.0
}
```

这组配置通过降低阈值并增大 residual 距离项权重，使更多 partition
在 merge 时进入 `recluster` 而不是 `append`。

## 实用建议

- 大规模流式实验建议从 [sift.json](/mnt/d/research/SA-WRQ/configs/sift.json) 开始。
- 如果 QPS 偏低，优先调节：
  - `nprobe`
  - `exact_rerank_candidates_per_route`
- 如果多次 merge 后召回明显下降，可以重点检查：
  - `merge_score_threshold`
  - `merge_trigger_imbalance_ratio`
  - `enable_global_rebuild`
- 如果 OnlinePQ 表现不稳定，建议先关闭部分更新策略，并保持：
  - `online_pq_alpha = 1.0`
  - `online_pq_lambda = 1.0`
  - `online_pq_warmup_enable = false`
