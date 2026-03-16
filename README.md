## SA-WRQ: Online Streaming ANN with Drift Triggers

当前实现支持在线流式插入、drift 监控与动作触发，主流程位于 `apps/run_eval.cpp`。

## 1. 系统能力

- `Main Index`：初始用 base 前缀（`main_index_rows`）构建。
- `Active Delta Index`：后续流式数据写入 active delta。
- `Closed Delta Index`：active delta 轮换后转为 closed，先归档保留（暂不参与检索）。
- delta 构建复用 `C_main`（固定 coarse centroids），仅在新窗口上更新 PQ。
- `Main + Active` 并行检索并按 `doc_id` 去重合并 top-k。
- `streaming_mode` 支持两种插入方式：
  - `streaming`：单条插入
  - `batch`：小批量插入（`stream_batch_size`）
- drift 触发三类动作：
  - `new delta`
  - `seal delta`
  - `rebuild main`

## 2. Drift 指标

每个 drift window（`drift_window_size`）计算：

- `NRE`（Residual Energy）  
  `NRE = E[ ||z-c1||^2 / (||z||^2 + eps) ]`
- `NPD`（PQ Distortion）  
  `NPD = E[ ||r-r_hat||^2 / (||r||^2 + eps) ]`
- `CM`（Centroid Margin）  
  `CM = E[ (d2-d1)/(d1+eps) ]`
- `LDS`（List Distribution Shift）  
  `LDS = JS(P_t || P_0)`（log2 版本）
- `CUE`（Code Usage Entropy）  
  子量化器码字使用熵（归一化后求平均）

同时输出归一化信号：

- `nre_ratio = NRE / baseline_NRE`
- `npd_ratio = NPD / baseline_NPD`
- `cm_z = (CM - baseline_CM_mean) / baseline_CM_std`
- `cue_ratio = CUE / baseline_CUE`

baseline 由初始 main 数据建立，main rebuild 后重置 baseline。

## 3. Trigger 逻辑

### 3.1 Soft Drift

- 条件：`npd_ratio > drift_soft_npd_ratio` 且 `nre_ratio < drift_hard_nre_ratio`
- 确认机制：`drift_confirm_k` 连续命中，或 `drift_confirm_m` 窗口命中占比超过 `drift_confirm_ratio`
- 最小生存期：`drift_min_delta_lifetime_windows`
- 候选收益：`candidate_gain >= drift_soft_gain`  
  `gain = (NPD_old - NPD_candidate) / (NPD_old + eps)`
- 动作：`trigger_new_delta = true`，并 seal 当前 active delta

### 3.2 Hard Drift

满足任一条件并通过确认机制后触发：

- `nre_ratio > drift_hard_nre_ratio`
- `cm_z > drift_hard_cm_z`
- `lds > drift_hard_lds`
- `closed_delta_count >= drift_max_closed_deltas`
- `closed_delta_docs / total_docs > drift_max_closed_ratio`

动作：`trigger_rebuild_main = true`

### 3.3 Seal Active Delta

- `trigger_new_delta = true` 时 seal，并创建新的 active delta

## 4. 主流程行为

`run_eval` 在线循环：

1. 初始建 main（base 前缀）。
2. 创建 active delta。
3. 按 streaming/batch 插入流式数据。
4. 每个 drift window 计算 NRE/NPD/CM/LDS/CUE。
5. 命中 `new delta / rebuild main` 时执行对应动作并继续服务。
6. 结束后输出最终 JSON 指标。

## 5. 配置项

### 5.1 流式相关

- `enable_streaming`
- `main_index_rows`
- `streaming_mode`：`"streaming"` / `"batch"`
- `stream_batch_size`
- `snapshot_interval`（>0 时按插入间隔采样，写入最终 `online_eval.json` 的 `snapshots` 数组）

### 5.2 Drift 相关

- `drift_window_size`
- `drift_confirm_k`
- `drift_confirm_m`
- `drift_confirm_ratio`
- `drift_min_delta_lifetime_windows`
- `drift_soft_npd_ratio`
- `drift_soft_gain`
- `drift_hard_nre_ratio`
- `drift_hard_cm_z`
- `drift_hard_lds`
- `drift_max_closed_deltas`
- `drift_max_closed_ratio`
- `drift_active_delta_max_docs`（当前语义下不作为轮换触发条件，预留）

## 6. 运行方式

```bash
cmake -S . -B build
cmake --build build -j
./build/run_eval configs/cifar.json data/cifar
```

## 7. 输出文件

- 单个结果文件（每次运行覆盖）：
  - `result/<dataset>/online_eval.json`

结果文件包含：

- 检索指标（recall/qps/latency/scanned）
- drift 指标（NRE/NPD/CM/LDS/CUE 及 ratio/z-score）
- 触发动作统计（new delta / seal delta / rebuild main）
- 可选快照序列（`snapshot_interval > 0` 时的阶段性 performance/drift 记录）
