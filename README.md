## SA-WRQ: Online Streaming ANN with Drift Triggers

当前实现支持在线流式插入与 drift 监控，主流程位于 `apps/run_eval.cpp`。

## 1. 系统能力

- `Main Index`：初始用 base 前缀（`main_index_rows`）离线构建，运行期间保持不变。
- `Delta Index`：后续流式数据统一写入单一 delta index。
- delta 构建复用 `C_main`（固定 coarse centroids）。
- `Main + Delta` 并行检索并按 `doc_id` 去重合并 top-k。
- `streaming_mode` 支持两种插入方式：
  - `streaming`：单条插入
  - `batch`：小批量插入（`stream_batch_size`）
- drift 信号持续监控并输出，不触发多 delta 轮换或 main rebuild。

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

baseline 由初始 main 数据建立，运行期间不重置。

## 3. Drift 信号逻辑

### 3.1 Soft Drift Signal

- 条件：`npd_ratio > drift_soft_npd_ratio` 且 `nre_ratio < drift_hard_nre_ratio`
- 确认机制：`drift_confirm_k` 连续命中，或 `drift_confirm_m` 窗口命中占比超过 `drift_confirm_ratio`
- 最小生存期：`drift_min_delta_lifetime_windows`
- 候选收益：`candidate_gain >= drift_soft_gain`  
  `gain = (NPD_old - NPD_candidate) / (NPD_old + eps)`
- 输出：`trigger_new_delta = true`（仅作为信号输出）

### 3.2 Hard Drift Signal

满足任一条件并通过确认机制后触发：

- `nre_ratio > drift_hard_nre_ratio`
- `cm_z > drift_hard_cm_z`
- `lds > drift_hard_lds`
- `closed_delta_count >= drift_max_closed_deltas`
- `closed_delta_docs / total_docs > drift_max_closed_ratio`

- 输出：`trigger_rebuild_main = true`（仅作为信号输出）

## 4. 主流程行为

`run_eval` 在线循环：

1. 初始建 main（base 前缀）。
2. 创建单一 delta index。
3. 按 streaming/batch 插入流式数据。
4. 每个 drift window 计算 NRE/NPD/CM/LDS/CUE。
5. drift 指标持续更新，但结构动作不执行，所有更新只写入同一个 delta。
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
- `drift_max_closed_deltas`（当前单 delta 语义下仅保留配置兼容）
- `drift_max_closed_ratio`（当前单 delta 语义下仅保留配置兼容）
- `drift_active_delta_max_docs`（当前单 delta 语义下仅保留配置兼容）

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
- drift 信号与动作计数（当前语义下动作计数保持为 0）
- 可选快照序列（`snapshot_interval > 0` 时的阶段性 performance/drift 记录）
