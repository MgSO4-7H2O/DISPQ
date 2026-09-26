# SIFT1M PQ 布局消融实验

本目录提供 dimension-major codebook 与 subquantizer-major PQ code 的 2×2 对照实验。四个 JSON 配置固定相同的查询、数据流、随机种子和在线维护参数；每次运行按顺序执行四种组合，并为每组保存独立日志和原始结果。

| 配置 | `pq_codebook_dimension_major` | `pq_codes_subquantizer_major` |
|---|---:|---:|
| `sift1m_layout_aos_record` | `false` | `false` |
| `sift1m_layout_dim_record` | `true` | `false` |
| `sift1m_layout_aos_subq` | `false` | `true` |
| `sift1m_layout_dim_subq` | `true` | `true` |

## 运行要求

- 使用包含这次布局开关及计时改动的 SA-WRQ 源码；运行时会从当前仓库源码构建 `run_eval`，确保新布局配置和阶段计时生效。
- Linux、CMake、C++ 编译器、OpenMP、Python 3 和 `numactl`。
- SIFT1M drifted 数据目录中包含 `sift1M_drifted_base.fvecs` 和 `sift1M_drifted_query.fvecs`。
- 在目标机器上重新编译，避免把带有其他 CPU ISA 优化的二进制直接复制过来。

在仓库根目录编译：

```bash
cmake -S . -B build_release \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DANN_ENABLE_PROFILING=ON
cmake --build build_release --target run_eval --parallel
```

运行四组实验：

```bash
bash experiments/sift1m_layout_ablation/run.sh /path/to/sift1M/drifted
```

默认使用 36 个 OpenMP 线程和 NUMA node 0。脚本根据 node 的 CPU 拓扑和当前进程可用的 CPU 集合，自动选取 36 个 node-local CPU，并将线程和内存绑定到指定节点。若目标机器的 node 0 不足 36 个可用 CPU，脚本会停止并提示调整线程数或显式指定 CPU 集合。

可以通过环境变量覆盖默认值。`CPU_LIST` 必须包含恰好与线程数相同的 CPU，并且都属于指定 NUMA 节点且对当前进程可用：

```bash
OMP_NUM_THREADS=36 \
NUMA_NODE=0 \
CPU_LIST=0-31,64-67 \
RESULT_ROOT=/data/sa_wrq/sift1m_layout_run1 \
bash experiments/sift1m_layout_ablation/run.sh /path/to/sift1M/drifted
```

也可以设置 `BUILD_IF_MISSING=1`，让脚本在找不到 `run_eval` 时自动构建：

```bash
BUILD_IF_MISSING=1 BUILD_JOBS=36 \
bash experiments/sift1m_layout_ablation/run.sh /path/to/sift1M/drifted
```

支持的设置有 `OMP_NUM_THREADS`、`NUMA_NODE`、`CPU_LIST`、`RESULT_ROOT`、`RUN_ID`、`BUILD_DIR`、`RUN_EVAL_BIN`、`BUILD_IF_MISSING` 和 `BUILD_JOBS`。默认结果目录带有时间戳，避免覆盖已有实验；如果显式指定的结果目录含有已有运行文件，脚本会拒绝覆盖。

## 结果文件

每种配置的结果目录包含：

- `config.json`：本次运行使用的完整配置副本。
- `run_meta.txt`：主机名、代码 revision、线程数、CPU 列表、NUMA 节点和数据目录。
- `run.log`：完整运行日志。
- `online_eval.json`、`memory_trace.json`：查询、更新和内存测量结果。

运行结束后可生成 2×2 汇总表：

```bash
python3 experiments/sift1m_layout_ablation/summarize.py /path/to/result-root
```

汇总脚本将更新本目录中的 `SUMMARY.md`、`summary.csv` 和 `summary.json`。四组实验耗时较长；若中途出错，脚本会保留已完成组和失败组日志并立即退出，可通过日志检查失败位置。重新运行时请使用新的 `RESULT_ROOT` 或 `RUN_ID`。
