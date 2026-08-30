# mHC Expand 测试

测试按 Coverage、Correctness、Mock 三类组织，运行平台为 Atlas A2/A3。案例参数保存在三份 JSON 中，数据生成、Host 契约、分层构建、设备执行和结果复核共用同一套 Python 模块与 C++ ACLNN runner。

| 类别 | 案例文件 | 数量 | 用途 |
|---|---|---:|---|
| Coverage | `st/cases/coverage.json` | 16 | FP16/BF16、前向/反向、题面规模和组合边界 |
| Correctness | `st/cases/correctness.json` | 10 | FP32 累加精度、抵消、大值和敏感性输入 |
| Mock | `st/cases/mock.json` | 8 | 固定工作量、正确性检查和 Profiler |

`full` 是 Coverage 与 Correctness 的 26 项并集。Mock 使用独立性能入口。

## 目录

```text
tests/
├── README.md
├── common/
│   ├── cases.py                 # Case 类型、分类加载和选择
│   ├── mock_cases.py            # Mock 配置、fixture 和结果比较
│   ├── reference.py             # 确定性输入、CPU 公式和输出验证
│   ├── tensor_io.py             # FP16/BF16 编解码与二进制 I/O
│   └── source_baseline.json     # 官方 7 文件清单与 CMake 基线
├── st/
│   ├── cases/                   # coverage、correctness、mock
│   ├── runner/
│   │   └── mhc_expand_runner.cpp
│   └── scripts/
│       ├── audit_coverage.py    # Case 矩阵审计
│       ├── build.py             # Bootstrap、Host、Kernel、Final 构建
│       ├── kernel_cache.py      # 内容寻址 Kernel 缓存
│       ├── gen_data.py          # Raw fixture 和 metadata
│       ├── run_tests.py         # 分层构建与设备验证入口
│       └── profile_mock_workloads.py  # Mock 执行和 msprof 映射
└── ut/
    ├── host_contract.cpp
    ├── test_host_contract.py
    ├── test_case_matrix.py
    ├── test_reference.py
    ├── test_artifacts.py
    ├── test_mock_cases.py
    └── test_profile_mapping.py
```

UT 按职责组织：

| 文件 | 内容 |
|---|---|
| `test_host_contract.py`、`host_contract.cpp` | 真实 CANN Host Context、Tiling、Shape 和 DType 推导 |
| `test_case_matrix.py` | 分类隔离、选择规则和 Coverage 审计 |
| `test_reference.py` | 前向复制、反向 FP32 累加和精度输入 |
| `test_artifacts.py` | Fixture、Runner plan、dtype 过滤、输出验证和 Kernel 缓存 |
| `test_mock_cases.py` | 固定 Mock 配置、数据和结果比较 |
| `test_profile_mapping.py` | msprof 行筛选、耗时映射和错误字段检查 |

## 案例选择与静态验证

```bash
python3 -m tests.st.scripts.gen_data --suite coverage --list
python3 -m tests.st.scripts.gen_data --suite correctness --list
python3 -m tests.st.scripts.audit_coverage
python3 -m tests.st.scripts.run_tests --layer static
python3 -m tests.st.scripts.run_tests --layer ut
```

`--suite` 支持 `coverage`、`correctness`、`full` 和 `custom`。`--case` 可重复指定；`--tag` 要求案例同时包含给定标签。题面的大规模反向输入单个 Tensor 约 896 MiB，生成或执行时需显式使用 `--allow-large`。

静态层检查官方文件清单、CMake 基线、Git whitespace、Python 语法和 Case 覆盖。UT 在安装 CANN Context Builder 时编译 `host_contract.cpp`，直接调用项目的 Tiling、Shape 和 DType 推导；缺少相应 SDK 头文件时仅跳过 Host Context 用例。

## 分层构建与设备验证

默认工作目录为 `/tmp/cannjudge/mhcexpand/layered`，可通过 `MHC_WORK_ROOT` 指定。CANN 路径通过 `MHC_CANN_PATH` 指定，缺省为 `/usr/local/Ascend/cann-8.5.0`。

```bash
# CMake 配置、Host-only 包和代表 dtype Kernel
python3 -m tests.st.scripts.build --target bootstrap
python3 -m tests.st.scripts.build --target host
python3 -m tests.st.scripts.build --target kernel \
  --representative-case forward_scalar_float16

# Workspace 查询、单进程 Smoke 和 Correctness
python3 -m tests.st.scripts.run_tests --layer workspace
python3 -m tests.st.scripts.run_tests --layer smoke
python3 -m tests.st.scripts.run_tests --layer correctness

# 空 Final 目录中的全模板构建与完整设备验证
python3 -m tests.st.scripts.build --target final
python3 -m tests.st.scripts.run_tests --layer full-test \
  --suite custom --case forward_scalar_float16
```

`validate` 串行执行 Static、UT、Host 构建和选定 Correctness；`pipeline` 还执行 Workspace 与 Smoke。`--stage final --allow-large` 在开发阶段检查后构建 Final 包并执行完整矩阵。

Host 构建只生成 Schema、Tiling 和 Host-only ACLNN 库，并检查 Kernel 目录没有变化。代表 Kernel 按 dtype 生成前向/反向、对齐/非对齐四组 Tiling Key。Final 构建在不存在或空目录中复制官方的 7 个文件到 `tests/submission/code`，只从该快照编译全部模板。收据同时记录快照、源码、工具链和产物身份，额外头文件不会进入 Final 构建。

## Runner 与结果

同一 Runner 进程可连续执行多个 Case，ACL、device、stream 和动态库只初始化一次。Workspace 模式只查询 Workspace，不启动 Kernel。数值执行对输出使用 NaN 预填，在输入输出两侧设置 64B guard，并保存首轮与末轮输出。

每次设备运行在 `results/run-*/` 保存：

| 文件 | 内容 |
|---|---|
| `manifest.json` | 命令、源码身份、fixture、构建产物、计划、日志和实际输出哈希 |
| `runner.plan` | 单进程调用序列 |
| `runner.log` | Workspace 或 Kernel 执行记录 |
| `results.json` | 首轮、末轮、guard 和逐调用判定 |

运行入口直接比较每个已完成调用的首末实际输出。后续调用发生 RE 或超时，仍保留前面已确认的数值错误；调用未完成、缺失输出或整个进程失败时，不能判定整组通过。

## 构建身份与缓存

Kernel 缓存键包含 Kernel 源码、Host/Tiling ABI、已编译 Tiling 库、生成配置、CANN/BiSheng 身份和编译选项。缓存命中逐文件检查路径、大小和 SHA-256；损坏条目进入隔离目录。Fixture 缓存键包含 Case、生成器、参考实现、Case 加载器、Tensor 编解码模块和 NumPy 版本。

## Mock 与 Profiler

八个 Mock Case 使用固定 seed 和配置生成可复现的本地合成输入。每个 Case 先检查首轮、末轮和 GM guard；启用 Profiler 时，再独立映射 `Task Duration(us)` 样本。

`st/cases/mock.json` 的 `runtime` 是套件执行参数：普通页 `normal-only`、512B guard、
`msprof op` 的 `BasicInfo`、5 次 Profiler 预热，以及 3 个独立进程。Native 每进程调用一次，
由 Profiler 负责重放。全部观察保存在 `runs/<point>/process-<index>/`；数值与采样均通过时，
才报告各进程设备耗时中位数的中位数。单进程 Correctness 使用独立的 64B guard 配置。

历史材料与官方对照工具不属于测试套件。套件报告说明数值和采样有效性，不声称已达到官方性能目标。

```bash
python3 -m tests.st.scripts.profile_mock_workloads \
  --build-dir /path/to/verified/final-build \
  --output /tmp/mhc-expand-mock \
  --profile
```

`--case` 可重复指定 1～8；`--warmup` 和 `--repeat` 控制进程内次数，`--processes` 控制独立进程数。
`--profiler-backend`、`--allocation-policy` 和 `--guard-bytes` 可以显式覆盖默认协议，实际参数写入报告。
缺少记录、样本数量不符、非有限耗时、数值错误、guard 错误或构建身份不一致均不能通过。
