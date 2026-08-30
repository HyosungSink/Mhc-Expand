# 【B组简单题】mHC-expand 算子（前向与反向）

## 1. 算子说明

MHC Expand 是 Manifold HyperConnection（流形超连接）架构中的扩流算子，用于将残差连接的单一数据流扩展为多路并行流。在 DeepSeek 提出的 mHC（Manifold-Constrained Hyper-Connections）架构中，标准残差连接被扩展为多路并行残差流，以增强层间信息交互能力。本算子负责在前向传播时将输入张量从 (S, D) 扩展为 (S, m, D)，在反向传播时将梯度从 (S, m, D) 归约回 (S, D)。

该算子位于 mHC 架构的残差流扩展阶段：

```
Input x: [S, D]
        ↓
MHC Expand Forward（本算子前向）：将单流扩展为 m 路并行流
        ↓
o: [S, m, D] → 送入后续 H^res 混合矩阵计算
        ↓
MHC Expand Backward（本算子反向）：将 m 路梯度归约为单流梯度
        ↓
x_grad: [S, D]
```

**参考资料：**
- 代码实现：https://github.com/deepseek-ai/TileKernels/blob/main/tile_kernels/mhc/expand_kernel.py
- 论文：mHC: Manifold-Constrained Hyper-Connections（arXiv:2512.24880）

## 2.  输入输出说明

### 2.1 前向算子（MHC Expand Forward）

输入规格

| 参数名 | 类型 | 数据类型 | 维度(shape) | 说明 |
|--------|------|----------|-------------|------|
| x | 必选输入 | bfloat16、float16 | [S, D] | 输入张量，S 为 token 数，D 为隐藏层维度 |

输出规格

| 参数名 | 数据类型 | 维度(shape) | 说明 |
|--------|----------|-------------|------|
| o | bfloat16、float16 | [S, m, D] | 扩展后的输出张量，m 为扩展倍数 |

形状约束

- S：token 数量，取值为正整数
- D：隐藏层维度，取值为正整数
- m：扩展倍数（mhc_mult），取值为正整数
- 输出 o 的第 m 个副本与输入 x 完全相同：o[i, m, j] = x[i, j]

计算公式

```
o[i, m, j] = x[i, j]    对所有 m ∈ [0, mhc_mult)
```

即将输入 x 沿第1维复制 m 份，每个副本内容相同。

### 2.2 反向算子（MHC Expand Backward）

输入规格

| 参数名 | 类型 | 数据类型 | 维度(shape) | 说明 |
|--------|------|----------|-------------|------|
| o_grad | 必选输入 | bfloat16、float16 | [S, m, D] | 上游传来的梯度张量 |

输出规格

| 参数名 | 数据类型 | 维度(shape) | 说明 |
|--------|----------|-------------|------|
| x_grad | bfloat16、float16 | [S, D] | 归约后的梯度张量 |

形状约束

- o_grad 的形状必须与前向输出的形状一致
- x_grad 的形状必须与前向输入的形状一致

计算公式

```
x_grad[i, j] = Σ_{m=0}^{mhc_mult-1} o_grad[i, m, j]
```

即将 m 个副本的梯度沿第1维求和归约。

## 3. ⚙️ 算子逻辑说明

### 3.1 前向：数据扩展阶段

1. 将输入 x 的形状从 [S, D] 读取到本地缓存
2. 对 mhc_mult 个副本，逐个将 x 数据写入输出 o 的对应位置
3. 每个副本 o[:, m, :] 的内容与输入 x 完全相同

### 3.2 反向：梯度归约阶段

1. 初始化累加缓存为零
2. 遍历 mhc_mult 个副本，将每个 o_grad[:, m, :] 累加到缓存中
3. 将累加结果写入 x_grad

### 3.3 参考实现关键策略

参考 DeepSeek 的 TileLang 实现，采用了以下分块策略：

```
分块大小：blk_n = 32（token维度分块），blk_h = 128（隐藏维度分块）
并行策略：按 (ceildiv(S, blk_n), ceildiv(D, blk_h)) 启动 Kernel
前向：将 x 的分块读入 fragment，逐副本写入输出
反向：将 fragment 初始化为0，逐副本累加梯度，最终写回
```

## 4. 任务要求

- **精度保障**：针对不同 Data type 和不同 shape 维度，设计算子逻辑，保证算子精度正确
- **性能优化**：充分发挥系统带宽能力，算子性能更优。前向需优化数据复制效率，反向需优化归约求和效率
- **切分最优**：探索输入 tensor 的切分方式，找到不同输入 shape 场景下的最优解
- **泛化功能**：必须实现算子泛化功能，满足各类合法输入场景的计算需求
- **前向反向完整性**：需同时实现前向和反向算子，反向算子的梯度计算必须与参考实现一致

## 5. 功能示例

```python
import torch

# 示例1：前向扩展（基础用法）
# 输入形状：x=[4, 8]，mhc_mult=2
x = torch.randn(4, 8, dtype=torch.bfloat16)
mhc_mult = 2

# 前向：将 [4, 8] 扩展为 [4, 2, 8]
o = x.unsqueeze(1).expand(-1, mhc_mult, -1).clone()

# 输出形状：o=[4, 2, 8]
# 结果：o[:, 0, :] = x, o[:, 1, :] = x（两个副本内容相同）
# 验证：torch.allclose(o[:, 0, :], x) → True

# 示例2：反向归约
# 输入形状：o_grad=[4, 2, 8]
o_grad = torch.randn(4, 2, 8, dtype=torch.bfloat16)

# 反向：将 [4, 2, 8] 归约为 [4, 8]
x_grad = o_grad.sum(dim=1)

# 输出形状：x_grad=[4, 8]
# 结果：x_grad[i, j] = o_grad[i, 0, j] + o_grad[i, 1, j]

# 示例3：大模型典型规模
# 输入形状：x=[4096, 7168]，mhc_mult=4
x = torch.randn(4096, 7168, dtype=torch.bfloat16)
mhc_mult = 4

# 前向：将 [4096, 7168] 扩展为 [4096, 4, 7168]
o = x.unsqueeze(1).expand(-1, mhc_mult, -1).clone()

# 反向：将 [4096, 4, 7168] 归约为 [4096, 7168]
x_grad = o_grad.sum(dim=1)

# 验证梯度正确性
o_grad = torch.randn(4096, 4, 7168, dtype=torch.bfloat16)
x_grad = o_grad.sum(dim=1)
# 等价于：x_grad = o_grad[:, 0, :] + o_grad[:, 1, :] + o_grad[:, 2, :] + o_grad[:, 3, :]
```

## 6.  测试用例覆盖范围

- **数据类型**：bfloat16、float16
- **维度场景**：小规模(S=64, D=256, m=2)、中规模(S=1024, D=4096, m=4)、大规模(S=8192, D=7168, m=8)
- **扩展倍数**：m=2、m=4、m=8
- **前向验证**：输出每个副本与输入是否一致
- **反向验证**：梯度归约求和是否正确
- **边界场景**：S=1（单token）、D=1（单维度）、非对齐维度
- **精度场景**：float16 累加精度、bfloat16 累加精度

