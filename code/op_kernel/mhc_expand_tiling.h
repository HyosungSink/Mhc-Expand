// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

// S 与列 tile 个数在 Kernel 侧未被使用/可由 rowLen、tileCols 推出，不进入
// TilingData：GET_TILING_DATA 按 32B 对齐从 GM 搬到栈上，结构体越小、小 shape
// 下这次搬运的开销越低（8 个 uint32_t 字段正好压满一个 32B block）。
struct MhcExpandTilingData {
    uint32_t rowLen;        // D：单个隐藏行的元素个数
    uint32_t mult;          // mhc_mult：扩展倍数
    uint32_t tileRows;      // 单个 UB tile 暂存的行数
    uint32_t tileCols;      // 单个 UB tile 暂存的列数
    uint32_t unitsPerCore;  // 每个 Block 的基础工作单元数
    uint32_t tailUnits;     // 多分配一个工作单元的 Block 个数
    uint32_t ubPitch;       // UB 内相邻行的元素间距
    uint32_t laneGroup;     // 单次 UB 暂存覆盖的副本份数：前向是预排布份数，反向是合并读取份数
};
