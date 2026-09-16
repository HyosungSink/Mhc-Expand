// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MhcExpandTilingData {
    uint32_t rowLen;        // D：单个隐藏行的元素个数
    uint32_t mult;          // mhc_mult：扩展倍数
    uint32_t rowCount;      // S：token 行数
    uint32_t tileRows;      // 单个 UB tile 暂存的行数
    uint32_t tileCols;      // 单个 UB tile 暂存的列数
    uint32_t colTiles;      // 每行拆分出的列 tile 个数
    uint32_t unitsPerCore;  // 每个 Block 的基础工作单元数
    uint32_t tailUnits;     // 多分配一个工作单元的 Block 个数
    uint32_t ubPitch;       // UB 内相邻行的元素间距
    uint32_t repLanes;      // 前向在 UB 内预排布的副本份数
};
