// TilingKey模板定义的头文件
#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(MhcExpand,
    ASCENDC_TPL_DATATYPE_DECL(DT_X, C_DT_FLOAT16, C_DT_BF16),
    ASCENDC_TPL_BOOL_DECL(IS_BACKWARD, 0, 1),
    ASCENDC_TPL_BOOL_DECL(IS_ALIGNED, 0, 1),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_X, C_DT_FLOAT16, C_DT_BF16),
        ASCENDC_TPL_BOOL_SEL(IS_BACKWARD, 0, 1),
        ASCENDC_TPL_BOOL_SEL(IS_ALIGNED, 0, 1),
        // 纯矢量算子，声明只启动 Vector 核，省掉每个 Block 的 Cube 核启动开销。
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
    ),
);
