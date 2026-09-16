// TilingKey template definition header.
#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

// The tiling key only carries the element type. The direction and the column
// split stay in the tiling data, so the reachable key set is exactly the set of
// supported element types.
ASCENDC_TPL_ARGS_DECL(MhcExpand,
    ASCENDC_TPL_DATATYPE_DECL(DT_X, C_DT_FLOAT16, C_DT_BF16),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_X, C_DT_FLOAT16),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_X, C_DT_BF16),
    ),
);
