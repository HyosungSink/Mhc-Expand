// Host side Tiling implementation for mHC Expand.
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mhc_expand_tiling.h"
#include "../op_kernel/tiling_key_mhc_expand.h"

namespace {
constexpr uint32_t MHC_TILE_MIN = 1024;
constexpr uint32_t MHC_TILE_MAX = 4096;
constexpr uint32_t MHC_UB_BUDGET = 160 * 1024;
}  // namespace

namespace optiling {
static uint32_t AlignUp(uint32_t value, uint32_t unit) {
    return (value + unit - 1) / unit * unit;
}

static uint32_t AlignDown(uint32_t value, uint32_t unit) {
    return value / unit * unit;
}

// The largest column tile whose live buffers still fit the UB budget.
//
// The forward direction stages an input tile and an output tile, each with a
// depth of two, while the reduction direction additionally keeps one queued
// input tile, a fp32 running total and a fp32 widened tile.
//
// A column tile is always a whole number of 32B blocks so that every transfer
// of a tile moves complete blocks and a vector store never crosses the end of
// its tile. The only exception is a row that is not block aligned at all: such
// a row becomes a single tile that the element wise transfer path moves on its
// own, and it is never split.
static uint32_t ChooseTileLen(uint32_t rowLen, uint32_t byteSize, bool backward) {
    const uint32_t unit = 32 / byteSize;
    const uint32_t candidates[] = {MHC_TILE_MAX, 2048, MHC_TILE_MIN};
    for (uint32_t index = 0; index < sizeof(candidates) / sizeof(candidates[0]); ++index) {
        uint32_t candidate = candidates[index];
        if (candidate > rowLen) {
            candidate = rowLen;
        }
        // A row that fits in one block aligned tile is kept whole, so a narrow
        // row is never split into a block sized tile plus a tiny tail tile that
        // would force an element wise transfer across a row boundary.
        const uint32_t clamped = AlignUp(candidate, unit);
        if (clamped <= rowLen) {
            candidate = clamped;
        } else if (rowLen > unit) {
            candidate = AlignDown(candidate, unit);
        }
        if (candidate == 0) {
            candidate = AlignUp(rowLen, unit);
        }
        const uint32_t tileBytes = AlignUp(candidate * byteSize, 32);
        uint32_t live = 4 * tileBytes;
        if (backward) {
            const uint32_t wideBytes = AlignUp(candidate * sizeof(float), 32);
            live += 2 * tileBytes + 2 * wideBytes;
        }
        if (live <= MHC_UB_BUDGET) {
            return candidate;
        }
    }
    return rowLen;
}

// A block owns `rowTile` whole rows of every column tile and reads its own
// index, so its row range follows from the block index alone. Splitting by row
// keeps every column tile of a row inside one block.
static uint32_t RowsPerBlock(uint32_t rows, uint32_t maxBlocks) {
    if (maxBlocks == 0) {
        maxBlocks = 1;
    }
    const uint32_t rowsPerBlock = (rows + maxBlocks - 1) / maxBlocks;
    return rowsPerBlock == 0 ? 1 : rowsPerBlock;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t numCores = static_cast<uint32_t>(platform.GetCoreNumAiv());
    const gert::Tensor *tensorX = context->GetRequiredInputTensor(0);
    const gert::StorageShape *shapeX = context->GetInputShape(0);
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (tensorX == nullptr || shapeX == nullptr || attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t *attrMult = attrs->GetInt(0);
    const bool *attrBackward = attrs->GetBool(1);
    const int64_t multiplier = attrMult == nullptr ? 0 : *attrMult;
    const bool backward = attrBackward != nullptr && *attrBackward;

    const ge::DataType dtypeX = tensorX->GetDataType();
    const int dtypeWidth = ge::GetSizeByDataType(dtypeX);
    if (dtypeWidth != 2) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t byteSize = static_cast<uint32_t>(dtypeWidth);

    // `rows` is the number of output rows of this direction and `d` is the
    // length of every row, which is also the innermost stride of both operands.
    // Forward reads one stream of S rows and writes S * multiplier rows;
    // backward reads S * multiplier rows and writes S rows.
    const gert::Shape &shape = shapeX->GetOriginShape();
    const size_t rank = shape.GetDimNum();
    if (rank != 2 && rank != 3) {
        return ge::GRAPH_FAILED;
    }
    const int64_t elements = tensorX->GetShapeSize();
    const int64_t d = shape.GetDim(rank - 1);
    int64_t streams = 0;
    size_t rows = 0;
    if (backward) {
        if (rank == 3) {
            if (shape.GetDim(1) != multiplier) {
                return ge::GRAPH_FAILED;
            }
            // Backward reads S * multiplier rows and writes S rows, so the
            // output rows of this direction are the S rows of the input stream.
            streams = shape.GetDim(0);
            rows = static_cast<size_t>(streams);
        } else {
            // In the rank-2 form the lanes of one row are adjacent, so the
            // operand only has to divide evenly into full rows.
            if (d <= 0 || elements % d != 0 || multiplier <= 0 || (elements / d) % multiplier != 0) {
                return ge::GRAPH_FAILED;
            }
            streams = elements / d / multiplier;
            rows = static_cast<size_t>(streams);
        }
    } else {
        if (rank != 2) {
            return ge::GRAPH_FAILED;
        }
        streams = shape.GetDim(0);
        rows = static_cast<size_t>(streams) * static_cast<size_t>(multiplier);
        // The forward operand carries the single stream that is replicated, so
        // it has to cover S * D elements. A rank-2 declaration whose leading
        // extent is the collapsed lane axis reports S * D here as well, and this
        // direction never reads past the S * D elements it folds.
        if (elements < static_cast<int64_t>(streams) * d) {
            return ge::GRAPH_FAILED;
        }
    }
    if (streams <= 0 || d <= 0 || multiplier <= 0 || rows == 0) {
        return ge::GRAPH_FAILED;
    }
    if (d > 0x7FFFFFFF || rows > 0x7FFFFFFF) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t tileLen = ChooseTileLen(static_cast<uint32_t>(d), byteSize, backward);
    const uint32_t rowCount = static_cast<uint32_t>(rows);
    const uint64_t tileCount = (static_cast<uint64_t>(d) + tileLen - 1) / tileLen;
    if (tileCount == 0 || tileCount > 0xFFFF) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t colTiles = static_cast<uint32_t>(tileCount);

    const uint32_t maxCores = numCores == 0 ? 1 : numCores;
    const uint32_t maxBlocks = rowCount < maxCores ? rowCount : maxCores;
    const uint32_t rowTile = RowsPerBlock(rowCount, maxBlocks);
    const uint32_t blocks = (rowCount + rowTile - 1) / rowTile;

    MhcExpandTilingData *tiling = context->GetTilingData<MhcExpandTilingData>();
    tiling->rowTile = rowTile;
    tiling->rowTotal = rowCount;
    tiling->rowLen = static_cast<uint32_t>(d);
    tiling->tileLen = tileLen;
    tiling->colTiles = colTiles;
    tiling->lanes = static_cast<uint32_t>(multiplier);
    tiling->backward = backward ? 1u : 0u;

    context->SetBlockDim(blocks);
    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dtypeX));
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *inputShape = context->GetInputShape(0);
    gert::Shape *outputShape = context->GetOutputShape(0);
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (inputShape == nullptr || outputShape == nullptr || attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t *attrMult = attrs->GetInt(0);
    const bool *attrBackward = attrs->GetBool(1);
    const int64_t multiplier = attrMult == nullptr ? 0 : *attrMult;
    const bool backward = attrBackward != nullptr && *attrBackward;
    const size_t rank = inputShape->GetDimNum();
    *outputShape = *inputShape;
    if (multiplier <= 0) {
        return ge::GRAPH_FAILED;
    }
    if (backward) {
        if (rank != 3 && rank != 2) {
            return ge::GRAPH_FAILED;
        }
        if (rank == 3) {
            if (inputShape->GetDim(1) != multiplier) {
                return ge::GRAPH_FAILED;
            }
            outputShape->SetDimNum(2);
            outputShape->SetDim(0, inputShape->GetDim(0));
            outputShape->SetDim(1, inputShape->GetDim(2));
        } else {
            outputShape->SetDimNum(2);
            outputShape->SetDim(0, inputShape->GetDim(0) / multiplier);
            outputShape->SetDim(1, inputShape->GetDim(1));
        }
    } else {
        if (rank != 2) {
            return ge::GRAPH_FAILED;
        }
        outputShape->SetDimNum(3);
        outputShape->SetDim(0, inputShape->GetDim(0));
        outputShape->SetDim(1, multiplier);
        outputShape->SetDim(2, inputShape->GetDim(1));
    }
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class MhcExpand : public OpDef {
public:
    explicit MhcExpand(const char *name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("o")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("mhc_mult").AttrType(OPTIONAL).Int(2);
        this->Attr("backward").AttrType(OPTIONAL).Bool();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(MhcExpand);
}  // namespace ops
