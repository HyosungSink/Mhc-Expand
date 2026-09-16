// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mhc_expand_tiling.h"
#include "../op_kernel/tiling_key_mhc_expand.h"

namespace {
constexpr uint32_t BLOCK_BYTES = 32U;
constexpr uint64_t MAX_BLOCK_COUNT = 4095U;
constexpr uint64_t MAX_GAP_BLOCKS = 65535U;
constexpr uint64_t FORWARD_STAGE_ELEMS = 32768U;  // 前向：2 份暂存缓冲
constexpr uint64_t BACKWARD_STAGE_ELEMS = 10240U; // 反向：lane/累加/输出共 16B 每元素
constexpr uint64_t MIN_TILE_ELEMS = 2048U;        // 单个 tile 的元素下限，用于收敛小形状的核数
constexpr uint64_t FORWARD_REPLICA_ELEMS = 45056U; // 前向副本路径：输入加副本的单份元素上限
constexpr int64_t DEFAULT_MULT = 2;

inline uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return (value + divisor - 1U) / divisor;
}

// 从属性中读取扩展倍数与前反向开关，缺省值与算子原型保持一致。
inline bool ReadAttrs(const gert::RuntimeAttrs *attrs, int64_t &mult, bool &backward)
{
    if (attrs == nullptr) {
        return false;
    }
    const int64_t *multAttr = attrs->GetInt(0);
    const bool *backwardAttr = attrs->GetBool(1);
    mult = (multAttr == nullptr) ? DEFAULT_MULT : *multAttr;
    backward = (backwardAttr == nullptr) ? false : *backwardAttr;
    return mult > 0;
}

// 由输入 Shape 推导 (S, D)。前向接受 [D] 与 [S, D]；反向同时接受题面布局
// [.., mhc_mult, D] 与评测布局 [S * mhc_mult, D]，不依赖任何固定 Shape 名单。
bool ResolveGeometry(const gert::Shape &shape, int64_t mult, bool backward, int64_t &rows, int64_t &cols)
{
    const size_t rank = shape.GetDimNum();
    if (rank == 0U) {
        return false;
    }
    int64_t total = 1;
    for (size_t index = 0U; index < rank; ++index) {
        const int64_t dim = shape.GetDim(index);
        if (dim < 0) {
            return false;
        }
        total *= dim;
    }
    if (!backward) {
        if (rank > 2U) {
            return false;
        }
        cols = shape.GetDim(rank - 1U);
        if (cols <= 0) {
            rows = 0;
            cols = 0;
            return total == 0;
        }
        rows = total / cols;
        return true;
    }
    if (rank >= 3U) {
        if (shape.GetDim(rank - 2U) != mult) {
            return false;
        }
        cols = shape.GetDim(rank - 1U);
    } else {
        if (shape.GetDim(0) % mult != 0) {
            return false;
        }
        cols = (rank == 2U) ? shape.GetDim(1) : (shape.GetDim(0) / mult);
    }
    if (cols <= 0) {
        rows = 0;
        cols = 0;
        return total == 0;
    }
    const int64_t lanes = mult * cols;
    if (total % lanes != 0) {
        return false;
    }
    rows = total / lanes;
    return true;
}
}  // namespace

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        if (context == nullptr) {
            return ge::GRAPH_FAILED;
        }
        const gert::Tensor *tensorX = context->GetInputTensor(0);
        if (tensorX == nullptr) {
            return ge::GRAPH_FAILED;
        }
        const ge::DataType dtypeX = tensorX->GetDataType();
        if (dtypeX != ge::DT_FLOAT16 && dtypeX != ge::DT_BF16) {
            return ge::GRAPH_FAILED;
        }
        int64_t mult = DEFAULT_MULT;
        bool backward = false;
        if (!ReadAttrs(context->GetAttrs(), mult, backward)) {
            return ge::GRAPH_FAILED;
        }
        int64_t rows = 0;
        int64_t cols = 0;
        if (!ResolveGeometry(tensorX->GetStorageShape(), mult, backward, rows, cols)) {
            return ge::GRAPH_FAILED;
        }

        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t coreNum = platform.GetCoreNumAiv();
        if (coreNum == 0U) {
            coreNum = 1U;
        }
        const uint64_t elemSize = static_cast<uint64_t>(ge::GetSizeByDataType(dtypeX));
        const uint64_t elemsPerBlock = BLOCK_BYTES / elemSize;

        MhcExpandTilingData *tiling = context->GetTilingData<MhcExpandTilingData>();
        if (tiling == nullptr) {
            return ge::GRAPH_FAILED;
        }
        uint32_t blockDim = 1U;
        if (rows > 0 && cols > 0) {
            const uint64_t colCount = static_cast<uint64_t>(cols);
            const bool alignedRow = (colCount * elemSize) % BLOCK_BYTES == 0U;
            const uint64_t pitch = alignedRow ? colCount : CeilDiv(colCount, elemsPerBlock) * elemsPerBlock;
            const uint64_t budget = backward ? BACKWARD_STAGE_ELEMS : FORWARD_STAGE_ELEMS;

            // 工作单元：列未切分时是一整行，列切分后是一行中的一个列块。
            uint64_t tileCols = colCount;
            uint64_t colTiles = 1U;
            uint64_t ubPitch = pitch;
            uint64_t maxRows = 1U;
            if (pitch <= budget) {
                maxRows = budget / pitch;
                // 多行搬运依赖 uint16 的 blockLen/Gap 字段，越界时退回单行。
                if (!alignedRow || (colCount * elemSize) / BLOCK_BYTES > MAX_GAP_BLOCKS ||
                    (static_cast<uint64_t>(mult - 1) * colCount * elemSize) / BLOCK_BYTES > MAX_GAP_BLOCKS) {
                    maxRows = 1U;
                }
                if (maxRows > MAX_BLOCK_COUNT) {
                    maxRows = MAX_BLOCK_COUNT;
                }
            } else {
                tileCols = (budget / elemsPerBlock) * elemsPerBlock;
                if (tileCols == 0U) {
                    tileCols = elemsPerBlock;
                }
                colTiles = CeilDiv(colCount, tileCols);
                ubPitch = tileCols;
            }

            // 前向把 repLanes 份副本先在 UB 内排好，MTE3 就能按 repLanes*D 连续写出。
            uint64_t repLanes = 1U;
            if (!backward && alignedRow && colTiles == 1U && mult > 1) {
                for (uint64_t candidate = static_cast<uint64_t>(mult); candidate >= 2U; --candidate) {
                    if (static_cast<uint64_t>(mult) % candidate != 0U) {
                        continue;
                    }
                    if (colCount * (1U + candidate) > FORWARD_REPLICA_ELEMS) {
                        continue;
                    }
                    if (((static_cast<uint64_t>(mult) - candidate) * colCount * elemSize) / BLOCK_BYTES >
                        MAX_GAP_BLOCKS) {
                        continue;
                    }
                    repLanes = candidate;
                    break;
                }
            }
            if (repLanes > 1U) {
                maxRows = FORWARD_REPLICA_ELEMS / (colCount * (1U + repLanes));
                if (maxRows == 0U) {
                    maxRows = 1U;
                }
                if (maxRows > MAX_BLOCK_COUNT) {
                    maxRows = MAX_BLOCK_COUNT;
                }
            }

            // 小形状下每个核的固定开销主导耗时，给每个核设一个工作量下限来收敛核数。
            const uint64_t totalUnits = static_cast<uint64_t>(rows) * colTiles;
            uint64_t denseUnits = 1U;
            if (colTiles == 1U) {
                denseUnits = CeilDiv(MIN_TILE_ELEMS, pitch);
            }
            uint64_t blocks = CeilDiv(totalUnits, denseUnits);
            if (blocks > coreNum) {
                blocks = coreNum;
            }
            if (blocks == 0U) {
                blocks = 1U;
            }
            blockDim = static_cast<uint32_t>(blocks);

            // 工作单元按核均分，最忙的核只比平均多一个单元。
            const uint64_t unitsPerCore = totalUnits / blocks;
            const uint64_t tailUnits = totalUnits % blocks;
            uint64_t tileRows = 1U;
            if (colTiles == 1U) {
                const uint64_t busiest = unitsPerCore + ((tailUnits != 0U) ? 1U : 0U);
                const uint64_t chunks = CeilDiv(busiest, maxRows);
                tileRows = (chunks == 0U) ? 1U : CeilDiv(busiest, chunks);
                if (tileRows == 0U) {
                    tileRows = 1U;
                }
            }

            tiling->rowLen = static_cast<uint32_t>(cols);
            tiling->mult = static_cast<uint32_t>(mult);
            tiling->rowCount = static_cast<uint32_t>(rows);
            tiling->tileRows = static_cast<uint32_t>(tileRows);
            tiling->tileCols = static_cast<uint32_t>(tileCols);
            tiling->colTiles = static_cast<uint32_t>(colTiles);
            tiling->unitsPerCore = static_cast<uint32_t>(unitsPerCore);
            tiling->tailUnits = static_cast<uint32_t>(tailUnits);
            tiling->ubPitch = static_cast<uint32_t>(ubPitch);
            tiling->repLanes = static_cast<uint32_t>(repLanes);
        } else {
            tiling->rowLen = 0U;
            tiling->mult = static_cast<uint32_t>(mult);
            tiling->rowCount = 0U;
            tiling->tileRows = 1U;
            tiling->tileCols = 1U;
            tiling->colTiles = 1U;
            tiling->unitsPerCore = 0U;
            tiling->tailUnits = 0U;
            tiling->ubPitch = static_cast<uint32_t>(elemsPerBlock);
            tiling->repLanes = 1U;
        }

        uint32_t dtypeKey = static_cast<uint32_t>(dtypeX);
        uint32_t backwardKey = backward ? 1U : 0U;
        uint32_t alignedKey =
            (cols > 0 && (static_cast<uint64_t>(cols) * elemSize) % BLOCK_BYTES == 0U) ? 1U : 0U;
        ASCENDC_TPL_SEL_PARAM(context, dtypeKey, backwardKey, alignedKey);

        context->SetBlockDim(blockDim);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        if (currentWorkspace != nullptr) {
            currentWorkspace[0] = 0;
        }
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context)
    {
        if (context == nullptr) {
            return GRAPH_FAILED;
        }
        const gert::Shape *inputShape = context->GetInputShape(0);
        gert::Shape *outputShape = context->GetOutputShape(0);
        if (inputShape == nullptr || outputShape == nullptr) {
            return GRAPH_FAILED;
        }
        int64_t mult = DEFAULT_MULT;
        bool backward = false;
        if (!ReadAttrs(context->GetAttrs(), mult, backward)) {
            return GRAPH_FAILED;
        }
        const size_t rank = inputShape->GetDimNum();
        if (rank == 0U) {
            return GRAPH_FAILED;
        }
        if (!backward) {
            if (rank > 2U) {
                return GRAPH_FAILED;
            }
            outputShape->SetDimNum(rank + 1U);
            for (size_t index = 0U; index + 1U < rank; ++index) {
                outputShape->SetDim(index, inputShape->GetDim(index));
            }
            outputShape->SetDim(rank - 1U, mult);
            outputShape->SetDim(rank, inputShape->GetDim(rank - 1U));
            return GRAPH_SUCCESS;
        }
        if (rank >= 3U) {
            if (inputShape->GetDim(rank - 2U) != mult) {
                return GRAPH_FAILED;
            }
            outputShape->SetDimNum(rank - 1U);
            for (size_t index = 0U; index + 2U < rank; ++index) {
                outputShape->SetDim(index, inputShape->GetDim(index));
            }
            outputShape->SetDim(rank - 2U, inputShape->GetDim(rank - 1U));
            return GRAPH_SUCCESS;
        }
        if (inputShape->GetDim(0) % mult != 0) {
            return GRAPH_FAILED;
        }
        if (rank == 2U) {
            outputShape->SetDimNum(2U);
            outputShape->SetDim(0, inputShape->GetDim(0) / mult);
            outputShape->SetDim(1, inputShape->GetDim(1));
        } else {
            outputShape->SetDimNum(1U);
            outputShape->SetDim(0, inputShape->GetDim(0) / mult);
        }
        return GRAPH_SUCCESS;
    }

    static graphStatus InferDataType(gert::InferDataTypeContext *context)
    {
        if (context == nullptr) {
            return GRAPH_FAILED;
        }
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
            this->Attr("backward").AttrType(OPTIONAL).Bool(false);
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(MhcExpand);
}  // namespace ops
