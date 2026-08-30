// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mhc_expand_tiling.h"
#include "../op_kernel/tiling_key_mhc_expand.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 示例: 获取平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        // 示例: 获取算子输入数组信息
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_x = tensor_x->GetDataType(); // 获取数据类型
        int dtype_size_x = ge::GetSizeByDataType(dtype_x); // 获取数据类型的字长
        uint32_t length_x = tensor_x->GetShapeSize(); // 获取元素个数
        uint32_t size_x = tensor_x->GetSize(); // 获取内存大小
        // 示例: 获取算子输入属性
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        const int64_t *attr_mhc_mult = attrs->GetInt(0);
        const bool *attr_backward = attrs->GetBool(1);
        // 示例: 配置tiling key, 从而实现kernel侧不同数据类型/算法的区分
        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);
        // 示例: 计算tiling方案并填充tiling结构体
        MhcExpandTilingData *tiling = context->GetTilingData<MhcExpandTilingData>();
        tiling->length = length_x;
        // 配置启动核数
        context->SetBlockDim(num_cores_aiv);
        // 配置workspace大小
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class MhcExpand : public OpDef {
    public:
        explicit MhcExpand(const char *name) : OpDef(name) {
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND});
            this->Output("o")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND});
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
