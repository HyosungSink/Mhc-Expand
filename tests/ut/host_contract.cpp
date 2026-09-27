// Host contract checks using CANN metadata contexts.
#include "base/context_builder/op_infer_datatype_context_builder.h"
#include "base/context_builder/op_infer_shape_context_builder.h"
#include "base/context_builder/op_tiling_context_builder.h"
#include "platform/platform_info.h"
#include "../../code/op_host/mhc_expand.cpp"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void Require(bool condition, const std::string &message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

ge::DataType DType(const std::string &name)
{
    if (name == "float16") {
        return ge::DT_FLOAT16;
    }
    if (name == "bfloat16") {
        return ge::DT_BF16;
    }
    throw std::runtime_error("unknown dtype: " + name);
}

gert::Tensor Tensor(const gert::Shape &shape, ge::DataType dtype)
{
    gert::StorageShape storage;
    storage.MutableOriginShape() = shape;
    storage.MutableStorageShape() = shape;
    return gert::Tensor(
        storage,
        gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, {}),
        gert::kOnDeviceHbm,
        dtype,
        nullptr);
}

void Check(const std::vector<std::string> &fields, fe::PlatFormInfos &platform)
{
    Require(fields.size() == 7, "expected label dtype backward S D multiplier check");
    const bool backward = std::stoi(fields[2]) != 0;
    const int64_t s = std::stoll(fields[3]);
    const int64_t d = std::stoll(fields[4]);
    int64_t multiplier = std::stoll(fields[5]);
    const std::string check = fields[6];
    const bool accepted = check == "accept" || check == "flat";
    Require(
        accepted || check == "rank" || check == "dtype" || check == "missing" ||
            check == "zero_multiplier" || check == "lane_mismatch",
        "unknown contract check: " + check);

    // Tiling reads the operand that actually carries the lanes: the expanded
    // stream for backward and the flattened expansion of the forward operand.
    // Shape inference reads the declared operand, which is what the semantic
    // shapes of the two directions describe.
    gert::Shape inputShape = backward ? gert::Shape{s, multiplier, d}
                                      : gert::Shape{s * multiplier, d};
    gert::Shape declaredShape = backward ? gert::Shape{s, multiplier, d} : gert::Shape{s, d};
    if (check == "flat") {
        Require(backward, "flat storage applies only to backward");
        inputShape = gert::Shape{s * multiplier, d};
    } else if (check == "rank") {
        inputShape = gert::Shape{s, d, 1};
        declaredShape = gert::Shape{s, d, 1};
    } else if (check == "lane_mismatch") {
        Require(backward, "lane mismatch applies only to backward");
        inputShape = gert::Shape{s, multiplier + 1, d};
    } else if (check == "zero_multiplier") {
        multiplier = 0;
    }

    const ge::DataType dtype = check == "dtype" ? ge::DT_FLOAT : DType(fields[1]);
    auto input = Tensor(inputShape, dtype);
    auto declared = Tensor(declaredShape, dtype);
    const gert::Shape outputShape = backward ? gert::Shape{s, d} : gert::Shape{s, multiplier, d};
    auto output = Tensor(outputShape, dtype);
    std::vector<uint32_t> inputInstances = {check == "missing" ? 0U : 1U};
    const std::vector<uint32_t> outputInstances = {1U};
    std::vector<gert::Tensor *> inputs;
    if (inputInstances[0] != 0) {
        inputs.push_back(&input);
    }
    std::vector<gert::Tensor *> declaredInputs;
    if (inputInstances[0] != 0) {
        declaredInputs.push_back(&declared);
    }
    std::vector<gert::Tensor *> outputs = {&output};
    int compileInfo = 0;
    auto workspace = gert::ContinuousVector::Create<size_t>(1);
    gert::OpTilingContextBuilder tilingBuilder;
    auto tilingHolder = tilingBuilder.OpType("MhcExpand")
        .OpName("host_contract")
        .IOInstanceNum(inputInstances, outputInstances)
        .AppendAttr(multiplier)
        .AppendAttr(backward)
        .InputTensors(inputs)
        .OutputTensors(outputs)
        .PlatformInfo(&platform)
        .CompileInfo(&compileInfo)
        .Workspace(reinterpret_cast<gert::ContinuousVector *>(workspace.get()))
        .TilingDataSize(sizeof(MhcExpandTilingData))
        .Build();
    auto *tilingContext = tilingHolder.GetContext();
    Require(tilingContext != nullptr, "could not construct TilingContext");
    Require(
        (optiling::TilingFunc(tilingContext) == ge::GRAPH_SUCCESS) == accepted,
        "Tiling acceptance disagrees with " + check);
    if (!accepted) {
        return;
    }

    gert::OpInferShapeContextBuilder shapeBuilder;
    auto shapeHolder = shapeBuilder.OpType("MhcExpand")
        .OpName("host_shape")
        .IOInstanceNum(inputInstances, outputInstances)
        .AppendAttr(multiplier)
        .AppendAttr(backward)
        .InputTensors(declaredInputs)
        .OutputTensorDesc(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .Build();
    gert::OpInferDataTypeContextBuilder dtypeBuilder;
    auto dtypeHolder = dtypeBuilder.OpType("MhcExpand")
        .OpName("host_dtype")
        .IOInstanceNum(inputInstances, outputInstances)
        .AppendAttr(multiplier)
        .AppendAttr(backward)
        .InputTensorDesc(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND)
        .Build();
    auto *shapeContext = shapeHolder.GetContext();
    auto *dtypeContext = dtypeHolder.GetContext();
    Require(shapeContext != nullptr && dtypeContext != nullptr, "could not construct inference contexts");
    Require(ge::InferShape(shapeContext) == ge::GRAPH_SUCCESS, "InferShape failed");
    Require(ge::InferDataType(dtypeContext) == ge::GRAPH_SUCCESS, "InferDataType failed");
    const auto *actualShape = shapeContext->GetOutputShape(0);
    Require(actualShape != nullptr && *actualShape == outputShape, "incorrect output shape");
    Require(dtypeContext->GetOutputDataType(0) == dtype, "incorrect output dtype");
}
}  // namespace

int main(int argc, char **argv)
{
    try {
        Require(argc == 2, "expected SoC name; specifications are read from stdin");
        fe::PlatFormInfos platform;
        fe::OptionalInfos optional;
        auto &manager = fe::PlatformInfoManager::Instance();
        Require(
            manager.InitializePlatformInfo() == 0 &&
                manager.GetPlatformInfos(argv[1], platform, optional) == 0,
            "CANN platform metadata is unavailable: " + std::string(argv[1]));
        std::string line;
        size_t count = 0;
        size_t failures = 0;
        while (std::getline(std::cin, line)) {
            std::istringstream stream(line);
            std::vector<std::string> fields;
            for (std::string field; stream >> field;) {
                fields.push_back(field);
            }
            if (fields.empty()) {
                continue;
            }
            ++count;
            try {
                Check(fields, platform);
                std::cout << "PASS " << fields[0] << '\n';
            } catch (const std::exception &error) {
                ++failures;
                std::cerr << "FAIL " << fields[0] << ": " << error.what() << '\n';
            }
        }
        Require(count > 0, "no Host contract specifications supplied");
        return failures == 0 ? 0 : 1;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
