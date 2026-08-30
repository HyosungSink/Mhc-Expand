// One-process multi-case ACLNN runner. Numerical comparison remains in Python.
#include <acl/acl.h>
#include <aclnn/acl_meta.h>

#include <chrono>
#include <cstdint>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using WorkspaceFn = aclError (*)(const aclTensor*, int64_t, bool, const aclTensor*, uint64_t*, aclOpExecutor**);
using ExecuteFn = aclError (*)(void*, uint64_t, aclOpExecutor*, aclrtStream);

struct CaseSpec {
    std::string name = "single";
    std::string inputPath;
    std::string outputPath;
    int64_t s = 0;
    int64_t d = 0;
    int64_t m = 0;
    bool backward = false;
    bool bf16 = false;
    bool flatBackward = false;
    int warmup = 5;
    int iterations = 20;
    size_t guardBytes = 64;
    uint16_t poison = 0;
};

static std::string JsonEscape(const std::string& value) {
    std::ostringstream output;
    for (const unsigned char character : value) {
        switch (character) {
            case '\\': output << "\\\\"; break;
            case '"': output << "\\\""; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (character < 0x20) {
                    output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                           << static_cast<int>(character) << std::dec;
                } else {
                    output << character;
                }
        }
    }
    return output.str();
}

static void Check(aclError status, const char* stage) {
    if (status != ACL_SUCCESS) {
        throw std::runtime_error(std::string(stage) + " error_code=" + std::to_string(status));
    }
}

class Runtime {
public:
    Runtime(const std::string& library, int device, int vectorCores,
            const std::string& resourceMode, const std::string& allocationName)
        : device_(device), vectorCores_(vectorCores), resourceMode_(resourceMode) {
        if (resourceMode != "device" && resourceMode != "stream") {
            throw std::runtime_error("invalid resource mode");
        }
        if (allocationName == "normal-only") {
            allocationPolicy = ACL_MEM_MALLOC_NORMAL_ONLY;
        } else if (allocationName == "huge-first") {
            allocationPolicy = ACL_MEM_MALLOC_HUGE_FIRST;
        } else {
            throw std::runtime_error("invalid allocation policy");
        }
        handle_ = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (handle_ == nullptr) throw std::runtime_error(dlerror());
        workspace = reinterpret_cast<WorkspaceFn>(dlsym(handle_, "aclnnMhcExpandGetWorkspaceSize"));
        execute = reinterpret_cast<ExecuteFn>(dlsym(handle_, "aclnnMhcExpand"));
        if (workspace == nullptr || execute == nullptr) throw std::runtime_error("missing ACLNN symbol");
        Dl_info loaded{};
        dladdr(reinterpret_cast<void*>(execute), &loaded);
        loadedLibrary = loaded.dli_fname == nullptr ? library : loaded.dli_fname;
        Check(aclInit(nullptr), "aclInit");
        initialized_ = true;
        Check(aclrtSetDevice(device_), "aclrtSetDevice");
        deviceSet_ = true;
        Check(aclrtGetDeviceResLimit(device_, ACL_RT_DEV_RES_VECTOR_CORE, &defaultVectorCores),
              "get default vector resources");
        if (vectorCores_ > 0 && resourceMode_ == "device") {
            Check(aclrtSetDeviceResLimit(device_, ACL_RT_DEV_RES_VECTOR_CORE, vectorCores_),
                  "set process vector resources");
            deviceLimitSet_ = true;
        }
        Check(aclrtGetDeviceResLimit(device_, ACL_RT_DEV_RES_VECTOR_CORE, &effectiveVectorCores),
              "get effective vector resources");
        Check(aclrtCreateStream(&stream), "aclrtCreateStream");
        streamCreated_ = true;
        if (vectorCores_ > 0 && resourceMode_ == "stream") {
            Check(aclrtSetStreamResLimit(stream, ACL_RT_DEV_RES_VECTOR_CORE, vectorCores_),
                  "set stream vector resources");
            streamLimitSet_ = true;
        }
        Check(aclrtGetStreamResLimit(stream, ACL_RT_DEV_RES_VECTOR_CORE, &streamVectorCores),
              "get stream vector resources");
    }

    ~Runtime() {
        if (streamLimitSet_) aclrtResetStreamResLimit(stream);
        if (streamCreated_) aclrtDestroyStream(stream);
        if (deviceLimitSet_) aclrtResetDeviceResLimit(device_);
        if (deviceSet_) aclrtResetDevice(device_);
        if (initialized_) aclFinalize();
        if (handle_ != nullptr) dlclose(handle_);
    }

    WorkspaceFn workspace = nullptr;
    ExecuteFn execute = nullptr;
    aclrtStream stream = nullptr;
    aclrtMemMallocPolicy allocationPolicy = ACL_MEM_MALLOC_HUGE_FIRST;
    std::string loadedLibrary;
    uint32_t defaultVectorCores = 0;
    uint32_t effectiveVectorCores = 0;
    uint32_t streamVectorCores = 0;

private:
    int device_ = 0;
    int vectorCores_ = 0;
    std::string resourceMode_;
    void* handle_ = nullptr;
    bool initialized_ = false;
    bool deviceSet_ = false;
    bool streamCreated_ = false;
    bool deviceLimitSet_ = false;
    bool streamLimitSet_ = false;
};

static std::vector<CaseSpec> ReadPlan(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open plan: " + path);
    std::vector<CaseSpec> cases;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        CaseSpec item;
        int backward = 0, bf16 = 0, flat = 0;
        std::string poison;
        std::istringstream fields(line);
        fields >> std::quoted(item.name) >> std::quoted(item.inputPath) >> std::quoted(item.outputPath)
               >> item.s >> item.d >> item.m >> backward >> bf16 >> flat
               >> item.warmup >> item.iterations >> item.guardBytes >> poison;
        if (!fields || item.s <= 0 || item.d <= 0 || item.m <= 0 || item.warmup < 0 ||
            item.iterations < 1) {
            throw std::runtime_error("invalid plan row: " + line);
        }
        item.backward = backward != 0;
        item.bf16 = bf16 != 0;
        item.flatBackward = flat != 0;
        item.poison = static_cast<uint16_t>(std::stoul(poison, nullptr, 0));
        cases.push_back(item);
    }
    if (cases.empty()) throw std::runtime_error("plan contains no cases");
    return cases;
}

static bool RunCase(Runtime& runtime, const CaseSpec& item, bool workspaceOnly) {
    const size_t inputCount = static_cast<size_t>(item.s) * item.d * (item.backward ? item.m : 1);
    const size_t outputCount = static_cast<size_t>(item.s) * item.d * (item.backward ? 1 : item.m);
    std::vector<uint16_t> hostInput(inputCount);
    std::ifstream inputFile(item.inputPath, std::ios::binary | std::ios::ate);
    if (!inputFile || static_cast<size_t>(inputFile.tellg()) != inputCount * 2) {
        throw std::runtime_error("input size mismatch: " + item.inputPath);
    }
    inputFile.seekg(0);
    inputFile.read(reinterpret_cast<char*>(hostInput.data()), inputCount * 2);

    void* inputAllocation = nullptr;
    void* outputAllocation = nullptr;
    Check(aclrtMalloc(&inputAllocation, inputCount * 2 + 2 * item.guardBytes,
                      runtime.allocationPolicy), "input allocation");
    Check(aclrtMalloc(&outputAllocation, outputCount * 2 + 2 * item.guardBytes,
                      runtime.allocationPolicy), "output allocation");
    auto* inputData = static_cast<uint8_t*>(inputAllocation) + item.guardBytes;
    auto* outputData = static_cast<uint8_t*>(outputAllocation) + item.guardBytes;
    Check(aclrtMemset(inputAllocation, inputCount * 2 + 2 * item.guardBytes, 0xa5,
                      inputCount * 2 + 2 * item.guardBytes), "input guard fill");
    Check(aclrtMemset(outputAllocation, outputCount * 2 + 2 * item.guardBytes, 0xa5,
                      outputCount * 2 + 2 * item.guardBytes), "output guard fill");
    Check(aclrtMemcpy(inputData, inputCount * 2, hostInput.data(), inputCount * 2,
                      ACL_MEMCPY_HOST_TO_DEVICE), "copy input");
    std::vector<uint16_t> hostOutput(outputCount, item.poison);
    Check(aclrtMemcpy(outputData, outputCount * 2, hostOutput.data(), outputCount * 2,
                      ACL_MEMCPY_HOST_TO_DEVICE), "poison output");

    std::vector<int64_t> inputShape = item.backward
        ? (item.flatBackward ? std::vector<int64_t>{item.s * item.m, item.d}
                             : std::vector<int64_t>{item.s, item.m, item.d})
        : std::vector<int64_t>{item.s, item.d};
    std::vector<int64_t> outputShape = item.backward
        ? std::vector<int64_t>{item.s, item.d}
        : std::vector<int64_t>{item.s, item.m, item.d};
    auto dtype = item.bf16 ? ACL_BF16 : ACL_FLOAT16;
    aclTensor* input = aclCreateTensor(inputShape.data(), inputShape.size(), dtype, nullptr, 0,
                                       ACL_FORMAT_ND, inputShape.data(), inputShape.size(), inputData);
    aclTensor* output = aclCreateTensor(outputShape.data(), outputShape.size(), dtype, nullptr, 0,
                                        ACL_FORMAT_ND, outputShape.data(), outputShape.size(), outputData);
    if (input == nullptr || output == nullptr) throw std::runtime_error("aclCreateTensor");

    uint64_t workspaceBytes = 0;
    aclOpExecutor* executor = nullptr;
    if (workspaceOnly) {
        Check(runtime.workspace(input, item.m, item.backward, output, &workspaceBytes, &executor),
              "GetWorkspaceSize");
        std::cout << "{\"case\":\"" << JsonEscape(item.name)
                  << "\",\"status\":\"workspace\",\"workspace_bytes\":" << workspaceBytes
                  << ",\"library\":\"" << JsonEscape(runtime.loadedLibrary) << "\"}" << std::endl;
        aclDestroyTensor(input);
        aclDestroyTensor(output);
        aclrtFree(inputAllocation);
        aclrtFree(outputAllocation);
        return true;
    }

    std::vector<double> wallUs;
    for (int iteration = 0; iteration < item.warmup + item.iterations; ++iteration) {
        workspaceBytes = 0;
        executor = nullptr;
        Check(runtime.workspace(input, item.m, item.backward, output, &workspaceBytes, &executor),
              "GetWorkspaceSize");
        void* workspace = nullptr;
        if (workspaceBytes != 0) {
            Check(aclrtMalloc(&workspace, workspaceBytes, runtime.allocationPolicy), "workspace allocation");
        }
        const auto started = std::chrono::steady_clock::now();
        Check(runtime.execute(workspace, workspaceBytes, executor, runtime.stream), "aclnnMhcExpand");
        Check(aclrtSynchronizeStream(runtime.stream), "aclrtSynchronizeStream");
        const auto finished = std::chrono::steady_clock::now();
        if (iteration == 0) {
            Check(aclrtMemcpy(hostOutput.data(), outputCount * 2, outputData, outputCount * 2,
                              ACL_MEMCPY_DEVICE_TO_HOST), "read first output");
            std::ofstream first(item.outputPath, std::ios::binary);
            first.write(reinterpret_cast<char*>(hostOutput.data()), outputCount * 2);
            if (!first) throw std::runtime_error("first output write failed");
        }
        if (iteration >= item.warmup) {
            wallUs.push_back(std::chrono::duration<double, std::micro>(finished - started).count());
        }
        if (workspace != nullptr) Check(aclrtFree(workspace), "workspace free");
    }

    Check(aclrtMemcpy(hostOutput.data(), outputCount * 2, outputData, outputCount * 2,
                      ACL_MEMCPY_DEVICE_TO_HOST), "read final output");
    bool guardsOk = item.guardBytes != 0;
    if (item.guardBytes != 0) {
        for (const auto& allocation : std::vector<std::pair<void*, size_t>>{
                 {inputAllocation, inputCount * 2}, {outputAllocation, outputCount * 2}}) {
            for (const size_t offset : std::vector<size_t>{0, item.guardBytes + allocation.second}) {
                std::vector<uint8_t> guard(item.guardBytes);
                Check(aclrtMemcpy(guard.data(), item.guardBytes,
                                  static_cast<uint8_t*>(allocation.first) + offset,
                                  item.guardBytes, ACL_MEMCPY_DEVICE_TO_HOST), "read guard");
                for (const auto byte : guard) guardsOk = guardsOk && byte == 0xa5;
            }
        }
    }
    std::ofstream final(item.outputPath + ".final.bin", std::ios::binary);
    final.write(reinterpret_cast<char*>(hostOutput.data()), outputCount * 2);
    if (!final) throw std::runtime_error("final output write failed");

    std::cout << "{\"case\":\"" << JsonEscape(item.name)
              << "\",\"status\":\"executed\",\"library\":\""
              << JsonEscape(runtime.loadedLibrary) << "\",\"output_elements\":" << outputCount
              << ",\"default_vector_cores\":" << runtime.defaultVectorCores
              << ",\"effective_vector_cores\":" << runtime.effectiveVectorCores
              << ",\"stream_vector_cores\":" << runtime.streamVectorCores
              << ",\"guard_bytes\":" << item.guardBytes
              << ",\"guards_ok\":" << (guardsOk ? "true" : "false")
              << ",\"warmup\":" << item.warmup << ",\"iterations\":" << item.iterations
              << ",\"host_wall_us\":[";
    for (size_t index = 0; index < wallUs.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << wallUs[index];
    }
    std::cout << "]}" << std::endl;

    aclDestroyTensor(input);
    aclDestroyTensor(output);
    aclrtFree(inputAllocation);
    aclrtFree(outputAllocation);
    return guardsOk;
}

int main(int argc, char** argv) {
    std::string library;
    std::string plan;
    std::string resourceMode = "device";
    std::string allocationName = "huge-first";
    int device = 0;
    int vectorCores = 0;
    bool workspaceOnly = false;
    CaseSpec single;
    try {
        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            auto next = [&]() {
                if (++index >= argc) throw std::runtime_error("missing argument after " + argument);
                return std::string(argv[index]);
            };
            if (argument == "--library") library = next();
            else if (argument == "--plan") plan = next();
            else if (argument == "--workspace-only") workspaceOnly = true;
            else if (argument == "--device") device = std::stoi(next());
            else if (argument == "--vector-cores") vectorCores = std::stoi(next());
            else if (argument == "--resource-mode") resourceMode = next();
            else if (argument == "--allocation-policy") allocationName = next();
            else if (argument == "--input") single.inputPath = next();
            else if (argument == "--output") single.outputPath = next();
            else if (argument == "--s") single.s = std::stoll(next());
            else if (argument == "--d") single.d = std::stoll(next());
            else if (argument == "--m") single.m = std::stoll(next());
            else if (argument == "--backward") single.backward = true;
            else if (argument == "--bf16") single.bf16 = true;
            else if (argument == "--flat-backward") single.flatBackward = true;
            else if (argument == "--warmup") single.warmup = std::stoi(next());
            else if (argument == "--iterations") single.iterations = std::stoi(next());
            else if (argument == "--guard-bytes") single.guardBytes = std::stoul(next());
            else if (argument == "--poison-bits") single.poison = static_cast<uint16_t>(std::stoul(next(), nullptr, 0));
            else throw std::runtime_error("unknown argument: " + argument);
        }
        if (library.empty()) throw std::runtime_error("--library is required");
        std::vector<CaseSpec> cases;
        if (!plan.empty()) {
            cases = ReadPlan(plan);
        } else {
            if (single.inputPath.empty() || single.outputPath.empty() || single.s <= 0 || single.d <= 0 ||
                single.m <= 0 || single.warmup < 0 || single.iterations < 1) {
                throw std::runtime_error("invalid single-case arguments");
            }
            cases.push_back(single);
        }
        Runtime runtime(library, device, vectorCores, resourceMode, allocationName);
        bool passed = true;
        for (const auto& item : cases) {
            try {
                passed = RunCase(runtime, item, workspaceOnly) && passed;
            } catch (const std::exception& error) {
                std::cout << "{\"case\":\"" << JsonEscape(item.name)
                          << "\",\"status\":\"Runtime Error\",\"diagnostic\":\""
                          << JsonEscape(error.what()) << "\"}" << std::endl;
                return 1;
            }
        }
        return passed ? 0 : 4;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
