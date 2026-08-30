// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mhc_expand_tiling.h"
#include "tiling_key_mhc_expand.h"

template <class DT_X>
class KernelMhcExpand {
public:
    __aicore__ inline KernelMhcExpand() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR o, uint32_t length) {

    }
    __aicore__ inline void Process() {

    }
private:

};

template <typename DT_X>
 __global__ __aicore__ void mhc_expand(GM_ADDR x, GM_ADDR o, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MhcExpandTilingData);
    GET_TILING_DATA_WITH_STRUCT(MhcExpandTilingData, tiling_data, tiling);
    KernelMhcExpand<DT_X> op;
    op.Init(x, o, tiling_data.length);
    op.Process();
}
