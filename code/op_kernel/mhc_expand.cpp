// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mhc_expand_tiling.h"
#include "tiling_key_mhc_expand.h"

namespace {
constexpr uint32_t MHC_BLOCK_BYTES = 32U;

// 前向按行块把 x 暂存到 UB 后写出 mhc_mult 份副本；反向把 mhc_mult 个 lane 依次
// 读入 UB，在 FP32 上累加后写回。两条路径共用同一套 tile 定位与搬运参数。
template <class DT_X, bool IS_BACKWARD, bool IS_ALIGNED>
class KernelMhcExpand {
public:
    __aicore__ inline KernelMhcExpand() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR o, const MhcExpandTilingData &tiling)
    {
        rowLen_ = tiling.rowLen;
        mult_ = tiling.mult;
        rowCount_ = tiling.rowCount;
        tileRows_ = tiling.tileRows;
        tileCols_ = tiling.tileCols;
        colTiles_ = tiling.colTiles;
        ubPitch_ = tiling.ubPitch;
        stageElems_ = tileRows_ * ubPitch_;
        const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
        tileCount_ = tiling.tilesPerCore + ((block < tiling.tailTiles) ? 1U : 0U);
        tileStart_ = block * tiling.tilesPerCore + ((block < tiling.tailTiles) ? block : tiling.tailTiles);
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        oGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(o));
    }

    __aicore__ inline void Process()
    {
        if (tileCount_ == 0U) {
            return;
        }
        if constexpr (IS_BACKWARD) {
            Reduce();
        } else {
            Expand();
        }
    }

private:
    // 把线性 tile 序号还原成行区间与列区间。
    __aicore__ inline void Locate(uint32_t tile, uint32_t &rows, uint32_t &cols,
                                  uint32_t &rowBase, uint32_t &colBase)
    {
        uint32_t rowTile = tile;
        uint32_t colTile = 0U;
        if (colTiles_ != 1U) {
            rowTile = tile / colTiles_;
            colTile = tile - rowTile * colTiles_;
        }
        rowBase = rowTile * tileRows_;
        rows = tileRows_;
        if (rowBase + rows > rowCount_) {
            rows = rowCount_ - rowBase;
        }
        colBase = colTile * tileCols_;
        cols = tileCols_;
        if (colBase + cols > rowLen_) {
            cols = rowLen_ - colBase;
        }
    }

    // GM -> UB：行内连续，行间按 rowLen_ 跨步。
    __aicore__ inline void LoadTile(const AscendC::LocalTensor<DT_X> &dst, int64_t offset,
                                    uint32_t rows, uint32_t cols, uint32_t gmPitch)
    {
        if constexpr (IS_ALIGNED) {
            if (rows == 1U || gmPitch == cols) {
                AscendC::DataCopy(dst, xGm_[offset], rows * cols);
            } else {
                AscendC::DataCopyParams params(
                    static_cast<uint16_t>(rows),
                    static_cast<uint16_t>(cols * sizeof(DT_X) / MHC_BLOCK_BYTES),
                    static_cast<uint16_t>((gmPitch - cols) * sizeof(DT_X) / MHC_BLOCK_BYTES),
                    0U);
                AscendC::DataCopy(dst, xGm_[offset], params);
            }
        } else {
            AscendC::DataCopyExtParams params(
                static_cast<uint16_t>(rows), cols * sizeof(DT_X),
                (gmPitch - cols) * sizeof(DT_X), 0U, 0U);
            AscendC::DataCopyPadExtParams<DT_X> pad;
            AscendC::DataCopyPad(dst, xGm_[offset], params, pad);
        }
    }

    // UB -> GM：行内连续，行间按 gmPitch 跨步。
    __aicore__ inline void StoreTile(int64_t offset, const AscendC::LocalTensor<DT_X> &src,
                                     uint32_t rows, uint32_t cols, uint32_t gmPitch)
    {
        if constexpr (IS_ALIGNED) {
            if (rows == 1U || gmPitch == cols) {
                AscendC::DataCopy(oGm_[offset], src, rows * cols);
            } else {
                AscendC::DataCopyParams params(
                    static_cast<uint16_t>(rows),
                    static_cast<uint16_t>(cols * sizeof(DT_X) / MHC_BLOCK_BYTES),
                    0U,
                    static_cast<uint16_t>((gmPitch - cols) * sizeof(DT_X) / MHC_BLOCK_BYTES));
                AscendC::DataCopy(oGm_[offset], src, params);
            }
        } else {
            AscendC::DataCopyExtParams params(
                static_cast<uint16_t>(rows), cols * sizeof(DT_X), 0U,
                (gmPitch - cols) * sizeof(DT_X), 0U);
            AscendC::DataCopyPad(oGm_[offset], src, params);
        }
    }

    __aicore__ inline void Expand()
    {
        AscendC::TBuf<AscendC::TPosition::VECCALC> stage;
        pipe_.InitBuffer(stage, stageElems_ * sizeof(DT_X) * 2U);
        AscendC::LocalTensor<DT_X> buffer = stage.Get<DT_X>();
        AscendC::TQueSync<AscendC::PIPE_MTE2, AscendC::PIPE_MTE3> ready;
        AscendC::TQueSync<AscendC::PIPE_MTE3, AscendC::PIPE_MTE2> reuse;

        for (uint32_t index = 0U; index < tileCount_; ++index) {
            const uint32_t slot = index & 1U;
            const AscendC::TEventID event = static_cast<AscendC::TEventID>(slot);
            if (index >= 2U) {
                reuse.WaitFlag(event);
            }
            uint32_t rows = 0U;
            uint32_t cols = 0U;
            uint32_t rowBase = 0U;
            uint32_t colBase = 0U;
            Locate(tileStart_ + index, rows, cols, rowBase, colBase);
            AscendC::LocalTensor<DT_X> tile = buffer[slot * stageElems_];
            const int64_t source = static_cast<int64_t>(rowBase) * rowLen_ + colBase;
            LoadTile(tile, source, rows, cols, rowLen_);
            ready.SetFlag(event);
            ready.WaitFlag(event);
            const int64_t base = static_cast<int64_t>(rowBase) * mult_ * rowLen_ + colBase;
            for (uint32_t lane = 0U; lane < mult_; ++lane) {
                StoreTile(base + static_cast<int64_t>(lane) * rowLen_, tile, rows, cols,
                          mult_ * rowLen_);
            }
            reuse.SetFlag(event);
        }
        Drain(reuse, tileCount_);
    }

    __aicore__ inline void Reduce()
    {
        const uint32_t stageBytes = stageElems_ * sizeof(DT_X);
        AscendC::TBuf<AscendC::TPosition::VECCALC> laneBuf;
        AscendC::TBuf<AscendC::TPosition::VECCALC> accBuf;
        AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
        AscendC::TBuf<AscendC::TPosition::VECCALC> outBuf;
        pipe_.InitBuffer(laneBuf, stageBytes * 2U);
        pipe_.InitBuffer(accBuf, stageElems_ * sizeof(float));
        pipe_.InitBuffer(tmpBuf, stageElems_ * sizeof(float));
        pipe_.InitBuffer(outBuf, stageBytes * 2U);
        AscendC::LocalTensor<DT_X> lanes = laneBuf.Get<DT_X>();
        AscendC::LocalTensor<float> acc = accBuf.Get<float>();
        AscendC::LocalTensor<float> tmp = tmpBuf.Get<float>();
        AscendC::LocalTensor<DT_X> outs = outBuf.Get<DT_X>();

        AscendC::TQueSync<AscendC::PIPE_MTE2, AscendC::PIPE_V> loaded;
        AscendC::TQueSync<AscendC::PIPE_V, AscendC::PIPE_MTE2> laneFree;
        AscendC::TQueSync<AscendC::PIPE_V, AscendC::PIPE_MTE3> reduced;
        AscendC::TQueSync<AscendC::PIPE_MTE3, AscendC::PIPE_V> outFree;

        for (uint32_t index = 0U; index < tileCount_; ++index) {
            uint32_t rows = 0U;
            uint32_t cols = 0U;
            uint32_t rowBase = 0U;
            uint32_t colBase = 0U;
            Locate(tileStart_ + index, rows, cols, rowBase, colBase);
            const uint32_t elems = (rows > 1U) ? (rows * ubPitch_) : cols;
            const int64_t base = static_cast<int64_t>(rowBase) * mult_ * rowLen_ + colBase;
            for (uint32_t lane = 0U; lane < mult_; ++lane) {
                const uint32_t slot = lane & 1U;
                const AscendC::TEventID event = static_cast<AscendC::TEventID>(slot);
                if (lane >= 2U) {
                    laneFree.WaitFlag(event);
                }
                AscendC::LocalTensor<DT_X> tile = lanes[slot * stageElems_];
                LoadTile(tile, base + static_cast<int64_t>(lane) * rowLen_, rows, cols,
                         mult_ * rowLen_);
                loaded.SetFlag(event);
                loaded.WaitFlag(event);
                if (lane == 0U) {
                    AscendC::Cast(acc, tile, AscendC::RoundMode::CAST_NONE, elems);
                } else {
                    AscendC::Cast(tmp, tile, AscendC::RoundMode::CAST_NONE, elems);
                    AscendC::Add(acc, acc, tmp, elems);
                }
                laneFree.SetFlag(event);
            }
            Drain(laneFree, mult_);
            const uint32_t slot = index & 1U;
            const AscendC::TEventID event = static_cast<AscendC::TEventID>(slot);
            if (index >= 2U) {
                outFree.WaitFlag(event);
            }
            AscendC::LocalTensor<DT_X> out = outs[slot * stageElems_];
            AscendC::Cast(out, acc, AscendC::RoundMode::CAST_RINT, elems);
            reduced.SetFlag(event);
            reduced.WaitFlag(event);
            StoreTile(static_cast<int64_t>(rowBase) * rowLen_ + colBase, out, rows, cols, rowLen_);
            outFree.SetFlag(event);
        }
        Drain(outFree, tileCount_);
    }

    // 收尾时补齐尚未配对的 WaitFlag，保证同步标志位归零。
    template <class SYNC>
    __aicore__ inline void Drain(SYNC &sync, uint32_t issued)
    {
        const uint32_t pending = (issued < 2U) ? issued : 2U;
        for (uint32_t index = 0U; index < pending; ++index) {
            sync.WaitFlag(static_cast<AscendC::TEventID>((issued - 1U - index) & 1U));
        }
    }

    AscendC::TPipe pipe_;
    AscendC::GlobalTensor<DT_X> xGm_;
    AscendC::GlobalTensor<DT_X> oGm_;
    uint32_t rowLen_ = 0U;
    uint32_t mult_ = 0U;
    uint32_t rowCount_ = 0U;
    uint32_t tileRows_ = 0U;
    uint32_t tileCols_ = 0U;
    uint32_t colTiles_ = 0U;
    uint32_t ubPitch_ = 0U;
    uint32_t stageElems_ = 0U;
    uint32_t tileCount_ = 0U;
    uint32_t tileStart_ = 0U;
};
}  // namespace

template <typename DT_X, int IS_BACKWARD, int IS_ALIGNED>
 __global__ __aicore__ void mhc_expand(GM_ADDR x, GM_ADDR o, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MhcExpandTilingData);
    GET_TILING_DATA_WITH_STRUCT(MhcExpandTilingData, tiling_data, tiling);
    KernelMhcExpand<DT_X, IS_BACKWARD != 0, IS_ALIGNED != 0> op;
    op.Init(x, o, tiling_data);
    op.Process();
}
