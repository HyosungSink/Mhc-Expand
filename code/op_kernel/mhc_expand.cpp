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
        laneGroup_ = tiling.laneGroup;
        slots_ = tiling.slots;
        stageElems_ = tileRows_ * ubPitch_;
        const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
        unitStart_ = block * tiling.unitsPerCore + ((block < tiling.tailUnits) ? block : tiling.tailUnits);
        unitEnd_ = unitStart_ + tiling.unitsPerCore + ((block < tiling.tailUnits) ? 1U : 0U);
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        oGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(o));
    }

    __aicore__ inline void Process()
    {
        if (unitEnd_ == unitStart_) {
            return;
        }
        if constexpr (IS_BACKWARD) {
            if (laneGroup_ > 1U) {
                ReduceGrouped();
            } else {
                Reduce();
            }
        } else if (laneGroup_ > 1U) {
            ExpandReplicated();
        } else {
            Expand();
        }
    }

private:
    // 把当前工作单元展开成行区间与列区间，并返回本次消耗的单元数。
    __aicore__ inline uint32_t Locate(uint32_t unit, uint32_t &rows, uint32_t &cols,
                                      uint32_t &rowBase, uint32_t &colBase)
    {
        if (colTiles_ == 1U) {
            rowBase = unit;
            colBase = 0U;
            cols = rowLen_;
            rows = unitEnd_ - unit;
            if (rows > tileRows_) {
                rows = tileRows_;
            }
            return rows;
        }
        rowBase = unit / colTiles_;
        const uint32_t colTile = unit - rowBase * colTiles_;
        rows = 1U;
        colBase = colTile * tileCols_;
        cols = tileCols_;
        if (colBase + cols > rowLen_) {
            cols = rowLen_ - colBase;
        }
        return 1U;
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

    // GM -> UB：一次读入 rows*mult_ 个数据块，块内连续，块间按 rowLen_ 跨步。
    __aicore__ inline void LoadLanes(const AscendC::LocalTensor<DT_X> &dst, int64_t offset,
                                     uint32_t rows, uint32_t cols)
    {
        const uint32_t blocks = rows * mult_;
        if constexpr (IS_ALIGNED) {
            if (cols == rowLen_) {
                AscendC::DataCopy(dst, xGm_[offset], blocks * cols);
            } else {
                const AscendC::DataCopyParams params(
                    static_cast<uint16_t>(blocks),
                    static_cast<uint16_t>(cols * sizeof(DT_X) / MHC_BLOCK_BYTES),
                    static_cast<uint16_t>((rowLen_ - cols) * sizeof(DT_X) / MHC_BLOCK_BYTES), 0U);
                AscendC::DataCopy(dst, xGm_[offset], params);
            }
        } else {
            const AscendC::DataCopyExtParams params(
                static_cast<uint16_t>(blocks), cols * sizeof(DT_X),
                (rowLen_ - cols) * sizeof(DT_X), 0U, 0U);
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
        AscendC::TQueSync<PIPE_MTE2, PIPE_MTE3> ready;
        AscendC::TQueSync<PIPE_MTE3, PIPE_MTE2> reuse;

        uint32_t index = 0U;
        for (uint32_t unit = unitStart_; unit < unitEnd_; ++index) {
            const uint32_t slot = index & 1U;
            const AscendC::TEventID event = static_cast<AscendC::TEventID>(slot);
            if (index >= 2U) {
                reuse.WaitFlag(event);
            }
            uint32_t rows = 0U;
            uint32_t cols = 0U;
            uint32_t rowBase = 0U;
            uint32_t colBase = 0U;
            unit += Locate(unit, rows, cols, rowBase, colBase);
            AscendC::LocalTensor<DT_X> tile = buffer[slot * stageElems_];
            const int64_t source = static_cast<int64_t>(rowBase) * rowLen_ + colBase;
            LoadTile(tile, source, rows, cols, rowLen_);
            ready.SetFlag(event);
            ready.WaitFlag(event);
            const int64_t base = static_cast<int64_t>(rowBase) * mult_ * rowLen_ + colBase;
            const uint32_t gap = mult_ * rowLen_ - cols;
            if constexpr (IS_ALIGNED) {
                if (rows == 1U) {
                    for (uint32_t lane = 0U; lane < mult_; ++lane) {
                        AscendC::DataCopy(oGm_[base + static_cast<int64_t>(lane) * rowLen_], tile,
                                          cols);
                    }
                } else {
                    const AscendC::DataCopyParams emit(
                        static_cast<uint16_t>(rows),
                        static_cast<uint16_t>(cols * sizeof(DT_X) / MHC_BLOCK_BYTES), 0U,
                        static_cast<uint16_t>(gap * sizeof(DT_X) / MHC_BLOCK_BYTES));
                    for (uint32_t lane = 0U; lane < mult_; ++lane) {
                        AscendC::DataCopy(oGm_[base + static_cast<int64_t>(lane) * rowLen_], tile,
                                          emit);
                    }
                }
            } else {
                const AscendC::DataCopyExtParams emit(
                    static_cast<uint16_t>(rows), cols * sizeof(DT_X), 0U, gap * sizeof(DT_X), 0U);
                for (uint32_t lane = 0U; lane < mult_; ++lane) {
                    AscendC::DataCopyPad(oGm_[base + static_cast<int64_t>(lane) * rowLen_], tile,
                                         emit);
                }
            }
            reuse.SetFlag(event);
        }
        Drain(reuse, index);
    }

    // 直接把 x 读进输出 tile 的第 0 份副本，再在 UB 内铺开其余副本，
    // MTE3 就能按 laneGroup*D 的连续块写回，省掉一块暂存和两次同步。
    __aicore__ inline void ExpandReplicated()
    {
        const uint32_t outElems = stageElems_ * laneGroup_;
        AscendC::TBuf<AscendC::TPosition::VECCALC> outStage;
        pipe_.InitBuffer(outStage, outElems * sizeof(DT_X) * slots_);
        AscendC::LocalTensor<DT_X> outAll = outStage.Get<DT_X>();
        AscendC::TQueSync<PIPE_MTE2, PIPE_V> loaded;
        AscendC::TQueSync<PIPE_V, PIPE_MTE3> filled;
        AscendC::TQueSync<PIPE_MTE3, PIPE_MTE2> recycle;

        const uint32_t laneBlocks = rowLen_ * sizeof(DT_X) / MHC_BLOCK_BYTES;
        const uint32_t groupBlocks = laneBlocks * laneGroup_;
        const uint32_t laneGap = groupBlocks - laneBlocks;
        const uint32_t groups = mult_ / laneGroup_;
        const uint32_t groupStride = laneGroup_ * rowLen_;
        const uint32_t rowStride = mult_ * rowLen_;
        // 整块 tile 的搬运参数在循环外构造，尾块只改 blockCount。
        AscendC::DataCopyParams gather(static_cast<uint16_t>(tileRows_),
                                       static_cast<uint16_t>(laneBlocks), 0U,
                                       static_cast<uint16_t>(laneGap));
        AscendC::DataCopyParams spread(static_cast<uint16_t>(tileRows_),
                                       static_cast<uint16_t>(laneBlocks),
                                       static_cast<uint16_t>(laneGap),
                                       static_cast<uint16_t>(laneGap));
        AscendC::DataCopyParams emit(static_cast<uint16_t>(tileRows_),
                                     static_cast<uint16_t>(groupBlocks), 0U,
                                     static_cast<uint16_t>(laneBlocks * (mult_ - laneGroup_)));
        uint32_t index = 0U;
        uint32_t slot = 0U;
        for (uint32_t unit = unitStart_; unit < unitEnd_; ++index) {
            const AscendC::TEventID event = static_cast<AscendC::TEventID>(slot);
            if (index >= slots_) {
                recycle.WaitFlag(event);
            }
            const uint32_t rowBase = unit;
            uint32_t rows = unitEnd_ - unit;
            if (rows > tileRows_) {
                rows = tileRows_;
            } else {
                gather.blockCount = static_cast<uint16_t>(rows);
                spread.blockCount = static_cast<uint16_t>(rows);
                emit.blockCount = static_cast<uint16_t>(rows);
            }
            unit += rows;
            AscendC::LocalTensor<DT_X> out = outAll[slot * outElems];
            AscendC::DataCopy(out, xGm_[static_cast<int64_t>(rowBase) * rowLen_], gather);
            loaded.SetFlag(event);
            loaded.WaitFlag(event);
            for (uint32_t lane = 1U; lane < laneGroup_; ++lane) {
                AscendC::DataCopy(out[lane * rowLen_], out, spread);
            }
            filled.SetFlag(event);
            filled.WaitFlag(event);
            const int64_t base = static_cast<int64_t>(rowBase) * rowStride;
            if (groups == 1U) {
                AscendC::DataCopy(oGm_[base], out, rows * rowStride);
            } else {
                for (uint32_t group = 0U; group < groups; ++group) {
                    AscendC::DataCopy(oGm_[base + static_cast<int64_t>(group) * groupStride], out,
                                      emit);
                }
            }
            recycle.SetFlag(event);
            ++slot;
            if (slot == slots_) {
                slot = 0U;
            }
        }
        DrainRing(recycle, index, slot);
    }

    // 一次把 laneGroup 份梯度连续读进 UB，再在 FP32 上逐 lane 累加，
    // 把原来跨 lane 的跳读并成一次连续搬运。
    __aicore__ inline void ReduceGrouped()
    {
        constexpr uint32_t elemsPerBlock = MHC_BLOCK_BYTES / sizeof(DT_X);
        const uint32_t groupElems = stageElems_ * laneGroup_;
        AscendC::TBuf<AscendC::TPosition::VECCALC> inStage;
        AscendC::TBuf<AscendC::TPosition::VECCALC> accBuf;
        AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
        AscendC::TBuf<AscendC::TPosition::VECCALC> outStage;
        pipe_.InitBuffer(inStage, groupElems * sizeof(DT_X) * 2U);
        pipe_.InitBuffer(accBuf, stageElems_ * sizeof(float));
        pipe_.InitBuffer(tmpBuf, ubPitch_ * sizeof(float));
        pipe_.InitBuffer(outStage, stageElems_ * sizeof(DT_X) * 2U);
        AscendC::LocalTensor<DT_X> inAll = inStage.Get<DT_X>();
        AscendC::LocalTensor<float> acc = accBuf.Get<float>();
        AscendC::LocalTensor<float> tmp = tmpBuf.Get<float>();
        AscendC::LocalTensor<DT_X> outAll = outStage.Get<DT_X>();

        AscendC::TQueSync<PIPE_MTE2, PIPE_V> loaded;
        AscendC::TQueSync<PIPE_V, PIPE_MTE2> inFree;
        AscendC::TQueSync<PIPE_V, PIPE_MTE3> reduced;
        AscendC::TQueSync<PIPE_MTE3, PIPE_V> outFree;

        uint32_t index = 0U;
        for (uint32_t unit = unitStart_; unit < unitEnd_; ++index) {
            const uint32_t slot = index & 1U;
            const AscendC::TEventID event = static_cast<AscendC::TEventID>(slot);
            if (index >= 2U) {
                inFree.WaitFlag(event);
            }
            uint32_t rows = 0U;
            uint32_t cols = 0U;
            uint32_t rowBase = 0U;
            uint32_t colBase = 0U;
            unit += Locate(unit, rows, cols, rowBase, colBase);
            AscendC::LocalTensor<DT_X> in = inAll[slot * groupElems];
            LoadLanes(in, static_cast<int64_t>(rowBase) * mult_ * rowLen_ + colBase, rows, cols);
            loaded.SetFlag(event);
            loaded.WaitFlag(event);
            // DataCopy/DataCopyPad 把每个数据块按 32B 对齐落在 UB 上，lane 间距按此计算。
            const uint32_t lanePitch =
                IS_ALIGNED ? cols : ((cols + elemsPerBlock - 1U) / elemsPerBlock) * elemsPerBlock;
            for (uint32_t row = 0U; row < rows; ++row) {
                const uint32_t lanes = row * mult_ * lanePitch;
                const uint32_t sink = row * cols;
                AscendC::Cast(acc[sink], in[lanes], AscendC::RoundMode::CAST_NONE, cols);
                for (uint32_t lane = 1U; lane < mult_; ++lane) {
                    AscendC::Cast(tmp, in[lanes + lane * lanePitch], AscendC::RoundMode::CAST_NONE,
                                  cols);
                    AscendC::Add(acc[sink], acc[sink], tmp, cols);
                }
            }
            inFree.SetFlag(event);
            if (index >= 2U) {
                outFree.WaitFlag(event);
            }
            AscendC::LocalTensor<DT_X> out = outAll[slot * stageElems_];
            AscendC::Cast(out, acc, AscendC::RoundMode::CAST_RINT, rows * cols);
            reduced.SetFlag(event);
            reduced.WaitFlag(event);
            StoreTile(static_cast<int64_t>(rowBase) * rowLen_ + colBase, out, rows, cols, rowLen_);
            outFree.SetFlag(event);
        }
        Drain(inFree, index);
        Drain(outFree, index);
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

        AscendC::TQueSync<PIPE_MTE2, PIPE_V> loaded;
        AscendC::TQueSync<PIPE_V, PIPE_MTE2> laneFree;
        AscendC::TQueSync<PIPE_V, PIPE_MTE3> reduced;
        AscendC::TQueSync<PIPE_MTE3, PIPE_V> outFree;

        // lane 缓冲的槽位跨 tile 连续轮转，避免每个 tile 结束时把 MTE2 拦在 V 后面。
        uint32_t laneSeq = 0U;
        uint32_t index = 0U;
        for (uint32_t unit = unitStart_; unit < unitEnd_; ++index) {
            uint32_t rows = 0U;
            uint32_t cols = 0U;
            uint32_t rowBase = 0U;
            uint32_t colBase = 0U;
            unit += Locate(unit, rows, cols, rowBase, colBase);
            const uint32_t elems = (rows > 1U) ? (rows * ubPitch_) : cols;
            const int64_t base = static_cast<int64_t>(rowBase) * mult_ * rowLen_ + colBase;
            for (uint32_t lane = 0U; lane < mult_; ++lane, ++laneSeq) {
                const uint32_t slot = laneSeq & 1U;
                const AscendC::TEventID event = static_cast<AscendC::TEventID>(slot);
                if (laneSeq >= 2U) {
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
        Drain(laneFree, laneSeq);
        Drain(outFree, index);
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

    // 副本路径按 slots_ 轮转，收尾时逐个归零尚未配对的标志位。
    template <class SYNC>
    __aicore__ inline void DrainRing(SYNC &sync, uint32_t issued, uint32_t slot)
    {
        const uint32_t pending = (issued < slots_) ? issued : slots_;
        for (uint32_t index = 0U; index < pending; ++index) {
            slot = (slot == 0U) ? (slots_ - 1U) : (slot - 1U);
            sync.WaitFlag(static_cast<AscendC::TEventID>(slot));
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
    uint32_t laneGroup_ = 1U;
    uint32_t slots_ = 2U;
    uint32_t stageElems_ = 0U;
    uint32_t unitStart_ = 0U;
    uint32_t unitEnd_ = 0U;
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
