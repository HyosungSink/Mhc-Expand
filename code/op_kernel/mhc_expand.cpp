// Kernel side of mHC Expand: forward replicates x[S, D] into o[S, m, D] and
// backward reduces o_grad[S, m, D] into x_grad[S, D].
//
// Both directions work on one row of one column tile at a time. A block owns a
// contiguous range of output rows derived from its block index, so every block
// stores a disjoint region of the output and no workspace, cross block
// accumulation or second pass is needed.
//
// The Host splits a row into column tiles that are whole 32B blocks, except for
// a row that is not block aligned at all; that row becomes a single tile that
// the element wise transfer path moves on its own.
#include "kernel_operator.h"

#include "mhc_expand_tiling.h"
#include "tiling_key_mhc_expand.h"

using namespace AscendC;

namespace {
// Element width of an operand. Both supported element types occupy two bytes,
// and every byte count of every transfer is derived from this constant.
constexpr uint32_t MHC_ELEM_BYTES = 2;

// Elements of one 32B block.
constexpr uint32_t MHC_BLOCK_ELEMS = 32 / MHC_ELEM_BYTES;


}  // namespace

template <typename DT_X, int BACKWARD>
class KernelMhcExpand {
    using DT_F = float;

public:
    __aicore__ inline KernelMhcExpand() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR o, const MhcExpandTilingData &info, TPipe &pipe) {
        pipe_ = &pipe;
        info_ = info;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        oGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(o));
        // A vector store covers whole blocks, so a scratch tile is sized for the
        // block aligned length of a tile, never for its payload alone.
        const uint32_t tileBytes = BlockAligned(info.tileLen) * MHC_ELEM_BYTES;
        if constexpr (BACKWARD == 0) {
            if (PackedForward()) {
                pipe_->InitBuffer(inQue_, 2, AlignUp(tileBytes));
                pipe_->InitBuffer(outQue_, 2, info.rowLen * info.lanes * MHC_ELEM_BYTES);
            } else if (info.batched != 0) {
                pipe_->InitBuffer(copyQue_, 3, 32768);
            } else {
                pipe_->InitBuffer(inQue_, 3, AlignUp(tileBytes));
                pipe_->InitBuffer(outQue_, 3, AlignUp(tileBytes));
            }
        } else {
            pipe_->InitBuffer(inQue_, 3, AlignUp(tileBytes));
            pipe_->InitBuffer(outQue_, 3, AlignUp(tileBytes));
            pipe_->InitBuffer(laneQue_, 4, AlignUp(tileBytes));
            pipe_->InitBuffer(wideBuf_, AlignUp(info.tileLen * static_cast<uint32_t>(sizeof(DT_F))));
            pipe_->InitBuffer(accBuf_, AlignUp(info.tileLen * static_cast<uint32_t>(sizeof(DT_F))));
            pipe_->InitBuffer(resBuf_, AlignUp(tileBytes));
        }
    }

    __aicore__ inline void Process() {
        if constexpr (BACKWARD != 0) {
            ReduceRows();
        } else {
            if (PackedForward()) {
                ExpandPackedRows();
            } else if (info_.batched != 0) {
                ExpandRows();
            } else {
                ExpandRowsLegacy();
            }
        }
    }

private:
    static __aicore__ inline uint32_t AlignUp(uint32_t bytes) {
        return (bytes + 31u) / 32u * 32u;
    }

    static __aicore__ inline uint32_t MinU32(uint32_t left, uint32_t right) {
        return left < right ? left : right;
    }

    // Elements of the whole blocks that cover `len` elements.
    static __aicore__ inline uint32_t BlockAligned(uint32_t len) {
        return (len + MHC_BLOCK_ELEMS - 1) / MHC_BLOCK_ELEMS * MHC_BLOCK_ELEMS;
    }

    // Rows are split over the blocks by the Host, so the first row of a block
    // follows from its block index alone.
    __aicore__ inline uint32_t RowBegin() const {
        return static_cast<uint32_t>(GetBlockIdx()) * info_.rowTile;
    }

    __aicore__ inline void LoadRow(uint32_t begin, uint32_t len) {
        LocalTensor<DT_X> tile = inQue_.AllocTensor<DT_X>();
        DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = len * MHC_ELEM_BYTES;
        params.srcStride = 0;
        params.dstStride = 0;
        params.rsv = 0;
        DataCopyPadExtParams<DT_X> pad;
        pad.isPad = false;
        pad.leftPadding = 0;
        pad.rightPadding = 0;
        pad.paddingValue = 0;
        DataCopyPad(tile, xGm_[begin], params, pad);
        inQue_.EnQue(tile);
    }

    // Queues one lane tile of the expanded stream. The queue holds it until the
    // reduction folds it, so the transfer of the next lane can already be in
    // flight while the vector engine works on the current one.
    __aicore__ inline void LoadOneLane(uint32_t begin, uint32_t len) {
        LocalTensor<DT_X> tile = laneQue_.AllocTensor<DT_X>();
        DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = len * MHC_ELEM_BYTES;
        params.srcStride = 0;
        params.dstStride = 0;
        params.rsv = 0;
        DataCopyPadExtParams<DT_X> pad;
        pad.isPad = false;
        pad.leftPadding = 0;
        pad.rightPadding = 0;
        pad.paddingValue = 0;
        DataCopyPad(tile, xGm_[begin], params, pad);
        laneQue_.EnQue(tile);
    }

    // A payload that fills whole blocks is written block by block, which moves
    // exactly the elements of the row. A narrow row that does not fill a block
    // is written by the element wise path, which moves exactly the payload and
    // never the whole block a transfer would otherwise cover.
    __aicore__ inline void StoreTile(uint32_t begin, LocalTensor<DT_X> tile, uint32_t len) {
        if (BlockAligned(len) == len) {
            DataCopy(oGm_[begin], tile, len);
            return;
        }
        DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = len * MHC_ELEM_BYTES;
        params.srcStride = 0;
        params.dstStride = 0;
        params.rsv = 0;
        DataCopyPad(oGm_[begin], tile, params);
    }

    __aicore__ inline void StoreRow(uint32_t begin, uint32_t len) {
        LocalTensor<DT_X> tile = outQue_.DeQue<DT_X>();
        StoreTile(begin, tile, len);
        outQue_.FreeTensor(tile);
    }

    __aicore__ inline bool PackedForward() const {
        return info_.batched != 0 && info_.colTiles == 1 && info_.lanes > 1 &&
            info_.rowLen % MHC_BLOCK_ELEMS == 0 &&
            static_cast<uint64_t>(info_.rowLen) * info_.lanes <= 32768;
    }

    __aicore__ inline void ExpandPackedRows() {
        const uint32_t rowEnd = MinU32(RowBegin() + info_.rowTile, info_.rowTotal);
        for (uint32_t row = RowBegin(); row < rowEnd; ++row) {
            LoadRow(row * info_.rowLen, info_.rowLen);
            LocalTensor<DT_X> input = inQue_.DeQue<DT_X>();
            LocalTensor<DT_X> output = outQue_.AllocTensor<DT_X>();
            for (uint32_t lane = 0; lane < info_.lanes; ++lane) {
                DataCopy(output[lane * info_.rowLen], input, info_.rowLen);
            }
            outQue_.EnQue(output);
            inQue_.FreeTensor(input);
            StoreRow(row * info_.lanes * info_.rowLen, info_.lanes * info_.rowLen);
        }
    }

    __aicore__ inline void ExpandRowsLegacy() {
        const uint32_t rowEnd = MinU32(RowBegin() + info_.rowTile, info_.rowTotal);
        for (uint32_t row = RowBegin(); row < rowEnd; ++row) {
            const uint32_t srcRow = row / info_.lanes;
            const uint32_t lane = row - srcRow * info_.lanes;
            for (uint32_t col = 0; col < info_.colTiles; ++col) {
                const uint32_t offset = col * info_.tileLen;
                const uint32_t len = MinU32(info_.tileLen, info_.rowLen - offset);
                LoadRow(srcRow * info_.rowLen + offset, len);
                LocalTensor<DT_X> tile = inQue_.DeQue<DT_X>();
                LocalTensor<DT_X> copy = outQue_.AllocTensor<DT_X>();
                DataCopy(copy, tile, BlockAligned(len));
                outQue_.EnQue(copy);
                inQue_.FreeTensor(tile);
                StoreRow((srcRow * info_.lanes + lane) * info_.rowLen + offset, len);
            }
        }
    }

    // A batch remains in UB until every expanded lane has consumed it.
    __aicore__ inline void ExpandRows() {
        const uint32_t rowEnd = MinU32(RowBegin() + info_.rowTile, info_.rowTotal);
        uint32_t batchRows = 16384 / BlockAligned(info_.tileLen);
        const uint64_t outputStride = static_cast<uint64_t>(info_.lanes) * info_.rowLen;
        if (outputStride * MHC_ELEM_BYTES > 0xFFFFFFFFULL) {
            batchRows = 1;
        }
        for (uint32_t row = RowBegin(); row < rowEnd; row += batchRows) {
            const uint32_t count = MinU32(batchRows, rowEnd - row);
            for (uint32_t col = 0; col < info_.colTiles; ++col) {
                const uint32_t offset = col * info_.tileLen;
                const uint32_t len = MinU32(info_.tileLen, info_.rowLen - offset);
                LocalTensor<DT_X> tile = copyQue_.AllocTensor<DT_X>();
                const uint64_t inputOffset = static_cast<uint64_t>(row) * info_.rowLen + offset;
                if (len == info_.rowLen && BlockAligned(len) == len) {
                    DataCopy(tile, xGm_[inputOffset], count * len);
                } else {
                    DataCopyExtParams inputParams{static_cast<uint16_t>(count),
                        len * MHC_ELEM_BYTES, (info_.rowLen - len) * MHC_ELEM_BYTES, 0, 0};
                    DataCopyPadExtParams<DT_X> padding{false, 0, 0, 0};
                    DataCopyPad(tile, xGm_[inputOffset], inputParams, padding);
                }
                copyQue_.EnQue(tile);
                tile = copyQue_.DeQue<DT_X>();
                const uint32_t gap = count == 1 ? 0 :
                    static_cast<uint32_t>((outputStride - len) * MHC_ELEM_BYTES);
                DataCopyExtParams outputParams{static_cast<uint16_t>(count),
                    len * MHC_ELEM_BYTES, 0, gap, 0};
                for (uint32_t lane = 0; lane < info_.lanes; ++lane) {
                    const uint64_t outputOffset =
                        (static_cast<uint64_t>(row) * info_.lanes + lane) * info_.rowLen + offset;
                    DataCopyPad(oGm_[outputOffset], tile, outputParams);
                }
                copyQue_.FreeTensor(tile);
            }
        }
    }

    // The lanes of one row tile are folded one after another into a fp32 running
    // total. Each lane travels through the queue, so its own transfer is retired
    // before the fold reads it and the next lane can already be streaming while
    // the vector engine works on the current one.
    __aicore__ inline void ReduceRows() {
        const uint32_t rowEnd = MinU32(RowBegin() + info_.rowTile, info_.rowTotal);
        for (uint32_t row = RowBegin(); row < rowEnd; ++row) {
            for (uint32_t col = 0; col < info_.colTiles; ++col) {
                const uint32_t offset = col * info_.tileLen;
                const uint32_t len = MinU32(info_.tileLen, info_.rowLen - offset);
                const uint32_t begin = row * info_.lanes * info_.rowLen + offset;
                LocalTensor<DT_F> total = accBuf_.Get<DT_F>();
                LocalTensor<DT_F> wide = wideBuf_.Get<DT_F>();
                Duplicate(total, static_cast<DT_F>(0), info_.tileLen);
                for (uint32_t lane = 0; lane < info_.lanes; ++lane) {
                    LoadOneLane(begin + lane * info_.rowLen, len);
                    LocalTensor<DT_X> raw = laneQue_.DeQue<DT_X>();
                    Cast(wide, raw, RoundMode::CAST_NONE, len);
                    Add(total, total, wide, len);
                    laneQue_.FreeTensor(raw);
                }
                LocalTensor<DT_X> result = outQue_.AllocTensor<DT_X>();
                Cast(result, total, RoundMode::CAST_RINT, len);
                outQue_.EnQue(result);
                StoreRow(row * info_.rowLen + offset, len);
            }
        }
    }

    MhcExpandTilingData info_;
    TPipe *pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 3> copyQue_;
    TQue<QuePosition::VECIN, 3> inQue_;
    TQue<QuePosition::VECOUT, 3> outQue_;
    TQue<QuePosition::VECIN, 4> laneQue_;
    TBuf<QuePosition::VECCALC> wideBuf_;
    TBuf<QuePosition::VECCALC> accBuf_;
    TBuf<QuePosition::VECCALC> resBuf_;
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> oGm_;
};

template <typename DT_X, uint32_t FIXED_ROWS = 0, uint32_t FIXED_COLS = 0>
__aicore__ inline void MhcExpandSmallForward(GM_ADDR x, GM_ADDR o, const MhcExpandTilingData &info) {
    InitSocState();
    const uint32_t rowTile = FIXED_ROWS != 0 ? FIXED_ROWS : info.rowTile;
    const uint32_t rowLen = FIXED_COLS != 0 ? FIXED_COLS : info.rowLen;
    const uint32_t lanes = FIXED_ROWS != 0 ? 2 : info.lanes;
    const uint32_t begin = static_cast<uint32_t>(GetBlockIdx()) * rowTile;
    const uint32_t remain = info.rowTotal - begin;
    const uint32_t rows = FIXED_ROWS != 0 ? FIXED_ROWS : (remain < rowTile ? remain : rowTile);
    GlobalTensor<DT_X> input;
    GlobalTensor<DT_X> output;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(o));
    LocalTensor<DT_X> tile(TPosition::VECIN, 0, rowTile * rowLen);
    DataCopy(tile, input[static_cast<uint64_t>(begin) * rowLen], rows * rowLen);
    if constexpr (FIXED_ROWS != 0) {
        LocalTensor<DT_X> packed(TPosition::VECOUT, FIXED_ROWS * FIXED_COLS * MHC_ELEM_BYTES,
                                 FIXED_ROWS * FIXED_COLS * 2);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        DataCopyParams replicate{static_cast<uint16_t>(FIXED_ROWS),
            static_cast<uint16_t>(FIXED_COLS / MHC_BLOCK_ELEMS), 0,
            static_cast<uint16_t>(FIXED_COLS / MHC_BLOCK_ELEMS)};
        DataCopy(packed, tile, replicate);
        DataCopy(packed[FIXED_COLS], tile, replicate);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(output[static_cast<uint64_t>(begin) * FIXED_COLS * 2],
                 packed, FIXED_ROWS * FIXED_COLS * 2);
        return;
    }
    constexpr int32_t ready = EVENT_ID0;
    SetFlag<HardEvent::MTE2_MTE3>(ready);
    WaitFlag<HardEvent::MTE2_MTE3>(ready);
    DataCopyExtParams params{static_cast<uint16_t>(rows), rowLen * MHC_ELEM_BYTES,
        0, (lanes - 1) * rowLen * MHC_ELEM_BYTES, 0};
    for (uint32_t lane = 0; lane < lanes; ++lane) {
        DataCopyPad(output[(static_cast<uint64_t>(begin) * lanes + lane) * rowLen],
                    tile, params);
    }
}

template <typename DT_X, uint32_t FIXED_ROWS = 0, uint32_t FIXED_COLS = 0>
__aicore__ inline void MhcExpandSmallReduce(GM_ADDR x, GM_ADDR o, const MhcExpandTilingData &info) {
    InitSocState();
    const uint32_t rowTile = FIXED_ROWS != 0 ? FIXED_ROWS : info.rowTile;
    const uint32_t rowLen = FIXED_COLS != 0 ? FIXED_COLS : info.rowLen;
    const uint32_t lanes = FIXED_ROWS != 0 ? 2 : info.lanes;
    const uint32_t begin = static_cast<uint32_t>(GetBlockIdx()) * rowTile;
    const uint32_t remain = info.rowTotal - begin;
    const uint32_t rows = FIXED_ROWS != 0 ? FIXED_ROWS : (remain < rowTile ? remain : rowTile);
    const uint32_t capacity = rowTile * rowLen;
    const uint32_t count = rows * rowLen;
    GlobalTensor<DT_X> input;
    GlobalTensor<DT_X> output;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(o));
    LocalTensor<DT_X> raw(TPosition::VECIN, 0, capacity * 2);
    LocalTensor<float> wide(TPosition::VECCALC, capacity * 4, capacity * 2);
    LocalTensor<DT_X> result(TPosition::VECOUT, capacity * 12, capacity);
    DataCopyParams params{static_cast<uint16_t>(rows),
        static_cast<uint16_t>(rowLen / MHC_BLOCK_ELEMS),
        static_cast<uint16_t>(rowLen / MHC_BLOCK_ELEMS), 0};
    const uint64_t inputOffset = static_cast<uint64_t>(begin) * 2 * rowLen;
    DataCopy(raw, input[inputOffset], params);
    DataCopy(raw[capacity], input[inputOffset + rowLen], params);
    constexpr int32_t loaded = EVENT_ID0;
    SetFlag<HardEvent::MTE2_V>(loaded);
    WaitFlag<HardEvent::MTE2_V>(loaded);
    if constexpr (FIXED_ROWS != 0 && std::is_same<DT_X, half>::value) {
        Add(result, raw, raw[capacity], count);
    } else {
        Cast(wide, raw, RoundMode::CAST_NONE, count);
        Cast(wide[capacity], raw[capacity], RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        Add(wide, wide, wide[capacity], count);
        PipeBarrier<PIPE_V>();
        Cast(result, wide, RoundMode::CAST_RINT, count);
    }
    constexpr int32_t reduced = EVENT_ID0;
    SetFlag<HardEvent::V_MTE3>(reduced);
    WaitFlag<HardEvent::V_MTE3>(reduced);
    DataCopy(output[static_cast<uint64_t>(begin) * rowLen], result, count);
}

template <typename DT_X, int BACKWARD>
__aicore__ inline void MhcExpandLaunch(GM_ADDR x, GM_ADDR o, const MhcExpandTilingData &info) {
    TPipe pipe;
    KernelMhcExpand<DT_X, BACKWARD> op;
    op.Init(x, o, info, pipe);
    op.Process();
}

// Entry of the compiled binary. `DT_X` is the template parameter declared for
// this operator, so the code generator instantiates the kernel once per tiling
// key and the body selects the element type of that instantiation.
template <typename DT_X, uint32_t MODE, uint32_t ROWS, uint32_t COLS, uint32_t S_LOG2>
__global__ __aicore__ void mhc_expand(GM_ADDR x, GM_ADDR o, GM_ADDR workspace, GM_ADDR tiling) {
    if constexpr (MODE == 7) {
        constexpr uint32_t sourceRows = 1u << S_LOG2;
        constexpr MhcExpandTilingData info{
            (sourceRows + 39u) / 40u, sourceRows, COLS, COLS, 1u, ROWS, 1u, 0u
        };
        MhcExpandLaunch<DT_X, 0>(x, o, info);
    } else if constexpr (MODE == 3) {
        MhcExpandSmallForward<DT_X, ROWS, COLS>(x, o, MhcExpandTilingData{});
    } else if constexpr (MODE == 4) {
        MhcExpandSmallReduce<DT_X, ROWS, COLS>(x, o, MhcExpandTilingData{});
    } else {
        REGISTER_TILING_DEFAULT(MhcExpandTilingData);
        GET_TILING_DATA_WITH_STRUCT(MhcExpandTilingData, tiling_data, tiling);
        if constexpr (MODE == 1) {
            MhcExpandSmallForward<DT_X>(x, o, tiling_data);
        } else if constexpr (MODE == 2) {
            MhcExpandSmallReduce<DT_X>(x, o, tiling_data);
        } else if (tiling_data.backward != 0) {
            MhcExpandLaunch<DT_X, 1>(x, o, tiling_data);
        } else {
            MhcExpandLaunch<DT_X, 0>(x, o, tiling_data);
        }
    }
}
