// Tiling structure shared by the Host tiling function and the Kernel.
#pragma once

#include <cstdint>

// A block owns `rowTile` consecutive output rows starting at its own block
// index times `rowTile`, up to `rowTotal` rows in total. Every row is `rowLen`
// elements long (the innermost stride of both operands) and is covered by
// `colTiles` column tiles of at most `tileLen` elements.
//
// Backward folds the `lanes` copies of every output row into one total.
struct MhcExpandTilingData {
    uint32_t rowTile;    // number of output rows handled by this block
    uint32_t rowTotal;   // number of output rows of the whole operator
    uint32_t rowLen;     // elements per row (D)
    uint32_t tileLen;    // elements per column tile
    uint32_t colTiles;   // number of column tiles covering one row
    uint32_t lanes;      // expansion factor (mhc_mult)
    uint32_t batched;   // forward source rows are grouped in UB
    uint32_t backward;   // 1 for the reduction direction
};
