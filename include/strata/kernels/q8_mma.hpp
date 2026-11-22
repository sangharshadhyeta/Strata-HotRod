// include/strata/kernels/q8_mma.hpp - Q8_0 weights x up to 8 int8-quantized activation rows on the tensor cores (mma.sync m16n8k32 s8), model-independent.
//
// A block of n <= 8 tokens is exactly one N = 8 tile: 16 weight rows x 32 k against the 8 tokens' k, so a weight is read once for the whole block and its products are int32 mma sums, not dp4a
// ALU work (a block of n cost ~n x one token with dp4a on the 3060).  A Q8_0 block (half scale + 32 int8) is one k32 step; the activation is q8_1-like blocks (36 bytes: half scale, half sum, 32 int8;
// the layout of moe::quantize_act and the engine's quantize_q8_1_rows).  Each block's int32 sum is scaled by weight scale x activation scale and accumulated in float, block after block.
#pragma once

#include <cstdint>

namespace strata::kernels::mma {

/// y[c][r] = sum_b dw[r][b] dx[c][b] (W[r][b] . X[c][b]) for r < n_out, c < cols (1 .. 8); W: n_out rows of n_in / 32 Q8_0 blocks (34 bytes), xq: cols rows of n_in / 32 blocks (36 bytes).
/// n_out must be a multiple of 16 and n_in of 32.  Returns false (nothing launched) otherwise.
bool q8_matmul(const uint8_t* W, const void* xq, float* y, int n_in, int n_out, int cols, void* stream);

}  // namespace strata::kernels::mma
