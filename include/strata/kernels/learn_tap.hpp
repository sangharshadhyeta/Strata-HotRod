// include/strata/kernels/learn_tap.hpp - STRATA_STANDIN_LEARN's output tap, read by the combine kernels.
//
// The combine kernels already read every rank's expert output (parts, plus the hit rows when they are folded in) and
// multiply it by its routing weight.  With a tap, the same kernel also copies, per row, the output of one rank:
//   - y_s: the row's first stand-in (rec[row][0] == layer) at rank rec[row][1]; written to ys[row][:] when gate[0] == 1.
//   - y_e: row 0 of the window, rank gate[2] at layer gate[1]; written to ye[:] when gate[0] == 2 and row0 is set.
// The copy is the raw output before the routing weight: one predicated store per row per layer, inside the launch that
// already runs, so the graph shape is unchanged.  A null tap (learning off) takes the kernels' original path.
#pragma once
#include <cstdint>

namespace strata::kernels {
struct MoeLearnTap {
    float* ys = nullptr;           ///< [rows][N] (from the call's first row): the y_s copies
    float* ye = nullptr;           ///< [N]: the exact successor's y_e copy (row 0 of the window)
    const int32_t* rec = nullptr;  ///< [rows][4]: each row's first stand-in (layer, rank, true expert, stand-in); layer -1 none
    const int32_t* gate = nullptr; ///< [4]: mode (0 off, 1 y_s, 2 y_e), layer, rank (device, set between windows)
    int32_t layer = 0;             ///< this layer's index
    int32_t row0 = 0;              ///< 1: the call's first row is the window's row 0 (the y_e row)
};
}  // namespace strata::kernels

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#elif defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif
#if defined(__CUDACC__) || defined(__HIPCC__)
// Device side: the rank whose raw part is this row's y_s (-1: none) and the rank of y_e (-1: none), then the copies.
__device__ inline int lrn_ys_rank(const strata::kernels::MoeLearnTap& t, int row) {
    if (t.rec == nullptr || t.gate[0] != 1) return -1;
    return t.rec[row * 4] == t.layer ? t.rec[row * 4 + 1] : -1;
}
__device__ inline int lrn_ye_rank(const strata::kernels::MoeLearnTap& t, int row) {
    if (t.rec == nullptr || !t.row0 || row != 0 || t.gate[0] != 2 || t.gate[1] != t.layer) return -1;
    return t.gate[2];
}
// one part's raw value (before its routing weight); a rank that is neither takes nothing
__device__ inline void lrn_put(const strata::kernels::MoeLearnTap& t, int ys_rank, int ye_rank, int row, int rank, int64_t n_embd,
                               int64_t col, float v) {
    if (rank == ys_rank) t.ys[int64_t(row) * n_embd + col] = v;
    if (rank == ye_rank) t.ye[col] = v;
}
__device__ inline void lrn_put4(const strata::kernels::MoeLearnTap& t, int ys_rank, int ye_rank, int row, int rank, int64_t n_embd,
                                int64_t c4, float4 v) {
    if (rank == ys_rank) {
        float* d = t.ys + int64_t(row) * n_embd + c4 * 4;
        d[0] = v.x; d[1] = v.y; d[2] = v.z; d[3] = v.w;
    }
    if (rank == ye_rank) {
        float* d = t.ye + c4 * 4;
        d[0] = v.x; d[1] = v.y; d[2] = v.z; d[3] = v.w;
    }
}
#endif
