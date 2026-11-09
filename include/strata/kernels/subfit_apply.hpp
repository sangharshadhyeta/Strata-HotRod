// include/strata/kernels/subfit_apply.hpp - SPEC-substitute-fit stage 2, steps 2-5: the device side of the per-pair corrections.
//
// Pieces:
//   - SubfitRoute: what the router kernels (native_router.cu) need per call: the group's orig / flag rows (written by the route kernels,
//     consumed by subfit_apply), the layer's LUT (e*512+s -> slot or -1), the per-slot residual, and the mode (1 dynamic, 2 all).
//     All null / 0 = no wiring: the kernels do exactly what they did before the learner.
//   - subfit_apply: one launch per layer, after the experts and their join, before the combine (verify.cpp).  For each row and slot k
//     with orig >= 0, LUT[l][orig*512+ids] = slot >= 0 and (mode all or the slot's flag): z = A_l x_row (x = the layer's MoE input),
//     parts[row,k] = c * parts[row,k] + B z.  It also clears orig and flag for the rows it reads (they are consumed).  The launch is
//     made whenever the mode is on (the window graph is captured once); the device count nfit[l] skips a layer with no fitted slot.
//   - subfit_tables_alloc: the device tables (LUT, A, B, c, resid, nfit).  Writes are cudaMemcpy calls made between windows.
//
// Layouts: parts [rows][K][N] float, raw expert outputs (before the router weight); x [rows][N] float (mixed_); ids [rows][K] int32;
// orig [rows][K] int16 (-1: not substituted); flag [rows][K] uint8; lut [512*512] int16 for the layer; A [r][N] fp16; B [slots][N][r]
// fp16 (B[j*r+m]); c [slots] float; resid [slots] float; nfit [1] int32.
#pragma once

#include "strata/core/subfit_rule.hpp"

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

/// The router kernels' view of the learner (per call; nullptr members = off).
struct SubfitRoute {
    int16_t* orig = nullptr;        ///< [rows][10] (this call's row base): the router's pick of a substituted slot, -1 otherwise
    uint8_t* flag = nullptr;        ///< [rows][10] (this call's row base): 1 = the move is corrected
    const int16_t* lut = nullptr;   ///< this layer's LUT [512*512]: e*512+s -> slot, or -1
    const float* resid = nullptr;   ///< [slots]: the held-out residual sqrt(E1/E0) of each slot's pair
    int mode = 0;                   ///< strata::core::subfit_rule::kModeDynamic (1) or kModeAll (2); 0 = off
};

struct SubfitApplyArgs {
    float* parts = nullptr;             ///< [n][K][N], this call's rows
    const float* x = nullptr;           ///< [n][N] the layer's MoE input (mixed_) of this call's rows
    const int32_t* ids = nullptr;       ///< [n][K] the picks after the substitutions
    int16_t* orig = nullptr;            ///< [n][K] consumed (reset to -1)
    uint8_t* flag = nullptr;            ///< [n][K] consumed (reset to 0)
    const int16_t* lut = nullptr;       ///< this layer's LUT
    const uint16_t* A = nullptr;        ///< this layer's A (r x N, fp16)
    const uint16_t* B = nullptr;        ///< all slots' B (slots x N x r, fp16)
    const float* c = nullptr;           ///< [slots]
    const int32_t* nfit = nullptr;      ///< this layer's fitted count (device): 0 = nothing to apply
    int r = 16, N = 2560, K = 10, n = 0;
    int mode = 1;                       ///< 1 dynamic (flagged moves), 2 all (every fitted move)
};

/// Launch for this layer's rows on `stream` (no host sync).  Null members or n <= 0: nothing is launched.
void subfit_apply(const SubfitApplyArgs& a, void* stream);

/// Device tables for `slots` slots of N x r, `layers` layers.  The LUT is set to -1, orig to -1, flag to 0 (the caller passes its own
/// buffers); the others zero.  Returns false (nothing kept) when one allocation fails.
struct SubfitTables {
    int16_t* lut = nullptr;      ///< [layers][512][512]
    uint16_t* A = nullptr;       ///< [layers][r][N]
    uint16_t* B = nullptr;       ///< [slots][N][r]
    float* c = nullptr;          ///< [slots]
    float* resid = nullptr;      ///< [slots]
    int32_t* nfit = nullptr;     ///< [layers]
    int layers = 0, slots = 0, N = 0, r = 0;
};
bool subfit_tables_alloc(SubfitTables& t, int layers, int slots, int N, int r);
void subfit_tables_free(SubfitTables& t);

}  // namespace strata::kernels
