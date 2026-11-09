// include/strata/kernels/subfit_capture.hpp - SPEC-substitute-fit stage 1a: the device side of STRATA_SUBFIT_LOG=1.
//
// Four predicated launches, all graph-safe (the flags are device ints set between windows; with the logger off none is launched):
//   - subfit_capture_x: in a stand-in window (gate[0] == 1), each row whose first stand-in is at this layer copies its layer input
//     (the normed row mixed_) into the x buffer.  Runs after the route-resident kernel, before the experts.
//   - subfit_probe_pre / subfit_probe_post: in a probe window (gate[3] == 1) for row 0 only: the router's picks before the
//     route-resident kernel (pre), then after it: flag[k] = the pick is still the router's (not replaced), resident, and row 0 had no
//     stand-in before this layer (clean); x = row 0's layer input; w = the router weights of row 0.
//   - subfit_probe_y: the raw expert outputs of row 0 (parts_, [K][N]) after the experts, before the combine (the host reads them after the window).
#pragma once

#include <cstdint>


namespace strata::kernels {

/// Device buffers of the probe (allocated only with the logger).  L = layers, K = routed experts per row (10), N = n_embd.
struct SubfitProbeBufs {
    float* x = nullptr;        ///< [L][N]: row 0's layer input at each layer
    float* y = nullptr;        ///< [L][K][N]: row 0's raw expert outputs (before the router weight)
    float* w = nullptr;        ///< [L][K]: row 0's router weights
    int32_t* flag = nullptr;   ///< [L][K]: 1 = resident, exact pick of row 0 at a clean row
    int32_t* pre = nullptr;    ///< [L][K]: row 0's router picks before the route-resident kernel
    int32_t* clean = nullptr;  ///< [L]: 1 = row 0 had no stand-in before this layer
};

/// rows [0, n_rows): mixed (row-major [n_rows][N]) -> x for each row with rec[row*4] == layer while gate[0] == 1.
void subfit_capture_x(const float* mixed, const int32_t* rec, const int32_t* gate, int layer, float* x, int64_t N, int n_rows,
                      void* stream);
/// Row 0's picks before the route-resident kernel: pre[layer][k] = ids[k]; clean[layer] = 1 when rec[0] (row 0's first stand-in layer) is -1;
/// flag[layer][*] = 0 (a layer whose post did not run stays unsampled).
void subfit_probe_pre(const int32_t* ids, const int32_t* rec, const int32_t* gate, int layer, int K, int32_t* pre, int32_t* clean,
                      int32_t* flag, void* stream);
/// After the route-resident kernel: flag, w and x of row 0 at this layer (see the header comment).  res: the layer's d_res [n_expert].
void subfit_probe_post(const int32_t* ids, const float* w, const int32_t* res, const int32_t* pre, const int32_t* clean,
                       const int32_t* gate, const float* mixed0, int layer, int K, int64_t N, float* x, float* w_out, int32_t* flag,
                       void* stream);
/// Row 0's K raw expert outputs (parts: [K][N]) into y (the layer's [K][N] slab), while gate[3] == 1.
void subfit_probe_y(const float* parts, const int32_t* gate, int64_t K_N, float* y, void* stream);

/// SPEC-substitute-fit stage 2, shadow samples: while gate[3] == 2 (the exact successor of a cut whose row listed its substitutions),
/// row 0's picks (ids [K]) and MoE input (mixed0 [N]) of this layer into ids_out[layer][K] and x_out[layer][N] ...
void subfit_shadow_xi(const int32_t* ids, const float* mixed0, const int32_t* gate, int layer, int K, int64_t N, int32_t* ids_out,
                      float* x_out, void* stream);
/// ... and its K raw expert outputs (parts [K][N]) into y_out[layer][K][N].
void subfit_shadow_y(const float* parts, const int32_t* gate, int layer, int64_t K_N, float* y_out, void* stream);

}  // namespace strata::kernels
