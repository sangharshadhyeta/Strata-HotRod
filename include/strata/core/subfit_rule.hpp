// include/strata/core/subfit_rule.hpp - SPEC-substitute-fit stage 2, step 4/5: the shift cap's decision for one substitute move.
//
// One function, compiled for the device (native_router.cu, route_resident_warp and route_mask_warp) and for the CPU tests, so the
// rule the tests check is the rule the kernels run.  No CUDA here: __host__ __device__ only when nvcc compiles it.
//
// A move is a substitution of a router pick e by a stand-in s with the move's shift `plain` (w x max(gap, 0) x the pair's factor).
//   - fitted: the pair (e, s) has a fitted correction (its LUT entry has a slot), with the held-out residual `resid` = sqrt(E1/E0).
//   - mode all: every fitted move is taken and corrected (no cap test); its shift counts as plain x resid.
//   - mode dynamic: a move past the row's cap (shift_acc + plain > cap) is taken corrected when fitted (its shift counts as plain x resid),
//     and refused otherwise (the router's pick stays).  A move under the cap is taken plain, as today.
//   - mode off (or no fit): the cap refuses a move past the cap, exactly as before the learner existed.
#pragma once

#ifdef __CUDACC__
#define STRATA_SUBFIT_HD __host__ __device__ __forceinline__
#else
#define STRATA_SUBFIT_HD inline
#endif

namespace strata::core::subfit_rule {

constexpr int kModeOff = 0;
constexpr int kModeDynamic = 1;
constexpr int kModeAll = 2;

struct Move {
    bool take;       ///< the move is made (false: refused, the router's pick stays)
    bool corrected;  ///< the move is made with its fitted correction (the apply kernel corrects its slot)
    float add;       ///< the shift this move adds to the row's accumulated shift (plain, or plain x resid when corrected)
};

/// The decision for one move.  capped: the row has a cap (cap < +inf); shift_acc: the row's shift so far; plain: this move's shift.
STRATA_SUBFIT_HD Move decide(float shift_acc, float cap, bool capped, float plain, bool fitted, float resid, int mode) {
    Move d{true, false, plain};
    if (fitted && mode == kModeAll) {
        d.corrected = true;
    } else if (capped && shift_acc + plain > cap) {
        if (fitted && mode == kModeDynamic) {
            d.corrected = true;
        } else {
            d.take = false;
            d.add = 0.0f;
            return d;
        }
    }
    if (d.corrected) d.add = plain * resid;
    return d;
}

}  // namespace strata::core::subfit_rule
