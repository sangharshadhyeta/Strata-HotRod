// The control windows' rules (SPEC-control-windows), shared by verify.cpp and its CPU test (tests/core/route_control_test.cpp).
// Pure functions only: no CUDA, no globals.
//
// A fragile control window (STRATA_FRAGILE_CTRL_EVERY) and a mask control window (STRATA_MASK_CTRL / STRATA_MASK_RESIDENT) are decided
// before their decode window runs; the kernels read the flag at launch.  The fragile control is the one that wins a window: the mask
// control waits for the next one and its cadence counter does not advance in a window the fragile control takes.
#pragma once

#include <algorithm>
#include <cmath>

namespace strata::core {

/// Cadence: counts one eligible window and is true on every `every`-th count (every <= 0: never, and the count is not advanced).
inline bool route_ctrl_due(long long& count, long long every) {
    if (every <= 0) return false;
    return ++count % every == 0;
}

/// The first row r >= 1 of rows 0..n-1 that no stand-in moved (shift not above 0; a NaN shift counts as unmoved), or -1.  Row 0 is never
/// a cut (a re-run would step the GDN / PLE state twice).
inline int route_ctrl_first_unmoved(const float* shift, int n) {
    for (int r = 1; r < n; ++r)
        if (!(shift[r] > 0.0f)) return r;
    return -1;
}

/// Wilson score interval at 90% (z = 1.645) for f flips in n re-checks: the flip rate's lower and upper bounds (0 and 1 when n is 0).
inline void route_wilson90(long long f, long long n, double& lo, double& hi) {
    if (n <= 0) {
        lo = 0.0;
        hi = 1.0;
        return;
    }
    const double z = 1.645, z2 = z * z, nn = (double) n, p = (double) f / nn;
    const double centre = p + z2 / (2.0 * nn);
    const double half = z * std::sqrt(p * (1.0 - p) / nn + z2 / (4.0 * nn * nn));
    const double den = 1.0 + z2 / nn;
    lo = (centre - half) / den;
    hi = (centre + half) / den;
}

/// The noise floor: Wilson 90% bounds of the flip rate of the first n entries of a control ring (true = flipped).
/// Zero entries give [0, 1], so nothing above noise is shown before the ring holds a re-check.
inline void route_ctrl_bounds(const bool* ring, int n, double& lo, double& hi) {
    long long f = 0;
    for (int i = 0; i < n; ++i)
        if (ring[i]) ++f;
    route_wilson90(f, n, lo, hi);
}

/// A k raise passes only when the flips are above the noise: the ring's flip-rate lower bound over the control's upper bound.
inline bool route_k_raise_ok(double ring_lo, double ctrl_hi) { return ring_lo > ctrl_hi; }

/// Which control kind a decode window is.  The fragile control comes first (fragile_due); the mask control needs its gate and its own
/// cadence, and a fragile window does not advance the mask's counter (the mask control waits for the next window).
struct WindowControl {
    bool fragile;
    bool mask;
};
inline WindowControl route_window_control(bool fragile_due, bool mask_gate, long long& mask_count, long long mask_every) {
    WindowControl w{fragile_due, false};
    w.mask = !fragile_due && mask_gate && route_ctrl_due(mask_count, mask_every);
    return w;
}

}  // namespace strata::core
