// route_control_test - the control windows' rules (SPEC-control-windows) on their own: cadence, row selection, the two kinds never in one
// window, the settings at 0 meaning no control windows, and the noise bounds of the k raise.  CPU only; no model, no GPU.
#include "strata/core/route_control.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace {
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "route_control_test: FAILED: %s\n", what);
        std::exit(1);
    }
}
}  // namespace

int main() {
    using namespace strata::core;
    const float nan = std::numeric_limits<float>::quiet_NaN();

    // cadence: one true per `every` eligible windows, the count advances only with a positive every
    {
        long long n = 0;
        int hits = 0;
        for (int i = 0; i < 128 * 10; ++i) hits += route_ctrl_due(n, 128) ? 1 : 0;
        check(hits == 10, "cadence: 10 control windows in 1280 eligible windows at every 128");
        check(n == 1280, "cadence: the count advances on every eligible window");
        long long z = 5;
        check(!route_ctrl_due(z, 0), "cadence: every 0 is never due");
        check(z == 5, "cadence: every 0 leaves the count alone");
        long long first = 0;
        check(route_ctrl_due(first, 1) && route_ctrl_due(first, 1), "cadence: every 1 is due in each window");
    }

    // selection: the first row r >= 1 that no stand-in moved; row 0 is never a cut; a NaN shift counts as unmoved
    {
        const float moved[4] = {0.0f, 0.0f, 1.5f, 0.0f};
        check(route_ctrl_first_unmoved(moved, 4) == 1, "selection: row 0 is skipped, row 1 unmoved is taken");
        const float all_moved[4] = {0.0f, 0.2f, 1.5f, 0.1f};
        check(route_ctrl_first_unmoved(all_moved, 4) == -1, "selection: rows 1.. all moved gives none");
        const float one[1] = {0.0f};
        check(route_ctrl_first_unmoved(one, 1) == -1, "selection: a single row gives none");
        const float nan_row[3] = {0.0f, 0.4f, nan};
        check(route_ctrl_first_unmoved(nan_row, 3) == 2, "selection: a NaN shift counts as unmoved");
        // a control window (every shift 0) selects r = 1 when at least two rows were kept
        const float ctl[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        check(route_ctrl_first_unmoved(ctl, 5) == 1, "selection: a control window picks r = 1");
    }

    // no coincidence: a fragile control window takes its window and the mask cadence does not advance in it; the mask control then waits
    {
        long long fcount = 0, mcount = 0;
        const long long fevery = 128, mevery = 8;
        long long mask_hits = 0, frag_hits = 0, both = 0, nonfrag = 0;
        for (int i = 0; i < 20000; ++i) {
            const bool fdue = route_ctrl_due(fcount, fevery);
            const WindowControl w = route_window_control(fdue, true, mcount, mevery);
            both += (w.fragile && w.mask) ? 1 : 0;
            frag_hits += w.fragile ? 1 : 0;
            mask_hits += w.mask ? 1 : 0;
            if (!w.fragile) ++nonfrag;
        }
        check(both == 0, "no coincidence: a window is never both kinds");
        check(frag_hits == 20000 / 128, "fragile: one window in 128 eligible windows");
        check(mcount == nonfrag, "no coincidence: the mask count advances only in windows the fragile control did not take");
        check(mask_hits == nonfrag / mevery, "no coincidence: the mask is due at its own cadence over the windows left to it");
        check(mask_hits > 0, "no coincidence: the mask still gets control windows");
    }

    // the mask's gate off (STRATA_MASK_CTRL=0 without STRATA_MASK_RESIDENT): no mask control window, and its count does not move
    {
        long long mcount = 0;
        int hits = 0;
        for (int i = 0; i < 1000; ++i) hits += route_window_control(false, false, mcount, 8).mask ? 1 : 0;
        check(hits == 0 && mcount == 0, "settings off: no mask control window and no cadence");
        long long fcount = 0;
        int fh = 0;
        for (int i = 0; i < 1000; ++i) fh += route_ctrl_due(fcount, 0) ? 1 : 0;   // STRATA_FRAGILE_CTRL_EVERY=0
        check(fh == 0 && fcount == 0, "settings off: no fragile control window and no cadence");
    }

    // noise bounds: an empty ring gives [0, 1], so no k raise passes; a filled ring gives a finite upper bound; a raise needs the ring's
    // lower bound above it
    {
        double lo = 0.0, hi = 0.0;
        route_ctrl_bounds(nullptr, 0, lo, hi);
        check(lo == 0.0 && hi == 1.0, "noise bounds: an empty control ring is [0, 1]");
        check(!route_k_raise_ok(0.0, hi), "noise bounds: an empty control ring blocks every k raise");

        bool quiet[200] = {};   // 200 control re-checks, none flipped
        route_ctrl_bounds(quiet, 200, lo, hi);
        check(std::isfinite(hi) && hi < 0.03, "noise bounds: 0 of 200 flips gives a small finite upper bound");
        bool noisy[200] = {};
        for (int i = 0; i < 60; ++i) noisy[i] = true;   // 30% flipped
        double ctl_lo = 0.0, ctl_hi = 0.0;
        route_ctrl_bounds(noisy, 200, ctl_lo, ctl_hi);
        check(std::isfinite(ctl_hi) && ctl_hi < 1.0 && ctl_hi > 0.3, "noise bounds: 30% flips gives a finite upper bound above 0.3");
        check(!route_k_raise_ok(0.25, ctl_hi), "a raise is refused when the ring's flips are within the control's noise");
        check(route_k_raise_ok(0.5, hi), "a raise passes when the ring's lower bound clears the control's upper bound");
    }

    // Wilson 90% on a known case: 0 of 10 gives lower 0 and an upper bound near 0.21
    {
        double lo = -1.0, hi = -1.0;
        route_wilson90(0, 10, lo, hi);
        check(std::fabs(lo) < 1e-12 && hi > 0.20 && hi < 0.23, "wilson90: 0 of 10 is [0, ~0.213]");
    }

    std::puts("route_control_test: passed (cadence, selection, no coincidence, settings off, noise bounds)");
    return 0;
}
