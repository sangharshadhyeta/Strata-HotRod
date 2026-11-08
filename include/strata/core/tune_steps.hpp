// include/strata/core/tune_steps.hpp - SPEC-recheck-gated step G: the rotation of the paired tuner and the candidates of its two new tests
// (the shift cap's multipliers, the prefetch count), header-only, no CUDA.  The decisions themselves are paired_test.hpp's (decide, the quality
// veto, the cut guard) as the budget and mask tests use them; this header only says which test runs next and what its candidate is.
//
// Rotation: budget -> mask -> cap -> prefetch -> budget.  A kind that is off is skipped; the budget test is always on.
#pragma once

#include "strata/core/paired_test.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace strata::core::tune {

// SPEC-recheck-gated step H2: KMult is the margin m of the cut threshold (k_eff = m x k_safe), the fifth kind of the rotation.
enum Kind : int { Budget = 0, Mask = 1, Cap = 2, Prefetch = 3, KMult = 4 };

/// The next test kind after `cur` in the rotation budget -> mask -> cap -> prefetch -> k -> budget, skipping the kinds that are off.
/// Budget is always on, so this always returns one.
inline int next_kind(int cur, bool mask_on, bool cap_on, bool prefetch_on, bool k_on = false) {
    const bool on[5] = {true, mask_on, cap_on, prefetch_on, k_on};
    for (int s = 1; s <= 5; ++s) {
        const int k = (cur + s) % 5;
        if (on[k]) return k;
    }
    return Budget;
}

// ---- the margin m (H2): candidates m x 1.25^d, range [0.6, 1.5], start 1.0

constexpr double kKMultLo = 0.6;
constexpr double kKMultHi = 1.5;

/// The candidate of a margin test in direction d (+1 up, -1 down): m x step^d (step: the test's step factor, 1.25 at the start).  false when
/// it clamps to m (at a bound): nothing to test.
inline bool k_mult_candidate(double m, double step, int d, double& out) {
    if (!(step > 0.0)) return false;
    const double cand = std::max(kKMultLo, std::min(kKMultHi, m * std::pow(step, (double) (d < 0 ? -1 : 1))));
    if (std::fabs(cand - m) < 1e-12) return false;
    out = cand;
    return true;
}

/// H2 quality veto (the explore cuts): the candidate arm's explore flip rate has a Wilson 90% lower bound above the larger of the control's
/// upper bound and the base arm's explore upper bound (the rows it stopped re-checking flip above noise).  Needs >= min_rc explore cuts in
/// the candidate arm; else not blocked.  Counts: flips and cuts per arm (base, candidate).
inline bool explore_veto(long long xfl_base, long long xrc_base, long long xfl_cand, long long xrc_cand, double ctrl_hi, long long min_rc) {
    if (xrc_cand < min_rc) return false;
    double b_lo = 0.0, b_hi = 0.0, c_lo = 0.0, c_hi = 0.0;
    paired::wilson90(xfl_base, xrc_base, b_lo, b_hi);
    paired::wilson90(xfl_cand, xrc_cand, c_lo, c_hi);
    return c_lo > std::max(ctrl_hi, b_hi);
}

// ---- the shift cap's multipliers (both move by the same factor: the ratio unc / none is kept; each stays in [0.25, 8] x the median)
constexpr double kCapLo = 0.25;
constexpr double kCapHi = 8.0;

/// The candidate of a cap test: both multipliers times f = step^d, f limited so that both stay in [kCapLo, kCapHi].  false when the candidate
/// is the base (f clamps to 1, or a multiplier is not positive): nothing to test.
inline bool cap_candidate(double unc, double none, double step, int d, double& cand_unc, double& cand_none) {
    if (!(unc > 0.0) || !(none > 0.0) || !(step > 0.0)) return false;
    const double hi = std::min(kCapHi / unc, kCapHi / none);
    const double lo = std::max(kCapLo / unc, kCapLo / none);
    const double f = std::min(hi, std::max(lo, std::pow(step, (double) d)));
    if (std::fabs(f - 1.0) < 1e-12) return false;
    cand_unc = unc * f;
    cand_none = none * f;
    return true;
}

// ---- the prefetch count (per cut row, STRATA_RECHECK_PREFETCH_MAX): x or / 1.5, rounded, at least one step; 0 is a candidate; range 0-64

constexpr long long kPrefetchLo = 0;
constexpr long long kPrefetchHi = 64;

/// The candidate of a prefetch test in direction d (+1 up, -1 down).  false when it equals n (at a bound): nothing to test.
inline bool prefetch_candidate(long long n, int d, long long& out) {
    const int dir = d < 0 ? -1 : 1;
    long long m = std::llround((double) n * std::pow(1.5, (double) dir));
    if (m == n) m = n + dir;
    m = std::max(kPrefetchLo, std::min(kPrefetchHi, m));
    if (m == n) return false;
    out = m;
    return true;
}

/// The persisted state of the two tests: the chosen values, their step and direction states (paired::BudgetStep, paired::DirState).
struct Tune {
    bool set = false;          ///< a test has been decided or the values were saved: the env values are no longer the start
    double cap_unc = 1.5;      ///< the uncertain-row cap multiplier (x the running median)
    double cap_none = 1.0;     ///< the none-row cap multiplier
    float cap_step = 1.25f;    ///< the cap test's step factor
    int cap_last_win = 0;
    int cap_first = 1, cap_stage = 0;   ///< the cap test's direction state
    long long prefetch = 12;   ///< the prefetch count per cut row
    int pf_first = 1, pf_stage = 0;     ///< the prefetch test's direction state
    double k_mult = 1.0;       ///< step H2: the margin m (k_eff = m x k_safe), range 0.6-1.5
    float k_step = 1.25f;      ///< the margin test's step factor
    int k_last_win = 0;
    int k_first = 1, k_stage = 0;       ///< the margin test's direction state
};

}  // namespace strata::core::tune
