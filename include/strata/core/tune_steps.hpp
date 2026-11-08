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

enum Kind : int { Budget = 0, Mask = 1, Cap = 2, Prefetch = 3 };

/// The next test kind after `cur` in the rotation, skipping the kinds that are off.  Budget is always on, so this always returns one.
inline int next_kind(int cur, bool mask_on, bool cap_on, bool prefetch_on) {
    const bool on[4] = {true, mask_on, cap_on, prefetch_on};
    for (int s = 1; s <= 4; ++s) {
        const int k = (cur + s) % 4;
        if (on[k]) return k;
    }
    return Budget;
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
};

}  // namespace strata::core::tune
