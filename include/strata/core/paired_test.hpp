// include/strata/core/paired_test.hpp - the decision routine of the on-line tuners (STRATA_ROUTE_BUDGET_AUTO and STRATA_MASK=learn).
//
// A test compares a candidate with the base on adjacent interleaved runs.  Each pair gives d = ln(rate_cand / rate_base).  The test
// moves to the candidate only when the mean of d is above the minimum effect delta with 95% confidence (mean - 2 se > delta), keeps
// the base when the mean is below it with the same confidence (mean + 2 se < delta) or at the pair cap, and otherwise continues.
// A quality veto (flips per committed token) ends the test as a keep.  Self-contained: no CUDA, no engine state, so the CPU tests
// (tests/core/paired_test_test.cpp) run the same code the tuners run.
#pragma once

#include <algorithm>
#include <cmath>

namespace strata::core::paired {

// Welford mean and variance over the pair differences.
struct PairStats {
    long long n = 0;
    double mean = 0.0;
    double m2 = 0.0;
    void add(double d) {
        ++n;
        const double dl = d - mean;
        mean += dl / (double) n;
        m2 += dl * (d - mean);
    }
    double var() const { return n > 1 ? m2 / (double) (n - 1) : 0.0; }
    double se() const { return n > 1 ? std::sqrt(var() / (double) n) : 0.0; }
};

// The rule of one kind of test.  z = 2.0: one-sided ~97.7% for each look, conservative for the repeated looks.
struct Rule {
    long long min_pairs;   // no decision before this many pairs
    long long cap;         // at this many pairs without a decision: keep
    double delta;          // the minimum effect, ln(1 + min effect)
    double z;
};

enum class Verdict { Continue, Move, Keep };
enum class Reason { None, Move, BelowDelta, Cap, Quality, Cuts };

// The budget test's rule: 12 pairs at least, 160 at most.  The mask test: the same minimum, 240 at most (its effect is measured on
// a slower-moving quantity).  min_effect is the fraction (STRATA_TUNE_MIN_EFFECT, 0.03 = 3%).
inline Rule budget_rule(double min_effect) { return Rule{12, 160, std::log(1.0 + min_effect), 2.0}; }
inline Rule mask_rule(double min_effect) { return Rule{12, 240, std::log(1.0 + min_effect), 2.0}; }

// The per-pair value.  False (no pair) when a rate is not positive: a run with no committed tokens or no wall time.
inline bool pair_diff(double tok_base, double secs_base, double tok_cand, double secs_cand, double& d) {
    if (!(tok_base > 0.0) || !(secs_base > 0.0) || !(tok_cand > 0.0) || !(secs_cand > 0.0)) return false;
    d = std::log((tok_cand / secs_cand) / (tok_base / secs_base));
    return true;
}

// Wilson score bounds of f successes in n trials at z = 1.645 (the 90% two-sided interval the tuners use).
inline void wilson90(long long f, long long n, double& lo, double& hi) {
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

// The quality veto on flips per committed token.  It needs >= min_tok committed tokens in both arms; then it blocks when the
// candidate's Wilson lower bound exceeds the base's upper bound, or when its point rate exceeds 1.5 times the base's with >= min_flips
// flips.  Returns true when the candidate is blocked.
inline bool quality_veto(long long flips_base, long long tok_base, long long flips_cand, long long tok_cand, long long min_tok,
                         long long min_flips) {
    if (tok_base < min_tok || tok_cand < min_tok) return false;
    double base_lo = 0.0, base_hi = 0.0, cand_lo = 0.0, cand_hi = 0.0;
    wilson90(flips_base, tok_base, base_lo, base_hi);
    wilson90(flips_cand, tok_cand, cand_lo, cand_hi);
    const double pb = (double) flips_base / (double) tok_base;
    const double pc = (double) flips_cand / (double) tok_cand;
    return cand_lo > base_hi || (pc > 1.5 * pb && flips_cand >= min_flips);
}

// The cut guard (STRATA_RECHECK_ROW_ONLY's tuner, SPEC-recheck-cheap change 4): the candidate's cuts per committed token. Blocked when
// the candidate's Wilson 90% lower bound exceeds twice the base's upper bound. Needs >= min_tok tokens in both arms (else not blocked).
inline bool cut_veto(long long cuts_base, long long tok_base, long long cuts_cand, long long tok_cand, long long min_tok) {
    if (tok_base < min_tok || tok_cand < min_tok) return false;
    double base_lo = 0.0, base_hi = 0.0, cand_lo = 0.0, cand_hi = 0.0;
    wilson90(std::min(cuts_base, tok_base), tok_base, base_lo, base_hi);
    wilson90(std::min(cuts_cand, tok_cand), tok_cand, cand_lo, cand_hi);
    return cand_lo > 2.0 * base_hi;
}

// One look after a pair.  quality_blocked: the quality veto holds (the test ends as a keep, the base's own outcome today).
// cuts_blocked: the cut guard holds (a keep, reason Cuts; the quality veto takes precedence).
// The sequential rule: move when mean - z se > delta; keep when mean + z se < delta, or at the cap; else continue.
inline Verdict decide(const PairStats& s, const Rule& r, bool quality_blocked, Reason* why = nullptr, bool cuts_blocked = false) {
    Reason reason = Reason::None;
    Verdict v = Verdict::Continue;
    if (quality_blocked) {
        v = Verdict::Keep;
        reason = Reason::Quality;
    } else if (cuts_blocked) {
        v = Verdict::Keep;
        reason = Reason::Cuts;
    } else if (s.n >= r.min_pairs) {
        const double half = r.z * s.se();
        if (s.mean - half > r.delta) {
            v = Verdict::Move;
            reason = Reason::Move;
        } else if (s.mean + half < r.delta) {
            v = Verdict::Keep;
            reason = Reason::BelowDelta;
        } else if (s.n >= r.cap) {
            v = Verdict::Keep;
            reason = Reason::Cap;
        }
    } else if (s.n >= r.cap) {
        v = Verdict::Keep;
        reason = Reason::Cap;
    }
    if (why != nullptr) *why = reason;
    return v;
}

// The direction state of a test (the budget's and the mask's, the same rule): a win keeps the direction first; a loss tries the other
// direction against the same base; both lost: the other direction first next round.  The candidate's direction is first (stage 0)
// or -first (stage 1).
struct DirState {
    int first = 1;
    int stage = 0;
    int dir() const { return stage == 0 ? first : -first; }
    void update(bool won, int d) {
        if (won) {
            first = d;
            stage = 0;
        } else if (stage == 0) {
            stage = 1;
        } else {
            first = -first;
            stage = 0;
        }
    }
};

// The budget's step factor s (candidate = b * s^d): a win in the same direction as the last win grows it (x1.25, to 2); a loss
// shrinks it toward 1.1 (square root, floor 1.1).
struct BudgetStep {
    float step = 1.25f;
    int last_win = 0;
    void update(bool won, int d) {
        if (won) {
            if (d == last_win) step = std::min(step * 1.25f, 2.0f);
            last_win = d;
        } else {
            step = std::max(std::sqrt(step), 1.1f);
        }
    }
};

// The mask's depth step: a move in the same direction as the last move doubles the step (to 8); a loss halves it (to 1).
struct MaskStep {
    double step = 1.0;
    int last_move = 0;
    void on_move(int d) {
        if (d == last_move) step = std::min(8.0, step * 2.0);
        last_move = d;
    }
    void on_loss() { step = std::max(1.0, step * 0.5); }
};

}  // namespace strata::core::paired
