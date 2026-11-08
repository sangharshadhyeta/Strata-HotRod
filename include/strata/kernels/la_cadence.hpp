// la_cadence.hpp - the look-ahead learner's training cadence (SPEC-adaptive-la, change 2): the rule the device batch
// (native_router.cu la_b_prep_k / la_b_decide_k) and the CPU unit test (tests/core/la_cadence_test.cpp) both use, so the
// test runs the same arithmetic the GPU does.  Everything here is integer or the 90% Wilson / Newcombe bound (float), no
// text matching.
//
// A frozen (layer, dist) trains its leader only in the windows its state names ("due").  The leader's period starts at 4
// windows and doubles after each kFreezePicks picks without a change (4 -> 8 -> 16, at most 16).  A recent hit rate below
// the long-run rate (Newcombe 90%, over a recent block of >= kRecentMin picks) sends the period back to 4 at once, and so
// does a change or a reset (a live (layer, dist) trains every window: period 1).
//
// A window's batch is skipped when no (layer, dist) trains in it (launch_needed): the host reads the decide kernel's
// next-training window (a host-mapped word) and launches only when that window has a trainer.
#pragma once

#include <climits>
#include <cmath>

#if defined(__CUDACC__)
#define STRATA_LA_HD __host__ __device__ __forceinline__
#else
#define STRATA_LA_HD inline
#endif

namespace strata::kernels::la_cadence {

inline constexpr int kFreezePicks = 2000;   // picks without a change before a (layer, dist) freezes; also the doubling step
inline constexpr int kBasePeriod = 4;       // a frozen leader's windows between trainings to start, and after a drop
inline constexpr int kMaxPeriod = 16;       // the cap
inline constexpr int kRecentMin = 100;      // the recent block must hold this many picks before the drop test runs
inline constexpr int kNever = INT_MAX;      // due of a set that never trains (no target layer, or a plain leader while frozen)
inline constexpr float kZ90 = 1.6448536f;

// the 90% Wilson interval of a proportion p over n picks
STRATA_LA_HD void wilson(float p, float n, float& lo, float& hi) {
    const float z2 = kZ90 * kZ90, den = 1.0f + z2 / n;
    const float c = (p + z2 / (2.0f * n)) / den;
    const float h = kZ90 * sqrtf(p * (1.0f - p) / n + z2 / (4.0f * n * n)) / den;
    lo = c - h;
    hi = c + h;
}

// Newcombe's lower bound for p1 - p2 from two Wilson 90% intervals over n1 and n2 picks (> 0: p1 is above p2)
STRATA_LA_HD float diff_lo(float p1, float n1, float p2, float n2) {
    float l1, u1, l2, u2;
    wilson(p1, n1, l1, u1);
    wilson(p2, n2, l2, u2);
    return (p1 - p2) - sqrtf((p1 - l1) * (p1 - l1) + (u2 - p2) * (u2 - p2));
}

// the frozen leader's windows between trainings, given the picks since its cadence base (its freeze, or its last drop)
STRATA_LA_HD int frozen_period(unsigned long long picks, unsigned long long base) {
    const unsigned long long k = picks > base ? (picks - base) / (unsigned long long) kFreezePicks : 0ull;
    if (k == 0) return kBasePeriod;
    return k == 1 ? 2 * kBasePeriod : kMaxPeriod;
}

// the drop test: the recent block's hit rate is below the long-run rate (the leader's hits and picks since its change)
STRATA_LA_HD bool recent_drop(unsigned long long hits_long, unsigned long long picks_long, unsigned long long hits_rec,
                              unsigned long long picks_rec) {
    if (picks_long == 0 || picks_rec < (unsigned long long) kRecentMin) return false;
    const float pl = (float) hits_long / (float) picks_long, pr = (float) hits_rec / (float) picks_rec;
    return diff_lo(pl, (float) picks_long, pr, (float) picks_rec) > 0.0f;
}

// one (layer, dist)'s cadence state (device memory: one per (s,d))
struct State {
    unsigned long long base;   // picks at the start of the current period step (the freeze, or the last drop; 0 when live)
    int due;                   // the window index whose batch trains the frozen leader next (kNever: never)
    int per;                   // the current period in windows (1 live, 4..16 frozen); the stats line's median
};

// a live (layer, dist) trains every window: the next window is due
STRATA_LA_HD void set_live(State& st, int w) {
    st.base = 0;
    st.due = w + 1;
    st.per = 1;
}

// the (layer, dist) freezes at window w with `picks` picks since its last change: the leader's first period
STRATA_LA_HD void set_frozen(State& st, int w, unsigned long long picks) {
    st.base = picks;
    st.due = w + kBasePeriod;
    st.per = kBasePeriod;
}

// the frozen leader trained in window w: the next training by the period the picks give, or back to kBasePeriod on a drop
STRATA_LA_HD void after_train(State& st, int w, unsigned long long picks, unsigned long long hits_long, unsigned long long picks_rec,
                              unsigned long long hits_rec) {
    if (recent_drop(hits_long, picks, hits_rec, picks_rec)) {
        st.base = picks;
        st.per = kBasePeriod;
        st.due = w + kBasePeriod;
        return;
    }
    st.per = frozen_period(picks, st.base);
    st.due = w + st.per;
}

// does the frozen leader train in window w (its state's due window has come)
STRATA_LA_HD bool trains_at(int due, int w) { return due <= w; }

// the batch of window w is launched unless no set trains in it.  gate_valid: the last launched batch has finished, so min_due (its
// decide's next-training window over all sets) is current; without it the batch is launched as before
STRATA_LA_HD bool launch_needed(bool gate_valid, int min_due, int w) { return !gate_valid || min_due <= w; }

// the legacy frozen rule (STRATA_LA_CADENCE=0): the leader trains every 4th batch, by the batch counter
inline constexpr int kLegacyEvery = 4;
STRATA_LA_HD bool legacy_trains(int tick) { return tick % kLegacyEvery == 0; }

}  // namespace strata::kernels::la_cadence

#undef STRATA_LA_HD
