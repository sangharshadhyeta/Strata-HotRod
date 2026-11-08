// tests/core/la_cadence_test.cpp - the look-ahead learner's training cadence (SPEC-adaptive-la change 2), CPU only.
//
// The rule is strata::kernels::la_cadence (include/strata/kernels/la_cadence.hpp): the same functions the device batch calls
// (la_b_prep_k, la_b_decide_k) and the host gate (la_learn_post_launch) use.  Checked here:
//   * the period: 4 windows after a freeze, 8 after 2,000 more picks, 16 after 4,000 more, never more;
//   * the drop test: a recent block below the long-run rate (Newcombe 90%, >= 100 recent picks) sends the period back to 4;
//     a stable block, or a short one, does not;
//   * a scripted frozen (layer, dist): the windows it trains in, the period through a freeze, a doubling, the cap, a drop and a change;
//   * the gate: a window with no trainer has no training launch; a window with one does; the legacy every-4th-batch rule is unchanged;
//   * the scoring and the cadence step run in every window while the training is gated: the picks count every window, and a
//     recent drop is seen in the window it happens, even when that window trains nothing;
//   * a 50-window run with a frozen leader and no live set launches 12 training batches (the due windows), not 50.
#include "strata/kernels/la_cadence.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace la = strata::kernels::la_cadence;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-96s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

// the hits of a block of n picks at rate p (rounded)
unsigned long long hits_at(double p, unsigned long long n) { return (unsigned long long) (p * (double) n + 0.5); }
}  // namespace

int main() {
    std::printf("la_cadence_test\n");

    // 1. the period: from a base of 0 (a freeze at 0 picks), doubling each 2,000 picks, capped at 16
    check(la::frozen_period(0, 0) == 4, "period 4 at the freeze (0 picks since the base)");
    check(la::frozen_period(1999, 0) == 4, "period 4 just before 2,000 picks");
    check(la::frozen_period(2000, 0) == 8, "period 8 at 2,000 picks");
    check(la::frozen_period(3999, 0) == 8, "period 8 just before 4,000 picks");
    check(la::frozen_period(4000, 0) == 16, "period 16 at 4,000 picks");
    check(la::frozen_period(1000000, 0) == 16, "period stays at the cap of 16 far past 4,000 picks");
    check(la::frozen_period(2000, 2000) == 4, "a base of 2,000 (a freeze at 2,000 picks) starts at 4 again");
    check(la::frozen_period(4000, 2000) == 8, "from base 2,000: 8 at 4,000");
    check(la::frozen_period(6000, 2000) == 16, "from base 2,000: 16 at 6,000");
    check(la::frozen_period(100, 200) == 4, "picks below the base (a reset in between) read as 4, not a wrap");

    // 2. the drop test
    check(la::recent_drop(hits_at(0.9, 2000), 2000, hits_at(0.5, 500), 500), "a recent 50% block against a long-run 90% is a drop");
    check(!la::recent_drop(hits_at(0.9, 2000), 2000, hits_at(0.9, 500), 500), "a recent block at the long-run rate is not a drop");
    check(!la::recent_drop(hits_at(0.9, 2000), 2000, hits_at(0.88, 500), 500), "a 2-point dip over 500 picks is within the interval");
    check(!la::recent_drop(hits_at(0.9, 2000), 2000, 0, 50), "a recent block of 50 picks (< 100) never drops");
    check(!la::recent_drop(0, 0, 0, 500), "no long-run picks: no drop");

    // 3. a scripted frozen (layer, dist): window w, picks, the recent block; it trains only in its due windows
    la::State st{0, 0, 1};
    la::set_frozen(st, 10, 2000);   // frozen at window 10 with 2,000 picks
    check(st.due == 14 && st.per == 4 && st.base == 2000, "a freeze at window 10 (2,000 picks): due 14, period 4");
    bool trained_early = false;
    for (int w = 11; w < 14; ++w) {
        trained_early |= la::trains_at(st.due, w);
        la::after_window(st, w, false, 2000 + 10 * (w - 10), hits_at(0.9, 2000), 500, hits_at(0.9, 500));
    }
    check(!trained_early && st.due == 14, "the frozen leader does not train in windows 11-13 (their cadence steps still run)");
    check(la::trains_at(st.due, 14), "the frozen leader trains in window 14");
    la::after_window(st, 14, true, 2400, hits_at(0.9, 2400), 500, hits_at(0.9, 500));
    check(st.due == 18 && st.per == 4, "stable after window 14 (400 picks since the base): due 18, period 4");
    la::after_window(st, 18, true, 4100, hits_at(0.9, 4100), 500, hits_at(0.9, 500));
    check(st.due == 26 && st.per == 8, "2,100 picks since the base at window 18: period 8, due 26");
    la::after_window(st, 26, true, 6100, hits_at(0.9, 6100), 500, hits_at(0.9, 500));
    check(st.due == 42 && st.per == 16, "4,100 picks since the base at window 26: period 16, due 42");
    la::after_window(st, 42, true, 9000, hits_at(0.9, 9000), 500, hits_at(0.9, 500));
    check(st.due == 58 && st.per == 16, "still at the cap of 16 at window 42: due 58");
    la::after_window(st, 58, true, 9500, hits_at(0.9, 9500), 500, hits_at(0.4, 500));
    check(st.due == 62 && st.per == 4 && st.base == 9500, "a recent drop (40% against 90%) at window 58: back to 4, due 62");
    la::after_window(st, 59, false, 9510, hits_at(0.9, 9510), 500, hits_at(0.4, 500));
    check(st.due == 62 && st.base == 9510, "a persisting drop at window 59 (not trained) keeps the due at 62, no stall");
    la::set_live(st, 62);
    check(st.per == 1 && st.due == 63 && la::trains_at(st.due, 63), "a change or reset (set_live): every window trains");

    // 4. the gate: no trainer in the window, no launch; the gate's validity; the legacy rule
    check(la::launch_needed(false, la::kNever, 5), "the gate not valid (no batch done yet): launch");
    check(!la::launch_needed(true, la::kNever, 5), "valid gate, no set due ever: no launch");
    check(!la::launch_needed(true, 9, 5), "valid gate, next trainer at window 9, this window 5: no launch");
    check(la::launch_needed(true, 5, 5), "valid gate, trainer due this window: launch");
    check(la::launch_needed(true, 4, 5), "valid gate, a trainer already due (late): launch");
    check(la::legacy_trains(0) && !la::legacy_trains(1) && !la::legacy_trains(3) && la::legacy_trains(4) && la::legacy_trains(8),
          "the legacy rule (STRATA_LA_CADENCE=0) trains every 4th batch: 0, 4, 8");

    // 5. a run: one frozen leader, no live set, 50 windows.  Every window scores (picks += 10) and steps the cadence; the training
    // launch happens only in the windows the gate lets through (the decide's next-training window, valid as the last batch is done)
    {
        la::State s2{0, 0, 1};
        unsigned long long picks = 2000;   // frozen at window 0
        la::set_frozen(s2, 0, picks);
        int min_due = s2.due;
        int launches = 0;
        std::vector<int> when;
        for (int w = 1; w <= 50; ++w) {
            picks += 10;
            const bool train = la::launch_needed(true, min_due, w);
            if (train) {
                ++launches;
                when.push_back(w);
            }
            la::after_window(s2, w, train && la::trains_at(s2.due, w), picks, hits_at(0.9, picks), 500, hits_at(0.9, 500));
            min_due = s2.due;
        }
        check(launches == 12 && when.size() >= 2 && when[0] == 4 && when[1] == 8,
              "50 windows with a frozen leader: training launches at its due windows (4, 8, 12, ...), not every window");
        check(picks == 2000 + 10 * 50, "the picks advance in every window (the scoring is not gated)");
        std::printf("    (training launches in 50 windows: %d, first windows: %d %d %d)\n", launches, when.size() > 0 ? when[0] : -1,
                    when.size() > 1 ? when[1] : -1, when.size() > 2 ? when[2] : -1);
    }

    // 6. a drop in a window that trains nothing is seen in that window: a frozen leader due at window 20 (next due 24), a recent block
    // that goes to 0 hits at window 21: the base moves to the picks of window 21 and the due comes in no later than 25
    {
        la::State s3{0, 0, 1};
        unsigned long long picks = 2000;
        la::set_frozen(s3, 0, picks);
        s3.due = 20;
        s3.per = 4;
        for (int w = 1; w <= 21; ++w) {
            picks += 10;
            const bool train = la::trains_at(s3.due, w);
            const unsigned long long rec_hits = w < 21 ? hits_at(0.9, 500) : 0;
            la::after_window(s3, w, train, picks, hits_at(0.9, picks), 500, rec_hits);
            if (w == 20) check(s3.due == 24, "a trained window 20 (4 windows on): due 24");
            if (w == 21) {
                check(!train, "window 21 trains nothing (no training launch)");
                check(s3.base == picks && s3.due <= 25, "the drop at window 21 is seen in that window: base moved, due <= 25");
            }
        }
    }

    std::printf("%s (%d failures)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
