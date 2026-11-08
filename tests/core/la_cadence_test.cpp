// tests/core/la_cadence_test.cpp - the look-ahead learner's training cadence (SPEC-adaptive-la change 2), CPU only.
//
// The rule is strata::kernels::la_cadence (include/strata/kernels/la_cadence.hpp): the same functions the device batch calls
// (la_b_prep_k, la_b_decide_k) and the host gate (la_learn_post_launch) use.  Checked here:
//   * the period: 4 windows after a freeze, 8 after 2,000 more picks, 16 after 4,000 more, never more;
//   * the drop test: a recent block below the long-run rate (Newcombe 90%, >= 100 recent picks) sends the period back to 4;
//     a stable block, or a short one, does not;
//   * a scripted frozen (layer, dist): the windows it trains in, the period through a freeze, a doubling, the cap, a drop and a change;
//   * the gate: a window with no trainer has no launch; a window with one does; the legacy every-4th-batch rule is unchanged;
//   * a 50-window run with a frozen leader and no live set launches 5 batches (the trained windows), not 50.
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
    for (int w = 11; w < 14; ++w) trained_early |= la::trains_at(st.due, w);
    check(!trained_early, "the frozen leader does not train in windows 11-13");
    check(la::trains_at(st.due, 14), "the frozen leader trains in window 14");
    la::after_train(st, 14, 2400, hits_at(0.9, 2400), 500, hits_at(0.9, 500));
    check(st.due == 18 && st.per == 4, "stable after window 14 (400 picks since the base): due 18, period 4");
    la::after_train(st, 18, 4100, hits_at(0.9, 4100), 500, hits_at(0.9, 500));
    check(st.due == 26 && st.per == 8, "2,100 picks since the base at window 18: period 8, due 26");
    la::after_train(st, 26, 6100, hits_at(0.9, 6100), 500, hits_at(0.9, 500));
    check(st.due == 42 && st.per == 16, "4,100 picks since the base at window 26: period 16, due 42");
    la::after_train(st, 42, 9000, hits_at(0.9, 9000), 500, hits_at(0.9, 500));
    check(st.due == 58 && st.per == 16, "still at the cap of 16 at window 42: due 58");
    la::after_train(st, 58, 9500, hits_at(0.9, 9500), 500, hits_at(0.4, 500));
    check(st.due == 62 && st.per == 4 && st.base == 9500, "a recent drop (40% against 90%) at window 58: back to 4, due 62");
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

    // 5. a run: one frozen leader, no live set, 50 windows of stable hits (the decide's next-training window gates each launch)
    {
        la::State s2{0, 0, 1};
        unsigned long long picks = 2000;   // frozen at window 0
        la::set_frozen(s2, 0, picks);
        int min_due = s2.due;
        int launches = 0;
        std::vector<int> when;
        for (int w = 1; w <= 50; ++w) {
            picks += 10;   // ten picks per launched window: a window's batch scores the picks it saw
            if (la::launch_needed(true, min_due, w)) {
                ++launches;
                when.push_back(w);
                if (la::trains_at(s2.due, w)) la::after_train(s2, w, picks, hits_at(0.9, picks), 500, hits_at(0.9, 500));
                min_due = s2.due;
            }
        }
        // the leader trains at the due windows only: 4, then 8 after 2,000 more picks, ... ; with 10 picks per launch and no
        // extra windows, the periods never reach 2,000 picks in 50 windows, so the trains are every 4 windows
        check(launches == 12 && when.size() >= 2 && when[0] == 4 && when[1] == 8,
              "50 windows with a frozen leader: the batch launches at its due windows (4, 8, 12, ...), not every window");
        std::printf("    (launches in 50 windows: %d, first windows: %d %d %d)\n", launches, when.size() > 0 ? when[0] : -1,
                    when.size() > 1 ? when[1] : -1, when.size() > 2 ? when[2] : -1);
    }

    std::printf("%s (%d failures)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
