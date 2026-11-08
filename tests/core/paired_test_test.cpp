// tests/core/paired_test_test.cpp - the on-line tuners' decision routine (include/strata/core/paired_test.hpp) on the CPU: no CUDA, no
// engine.  (a) the A/A false-move rate over 10,000 simulated tests per sigma; (b) the power of a true 6% gain; (c) the quality veto;
// (d) the sequential rule's edges (minimum pairs, keep, cap); (e) Welford against a two-pass computation; (f) the per-pair value;
// (g) scripted sequences through the stage and step rules the tuners use (budget and mask).
#include "strata/core/paired_test.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace pt = strata::core::paired;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-96s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

// One simulated test: pairs drawn from draw() until a decision or the cap.  Returns the verdict and the pairs used.
struct SimResult {
    pt::Verdict v;
    long long pairs;
};
template <class Draw>
SimResult simulate(const pt::Rule& r, Draw draw) {
    pt::PairStats s;
    for (long long k = 1;; ++k) {
        s.add(draw());
        const pt::Verdict v = pt::decide(s, r, false);
        if (v != pt::Verdict::Continue) return {v, k};
    }
}

const double kEffect = 0.03;   // STRATA_TUNE_MIN_EFFECT's default
}  // namespace

int main() {
    std::printf("paired_test_test\n");
    const pt::Rule budget = pt::budget_rule(kEffect);
    const pt::Rule mask = pt::mask_rule(kEffect);

    // (e) Welford against a two-pass mean and sample variance
    {
        std::mt19937_64 rng(7);
        std::normal_distribution<double> nd(0.02, 0.05);
        std::vector<double> xs;
        pt::PairStats s;
        for (int i = 0; i < 500; ++i) {
            xs.push_back(nd(rng));
            s.add(xs.back());
        }
        double m = 0.0;
        for (double x : xs) m += x;
        m /= (double) xs.size();
        double v = 0.0;
        for (double x : xs) v += (x - m) * (x - m);
        v /= (double) (xs.size() - 1);
        check(std::fabs(s.mean - m) < 1e-12 && std::fabs(s.var() - v) < 1e-12 && std::fabs(s.se() - std::sqrt(v / 500.0)) < 1e-12,
              "Welford mean, variance and se match a two-pass computation (500 draws)");
        pt::PairStats one;
        one.add(0.5);
        check(one.se() == 0.0 && one.mean == 0.5, "one pair: se 0 (no variance yet), mean is the pair");
    }

    // (f) the per-pair value
    {
        double d = 0.0;
        check(pt::pair_diff(800.0, 10.0, 1000.0, 10.0, d) && std::fabs(d - std::log(1.25)) < 1e-12,
              "pair value is ln(rate_cand / rate_base): 1000 tok in 10 s vs 800 tok in 10 s is ln(1.25)");
        check(pt::pair_diff(800.0, 10.0, 600.0, 20.0, d) && std::fabs(d - std::log(0.375)) < 1e-12,
              "pair value: a candidate at 30 tok/s (600 tok in 20 s) vs a base at 80 tok/s is ln(0.375)");
        check(!pt::pair_diff(0.0, 10.0, 1000.0, 10.0, d) && !pt::pair_diff(800.0, 0.0, 1000.0, 10.0, d) &&
                  !pt::pair_diff(800.0, 10.0, 0.0, 10.0, d) && !pt::pair_diff(800.0, 10.0, 1000.0, -1.0, d),
              "a block with no committed tokens or no wall time is no pair");
    }

    // (d) the sequential rule's edges
    {
        pt::PairStats s;
        pt::Verdict v = pt::Verdict::Continue;
        for (int i = 0; i < 11; ++i) {
            s.add(0.5);
            v = pt::decide(s, budget, false);
        }
        check(v == pt::Verdict::Continue, "no decision before 12 pairs, even for a huge difference (11 pairs of +0.5)");
        s.add(0.5);
        v = pt::decide(s, budget, false);
        check(v == pt::Verdict::Move, "a huge difference moves at the 12th pair (zero spread, mean - 2 se > delta)");

        pt::PairStats z;
        for (int i = 0; i < 12; ++i) z.add(0.0);
        check(pt::decide(z, budget, false) == pt::Verdict::Keep, "zero difference keeps at the 12th pair (mean + 2 se < delta)");

        pt::PairStats cap;
        for (int i = 0; i < 160; ++i) cap.add((i % 2) ? 0.5 : -0.5);   // mean 0, sd 0.5: never decided
        pt::Reason why = pt::Reason::None;
        const pt::Verdict vc = pt::decide(cap, budget, false, &why);
        check(vc == pt::Verdict::Keep && why == pt::Reason::Cap, "at the budget cap (160 pairs) an undecided test keeps");
        pt::PairStats capm;
        for (int i = 0; i < 240; ++i) capm.add((i % 2) ? 0.5 : -0.5);
        check(pt::decide(capm, mask, false) == pt::Verdict::Keep && pt::decide(capm, mask, false) != pt::Verdict::Move &&
                  pt::decide(cap, mask, false) == pt::Verdict::Continue,
              "mask rule: 160 undecided pairs continue (cap 240), 240 keep");

        // the minimum effect: a 2% gain with tight spread does not move, a 4% gain does
        pt::PairStats two, four;
        std::mt19937_64 rng(11);
        std::normal_distribution<double> n2(std::log(1.02), 0.002), n4(std::log(1.04), 0.002);
        pt::Verdict v2 = pt::Verdict::Continue, v4 = pt::Verdict::Continue;
        for (int i = 0; i < 160 && v2 == pt::Verdict::Continue; ++i) {
            two.add(n2(rng));
            v2 = pt::decide(two, budget, false);
        }
        for (int i = 0; i < 160 && v4 == pt::Verdict::Continue; ++i) {
            four.add(n4(rng));
            v4 = pt::decide(four, budget, false);
        }
        check(v2 == pt::Verdict::Keep, "a true 2% gain (tight spread) is kept: below the 3% minimum effect");
        check(v4 == pt::Verdict::Move, "a true 4% gain (tight spread) moves: above the 3% minimum effect");
    }

    // (a) A/A: two identical arms, pairs ~ N(0, sigma).  False-move rate over 10,000 tests per sigma (fixed seed).
    std::printf("  A/A false-move rate, budget rule (10,000 tests per sigma, cap 160 pairs):\n");
    for (double sigma : {0.02, 0.05, 0.1}) {
        std::mt19937_64 rng(20261108ULL + (unsigned long long) (sigma * 1000));
        std::normal_distribution<double> nd(0.0, sigma);
        int moves = 0, capped = 0;
        double pairs_sum = 0.0;
        for (int t = 0; t < 10000; ++t) {
            const SimResult r = simulate(budget, [&] { return nd(rng); });
            if (r.v == pt::Verdict::Move) ++moves;
            if (r.v == pt::Verdict::Keep && r.pairs == budget.cap) ++capped;
            pairs_sum += (double) r.pairs;
        }
        const double rate = (double) moves / 10000.0;
        std::printf("    sigma %.2f: false moves %d / 10000 = %.2f%%, capped %d, mean pairs %.1f\n", sigma, moves, 100.0 * rate, capped,
                    pairs_sum / 10000.0);
        char what[160];
        std::snprintf(what, sizeof what, "A/A false-move rate <= 5%% at sigma %.2f (measured %.2f%%)", sigma, 100.0 * rate);
        check(rate <= 0.05, what);
    }

    // (b) power: a true 6% gain (d ~ N(ln 1.06, 0.05)) moves before the cap in >= 80% of tests
    {
        std::mt19937_64 rng(4242);
        std::normal_distribution<double> nd(std::log(1.06), 0.05);
        int moves = 0, keeps = 0;
        double pairs_sum = 0.0;
        for (int t = 0; t < 10000; ++t) {
            const SimResult r = simulate(budget, [&] { return nd(rng); });
            if (r.v == pt::Verdict::Move) ++moves;
            else ++keeps;
            pairs_sum += (double) r.pairs;
        }
        const double power = (double) moves / 10000.0;
        std::printf("    power, true 6%% gain at sigma 0.05: moves %d / 10000 = %.2f%% (keeps %d), mean pairs %.1f\n", moves, 100.0 * power,
                    keeps, pairs_sum / 10000.0);
        char what[160];
        std::snprintf(what, sizeof what, "a true 6%% gain at sigma 0.05 moves in >= 80%% of tests (measured %.2f%%)", 100.0 * power);
        check(power >= 0.80, what);
    }

    // (c) the quality veto
    {
        // a clear flip difference: 2 flips in 1000 tokens (base) vs 30 in 1000 (cand): the candidate's lower bound is above the base's upper
        check(pt::quality_veto(2, 1000, 30, 1000, 200, 3), "quality veto: 30 vs 2 flips per 1000 tokens blocks the candidate");
        // a 1.5x point rate with >= 3 flips, and the interval overlaps (10/10000 base vs 3/1000 cand: cand 0.3% vs 0.1%)
        check(pt::quality_veto(10, 10000, 3, 1000, 200, 3), "quality veto: 3 flips at 3x the base's rate (>= 3 flips) blocks");
        check(!pt::quality_veto(10, 10000, 2, 1000, 200, 3), "quality veto: 2 flips (under 3) at 2x the rate does not block");
        check(!pt::quality_veto(20, 1000, 20, 1000, 200, 3), "quality veto: equal flip rates do not block");
        check(!pt::quality_veto(0, 100, 5, 100, 200, 3), "quality veto: needs >= 200 committed tokens in both arms (none here)");

        // the veto ends a test as a keep even when the candidate is 6% faster (a sequence of fast pairs)
        pt::PairStats s;
        for (int i = 0; i < 40; ++i) s.add(std::log(1.06));
        pt::Reason why = pt::Reason::None;
        const pt::Verdict blocked = pt::decide(s, budget, true, &why);
        check(blocked == pt::Verdict::Keep && why == pt::Reason::Quality, "a faster candidate with the quality veto holding is kept");
        check(pt::decide(s, budget, false) == pt::Verdict::Move, "the same pairs without the veto move (control)");
    }

    // (g) scripted sequences through the state rules
    {
        // budget: pair sequences per test -> (move?) through the DirState and BudgetStep the tuner uses (verify.cpp route_auto_test_done)
        float b = 0.1f;
        pt::DirState dir;
        pt::BudgetStep step;
        auto run_test = [&](const std::vector<double>& ds) -> bool {   // true: move
            pt::PairStats s;
            pt::Verdict v = pt::Verdict::Continue;
            for (double d : ds) {
                s.add(d);
                v = pt::decide(s, budget, false);
                if (v != pt::Verdict::Continue) break;
            }
            if (v == pt::Verdict::Continue) v = pt::Verdict::Keep;   // the tuner keeps when the pairs end without a decision
            const bool won = v == pt::Verdict::Move;
            const int d = dir.dir();
            if (won) b = b * std::pow(step.step, (float) d);
            step.update(won, d);
            dir.update(won, d);
            return won;
        };
        // 1. a clear win in the candidate's direction (+1) moves b up and keeps the direction first
        std::vector<double> up(14, std::log(1.10));
        check(run_test(up) && dir.first == 1 && dir.stage == 0 && b > 0.1f, "scripted: a clear win moves b up, direction +1 stays first");
        // 2. a second clear win the same way grows the step (x1.25: 1.25 -> 1.5625)
        const float s_before = step.step;
        check(run_test(up) && step.step > s_before && step.last_win == 1, "scripted: a second win the same way grows the step");
        // 3. noise: five tests of zero difference, b does not move, the first direction alternates every second test
        const float b_mid = b;
        const int first0 = dir.first;
        std::vector<double> zero(14, 0.0);
        bool any_move = false;
        for (int t = 0; t < 2; ++t) any_move = any_move || run_test(zero);   // stage 0 keep, stage 1 keep: the round ends, first flips
        check(!any_move && b == b_mid && dir.first == -first0 && dir.stage == 0, "scripted: noise keeps b and flips the first direction after both lost");
        // 4. a loss shrinks the step toward 1.1
        const float s_loss = step.step;
        check(!run_test(zero) && step.step < s_loss && step.step >= 1.1f, "scripted: a keep shrinks the step toward 1.1 (never below)");
        // 5. a zero difference decides keep at 12 pairs (no sign of a gain); b does not move
        std::vector<double> flat(300, std::log(1.0));
        check(!run_test(flat) && b == b_mid, "scripted: a zero difference keeps b");
        // 5b. a noisy test that never decides keeps at the cap (alternating +-0.5 pairs: 160 pairs, no decision)
        std::vector<double> noisy;
        for (int i = 0; i < 200; ++i) noisy.push_back((i % 2) ? 0.5 : -0.5);
        check(!run_test(noisy) && b == b_mid, "scripted: an undecided noisy test keeps b at the cap");
        // 6. a quality-blocked candidate: 40 fast pairs, the veto holds: keep, the direction moves on
        {
            pt::PairStats s;
            for (int i = 0; i < 40; ++i) s.add(std::log(1.10));
            const pt::Verdict v = pt::decide(s, budget, true);
            check(v == pt::Verdict::Keep, "scripted: the quality veto keeps a fast candidate (b unchanged by the verdict)");
        }
        std::printf("    scripted budget end: b %.4f, step %.3f, direction %d stage %d\n", (double) b, (double) step.step, dir.first, dir.stage);
    }

    {
        // mask: a single move takes the candidate (no two-wins rule); a second move the same way doubles the step to 2; a keep halves it
        pt::MaskStep ms;
        ms.on_move(1);
        check(ms.step == 1.0 && ms.last_move == 1, "scripted mask: the first move takes one step (no second win needed), last move +1");
        ms.on_move(1);
        check(ms.step == 2.0, "scripted mask: a second move the same way doubles the step");
        ms.on_loss();
        check(ms.step == 1.0, "scripted mask: a keep halves the step (floor 1)");
        ms.on_loss();
        check(ms.step == 1.0, "scripted mask: the step never falls below 1");
        ms.on_move(-1);
        check(ms.step == 1.0 && ms.last_move == -1, "scripted mask: a move in the other direction keeps the step and records -1");
        pt::Rule r = pt::mask_rule(kEffect);
        pt::PairStats s;
        for (int i = 0; i < 12; ++i) s.add(std::log(1.08));
        check(pt::decide(s, r, false) == pt::Verdict::Move, "scripted mask: a clear 8% gain moves the mask at 12 pairs");
    }

    std::printf("paired_test_test: %s (%d failed)\n", g_fail ? "FAIL" : "ok", g_fail);
    return g_fail ? 1 : 0;
}
