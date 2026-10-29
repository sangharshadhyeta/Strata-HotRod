// Strata. src/dsv41/ds4_standin_tune.hpp - the DeepSeek engine's stand-in budget tuner (host logic, no CUDA).
//
// The rules are the Qwen engine's (src/core/verify.cpp: STRATA_ROUTE_BUDGET_AUTO, STRATA_FRAGILE_BRAKE, the fragile_k learning,
// the standin-tune.txt file), ported to one token per decode step.  Left out on purpose: the per-size-class learning, the control
// ring (the noise floor), the explore cuts, the rollback and the DYN budget.
//
// The engine asks budget_for_next_token() before each decode token, runs the token at that budget, and reports it with
// end_token(): its wall seconds, whether stand-ins were taken and, when the token was re-checked, its top-two logit gap, its risk
// and whether the top token flipped.  A re-check happens when gap < k * risk (k: this header's k()).
//
// Two loops run on top of each other:
//  - The AUTO search keeps ONE base budget b.  An INTERLEAVED test runs two arms over the same stretch of decode, so content drift
//    cancels: the base arm at req_b and the candidate arm at req_b * s^d (s: the step, 1.25 at the start; d: the direction, +1 up
//    or -1 down).  Runs of kAutoRun tokens alternate between the arms.  A test ends when both arms have kAutoMin tokens.  The
//    candidate wins at >= 1.02 times the base's speed (tokens per second); then b moves to it.
//  - The BRAKE gives each request its own budget req_b, which the arms scale.  Re-check flips in a request cut req_b (Wilson 90%
//    bounds decide, so a few flips do not act); clean stretches grow it back.
// k (the close-call factor) learns from the flips of every 50 re-checks.
//
// State that outlives a process is kept in a text file (save / load): lines "budget", "k", "step", "clean", "tests", "lastwin".
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

namespace strata::ds4 {

class StandinTune {
public:
    // b0: the start of the base budget (also the ceiling's base while nothing is learned); k0: the start of k (k_pinned: k never
    // learns or updates); brake_on: the per-request brake (off: every request runs at b).  Call once, before the first request.
    void configure(float b0, float k0, bool k_pinned, bool brake_on) {
        b0_ = clamp_b((double) b0);
        b_ = b0_;
        k_ = clamp_k((double) k0);
        k_pinned_ = k_pinned;
        brake_on_ = brake_on;
        req_b_ = brake_on_ ? 0.5f * b_ : b_;
    }

    // A request starts: its budget is half the base budget (the class start of the Qwen engine) when the brake is on, else b.
    void begin_request() {
        req_b_ = brake_on_ ? 0.5f * b_ : b_;
        req_rc_ = req_fl_ = req_tok_ = req_wins_ = 0;
    }

    // The budget the next decode token runs at.  Chooses its arm (a scheduled run of the running test, or the base arm when no test
    // can start); the brake's grow check runs every kBrakeWindows tokens of the request.
    float budget_for_next_token() {
        if (brake_on_ && ++req_wins_ % kBrakeWindows == 0) grow_check();
        schedule();
        return arm_budget();
    }

    float k() const { return k_; }
    float b() const { return b_; }           ///< the base budget (the search's b)
    float req_b() const { return req_b_; }   ///< this request's budget (the base arm's)

    // After a decode token.  secs: its wall time, a re-run included.  rechecked: stand-ins were taken and the gap test sent it to a
    // re-run; gap / risk: its top-two logit gap and risk (used when rechecked); flipped: the re-run's top token differs.
    void end_token(double secs, bool rechecked, float gap, float risk, bool flipped) {
        const int a = arm_;
        if (testing_) {   // an arm's time and tokens: the test's own tokens only
            secs_[a] += secs;
            ++tok_[a];
            ++n_[a];
        }
        ++req_tok_;
        if (rechecked) {
            ++req_rc_;
            if (flipped) ++req_fl_;
            ++rc_[a];
            if (flipped) ++fl_[a];
            // the k ring: this re-check's ratio and flip, the last kKWindow re-checks
            ring_ratio_[ring_pos_] = risk > 0.0f && std::isfinite(gap / risk) ? gap / risk : 0.0f;
            ring_flip_[ring_pos_] = flipped;
            ring_pos_ = (ring_pos_ + 1) % kKWindow;
            if (++rc_total_ % kKWindow == 0) k_update();
            brake_check();
        }
        if (testing_ && n_[0] >= kAutoMin && n_[1] >= kAutoMin) test_done();
    }

    // Writes the state to path (through a temp file, then renamed).  False when the file cannot be written.
    bool save(const std::string& path) const {
        const std::string tmp = path + ".tmp";
        FILE* f = std::fopen(tmp.c_str(), "w");
        if (f == nullptr) return false;
        bool ok = std::fprintf(f, "budget %.9g\n", (double) b_) > 0;
        ok = ok && std::fprintf(f, "k %.9g\n", (double) k_) > 0;
        ok = ok && std::fprintf(f, "step %.9g\n", (double) step_) > 0;
        ok = ok && std::fprintf(f, "clean %.9g\n", b_clean_) > 0;
        ok = ok && std::fprintf(f, "tests %lld\n", tests_) > 0;
        ok = ok && std::fprintf(f, "lastwin %d\n", last_win_) > 0;
        ok = std::fclose(f) == 0 && ok;
        if (ok && std::rename(tmp.c_str(), path.c_str()) == 0) return true;
        std::remove(tmp.c_str());
        return false;
    }

    // Reads the state from path.  False when the file is missing or has no budget line.  b and clean, step, tests and lastwin are
    // taken as saved; k is taken unless k is pinned.  Call before the first request.
    bool load(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "r");
        if (f == nullptr) return false;
        char line[256];
        double budget = -1.0, k = -1.0, step = -1.0, clean = -1.0, tests = -1.0, lastwin = 2.0;
        while (std::fgets(line, sizeof line, f) != nullptr) {
            char key[32] = "";
            double v = 0.0;
            if (std::sscanf(line, "%31s %lf", key, &v) != 2) continue;
            if (std::strcmp(key, "budget") == 0) budget = v;
            else if (std::strcmp(key, "k") == 0) k = v;
            else if (std::strcmp(key, "step") == 0) step = v;
            else if (std::strcmp(key, "clean") == 0) clean = v;
            else if (std::strcmp(key, "tests") == 0) tests = v;
            else if (std::strcmp(key, "lastwin") == 0) lastwin = v;
        }
        std::fclose(f);
        if (budget < 0.0) return false;
        b_ = clamp_b(budget);
        req_b_ = brake_on_ ? 0.5f * b_ : b_;
        if (k > 0.0 && !k_pinned_) k_ = clamp_k(k);
        if (step > 0.0) step_ = std::min(2.0f, std::max(1.1f, (float) step));
        if (clean >= 0.0) b_clean_ = clean;
        if (tests >= 0.0) tests_ = (long long) tests;
        if (lastwin == -1.0 || lastwin == 0.0 || lastwin == 1.0) last_win_ = (int) lastwin;
        return true;
    }

    // One line: b, k, req_b, tests, wins, brakes.
    std::string stats() const {
        char buf[256];
        std::snprintf(buf, sizeof buf, "b %.3f k %.3f req_b %.3f tests %lld wins %lld brakes %lld", (double) b_, (double) k_,
                      (double) req_b_, tests_, wins_, brakes_);
        return buf;
    }

private:
    // (kAutoMin counts tokens here: our Qwen engine's 96 were decode windows of several tokens each.  At 96 tokens an arm never reaches
    // kAutoTokMin or kCleanTok, so the quality gate and the clean ceiling would never act; 400 lets both act as they do there.)
    enum { kAutoRun = 8, kAutoMin = 400, kAutoTokMin = 200, kCleanTok = 400 };
    enum { kBrakeMin = 15, kBrakeWindows = 64, kBrakeTokMin = 100, kBrakeFlips = 3, kKWindow = 50 };
    static constexpr double kZ = 1.6448536;   // the Wilson score's z for 90%

    // ---- the search state ----
    float b0_ = 0.1f;                  ///< the start of b: the ceiling's base while no clean level is learned
    float b_ = 0.1f;                   ///< the base budget (the search's b)
    float step_ = 1.25f;               ///< the candidate's step factor s (after a win in the same direction it grows to 2; after a loss it shrinks toward 1.1)
    int last_win_ = 0;                 ///< the direction of the last win (0: none yet)
    int first_ = 1;                    ///< the direction the next round tests first
    int stage_ = 0;                    ///< 0: the round tests first_; 1: the other direction against the same base
    double b_clean_ = 0.0;             ///< the learned ceiling: the largest candidate of a clean test (0: none yet)
    long long tests_ = 0, wins_ = 0, brakes_ = 0;   ///< tests finished, candidates that won, brakes (process total)

    // ---- the running test ----
    bool testing_ = false;             ///< a test is running
    float cand_b_ = 0.1f;              ///< its candidate's search budget (b * s^d, clamped)
    int dir_ = 1;                      ///< its direction d
    long long n_[2] = {0, 0};          ///< its scheduled tokens per arm (0 base, 1 candidate)
    long long tok_[2] = {0, 0};        ///< its tokens per arm
    double secs_[2] = {0.0, 0.0};      ///< its wall seconds per arm
    long long rc_[2] = {0, 0};         ///< its re-checks per arm
    long long fl_[2] = {0, 0};         ///< its flips per arm
    long long runs_ = 0;               ///< its runs started
    int run_left_ = 0;                 ///< tokens left in the current run
    int run_arm_ = 0;                  ///< the current run's arm
    int arm_ = 0;                      ///< the arm of the token being run (set by budget_for_next_token)

    // ---- the brake (per request) ----
    bool brake_on_ = true;
    float req_b_ = 0.05f;              ///< this request's budget
    long long req_rc_ = 0, req_fl_ = 0;   ///< this request's re-checks and flips since its last brake
    long long req_tok_ = 0;            ///< this request's tokens since its last brake
    long long req_wins_ = 0;           ///< this request's tokens (the grow check every kBrakeWindows)

    // ---- k ----
    float k_ = 4.0f;
    bool k_pinned_ = false;
    float ring_ratio_[kKWindow] = {};  ///< the last kKWindow re-checks: gap / risk
    bool ring_flip_[kKWindow] = {};    ///< and whether each flipped
    int ring_pos_ = 0;
    long long rc_total_ = 0;           ///< re-checks in this process (the k update every kKWindow)

    static double clamp_d(double v, double lo, double hi) { return std::min(hi, std::max(lo, v)); }
    static float clamp_b(double b) { return (float) clamp_d(b, 0.02, 4.0); }
    static float clamp_k(double k) { return (float) clamp_d(k, 0.005, 50.0); }

    // Wilson score interval at 90% for f flips in n: the flip rate's lower and upper bounds (0 and 1 when n is 0).
    static void wilson90(long long f, long long n, double& lo, double& hi) {
        if (n <= 0) {
            lo = 0.0;
            hi = 1.0;
            return;
        }
        const double z2 = kZ * kZ, nn = (double) n, p = (double) f / nn;
        const double centre = p + z2 / (2.0 * nn);
        const double half = kZ * std::sqrt(p * (1.0 - p) / nn + z2 / (4.0 * nn * nn));
        const double den = 1.0 + z2 / nn;
        lo = (centre - half) / den;
        hi = (centre + half) / den;
    }

    // The budget of the arm of the token being run: the base arm runs at req_b, the candidate at req_b * s^d (clamped; 0 stays 0).
    float arm_budget() const {
        if (arm_ == 0 || req_b_ <= 0.0f) return req_b_;
        return clamp_b((double) req_b_ * std::pow((double) step_, (double) dir_));
    }

    // Starts a test of direction d at the base b: the arms' counts and times are cleared.
    void begin_test(int d) {
        dir_ = d;
        cand_b_ = clamp_b((double) b_ * std::pow((double) step_, (double) d));
        n_[0] = n_[1] = tok_[0] = tok_[1] = rc_[0] = rc_[1] = fl_[0] = fl_[1] = 0;
        secs_[0] = secs_[1] = 0.0;
        runs_ = 0;
        run_left_ = 0;
        testing_ = true;
    }

    // The next direction: a win moves b to the candidate and tests that direction first next round; a loss tests the other direction
    // against the same base (stage 1); a second loss keeps b and tests the other direction first next round.
    void next_stage(bool won, int d) {
        if (won) {
            b_ = cand_b_;
            first_ = d;
            stage_ = 0;
        } else if (stage_ == 0) {
            stage_ = 1;
        } else {
            first_ = -first_;
            stage_ = 0;
        }
    }

    // Chooses the arm of the next token.  When no test is running, a test starts in a direction that can be tested: a candidate that
    // clamps to the base is not tested, nor is one above the ceiling (2 * b_clean, or 2 * b0 while none is learned).  Each call tries
    // both directions at most once; when neither can be tested the token runs the base arm unscheduled.
    void schedule() {
        for (int tries = 0; !testing_ && tries < 2; ++tries) {
            const int d = stage_ == 0 ? first_ : -first_;
            const double cand = clamp_d((double) b_ * std::pow((double) step_, (double) d), 0.02, 4.0);
            const double ceiling = b_clean_ > 0.0 ? 2.0 * b_clean_ : 2.0 * (double) b0_;
            if (cand != (double) b_ && !(cand > (double) b_ && cand > ceiling))
                begin_test(d);
            else
                next_stage(false, d);   // clamped to the base, or above the ceiling: a loss without a test
        }
        if (!testing_) {
            arm_ = 0;
            return;
        }
        if (run_left_ == 0) {   // runs of kAutoRun tokens; run r is arm (r % 2) ^ ((r / 2) % 2), so each pair of runs alternates which arm is first
            run_arm_ = (int) ((runs_ % 2) ^ ((runs_ / 2) % 2));
            ++runs_;
            run_left_ = kAutoRun;
        }
        --run_left_;
        arm_ = run_arm_;
    }

    // A grow check every kBrakeWindows tokens of the request: with >= kBrakeMin re-checks and the Wilson 90% upper bounds of its flip
    // rate per re-check (<= 0.2) and of its flips per token (<= 0.01) the budget grows: x 1.25, never above b; from 0 to 0.25 b.
    // The first check of a request (token kBrakeWindows) sets req_b = b at once when the bounds hold.
    void grow_check() {
        if (req_rc_ < kBrakeMin) return;
        double lo = 0.0, hi = 0.0, tok_lo = 0.0, tok_hi = 0.0;
        wilson90(req_fl_, req_rc_, lo, hi);
        wilson90(req_fl_, req_tok_, tok_lo, tok_hi);
        if (hi > 0.2 || tok_hi > 0.01) return;
        if (req_wins_ == kBrakeWindows) {
            req_b_ = b_;
            return;
        }
        req_b_ = req_b_ <= 0.0f ? 0.25f * b_ : std::min(b_, req_b_ * 1.25f);
    }

    // After a re-check: the request brakes when (>= kBrakeMin re-checks and the Wilson 90% lower bound of its flip rate per re-check is
    // above 0.2) or (>= kBrakeFlips flips, >= kBrakeTokMin tokens and the lower bound of flips per token is above 0.01).  The budget
    // halves (below 0.01 it is 0) and the request's counts reset.
    void brake_check() {
        if (!brake_on_ || req_b_ <= 0.0f) return;
        double lo = 0.0, hi = 0.0, tok_lo = 0.0, tok_hi = 0.0;
        wilson90(req_fl_, req_rc_, lo, hi);
        wilson90(req_fl_, req_tok_, tok_lo, tok_hi);
        const bool by_recheck = req_rc_ >= kBrakeMin && lo > 0.2;
        const bool by_token = req_tok_ >= kBrakeTokMin && req_fl_ >= kBrakeFlips && tok_lo > 0.01;
        if (!by_recheck && !by_token) return;
        const float cut = req_b_ * 0.5f;
        req_b_ = cut < 0.01f ? 0.0f : cut;
        ++brakes_;
        req_rc_ = req_fl_ = req_tok_ = 0;
    }

    // Every kKWindow re-checks: when any of the last kKWindow flipped, k = clamp(1.25 * the largest flipped gap/risk ratio, 0.5 k,
    // 1.5 k).  No flip: k is kept.
    void k_update() {
        if (k_pinned_) return;
        bool any = false;
        float largest = 0.0f;
        for (int i = 0; i < kKWindow; ++i)
            if (ring_flip_[i]) {
                any = true;
                largest = std::max(largest, ring_ratio_[i]);
            }
        if (!any) return;
        const float old = k_;
        const float target = std::min(1.5f * old, std::max(0.5f * old, 1.25f * largest));
        k_ = clamp_k((double) target);
    }

    // The test's end (both arms have kAutoMin tokens): the candidate wins at >= 1.02 times the base's speed, unless its quality is lost.
    // Quality (Wilson 90% on flips per token, once both arms have kAutoTokMin tokens): lost when the candidate's lower bound is above
    // the base's upper bound, or its rate is above 1.5 times the base's with >= kBrakeFlips flips.  A candidate with <= 0.5 flips per
    // 100 tokens over >= kCleanTok tokens raises the learned ceiling b_clean.  The step: a win in the same direction as the last win
    // grows it (x 1.25, to 2); a loss shrinks it toward 1.1 (square root, at least 1.1).
    void test_done() {
        const int d = dir_;
        const double sb = secs_[0] > 0.0 ? (double) tok_[0] / secs_[0] : 0.0;
        const double sc = secs_[1] > 0.0 ? (double) tok_[1] / secs_[1] : 0.0;
        const double pb = tok_[0] > 0 ? (double) fl_[0] / (double) tok_[0] : 0.0;
        const double pc = tok_[1] > 0 ? (double) fl_[1] / (double) tok_[1] : 0.0;
        double base_lo = 0.0, base_hi = 0.0, cand_lo = 0.0, cand_hi = 0.0;
        wilson90(fl_[0], tok_[0], base_lo, base_hi);
        wilson90(fl_[1], tok_[1], cand_lo, cand_hi);
        const bool enough = tok_[0] >= kAutoTokMin && tok_[1] >= kAutoTokMin;
        const bool quality = !(enough && (cand_lo > base_hi || (pc > 1.5 * pb && fl_[1] >= kBrakeFlips)));
        if (tok_[1] >= kCleanTok && 100.0 * (double) fl_[1] <= 0.5 * (double) tok_[1] && (double) cand_b_ > b_clean_)
            b_clean_ = (double) cand_b_;
        const bool won = sc >= sb * 1.02 && quality;
        if (won) {
            if (d == last_win_) step_ = std::min(step_ * 1.25f, 2.0f);
            last_win_ = d;
            ++wins_;
        } else {
            step_ = std::max((float) std::sqrt((double) step_), 1.1f);
        }
        next_stage(won, d);
        testing_ = false;
        ++tests_;
    }
};

}  // namespace strata::ds4
