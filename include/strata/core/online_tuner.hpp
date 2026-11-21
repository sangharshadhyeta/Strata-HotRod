// include/strata/core/online_tuner.hpp - the engine settles on a good configuration by itself, while it decodes (HotRod, 21 Nov 2026).
//
// WHY.  One-factor-at-a-time calibration runs chose settings that each won alone on short answers and together decoded about 25% slower, and the
// same setting read 19.1 or 26.3 tok/s depending on whether its arm ran first or last.  Restarting the engine thirty times is also slow (80-90
// minutes).  The settings that matter most can be changed between two decode windows, so the engine can compare them on the real windows it is
// running anyway, and keep what wins.
//
// HOW.  A knob is a setting with a few ordered levels (the swap count, the swap decay, the draft floor, the route tail skip).  The tuner runs
// TESTS.  A test compares the configuration in use (A) with a candidate (B: one knob moved one level, or, for the guard, every knob back at its
// product default) on interleaved blocks of decode windows: A B A B A ... A.  A block's score is milliseconds per accepted token.  Each B block
// is compared with the mean of the two A blocks around it, so a drift in the machine or the workload (the page cache warming, a longer
// context) cancels to first order.  The candidate is adopted only when the mean gain beats min_gain AND a one-sided paired t-test (1%) passes;
// an inconclusive test is extended by more pairs, up to max_pairs; a candidate block that is much slower than its references ends the test at
// once (a bad candidate costs one block, not a test).  Knobs are walked round robin, each up and down, and a knob that just won is pushed on in the
// same direction (hill climbing).  When a whole cycle adopts nothing the configuration is SETTLED: from then on only one test is run every
// settled_gap windows (about 1.5% of the windows), so a change of workload is followed without a constant cost.
// After a cycle in which two or more knobs moved away from their defaults, the GUARD runs: the configuration in use against all defaults.  The
// settings interact, and this is the check that the sum of the adopted choices is no worse than doing nothing; when it is, every knob goes back to
// its default and stays there for burn_windows.
//
// What is learned is saved (save / load) and applied at the next start, which then begins settled and only re-checks.  The calibration run is the same
// tuner driven in a burst (a smaller block, no waiting), so what a calibration writes and what live use writes are one file.
//
// This header is the whole logic and has no engine dependency: online_tuner_test.cpp drives it with simulated engines.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

namespace strata::core {

struct TunerKnob {
    std::string name;
    std::vector<double> levels;   ///< ordered; moving a knob is a step to a neighbouring level
    int def = 0;                  ///< the product default's index
    int cur = 0;                  ///< the level in use
    bool on = true;               ///< false: not tuned (the engine cannot change it here)
    int tried = 0;                ///< bit 0: a step up was rejected since the last move, bit 1: a step down
    int prefer = 0;               ///< +1 / -1: the direction of the last adoption (tried first), 0 none
    int64_t burn_until = 0;       ///< window count before which this knob is not tested again (after a guard reset)
};

struct TunerConfig {
    int block = 48;               ///< decode windows per block
    int min_pairs = 3;            ///< B blocks a test has at least
    int max_pairs = 8;            ///< ... and at most, when it stays inconclusive
    double min_gain = 0.03;       ///< a candidate must be faster than its references by this much (mean over the pairs)
    double abort_loss = 0.25;     ///< a B block this much slower than its reference ends the test as rejected
    int64_t settled_gap = 20000;  ///< windows between two tests once settled
    int64_t burn_windows = 100000;///< a guard reset keeps the knobs at their defaults this long
    int warmup = 64;              ///< windows ignored at the start (the first windows after a start are not the steady state)
};

class OnlineTuner {
public:
    explicit OnlineTuner(TunerConfig c = TunerConfig()) : c_(c) {}

    /// Registers a knob (before the first window).  Returns its index.
    int add_knob(const std::string& name, std::vector<double> levels, int def, bool on = true) {
        TunerKnob k;
        k.name = name;
        k.levels = std::move(levels);
        k.def = std::clamp(def, 0, std::max(0, (int) k.levels.size() - 1));
        k.cur = k.def;
        k.on = on && k.levels.size() > 1;
        knobs_.push_back(std::move(k));
        return (int) knobs_.size() - 1;
    }

    int knob_count() const { return (int) knobs_.size(); }
    const TunerKnob& knob(int i) const { return knobs_[(size_t) i]; }
    bool settled() const { return settled_; }
    int64_t windows() const { return total_; }
    int64_t tests() const { return tests_; }
    int64_t adoptions() const { return adoptions_; }
    int64_t resets() const { return resets_; }
    bool testing() const { return test_.active; }

    /// The level the engine should use NOW for knob `i`: the candidate's during a B block, else the one in use.
    double value(int i) const { return knobs_[(size_t) i].levels[(size_t) level_index(i)]; }
    int level_index(int i) const {
        const TunerKnob& k = knobs_[(size_t) i];
        if (test_.active && test_.in_b) {
            if (test_.guard) return k.def;
            if (test_.knob == i) return test_.level;
        }
        return k.cur;
    }

    /// One finished decode window: its time and the tokens it accepted.  Call it for steady-state decode windows only (not the first window of a
    /// request, not a window that carried prompt rows).
    void window(double ms, int tokens) {
        if (knobs_.empty() || !(ms > 0.0) || tokens <= 0) return;
        ++total_;
        if (total_ <= c_.warmup) return;
        blk_ms_ += ms;
        blk_tok_ += tokens;
        if (++blk_n_ < c_.block) return;
        const double score = blk_ms_ / (double) blk_tok_;
        blk_ms_ = 0.0;
        blk_tok_ = 0;
        blk_n_ = 0;
        on_block(score);
    }

    /// Messages for the log (adoptions, rejections of note, guard resets); the caller prints and clears them.
    std::vector<std::string> take_events() { std::vector<std::string> e; e.swap(events_); return e; }

    /// What was learned, as text: "knob <name> <level>" per knob and "settled 0|1".
    std::string save() const {
        std::ostringstream o;
        for (const TunerKnob& k : knobs_) o << "knob " << k.name << " " << k.levels[(size_t) k.cur] << "\n";
        o << "settled " << (settled_ ? 1 : 0) << "\n";
        return o.str();
    }

    /// Puts a saved state back: each known knob takes its nearest level, and the tuner starts settled (only maintenance tests) when the state
    /// names every tuned knob.  Unknown names and malformed lines are ignored.  Returns how many knobs were set.
    int load(const std::string& text) {
        std::istringstream in(text);
        std::string line;
        int set = 0;
        while (std::getline(in, line)) {
            std::istringstream ls(line);
            std::string tag, name;
            double v = 0.0;
            if (!(ls >> tag >> name) || tag != "knob" || !(ls >> v)) continue;
            for (TunerKnob& k : knobs_) {
                if (k.name != name || !k.on) continue;
                int best = k.def;
                double bd = 1e300;
                for (int i = 0; i < (int) k.levels.size(); ++i)
                    if (std::fabs(k.levels[(size_t) i] - v) < bd) { bd = std::fabs(k.levels[(size_t) i] - v); best = i; }
                k.cur = best;
                ++set;
            }
        }
        int tuned = 0;
        for (const TunerKnob& k : knobs_) tuned += k.on ? 1 : 0;
        if (set > 0 && set >= tuned) {
            settled_ = true;
            next_test_at_ = total_ + c_.settled_gap / 4;   // a loaded state is re-checked sooner than a settled one
            guard_pending_ = moved_knobs() >= 2;           // two or more knobs away from their defaults: the first check is the guard
        }
        return set;
    }

    /// The calibration burst: no waiting between tests and smaller blocks, so a few thousand windows settle the knobs.
    void set_burst(bool on) {
        burst_ = on;
        if (on) { c_.block = std::min(c_.block, 32); c_.warmup = std::min(c_.warmup, 16); }
    }

private:
    struct Test {
        bool active = false;
        bool guard = false;
        int knob = -1;
        int level = 0;
        int dir = 0;
        bool in_b = false;                 // the block being measured is a B block
        int n_pairs = 0;                   // B blocks this test will have (grows when inconclusive)
        std::vector<double> a, b;          // block scores, in order
    };

    static double t_crit(int df) {         // one-sided 1%
        static const double t[] = {0.0, 0.0, 6.965, 4.541, 3.747, 3.365, 3.143, 2.998, 2.896, 2.821, 2.764};
        return df >= 10 ? 2.6 : (df < 2 ? 1e9 : t[df]);
    }

    int active_knobs() const { int n = 0; for (const TunerKnob& k : knobs_) n += k.on ? 1 : 0; return n; }
    int moved_knobs() const { int n = 0; for (const TunerKnob& k : knobs_) n += (k.on && k.cur != k.def) ? 1 : 0; return n; }

    void note(const std::string& s) { events_.push_back(s); }

    // the next test: round robin over the knobs, each in the directions not yet rejected since its last move
    bool start_move_test() {
        const int nk = (int) knobs_.size();
        for (int spin = 0; spin < nk * 2 + 1; ++spin) {
            TunerKnob& k = knobs_[(size_t) cursor_];
            if (k.on && total_ >= k.burn_until) {
                const int dirs[2] = {k.prefer != 0 ? k.prefer : 1, k.prefer != 0 ? -k.prefer : -1};
                for (int d : dirs) {
                    const int bit = d > 0 ? 1 : 2;
                    const int to = k.cur + d;
                    if ((k.tried & bit) || to < 0 || to >= (int) k.levels.size()) continue;
                    begin(false, cursor_, to, d);
                    return true;
                }
            }
            k.tried = 0;                                   // both directions done for this round: the next round starts clean
            k.prefer = 0;
            cursor_ = (cursor_ + 1) % nk;
            ++walked_;
            if (walked_ >= nk) {                           // a whole cycle without a test to start: nothing left to try
                walked_ = 0;
                return false;
            }
        }
        return false;
    }

    void begin(bool guard, int knob, int level, int dir) {
        test_ = Test();
        test_.active = true;
        test_.guard = guard;
        test_.knob = knob;
        test_.level = level;
        test_.dir = dir;
        test_.in_b = false;
        test_.n_pairs = c_.min_pairs;
        ++tests_;
        if (settled_) next_test_at_ = total_ + c_.settled_gap;   // once settled, one test per gap
    }

    void finish_test(bool adopt) {
        const Test t = test_;
        test_ = Test();
        if (t.guard) {
            if (adopt) {
                for (TunerKnob& k : knobs_) { k.cur = k.def; k.tried = 0; k.prefer = 0; k.burn_until = total_ + c_.burn_windows; }
                ++resets_;
                note("online tuner: the adopted settings together are slower than the defaults - every knob is back at its default");
            }
            return;
        }
        TunerKnob& k = knobs_[(size_t) t.knob];
        if (adopt) {
            const double was = k.levels[(size_t) k.cur];
            k.cur = t.level;
            k.tried = t.dir > 0 ? 2 : 1;                   // the way back is the level this just beat: not worth a test
            k.prefer = t.dir;
            ++adoptions_;
            cycle_adopted_ = true;
            settled_ = false;
            char buf[200];
            std::snprintf(buf, sizeof buf, "online tuner: %s %g -> %g (%+.1f%% over %d pairs)", k.name.c_str(), was, k.levels[(size_t) k.cur],
                          last_gain_ * 100.0, last_pairs_);
            note(buf);
        } else {
            k.tried |= t.dir > 0 ? 1 : 2;
        }
    }

    // a block of the test (or, when nothing is running, the idle bookkeeping) just finished
    void on_block(double score) {
        if (!test_.active) {
            // idle: decide whether to start a test
            if (settled_ && !burst_ && total_ < next_test_at_) return;
            if (active_knobs() == 0) return;
            if (guard_pending_) {                           // after a cycle that adopted something: is the sum of the choices any good?
                guard_pending_ = false;
                if (moved_knobs() >= 2) { begin(true, -1, 0, 0); return; }
            }
            if (start_move_test()) return;
            // every knob has been tried in both directions: the cycle is over
            if (cycle_adopted_) { cycle_adopted_ = false; guard_pending_ = true; return; }   // something moved: guard it, then another cycle
            if (!settled_) {
                settled_ = true;
                note("online tuner: settled");
            }
            next_test_at_ = total_ + c_.settled_gap;
            return;
        }
        // a test is running: this block belongs to A (even position) or B (odd)
        (test_.in_b ? test_.b : test_.a).push_back(score);
        const bool was_b = test_.in_b;
        test_.in_b = !test_.in_b;
        if (was_b) {
            const size_t i = test_.b.size() - 1;
            const double ref = test_.a.size() > i ? test_.a[i] : test_.a.back();
            if (ref > 0.0 && score > ref * (1.0 + c_.abort_loss)) { finish_test(false); return; }   // a bad candidate costs one block
            return;
        }
        // an A block finished: the test can be judged once every B block has both of its neighbours
        if ((int) test_.a.size() < test_.n_pairs + 1) return;
        std::vector<double> g;
        for (int i = 0; i < test_.n_pairs; ++i) {
            const double ref = 0.5 * (test_.a[(size_t) i] + test_.a[(size_t) i + 1]);
            g.push_back(ref > 0.0 ? (ref - test_.b[(size_t) i]) / ref : 0.0);
        }
        double mean = 0.0;
        for (double x : g) mean += x;
        mean /= (double) g.size();
        double var = 0.0;
        for (double x : g) var += (x - mean) * (x - mean);
        var /= (double) std::max<size_t>(1, g.size() - 1);
        const double se = std::sqrt(var / (double) g.size());
        const double t = se > 0.0 ? mean / se : (mean > 0.0 ? 1e9 : 0.0);
        last_gain_ = mean;
        last_pairs_ = (int) g.size();
        // adoption: the mean gain beats min_gain, the paired t-test passes, AND the candidate was faster in (nearly) every pair - at most one
        // pair in six may be slower.  Extending an inconclusive test inflates the error rate of the t-test alone; the sign rule keeps the
        // false adoptions of a flat knob rare (found on the simulated engines: 4 in 20 runs of 30000 windows with the t-test alone).
        int slower = 0;
        for (double x : g) slower += x <= 0.0;
        const bool win = mean > c_.min_gain && t > t_crit((int) g.size() - 1) && slower <= (int) g.size() / 6;
        if (win) { finish_test(true); return; }
        if (mean > c_.min_gain * 0.5 && test_.n_pairs < c_.max_pairs) { ++test_.n_pairs; return; }   // inconclusive: more pairs
        finish_test(false);
    }

    TunerConfig c_;
    std::vector<TunerKnob> knobs_;
    Test test_;
    std::vector<std::string> events_;
    double blk_ms_ = 0.0;
    int64_t blk_tok_ = 0;
    int blk_n_ = 0;
    int64_t total_ = 0, tests_ = 0, adoptions_ = 0, resets_ = 0;
    int64_t next_test_at_ = 0;
    int cursor_ = 0, walked_ = 0;
    bool settled_ = false, burst_ = false, cycle_adopted_ = false, guard_pending_ = false;
    double last_gain_ = 0.0;
    int last_pairs_ = 0;
};

}  // namespace strata::core
