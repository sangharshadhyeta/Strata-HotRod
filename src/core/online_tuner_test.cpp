// online_tuner_test.cpp - the online tuner (include/strata/core/online_tuner.hpp) on simulated engines: noisy windows, a drifting machine, knobs
// that help, knobs that do nothing, knobs that help alone and hurt together.  CPU only.
#include "strata/core/online_tuner.hpp"

#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>

using strata::core::OnlineTuner;
using strata::core::TunerConfig;

static int g_fail = 0;
#define CHECK(c, ...)                                                                    \
    do {                                                                                 \
        if (!(c)) {                                                                      \
            ++g_fail;                                                                    \
            std::fprintf(stderr, "FAIL %s:%d: %s  ", __FILE__, __LINE__, #c);            \
            std::fprintf(stderr, __VA_ARGS__);                                           \
            std::fprintf(stderr, "\n");                                                  \
        }                                                                                \
    } while (0)

// a simulated engine: the time of a window is (ms per token at these levels) x tokens x lognormal noise, with a drift over the windows
struct Sim {
    std::function<double(const std::vector<int>&)> mult;   // ms-per-token multiplier for the knobs' level indices (1.0 = the default's speed)
    double cv = 0.25;                                        // per-window noise
    double drift = 0.0;                                      // relative slowdown per 1000 windows
    std::mt19937_64 rng;
    explicit Sim(uint64_t seed) : rng(seed) {}
    void run(OnlineTuner& t, int64_t windows) {
        std::normal_distribution<double> nd(0.0, 1.0);
        for (int64_t w = 0; w < windows; ++w) {
            std::vector<int> idx;
            for (int i = 0; i < t.knob_count(); ++i) idx.push_back(t.level_index(i));
            const int tokens = 1 + (int) (rng() % 4);
            const double per_tok = 100.0 * mult(idx) * (1.0 + drift * (double) w / 1000.0);
            const double noise = std::exp(cv * nd(rng) - 0.5 * cv * cv);
            t.window(per_tok * tokens * noise, tokens);
        }
    }
};

static void one_good_knob() {
    int found = 0, settled = 0, flat_moved = 0;
    const int seeds = 20;
    for (int s = 0; s < seeds; ++s) {
        OnlineTuner t;
        t.add_knob("good", {0, 1, 2, 3, 4}, 1);
        t.add_knob("flat", {0, 1, 2}, 1);
        Sim sim(1000 + (uint64_t) s);
        const double m[5] = {1.05, 1.0, 0.92, 0.88, 0.95};   // the best level is index 3
        sim.mult = [&](const std::vector<int>& i) { return m[i[0]]; };
        sim.run(t, 14000);
        found += m[t.knob(0).cur] <= 0.93;
        settled += t.settled();
        flat_moved += t.knob(1).cur != 1;
    }
    std::printf("one good knob: ended on a level at least 7%% faster in %d/%d seeds, settled in %d, the flat knob moved in %d\n", found, seeds, settled, flat_moved);
    CHECK(found >= 17, "found %d", found);
    CHECK(settled >= 17, "settled %d", settled);
    CHECK(flat_moved <= 3, "flat moved %d", flat_moved);
}

static void nothing_to_find() {
    int adopted = 0;
    const int seeds = 20;
    for (int s = 0; s < seeds; ++s) {
        OnlineTuner t;
        t.add_knob("a", {0, 1, 2}, 1);
        t.add_knob("b", {0, 1, 2}, 1);
        t.add_knob("c", {0, 1}, 0);
        Sim sim(2000 + (uint64_t) s);
        sim.mult = [](const std::vector<int>&) { return 1.0; };
        sim.run(t, 30000);
        adopted += (int) t.adoptions();
    }
    std::printf("nothing to find: %d adoptions over %d seeds x 30000 windows\n", adopted, seeds);
    // an adoption here is an equal-speed level taken by chance (about 1 test in 130 at the 1% level): harmless, and the guard and the next
    // tests walk on; what matters is that it stays rare
    CHECK(adopted <= 14, "false adoptions %d", adopted);
}

static void drift_is_not_a_win() {
    int adopted = 0;
    const int seeds = 20;
    for (int s = 0; s < seeds; ++s) {
        OnlineTuner t;
        t.add_knob("a", {0, 1, 2}, 1);
        t.add_knob("b", {0, 1, 2}, 1);
        Sim sim(3000 + (uint64_t) s);
        sim.drift = 0.03;   // 3% slower every 1000 windows, whatever the setting
        sim.mult = [](const std::vector<int>&) { return 1.0; };
        sim.run(t, 20000);
        adopted += (int) t.adoptions();
    }
    std::printf("drift only: %d adoptions over %d seeds\n", adopted, seeds);
    CHECK(adopted <= 2, "drift adopted %d", adopted);
}

static void helps_alone_hurts_together() {
    int worse_than_default = 0, best_single = 0;
    const int seeds = 20;
    for (int s = 0; s < seeds; ++s) {
        OnlineTuner t;
        t.add_knob("x", {0, 1}, 0);
        t.add_knob("y", {0, 1}, 0);
        Sim sim(4000 + (uint64_t) s);
        const auto table = [](const std::vector<int>& i) { return i[0] && i[1] ? 1.20 : (i[0] || i[1]) ? 0.90 : 1.0; };
        sim.mult = table;
        sim.run(t, 16000);
        const double final_mult = table({t.knob(0).cur, t.knob(1).cur});
        worse_than_default += final_mult > 1.0 + 1e-9;
        best_single += final_mult < 0.95;
    }
    std::printf("helps alone, hurts together: ended worse than the defaults in %d/%d seeds, at a single knob's win in %d\n", worse_than_default, seeds,
                best_single);
    CHECK(worse_than_default == 0, "ended worse than the defaults %d", worse_than_default);
    CHECK(best_single >= 16, "took the single win %d", best_single);
}

static void the_guard_undoes_a_bad_local_optimum() {
    // a loaded state that is a local optimum worse than the defaults: moving ONE knob back makes it worse, so only the guard can find the way out
    int reset = 0;
    const int seeds = 12;
    for (int s = 0; s < seeds; ++s) {
        OnlineTuner t;
        t.add_knob("x", {0, 1}, 0);
        t.add_knob("y", {0, 1}, 0);
        const int n = t.load("knob x 1\nknob y 1\nsettled 1\n");
        CHECK(n == 2, "loaded %d knobs", n);
        CHECK(t.settled(), "a loaded state starts settled");
        Sim sim(5000 + (uint64_t) s);
        const auto table = [](const std::vector<int>& i) { return i[0] && i[1] ? 1.30 : (i[0] || i[1]) ? 1.45 : 1.0; };
        sim.mult = table;
        sim.run(t, 20000);
        reset += t.knob(0).cur == 0 && t.knob(1).cur == 0;
    }
    std::printf("bad local optimum: back at the defaults in %d/%d seeds\n", reset, seeds);
    CHECK(reset >= 10, "reset %d", reset);
}

static void state_round_trip() {
    OnlineTuner a;
    a.add_knob("good", {0, 1, 2, 3}, 1);
    a.add_knob("other", {0.5, 0.6, 0.7}, 0);
    Sim sim(6000);
    sim.mult = [](const std::vector<int>& i) { return i[0] == 3 ? 0.85 : 1.0; };
    sim.run(a, 12000);
    const std::string text = a.save();
    OnlineTuner b;
    b.add_knob("good", {0, 1, 2, 3}, 1);
    b.add_knob("other", {0.5, 0.6, 0.7}, 0);
    const int n = b.load(text);
    std::printf("round trip: saved\n%s", text.c_str());
    CHECK(n == 2, "loaded %d", n);
    CHECK(b.knob(0).cur == a.knob(0).cur && b.knob(1).cur == a.knob(1).cur, "levels differ");
    CHECK(b.settled(), "the loaded tuner starts settled");
    OnlineTuner c;
    c.add_knob("good", {0, 1, 2, 3}, 1);
    CHECK(c.load("garbage\nknob unknown 3\nknob good notanumber\n") == 0, "malformed lines must be ignored");
}

static void burst_settles_fast() {
    int found = 0, fast = 0;
    const int seeds = 12;
    for (int s = 0; s < seeds; ++s) {
        OnlineTuner t;
        t.set_burst(true);
        t.add_knob("good", {0, 1, 2, 3, 4}, 1);
        t.add_knob("flat", {0, 1, 2}, 1);
        Sim sim(7000 + (uint64_t) s);
        const double m[5] = {1.05, 1.0, 0.92, 0.88, 0.95};
        sim.mult = [&](const std::vector<int>& i) { return m[i[0]]; };
        int64_t w = 0;
        while (!t.settled() && w < 12000) { sim.run(t, 100); w += 100; }
        found += m[t.knob(0).cur] <= 0.93;
        fast += t.settled() && w <= 9000;
    }
    std::printf("burst: a level at least 7%% faster in %d/%d seeds, settled within 9000 windows in %d\n", found, seeds, fast);
    CHECK(found >= 9, "burst found %d", found);   // (block 32 in the burst)
    CHECK(fast >= 9, "burst fast %d", fast);
}

static void a_bad_candidate_costs_one_block() {
    // the candidate is 2x slower: the test must end at its first B block, so the windows spent on bad candidates stay a small share
    OnlineTuner t;
    t.add_knob("bad", {0, 1, 2}, 1);
    Sim sim(8000);
    sim.mult = [](const std::vector<int>& i) { return i[0] == 1 ? 1.0 : 2.0; };
    sim.cv = 0.1;
    int64_t bad = 0, total = 0;
    for (int w = 0; w < 6000; ++w) {
        const int idx = t.level_index(0);
        bad += idx != 1;
        ++total;
        sim.run(t, 1);
    }
    std::printf("a 2x slower candidate: %.1f%% of the first 6000 windows ran on it\n", 100.0 * (double) bad / (double) total);
    CHECK((double) bad / (double) total < 0.20, "bad share %.3f", (double) bad / (double) total);
    CHECK(t.knob(0).cur == 1, "must stay at the default");
}

int main() {
    one_good_knob();
    nothing_to_find();
    drift_is_not_a_win();
    helps_alone_hurts_together();
    the_guard_undoes_a_bad_local_optimum();
    state_round_trip();
    burst_settles_fast();
    a_bad_candidate_costs_one_block();
    if (g_fail != 0) {
        std::fprintf(stderr, "online_tuner_test: %d check(s) failed\n", g_fail);
        return 1;
    }
    std::printf("online_tuner_test: all passed\n");
    return 0;
}
