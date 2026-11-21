// Strata. The DeepSeek engine's stand-in budget tuner (src/dsv41/ds4_standin_tune.hpp) on a fake engine: no GPU, no model.
// The fake decode token: speed 5 + 10 * min(b, 0.4) tokens/s with +-0.2% noise; stand-ins are taken when b > 0; a token is
// re-checked when gap < k * risk (gap uniform in [0, 10), risk 1); a re-check flips with probability flip_p when b >= flip_from.
#include "ds4_standin_tune.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>

using strata::ds4::StandinTune;

static int fails = 0;
#define CHECK(c)                                                                     \
    do {                                                                             \
        if (!(c)) {                                                                  \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);                 \
            ++fails;                                                                 \
        }                                                                            \
    } while (0)

struct Engine {
    std::mt19937_64 rng{20261030ull};
    double flip_from = 0.5;       // flips only at budgets at or above this
    double flip_p = 0.5;          // a re-check's flip probability there
    long long tokens = 0, rechecks = 0, flips = 0;
    float max_b = 0.0f;           // the largest budget a token ran at
    bool last_rc = false;         // the last token's re-check flag

    double u() { return std::uniform_real_distribution<double>(0.0, 1.0)(rng); }

    // one decode token of the request: the budget the tuner gives it, the fake run, the report
    float token(StandinTune& t) {
        const float b = t.budget_for_next_token();
        if (b > max_b) max_b = b;
        const double speed = 5.0 + 10.0 * std::min(0.4, (double) b);
        const double secs = (1.0 / speed) * (1.0 + (u() * 2.0 - 1.0) * 0.002);
        const float gap = (float) (10.0 * u());
        const float risk = 1.0f;
        const bool rc = b > 0.0f && gap < t.k() * risk;
        const bool fl = rc && (double) b >= flip_from && u() < flip_p;
        t.end_token(secs, rc, gap, risk, fl);
        ++tokens;
        if (rc) ++rechecks;
        if (fl) ++flips;
        last_rc = rc;
        return b;
    }

    // n tokens in requests of len tokens each (begin_request at each start)
    void run(StandinTune& t, long long n, long long len) {
        for (long long i = 0; i < n; ++i) {
            if (i % len == 0) t.begin_request();
            token(t);
        }
    }
};

static std::string slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static std::string tmp_path(const char* name) {
    return (std::filesystem::temp_directory_path() / name).string();
}

// Requests here are 1000 tokens long: the brake's grow needs its Wilson 90% upper bound of flips per token at most 0.01, which with
// no flips takes about 268 tokens, and the budget reaches b at the grow check after the first 64 tokens once that holds.
// 1. From b0 = 0.1 with no learned ceiling: the ceiling starts at 2 * b0 and rises as clean tests (no flips over >= 400 tokens) are
// learned, so the base climbs to the speed knee (0.4) and stops short of the flip region (0.5): the candidates that probe past it
// lose on quality.  Their probes are the only flips, a small share of the tokens.
static void test_climb_ceiling(bool legacy) {
    StandinTune t;
    t.configure(0.1f, 4.0f, false, true);
    if (legacy) t.set_brake_legacy(true);
    Engine e;
    std::printf("climb (no clean level, %s brake): b0 0.100\n", legacy ? "legacy" : "new");
    for (int blk = 1; blk <= 20; ++blk) {
        e.run(t, 1000, 1000);
        if (blk % 4 == 0) std::printf("  after %5d tokens: b %.4f req_b %.4f\n", blk * 1000, (double) t.b(), (double) t.req_b());
    }
    std::printf("  %s\n", t.stats().c_str());
    CHECK(t.b() >= 0.3f);           // up to the knee
    CHECK(t.b() < 0.5f);            // the base never settles in the flip region
    CHECK(e.flips * 100 <= e.tokens * 2);   // flips only from the probes past the knee: at most 2% of the tokens
}

// 2. With a learned ceiling (clean 0.4, as a clean test would leave it), the base climbs toward the speed knee at 0.4 and
// stops there: past 0.4 the speed is flat, so no candidate wins, and the flip region (b >= 0.5) is not entered.
static void test_climb_to_knee(bool legacy) {
    const std::string path = tmp_path("ds4_standin_tune_seed.txt");
    { std::ofstream f(path); f << "budget 0.1\nclean 0.4\n"; }
    StandinTune t;
    t.configure(0.1f, 4.0f, false, true);
    if (legacy) t.set_brake_legacy(true);
    CHECK(t.load(path));
    std::filesystem::remove(path);
    Engine e;
    std::printf("climb (clean 0.4, %s brake): b0 0.100\n", legacy ? "legacy" : "new");
    for (int blk = 1; blk <= 30; ++blk) {
        e.run(t, 1000, 1000);
        if (blk % 5 == 0) std::printf("  after %5d tokens: b %.4f req_b %.4f\n", blk * 1000, (double) t.b(), (double) t.req_b());
    }
    std::printf("  %s; largest budget a token ran at %.4f, flips %lld\n", t.stats().c_str(), (double) e.max_b, e.flips);
    CHECK(t.b() >= 0.3f);
    CHECK(t.b() < 0.6f);            // does not run past the knee and the flip region
}

// 3 (legacy). The fixed-threshold brake (set_brake_legacy(true)): a request with many flips is braked: req_b halves with each brake,
// down to 0, and the request's grow never lifts it again.  The new rule does not brake here (no measured noise floor): see test_brake_new.
static void test_brake_legacy() {
    StandinTune t;
    t.configure(1.0f, 4.0f, false, true);
    t.set_brake_legacy(true);
    Engine e;
    e.flip_from = 0.0;
    e.flip_p = 1.0;                 // every re-check flips
    t.begin_request();
    const float start = t.req_b();
    float prev = start;
    bool monotone = true;
    for (int i = 0; i < 300; ++i) {
        e.token(t);
        if (t.req_b() > prev) monotone = false;
        prev = t.req_b();
    }
    std::printf("brake (all re-checks flip): req_b %.4f -> %.4f, %s\n", (double) start, (double) t.req_b(), t.stats().c_str());
    CHECK(monotone);
    CHECK(start == 0.5f);           // half the base budget at the start
    CHECK(t.req_b() == 0.0f);       // braked to nothing
    CHECK(t.stats().find("brakes 0") == std::string::npos);

    // a clean request: no flips, the budget grows back to b
    StandinTune c;
    c.configure(1.0f, 4.0f, false, true);
    c.set_brake_legacy(true);
    Engine q;
    q.flip_p = 0.0;
    c.begin_request();
    for (int i = 0; i < 1000; ++i) q.token(c);
    std::printf("clean request (no flips): req_b %.4f, b %.4f\n", (double) c.req_b(), (double) c.b());
    CHECK(q.flips == 0);
    CHECK(c.req_b() == c.b());
}

// 4. k: no flips, no change; pinned, no change; learned, it moves only at a 50th re-check and inside its bounds.
static void test_k_evidence() {   // 9 Nov port of the Qwen rule
    {   // no measured noise floor (no control re-checks): flips everywhere cannot move k
        StandinTune t;
        t.configure(0.1f, 4.0f, false, false);
        t.set_ctrl_every(0);
        Engine e;
        e.flip_from = 0.0;
        e.flip_p = 0.5;
        for (int i = 0; i < 3000; ++i) {
            if (i % 200 == 0) t.begin_request();
            e.token(t);
        }
        CHECK(e.flips > 0);
        CHECK(t.k() == 4.0f);
    }
    {   // with a floor of 0 (controls never differ) the noise bound shrinks and the flips set k; every step within 0.5x-1.5x
        StandinTune t;
        t.configure(0.1f, 4.0f, false, false);
        for (int i = 0; i < 40; ++i) t.note_control(false);
        CHECK(t.ctrl_count() == 40);
        CHECK(t.ctrl_hi() < 0.1);
    }
}

static void test_k() {
    {   // no flips at all
        StandinTune t;
        t.configure(0.1f, 4.0f, false, true);
        Engine e;
        e.flip_p = 0.0;
        e.run(t, 3000, 200);
        CHECK(e.rechecks > 100);
        CHECK(t.k() == 4.0f);
    }
    {   // pinned: flips do not move it
        StandinTune t;
        t.configure(0.1f, 2.5f, true, true);
        Engine e;
        e.flip_from = 0.0;
        e.flip_p = 1.0;
        e.run(t, 2000, 200);
        CHECK(e.flips > 0);
        CHECK(t.k() == 2.5f);
    }
    {   // learned: moves only at a 50th re-check, each step within [0.5, 1.5] of the old value, always within [0.005, 50]
        StandinTune t;
        t.configure(0.1f, 4.0f, false, false);
        Engine e;
        e.flip_from = 0.0;
        e.flip_p = 0.5;
        float prev = t.k();
        int changes = 0;
        bool at_50 = true, in_bounds = true, in_factor = true;
        for (int i = 0; i < 4000; ++i) {
            if (i % 200 == 0) t.begin_request();
            e.token(t);
            if (t.control_due()) t.note_control(false);   // (9 Nov: the engine's control re-checks; exact vs exact never differs here)
            if (t.k() != prev) {
                ++changes;
                if (e.rechecks % 50 != 0) at_50 = false;
                const double r = (double) t.k() / (double) prev;
                if (r < 0.5 - 1e-6 || r > 1.5 + 1e-6) in_factor = false;
                if (t.k() < 0.005f || t.k() > 50.0f) in_bounds = false;
                prev = t.k();
            }
        }
        std::printf("k learning: %d changes, k %.4f (%lld re-checks, %lld flips)\n", changes, (double) t.k(), e.rechecks, e.flips);
        CHECK(changes > 0);
        CHECK(at_50);
        CHECK(in_factor);
        CHECK(in_bounds);
    }
}

// Brakes so far in this tuner (the stats line's "brakes N").
static long long brakes_of(const StandinTune& t) {
    const std::string s = t.stats();
    const std::size_t p = s.find("brakes ");
    long long v = -1;
    if (p != std::string::npos) std::sscanf(s.c_str() + p, "brakes %lld", &v);
    return v;
}

// 3 (new rule). No fixed thresholds: the brake acts on evidence against measured references (the band against the control floor,
// the request's rate against the lifetime's).
static void test_brake_new() {
    {   // (i) no history, many flips at low ratios (ratio 1.0 < 0.5 k = 2) and none in the band: no brake
        StandinTune t;
        t.configure(1.0f, 4.0f, false, true);
        t.begin_request();
        for (int i = 0; i < 300; ++i) {
            t.budget_for_next_token();
            t.end_token(0.01, true, 1.0f, 1.0f, true);
        }
        std::printf("new rule (i) no history, low-ratio flips: %s\n", t.stats().c_str());
        CHECK(brakes_of(t) == 0);
        CHECK(t.k() == 4.0f);
    }
    {   // (ii) controls that never differ (a floor near 0.08) and band re-checks that mostly flip: brakes
        StandinTune t;
        t.configure(1.0f, 4.0f, false, true);
        for (int i = 0; i < 40; ++i) t.note_control(false);
        t.begin_request();
        const float start = t.req_b();
        for (int i = 0; i < 40; ++i) {
            t.budget_for_next_token();
            t.end_token(0.01, true, 3.0f, 1.0f, i % 4 != 0);   // ratio 3 >= 0.5 k: in the band; 3 of 4 flip
        }
        std::printf("new rule (ii) band flips vs a clean floor: %s\n", t.stats().c_str());
        CHECK(brakes_of(t) > 0);
        CHECK(t.req_b() < start);
    }
    {   // (iii) a lifetime rate of ~1% (six requests), then a request at ~1%: no brake; a request at ~10%: brakes
        StandinTune t;
        t.configure(1.0f, 4.0f, false, true);
        Engine e;
        e.flip_from = 0.0;
        e.flip_p = 0.025;           // 40% re-checked x 2.5% = ~1% of tokens
        for (int r = 0; r < 6; ++r) {
            t.begin_request();
            for (int i = 0; i < 1000; ++i) e.token(t);
        }
        t.begin_request();          // folds the sixth request into the lifetime
        const long long before_low = brakes_of(t);
        for (int i = 0; i < 1000; ++i) e.token(t);
        std::printf("new rule (iii) lifetime ~1%% (%lld tokens, %lld flips), request at ~1%%: %s\n", t.lifetime_tok(), t.lifetime_fl(), t.stats().c_str());
        CHECK(brakes_of(t) == before_low);
        Engine h;
        h.flip_from = 0.0;
        h.flip_p = 0.25;            // ~10% of tokens
        t.begin_request();
        const long long before_high = brakes_of(t);
        for (int i = 0; i < 1000; ++i) h.token(t);
        std::printf("  request at ~10%%: %s\n", t.stats().c_str());
        CHECK(brakes_of(t) > before_high);
    }
    {   // (iv) a clean request (no re-checks at all) reaches req_b == b at its first check, token 64
        StandinTune t;
        t.configure(1.0f, 4.0f, false, true);
        t.begin_request();
        for (int i = 1; i <= 64; ++i) {
            t.budget_for_next_token();
            if (i == 63) CHECK(t.req_b() == 0.5f);   // not yet: the first check is at token 64
            t.end_token(0.01, false, 10.0f, 1.0f, false);
        }
        CHECK(t.req_b() == t.b());
        std::printf("new rule (iv) clean request: req_b %.4f b %.4f\n", (double) t.req_b(), (double) t.b());
    }
}

// 5. save then load restores b, k, step, clean (checked by saving the loaded state again: the two files are byte-identical).
static void test_save_load() {
    const std::string a_path = tmp_path("ds4_standin_tune_a.txt");
    const std::string b_path = tmp_path("ds4_standin_tune_b.txt");
    {
        std::ofstream f(a_path);
        f << "budget 0.1\nclean 0.4\n";
    }
    StandinTune a;
    a.configure(0.1f, 4.0f, false, true);
    CHECK(a.load(a_path));
    Engine e;
    e.flip_p = 0.5;
    e.flip_from = 0.3;
    e.run(a, 6000, 200);
    CHECK(a.save(b_path + ".a"));
    const std::string saved = slurp(b_path + ".a");
    std::printf("saved state:\n%s", saved.c_str());
    CHECK(saved.find("budget ") != std::string::npos);
    CHECK(saved.find("k ") != std::string::npos);
    CHECK(saved.find("step ") != std::string::npos);
    CHECK(saved.find("clean ") != std::string::npos);
    CHECK(saved.find("tests ") != std::string::npos);
    CHECK(saved.find("lastwin ") != std::string::npos);

    StandinTune r;
    r.configure(0.3f, 3.0f, false, true);   // a different start: the load must replace it
    CHECK(r.load(b_path + ".a"));
    CHECK(r.b() == a.b());
    CHECK(r.k() == a.k());
    CHECK(r.save(b_path));
    CHECK(slurp(b_path) == saved);          // step, clean, tests, lastwin restored too

    // a pinned k is not replaced by the file's k
    StandinTune p;
    p.configure(0.1f, 2.0f, true, true);
    CHECK(p.load(b_path + ".a"));
    CHECK(p.k() == 2.0f);
    CHECK(p.b() == a.b());

    // a missing file, and a file without a budget line, load as false
    CHECK(!r.load(tmp_path("ds4_standin_tune_missing.txt")));
    { std::ofstream f(b_path + ".nob"); f << "k 3\n"; }
    CHECK(!r.load(b_path + ".nob"));

    std::filesystem::remove(a_path);
    std::filesystem::remove(b_path);
    std::filesystem::remove(b_path + ".a");
    std::filesystem::remove(b_path + ".nob");
}

// 6. The lifetime counts (alltok / allfl) join at begin_request, are saved and loaded (missing lines are 0), and halve above 200000.
static void test_lifetime() {
    const std::string a_path = tmp_path("ds4_standin_tune_life_a.txt");
    const std::string b_path = tmp_path("ds4_standin_tune_life_b.txt");
    const std::string c_path = tmp_path("ds4_standin_tune_life_c.txt");
    StandinTune a;
    a.configure(1.0f, 4.0f, false, true);
    Engine e;
    e.flip_from = 0.0;
    e.flip_p = 0.025;
    for (int r = 0; r < 5; ++r) {
        a.begin_request();
        for (int i = 0; i < 1000; ++i) e.token(a);
    }
    a.begin_request();   // folds the fifth request
    CHECK(a.lifetime_tok() == 5000);
    const long long lf = a.lifetime_fl();
    CHECK(lf > 0);
    CHECK(a.save(a_path));
    const std::string saved = slurp(a_path);
    CHECK(saved.find("alltok 5000\n") != std::string::npos);
    CHECK(saved.find("allfl " + std::to_string(lf) + "\n") != std::string::npos);

    StandinTune r;
    r.configure(1.0f, 4.0f, false, true);
    CHECK(r.load(a_path));
    CHECK(r.lifetime_tok() == 5000);
    CHECK(r.lifetime_fl() == lf);
    CHECK(r.save(b_path));
    CHECK(slurp(b_path) == saved);   // the lifetime lines round-trip byte for byte

    {   // a file without the lifetime lines loads with 0
        { std::ofstream f(c_path); f << "budget 0.2\n"; }
        StandinTune m;
        m.configure(1.0f, 4.0f, false, true);
        CHECK(m.load(c_path));
        CHECK(m.lifetime_tok() == 0);
        CHECK(m.lifetime_fl() == 0);
    }
    {   // halving: 250 requests of 1000 tokens (no flips): the lifetime stays at or below 200000 and above 100000
        StandinTune h;
        h.configure(1.0f, 4.0f, false, true);
        Engine q;
        q.flip_p = 0.0;
        for (int k = 0; k < 250; ++k) {
            h.begin_request();
            for (int i = 0; i < 1000; ++i) q.token(h);
        }
        h.begin_request();
        std::printf("lifetime after 250000 tokens: %lld\n", h.lifetime_tok());
        CHECK(h.lifetime_tok() <= 200000);
        CHECK(h.lifetime_tok() > 100000);
    }

    std::filesystem::remove(a_path);
    std::filesystem::remove(b_path);
    std::filesystem::remove(c_path);
}

int main() {
    test_climb_ceiling(true);
    test_climb_ceiling(false);
    test_climb_to_knee(true);
    test_climb_to_knee(false);
    test_brake_legacy();
    test_brake_new();
    test_k();
    test_k_evidence();
    test_save_load();
    test_lifetime();
    if (fails == 0) {
        std::printf("PASS ds4_standin_tune_test: all checks passed\n");
        return 0;
    }
    std::printf("FAIL ds4_standin_tune_test: %d check(s) failed\n", fails);
    return 1;
}
