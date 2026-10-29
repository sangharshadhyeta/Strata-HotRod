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
static void test_climb_ceiling() {
    StandinTune t;
    t.configure(0.1f, 4.0f, false, true);
    Engine e;
    std::printf("climb (no clean level): b0 0.100\n");
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
static void test_climb_to_knee() {
    const std::string path = tmp_path("ds4_standin_tune_seed.txt");
    { std::ofstream f(path); f << "budget 0.1\nclean 0.4\n"; }
    StandinTune t;
    t.configure(0.1f, 4.0f, false, true);
    CHECK(t.load(path));
    std::filesystem::remove(path);
    Engine e;
    std::printf("climb (clean 0.4): b0 0.100\n");
    for (int blk = 1; blk <= 30; ++blk) {
        e.run(t, 1000, 1000);
        if (blk % 5 == 0) std::printf("  after %5d tokens: b %.4f req_b %.4f\n", blk * 1000, (double) t.b(), (double) t.req_b());
    }
    std::printf("  %s; largest budget a token ran at %.4f, flips %lld\n", t.stats().c_str(), (double) e.max_b, e.flips);
    CHECK(t.b() >= 0.3f);
    CHECK(t.b() < 0.6f);            // does not run past the knee and the flip region
}

// 3. A request with many flips is braked: req_b halves with each brake, down to 0, and the request's grow never lifts it again.
static void test_brake() {
    StandinTune t;
    t.configure(1.0f, 4.0f, false, true);
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
    Engine q;
    q.flip_p = 0.0;
    c.begin_request();
    for (int i = 0; i < 1000; ++i) q.token(c);
    std::printf("clean request (no flips): req_b %.4f, b %.4f\n", (double) c.req_b(), (double) c.b());
    CHECK(q.flips == 0);
    CHECK(c.req_b() == c.b());
}

// 4. k: no flips, no change; pinned, no change; learned, it moves only at a 50th re-check and inside its bounds.
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

int main() {
    test_climb_ceiling();
    test_climb_to_knee();
    test_brake();
    test_k();
    test_save_load();
    if (fails == 0) {
        std::printf("PASS ds4_standin_tune_test: all checks passed\n");
        return 0;
    }
    std::printf("FAIL ds4_standin_tune_test: %d check(s) failed\n", fails);
    return 1;
}
