// tests/core/subfit_ladder_test.cpp - SPEC-substitute-fit stage 2, the fallback ladder and the shadow-sample queue, CPU only.
// Covers: a layer group fits while every pair is thin and a never-seen pair then uses the layer's slot; a (layer, expert) group is
// used before the layer's; a dense pair uses its own slot first; nothing is used when nothing fits (noise); the packer writes the
// expert and layer slots into their regions; save/load keeps the ladder; the queue evaluates in the background, counts failures,
// drops without an evaluator and past its cap.
#include "strata/core/subfit_learn.hpp"
#include "strata/core/subfit_shadow.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

using namespace strata::core::subfit_learn;
namespace sh = strata::core::subfit_shadow;

namespace {

int failures = 0;
#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); \
            ++failures;                                                        \
        }                                                                      \
    } while (0)

constexpr int N = 32, R = 16, NL = 2;

Config cfg() {
    Config c;
    c.n = N;
    c.rank = R;
    c.n_layers = NL;
    c.max_pairs = 64;
    c.max_experts = 64;
    return c;
}

// y_e = c y_s + B z(x) + small noise, with z the learner's own features of layer `l` (a learnable map), or pure noise
struct Map {
    std::vector<float> A, B;   // A: R x N (layer's features), B: N x R
    float c;
    Map(int layer, uint32_t seed, float c_) : B((size_t) N * R), c(c_) {
        features(layer, N, R, 1, A);
        std::mt19937 g(seed);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (float& v : B) v = 2.0f * nd(g);
    }
    void make(std::mt19937& g, std::vector<float>& x, std::vector<float>& ys, std::vector<float>& ye, bool noise) const {
        std::normal_distribution<float> nd(0.0f, 1.0f);
        x.resize(N); ys.resize(N); ye.assign(N, 0.0f);
        for (float& v : x) v = nd(g);
        for (float& v : ys) v = nd(g);
        if (noise) { for (float& v : ye) v = nd(g); return; }
        std::vector<float> z(R, 0.0f);
        for (int k = 0; k < R; ++k)
            for (int j = 0; j < N; ++j) z[(size_t) k] += A[(size_t) k * N + j] * x[(size_t) j];
        for (int j = 0; j < N; ++j) {
            float v = c * ys[(size_t) j];
            for (int k = 0; k < R; ++k) v += B[(size_t) j * R + k] * z[(size_t) k];
            ye[(size_t) j] = v + 0.01f * nd(g);
        }
    }
};

void test_layer_fits_while_pairs_thin() {
    Learner L(cfg());
    Map m(0, 7, 0.8f);
    std::mt19937 g(1);
    std::vector<float> x, ys, ye;
    // 400 samples spread over 400 different (e, s) pairs: every pair has 1 sample, the layer group has 400
    for (int i = 0; i < 400; ++i) {
        m.make(g, x, ys, ye, false);
        L.add_sample(0, i % 200, 200 + (i % 300) % 300, x.data(), ys.data(), ye.data());
        if (i % 50 == 49) L.refit();
    }
    L.refit();
    CHECK(L.level_used(0, 5, 401) == 2, "a never-seen pair uses its layer's slot when only the layer group fits");
    CHECK(L.lookup(0, 5, 401) >= 0, "the layer's slot is a real slot");
    CHECK(L.level_used(1, 5, 401) == -1, "another layer (no samples) has no slot");
    const Summary s = L.summary();
    CHECK(s.fitted_lv[2] == 1 && s.fitted_lv[0] == 0, "one layer group fitted, no pair fitted");
}

void test_expert_before_layer_and_pair_first() {
    Learner L(cfg());
    Map m(0, 11, 0.5f);
    std::mt19937 g(2);
    std::vector<float> x, ys, ye;
    // expert 3: 300 samples over many substitutes (its group fits); pair (3, 300): 200 of them (fits too); the layer group sees all
    for (int i = 0; i < 300; ++i) {
        m.make(g, x, ys, ye, false);
        const int s = i < 200 ? 300 : 301 + i % 10;   // (10 other substitutes: the pair cap must not evict the dense pair)
        L.add_sample(0, 3, s % 512, x.data(), ys.data(), ye.data());
        if (i % 25 == 24) L.refit();
    }
    L.refit();
    CHECK(L.level_used(0, 3, 300) == 0, "a dense fitted pair uses its own slot");
    CHECK(L.level_used(0, 3, 50) == 1, "an unseen substitute of a fitted expert group uses the expert's slot before the layer's");
    CHECK(L.level_used(0, 9, 50) == 2, "another expert falls back to the layer's slot");
    CHECK(L.lookup(0, 3, 300) != L.lookup(0, 3, 50), "pair and expert slots differ");
}

void test_noise_uses_nothing() {
    Learner L(cfg());
    Map m(0, 13, 0.0f);
    std::mt19937 g(3);
    std::vector<float> x, ys, ye;
    for (int i = 0; i < 400; ++i) {
        m.make(g, x, ys, ye, true);
        L.add_sample(0, i % 20, 100 + i % 7, x.data(), ys.data(), ye.data());
    }
    L.refit();
    CHECK(L.level_used(0, 1, 101) == -1, "no level is used when nothing fits (noise)");
    CHECK(L.summary().fitted == 0, "nothing fitted on noise");
}

void test_packer_regions_and_persistence() {
    UploadPacker P(NL);
    Changes ch;
    ch.lut.push_back(LutChange{1, 7, kAny, 5});
    ch.lut.push_back(LutChange{1, kAny, kAny, 6});
    ch.lut.push_back(LutChange{0, 2, 3, 4});
    std::vector<LutWrite> lw;
    std::vector<SlotWrite> sw;
    P.pack(ch, lw, sw);
    CHECK(lw.size() == 3, "three LUT writes");
    CHECK(lw[0].index == (size_t) NL * 512 * 512 + 1 * 512 + 7, "the expert slot's index is in the expert region");
    CHECK(lw[1].index == (size_t) NL * 512 * 512 + (size_t) NL * 512 + 1, "the layer slot's index is in the layer region");
    CHECK(lw[2].index == ((size_t) 0 * 512 + 2) * 512 + 3, "the pair's index is unchanged");
    CHECK(P.fit_n()[1] == 2 && P.fit_n()[0] == 1, "fitted counts per layer include the expert and layer slots");
    CHECK(table_size(NL) == (size_t) NL * 512 * 512 + (size_t) NL * 512 + (size_t) NL, "table size");

    Learner L(cfg());
    Map m(1, 17, 0.7f);
    std::mt19937 g(4);
    std::vector<float> x, ys, ye;
    for (int i = 0; i < 400; ++i) {
        m.make(g, x, ys, ye, false);
        L.add_sample(1, i % 150, 300 + i % 100, x.data(), ys.data(), ye.data());
        if (i % 25 == 24) L.refit();
    }
    L.refit();
    const int lv = L.level_used(1, 4, 499), slot = L.lookup(1, 4, 499);
    CHECK(lv >= 1, "a pooled level fitted before saving");
    const std::string path = (std::filesystem::temp_directory_path() / "subfit_ladder_test.bin").string();
    std::string err;
    CHECK(L.save(path, 42, err), "save");
    Learner L2(cfg());
    CHECK(L2.load(path, 42, err), err.c_str());
    CHECK(L2.level_used(1, 4, 499) == lv && L2.lookup(1, 4, 499) == slot, "the ladder survives save/load");
    std::filesystem::remove(path);
}

void test_pair_admission() {
    Config c = cfg();
    c.pair_min_seen = 5;
    Learner L(c);
    Map m(0, 21, 0.6f);
    std::mt19937 g(9);
    std::vector<float> x, ys, ye;
    for (int i = 0; i < 4; ++i) { m.make(g, x, ys, ye, false); L.add_sample(0, 1, 2, x.data(), ys.data(), ye.data()); }
    double cu, cr, gn; bool f;
    CHECK(!L.pair_info(0, 1, 2, cu, cr, gn, f), "a pair seen 4 times (min 5) has no group yet");
    CHECK(L.pair_info(0, 1, kAny, cu, cr, gn, f), "its expert group learns from the start");
    m.make(g, x, ys, ye, false); L.add_sample(0, 1, 2, x.data(), ys.data(), ye.data());
    CHECK(L.pair_info(0, 1, 2, cu, cr, gn, f), "the 5th sighting admits the pair");
}

void test_queue() {
    sh::Queue q;
    sh::Item it;
    it.layer = 1; it.e = 2; it.s = 3;
    it.x.assign(N, 1.5f);
    it.ye.assign(N, 0.0f);
    q.push(sh::Item(it));
    CHECK(q.take_stats().dropped == 1, "without an evaluator an item is dropped");
    q.set_eval([](int layer, int s, const float* x, float* ys) {
        if (s == 99) return false;
        for (int j = 0; j < N; ++j) ys[j] = 2.0f * x[j] + (float) layer;
        return true;
    }, false);
    for (int i = 0; i < 10; ++i) q.push(sh::Item(it));
    sh::Item bad = it;
    bad.s = 99;
    q.push(std::move(bad));
    CHECK(q.drain_for_tests(5000), "the worker drains the queue");
    const auto done = q.take_done();
    CHECK(done.size() == 10, "ten finished samples");
    CHECK(!done.empty() && std::fabs(done[0].ys[0] - 4.0f) < 1e-6f, "the evaluator's y_s is kept");
    const sh::Stats st = q.take_stats();
    CHECK(st.queued == 11 && st.evaluated == 10 && st.failed == 1, "queued / evaluated / failed counts");
    sh::Item ny = it;
    ny.need_ye = true;
    ny.ye.clear();
    q.push(std::move(ny));
    CHECK(q.drain_for_tests(5000), "drains the y_e item");
    const auto d2 = q.take_done();
    CHECK(d2.size() == 1 && d2[0].ye.size() == (size_t) N && std::fabs(d2[0].ye[0] - 4.0f) < 1e-6f, "need_ye: the worker computes y_e = e(x) too");
    CHECK(!sh::cpu_ye_env(nullptr) && sh::cpu_ye_env("1") && !sh::cpu_ye_env("0"), "cpu y_e env: off by default");
    q.set_paused(true);
    q.push(sh::Item(it));
    CHECK(!q.drain_for_tests(200), "paused: the item waits");
    q.set_paused(false);
    CHECK(q.drain_for_tests(5000), "resumed: the item is evaluated");
    CHECK(q.take_done().size() == 1, "one finished after the resume");
    CHECK(sh::per_cut_env(nullptr) == sh::kDefaultPerCut && sh::per_cut_env("0") == 0 && sh::per_cut_env("999") == 256, "env");
}

}  // namespace

int main() {
    test_layer_fits_while_pairs_thin();
    test_expert_before_layer_and_pair_first();
    test_noise_uses_nothing();
    test_packer_regions_and_persistence();
    test_pair_admission();
    test_queue();
    if (failures == 0) std::printf("subfit_ladder_test: all checks passed\n");
    return failures == 0 ? 0 : 1;
}
