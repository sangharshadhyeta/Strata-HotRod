// tests/core/subfit_learn_test.cpp - SPEC-substitute-fit stage 2, step 1: the live learner, CPU only.
// Covers: env parsing, fp16 round trip, the feature matrix, the streaming ridge (recovers a known B, exactly at lambda 0 and within
// the shrinkage at 0.1), the held-out gate (a learnable pair fits, a noise pair is refused, a pair whose map changes is revoked),
// eviction (least recently sampled, a fitted victim frees its slot), the slot cap and the lookup table, and the save/load round trip.
#include "strata/core/subfit_learn.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <random>
#include <string>
#include <vector>

using namespace strata::core::subfit_learn;

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

// A synthetic world: y_e = c * y_s + Bt z (+ noise), z = A_l x with the learner's own features.
struct World {
    std::mt19937 g;
    std::vector<float> A;        // R x N, the learner's features of layer 0
    std::vector<float> Bt;       // N x R, the true correction
    World(uint32_t seed, float bscale) : g(seed), Bt((size_t) N * R) {
        features(0, N, R, 1, A);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (float& v : Bt) v = bscale * nd(g);
    }
    void sample(std::vector<float>& x, std::vector<float>& ys) {
        std::normal_distribution<float> nd(0.0f, 1.0f);
        x.resize(N);
        ys.resize(N);
        for (float& v : x) v = nd(g);
        for (float& v : ys) v = nd(g);
    }
    std::vector<float> z(const std::vector<float>& x) const {
        std::vector<float> out(R, 0.0f);
        for (int k = 0; k < R; ++k)
            for (int j = 0; j < N; ++j) out[(size_t) k] += A[(size_t) k * N + j] * x[(size_t) j];
        return out;
    }
};

// y_e = sign * (c y_s + Bt z + small noise) for the learnable map (sign -1 is a map that changes sign), or independent noise
void make_ye(World& w, const std::vector<float>& x, const std::vector<float>& ys, float c, bool noise, float sign, std::vector<float>& ye) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    ye.assign(N, 0.0f);
    if (noise) {
        for (float& v : ye) v = nd(w.g);
        return;
    }
    const std::vector<float> z = w.z(x);
    for (int j = 0; j < N; ++j) {
        float acc = c * ys[(size_t) j] + 0.02f * nd(w.g);
        for (int k = 0; k < R; ++k) acc += w.Bt[(size_t) j * R + k] * z[(size_t) k];
        ye[(size_t) j] = sign * acc;
    }
}

Config cfg(int max_pairs = 64, int max_fitted = 512) {
    Config c;
    c.n = N;
    c.rank = R;
    c.max_pairs = max_pairs;
    c.max_fitted = max_fitted;
    c.seed = 1;
    c.n_layers = NL;
    return c;
}

// one request: `count` samples of a pair, then the request end's refit
Changes request(Learner& L, World& w, int layer, int e, int s, int count, float c, bool noise, float sign = 1.0f) {
    std::vector<float> x, ys, ye;
    for (int i = 0; i < count; ++i) {
        w.sample(x, ys);
        make_ye(w, x, ys, c, noise, sign, ye);
        L.add_sample(layer, e, s, x.data(), ys.data(), ye.data());
    }
    return L.refit();
}

void test_env_and_fp16() {
    CHECK(apply_env(nullptr) == Apply::Dynamic, "apply unset: dynamic");
    CHECK(apply_env("off") == Apply::Off && apply_env("all") == Apply::All && apply_env("dynamic") == Apply::Dynamic, "apply names");
    CHECK(apply_env("bogus") == Apply::Dynamic, "apply other value: dynamic");
    CHECK(std::string(apply_name(Apply::All)) == "all", "apply name");
    CHECK(rank_env(nullptr) == 16 && rank_env("8") == 8 && rank_env("0") == 16 && rank_env("99") == 32, "rank env");
    CHECK(pairs_env(nullptr) == 512 && pairs_env("64") == 64 && pairs_env("-1") == 512, "pairs env");
    CHECK(f16_to_f32(f32_to_f16(1.0f)) == 1.0f && f16_to_f32(f32_to_f16(-2.5f)) == -2.5f, "fp16 exact values");
    CHECK(f16_to_f32(f32_to_f16(0.0f)) == 0.0f, "fp16 zero");
    bool rel_ok = true;
    for (int i = 1; i < 4000; ++i) {
        const float v = std::sin((float) i) * std::pow(10.0f, (float) (i % 9) - 4.0f);
        const float b = f16_to_f32(f32_to_f16(v));
        if (std::fabs(b - v) > std::fabs(v) * (1.0f / 1024.0f) + 1e-7f) rel_ok = false;   // half an ulp: 2^-11 relative for normals
    }
    CHECK(rel_ok, "fp16 round trip within 2^-11 relative (normal range)");
    CHECK(std::isinf(f16_to_f32(f32_to_f16(1e6f))), "fp16 overflow goes to infinity");
    const float sub = f16_to_f32(f32_to_f16(3.0e-6f));
    CHECK(sub > 0.0f && std::fabs(sub - 3.0e-6f) < 6e-8f, "fp16 subnormal close to the input");
}

void test_features() {
    std::vector<float> a, b, c;
    features(3, N, R, 7, a);
    features(3, N, R, 7, b);
    features(4, N, R, 7, c);
    CHECK(a == b, "features: deterministic for a layer and seed");
    CHECK(a != c, "features: different layers differ");
    double ms = 0.0;
    for (float v : a) ms += (double) v * (double) v;
    ms /= (double) a.size();
    CHECK(std::fabs(ms * N - 1.0) < 0.15, "features: entries scaled so E[a^2] = 1/N");
}

void test_ridge_recovers_b() {
    World w(5, 0.5f);
    const float c_true = 0.8f;
    std::vector<double> ZtZ((size_t) R * R, 0.0);
    std::vector<float> ZtYe((size_t) R * N, 0.0f), ZtYs((size_t) R * N, 0.0f);
    std::vector<float> x, ys, ye;
    for (int i = 0; i < 3000; ++i) {
        w.sample(x, ys);
        make_ye(w, x, ys, c_true, false, 1.0f, ye);
        // (the 0.02 noise of make_ye is part of the data here: the exact-recovery bound is loose enough for it)
        const std::vector<float> z = w.z(x);
        for (int k = 0; k < R; ++k) {
            for (int m = 0; m < R; ++m) ZtZ[(size_t) k * R + m] += (double) z[k] * (double) z[m];
            for (int j = 0; j < N; ++j) {
                ZtYe[(size_t) k * N + j] += z[k] * ye[(size_t) j];
                ZtYs[(size_t) k * N + j] += z[k] * ys[(size_t) j];
            }
        }
    }
    // exact at lambda 0: the target is exactly B_true z
    std::vector<float> B;
    CHECK(ridge_solve(ZtZ.data(), ZtYe.data(), ZtYs.data(), c_true, R, N, 0.0, B), "ridge at lambda 0 solves");
    double err = 0.0, den = 0.0;
    for (size_t i = 0; i < B.size(); ++i) {
        err += (double) (B[i] - w.Bt[i]) * (double) (B[i] - w.Bt[i]);
        den += (double) w.Bt[i] * (double) w.Bt[i];
    }
    CHECK(std::sqrt(err / den) < 1e-3, "ridge recovers the known B (relative error < 1e-3 at lambda 0)");
    // with the learner's lambda (0.1 trace / r) the solution must satisfy the regularised normal equations exactly:
    // (ZtZ + lam I) b_j = T_j for every output j, where T_j = ZtYe_j - c ZtYs_j (the fit is shrunk, so B itself is not Bt)
    std::vector<float> Bs;
    CHECK(ridge_solve(ZtZ.data(), ZtYe.data(), ZtYs.data(), c_true, R, N, 0.1, Bs), "ridge at lambda 0.1 solves");
    double trace = 0.0;
    for (int k = 0; k < R; ++k) trace += ZtZ[(size_t) k * R + k];
    const double lam = 0.1 * trace / R;
    double worst = 0.0, scale = 0.0;
    for (int j = 0; j < N; ++j) {
        for (int k = 0; k < R; ++k) {
            double lhs = lam * (double) Bs[(size_t) j * R + k], rhs = 0.0;
            for (int m = 0; m < R; ++m) lhs += ZtZ[(size_t) k * R + m] * (double) Bs[(size_t) j * R + m];
            rhs = (double) ZtYe[(size_t) k * N + j] - c_true * (double) ZtYs[(size_t) k * N + j];
            worst = std::max(worst, std::fabs(lhs - rhs));
            scale = std::max(scale, std::fabs(rhs));
        }
    }
    CHECK(worst <= 1e-4 * scale, "ridge at lambda 0.1 satisfies (ZtZ + lam I) B = T (normal equations)");
    // no samples: no fit
    std::vector<double> zero((size_t) R * R, 0.0);
    std::vector<float> none;
    CHECK(!ridge_solve(zero.data(), ZtYe.data(), ZtYs.data(), c_true, R, N, 0.1, none), "ridge with no statistics refuses");
}

void test_gate() {
    Learner L(cfg());
    World learn(11, 0.5f), noise_w(12, 0.0f), flip(11, 0.5f);
    // learnable pair (0, 1, 2): fitted after two requests of 25 held-out samples each (the second request's gate sees 32)
    Changes c1 = request(L, learn, 0, 1, 2, 100, 0.8f, false);
    CHECK(!L.fitted(0, 1, 2), "learnable pair: not fitted before the gate has 32 held-out samples");
    (void) c1;
    Changes c2 = request(L, learn, 0, 1, 2, 100, 0.8f, false);
    CHECK(L.fitted(0, 1, 2), "learnable pair: fitted after its held-out gain passes 0.20");
    CHECK(L.lookup(0, 1, 2) >= 0, "learnable pair: has a device slot");
    double c_used = 0, c_raw = 0, gain = 0;
    bool is_fit = false;
    CHECK(L.pair_info(0, 1, 2, c_used, c_raw, gain, is_fit), "pair info");
    CHECK(std::fabs(c_raw - 0.8) < 0.1 && std::fabs(c_used - 0.8) < 0.1, "learnable pair: the scalar recovers 0.8");
    CHECK(gain >= 0.20, "learnable pair: the recent held-out gain is at least 0.20");
    bool slot_change_seen = false;
    for (const SlotChange& sc : c2.slots)
        if (sc.slot >= 0 && sc.layer == 0 && sc.e == 1 && sc.s == 2 && sc.b.size() == (size_t) N * R) slot_change_seen = true;
    CHECK(slot_change_seen, "the fitted pair's slot is uploaded with its fp16 B (N x r)");

    // noise pair (0, 3, 4): its held-out gain stays below the bar, so it is refused
    for (int i = 0; i < 3; ++i) request(L, noise_w, 0, 3, 4, 100, 0.0f, true);
    CHECK(!L.fitted(0, 3, 4), "noise pair: refused (held-out gain below 0.20)");
    CHECK(L.lookup(0, 3, 4) == -1, "noise pair: no slot");

    // revoke: the learnable pair's map changes sign; its held-out samples (scored by the old fit) fall below 0.10
    request(L, flip, 0, 1, 2, 100, 0.8f, false, -1.0f);
    CHECK(!L.fitted(0, 1, 2), "a pair whose map changes is revoked");
    CHECK(L.lookup(0, 1, 2) == -1, "a revoked pair frees its slot");
}

void test_eviction_and_slots() {
    Learner L(cfg(3, 2));
    World w(21, 0.5f);
    request(L, w, 0, 10, 11, 60, 1.0f, false);
    request(L, w, 0, 10, 12, 60, 1.0f, false);
    request(L, w, 0, 10, 13, 60, 1.0f, false);
    CHECK(L.pair_count() == 3, "three pairs kept");
    // a fourth pair evicts the least recently sampled (10, 11)
    request(L, w, 0, 10, 14, 4, 1.0f, false);
    CHECK(L.pair_count() == 3, "the cap holds at three pairs");
    double c_used = 0, c_raw = 0, gain = 0;
    bool f = false;
    CHECK(!L.pair_info(0, 10, 11, c_used, c_raw, gain, f), "the least recently sampled pair is evicted");
    CHECK(L.pair_info(0, 10, 12, c_used, c_raw, gain, f), "the others stay");

    // slot cap: three learnable pairs, two slots; the third stays fitted and unslotted
    Learner S(cfg(64, 2));
    World sw(31, 0.5f);
    for (int round = 0; round < 3; ++round) {
        request(S, sw, 1, 5, 6, 100, 0.9f, false);
        request(S, sw, 1, 5, 7, 100, 0.9f, false);
        request(S, sw, 1, 5, 8, 100, 0.9f, false);
    }
    const Summary sm = S.summary();
    CHECK(sm.fitted == 3, "three learnable pairs fitted");
    CHECK(S.slots_used() == 2, "two slots used of two");
    CHECK(sm.unslotted == 1, "one fitted pair has no slot (the table is full)");
    int slotted = 0;
    for (int s = 6; s <= 8; ++s)
        if (S.lookup(1, 5, s) >= 0) ++slotted;
    CHECK(slotted == 2, "the lookup gives a slot to two pairs");
    {
        std::vector<int> got;
        for (int s = 6; s <= 8; ++s)
            if (S.lookup(1, 5, s) >= 0) got.push_back(S.lookup(1, 5, s));
        CHECK(got.size() == 2 && got[0] != got[1] && got[0] < 2 && got[1] < 2 && got[0] >= 0 && got[1] >= 0,
              "the two slotted pairs have distinct slots in 0..1");
    }
    CHECK(S.lookup(9, 5, 7) == -1 && S.lookup(1, 600, 7) == -1 && S.lookup(-1, 0, 0) == -1, "out-of-range keys give -1");
    const Changes all = S.all_fitted();
    CHECK(all.slots.size() == 2 && all.lut.size() == 2, "the full upload carries the slotted pairs");
}

void test_save_load() {
    const std::string path = (std::filesystem::temp_directory_path() / "subfit_learn_test.bin").string();
    std::filesystem::remove(path);
    std::string err;
    Learner L(cfg());
    World w(41, 0.5f);
    request(L, w, 0, 2, 3, 100, 0.7f, false);
    request(L, w, 0, 2, 3, 100, 0.7f, false);
    request(L, w, 1, 4, 5, 60, 0.7f, false);   // not yet fitted (too few held-out samples)
    CHECK(L.save(path, 77, err), ("save: " + err).c_str());
    Learner M(cfg());
    CHECK(!M.load(path, 78, err), "another model's fingerprint is refused");
    CHECK(M.load(path, 77, err), ("load: " + err).c_str());
    CHECK(M.fitted(0, 2, 3) == L.fitted(0, 2, 3) && M.fitted(0, 2, 3), "the fit comes back");
    CHECK(M.lookup(0, 2, 3) == L.lookup(0, 2, 3) && M.lookup(0, 2, 3) >= 0, "the slot comes back");
    CHECK(M.pair_count() == L.pair_count(), "the pair count comes back");
    const std::vector<float>* b1 = L.pair_b(0, 2, 3);
    const std::vector<float>* b2 = M.pair_b(0, 2, 3);
    CHECK(b1 != nullptr && b2 != nullptr && *b1 == *b2, "B comes back exactly");
    double c1 = 0, r1 = 0, g1 = 0, c2 = 0, r2 = 0, g2 = 0;
    bool f1 = false, f2 = false;
    L.pair_info(1, 4, 5, c1, r1, g1, f1);
    M.pair_info(1, 4, 5, c2, r2, g2, f2);
    CHECK(c1 == c2 && r1 == r2 && g1 == g2 && f1 == f2, "the stats of the unfitted pair come back exactly");
    // a learner loaded from the file keeps learning: the next refit runs without error on the same pair
    request(M, w, 0, 2, 3, 40, 0.7f, false);
    CHECK(M.fitted(0, 2, 3), "a loaded fit stays fitted under more samples of the same map");
    std::filesystem::remove(path);
}

}  // namespace

int main() {
    test_env_and_fp16();
    test_features();
    test_ridge_recovers_b();
    test_gate();
    test_eviction_and_slots();
    test_save_load();
    if (failures == 0) {
        std::printf("subfit_learn_test: all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "subfit_learn_test: %d check(s) failed\n", failures);
    return 1;
}
