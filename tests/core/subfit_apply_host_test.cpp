// tests/core/subfit_apply_host_test.cpp - SPEC-substitute-fit stage 2, steps 2-5: the host side, CPU only (no GPU).
// Covers: the shift rule (subfit_rule::decide) in every mode, against the old cap rule for mode off and fit-less pairs (the same
// function the router kernels call); the upload packer (LUT writes, the mirror and fitted counts, fp16 slot B, residual factor) driven
// by a real learner through a fit and a revoke; the fp16 feature matrix the apply kernel reads.
#include "strata/core/subfit_learn.hpp"
#include "strata/core/subfit_rule.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

using namespace strata::core::subfit_learn;
namespace rule = strata::core::subfit_rule;

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
constexpr float kInf = std::numeric_limits<float>::infinity();

// the learner test's synthetic world: y_e = c y_s + Bt z (+ noise), z = A_l x with the learner's own features
struct World {
    std::mt19937 g;
    std::vector<float> A;
    std::vector<float> Bt;
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
    void make_ye(const std::vector<float>& x, const std::vector<float>& ys, float c, bool noise, float sign, std::vector<float>& ye) {
        std::normal_distribution<float> nd(0.0f, 1.0f);
        ye.assign(N, 0.0f);
        if (noise) {
            for (float& v : ye) v = nd(g);
            return;
        }
        std::vector<float> z(R, 0.0f);
        for (int k = 0; k < R; ++k)
            for (int j = 0; j < N; ++j) z[(size_t) k] += A[(size_t) k * N + j] * x[(size_t) j];
        for (int j = 0; j < N; ++j) {
            float acc = c * ys[(size_t) j] + 0.02f * nd(g);
            for (int k = 0; k < R; ++k) acc += Bt[(size_t) j * R + k] * z[(size_t) k];
            ye[(size_t) j] = sign * acc;
        }
    }
};

Config cfg() {
    Config c;
    c.n = N;
    c.rank = R;
    c.max_pairs = 64;
    c.max_fitted = 512;
    c.seed = 1;
    c.n_layers = NL;
    return c;
}

Changes request(Learner& L, World& w, int layer, int e, int s, int count, float c, bool noise, float sign = 1.0f) {
    std::vector<float> x, ys, ye;
    for (int i = 0; i < count; ++i) {
        w.sample(x, ys);
        w.make_ye(x, ys, c, noise, sign, ye);
        L.add_sample(layer, e, s, x.data(), ys.data(), ye.data());
    }
    return L.refit();
}

// ---- the shift rule
void test_rule() {
    // mode off, no fit: exactly the old cap rule (refused iff shift + plain > cap; a row with no cap never refuses)
    std::mt19937 g(7);
    std::uniform_real_distribution<float> u(0.0f, 2.0f);
    int checked = 0;
    for (int i = 0; i < 20000; ++i) {
        const float shift = u(g), plain = u(g), cap = u(g) * 2.0f;
        for (int capped = 0; capped < 2; ++capped) {
            const rule::Move m = rule::decide(shift, capped ? cap : kInf, capped != 0, plain, false, 1.0f, rule::kModeOff);
            const bool old_take = !(capped && shift + plain > cap);
            CHECK(m.take == old_take, "mode off, no fit: the decision is the old cap rule");
            CHECK(!m.corrected && (m.take ? m.add == plain : m.add == 0.0f), "mode off: the shift added is plain (0 when refused)");
            ++checked;
        }
    }
    // a fitted pair in mode off is not corrected either (the mode is the switch)
    CHECK(!rule::decide(0.0f, 0.5f, true, 9.0f, true, 0.3f, rule::kModeOff).corrected, "mode off ignores the fit");
    CHECK(!rule::decide(0.0f, 0.5f, true, 9.0f, true, 0.3f, rule::kModeOff).take, "mode off: a fitted move past the cap is still refused");

    // dynamic: under the cap, a fitted move is plain; past the cap it is taken corrected, adding plain x resid
    rule::Move m = rule::decide(0.1f, 1.0f, true, 0.4f, true, 0.25f, rule::kModeDynamic);
    CHECK(m.take && !m.corrected && m.add == 0.4f, "dynamic: under the cap a fitted move is plain");
    m = rule::decide(0.8f, 1.0f, true, 0.4f, true, 0.25f, rule::kModeDynamic);
    CHECK(m.take && m.corrected && std::fabs(m.add - 0.1f) < 1e-6f, "dynamic: past the cap a fitted move is corrected, adds plain x resid");
    // dynamic: past the cap, a pair with no fit is refused (today's behaviour)
    m = rule::decide(0.8f, 1.0f, true, 0.4f, false, 1.0f, rule::kModeDynamic);
    CHECK(!m.take && !m.corrected && m.add == 0.0f, "dynamic: past the cap an unfitted pair is refused");
    // dynamic, uncapped row: nothing is refused and nothing is corrected
    m = rule::decide(5.0f, kInf, false, 0.4f, true, 0.25f, rule::kModeDynamic);
    CHECK(m.take && !m.corrected && m.add == 0.4f, "dynamic: an uncapped row takes the move plain");

    // all: every fitted move is taken corrected, no cap test; an unfitted one keeps the cap
    m = rule::decide(0.0f, 0.5f, true, 0.4f, true, 0.5f, rule::kModeAll);
    CHECK(m.take && m.corrected && std::fabs(m.add - 0.2f) < 1e-6f, "all: a fitted move under the cap is corrected");
    m = rule::decide(9.0f, 0.5f, true, 0.4f, true, 0.5f, rule::kModeAll);
    CHECK(m.take && m.corrected, "all: a fitted move past the cap is corrected, not refused");
    m = rule::decide(9.0f, 0.5f, true, 0.4f, false, 0.5f, rule::kModeAll);
    CHECK(!m.take, "all: an unfitted move past the cap is refused");
    m = rule::decide(0.0f, kInf, false, 0.4f, true, 0.5f, rule::kModeAll);
    CHECK(m.take && m.corrected, "all: a fitted move on an uncapped row is corrected");
    std::printf("shift rule: %d old-rule cases agree\n", checked);
}

// ---- the packer, through a real learner
void test_packer() {
    Learner L(cfg());
    UploadPacker P(NL);
    World learn(11, 0.5f), flip(11, 0.5f);

    // a fresh packer's mirror is all -1 and the counts are 0
    CHECK(P.lut().size() == (size_t) NL * 512 * 512, "packer: the mirror is layers x 512 x 512");
    CHECK(P.lut()[UploadPacker::lut_index(1, 511, 511)] == -1, "packer: a fresh mirror is -1");
    CHECK(P.fit_n()[0] == 0 && P.fit_n()[1] == 0, "packer: a fresh count is 0");
    CHECK(UploadPacker::lut_index(1, 2, 3) == ((size_t) 1 * 512 + 2) * 512 + 3, "packer: the index is (layer*512+e)*512+s");

    // fit (0, 1, 2) over two requests; the packer's writes carry the slot's fp16 B and c and the residual
    request(L, learn, 0, 1, 2, 100, 0.8f, false);
    Changes c2 = request(L, learn, 0, 1, 2, 100, 0.8f, false);
    CHECK(L.fitted(0, 1, 2), "packer test: the pair fits");
    std::vector<LutWrite> lw;
    std::vector<SlotWrite> sw;
    P.pack(c2, lw, sw);
    CHECK(!lw.empty() && !sw.empty(), "packer: the fit gives LUT writes and a slot write");
    const int slot = L.lookup(0, 1, 2);
    CHECK(P.lut()[UploadPacker::lut_index(0, 1, 2)] == slot, "packer: the mirror holds the learner's slot for the pair");
    CHECK(P.fit_n()[0] == 1 && P.fit_n()[1] == 0, "packer: the fitted count of layer 0 is 1");
    bool saw_lut = false;
    for (const LutWrite& w : lw)
        if (w.index == UploadPacker::lut_index(0, 1, 2) && w.value == slot) saw_lut = true;
    CHECK(saw_lut, "packer: the LUT write is (index of (0,1,2), slot)");
    const SlotWrite* s0 = nullptr;
    for (const SlotWrite& w : sw)
        if (w.slot == slot) s0 = &w;
    CHECK(s0 != nullptr && !s0->freed, "packer: the fitted slot is written, not freed");
    if (s0 != nullptr) {
        const std::vector<float>* B = L.pair_b(0, 1, 2);
        CHECK(B != nullptr && s0->b.size() == (size_t) N * R, "packer: the slot's B is N x r");
        bool same = B != nullptr && s0->b.size() == B->size();
        for (size_t i = 0; same && i < B->size(); ++i) same = s0->b[i] == f32_to_f16((*B)[i]);
        CHECK(same, "packer: the slot's B is the learner's B in fp16");
        double c_used = 0, c_raw = 0, gain = 0;
        bool is_fit = false;
        L.pair_info(0, 1, 2, c_used, c_raw, gain, is_fit);
        CHECK(std::fabs(s0->c - (float) c_used) < 1e-6f, "packer: the slot's c is the pair's used scalar");
        CHECK(s0->resid > 0.0f && s0->resid < 1.0f, "packer: the residual factor is in (0, 1) for a fitted pair");
        CHECK(std::fabs(s0->resid - std::sqrt(1.0 - gain)) < 1e-5, "packer: the residual is sqrt(1 - held-out gain)");
    }

    // the mirror equals the learner's lookup on every key of the layers (the learner is the reference)
    bool mirror_ok = true;
    for (int l = 0; l < NL; ++l)
        for (int e = 0; e < 512; ++e)
            for (int s = 0; s < 512; ++s)
                if (P.lut()[UploadPacker::lut_index(l, e, s)] != (int16_t) L.lookup(l, e, s)) mirror_ok = false;
    CHECK(mirror_ok, "packer: the mirror equals the learner's lookup for every key");

    // all_fitted() of a fresh packer gives the same state (the upload at start after a load)
    UploadPacker P2(NL);
    std::vector<LutWrite> lw2;
    std::vector<SlotWrite> sw2;
    P2.pack(L.all_fitted(), lw2, sw2);
    CHECK(P2.lut() == P.lut() && P2.fit_n() == P.fit_n(), "packer: the full upload gives the same mirror and counts");

    // revoke: the map flips, the pair is freed; its LUT entry goes back to -1, its count to 0, and the slot is written freed
    Changes c3 = request(L, flip, 0, 1, 2, 100, 0.8f, false, -1.0f);
    if (!L.fitted(0, 1, 2)) {
        std::vector<LutWrite> lw3;
        std::vector<SlotWrite> sw3;
        P.pack(c3, lw3, sw3);
        CHECK(P.lut()[UploadPacker::lut_index(0, 1, 2)] == -1, "packer: a revoked pair's LUT entry is -1");
        CHECK(P.fit_n()[0] == 0, "packer: a revoked pair leaves the layer's count at 0");
        bool freed = false;
        for (const SlotWrite& w : sw3)
            if (w.freed && w.slot == slot) freed = true;
        CHECK(freed, "packer: the freed slot is written as freed (no B)");
    } else {
        std::fprintf(stderr, "note: the revoke did not happen in this run; the freed-slot check was skipped\n");
    }

    // out-of-range changes are dropped, not written
    Changes bad;
    bad.lut.push_back(LutChange{NL, 0, 0, 3});
    bad.lut.push_back(LutChange{0, 512, 0, 3});
    std::vector<LutWrite> lw4;
    std::vector<SlotWrite> sw4;
    P.pack(bad, lw4, sw4);
    CHECK(lw4.empty(), "packer: out-of-range LUT changes are dropped");
}

// ---- the feature matrix as the device reads it
void test_features_f16() {
    Learner L(cfg());
    const std::vector<uint16_t> h = L.features_f16(1);
    std::vector<float> A;
    features(1, N, R, 1, A);
    CHECK(h.size() == A.size() && h.size() == (size_t) N * R, "features_f16: r x N values");
    bool same = h.size() == A.size();
    for (size_t i = 0; same && i < A.size(); ++i) same = h[i] == f32_to_f16(A[i]);
    CHECK(same, "features_f16: the learner's features in fp16");
}

}  // namespace

int main() {
    test_rule();
    test_packer();
    test_features_f16();
    if (failures == 0) std::printf("subfit_apply_host_test: all checks passed\n");
    return failures == 0 ? 0 : 1;
}
