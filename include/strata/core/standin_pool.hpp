// include/strata/core/standin_pool.hpp - STRATA_STANDIN_LEARN's host-side estimates (no CUDA).
//
// A pair (layer, true expert e, stand-in s) keeps decayed least-squares sums over its samples (num = y_s.y_e, den = y_s.y_s,
// ee = y_e.y_e).  The same sums are kept per group: (layer, e) and per layer.  A pair's estimate adds a prior from its parent
// group: the expert group when it has >= kReady samples, else the layer group, else no prior.  The prior is the parent's mean
// sums (its sums over its decayed weight w) times a weight of K*K/(K+n) pseudo-samples, where n is the pair's own count: 16
// pseudo-samples for a pair with one sample, fading as its own samples grow (with the decay the pair's own sums weigh at most
// about 50, so a fixed 16 would keep a quarter of the parent's mean forever).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <utility>
#include <vector>

namespace strata::standin {
constexpr int kReady = 32;          // samples a pair (or a group) needs before its estimate is used
constexpr double kShrinkK = 16.0;   // the prior's pseudo-samples at n = 0
constexpr double kDecay = 0.98;     // every sum decays by this per sample
constexpr int kPerLayer = 4096;     // pairs kept per layer (least recently used goes)
constexpr uint32_t kMagic = 0x4C535453u;   // "STSL"
constexpr int kExperts = 512;

struct Pair {
    double num = 0.0, den = 0.0, ee = 0.0;
    int n = 0;                       // raw samples
    float c = 1.0f;                  // the own estimate (as the corrections table used it before pooling)
    uint64_t used = 0;               // LRU tick
};

// A group's decayed sums.  w is the decayed sample weight (the mean is the sums over w); n the raw sample count.
struct Sums {
    double num = 0.0, den = 0.0, ee = 0.0, w = 0.0;
    int64_t n = 0;
};

inline uint64_t pair_key(int64_t l, int64_t e, int64_t s) {
    return ((uint64_t) l << 40) | ((uint64_t) e << 20) | (uint64_t) s;
}
inline uint64_t expert_key(int64_t l, int64_t e) { return ((uint64_t) l << 20) | (uint64_t) e; }

// One sample into a group (the same decay as a pair's sums).
inline void add_sample(Sums& g, double sxy, double sxx, double syy) {
    g.num = kDecay * g.num + sxy;
    g.den = kDecay * g.den + sxx;
    g.ee = kDecay * g.ee + syy;
    g.w = kDecay * g.w + 1.0;
    ++g.n;
}

// The decayed weight of n unit samples: what a pair (or a rebuilt group) holds for its count.
inline double decayed_weight(int64_t n) { return (1.0 - std::pow(kDecay, (double) n)) / (1.0 - kDecay); }

struct Est {
    bool ready = false;    // usable: the parent group exists, or the pair's own count reached kReady (and ee > 0)
    bool pooled = false;   // the estimate took a parent's prior
    double c = 1.0;        // clipped to [0.5, 2]
    double err = 0.0;      // sqrt((ee - 2 c num + c^2 den) / ee) over the shrunk sums
};

// The pair's estimate.  expert: the (layer, e) group, layer: the layer group (either may be null).
inline Est estimate(const Pair& p, const Sums* expert, const Sums* layer) {
    Est r;
    const Sums* par = nullptr;
    if (expert != nullptr && expert->n >= kReady) par = expert;
    else if (layer != nullptr && layer->n >= kReady) par = layer;
    double num = p.num, den = p.den, ee = p.ee;
    if (par != nullptr && par->w > 0.0) {
        const double prior = kShrinkK * kShrinkK / (kShrinkK + (double) p.n);
        num += prior * par->num / par->w;
        den += prior * par->den / par->w;
        ee += prior * par->ee / par->w;
        r.pooled = true;
    }
    r.ready = (r.pooled || p.n >= kReady) && ee > 0.0;
    r.c = std::min(2.0, std::max(0.5, num / std::max(den, 1e-30)));
    if (ee > 0.0) {
        double e2 = (ee - 2.0 * r.c * num + r.c * r.c * den) / ee;
        if (!(e2 > 0.0)) e2 = 0.0;
        r.err = std::sqrt(e2);
    }
    return r;
}

// The groups from the pairs: each pair's sums go to its (layer, e) group and its layer.  Replaces the groups.
template <class Map>
inline void rebuild_groups(const Map& pairs, std::vector<Sums>& layer_g, std::unordered_map<uint64_t, Sums>& exp_g) {
    for (auto& g : layer_g) g = Sums{};
    exp_g.clear();
    const int64_t nl = (int64_t) layer_g.size();
    for (const auto& kv : pairs) {
        const int64_t l = (int64_t) (kv.first >> 40), e = (int64_t) ((kv.first >> 20) & 0xFFFFFu);
        if (l < 0 || l >= nl || e >= kExperts) continue;
        const Pair& p = kv.second;
        Sums add;
        add.num = p.num;
        add.den = p.den;
        add.ee = p.ee;
        add.w = decayed_weight(p.n);
        add.n = p.n;
        for (Sums* g : {&layer_g[(size_t) l], &exp_g[expert_key(l, e)]}) {
            g->num += add.num;
            g->den += add.den;
            g->ee += add.ee;
            g->w += add.w;
            g->n += add.n;
        }
    }
}

// The balance's state step for every layer (one round).  best[i]: the best err of expert i's ready pairs (< 0: none ready);
// use[i]: the table's use factor; miss_dec[i]: the decayed demand misses.  Layers with fewer than 10 ranked experts make no
// decision.  Value = use x best err; cost = its misses + 1.  Top decile desires +1, bottom decile -1; a state moves one step a
// round, and leaves +1 / -1 for 0 only after 2 rounds on the other side.  Experts outside the ranking keep their state.
inline void balance_step(std::vector<int8_t>& state, std::vector<uint8_t>& opp, const std::vector<double>& best,
                         const std::vector<float>& use, const std::vector<float>& miss_dec, int64_t layers) {
    for (int64_t l = 0; l < layers; ++l) {
        std::vector<std::pair<double, int>> rk;
        for (int e = 0; e < kExperts; ++e) {
            const int64_t i = l * kExperts + e;
            if (best[(size_t) i] < 0.0 || !(use[(size_t) i] > 0.0f)) continue;
            const double value = (double) use[(size_t) i] * best[(size_t) i];
            rk.emplace_back(value / ((double) miss_dec[(size_t) i] + 1.0), e);
        }
        if (rk.size() < 10) continue;
        std::sort(rk.begin(), rk.end(), [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
            return a.first != b.first ? a.first > b.first : a.second < b.second;
        });
        const size_t k = std::max<size_t>(1, rk.size() / 10);
        for (size_t pos = 0; pos < rk.size(); ++pos) {
            const int d = pos < k ? 1 : pos + k >= rk.size() ? -1 : 0;
            const size_t i = (size_t) l * kExperts + (size_t) rk[pos].second;
            int8_t& s = state[i];
            uint8_t& op = opp[i];
            if (d == 0 || d == s) op = 0;
            else if (s == 0) { s = (int8_t) d; op = 0; }
            else if (++op >= 2) { s = 0; op = 0; }   // the opposite side for 2 rounds: back to 0
        }
    }
}

struct Loaded {
    uint64_t key = 0;
    Pair p;
};

// The saved pairs: "STSL", layers, experts, count (u32 each), then per pair key (u64), num, den, ee (f64), n (u32), c (f32).
// False when the header is not this model's (layers, 512 experts).  A record that does not fit the bytes, or whose ids or c
// are out of range, is skipped (a short file stops at its last whole record).
inline bool parse_table(const uint8_t* d, size_t size, int64_t layers, std::vector<Loaded>& out) {
    if (size < 16) return false;
    uint32_t hd[4] = {0, 0, 0, 0};
    std::memcpy(hd, d, sizeof hd);
    if (hd[0] != kMagic || (int64_t) hd[1] != layers || hd[2] != (uint32_t) kExperts) return false;
    size_t pos = 16;
    for (uint32_t i = 0; i < hd[3]; ++i) {
        if (size - pos < 40) break;
        Loaded L;
        uint32_t n = 0;
        std::memcpy(&L.key, d + pos, 8);
        std::memcpy(&L.p.num, d + pos + 8, 8);
        std::memcpy(&L.p.den, d + pos + 16, 8);
        std::memcpy(&L.p.ee, d + pos + 24, 8);
        std::memcpy(&n, d + pos + 32, 4);
        std::memcpy(&L.p.c, d + pos + 36, 4);
        pos += 40;
        L.p.n = (int) n;
        const int64_t l = (int64_t) (L.key >> 40), e = (int64_t) ((L.key >> 20) & 0xFFFFFu), s = (int64_t) (L.key & 0xFFFFFu);
        if (l >= layers || e >= kExperts || s >= kExperts || !std::isfinite(L.p.c)) continue;
        out.push_back(L);
    }
    return true;
}

}  // namespace strata::standin
