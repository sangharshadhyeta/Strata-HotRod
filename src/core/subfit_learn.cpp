// src/core/subfit_learn.cpp - SPEC-substitute-fit stage 2, step 1: the live per-pair learner (see the header for the model and the gate).
#include "strata/core/subfit_learn.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::core::subfit_learn {

namespace {

constexpr uint32_t kFileMagic = 0x4e4c4653u;   // bytes "SFLN" in file order (little endian)
constexpr uint32_t kFileVersion = 1;

uint64_t splitmix(uint64_t& s) {
    uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

double uniform_open(uint64_t& s) { return ((double) (splitmix(s) >> 11) + 0.5) * (1.0 / 9007199254740992.0); }

// --- file helpers (little endian, the host's layout; the file is read back by the same engine)
void put(std::vector<uint8_t>& o, const void* p, size_t n) {
    const uint8_t* b = static_cast<const uint8_t*>(p);
    o.insert(o.end(), b, b + n);
}
template <class T> void put_v(std::vector<uint8_t>& o, T v) { put(o, &v, sizeof v); }
void put_f32s(std::vector<uint8_t>& o, const std::vector<float>& v) { put(o, v.data(), v.size() * sizeof(float)); }
void put_f64s(std::vector<uint8_t>& o, const double* v, size_t n) { put(o, v, n * sizeof(double)); }

struct Reader {
    const uint8_t* p;
    size_t left;
    bool ok = true;
    void take(void* dst, size_t n) {
        if (!ok || left < n) {
            ok = false;
            return;
        }
        std::memcpy(dst, p, n);
        p += n;
        left -= n;
    }
    template <class T> T get() {
        T v{};
        take(&v, sizeof v);
        return v;
    }
    void get_f32s(std::vector<float>& v, size_t n) {
        v.assign(n, 0.0f);
        take(v.data(), n * sizeof(float));
    }
    void get_f64s(double* v, size_t n) { take(v, n * sizeof(double)); }
};

}  // namespace

Apply apply_env(const char* v) {
    if (v == nullptr) return Apply::Dynamic;
    if (std::strcmp(v, "off") == 0) return Apply::Off;
    if (std::strcmp(v, "all") == 0) return Apply::All;
    if (std::strcmp(v, "flip") == 0) return Apply::Flip;
    return Apply::Dynamic;
}

const char* apply_name(Apply a) {
    switch (a) {
        case Apply::Off: return "off";
        case Apply::All: return "all";
        case Apply::Flip: return "flip";
        default: return "dynamic";
    }
}

int rank_env(const char* v) {
    if (v == nullptr) return kDefaultRank;
    const int r = std::atoi(v);
    if (r <= 0) return kDefaultRank;
    return std::min(32, std::max(1, r));
}

double gain_env(const char* v, double dflt) {
    if (v == nullptr || *v == 0) return dflt;
    const double g = std::atof(v);
    return std::isfinite(g) ? std::max(-1.0, std::min(1.0, g)) : dflt;
}

bool ladder_env(const char* v) { return v == nullptr || std::atoi(v) != 0; }

int pairs_env(const char* v) {
    if (v == nullptr) return kDefaultPairs;
    const int p = std::atoi(v);
    if (p <= 0) return kDefaultPairs;
    return std::min(8192, std::max(1, p));
}

// The fitted pair's residual factor: sqrt(E1/E0) = sqrt(1 - gain) over the held-out window (gain 0 while the window is short: 1).
static float residual_factor(double gain) { return (float) std::sqrt(std::max(0.0, 1.0 - gain)); }

uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof x);
    const uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t mant = x & 0x7fffffu;
    const int32_t exp = (int32_t) ((x >> 23) & 0xffu);
    if (exp == 0xff) return (uint16_t) (sign | 0x7c00u | (mant ? 0x200u : 0u));
    const int32_t e = exp - 127 + 15;
    if (e >= 31) return (uint16_t) (sign | 0x7c00u);
    if (e <= 0) {
        if (e < -10) return (uint16_t) sign;
        mant |= 0x800000u;
        const int shift = 14 - e;   // 13 + (1 - e)
        uint32_t m = mant >> shift;
        const uint32_t rem = mant & ((1u << shift) - 1u), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (m & 1u))) ++m;
        return (uint16_t) (sign | m);
    }
    uint32_t h = sign | ((uint32_t) e << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;   // a carry into the exponent is the right value
    return (uint16_t) h;
}

float f16_to_f32(uint16_t h) {
    const uint32_t sign = ((uint32_t) h & 0x8000u) << 16;
    const uint32_t e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
    uint32_t x;
    if (e == 0) {
        if (m == 0) {
            x = sign;
        } else {   // subnormal: normalise
            int32_t ee = -1;
            uint32_t mm = m;
            while (!(mm & 0x400u)) {
                mm <<= 1;
                --ee;
            }
            mm &= 0x3ffu;
            // value = (1.f) * 2^(ee - 13): the float's biased exponent is ee - 13 + 127
            x = sign | (uint32_t) (ee - 13 + 127) << 23 | mm << 13;
        }
    } else if (e == 31) {
        x = sign | 0x7f800000u | (m << 13);
    } else {
        x = sign | (e + 127 - 15) << 23 | m << 13;
    }
    float f;
    std::memcpy(&f, &x, sizeof f);
    return f;
}

void features(int layer, int n, int r, uint64_t seed, std::vector<float>& A) {
    uint64_t s = seed * 0x100000001b3ULL + (uint64_t) (layer + 1) * 0x9e3779b97f4a7c15ULL;
    A.assign((size_t) r * (size_t) n, 0.0f);
    const double scale = 1.0 / std::sqrt((double) n);
    for (size_t i = 0; i < A.size(); i += 2) {
        const double u1 = uniform_open(s), u2 = uniform_open(s);
        const double rad = std::sqrt(-2.0 * std::log(u1)), th = 6.283185307179586 * u2;
        A[i] = (float) (rad * std::cos(th) * scale);
        if (i + 1 < A.size()) A[i + 1] = (float) (rad * std::sin(th) * scale);
    }
}

bool ridge_solve(const double* ZtZ, const float* ZtYe, const float* ZtYs, double c, int r, int n, double lambda_rel, std::vector<float>& B) {
    double trace = 0.0;
    for (int k = 0; k < r; ++k) trace += ZtZ[k * r + k];
    if (!(trace > 0.0)) return false;
    const double lam = lambda_rel * trace / (double) r;
    // Cholesky of ZtZ + lam I (lower, row-major)
    std::vector<double> L((size_t) r * r, 0.0);
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = ZtZ[i * r + j] + (i == j ? lam : 0.0);
            for (int k = 0; k < j; ++k) sum -= L[i * r + k] * L[j * r + k];
            if (i == j) {
                if (!(sum > 0.0)) return false;
                L[i * r + i] = std::sqrt(sum);
            } else {
                L[i * r + j] = sum / L[j * r + j];
            }
        }
    }
    // M = (ZtZ + lam I)^-1 by solving L L^T m = e_k for each unit vector
    std::vector<double> M((size_t) r * r, 0.0), y((size_t) r);
    for (int col = 0; col < r; ++col) {
        for (int i = 0; i < r; ++i) {
            double sum = (i == col) ? 1.0 : 0.0;
            for (int k = 0; k < i; ++k) sum -= L[i * r + k] * y[k];
            y[i] = sum / L[i * r + i];
        }
        for (int i = r - 1; i >= 0; --i) {
            double sum = y[i];
            for (int k = i + 1; k < r; ++k) sum -= L[k * r + i] * M[k * r + col];
            M[i * r + col] = sum / L[i * r + i];
        }
    }
    // B[j][k] = sum_m M[k][m] * (ZtYe[m][j] - c ZtYs[m][j])
    B.assign((size_t) n * r, 0.0f);
    std::vector<double> t((size_t) r);
    for (int j = 0; j < n; ++j) {
        for (int m = 0; m < r; ++m) t[(size_t) m] = (double) ZtYe[(size_t) m * n + j] - c * (double) ZtYs[(size_t) m * n + j];
        for (int k = 0; k < r; ++k) {
            double sum = 0.0;
            for (int m = 0; m < r; ++m) sum += M[(size_t) k * r + m] * t[(size_t) m];
            B[(size_t) j * r + k] = (float) sum;
        }
    }
    return true;
}

Learner::Learner(const Config& cfg) : cfg_(cfg) {
    cfg_.n = std::max(1, cfg_.n);
    cfg_.rank = std::min(32, std::max(1, cfg_.rank));
    cfg_.max_pairs = std::max(1, cfg_.max_pairs);
    cfg_.max_fitted = std::min(kMaxFitted, std::max(1, cfg_.max_fitted));
    cfg_.n_layers = std::max(1, cfg_.n_layers);
    slot_used_.assign((size_t) cfg_.max_fitted, false);
    cfg_.max_experts = std::max(1, cfg_.max_experts);
    if (cfg_.gain_revoke >= cfg_.gain_fit) cfg_.gain_revoke = cfg_.gain_fit - 0.05;   // a fit must not flap at the bar
    lut_.assign(table_size(cfg_.n_layers), (int16_t) -1);
    A_.assign((size_t) cfg_.n_layers, {});
}

const std::vector<float>& Learner::features_of(int layer) {
    std::vector<float>& A = A_[(size_t) layer];
    if (A.empty()) features(layer, cfg_.n, cfg_.rank, cfg_.seed, A);
    return A;
}

Learner::Pair* Learner::find(int layer, int e, int s) {
    auto it = pairs_.find(key(layer, e, s));
    return it == pairs_.end() ? nullptr : &it->second;
}

const Learner::Pair* Learner::find(int layer, int e, int s) const {
    auto it = pairs_.find(key(layer, e, s));
    return it == pairs_.end() ? nullptr : &it->second;
}

void Learner::set_lut(const Pair& p, int slot, Changes* ch) {
    const size_t idx = table_index(cfg_.n_layers, p.layer, p.e, p.s);
    lut_[idx] = (int16_t) slot;
    if (ch != nullptr) ch->lut.push_back(LutChange{p.layer, p.e, p.s, slot});
}

SlotChange Learner::slot_change(const Pair& p, int slot) const {
    SlotChange sc;
    sc.slot = slot;
    sc.layer = p.layer;
    sc.e = p.e;
    sc.s = p.s;
    sc.c = (float) p.c_used;
    sc.resid = residual_factor(gain_recent(p));
    sc.b.resize(p.B.size());
    for (size_t i = 0; i < p.B.size(); ++i) sc.b[i] = f32_to_f16(p.B[i]);
    return sc;
}

void Learner::free_slot(Pair& p, Changes* ch) {
    if (p.slot < 0) return;
    slot_used_[(size_t) p.slot] = false;
    set_lut(p, -1, ch);
    if (ch != nullptr) {
        SlotChange sc;
        sc.slot = p.slot;
        sc.layer = p.layer;
        sc.e = p.e;
        sc.s = p.s;
        ch->slots.push_back(sc);   // empty b: the slot is freed
    }
    p.slot = -1;
}

bool Learner::assign_slot(Pair& p, Changes* ch) {
    if (p.slot >= 0) return true;
    if (!p.has_b) return false;
    for (int i = 0; i < cfg_.max_fitted; ++i) {
        if (slot_used_[(size_t) i]) continue;
        slot_used_[(size_t) i] = true;
        p.slot = i;
        set_lut(p, i, ch);
        if (ch != nullptr) ch->slots.push_back(slot_change(p, i));
        return true;
    }
    return false;
}

void Learner::evict_one(int level, Changes* ch) {
    auto victim = pairs_.end();
    for (auto it = pairs_.begin(); it != pairs_.end(); ++it) {
        if (level_of(it->second.e, it->second.s) != level) continue;
        if (victim == pairs_.end() || it->second.last_tick < victim->second.last_tick) victim = it;
    }
    if (victim == pairs_.end()) return;
    free_slot(victim->second, ch);
    --level_n_[level];
    pairs_.erase(victim);
}

Learner::Pair& Learner::get_or_create(int layer, int e, int s, Changes* ch) {
    auto it = pairs_.find(key(layer, e, s));
    if (it != pairs_.end()) return it->second;
    const int lv = level_of(e, s);
    if (lv == 0 && (int) level_n_[0] >= cfg_.max_pairs) evict_one(0, ch);
    if (lv == 1 && (int) level_n_[1] >= cfg_.max_experts) evict_one(1, ch);
    ++level_n_[lv];
    Pair p;
    p.layer = layer;
    p.e = e;
    p.s = s;
    p.zz.assign((size_t) cfg_.rank * cfg_.rank, 0.0);
    p.zye.assign((size_t) cfg_.rank * cfg_.n, 0.0f);
    p.zys.assign((size_t) cfg_.rank * cfg_.n, 0.0f);
    return pairs_.emplace(key(layer, e, s), std::move(p)).first->second;
}

void Learner::push_ho(Pair& p, double e0, double e0s, double e1) {
    p.ring0[p.ring_pos] = e0;
    p.ring0s[p.ring_pos] = e0s;
    p.ring1[p.ring_pos] = e1;
    p.ring_pos = (p.ring_pos + 1) % kHoldWindow;
    if (p.ring_n < kHoldWindow) ++p.ring_n;
}

// The gate's gain: 1 - E1 / E0 over the last kHoldWindow held-out samples, with E0 the scalar-only baseline (see the Pair comment).
double Learner::gain_recent(const Pair& p) const {
    if (p.ring_n < kHoldWindow) return 0.0;
    double e0 = 0.0, e1 = 0.0;
    for (int i = 0; i < kHoldWindow; ++i) {
        e0 += p.ring0s[i];
        e1 += p.ring1[i];
    }
    return e0 > 0.0 ? 1.0 - e1 / e0 : 0.0;
}

// The record's gain: the same over E0 = |ye - ys|^2 as is (the spec's definition: it credits a shrink toward zero as well).
double Learner::gain_asis(const Pair& p) const {
    if (p.ring_n < kHoldWindow) return 0.0;
    double e0 = 0.0, e1 = 0.0;
    for (int i = 0; i < kHoldWindow; ++i) {
        e0 += p.ring0[i];
        e1 += p.ring1[i];
    }
    return e0 > 0.0 ? 1.0 - e1 / e0 : 0.0;
}

void Learner::add_sample(int layer, int e, int s, const float* x, const float* ys, const float* ye) {
    if (layer < 0 || layer >= cfg_.n_layers || e < 0 || e >= kExperts || s < 0 || s >= kExperts) return;
    const int n = cfg_.n, r = cfg_.rank;
    for (int j = 0; j < n; ++j)
        if (!std::isfinite(x[j]) || !std::isfinite(ys[j]) || !std::isfinite(ye[j])) return;
    const std::vector<float>& A = features_of(layer);
    std::vector<double> z((size_t) r, 0.0);
    for (int k = 0; k < r; ++k) {
        double acc = 0.0;
        const float* a = &A[(size_t) k * n];
        for (int j = 0; j < n; ++j) acc += (double) a[j] * (double) x[j];
        z[(size_t) k] = acc;
    }
    // the fallback ladder: the pair, its (layer, e) group and its layer group each learn from the sample (own held-out schedule)
    add_to(get_or_create(layer, e, s, &pending_), z, ys, ye);
    if (!cfg_.ladder) return;
    add_to(get_or_create(layer, e, kAny, &pending_), z, ys, ye);
    add_to(get_or_create(layer, kAny, kAny, &pending_), z, ys, ye);
}

void Learner::add_to(Pair& p, const std::vector<double>& z, const float* ys, const float* ye) {
    const int n = cfg_.n, r = cfg_.rank;
    p.last_tick = ++tick_;
    p.dirty = true;
    const uint64_t idx = p.seen++;
    if (idx % (uint64_t) kHoldEvery == (uint64_t) kHoldEvery - 1) {   // held out: the pair's current fit, not trained on
        double e0 = 0.0, e0s = 0.0, e1 = 0.0;
        for (int j = 0; j < n; ++j) {
            const double scalar = p.c_used * (double) ys[j];
            double pred = scalar;
            if (p.has_b) {
                const float* bj = &p.B[(size_t) j * r];
                for (int k = 0; k < r; ++k) pred += (double) bj[k] * z[(size_t) k];
            }
            const double d0 = (double) ye[j] - (double) ys[j], d0s = (double) ye[j] - scalar, d1 = (double) ye[j] - pred;
            e0 += d0 * d0;
            e0s += d0s * d0s;
            e1 += d1 * d1;
        }
        push_ho(p, e0, e0s, e1);
        p.E0 += e0;
        p.E1 += e1;
        ++p.n_ho;
        return;
    }
    ++p.n_train;
    double sxy = 0.0, sxx = 0.0;
    for (int j = 0; j < n; ++j) {
        sxy += (double) ys[j] * (double) ye[j];
        sxx += (double) ys[j] * (double) ys[j];
    }
    p.sxy += sxy;
    p.sxx += sxx;
    for (int k = 0; k < r; ++k) {
        for (int m = 0; m < r; ++m) p.zz[(size_t) k * r + m] += z[(size_t) k] * z[(size_t) m];
        float* zy = &p.zye[(size_t) k * n];
        float* zs = &p.zys[(size_t) k * n];
        const double zk = z[(size_t) k];
        for (int j = 0; j < n; ++j) {
            zy[j] += (float) (zk * (double) ye[j]);
            zs[j] += (float) (zk * (double) ys[j]);
        }
    }
}

UploadPacker::UploadPacker(int n_layers)
    : n_layers_(std::max(1, n_layers)),
      lut_(table_size(std::max(1, n_layers)), (int16_t) -1),
      fit_n_((size_t) std::max(1, n_layers), 0) {}

void UploadPacker::pack(const Changes& ch, std::vector<LutWrite>& lut_out, std::vector<SlotWrite>& slot_out) {
    for (const LutChange& c : ch.lut) {
        if (c.layer < 0 || c.layer >= n_layers_) continue;
        if (!(c.e == kAny ? c.s == kAny : (c.e >= 0 && c.e < kExperts && (c.s == kAny || (c.s >= 0 && c.s < kExperts))))) continue;
        const size_t idx = index(c.layer, c.e, c.s);
        const int16_t old = lut_[idx];
        if (old >= 0 && c.slot < 0) --fit_n_[(size_t) c.layer];
        if (old < 0 && c.slot >= 0) ++fit_n_[(size_t) c.layer];
        lut_[idx] = (int16_t) c.slot;
        lut_out.push_back(LutWrite{idx, (int16_t) c.slot});
    }
    for (const SlotChange& sc : ch.slots) {
        if (sc.slot < 0) continue;
        SlotWrite w;
        w.slot = sc.slot;
        w.freed = sc.b.empty();
        w.c = sc.c;
        w.resid = sc.resid;
        if (!w.freed) w.b = sc.b;
        slot_out.push_back(std::move(w));
    }
}

std::vector<uint16_t> Learner::features_f16(int layer) const {
    std::vector<float> A;
    features(layer, cfg_.n, cfg_.rank, cfg_.seed, A);
    std::vector<uint16_t> out(A.size());
    for (size_t i = 0; i < A.size(); ++i) out[i] = f32_to_f16(A[i]);
    return out;
}

void Learner::refit_pair(Pair& p, Changes* ch) {
    const int r = cfg_.rank, n = cfg_.n;
    bool changed = false;
    if (p.dirty) {
        if (p.sxx > 0.0) {
            p.c_raw = p.sxy / p.sxx;
            p.c_used = std::min(kScalarClip, std::max(-kScalarClip, p.c_raw));
            changed = true;
        }
        if (p.n_train >= (uint64_t) r + 1) {
            std::vector<float> B;
            if (ridge_solve(p.zz.data(), p.zye.data(), p.zys.data(), p.c_used, r, n, kLambdaRel, B)) {
                p.B = std::move(B);
                p.has_b = true;
                changed = true;
            }
        }
        p.dirty = false;
    }
    if (p.n_ho >= (uint64_t) kHoldWindow) {
        const double g = gain_recent(p);
        if (!p.is_fitted && g >= cfg_.gain_fit) {
            p.is_fitted = true;
            changed = true;
        } else if (p.is_fitted && g < cfg_.gain_revoke) {
            p.is_fitted = false;
            free_slot(p, ch);
            return;
        }
    }
    if (!p.is_fitted) return;
    if (p.slot < 0) {
        assign_slot(p, ch);   // a fitted pair with no slot (the table was full) takes a free one
        return;
    }
    if (changed && ch != nullptr) {
        ch->slots.push_back(slot_change(p, p.slot));
    }
}

Changes Learner::refit() {
    // refit in key order so the slot assignment is deterministic across runs
    std::vector<uint64_t> keys;
    keys.reserve(pairs_.size());
    for (const auto& kv : pairs_)
        if (kv.second.dirty || (kv.second.is_fitted && kv.second.slot < 0)) keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end());
    for (uint64_t k : keys) {
        auto it = pairs_.find(k);
        if (it != pairs_.end()) refit_pair(it->second, &pending_);
    }
    Changes out = std::move(pending_);
    pending_ = Changes{};
    return out;
}

Changes Learner::all_fitted() const {
    Changes out;
    for (const auto& kv : pairs_) {
        const Pair& p = kv.second;
        if (!p.is_fitted || p.slot < 0) continue;
        out.slots.push_back(slot_change(p, p.slot));
        out.lut.push_back(LutChange{p.layer, p.e, p.s, p.slot});
    }
    return out;
}

int Learner::lookup(int layer, int e, int s) const {
    if (layer < 0 || layer >= cfg_.n_layers || e < 0 || e >= kExperts || s < 0 || s >= kExperts) return -1;
    int16_t v = lut_[table_index(cfg_.n_layers, layer, e, s)];
    if (v < 0) v = lut_[table_index(cfg_.n_layers, layer, e, kAny)];
    if (v < 0) v = lut_[table_index(cfg_.n_layers, layer, kAny, kAny)];
    return v;
}

int Learner::level_used(int layer, int e, int s) const {
    if (layer < 0 || layer >= cfg_.n_layers || e < 0 || e >= kExperts || s < 0 || s >= kExperts) return -1;
    if (lut_[table_index(cfg_.n_layers, layer, e, s)] >= 0) return 0;
    if (lut_[table_index(cfg_.n_layers, layer, e, kAny)] >= 0) return 1;
    if (lut_[table_index(cfg_.n_layers, layer, kAny, kAny)] >= 0) return 2;
    return -1;
}

double Learner::gain_as_is(int layer, int e, int s) const {
    const Pair* p = find(layer, e, s);
    return p != nullptr ? gain_asis(*p) : 0.0;
}

bool Learner::fitted(int layer, int e, int s) const {
    const Pair* p = find(layer, e, s);
    return p != nullptr && p->is_fitted;
}

bool Learner::pair_info(int layer, int e, int s, double& c_used, double& c_raw, double& gain, bool& is_fitted) const {
    const Pair* p = find(layer, e, s);
    if (p == nullptr) return false;
    c_used = p->c_used;
    c_raw = p->c_raw;
    gain = gain_recent(*p);
    is_fitted = p->is_fitted;
    return true;
}

const std::vector<float>* Learner::pair_b(int layer, int e, int s) const {
    const Pair* p = find(layer, e, s);
    return p != nullptr && p->has_b ? &p->B : nullptr;
}

size_t Learner::pairs_with_holdout(int min_ho) const {
    size_t n = 0;
    for (const auto& kv : pairs_)
        if (kv.second.n_ho >= (uint64_t) std::max(0, min_ho)) ++n;
    return n;
}

size_t Learner::slots_used() const {
    size_t n = 0;
    for (bool u : slot_used_) n += u ? 1u : 0u;
    return n;
}

Summary Learner::summary() const {
    Summary s;
    s.pairs = pairs_.size();
    std::vector<double> gains, glv[3], rlv[3];
    for (const auto& kv : pairs_) {
        const Pair& p = kv.second;
        const int lv = level_of(p.e, p.s);
        ++s.groups[lv];
        if (p.ring_n >= kHoldWindow) rlv[lv].push_back(gain_recent(p));
        if (!p.is_fitted) continue;
        ++s.fitted;
        ++s.fitted_lv[lv];
        if (p.slot < 0) ++s.unslotted;
        gains.push_back(gain_recent(p));
        glv[lv].push_back(gain_recent(p));
    }
    for (int lv = 0; lv < 3; ++lv)
        if (!rlv[lv].empty()) {
            std::sort(rlv[lv].begin(), rlv[lv].end());
            s.ready_lv[lv] = rlv[lv].size();
            s.ready_median_lv[lv] = rlv[lv][rlv[lv].size() / 2];
            s.ready_best_lv[lv] = rlv[lv].back();
        }
    for (int lv = 0; lv < 3; ++lv)
        if (!glv[lv].empty()) {
            std::sort(glv[lv].begin(), glv[lv].end());
            s.gain_median_lv[lv] = glv[lv][glv[lv].size() / 2];
        }
    if (!gains.empty()) {
        std::sort(gains.begin(), gains.end());
        const size_t m = gains.size();
        s.gain_median = m % 2 == 1 ? gains[m / 2] : 0.5 * (gains[m / 2 - 1] + gains[m / 2]);
        s.gain_min = gains.front();
    }
    return s;
}

// File: "SFLN", version, fingerprint (u64), n, r, n_layers (u32), tick (u64), count (u32); then per pair: layer, e, s (i32), last_tick,
// seen, n_train, n_ho (u64), sxy, sxx, c_raw, c_used, E0, E1 (f64), ring_pos, ring_n (i32), ring0, ring1 (kHoldWindow f64 each),
// is_fitted, has_b (u8), slot (i32), zz (r*r f64), zye, zys (r*n f32), B (n*r f32 when has_b).
bool Learner::save(const std::string& path, uint64_t fp, std::string& err) const {
    std::vector<uint8_t> o;
    put_v(o, kFileMagic);
    put_v(o, kFileVersion);
    put_v(o, fp);
    put_v(o, (uint32_t) cfg_.n);
    put_v(o, (uint32_t) cfg_.rank);
    put_v(o, (uint32_t) cfg_.n_layers);
    put_v(o, tick_);
    put_v(o, (uint32_t) pairs_.size());
    const int r = cfg_.rank;
    for (const auto& kv : pairs_) {
        const Pair& p = kv.second;
        put_v(o, (int32_t) p.layer);
        put_v(o, (int32_t) p.e);
        put_v(o, (int32_t) p.s);
        put_v(o, p.last_tick);
        put_v(o, p.seen);
        put_v(o, p.n_train);
        put_v(o, p.n_ho);
        put_v(o, p.sxy);
        put_v(o, p.sxx);
        put_v(o, p.c_raw);
        put_v(o, p.c_used);
        put_v(o, p.E0);
        put_v(o, p.E1);
        put_v(o, (int32_t) p.ring_pos);
        put_v(o, (int32_t) p.ring_n);
        put_f64s(o, p.ring0, kHoldWindow);
        put_f64s(o, p.ring0s, kHoldWindow);
        put_f64s(o, p.ring1, kHoldWindow);
        put_v(o, (uint8_t) (p.is_fitted ? 1 : 0));
        put_v(o, (uint8_t) (p.has_b ? 1 : 0));
        put_v(o, (int32_t) p.slot);
        put_f64s(o, p.zz.data(), (size_t) r * r);
        put_f32s(o, p.zye);
        put_f32s(o, p.zys);
        if (p.has_b) put_f32s(o, p.B);
    }
    const std::string tmp = path + ".tmp";
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (f == nullptr) {
        err = "subfit learn: cannot write " + tmp;
        return false;
    }
    const bool ok = std::fwrite(o.data(), 1, o.size(), f) == o.size();
    std::fclose(f);
    if (!ok || std::rename(tmp.c_str(), path.c_str()) != 0) {
        err = "subfit learn: cannot finish " + path;
        std::remove(tmp.c_str());
        return false;
    }
    return true;
}

bool Learner::load(const std::string& path, uint64_t fp, std::string& err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = "no saved learner state";
        return false;
    }
    std::vector<uint8_t> buf;
    uint8_t chunk[1 << 14];
    for (size_t k; (k = std::fread(chunk, 1, sizeof chunk, f)) > 0;) buf.insert(buf.end(), chunk, chunk + k);
    std::fclose(f);
    Reader rd{buf.data(), buf.size()};
    if (rd.get<uint32_t>() != kFileMagic || rd.get<uint32_t>() != kFileVersion) {
        err = "not a subfit learner file";
        return false;
    }
    const uint64_t file_fp = rd.get<uint64_t>();
    const uint32_t n = rd.get<uint32_t>(), r = rd.get<uint32_t>(), nl = rd.get<uint32_t>();
    if (file_fp != fp || n != (uint32_t) cfg_.n || r != (uint32_t) cfg_.rank || nl != (uint32_t) cfg_.n_layers) {
        err = "subfit learner file of another model or shape; not loaded";
        return false;
    }
    const uint64_t tick = rd.get<uint64_t>();
    const uint32_t count = rd.get<uint32_t>();
    // a fresh state: everything is rebuilt from the file
    std::unordered_map<uint64_t, Pair> pairs;
    std::vector<bool> used((size_t) cfg_.max_fitted, false);
    for (uint32_t i = 0; i < count && rd.ok; ++i) {
        Pair p;
        p.layer = rd.get<int32_t>();
        p.e = rd.get<int32_t>();
        p.s = rd.get<int32_t>();
        p.last_tick = rd.get<uint64_t>();
        p.seen = rd.get<uint64_t>();
        p.n_train = rd.get<uint64_t>();
        p.n_ho = rd.get<uint64_t>();
        p.sxy = rd.get<double>();
        p.sxx = rd.get<double>();
        p.c_raw = rd.get<double>();
        p.c_used = rd.get<double>();
        p.E0 = rd.get<double>();
        p.E1 = rd.get<double>();
        p.ring_pos = rd.get<int32_t>();
        p.ring_n = rd.get<int32_t>();
        rd.get_f64s(p.ring0, kHoldWindow);
        rd.get_f64s(p.ring0s, kHoldWindow);
        rd.get_f64s(p.ring1, kHoldWindow);
        p.is_fitted = rd.get<uint8_t>() != 0;
        p.has_b = rd.get<uint8_t>() != 0;
        p.slot = rd.get<int32_t>();
        p.zz.assign((size_t) r * r, 0.0);
        rd.get_f64s(p.zz.data(), (size_t) r * r);
        rd.get_f32s(p.zye, (size_t) r * n);
        rd.get_f32s(p.zys, (size_t) r * n);
        if (p.has_b) rd.get_f32s(p.B, (size_t) n * r);
        if (!rd.ok || p.layer < 0 || p.layer >= cfg_.n_layers || !(p.e == kAny ? p.s == kAny : (p.e >= 0 && p.e < kExperts && (p.s == kAny || (p.s >= 0 && p.s < kExperts)))) ||
            p.ring_pos < 0 || p.ring_pos >= kHoldWindow || p.ring_n < 0 || p.ring_n > kHoldWindow) {
            err = "subfit learner file is malformed; not loaded";
            return false;
        }
        if (p.slot >= 0) {
            if (p.slot >= cfg_.max_fitted || used[(size_t) p.slot]) {
                err = "subfit learner file has a bad slot table; not loaded";
                return false;
            }
            used[(size_t) p.slot] = true;
        }
        pairs.emplace(key(p.layer, p.e, p.s), std::move(p));
    }
    if (!rd.ok) {
        err = "subfit learner file is truncated; not loaded";
        return false;
    }
    // commit
    pairs_ = std::move(pairs);
    tick_ = tick;
    slot_used_ = used;
    std::fill(lut_.begin(), lut_.end(), (int16_t) -1);
    level_n_[0] = level_n_[1] = level_n_[2] = 0;
    for (const auto& kv : pairs_) {
        const Pair& p = kv.second;
        ++level_n_[level_of(p.e, p.s)];
        if (p.slot >= 0) lut_[table_index(cfg_.n_layers, p.layer, p.e, p.s)] = (int16_t) p.slot;
    }
    pending_ = Changes{};
    return true;
}

}  // namespace strata::core::subfit_learn
