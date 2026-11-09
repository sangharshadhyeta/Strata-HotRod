// include/strata/core/subfit_learn.hpp - SPEC-substitute-fit stage 2, step 1: the live learner of the per-pair corrections.  CPU only:
// no CUDA here.  The device side (the apply kernel and the table upload) takes the Changes this learner returns, between windows.
//
// Per (layer l, missing expert e, substitute s) pair, from the samples standin_learn_settle gets (x = the row's MoE input, ys = the
// substitute's output, ye = the exact output; all N wide):
//   - features z = A_l x, A_l a fixed seeded Gaussian r x N matrix per layer (scaled 1/sqrt(N); recomputable, not saved);
//   - scalar c = sxy / sxx of (ys, ye) over the training samples (recorded unclipped; used clipped to [-4, 4]);
//   - residual target t = ye - c ys; ridge fit B (N x r) from the streaming sums (ZtZ double, ZtYe and ZtYs kept as float sums so
//     the target is recomputed with the current c), lambda = 0.1 trace(ZtZ) / r, solved by Cholesky;
//   - correction y = c ys + B z.
// Held-out: every 4th sample of a pair (its seen count mod 4 == 3) is not trained on; with the pair's current fit it adds
// E1 += |ye - (c ys + B z)|^2 and the baseline E0 = |ye - c ys|^2 (the scalar alone).  A pair is fitted when its last 32 held-out
// samples give a gain 1 - E1/E0 >= 0.20, and revoked when that gain falls below 0.10.  The gate is evaluated at refit (the request end).
// DEVIATION from the spec's E0 = |ye - ys|^2: that baseline credits a shrink toward zero, and an uncorrelated pair's scalar goes to ~0,
// which halves its error - the gate then accepted a pure-noise pair (the test). The as-is gain is still kept and reported
// (gain_as_is).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::core::subfit_learn {

constexpr int kExperts = 512;                 ///< experts per layer (the LUT's side)
constexpr int kDefaultRank = 16;              ///< STRATA_SUBFIT_RANK default
constexpr int kDefaultPairs = 256;            ///< STRATA_SUBFIT_PAIRS default: pair groups (0.5 MB of statistics each on a 31 GB box)
constexpr int kDefaultExperts = 256;          ///< (layer, expert) groups kept (the ladder's middle level; least recently sampled evicted)
/// The fallback ladder (9 Nov): every sample also feeds its (layer, missing expert) group and its layer group.  A group is keyed
/// like a pair with kAny in place of s (expert level) or of both e and s (layer level).  At apply time a substituted slot uses the
/// most specific FITTED level: the pair's slot, else its (layer, e) slot, else its layer's slot, else none.
constexpr int kAny = 1023;
/// The one device table for the three levels: the pair LUT [L][512][512], then the expert slots [L][512], then the layer slots [L].
inline size_t table_size(int n_layers) { return (size_t) n_layers * 512 * 512 + (size_t) n_layers * 512 + (size_t) n_layers; }
inline size_t table_index(int n_layers, int layer, int e, int s) {
    if (e == kAny) return (size_t) n_layers * 512 * 512 + (size_t) n_layers * 512 + (size_t) layer;
    if (s == kAny) return (size_t) n_layers * 512 * 512 + (size_t) layer * 512 + (size_t) e;
    return ((size_t) layer * 512 + (size_t) e) * 512 + (size_t) s;
}
inline int level_of(int e, int s) { return e == kAny ? 2 : s == kAny ? 1 : 0; }
constexpr int kMaxFitted = 512;               ///< fitted pairs in the device table (the LUT's slots)
constexpr int kHoldEvery = 4;                 ///< every 4th sample of a pair is held out
constexpr int kHoldWindow = 32;               ///< held-out samples in the gate's window (and the minimum before any decision)
constexpr double kGainFit = 0.20;             ///< gain needed to become fitted
constexpr double kGainRevoke = 0.10;          ///< gain below which a fitted pair is revoked
constexpr double kScalarClip = 4.0;           ///< the scalar used is clipped to [-kScalarClip, kScalarClip] (the record keeps the raw value)
constexpr double kLambdaRel = 0.1;            ///< lambda = kLambdaRel * trace(ZtZ) / r

/// The learner's switch (STRATA_SUBFIT_APPLY, read once at start): off, dynamic (the step-F cap's refusals are taken corrected), all.
enum class Apply { Off = 0, Dynamic = 1, All = 2, Flip = 3 };
/// "off", "dynamic", "all"; unset or anything else is dynamic (the default).
Apply apply_env(const char* v);
const char* apply_name(Apply a);
/// STRATA_SUBFIT_RANK: default 16, clamped to [1, 32].
int rank_env(const char* v);
/// STRATA_SUBFIT_PAIRS: default 512, clamped to [1, 8192].
int pairs_env(const char* v);
bool ladder_env(const char* v);
double gain_env(const char* v, double dflt);   ///< a gain bar from the environment (-1..1), else dflt

/// IEEE half of a float (round to nearest even; overflow to infinity, tiny values to subnormal or zero).
uint16_t f32_to_f16(float f);
float f16_to_f32(uint16_t h);

/// The layer's feature matrix A (r rows of n): splitmix64 uniforms, Box-Muller normals, scaled by 1/sqrt(n).  Deterministic in
/// (layer, seed, r, n).
void features(int layer, int n, int r, uint64_t seed, std::vector<float>& A);

/// Solves (ZtZ + lambda I) M = I by Cholesky (lambda = lambda_rel * trace / r) and returns B = M * T where T[k][j] = ZtYe[k][j] - c ZtYs[k][j]
/// (B is N x r: B[j * r + k]).  Returns false (B untouched) when the system has no positive pivot.
bool ridge_solve(const double* ZtZ, const float* ZtYe, const float* ZtYs, double c, int r, int n, double lambda_rel, std::vector<float>& B);

/// The learner's configuration (one per process; the tests use small n).
struct Config {
    int n = 2560;                   ///< the MoE input and output width
    int rank = kDefaultRank;        ///< r
    int max_pairs = kDefaultPairs;  ///< pair groups with statistics (least recently sampled evicted)
    int max_experts = kDefaultExperts;   ///< (layer, expert) groups with statistics (least recently sampled evicted); layers: all
    bool ladder = true;             ///< STRATA_SUBFIT_LADDER (default on): the (layer, expert) and layer groups learn too (off: pairs only)
    double gain_fit = kGainFit;     ///< STRATA_SUBFIT_GAIN_FIT: held-out gain needed to become fitted (default 0.20)
    double gain_revoke = kGainRevoke;   ///< STRATA_SUBFIT_GAIN_REVOKE: below it a fitted group is revoked (default 0.10; kept < gain_fit)
    int max_fitted = kMaxFitted;    ///< pairs with a device slot (the device table's rows)
    uint64_t seed = 1;              ///< the feature seed (the layer index is mixed in)
    int n_layers = 48;
};

/// One device slot change for the upload (between windows): the pair's scalar and its fp16 B (N x r), or the slot freed.
struct SlotChange {
    int slot = -1;
    int layer = 0, e = 0, s = 0;
    float c = 1.0f;
    float resid = 1.0f;        ///< the held-out residual sqrt(E1/E0) of the pair (the shift accounting's factor for a corrected move)
    std::vector<uint16_t> b;   ///< fp16 B, N x r (empty when the slot is freed)
};
/// One LUT entry change: (layer, e, s) -> slot, or -1 for none.
struct LutChange {
    int layer = 0, e = 0, s = 0;
    int slot = -1;
};
struct Changes {
    std::vector<SlotChange> slots;
    std::vector<LutChange> lut;
};

/// The host side of the device upload (CPU only; the verifier copies the writes between windows).  It keeps the mirror of the LUT
/// (n_layers x 512 x 512, -1 none) and the fitted count per layer (the apply kernel's launch check), and turns each Changes into the
/// writes.  A fresh packer matches a device table filled with -1.
struct LutWrite {
    size_t index;    ///< (layer * 512 + e) * 512 + s
    int16_t value;   ///< slot, or -1
};
struct SlotWrite {
    int slot = -1;
    bool freed = false;        ///< the slot is free: its LUT entries are written -1, its B is not written
    float c = 1.0f;
    float resid = 1.0f;
    std::vector<uint16_t> b;   ///< fp16 B, N x r (empty when freed)
};
class UploadPacker {
public:
    explicit UploadPacker(int n_layers);
    /// The writes of one Changes (LUT entries in order, then the slots); the mirror and the fitted counts are updated.  Entries out of
    /// range are dropped (none come from the learner).
    void pack(const Changes& ch, std::vector<LutWrite>& lut_out, std::vector<SlotWrite>& slot_out);
    const std::vector<int16_t>& lut() const { return lut_; }
    const std::vector<int32_t>& fit_n() const { return fit_n_; }
    static size_t lut_index(int layer, int e, int s) { return ((size_t) layer * kExperts + (size_t) e) * kExperts + (size_t) s; }
    size_t index(int layer, int e, int s) const { return table_index(n_layers_, layer, e, s); }

private:
    int n_layers_;
    std::vector<int16_t> lut_;
    std::vector<int32_t> fit_n_;
};

/// The request-end summary: pairs with statistics, fitted pairs, the gains of the fitted pairs (median, minimum), and the fitted pairs
/// that had no device slot (the table is full).
struct Summary {
    size_t pairs = 0, fitted = 0, unslotted = 0;
    double gain_median = 0.0, gain_min = 0.0;
    size_t groups[3] = {0, 0, 0}, fitted_lv[3] = {0, 0, 0};   ///< per level (pair, expert, layer)
    double gain_median_lv[3] = {0.0, 0.0, 0.0};
    // every group with a full held-out window (fitted or not): how many per level and their held-out gain (median, best)
    size_t ready_lv[3] = {0, 0, 0};
    double ready_median_lv[3] = {0.0, 0.0, 0.0}, ready_best_lv[3] = {0.0, 0.0, 0.0};
};

class Learner {
public:
    explicit Learner(const Config& cfg);
    Learner(const Learner&) = delete;
    Learner& operator=(const Learner&) = delete;

    const Config& config() const { return cfg_; }

    /// One sample of pair (layer, e, s).  x, ys, ye: N floats each.  Non-finite samples are dropped.  Between windows only.
    void add_sample(int layer, int e, int s, const float* x, const float* ys, const float* ye);
    /// the level whose slot a (layer, e, s) slot would use (0 pair, 1 expert, 2 layer), -1 none (the host mirror)
    int level_used(int layer, int e, int s) const;

    /// The request end: every pair with new samples is refit; the fitted/revoked state is updated; the changes for the device are
    /// returned (slots whose fit changed or were freed, and the LUT entries that moved).
    Changes refit();

    /// Every fitted pair with a slot, as changes (the device table's full upload, e.g. at start after load).
    Changes all_fitted() const;

    /// Layer `layer`'s feature matrix A as fp16 (r x n, the device copy the apply kernel reads); the same values features() gives.
    std::vector<uint16_t> features_f16(int layer) const;

    /// (layer, e, s) -> slot, or -1 (fitted with a device slot).  Out-of-range keys give -1.
    int lookup(int layer, int e, int s) const;
    bool fitted(int layer, int e, int s) const;
    /// The record's gain of the pair as is (1 - E1/E0 with E0 = |ye - ys|^2 over the recent window), 0 when unknown.
    double gain_as_is(int layer, int e, int s) const;
    /// The pair's scalar c (used) and held-out gain (recent window, over the scalar-only baseline), or false when the pair is unknown.
    bool pair_info(int layer, int e, int s, double& c_used, double& c_raw, double& gain, bool& is_fitted) const;

    Summary summary() const;
    size_t pair_count() const { return pairs_.size(); }
    /// Pairs with at least `min_ho` held-out samples so far (the request-end line's count).
    size_t pairs_with_holdout(int min_ho) const;
    size_t slots_used() const;

    /// Persistence per model fingerprint: stats and fits (the features are recomputed, not saved).
    bool save(const std::string& path, uint64_t fp, std::string& err) const;
    bool load(const std::string& path, uint64_t fp, std::string& err);

    /// Test hooks: the pair's B (fitted or not, N x r floats; empty when none) and the LUT size.
    const std::vector<float>* pair_b(int layer, int e, int s) const;

private:
    struct Pair {
        int layer = 0, e = 0, s = 0;
        uint64_t last_tick = 0;
        uint64_t seen = 0, n_train = 0, n_ho = 0;
        double sxy = 0.0, sxx = 0.0;
        double c_raw = 1.0, c_used = 1.0;
        bool dirty = false;
        bool has_b = false;
        bool is_fitted = false;
        int slot = -1;
        double E0 = 0.0, E1 = 0.0;     // cumulative held-out sums: E0 = |ye - ys|^2 (as is), E1 = |ye - (c ys + B z)|^2
        // the last kHoldWindow held-out sums: as is (ring0, the record), the scalar-only baseline |ye - c ys|^2 (ring0s, the gate's E0)
        // and E1 (ring1).  The gate uses the scalar-only baseline: the as-is error credits a pure shrink (c -> 0 on an uncorrelated pair
        // halves it), which accepted noise pairs in the test.
        double ring0[kHoldWindow] = {}, ring0s[kHoldWindow] = {}, ring1[kHoldWindow] = {};
        int ring_pos = 0, ring_n = 0;
        std::vector<double> zz;        // r x r
        std::vector<float> zye, zys;   // r x n each
        std::vector<float> B;          // n x r (empty until the first refit with enough samples)
    };

    uint64_t key(int layer, int e, int s) const { return ((uint64_t) layer << 20) | ((uint64_t) e << 10) | (uint64_t) s; }
    Pair* find(int layer, int e, int s);
    const Pair* find(int layer, int e, int s) const;
    Pair& get_or_create(int layer, int e, int s, Changes* ch);
    void evict_one(int level, Changes* ch);
    void add_to(Pair& p, const std::vector<double>& z, const float* ys, const float* ye);
    void free_slot(Pair& p, Changes* ch);
    bool assign_slot(Pair& p, Changes* ch);
    void set_lut(const Pair& p, int slot, Changes* ch);
    const std::vector<float>& features_of(int layer);
    double gain_recent(const Pair& p) const;
    double gain_asis(const Pair& p) const;
    void refit_pair(Pair& p, Changes* ch);
    void push_ho(Pair& p, double e0, double e0s, double e1);
    SlotChange slot_change(const Pair& p, int slot) const;

    Config cfg_;
    Changes pending_;                         // the device changes made by add_sample (evictions); refit() returns them
    uint64_t tick_ = 0;
    std::unordered_map<uint64_t, Pair> pairs_;
    std::vector<std::vector<float>> A_;       // per layer, built on first use
    std::vector<bool> slot_used_;             // cfg_.max_fitted slots
    std::vector<int16_t> lut_;                // table_size(n_layers): the three levels' slots (or -1)
    size_t level_n_[3] = {0, 0, 0};           // groups per level
};

}  // namespace strata::core::subfit_learn
