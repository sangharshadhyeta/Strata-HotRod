// include/strata/core/recheck_gate.hpp - the re-check gated by drafter confidence (docs/SPEC-recheck-gated.md: the rule 1-6, step B's
// eligibility, step C's early-margin deciles), header-only, no CUDA.  Everything the decode loop decides per row or per window lives here,
// so it can be tested on the CPU: the row's rule from its drafter probability, the hold bitmask of the uncertain rows, the audit cadence,
// the eligibility of a moved row for a cut, the threshold mover (rule 5, Wilson bounds), the early-margin deciles, and the state that
// persists across requests and restarts (text, keyed by the model's fingerprint).
//
// The counts the mover learns from are the eligible rows only: a row that was not eligible for a re-check was not measured, so it is not
// counted.  With the gate on, a confident row is eligible only in an audit window; an uncertain row and a "none" row are always eligible
// (8 Nov: uncertain rows are no longer held up front).  With the gate off every moved row is eligible in every window.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

#include "strata/core/tune_steps.hpp"

namespace strata::core::gate {

constexpr int kNone = 10;                      ///< the bucket of a row with no drafter probability (fp_bucket's 10)
constexpr int kBuckets = 10;                   ///< drafter probability buckets 0.0-0.1 .. 0.9-1.0
constexpr uint32_t kAllBits = 0xFFFFFFFFu;     ///< every row held (the control window)
constexpr float kPHiDefault = 0.9f;            ///< rule 1's start value (STRATA_RECHECK_P_HI)
constexpr float kPHiLo = 0.6f, kPHiHi = 0.95f; ///< rule 5's range
constexpr float kPHiStep = 0.1f;               ///< one bucket
constexpr double kZ90 = 1.645;                 ///< the Wilson z of the 90% bounds (as route_wilson90)
constexpr double kFlipBound = 0.005;           ///< rule 5's flip-rate bound (0.5% per moved row)
constexpr long long kMoveRows = 2000;          ///< at most one move per this many audited moved rows
constexpr int kMarginRing = 4096;              ///< the early-margin samples kept for the running deciles

enum class Rule { None, Confident, Uncertain };

/// The row's rule.  bucket: fp_bucket (0-9, or kNone); p: the drafter's probability of the row's token (valid when bucket != kNone).
inline Rule classify(int bucket, float p, float p_hi) {
    if (bucket >= kNone) return Rule::None;
    return p >= p_hi ? Rule::Confident : Rule::Uncertain;
}

/// Eligible for a cut: every row in an audit window; outside one, the uncertain rows (rule 2) and the rows with no drafter probability (rule 3).
/// Only the confident rows (rule 1) are not eligible outside an audit window.
inline bool eligible(Rule r, bool audit) { return audit || r != Rule::Confident; }

/// Rule 2 as the hold bitmask.  Changed on 8 Nov (coordinator, from the live data): the uncertain rows are NOT held (exact up front cost
/// ~100 expert reads per row, against the re-check paying only for the ~14% of them that are cut); they stay eligible for a cut, as today.
/// This returns 0 (the bitmask of step A stays for a later rank-partial re-check).
inline uint32_t uncertain_bits(const int* bucket, const float* p, int T, float p_hi, bool audit) {
    (void) bucket; (void) p; (void) T; (void) p_hi; (void) audit;
    return 0u;
}

/// Rule 4: decode window number `window` (1-based, counted over the decode windows of the process) is an audit window when every > 0 and
/// window is a multiple of every.
inline bool audit_due(long long window, int every) { return every > 0 && window > 0 && window % every == 0; }

/// Clamps P_HI into the range 0.6-0.95.
inline float clamp_p_hi(float p) {
    if (!(p >= kPHiLo)) return kPHiLo;   // (NaN too)
    return p > kPHiHi ? kPHiHi : p;
}

/// The bucket just below P_HI (the one rule 5 tests for a move down): floor(p_hi * 10), taken a hair below, so 0.9 gives 8.
inline int below_bucket(float p_hi) { return (int) std::floor(p_hi * 10.0f - 1e-4f); }
/// The first confident bucket of P_HI (rule 5 tests the confident buckets for a move up): floor(p_hi * 10), a hair above, so 0.9 gives 9.
inline int confident_bucket(float p_hi) { return (int) std::floor(p_hi * 10.0f + 1e-4f); }

/// The Wilson 90% bound of k flips in n moved rows (upper or lower).  n = 0: 1 (upper) or 0 (lower).
inline double wilson(long long k, long long n, bool upper) {
    if (n <= 0) return upper ? 1.0 : 0.0;
    const double z2 = kZ90 * kZ90, nn = (double) n, p = (double) k / nn;
    const double centre = p + z2 / (2.0 * nn);
    const double half = kZ90 * std::sqrt(p * (1.0 - p) / nn + z2 / (4.0 * nn * nn));
    const double denom = 1.0 + z2 / nn;
    return upper ? (centre + half) / denom : (centre - half) / denom;
}

/// Per-bucket counts: index 0-9 the drafter buckets, 10 the none rows.
struct Counts {
    long long moved[kNone + 1] = {};
    long long cut[kNone + 1] = {};
    long long flip[kNone + 1] = {};
};

/// Early-margin deciles (step C): the none rows' moved / cut / flipped by the decile of their margin.
struct Deciles {
    long long moved[kBuckets] = {};
    long long cut[kBuckets] = {};
    long long flip[kBuckets] = {};
};

/// SPEC-recheck-gated (D): the flip attribution by rank and layer band, [rank - 3 (ranks 3-9)][band (layer / 12, 0-3)][0 not flipped, 1 flipped],
/// counted over the swaps (replaced picks) of each cut row.
constexpr int kRankSlots = 7;
constexpr int kBands = 4;
struct RankTable {
    long long n[kRankSlots][kBands][2] = {};
};
/// The rank's slot (ranks 3-9 -> 0-6), or -1.
inline int rank_slot(int rank) { return rank >= 3 && rank <= 9 ? rank - 3 : -1; }
/// The layer band (layers 0-11 -> 0, 12-23 -> 1, 24-35 -> 2, 36-47 -> 3; past 47 -> 3, below 0 -> 0).
inline int band_of(int layer) {
    const int b = layer < 0 ? 0 : layer / 12;
    return b > 3 ? 3 : b;
}

// ---- SPEC-recheck-gated step F: the per-row shift cap.  The caps are multiples of a running median of the moved rows' final row_shift.

/// A ring of samples (the newest `cap` kept): the running median and the running deciles.
struct Ring {
    std::vector<float> v;
    size_t pos = 0;
    void add(float x, size_t cap) {
        if (!(x == x) || !(x > 0.0f)) return;   // NaN and non-positive values are not kept
        if (v.size() < cap) v.push_back(x);
        else {
            v[pos] = x;
            pos = (pos + 1) % cap;
        }
    }
    size_t n() const { return v.size(); }
    /// The upper median (element n/2 of the sorted samples); 0 with no sample.
    float median() const {
        if (v.empty()) return 0.0f;
        std::vector<float> c = v;
        std::nth_element(c.begin(), c.begin() + (std::ptrdiff_t) (c.size() / 2), c.end());
        return c[c.size() / 2];
    }
    /// The decile of x against the samples: the share below x, in tenths (0 with no sample; 9 at most).
    int decile(float x) const {
        if (v.empty()) return 0;
        size_t less = 0;
        for (float s : v) less += s < x ? 1u : 0u;
        const int d = (int) (10.0 * (double) less / (double) v.size());
        return d > 9 ? 9 : d;
    }
};
constexpr size_t kShiftRing = 4096;   ///< the moved rows' final shifts kept for the median and the deciles
constexpr size_t kShiftMin = 64;      ///< no cap before this many samples

/// The cap multipliers of the running median, per row class (+inf = no cap for the class).  Defaults: confident none, uncertain 1.5, none 1.0.
struct CapMult {
    float conf = INFINITY;
    float unc = 1.5f;
    float none = 1.0f;
};

/// The cap of a row of class r: its multiplier times the running median; +inf until kShiftMin samples and a positive median.
inline float cap_for(Rule r, float median, size_t n, const CapMult& m) {
    if (!(median > 0.0f) || n < kShiftMin) return INFINITY;
    const float k = r == Rule::Confident ? m.conf : r == Rule::Uncertain ? m.unc : m.none;
    return k * median;
}

/// The 32 caps of this window's rows (row r: its class from bucket / p / P_HI).  An audit window caps nothing (the audit measures the
/// uncapped behaviour).  Rows past T and past 32 stay +inf.
inline void fill_caps(float* out32, const int* bucket, const float* p, int T, float p_hi, bool audit, float median, size_t n, const CapMult& m) {
    for (int r = 0; r < 32; ++r) out32[r] = INFINITY;
    if (audit) return;
    const int rows = T < 32 ? T : 32;
    for (int r = 0; r < rows; ++r) out32[r] = cap_for(classify(bucket[r], p[r], p_hi), median, n, m);
}

/// The calibration counter (step F): the cut rows' final shift by decile of the running shifts, split by the verdict.
struct ShiftCal {
    long long flip[kBuckets] = {};
    long long not_flip[kBuckets] = {};
};

// ---- SPEC-recheck-gated step H1: k from evidence above the noise floor (replaces "1.25 x the largest flipped ratio").
// Every 50 re-checks the ring (the last 200 re-checks: ratio gap/shift, flipped or not) gives r*: the LARGEST flipped ratio t such that the
// entries with ratio >= t number n_t >= 8 and their flips f_t have a Wilson 90% lower bound above the control's upper bound.  Noise flips
// (about 7% of exact-vs-exact re-checks, at any ratio) cannot clear that bound at high ratios, so they no longer set k.

constexpr int kEvidenceMinN = 8;     ///< entries with ratio >= t needed for the threshold t
constexpr int kKRingMin = 50;        ///< a ring this size with no evidence lowers k by half (its step limits are the caller's)

struct KEvidence {
    bool found = false;   ///< a threshold qualified (r* is set)
    double r_star = 0.0;  ///< the largest qualifying flipped ratio
    long long n_t = 0;    ///< entries with ratio >= r*
    long long f_t = 0;    ///< of those, flipped
};

/// r* over the first n ring entries (ratios[i], flips[i]).  ctrl_hi: the control's Wilson 90% upper bound of its flip rate (with fewer than
/// 20 control re-checks it is near 1, so nothing qualifies: no raise, as before).  NaN ratios are not thresholds.
inline KEvidence k_evidence(const float* ratios, const bool* flips, int n, double ctrl_hi) {
    KEvidence e;
    for (int i = 0; i < n; ++i) {
        if (!flips[i]) continue;
        const double t = (double) ratios[i];
        if (!(t == t)) continue;
        long long nt = 0, ft = 0;
        for (int j = 0; j < n; ++j)
            if ((double) ratios[j] >= t) {   // (a NaN ratio is never >= t)
                ++nt;
                if (flips[j]) ++ft;
            }
        if (nt < kEvidenceMinN) continue;
        double lo = 0.0, hi = 0.0;
        paired::wilson90(ft, nt, lo, hi);
        if (lo > ctrl_hi && (!e.found || t > e.r_star)) {
            e.found = true;
            e.r_star = t;
            e.n_t = nt;
            e.f_t = ft;
        }
    }
    return e;
}

/// The target k (before the caller's step limits: up to 1.5x, down to 0.5x per update): 1.25 x r* clamped to [0.005, 50] with evidence;
/// else half of the old k when the ring has kKRingMin entries; else the old k.
/// 8 Nov fix (live: k fell 0.039 -> 0.0012 in 15 minutes): "no flip above the noise" is evidence only when the noise is measured - with
/// fewer than kKCtrlMin control samples the Wilson bound is huge, nothing can exceed it, and halving on every update collapsed k.  So the
/// lowering needs kKCtrlMin control samples and never goes below kKFloor; otherwise k is held.
constexpr int kKCtrlMin = 20;
constexpr float kKFloor = 0.005f;
/// SPEC-substitute-fit stage 2 step 3: k may only RAISE when the evidence group behind r* has at least this many re-checks (n_t).  A
/// threshold resting on 8 re-checks can hold k where it is but cannot push it up (the 0.092-from-8 case).  Lowering is unchanged.
constexpr long long kKRaiseMinN = 30;
inline float k_target(const KEvidence& e, int n_ring, float old, int n_ctrl) {
    if (e.found) {
        const double v = 1.25 * e.r_star;
        const float target = (float) std::min(50.0, std::max(0.005, v));
        if (target > old && e.n_t < kKRaiseMinN) return old;   // a raise needs kKRaiseMinN re-checks behind r*
        return target;
    }
    if (n_ring >= kKRingMin && n_ctrl >= kKCtrlMin) return old > kKFloor ? std::max(kKFloor, 0.5f * old) : old;
    return old;
}

/// SPEC-substitute-fit stage 2 step 3: the k and the control-noise ring of each APPLY mode (0 off, 1 dynamic, 2 all).  While the process
/// runs, the ring of its own mode is State::ctrl (what the window seeds from and the control re-checks fill); the k is the verifier's
/// fragile_k.  store_active() puts both back into the slot before a save; activate() takes a mode's slot when the process starts.
/// Old files (no mode lines): the tune file's k and the gate's ring are the off slot, and they seed the other modes.
constexpr int kApplyModes = 4;   // off, dynamic, all, flip
struct KByMode {
    float k[kApplyModes] = {};
    bool k_set[kApplyModes] = {};
    std::vector<unsigned char> ctrl[kApplyModes];
    bool ctrl_set[kApplyModes] = {};
};

/// The legacy values (the tune file's k when > 0, the gate's ring when the file had no mode lines) go to the off slot when it is not set,
/// and every unset mode takes the off slot's values.
inline void seed_by_mode(KByMode& km, float legacy_k, const std::vector<unsigned char>* legacy_ctrl) {
    if (legacy_k > 0.0f && !km.k_set[0]) {
        km.k[0] = legacy_k;
        km.k_set[0] = true;
    }
    if (legacy_ctrl != nullptr && !km.ctrl_set[0]) {
        km.ctrl[0] = *legacy_ctrl;
        km.ctrl_set[0] = true;
    }
    for (int m = 1; m < kApplyModes; ++m) {
        if (!km.k_set[m] && km.k_set[0]) {
            km.k[m] = km.k[0];
            km.k_set[m] = true;
        }
        if (!km.ctrl_set[m] && km.ctrl_set[0]) {
            km.ctrl[m] = km.ctrl[0];
            km.ctrl_set[m] = true;
        }
    }
}

/// The mode index of an APPLY value (clamped to the three modes).
inline int mode_slot(int apply_mode) { return apply_mode < 0 ? 0 : (apply_mode >= kApplyModes ? kApplyModes - 1 : apply_mode); }

/// Everything the gate keeps across requests and restarts.  total: the eligible rows, cumulative (what the logs print and the totals
/// file keeps).  since: the same, since the last P_HI move (the mover's evidence).  margin_ring: the none rows' margins (running estimate).
struct State {
    float p_hi = kPHiDefault;
    long long moves = 0;
    Counts total;
    Counts since;
    Deciles margin;
    RankTable rank;
    tune::Tune tune;     ///< step G: the cap multipliers and the prefetch count chosen by the paired tuner (persisted)
    // the control re-checks (the noise floor of k and the brake): the last kCtrlKeep exact-vs-exact re-checks, oldest first (1 = the top-1
    // flipped), and their cumulative count and flips.  Saved with the state and seeding the window's control ring at load.
    std::vector<unsigned char> ctrl;
    long long ctrl_seen = 0, ctrl_flips_total = 0;
    Ring shift;          ///< step F: the moved rows' final row_shift (the cap's median, the calibration deciles)
    ShiftCal cal;        ///< step F: the cut rows' shift decile, split by the verdict
    std::vector<float> margin_ring;
    size_t margin_pos = 0;
    KByMode km;          ///< step 3: the k and the control ring of every APPLY mode (the active one's ring is ctrl)
};

/// Entering mode m at start: its ring becomes State::ctrl (when the mode has one).  Returns true and sets k_out when the mode has a k.
inline bool activate(State& s, int m, float& k_out) {
    if (s.km.ctrl_set[m]) s.ctrl = s.km.ctrl[m];
    if (!s.km.k_set[m]) return false;
    k_out = s.km.k[m];
    return true;
}

/// Before a save: the active mode's k and ring go back into its slot (the other slots are kept as they were).
inline void store_active(State& s, int m, float k) {
    s.km.k[m] = k;
    s.km.k_set[m] = true;
    s.km.ctrl[m] = s.ctrl;
    s.km.ctrl_set[m] = true;
}

inline void add_moved(State& s, int bucket) {
    ++s.total.moved[bucket];
    ++s.since.moved[bucket];
}
inline void add_cut(State& s, int bucket) {
    ++s.total.cut[bucket];
    ++s.since.cut[bucket];
}
inline void add_flip(State& s, int bucket) {
    ++s.total.flip[bucket];
    ++s.since.flip[bucket];
}

/// The early margin's decile against the running estimate: the share of the ring below m, in tenths (0 with no sample; NaN is decile 0).
inline int margin_decile(const State& s, float m) {
    const size_t n = s.margin_ring.size();
    if (n == 0) return 0;
    size_t less = 0;
    for (float v : s.margin_ring) less += v < m ? 1u : 0u;
    const int d = (int) (10.0 * (double) less / (double) n);
    return d > 9 ? 9 : d;
}
/// Adds a margin sample to the ring (NaN is not kept).
inline void add_margin_sample(State& s, float m) {
    if (m != m) return;
    if (s.margin_ring.size() < (size_t) kMarginRing) {
        s.margin_ring.push_back(m);
    } else {
        s.margin_ring[s.margin_pos] = m;
        s.margin_pos = (s.margin_pos + 1) % (size_t) kMarginRing;
    }
}

/// Rule 5, the threshold mover.  Evidence: `since` (the eligible rows since the last move) and at least kMoveRows moved rows in the
/// drafter buckets.  Up (P_HI + one bucket, capped at 0.95) when the Wilson 90% lower bound of the confident buckets' flip rate exceeds
/// 0.5%.  Otherwise down (P_HI - one bucket, floored at 0.6) when the Wilson 90% upper bound of the bucket just below P_HI is at most 0.5%.
/// A move resets `since` and counts one move.  Returns +1 (up), -1 (down), 0 (no move).
inline int step(State& s) {
    long long moved_all = 0;
    for (int b = 0; b < kBuckets; ++b) moved_all += s.since.moved[b];
    if (moved_all < kMoveRows) return 0;
    const int kc = confident_bucket(s.p_hi), kb = below_bucket(s.p_hi);
    int dir = 0;
    long long cm = 0, cf = 0;
    for (int b = kc < 0 ? 0 : kc; b < kBuckets; ++b) {
        cm += s.since.moved[b];
        cf += s.since.flip[b];
    }
    if (cm > 0 && s.p_hi < kPHiHi - 1e-6f && wilson(cf, cm, false) > kFlipBound) {
        dir = 1;
    } else if (kb >= 0 && kb < kBuckets && s.p_hi > kPHiLo + 1e-6f && s.since.moved[kb] > 0 &&
               wilson(s.since.flip[kb], s.since.moved[kb], true) <= kFlipBound) {
        dir = -1;
    }
    if (dir == 0) return 0;
    s.p_hi = clamp_p_hi(s.p_hi + (float) dir * kPHiStep);
    ++s.moves;
    s.since = Counts{};
    return dir;
}

// ---- the control re-checks (the noise floor), persisted with the state.

constexpr size_t kCtrlKeep = 400;   ///< the control re-checks kept (the newest; a decayed window of the noise, not a cumulative sum)

/// One control re-check (flipped or not): the ring keeps the newest kCtrlKeep, the counters are cumulative.
inline void ctrl_add(State& s, bool flipped) {
    ++s.ctrl_seen;
    if (flipped) ++s.ctrl_flips_total;
    if (s.ctrl.size() >= kCtrlKeep) s.ctrl.erase(s.ctrl.begin());
    s.ctrl.push_back(flipped ? 1 : 0);
}

/// Seeds a window's control ring (ring[0..cap), the layout of the window's fr_ctrl): the newest min(size, cap) saved re-checks, oldest first.
/// Returns the count n (the ring's fill) and sets pos to the next slot (n mod cap), as the ring keeps it.
inline int ctrl_seed(const State& s, bool* ring, int cap, int& pos) {
    const int n = (int) std::min<size_t>(s.ctrl.size(), (size_t) cap);
    const size_t start = s.ctrl.size() - (size_t) n;
    for (int i = 0; i < n; ++i) ring[i] = s.ctrl[start + (size_t) i] != 0;
    pos = cap > 0 ? n % cap : 0;
    return n;
}

/// The control cadence (STRATA_FRAGILE_CTRL_EVERY = configured, 0 = off): while fewer than kCtrlEarlyN control re-checks exist, at most
/// kCtrlEarlyEvery eligible windows apart, so the noise floor is measured early; the configured cadence after that.
constexpr long long kCtrlEarlyN = 20;
constexpr long long kCtrlEarlyEvery = 128;
inline long long ctrl_every(long long configured, long long n_ctrl) {
    if (configured <= 0) return 0;
    return n_ctrl < kCtrlEarlyN && configured > kCtrlEarlyEvery ? kCtrlEarlyEvery : configured;
}

// ---- persistence: text, one key per line, keyed by the model's fingerprint (a file of another model is refused).

/// The state as text (the fingerprint on the first line).  Floats are written with 9 digits, so a round trip is exact.
inline std::string serialize(const State& s, uint64_t fp) {
    std::string out;
    char buf[64];
    auto counts = [&](const char* name, const long long* v, int n) {
        out += name;
        for (int i = 0; i < n; ++i) {
            std::snprintf(buf, sizeof buf, " %lld", v[i]);
            out += buf;
        }
        out += "\n";
    };
    std::snprintf(buf, sizeof buf, "fingerprint %016llx\n", (unsigned long long) fp);
    out += buf;
    std::snprintf(buf, sizeof buf, "p_hi %.9g\n", (double) s.p_hi);
    out += buf;
    std::snprintf(buf, sizeof buf, "moves %lld\n", s.moves);
    out += buf;
    counts("total_moved", s.total.moved, kNone + 1);
    counts("total_cut", s.total.cut, kNone + 1);
    counts("total_flip", s.total.flip, kNone + 1);
    counts("since_moved", s.since.moved, kNone + 1);
    counts("since_cut", s.since.cut, kNone + 1);
    counts("since_flip", s.since.flip, kNone + 1);
    counts("margin_moved", s.margin.moved, kBuckets);
    counts("margin_cut", s.margin.cut, kBuckets);
    counts("margin_flip", s.margin.flip, kBuckets);
    counts("rank", &s.rank.n[0][0][0], kRankSlots * kBands * 2);
    counts("cal_flip", s.cal.flip, kBuckets);
    counts("cal_not", s.cal.not_flip, kBuckets);
    std::snprintf(buf, sizeof buf, "tune_set %d\n", s.tune.set ? 1 : 0);
    out += buf;
    std::snprintf(buf, sizeof buf, "tune_cap %.9g %.9g %.9g %d %d %d\n", s.tune.cap_unc, s.tune.cap_none, (double) s.tune.cap_step,
                  s.tune.cap_last_win, s.tune.cap_first, s.tune.cap_stage);
    out += buf;
    std::snprintf(buf, sizeof buf, "tune_prefetch %lld %d %d\n", s.tune.prefetch, s.tune.pf_first, s.tune.pf_stage);
    out += buf;
    std::snprintf(buf, sizeof buf, "tune_k %.9g %.9g %d %d %d\n", s.tune.k_mult, (double) s.tune.k_step, s.tune.k_last_win, s.tune.k_first,
                  s.tune.k_stage);
    out += buf;
    std::snprintf(buf, sizeof buf, "ctrl_seen %lld\nctrl_flips %lld\n", s.ctrl_seen, s.ctrl_flips_total);
    out += buf;
    out += "ctrl_ring";
    for (unsigned char v : s.ctrl) out += v ? " 1" : " 0";
    out += "\n";
    for (int m = 0; m < kApplyModes; ++m) {   // step 3: the modes' k and rings (a mode that has none writes no line)
        if (s.km.k_set[m]) {
            std::snprintf(buf, sizeof buf, "k_mode %d %.9g\n", m, (double) s.km.k[m]);
            out += buf;
        }
        if (s.km.ctrl_set[m]) {
            std::snprintf(buf, sizeof buf, "ctrl_mode %d", m);
            out += buf;
            for (unsigned char v : s.km.ctrl[m]) out += v ? " 1" : " 0";
            out += "\n";
        }
    }
    std::snprintf(buf, sizeof buf, "shift_pos %lld\n", (long long) s.shift.pos);
    out += buf;
    out += "shift_ring";
    for (float v : s.shift.v) {
        std::snprintf(buf, sizeof buf, " %.9g", (double) v);
        out += buf;
    }
    out += "\n";
    std::snprintf(buf, sizeof buf, "margin_pos %lld\n", (long long) s.margin_pos);
    out += buf;
    out += "margin_ring";
    for (float v : s.margin_ring) {
        std::snprintf(buf, sizeof buf, " %.9g", (double) v);
        out += buf;
    }
    out += "\n";
    return out;
}

/// Reads serialize's text into s.  false (s unchanged) when the fingerprint is not fp, or a line is malformed.
inline bool parse(const std::string& text, uint64_t fp, State& s) {
    State t;
    bool fp_ok = false;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::string key;
        if (!(ls >> key)) continue;
        if (key == "fingerprint") {
            std::string hex;
            if (!(ls >> hex)) return false;
            fp_ok = std::stoull(hex, nullptr, 16) == (unsigned long long) fp;
        } else if (key == "p_hi") {
            double v = 0;
            if (!(ls >> v)) return false;
            t.p_hi = clamp_p_hi((float) v);
        } else if (key == "moves") {
            if (!(ls >> t.moves)) return false;
        } else if (key == "margin_pos") {
            long long v = 0;
            if (!(ls >> v) || v < 0) return false;
            t.margin_pos = (size_t) v;
        } else if (key == "margin_ring") {
            double v = 0;
            while (ls >> v) t.margin_ring.push_back((float) v);
        } else if (key == "tune_set") {
            int v = 0;
            if (!(ls >> v)) return false;
            t.tune.set = v != 0;
        } else if (key == "tune_cap") {
            double u = 0, n = 0, st = 0;
            int lw = 0, f = 1, sg = 0;
            if (!(ls >> u >> n >> st >> lw >> f >> sg)) return false;
            if (!(u > 0.0) || !(n > 0.0) || !(st > 0.0)) return false;
            t.tune.cap_unc = u;
            t.tune.cap_none = n;
            t.tune.cap_step = (float) st;
            t.tune.cap_last_win = lw;
            t.tune.cap_first = f < 0 ? -1 : 1;
            t.tune.cap_stage = sg != 0 ? 1 : 0;
        } else if (key == "tune_prefetch") {
            long long p = 0;
            int f = 1, sg = 0;
            if (!(ls >> p >> f >> sg)) return false;
            if (p < tune::kPrefetchLo || p > tune::kPrefetchHi) return false;
            t.tune.prefetch = p;
            t.tune.pf_first = f < 0 ? -1 : 1;
            t.tune.pf_stage = sg != 0 ? 1 : 0;
        } else if (key == "ctrl_seen") {
            if (!(ls >> t.ctrl_seen) || t.ctrl_seen < 0) return false;
        } else if (key == "ctrl_flips") {
            if (!(ls >> t.ctrl_flips_total) || t.ctrl_flips_total < 0) return false;
        } else if (key == "ctrl_ring") {
            long long v = 0;
            while (ls >> v) {
                if ((v != 0 && v != 1) || t.ctrl.size() >= kCtrlKeep) return false;   // a 0/1 per re-check, at most kCtrlKeep
                t.ctrl.push_back((unsigned char) v);
            }
        } else if (key == "k_mode") {
            int m = -1;
            double v = 0;
            if (!(ls >> m >> v) || m < 0 || m >= kApplyModes || !(v > 0.0)) return false;
            t.km.k[m] = (float) v;
            t.km.k_set[m] = true;
        } else if (key == "ctrl_mode") {
            int m = -1;
            if (!(ls >> m) || m < 0 || m >= kApplyModes) return false;
            std::vector<unsigned char> ring;
            long long v = 0;
            while (ls >> v) {
                if ((v != 0 && v != 1) || ring.size() >= kCtrlKeep) return false;
                ring.push_back((unsigned char) v);
            }
            t.km.ctrl[m] = std::move(ring);
            t.km.ctrl_set[m] = true;
        } else if (key == "tune_k") {
            double m = 0, st = 0;
            int lw = 0, f = 1, sg = 0;
            if (!(ls >> m >> st >> lw >> f >> sg)) return false;
            if (!(m >= tune::kKMultLo && m <= tune::kKMultHi) || !(st > 0.0)) return false;
            t.tune.k_mult = m;
            t.tune.k_step = (float) st;
            t.tune.k_last_win = lw;
            t.tune.k_first = f < 0 ? -1 : 1;
            t.tune.k_stage = sg != 0 ? 1 : 0;
        } else if (key == "shift_pos") {
            long long v = 0;
            if (!(ls >> v) || v < 0) return false;
            t.shift.pos = (size_t) v;
        } else if (key == "shift_ring") {
            double v = 0;
            while (ls >> v) t.shift.v.push_back((float) v);
        } else {
            long long* dst = nullptr;
            int n = 0;
            if (key == "total_moved") { dst = t.total.moved; n = kNone + 1; }
            else if (key == "total_cut") { dst = t.total.cut; n = kNone + 1; }
            else if (key == "total_flip") { dst = t.total.flip; n = kNone + 1; }
            else if (key == "since_moved") { dst = t.since.moved; n = kNone + 1; }
            else if (key == "since_cut") { dst = t.since.cut; n = kNone + 1; }
            else if (key == "since_flip") { dst = t.since.flip; n = kNone + 1; }
            else if (key == "margin_moved") { dst = t.margin.moved; n = kBuckets; }
            else if (key == "margin_cut") { dst = t.margin.cut; n = kBuckets; }
            else if (key == "margin_flip") { dst = t.margin.flip; n = kBuckets; }
            else if (key == "rank") { dst = &t.rank.n[0][0][0]; n = kRankSlots * kBands * 2; }
            else if (key == "cal_flip") { dst = t.cal.flip; n = kBuckets; }
            else if (key == "cal_not") { dst = t.cal.not_flip; n = kBuckets; }
            if (dst == nullptr) continue;   // an unknown key: kept out (forward compatibility)
            for (int i = 0; i < n; ++i)
                if (!(ls >> dst[i])) return false;
        }
    }
    if (!fp_ok) return false;
    if (t.margin_pos >= (size_t) kMarginRing || t.margin_ring.size() > (size_t) kMarginRing) return false;
    if (t.shift.pos >= kShiftRing || t.shift.v.size() > kShiftRing) return false;
    s = t;
    return true;
}

}  // namespace strata::core::gate
