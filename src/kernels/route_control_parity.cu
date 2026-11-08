// route_control_parity - the control windows' flags (SPEC-control-windows) on the two routing kernels, GPU, synthetic, no model.
//
// A fragile control window (route_fragile_control(true)) and a mask control window (route_mask_control(true)) move no pick: route_resident_k
// and the mask kernel return the router's own ids, leave the weights as they were, and add nothing to row_shift.  With both flags clear the
// kernels are the ones that shipped: route_resident_k is checked against the scalar reference of route_resident_parity (ids and counters
// exact, weights within a few ulp), and the mask kernel against a serial write-out of its rule (ids exact, weights and row_shift within 1e-5).
//
// Run: route_control_parity   (ctest: route_control_parity; GPU, synthetic, no model)
#include "strata/kernels/native_router.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(2); }
}

constexpr int kExperts = 512;
constexpr int kTop = 10;
constexpr int kMaxTok = 8;
constexpr int kWeightUlp = 8;   // the scalar reference is a second compilation of the softmax (route_resident_parity's bound)
constexpr float kMaskTol = 1e-5f;
constexpr int kMaskLo = 6;      // the mask's floor (route_mask_set): ranks lo-1 and up are the moves

// The spec of route_resident_k with every option off but the floor and the margin: a serial scan, a strictly greater logit wins, the
// smallest index keeps a tie, and the already-picked test runs against the ids as they are at that rank.  (route_resident_parity's reference.)
__global__ void route_resident_ref_k(const float* __restrict__ logits, int32_t* __restrict__ ids, float* __restrict__ weights,
                                     const int32_t* __restrict__ res, int n_tok, float margin, int lo, int hi,
                                     unsigned long long* __restrict__ stats) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_tok) return;
    const float* l = logits + (size_t) t * kExperts;
    int32_t* id = ids + (size_t) t * kTop;
    int32_t my[kTop];
    for (int r = 0; r < kTop; ++r) my[r] = id[r];
    int swaps = 0, tail = 0, before = 0;
    for (int r = 0; r < kTop; ++r) before += res[my[r]] < 0;
    for (int r = lo; r <= hi && r < kTop; ++r) {
        const int e = my[r];
        if (res[e] >= 0) continue;
        ++tail;
        float bv = -INFINITY;
        int bi = -1;
        for (int f = 0; f < kExperts; ++f) {
            if (res[f] < 0) continue;
            bool used = false;
            for (int q = 0; q < kTop; ++q) used |= my[q] == f;
            if (used) continue;
            if (l[f] > bv) { bv = l[f]; bi = f; }
        }
        if (bi >= 0 && l[e] - bv <= margin) { my[r] = bi; ++swaps; }
    }
    int after = 0;
    for (int r = 0; r < kTop; ++r) after += res[my[r]] < 0;
    if (swaps) {
        float m = -INFINITY;
        for (int r = 0; r < kTop; ++r) m = fmaxf(m, l[my[r]]);
        float ex[kTop], sum = 0.0f;
        for (int r = 0; r < kTop; ++r) { ex[r] = expf(l[my[r]] - m); sum += ex[r]; }
        for (int r = 0; r < kTop; ++r) weights[(size_t) t * kTop + r] = ex[r] / sum;
    }
    for (int r = 0; r < kTop; ++r) id[r] = my[r];
    if (stats) {
        atomicAdd(stats + 0, (unsigned long long) tail);
        atomicAdd(stats + 1, (unsigned long long) swaps);
        atomicAdd(stats + 2, (unsigned long long) before);
        atomicAdd(stats + 3, (unsigned long long) after);
    }
}

// The router's top-10 of a row, ties to the smallest index.
void top10(const float* l, int32_t* out) {
    for (int r = 0; r < kTop; ++r) {
        float bv = -INFINITY;
        int bi = -1;
        for (int f = 0; f < kExperts; ++f) {
            bool used = false;
            for (int q = 0; q < r; ++q) used |= out[q] == f;
            if (used) continue;
            if (l[f] > bv) { bv = l[f]; bi = f; }
        }
        out[r] = bi;
    }
}

// The softmax over the ten picks, as the kernels write their weights.
void softmax_picks(const float* l, const int32_t* id, float* w) {
    float m = -INFINITY;
    for (int r = 0; r < kTop; ++r) m = std::max(m, l[id[r]]);
    float ex[kTop], sum = 0.0f;
    for (int r = 0; r < kTop; ++r) { ex[r] = std::exp(l[id[r]] - m); sum += ex[r]; }
    for (int r = 0; r < kTop; ++r) w[r] = ex[r] / sum;
}

// The mask's rule written out serially for one row (level 1 only, no residency, no edge): a masked pick of rank >= lo moves to the best
// unmasked expert the token did not pick (the strictly greater logit wins, the smallest index keeps a tie); rank lo - 1 never moves without an
// edge.  Returns the moves and the row's shift (w x gap of each move, w the router's weight on the pick).
int mask_ref_row(const float* l, int32_t* id, const float* w_in, const uint8_t* mask, float& shift) {
    int32_t my[kTop];
    for (int r = 0; r < kTop; ++r) my[r] = id[r];
    int moved = 0;
    shift = 0.0f;
    for (int r = 0; r < kTop; ++r) {
        const int e = my[r];
        if (e < 0 || e >= kExperts || mask[e] == 0) continue;
        if (r <= kMaskLo - 1) continue;   // (rank lo - 1 never moves: no edge)
        float bv = -INFINITY;
        int bi = -1;
        for (int f = 0; f < kExperts; ++f) {
            if (mask[f] != 0) continue;
            bool used = false;
            for (int q = 0; q < kTop; ++q) used |= my[q] == f;
            if (used) continue;
            if (bi < 0 || l[f] > bv) { bv = l[f]; bi = f; }
        }
        if (bi < 0) continue;
        const float gap = std::max(0.0f, l[e] - bv);
        shift += w_in[r] * gap;
        my[r] = bi;
        ++moved;
    }
    if (moved) for (int r = 0; r < kTop; ++r) id[r] = my[r];
    return moved;
}
}  // namespace

int main() {
    std::mt19937 rng(20261108u);
    std::uniform_real_distribution<float> ud(-3.0f, 3.0f);
    std::uniform_int_distribution<int> pick(0, 1);
    std::uniform_int_distribution<int> coin(0, 99);
    const float margins[] = {0.5f, 1.0f, 1.5f};

    std::vector<float> logits((size_t) kMaxTok * kExperts), w_in((size_t) kMaxTok * kTop), w_out((size_t) kMaxTok * kTop);
    std::vector<int32_t> ids((size_t) kMaxTok * kTop), ids_out((size_t) kMaxTok * kTop), res(kExperts);
    std::vector<uint8_t> mask(kExperts);
    std::vector<float> rs((size_t) kMaxTok, 0.0f), rs_out((size_t) kMaxTok, 0.0f);

    float *d_logits = nullptr, *d_w = nullptr, *d_rs = nullptr;
    int32_t *d_ids = nullptr, *d_res = nullptr;
    unsigned long long* d_stats = nullptr;
    ck(cudaMalloc(&d_logits, logits.size() * 4), "logits");
    ck(cudaMalloc(&d_w, w_in.size() * 4), "w");
    ck(cudaMalloc(&d_rs, rs.size() * 4), "row_shift");
    ck(cudaMalloc(&d_ids, ids.size() * 4), "ids");
    ck(cudaMalloc(&d_res, res.size() * 4), "res");
    ck(cudaMalloc(&d_stats, 4 * sizeof(unsigned long long)), "stats");
    cudaStream_t stream = nullptr;
    ck(cudaStreamCreate(&stream), "stream");
    // the fragile control flag, allocated before the kernels run (the verifier does it at its init, before any graph capture)
    if (!strata::kernels::route_fragile_control_alloc()) {
        std::fprintf(stderr, "route_fragile_control_alloc: no host-mapped flag\n");
        return 2;
    }

    // the mask (one layer) and its learner-side host copy: route_mask_set once, then route_mask_update per case
    std::string msg;
    std::vector<uint8_t> mask_host(kExperts, 0);
    if (!strata::kernels::route_mask_set(mask_host.data(), 1, kExperts, kMaskLo, -1.0f, msg)) {
        std::fprintf(stderr, "route_mask_set: %s\n", msg.c_str());
        return 2;
    }
    const uint8_t* d_mask = strata::kernels::route_mask_layer(0, kExperts);
    if (d_mask == nullptr) {
        std::fprintf(stderr, "route_mask_layer: no device mask\n");
        return 2;
    }

    const int cases = 120;
    int bad = 0, flag_clear_swaps = 0, mask_clear_moves = 0, flag_set_checks = 0;
    unsigned long long seen_set = 0, moved_set = 0;
    for (int c = 0; c < cases; ++c) {
        const int n_tok = 1 + (c % kMaxTok);
        const bool ties = pick(rng) != 0;
        for (size_t i = 0; i < logits.size(); ++i) {
            float v = ud(rng);
            if (ties) v = std::round(v * 2.0f) * 0.5f;
            logits[i] = v;
        }
        for (int t = 0; t < n_tok; ++t) {
            int32_t* row = ids.data() + (size_t) t * kTop;
            top10(logits.data() + (size_t) t * kExperts, row);
            softmax_picks(logits.data() + (size_t) t * kExperts, row, w_in.data() + (size_t) t * kTop);
        }
        const int pct = 40 + coin(rng) % 45;
        for (int e = 0; e < kExperts; ++e) res[e] = coin(rng) < pct ? (int32_t) (e % 7) : -1;
        for (int e = 0; e < kExperts; ++e) mask[e] = coin(rng) < 30 ? 1 : 0;
        strata::kernels::route_mask_update(mask);

        ck(cudaMemcpy(d_logits, logits.data(), logits.size() * 4, cudaMemcpyHostToDevice), "up logits");
        ck(cudaMemcpy(d_res, res.data(), res.size() * 4, cudaMemcpyHostToDevice), "up res");

        // ---- 1. route_resident_k, fragile flag CLEAR: the shipped kernel against the scalar reference (ids and counters exact, weights ulp)
        strata::kernels::route_fragile_control(false);
        strata::kernels::route_mask_control(false);
        const float margin = margins[c % 3];
        const int lo = pick(rng) ? 4 + (c % 4) : 0;
        const int hi = kTop - 1;
        ck(cudaMemcpy(d_ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice), "up ids");
        ck(cudaMemcpy(d_w, w_in.data(), w_in.size() * 4, cudaMemcpyHostToDevice), "up w");
        ck(cudaMemset(d_stats, 0, 4 * sizeof(unsigned long long)), "zero stats");
        strata::kernels::route_cfg_init(lo, hi, margin, -1.0f);
        strata::kernels::native_route_resident(d_logits, d_ids, d_w, d_res, n_tok, d_stats, (void*) stream);
        ck(cudaStreamSynchronize(stream), "sync clear");
        unsigned long long st_clear[4] = {0, 0, 0, 0};
        ck(cudaMemcpy(st_clear, d_stats, sizeof st_clear, cudaMemcpyDeviceToHost), "down stats clear");
        std::vector<int32_t> i_clear(ids.size());
        std::vector<float> w_clear(w_in.size());
        ck(cudaMemcpy(i_clear.data(), d_ids, i_clear.size() * 4, cudaMemcpyDeviceToHost), "down ids clear");
        ck(cudaMemcpy(w_clear.data(), d_w, w_clear.size() * 4, cudaMemcpyDeviceToHost), "down w clear");

        ck(cudaMemcpy(d_ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice), "up ids ref");
        ck(cudaMemcpy(d_w, w_in.data(), w_in.size() * 4, cudaMemcpyHostToDevice), "up w ref");
        ck(cudaMemset(d_stats, 0, 4 * sizeof(unsigned long long)), "zero stats ref");
        route_resident_ref_k<<<1, 32, 0, stream>>>(d_logits, d_ids, d_w, d_res, n_tok, margin, lo, hi, d_stats);
        ck(cudaStreamSynchronize(stream), "sync ref");
        unsigned long long st_ref[4] = {0, 0, 0, 0};
        ck(cudaMemcpy(st_ref, d_stats, sizeof st_ref, cudaMemcpyDeviceToHost), "down stats ref");
        std::vector<int32_t> i_ref(ids.size());
        std::vector<float> w_ref(w_in.size());
        ck(cudaMemcpy(i_ref.data(), d_ids, i_ref.size() * 4, cudaMemcpyDeviceToHost), "down ids ref");
        ck(cudaMemcpy(w_ref.data(), d_w, w_ref.size() * 4, cudaMemcpyDeviceToHost), "down w ref");

        int w_ulp = 0;
        for (size_t i = 0; i < (size_t) n_tok * kTop; ++i) {
            uint32_t a = 0, b = 0;
            std::memcpy(&a, &w_ref[i], 4);
            std::memcpy(&b, &w_clear[i], 4);
            if (a != b) w_ulp = std::max(w_ulp, (int) (a > b ? a - b : b - a));
        }
        const size_t nid = (size_t) n_tok * kTop;
        const bool ids_ok = std::memcmp(i_clear.data(), i_ref.data(), nid * 4) == 0;
        const bool st_ok = std::memcmp(st_clear, st_ref, sizeof st_clear) == 0;
        if (!ids_ok || !st_ok || w_ulp > kWeightUlp) {
            ++bad;
            std::printf("  case %d (resident, flag clear): ids %s, counters %s, weights %d ulp\n", c, ids_ok ? "same" : "DIFFER",
                        st_ok ? "same" : "DIFFER", w_ulp);
        }
        flag_clear_swaps += (int) st_clear[1];

        // ---- 2. route_resident_k, fragile flag SET, with the margin wide open (every non-resident pick would move if the flag were clear)
        strata::kernels::route_fragile_control(true);
        strata::kernels::route_cfg_init(0, kTop - 1, 1e30f, -1.0f);
        ck(cudaMemcpy(d_ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice), "up ids set");
        ck(cudaMemcpy(d_w, w_in.data(), w_in.size() * 4, cudaMemcpyHostToDevice), "up w set");
        ck(cudaMemcpy(d_rs, rs.data(), rs.size() * 4, cudaMemcpyHostToDevice), "up rs set");
        ck(cudaMemset(d_stats, 0, 4 * sizeof(unsigned long long)), "zero stats set");
        strata::kernels::native_route_resident(d_logits, d_ids, d_w, d_res, n_tok, d_stats, (void*) stream, nullptr, nullptr, nullptr, nullptr,
                                               nullptr, nullptr, d_rs);
        ck(cudaStreamSynchronize(stream), "sync set");
        unsigned long long st_set[4] = {0, 0, 0, 0};
        ck(cudaMemcpy(st_set, d_stats, sizeof st_set, cudaMemcpyDeviceToHost), "down stats set");
        std::vector<int32_t> i_set(ids.size());
        std::vector<float> w_set(w_in.size());
        ck(cudaMemcpy(i_set.data(), d_ids, i_set.size() * 4, cudaMemcpyDeviceToHost), "down ids set");
        ck(cudaMemcpy(w_set.data(), d_w, w_set.size() * 4, cudaMemcpyDeviceToHost), "down w set");
        ck(cudaMemcpy(rs_out.data(), d_rs, rs_out.size() * 4, cudaMemcpyDeviceToHost), "down rs set");
        bool set_ok = std::memcmp(i_set.data(), ids.data(), nid * 4) == 0;   // the router's own picks
        set_ok = set_ok && std::memcmp(w_set.data(), w_in.data(), nid * 4) == 0;   // weights untouched
        for (int t = 0; t < n_tok; ++t) set_ok = set_ok && rs_out[(size_t) t] == 0.0f;   // no stand-in moved anything
        set_ok = set_ok && st_set[0] == 0 && st_set[1] == 0 && st_set[3] == st_set[2];   // no tail seen, no swap
        if (!set_ok) {
            ++bad;
            std::printf("  case %d (resident, flag set): router picks kept %s, weights %s, swaps %llu\n", c,
                        std::memcmp(i_set.data(), ids.data(), nid * 4) == 0 ? "yes" : "NO",
                        std::memcmp(w_set.data(), w_in.data(), nid * 4) == 0 ? "untouched" : "CHANGED", st_set[1]);
        }
        ++flag_set_checks;

        // ---- 3. the mask kernel, flags CLEAR: ids exact against the serial rule, weights and row_shift within tolerance
        strata::kernels::route_fragile_control(false);
        strata::kernels::route_mask_control(false);
        ck(cudaMemcpy(d_ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice), "up ids mask");
        ck(cudaMemcpy(d_w, w_in.data(), w_in.size() * 4, cudaMemcpyHostToDevice), "up w mask");
        ck(cudaMemcpy(d_rs, rs.data(), rs.size() * 4, cudaMemcpyHostToDevice), "zero rs mask");
        unsigned long long seen = 0, moved = 0;
        strata::kernels::route_mask_read(seen, moved);   // (drops the counts of earlier windows)
        strata::kernels::native_route_mask(d_logits, d_ids, d_w, d_mask, n_tok, d_rs, (void*) stream);
        ck(cudaStreamSynchronize(stream), "sync mask");
        ck(cudaMemcpy(i_set.data(), d_ids, i_set.size() * 4, cudaMemcpyDeviceToHost), "down ids mask");
        ck(cudaMemcpy(w_set.data(), d_w, w_set.size() * 4, cudaMemcpyDeviceToHost), "down w mask");
        ck(cudaMemcpy(rs_out.data(), d_rs, rs_out.size() * 4, cudaMemcpyDeviceToHost), "down rs mask");
        strata::kernels::route_mask_read(seen, moved);

        std::vector<int32_t> i_mref(ids.size());
        std::vector<float> w_mref(w_in.size()), rs_mref((size_t) kMaxTok, 0.0f);
        int moved_ref = 0;
        for (int t = 0; t < n_tok; ++t) {
            int32_t* row = i_mref.data() + (size_t) t * kTop;
            std::memcpy(row, ids.data() + (size_t) t * kTop, kTop * 4);
            float sh = 0.0f;
            const int m = mask_ref_row(logits.data() + (size_t) t * kExperts, row, w_in.data() + (size_t) t * kTop, mask.data(), sh);
            moved_ref += m;
            rs_mref[(size_t) t] = sh;
            if (m) softmax_picks(logits.data() + (size_t) t * kExperts, row, w_mref.data() + (size_t) t * kTop);
            else std::memcpy(w_mref.data() + (size_t) t * kTop, w_in.data() + (size_t) t * kTop, kTop * 4);
        }
        bool mask_ok = std::memcmp(i_set.data(), i_mref.data(), nid * 4) == 0;
        for (size_t i = 0; i < nid && mask_ok; ++i) mask_ok = std::fabs(w_set[i] - w_mref[i]) <= kMaskTol;
        for (int t = 0; t < n_tok && mask_ok; ++t)
            mask_ok = std::fabs(rs_out[(size_t) t] - rs_mref[(size_t) t]) <= kMaskTol * (1.0f + std::fabs(rs_mref[(size_t) t]));
        mask_ok = mask_ok && (long long) moved == moved_ref;
        if (!mask_ok) {
            ++bad;
            std::printf("  case %d (mask, flags clear): ids %s, moves %llu (serial %d)\n", c,
                        std::memcmp(i_set.data(), i_mref.data(), nid * 4) == 0 ? "same" : "DIFFER", moved, moved_ref);
        }
        mask_clear_moves += moved_ref;

        // ---- 4. the mask kernel with a control flag (mask control, then the fragile one): the router's own picks, no move, no shift
        for (int which = 0; which < 2; ++which) {
            strata::kernels::route_fragile_control(which == 1);
            strata::kernels::route_mask_control(which == 0);
            ck(cudaMemcpy(d_ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice), "up ids ctl");
            ck(cudaMemcpy(d_w, w_in.data(), w_in.size() * 4, cudaMemcpyHostToDevice), "up w ctl");
            ck(cudaMemcpy(d_rs, rs.data(), rs.size() * 4, cudaMemcpyHostToDevice), "zero rs ctl");
            strata::kernels::route_mask_read(seen, moved);   // (drops the earlier counts)
            strata::kernels::native_route_mask(d_logits, d_ids, d_w, d_mask, n_tok, d_rs, (void*) stream);
            ck(cudaStreamSynchronize(stream), "sync ctl");
            ck(cudaMemcpy(i_set.data(), d_ids, i_set.size() * 4, cudaMemcpyDeviceToHost), "down ids ctl");
            ck(cudaMemcpy(w_set.data(), d_w, w_set.size() * 4, cudaMemcpyDeviceToHost), "down w ctl");
            ck(cudaMemcpy(rs_out.data(), d_rs, rs_out.size() * 4, cudaMemcpyDeviceToHost), "down rs ctl");
            strata::kernels::route_mask_read(seen, moved);
            bool ctl_ok = std::memcmp(i_set.data(), ids.data(), nid * 4) == 0;
            ctl_ok = ctl_ok && std::memcmp(w_set.data(), w_in.data(), nid * 4) == 0;
            for (int t = 0; t < n_tok; ++t) ctl_ok = ctl_ok && rs_out[(size_t) t] == 0.0f;
            ctl_ok = ctl_ok && moved == 0;
            if (!ctl_ok) {
                ++bad;
                std::printf("  case %d (mask, %s control): picks kept %s, moves %llu\n", c, which == 0 ? "mask" : "fragile",
                            std::memcmp(i_set.data(), ids.data(), nid * 4) == 0 ? "yes" : "NO", moved);
            }
            seen_set += seen;
            moved_set += moved;
            ++flag_set_checks;
        }
    }

    strata::kernels::route_fragile_control(false);
    strata::kernels::route_mask_control(false);
    std::printf("route_control_parity: %d cases, flag-clear swaps %d, flag-clear mask moves %d, flag-set checks %d, masked picks seen under control %llu "
                "(moves %llu), mismatches %d\n",
                cases, flag_clear_swaps, mask_clear_moves, flag_set_checks, seen_set, moved_set, bad);
    if (flag_clear_swaps == 0 || mask_clear_moves == 0) {
        std::printf("route_control_parity: the clear-flag runs moved nothing - the test proved nothing\n");
        ++bad;
    }
    std::printf(bad ? "route_control_parity: %d FAILURES\n" : "route_control_parity OK\n", bad);
    cudaFree(d_logits); cudaFree(d_w); cudaFree(d_rs); cudaFree(d_ids); cudaFree(d_res); cudaFree(d_stats);
    cudaStreamDestroy(stream);
    return bad ? 1 : 0;
}
