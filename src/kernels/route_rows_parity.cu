// route_rows_parity - SPEC-recheck-cheap change 1: the fragile flag as a row count on the two routing kernels (GPU, synthetic, no model).
//
// route_fragile_rows(n): a row t of the window with t < n moves nothing (route_resident_k and the mask kernel return the router's own ids,
// the weights as they were, row_shift 0); the rows t >= n are the flag-clear run's, bit for bit (a row is independent of the others: one
// warp per token).  Checked for every n from 0 to the rows of the window and for n = a large value, which must equal the old control
// window (route_fragile_control(true)) bit for bit, ids, weights, row_shift and counters.
//
// Run: route_rows_parity   (ctest: route_rows_parity; GPU, synthetic, no model)
#include "strata/core/recheck_cheap.hpp"
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
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

constexpr int kExperts = 512;
constexpr int kTop = 10;
constexpr int kMaxTok = 8;
constexpr int kMaskLo = 6;   // the mask's floor (route_mask_set): ranks lo-1 and up are the moves
constexpr int kAll = strata::core::recheck::kAllRows;

// The router's top-10 of a row, ties to the smallest index.
void top10(const float* l, int32_t* out) {
    for (int r = 0; r < kTop; ++r) {
        float bv = -INFINITY;
        int bi = -1;
        for (int f = 0; f < kExperts; ++f) {
            bool used = false;
            for (int q = 0; q < r; ++q) used |= out[q] == f;
            if (used) continue;
            if (l[f] > bv) {
                bv = l[f];
                bi = f;
            }
        }
        out[r] = bi;
    }
}

// The softmax over the ten picks, as the kernels write their weights.
void softmax_picks(const float* l, const int32_t* id, float* w) {
    float m = -INFINITY;
    for (int r = 0; r < kTop; ++r) m = std::max(m, l[id[r]]);
    float ex[kTop], sum = 0.0f;
    for (int r = 0; r < kTop; ++r) {
        ex[r] = std::exp(l[id[r]] - m);
        sum += ex[r];
    }
    for (int r = 0; r < kTop; ++r) w[r] = ex[r] / sum;
}

// One run's outputs (the device buffers read back).
struct Out {
    std::vector<int32_t> ids;
    std::vector<float> w, rs;
    unsigned long long st[4] = {0, 0, 0, 0};
};

bool same_bits(const float* a, const float* b, size_t n) { return std::memcmp(a, b, n * sizeof(float)) == 0; }

}  // namespace

int main() {
    std::mt19937 rng(20261108u + 11u);
    std::uniform_real_distribution<float> ud(-3.0f, 3.0f);
    std::uniform_int_distribution<int> coin(0, 99);

    std::vector<float> logits((size_t) kMaxTok * kExperts), w_in((size_t) kMaxTok * kTop, 0.0f);
    std::vector<int32_t> ids_in((size_t) kMaxTok * kTop, -1), res(kExperts);
    std::vector<uint8_t> mask(kExperts, 0);
    std::vector<float> rs0((size_t) kMaxTok, 0.0f);

    float *d_logits = nullptr, *d_w = nullptr, *d_rs = nullptr;
    int32_t *d_ids = nullptr, *d_res = nullptr;
    unsigned long long* d_stats = nullptr;
    ck(cudaMalloc(&d_logits, logits.size() * sizeof(float)), "alloc logits");
    ck(cudaMalloc(&d_w, w_in.size() * sizeof(float)), "alloc weights");
    ck(cudaMalloc(&d_rs, rs0.size() * sizeof(float)), "alloc row_shift");
    ck(cudaMalloc(&d_ids, ids_in.size() * sizeof(int32_t)), "alloc ids");
    ck(cudaMalloc(&d_res, res.size() * sizeof(int32_t)), "alloc res");
    ck(cudaMalloc(&d_stats, 4 * sizeof(unsigned long long)), "alloc stats");
    cudaStream_t stream = nullptr;
    ck(cudaStreamCreate(&stream), "stream");

    if (!strata::kernels::route_fragile_control_alloc()) {
        std::fprintf(stderr, "route_fragile_control_alloc: no host-mapped flag\n");
        return 2;
    }
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

    // one run's inputs: the flag (a row count n, or the old control window when ctl), the router's picks, zero shift and counters
    auto set_run = [&](int n, bool ctl) {
        if (ctl) strata::kernels::route_fragile_control(true);
        else strata::kernels::route_fragile_rows(n);
        strata::kernels::route_cfg_init(0, kTop - 1, 1e30f, -1.0f);   // every non-resident pick would move if the flag were clear
        ck(cudaMemcpy(d_ids, ids_in.data(), ids_in.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "up ids");
        ck(cudaMemcpy(d_w, w_in.data(), w_in.size() * sizeof(float), cudaMemcpyHostToDevice), "up w");
        ck(cudaMemcpy(d_rs, rs0.data(), rs0.size() * sizeof(float), cudaMemcpyHostToDevice), "up rs");
        ck(cudaMemset(d_stats, 0, 4 * sizeof(unsigned long long)), "zero stats");
    };
    auto launch_resident = [&](int n_tok) {
        strata::kernels::native_route_resident(d_logits, d_ids, d_w, d_res, n_tok, d_stats, (void*) stream, nullptr, nullptr, nullptr,
                                               nullptr, nullptr, nullptr, d_rs);
    };
    auto read_resident = [&](int n_tok, Out& o) {
        ck(cudaStreamSynchronize(stream), "sync resident");
        o.ids.assign((size_t) n_tok * kTop, 0);
        o.w.assign((size_t) n_tok * kTop, 0.0f);
        o.rs.assign((size_t) n_tok, 0.0f);
        ck(cudaMemcpy(o.ids.data(), d_ids, o.ids.size() * sizeof(int32_t), cudaMemcpyDeviceToHost), "down ids");
        ck(cudaMemcpy(o.w.data(), d_w, o.w.size() * sizeof(float), cudaMemcpyDeviceToHost), "down w");
        ck(cudaMemcpy(o.rs.data(), d_rs, o.rs.size() * sizeof(float), cudaMemcpyDeviceToHost), "down rs");
        ck(cudaMemcpy(o.st, d_stats, sizeof o.st, cudaMemcpyDeviceToHost), "down stats");
    };

    const int cases = 60;
    int bad = 0;
    long long rows_held = 0, rows_checked = 0, clear_swaps = 0, clear_mask_moves = 0, mask_rows_held = 0;
    for (int c = 0; c < cases; ++c) {
        const int n_tok = 1 + (c % kMaxTok);
        const bool ties = coin(rng) < 50;
        for (size_t i = 0; i < logits.size(); ++i) {
            float v = ud(rng);
            if (ties) v = std::round(v * 2.0f) * 0.5f;
            logits[i] = v;
        }
        for (int t = 0; t < n_tok; ++t) {
            int32_t* row = ids_in.data() + (size_t) t * kTop;
            top10(logits.data() + (size_t) t * kExperts, row);
            softmax_picks(logits.data() + (size_t) t * kExperts, row, w_in.data() + (size_t) t * kTop);
        }
        for (int e = 0; e < kExperts; ++e) res[e] = coin(rng) < 50 ? (int32_t) (e % 7) : -1;
        for (int e = 0; e < kExperts; ++e) mask[e] = coin(rng) < 30 ? 1 : 0;
        strata::kernels::route_mask_update(mask);
        ck(cudaMemcpy(d_logits, logits.data(), logits.size() * sizeof(float), cudaMemcpyHostToDevice), "up logits");
        ck(cudaMemcpy(d_res, res.data(), res.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "up res");

        // ---- route_resident_k: the flag-clear run (row count 0) is the reference of every row >= n
        Out ref;
        strata::kernels::route_mask_control(false);
        set_run(0, false);
        launch_resident(n_tok);
        read_resident(n_tok, ref);
        clear_swaps += (long long) ref.st[1];

        // n = 0 is the flag-clear run (again, after a row count was set and cleared)
        for (int n = 0; n <= n_tok + 1; ++n) {
            Out got;
            set_run(n, false);
            launch_resident(n_tok);
            read_resident(n_tok, got);
            ++rows_checked;
            for (int t = 0; t < n_tok; ++t) {
                const int32_t* gi = got.ids.data() + (size_t) t * kTop;
                const int32_t* ii = ids_in.data() + (size_t) t * kTop;
                const float* gw = got.w.data() + (size_t) t * kTop;
                const float* iw = w_in.data() + (size_t) t * kTop;
                const float* rw = ref.w.data() + (size_t) t * kTop;
                const bool held = t < n;
                bool ok;
                if (held) {
                    // a held row: the router's picks, the weights untouched, no shift
                    ok = std::memcmp(gi, ii, kTop * sizeof(int32_t)) == 0 && same_bits(gw, iw, kTop) && got.rs[(size_t) t] == 0.0f;
                    ++rows_held;
                } else {
                    // a row past the count: the flag-clear run, bit for bit
                    ok = std::memcmp(gi, ref.ids.data() + (size_t) t * kTop, kTop * sizeof(int32_t)) == 0 && same_bits(gw, rw, kTop) &&
                         got.rs[(size_t) t] == ref.rs[(size_t) t];
                }
                if (!ok) {
                    ++bad;
                    std::printf("  case %d n_tok %d count %d row %d (%s): differs\n", c, n_tok, n, t, held ? "held" : "free");
                }
            }
        }

        // n = a large value is the old control window, bit for bit (ids, weights, row_shift, counters)
        Out ctl, big;
        set_run(0, true);
        launch_resident(n_tok);
        read_resident(n_tok, ctl);
        set_run(kAll, false);
        launch_resident(n_tok);
        read_resident(n_tok, big);
        const bool same_ctl = big.ids == ctl.ids && same_bits(big.w.data(), ctl.w.data(), big.w.size()) && big.rs == ctl.rs &&
                              std::memcmp(big.st, ctl.st, sizeof big.st) == 0;
        if (!same_ctl) {
            ++bad;
            std::printf("  case %d: a large row count is not the control window\n", c);
        }
        // the control window moves nothing: every pick is the router's
        if (std::memcmp(big.ids.data(), ids_in.data(), (size_t) n_tok * kTop * sizeof(int32_t)) != 0 || big.st[1] != 0) {
            ++bad;
            std::printf("  case %d: the control window moved a pick\n", c);
        }

        // ---- the mask kernel: the same rule on its own (flags clear: the reference; a row count n keeps rows < n)
        auto run_mask = [&](int n, bool ctl_flag, Out& o) {
            strata::kernels::route_mask_control(ctl_flag);
            strata::kernels::route_fragile_rows(n);
            ck(cudaMemcpy(d_ids, ids_in.data(), ids_in.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "up ids mask");
            ck(cudaMemcpy(d_w, w_in.data(), w_in.size() * sizeof(float), cudaMemcpyHostToDevice), "up w mask");
            ck(cudaMemcpy(d_rs, rs0.data(), rs0.size() * sizeof(float), cudaMemcpyHostToDevice), "up rs mask");
            unsigned long long seen = 0, moved = 0;
            strata::kernels::route_mask_read(seen, moved);   // drops the earlier counts
            strata::kernels::native_route_mask(d_logits, d_ids, d_w, d_mask, n_tok, d_rs, (void*) stream);
            ck(cudaStreamSynchronize(stream), "sync mask");
            o.ids.assign((size_t) n_tok * kTop, 0);
            o.w.assign((size_t) n_tok * kTop, 0.0f);
            o.rs.assign((size_t) n_tok, 0.0f);
            ck(cudaMemcpy(o.ids.data(), d_ids, o.ids.size() * sizeof(int32_t), cudaMemcpyDeviceToHost), "down ids mask");
            ck(cudaMemcpy(o.w.data(), d_w, o.w.size() * sizeof(float), cudaMemcpyDeviceToHost), "down w mask");
            ck(cudaMemcpy(o.rs.data(), d_rs, o.rs.size() * sizeof(float), cudaMemcpyDeviceToHost), "down rs mask");
            strata::kernels::route_mask_read(seen, moved);
            o.st[0] = seen;
            o.st[1] = moved;
        };
        Out mref;
        run_mask(0, false, mref);
        clear_mask_moves += (long long) mref.st[1];
        for (int n = 0; n <= n_tok + 1; ++n) {
            Out got;
            run_mask(n, false, got);
            for (int t = 0; t < n_tok; ++t) {
                const int32_t* gi = got.ids.data() + (size_t) t * kTop;
                const float* gw = got.w.data() + (size_t) t * kTop;
                bool ok;
                if (t < n) {
                    ok = std::memcmp(gi, ids_in.data() + (size_t) t * kTop, kTop * sizeof(int32_t)) == 0 &&
                         same_bits(gw, w_in.data() + (size_t) t * kTop, kTop) && got.rs[(size_t) t] == 0.0f;
                    ++mask_rows_held;
                } else {
                    ok = std::memcmp(gi, mref.ids.data() + (size_t) t * kTop, kTop * sizeof(int32_t)) == 0 &&
                         same_bits(gw, mref.w.data() + (size_t) t * kTop, kTop) && got.rs[(size_t) t] == mref.rs[(size_t) t];
                }
                if (!ok) {
                    ++bad;
                    std::printf("  case %d mask n_tok %d count %d row %d: differs\n", c, n_tok, n, t);
                }
            }
        }
        // ---- SPEC-recheck-gated step A: the row bitmask (route_fragile_hold).  Bit r holds row r: its picks, weights and row_shift stay the
        // router's; every other row is the flag-clear run's, bit for bit.  0xFFFFFFFF is the control window (checked above as a count).
        {
            std::uniform_int_distribution<uint32_t> ub(0u, 0xFFFFFFFFu);
            const uint32_t rows_mask = (1u << n_tok) - 1u;
            for (int m = 0; m < 4; ++m) {
                const uint32_t bits = m == 0 ? 0xFFFFFFFFu : (m == 1 ? 0u : (ub(rng) & rows_mask));
                // the resident kernel
                strata::kernels::route_fragile_hold(bits);
                ck(cudaMemcpy(d_ids, ids_in.data(), ids_in.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "up ids bits");
                ck(cudaMemcpy(d_w, w_in.data(), w_in.size() * sizeof(float), cudaMemcpyHostToDevice), "up w bits");
                ck(cudaMemcpy(d_rs, rs0.data(), rs0.size() * sizeof(float), cudaMemcpyHostToDevice), "up rs bits");
                ck(cudaMemset(d_stats, 0, 4 * sizeof(unsigned long long)), "zero stats bits");
                Out got;
                launch_resident(n_tok);
                read_resident(n_tok, got);
                for (int t = 0; t < n_tok; ++t) {
                    const bool held = t < 32 ? ((bits >> t) & 1u) != 0u : bits == 0xFFFFFFFFu;
                    const int32_t* gi = got.ids.data() + (size_t) t * kTop;
                    const float* gw = got.w.data() + (size_t) t * kTop;
                    bool ok;
                    if (held) {
                        ok = std::memcmp(gi, ids_in.data() + (size_t) t * kTop, kTop * sizeof(int32_t)) == 0 &&
                             same_bits(gw, w_in.data() + (size_t) t * kTop, kTop) && got.rs[(size_t) t] == 0.0f;
                        ++rows_held;
                    } else {
                        ok = std::memcmp(gi, ref.ids.data() + (size_t) t * kTop, kTop * sizeof(int32_t)) == 0 &&
                             same_bits(gw, ref.w.data() + (size_t) t * kTop, kTop) && got.rs[(size_t) t] == ref.rs[(size_t) t];
                    }
                    if (!ok) {
                        ++bad;
                        std::printf("  case %d resident bits 0x%08x row %d (%s): differs\n", c, bits, t, held ? "held" : "free");
                    }
                }
                // the mask kernel
                strata::kernels::route_mask_control(false);
                strata::kernels::route_fragile_hold(bits);
                ck(cudaMemcpy(d_ids, ids_in.data(), ids_in.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "up ids mask bits");
                ck(cudaMemcpy(d_w, w_in.data(), w_in.size() * sizeof(float), cudaMemcpyHostToDevice), "up w mask bits");
                ck(cudaMemcpy(d_rs, rs0.data(), rs0.size() * sizeof(float), cudaMemcpyHostToDevice), "up rs mask bits");
                strata::kernels::native_route_mask(d_logits, d_ids, d_w, d_mask, n_tok, d_rs, (void*) stream);
                ck(cudaStreamSynchronize(stream), "sync mask bits");
                Out gm;
                gm.ids.assign((size_t) n_tok * kTop, 0);
                gm.w.assign((size_t) n_tok * kTop, 0.0f);
                gm.rs.assign((size_t) n_tok, 0.0f);
                ck(cudaMemcpy(gm.ids.data(), d_ids, gm.ids.size() * sizeof(int32_t), cudaMemcpyDeviceToHost), "down ids mask bits");
                ck(cudaMemcpy(gm.w.data(), d_w, gm.w.size() * sizeof(float), cudaMemcpyDeviceToHost), "down w mask bits");
                ck(cudaMemcpy(gm.rs.data(), d_rs, gm.rs.size() * sizeof(float), cudaMemcpyDeviceToHost), "down rs mask bits");
                for (int t = 0; t < n_tok; ++t) {
                    const bool held = t < 32 ? ((bits >> t) & 1u) != 0u : bits == 0xFFFFFFFFu;
                    const int32_t* gi = gm.ids.data() + (size_t) t * kTop;
                    const float* gw = gm.w.data() + (size_t) t * kTop;
                    bool ok;
                    if (held) {
                        ok = std::memcmp(gi, ids_in.data() + (size_t) t * kTop, kTop * sizeof(int32_t)) == 0 &&
                             same_bits(gw, w_in.data() + (size_t) t * kTop, kTop) && gm.rs[(size_t) t] == 0.0f;
                        ++mask_rows_held;
                    } else {
                        ok = std::memcmp(gi, mref.ids.data() + (size_t) t * kTop, kTop * sizeof(int32_t)) == 0 &&
                             same_bits(gw, mref.w.data() + (size_t) t * kTop, kTop) && gm.rs[(size_t) t] == mref.rs[(size_t) t];
                    }
                    if (!ok) {
                        ++bad;
                        std::printf("  case %d mask bits 0x%08x row %d (%s): differs\n", c, bits, t, held ? "held" : "free");
                    }
                }
            }
        }
        Out mbig, mctl;
        run_mask(kAll, false, mbig);
        run_mask(0, true, mctl);   // the old control window (the mask's own flag)
        if (mbig.ids != mctl.ids || !same_bits(mbig.w.data(), mctl.w.data(), mbig.w.size()) || mbig.rs != mctl.rs) {
            ++bad;
            std::printf("  case %d: a large mask row count is not the mask control window\n", c);
        }
    }
    strata::kernels::route_mask_control(false);
    strata::kernels::route_fragile_rows(0);

    std::printf("route_rows_parity: %d cases, %lld row counts, %lld held resident rows, %lld held mask rows; flag-clear swaps %lld, "
                "flag-clear mask moves %lld\n",
                cases, rows_checked, rows_held, mask_rows_held, clear_swaps, clear_mask_moves);
    if (clear_swaps == 0 || clear_mask_moves == 0) {
        std::printf("route_rows_parity: the flag-clear runs moved nothing - the test proved nothing\n");
        return 1;
    }
    std::printf(bad ? "route_rows_parity: %d FAILURES\n" : "route_rows_parity OK\n", bad);
    return bad ? 1 : 0;
}
