// router_fuse_parity - SPEC-router-fuse step 1: the look-ahead's fused path against today's per-distance chain, bit for bit.
//
// Today's chain, per dist d in {1, 2, 4} of a layer l: bf16_gemv_fp32_mmvf_multi (the target layer's router on layer l's MoE input),
// la_learn_forward (only when the learner is on), native_router_top10_multi, native_route_mask (the target layer's mask, resident-aware
// or not), route_pred_edge_skip, route_pred_tail_read.  The fused path: bf16_gemv_fp32_mmvf_multi3 (all dists in one launch), the
// learner per dist, route_pred_select (all dists' top-10 / mask / edge / tail in one launch).
// Checked on synthetic router-shaped inputs (random activations, weights at router scale, residency ~30%, the mask on the last 128 experts):
// the logits (after the learner) bitwise, the weights bitwise, the ids exactly, the edge-skip and tail-read counters equal.
// Exact float ties are counted and reported (a row whose top-11 logits hold two equal values): a tie is the one place an id could
// legitimately differ between two correct orders, so it is reported, and an id mismatch on a row without a tie fails the test.
//
// Run: router_fuse_parity   (GPU, synthetic, no model; needs ~40 MB)
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/native_router.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace K = strata::kernels;

namespace {
constexpr int kL = 8, kNE = 512, kK = 10, kMaxN = 8, kN = 2560;
constexpr int kLo = 3, kHi = 9;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(2); }
}
uint16_t bf16(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t) ((u + 0x7fff + ((u >> 16) & 1)) >> 16); }

struct Config {
    const char* name;
    float margin, edge;
    bool learn;     // the learner's forward per dist
    bool mres;      // the mask is resident-aware (its res / ring / cheap slices)
    int tail_all;   // route_pred_tail_read's all_ranks
};

// the host table's index of (target layer, slot, row, rank): route_pred_dev's slab order
size_t slab(int64_t target, int slot) { return ((size_t) target * K::kRoutePredDists + (size_t) slot) * 80; }
}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    int bad = 0;
    std::mt19937 rng(17);
    std::normal_distribution<float> nd(0.f, 1.f);

    // weights: one router per target layer (bf16 [512 x N]) at router scale, activations ~N(0,1)
    std::vector<uint16_t> wh((size_t) kL * kNE * kN);
    for (auto& v : wh) v = bf16(0.05f * nd(rng));
    uint16_t* dwall;
    ck(cudaMalloc((void**) &dwall, wh.size() * 2), "weights");
    ck(cudaMemcpy(dwall, wh.data(), wh.size() * 2, cudaMemcpyHostToDevice), "weights up");
    std::vector<float> xh((size_t) kMaxN * kN);
    for (auto& v : xh) v = nd(rng);
    float* dx;
    ck(cudaMalloc((void**) &dx, xh.size() * 4), "x");
    ck(cudaMemcpy(dx, xh.data(), xh.size() * 4, cudaMemcpyHostToDevice), "x up");

    std::vector<int32_t> res((size_t) kL * kNE);
    std::vector<uint8_t> mask((size_t) kL * kNE);
    for (int l = 0; l < kL; ++l)
        for (int e = 0; e < kNE; ++e) {
            res[(size_t) l * kNE + e] = (rng() % 10) < 3 ? (int32_t) (e % 64) : -1;
            mask[(size_t) l * kNE + e] = e >= kNE - 128 ? 1 : 0;
        }
    int32_t* dres;
    ck(cudaMalloc((void**) &dres, res.size() * 4), "res");
    ck(cudaMemcpy(dres, res.data(), res.size() * 4, cudaMemcpyHostToDevice), "res up");

    std::string msg;
    if (!K::route_mask_set(mask.data(), kL, kNE, kLo, 1.0f, msg, kLo)) { std::fprintf(stderr, "route_mask_set: %s\n", msg.c_str()); return 2; }
    K::route_pred_alloc(kL);
    K::route_pred_edge_skip_alloc();
    K::route_pred_tail_read_alloc();
    const bool learner_ok = K::la_learn_alloc(kL, kN, kLo, false, msg);
    if (!learner_ok) std::printf("note: learner not allocated (%s); the learner configs are skipped\n", msg.c_str());

    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream");
    unsigned long long* dstat;   // route_resident is not part of this step; the mask's counters take a scratch
    ck(cudaMalloc((void**) &dstat, 16 * 8), "stats");
    ck(cudaMemset(dstat, 0, 16 * 8), "stats zero");

    const Config cfgs[] = {
        {"open rule (margin 1e30), edge 1, tail ranks >= lo", 1e30f, 1.0f, false, false, 0},
        {"finite margin, learner, mask resident-aware, tail all ranks", 1.0f, 1.0f, true, true, 1},
        {"margin 0.5, no edge (edge skip off), tail ranks >= lo", 0.5f, -1.0f, false, false, 0},
    };
    const int ns[] = {1, 2, 3, 5, 8};
    const int l = 0;
    const int dists[3] = {1, 2, 4};
    for (const Config& cfg : cfgs) {
        if (cfg.learn && !learner_ok) continue;
        K::route_cfg_init(kLo, kHi, cfg.margin, cfg.edge, 0.f, 0, 1.f, 0.f, 0.f, 0.f, 0.f);
        for (int n : ns) {
            // ---- today's chain (dists sequential, the one route scratch) ----
            std::vector<float> ref_logits(3 * (size_t) kMaxN * kNE), ref_w(3 * (size_t) kMaxN * kK);
            std::vector<int32_t> ref_ids(3 * (size_t) kMaxN * kK);
            ck(cudaDeviceSynchronize(), "sync");
            K::route_pred_edge_skips_take();
            K::route_pred_tail_read_take();
            for (int di = 0; di < 3; ++di) {
                const int dist = dists[di], t = l + dist;
                K::bf16_gemv_fp32_mmvf_multi(dx, kN, dwall + (size_t) t * kNE * kN, K::route_pred_logits(), kNE, kN, kNE, n, s);
                if (cfg.learn) K::la_learn_forward(l, dist, dx, n, s);
                ck(cudaStreamSynchronize(s), "sync l");
                ck(cudaMemcpy(ref_logits.data() + (size_t) di * kMaxN * kNE, K::route_pred_logits(), (size_t) n * kNE * 4, cudaMemcpyDeviceToHost), "cp l");
                K::native_router_top10_multi(K::route_pred_logits(), K::route_pred_dev(t, dist), K::route_pred_weights(), n, s);
                K::native_route_mask(K::route_pred_logits(), K::route_pred_dev(t, dist), K::route_pred_weights(),
                                     K::route_mask_layer(t, kL), n, nullptr, s, nullptr, t, 0,
                                     cfg.mres ? dres + (size_t) t * kNE : nullptr, nullptr, nullptr);
                if (cfg.edge >= 0.f) K::route_pred_edge_skip(K::route_pred_logits(), K::route_pred_dev(t, dist), dres + (size_t) t * kNE, n, s);
                K::route_pred_tail_read(dres + (size_t) t * kNE, K::route_pred_dev(t, dist), n, s, cfg.tail_all);
                ck(cudaStreamSynchronize(s), "sync chain");
                const int32_t* hslab = K::route_pred_host() + slab(t, K::route_pred_slot(dist));
                for (int r = 0; r < n; ++r)
                    for (int k = 0; k < kK; ++k) ref_ids[((size_t) di * kMaxN + r) * kK + k] = hslab[r * kK + k];
                ck(cudaMemcpy(ref_w.data() + (size_t) di * kMaxN * kK, K::route_pred_weights(), (size_t) n * kK * 4, cudaMemcpyDeviceToHost), "cp w");
            }
            const unsigned long long ref_edge = K::route_pred_edge_skips_take(), ref_tail = K::route_pred_tail_read_take();

            // ---- fused: the same slabs cleared on the host (the mapped table), then the one GEMV and the one selection ----
            for (int di = 0; di < 3; ++di) {
                const int t = l + dists[di];
                int32_t* host_table = const_cast<int32_t*>(K::route_pred_host());   // (mapped memory: the test's own reset)
                for (size_t i = 0; i < 80; ++i) host_table[slab(t, K::route_pred_slot(dists[di])) + i] = -1;
            }
            const uint16_t* w3[3] = {dwall + (size_t) (l + 1) * kNE * kN, dwall + (size_t) (l + 2) * kNE * kN, dwall + (size_t) (l + 4) * kNE * kN};
            const bool gemv_ok = K::bf16_gemv_fp32_mmvf_multi3(dx, kN, w3, 7u, K::route_pred_logits(), kNE, (int64_t) 8 * kNE, kN, kNE, n, s);
            if (!gemv_ok) {
                std::printf("FAIL %s n=%d: bf16_gemv_fp32_mmvf_multi3 declined\n", cfg.name, n);
                ++bad;
                continue;
            }
            if (cfg.learn)
                for (int di = 0; di < 3; ++di)
                    K::la_learn_forward_at(l, dists[di], dx, n, s, K::route_pred_logits_at(K::route_pred_slot(dists[di])));
            K::RoutePredDist pd[3];
            for (int di = 0; di < 3; ++di) {
                const int t = l + dists[di], slot = K::route_pred_slot(dists[di]);
                pd[di].logits = K::route_pred_logits_at(slot);
                pd[di].ids = K::route_pred_dev(t, dists[di]);
                pd[di].w = K::route_pred_weights_at(slot);
                pd[di].mask = K::route_mask_layer(t, kL);
                pd[di].mres = cfg.mres ? dres + (size_t) t * kNE : nullptr;
                pd[di].ring = nullptr;
                pd[di].cheap = nullptr;
                pd[di].res = dres + (size_t) t * kNE;
                pd[di].layer = t;
                pd[di].edge_on = cfg.edge >= 0.f;
                pd[di].tail_on = true;
            }
            K::route_pred_select(pd, n, cfg.tail_all, s);
            ck(cudaStreamSynchronize(s), "sync fused");
            const unsigned long long fused_edge = K::route_pred_edge_skips_take(), fused_tail = K::route_pred_tail_read_take();

            // ---- compare ----
            int id_bad = 0, w_bad = 0, l_bad = 0, ties = 0, tie_rows_id_diff = 0;
            for (int di = 0; di < 3; ++di) {
                const int t = l + dists[di], slot = K::route_pred_slot(dists[di]);
                std::vector<float> fl((size_t) n * kNE), fw((size_t) n * kK);
                ck(cudaMemcpy(fl.data(), K::route_pred_logits_at(slot), fl.size() * 4, cudaMemcpyDeviceToHost), "cp fl");
                ck(cudaMemcpy(fw.data(), K::route_pred_weights_at(slot), fw.size() * 4, cudaMemcpyDeviceToHost), "cp fw");
                const int32_t* hslab = K::route_pred_host() + slab(t, slot);
                for (int r = 0; r < n; ++r) {
                    const float* rl = ref_logits.data() + ((size_t) di * kMaxN + r) * kNE;
                    const float* fr = fl.data() + (size_t) r * kNE;
                    if (std::memcmp(rl, fr, kNE * 4) != 0) ++l_bad;
                    // exact ties in the top 11 of the row's logits (the boundary and any pair inside the ten)
                    std::vector<float> top(rl, rl + kNE);
                    std::partial_sort(top.begin(), top.begin() + 11, top.end(), std::greater<float>());
                    bool tie = false;
                    for (int q = 0; q + 1 < 11; ++q) tie |= top[q] == top[q + 1];
                    if (tie) ++ties;
                    bool row_id = true;
                    for (int k = 0; k < kK; ++k)
                        if (ref_ids[((size_t) di * kMaxN + r) * kK + k] != hslab[r * kK + k]) row_id = false;
                    if (!row_id) {
                        ++id_bad;
                        if (tie) ++tie_rows_id_diff;
                    }
                    if (std::memcmp(ref_w.data() + ((size_t) di * kMaxN + r) * kK, fw.data() + (size_t) r * kK, kK * 4) != 0) ++w_bad;
                }
            }
            const bool counters = ref_edge == fused_edge && ref_tail == fused_tail;
            const bool ok = l_bad == 0 && w_bad == 0 && id_bad == 0 && counters;
            std::printf("%s %-58s n=%d  logits-diff %d  weights-diff %d  id-rows-diff %d (on exact ties %d)  edge %llu/%llu  tail %llu/%llu  ties(top11) %d\n",
                        ok ? "PASS" : "FAIL", cfg.name, n, l_bad, w_bad, id_bad, tie_rows_id_diff, ref_edge, fused_edge, ref_tail, fused_tail,
                        ties);
            if (!ok) ++bad;
        }
    }
    std::printf("router_fuse_parity (step 1, look-ahead): %s (%d failing cases)\n", bad == 0 ? "all bit-identical" : "MISMATCH", bad);
    return bad == 0 ? 0 : 1;
}
