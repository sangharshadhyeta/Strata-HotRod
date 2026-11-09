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

// ---- step 2: the layer's own post-routing stages, the separate calls against native_route_post (bit for bit) ----
// The own routing of layer l: the router GEMV and the top-10 (the same ids and weights start both chains), then the mask, the resident
// stage and (with the learner) the label copy: the separate calls, or the one launch.  Compared: ids exact, weights and row_shift bitwise,
// the resident counters and the mask's seen / moved counts equal, the label bitwise.
int run_post_checks(const float* dx, const uint16_t* dwall, const int32_t* dres, bool learner_ok, const Config& cfg, cudaStream_t s,
                    int n, int l) {
    const size_t nl = (size_t) kMaxN * kNE;
    float *dlog, *dw_a, *dw_b, *dsh_a, *dsh_b;
    int32_t *dids_a, *dids_b;
    unsigned long long *dst_a, *dst_b;
    ck(cudaMalloc((void**) &dlog, nl * 4), "post logits");
    ck(cudaMalloc((void**) &dw_a, kMaxN * kK * 4), "post w a");
    ck(cudaMalloc((void**) &dw_b, kMaxN * kK * 4), "post w b");
    ck(cudaMalloc((void**) &dsh_a, kMaxN * 4), "post shift a");
    ck(cudaMalloc((void**) &dsh_b, kMaxN * 4), "post shift b");
    ck(cudaMalloc((void**) &dids_a, kMaxN * kK * 4), "post ids a");
    ck(cudaMalloc((void**) &dids_b, kMaxN * kK * 4), "post ids b");
    ck(cudaMalloc((void**) &dst_a, 16 * 8), "post stats a");
    ck(cudaMalloc((void**) &dst_b, 16 * 8), "post stats b");
    ck(cudaMemsetAsync(dsh_a, 0, kMaxN * 4, s), "z");
    ck(cudaMemsetAsync(dsh_b, 0, kMaxN * 4, s), "z");
    ck(cudaMemsetAsync(dst_a, 0, 16 * 8, s), "z");
    ck(cudaMemsetAsync(dst_b, 0, 16 * 8, s), "z");
    K::bf16_gemv_fp32_mmvf_multi(dx, kN, dwall + (size_t) l * kNE * kN, dlog, kNE, kN, kNE, n, s);
    K::native_router_top10_multi(dlog, dids_a, dw_a, n, s);
    ck(cudaMemcpyAsync(dids_b, dids_a, kMaxN * kK * 4, cudaMemcpyDeviceToDevice, s), "ids copy");
    ck(cudaMemcpyAsync(dw_b, dw_a, kMaxN * kK * 4, cudaMemcpyDeviceToDevice, s), "w copy");
    ck(cudaStreamSynchronize(s), "post start");
    const bool label_on = cfg.learn && learner_ok;
    const uint8_t* mask_l = K::route_mask_layer(l, kL);
    const int32_t* res_l = dres + (size_t) l * kNE;
    const int32_t* mres = cfg.mres ? dres + (size_t) l * kNE : nullptr;
    unsigned long long seen = 0, moved = 0;
    K::route_mask_read(seen, moved);   // (reset the counters)

    // ---- the separate calls ----
    if (label_on) K::la_learn_label(l, dlog, n, s);
    K::native_route_mask(dlog, dids_a, dw_a, mask_l, n, dsh_a, s, nullptr, l, 0, mres, nullptr, nullptr);
    K::native_route_resident(dlog, dids_a, dw_a, res_l, n, dst_a, s, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, dsh_a, nullptr,
                             nullptr, l, nullptr, nullptr, 0);
    ck(cudaStreamSynchronize(s), "post ref");
    unsigned long long ref_seen = 0, ref_moved = 0;
    K::route_mask_read(ref_seen, ref_moved);
    std::vector<float> lab_ref((size_t) n * kNE, 0.f), lab_fus((size_t) n * kNE, 0.f);
    if (label_on) ck(cudaMemcpy(lab_ref.data(), K::la_learn_label_dev(l, n), lab_ref.size() * 4, cudaMemcpyDeviceToHost), "lab ref");

    // ---- the one launch ----
    K::RoutePostArgs pa{};
    pa.logits = dlog; pa.ids = dids_b; pa.weights = dw_b; pa.n_tok = n; pa.layer = l; pa.row_base = 0;
    pa.mask = mask_l; pa.mres = mres; pa.row_shift = dsh_b; pa.res = res_l; pa.rstats = dst_b;
    pa.label = label_on ? K::la_learn_label_dev(l, n) : nullptr;
    K::native_route_post(pa, s);
    ck(cudaStreamSynchronize(s), "post fused");
    unsigned long long fus_seen = 0, fus_moved = 0;
    K::route_mask_read(fus_seen, fus_moved);
    if (label_on) ck(cudaMemcpy(lab_fus.data(), K::la_learn_label_dev(l, n), lab_fus.size() * 4, cudaMemcpyDeviceToHost), "lab fus");

    // ---- compare ----
    std::vector<int32_t> ia(kMaxN * kK), ib(kMaxN * kK);
    std::vector<float> wa(kMaxN * kK), wb(kMaxN * kK), sa(kMaxN), sb(kMaxN);
    unsigned long long ta[16], tb2[16];
    ck(cudaMemcpy(ia.data(), dids_a, ia.size() * 4, cudaMemcpyDeviceToHost), "ia");
    ck(cudaMemcpy(ib.data(), dids_b, ib.size() * 4, cudaMemcpyDeviceToHost), "ib");
    ck(cudaMemcpy(wa.data(), dw_a, wa.size() * 4, cudaMemcpyDeviceToHost), "wa");
    ck(cudaMemcpy(wb.data(), dw_b, wb.size() * 4, cudaMemcpyDeviceToHost), "wb");
    ck(cudaMemcpy(sa.data(), dsh_a, sa.size() * 4, cudaMemcpyDeviceToHost), "sa");
    ck(cudaMemcpy(sb.data(), dsh_b, sb.size() * 4, cudaMemcpyDeviceToHost), "sb");
    ck(cudaMemcpy(ta, dst_a, sizeof ta, cudaMemcpyDeviceToHost), "ta");
    ck(cudaMemcpy(tb2, dst_b, sizeof tb2, cudaMemcpyDeviceToHost), "tb");
    int id_bad = 0, w_bad = 0, sh_bad = 0, lab_bad = 0, ties = 0;
    for (int r = 0; r < n; ++r) {
        bool same = true;
        for (int k = 0; k < kK; ++k) same &= ia[r * kK + k] == ib[r * kK + k];
        if (!same) ++id_bad;
        if (std::memcmp(&wa[r * kK], &wb[r * kK], kK * 4) != 0) ++w_bad;
        if (std::memcmp(&sa[r], &sb[r], 4) != 0) ++sh_bad;
    }
    if (label_on && std::memcmp(lab_ref.data(), lab_fus.data(), lab_ref.size() * 4) != 0) lab_bad = 1;
    for (int i = 0; i < 16; ++i)
        if (ta[i] != tb2[i]) { ++id_bad; break; }
    if (ref_seen != fus_seen || ref_moved != fus_moved) ++id_bad;
    // the ties of the own logits (as above): reported only
    for (int r = 0; r < n; ++r) {
        std::vector<float> row(kNE);
        ck(cudaMemcpy(row.data(), dlog + (size_t) r * kNE, kNE * 4, cudaMemcpyDeviceToHost), "row");
        std::partial_sort(row.begin(), row.begin() + 11, row.end(), std::greater<float>());
        for (int q = 0; q + 1 < 11; ++q)
            if (row[q] == row[q + 1]) { ++ties; break; }
    }
    const bool ok = id_bad == 0 && w_bad == 0 && sh_bad == 0 && lab_bad == 0;
    std::printf("%s post  %-58s n=%d  ids-diff %d  weights-diff %d  shift-diff %d  label-diff %d  mask seen/moved %llu/%llu vs %llu/%llu  stats %s  ties(top11) %d\n",
                ok ? "PASS" : "FAIL", cfg.name, n, id_bad, w_bad, sh_bad, lab_bad, ref_seen, ref_moved, fus_seen, fus_moved,
                std::memcmp(ta, tb2, sizeof ta) == 0 ? "equal" : "DIFFER", ties);
    cudaFree(dlog); cudaFree(dw_a); cudaFree(dw_b); cudaFree(dsh_a); cudaFree(dsh_b); cudaFree(dids_a); cudaFree(dids_b);
    cudaFree(dst_a); cudaFree(dst_b);
    return ok ? 0 : 1;
}

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
    // SPEC-recheck-gated: the hold bitmask and the per-row shift caps the post-routing comparison runs with (their third pass, below)
    if (!K::route_fragile_control_alloc() || !K::route_shift_cap_alloc()) { std::fprintf(stderr, "hold / cap flags: no host-mapped memory\n"); return 2; }
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
            bad += run_post_checks(dx, dwall, dres, learner_ok, cfg, s, n, l);
            // SPEC-recheck-gated: the same comparison with the hold (rows 0 and 2 held) and the finite per-row shift caps (small, so many moves are
            // refused): the fused post kernel against the separate calls, both reading the same device flags.  Then the flags are reset.
            {
                K::route_fragile_hold(0x5u);
                float caps[32];
                for (int i = 0; i < 32; ++i) caps[i] = i < 8 ? 0.02f * (float) (1 + i) : INFINITY;
                K::route_shift_cap_set(caps, 32);
                const int held_bad = run_post_checks(dx, dwall, dres, learner_ok, cfg, s, n, l);
                bad += held_bad;
                std::printf("%s %-58s n=%d  hold 0x5 and shift caps: %s\n", held_bad == 0 ? "PASS" : "FAIL", cfg.name, n,
                            held_bad == 0 ? "fused == separate" : "MISMATCH");
                K::route_fragile_hold(0u);
                K::route_shift_cap_set(nullptr, 0);
            }
        }
    }
    std::printf("router_fuse_parity (step 1 look-ahead, step 2 post-routing): %s (%d failing cases)\n",
                bad == 0 ? "all bit-identical" : "MISMATCH", bad);
    return bad == 0 ? 0 : 1;
}
