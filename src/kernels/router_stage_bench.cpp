// router_stage_bench - what each kernel of the router stage costs at the live shapes (SPEC-router-fuse, step 0).
//
// The stage per decode layer: the layer's own routing (the router GEMV, the top-10, the layer's mask, route_resident_k) and the
// look-ahead node (for dists 1 and 2: the next layers' router GEMV, top-10, the mask of the target layer, route_pred_edge_skip,
// route_pred_tail_read; with STRATA_LA_LEARN also the learner's forward, and the drafter's forward on its source layer).
// Each kernel is timed alone: 48 launches (48 layers' pointers, 8 distinct weight copies) captured in one CUDA graph, median of
// `reps` graph launches, microseconds per launch.  Then the two per-layer sequences are timed as one graph each.
//
// The config is the one of the 8095 defaults as far as they are known: lo 3, hi 9, the open margin (1e30, route-resident on),
// edge 1, the mask on, the residency table ~30% resident.  The look-ahead's residency and the mask are synthetic (no model here).
// A kernel with a prerequisite that cannot be had here (the learner, the drafter) prints why and is skipped.
// The edge-skip and tail-read rows reset their ids (a device copy from a golden slab, a graph node of its own, printed as
// "reset ids" so it can be subtracted).
//
//     build/router_stage_bench [N=2560] [reps=20]
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/native_router.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace K = strata::kernels;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); std::exit(1); } } while (0)

namespace {
constexpr int L = 48, NE = 512, KK = 10, NW = 8;
int g_N = 2560;

uint16_t bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t) ((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

// one timed kernel: `body(l, n, s)` launches it for layer l; 48 of them in one graph
template <class Body>
float time_graph(cudaStream_t s, cudaEvent_t e0, cudaEvent_t e1, int reps, Body body) {
    cudaGraph_t g;
    cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    for (int l = 0; l < L; ++l) body(l, s);
    CK(cudaStreamEndCapture(s, &g));
    CK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
    for (int i = 0; i < 3; ++i) CK(cudaGraphLaunch(ge, s));
    CK(cudaStreamSynchronize(s));
    std::vector<float> ms;
    for (int r = 0; r < reps; ++r) {
        CK(cudaEventRecord(e0, s));
        CK(cudaGraphLaunch(ge, s));
        CK(cudaEventRecord(e1, s));
        CK(cudaEventSynchronize(e1));
        float m = 0;
        CK(cudaEventElapsedTime(&m, e0, e1));
        ms.push_back(m);
    }
    std::sort(ms.begin(), ms.end());
    CK(cudaGraphExecDestroy(ge));
    CK(cudaGraphDestroy(g));
    return 1e3f * ms[ms.size() / 2] / L;   // us per layer
}
}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    g_N = argc > 1 ? std::atoi(argv[1]) : 2560;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 20;
    const int N = g_N;
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);

    // weights: 8 distinct router copies (the router is [512 x N] bf16), the rest of the layers reuse them round-robin
    std::vector<uint16_t> wh((size_t) NE * N);
    for (auto& v : wh) v = bf16(0.05f * nd(rng));
    std::vector<uint16_t*> dw(NW);
    for (int i = 0; i < NW; ++i) {
        CK(cudaMalloc((void**) &dw[i], wh.size() * 2));
        CK(cudaMemcpy(dw[i], wh.data(), wh.size() * 2, cudaMemcpyHostToDevice));
    }
    // activations, logits, ids, weights, row shift (device, max 8 rows)
    std::vector<float> xh((size_t) 8 * N);
    for (auto& v : xh) v = nd(rng);
    float *dx, *dlog, *dw_out, *dshift, *dbias, *dgold_logits;
    int32_t *dids, *dgold_ids;
    CK(cudaMalloc((void**) &dx, xh.size() * 4));
    CK(cudaMemcpy(dx, xh.data(), xh.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc((void**) &dlog, (size_t) 8 * NE * 4));
    CK(cudaMalloc((void**) &dw_out, (size_t) 8 * KK * 4));
    CK(cudaMalloc((void**) &dids, (size_t) 8 * KK * 4));
    CK(cudaMalloc((void**) &dshift, 8 * 4));
    CK(cudaMalloc((void**) &dbias, NE * 4));
    CK(cudaMemset(dbias, 0, NE * 4));
    CK(cudaMalloc((void**) &dgold_logits, (size_t) 8 * NE * 4));
    CK(cudaMalloc((void**) &dgold_ids, (size_t) 8 * KK * 4));
    CK(cudaMemset(dshift, 0, 8 * 4));

    // residency per layer (~30% resident: slot >= 0), mask per layer (the last 128 experts, the least used, are masked)
    std::vector<int32_t> res((size_t) L * NE);
    std::vector<uint8_t> mask((size_t) L * NE);
    for (int l = 0; l < L; ++l)
        for (int e = 0; e < NE; ++e) {
            res[(size_t) l * NE + e] = (rng() % 10) < 3 ? (int32_t) (e % 64) : -1;
            mask[(size_t) l * NE + e] = e >= NE - 128 ? 1 : 0;
        }
    int32_t* dres;
    CK(cudaMalloc((void**) &dres, res.size() * 4));
    CK(cudaMemcpy(dres, res.data(), res.size() * 4, cudaMemcpyHostToDevice));

    std::string msg;
    if (!K::route_mask_set(mask.data(), L, NE, 3, 1.0f, msg, 3)) { std::fprintf(stderr, "route_mask_set: %s\n", msg.c_str()); return 1; }
    K::route_cfg_init(3, 9, 1e30f, 1.0f, 0.f, 0, 1.f, 0.f, 0.f, 0.f, 0.f);
    K::route_pred_alloc(L);
    K::route_pred_edge_skip_alloc();
    K::route_pred_tail_read_alloc();
    unsigned long long* dstats;
    CK(cudaMalloc((void**) &dstats, 16 * 8));
    CK(cudaMemset(dstats, 0, 16 * 8));
    if (!K::la_learn_alloc(6, N, 3, false, msg)) std::printf("SKIP learner forward (la_learn_alloc: %s)\n", msg.c_str());
    const bool learner = K::la_learn_active();
    if (learner && !K::la_mtp_alloc(N, false, msg)) std::printf("SKIP drafter forward (la_mtp_alloc: %s)\n", msg.c_str());
    const bool drafter = K::la_mtp_active();

    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    CK(cudaMemcpy(dgold_ids, dids, 8 * KK * 4, cudaMemcpyDeviceToDevice));

    const int ns[] = {1, 2, 4, 8};
    struct Row { const char* name; float us[4]; bool ok; };
    std::vector<Row> rows;
    auto run_row = [&](const char* name, auto body) {
        Row r{name, {0, 0, 0, 0}, true};
        for (int i = 0; i < 4; ++i) {
            const int n = ns[i];
            r.us[i] = time_graph(s, e0, e1, reps, [&](int l, cudaStream_t st) { body(l, n, st); });
        }
        rows.push_back(r);
    };
    // the look-ahead's pred logits and ids: dist 1 (slot 0) at layer l -> target l+1
    auto pred_ids = [&](int l, int dist) { return K::route_pred_dev((l + dist) % L, dist); };
    auto gemv = [&](int l, int n, cudaStream_t st) {
        K::bf16_gemv_fp32_mmvf_multi(dx, N, dw[l % NW], dlog, NE, N, NE, n, st);
    };

    run_row("router GEMV (512 x N)", [&](int l, int n, cudaStream_t st) { gemv(l, n, st); });
    run_row("top-10 (multi)", [&](int l, int n, cudaStream_t st) { K::native_router_top10_multi(dlog, dids, dw_out, n, st); });
    run_row("layer mask (native_route_mask)", [&](int l, int n, cudaStream_t st) {
        K::native_route_mask(dlog, dids, dw_out, K::route_mask_layer(l, L), n, dshift, st, nullptr, l, 0,
                             dres + (size_t) l * NE, nullptr, nullptr);
    });
    run_row("route_resident_k (open margin)", [&](int l, int n, cudaStream_t st) {
        K::native_route_resident(dlog, dids, dw_out, dres + (size_t) l * NE, n, dstats, st, nullptr, nullptr, nullptr, nullptr, nullptr,
                                 nullptr, dshift, nullptr, nullptr, l, nullptr, nullptr, 0);
    });
    run_row("look-ahead: pred GEMV (dist 1 slot)", [&](int l, int n, cudaStream_t st) {
        K::bf16_gemv_fp32_mmvf_multi(dx, N, dw[(l + 1) % NW], K::route_pred_logits(), NE, N, NE, n, st);
    });
    run_row("look-ahead: pred top-10 (dist 1)", [&](int l, int n, cudaStream_t st) {
        K::native_router_top10_multi(K::route_pred_logits(), pred_ids(l, 1), K::route_pred_weights(), n, st);
    });
    run_row("look-ahead: pred mask (dist 1)", [&](int l, int n, cudaStream_t st) {
        const int t = (l + 1) % L;
        K::native_route_mask(K::route_pred_logits(), pred_ids(l, 1), K::route_pred_weights(), K::route_mask_layer(t, L), n, nullptr, st,
                             nullptr, t, 0, dres + (size_t) t * NE, nullptr, nullptr);
    });
    run_row("look-ahead: edge skip (dist 1)", [&](int l, int n, cudaStream_t st) {
        const int t = (l + 1) % L;
        K::route_pred_edge_skip(K::route_pred_logits(), pred_ids(l, 1), dres + (size_t) t * NE, n, st);
    });
    run_row("look-ahead: tail read (dist 1)", [&](int l, int n, cudaStream_t st) {
        const int t = (l + 1) % L;
        K::route_pred_tail_read(dres + (size_t) t * NE, pred_ids(l, 1), n, st, 0);
    });
    run_row("reset ids (pred slab, the edge/tail rows' input)", [&](int l, int n, cudaStream_t st) {
        const int t = (l + 1) % L;
        CK(cudaMemcpyAsync(pred_ids(l, 1), dgold_ids, (size_t) n * KK * 4, cudaMemcpyDeviceToDevice, st));
        (void) t;
    });
    if (learner) {
        run_row("look-ahead: learner forward (dist 1)", [&](int l, int n, cudaStream_t st) {
            K::la_learn_forward(l % 6, 1, dx, n, st);
        });
    }
    if (drafter) {
        run_row("drafter forward (la_mtp)", [&](int l, int n, cudaStream_t st) {
            (void) l;
            K::la_mtp_forward(dx, n, st);
        });
    }

    std::printf("router_stage_bench N=%d NE=%d K=%d layers=%d reps=%d learner=%d drafter=%d\n", N, NE, KK, L, reps, learner ? 1 : 0,
                drafter ? 1 : 0);
    std::printf("%-52s %10s %10s %10s %10s   (us per launch)\n", "kernel", "n=1", "n=2", "n=4", "n=8");
    for (const auto& r : rows)
        std::printf("%-52s %10.2f %10.2f %10.2f %10.2f\n", r.name, r.us[0], r.us[1], r.us[2], r.us[3]);

    // the per-layer sequences as one graph each (us per layer)
    auto own_seq = [&](int l, int n, cudaStream_t st) {
        gemv(l, n, st);
        K::native_router_top10_multi(dlog, dids, dw_out, n, st);
        K::native_route_mask(dlog, dids, dw_out, K::route_mask_layer(l, L), n, dshift, st, nullptr, l, 0, dres + (size_t) l * NE, nullptr,
                             nullptr);
        K::native_route_resident(dlog, dids, dw_out, dres + (size_t) l * NE, n, dstats, st, nullptr, nullptr, nullptr, nullptr, nullptr,
                                 nullptr, dshift, nullptr, nullptr, l, nullptr, nullptr, 0);
    };
    auto la_seq = [&](int l, int n, cudaStream_t st) {
        for (int dist : {1, 2}) {
            const int t = (l + dist) % L;
            K::bf16_gemv_fp32_mmvf_multi(dx, N, dw[t % NW], K::route_pred_logits(), NE, N, NE, n, st);
            K::native_router_top10_multi(K::route_pred_logits(), pred_ids(l, dist), K::route_pred_weights(), n, st);
            K::native_route_mask(K::route_pred_logits(), pred_ids(l, dist), K::route_pred_weights(), K::route_mask_layer(t, L), n, nullptr, st,
                                 nullptr, t, 0, dres + (size_t) t * NE, nullptr, nullptr);
            K::route_pred_edge_skip(K::route_pred_logits(), pred_ids(l, dist), dres + (size_t) t * NE, n, st);
            K::route_pred_tail_read(dres + (size_t) t * NE, pred_ids(l, dist), n, st, 0);
        }
    };
    std::printf("\nper-layer sequences (us per layer, one graph of %d layers):\n", L);
    std::printf("%-52s %10s %10s %10s %10s\n", "sequence", "n=1", "n=2", "n=4", "n=8");
    auto seq_row = [&](const char* name, auto body) {
        float us[4];
        for (int i = 0; i < 4; ++i) us[i] = time_graph(s, e0, e1, reps, [&](int l, cudaStream_t st) { body(l, ns[i], st); });
        std::printf("%-52s %10.2f %10.2f %10.2f %10.2f\n", name, us[0], us[1], us[2], us[3]);
    };
    seq_row("own routing: GEMV + top-10 + mask + resident", [&](int l, int n, cudaStream_t st) { own_seq(l, n, st); });
    seq_row("look-ahead: dists 1, 2 (GEMV, top-10, mask, edge, tail)", [&](int l, int n, cudaStream_t st) { la_seq(l, n, st); });
    seq_row("whole stage (own + look-ahead), today's chain", [&](int l, int n, cudaStream_t st) {
        own_seq(l, n, st);
        la_seq(l, n, st);
    });
    (void) gemv;
    return 0;
}
