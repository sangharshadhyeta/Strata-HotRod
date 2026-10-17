// route_resident_options_parity: the merged stand-in kernel (one warp per token) against OUR PREVIOUS kernel (one thread per token, copied here
// unchanged as route_resident_old_k), over random inputs and every combination of our options: the rank window, the margin, the edge rank, the CCE
// bias, the rank gate (maxrank), the weight floor (wmax), the wanted-expert weight (want_lo), the quality-swap gap (repl_gap), the ring flags, and the
// outputs they drive: ids, weights, stats, the wanted-expert counter, the per-row record, the quality-swap counter and the detail counters.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <random>
#include <vector>
#include "strata/kernels/native_router.hpp"
using strata::kernels::RouteCfg;
namespace {
__global__ void route_resident_old_k(const float* __restrict__ logits, int32_t* __restrict__ ids, float* __restrict__ weights,
                                 const int32_t* __restrict__ res, int n_tok, const RouteCfg* cfg,
                                 unsigned long long* __restrict__ stats, float* __restrict__ want,
                                 unsigned long long* __restrict__ detail, int32_t* __restrict__ rows, float* __restrict__ repl,
                                 const uint8_t* __restrict__ ring) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_tok) return;
    // the floor lives in host-mapped memory, read at every launch, so a captured graph follows a change the host makes between
    // windows: the floor, the margin and the edge margin
    const int lo = ((const volatile RouteCfg*) cfg)->lo;
    const int hi = ((const volatile RouteCfg*) cfg)->hi;
    const float margin = ((const volatile RouteCfg*) cfg)->margin;
    const float edge = ((const volatile RouteCfg*) cfg)->edge;
    const float bias = ((const volatile RouteCfg*) cfg)->bias;
    const int maxrank = ((const volatile RouteCfg*) cfg)->maxrank;
    const float want_lo = ((const volatile RouteCfg*) cfg)->want_lo;
    const float repl_gap = ((const volatile RouteCfg*) cfg)->repl_gap;
    const float wmax = ((const volatile RouteCfg*) cfg)->wmax;
    const float* l = logits + (size_t) t * 512;
    int32_t* id = ids + (size_t) t * 10;
    // detail (decode rows, cumulative, STRATA_TRACE_ADAPT): [r] picks of rank r not resident, [10] the router weight on them
    // (x1e6), [11] rows, [12] rows whose top-1 is not resident, [13] rows with a non-resident pick among the top 5
    if (detail != nullptr) {
        int top5 = 0;
        for (int r = 0; r < 10; ++r) {
            if (res[id[r]] < 0) {
                atomicAdd(detail + r, 1ull);
                atomicAdd(detail + 10, (unsigned long long) (weights[(size_t) t * 10 + r] * 1e6f));
                if (r == 0) atomicAdd(detail + 12, 1ull);
                if (r < 5) top5 = 1;
            }
        }
        atomicAdd(detail + 11, 1ull);
        if (top5) atomicAdd(detail + 13, 1ull);
        // the router weight at each rank (x1e6, summed over rows): [33 + r]
        for (int r = 0; r < 10; ++r) atomicAdd(detail + 33 + r, (unsigned long long) (weights[(size_t) t * 10 + r] * 1e6f));
    }
    // what the router WANTS, counted before any swap: the picks the host sees afterwards are confined to resident experts
    // and cannot say which expert the swap round should bring in (`want` is the layer's slice of the usage counter)
    if (want != nullptr)
        for (int r = 0; r < 10; ++r) atomicAdd(want + id[r], r < lo ? 1.0f : want_lo);   // STRATA_WANT_LOW: a pick the floor replaces counts less
    // SPEC 13.40: the same picks, kept per ROW of the window instead of summed, so the host can count only the rows whose input token was
    // kept once the acceptance is known (`rows` is the layer's [8 rows][10] slice; the atomic count above is then off)
    if (rows != nullptr)
        for (int r = 0; r < 10; ++r) rows[t * 10 + r] = id[r];
    // CCE router bias (Cache-Conditional Experts, arXiv 2412.00099; solid.cpp LLAMA_MOE_CACHE_BIAS): the top-10 is chosen again on the
    // logits with `bias` (= ln(1 + beta), i.e. the score times 1 + beta) added to every resident expert, so a near tie goes to the expert
    // that costs nothing to fetch.  Counted above (want, rows, detail) from the router's own picks.  The weights stay the softmax of the
    // UNBIASED logits of the chosen experts: every chosen expert is computed exactly as the router would score it.
    if (bias > 0.f) {
        int32_t pick[10];
        for (int r = 0; r < 10; ++r) {
            int best = -1;
            float bs = -INFINITY;
            for (int f = 0; f < 512; ++f) {
                bool used = false;
                for (int q = 0; q < r; ++q) used |= pick[q] == f;
                const float s = l[f] + (res[f] >= 0 ? bias : 0.f);
                if (!used && s > bs) { bs = s; best = f; }
            }
            pick[r] = best;
        }
        float m = -INFINITY;
        for (int r = 0; r < 10; ++r) m = fmaxf(m, l[pick[r]]);
        float ex[10], sum = 0.f;
        for (int r = 0; r < 10; ++r) { ex[r] = expf(l[pick[r]] - m); sum += ex[r]; }
        for (int r = 0; r < 10; ++r) { id[r] = pick[r]; weights[(size_t) t * 10 + r] = ex[r] / sum; }
    }
    // STRATA_RING: an expert whose look-ahead ring copy has landed counts as resident (the host plan finds it in the ring)
    auto have = [&](int f) { return res[f] >= 0 || (ring != nullptr && ring[f] != 0); };
    int swaps = 0, tail = 0, before = 0;
    for (int r = 0; r < 10; ++r) before += !have(id[r]);
    for (int r = (edge >= 0.f && lo > 0) ? lo - 1 : lo; r <= hi && r < 10; ++r) {
        const int e = id[r];
        if (have(e)) continue;
        if (wmax > 0.f && weights[(size_t) t * 10 + r] >= wmax) continue;   // a heavy pick stays exact: fetched / computed on the CPU
        ++tail;
        int best = -1;
        float bl = -INFINITY;
        for (int f = 0; f < 512; ++f) {
            if (!have(f) || l[f] <= bl) continue;
            bool used = false;
            for (int q = 0; q < 10; ++q) used |= id[q] == f;
            if (!used) { best = f; bl = l[f]; }
        }
        // the edge rank (lo - 1, a fractional floor such as 1.5) is replaced only by a close substitute (within `edge`)
        // item 21: the stand-in's place in the router's own ranking of all 512 experts (10 = the router's next choice after its ten
        // picks); a stand-in further down than `maxrank` is refused and the exact expert is fetched instead
        if (best >= 0 && maxrank > 0) {
            int place = 0;
            for (int f = 0; f < 512; ++f) place += l[f] > bl;
            if (place >= maxrank) best = -1;
        }
        if (best >= 0 && l[e] - bl <= (r < lo ? edge : margin)) {
            if (detail != nullptr) {
                // what was replaced and by what (per replacement): the weight of the replaced pick, the substitute's place in
                // the router's own ranking of all 512 experts, and the score gap to it
                const float w = weights[(size_t) t * 10 + r];
                const int wb = w < 0.01f ? 0 : w < 0.02f ? 1 : w < 0.05f ? 2 : w < 0.10f ? 3 : w < 0.20f ? 4 : 5;
                int gr = 0;
                for (int f = 0; f < 512; ++f) gr += l[f] > bl;
                const int rb = gr < 20 ? 0 : gr < 40 ? 1 : gr < 80 ? 2 : gr < 160 ? 3 : 4;
                const float gap = l[e] - bl;
                const int gb = gap < 0.5f ? 0 : gap < 1.f ? 1 : gap < 2.f ? 2 : gap < 4.f ? 3 : gap < 8.f ? 4 : 5;
                atomicAdd(detail + 16 + wb, 1ull);
                atomicAdd(detail + 22 + rb, 1ull);
                atomicAdd(detail + 27 + gb, 1ull);
            }
            // SWAP2: this expert was wanted and a stand-in served it; counted for a quality swap only when the stand-in is poor (STRATA_SWAP2_GAP logits)
            if (repl != nullptr && l[e] - bl >= repl_gap) atomicAdd(repl + e, 1.0f);
            id[r] = best;
            ++swaps;
        }
    }
    int after = 0;
    for (int r = 0; r < 10; ++r) after += !have(id[r]);
    if (swaps) {
        float m = -INFINITY;
        for (int r = 0; r < 10; ++r) m = fmaxf(m, l[id[r]]);
        float ex[10], sum = 0.0f;
        for (int r = 0; r < 10; ++r) { ex[r] = expf(l[id[r]] - m); sum += ex[r]; }
        for (int r = 0; r < 10; ++r) weights[(size_t) t * 10 + r] = ex[r] / sum;
    }
    if (stats) {
        atomicAdd(stats + 0, (unsigned long long) tail);
        atomicAdd(stats + 1, (unsigned long long) swaps);
        atomicAdd(stats + 2, (unsigned long long) before);
        atomicAdd(stats + 3, (unsigned long long) after);
    }
}

int ulp_dist(float a, float b) {
    uint32_t x, y;
    std::memcpy(&x, &a, 4); std::memcpy(&y, &b, 4);
    return (int) (x > y ? x - y : y - x);
}
void ck(cudaError_t e, const char* what) { if (e != cudaSuccess) { std::fprintf(stderr, "CUDA %s: %s\n", what, cudaGetErrorString(e)); std::exit(2); } }
constexpr int kE = 512, kT = 10, kMaxTok = 8;
void top10(const float* l, int32_t* out, float* w) {
    std::vector<int> o(kE);
    for (int i = 0; i < kE; ++i) o[i] = i;
    std::partial_sort(o.begin(), o.begin() + kT, o.end(), [&](int a, int b) { return l[a] > l[b] || (l[a] == l[b] && a < b); });
    float m = -INFINITY;
    for (int i = 0; i < kE; ++i) m = std::fmax(m, l[i]);
    double all = 0;
    for (int i = 0; i < kE; ++i) all += std::exp((double) l[i] - m);
    double sel = 0;
    for (int r = 0; r < kT; ++r) { out[r] = o[r]; sel += std::exp((double) l[o[r]] - m) / all; }
    for (int r = 0; r < kT; ++r) w[r] = (float) ((std::exp((double) l[o[r]] - m) / all) / sel);
}
}
int main(int argc, char** argv) {
    const int cases = argc > 1 ? std::atoi(argv[1]) : 600;
    std::mt19937 rng(12345);
    std::normal_distribution<float> nd(0.f, 2.f);
    std::uniform_int_distribution<int> coin(0, 99);
    float *d_logits, *d_w_old, *d_w_new, *d_want_old, *d_want_new, *d_repl_old, *d_repl_new;
    int32_t *d_ids_old, *d_ids_new, *d_res, *d_rows_old, *d_rows_new;
    unsigned long long *d_st_old, *d_st_new, *d_det_old;
    uint8_t* d_ring;
    ck(cudaMalloc(&d_logits, kMaxTok * kE * 4), "logits");
    ck(cudaMalloc(&d_w_old, kMaxTok * kT * 4), "w"); ck(cudaMalloc(&d_w_new, kMaxTok * kT * 4), "w");
    ck(cudaMalloc(&d_ids_old, kMaxTok * kT * 4), "ids"); ck(cudaMalloc(&d_ids_new, kMaxTok * kT * 4), "ids");
    ck(cudaMalloc(&d_res, kE * 4), "res"); ck(cudaMalloc(&d_ring, kE), "ring");
    ck(cudaMalloc(&d_want_old, kE * 4), "want"); ck(cudaMalloc(&d_want_new, kE * 4), "want");
    ck(cudaMalloc(&d_repl_old, kE * 4), "repl"); ck(cudaMalloc(&d_repl_new, kE * 4), "repl");
    ck(cudaMalloc(&d_rows_old, kMaxTok * kT * 4), "rows"); ck(cudaMalloc(&d_rows_new, kMaxTok * kT * 4), "rows");
    ck(cudaMalloc(&d_st_old, 32), "stats"); ck(cudaMalloc(&d_st_new, 32), "stats");
    ck(cudaMalloc(&d_det_old, 48 * 8), "detail");
    strata::kernels::route_detail_alloc();           // the library's own detail counters (read back with route_detail_read) are the new kernel's
    cudaStream_t s = nullptr;
    ck(cudaStreamCreate(&s), "stream");
    int worst_ulp = 0, bad = 0, with_swaps = 0, used_ring = 0, used_bias = 0, used_wmax = 0, used_maxrank = 0, used_edge = 0;
    unsigned long long swaps_total = 0;
    std::vector<float> logits(kMaxTok * kE), w(kMaxTok * kT);
    std::vector<int32_t> ids(kMaxTok * kT), res(kE);
    std::vector<uint8_t> ring(kE);
    for (int c = 0; c < cases; ++c) {
        const int n = 1 + (c % kMaxTok);
        const int lo = (int[]){0, 2, 2, 4, 6}[coin(rng) % 5];
        const int hi = coin(rng) < 80 ? 9 : lo + coin(rng) % (10 - lo);
        const float margin = (float[]){1e30f, 1e30f, 4.f, 1.5f, 0.5f}[coin(rng) % 5];
        const float edge = (lo > 0 && coin(rng) < 30) ? (float[]){0.5f, 1.f, 3.f}[coin(rng) % 3] : -1.f;
        const float bias = coin(rng) < 20 ? (float[]){0.1f, 0.3f, 0.7f}[coin(rng) % 3] : 0.f;
        const int maxrank = coin(rng) < 25 ? (int[]){12, 20, 40}[coin(rng) % 3] : 0;
        const float want_lo = coin(rng) < 30 ? 0.2f : 1.f;
        const float repl_gap = coin(rng) < 50 ? 0.f : (float[]){0.5f, 1.f, 2.f}[coin(rng) % 3];
        const float wmax = coin(rng) < 25 ? (float[]){0.03f, 0.08f, 0.2f}[coin(rng) % 3] : 0.f;
        const bool use_ring = coin(rng) < 40;
        const bool tie_heavy = coin(rng) < 30;
        const int resident_pct = 30 + coin(rng) % 55;
        for (int e = 0; e < kE; ++e) { res[e] = coin(rng) < resident_pct ? (int32_t) (e % 7) : -1; ring[e] = (use_ring && coin(rng) < 6) ? 1 : 0; }
        for (size_t i = 0; i < logits.size(); ++i) { float v = nd(rng); if (tie_heavy) v = std::round(v * 2.f) * 0.5f; logits[i] = v; }
        for (int t = 0; t < n; ++t) top10(logits.data() + (size_t) t * kE, ids.data() + (size_t) t * kT, w.data() + (size_t) t * kT);
        ck(cudaMemcpy(d_logits, logits.data(), logits.size() * 4, cudaMemcpyHostToDevice), "up");
        ck(cudaMemcpy(d_res, res.data(), res.size() * 4, cudaMemcpyHostToDevice), "up");
        ck(cudaMemcpy(d_ring, ring.data(), kE, cudaMemcpyHostToDevice), "up");
        strata::kernels::route_cfg_init(lo, hi, margin, edge, bias, maxrank, want_lo, repl_gap, wmax);
        for (int which = 0; which < 2; ++which) {
            int32_t* di = which ? d_ids_new : d_ids_old; float* dw = which ? d_w_new : d_w_old;
            ck(cudaMemcpy(di, ids.data(), n * kT * 4, cudaMemcpyHostToDevice), "up ids");
            ck(cudaMemcpy(dw, w.data(), n * kT * 4, cudaMemcpyHostToDevice), "up w");
            ck(cudaMemset(which ? d_st_new : d_st_old, 0, 32), "z");
            ck(cudaMemset(which ? d_want_new : d_want_old, 0, kE * 4), "z");
            ck(cudaMemset(which ? d_repl_new : d_repl_old, 0, kE * 4), "z");
            ck(cudaMemset(which ? d_rows_new : d_rows_old, 0xff, kMaxTok * kT * 4), "z");
        }
        ck(cudaMemset(d_det_old, 0, 48 * 8), "z");
        unsigned long long det_before[48], det_new[48], det_old[48];
        strata::kernels::route_detail_read(det_before);
        route_resident_old_k<<<(unsigned) ((n + 31) / 32), 32, 0, s>>>(d_logits, d_ids_old, d_w_old, d_res, n, strata::kernels::route_cfg_device(), d_st_old,
                                                                      d_want_old, d_det_old, d_rows_old, d_repl_old, use_ring ? d_ring : nullptr);
        ck(cudaStreamSynchronize(s), "old");
        strata::kernels::native_route_resident(d_logits, d_ids_new, d_w_new, d_res, n, d_st_new, (void*) s, d_want_new, d_rows_new, d_repl_new,
                                               use_ring ? d_ring : nullptr);
        ck(cudaStreamSynchronize(s), "new");
        strata::kernels::route_detail_read(det_new);
        for (int i = 0; i < 48; ++i) det_new[i] -= det_before[i];
        ck(cudaMemcpy(det_old, d_det_old, 48 * 8, cudaMemcpyDeviceToHost), "dl");
        std::vector<int32_t> io(n * kT), in(n * kT), ro(n * kT), rn(n * kT);
        std::vector<float> wo(n * kT), wn(n * kT), qo(kE), qn(kE), ao(kE), an(kE);
        unsigned long long so[4], sn[4];
        ck(cudaMemcpy(io.data(), d_ids_old, n * kT * 4, cudaMemcpyDeviceToHost), "dl"); ck(cudaMemcpy(in.data(), d_ids_new, n * kT * 4, cudaMemcpyDeviceToHost), "dl");
        ck(cudaMemcpy(wo.data(), d_w_old, n * kT * 4, cudaMemcpyDeviceToHost), "dl"); ck(cudaMemcpy(wn.data(), d_w_new, n * kT * 4, cudaMemcpyDeviceToHost), "dl");
        ck(cudaMemcpy(so, d_st_old, 32, cudaMemcpyDeviceToHost), "dl"); ck(cudaMemcpy(sn, d_st_new, 32, cudaMemcpyDeviceToHost), "dl");
        ck(cudaMemcpy(ao.data(), d_want_old, kE * 4, cudaMemcpyDeviceToHost), "dl"); ck(cudaMemcpy(an.data(), d_want_new, kE * 4, cudaMemcpyDeviceToHost), "dl");
        ck(cudaMemcpy(qo.data(), d_repl_old, kE * 4, cudaMemcpyDeviceToHost), "dl"); ck(cudaMemcpy(qn.data(), d_repl_new, kE * 4, cudaMemcpyDeviceToHost), "dl");
        ck(cudaMemcpy(ro.data(), d_rows_old, n * kT * 4, cudaMemcpyDeviceToHost), "dl"); ck(cudaMemcpy(rn.data(), d_rows_new, n * kT * 4, cudaMemcpyDeviceToHost), "dl");
        int w_ulp = 0;
        for (size_t i = 0; i < wo.size(); ++i) w_ulp = std::max(w_ulp, ulp_dist(wo[i], wn[i]));
        worst_ulp = std::max(worst_ulp, w_ulp);
        bool ok = io == in && ro == rn && std::memcmp(so, sn, 32) == 0 && std::memcmp(det_old, det_new, 48 * 8) == 0 &&
                  w_ulp <= 8 && std::memcmp(qo.data(), qn.data(), kE * 4) == 0;
        for (int e = 0; e < kE && ok; ++e) if (std::fabs(ao[e] - an[e]) > 1e-4f * (1.f + std::fabs(ao[e]))) ok = false;   // float atomics: order differs
        if (!ok && bad < 6) {
            std::printf("MISMATCH case %d (n=%d lo=%d hi=%d margin=%g edge=%g bias=%g maxrank=%d want_lo=%g repl_gap=%g wmax=%g ring=%d): ids %s rows %s stats %s detail %s weights %s repl %s want %s\n",
                        c, n, lo, hi, margin, edge, bias, maxrank, want_lo, repl_gap, wmax, (int) use_ring, io == in ? "ok" : "DIFF", ro == rn ? "ok" : "DIFF",
                        std::memcmp(so, sn, 32) == 0 ? "ok" : "DIFF", std::memcmp(det_old, det_new, 48 * 8) == 0 ? "ok" : "DIFF",
                        w_ulp <= 8 ? "ok" : "DIFF", std::memcmp(qo.data(), qn.data(), kE * 4) == 0 ? "ok" : "DIFF",
                        "see");
        }
        bad += !ok;
        with_swaps += sn[1] > 0;
        swaps_total += sn[1];
        used_ring += use_ring; used_bias += bias > 0.f; used_wmax += wmax > 0.f; used_maxrank += maxrank > 0; used_edge += edge >= 0.f;
    }
    std::printf("route_resident_options_parity: %d cases (ring %d, bias %d, weight floor %d, rank gate %d, edge rank %d), %d with swaps (%llu swaps), weights within %d ulp, %d mismatches\n",
                cases, used_ring, used_bias, used_wmax, used_maxrank, used_edge, with_swaps, swaps_total, worst_ulp, bad);
    std::printf(bad ? "route_resident_options_parity FAILED\n" : "route_resident_options_parity OK\n");
    return bad ? 1 : 0;
}
