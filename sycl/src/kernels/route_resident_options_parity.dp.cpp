// route_resident_options_parity: the merged stand-in kernel (one warp per token) against OUR PREVIOUS kernel (one thread per token, copied here
// unchanged as route_resident_old_k), over random inputs and every combination of our options: the rank window, the margin, the edge rank, the CCE
// bias, the rank gate (maxrank), the weight floor (wmax), the wanted-expert weight (want_lo), the quality-swap gap (repl_gap), the ring flags, and the
// outputs they drive: ids, weights, stats, the wanted-expert counter, the per-row record, the quality-swap counter and the detail counters.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
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
/*
DPCT1110: The total declared local variable size in device function
route_resident_old_k exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void route_resident_old_k(
    const float *__restrict__ logits, int32_t *__restrict__ ids,
    float *__restrict__ weights, const int32_t *__restrict__ res, int n_tok,
    const RouteCfg *cfg, unsigned long long *__restrict__ stats,
    float *__restrict__ want, unsigned long long *__restrict__ detail,
    int32_t *__restrict__ rows, float *__restrict__ repl,
    const uint8_t *__restrict__ ring) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
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
#pragma unroll
        for (int r = 0; r < 10; ++r) {
            if (res[id[r]] < 0) {
                dpct::atomic_fetch_add<
                    sycl::access::address_space::generic_space>(detail + r,
                                                                1ull);
                dpct::atomic_fetch_add<
                    sycl::access::address_space::generic_space>(
                    detail + 10,
                    (unsigned long long)(weights[(size_t)t * 10 + r] * 1e6f));
                if (r == 0) dpct::atomic_fetch_add<
                    sycl::access::address_space::generic_space>(detail + 12,
                                                                1ull);
                if (r < 5) top5 = 1;
            }
        }
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            detail + 11, 1ull);
        if (top5)
            dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                detail + 13, 1ull);
        // the router weight at each rank (x1e6, summed over rows): [33 + r]
#pragma unroll
        for (int r = 0; r < 10; ++r)
            dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                detail + 33 + r,
                (unsigned long long)(weights[(size_t)t * 10 + r] * 1e6f));
    }
    // what the router WANTS, counted before any swap: the picks the host sees afterwards are confined to resident experts
    // and cannot say which expert the swap round should bring in (`want` is the layer's slice of the usage counter)
    if (want != nullptr)
#pragma unroll
        for (int r = 0; r < 10; ++r)
            dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                want + id[r],
                r < lo ? 1.0f : want_lo); // STRATA_WANT_LOW: a pick the floor
                                          // replaces counts less
    // SPEC 13.40: the same picks, kept per ROW of the window instead of summed, so the host can count only the rows whose input token was
    // kept once the acceptance is known (`rows` is the layer's [8 rows][10] slice; the atomic count above is then off)
    if (rows != nullptr)
#pragma unroll
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
#pragma unroll
                for (int q = 0; q < r; ++q) used |= pick[q] == f;
                const float s = l[f] + (res[f] >= 0 ? bias : 0.f);
                if (!used && s > bs) { bs = s; best = f; }
            }
            pick[r] = best;
        }
        float m = -INFINITY;
#pragma unroll
        for (int r = 0; r < 10; ++r) m = sycl::fmax(m, l[pick[r]]);
        float ex[10], sum = 0.f;
#pragma unroll
        for (int r = 0; r < 10; ++r) {
            ex[r] = sycl::native::exp(l[pick[r]] - m); sum += ex[r];
        }
#pragma unroll
        for (int r = 0; r < 10; ++r) {
            id[r] = pick[r]; weights[(size_t)t * 10 + r] = ex[r] / sum;
        }
    }
    // STRATA_RING: an expert whose look-ahead ring copy has landed counts as resident (the host plan finds it in the ring)
    auto have = [&](int f) { return res[f] >= 0 || (ring != nullptr && ring[f] != 0); };
    int swaps = 0, tail = 0, before = 0;
#pragma unroll
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
#pragma unroll
            for (int q = 0; q < 10; ++q) used |= id[q] == f;
            if (!used) { best = f; bl = l[f]; }
        }
        // the edge rank (lo - 1, a fractional floor such as 1.5) is replaced only by a close substitute (within `edge`)
        // item 21: the stand-in's place in the router's own ranking of all 512 experts (10 = the router's next choice after its ten
        // picks); a stand-in further down than `maxrank` is refused and the exact expert is fetched instead
        if (best >= 0 && maxrank > 0) {
            int place = 0;
#pragma unroll
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
#pragma unroll
                for (int f = 0; f < 512; ++f) gr += l[f] > bl;
                const int rb = gr < 20 ? 0 : gr < 40 ? 1 : gr < 80 ? 2 : gr < 160 ? 3 : 4;
                const float gap = l[e] - bl;
                const int gb = gap < 0.5f ? 0 : gap < 1.f ? 1 : gap < 2.f ? 2 : gap < 4.f ? 3 : gap < 8.f ? 4 : 5;
                dpct::atomic_fetch_add<
                    sycl::access::address_space::generic_space>(
                    detail + 16 + wb, 1ull);
                dpct::atomic_fetch_add<
                    sycl::access::address_space::generic_space>(
                    detail + 22 + rb, 1ull);
                dpct::atomic_fetch_add<
                    sycl::access::address_space::generic_space>(
                    detail + 27 + gb, 1ull);
            }
            // SWAP2: this expert was wanted and a stand-in served it; counted for a quality swap only when the stand-in is poor (STRATA_SWAP2_GAP logits)
            if (repl != nullptr && l[e] - bl >= repl_gap)
                dpct::atomic_fetch_add<
                    sycl::access::address_space::generic_space>(repl + e, 1.0f);
            id[r] = best;
            ++swaps;
        }
    }
    int after = 0;
#pragma unroll
    for (int r = 0; r < 10; ++r) after += !have(id[r]);
    if (swaps) {
        float m = -INFINITY;
#pragma unroll
        for (int r = 0; r < 10; ++r) m = sycl::fmax(m, l[id[r]]);
        float ex[10], sum = 0.0f;
#pragma unroll
        for (int r = 0; r < 10; ++r) {
            ex[r] = sycl::native::exp(l[id[r]] - m); sum += ex[r];
        }
#pragma unroll
        for (int r = 0; r < 10; ++r) weights[(size_t)t * 10 + r] = ex[r] / sum;
    }
    if (stats) {
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            stats + 0, (unsigned long long)tail);
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            stats + 1, (unsigned long long)swaps);
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            stats + 2, (unsigned long long)before);
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            stats + 3, (unsigned long long)after);
    }
}

int ulp_dist(float a, float b) {
    uint32_t x, y;
    std::memcpy(&x, &a, 4); std::memcpy(&y, &b, 4);
    return (int) (x > y ? x - y : y - x);
}
/*
DPCT1009: SYCL reports errors using exceptions and does not use error
codes. Please replace the "get_error_string_dummy(...)" with a real
error-handling function.
*/
void ck(dpct::err0 e, const char *what) {}
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
    ck(DPCT_CHECK_ERROR(d_logits = (float *)sycl::malloc_device(
                            kMaxTok * kE * 4, dpct::get_in_order_queue())),
       "logits");
    ck(DPCT_CHECK_ERROR(d_w_old = (float *)sycl::malloc_device(
                            kMaxTok * kT * 4, dpct::get_in_order_queue())),
       "w");
        ck(DPCT_CHECK_ERROR(d_w_new = (float *)sycl::malloc_device(
                                kMaxTok * kT * 4, dpct::get_in_order_queue())),
           "w");
    ck(DPCT_CHECK_ERROR(d_ids_old = (int32_t *)sycl::malloc_device(
                            kMaxTok * kT * 4, dpct::get_in_order_queue())),
       "ids");
        ck(DPCT_CHECK_ERROR(d_ids_new = (int32_t *)sycl::malloc_device(
                                kMaxTok * kT * 4, dpct::get_in_order_queue())),
           "ids");
    ck(DPCT_CHECK_ERROR(d_res = (int32_t *)sycl::malloc_device(
                            kE * 4, dpct::get_in_order_queue())),
       "res");
        ck(DPCT_CHECK_ERROR(d_ring = (uint8_t *)sycl::malloc_device(
                                kE, dpct::get_in_order_queue())),
           "ring");
    ck(DPCT_CHECK_ERROR(d_want_old = (float *)sycl::malloc_device(
                            kE * 4, dpct::get_in_order_queue())),
       "want");
        ck(DPCT_CHECK_ERROR(d_want_new = (float *)sycl::malloc_device(
                                kE * 4, dpct::get_in_order_queue())),
           "want");
    ck(DPCT_CHECK_ERROR(d_repl_old = (float *)sycl::malloc_device(
                            kE * 4, dpct::get_in_order_queue())),
       "repl");
        ck(DPCT_CHECK_ERROR(d_repl_new = (float *)sycl::malloc_device(
                                kE * 4, dpct::get_in_order_queue())),
           "repl");
    ck(DPCT_CHECK_ERROR(d_rows_old = (int32_t *)sycl::malloc_device(
                            kMaxTok * kT * 4, dpct::get_in_order_queue())),
       "rows");
        ck(DPCT_CHECK_ERROR(d_rows_new = (int32_t *)sycl::malloc_device(
                                kMaxTok * kT * 4, dpct::get_in_order_queue())),
           "rows");
    ck(DPCT_CHECK_ERROR(d_st_old = (unsigned long long *)sycl::malloc_device(
                            32, dpct::get_in_order_queue())),
       "stats");
        ck(DPCT_CHECK_ERROR(d_st_new =
                                (unsigned long long *)sycl::malloc_device(
                                    32, dpct::get_in_order_queue())),
           "stats");
    ck(DPCT_CHECK_ERROR(d_det_old = (unsigned long long *)sycl::malloc_device(
                            48 * 8, dpct::get_in_order_queue())),
       "detail");
    strata::kernels::route_detail_alloc();           // the library's own detail counters (read back with route_detail_read) are the new kernel's
    dpct::queue_ptr s = &dpct::get_in_order_queue();
    ck(DPCT_CHECK_ERROR(s = dpct::get_current_device().create_queue(true)),
       "stream");
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
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        ck(DPCT_CHECK_ERROR(dpct::get_in_order_queue().memcpy(
               d_logits, logits.data(), logits.size() * 4).wait()),
           "up");
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        ck(DPCT_CHECK_ERROR(dpct::get_in_order_queue().memcpy(d_res, res.data(),
                                                              res.size() * 4).wait()),
           "up");
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        ck(DPCT_CHECK_ERROR(
               dpct::get_in_order_queue().memcpy(d_ring, ring.data(), kE).wait()),
           "up");
        strata::kernels::route_cfg_init(lo, hi, margin, edge, bias, maxrank, want_lo, repl_gap, wmax);
        for (int which = 0; which < 2; ++which) {
            int32_t* di = which ? d_ids_new : d_ids_old; float* dw = which ? d_w_new : d_w_old;
            /*
            DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
            assuming in the original code the source host memory is pageable
            memory. If the memory is not pageable, call wait() on event return
            by memcpy API to ensure synchronization behavior.
            */
            ck(DPCT_CHECK_ERROR(dpct::get_in_order_queue().memcpy(
                   di, ids.data(), n * kT * 4).wait()),
               "up ids");
            /*
            DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
            assuming in the original code the source host memory is pageable
            memory. If the memory is not pageable, call wait() on event return
            by memcpy API to ensure synchronization behavior.
            */
            ck(DPCT_CHECK_ERROR(
                   dpct::get_in_order_queue().memcpy(dw, w.data(), n * kT * 4).wait()),
               "up w");
            ck(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                    .memset(which ? d_st_new : d_st_old, 0, 32)
                                    .wait()),
               "z");
            ck(DPCT_CHECK_ERROR(
                   dpct::get_in_order_queue()
                       .memset(which ? d_want_new : d_want_old, 0, kE * 4)
                       .wait()),
               "z");
            ck(DPCT_CHECK_ERROR(
                   dpct::get_in_order_queue()
                       .memset(which ? d_repl_new : d_repl_old, 0, kE * 4)
                       .wait()),
               "z");
            ck(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                    .memset(which ? d_rows_new : d_rows_old,
                                            0xff, kMaxTok * kT * 4)
                                    .wait()),
               "z");
        }
        ck(DPCT_CHECK_ERROR(
               dpct::get_in_order_queue().memset(d_det_old, 0, 48 * 8).wait()),
           "z");
        unsigned long long det_before[48], det_new[48], det_old[48];
        strata::kernels::route_detail_read(det_before);
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            s->submit([&](sycl::handler &cgh) {
                auto strata_kernels_route_cfg_device_ct5 =
                    strata::kernels::route_cfg_device();

                cgh.parallel_for<
                    dpct_kernel_name<class route_resident_old_k_c6517e>>(
                    sycl::nd_range<3>(
                        sycl::range(1, 1, (unsigned)((n + 31) / 32)) *
                            sycl::range(1, 1, 32),
                        sycl::range(1, 1, 32)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        route_resident_old_k(
                            d_logits, d_ids_old, d_w_old, d_res, n,
                            strata_kernels_route_cfg_device_ct5, d_st_old,
                            d_want_old, d_det_old, d_rows_old, d_repl_old,
                            use_ring ? d_ring : nullptr);
                    });
            });
        }
        ck(DPCT_CHECK_ERROR(s->wait()), "old");
        strata::kernels::native_route_resident(d_logits, d_ids_new, d_w_new, d_res, n, d_st_new, (void*) s, d_want_new, d_rows_new, d_repl_new,
                                               use_ring ? d_ring : nullptr);
        ck(DPCT_CHECK_ERROR(s->wait()), "new");
        strata::kernels::route_detail_read(det_new);
        for (int i = 0; i < 48; ++i) det_new[i] -= det_before[i];
        ck(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                .memcpy(det_old, d_det_old, 48 * 8)
                                .wait()),
           "dl");
        std::vector<int32_t> io(n * kT), in(n * kT), ro(n * kT), rn(n * kT);
        std::vector<float> wo(n * kT), wn(n * kT), qo(kE), qn(kE), ao(kE), an(kE);
        unsigned long long so[4], sn[4];
        ck(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                .memcpy(io.data(), d_ids_old, n * kT * 4)
                                .wait()),
           "dl");
            ck(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                    .memcpy(in.data(), d_ids_new, n * kT * 4)
                                    .wait()),
               "dl");
        ck(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                .memcpy(wo.data(), d_w_old, n * kT * 4)
                                .wait()),
           "dl");
            ck(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                    .memcpy(wn.data(), d_w_new, n * kT * 4)
                                    .wait()),
               "dl");
        ck(DPCT_CHECK_ERROR(
               dpct::get_in_order_queue().memcpy(so, d_st_old, 32).wait()),
           "dl");
            ck(DPCT_CHECK_ERROR(
                   dpct::get_in_order_queue().memcpy(sn, d_st_new, 32).wait()),
               "dl");
        ck(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                .memcpy(ao.data(), d_want_old, kE * 4)
                                .wait()),
           "dl");
            ck(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                    .memcpy(an.data(), d_want_new, kE * 4)
                                    .wait()),
               "dl");
        ck(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                .memcpy(qo.data(), d_repl_old, kE * 4)
                                .wait()),
           "dl");
            ck(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                    .memcpy(qn.data(), d_repl_new, kE * 4)
                                    .wait()),
               "dl");
        ck(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                .memcpy(ro.data(), d_rows_old, n * kT * 4)
                                .wait()),
           "dl");
            ck(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                    .memcpy(rn.data(), d_rows_new, n * kT * 4)
                                    .wait()),
               "dl");
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
