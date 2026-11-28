// src/kernels/cuda/route_prior.cu - STRATA_ROUTE_PRIOR=lambda: cache-conditioned routing (opt-in, changes the output).
//
// "Mixture of Cache-Conditional Experts" (arXiv 2412.00099), without training: when a token's 10 experts are picked,
// an expert already in VRAM gets a `lambda` bonus on its logit.  The SELECTION uses the biased logits; the WEIGHTS stay
// the router's (softmax over the 512 unbiased logits), renormalised over the 10 picked like the native router.  Fewer
// missed experts to copy over PCIe or compute on the CPU.  Unlike STRATA_ROUTE_RESIDENT (a missed tail expert swapped
// for the best resident one within a logit margin), every rank can move, by an amount the logits decide.
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/kernels/route_prior.hpp"

#include <cfloat>
#include <cmath>

namespace strata::kernels {
namespace {

__dpct_inline__ float wmax(float v) {
#pragma unroll
    /*
    DPCT1108:0: '__shfl_xor_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    for (int m = 16; m; m >>= 1) v = sycl::fmax(
        v, dpct::experimental::permute_sub_group_by_xor(
               0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
               v, m));
    return v;
}
__dpct_inline__ float wsum(float v) {
#pragma unroll
    /*
    DPCT1108:1: '__shfl_xor_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    for (int m = 16; m; m >>= 1) v +=
        dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), v,
            m);
    return v;
}

// one warp per token; 512 experts = 16 values per lane (as the native router)
/*
DPCT1110:2: The total declared local variable size in device function
route_prior_k exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void route_prior_k(const float *__restrict__ logits,
                                   const int32_t *__restrict__ res,
                                   float lambda, int32_t *__restrict__ ids,
                                   float *__restrict__ weights) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const float *lg = logits + (size_t)item_ct1.get_group(2) * 512;
    int32_t *id = ids + (size_t)item_ct1.get_group(2) * 10;
    float *w = weights + (size_t)item_ct1.get_group(2) * 10;
    const int lane = item_ct1.get_local_id(2);
    float p[16], s[16];
    float mx = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      p[i] = lg[lane + 32 * i]; mx = sycl::fmax(mx, p[i]);
    }
    mx = wmax(mx);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int e = lane + 32 * i;
        s[i] = p[i] + (res[e] >= 0 ? lambda : 0.0f);      // the selection score
        p[i] = sycl::native::exp(p[i] -
                                 mx); // the router's probability (unnormalised)
        sum += p[i];
    }
    (void) wsum(sum);
    float sel_p = 0.0f, sel_sum = 0.0f;
    for (int rank = 0; rank < 10; ++rank) {
        float best = -FLT_MAX, bp = 0.0f;
        int be = 1 << 30;
#pragma unroll
        for (int i = 0; i < 16; ++i)
            if (s[i] > best) { best = s[i]; be = lane + 32 * i; bp = p[i]; }
#pragma unroll
        for (int m = 16; m; m >>= 1) {
            /*
            DPCT1108:3: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            const float ob = dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                best, m);
            /*
            DPCT1108:4: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            const int oe = dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                be, m);
            /*
            DPCT1108:5: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            const float op = dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                bp, m);
            if (ob > best || (ob == best && oe < be)) { best = ob; be = oe; bp = op; }
        }
        if ((be & 31) == lane) s[be >> 5] = -FLT_MAX;
        if (lane == rank) { id[rank] = be; sel_p = bp; }
        sel_sum += bp;                                      // the same on every lane
    }
    if (lane < 10) w[lane] = sel_p / sycl::fmax(sel_sum, 1e-30f);
}

}  // namespace

void route_prior_top10(const float* logits, const int32_t* res_layer, float lambda, int32_t* ids, float* weights,
                             int n_tok, void* stream) {
    if (n_tok <= 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class route_prior_k_8873b1>>(
                sycl::nd_range<3>(sycl::range(1, 1, n_tok) *
                                      sycl::range(1, 1, 32),
                                  sycl::range(1, 1, 32)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        route_prior_k(logits, res_layer, lambda, ids, weights);
                    });
    }
}

}  // namespace strata::kernels
