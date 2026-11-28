// Arithmetic adapted from the MIT-licensed pinned ggml CUDA
// moe-weighted-reduction.cu at 3cf03257f219afbe7334045ff7c6a06ac68c627d.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/kernels/native_moe.hpp"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <cmath>

namespace strata::kernels {
namespace {
std::atomic<bool> enabled{false};
// GATED (S26 STRATA_LFUSE): `shared` is the shared expert's unscaled output and sg its raw gate logit per token;
// the row is scaled here as shared_expert_multi did (native_scalar_sigmoid_multi_kernel's expression, then one
// rounded multiply = scale_rows_kernel's `out *= g`), then added as before
template <bool ZADD = false, bool GATED = false>
__dpct_inline__ void
combine(const float *__restrict__ parts, const float *__restrict__ weights,
        const float *__restrict__ shared, float *__restrict__ output,
        int64_t n_embd, int k, const float *__restrict__ sg, MoeLearnTap tap) {
    // blockIdx.y = the token of a multi-token launch (0 for the single one)
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t tk = item_ct1.get_group(1);
    parts += tk * k * n_embd; weights += tk * k; if (shared) shared += tk * n_embd; output += tk * n_embd;
    const int64_t col =
        int64_t(item_ct1.get_group(2)) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (col >= n_embd) return;
    const int ys_r = lrn_ys_rank(tap, (int) tk), ye_r = lrn_ye_rank(tap, (int) tk);
    // ZADD: a part is 0.0f + hit (the zeroed row plus moe_hit_add of the verify window), rounded as that add
    float p0 = parts[col];
    if (ZADD) p0 = 0.0f + p0;
    lrn_put(tap, ys_r, ye_r, (int) tk, 0, n_embd, col, p0);
    float sum = p0 * weights[0];
    for (int expert = 1; expert < k; ++expert) {
        float p = parts[int64_t(expert) * n_embd + col];
        if (ZADD) p = 0.0f + p;
        lrn_put(tap, ys_r, ye_r, (int) tk, expert, n_embd, col, p);
        sum += p * weights[expert];
    }
    if constexpr (GATED) {
        // contraction off in this block only: the product is rounded on its own (scale_rows' store), then added
#if defined(__HIPCC__)
#pragma clang fp contract(off)
#endif
        const float g = 1.0f / (1.0f + sycl::native::exp(-sg[tk]));
#if defined(__HIPCC__)
        const float sh = shared[col] * g;
#else
        /*
        DPCT1013: The rounding mode could not be specified and the
        generated code may have different accuracy than the original code.
        Verify the correctness. SYCL math built-in function rounding mode is
        aligned with OpenCL C 1.2 standard.
        */
        const float sh =
            shared[col] *
            g; // nvcc ignores the clang pragma and fuses the product into the
               // add (upstream #5100): a rounded multiply by hand
#endif
        sum += sh;
    } else {
        if (shared) sum += shared[col];
    }
    output[col] = sum;
}
__dpct_inline__ void combine_k10_vec4(const sycl::float4 *__restrict__ parts4,
                                      const float *__restrict__ weights,
                                      const sycl::float4 *__restrict__ shared4,
                                      sycl::float4 *__restrict__ output4,
                                      int64_t n4, MoeLearnTap tap) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t tk = item_ct1.get_group(1);
    parts4 += tk * 10 * n4;
    weights += tk * 10;
    shared4 += tk * n4;
    output4 += tk * n4;
    const int64_t c4 =
        int64_t(item_ct1.get_group(2)) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (c4 >= n4) return;
    const int ys_r = lrn_ys_rank(tap, (int) tk), ye_r = lrn_ye_rank(tap, (int) tk);
    /*
    DPCT1098: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    const float w0 = *(weights + 0);
    const sycl::float4 p0 = parts4[c4];
    ::lrn_put4(tap, ys_r, ye_r, (int)tk, 0, n4 * 4, c4, p0);
    sycl::float4 sum =
        sycl::float4(p0.x() * w0, p0.y() * w0, p0.z() * w0, p0.w() * w0);
#pragma unroll
    for (int expert = 1; expert < 10; ++expert) {
        /*
        DPCT1098: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        const float w = *(weights + expert);
        const sycl::float4 p = parts4[int64_t(expert) * n4 + c4];
        ::lrn_put4(tap, ys_r, ye_r, (int)tk, expert, n4 * 4, c4, p);
        // the documented contract spelled out: the first product rounded, then one FMA per expert in order. Left to the
        // compiler's contraction, a build may fuse a different product (gfx1151: the float4 kernel then differed from the
        // scalar one, native_multi_parity)
        sum.x() = sycl::fma((float)(p.x()), (float)w, sum.x());
        sum.y() = sycl::fma((float)(p.y()), (float)w, sum.y());
        sum.z() = sycl::fma((float)(p.z()), (float)w, sum.z());
        sum.w() = sycl::fma((float)(p.w()), (float)w, sum.w());
    }
    const sycl::float4 sh = shared4[c4];
    sum.x() += sh.x();
    sum.y() += sh.y();
    sum.z() += sh.z();
    sum.w() += sh.w();
    output4[c4] = sum;
}
<<<<<<< sycl
||||||| base
/*
DPCT1110: The total declared local variable size in device function
combine_k10_vec4_hits exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void combine_k10_vec4_hits(
    const sycl::float4 *__restrict__ parts4,
    const sycl::float4 *__restrict__ hits4, const int32_t *__restrict__ dst,
    const int32_t *__restrict__ count, const float *__restrict__ weights,
    const sycl::float4 *__restrict__ shared4,
    sycl::float4 *__restrict__ output4, int64_t n4) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t tk = item_ct1.get_group(1);
    auto &hit_mask = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    if (item_ct1.get_local_id(2) == 0) hit_mask = 0;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    const int nh = *count;
    for (int h = item_ct1.get_local_id(2); h < nh;
         h += item_ct1.get_local_range(2)) {
        const int r = dst[h];
        if (r / 10 == (int)tk)
            dpct::atomic_fetch_or<sycl::access::address_space::generic_space>(
                &hit_mask, 1 << (r % 10));
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    const int mask = hit_mask;
    parts4 += tk * 10 * n4;
    hits4 += tk * 10 * n4;
    weights += tk * 10;
    shared4 += tk * n4;
    output4 += tk * n4;
    const int64_t c4 =
        int64_t(item_ct1.get_group(2)) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (c4 >= n4) return;
    /*
    DPCT1098: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    const float w0 = *(weights + 0);
    sycl::float4 p0 = parts4[c4];
    if (mask & 1) {
        const sycl::float4 hh = hits4[c4]; p0.x() += hh.x(); p0.y() += hh.y();
        p0.z() += hh.z();
        p0.w() += hh.w();
    }
    sycl::float4 sum =
        sycl::float4(p0.x() * w0, p0.y() * w0, p0.z() * w0, p0.w() * w0);
#pragma unroll
    for (int expert = 1; expert < 10; ++expert) {
        /*
        DPCT1098: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        const float w = *(weights + expert);
        sycl::float4 p = parts4[int64_t(expert) * n4 + c4];
        if ((mask >> expert) & 1) {
            const sycl::float4 hh = hits4[int64_t(expert) * n4 + c4];
            p.x() += hh.x();
            p.y() += hh.y();
            p.z() += hh.z();
            p.w() += hh.w();
        }
        sum.x() = sycl::fma(p.x(), (float)w, sum.x());
        sum.y() = sycl::fma(p.y(), (float)w, sum.y());
        sum.z() = sycl::fma(p.z(), (float)w, sum.z());
        sum.w() = sycl::fma(p.w(), (float)w, sum.w());
    }
    const sycl::float4 sh = shared4[c4];
    sum.x() += sh.x();
    sum.y() += sh.y();
    sum.z() += sh.z();
    sum.w() += sh.w();
    output4[c4] = sum;
}
=======
/*
DPCT1110: The total declared local variable size in device function
combine_k10_vec4_hits exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void combine_k10_vec4_hits(
    const sycl::float4 *__restrict__ parts4,
    const sycl::float4 *__restrict__ hits4, const int32_t *__restrict__ dst,
    const int32_t *__restrict__ count, const float *__restrict__ weights,
    const sycl::float4 *__restrict__ shared4,
    sycl::float4 *__restrict__ output4, int64_t n4, MoeLearnTap tap) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t tk = item_ct1.get_group(1);
    auto &hit_mask = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    if (item_ct1.get_local_id(2) == 0) hit_mask = 0;
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const int nh = *count;
    for (int h = item_ct1.get_local_id(2); h < nh;
         h += item_ct1.get_local_range(2)) {
        const int r = dst[h];
        if (r / 10 == (int)tk)
            dpct::atomic_fetch_or<sycl::access::address_space::generic_space>(
                &hit_mask, 1 << (r % 10));
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const int mask = hit_mask;
    parts4 += tk * 10 * n4;
    hits4 += tk * 10 * n4;
    weights += tk * 10;
    shared4 += tk * n4;
    output4 += tk * n4;
    const int64_t c4 =
        int64_t(item_ct1.get_group(2)) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (c4 >= n4) return;
    const int ys_r = lrn_ys_rank(tap, (int) tk), ye_r = lrn_ye_rank(tap, (int) tk);
    /*
    DPCT1098: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    const float w0 = *(weights + 0);
    sycl::float4 p0 = parts4[c4];
    if (mask & 1) {
        const sycl::float4 hh = hits4[c4]; p0.x() += hh.x(); p0.y() += hh.y();
        p0.z() += hh.z();
        p0.w() += hh.w();
    }
    ::lrn_put4(tap, ys_r, ye_r, (int)tk, 0, n4 * 4, c4, p0);
    sycl::float4 sum =
        sycl::float4(p0.x() * w0, p0.y() * w0, p0.z() * w0, p0.w() * w0);
#pragma unroll
    for (int expert = 1; expert < 10; ++expert) {
        /*
        DPCT1098: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        const float w = *(weights + expert);
        sycl::float4 p = parts4[int64_t(expert) * n4 + c4];
        if ((mask >> expert) & 1) {
            const sycl::float4 hh = hits4[int64_t(expert) * n4 + c4];
            p.x() += hh.x();
            p.y() += hh.y();
            p.z() += hh.z();
            p.w() += hh.w();
        }
        ::lrn_put4(tap, ys_r, ye_r, (int)tk, expert, n4 * 4, c4, p);
        sum.x() = sycl::fma(p.x(), (float)w, sum.x());
        sum.y() = sycl::fma(p.y(), (float)w, sum.y());
        sum.z() = sycl::fma(p.z(), (float)w, sum.z());
        sum.w() = sycl::fma(p.w(), (float)w, sum.w());
    }
    const sycl::float4 sh = shared4[c4];
    sum.x() += sh.x();
    sum.y() += sh.y();
    sum.z() += sh.z();
    sum.w() += sh.w();
    output4[c4] = sum;
}
>>>>>>> ours-now
bool valid_span(const void* p, size_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(p);
    return p && address % alignof(float) == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const auto ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}
}
void native_moe_combine_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_moe_combine_enabled() { return enabled.load(std::memory_order_relaxed); }
void native_moe_combine(const float* parts, const float* weights, const float* shared,
                        float* output, int64_t n_embd, int64_t k, void* stream, const MoeLearnTap* tap) {
    if (!stream || n_embd <= 0 || n_embd > std::numeric_limits<int>::max() || k < 1 || k > 15)
        throw std::invalid_argument("native MoE combine requires a stream, positive width and 1..15 experts");
    const size_t row_bytes = size_t(n_embd) * sizeof(float);
    const size_t part_bytes = row_bytes * size_t(k), weight_bytes = size_t(k) * sizeof(float);
    if (!valid_span(parts, part_bytes) || !valid_span(weights, weight_bytes) || !valid_span(output, row_bytes)
            || (shared && !valid_span(shared, row_bytes))
            || overlap(output, row_bytes, parts, part_bytes)
            || overlap(output, row_bytes, weights, weight_bytes)
            || (shared && overlap(output, row_bytes, shared, row_bytes)))
        throw std::invalid_argument("native MoE combine requires aligned spans and disjoint output");
    if (k == 10 && shared != nullptr && (n_embd & 3) == 0 &&
        (((uintptr_t) parts | (uintptr_t) shared | (uintptr_t) output) & 15u) == 0) {
        const int64_t n4 = n_embd >> 2;
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            ((sycl::queue *)(strata::q_of(stream)))
<<<<<<< sycl
                ->parallel_for<dpct_kernel_name<class combine_k10_vec4_503ec2>>(
                    sycl::nd_range<3>(
                        sycl::range(1, 1, unsigned((n4 + 127) / 128)) *
                            sycl::range(1, 1, 128),
                        sycl::range(1, 1, 128)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        combine_k10_vec4(
                            reinterpret_cast<const sycl::float4 *>(parts),
                            weights,
                            reinterpret_cast<const sycl::float4 *>(shared),
                            reinterpret_cast<sycl::float4 *>(output), n4);
                    });
||||||| base
                ->parallel_for<dpct_kernel_name<class combine_k10_vec4_8eb4ee>>(
                    sycl::nd_range<3>(
                        sycl::range(1, 1, unsigned((n4 + 127) / 128)) *
                            sycl::range(1, 1, 128),
                        sycl::range(1, 1, 128)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        combine_k10_vec4(
                            reinterpret_cast<const sycl::float4 *>(parts),
                            weights,
                            reinterpret_cast<const sycl::float4 *>(shared),
                            reinterpret_cast<sycl::float4 *>(output), n4);
                    });
=======
                ->submit([&](sycl::handler &cgh) {
                    MoeLearnTap tap_tap_MoeLearnTap_ct5 =
                        tap ? *tap : MoeLearnTap{};

                    cgh.parallel_for<
                        dpct_kernel_name<class combine_k10_vec4_bd48fc>>(
                        sycl::nd_range<3>(
                            sycl::range(1, 1, unsigned((n4 + 127) / 128)) *
                                sycl::range(1, 1, 128),
                            sycl::range(1, 1, 128)),
                        exp_props, [=](sycl::nd_item<3> item_ct1) {
                            combine_k10_vec4(
                                reinterpret_cast<const sycl::float4 *>(parts),
                                weights,
                                reinterpret_cast<const sycl::float4 *>(shared),
                                reinterpret_cast<sycl::float4 *>(output), n4,
                                tap_tap_MoeLearnTap_ct5);
                        });
                });
>>>>>>> ours-now
        }
    } else {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        ((sycl::queue *)(strata::q_of(stream)))
<<<<<<< sycl
            ->/*
DPCT1050: The template argument of the dpct_kernel_name could not be
deduced. You need to update this code.
*/
            parallel_for<
                dpct_kernel_name<class combine_f74f36,
                                 dpct_kernel_scalar<false>,
                                 dpct_kernel_scalar<false>>>(
                sycl::nd_range<3>(
                    sycl::range(1, 1, unsigned((n_embd + 255) / 256)) *
                        sycl::range(1, 1, 256),
                    sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    combine(parts, weights, shared, output, n_embd, int(k),
                            nullptr);
                });
||||||| base
            ->/*
DPCT1050: The template argument of the dpct_kernel_name could not be
deduced. You need to update this code.
*/
            parallel_for<
                dpct_kernel_name<class combine_efe08f,
                                 dpct_placeholder /*Fix the type mannually*/,
                                 dpct_placeholder /*Fix the type mannually*/>>(
                sycl::nd_range<3>(
                    sycl::range(1, 1, unsigned((n_embd + 255) / 256)) *
                        sycl::range(1, 1, 256),
                    sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    combine(parts, weights, shared, output, n_embd, int(k),
                            nullptr);
                });
=======
            ->submit([&](sycl::handler &cgh) {
                MoeLearnTap tap_tap_MoeLearnTap_ct7 =
                    tap ? *tap : MoeLearnTap{};

                /*
                DPCT1050: The template argument of the dpct_kernel_name
                could not be deduced. You need to update this code.
                */
                cgh.parallel_for<dpct_kernel_name<
                    class combine_3ed626,
                    dpct_placeholder /*Fix the type mannually*/,
                    dpct_placeholder /*Fix the type mannually*/>>(
                    sycl::nd_range<3>(
                        sycl::range(1, 1, unsigned((n_embd + 255) / 256)) *
                            sycl::range(1, 1, 256),
                        sycl::range(1, 1, 256)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        combine(parts, weights, shared, output, n_embd, int(k),
                                nullptr, tap_tap_MoeLearnTap_ct7);
                    });
            });
>>>>>>> ours-now
    }
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const auto error = 0;
    /*
    DPCT1009: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
    /*
    DPCT1001: The statement could not be removed.
    */
    /*
    DPCT1000: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (error !=
        0) throw std::runtime_error(dpct::get_error_string_dummy(error));
}
<<<<<<< sycl
||||||| base
bool native_moe_combine_hits_supported(const float* parts, const float* shared, const float* output, int64_t n_embd, int64_t k, int n_tok) {
    return k == 10 && shared != nullptr && n_tok >= 1 && n_tok <= 8 && n_embd > 0 && (n_embd & 3) == 0 &&
           (((uintptr_t) parts | (uintptr_t) shared | (uintptr_t) output) & 15u) == 0;
}
void native_moe_combine_multi_hits_added(const float* parts, const float* hits, const int32_t* dst, const int32_t* count,
                                         const float* weights, const float* shared, float* output, int64_t n_embd, int64_t k,
                                         int n_tok, void* stream) {
    if (!stream || !native_moe_combine_hits_supported(parts, shared, output, n_embd, k, n_tok) || ((uintptr_t) hits & 15u) != 0)
        throw std::invalid_argument("native MoE combine (hits added) requires k = 10, aligned spans, 1..8 tokens");
    const int64_t n4 = n_embd >> 2;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        ((sycl::queue *)(strata::q_of(stream)))
            ->parallel_for<
                dpct_kernel_name<class combine_k10_vec4_hits_e7ddc5>>(
                sycl::nd_range<3>(sycl::range(1, (unsigned)n_tok,
                                              unsigned((n4 + 127) / 128)) *
                                      sycl::range(1, 1, 128),
                                  sycl::range(1, 1, 128)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    combine_k10_vec4_hits(
                        reinterpret_cast<const sycl::float4 *>(parts),
                        reinterpret_cast<const sycl::float4 *>(hits), dst,
                        count, weights,
                        reinterpret_cast<const sycl::float4 *>(shared),
                        reinterpret_cast<sycl::float4 *>(output), n4);
                });
    }
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const auto error = 0;
    /*
    DPCT1009: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
    /*
    DPCT1001: The statement could not be removed.
    */
    /*
    DPCT1000: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (error !=
        0) throw std::runtime_error(dpct::get_error_string_dummy(error));
}
=======
bool native_moe_combine_hits_supported(const float* parts, const float* shared, const float* output, int64_t n_embd, int64_t k, int n_tok) {
    return k == 10 && shared != nullptr && n_tok >= 1 && n_tok <= 8 && n_embd > 0 && (n_embd & 3) == 0 &&
           (((uintptr_t) parts | (uintptr_t) shared | (uintptr_t) output) & 15u) == 0;
}
void native_moe_combine_multi_hits_added(const float* parts, const float* hits, const int32_t* dst, const int32_t* count,
                                         const float* weights, const float* shared, float* output, int64_t n_embd, int64_t k,
                                         int n_tok, void* stream, const MoeLearnTap* tap) {
    if (!stream || !native_moe_combine_hits_supported(parts, shared, output, n_embd, k, n_tok) || ((uintptr_t) hits & 15u) != 0)
        throw std::invalid_argument("native MoE combine (hits added) requires k = 10, aligned spans, 1..8 tokens");
    const int64_t n4 = n_embd >> 2;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        ((sycl::queue *)(strata::q_of(stream)))
            ->submit([&](sycl::handler &cgh) {
                MoeLearnTap tap_tap_MoeLearnTap_ct8 =
                    tap ? *tap : MoeLearnTap{};

                cgh.parallel_for<
                    dpct_kernel_name<class combine_k10_vec4_hits_982445>>(
                    sycl::nd_range<3>(sycl::range(1, (unsigned)n_tok,
                                                  unsigned((n4 + 127) / 128)) *
                                          sycl::range(1, 1, 128),
                                      sycl::range(1, 1, 128)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        combine_k10_vec4_hits(
                            reinterpret_cast<const sycl::float4 *>(parts),
                            reinterpret_cast<const sycl::float4 *>(hits), dst,
                            count, weights,
                            reinterpret_cast<const sycl::float4 *>(shared),
                            reinterpret_cast<sycl::float4 *>(output), n4,
                            tap_tap_MoeLearnTap_ct8);
                    });
            });
    }
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const auto error = 0;
    /*
    DPCT1009: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
    /*
    DPCT1001: The statement could not be removed.
    */
    /*
    DPCT1000: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (error !=
        0) throw std::runtime_error(dpct::get_error_string_dummy(error));
}
>>>>>>> ours-now
void native_moe_combine_multi(const float* parts, const float* weights, const float* shared, float* output,
                              int64_t n_embd, int64_t k, int n_tok, void* stream, const MoeLearnTap* tap) {
    if (!stream || n_embd <= 0 || k < 1 || k > 15 || n_tok < 1)
        throw std::invalid_argument("native MoE combine (multi) requires a stream, width, 1..15 experts, tokens");
    if (k == 10 && shared != nullptr && (n_embd & 3) == 0 &&
        (((uintptr_t) parts | (uintptr_t) shared | (uintptr_t) output) & 15u) == 0) {
        const int64_t n4 = n_embd >> 2;
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            ((sycl::queue *)(strata::q_of(stream)))
<<<<<<< sycl
                ->parallel_for<dpct_kernel_name<class combine_k10_vec4_44a528>>(
                    sycl::nd_range<3>(sycl::range(1, (unsigned)n_tok,
                                                  unsigned((n4 + 127) / 128)) *
                                          sycl::range(1, 1, 128),
                                      sycl::range(1, 1, 128)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        combine_k10_vec4(
                            reinterpret_cast<const sycl::float4 *>(parts),
                            weights,
                            reinterpret_cast<const sycl::float4 *>(shared),
                            reinterpret_cast<sycl::float4 *>(output), n4);
                    });
||||||| base
                ->parallel_for<dpct_kernel_name<class combine_k10_vec4_73e555>>(
                    sycl::nd_range<3>(sycl::range(1, (unsigned)n_tok,
                                                  unsigned((n4 + 127) / 128)) *
                                          sycl::range(1, 1, 128),
                                      sycl::range(1, 1, 128)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        combine_k10_vec4(
                            reinterpret_cast<const sycl::float4 *>(parts),
                            weights,
                            reinterpret_cast<const sycl::float4 *>(shared),
                            reinterpret_cast<sycl::float4 *>(output), n4);
                    });
=======
                ->submit([&](sycl::handler &cgh) {
                    MoeLearnTap tap_tap_MoeLearnTap_ct5 =
                        tap ? *tap : MoeLearnTap{};

                    cgh.parallel_for<
                        dpct_kernel_name<class combine_k10_vec4_fa405a>>(
                        sycl::nd_range<3>(
                            sycl::range(1, (unsigned)n_tok,
                                        unsigned((n4 + 127) / 128)) *
                                sycl::range(1, 1, 128),
                            sycl::range(1, 1, 128)),
                        exp_props, [=](sycl::nd_item<3> item_ct1) {
                            combine_k10_vec4(
                                reinterpret_cast<const sycl::float4 *>(parts),
                                weights,
                                reinterpret_cast<const sycl::float4 *>(shared),
                                reinterpret_cast<sycl::float4 *>(output), n4,
                                tap_tap_MoeLearnTap_ct5);
                        });
                });
>>>>>>> ours-now
        }
    } else {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        ((sycl::queue *)(strata::q_of(stream)))
<<<<<<< sycl
            ->/*
DPCT1050: The template argument of the dpct_kernel_name could not be
deduced. You need to update this code.
*/
            parallel_for<
                dpct_kernel_name<class combine_d78bb3,
                                 dpct_kernel_scalar<false>,
                                 dpct_kernel_scalar<false>>>(
                sycl::nd_range<3>(sycl::range(1, (unsigned)n_tok,
                                              unsigned((n_embd + 255) / 256)) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    combine(parts, weights, shared, output, n_embd, int(k),
                            nullptr);
                });
||||||| base
            ->/*
DPCT1050: The template argument of the dpct_kernel_name could not be
deduced. You need to update this code.
*/
            parallel_for<
                dpct_kernel_name<class combine_87fa0f,
                                 dpct_placeholder /*Fix the type mannually*/,
                                 dpct_placeholder /*Fix the type mannually*/>>(
                sycl::nd_range<3>(sycl::range(1, (unsigned)n_tok,
                                              unsigned((n_embd + 255) / 256)) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    combine(parts, weights, shared, output, n_embd, int(k),
                            nullptr);
                });
=======
            ->submit([&](sycl::handler &cgh) {
                MoeLearnTap tap_tap_MoeLearnTap_ct7 =
                    tap ? *tap : MoeLearnTap{};

                /*
                DPCT1050: The template argument of the dpct_kernel_name
                could not be deduced. You need to update this code.
                */
                cgh.parallel_for<dpct_kernel_name<
                    class combine_42722c,
                    dpct_placeholder /*Fix the type mannually*/,
                    dpct_placeholder /*Fix the type mannually*/>>(
                    sycl::nd_range<3>(
                        sycl::range(1, (unsigned)n_tok,
                                    unsigned((n_embd + 255) / 256)) *
                            sycl::range(1, 1, 256),
                        sycl::range(1, 1, 256)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        combine(parts, weights, shared, output, n_embd, int(k),
                                nullptr, tap_tap_MoeLearnTap_ct7);
                    });
            });
>>>>>>> ours-now
    }
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const auto error = 0;
    /*
    DPCT1009: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
    /*
    DPCT1001: The statement could not be removed.
    */
    /*
    DPCT1000: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (error !=
        0) throw std::runtime_error(dpct::get_error_string_dummy(error));
}
void native_moe_combine_multi_hits_gated(const float* hits, const float* weights, const float* shared,
                                         const float* shared_gate, float* output, int64_t n_embd, int64_t k, int n_tok,
                                         void* stream, const MoeLearnTap* tap) {
    if (!stream || n_embd <= 0 || k < 1 || k > 15 || n_tok < 1 || !shared || !shared_gate)
        throw std::invalid_argument("native MoE combine (hits, gated) requires a stream, width, 1..15 experts, tokens");
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        ((sycl::queue *)(strata::q_of(stream)))
<<<<<<< sycl
            ->parallel_for<
                dpct_kernel_name<class combine_deb31a, dpct_kernel_scalar<true>,
                                 dpct_kernel_scalar<true>>>(
                sycl::nd_range<3>(sycl::range(1, (unsigned)n_tok,
                                              unsigned((n_embd + 255) / 256)) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    combine<true, true>(hits, weights, shared, output, n_embd,
                                        int(k), shared_gate);
                });
||||||| base
            ->parallel_for<
                dpct_kernel_name<class combine_580e30, dpct_kernel_scalar<true>,
                                 dpct_kernel_scalar<true>>>(
                sycl::nd_range<3>(sycl::range(1, (unsigned)n_tok,
                                              unsigned((n_embd + 255) / 256)) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    combine<true, true>(hits, weights, shared, output, n_embd,
                                        int(k), shared_gate);
                });
=======
            ->submit([&](sycl::handler &cgh) {
                MoeLearnTap tap_tap_MoeLearnTap_ct7 =
                    tap ? *tap : MoeLearnTap{};

                cgh.parallel_for<dpct_kernel_name<class combine_2f00ac,
                                                  dpct_kernel_scalar<true>,
                                                  dpct_kernel_scalar<true>>>(
                    sycl::nd_range<3>(
                        sycl::range(1, (unsigned)n_tok,
                                    unsigned((n_embd + 255) / 256)) *
                            sycl::range(1, 1, 256),
                        sycl::range(1, 1, 256)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        combine<true, true>(hits, weights, shared, output,
                                            n_embd, int(k), shared_gate,
                                            tap_tap_MoeLearnTap_ct7);
                    });
            });
>>>>>>> ours-now
    }
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const auto error = 0;
    /*
    DPCT1009: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
    /*
    DPCT1001: The statement could not be removed.
    */
    /*
    DPCT1000: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (error !=
        0) throw std::runtime_error(dpct::get_error_string_dummy(error));
}
void native_moe_combine_multi_hits(const float* hits, const float* weights, const float* shared, float* output,
                                   int64_t n_embd, int64_t k, int n_tok, void* stream) {
    if (!stream || n_embd <= 0 || k < 1 || k > 15 || n_tok < 1)
        throw std::invalid_argument("native MoE combine (hits) requires a stream, width, 1..15 experts, tokens");
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        ((sycl::queue *)(strata::q_of(stream)))
            ->/*
  DPCT1050: The template argument of the dpct_kernel_name could not be
  deduced. You need to update this code.
  */
            parallel_for<
<<<<<<< sycl
                dpct_kernel_name<class combine_772c4f, dpct_kernel_scalar<true>,
                                 dpct_kernel_scalar<false>>>(
||||||| base
                dpct_kernel_name<class combine_755b16, dpct_kernel_scalar<true>,
                                 dpct_placeholder /*Fix the type mannually*/>>(
=======
                dpct_kernel_name<class combine_a2ddbc, dpct_kernel_scalar<true>,
                                 dpct_placeholder /*Fix the type mannually*/>>(
>>>>>>> ours-now
                sycl::nd_range<3>(sycl::range(1, (unsigned)n_tok,
                                              unsigned((n_embd + 255) / 256)) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    combine<true>(hits, weights, shared, output, n_embd, int(k),
                                  nullptr, MoeLearnTap{});
                });
    }
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const auto error = 0;
    /*
    DPCT1009: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
    /*
    DPCT1001: The statement could not be removed.
    */
    /*
    DPCT1000: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (error !=
        0) throw std::runtime_error(dpct::get_error_string_dummy(error));
}
}
