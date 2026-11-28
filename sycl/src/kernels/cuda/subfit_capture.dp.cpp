// src/kernels/cuda/subfit_capture.cu - the device side of STRATA_SUBFIT_LOG=1 (see include/strata/kernels/subfit_capture.hpp).
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/kernels/subfit_capture.hpp"

namespace strata::kernels {
namespace {

constexpr int kThreads = 256;

__dpct_inline__ void k_capture_x(const float *__restrict__ mixed,
                                 const int32_t *__restrict__ rec,
                                 const int32_t *__restrict__ gate, int layer,
                                 float *__restrict__ x, int64_t N, int n_rows) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = item_ct1.get_group(1);
    if (row >= n_rows || gate[0] != 1 || rec[row * 4] != layer) return;
#pragma unroll
    for (int64_t c =
             item_ct1.get_group(2) * (int64_t)item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         c < N; c += (int64_t)item_ct1.get_group_range(2) *
                     item_ct1.get_local_range(2))
        x[row * N + c] = mixed[row * N + c];
}

__dpct_inline__ void k_probe_pre(const int32_t *__restrict__ ids,
                                 const int32_t *__restrict__ rec,
                                 const int32_t *__restrict__ gate, int layer,
                                 int K, int32_t *__restrict__ pre,
                                 int32_t *__restrict__ clean,
                                 int32_t *__restrict__ flag) {
    if (gate[3] != 1) return;
    const int k =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2);
    if (k < K) {
        pre[layer * K + k] = ids[k];
        flag[layer * K + k] = 0;
    }
    if (k == 0) clean[layer] = rec[0] < 0 ? 1 : 0;   // rec[0]: row 0's first stand-in layer (-1 none), before this layer's route-resident
}

__dpct_inline__ void
k_probe_post(const int32_t *__restrict__ ids, const float *__restrict__ w,
             const int32_t *__restrict__ res, const int32_t *__restrict__ pre,
             const int32_t *__restrict__ clean,
             const int32_t *__restrict__ gate, const float *__restrict__ mixed0,
             int layer, int K, int64_t N, float *__restrict__ x,
             float *__restrict__ w_out, int32_t *__restrict__ flag) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    if (gate[3] != 1) return;
#pragma unroll
    for (int64_t c = item_ct1.get_local_id(2); c < N;
         c += item_ct1.get_local_range(2)) x[layer * N + c] = mixed0[c];
    const int k = item_ct1.get_local_id(2);
    if (k < K) {
        const int32_t e = ids[k];
        // the pick must be the router's own (route-resident did not replace it), its expert resident (d_res >= 0), row 0 clean
        const bool ok = clean[layer] != 0 && pre[layer * K + k] == e && e >= 0 && res[e] >= 0;
        flag[layer * K + k] = ok ? 1 : 0;
        w_out[layer * K + k] = w[k];
    }
}

__dpct_inline__ void k_probe_y(const float *__restrict__ parts,
                               const int32_t *__restrict__ gate, int64_t K_N,
                               float *__restrict__ y) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    if (gate[3] != 1) return;
#pragma unroll
    for (int64_t i =
             item_ct1.get_group(2) * (int64_t)item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         i < K_N; i += (int64_t)item_ct1.get_group_range(2) *
                       item_ct1.get_local_range(2)) y[i] = parts[i];
}

__dpct_inline__ void
k_shadow_xi(const int32_t *__restrict__ ids, const float *__restrict__ mixed0,
            const int32_t *__restrict__ gate, int layer, int K, int64_t N,
            int32_t *__restrict__ ids_out, float *__restrict__ x_out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    if (gate[3] != 2) return;
    if (item_ct1.get_group(2) == 0 && item_ct1.get_local_id(2) < (unsigned)K)
        ids_out[(int64_t)layer * K + item_ct1.get_local_id(2)] =
            ids[item_ct1.get_local_id(2)];
#pragma unroll
    for (int64_t c =
             item_ct1.get_group(2) * (int64_t)item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         c < N; c += (int64_t)item_ct1.get_group_range(2) *
                     item_ct1.get_local_range(2))
        x_out[(int64_t) layer * N + c] = mixed0[c];
}

__dpct_inline__ void k_shadow_y(const float *__restrict__ parts,
                                const int32_t *__restrict__ gate, int layer,
                                int64_t K_N, float *__restrict__ y_out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    if (gate[3] != 2) return;
#pragma unroll
    for (int64_t i =
             item_ct1.get_group(2) * (int64_t)item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         i < K_N; i += (int64_t)item_ct1.get_group_range(2) *
                       item_ct1.get_local_range(2))
        y_out[(int64_t) layer * K_N + i] = parts[i];
}

}  // namespace

void subfit_shadow_xi(const int32_t* ids, const float* mixed0, const int32_t* gate, int layer, int K, int64_t N, int32_t* ids_out,
                      float* x_out, void* stream) {
    const unsigned blocks = (unsigned) ((N + kThreads - 1) / kThreads);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        ((sycl::queue *)(strata::q_of(stream)))
            ->parallel_for<dpct_kernel_name<class k_shadow_xi_cc0df2>>(
                sycl::nd_range<3>(sycl::range(1, 1, blocks < 1 ? 1 : blocks) *
                                      sycl::range(1, 1, kThreads),
                                  sycl::range(1, 1, kThreads)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    k_shadow_xi(ids, mixed0, gate, layer, K, N, ids_out, x_out);
                });
    }
}

void subfit_shadow_y(const float* parts, const int32_t* gate, int layer, int64_t K_N, float* y_out, void* stream) {
    const int64_t blocks = (K_N + kThreads - 1) / kThreads;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        ((sycl::queue *)(strata::q_of(stream)))
            ->parallel_for<dpct_kernel_name<class k_shadow_y_1f5fc8>>(
                sycl::nd_range<3>(
                    sycl::range(1, 1, (unsigned)(blocks < 1 ? 1 : blocks)) *
                        sycl::range(1, 1, kThreads),
                    sycl::range(1, 1, kThreads)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    k_shadow_y(parts, gate, layer, K_N, y_out);
                });
    }
}

void subfit_capture_x(const float* mixed, const int32_t* rec, const int32_t* gate, int layer, float* x, int64_t N, int n_rows,
                      void* stream) {
    if (n_rows <= 0) return;
    const dpct::dim3 grid((unsigned)((N + kThreads - 1) / kThreads),
                          (unsigned)n_rows);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        ((sycl::queue *)(strata::q_of(stream)))
            ->parallel_for<dpct_kernel_name<class k_capture_x_01946d>>(
                sycl::nd_range<3>(grid * sycl::range(1, 1, kThreads),
                                  sycl::range(1, 1, kThreads)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    k_capture_x(mixed, rec, gate, layer, x, N, n_rows);
                });
    }
}

void subfit_probe_pre(const int32_t* ids, const int32_t* rec, const int32_t* gate, int layer, int K, int32_t* pre, int32_t* clean,
                      int32_t* flag, void* stream) {
    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::use_root_sync};

    ((sycl::queue *)(strata::q_of(stream)))
        ->parallel_for<dpct_kernel_name<class k_probe_pre_3fa99f>>(
            sycl::nd_range<3>(sycl::range(1, 1, 32), sycl::range(1, 1, 32)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                k_probe_pre(ids, rec, gate, layer, K, pre, clean, flag);
            });
}

void subfit_probe_post(const int32_t* ids, const float* w, const int32_t* res, const int32_t* pre, const int32_t* clean,
                       const int32_t* gate, const float* mixed0, int layer, int K, int64_t N, float* x, float* w_out, int32_t* flag,
                       void* stream) {
    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::use_root_sync};

    ((sycl::queue *)(strata::q_of(stream)))
        ->parallel_for<dpct_kernel_name<class k_probe_post_632f11>>(
            sycl::nd_range<3>(sycl::range(1, 1, kThreads),
                              sycl::range(1, 1, kThreads)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                k_probe_post(ids, w, res, pre, clean, gate, mixed0, layer, K, N,
                             x, w_out, flag);
            });
}

void subfit_probe_y(const float* parts, const int32_t* gate, int64_t K_N, float* y, void* stream) {
    const int64_t blocks = (K_N + kThreads - 1) / kThreads;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        ((sycl::queue *)(strata::q_of(stream)))
            ->parallel_for<dpct_kernel_name<class k_probe_y_96730e>>(
                sycl::nd_range<3>(
                    sycl::range(1, 1, (unsigned)(blocks < 1 ? 1 : blocks)) *
                        sycl::range(1, 1, kThreads),
                    sycl::range(1, 1, kThreads)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    k_probe_y(parts, gate, K_N, y);
                });
    }
}

}  // namespace strata::kernels
