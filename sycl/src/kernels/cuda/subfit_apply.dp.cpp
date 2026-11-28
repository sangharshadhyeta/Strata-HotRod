// src/kernels/cuda/subfit_apply.cu - SPEC-substitute-fit stage 2: the per-pair correction of a substituted expert's output (see the header).
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/kernels/subfit_apply.hpp"

#include <cstring>

namespace strata::kernels {
namespace {

constexpr int kThreads = 256;
constexpr int kMaxR = 32;
constexpr int kMaxK = 16;
constexpr int kWarps = kThreads / 32;

__dpct_inline__ float h2f(uint16_t h) {
    return sycl::vec<sycl::half, 1>(
               sycl::bit_cast<sycl::half, unsigned short>(h))
        .convert<float, sycl::rounding_mode::automatic>()[0];
}

// One block per row.  Every row's orig and flag are consumed (reset) here, whatever the layer's fit count, so the next window's route
// kernels start from -1 / 0.
/*
DPCT1110: The total declared local variable size in device function
k_subfit_apply exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void k_subfit_apply(SubfitApplyArgs a) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = (int)item_ct1.get_group(2);
    if (row >= a.n) return;
    auto &slot_k =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[kMaxK]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &z =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[kMaxR]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &red = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[kMaxR][kWarps]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int tid = (int)item_ct1.get_local_id(2);
    const int nfit = *a.nfit;
    if (tid < a.K) {
        const int idx = row * a.K + tid;
        const int e = a.orig[idx];
        const int s = a.ids[idx];
        const int fl = a.flag[idx];
        int slot = -1;
        if (nfit > 0 && e >= 0 && e < 512 && s >= 0 && s < 512) {
            slot = a.lut[e * 512 + s];                                  // the ladder: the pair,
            if (slot < 0 && a.elut != nullptr) slot = a.elut[e];        // else the (layer, expert) group,
            if (slot < 0 && a.llut != nullptr) slot = a.llut[0];        // else the layer's group
        }
        slot_k[tid] = (slot >= 0 && (a.mode == strata::core::subfit_rule::kModeAll || fl != 0)) ? slot : -1;
        a.orig[idx] = -1;
        a.flag[idx] = 0;
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    bool any = false;
#pragma unroll
    for (int k = 0; k < a.K; ++k) any =
        any ||
        slot_k[k] >= 0; // (every thread reads the same shared values: uniform)
    if (!any) return;

    // z = A x: one pass over the row's input, r accumulators per thread, then a block reduction of each
    const int r = a.r, N = a.N;
    float acc[kMaxR];
#pragma unroll
    for (int m = 0; m < kMaxR; ++m) acc[m] = 0.0f;
    const float* xr = a.x + (size_t) row * N;
    for (int j = tid; j < N; j += kThreads) {
        const float xv = xr[j];
#pragma unroll
        for (int m = 0; m < r; ++m) acc[m] += h2f(a.A[(size_t)m * N + j]) * xv;
    }
    for (int m = 0; m < r; ++m) {
        float v = acc[m];
        /*
DPCT1108: '__shfl_down_sync' was migrated with the experimental feature masked
sub_group function which may not be supported by all compilers or runtimes. You
may need to adjust the code.
*/
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) v +=
            dpct::experimental::shift_sub_group_left(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                v, off);
        if ((tid & 31) == 0) red[m][tid >> 5] = v;
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (tid < r) {
        float s = 0.0f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) s += red[tid][w];
        z[tid] = s;
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // parts[row,k] = c * parts[row,k] + B z for each corrected slot of this row
    for (int k = 0; k < a.K; ++k) {
        const int slot = slot_k[k];
        if (slot < 0) continue;
        const float cs = a.c[slot];
        float* p = a.parts + ((size_t) row * a.K + k) * N;
        const uint16_t* Bs = a.B + (size_t) slot * N * r;
        for (int j = tid; j < N; j += kThreads) {
            float v = 0.0f;
#pragma unroll
            for (int m = 0; m < r; ++m) v += h2f(Bs[(size_t)j * r + m]) * z[m];
            p[j] = cs * p[j] + v;
        }
    }
}

}  // namespace

void subfit_apply(const SubfitApplyArgs& a, void* stream) {
    if (a.parts == nullptr || a.x == nullptr || a.ids == nullptr || a.orig == nullptr || a.flag == nullptr || a.lut == nullptr ||
        a.A == nullptr || a.B == nullptr || a.c == nullptr || a.nfit == nullptr || a.n <= 0)
        return;
    if (a.K < 1 || a.K > kMaxK || a.r < 1 || a.r > kMaxR) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        ((sycl::queue *)(strata::q_of(stream)))
            ->parallel_for<dpct_kernel_name<class k_subfit_apply_98d9f4>>(
                sycl::nd_range<3>(sycl::range(1, 1, (unsigned)a.n) *
                                      sycl::range(1, 1, kThreads),
                                  sycl::range(1, 1, kThreads)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        k_subfit_apply(a);
                    });
    }
}

// One table: VRAM when it fits, else pinned host memory the GPU reads over PCIe (the expert cache fills the card at start, and the
// tables are ~70 MB: the lookups are one small read per substitution, a slot's B is read only for a corrected move).  `host` says which.
static bool alloc_table(size_t bytes, void **dev, void **host_ptr,
                        bool vram_ok) try {
    *dev = nullptr;
    *host_ptr = nullptr;
    if (vram_ok && DPCT_CHECK_ERROR(*dev = (void *)sycl::malloc_device(
                                        bytes, dpct::get_in_order_queue())) ==
                       0) return true;
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    (void)0;
    *dev = nullptr;
    void* h = nullptr;
    /*
    DPCT1048: The original value cudaHostAllocMapped is not meaningful in the
    migrated code and was removed or replaced with 0. You may need to check the
    migrated code.
    */
    if (DPCT_CHECK_ERROR(h = (void *)sycl::malloc_host(
                             bytes, dpct::get_in_order_queue())) != 0) {
        /*
        DPCT1010: SYCL uses exceptions to report errors and does not use the
        error codes. The cudaGetLastError function call was replaced with 0. You
        need to rewrite this code.
        */
        (void)0;
        return false;
    }
    void* d = nullptr;
    if (DPCT_CHECK_ERROR(d = (void *)h) != 0) {
        /*
        DPCT1010: SYCL uses exceptions to report errors and does not use the
        error codes. The cudaGetLastError function call was replaced with 0. You
        need to rewrite this code.
        */
        (void)0;
        (void)DPCT_CHECK_ERROR(sycl::free(h, dpct::get_in_order_queue()));
        return false;
    }
    *dev = d;
    *host_ptr = h;
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static bool fill_table(void *dev, void *host_ptr, int byte, size_t bytes) try {
    if (host_ptr != nullptr) {
        std::memset(host_ptr, byte, bytes);
        return true;
    }
    return DPCT_CHECK_ERROR(
               dpct::get_in_order_queue().memset(dev, byte, bytes).wait()) == 0;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool subfit_tables_alloc(SubfitTables& t, int layers, int slots, int N, int r) {
    subfit_tables_free(t);
    t.layers = layers;
    t.slots = slots;
    t.N = N;
    t.r = r;
    const size_t lut_n = (size_t) layers * 512 * 512 + (size_t) layers * 512 + (size_t) layers, a_n = (size_t) layers * r * N, b_n = (size_t) slots * N * r;
    const size_t bytes[6] = {lut_n * sizeof(int16_t), a_n * sizeof(uint16_t), b_n * sizeof(uint16_t), (size_t) slots * sizeof(float),
                             (size_t) slots * sizeof(float), (size_t) layers * sizeof(int32_t)};
    const int fill[6] = {0xff, 0, 0, 0, 0, 0};   // the LUT is int16 -1, the rest zero
    void* dev[6] = {};
    bool ok = true;
    for (int i = 0; i < 6 && ok; ++i) {
        // the LUT (25 MB) and B (40 MB) always in pinned host memory: in VRAM they took the margin the window graphs need later
        // ('verify: instantiate: out of memory', 9 Nov); the small tables (A, c, resid, nfit) in VRAM when they fit
        ok = alloc_table(bytes[i], &dev[i], &t.host[i], i != 0 && i != 2);
        if (ok) ok = fill_table(dev[i], t.host[i], fill[i], bytes[i]);
        if (ok && t.host[i] != nullptr) t.host_bytes += bytes[i];
        if (!ok) {
            // keep what was taken so subfit_tables_free releases it
            if (dev[i] != nullptr && t.host[i] == nullptr)(void)
                DPCT_CHECK_ERROR(
                    sycl::free(dev[i], dpct::get_in_order_queue()));
            else if (t.host[i] != nullptr)(void) DPCT_CHECK_ERROR(
                sycl::free(t.host[i], dpct::get_in_order_queue()));
            t.host[i] = nullptr;
            dev[i] = nullptr;
        }
    }
    t.lut = static_cast<int16_t*>(dev[0]);
    t.A = static_cast<uint16_t*>(dev[1]);
    t.B = static_cast<uint16_t*>(dev[2]);
    t.c = static_cast<float*>(dev[3]);
    t.resid = static_cast<float*>(dev[4]);
    t.nfit = static_cast<int32_t*>(dev[5]);
    if (!ok) {
        /*
        DPCT1010: SYCL uses exceptions to report errors and does not use the
        error codes. The cudaGetLastError function call was replaced with 0. You
        need to rewrite this code.
        */
        (void)0;
        subfit_tables_free(t);
        return false;
    }
    return true;
}

void subfit_tables_free(SubfitTables& t) {
    void* dev[6] = {t.lut, t.A, t.B, t.c, t.resid, t.nfit};
    for (int i = 0; i < 6; ++i) {
        if (t.host[i] != nullptr)(void)
            DPCT_CHECK_ERROR(sycl::free(t.host[i], dpct::get_in_order_queue()));
        else if (dev[i] != nullptr)(void)
            DPCT_CHECK_ERROR(sycl::free(dev[i], dpct::get_in_order_queue()));
    }
    t = SubfitTables{};
}

}  // namespace strata::kernels
