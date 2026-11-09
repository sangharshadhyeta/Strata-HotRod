// src/kernels/cuda/subfit_apply.cu - SPEC-substitute-fit stage 2: the per-pair correction of a substituted expert's output (see the header).
#include "strata/kernels/subfit_apply.hpp"

#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace strata::kernels {
namespace {

constexpr int kThreads = 256;
constexpr int kMaxR = 32;
constexpr int kMaxK = 16;
constexpr int kWarps = kThreads / 32;

__device__ __forceinline__ float h2f(uint16_t h) { return __half2float(__ushort_as_half(h)); }

// One block per row.  Every row's orig and flag are consumed (reset) here, whatever the layer's fit count, so the next window's route
// kernels start from -1 / 0.
__global__ void k_subfit_apply(SubfitApplyArgs a) {
    const int row = (int) blockIdx.x;
    if (row >= a.n) return;
    __shared__ int slot_k[kMaxK];
    __shared__ float z[kMaxR];
    __shared__ float red[kMaxR][kWarps];
    const int tid = (int) threadIdx.x;
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
    __syncthreads();
    bool any = false;
    for (int k = 0; k < a.K; ++k) any = any || slot_k[k] >= 0;   // (every thread reads the same shared values: uniform)
    if (!any) return;

    // z = A x: one pass over the row's input, r accumulators per thread, then a block reduction of each
    const int r = a.r, N = a.N;
    float acc[kMaxR];
    for (int m = 0; m < kMaxR; ++m) acc[m] = 0.0f;
    const float* xr = a.x + (size_t) row * N;
    for (int j = tid; j < N; j += kThreads) {
        const float xv = xr[j];
        for (int m = 0; m < r; ++m) acc[m] += h2f(a.A[(size_t) m * N + j]) * xv;
    }
    for (int m = 0; m < r; ++m) {
        float v = acc[m];
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
        if ((tid & 31) == 0) red[m][tid >> 5] = v;
    }
    __syncthreads();
    if (tid < r) {
        float s = 0.0f;
        for (int w = 0; w < kWarps; ++w) s += red[tid][w];
        z[tid] = s;
    }
    __syncthreads();

    // parts[row,k] = c * parts[row,k] + B z for each corrected slot of this row
    for (int k = 0; k < a.K; ++k) {
        const int slot = slot_k[k];
        if (slot < 0) continue;
        const float cs = a.c[slot];
        float* p = a.parts + ((size_t) row * a.K + k) * N;
        const uint16_t* Bs = a.B + (size_t) slot * N * r;
        for (int j = tid; j < N; j += kThreads) {
            float v = 0.0f;
            for (int m = 0; m < r; ++m) v += h2f(Bs[(size_t) j * r + m]) * z[m];
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
    k_subfit_apply<<<(unsigned) a.n, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(a);
}

// One table: VRAM when it fits, else pinned host memory the GPU reads over PCIe (the expert cache fills the card at start, and the
// tables are ~70 MB: the lookups are one small read per substitution, a slot's B is read only for a corrected move).  `host` says which.
static bool alloc_table(size_t bytes, void** dev, void** host_ptr, bool vram_ok) {
    *dev = nullptr;
    *host_ptr = nullptr;
    if (vram_ok && cudaMalloc(dev, bytes) == cudaSuccess) return true;
    (void) cudaGetLastError();
    *dev = nullptr;
    void* h = nullptr;
    if (cudaHostAlloc(&h, bytes, cudaHostAllocMapped) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    void* d = nullptr;
    if (cudaHostGetDevicePointer(&d, h, 0) != cudaSuccess) {
        (void) cudaGetLastError();
        (void) cudaFreeHost(h);
        return false;
    }
    *dev = d;
    *host_ptr = h;
    return true;
}

static bool fill_table(void* dev, void* host_ptr, int byte, size_t bytes) {
    if (host_ptr != nullptr) {
        std::memset(host_ptr, byte, bytes);
        return true;
    }
    return cudaMemset(dev, byte, bytes) == cudaSuccess;
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
            if (dev[i] != nullptr && t.host[i] == nullptr) (void) cudaFree(dev[i]);
            else if (t.host[i] != nullptr) (void) cudaFreeHost(t.host[i]);
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
        (void) cudaGetLastError();
        subfit_tables_free(t);
        return false;
    }
    return true;
}

void subfit_tables_free(SubfitTables& t) {
    void* dev[6] = {t.lut, t.A, t.B, t.c, t.resid, t.nfit};
    for (int i = 0; i < 6; ++i) {
        if (t.host[i] != nullptr) (void) cudaFreeHost(t.host[i]);
        else if (dev[i] != nullptr) (void) cudaFree(dev[i]);
    }
    t = SubfitTables{};
}

}  // namespace strata::kernels
