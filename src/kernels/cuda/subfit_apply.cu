// src/kernels/cuda/subfit_apply.cu - SPEC-substitute-fit stage 2: the per-pair correction of a substituted expert's output (see the header).
#include "strata/kernels/subfit_apply.hpp"

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

bool subfit_tables_alloc(SubfitTables& t, int layers, int slots, int N, int r) {
    subfit_tables_free(t);
    t.layers = layers;
    t.slots = slots;
    t.N = N;
    t.r = r;
    void* p = nullptr;
    const size_t lut_n = (size_t) layers * 512 * 512 + (size_t) layers * 512 + (size_t) layers, a_n = (size_t) layers * r * N, b_n = (size_t) slots * N * r;
    bool ok = cudaMalloc(&p, lut_n * sizeof(int16_t)) == cudaSuccess;
    if (ok) {
        t.lut = static_cast<int16_t*>(p);
        ok = cudaMemset(t.lut, 0xff, lut_n * sizeof(int16_t)) == cudaSuccess;   // int16 -1
    }
    if (ok) ok = cudaMalloc(&p, a_n * sizeof(uint16_t)) == cudaSuccess;
    if (ok) {
        t.A = static_cast<uint16_t*>(p);
        ok = cudaMemset(t.A, 0, a_n * sizeof(uint16_t)) == cudaSuccess;
    }
    if (ok) ok = cudaMalloc(&p, b_n * sizeof(uint16_t)) == cudaSuccess;
    if (ok) {
        t.B = static_cast<uint16_t*>(p);
        ok = cudaMemset(t.B, 0, b_n * sizeof(uint16_t)) == cudaSuccess;
    }
    if (ok) ok = cudaMalloc(&p, (size_t) slots * sizeof(float)) == cudaSuccess;
    if (ok) {
        t.c = static_cast<float*>(p);
        ok = cudaMemset(t.c, 0, (size_t) slots * sizeof(float)) == cudaSuccess;
    }
    if (ok) ok = cudaMalloc(&p, (size_t) slots * sizeof(float)) == cudaSuccess;
    if (ok) {
        t.resid = static_cast<float*>(p);
        ok = cudaMemset(t.resid, 0, (size_t) slots * sizeof(float)) == cudaSuccess;
    }
    if (ok) ok = cudaMalloc(&p, (size_t) layers * sizeof(int32_t)) == cudaSuccess;
    if (ok) {
        t.nfit = static_cast<int32_t*>(p);
        ok = cudaMemset(t.nfit, 0, (size_t) layers * sizeof(int32_t)) == cudaSuccess;
    }
    if (!ok) {
        (void) cudaGetLastError();
        subfit_tables_free(t);
        return false;
    }
    return true;
}

void subfit_tables_free(SubfitTables& t) {
    if (t.lut) (void) cudaFree(t.lut);
    if (t.A) (void) cudaFree(t.A);
    if (t.B) (void) cudaFree(t.B);
    if (t.c) (void) cudaFree(t.c);
    if (t.resid) (void) cudaFree(t.resid);
    if (t.nfit) (void) cudaFree(t.nfit);
    t = SubfitTables{};
}

}  // namespace strata::kernels
