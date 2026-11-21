// src/kernels/cuda/inkling2_router.cu - see include/strata/kernels/inkling2_router.hpp.
// Hand-ported from the DeepSeek branch's route_kernel (src/dsv41/ds4_kernels.cu): the biased iterative top-k, the warp-shuffle argmax, the guard
// for a non-finite gate input (Project Maya #4825's chain: NaN routing, then an illegal access); sigmoid scoring for Inkling 2.
#include "strata/kernels/inkling2_router.hpp"

#include <cuda_runtime.h>

#include <cfloat>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int kMaxN = 1024, kMaxK = 16;

// A pick that never came out (every score left is NaN): the lowest expert not picked yet, a valid index, so nothing reads outside the tables.
__device__ __forceinline__ int fallback_pick(const int* pick, int r, int n) {
    int ix = 0;
    for (bool again = true; again;) {
        again = false;
        for (int q = 0; q < r && !again; ++q)
            if (pick[q] == ix) ++ix, again = true;
    }
    return ix < n ? ix : 0;
}

__global__ void route_kernel(const float* __restrict__ logits, const float* __restrict__ bias, int n, int k, float route_scale, int32_t* idx, float* w) {
    __shared__ float sc[kMaxN], bs[kMaxN];
    __shared__ float bv[32];
    __shared__ int bi[32];
    __shared__ int pick[kMaxK];
    __shared__ unsigned char taken[kMaxN];   // a picked expert is skipped by a flag: marking bs with -FLT_MAX (the DeepSeek original) ties with the
                                             // scan's start value, so with non-finite scores the same expert came out again and again
    const int t = blockIdx.x;
    logits += (size_t) t * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        sc[i] = 1.0f / (1.0f + expf(-logits[i]));   // Inkling 2's score: sigmoid
        bs[i] = sc[i] + bias[i];                    // the selection key; the weights use sc alone
        taken[i] = 0;
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    for (int r = 0; r < k; ++r) {
        float best = -FLT_MAX;
        int bix = 0x7fffffff;
        for (int i = threadIdx.x; i < n; i += blockDim.x)
            if (!taken[i] && (bs[i] > best || (bs[i] == best && i < bix))) best = bs[i], bix = i;
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bix, o);
            if (ov > best || (ov == best && oi < bix)) best = ov, bix = oi;
        }
        if (lane == 0) bv[wp] = best, bi[wp] = bix;
        __syncthreads();
        if (threadIdx.x == 0) {
            float b = -FLT_MAX;
            int ix = 0x7fffffff;
            for (int q = 0; q < (int) (blockDim.x >> 5); ++q)
                if (bv[q] > b || (bv[q] == b && bi[q] < ix)) b = bv[q], ix = bi[q];
            if (ix >= n) ix = fallback_pick(pick, r, n);
            pick[r] = ix;
            taken[ix] = 1;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int r = 0; r < k; ++r) s += sc[pick[r]];
        for (int r = 0; r < k; ++r) {
            idx[(size_t) t * k + r] = pick[r];
            w[(size_t) t * k + r] = sc[pick[r]] / (s + 1e-20f) * route_scale;
        }
    }
}

}  // namespace

void inkling2_route(const float* logits, const float* bias, int T, int n, int k, float route_scale, int32_t* idx, float* w, void* stream) {
    if (T <= 0) return;
    if (n <= 0 || n > kMaxN || k <= 0 || k > kMaxK || k > n) {
        std::fprintf(stderr, "inkling2_route: n=%d k=%d outside 1..%d / 1..%d\n", n, k, kMaxN, kMaxK);
        std::exit(1);
    }
    const int threads = n <= 128 ? 128 : 512;   // a multiple of 32: the block reduction walks blockDim / 32 warps
    route_kernel<<<T, threads, 0, (cudaStream_t) stream>>>(logits, bias, n, k, route_scale, idx, w);
}

}  // namespace strata::kernels
