// Adapted from topk-moe.cu/common.cuh in llama.cpp
// 3cf03257f219afbe7334045ff7c6a06ac68c627d; finite F32, 512-expert/10-output path.
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
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/la_cadence.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <cstdio>
#include <iterator>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace strata::kernels {
namespace {
std::atomic<bool> enabled{false};
__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) value += __shfl_xor_sync(0xffffffffu, value, mask, 32);
    return value;
}
__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, mask, 32));
    return value;
}
__launch_bounds__(256, 1)
__global__ void route(const float* __restrict__ logits, int32_t* __restrict__ ids,
                      float* __restrict__ weights) {
    // Preserve the pinned 32x8 block geometry; only row zero is active here.
    // blockIdx.x = the token (a multi-token launch; 0 for the single one)
    logits += (size_t) blockIdx.x * 512; ids += (size_t) blockIdx.x * 10; weights += (size_t) blockIdx.x * 10;
    if (threadIdx.y != 0) return;
    const int lane = threadIdx.x;
    float values[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) values[i] = logits[lane + i * 32];
    __syncthreads();
    float maximum = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; ++i) maximum = max(maximum, values[i]);
    maximum = warp_max(maximum);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] = expf(values[i] - maximum);
        sum += values[i];
    }
    const float reciprocal = 1.0f / warp_sum(sum);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] *= reciprocal;
        if (__isnanf(values[i])) values[i] = -FLT_MAX;
    }
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < 10; ++rank) {
        float best = values[0];
        int expert = lane;
#pragma unroll
        for (int i = 1; i < 16; ++i) {
            if (values[i] > best) { best = values[i]; expert = lane + i * 32; }
        }
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, best, mask, 32);
            const int other_id = __shfl_xor_sync(0xffffffffu, expert, mask, 32);
            if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
        }
        if ((expert & 31) == lane) {
            values[expert / 32] = -INFINITY;
            ids[rank] = expert;
            // Deliberately accumulate by WINNING EXPERT lane, not output rank.
            // Multiple selected experts in one lane add in selection order.
            selected_sum += best;
        }
        if (rank == lane) selected = best;
    }
    selected_sum = max(warp_sum(selected_sum), 6.103515625e-5f);
    const float inverse_selected_sum = 1.0f / selected_sum;
    if (lane < 10) weights[lane] = selected * inverse_selected_sum;
}
// One warp's top-10 of one token's 512 logits (the route_multi body, shared with the fused router below): lane = the warp's lane.
// The logits are read through L2 (__ldcg), so a warp may read a row another CTA of the same launch just wrote.
__device__ __forceinline__ void route_warp_top10(const float* __restrict__ logits, int32_t* __restrict__ ids,
                                                 float* __restrict__ weights, const int lane) {
    float values[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) values[i] = __ldcg(logits + lane + i * 32);
    __syncwarp();
    float maximum = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; ++i) maximum = max(maximum, values[i]);
    maximum = warp_max(maximum);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] = expf(values[i] - maximum);
        sum += values[i];
    }
    const float reciprocal = 1.0f / warp_sum(sum);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] *= reciprocal;
        if (__isnanf(values[i])) values[i] = -FLT_MAX;
    }
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < 10; ++rank) {
        float best = values[0];
        int expert = lane;
#pragma unroll
        for (int i = 1; i < 16; ++i) {
            if (values[i] > best) { best = values[i]; expert = lane + i * 32; }
        }
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, best, mask, 32);
            const int other_id = __shfl_xor_sync(0xffffffffu, expert, mask, 32);
            if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
        }
        if ((expert & 31) == lane) {
            values[expert / 32] = -INFINITY;
            ids[rank] = expert;
            selected_sum += best;
        }
        if (rank == lane) selected = best;
    }
    selected_sum = max(warp_sum(selected_sum), 6.103515625e-5f);
    const float inverse_selected_sum = 1.0f / selected_sum;
    if (lane < 10) weights[lane] = selected * inverse_selected_sum;
}
__launch_bounds__(256, 1)
__global__ void route_multi(const float* __restrict__ logits, int32_t* __restrict__ ids,
                            float* __restrict__ weights, int n_tok) {
    const int tk = (int) blockIdx.x * 8 + (int) threadIdx.y;
    if (tk >= n_tok) return;
    route_warp_top10(logits + (size_t) tk * 512, ids + (size_t) tk * 10, weights + (size_t) tk * 10, threadIdx.x);
}
// STRATA_ROUTER_FUSED (opt-in): the router GEMV (bf16_f32_mmvf_multi_kernel, NT = n_tok, n_in = 2560, 256 threads: the 224/256
// choice is mmvf_block_size's) and the 512 x 10 top-10 in ONE launch.  kFusedRows rows per CTA, one warp per row.  The CTA
// stages the n_tok activation rows in shared memory once (40 KB at 4 x 2560).  Each row keeps the GEMV's arithmetic exactly: the
// 256 "virtual threads" of the 256-thread kernel are emulated by the warp's 32 lanes x 8 (virtual thread v = lane + 32 s, pairs
// v, v + 256, ...), each with its own fp32 accumulator per token, in the same pair order and two ordered FMAs; the 8 virtual
// warps are then reduced with the same xor butterfly, the warp partials sit in lanes 0..7 and lanes 8..31 hold 0.  The logits are
// therefore bit-identical to the two-launch path.  The CTA that finishes LAST (atomic counter, fenced) runs the top-10 of every
// token with route_warp_top10, the very code route_multi runs, so the ids and weights (ties: the lower index) are identical too.
// R = 16 rows (warps) per CTA: the 40 KB of staged activations allow 2 CTAs per SM (100 KB shared on GA106), so 512 / 16 = 32 CTAs
// are all resident in one wave (R = 8 would be 64 CTAs, 8 of them queued behind a full wave; R = 4 or 2 queue far more).
constexpr int kFusedRows = 16;
constexpr int kFusedThreads = 32 * kFusedRows;
constexpr int kFusedMaxTok = 4;               // the shared-memory staging bound (n_tok * n_in * 4 bytes, under 48 KB)
static unsigned int* g_fused_done = nullptr;  // the CTA counter: the last CTA to finish resets it to 0 (allocated at init)
__launch_bounds__(kFusedThreads, 1)
__global__ void router_gemv_top10_fused_k(const float* __restrict__ x, int64_t ldx, const uint16_t* __restrict__ w,
                                          float* __restrict__ logits, int32_t* __restrict__ ids, float* __restrict__ weights,
                                          int n_in, int n_tok, unsigned int* __restrict__ done) {
    extern __shared__ float2 sx[];   // [n_tok][n_in / 2]: the activations, read once per CTA
    __shared__ bool last;
    const int tid = (int) threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int half = n_in / 2;
    for (int i = tid; i < n_tok * half; i += kFusedThreads) {
        const int k = i / half, p = i - k * half;
        sx[i] = reinterpret_cast<const float2*>(x + (size_t) k * ldx)[p];
    }
    __syncthreads();
    const int row = (int) blockIdx.x * kFusedRows + warp;
    const uint32_t* weights2 = reinterpret_cast<const uint32_t*>(w + (size_t) row * n_in);
    float acc[8][kFusedMaxTok];      // [virtual warp s][token k]
#pragma unroll
    for (int s = 0; s < 8; ++s)
#pragma unroll
        for (int k = 0; k < kFusedMaxTok; ++k) acc[s][k] = 0.0f;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        for (int pair = lane + 32 * s; pair < half; pair += 256) {   // virtual thread v = lane + 32 s, stride 256 pairs
            const uint32_t weight = weights2[pair];
            const float w0 = f32_from_bf16((uint16_t) weight), w1 = f32_from_bf16((uint16_t) (weight >> 16));
#pragma unroll
            for (int k = 0; k < kFusedMaxTok; ++k) {
                if (k < n_tok) {
                    const float2 input = sx[k * half + pair];
                    acc[s][k] = __fmaf_rn(w0, input.x, acc[s][k]);
                    acc[s][k] = __fmaf_rn(w1, input.y, acc[s][k]);
                }
            }
        }
    }
#pragma unroll
    for (int k = 0; k < kFusedMaxTok; ++k) {
        // the 8 virtual warps' sums: warp s's total is in every lane; lane s keeps it, lanes 8..31 keep 0 (as partials[k][8..31])
        float keep = 0.0f;
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            const float total = warp_sum(acc[s][k]);
            if (lane == s) keep = total;
        }
        const float out = warp_sum(keep);
        if (k < n_tok && lane == 0) logits[(size_t) k * 512 + row] = out;
    }
    __threadfence();                 // this CTA's logits are visible device-wide before the counter moves
    __syncthreads();
    if (tid == 0) last = atomicAdd(done, 1u) == gridDim.x - 1;
    __syncthreads();
    if (!last) return;
    // the last CTA: one warp per token, the route_multi selection on the 512 logits every CTA wrote
    if (tid == 0) *done = 0u;        // the next launch starts from zero
    if (warp < n_tok) route_warp_top10(logits + (size_t) warp * 512, ids + (size_t) warp * 10, weights + (size_t) warp * 10, lane);
}
// The same contract for other (experts, top-k) shapes - Qwen3.6-35B-A3B's 256 / 8 - one warp per token, n_tok tokens
// (8 per block).  The 512 / 10 path keeps its own kernels above (the Flash-Next bits), this template is only used
// for the other shapes.
template <int NE, int K>
__launch_bounds__(256, 1)
__global__ void route_topk(const float* __restrict__ logits, int32_t* __restrict__ ids, float* __restrict__ weights,
                           int n_tok) {
    static_assert(NE % 32 == 0 && K <= 32, "experts in whole warps, at most 32 picks");
    constexpr int V = NE / 32;
    const int tk = (int) blockIdx.x * 8 + (int) threadIdx.y;
    if (tk >= n_tok) return;
    logits += (size_t) tk * NE; ids += (size_t) tk * K; weights += (size_t) tk * K;
    const int lane = threadIdx.x;
    float values[V];
#pragma unroll
    for (int i = 0; i < V; ++i) values[i] = logits[lane + i * 32];
    float maximum = -INFINITY;
#pragma unroll
    for (int i = 0; i < V; ++i) maximum = max(maximum, values[i]);
    maximum = warp_max(maximum);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < V; ++i) {
        values[i] = expf(values[i] - maximum);
        sum += values[i];
    }
    const float reciprocal = 1.0f / warp_sum(sum);
#pragma unroll
    for (int i = 0; i < V; ++i) {
        values[i] *= reciprocal;
        if (__isnanf(values[i])) values[i] = -FLT_MAX;
    }
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < K; ++rank) {
        float best = values[0];
        int expert = lane;
#pragma unroll
        for (int i = 1; i < V; ++i) {
            if (values[i] > best) { best = values[i]; expert = lane + i * 32; }
        }
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, best, mask, 32);
            const int other_id = __shfl_xor_sync(0xffffffffu, expert, mask, 32);
            if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
        }
        if ((expert & 31) == lane) {
            values[expert / 32] = -INFINITY;
            ids[rank] = expert;
            selected_sum += best;
        }
        if (rank == lane) selected = best;
    }
    selected_sum = max(warp_sum(selected_sum), 6.103515625e-5f);
    const float inverse_selected_sum = 1.0f / selected_sum;
    if (lane < K) weights[lane] = selected * inverse_selected_sum;
}
bool valid(const void* p, size_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(p);
    return p && address % 4 == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const auto ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}
}
void native_router_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_router_enabled() { return enabled.load(std::memory_order_relaxed); }
void native_router_top10(const float* logits, int32_t* ids, float* weights, void* stream) {
    if (!stream || !valid(logits, 512 * 4) || !valid(ids, 10 * 4) || !valid(weights, 10 * 4)
        || overlap(logits, 512 * 4, ids, 10 * 4) || overlap(logits, 512 * 4, weights, 10 * 4)
        || overlap(ids, 10 * 4, weights, 10 * 4))
        throw std::invalid_argument("native router requires a stream, aligned spans, and disjoint outputs");
    route<<<1, dim3(32, 8), 0, static_cast<cudaStream_t>(stream)>>>(logits, ids, weights);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void native_router_top10_multi(const float* logits, int32_t* ids, float* weights, int n_tok, void* stream) {
    if (!stream || n_tok < 1 || !valid(logits, (size_t) n_tok * 512 * 4) || !valid(ids, (size_t) n_tok * 10 * 4) ||
        !valid(weights, (size_t) n_tok * 10 * 4))
        throw std::invalid_argument("native router (multi) requires a stream and aligned [n,512]/[n,10] buffers");
    route_multi<<<(unsigned) ((n_tok + 7) / 8), dim3(32, 8), 0, static_cast<cudaStream_t>(stream)>>>(logits, ids, weights, n_tok);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
bool native_router_fused_on() {
    static const bool on = [] { const char* v = std::getenv("STRATA_ROUTER_FUSED"); return v != nullptr && v[0] == '1'; }();
    return on;
}
void native_router_fused_init() {   // before any graph capture: the CTA counter is a device allocation
    if (g_fused_done != nullptr) return;
    if (cudaMalloc((void**) &g_fused_done, sizeof(unsigned int)) != cudaSuccess) { cudaGetLastError(); g_fused_done = nullptr; return; }
    cudaMemset(g_fused_done, 0, sizeof(unsigned int));
}
bool native_router_fused_supported(int64_t n_in, int64_t n_out, int n_tok, const void* x, int64_t ldx, const void* w) {
    // n_out 512 (the 512 x 10 router), n_in even and staged in shared memory for up to kFusedMaxTok rows (under 48 KB), aligned
    return g_fused_done != nullptr && n_out == 512 && n_tok >= 1 && n_tok <= kFusedMaxTok && n_in > 0 && (n_in & 1) == 0 &&
           n_in <= 3072 && (ldx & 1) == 0 && x != nullptr && w != nullptr && (reinterpret_cast<uintptr_t>(x) & 7u) == 0 &&
           (reinterpret_cast<uintptr_t>(w) & 3u) == 0;
}
void native_router_gemv_top10_fused(const float* x, int64_t ldx, const uint16_t* w, float* logits, int32_t* ids, float* weights,
                                    int64_t n_in, int64_t n_out, int n_tok, void* stream) {
    if (!stream || g_fused_done == nullptr || n_out != 512 || n_tok < 1 || n_tok > kFusedMaxTok)
        throw std::invalid_argument("native router (fused): needs a stream, init, 512 outputs and 1..4 rows");
    const size_t shared = (size_t) n_tok * (size_t) n_in * sizeof(float);   // [n_tok][n_in / 2] float2
    router_gemv_top10_fused_k<<<(unsigned) (512 / kFusedRows), kFusedThreads, shared, static_cast<cudaStream_t>(stream)>>>(
        x, ldx, w, logits, ids, weights, (int) n_in, n_tok, g_fused_done);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
bool native_router_topk_supported(int n_expert, int k) { return (n_expert == 512 && k == 10) || (n_expert == 256 && k == 8); }
void native_router_topk(const float* logits, int32_t* ids, float* weights, int n_expert, int k, int n_tok, void* stream) {
    if (n_expert == 512 && k == 10) {   // Flash-Next: the pinned kernels above, unchanged
        if (n_tok == 1) native_router_top10(logits, ids, weights, stream);
        else native_router_top10_multi(logits, ids, weights, n_tok, stream);
        return;
    }
    if (!stream || n_tok < 1 || !native_router_topk_supported(n_expert, k) ||
        !valid(logits, (size_t) n_tok * n_expert * 4) || !valid(ids, (size_t) n_tok * k * 4) ||
        !valid(weights, (size_t) n_tok * k * 4))
        throw std::invalid_argument("native_router_topk: a stream, a supported (experts, k) and aligned buffers");
    route_topk<256, 8><<<(unsigned) ((n_tok + 7) / 8), dim3(32, 8), 0, static_cast<cudaStream_t>(stream)>>>(
        logits, ids, weights, n_tok);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

// STRATA_ROUTE_RESIDENT (EXPERIMENTAL, opt-in, changes the output): residency-biased routing for 512 x 10.
// After the top-10: for the ranks lo..hi a non-resident expert is replaced by the best resident expert not already
// picked when its logit is within `margin` of the replaced one; the weights are then the softmax of the selected
// logits (the router's renormalisation).  Tokens without a swap keep the router's exact bits.
// stats (device, 4 x uint64): [0] tail entries seen, [1] swaps, [2] non-resident entries before, [3] after.
static unsigned long long* g_detail = nullptr;   // the decode rows' detail counters (route_detail_alloc), see below
// The router's device scratch (the look-ahead's logits and weights, its skip counter, the ring flags, and the want / rows / repl /
// detail counters) is allocated once, on the GPU of the first verifier that asks.  A multi-GPU layer split's other stages run on
// other GPUs: there these buffers are another device's memory (an illegal address without peer access, a shared scratch with it),
// so their accessors answer null on any other GPU and the stage runs without them (as with the features off).
static int g_scratch_dev = -1;
static void note_scratch_dev() {
    if (g_scratch_dev < 0 && cudaGetDevice(&g_scratch_dev) != cudaSuccess) { (void) cudaGetLastError(); g_scratch_dev = -1; }
}
static bool scratch_here() {
    if (g_scratch_dev < 0) return true;
    int dev = -1;
    if (cudaGetDevice(&dev) != cudaSuccess) { (void) cudaGetLastError(); return false; }
    return dev == g_scratch_dev;
}
//
// One warp per token (upstream 0.1.45's form).  This was one THREAD per token scanning all 512 experts for every rank (4 ranks x 512 x 12 scalar
// ops on up to 6 threads, on the critical stream, once per layer): upstream measured the swap at 2.2-2.75 ms of a 19-22 ms window and 0.33 ms in
// the warp form.  The ranks are still visited in order (a swap changes the set already picked), but the scan is coalesced (lane j reads f = j,
// j + 32, ...) and reduced within the warp.  The bits are unchanged: a strictly greater logit wins, the smallest expert index breaks ties, and the
// softmax is the same expression.  Our options ride on lane 0 (the counters, the wanted-expert count, the per-row record, the CCE bias) or are
// reduced in the warp (the stand-in's place in the router's ranking).
// The fragile hold bitmask (route_fragile_hold; SPEC-recheck-gated step A): bit r set = window row r keeps the router's picks; 0xFFFFFFFF
// holds every row (the control window).  Rows >= 32 are held only by the all-ones mask.  Read at each launch (uniform across the warp).
__device__ __forceinline__ bool fragile_held(const uint32_t* fctrl, int row) {
    if (fctrl == nullptr) return false;
    const uint32_t b = *fctrl;
    return row < 32 ? ((b >> row) & 1u) != 0u : b == 0xFFFFFFFFu;
}
// one warp of token t (SPEC-router-fuse step 2: a device function, shared by route_resident_k and the post-routing kernel).
// margin_out / cap / cap_stats (SPEC-recheck-gated C, F; nullptr = off): the early margin (layers 0-3), the row's shift cap and its refusal counters.
__device__ __forceinline__ void route_resident_warp(const float* __restrict__ logits, int32_t* __restrict__ ids, float* __restrict__ weights,
                                 const int32_t* __restrict__ res, int t, const RouteCfg* cfg,
                                 unsigned long long* __restrict__ stats, float* __restrict__ want,
                                 unsigned long long* __restrict__ detail, int32_t* __restrict__ rows, float* __restrict__ repl,
                                 const uint8_t* __restrict__ ring, const uint8_t* __restrict__ cheap, float cheap_margin,
                                 int cheap_rank0, int cheap_anytier, unsigned long long* __restrict__ bstats,
                                 float* __restrict__ row_shift, const float* __restrict__ pair_c,
                                 int32_t* __restrict__ lrn_rec, int layer, const float* __restrict__ pair_pe,
                                 MaskEvs* __restrict__ mev, int row_base, const uint32_t* __restrict__ fctrl,
                                 float* __restrict__ margin_out = nullptr, const float* __restrict__ cap = nullptr,
                                 unsigned long long* __restrict__ cap_stats = nullptr) {
    const int lane = (int) threadIdx.x;
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
    const float budget = ((const volatile RouteCfg*) cfg)->budget;
    const float budget_tail = ((const volatile RouteCfg*) cfg)->budget_tail;
    const float* l = logits + (size_t) t * 512;
    int32_t* id = ids + (size_t) t * 10;
    // SPEC-recheck-gated (C): the early margin, the router's top-1 minus top-2 raw logit over all 512 experts, before any stand-in, min'd into
    // margin_out[row] by the routers of layers 0-3 (the host sets the array to a large value before each window; predicated, uniform per warp)
    if (margin_out != nullptr && layer < 4) {
        float m1 = -INFINITY, m2 = -INFINITY;
        for (int f = lane; f < 512; f += 32) {
            const float v = l[f];
            if (v > m1) { m2 = m1; m1 = v; }
            else if (v > m2) m2 = v;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float o1 = __shfl_xor_sync(0xffffffffu, m1, off);
            const float o2 = __shfl_xor_sync(0xffffffffu, m2, off);
            if (o1 > m1) { m2 = fmaxf(m1, o2); m1 = o1; }
            else m2 = fmaxf(m2, o1);
        }
        if (lane == 0) margin_out[row_base + t] = fminf(margin_out[row_base + t], m1 - m2);
    }
    if (lane == 0) {
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
                    // (best < 0: the first unused expert is taken even when every score is NaN or -inf - else best stayed -1
                    // and l[-1] / res[-1] were read)
                    if (!used && (best < 0 || s > bs)) { bs = s; best = f; }
                }
                pick[r] = best;
            }
            float m = -INFINITY;
            for (int r = 0; r < 10; ++r) m = fmaxf(m, l[pick[r]]);
            float ex[10], sum = 0.f;
            for (int r = 0; r < 10; ++r) { ex[r] = expf(l[pick[r]] - m); sum += ex[r]; }
            for (int r = 0; r < 10; ++r) { id[r] = pick[r]; weights[(size_t) t * 10 + r] = ex[r] / sum; }
        }
    }
    __syncwarp();   // lane 0's writes to the ids and weights are what the other lanes read next
    int32_t my[10];
    for (int r = 0; r < 10; ++r) my[r] = id[r];
    // STRATA_STANDIN_LEARN: the router's picks (orig) and the stand-in each swapped rank took (sub_s, -1: none)
    int32_t orig[10], sub_s[10];
    for (int r = 0; r < 10; ++r) { orig[r] = my[r]; sub_s[r] = -1; }
    // STRATA_RING: an expert whose look-ahead ring copy has landed counts as resident (the host plan finds it in the ring)
    auto have = [&](int f) { return res[f] >= 0 || (ring != nullptr && ring[f] != 0); };
    int swaps = 0, tail = 0, before = 0;
    if (lane == 0)
        for (int r = 0; r < 10; ++r) before += !have(my[r]);
    // STRATA_ROUTE_RAM_FLOOR (cheap != nullptr): the loop starts at rank 0, but only ranks 0-1 are new here.  A non-resident rank 0 or 1 pick
    // whose expert is in the RAM tier stays exact (the CPU computes it from RAM); any other one takes the stand-in like a rank 2-9 pick.
    // Measured 20 Oct (210 MMLU-Pro questions): 26% of decode time was the CPU path waiting on SSD fetches for such picks.  Ranks 2 and up
    // below the floor are skipped as before.  `first` is the floor's first rank: the fractional edge rank lo - 1, else lo.
    const int first = (edge >= 0.f && lo > 0) ? lo - 1 : lo;
    int cheap_subs = 0;   // lane 0 only: rank 0-1 stand-ins made because the expert was not in the RAM tier
    // STRATA_ROUTE_BUDGET (budget > 0): the loop starts at rank 0, as with `cheap`.  A rank below lo (`brank`) takes the budget test below
    // instead of the margin / edge, and skips none of the RAM floor's rules (the budget wins for those ranks); ranks lo and up are as before.
    const bool bud = budget > 0.f;
    // STRATA_FRAGILE_CTRL_EVERY (route_fragile_control): a control window moves no pick, so every pick stays at its router rank and row_shift
    // stays 0.  *fctrl is the same for every lane (uniform), so the loop is skipped by the whole warp.
    const bool fctl = fragile_held(fctrl, row_base + t);
    // SPEC-recheck-gated step F: this row's shift cap (+inf = none; rows past 32 are never capped) and the row's shift so far (a uniform value:
    // read once, then followed by each applied move).  A refused move is counted in cap_stats[0] (moves), the row once in cap_stats[1].
    const float cap_row = (cap != nullptr && row_base + t < 32) ? cap[row_base + t] : INFINITY;
    float shift_acc = row_shift != nullptr ? row_shift[t] : 0.0f;
    bool cap_hit = false;
    for (int r = (cheap != nullptr || bud) ? 0 : first; r <= hi && r < 10 && !fctl; ++r) {
        const int e = my[r];
        if (have(e)) continue;                          // already resident: uniform across the warp
        const bool brank = bud && r < lo;               // STRATA_ROUTE_BUDGET: this rank takes the budget test (uniform across the warp)
        const bool trank = budget_tail > 0.f && r >= lo;   // STRATA_ROUTE_BUDGET_TAIL: this rank (at or above lo) takes the tail test (uniform across the warp)
        if (!brank) {
            if (r >= 2 && r < first) continue;          // below the floor and not rank 0-1: exact, as before (never reached without `cheap`)
            // RAM-tier expert: exact, the CPU computes it from RAM (e is the same for the warp); STRATA_ROUTE_RAM_FLOOR_ANYTIER=1 lets a close
            // stand-in (within the margin) replace it too, so the CPU skips it as well
            if (cheap != nullptr && r < 2 && cheap[e] != 0 && cheap_anytier == 0) continue;
            // the safety margin (SPEC E1, 10 Oct: rank 1 replaced only by a stand-in within 1 logit stayed sound, 49/50 maths, 29/30 code;
            // an unlimited margin on ranks 0-1 collapsed the answers, T22/T27/9v): rank 0 stays exact unless STRATA_ROUTE_RAM_FLOOR_RANK0=1
            if (cheap != nullptr && r == 0 && cheap_rank0 == 0) continue;
        }
        if (wmax > 0.f && weights[(size_t) t * 10 + r] >= wmax) continue;   // a heavy pick stays exact: fetched / computed on the CPU
        if (lane == 0) ++tail;
        float bv = -INFINITY;
        int bi = -1;
        for (int f = lane; f < 512; f += 32) {
            if (!have(f)) continue;
            bool used = false;
            for (int q = 0; q < 10; ++q) used |= my[q] == f;
            if (used) continue;
            if (l[f] > bv) { bv = l[f]; bi = f; }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_down_sync(0xffffffffu, bv, off);
            const int oi = __shfl_down_sync(0xffffffffu, bi, off);
            if (ov > bv || (ov == bv && oi >= 0 && (bi < 0 || oi < bi))) { bv = ov; bi = oi; }
        }
        bv = __shfl_sync(0xffffffffu, bv, 0);           // every lane decides the same way from here
        bi = __shfl_sync(0xffffffffu, bi, 0);
        // item 21: the stand-in's place in the router's own ranking of all 512 experts (10 = the router's next choice after its ten
        // picks); a stand-in further down than `maxrank` is refused and the exact expert is fetched instead
        if (bi >= 0 && maxrank > 0) {
            int place = 0;
            for (int f = lane; f < 512; f += 32) place += l[f] > bv;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) place += __shfl_xor_sync(0xffffffffu, place, off);
            if (place >= maxrank) bi = -1;
        }
        // the edge rank (lo - 1, a fractional floor such as 1.5) is replaced only by a close substitute (within `edge`); the other
        // ranks below it (STRATA_ROUTE_RAM_FLOOR's ranks 0-1) take the margin, as the ranks 2-9 do
        // STRATA_ROUTE_RAM_FLOOR's ranks 0-1 take its own safety margin (STRATA_ROUTE_RAM_FLOOR_MARGIN, default 1 logit): no stand-in
        // that close = the exact expert, computed on the CPU after its fetch as before
        // STRATA_ROUTE_BUDGET: a rank below lo is replaced when w * gap <= budget, w = the pick's routing weight (the router's renormalised
        // softmax over the ten, read here before any swap changes the weights), gap = the pick's logit minus the stand-in's (>= 0; clamped)
        const float gap = l[e] - bv;
        // STRATA_ROUTE_BUDGET_TAIL: a rank at or above lo is replaced when w * max(gap, 0) <= budget_tail (the same rule, its own bound); the
        // margin does not apply to it.  A tail pick that the bound refuses stays exact (counted below when a stand-in was there)
        bool take = brank ? weights[(size_t) t * 10 + r] * fmaxf(gap, 0.f) <= budget
                        : trank ? weights[(size_t) t * 10 + r] * fmaxf(gap, 0.f) <= budget_tail
                        : gap <= ((cheap != nullptr && r < 2) ? cheap_margin
                                  : (edge >= 0.f && lo > 0 && r == lo - 1) ? edge : margin);
        if (bi >= 0 && trank && bstats != nullptr && lane == 0) atomicAdd(bstats + (take ? 3 : 4), 1ull);   // bstats [3] tail replaced, [4] tail kept exact by the bound
        // SPEC-recheck-gated step F: a move that would take the row's accumulated shift past its cap stays the router's pick (refused).
        // cap_row and shift_acc are the same in every lane (warp-reduced values; shift_acc starts at the row's shift and follows each move).
        if (bi >= 0 && take && cap_row < INFINITY) {
            const float pf_c = pair_pe != nullptr ? pair_pe[(size_t) orig[r] * 512 + bi] : 1.0f;
            if (shift_acc + weights[(size_t) t * 10 + r] * fmaxf(gap, 0.f) * pf_c > cap_row) {
                take = false;
                if (lane == 0) {
                    cap_hit = true;
                    if (cap_stats != nullptr) atomicAdd(cap_stats, 1ull);
                }
            }
        }
        if (bi >= 0 && take) {
            if (detail != nullptr) {
                // what was replaced and by what (per replacement): the weight of the replaced pick, the substitute's place in
                // the router's own ranking of all 512 experts, and the score gap to it
                int gr = 0;
                for (int f = lane; f < 512; f += 32) gr += l[f] > bv;
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) gr += __shfl_xor_sync(0xffffffffu, gr, off);
                if (lane == 0) {
                    const float w = weights[(size_t) t * 10 + r];
                    const int wb = w < 0.01f ? 0 : w < 0.02f ? 1 : w < 0.05f ? 2 : w < 0.10f ? 3 : w < 0.20f ? 4 : 5;
                    const int rb = gr < 20 ? 0 : gr < 40 ? 1 : gr < 80 ? 2 : gr < 160 ? 3 : 4;
                    const float gap = l[e] - bv;
                    const int gb = gap < 0.5f ? 0 : gap < 1.f ? 1 : gap < 2.f ? 2 : gap < 4.f ? 3 : gap < 8.f ? 4 : 5;
                    atomicAdd(detail + 16 + wb, 1ull);
                    atomicAdd(detail + 22 + rb, 1ull);
                    atomicAdd(detail + 27 + gb, 1ull);
                }
            }
            // SWAP2: this expert was wanted and a stand-in served it; counted for a quality swap only when the stand-in is poor (STRATA_SWAP2_GAP logits)
            if (lane == 0 && repl != nullptr && l[e] - bv >= repl_gap) atomicAdd(repl + e, 1.0f);
            if (lane == 0 && brank && bstats != nullptr && r < 3) atomicAdd(bstats + r, 1ull);   // budget replacements per rank 0-2
            // STRATA_FRAGILE_RECHECK: how far this row's picks moved (any rule): the replaced pick's router weight times its score gap to
            // the stand-in (the quantity the budget test uses), summed over the layers and the picks of the row (read by the host)
            // STRATA_STANDIN_BALANCE (pair_pe, opt-in): a ready pair's risk is scaled by its relative error (clip 0.25-4; 1 until ready)
            const float pf = pair_pe != nullptr ? pair_pe[(size_t) orig[r] * 512 + bi] : 1.0f;
            if (lane == 0 && row_shift != nullptr) atomicAdd(row_shift + t, weights[(size_t) t * 10 + r] * fmaxf(gap, 0.f) * pf);
            shift_acc += weights[(size_t) t * 10 + r] * fmaxf(gap, 0.f) * pf;   // (the cap's running shift, every lane)
            // STRATA_MASK=learn: the replaced pick (orig[r]) and its risk, in this row's list
            if (lane == 0 && mev != nullptr && row_base + t < kMaskEvRows) {
                const unsigned q = mev->n[row_base + t]++;
                if (q < kMaskEvCap) mev->e[row_base + t][q] = MaskEv{layer, orig[r], weights[(size_t) t * 10 + r] * fmaxf(gap, 0.f), (int32_t) ((r & 15) << 1) /* mask_ev_pack(0, r) */};
            }
            my[r] = bi;
            sub_s[r] = bi;
            ++swaps;
            // STRATA_STANDIN_LEARN: this row's first stand-in (layer, rank, true expert, stand-in); the buffer is reset each decode window
            if (lane == 0 && lrn_rec != nullptr && lrn_rec[(size_t) t * 4] < 0) {
                lrn_rec[(size_t) t * 4] = layer;
                lrn_rec[(size_t) t * 4 + 1] = r;
                lrn_rec[(size_t) t * 4 + 2] = orig[r];
                lrn_rec[(size_t) t * 4 + 3] = bi;
            }
            if (lane == 0 && cheap != nullptr && r < 2) ++cheap_subs;
        }
    }
    int after = 0;
    if (lane == 0)
        for (int r = 0; r < 10; ++r) after += !have(my[r]);
    if (lane == 0) {
        if (swaps) {
            float m = -INFINITY;
            for (int r = 0; r < 10; ++r) m = fmaxf(m, l[my[r]]);
            float ex[10], sum = 0.0f;
            for (int r = 0; r < 10; ++r) { ex[r] = expf(l[my[r]] - m); sum += ex[r]; }
            for (int r = 0; r < 10; ++r) weights[(size_t) t * 10 + r] = ex[r] / sum;
            // STRATA_STANDIN_LEARN: a stand-in's weight is scaled by its pair's learned correction (1 until the pair is ready)
            if (pair_c != nullptr)
                for (int r = 0; r < 10; ++r)
                    if (sub_s[r] >= 0) weights[(size_t) t * 10 + r] *= pair_c[(size_t) orig[r] * 512 + sub_s[r]];
        }
        for (int r = 0; r < 10; ++r) id[r] = my[r];
        if (stats) {
            atomicAdd(stats + 0, (unsigned long long) tail);
            atomicAdd(stats + 1, (unsigned long long) swaps);
            atomicAdd(stats + 2, (unsigned long long) before);
            atomicAdd(stats + 3, (unsigned long long) after);
            // STRATA_ROUTE_RAM_FLOOR only (the caller's block then has 5+ entries): rank 0-1 stand-ins for an expert not in the RAM tier
            if (cheap != nullptr) atomicAdd(stats + 4, (unsigned long long) cheap_subs);
        }
    }
    if (lane == 0 && cap_hit && cap_stats != nullptr) atomicAdd(cap_stats + 1, 1ull);   // step F: this row had a refused move in this launch
}
__global__ void route_resident_k(const float* __restrict__ logits, int32_t* __restrict__ ids, float* __restrict__ weights,
                                 const int32_t* __restrict__ res, int n_tok, const RouteCfg* cfg,
                                 unsigned long long* __restrict__ stats, float* __restrict__ want,
                                 unsigned long long* __restrict__ detail, int32_t* __restrict__ rows, float* __restrict__ repl,
                                 const uint8_t* __restrict__ ring, const uint8_t* __restrict__ cheap, float cheap_margin,
                                 int cheap_rank0, int cheap_anytier, unsigned long long* __restrict__ bstats,
                                 float* __restrict__ row_shift, const float* __restrict__ pair_c,
                                 int32_t* __restrict__ lrn_rec, int layer, const float* __restrict__ pair_pe,
                                 MaskEvs* __restrict__ mev, int row_base, const uint32_t* __restrict__ fctrl,
                                 float* __restrict__ margin_out, const float* __restrict__ cap,
                                 unsigned long long* __restrict__ cap_stats) {
    const int t = blockIdx.x;   // one warp (32 threads) per token
    if (t >= n_tok) return;
    route_resident_warp(logits, ids, weights, res, t, cfg, stats, want, detail, rows, repl, ring, cheap, cheap_margin, cheap_rank0,
                        cheap_anytier, bstats, row_shift, pair_c, lrn_rec, layer, pair_pe, mev, row_base, fctrl, margin_out, cap, cap_stats);
}
// ---- the floor: one small struct in host-mapped memory, read by the kernel at every launch.  The host sets it once (route_cfg_init)
// and could move it between windows (route_cfg_set_floor; nothing does today).  A higher floor replaces less.  SPEC 17's per-part
// floors (thinking / answer / code, a fence tracker) lived here and were removed after the benchmark showed no gain (SPEC 18.1).
static RouteCfg* g_cfg_host = nullptr;
static RouteCfg* g_cfg_dev = nullptr;
void route_cfg_init(int lo, int hi, float margin, float edge, float bias, int maxrank, float want_lo, float repl_gap, float wmax, float budget, float budget_tail) {
    if (g_cfg_host == nullptr) {
        if (cudaHostAlloc((void**) &g_cfg_host, sizeof(RouteCfg), cudaHostAllocMapped) != cudaSuccess) { g_cfg_host = nullptr; return; }
        if (cudaHostGetDevicePointer((void**) &g_cfg_dev, g_cfg_host, 0) != cudaSuccess) { g_cfg_dev = nullptr; return; }
    }
    g_cfg_host->lo = lo;
    g_cfg_host->hi = hi;
    g_cfg_host->margin = margin;
    g_cfg_host->edge = edge;
    g_cfg_host->bias = bias;
    g_cfg_host->maxrank = maxrank;
    g_cfg_host->want_lo = want_lo;
    g_cfg_host->repl_gap = repl_gap;
    g_cfg_host->wmax = wmax;
    g_cfg_host->budget = budget;
    g_cfg_host->budget_tail = budget_tail;
    __sync_synchronize();
}
void route_cfg_set_floor(int lo) {
    if (g_cfg_host == nullptr) return;
    g_cfg_host->lo = lo;
    __sync_synchronize();
}
// STRATA_ROUTE_BUDGET_DYN: the verifier moves the budget between windows (after a window's sync, before the next launch).
void route_cfg_set_budget(float budget) {
    if (g_cfg_host == nullptr) return;
    g_cfg_host->budget = budget;
    __sync_synchronize();
}
// STRATA_ROUTE_BUDGET_AUTO: both budgets together, between windows (as route_cfg_set_budget)
void route_cfg_set_budgets(float budget, float budget_tail) {
    if (g_cfg_host == nullptr) return;
    g_cfg_host->budget = budget;
    g_cfg_host->budget_tail = budget_tail;
    __sync_synchronize();
}
const RouteCfg* route_cfg_device() { return g_cfg_dev; }
// STRATA_LA_TAIL_READ: the open rule as the host holds it now (the same values the kernels read): any resident stand-in serves a rank at or above lo
bool route_cfg_tail_open() { return g_cfg_host != nullptr && g_cfg_host->margin >= 1e30f && g_cfg_host->budget_tail <= 0.f; }
namespace {
// STRATA_MTP_TALLY (measurement only): per (rank, expert) the router weight the drafter gave it; row 10 the rank-2 picks within one
// logit of rank 1; [11*512] the routed tokens.  One warp per token.
__global__ void route_tally_k(const float* __restrict__ logits, const int32_t* __restrict__ ids, const float* __restrict__ w,
                              float* __restrict__ acc, int32_t* __restrict__ seq_ids, float* __restrict__ seq_w,
                              unsigned int* __restrict__ seq_n, unsigned int seq_cap) {
    const int t = (int) blockIdx.x, lane = (int) threadIdx.x;
    unsigned int slot = 0;
    if (lane == 0 && seq_n != nullptr) slot = atomicAdd(seq_n, 1u);
    slot = __shfl_sync(0xffffffffu, slot, 0);
    if (lane >= 10) return;
    const int e = ids[t * 10 + lane];
    const float v = w[t * 10 + lane];
    if (seq_n != nullptr && slot < seq_cap) { seq_ids[(size_t) slot * 10 + lane] = e; seq_w[(size_t) slot * 10 + lane] = v; }
    atomicAdd(&acc[lane * 512 + e], v);
    if (lane == 2 && logits[(size_t) t * 512 + e] >= logits[(size_t) t * 512 + ids[t * 10 + 1]] - 1.0f) atomicAdd(&acc[10 * 512 + e], v);
    if (lane == 0) atomicAdd(&acc[11 * 512], 1.0f);
}
}
void native_route_tally(const float* logits, const int32_t* ids, const float* weights, int n_tok, float* acc, void* stream,
                        int32_t* seq_ids, float* seq_w, unsigned int* seq_n, unsigned int seq_cap) {
    route_tally_k<<<n_tok, 32, 0, static_cast<cudaStream_t>(stream)>>>(logits, ids, weights, acc, seq_ids, seq_w, seq_n, seq_cap);
}
namespace {
// STRATA_MTP_SLOTS: one warp per token (see mtp_slot_remap).  Every lane runs the same decisions (the argmax is a butterfly, so all
// lanes hold the winner); lane 0 does the atomics and the final writes, after all ten picks were read into registers.
__global__ void mtp_slot_remap_k(const float* __restrict__ logits, int32_t* __restrict__ ids, const int32_t* __restrict__ d_slot,
                                 uint8_t* __restrict__ ref, int32_t* __restrict__ want, unsigned int* __restrict__ want_n,
                                 unsigned int want_cap, unsigned long long* __restrict__ stats) {
    const int t = (int) blockIdx.x, lane = (int) threadIdx.x;
    int orig[10], chosen[10], fin[10];
    for (int r = 0; r < 10; ++r) { orig[r] = ids[t * 10 + r]; chosen[r] = -1; }
    for (int r = 0; r < 10; ++r) {
        const int e = orig[r];
        const int s = (e >= 0 && e < 512) ? d_slot[e] : -1;
        if (lane == 0) atomicAdd(&stats[r], 1ull);
        if (s >= 0) {
            fin[r] = s;
            if (lane == 0) { ref[s] = 1; atomicAdd(&stats[10 + r], 1ull); }
            continue;
        }
        if (r < 2 && lane == 0) {
            const unsigned int i = atomicAdd(want_n, 1u);
            if (i < want_cap) want[i] = e;
        }
        float bv = -FLT_MAX;
        int bi = -1;
        for (int x = lane; x < 512; x += 32) {
            if (d_slot[x] < 0) continue;
            bool used = false;
            for (int q = 0; q < 10; ++q) if (orig[q] == x || chosen[q] == x) { used = true; break; }
            if (used) continue;
            const float v = logits[(size_t) t * 512 + x];
            if (bi < 0 || v > bv) { bv = v; bi = x; }   // this lane's x ascend: the first (lowest id) of equal values stays
        }
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
            if (oi >= 0 && (bi < 0 || ov > bv || (ov == bv && oi < bi))) { bv = ov; bi = oi; }
        }
        const int best = __shfl_sync(0xffffffffu, bi, 0);
        chosen[r] = best;
        fin[r] = best >= 0 ? d_slot[best] : 0;
    }
    if (lane == 0) for (int r = 0; r < 10; ++r) ids[t * 10 + r] = fin[r];
}
}  // namespace
void mtp_slot_remap(const float* logits, int32_t* ids, const int32_t* d_slot, int n_tok, uint8_t* ref, int32_t* want, unsigned int* want_n,
                    unsigned int want_cap, unsigned long long* stats, void* stream) {
    mtp_slot_remap_k<<<n_tok, 32, 0, static_cast<cudaStream_t>(stream)>>>(logits, ids, d_slot, ref, want, want_n, want_cap, stats);
}
// STRATA_MTP_ZC (opt-in, with STRATA_MTP_SLOTS): one thread per (token, rank).  Marks the slot of a resident pick (ref = 1, a hit) and
// writes the rank 0-1 misses to `want`; the ids are NOT changed (a miss is computed from its pinned RAM copy, see moe_group_ptr).
__global__ void mtp_zc_mark_k(const int32_t* __restrict__ ids, const int32_t* __restrict__ d_slot, int n_thr,
                              uint8_t* __restrict__ ref, int32_t* __restrict__ want, unsigned int* __restrict__ want_n,
                              unsigned int want_cap, unsigned long long* __restrict__ stats) {
    const int idx = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n_thr) return;
    const int r = idx % 10;
    const int e = ids[idx];
    atomicAdd(&stats[r], 1ull);
    const int s = (e >= 0 && e < 512) ? d_slot[e] : -1;
    if (s >= 0) {
        ref[s] = 1;
        atomicAdd(&stats[10 + r], 1ull);
    } else if (r < 2) {
        const unsigned int i = atomicAdd(want_n, 1u);
        if (i < want_cap) want[i] = e;
    }
}
void mtp_zc_mark(const int32_t* ids, const int32_t* d_slot, int n_tok, uint8_t* ref, int32_t* want, unsigned int* want_n,
                 unsigned int want_cap, unsigned long long* stats, void* stream) {
    const int n_thr = n_tok * 10;
    const int block = 128;
    mtp_zc_mark_k<<<(n_thr + block - 1) / block, block, 0, static_cast<cudaStream_t>(stream)>>>(ids, d_slot, n_thr, ref, want, want_n,
                                                                                              want_cap, stats);
}
// STRATA_FRAGILE_CTRL_EVERY: the fragile control flag, a host-mapped int (1 = this window is a control: no stand-in or mask move), written
// between windows by route_fragile_control.  Its device alias is read by route_resident_k and route_mask_k at each launch, as the mask's
// control flag is.  Allocated on the first request for it; nullptr until then (no control windows).
static uint32_t* g_frag_ctrl_host = nullptr;   // the hold bitmask (route_fragile_hold)
static uint32_t* g_frag_ctrl_dev = nullptr;
// SPEC-recheck-gated step F: the per-row shift cap, host-mapped float [32] (+inf = none), written between windows; and the refusal counters
// [2] (device: [0] refused moves, [1] row launches with a refusal), read and reset by route_shift_cap_read.  Allocated before any capture.
static float* g_cap_host = nullptr;
static float* g_cap_dev = nullptr;
static unsigned long long* g_cap_stats = nullptr;
// The RAM floor's environment (read once): shared by native_route_resident and native_route_post.
static void resident_env(float& cheap_margin, int& cheap_rank0, int& cheap_anytier) {
    // STRATA_ROUTE_RAM_FLOOR's safety: the stand-in for a rank 0-1 pick must be within STRATA_ROUTE_RAM_FLOOR_MARGIN logits (default 1,
    // the margin SPEC E1 found sound for rank 1), and rank 0 is replaced only with STRATA_ROUTE_RAM_FLOOR_RANK0=1 (off: it leaked
    // <|im_start|>user into an answer at 94K tokens, 20 Oct)
    static const float cm = [] { const char* v = std::getenv("STRATA_ROUTE_RAM_FLOOR_MARGIN"); return v ? (float) std::atof(v) : 1.0f; }();
    static const int r0 = [] { const char* v = std::getenv("STRATA_ROUTE_RAM_FLOOR_RANK0"); return v != nullptr && std::atoi(v) != 0 ? 1 : 0; }();
    // STRATA_ROUTE_RAM_FLOOR_ANYTIER (default ON since 20 Oct, =0 turns it off): a close stand-in replaces a rank 0-1 pick in the
    // RAM tier too, not only one that would come from the SSD
    static const int at = [] { const char* v = std::getenv("STRATA_ROUTE_RAM_FLOOR_ANYTIER"); return v == nullptr || std::atoi(v) != 0 ? 1 : 0; }();
    cheap_margin = cm;
    cheap_rank0 = r0;
    cheap_anytier = at;
}
void native_route_resident(const float* logits, int32_t* ids, float* weights, const int32_t* res_layer, int n_tok,
                           unsigned long long* stats, void* stream, float* want, int32_t* rows, float* repl,
                           const uint8_t* ring, const uint8_t* cheap, unsigned long long* bstats, float* row_shift,
                           const float* pair_c, int32_t* lrn_rec, int layer, const float* pair_pe, MaskEvs* mev, int row_base,
                           float* margin_out) {
    if (!stream || n_tok < 1 || !res_layer || g_cfg_dev == nullptr) throw std::invalid_argument("native_route_resident: bad arguments");
    float cheap_margin;
    int cheap_rank0, cheap_anytier;
    resident_env(cheap_margin, cheap_rank0, cheap_anytier);
    route_resident_k<<<(unsigned) n_tok, 32, 0, static_cast<cudaStream_t>(stream)>>>(   // one warp (32 threads) per token
        logits, ids, weights, res_layer, n_tok, g_cfg_dev, stats, want, n_tok <= 8 ? g_detail : nullptr, n_tok <= 8 ? rows : nullptr,
        n_tok <= 8 ? repl : nullptr, ring, cheap, cheap_margin, cheap_rank0, cheap_anytier, bstats, row_shift,
        pair_c, lrn_rec, layer, pair_pe, mev, row_base, g_frag_ctrl_dev, margin_out, g_cap_dev, g_cap_stats);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
// STRATA_MASK=<share> (opt-in, changes the output): the least-used experts of each layer (the tail of the expert profile) are masked.
// After the router's top-10, a masked pick of rank >= lo moves to the best unmasked expert the token did not pick; rank lo-1 moves only
// when that expert is within `edge` logits (edge > 0); ranks below lo-1 stay.  The rule and the settings are route-resident's.  A move is
// a substitution: its w x gap (w the router's weight on the pick, gap the pick's logit minus the substitute's, >= 0) joins the row's
// row_shift, so the flip re-check sees the close calls the mask makes.  The flip re-check's exact re-run is launched without the mask by
// the verifier.  One warp per token; the weights are the softmax of the new picks when any moved (as route-resident does).
static uint8_t* g_mask_dev = nullptr;                // device [layers][n_expert], 1 = masked
static unsigned long long* g_mask_stats = nullptr;  // device [2]: masked picks seen, picks moved
static int* g_mask_ctrl_host = nullptr;             // host-mapped int: 1 = a control window (no mask moves); g_mask_ctrl_dev is its device alias (STRATA_MASK_RESIDENT)
static int* g_mask_ctrl_dev = nullptr;
static uint32_t* g_mask_cost = nullptr;             // device [layers][512]: the router's picks that are not cheap (the SSD demand; STRATA_MASK_RESIDENT)
static size_t g_mask_cost_n = 0;
static int g_mask_dev_id = -1;
static int64_t g_mask_layers = 0, g_mask_ne = 0;
static int g_mask_lo = 6;
static int g_mask_cost_lo = 6;                      // STRATA_MASK_RESIDENT: the SSD demand counts ranks below this (the stand-ins' first rank)
static std::vector<uint8_t> g_mask_host;            // the same mask on the host, [layers][n_expert] (the RAM tier reads it; empty: unset)
static float g_mask_edge = -1.0f;
namespace {
// STRATA_MASK_RESIDENT: an expert is cheap when the router reads it without an SSD fetch: on the card (res), in the look-ahead ring, or in the
// RAM tier (route_resident_k's `have` and the RAM floor's table, as one rule).  `res` is the layer's slice, ring and cheap may be nullptr.
__device__ __forceinline__ bool mask_cheap(int f, const int32_t* res, const uint8_t* ring, const uint8_t* cheap) {
    return res[f] >= 0 || (ring != nullptr && ring[f] != 0) || (cheap != nullptr && cheap[f] != 0);
}
// The mask's rule for one token t of a warp (lane = threadIdx.x; one warp per token).  SPEC-router-fuse step 1/2: a device function, so the
// fused selection kernel and the post-routing kernel run the same code as the one-kernel launch below.
__device__ __forceinline__ void route_mask_warp(float* __restrict__ logits, int32_t* __restrict__ ids, float* __restrict__ weights,
                                                const uint8_t* __restrict__ mask, int t, int lo, float edge,
                                                float* __restrict__ row_shift, unsigned long long* __restrict__ stats, MaskEvs* __restrict__ mev,
                                                int layer, int row_base, bool reset, const int32_t* __restrict__ res,
                                                const uint8_t* __restrict__ ring, const uint8_t* __restrict__ cheap,
                                                uint32_t* __restrict__ cost, const int* __restrict__ ctrl, int cost_lo,
                                                const uint32_t* __restrict__ fctrl, const float* __restrict__ cap = nullptr,
                                                unsigned long long* __restrict__ cap_stats = nullptr) {
    const int lane = (int) threadIdx.x;
    const int row = row_base + t;
    // SPEC-recheck-gated step F: the row's shift cap and its running shift (as route_resident_warp: uniform values)
    const float cap_row = (cap != nullptr && row < 32) ? cap[row] : INFINITY;
    float shift_acc = row_shift != nullptr ? row_shift[t] : 0.0f;
    bool cap_hit = false;
    if (mev != nullptr && reset && lane == 0 && row < kMaskEvRows) mev->n[row] = 0u;   // (this row's list starts empty)
    const float* l = logits + (size_t) t * 512;
    int32_t* id = ids + (size_t) t * 10;
    int my[10];
    for (int r = 0; r < 10; ++r) my[r] = id[r];
    // STRATA_MASK_RESIDENT: the router's own picks that would be read from the SSD are this decode row's demand.  Counted before any move, so
    // a masked expert keeps its count and the mask does not flip it in and out.  Only decode rows (mev) count; the look-ahead's guesses do not.
    // Only ranks below cost_lo count: the stand-ins (ranks at or above it) serve those reads, so they are not paid SSD demand.
    if (res != nullptr && cost != nullptr && mev != nullptr && lane == 0)
        for (int r = 0; r < 10 && r < cost_lo; ++r)
            if (my[r] >= 0 && my[r] < 512 && !mask_cheap(my[r], res, ring, cheap)) atomicAdd(cost + (size_t) layer * 512 + my[r], 1u);
    // STRATA_MASK_RESIDENT control window (route_mask_control): nothing moves, the router's picks stay; the masked picks still count as seen
    // (moved adds nothing).  *ctrl is the same for every lane, so the return is uniform across the warp.
    // STRATA_FRAGILE_CTRL_EVERY (fctrl): a fragile control window moves nothing here either (its row_shift stays 0), and counts the same way.
    if ((ctrl != nullptr && *ctrl != 0) || fragile_held(fctrl, row)) {
        if (lane == 0 && stats != nullptr) {
            unsigned long long masked = 0;
            for (int r = 0; r < 10; ++r)
                if (my[r] >= 0 && my[r] < 512 && mask[my[r]] != 0) ++masked;
            atomicAdd(stats, masked);
        }
        return;
    }
    int seen = 0, moved = 0;
    for (int r = 0; r < 10; ++r) {
        const int e = my[r];
        if (e < 0 || e >= 512 || mask[e] == 0) continue;   // uniform across the warp
        ++seen;
        // STRATA_MASK_RESIDENT: a masked pick that is already cheap stays (it costs no SSD read); it is still counted as seen
        if (res != nullptr && mask_cheap(e, res, ring, cheap)) continue;   // uniform across the warp
        const bool all = mask[e] == 2;   // level 2 (graduated, STRATA_MASK=learn): moves at every rank
        if (!all && r < lo - 1) continue;
        float bv = -INFINITY;
        int bi = -1;
        for (int f = lane; f < 512; f += 32) {   // the best unmasked expert the token did not pick (a strictly greater logit wins; lowest id on ties)
            if (mask[f] != 0) continue;
            if (res != nullptr && !mask_cheap(f, res, ring, cheap)) continue;   // STRATA_MASK_RESIDENT: the substitute must be cheap too
            bool used = false;
            for (int q = 0; q < 10; ++q) used |= my[q] == f;
            if (used) continue;
            if (bi < 0 || l[f] > bv) { bv = l[f]; bi = f; }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
            if (oi >= 0 && (bi < 0 || ov > bv || (ov == bv && oi < bi))) { bv = ov; bi = oi; }
        }
        if (bi < 0) continue;                                // uniform
        const float gap = fmaxf(0.0f, l[e] - bv);
        if (!all && r == lo - 1 && !(edge > 0.0f && gap <= edge)) continue;
        // STRATA_MASK_RESIDENT: below the stand-ins' first rank (cost_lo) a move is gated as the stand-ins gate theirs - the substitute must be
        // within the stand-ins' edge (STRATA_ROUTE_RESIDENT_EDGE; none: no move) - at every level, so STRATA_MASK_FLOOR=all keeps the top ranks'
        // logit check (7 Nov: without it, rank-0 swaps cut answers short and looped the reasoning).  Uniform across the warp (gap is reduced).
        if (res != nullptr && r < cost_lo && !(edge > 0.0f && gap <= edge)) continue;
        // SPEC-recheck-gated step F: a move past the row's shift cap is refused (the pick stays; uniform: gap and the weight are warp-reduced)
        if (cap_row < INFINITY && shift_acc + weights[(size_t) t * 10 + r] * gap > cap_row) {
            if (lane == 0) {
                cap_hit = true;
                if (cap_stats != nullptr) atomicAdd(cap_stats, 1ull);
            }
            continue;
        }
        shift_acc += weights[(size_t) t * 10 + r] * gap;
        if (lane == 0 && row_shift != nullptr) atomicAdd(row_shift + t, weights[(size_t) t * 10 + r] * gap);
        if (lane == 0 && mev != nullptr && row < kMaskEvRows) {   // STRATA_MASK=learn: the replaced pick and its risk
            const unsigned q = mev->n[row]++;
            if (q < kMaskEvCap) mev->e[row][q] = MaskEv{layer, e, weights[(size_t) t * 10 + r] * gap, (int32_t) (1 | ((r & 15) << 1)) /* mask_ev_pack(1, r) */};
        }
        my[r] = bi;
        ++moved;
    }
    if (lane == 0 && stats != nullptr) {
        atomicAdd(stats, (unsigned long long) seen);
        atomicAdd(stats + 1, (unsigned long long) moved);
    }
    if (lane == 0 && cap_hit && cap_stats != nullptr) atomicAdd(cap_stats + 1, 1ull);   // step F: this row had a refused move in this launch
    if (moved == 0) return;                                  // uniform: no move, the router's bits stay
    if (lane == 0) {
        float m = -INFINITY;
        for (int r = 0; r < 10; ++r) m = fmaxf(m, l[my[r]]);
        float ex[10], sum = 0.0f;
        for (int r = 0; r < 10; ++r) { ex[r] = expf(l[my[r]] - m); sum += ex[r]; }
        for (int r = 0; r < 10; ++r) { id[r] = my[r]; weights[(size_t) t * 10 + r] = ex[r] / sum; }
    }
}
__global__ void route_mask_k(float* __restrict__ logits, int32_t* __restrict__ ids, float* __restrict__ weights,
                             const uint8_t* __restrict__ mask, int n_tok, int lo, float edge, float* __restrict__ row_shift,
                             unsigned long long* __restrict__ stats, MaskEvs* __restrict__ mev, int layer, int row_base, bool reset,
                             const int32_t* __restrict__ res, const uint8_t* __restrict__ ring, const uint8_t* __restrict__ cheap,
                             uint32_t* __restrict__ cost, const int* __restrict__ ctrl, int cost_lo, const uint32_t* __restrict__ fctrl,
                             const float* __restrict__ cap, unsigned long long* __restrict__ cap_stats) {
    const int t = (int) blockIdx.x;   // one warp (32 threads) per token
    if (t >= n_tok) return;
    route_mask_warp(logits, ids, weights, mask, t, lo, edge, row_shift, stats, mev, layer, row_base, reset, res, ring, cheap, cost, ctrl,
                    cost_lo, fctrl, cap, cap_stats);
}
}  // namespace
bool route_mask_set(const uint8_t* host, int64_t n_layers, int64_t n_expert, int lo, float edge, std::string& msg, int cost_lo) {
    if (g_mask_dev != nullptr) { msg = "already set"; return false; }
    int dev = -1;
    if (cudaGetDevice(&dev) != cudaSuccess) { (void) cudaGetLastError(); msg = "no CUDA device"; return false; }
    const size_t n = (size_t) n_layers * (size_t) n_expert;
    uint8_t* dm = nullptr;
    unsigned long long* st = nullptr;
    if (cudaMalloc((void**) &dm, n) != cudaSuccess || cudaMemcpy(dm, host, n, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMalloc((void**) &st, 2 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMemset(st, 0, 2 * sizeof(unsigned long long)) != cudaSuccess) {
        (void) cudaGetLastError();
        if (dm != nullptr) (void) cudaFree(dm);
        if (st != nullptr) (void) cudaFree(st);
        msg = "no room on the GPU for it";
        return false;
    }
    // the SSD-demand counters (STRATA_MASK_RESIDENT): [layers][512], zeroed.  The kernel indexes them by 512, so they exist for a 512-expert
    // router only; on any failure they stay nullptr and the feature has no cost data (the mask itself is unaffected).
    uint32_t* cst = nullptr;
    const size_t cn = (size_t) n_layers * 512;
    if (n_expert == 512) {
        if (cudaMalloc((void**) &cst, cn * sizeof(uint32_t)) != cudaSuccess || cudaMemset(cst, 0, cn * sizeof(uint32_t)) != cudaSuccess) {
            (void) cudaGetLastError();
            if (cst != nullptr) (void) cudaFree(cst);
            cst = nullptr;
        }
    }
    // the control flag (STRATA_MASK_RESIDENT): a host-mapped int beside the stats, as g_cheap_host is (read by the kernel at launch, written
    // between windows by route_mask_control).  On failure: nullptr, so no control windows.
    int* ctl = nullptr;
    int* ctl_d = nullptr;
    if (cudaHostAlloc((void**) &ctl, sizeof(int), cudaHostAllocMapped) != cudaSuccess) {
        (void) cudaGetLastError();
        ctl = nullptr;
    } else if (cudaHostGetDevicePointer((void**) &ctl_d, ctl, 0) != cudaSuccess) {
        (void) cudaGetLastError();
        (void) cudaFreeHost(ctl);
        ctl = nullptr;
        ctl_d = nullptr;
    } else {
        *ctl = 0;
    }
    if (ctl == nullptr) std::fprintf(stderr, "strata: STRATA_MASK_RESIDENT: no host-mapped control flag, so no control windows\n");
    g_mask_dev = dm;
    g_mask_host.assign(host, host + n);
    g_mask_stats = st;
    g_mask_ctrl_host = ctl;
    g_mask_ctrl_dev = ctl_d;
    g_mask_cost = cst;
    g_mask_cost_n = cst != nullptr ? cn : 0;
    g_mask_dev_id = dev;
    g_mask_layers = n_layers;
    g_mask_ne = n_expert;
    g_mask_lo = lo;
    g_mask_cost_lo = cost_lo < 0 ? lo : cost_lo;
    g_mask_edge = edge;
    return true;
}
const uint8_t* route_mask_layer(int64_t layer, int64_t n_expert) {
    if (g_mask_dev == nullptr || n_expert != g_mask_ne || layer < 0 || layer >= g_mask_layers) return nullptr;
    int dev = -1;
    if (cudaGetDevice(&dev) != cudaSuccess) { (void) cudaGetLastError(); return nullptr; }
    return dev == g_mask_dev_id ? g_mask_dev + (size_t) layer * (size_t) n_expert : nullptr;   // another GPU's router keeps its own picks
}
const std::vector<uint8_t>* route_mask_host() { return &g_mask_host; }
bool route_mask_cost_read(std::vector<uint32_t>& out) {   // the SSD-demand counters, [layers][512] (not reset: the learner takes the deltas)
    out.clear();
    if (g_mask_cost == nullptr) return false;
    out.resize(g_mask_cost_n);
    if (cudaMemcpy(out.data(), g_mask_cost, g_mask_cost_n * sizeof(uint32_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
        (void) cudaGetLastError();
        out.clear();
        return false;
    }
    return true;
}
void native_route_mask(float* logits, int32_t* ids, float* weights, const uint8_t* mask, int n_tok, float* row_shift, void* stream,
                       MaskEvs* mev, int layer, int row_base, const int32_t* res, const uint8_t* ring, const uint8_t* cheap) {
    if (!mask || !logits || !ids || !weights || n_tok < 1 || g_mask_stats == nullptr) return;
    // the mask's control flag (STRATA_MASK_RESIDENT, or STRATA_MASK_CTRL under the old rule: the window moves nothing in either case) and the
    // fragile control flag (STRATA_FRAGILE_CTRL_EVERY: the same window, no move), both read at launch
    route_mask_k<<<(unsigned) n_tok, 32, 0, static_cast<cudaStream_t>(stream)>>>(logits, ids, weights, mask, n_tok, g_mask_lo,
                                                                              g_mask_edge, row_shift, g_mask_stats, mev, layer, row_base,
                                                                              layer == 0, res, ring, cheap, g_mask_cost,
                                                                              g_mask_ctrl_dev, g_mask_cost_lo, g_frag_ctrl_dev, g_cap_dev, g_cap_stats);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
// STRATA_MASK_RESIDENT: the window's control flag, written between windows (no kernel of the previous window still reads it: run() syncs its
// stream and joins the look-ahead's before it returns).  The write is a host store to mapped memory, as route_cfg_set_budgets' is.
void route_mask_control(bool on) {
    if (g_mask_ctrl_host == nullptr) return;
    *(volatile int*) g_mask_ctrl_host = on ? 1 : 0;
    __sync_synchronize();
}
// STRATA_FRAGILE_CTRL_EVERY: the window's fragile control flag, written between windows (same contract as route_mask_control).  on = true
// allocates the flag on first use; false when it cannot be had (then this window is not a control window).  off is a no-op until it exists.
// The flag is allocated once, before any graph capture (route_fragile_control_alloc at the verifier's init): a kernel captured while the
// device alias is still nullptr would never read the flag.  Without it, route_fragile_control(true) is false and no window is a control.
bool route_fragile_control_alloc() {
    if (g_frag_ctrl_host != nullptr) return true;
    uint32_t* h = nullptr;
    uint32_t* d = nullptr;
    if (cudaHostAlloc((void**) &h, sizeof(uint32_t), cudaHostAllocMapped) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    if (cudaHostGetDevicePointer((void**) &d, h, 0) != cudaSuccess) {
        (void) cudaGetLastError();
        (void) cudaFreeHost(h);
        return false;
    }
    *h = 0;
    g_frag_ctrl_host = h;
    g_frag_ctrl_dev = d;
    return true;
}
bool route_fragile_hold_ready() { return g_frag_ctrl_host != nullptr; }
bool route_fragile_hold(uint32_t bits) {
    if (g_frag_ctrl_host == nullptr) return false;
    *(volatile uint32_t*) g_frag_ctrl_host = bits;
    __sync_synchronize();
    return true;
}
// A row count n (the SPEC-recheck-cheap form): rows 0..n-1 held.  n >= 32 holds every row (the all-ones mask).
bool route_fragile_rows(int n) {
    if (n < 0) n = 0;
    return route_fragile_hold(n >= 32 ? 0xFFFFFFFFu : ((1u << n) - 1u));
}
bool route_fragile_control(bool on) { return route_fragile_hold(on ? 0xFFFFFFFFu : 0u); }
// SPEC-recheck-gated step F: the cap array (host-mapped, all +inf to start) and the refusal counters (device).  false when one cannot be had;
// then the kernels get nullptr and nothing is capped.
bool route_shift_cap_alloc() {
    if (g_cap_host != nullptr) return true;
    float* h = nullptr;
    float* d = nullptr;
    if (cudaHostAlloc((void**) &h, 32 * sizeof(float), cudaHostAllocMapped) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    if (cudaHostGetDevicePointer((void**) &d, h, 0) != cudaSuccess) {
        (void) cudaGetLastError();
        (void) cudaFreeHost(h);
        return false;
    }
    for (int i = 0; i < 32; ++i) h[i] = INFINITY;
    unsigned long long* st = nullptr;
    if (cudaMalloc((void**) &st, 2 * sizeof(unsigned long long)) != cudaSuccess || cudaMemset(st, 0, 2 * sizeof(unsigned long long)) != cudaSuccess) {
        (void) cudaGetLastError();
        if (st != nullptr) (void) cudaFree(st);
        (void) cudaFreeHost(h);
        return false;
    }
    g_cap_host = h;
    g_cap_dev = d;
    g_cap_stats = st;
    return true;
}
// Between windows: the caps of rows 0..n-1 (n <= 32); the rest stay +inf.  Graph-safe as the fragile hold is (read at each launch).
bool route_shift_cap_set(const float* caps, int n) {
    if (g_cap_host == nullptr) return false;
    for (int i = 0; i < 32; ++i) ((volatile float*) g_cap_host)[i] = (caps != nullptr && i < n) ? caps[i] : INFINITY;
    __sync_synchronize();
    return true;
}
// Between requests: the refusals since the last read (moves, row launches), read and reset.  false when none exists.
bool route_shift_cap_read(unsigned long long& moves, unsigned long long& rows) {
    moves = rows = 0;
    if (g_cap_stats == nullptr) return false;
    unsigned long long v[2] = {0, 0};
    if (cudaMemcpy(v, g_cap_stats, sizeof v, cudaMemcpyDeviceToHost) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    if (cudaMemset(g_cap_stats, 0, sizeof v) != cudaSuccess) (void) cudaGetLastError();
    moves = v[0];
    rows = v[1];
    return true;
}
bool route_mask_update(const std::vector<uint8_t>& host) {
    if (g_mask_dev == nullptr || host.size() != g_mask_host.size()) return false;
    const size_t ne = (size_t) g_mask_ne;
    if (host == g_mask_host) return true;   // nothing differs: no wait, no copy
    // the look-ahead learner's batch reads these rows on its own non-blocking stream (la_b_grad_k): wait for all work before rewriting them
    // (7 Nov review: a synchronous copy does not order against non-blocking streams).  Rare: a mask change, every 256 tokens at most.
    if (cudaDeviceSynchronize() != cudaSuccess) { (void) cudaGetLastError(); return false; }
    for (int64_t il = 0; il < g_mask_layers; ++il) {
        const size_t o = (size_t) il * ne;
        if (std::equal(host.begin() + (ptrdiff_t) o, host.begin() + (ptrdiff_t) (o + ne), g_mask_host.begin() + (ptrdiff_t) o)) continue;
        if (cudaMemcpy(g_mask_dev + o, host.data() + o, ne, cudaMemcpyHostToDevice) != cudaSuccess) {
            (void) cudaGetLastError();
            return false;
        }
        std::copy(host.begin() + (ptrdiff_t) o, host.begin() + (ptrdiff_t) (o + ne), g_mask_host.begin() + (ptrdiff_t) o);
    }
    return true;
}
bool route_mask_read(unsigned long long& seen, unsigned long long& moved) {   // the request's counts (read and reset; between requests)
    if (g_mask_stats == nullptr) return false;
    unsigned long long h[2] = {0, 0};
    if (cudaMemcpy(h, g_mask_stats, sizeof h, cudaMemcpyDeviceToHost) != cudaSuccess) { (void) cudaGetLastError(); return false; }
    (void) cudaMemset(g_mask_stats, 0, sizeof h);
    seen = h[0];
    moved = h[1];
    return true;
}
struct PostArgs {
    float* logits;
    int32_t* ids;
    float* weights;
    int n_tok;
    int layer;
    int row_base;
    const uint8_t* mask;
    int mask_lo;
    float mask_edge;
    int cost_lo;
    const int* mctrl;
    unsigned long long* mstats;
    uint32_t* cost;
    MaskEvs* mev;
    const int32_t* mres;
    const uint8_t* mring;
    const uint8_t* mcheap;
    float* row_shift;
    const int32_t* res;
    unsigned long long* rstats;
    float* want;
    int32_t* rows;
    float* repl;
    const uint8_t* rring;
    const uint8_t* rcheap;
    float cheap_margin;
    int cheap_rank0;
    int cheap_anytier;
    unsigned long long* bstats;
    const float* pair_c;
    int32_t* lrn_rec;
    const float* pair_pe;
    unsigned long long* detail;
    const uint32_t* fctrl;
    const RouteCfg* cfg;
    float* label;
    float* margin_out;          // SPEC-recheck-gated C (layers 0-3; nullptr = off)
    const float* cap;           // SPEC-recheck-gated F: the per-row shift caps (nullptr = off)
    unsigned long long* cap_stats;
};
// SPEC-router-fuse step 2: the layer's own post-routing stages in ONE launch, one warp per token: the learner's label copy (the logits,
// unchanged by the stages), the layer's mask (route_mask_warp), then route_resident_warp.  The same bodies in the same order as the
// separate launches, so every output is the same.  Off (STRATA_ROUTER_FUSE=0 or STRATA_ROUTER_FUSE_POST=0) the verifier makes the calls.
__global__ void route_post_k(PostArgs a) {
    const int t = (int) blockIdx.x;   // one warp (32 threads) per token
    if (t >= a.n_tok) return;
    const int lane = (int) threadIdx.x;
    if (a.label != nullptr)
        for (int i = lane; i < 512; i += 32) a.label[(size_t) t * 512 + i] = a.logits[(size_t) t * 512 + i];
    if (a.mask != nullptr)
        route_mask_warp(a.logits, a.ids, a.weights, a.mask, t, a.mask_lo, a.mask_edge, a.row_shift, a.mstats, a.mev, a.layer, a.row_base,
                        a.layer == 0, a.mres, a.mring, a.mcheap, a.cost, a.mctrl, a.cost_lo, a.fctrl, a.cap, a.cap_stats);
    __syncwarp();   // the mask's ids and weights are what the resident stage reads
    if (a.res != nullptr)
        route_resident_warp(a.logits, a.ids, a.weights, a.res, t, a.cfg, a.rstats, a.want, a.detail, a.rows, a.repl, a.rring, a.rcheap,
                            a.cheap_margin, a.cheap_rank0, a.cheap_anytier, a.bstats, a.row_shift, a.pair_c, a.lrn_rec, a.layer, a.pair_pe,
                            a.mev, a.row_base, a.fctrl, a.margin_out, a.cap, a.cap_stats);
}
void native_route_post(const RoutePostArgs& r, void* stream) {
    if (!stream || r.n_tok < 1) throw std::invalid_argument("native_route_post: a stream and at least one row");
    const bool mask_on = r.mask != nullptr && r.logits != nullptr && r.ids != nullptr && r.weights != nullptr && g_mask_stats != nullptr;
    if (r.res != nullptr && g_cfg_dev == nullptr) throw std::invalid_argument("native_route_post: the resident stage needs route_cfg_init");
    const bool res_on = r.res != nullptr;
    if (!mask_on && !res_on && r.label == nullptr) return;
    PostArgs a{};
    a.logits = r.logits; a.ids = r.ids; a.weights = r.weights; a.n_tok = r.n_tok; a.layer = r.layer; a.row_base = r.row_base;
    a.mask = mask_on ? r.mask : nullptr;
    a.mask_lo = g_mask_lo; a.mask_edge = g_mask_edge; a.cost_lo = g_mask_cost_lo; a.mctrl = g_mask_ctrl_dev; a.mstats = g_mask_stats;
    a.cost = g_mask_cost; a.mev = r.mev; a.mres = r.mres; a.mring = r.mring; a.mcheap = r.mcheap; a.row_shift = r.row_shift;
    a.res = res_on ? r.res : nullptr;
    a.rstats = r.rstats; a.want = r.want; a.rows = r.n_tok <= 8 ? r.rows : nullptr; a.repl = r.n_tok <= 8 ? r.repl : nullptr;
    a.rring = r.rring; a.rcheap = r.rcheap; a.bstats = r.bstats; a.pair_c = r.pair_c; a.lrn_rec = r.lrn_rec; a.pair_pe = r.pair_pe;
    a.detail = r.n_tok <= 8 ? g_detail : nullptr; a.fctrl = g_frag_ctrl_dev; a.cfg = g_cfg_dev;
    resident_env(a.cheap_margin, a.cheap_rank0, a.cheap_anytier);
    a.label = r.label;
    a.margin_out = r.margin_out; a.cap = g_cap_dev; a.cap_stats = g_cap_stats;
    route_post_k<<<(unsigned) r.n_tok, 32, 0, static_cast<cudaStream_t>(stream)>>>(a);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
static int32_t* g_pred_host = nullptr;
static int32_t* g_pred_dev = nullptr;
static float* g_pred_logits = nullptr;
static float* g_pred_w = nullptr;
void route_pred_alloc(int64_t n_layers) {
    if (g_pred_host != nullptr || n_layers <= 0) return;
    const size_t n = (size_t) n_layers * kRoutePredDists * 8 * 10;
    if (cudaHostAlloc((void**) &g_pred_host, n * sizeof(int32_t), cudaHostAllocMapped) != cudaSuccess) { cudaGetLastError(); g_pred_host = nullptr; return; }
    if (cudaHostGetDevicePointer((void**) &g_pred_dev, g_pred_host, 0) != cudaSuccess ||
        cudaMalloc((void**) &g_pred_logits, (size_t) kRoutePredDists * 8 * 512 * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void**) &g_pred_w, (size_t) kRoutePredDists * 8 * 10 * sizeof(float)) != cudaSuccess) {
        cudaGetLastError();
        g_pred_host = nullptr;
        g_pred_dev = nullptr;
        return;
    }
    for (size_t i = 0; i < n; ++i) g_pred_host[i] = -1;
    note_scratch_dev();
}
bool route_pred_active() { return g_pred_host != nullptr; }
const int32_t* route_pred_host() { return g_pred_host; }
int32_t* route_pred_dev(int64_t target_layer, int dist) {
    const int slot = route_pred_slot(dist);
    return g_pred_dev && slot >= 0 ? g_pred_dev + ((size_t) target_layer * kRoutePredDists + (size_t) slot) * 80 : nullptr;
}
float* route_pred_logits() { return g_pred_logits; }
float* route_pred_weights() { return g_pred_w; }
// SPEC-router-fuse step 1: the per-slot scratch (slot = route_pred_slot(dist)); slot 0 is route_pred_logits / route_pred_weights.
float* route_pred_logits_at(int slot) { return g_pred_logits != nullptr && slot >= 0 && slot < kRoutePredDists ? g_pred_logits + (size_t) slot * 8 * 512 : nullptr; }
float* route_pred_weights_at(int slot) { return g_pred_w != nullptr && slot >= 0 && slot < kRoutePredDists ? g_pred_w + (size_t) slot * 8 * 10 : nullptr; }
// STRATA_LA_MODEL=<file> (opt-in): the published per-layer d=1 linear predictors (tools/predictor/export_la_model.py folds their
// standardisation in): for each target layer a bf16 [512 x H] matrix and a float bias, applied to the previous layer's router input.
static uint16_t* g_lam_w = nullptr;
static float* g_lam_b = nullptr;
static std::vector<char> g_lam_has;
static int64_t g_lam_h = 0;
static int g_lam_dev = -1;   // the GPU the matrices live on: another GPU's look-ahead keeps its router
bool route_pred_model_load(const char* path, int64_t n_layers, int64_t n_embd, std::string& msg) {
    if (g_lam_w != nullptr) {
        int dev = -1;
        const bool same = cudaGetDevice(&dev) == cudaSuccess && dev == g_lam_dev;
        msg = same ? "already loaded" : "loaded on another GPU only (this GPU's look-ahead keeps the router)";
        return same && g_lam_h == n_embd && (int64_t) g_lam_has.size() == n_layers;
    }
    FILE* f = std::fopen(path, "rb");
    if (f == nullptr) { msg = std::string("cannot open ") + path; return false; }
    uint32_t hdr[4] = {0, 0, 0, 0};
    const bool hdr_ok = std::fread(hdr, sizeof hdr, 1, f) == 1 && hdr[0] == 0x314D414Cu /* "LAM1" */ &&
                        (int64_t) hdr[1] == n_layers && hdr[2] == 512u && (int64_t) hdr[3] == n_embd;
    if (!hdr_ok) {
        std::fclose(f);
        msg = "not an LAM1 file for this model (layers, 512 experts, hidden size)";
        return false;
    }
    const size_t per_w = (size_t) 512 * (size_t) n_embd;
    std::vector<uint16_t> w((size_t) n_layers * per_w);
    std::vector<float> b((size_t) n_layers * 512, 0.0f);
    std::vector<char> has((size_t) n_layers, 0);
    bool ok = true;
    for (int64_t l = 0; l < n_layers && ok; ++l) {
        uint8_t present = 0;
        ok = std::fread(&present, 1, 1, f) == 1;
        if (!ok || present == 0) continue;
        ok = std::fread(w.data() + (size_t) l * per_w, sizeof(uint16_t), per_w, f) == per_w &&
             std::fread(b.data() + (size_t) l * 512, sizeof(float), 512, f) == 512;
        has[(size_t) l] = ok ? 1 : 0;
    }
    std::fclose(f);
    if (!ok) { msg = "the file is cut short"; return false; }
    if (cudaMalloc((void**) &g_lam_w, w.size() * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc((void**) &g_lam_b, b.size() * sizeof(float)) != cudaSuccess) {
        cudaGetLastError();
        if (g_lam_w != nullptr) (void) cudaFree(g_lam_w);
        g_lam_w = nullptr;
        g_lam_b = nullptr;
        msg = "no room on the GPU for it";
        return false;
    }
    if (cudaMemcpy(g_lam_w, w.data(), w.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(g_lam_b, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaGetLastError();
        (void) cudaFree(g_lam_w);
        (void) cudaFree(g_lam_b);
        g_lam_w = nullptr;
        g_lam_b = nullptr;
        msg = "the copy to the GPU failed";
        return false;
    }
    g_lam_has = std::move(has);
    g_lam_h = n_embd;
    if (cudaGetDevice(&g_lam_dev) != cudaSuccess) g_lam_dev = -1;
    int64_t n = 0;
    for (char c : g_lam_has) n += c;
    msg = std::to_string(n) + " layers, " + std::to_string((w.size() * 2 + b.size() * 4) >> 20) + " MiB of VRAM";
    return true;
}
const uint16_t* route_pred_model_w(int64_t target_layer) {
    if (g_lam_w == nullptr || target_layer < 0 || target_layer >= (int64_t) g_lam_has.size() || !g_lam_has[(size_t) target_layer]) return nullptr;
    int dev = -1;
    if (cudaGetDevice(&dev) != cudaSuccess || dev != g_lam_dev) return nullptr;   // recorded for another GPU
    return g_lam_w + (size_t) target_layer * 512 * (size_t) g_lam_h;
}
const float* route_pred_model_b(int64_t target_layer) {
    return route_pred_model_w(target_layer) != nullptr ? g_lam_b + (size_t) target_layer * 512 : nullptr;
}
__global__ void pred_bias_add_k(float* __restrict__ logits, const float* __restrict__ bias, int n_tok) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_tok * 512) logits[i] += bias[i % 512];
}
void route_pred_bias_add(float* logits, const float* bias, int n_tok, void* stream) {
    pred_bias_add_k<<<(n_tok * 512 + 255) / 256, 256, 0, (cudaStream_t) stream>>>(logits, bias, n_tok);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
// STRATA_LA_EDGE_SKIP (default on since 21 Oct, =0 off; decode look-ahead only: verify.cpp lookahead_node): the predicted rank lo-1 pick (the edge rank of the
// floor, route_resident_k's `edge`) is not read ahead when a resident expert outside the predicted ten scores within `edge` logits of it
// in the predicted logits: the router takes that stand-in, so the read would be wasted.  Its slot is set to -1 (the host's warm path skips
// a negative id) and counted.  One warp per token; the reads of ids finish before lane 0 writes (the shuffle is a warp barrier).
// Approximation: the router also refuses a stand-in past its maxrank or a heavy pick (wmax) and counts ring-landed experts as resident;
// neither is seen here.
// one warp of token t (SPEC-router-fuse step 1: a device function, shared by pred_edge_skip_k and the fused selection kernel)
__device__ __forceinline__ void pred_edge_skip_warp(const float* __restrict__ logits, int32_t* __restrict__ ids, const int32_t* __restrict__ res,
                                                    const RouteCfg* cfg, unsigned long long* __restrict__ skipped, int t) {
    const int lane = (int) threadIdx.x;
    const int lo = ((const volatile RouteCfg*) cfg)->lo;
    const float edge = ((const volatile RouteCfg*) cfg)->edge;
    if (edge < 0.f || lo <= 0 || lo - 1 >= 10) return;   // uniform across the warp
    const int r = lo - 1;
    const int32_t* id = ids + (size_t) t * 10;
    const int e = id[r];
    if (e < 0 || e >= 512 || res[e] >= 0) return;   // uniform: no pick, or already resident (not read ahead anyway)
    const float* l = logits + (size_t) t * 512;
    float bv = -INFINITY;
    for (int f = lane; f < 512; f += 32) {
        if (res[f] < 0) continue;   // not resident (the router's stand-in candidates are the resident experts)
        bool used = false;
        for (int q = 0; q < 10; ++q) used |= id[q] == f;   // the router's stand-in is not one of the ten picks
        if (!used && l[f] > bv) bv = l[f];
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) bv = fmaxf(bv, __shfl_xor_sync(0xffffffffu, bv, off));
    if (lane == 0 && bv > -INFINITY && l[e] - bv <= edge) {
        ids[(size_t) t * 10 + r] = -1;
        atomicAdd(skipped, 1ull);
    }
}
__global__ void pred_edge_skip_k(const float* __restrict__ logits, int32_t* __restrict__ ids, const int32_t* __restrict__ res,
                                 const RouteCfg* cfg, unsigned long long* __restrict__ skipped, int n_tok) {
    const int t = blockIdx.x;
    if (t >= n_tok) return;
    pred_edge_skip_warp(logits, ids, res, cfg, skipped, t);
}
static unsigned long long* g_pred_skip = nullptr;   // device counter: skipped predictions since the last take
void route_pred_edge_skip_alloc() {
    if (g_pred_skip != nullptr) return;
    if (cudaMalloc((void**) &g_pred_skip, sizeof(unsigned long long)) != cudaSuccess) { cudaGetLastError(); g_pred_skip = nullptr; return; }
    cudaMemset(g_pred_skip, 0, sizeof(unsigned long long));
    note_scratch_dev();
}
void route_pred_edge_skip(const float* pred_logits, int32_t* pred_ids, const int32_t* res_layer, int n_tok, void* stream) {
    const RouteCfg* cfg = route_cfg_device();
    if (cfg == nullptr || g_pred_skip == nullptr || !scratch_here() || pred_logits == nullptr || pred_ids == nullptr || res_layer == nullptr || !stream || n_tok < 1)
        return;
    pred_edge_skip_k<<<(unsigned) n_tok, 32, 0, static_cast<cudaStream_t>(stream)>>>(pred_logits, pred_ids, res_layer, cfg, g_pred_skip, n_tok);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
// the skips counted since the last take (the request's stats line), then the counter is zeroed.  Called between requests, when the
// decode's stream is idle.
unsigned long long route_pred_edge_skips_take() {
    if (g_pred_skip == nullptr) return 0;
    unsigned long long v = 0;
    if (cudaMemcpy(&v, g_pred_skip, sizeof v, cudaMemcpyDeviceToHost) != cudaSuccess) { cudaGetLastError(); return 0; }
    cudaMemset(g_pred_skip, 0, sizeof v);
    return v;
}
// STRATA_LA_TAIL_READ (opt-in; decode look-ahead only, verify.cpp lookahead_node): a predicted pick at rank lo..hi that the router would
// replace with a resident stand-in is not read ahead (its id becomes -1; the host warm skips a negative id).  The test is route_resident_k's
// for a rank at or above lo: the stand-in is the best resident expert outside the predicted ten, the place cap (maxrank) and the heavy-pick
// cap (wmax) refuse as there, and the replacement is taken when w * max(gap, 0) <= budget_tail (budget_tail > 0) or gap <= margin (else).
// The open rule (margin unlimited, tail budget 0) is not tested here: the host leaves every such guess unread (la_tail_read_mode 2).
// The config is the host-mapped RouteCfg, read at every launch, so the window's budgets reach this kernel as they reach the router.
// Approximations: the router's own picks before a swap (its `used` set) and the ring's landed copies (counted resident there) are not seen.
// one warp of token t (SPEC-router-fuse step 1: a device function, shared by pred_tail_read_k and the fused selection kernel)
__device__ __forceinline__ void pred_tail_read_warp(const float* __restrict__ logits, int32_t* __restrict__ ids, const float* __restrict__ w,
                                                    const int32_t* __restrict__ res, const RouteCfg* cfg, unsigned long long* __restrict__ skipped,
                                                    int t, int all_ranks) {
    const int lane = (int) threadIdx.x;
    const int lo = ((const volatile RouteCfg*) cfg)->lo;
    const int hi = ((const volatile RouteCfg*) cfg)->hi;
    const float margin = ((const volatile RouteCfg*) cfg)->margin;
    const float budget_tail = ((const volatile RouteCfg*) cfg)->budget_tail;
    const float budget = ((const volatile RouteCfg*) cfg)->budget;
    const int maxrank = ((const volatile RouteCfg*) cfg)->maxrank;
    const float wmax = ((const volatile RouteCfg*) cfg)->wmax;
    const bool tail_open = margin >= 1e30f && budget_tail <= 0.f;   // the open rule: the host leaves those ranks unread (no test)
    // STRATA_LA_TAIL_READ=2 (all_ranks): ranks below lo take the router's budget test too (w * gap <= budget), so a rank 0..lo-1
    // pick a stand-in will serve is not read either - only the picks that will be computed exactly are read ahead
    const int r0 = all_ranks != 0 && budget > 0.f ? 0 : lo;
    const float* l = logits + (size_t) t * 512;
    int32_t* id = ids + (size_t) t * 10;
    for (int r = r0; r <= hi && r < 10; ++r) {   // uniform across the warp: every value below is reduced or read from the same place
        if (r >= lo && tail_open) break;
        const int e = id[r];
        if (e < 0 || e >= 512 || res[e] >= 0) continue;   // no pick, or resident (no stand-in is needed)
        if (wmax > 0.f && w[(size_t) t * 10 + r] >= wmax) continue;   // a heavy pick stays exact (the router's wmax)
        float bv = -INFINITY;
        for (int f = lane; f < 512; f += 32) {
            if (res[f] < 0) continue;   // the router's stand-in candidates are the resident experts
            bool used = false;
            for (int q = 0; q < 10; ++q) used |= id[q] == f;
            if (!used && l[f] > bv) bv = l[f];
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) bv = fmaxf(bv, __shfl_xor_sync(0xffffffffu, bv, off));
        if (bv == -INFINITY) continue;   // no resident stand-in: the pick is computed exactly
        if (maxrank > 0) {
            int place = 0;   // the stand-in's place in the router's ranking of all 512 experts
            for (int f = lane; f < 512; f += 32) place += l[f] > bv;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) place += __shfl_xor_sync(0xffffffffu, place, off);
            if (place >= maxrank) continue;   // refused by the place cap: exact
        }
        const float gap = l[e] - bv;
        const bool take = r < lo ? w[(size_t) t * 10 + r] * fmaxf(gap, 0.f) <= budget
                        : budget_tail > 0.f ? w[(size_t) t * 10 + r] * fmaxf(gap, 0.f) <= budget_tail : gap <= margin;
        if (lane == 0 && take) {
            id[r] = -1;
            atomicAdd(skipped, 1ull);
        }
        __syncwarp();   // lane 0's write is what the next rank's reads see
    }
}
__global__ void pred_tail_read_k(const float* __restrict__ logits, int32_t* __restrict__ ids, const float* __restrict__ w, const int32_t* __restrict__ res,
                                 const RouteCfg* cfg, unsigned long long* __restrict__ skipped, int n_tok, int all_ranks) {
    const int t = blockIdx.x;
    if (t >= n_tok) return;
    pred_tail_read_warp(logits, ids, w, res, cfg, skipped, t, all_ranks);
}
static unsigned long long* g_pred_tail = nullptr;   // device counter: stand-in-served predictions not read since the last take
void route_pred_tail_read_alloc() {
    if (g_pred_tail != nullptr) return;
    if (cudaMalloc((void**) &g_pred_tail, sizeof(unsigned long long)) != cudaSuccess) { cudaGetLastError(); g_pred_tail = nullptr; return; }
    cudaMemset(g_pred_tail, 0, sizeof(unsigned long long));
    note_scratch_dev();
}
void route_pred_tail_read(const int32_t* res_layer, int32_t* pred_ids, int n_tok, void* stream, int all_ranks) {
    const RouteCfg* cfg = route_cfg_device();
    if (cfg == nullptr || g_pred_tail == nullptr || g_pred_w == nullptr || !scratch_here() || pred_ids == nullptr || res_layer == nullptr || !stream || n_tok < 1)
        return;
    pred_tail_read_k<<<(unsigned) n_tok, 32, 0, static_cast<cudaStream_t>(stream)>>>(g_pred_logits, pred_ids, g_pred_w, res_layer, cfg, g_pred_tail, n_tok, all_ranks);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
unsigned long long route_pred_tail_read_take() {
    if (g_pred_tail == nullptr) return 0;
    unsigned long long v = 0;
    if (cudaMemcpy(&v, g_pred_tail, sizeof v, cudaMemcpyDeviceToHost) != cudaSuccess) { cudaGetLastError(); return 0; }
    cudaMemset(g_pred_tail, 0, sizeof v);
    return v;
}
// SPEC-router-fuse step 1: the look-ahead's post-GEMV chain of ALL distances in one launch.  Block (token t, dist slot d) runs one warp:
// the top-10 of the dist's logits, then the same bodies as the one-kernel launches, in the same order: the layer mask of the target layer,
// route_pred_edge_skip, route_pred_tail_read (__syncwarp between them: each reads the ids the previous one wrote).  Dists are independent
// (their own logits, ids, weights, target layer), so the per-dist results equal the sequential chain's.
struct PredSelArgs {
    float* logits[kRoutePredDists];
    int32_t* ids[kRoutePredDists];
    float* w[kRoutePredDists];
    const uint8_t* mask[kRoutePredDists];
    const int32_t* mres[kRoutePredDists];
    const uint8_t* ring[kRoutePredDists];
    const uint8_t* cheap[kRoutePredDists];
    const int32_t* res[kRoutePredDists];
    int layer[kRoutePredDists];
    unsigned flags[kRoutePredDists];   // bit 0 mask, bit 1 edge skip, bit 2 tail read
    int n_tok;
    int lo_mask;
    float edge_mask;
    int cost_lo;
    const int* ctrl;
    const uint32_t* fctrl;
    unsigned long long* mstats;
    const RouteCfg* cfg;
    unsigned long long* eskip;
    unsigned long long* tskip;
    int tail_all;
};
__global__ void route_pred_sel_k(PredSelArgs a) {
    const int t = (int) blockIdx.x, d = (int) blockIdx.y;   // (one warp per block)
    if (t >= a.n_tok || a.logits[d] == nullptr) return;   // uniform across the block
    float* lg = a.logits[d];
    int32_t* id = a.ids[d];
    float* w = a.w[d];
    route_warp_top10(lg + (size_t) t * 512, id + (size_t) t * 10, w + (size_t) t * 10, (int) threadIdx.x);
    __syncwarp();
    if (a.flags[d] & 1u) {
        route_mask_warp(lg, id, w, a.mask[d], t, a.lo_mask, a.edge_mask, nullptr, a.mstats, nullptr, a.layer[d], 0, a.layer[d] == 0,
                        a.mres[d], a.ring[d], a.cheap[d], nullptr, a.ctrl, a.cost_lo, a.fctrl);
    }
    __syncwarp();
    if (a.flags[d] & 2u) pred_edge_skip_warp(lg, id, a.res[d], a.cfg, a.eskip, t);
    __syncwarp();
    if (a.flags[d] & 4u) pred_tail_read_warp(lg, id, w, a.res[d], a.cfg, a.tskip, t, a.tail_all);
}
void route_pred_select(const RoutePredDist (&dd)[kRoutePredDists], int n_tok, int tail_all, void* stream) {
    if (!stream || n_tok < 1 || n_tok > 8) throw std::invalid_argument("route_pred_select: a stream and 1..8 rows");
    PredSelArgs a{};
    const RouteCfg* cfg = route_cfg_device();
    bool any = false;
    for (int i = 0; i < kRoutePredDists; ++i) {
        const RoutePredDist& d = dd[i];
        a.logits[i] = d.logits; a.ids[i] = d.ids; a.w[i] = d.w; a.mask[i] = d.mask; a.mres[i] = d.mres; a.ring[i] = d.ring;
        a.cheap[i] = d.cheap; a.res[i] = d.res; a.layer[i] = d.layer; a.flags[i] = 0;
        if (d.logits == nullptr || d.ids == nullptr || d.w == nullptr) { a.logits[i] = nullptr; continue; }
        any = true;
        // the same guards as the one-kernel wrappers (native_route_mask, route_pred_edge_skip, route_pred_tail_read)
        if (d.mask != nullptr && g_mask_stats != nullptr) a.flags[i] |= 1u;
        if (d.res != nullptr && d.edge_on && cfg != nullptr && g_pred_skip != nullptr && scratch_here()) a.flags[i] |= 2u;
        if (d.res != nullptr && d.tail_on && cfg != nullptr && g_pred_tail != nullptr && scratch_here()) a.flags[i] |= 4u;
    }
    if (!any) return;
    a.n_tok = n_tok; a.lo_mask = g_mask_lo; a.edge_mask = g_mask_edge; a.cost_lo = g_mask_cost_lo; a.ctrl = g_mask_ctrl_dev;
    a.fctrl = g_frag_ctrl_dev; a.mstats = g_mask_stats; a.cfg = cfg; a.eskip = g_pred_skip; a.tskip = g_pred_tail; a.tail_all = tail_all;
    route_pred_sel_k<<<dim3((unsigned) n_tok, kRoutePredDists), 32, 0, static_cast<cudaStream_t>(stream)>>>(a);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
static uint8_t* g_ring = nullptr;
static size_t g_ring_n = 0;
void route_ring_alloc(size_t n) {
    if (g_ring != nullptr || n == 0) return;
    if (cudaMalloc((void**) &g_ring, n) != cudaSuccess) { cudaGetLastError(); g_ring = nullptr; return; }
    cudaMemset(g_ring, 0, n);
    g_ring_n = n;
    note_scratch_dev();
}
uint8_t* route_ring_ptr() { return g_ring; }
const uint8_t* route_ring_layer(int64_t layer, int64_t n_expert) { return g_ring && scratch_here() ? g_ring + (size_t) (layer * n_expert) : nullptr; }
// ---- STRATA_ROUTE_RAM_FLOOR: one byte per (layer, expert), 1 while the expert has a RAM-tier place (the CPU can compute it from RAM).  Host-mapped,
// not cudaMalloc like the ring above: the host writes it at every change of the tier and the kernel reads it at each launch (as RouteCfg does).
// Allocated once, outside graph capture; a null table is "off" in the kernel.
static uint8_t* g_cheap_host = nullptr;
static uint8_t* g_cheap_dev = nullptr;
static size_t g_cheap_n = 0;
void route_cheap_alloc(size_t n) {
    if (g_cheap_host != nullptr || n == 0) return;
    if (cudaHostAlloc((void**) &g_cheap_host, n, cudaHostAllocMapped) != cudaSuccess) { cudaGetLastError(); g_cheap_host = nullptr; return; }
    if (cudaHostGetDevicePointer((void**) &g_cheap_dev, g_cheap_host, 0) != cudaSuccess) {
        cudaGetLastError();
        cudaFreeHost(g_cheap_host);
        g_cheap_host = nullptr;
        g_cheap_dev = nullptr;
        return;
    }
    for (size_t i = 0; i < n; ++i) g_cheap_host[i] = 0;   // not zeroed by cudaHostAlloc: every expert starts outside the RAM tier
    g_cheap_n = n;
}
uint8_t* route_cheap_host() { return g_cheap_host; }
const uint8_t* route_cheap_layer(int64_t layer, int64_t n_expert) {
    if (g_cheap_dev == nullptr || layer < 0 || n_expert <= 0 || (size_t) (layer + 1) * (size_t) n_expert > g_cheap_n) return nullptr;
    return g_cheap_dev + (size_t) (layer * n_expert);
}
static float* g_repl = nullptr;
static size_t g_repl_n = 0;
void route_repl_alloc(size_t n) {
    if (g_repl != nullptr || n == 0) return;
    if (cudaMalloc((void**) &g_repl, n * sizeof(float)) != cudaSuccess) { g_repl = nullptr; return; }
    cudaMemset(g_repl, 0, n * sizeof(float));
    g_repl_n = n;
    note_scratch_dev();
}
float* route_repl_layer(int64_t layer, int64_t n_expert) { return g_repl && scratch_here() ? g_repl + (size_t) (layer * n_expert) : nullptr; }
bool route_repl_active() { return g_repl != nullptr; }
void route_repl_drain(float* usage2, size_t n) {
    if (g_repl == nullptr || usage2 == nullptr || n < g_repl_n) return;
    std::vector<float> h(g_repl_n);
    if (cudaMemcpy(h.data(), g_repl, g_repl_n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) return;
    cudaMemset(g_repl, 0, g_repl_n * sizeof(float));
    for (size_t i = 0; i < g_repl_n; ++i) usage2[i] += h[i];
}
void route_detail_alloc() {
    if (g_detail != nullptr) return;
    if (cudaMalloc((void**) &g_detail, 48 * sizeof(unsigned long long)) != cudaSuccess) { g_detail = nullptr; return; }
    cudaMemset(g_detail, 0, 48 * sizeof(unsigned long long));
    note_scratch_dev();
}
void route_detail_read(unsigned long long* out16) {
    if (g_detail == nullptr) { for (int i = 0; i < 48; ++i) out16[i] = 0; return; }
    cudaMemcpy(out16, g_detail, 48 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
}

// ---- the router's wanted-expert counter (step 3 of SPEC.md 13.20): n_layers * n_expert floats on the device, filled by
// route_resident_k, drained into the host usage table before each swap round.
static float* g_want = nullptr;
static size_t g_want_n = 0;
void route_want_alloc(size_t n) {
    if (g_want != nullptr || n == 0) return;
    if (cudaMalloc((void**) &g_want, n * sizeof(float)) != cudaSuccess) { g_want = nullptr; return; }
    cudaMemset(g_want, 0, n * sizeof(float));
    g_want_n = n;
    note_scratch_dev();
}
float* route_want_layer(int64_t layer, int64_t n_expert) { return g_want && scratch_here() ? g_want + (size_t) (layer * n_expert) : nullptr; }
bool route_want_active() { return g_want != nullptr; }
// A pick counted between the copy and the clear is lost: the counts are approximate by nature.
void route_want_drain(float* usage, size_t n) {
    if (g_want == nullptr || usage == nullptr || n < g_want_n) return;
    std::vector<float> h(g_want_n);
    if (cudaMemcpy(h.data(), g_want, g_want_n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) return;
    cudaMemset(g_want, 0, g_want_n * sizeof(float));
    for (size_t i = 0; i < g_want_n; ++i) usage[i] += h[i];
}
// ---- SPEC 13.40 (T5): the router's original top-10 per window row and layer ([layer][8 rows][10]), for counting only the rows whose input
// token was kept.  The kernel writes it every window (rows beyond the window keep older values: the host reads only the kept rows).
static int32_t* g_rows = nullptr;
static int64_t g_rows_layers = 0;
void route_rows_alloc(int64_t n_layers) {
    if (g_rows != nullptr || n_layers <= 0) return;
    if (cudaMalloc((void**) &g_rows, (size_t) n_layers * 8 * 10 * sizeof(int32_t)) != cudaSuccess) { g_rows = nullptr; return; }
    cudaMemset(g_rows, 0xff, (size_t) n_layers * 8 * 10 * sizeof(int32_t));   // -1 everywhere
    g_rows_layers = n_layers;
    note_scratch_dev();
}
int32_t* route_rows_layer(int64_t layer) { return g_rows && scratch_here() ? g_rows + (size_t) layer * 80 : nullptr; }
bool route_rows_active() { return g_rows != nullptr; }
void route_rows_apply(int kept, float* usage, int64_t n_layers, int64_t n_expert) {
    if (g_rows == nullptr || usage == nullptr || kept < 1 || n_layers > g_rows_layers) return;
    if (kept > 8) kept = 8;
    std::vector<int32_t> h((size_t) n_layers * 80);
    if (cudaMemcpy(h.data(), g_rows, h.size() * sizeof(int32_t), cudaMemcpyDeviceToHost) != cudaSuccess) return;
    for (int64_t l = 0; l < n_layers; ++l)
        for (int t = 0; t < kept; ++t)
            for (int r = 0; r < 10; ++r) {
                const int32_t e = h[(size_t) l * 80 + (size_t) t * 10 + (size_t) r];
                if (e >= 0 && e < n_expert) usage[(size_t) (l * n_expert + e)] += (g_cfg_host == nullptr || r < g_cfg_host->lo) ? 1.0f : g_cfg_host->want_lo;
            }
}
// STRATA_LA_LEARN=1 (opt-in): the look-ahead's learned correction.  For each source layer s and dist d (kLaDist) the read-ahead's guess
// of layer s+d's experts is that layer's router applied to x_s (the plain router; STRATA_LA_MODEL: its published matrix for d=1)
// plus a learned low-rank correction B (A x_s).  Four slots per (s,d): slot 0 is the plain router, slots 1-3 corrections of rank
// 4, 8 and 16, all trained on the same inputs and labels (AdamW; centred MSE on the true top 32 plus a hinge below their 32nd logit).
// Within a window: the forward (the slots' logits) runs in lookahead_node on cs, where the look-ahead already runs; each layer's
// router logits (the label of the sources behind it) are copied into a per-window buffer on cs.  The training is ONE batch per window,
// on a side stream (la_learn_post_launch): rank, score, prep, grad, the B and A updates, the window's decision (leader, freeze, reset)
// and the resets, one launch per kernel type over every (source, dist, slot).  The batch waits for the window's graph (an event recorded
// after its launch); the next window's launch waits for the batch's done event (la_learn_pre_launch, on the host side of the launch);
// the window's own completion never waits for the batch.  A frozen (layer, dist) trains its leader every 4th window and no other slot.
// Everything the graph and the batch use is device memory (leader, live flags, counters, tick), so nothing per window is baked.
constexpr int kLaSlots = 4;
constexpr int kLaRows = 28;      // the window's projection rows: slot 1 rows 0-3, slot 2 rows 4-11, slot 3 rows 12-27
constexpr int kLaE = 512;
constexpr int kLaTok = 8;
// the look-ahead's distances: the one table.  A distance is one entry here and needs its route_pred_slot (native_router.hpp: 1, 2, 4
// are slots 0, 1, 2 of the table); the (s,d) lists below grow with kLaNd.  Everything below takes its count from here.
constexpr int kLaDist[] = {1, 2, 4};
constexpr int kLaNd = (int) (sizeof kLaDist / sizeof kLaDist[0]);
constexpr int kLaNSlots = kLaSlots - 1;   // the trained slots per (s,d): ranks 4, 8, 16
constexpr int kLaKernels = 8;             // the batch's kernels, in launch order: rank, score, prep, grad, upd_b, upd_a, decide, reinit
constexpr int kLaCnt = 18;       // counters per (s,d): [0] picks since change, [1] picks in the recent 500, [2] picks all,
                                 // [3+c] hits since change, [7+c] hits recent, [11+c] hits all, [15] changes, [16] resets,
                                 // [17] recent blocks in a row where the plain router beat the leader
constexpr int kLaMinPicks = 200, kLaFreezePicks = 2000, kLaRecPicks = 500;
constexpr float kLaLr = 1e-3f, kLaB1 = 0.9f, kLaB2 = 0.999f, kLaEps = 1e-8f, kLaWd = 1e-3f;

static int la_dist_index(int dist) {
    for (int i = 0; i < kLaNd; ++i)
        if (kLaDist[i] == dist) return i;
    return -1;
}
__host__ __device__ inline int la_rank(int c) { return c == 1 ? 4 : c == 2 ? 8 : c == 3 ? 16 : 0; }
__host__ __device__ inline int la_row_off(int c) { return c == 1 ? 0 : c == 2 ? 4 : c == 3 ? 12 : 0; }
// uniform in [-sqrt(3)e-3, sqrt(3)e-3]: standard deviation 1e-3 (the initial draws and the reset draws)
__host__ __device__ inline float la_rand_pm(unsigned long long key) {
    key ^= key >> 33; key *= 0xff51afd7ed558ccdULL; key ^= key >> 33; key *= 0xc4ceb9fe1a85ec53ULL; key ^= key >> 33;
    return (2.0f * (float) (key >> 40) * (1.0f / 16777216.0f) - 1.0f) * 1.7320508e-3f;
}
__host__ __device__ inline unsigned long long la_key(int sd, int c, long long i, unsigned long long nre) {
    return (unsigned long long) (sd * kLaSlots + c) * 0x9E3779B97F4A7C15ULL + (unsigned long long) i * 0xD1B54A32D192ED03ULL +
           nre * 0x2545F4914F6CDD1DULL + 1ULL;
}
__device__ __forceinline__ void la_adam(float& p, float& m, float& v, float g, int t) {
    m = kLaB1 * m + (1.0f - kLaB1) * g;
    v = kLaB2 * v + (1.0f - kLaB2) * g * g;
    const float c1 = 1.0f - powf(kLaB1, (float) t), c2 = 1.0f - powf(kLaB2, (float) t);
    p -= kLaLr * ((m / c1) / (sqrtf(v / c2) + kLaEps) + kLaWd * p);   // AdamW: the decay is decoupled from the gradient
}
// the sum over the block's threads, returned to all of them; sh: 32 floats
__device__ __forceinline__ float la_block_sum(float v, float* sh) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int w = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) sh[w] = v;
    __syncthreads();
    if (w == 0) {
        float s = threadIdx.x < (blockDim.x >> 5) ? sh[threadIdx.x] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (threadIdx.x == 0) sh[0] = s;
    }
    __syncthreads();
    const float r = sh[0];
    __syncthreads();   // every thread has read sh[0] before a following call writes sh[w] (24 Oct review: a race on the second sum)
    return r;
}
__device__ __forceinline__ unsigned long long la_now() {   // the GPU's clock (ns), for the timing only
    unsigned long long now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    return now;
}

// forward (per source s and dist d, in lookahead_node after the router GEMV and its bias): u = A x for the 28 rows (a block per row,
// all n tokens), then each slot's logits = plain + B u (a thread per token and expert); the leader's go to route_pred_logits
struct LaFwdArgs {
    const float* x;              // the source's MoE input, n x H
    float* xsave;                // the first dist: the window's copy of x for the training batch (else nullptr)
    float* u;                    // n x 28
    float* pred;                 // 4 x 8 x 512: the slots' logits (slot 0: the plain router)
    float* route;                // route_pred_logits: the plain logits on entry, the leader's on exit
    const float* A[kLaSlots];
    const float* B[kLaSlots];
    const int* lead;
    int n, H;
};
__global__ void la_fwd_u_k(LaFwdArgs a) {   // grid 28, block 256
    const int k = blockIdx.x, c = k < 4 ? 1 : (k < 12 ? 2 : 3);
    const float* row = a.A[c] + (size_t) (k - la_row_off(c)) * a.H;
    __shared__ float sh[32];
    if (a.xsave != nullptr && k == 0)
        for (int i = threadIdx.x; i < a.n * a.H; i += blockDim.x) a.xsave[i] = a.x[i];
    for (int t = 0; t < a.n; ++t) {
        float s = 0.0f;
        for (int h = threadIdx.x; h < a.H; h += blockDim.x) s += row[h] * a.x[(size_t) t * a.H + h];
        s = la_block_sum(s, sh);
        if (threadIdx.x == 0) a.u[t * kLaRows + k] = s;
    }
}
__global__ void la_fwd_pred_k(LaFwdArgs a) {   // grid ceil(n * 512 / 256), block 256
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= a.n * kLaE) return;
    const int t = i / kLaE, e = i % kLaE;
    const int lead = *a.lead;
    const float p0 = a.route[i];
    float sel = p0;
    a.pred[i] = p0;
    for (int c = 1; c < kLaSlots; ++c) {
        const int r = la_rank(c);
        const float* b = a.B[c] + (size_t) e * r;
        const float* uu = a.u + t * kLaRows + la_row_off(c);
        float v = p0;
        for (int q = 0; q < r; ++q) v += b[q] * uu[q];
        a.pred[c * kLaTok * kLaE + i] = v;
        if (lead == c) sel = v;
    }
    a.route[i] = sel;
}

// the batch (one window, after its graph): every (source s, dist d, slot c) at once.  sd = s * kLaNd + dd is the (s,d) index, its target
// layer is s + kLaDist[dd] (skipped when past the last layer), and z = sd * kLaNSlots + (c - 1) the trained set.
struct LaBatch {
    float* A[kLaSlots]; float* B[kLaSlots]; float* mA[kLaSlots]; float* vA[kLaSlots]; float* mB[kLaSlots]; float* vB[kLaSlots];
    const float* x;              // [L][8][H]: each source's MoE input (this window)
    const float* u;              // [Nsd][8][28]
    const float* pred;           // [Nsd][4][8][512]: the slots' logits from the source's forward (this window)
    const float* lab;            // [L][8][512]: each layer's router logits (this window)
    const int* res;              // [L][512]: the card-resident flags (>= 0: resident)
    int* rk;                     // [L][8][512]: the true rank of each expert
    float* thr;                  // [L][8]: the true 32nd logit
    float* g;                    // [Nset][8][512]: d loss / d logit, batch-averaged
    float* du;                   // [Nset][8][16]: g B for the set's rows
    int* act;                    // [Nset]: 1 when the set trains this window
    int* tstep;                  // [Nset]: this window's Adam step
    int* steps;                  // [Nsd][4]
    int* live;                   // [Nsd][4]
    int* lead;                   // [Nsd]
    int* frozen;                 // [Nsd]
    int* reset;                  // [Nsd]
    int* tick;                   // [Nsd]: windows seen by the (s,d)
    unsigned long long* cnt;     // [Nsd][kLaCnt]
    unsigned long long* st;      // [3]: the batch's start, the window's end (timing)
    unsigned long long* tot_ns;  // timing: the batch's time on the side stream
    unsigned long long* wait_ns; // timing: how far the batch ran past the window's end (the next window's wait bound)
    unsigned long long* nwin;    // the windows whose batch ran
    unsigned long long* rinfo;   // [Nsd][4]: the last reset's recent picks, plain hits, old leader's hits, old leader's slot
    unsigned long long* kst;     // timing: the per-kernel profile (kst: starts, kacc: sums), [kLaKernels]
    unsigned long long* kacc;
    la_cadence::State* cad;      // [Nsd]: the cadence state (SPEC-adaptive-la change 2; read and written when cad_on)
    unsigned long long* ntrain;  // the batches where some set trained (the stats line's N)
    int* gate_min;               // host-mapped: the next window with a trainer over all sets (decide's output when cad_on)
    int* gate_done;              // host-mapped: the window of the batch whose decide wrote gate_min (written after it)
    int cad_on;                  // STRATA_LA_CADENCE (default 1): the due-based leader cadence; 0 = the legacy every-4th-batch rule
    int win;                     // this batch's window index (every window counts, batched or not)
    int dist[kLaNd];
    int L, H, lo, n, Nd, Nsd, Nset, timing;
    float exact_w;               // STRATA_LA_EXACT_W: the weight of a true pick below the stand-in floor in the loss (1: as the rest)
    const uint8_t* mask;         // [L][512]: the route mask's base (route_mask_layer(0, 512)); nullptr: no mask, or another GPU's
};
// the true rank of each expert (ties by index) and the true 32nd logit: grid (n, L), a thread per expert
__global__ void la_b_rank_k(LaBatch a) {
    __shared__ float s[kLaE];
    const int t = blockIdx.x, m = blockIdx.y, e = threadIdx.x;
    if (a.timing && t == 0 && m == 0 && e == 0) a.st[0] = la_now();
    const size_t row = (size_t) m * kLaTok + t;
    const float v = a.lab[row * kLaE + e];
    s[e] = v;
    __syncthreads();
    int c = 0;
    for (int f = 0; f < kLaE; ++f) c += (s[f] > v) || (s[f] == v && f < e);
    a.rk[row * kLaE + e] = c;
    if (c == 31) a.thr[row] = v;
}
// the prequential score: grid (n, Nsd, 4), a thread per expert that is a true pick below the floor and not resident; a hit when fewer
// than 2 * lo non-resident predictions of the slot are above it.  A non-live candidate is not scored.
__global__ void la_b_score_k(LaBatch a) {
    const int t = blockIdx.x, sd = blockIdx.y, c = blockIdx.z, e = threadIdx.x;
    const int s = sd / a.Nd, m = s + a.dist[sd % a.Nd];
    if (m >= a.L) return;
    if (c > 0 && !a.live[sd * kLaSlots + c]) return;
    const size_t row = (size_t) m * kLaTok + t;
    if (a.rk[row * kLaE + e] >= a.lo || a.res[(size_t) m * kLaE + e] >= 0) return;
    const float* p = a.pred + ((size_t) sd * kLaSlots + c) * kLaTok * kLaE + (size_t) t * kLaE;
    const float pe = p[e];
    int above = 0;
    for (int f = 0; f < kLaE; ++f) above += (a.res[(size_t) m * kLaE + f] < 0) && (p[f] > pe || (p[f] == pe && f < e));
    unsigned long long* cc = a.cnt + (size_t) sd * kLaCnt;
    if (c == 0) {
        atomicAdd(cc + 0, 1ull);
        atomicAdd(cc + 1, 1ull);
        atomicAdd(cc + 2, 1ull);
    }
    if (above < 2 * a.lo) {
        atomicAdd(cc + 3 + c, 1ull);
        atomicAdd(cc + 7 + c, 1ull);
        atomicAdd(cc + 11 + c, 1ull);
    }
}
// which sets train this window: the (s,d) has a target layer; a live candidate trains; a frozen (s,d) trains its leader only every 4th window
__global__ void la_b_prep_k(LaBatch a) {   // grid ceil(Nset / 64), block 64
    const int z = blockIdx.x * blockDim.x + threadIdx.x;
    if (z >= a.Nset) return;
    const int sd = z / kLaNSlots, c = z % kLaNSlots + 1;
    const int s = sd / a.Nd;
    int ok = s + a.dist[sd % a.Nd] < a.L && a.live[sd * kLaSlots + c];
    if (a.frozen[sd]) {   // the frozen leader alone, in its cadence's windows (STRATA_LA_CADENCE=0: every 4th batch)
        const bool due = a.cad_on ? la_cadence::trains_at(a.cad[sd].due, a.win) : la_cadence::legacy_trains(a.tick[sd]);
        if (a.lead[sd] != c || !due) ok = 0;
    }
    a.act[z] = ok;
    if (ok) a.tstep[z] = a.steps[sd * kLaSlots + c] + 1;
}
// the gradient: block per (token, set), thread per expert.  Loss per token: the centred MSE over the true top 32 (the centre is the mean
// over those 32; its gradient is 2/32 (pc - tc)) plus the hinge mean over the other 480 (gradient 1/480 where the logit is above thr - 0.1).
// du = g B for the set's rows (before B changes).
__global__ void la_b_grad_k(LaBatch a) {   // grid (n, Nset), block 512
    const int t = blockIdx.x, z = blockIdx.y, e = threadIdx.x;
    if (!a.act[z]) return;
    const int sd = z / kLaNSlots, c = z % kLaNSlots + 1, r = la_rank(c);
    const int s = sd / a.Nd, m = s + a.dist[sd % a.Nd];
    __shared__ float sh[32];
    __shared__ float P[16][kLaE];
    const size_t row = (size_t) m * kLaTok + t;
    const float p = a.pred[((size_t) sd * kLaSlots + c) * kLaTok * kLaE + (size_t) t * kLaE + e];
    const float tv = a.lab[row * kLaE + e];
    const bool top = a.rk[row * kLaE + e] < 32;
    const float mP = la_block_sum(top ? p : 0.0f, sh) / 32.0f;
    const float mT = la_block_sum(top ? tv : 0.0f, sh) / 32.0f;
    float g = top ? (2.0f / 32.0f) * ((p - mP) - (tv - mT)) : (p > a.thr[row] - 0.1f ? 1.0f / 480.0f : 0.0f);
    // STRATA_LA_EXACT_W (a.exact_w, 1: off): the true picks below the stand-in floor (ranks 0 .. lo-1, the ones a miss reads from RAM or
    // the SSD) weigh w times as much as the rest of the top 32 - the placement solver's rank weighting, in the learner's loss
    if (top && a.rk[row * kLaE + e] < a.lo) g *= a.exact_w;
    // STRATA_MASK=learn: an expert at level 2 (masked at every rank, never routed or read) has no weight in the loss.  m is the TARGET layer
    // (the layer whose label row this is); level 1 stays in the loss (exact below its rank lo-1)
    if (a.mask != nullptr && a.mask[(size_t) m * kLaE + e] == 2) g = 0.0f;
    g /= (float) a.n;
    a.g[((size_t) z * kLaTok + t) * kLaE + e] = g;
    for (int k = 0; k < r; ++k) P[k][e] = g * a.B[c][(size_t) sd * kLaE * r + e * r + k];
    __syncthreads();
    const int w = e >> 5;
    if (w < r) {
        float sum = 0.0f;
        for (int i = e & 31; i < kLaE; i += 32) sum += P[w][i];
        for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
        if ((e & 31) == 0) a.du[((size_t) z * kLaTok + t) * 16 + w] = sum;
    }
}
__global__ void la_b_upd_b_k(LaBatch a) {   // grid (32, Nset), block 256: 512 x 16 entries per set
    const int z = blockIdx.y;
    if (!a.act[z]) return;
    const int sd = z / kLaNSlots, c = z % kLaNSlots + 1, r = la_rank(c), off = la_row_off(c);
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int e = idx / 16, k = idx % 16;
    if (e >= kLaE || k >= r) return;
    float g = 0.0f;
    for (int t = 0; t < a.n; ++t) g += a.g[((size_t) z * kLaTok + t) * kLaE + e] * a.u[((size_t) sd * kLaTok + t) * kLaRows + off + k];
    const size_t i = (size_t) sd * kLaE * r + (size_t) e * r + k;
    la_adam(a.B[c][i], a.mB[c][i], a.vB[c][i], g, a.tstep[z]);
}
// the A update, tiled: a thread per column h updates all r rows of the set, so each x element is read once per window (not once per row);
// du for the set sits in shared memory.  grid (ceil(H / 256), Nset), block 256.
__global__ void la_b_upd_a_k(LaBatch a) {
    const int z = blockIdx.y;
    if (!a.act[z]) return;
    const int sd = z / kLaNSlots, c = z % kLaNSlots + 1, r = la_rank(c);
    const int s = sd / a.Nd;
    __shared__ float du_s[kLaTok][16];
    for (int i = threadIdx.x; i < kLaTok * 16; i += blockDim.x) du_s[i / 16][i % 16] = a.du[(size_t) z * kLaTok * 16 + i];
    __syncthreads();
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h < a.H) {
        float xt[kLaTok];
        for (int t = 0; t < a.n; ++t) xt[t] = a.x[((size_t) s * kLaTok + t) * a.H + h];
        for (int k = 0; k < r; ++k) {
            float g = 0.0f;
            for (int t = 0; t < a.n; ++t) g += du_s[t][k] * xt[t];
            const size_t i = ((size_t) sd * r + k) * a.H + h;
            la_adam(a.A[c][i], a.mA[c][i], a.vA[c][i], g, a.tstep[z]);
        }
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) a.steps[sd * kLaSlots + c] = a.tstep[z];
}
// the batch's per-kernel profile (STRATA_LA_LEARN_TIMING=1): kst[k] is the start of kernel k on the side stream, kacc[k] the sum of its spans
__global__ void la_b_mark_k(unsigned long long* kst, unsigned long long* kacc, int k, int end) {
    const unsigned long long now = la_now();
    if (!end) kst[k] = now;
    else kacc[k] += now - kst[k];
}

// the window's decision, one (s,d) at a time (la_b_decide_k): the leader (a candidate beats the leader by a Newcombe Wilson 90% interval
// on >= 200 picks since the last change), the freeze (2000 picks without a change), the reset (the plain router beats a leader over the
// recent 500), and (cad_on) the cadence of the frozen leader (la_cadence.hpp).  mn: the set's next training window, any: a set trained
__device__ __forceinline__ void la_decide_one(const LaBatch& a, int sd, int w, int& mn, int& any) {
    unsigned long long* c = a.cnt + (size_t) sd * kLaCnt;
    a.reset[sd] = 0;
    a.tick[sd] += 1;
    const int s = sd / a.Nd, dd = sd % a.Nd;
    const bool was_frozen = a.frozen[sd] != 0;
    int L = a.lead[sd];
    const bool tgt = s + a.dist[dd] < a.L;
    bool trained_leader = false;
    if (a.cad_on) {
        // what this window's training trains, from the state before this window's decision (the training kernels may be skipped
        // when nothing is due, so the act flags cannot be read): the same test as la_b_prep_k
        if (!was_frozen) {
            for (int k = 1; k < kLaSlots; ++k) any |= (tgt && a.live[sd * kLaSlots + k] != 0);
        } else {
            trained_leader = L != 0 && tgt && a.live[sd * kLaSlots + L] != 0 && la_cadence::trains_at(a.cad[sd].due, w);
            any |= trained_leader;
        }
    } else {
        for (int k = 0; k < kLaNSlots; ++k) any |= a.act[(size_t) sd * kLaNSlots + k] != 0;
        trained_leader = L != 0 && a.act[(size_t) sd * kLaNSlots + L - 1] != 0;
    }
    const float nS = (float) c[0];
    if (!a.frozen[sd] && nS >= kLaMinPicks) {
        int best = -1;
        float bestp = -1.0f;
        const float pL = (float) c[3 + L] / nS;
        for (int k = 0; k < kLaSlots; ++k) {
            if (k == L) continue;
            const float p = (float) c[3 + k] / nS;
            if (p > bestp && la_cadence::diff_lo(p, nS, pL, nS) > 0.0f) { best = k; bestp = p; }
        }
        if (best >= 0) {
            L = best;
            a.lead[sd] = L;
            c[15] += 1;
            c[0] = 0;
            for (int k = 0; k < kLaSlots; ++k) c[3 + k] = 0;
            for (int k = 1; k < kLaSlots; ++k) a.live[sd * kLaSlots + k] = 1;
        }
    }
    if (!a.frozen[sd] && c[0] >= kLaFreezePicks) {   // 2000 picks without a change: the non-leader candidates stop training
        a.frozen[sd] = 1;
        for (int k = 1; k < kLaSlots; ++k)
            if (k != L) a.live[sd * kLaSlots + k] = 0;
    }
    const float nR = (float) c[1];
    if (L != 0 && nR >= kLaRecPicks) {   // the plain router beats the leader over the recent picks: back to the plain router
        const float p0 = (float) c[7] / nR, pL = (float) c[7 + L] / nR;
        // two failing blocks in a row (25 Oct): one 90% test per 500 picks over 144 (layer, head) pairs fired ~1 in 20 by chance -
        // 35 resets in a 62K-token run while the leaders kept winning - each one throwing a good correction away
        c[17] = la_cadence::diff_lo(p0, nR, pL, nR) > 0.0f ? c[17] + 1 : 0;
        if (c[17] >= 2) {
            c[17] = 0;
            a.lead[sd] = 0;
            a.frozen[sd] = 0;
            a.reset[sd] = 1;
            c[16] += 1;
            // why it fired (the host logs the last one): the recent picks, the plain router's and the old leader's hits, the old leader's slot
            a.rinfo[sd * kLaSlots + 0] = (unsigned long long) nR;
            a.rinfo[sd * kLaSlots + 1] = c[7];
            a.rinfo[sd * kLaSlots + 2] = c[7 + L];
            a.rinfo[sd * kLaSlots + 3] = (unsigned long long) L;
            c[0] = 0;
            for (int k = 0; k < kLaSlots; ++k) c[3 + k] = 0;
            for (int k = 1; k < kLaSlots; ++k) {
                a.live[sd * kLaSlots + k] = 1;
                a.steps[sd * kLaSlots + k] = 0;
            }
        }
    }
    if (a.cad_on) {   // the cadence reads the recent block before its reset below
        la_cadence::State& st = a.cad[sd];
        if (!a.frozen[sd]) {
            la_cadence::set_live(st, w);
        } else if (!was_frozen) {
            la_cadence::set_frozen(st, w, c[0]);
            if (L == 0) st.due = la_cadence::kNever;   // a plain leader while frozen: no set trains
        } else if (L != 0) {
            la_cadence::after_window(st, w, trained_leader, c[0], c[3 + L], c[1], c[7 + L]);
        }
        const int next = !tgt ? la_cadence::kNever : (a.frozen[sd] ? st.due : w + 1);
        mn = min(mn, next);
    }
    if (nR >= kLaRecPicks) {
        c[1] = 0;
        for (int k = 0; k < kLaSlots; ++k) c[7 + k] = 0;
    }
}
// one block of 256 threads: every (s,d) in turn per thread, then the block's reductions: the next training window over all sets (the
// host's gate, written after it: the fence orders the two words for the host) and whether any set trained (the stats line's N)
__global__ void la_b_decide_k(LaBatch a) {
    __shared__ int s_min[256];
    __shared__ int s_any[256];
    if (threadIdx.x == 0) atomicAdd(a.nwin, 1ull);
    int mn = la_cadence::kNever, any = 0;
    for (int sd = threadIdx.x; sd < a.Nsd; sd += blockDim.x) la_decide_one(a, sd, a.win, mn, any);
    s_min[threadIdx.x] = mn;
    s_any[threadIdx.x] = any;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if ((int) threadIdx.x < o) {
            s_min[threadIdx.x] = min(s_min[threadIdx.x], s_min[threadIdx.x + o]);
            s_any[threadIdx.x] |= s_any[threadIdx.x + o];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (s_any[0]) atomicAdd(a.ntrain, 1ull);
        if (a.cad_on) {
            *a.gate_min = s_min[0];
            __threadfence_system();
            *a.gate_done = a.win;
        }
    }
}
__global__ void la_b_reinit_k(LaBatch a) {   // grid (160, Nsd), block 256: a reset slot starts fresh (A random, B and moments zero)
    const int sd = blockIdx.y;
    if (!a.reset[sd]) return;
    const unsigned long long nre = a.cnt[(size_t) sd * kLaCnt + 16];
    const int stride = gridDim.x * blockDim.x;
    for (int c = 1; c < kLaSlots; ++c) {
        const int r = la_rank(c);
        const size_t oa = (size_t) sd * r * a.H, ob = (size_t) sd * kLaE * r;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < r * a.H; i += stride) {
            a.A[c][oa + i] = la_rand_pm(la_key(sd, c, i, nre));
            a.mA[c][oa + i] = 0.0f;
            a.vA[c][oa + i] = 0.0f;
        }
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < kLaE * r; i += stride) {
            a.B[c][ob + i] = 0.0f;
            a.mB[c][ob + i] = 0.0f;
            a.vB[c][ob + i] = 0.0f;
        }
    }
}
// timing (STRATA_LA_LEARN_TIMING=1): the window's end stamp (on cs, after the last layer) and the batch's end (one thread, the last kernel)
__global__ void la_b_wend_k(unsigned long long* st) { st[1] = la_now(); }
__global__ void la_b_end_k(LaBatch a) {
    const unsigned long long now = la_now();
    a.tot_ns[0] += now - a.st[0];
    if (now > a.st[1]) a.wait_ns[0] += now - a.st[1];
}

struct LaLearn {
    int L = 0, H = 0, lo = 6, Nd = kLaNd, Nsd = 0, Nset = 0;
    int dist[kLaNd] = {};        // the instance's distances (kLaDist for the main learner; {1} for the drafter's)
    bool timing = false;
    bool pending = false;        // a batch was launched and its done event is not yet waited on by a launch
    cudaStream_t side = nullptr; // the batch's stream (non-blocking): never the stream the host waits on at a window's end
    cudaEvent_t ev_end = nullptr, ev_done = nullptr;
    float* A[kLaSlots] = {};     // slot c: [Nsd][r_c][H]
    float* B[kLaSlots] = {};     // slot c: [Nsd][512][r_c]
    float* mA[kLaSlots] = {};
    float* vA[kLaSlots] = {};
    float* mB[kLaSlots] = {};
    float* vB[kLaSlots] = {};
    float* x = nullptr;          // [L][8][H]
    float* u = nullptr;          // [Nsd][8][28]
    float* pred = nullptr;       // [Nsd][4][8][512]
    float* lab = nullptr;        // [L][8][512]
    int* rk = nullptr;           // [L][8][512]
    float* thr = nullptr;        // [L][8]
    float* g = nullptr;          // [Nset][8][512]
    float* du = nullptr;         // [Nset][8][16]
    int* act = nullptr;          // [Nset]
    int* tstep = nullptr;        // [Nset]
    int* steps = nullptr;        // [Nsd][4]
    int* tick = nullptr;         // [Nsd]
    int* live = nullptr;         // [Nsd][4]
    int* lead = nullptr;         // [Nsd]
    int* frozen = nullptr;       // [Nsd]
    int* reset = nullptr;        // [Nsd]
    unsigned long long* cnt = nullptr;   // [Nsd][kLaCnt]
    unsigned long long* st = nullptr;    // [3]
    unsigned long long* tot_ns = nullptr;
    unsigned long long* wait_ns = nullptr;
    unsigned long long* nwin = nullptr;
    unsigned long long* rinfo = nullptr;   // [Nsd][4]: the last reset's facts (LaBatch::rinfo)
    unsigned long long* kst = nullptr;     // [kLaKernels]: the per-kernel profile (timing)
    unsigned long long* kacc = nullptr;    // [kLaKernels]
    la_cadence::State* cad = nullptr;      // [Nsd]: the cadence state (device)
    unsigned long long* ntrain = nullptr;  // the batches where some set trained (device)
    int* gate_min_h = nullptr;             // host-mapped [2]: [0] the next window with a trainer (decide), [1] the window of that decide
    int* gate_min_d = nullptr;             // the device view of gate_min_h (its words: [0] gate_min, [1] gate_done)
    int win = 0;                           // windows seen (batched or skipped): the index of the next window
    int last_launch = -1;                  // the window of the last batch launched (-1: none)
    bool cad_on = true;                    // STRATA_LA_CADENCE (default on): the due-based cadence and the batch gate; 0 = as before
    unsigned long long skipped = 0;        // windows whose batch was not launched (no set trained in them)
    // host: the real main-stream waits (the batch not done when a window's launch came): windows that found it pending, and the timed ones
    unsigned long long win_pending = 0, win_waited = 0;
    double wait_ms_total = 0.0;   // the timed waits so far (ms)
    std::vector<std::pair<cudaEvent_t, cudaEvent_t>> wait_ev;   // timing: before/after the wait on cs, read at the request's end
    std::vector<unsigned long long> resets_seen;                  // host: per (s,d), the resets already logged
    std::vector<char> logged;    // host: the (s,d) whose chosen r was logged
    size_t bytes = 0;
};
static LaLearn* g_la = nullptr;

// one learner instance: L layers, H wide, Nd distances per layer (the caller sets the dists), lo the stand-in floor
static LaLearn* la_new(int L, int H, int lo, int Nd, bool timing, std::string& msg) {
    LaLearn* s = new LaLearn();
    s->L = L;
    s->H = H;
    s->lo = lo;
    s->timing = timing;
    s->Nd = Nd;
    s->Nsd = Nd * s->L;
    s->Nset = s->Nsd * kLaNSlots;
    bool ok = true;
    auto dalloc = [&](void** p, size_t bytes) {
        *p = nullptr;
        if (!ok) return;
        if (cudaMalloc(p, bytes) != cudaSuccess) { (void) cudaGetLastError(); ok = false; return; }
        cudaMemset(*p, 0, bytes);
        s->bytes += bytes;
    };
    for (int c = 1; c < kLaSlots; ++c) {
        const size_t ra = (size_t) s->Nsd * la_rank(c) * s->H * sizeof(float), rb = (size_t) s->Nsd * kLaE * la_rank(c) * sizeof(float);
        dalloc((void**) &s->A[c], ra);
        dalloc((void**) &s->B[c], rb);
        dalloc((void**) &s->mA[c], ra);
        dalloc((void**) &s->vA[c], ra);
        dalloc((void**) &s->mB[c], rb);
        dalloc((void**) &s->vB[c], rb);
    }
    dalloc((void**) &s->x, (size_t) s->L * kLaTok * s->H * sizeof(float));
    dalloc((void**) &s->u, (size_t) s->Nsd * kLaTok * kLaRows * sizeof(float));
    dalloc((void**) &s->pred, (size_t) s->Nsd * kLaSlots * kLaTok * kLaE * sizeof(float));
    dalloc((void**) &s->lab, (size_t) s->L * kLaTok * kLaE * sizeof(float));
    dalloc((void**) &s->rk, (size_t) s->L * kLaTok * kLaE * sizeof(int));
    dalloc((void**) &s->thr, (size_t) s->L * kLaTok * sizeof(float));
    dalloc((void**) &s->g, (size_t) s->Nset * kLaTok * kLaE * sizeof(float));
    dalloc((void**) &s->du, (size_t) s->Nset * kLaTok * 16 * sizeof(float));
    dalloc((void**) &s->act, (size_t) s->Nset * sizeof(int));
    dalloc((void**) &s->tstep, (size_t) s->Nset * sizeof(int));
    dalloc((void**) &s->steps, (size_t) s->Nsd * kLaSlots * sizeof(int));
    dalloc((void**) &s->tick, (size_t) s->Nsd * sizeof(int));
    dalloc((void**) &s->live, (size_t) s->Nsd * kLaSlots * sizeof(int));
    dalloc((void**) &s->lead, (size_t) s->Nsd * sizeof(int));
    dalloc((void**) &s->frozen, (size_t) s->Nsd * sizeof(int));
    dalloc((void**) &s->reset, (size_t) s->Nsd * sizeof(int));
    dalloc((void**) &s->cnt, (size_t) s->Nsd * kLaCnt * sizeof(unsigned long long));
    dalloc((void**) &s->st, 3 * sizeof(unsigned long long));
    dalloc((void**) &s->tot_ns, sizeof(unsigned long long));
    dalloc((void**) &s->wait_ns, sizeof(unsigned long long));
    dalloc((void**) &s->nwin, sizeof(unsigned long long));
    dalloc((void**) &s->rinfo, (size_t) s->Nsd * kLaSlots * sizeof(unsigned long long));
    dalloc((void**) &s->kst, kLaKernels * sizeof(unsigned long long));
    dalloc((void**) &s->kacc, kLaKernels * sizeof(unsigned long long));
    dalloc((void**) &s->cad, (size_t) s->Nsd * sizeof(la_cadence::State));
    dalloc((void**) &s->ntrain, sizeof(unsigned long long));
    if (ok && (cudaHostAlloc((void**) &s->gate_min_h, 2 * sizeof(int), cudaHostAllocMapped) != cudaSuccess ||
               cudaHostGetDevicePointer((void**) &s->gate_min_d, s->gate_min_h, 0) != cudaSuccess)) {
        (void) cudaGetLastError();
        ok = false;
    }
    if (ok) {
        s->gate_min_h[0] = la_cadence::kNever;
        s->gate_min_h[1] = -2;   // no decide yet: the gate is not valid
    }
    {   // STRATA_LA_CADENCE=0 (opt-out): the legacy every-4th-batch rule, every window's batch launched
        static const bool cad_env = [] { const char* v = std::getenv("STRATA_LA_CADENCE"); return v == nullptr || std::atoi(v) != 0; }();
        s->cad_on = cad_env;
    }
    if (ok && (cudaStreamCreateWithFlags(&s->side, cudaStreamNonBlocking) != cudaSuccess ||
               cudaEventCreateWithFlags(&s->ev_end, cudaEventDisableTiming) != cudaSuccess ||
               cudaEventCreateWithFlags(&s->ev_done, cudaEventDisableTiming) != cudaSuccess)) {
        (void) cudaGetLastError();
        ok = false;
    }
    if (!ok) {
        msg = "no room on the GPU for it (or no stream and events)";
        for (int c = 1; c < kLaSlots; ++c) {
            cudaFree(s->A[c]); cudaFree(s->B[c]); cudaFree(s->mA[c]); cudaFree(s->vA[c]); cudaFree(s->mB[c]); cudaFree(s->vB[c]);
        }
        cudaFree(s->x); cudaFree(s->u); cudaFree(s->pred); cudaFree(s->lab); cudaFree(s->rk); cudaFree(s->thr); cudaFree(s->g);
        cudaFree(s->du); cudaFree(s->act); cudaFree(s->tstep); cudaFree(s->steps); cudaFree(s->tick); cudaFree(s->live);
        cudaFree(s->lead); cudaFree(s->frozen); cudaFree(s->reset); cudaFree(s->cnt); cudaFree(s->st); cudaFree(s->tot_ns);
        cudaFree(s->wait_ns); cudaFree(s->nwin); cudaFree(s->rinfo); cudaFree(s->kst); cudaFree(s->kacc);
        cudaFree(s->cad); cudaFree(s->ntrain);
        if (s->gate_min_h != nullptr) cudaFreeHost(s->gate_min_h);
        if (s->side != nullptr) cudaStreamDestroy(s->side);
        if (s->ev_end != nullptr) cudaEventDestroy(s->ev_end);
        if (s->ev_done != nullptr) cudaEventDestroy(s->ev_done);
        (void) cudaGetLastError();
        delete s;
        return nullptr;
    }
    // the initial draws (the reset's function, nothing learned yet): A random, B zero; slot 0 has no parameters
    for (int c = 1; c < kLaSlots; ++c) {
        const int r = la_rank(c);
        std::vector<float> hA((size_t) s->Nsd * r * s->H);
        for (int sd = 0; sd < s->Nsd; ++sd)
            for (int i = 0; i < r * s->H; ++i) hA[(size_t) sd * r * s->H + i] = la_rand_pm(la_key(sd, c, i, 0));
        cudaMemcpy(s->A[c], hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice);
    }
    std::vector<int> live((size_t) s->Nsd * kLaSlots, 1);
    cudaMemcpy(s->live, live.data(), live.size() * sizeof(int), cudaMemcpyHostToDevice);
    s->logged.assign((size_t) s->Nsd, 0);
    char buf[160];
    std::snprintf(buf, sizeof buf, "%lld layers, %zu MiB of VRAM%s", (long long) L, s->bytes >> 20, timing ? ", timing on" : "");
    msg = buf;
    return s;
}
bool la_learn_alloc(int64_t n_layers, int64_t n_embd, int lo, bool timing, std::string& msg) {
    if (g_la != nullptr) { msg = "already on"; return true; }
    if (n_layers <= 0 || n_embd <= 0) { msg = "no layers"; return false; }
    note_scratch_dev();
    if (!scratch_here()) { msg = "not on the scratch GPU"; return false; }
    LaLearn* s = la_new((int) n_layers, (int) n_embd, lo, kLaNd, timing, msg);
    if (s == nullptr) return false;
    for (int i = 0; i < kLaNd; ++i) s->dist[i] = kLaDist[i];
    g_la = s;
    return true;
}
bool la_learn_active() { return g_la != nullptr && scratch_here(); }

// the forward, on cs (in the window's graph)
void la_learn_forward(int64_t l, int dist, const float* x, int n, void* stream) {
    la_learn_forward_at(l, dist, x, n, stream, route_pred_logits());
}
// SPEC-router-fuse step 1: the same forward with its route scratch given (the dist's slot: route_pred_logits_at).
void la_learn_forward_at(int64_t l, int dist, const float* x, int n, void* stream, float* route) {
    if (!la_learn_active() || n < 1 || n > kLaTok || la_dist_index(dist) < 0 || route == nullptr) return;
    LaLearn& s = *g_la;
    const int sd = (int) l * kLaNd + la_dist_index(dist);
    LaFwdArgs a{};
    a.x = x;
    a.xsave = dist == kLaDist[0] ? s.x + (size_t) l * kLaTok * s.H : nullptr;   // the first distance saves the input
    a.u = s.u + (size_t) sd * kLaTok * kLaRows;
    a.pred = s.pred + (size_t) sd * kLaSlots * kLaTok * kLaE;
    a.route = route;
    a.lead = s.lead + sd;
    a.n = n;
    a.H = s.H;
    for (int c = 1; c < kLaSlots; ++c) {
        a.A[c] = s.A[c] + (size_t) sd * la_rank(c) * s.H;
        a.B[c] = s.B[c] + (size_t) sd * kLaE * la_rank(c);
    }
    cudaStream_t cs = static_cast<cudaStream_t>(stream);
    la_fwd_u_k<<<kLaRows, 256, 0, cs>>>(a);
    la_fwd_pred_k<<<(n * kLaE + 255) / 256, 256, 0, cs>>>(a);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

// SPEC-router-fuse step 2: the label slab of layer l (rows 0..n-1 of this window), for the post-routing kernel's label copy; nullptr when off.
float* la_learn_label_dev(int64_t l, int n) {
    if (!la_learn_active() || n < 1 || n > kLaTok || l < 0 || l >= g_la->L) return nullptr;
    return g_la->lab + (size_t) l * kLaTok * kLaE;
}
// the label of layer l for this window (on cs, in the graph): its router logits, kept for the batch
void la_learn_label(int64_t l, const float* tl, int n, void* stream) {
    if (!la_learn_active() || n < 1 || n > kLaTok) return;
    LaLearn& s = *g_la;
    cudaStream_t cs = static_cast<cudaStream_t>(stream);
    if (cudaMemcpyAsync(s.lab + (size_t) l * kLaTok * kLaE, tl, (size_t) n * kLaE * sizeof(float), cudaMemcpyDeviceToDevice, cs) != cudaSuccess) {
        const auto error = cudaGetLastError();
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

// the window's end (on cs, in the graph): a stamp for the timing only
void la_learn_window_end(void* stream) {
    if (!la_learn_active() || g_la->timing == false) return;
    la_b_wend_k<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(g_la->st);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

// before a window's graph is launched on cs: the previous window's batch must be done (its weights, its inputs)
// The real wait: the batch is pending when this window's launch comes, so the query tells whether cs will stall on it.  A window that
// finds it not done is counted, and under TIMING an event before and after the wait on cs is kept (read at the request's end).
void la_learn_pre_launch(void* stream) {
    if (!la_learn_active() || !g_la->pending) return;
    LaLearn& s = *g_la;
    cudaStream_t cs = static_cast<cudaStream_t>(stream);
    s.win_pending += 1;
    if (cudaEventQuery(s.ev_done) == cudaSuccess) {   // done already: the wait costs nothing
        cudaStreamWaitEvent(cs, s.ev_done, 0);
        return;
    }
    (void) cudaGetLastError();   // cudaErrorNotReady: the batch is still running
    s.win_waited += 1;
    cudaEvent_t before = nullptr, after = nullptr;
    if (s.timing && cudaEventCreate(&before) == cudaSuccess && cudaEventCreate(&after) == cudaSuccess) {
        cudaEventRecord(before, cs);
        cudaStreamWaitEvent(cs, s.ev_done, 0);
        cudaEventRecord(after, cs);
        s.wait_ev.emplace_back(before, after);
        return;
    }
    (void) cudaGetLastError();
    cudaStreamWaitEvent(cs, s.ev_done, 0);
}

// after a learning window's graph is launched on cs: the batch on the side stream, which waits for the window's end and records done
static void la_batch(LaLearn& s, const int* res, int n, const int* dist, int Nd, const uint8_t* mask, int win, bool train);
void la_learn_post_launch(void* stream, const int* res, int n) {
    if (!la_learn_active() || n < 1 || n > kLaTok) return;
    LaLearn& s = *g_la;
    const int w = s.win++;   // every window counts
    // the batch always runs its scoring and decision (rank, score, decide, reinit: the counters and the cadence advance every window);
    // the training kernels only when a set is due (SPEC-adaptive-la change 2).  The gate: the last batch is done (its decide wrote the
    // next training window, a host-mapped word read with no CUDA call) and no set is due in w.  STRATA_LA_CADENCE=0: always trains.
    bool train = true;
    if (s.cad_on) {
        std::atomic_thread_fence(std::memory_order_acquire);
        const bool valid = *static_cast<volatile int*>(&s.gate_min_h[1]) == s.last_launch;
        const int min_due = *static_cast<volatile int*>(&s.gate_min_h[0]);
        train = la_cadence::launch_needed(valid, min_due, w);
        if (!train) s.skipped += 1;
    }
    cudaStream_t cs = static_cast<cudaStream_t>(stream);
    cudaEventRecord(s.ev_end, cs);
    cudaStreamWaitEvent(s.side, s.ev_end, 0);
    s.last_launch = w;
    // the mask's base ([layers][512]; route_mask_layer(0) is the base): nullptr when no mask is set or it lives on another GPU
    la_batch(s, res, n, kLaDist, kLaNd, route_mask_layer(0, kLaE), w, train);
}
// the batch of n token rows on the instance's side stream (the caller ordered it after the work it reads): the LaBatch from the instance,
// the dists (Nd of them), every kernel in launch order, and the done event.  The main learner's post launch and the drafter's batch share it.
// mask: the route mask's base for the target layers (nullptr: none; the drafter's label is its own router, not a main layer, so it passes none).
static void la_batch(LaLearn& s, const int* res, int n, const int* dist, int Nd, const uint8_t* mask, int win, bool train) {
    LaBatch a{};
    for (int c = 1; c < kLaSlots; ++c) {
        a.A[c] = s.A[c]; a.B[c] = s.B[c]; a.mA[c] = s.mA[c]; a.vA[c] = s.vA[c]; a.mB[c] = s.mB[c]; a.vB[c] = s.vB[c];
    }
    a.x = s.x; a.u = s.u; a.pred = s.pred; a.lab = s.lab; a.res = res;
    a.rk = s.rk; a.thr = s.thr; a.mask = mask; a.g = s.g; a.du = s.du; a.act = s.act; a.tstep = s.tstep;
    a.steps = s.steps; a.live = s.live; a.lead = s.lead; a.frozen = s.frozen; a.reset = s.reset; a.tick = s.tick;
    a.cnt = s.cnt; a.st = s.st; a.tot_ns = s.tot_ns; a.wait_ns = s.wait_ns; a.nwin = s.nwin;
    a.rinfo = s.rinfo; a.kst = s.kst; a.kacc = s.kacc;
    a.cad = s.cad; a.ntrain = s.ntrain; a.gate_min = s.gate_min_d; a.gate_done = s.gate_min_d + 1; a.cad_on = s.cad_on ? 1 : 0; a.win = win;
    for (int i = 0; i < Nd; ++i) a.dist[i] = dist[i];
    a.L = s.L; a.H = s.H; a.lo = s.lo; a.n = n; a.Nd = Nd; a.Nsd = s.Nsd; a.Nset = s.Nset; a.timing = s.timing ? 1 : 0;
    static const float exact_w = [] { const char* v = std::getenv("STRATA_LA_EXACT_W"); const float w = v != nullptr ? (float) std::atof(v) : 4.0f; return w > 0.0f ? w : 1.0f; }();   // (default 4 since 7 Nov: the weight v2 was trained with; =1 the old loss)
    a.exact_w = exact_w;
    cudaStream_t sd_s = s.side;
    // TIMING: a one-thread mark before and after each kernel, in launch order (la_b_mark_k: the per-kernel sums)
    auto mark = [&](int k, int end) {
        if (s.timing) la_b_mark_k<<<1, 1, 0, sd_s>>>(s.kst, s.kacc, k, end);
    };
    mark(0, 0);
    la_b_rank_k<<<dim3(n, s.L), kLaE, 0, sd_s>>>(a);
    mark(0, 1);
    mark(1, 0);
    la_b_score_k<<<dim3(n, s.Nsd, kLaSlots), kLaE, 0, sd_s>>>(a);
    mark(1, 1);
    mark(2, 0);
    if (train) {   // the training kernels: skipped when no set is due (their act flags would all be 0: nothing changes)
        la_b_prep_k<<<(s.Nset + 63) / 64, 64, 0, sd_s>>>(a);
        mark(2, 1);
        mark(3, 0);
        la_b_grad_k<<<dim3(n, s.Nset), kLaE, 0, sd_s>>>(a);
        mark(3, 1);
        mark(4, 0);
        la_b_upd_b_k<<<dim3(32, s.Nset), 256, 0, sd_s>>>(a);
        mark(4, 1);
        mark(5, 0);
        la_b_upd_a_k<<<dim3((s.H + 255) / 256, s.Nset), 256, 0, sd_s>>>(a);
        mark(5, 1);
    }
    mark(6, 0);
    la_b_decide_k<<<1, 256, 0, sd_s>>>(a);   // one block: the (s,d) loop and the block's gate and count
    mark(6, 1);
    mark(7, 0);
    la_b_reinit_k<<<dim3(160, s.Nsd), 256, 0, sd_s>>>(a);
    mark(7, 1);
    if (s.timing) la_b_end_k<<<1, 1, 0, sd_s>>>(a);
    cudaEventRecord(s.ev_done, sd_s);
    s.pending = true;
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

// the host's reads, between requests: the counters, the leaders and the freezes
static bool la_learn_read(LaLearn& s, std::vector<unsigned long long>& cnt, std::vector<int>& lead, std::vector<int>& frozen) {
    if (s.pending) cudaEventSynchronize(s.ev_done);   // the last batch (side stream) is not waited by cudaMemcpy: wait for it here, between requests
    cnt.assign((size_t) s.Nsd * kLaCnt, 0);
    lead.assign((size_t) s.Nsd, 0);
    frozen.assign((size_t) s.Nsd, 0);
    return cudaMemcpy(cnt.data(), s.cnt, cnt.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost) == cudaSuccess &&
           cudaMemcpy(lead.data(), s.lead, lead.size() * sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess &&
           cudaMemcpy(frozen.data(), s.frozen, frozen.size() * sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess;
}
// the stats line(s) of one learner instance since its start, "" when there are none: name is the instance's name in the lines
static std::string la_report_inst(LaLearn& s, const char* name) {
    const std::string pre = std::string("strata serve: ") + name;
    std::vector<unsigned long long> cnt;
    std::vector<int> lead, frozen;
    if (!la_learn_read(s, cnt, lead, frozen)) { (void) cudaGetLastError(); return ""; }
    std::string out;
    double hit_l[kLaNd] = {}, hit_p[kLaNd] = {}, tot[kLaNd] = {};
    unsigned long long resets = 0;
    for (int sd = 0; sd < s.Nsd; ++sd) {
        const int dd = sd % s.Nd, L = lead[sd];
        const unsigned long long* c = cnt.data() + (size_t) sd * kLaCnt;
        resets += c[16];
        tot[dd] += (double) c[0];
        hit_l[dd] += (double) c[3 + L];
        hit_p[dd] += (double) c[3];
        if (frozen[sd] && !s.logged[sd]) {   // the chosen r of each layer, logged once
            s.logged[sd] = 1;
            out += pre + ": layer " + std::to_string(sd / s.Nd) + " d=" + std::to_string(s.dist[dd]) + " chose r=" +
                   std::to_string(la_rank(L)) + " (" + std::to_string(c[15]) + " changes, " + std::to_string(c[2]) + " picks)\n";
        }
    }
    auto hist_of = [&](int dd) {
        std::string h = "{";
        for (int r : {0, 4, 8, 16}) {
            int k = 0;
            for (int sd = dd; sd < s.Nsd; sd += s.Nd) k += la_rank(lead[sd]) == r;
            if (k > 0) h += (h.size() > 1 ? "," : "") + std::to_string(r) + ":" + std::to_string(k);
        }
        return h + "}";
    };
    auto pct = [](double a, double b) {
        if (!(b > 0)) return std::string("-");
        char buf[16];
        std::snprintf(buf, sizeof buf, "%.1f%%", 100.0 * a / b);
        return std::string(buf);
    };
    out += pre + ":";
    for (int dd = 0; dd < s.Nd; ++dd)
        out += (dd > 0 ? ", " : " layers ") + std::string("d=") + std::to_string(s.dist[dd]) + " r=" + hist_of(dd);
    out += ", leader hit rate";
    for (int dd = 0; dd < s.Nd; ++dd)
        out += (dd > 0 ? ", " : " ") + std::string("d=") + std::to_string(s.dist[dd]) + " " + pct(hit_l[dd], tot[dd]) + " vs plain " +
               pct(hit_p[dd], tot[dd]);
    out += " (non-resident ranks < lo; picks since each layer's last change), resets " + std::to_string(resets) + "\n";
    // the training cadence (SPEC-adaptive-la change 2): the windows whose batch trained a set, of the windows seen; the median period of the
    // (s,d) with a target layer (1 live, 4..16 frozen; STRATA_LA_CADENCE=0: no period is kept)
    if (s.win > 0) {
        unsigned long long nt = 0;
        std::vector<la_cadence::State> cd((size_t) s.Nsd);
        if (cudaMemcpy(&nt, s.ntrain, sizeof nt, cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(cd.data(), s.cad, cd.size() * sizeof(la_cadence::State), cudaMemcpyDeviceToHost) != cudaSuccess) {
            (void) cudaGetLastError();
        } else {
            std::vector<int> per;
            for (int sd = 0; sd < s.Nsd; ++sd)
                if (sd / s.Nd + s.dist[sd % s.Nd] < s.L && cd[sd].per > 0) per.push_back(cd[sd].per);
            std::sort(per.begin(), per.end());
            const int med = per.empty() ? 0 : per[per.size() / 2];
            out += pre + ": trained in " + std::to_string(nt) + " of " + std::to_string(s.win) + " windows";
            if (s.cad_on) out += " (cadence median " + std::to_string(med) + ")";
            out += ", " + std::to_string(s.skipped) + " windows whose training launch was skipped (no set due)\n";
        }
    }
    // the resets: each (s,d)'s last one, logged when its count has grown (the recent picks, the plain router's hits, the old leader's)
    if (s.resets_seen.size() != (size_t) s.Nsd) s.resets_seen.assign((size_t) s.Nsd, 0);
    std::vector<unsigned long long> ri((size_t) s.Nsd * kLaSlots, 0);
    if (cudaMemcpy(ri.data(), s.rinfo, ri.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost) != cudaSuccess) (void) cudaGetLastError();
    for (int sd = 0; sd < s.Nsd; ++sd) {
        const unsigned long long n_rs = cnt[(size_t) sd * kLaCnt + 16];
        if (n_rs == s.resets_seen[(size_t) sd]) continue;
        s.resets_seen[(size_t) sd] = n_rs;
        const unsigned long long* q = ri.data() + (size_t) sd * kLaSlots;
        out += pre + ": layer " + std::to_string(sd / s.Nd) + " d=" + std::to_string(s.dist[sd % s.Nd]) +
               " reset (" + std::to_string(n_rs) + " so far): at " + std::to_string(q[0]) + " recent picks the plain router had " +
               std::to_string(q[1]) + " hits, the leader r=" + std::to_string(la_rank((int) q[3])) + " " + std::to_string(q[2]) + "\n";
    }
    // the main stream's real waits on the batch: the windows that found it pending, and (TIMING) the time cs spent waiting
    for (auto& p : s.wait_ev) {
        float ms = 0.0f;
        if (cudaEventSynchronize(p.second) == cudaSuccess && cudaEventElapsedTime(&ms, p.first, p.second) == cudaSuccess) s.wait_ms_total += ms;
        else (void) cudaGetLastError();
        cudaEventDestroy(p.first);
        cudaEventDestroy(p.second);
    }
    s.wait_ev.clear();
    char wbuf[200];
    std::snprintf(wbuf, sizeof wbuf, "%s: main stream waited on the batch in %llu of %llu windows%s\n", pre.c_str(),
                  s.win_waited, s.win_pending, s.timing ? "" : " (not timed: STRATA_LA_LEARN_TIMING=1 times it)");
    out += wbuf;
    if (s.timing) {
        std::snprintf(wbuf, sizeof wbuf, "%s: waited %.3f ms in total (measured on cs)\n", pre.c_str(), s.wait_ms_total);
        out += wbuf;
    }
    if (!s.timing) return out;
    unsigned long long tot_ns = 0, wait_ns = 0, nwin = 0, kacc[kLaKernels] = {};
    if (cudaMemcpy(&tot_ns, s.tot_ns, sizeof tot_ns, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(&wait_ns, s.wait_ns, sizeof wait_ns, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(&nwin, s.nwin, sizeof nwin, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(kacc, s.kacc, sizeof kacc, cudaMemcpyDeviceToHost) != cudaSuccess) {
        (void) cudaGetLastError();
        return out;
    }
    const double w = nwin > 0 ? (double) nwin : 1.0;
    static const char* const kname[kLaKernels] = {"rank", "score", "prep", "grad", "upd_b", "upd_a", "decide", "reinit"};
    std::string per;
    for (int k = 0; k < kLaKernels; ++k) {
        char kb[48];
        std::snprintf(kb, sizeof kb, "%s %.3f", kname[k], (double) kacc[k] / 1e6 / w);
        per += (k > 0 ? ", " : "") + std::string(kb);
    }
    char buf[320];
    std::snprintf(buf, sizeof buf, "%s: batch on the side stream %.3f ms/window (per kernel ms: %s), "
                  "end-of-window wait bound %.3f ms/window, over %llu windows\n",
                  pre.c_str(), (double) tot_ns / 1e6 / w, per.c_str(), (double) wait_ns / 1e6 / w, nwin);
    return out + buf;
}
std::string la_learn_report() {
    if (g_la == nullptr) return "";
    return la_report_inst(*g_la, "look-ahead learner");
}

// The file (STRATA_LA_LEARN): "LAL1", version 1, n_layers, n_embd, 512; then per (s,d) the leader's rank (0, 4, 8 or 16); then for each
// (s,d) with a rank, the leader's A (r x n_embd) and B (512 x r) as fp16.  Written to a temporary file and renamed.
static bool la_save_inst(LaLearn& s, const char* path, std::string& msg) {
    std::vector<unsigned long long> cnt;
    std::vector<int> lead, frozen;
    if (!la_learn_read(s, cnt, lead, frozen)) { (void) cudaGetLastError(); msg = "cannot read the GPU state"; return false; }
    const std::string tmp = std::string(path) + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (f == nullptr) { msg = "cannot write " + tmp; return false; }
    // version 3: one word per (s,d) of kLaDist (kLaNd per layer): the leader's rank, | 256 when the (s,d) is frozen (its leader alone trains,
    // every 4th window); version 2 had dists 1 and 2 only (la_learn_load maps it)
    const uint32_t hdr[5] = {0x314C414Cu, 3u, (uint32_t) s.L, (uint32_t) s.H, (uint32_t) kLaE};
    bool ok = std::fwrite(hdr, sizeof hdr, 1, f) == 1;
    for (int sd = 0; ok && sd < s.Nsd; ++sd) {
        const uint32_t r = (uint32_t) la_rank(lead[sd]) | (frozen[sd] ? 256u : 0u);
        ok = std::fwrite(&r, sizeof r, 1, f) == 1;
    }
    for (int sd = 0; ok && sd < s.Nsd; ++sd) {
        const int c = lead[sd], r = la_rank(c);
        if (c == 0) continue;
        std::vector<float> fa((size_t) r * s.H), fb((size_t) kLaE * r);
        std::vector<uint16_t> ha(fa.size()), hb(fb.size());
        ok = cudaMemcpy(fa.data(), s.A[c] + (size_t) sd * r * s.H, fa.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess &&
             cudaMemcpy(fb.data(), s.B[c] + (size_t) sd * kLaE * r, fb.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
        for (size_t i = 0; i < fa.size(); ++i) ha[i] = __half_as_ushort(__float2half(fa[i]));
        for (size_t i = 0; i < fb.size(); ++i) hb[i] = __half_as_ushort(__float2half(fb[i]));
        ok = ok && std::fwrite(ha.data(), sizeof(uint16_t), ha.size(), f) == ha.size() &&
             std::fwrite(hb.data(), sizeof(uint16_t), hb.size(), f) == hb.size();
    }
    const bool closed = std::fclose(f) == 0;
    if (!ok || !closed) { std::remove(tmp.c_str()); msg = "write failed"; return false; }
    if (std::rename(tmp.c_str(), path) != 0) { std::remove(tmp.c_str()); msg = "rename failed"; return false; }
    return true;
}
bool la_learn_save(const char* path, std::string& msg) {
    if (!la_learn_active()) { msg = "off"; return false; }
    return la_save_inst(*g_la, path, msg);
}
// Loads a file saved for this model shape.  Each (s,d) takes the file's leader in its rank's slot; the other slots start from the
// leader's first rows and columns (the rest random / zero), all live, with fresh moments.  Called between requests, before any window.
static bool la_load_inst(LaLearn& s, const char* path, std::string& msg) {
    FILE* f = std::fopen(path, "rb");
    if (f == nullptr) { msg = "no file"; return false; }
    uint32_t hdr[5] = {0, 0, 0, 0, 0};
    bool ok = std::fread(hdr, sizeof hdr, 1, f) == 1 && hdr[0] == 0x314C414Cu && (hdr[1] == 1u || hdr[1] == 2u || hdr[1] == 3u) &&
              (int) hdr[2] == s.L && (int) hdr[3] == s.H && hdr[4] == (uint32_t) kLaE && (hdr[1] == 3u || s.Nd == kLaNd);
    if (!ok) { std::fclose(f); msg = "not a LAL1 file for this model"; return false; }
    // the file's (s,d) list: version 3 has every dist of kLaDist; versions 1 and 2 had dists 1 and 2 only, which map onto the first two
    // entries of kLaDist (dist 4 then starts from the plain router, not frozen)
    const int file_nd = hdr[1] == 3u ? s.Nd : 2;
    const int n_file = file_nd * s.L;
    std::vector<int> rl((size_t) s.Nsd, 0), fzv((size_t) s.Nsd, 0), fsd((size_t) n_file, 0);
    for (int fi = 0; ok && fi < n_file; ++fi) {
        uint32_t w = 0;
        ok = std::fread(&w, sizeof w, 1, f) == 1;
        const uint32_t r = w & 255u;
        ok = ok && (r == 0 || r == 4 || r == 8 || r == 16) && (w >> 9) == 0;
        const int sd = (fi / file_nd) * s.Nd + fi % file_nd;
        fsd[(size_t) fi] = sd;
        rl[(size_t) sd] = (int) r;
        fzv[(size_t) sd] = (int) ((w >> 8) & 1u);
    }
    std::vector<float> hA[kLaSlots], hB[kLaSlots];
    for (int c = 1; c < kLaSlots; ++c) {
        const int r = la_rank(c);
        hA[c].assign((size_t) s.Nsd * r * s.H, 0.0f);
        hB[c].assign((size_t) s.Nsd * kLaE * r, 0.0f);
        for (int sd = 0; sd < s.Nsd; ++sd)
            for (int i = 0; i < r * s.H; ++i) hA[c][(size_t) sd * r * s.H + i] = la_rand_pm(la_key(sd, c, i, 0));
    }
    std::vector<int> lead((size_t) s.Nsd, 0);
    for (int fi = 0; ok && fi < n_file; ++fi) {   // the data follow the words in the file's order
        const int sd = fsd[(size_t) fi];
        const int rL = rl[(size_t) sd];
        if (rL == 0) continue;
        const int cL = rL == 4 ? 1 : (rL == 8 ? 2 : 3);
        lead[(size_t) sd] = cL;
        std::vector<uint16_t> ra((size_t) rL * s.H), rb((size_t) kLaE * rL);
        ok = std::fread(ra.data(), sizeof(uint16_t), ra.size(), f) == ra.size() && std::fread(rb.data(), sizeof(uint16_t), rb.size(), f) == rb.size();
        if (!ok) break;
        std::vector<float> al(ra.size()), bl(rb.size());
        for (size_t i = 0; i < ra.size(); ++i) al[i] = __half2float(__ushort_as_half(ra[i]));
        for (size_t i = 0; i < rb.size(); ++i) bl[i] = __half2float(__ushort_as_half(rb[i]));
        for (int c = 1; c < kLaSlots; ++c) {
            const int r = la_rank(c), m = r < rL ? r : rL;
            for (int kk = 0; kk < m; ++kk) {
                for (int h = 0; h < s.H; ++h) hA[c][((size_t) sd * r + kk) * s.H + h] = al[(size_t) kk * s.H + h];
                for (int e = 0; e < kLaE; ++e) hB[c][((size_t) sd * kLaE + e) * r + kk] = bl[(size_t) e * rL + kk];
            }
        }
    }
    std::fclose(f);
    if (!ok) { msg = "the file is cut short or damaged"; return false; }
    for (int c = 1; c < kLaSlots; ++c) {
        const size_t na = hA[c].size() * sizeof(float), nb = hB[c].size() * sizeof(float);
        cudaMemcpy(s.A[c], hA[c].data(), na, cudaMemcpyHostToDevice);
        cudaMemcpy(s.B[c], hB[c].data(), nb, cudaMemcpyHostToDevice);
        cudaMemset(s.mA[c], 0, na);
        cudaMemset(s.vA[c], 0, na);
        cudaMemset(s.mB[c], 0, nb);
        cudaMemset(s.vB[c], 0, nb);
    }
    std::vector<int> live((size_t) s.Nsd * kLaSlots, 1), zero((size_t) s.Nsd * kLaSlots, 0), zsd((size_t) s.Nsd, 0);
    // a frozen (s,d) keeps only its leader training (every 4th window); its candidates are not restarted
    for (int sd = 0; sd < s.Nsd; ++sd)
        if (fzv[(size_t) sd])
            for (int k = 1; k < kLaSlots; ++k) live[(size_t) sd * kLaSlots + k] = lead[(size_t) sd] == k ? 1 : 0;
    cudaMemcpy(s.lead, lead.data(), lead.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(s.live, live.data(), live.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(s.steps, zero.data(), zero.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(s.frozen, fzv.data(), fzv.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(s.reset, zsd.data(), zsd.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(s.tick, zsd.data(), zsd.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(s.cnt, 0, (size_t) s.Nsd * kLaCnt * sizeof(unsigned long long));
    {   // the cadence restarts: a loaded frozen (s,d) trains its leader in the next window; the batch gate is not valid until a batch has run
        std::vector<la_cadence::State> cd((size_t) s.Nsd);
        for (int sd = 0; sd < s.Nsd; ++sd) {
            la_cadence::State& st = cd[(size_t) sd];
            st.base = 0;
            st.per = fzv[(size_t) sd] ? la_cadence::kBasePeriod : 1;
            st.due = fzv[(size_t) sd] ? s.win : s.win + 1;
        }
        cudaMemcpy(s.cad, cd.data(), cd.size() * sizeof(la_cadence::State), cudaMemcpyHostToDevice);
        s.last_launch = -1;
    }
    s.logged.assign((size_t) s.Nsd, 0);
    s.resets_seen.assign((size_t) s.Nsd, 0);
    if (cudaGetLastError() != cudaSuccess) { msg = "the copy to the GPU failed"; return false; }
    int nfrozen = 0;
    for (int v : fzv) nfrozen += v;
    msg = "loaded, " + std::to_string(nfrozen) + " of " + std::to_string(s.Nsd) + " frozen";
    return true;
}
bool la_learn_load(const char* path, std::string& msg) {
    if (!la_learn_active()) { msg = "off"; return false; }
    return la_load_inst(*g_la, path, msg);
}

// ---- STRATA_MTP_LA=1 (opt-in; mtp.cpp, verify.cpp): the drafter's look-ahead.  A second instance g_lam: L = 2 (layer 0 is the main model's
// layer src's router input, layer 1 the drafter), Nd = 1, dist {1}, lo 2.  Each decode window's graph runs la_mtp_forward on that layer's
// input rows: the drafter's own router (its gate, the plain logits in la_mtp_route) with the learned correction, the leader's top 10 per row,
// written to the mapped record (pred_h, its sequence last) for the drafter's loader.  The drafter's first round writes its router logits as
// the label (la_mtp_label, in its graph) and starts from window row a: la_mtp_take_source copies that row's source, slot logits and u into
// token 0 of the instance's own buffers before the round.  The batch (la_mtp_batch) runs when the drafter's stream is idle and reads only
// those buffers: the window's forward buffers are written by the next window before the batch, so they cannot be what the batch reads.
static LaLearn* g_lam = nullptr;
struct MtpLaBuf {
    int H = 0;
    float* route = nullptr;     // [8][512]: the plain logits in, the leader's out
    float* xall = nullptr;      // [8][H]: the window's source rows
    float* pscr = nullptr;      // [4][8][512]: the window's slot logits (LaFwdArgs::pred)
    float* uscr = nullptr;      // [8][kLaRows]: the window's u (LaFwdArgs::u)
    int* res2 = nullptr;        // [2][512]: row 0 all -1, row 1 the drafter's slot map (device copy of d_slot_)
    int32_t* pids = nullptr;    // [8][10]: the leader's top 10 per row
    float* pw = nullptr;        // [8][10]: their weights
    int* pred_h = nullptr;      // mapped: [0] the sequence, [1] the rows, [2 + 10 t + r] row t's rank r id
    int* pred_d = nullptr;      // the device alias of pred_h
    cudaEvent_t ev_src = nullptr;   // recorded after take_source's copies
    bool src_recorded = false;
};
static MtpLaBuf g_mla;

// the window's publish, one block of 128 threads: the ids of the rows, then the rows, then the sequence (after the ids, system-wide)
__global__ void la_mtp_pub_k(const int32_t* pids, int n, int* rec) {
    const int i = threadIdx.x;
    if (i < n * 10) rec[2 + i] = pids[i];
    if (i == 0) rec[1] = n;
    __threadfence_system();
    __syncthreads();
    if (i == 0) {
        __threadfence_system();
        rec[0] = rec[0] + 1;
    }
}

static void la_mtp_free_buf() {
    MtpLaBuf& m = g_mla;
    cudaFree(m.route);
    cudaFree(m.xall);
    cudaFree(m.pscr);
    cudaFree(m.uscr);
    cudaFree(m.res2);
    cudaFree(m.pids);
    cudaFree(m.pw);
    if (m.pred_h != nullptr) cudaFreeHost(m.pred_h);
    if (m.ev_src != nullptr) cudaEventDestroy(m.ev_src);
    m = MtpLaBuf();
}

bool la_mtp_alloc(int64_t n_embd, bool timing, std::string& msg) {
    if (g_lam != nullptr) { msg = "already on"; return true; }
    if (g_la == nullptr) { msg = "the look-ahead learner is off"; return false; }
    if (n_embd <= 0) { msg = "no width"; return false; }
    note_scratch_dev();
    if (!scratch_here()) { msg = "not on the scratch GPU"; return false; }
    MtpLaBuf& m = g_mla;
    m.H = (int) n_embd;
    bool ok = true;
    auto dalloc = [&](void** p, size_t bytes) {
        *p = nullptr;
        if (!ok) return;
        if (cudaMalloc(p, bytes) != cudaSuccess) { (void) cudaGetLastError(); ok = false; return; }
        cudaMemset(*p, 0, bytes);
    };
    dalloc((void**) &m.route, (size_t) kLaTok * kLaE * sizeof(float));
    dalloc((void**) &m.xall, (size_t) kLaTok * m.H * sizeof(float));
    dalloc((void**) &m.pscr, (size_t) kLaSlots * kLaTok * kLaE * sizeof(float));
    dalloc((void**) &m.uscr, (size_t) kLaTok * kLaRows * sizeof(float));
    dalloc((void**) &m.res2, (size_t) 2 * kLaE * sizeof(int));
    dalloc((void**) &m.pids, (size_t) kLaTok * 10 * sizeof(int32_t));
    dalloc((void**) &m.pw, (size_t) kLaTok * 10 * sizeof(float));
    if (ok) {
        cudaMemset(m.res2, 0xff, (size_t) 2 * kLaE * sizeof(int));   // -1: not resident (row 1 is copied in by the drafter)
        if (cudaHostAlloc((void**) &m.pred_h, (2 + kLaTok * 10) * sizeof(int), cudaHostAllocMapped) != cudaSuccess ||
            cudaHostGetDevicePointer((void**) &m.pred_d, m.pred_h, 0) != cudaSuccess) {
            (void) cudaGetLastError();
            ok = false;
        } else {
            for (int i = 0; i < 2 + kLaTok * 10; ++i) m.pred_h[i] = 0;
        }
    }
    if (ok && cudaEventCreateWithFlags(&m.ev_src, cudaEventDisableTiming) != cudaSuccess) { (void) cudaGetLastError(); ok = false; }
    if (!ok) {
        la_mtp_free_buf();
        msg = "no room on the GPU for the drafter's look-ahead (or no events)";
        return false;
    }
    LaLearn* s = la_new(2, (int) n_embd, 2, 1, timing, msg);
    if (s == nullptr) {
        la_mtp_free_buf();
        return false;
    }
    s->dist[0] = 1;
    g_lam = s;
    return true;
}
bool la_mtp_active() { return g_lam != nullptr && scratch_here(); }
float* la_mtp_route() { return la_mtp_active() ? g_mla.route : nullptr; }
int* la_mtp_res_row() { return la_mtp_active() ? g_mla.res2 + kLaE : nullptr; }
const int* la_mtp_pred_host() { return la_mtp_active() ? g_mla.pred_h : nullptr; }

// the window's forward (on the window's stream, in its graph): the source rows into xall, the drafter's plain logits (la_mtp_route, written by
// the caller) through the leader of the instance's sd 0, then the top 10 per row and the record
void la_mtp_forward(const float* x, int n, void* stream) {
    if (!la_mtp_active() || n < 1 || n > kLaTok) return;
    LaLearn& s = *g_lam;
    MtpLaBuf& m = g_mla;
    cudaStream_t cs = static_cast<cudaStream_t>(stream);
    cudaMemcpyAsync(m.xall, x, (size_t) n * m.H * sizeof(float), cudaMemcpyDeviceToDevice, cs);
    LaFwdArgs a{};
    a.x = m.xall;
    a.xsave = nullptr;
    a.u = m.uscr;
    a.pred = m.pscr;
    a.route = m.route;
    a.lead = s.lead;
    a.n = n;
    a.H = m.H;
    for (int c = 1; c < kLaSlots; ++c) {
        a.A[c] = s.A[c];
        a.B[c] = s.B[c];
    }
    la_fwd_u_k<<<kLaRows, 256, 0, cs>>>(a);
    la_fwd_pred_k<<<(n * kLaE + 255) / 256, 256, 0, cs>>>(a);
    native_router_top10_multi(m.route, m.pids, m.pw, n, stream);
    la_mtp_pub_k<<<1, 128, 0, cs>>>(m.pids, n, m.pred_d);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

// the drafter's round starts from window row `row`: its source, the slot logits and u of that row into token 0 of the instance's buffers
// (after the last batch, which reads them), and the event the window waits on before it overwrites the window's buffers
void la_mtp_take_source(int row, void* stream) {
    if (!la_mtp_active() || row < 0 || row >= kLaTok) return;
    LaLearn& s = *g_lam;
    MtpLaBuf& m = g_mla;
    cudaStream_t cs = static_cast<cudaStream_t>(stream);
    if (s.pending) cudaStreamWaitEvent(cs, s.ev_done, 0);
    cudaMemcpyAsync(s.x, m.xall + (size_t) row * m.H, (size_t) m.H * sizeof(float), cudaMemcpyDeviceToDevice, cs);
    for (int c = 0; c < kLaSlots; ++c)   // slot c, token 0 (the batch's layout: sd 0, token 0 of each slot)
        cudaMemcpyAsync(s.pred + (size_t) c * kLaTok * kLaE, m.pscr + ((size_t) c * kLaTok + row) * kLaE, (size_t) kLaE * sizeof(float),
                        cudaMemcpyDeviceToDevice, cs);
    cudaMemcpyAsync(s.u, m.uscr + (size_t) row * kLaRows, (size_t) kLaRows * sizeof(float), cudaMemcpyDeviceToDevice, cs);
    cudaEventRecord(m.ev_src, cs);
    m.src_recorded = true;
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

// the drafter's first step's router logits (512) as the label of layer 1, token 0 (in the drafter's round graph)
void la_mtp_label(const float* logits_row, void* stream) {
    if (!la_mtp_active()) return;
    cudaStream_t cs = static_cast<cudaStream_t>(stream);
    cudaMemcpyAsync(g_lam->lab + (size_t) kLaTok * kLaE, logits_row, (size_t) kLaE * sizeof(float), cudaMemcpyDeviceToDevice, cs);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

// the batch of the last round, on the instance's side stream (the caller found the drafter's stream idle: the label and the copies are done)
void la_mtp_batch() {
    if (!la_mtp_active()) return;
    static const int kDist1[1] = {1};
    la_batch(*g_lam, g_mla.res2, 1, kDist1, 1, nullptr, g_lam->win++, true);
}

// before a window's graph is launched on its stream: the last batch is done, and the source copies are (the forward overwrites them)
void la_mtp_pre_launch(void* stream) {
    if (!la_mtp_active()) return;
    cudaStream_t cs = static_cast<cudaStream_t>(stream);
    if (g_lam->pending) cudaStreamWaitEvent(cs, g_lam->ev_done, 0);
    if (g_mla.src_recorded) cudaStreamWaitEvent(cs, g_mla.ev_src, 0);
}

bool la_mtp_save(const char* path, std::string& msg) {
    if (!la_mtp_active()) { msg = "off"; return false; }
    return la_save_inst(*g_lam, path, msg);
}
bool la_mtp_load(const char* path, std::string& msg) {
    if (!la_mtp_active()) { msg = "off"; return false; }
    return la_load_inst(*g_lam, path, msg);
}
std::string la_mtp_report() {
    if (!la_mtp_active()) return "";
    return la_report_inst(*g_lam, "drafter look-ahead learner");
}
}
