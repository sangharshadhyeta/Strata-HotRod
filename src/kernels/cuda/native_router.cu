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
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <atomic>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <cstdio>
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
__launch_bounds__(256, 1)
__global__ void route_multi(const float* __restrict__ logits, int32_t* __restrict__ ids,
                            float* __restrict__ weights, int n_tok) {
    const int tk = (int) blockIdx.x * 8 + (int) threadIdx.y;
    if (tk >= n_tok) return;
    logits += (size_t) tk * 512; ids += (size_t) tk * 10; weights += (size_t) tk * 10;
    const int lane = threadIdx.x;
    float values[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) values[i] = logits[lane + i * 32];
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
__global__ void route_resident_k(const float* __restrict__ logits, int32_t* __restrict__ ids, float* __restrict__ weights,
                                 const int32_t* __restrict__ res, int n_tok, const RouteCfg* cfg,
                                 unsigned long long* __restrict__ stats, float* __restrict__ want,
                                 unsigned long long* __restrict__ detail, int32_t* __restrict__ rows, float* __restrict__ repl,
                                 const uint8_t* __restrict__ ring, const uint8_t* __restrict__ cheap, float cheap_margin,
                                 int cheap_rank0, int cheap_anytier, unsigned long long* __restrict__ bstats,
                                 float* __restrict__ row_shift) {
    const int t = blockIdx.x;
    if (t >= n_tok) return;
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
    for (int r = (cheap != nullptr || bud) ? 0 : first; r <= hi && r < 10; ++r) {
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
        const bool take = brank ? weights[(size_t) t * 10 + r] * fmaxf(gap, 0.f) <= budget
                        : trank ? weights[(size_t) t * 10 + r] * fmaxf(gap, 0.f) <= budget_tail
                        : gap <= ((cheap != nullptr && r < 2) ? cheap_margin
                                  : (edge >= 0.f && lo > 0 && r == lo - 1) ? edge : margin);
        if (bi >= 0 && trank && bstats != nullptr && lane == 0) atomicAdd(bstats + (take ? 3 : 4), 1ull);   // bstats [3] tail replaced, [4] tail kept exact by the bound
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
            if (lane == 0 && row_shift != nullptr) atomicAdd(row_shift + t, weights[(size_t) t * 10 + r] * fmaxf(gap, 0.f));
            my[r] = bi;
            ++swaps;
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
void native_route_resident(const float* logits, int32_t* ids, float* weights, const int32_t* res_layer, int n_tok,
                           unsigned long long* stats, void* stream, float* want, int32_t* rows, float* repl,
                           const uint8_t* ring, const uint8_t* cheap, unsigned long long* bstats, float* row_shift) {
    if (!stream || n_tok < 1 || !res_layer || g_cfg_dev == nullptr) throw std::invalid_argument("native_route_resident: bad arguments");
    // STRATA_ROUTE_RAM_FLOOR's safety: the stand-in for a rank 0-1 pick must be within STRATA_ROUTE_RAM_FLOOR_MARGIN logits (default 1,
    // the margin SPEC E1 found sound for rank 1), and rank 0 is replaced only with STRATA_ROUTE_RAM_FLOOR_RANK0=1 (off: it leaked
    // <|im_start|>user into an answer at 94K tokens, 20 Oct)
    static const float cheap_margin = [] { const char* v = std::getenv("STRATA_ROUTE_RAM_FLOOR_MARGIN"); return v ? (float) std::atof(v) : 1.0f; }();
    static const int cheap_rank0 = [] { const char* v = std::getenv("STRATA_ROUTE_RAM_FLOOR_RANK0"); return v != nullptr && std::atoi(v) != 0 ? 1 : 0; }();
    // STRATA_ROUTE_RAM_FLOOR_ANYTIER (default ON since 20 Oct, =0 turns it off): a close stand-in replaces a rank 0-1 pick in the
    // RAM tier too, not only one that would come from the SSD
    static const int cheap_anytier = [] { const char* v = std::getenv("STRATA_ROUTE_RAM_FLOOR_ANYTIER"); return v == nullptr || std::atoi(v) != 0 ? 1 : 0; }();
    route_resident_k<<<(unsigned) n_tok, 32, 0, static_cast<cudaStream_t>(stream)>>>(   // one warp (32 threads) per token
        logits, ids, weights, res_layer, n_tok, g_cfg_dev, stats, want, n_tok <= 8 ? g_detail : nullptr, n_tok <= 8 ? rows : nullptr,
        n_tok <= 8 ? repl : nullptr, ring, cheap, cheap_margin, cheap_rank0, cheap_anytier, bstats, row_shift);
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
        cudaMalloc((void**) &g_pred_logits, 8 * 512 * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void**) &g_pred_w, 8 * 10 * sizeof(float)) != cudaSuccess) {
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
__global__ void pred_edge_skip_k(const float* __restrict__ logits, int32_t* __restrict__ ids, const int32_t* __restrict__ res,
                                 const RouteCfg* cfg, unsigned long long* __restrict__ skipped, int n_tok) {
    const int t = blockIdx.x;
    if (t >= n_tok) return;
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
constexpr int kLaCnt = 17;       // counters per (s,d): [0] picks since change, [1] picks in the recent 500, [2] picks all,
                                 // [3+c] hits since change, [7+c] hits recent, [11+c] hits all, [15] changes, [16] resets
constexpr int kLaMinPicks = 200, kLaFreezePicks = 2000, kLaRecPicks = 500, kLaLeaderEvery = 4;
constexpr float kLaLr = 1e-3f, kLaB1 = 0.9f, kLaB2 = 0.999f, kLaEps = 1e-8f, kLaWd = 1e-3f, kLaZ90 = 1.6448536f;

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
    return sh[0];
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
    unsigned long long* cnt;     // [Nsd][17]
    unsigned long long* st;      // [3]: the batch's start, the window's end (timing)
    unsigned long long* tot_ns;  // timing: the batch's time on the side stream
    unsigned long long* wait_ns; // timing: how far the batch ran past the window's end (the next window's wait bound)
    unsigned long long* nwin;    // the windows whose batch ran
    unsigned long long* rinfo;   // [Nsd][4]: the last reset's recent picks, plain hits, old leader's hits, old leader's slot
    unsigned long long* kst;     // timing: the per-kernel profile (kst: starts, kacc: sums), [kLaKernels]
    unsigned long long* kacc;
    int dist[kLaNd];
    int L, H, lo, n, Nd, Nsd, Nset, timing;
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
    if (a.frozen[sd] && (a.lead[sd] != c || a.tick[sd] % kLaLeaderEvery != 0)) ok = 0;
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

// the window's decision, one thread per (s,d): the leader (a candidate beats the leader by a Newcombe Wilson 90% interval on >= 200 picks
// since the last change), the freeze (2000 picks without a change), the reset (the plain router beats a leader over the recent 500)
__host__ __device__ inline void la_wilson(float p, float n, float& lo, float& hi) {
    const float z2 = kLaZ90 * kLaZ90, den = 1.0f + z2 / n;
    const float c = (p + z2 / (2.0f * n)) / den;
    const float h = kLaZ90 * sqrtf(p * (1.0f - p) / n + z2 / (4.0f * n * n)) / den;
    lo = c - h;
    hi = c + h;
}
// Newcombe's lower bound for p1 - p2 from two Wilson 90% intervals over n1 and n2 picks
__device__ __forceinline__ float la_diff_lo(float p1, float n1, float p2, float n2) {
    float l1, u1, l2, u2;
    la_wilson(p1, n1, l1, u1);
    la_wilson(p2, n2, l2, u2);
    return (p1 - p2) - sqrtf((p1 - l1) * (p1 - l1) + (u2 - p2) * (u2 - p2));
}
__global__ void la_b_decide_k(LaBatch a) {   // grid ceil(Nsd / 64), block 64
    const int sd = blockIdx.x * blockDim.x + threadIdx.x;
    if (sd == 0) atomicAdd(a.nwin, 1ull);
    if (sd >= a.Nsd) return;
    unsigned long long* c = a.cnt + (size_t) sd * kLaCnt;
    a.reset[sd] = 0;
    a.tick[sd] += 1;
    int L = a.lead[sd];
    const float nS = (float) c[0];
    if (!a.frozen[sd] && nS >= kLaMinPicks) {
        int best = -1;
        float bestp = -1.0f;
        const float pL = (float) c[3 + L] / nS;
        for (int k = 0; k < kLaSlots; ++k) {
            if (k == L) continue;
            const float p = (float) c[3 + k] / nS;
            if (p > bestp && la_diff_lo(p, nS, pL, nS) > 0.0f) { best = k; bestp = p; }
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
        if (la_diff_lo(p0, nR, pL, nR) > 0.0f) {
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
    if (nR >= kLaRecPicks) {
        c[1] = 0;
        for (int k = 0; k < kLaSlots; ++k) c[7 + k] = 0;
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
    unsigned long long* cnt = nullptr;   // [Nsd][17]
    unsigned long long* st = nullptr;    // [3]
    unsigned long long* tot_ns = nullptr;
    unsigned long long* wait_ns = nullptr;
    unsigned long long* nwin = nullptr;
    unsigned long long* rinfo = nullptr;   // [Nsd][4]: the last reset's facts (LaBatch::rinfo)
    unsigned long long* kst = nullptr;     // [kLaKernels]: the per-kernel profile (timing)
    unsigned long long* kacc = nullptr;    // [kLaKernels]
    // host: the real main-stream waits (the batch not done when a window's launch came): windows that found it pending, and the timed ones
    unsigned long long win_pending = 0, win_waited = 0;
    double wait_ms_total = 0.0;   // the timed waits so far (ms)
    std::vector<std::pair<cudaEvent_t, cudaEvent_t>> wait_ev;   // timing: before/after the wait on cs, read at the request's end
    std::vector<unsigned long long> resets_seen;                  // host: per (s,d), the resets already logged
    std::vector<char> logged;    // host: the (s,d) whose chosen r was logged
    size_t bytes = 0;
};
static LaLearn* g_la = nullptr;

bool la_learn_alloc(int64_t n_layers, int64_t n_embd, int lo, bool timing, std::string& msg) {
    if (g_la != nullptr) { msg = "already on"; return true; }
    if (n_layers <= 0 || n_embd <= 0) { msg = "no layers"; return false; }
    note_scratch_dev();
    if (!scratch_here()) { msg = "not on the scratch GPU"; return false; }
    LaLearn* s = new LaLearn();
    s->L = (int) n_layers;
    s->H = (int) n_embd;
    s->lo = lo;
    s->timing = timing;
    s->Nsd = kLaNd * s->L;
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
        if (s->side != nullptr) cudaStreamDestroy(s->side);
        if (s->ev_end != nullptr) cudaEventDestroy(s->ev_end);
        if (s->ev_done != nullptr) cudaEventDestroy(s->ev_done);
        (void) cudaGetLastError();
        delete s;
        return false;
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
    g_la = s;
    char buf[160];
    std::snprintf(buf, sizeof buf, "%lld layers, %zu MiB of VRAM%s", (long long) n_layers, s->bytes >> 20, timing ? ", timing on" : "");
    msg = buf;
    return true;
}
bool la_learn_active() { return g_la != nullptr && scratch_here(); }

// the forward, on cs (in the window's graph)
void la_learn_forward(int64_t l, int dist, const float* x, int n, void* stream) {
    if (!la_learn_active() || n < 1 || n > kLaTok || la_dist_index(dist) < 0) return;
    LaLearn& s = *g_la;
    const int sd = (int) l * kLaNd + la_dist_index(dist);
    LaFwdArgs a{};
    a.x = x;
    a.xsave = dist == kLaDist[0] ? s.x + (size_t) l * kLaTok * s.H : nullptr;   // the first distance saves the input
    a.u = s.u + (size_t) sd * kLaTok * kLaRows;
    a.pred = s.pred + (size_t) sd * kLaSlots * kLaTok * kLaE;
    a.route = route_pred_logits();
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
void la_learn_post_launch(void* stream, const int* res, int n) {
    if (!la_learn_active() || n < 1 || n > kLaTok) return;
    LaLearn& s = *g_la;
    cudaStream_t cs = static_cast<cudaStream_t>(stream);
    cudaEventRecord(s.ev_end, cs);
    cudaStreamWaitEvent(s.side, s.ev_end, 0);
    LaBatch a{};
    for (int c = 1; c < kLaSlots; ++c) {
        a.A[c] = s.A[c]; a.B[c] = s.B[c]; a.mA[c] = s.mA[c]; a.vA[c] = s.vA[c]; a.mB[c] = s.mB[c]; a.vB[c] = s.vB[c];
    }
    a.x = s.x; a.u = s.u; a.pred = s.pred; a.lab = s.lab; a.res = res;
    a.rk = s.rk; a.thr = s.thr; a.g = s.g; a.du = s.du; a.act = s.act; a.tstep = s.tstep;
    a.steps = s.steps; a.live = s.live; a.lead = s.lead; a.frozen = s.frozen; a.reset = s.reset; a.tick = s.tick;
    a.cnt = s.cnt; a.st = s.st; a.tot_ns = s.tot_ns; a.wait_ns = s.wait_ns; a.nwin = s.nwin;
    a.rinfo = s.rinfo; a.kst = s.kst; a.kacc = s.kacc;
    for (int i = 0; i < kLaNd; ++i) a.dist[i] = kLaDist[i];
    a.L = s.L; a.H = s.H; a.lo = s.lo; a.n = n; a.Nd = kLaNd; a.Nsd = s.Nsd; a.Nset = s.Nset; a.timing = s.timing ? 1 : 0;
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
    mark(6, 0);
    la_b_decide_k<<<(s.Nsd + 63) / 64, 64, 0, sd_s>>>(a);
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
std::string la_learn_report() {
    if (g_la == nullptr) return "";
    LaLearn& s = *g_la;
    std::vector<unsigned long long> cnt;
    std::vector<int> lead, frozen;
    if (!la_learn_read(s, cnt, lead, frozen)) { (void) cudaGetLastError(); return ""; }
    std::string out;
    double hit_l[kLaNd] = {}, hit_p[kLaNd] = {}, tot[kLaNd] = {};
    unsigned long long resets = 0;
    for (int sd = 0; sd < s.Nsd; ++sd) {
        const int dd = sd % kLaNd, L = lead[sd];
        const unsigned long long* c = cnt.data() + (size_t) sd * kLaCnt;
        resets += c[16];
        tot[dd] += (double) c[0];
        hit_l[dd] += (double) c[3 + L];
        hit_p[dd] += (double) c[3];
        if (frozen[sd] && !s.logged[sd]) {   // the chosen r of each layer, logged once
            s.logged[sd] = 1;
            out += "strata serve: look-ahead learner: layer " + std::to_string(sd / kLaNd) + " d=" + std::to_string(kLaDist[dd]) + " chose r=" +
                   std::to_string(la_rank(L)) + " (" + std::to_string(c[15]) + " changes, " + std::to_string(c[2]) + " picks)\n";
        }
    }
    auto hist_of = [&](int dd) {
        std::string h = "{";
        for (int r : {0, 4, 8, 16}) {
            int k = 0;
            for (int sd = dd; sd < s.Nsd; sd += kLaNd) k += la_rank(lead[sd]) == r;
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
    out += "strata serve: look-ahead learner:";
    for (int dd = 0; dd < kLaNd; ++dd)
        out += (dd > 0 ? ", " : " layers ") + std::string("d=") + std::to_string(kLaDist[dd]) + " r=" + hist_of(dd);
    out += ", leader hit rate";
    for (int dd = 0; dd < kLaNd; ++dd)
        out += (dd > 0 ? ", " : " ") + std::string("d=") + std::to_string(kLaDist[dd]) + " " + pct(hit_l[dd], tot[dd]) + " vs plain " +
               pct(hit_p[dd], tot[dd]);
    out += " (non-resident ranks < lo; picks since each layer's last change), resets " + std::to_string(resets) + "\n";
    // the resets: each (s,d)'s last one, logged when its count has grown (the recent picks, the plain router's hits, the old leader's)
    if (s.resets_seen.size() != (size_t) s.Nsd) s.resets_seen.assign((size_t) s.Nsd, 0);
    std::vector<unsigned long long> ri((size_t) s.Nsd * kLaSlots, 0);
    if (cudaMemcpy(ri.data(), s.rinfo, ri.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost) != cudaSuccess) (void) cudaGetLastError();
    for (int sd = 0; sd < s.Nsd; ++sd) {
        const unsigned long long n_rs = cnt[(size_t) sd * kLaCnt + 16];
        if (n_rs == s.resets_seen[(size_t) sd]) continue;
        s.resets_seen[(size_t) sd] = n_rs;
        const unsigned long long* q = ri.data() + (size_t) sd * kLaSlots;
        out += "strata serve: look-ahead learner: layer " + std::to_string(sd / kLaNd) + " d=" + std::to_string(kLaDist[sd % kLaNd]) +
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
    std::snprintf(wbuf, sizeof wbuf, "strata serve: look-ahead learner: main stream waited on the batch in %llu of %llu windows%s\n",
                  s.win_waited, s.win_pending, s.timing ? "" : " (not timed: STRATA_LA_LEARN_TIMING=1 times it)");
    out += wbuf;
    if (s.timing) {
        std::snprintf(wbuf, sizeof wbuf, "strata serve: look-ahead learner: waited %.3f ms in total (measured on cs)\n", s.wait_ms_total);
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
    std::snprintf(buf, sizeof buf, "strata serve: look-ahead learner: batch on the side stream %.3f ms/window (per kernel ms: %s), "
                  "end-of-window wait bound %.3f ms/window, over %llu windows\n",
                  (double) tot_ns / 1e6 / w, per.c_str(), (double) wait_ns / 1e6 / w, nwin);
    return out + buf;
}

// The file (STRATA_LA_LEARN): "LAL1", version 1, n_layers, n_embd, 512; then per (s,d) the leader's rank (0, 4, 8 or 16); then for each
// (s,d) with a rank, the leader's A (r x n_embd) and B (512 x r) as fp16.  Written to a temporary file and renamed.
bool la_learn_save(const char* path, std::string& msg) {
    if (!la_learn_active()) { msg = "off"; return false; }
    LaLearn& s = *g_la;
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
// Loads a file saved for this model shape.  Each (s,d) takes the file's leader in its rank's slot; the other slots start from the
// leader's first rows and columns (the rest random / zero), all live, with fresh moments.  Called between requests, before any window.
bool la_learn_load(const char* path, std::string& msg) {
    if (!la_learn_active()) { msg = "off"; return false; }
    LaLearn& s = *g_la;
    FILE* f = std::fopen(path, "rb");
    if (f == nullptr) { msg = "no file"; return false; }
    uint32_t hdr[5] = {0, 0, 0, 0, 0};
    bool ok = std::fread(hdr, sizeof hdr, 1, f) == 1 && hdr[0] == 0x314C414Cu && (hdr[1] == 1u || hdr[1] == 2u || hdr[1] == 3u) &&
              (int) hdr[2] == s.L && (int) hdr[3] == s.H && hdr[4] == (uint32_t) kLaE;
    if (!ok) { std::fclose(f); msg = "not a LAL1 file for this model"; return false; }
    // the file's (s,d) list: version 3 has every dist of kLaDist; versions 1 and 2 had dists 1 and 2 only, which map onto the first two
    // entries of kLaDist (dist 4 then starts from the plain router, not frozen)
    const int file_nd = hdr[1] == 3u ? kLaNd : 2;
    const int n_file = file_nd * s.L;
    std::vector<int> rl((size_t) s.Nsd, 0), fzv((size_t) s.Nsd, 0), fsd((size_t) n_file, 0);
    for (int fi = 0; ok && fi < n_file; ++fi) {
        uint32_t w = 0;
        ok = std::fread(&w, sizeof w, 1, f) == 1;
        const uint32_t r = w & 255u;
        ok = ok && (r == 0 || r == 4 || r == 8 || r == 16) && (w >> 9) == 0;
        const int sd = (fi / file_nd) * kLaNd + fi % file_nd;
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
    s.logged.assign((size_t) s.Nsd, 0);
    s.resets_seen.assign((size_t) s.Nsd, 0);
    if (cudaGetLastError() != cudaSuccess) { msg = "the copy to the GPU failed"; return false; }
    int nfrozen = 0;
    for (int v : fzv) nfrozen += v;
    msg = "loaded, " + std::to_string(nfrozen) + " of " + std::to_string(s.Nsd) + " frozen";
    return true;
}
}
