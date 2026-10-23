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
#include <cuda_runtime.h>
#include <atomic>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#include <string>
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
    const size_t n = (size_t) n_layers * 2 * 8 * 10;
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
int32_t* route_pred_dev(int64_t target_layer, int dist) { return g_pred_dev ? g_pred_dev + ((size_t) target_layer * 2 + (size_t) (dist - 1)) * 80 : nullptr; }
float* route_pred_logits() { return g_pred_logits; }
float* route_pred_weights() { return g_pred_w; }
// STRATA_LA_MODEL=<file> (opt-in): the published per-layer d=1 linear predictors (tools/predictor/export_la_model.py folds their
// standardisation in): for each target layer a bf16 [512 x H] matrix and a float bias, applied to the previous layer's router input.
static uint16_t* g_lam_w = nullptr;
static float* g_lam_b = nullptr;
static std::vector<char> g_lam_has;
static int64_t g_lam_h = 0;
bool route_pred_model_load(const char* path, int64_t n_layers, int64_t n_embd, std::string& msg) {
    if (g_lam_w != nullptr) {
        msg = "already loaded";
        return g_lam_h == n_embd && (int64_t) g_lam_has.size() == n_layers;
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
    int64_t n = 0;
    for (char c : g_lam_has) n += c;
    msg = std::to_string(n) + " layers, " + std::to_string((w.size() * 2 + b.size() * 4) >> 20) + " MiB of VRAM";
    return true;
}
const uint16_t* route_pred_model_w(int64_t target_layer) {
    if (g_lam_w == nullptr || target_layer < 0 || target_layer >= (int64_t) g_lam_has.size() || !g_lam_has[(size_t) target_layer]) return nullptr;
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
}
