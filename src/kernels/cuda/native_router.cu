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
#include <cstdlib>
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

// STRATA_ROUTE_RESIDENT (EXPERIMENTAL, opt-in, changes the output): residency-biased routing for 512 x 10.
// After the top-10: for the ranks lo..hi a non-resident expert is replaced by the best resident expert not already
// picked when its logit is within `margin` of the replaced one; the weights are then the softmax of the selected
// logits (the router's renormalisation).  Tokens without a swap keep the router's exact bits.
// stats (device, 4 x uint64): [0] tail entries seen, [1] swaps, [2] non-resident entries before, [3] after.
static unsigned long long* g_detail = nullptr;   // the decode rows' detail counters (route_detail_alloc), see below
__global__ void route_resident_k(const float* __restrict__ logits, int32_t* __restrict__ ids, float* __restrict__ weights,
                                 const int32_t* __restrict__ res, int n_tok, const RouteCfg* cfg,
                                 unsigned long long* __restrict__ stats, float* __restrict__ want,
                                 unsigned long long* __restrict__ detail) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_tok) return;
    // the floors live in host-mapped memory, read at every launch, so a captured graph follows a change the host makes between
    // windows (SPEC 17): the floor in force now (the host sets it from the part being written), the margin and the edge margin
    const int lo = ((const volatile RouteCfg*) cfg)->eff;
    const int hi = ((const volatile RouteCfg*) cfg)->hi;
    const float margin = ((const volatile RouteCfg*) cfg)->margin;
    const float edge = ((const volatile RouteCfg*) cfg)->edge;
    const float* l = logits + (size_t) t * 512;
    int32_t* id = ids + (size_t) t * 10;
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
        for (int r = 0; r < 10; ++r) atomicAdd(want + id[r], 1.0f);
    int swaps = 0, tail = 0, before = 0;
    for (int r = 0; r < 10; ++r) before += res[id[r]] < 0;
    for (int r = (edge >= 0.f && lo > 0) ? lo - 1 : lo; r <= hi && r < 10; ++r) {
        const int e = id[r];
        if (res[e] >= 0) continue;
        ++tail;
        int best = -1;
        float bl = -INFINITY;
        for (int f = 0; f < 512; ++f) {
            if (res[f] < 0 || l[f] <= bl) continue;
            bool used = false;
            for (int q = 0; q < 10; ++q) used |= id[q] == f;
            if (!used) { best = f; bl = l[f]; }
        }
        // the edge rank (lo - 1, a fractional floor such as 1.5) is replaced only by a close substitute (within `edge`)
        if (best >= 0 && l[e] - bl <= (r < lo ? edge : margin)) {
            if (detail != nullptr) {
                // what was replaced and by what (per replacement): the weight of the replaced pick, the substitute's place in
                // the router's own ranking of all 512 experts, and the score gap to it
                const float w = weights[(size_t) t * 10 + r];
                const int wb = w < 0.01f ? 0 : w < 0.02f ? 1 : w < 0.05f ? 2 : w < 0.10f ? 3 : w < 0.20f ? 4 : 5;
                int gr = 0;
                for (int f = 0; f < 512; ++f) gr += l[f] > bl;
                const int rb = gr < 20 ? 0 : gr < 40 ? 1 : gr < 80 ? 2 : gr < 160 ? 3 : 4;
                const float gap = l[e] - bl;
                const int gb = gap < 0.5f ? 0 : gap < 1.f ? 1 : gap < 2.f ? 2 : gap < 4.f ? 3 : gap < 8.f ? 4 : 5;
                atomicAdd(detail + 16 + wb, 1ull);
                atomicAdd(detail + 22 + rb, 1ull);
                atomicAdd(detail + 27 + gb, 1ull);
            }
            id[r] = best;
            ++swaps;
        }
    }
    int after = 0;
    for (int r = 0; r < 10; ++r) after += res[id[r]] < 0;
    if (swaps) {
        float m = -INFINITY;
        for (int r = 0; r < 10; ++r) m = fmaxf(m, l[id[r]]);
        float ex[10], sum = 0.0f;
        for (int r = 0; r < 10; ++r) { ex[r] = expf(l[id[r]] - m); sum += ex[r]; }
        for (int r = 0; r < 10; ++r) weights[(size_t) t * 10 + r] = ex[r] / sum;
    }
    if (stats) {
        atomicAdd(stats + 0, (unsigned long long) tail);
        atomicAdd(stats + 1, (unsigned long long) swaps);
        atomicAdd(stats + 2, (unsigned long long) before);
        atomicAdd(stats + 3, (unsigned long long) after);
    }
}
// ---- the floors (SPEC 17): one small struct in host-mapped memory, read by the kernel at every launch.  The host sets it once
// (route_cfg_init: the floors of thinking, of the answer and of code) and moves `eff`, the floor in force, between windows
// (route_cfg_set_state). A higher floor replaces less. Inside a code fence the floor is never less careful than the code floor,
// so a fence can only make the stand-in more careful, never less.
static RouteCfg* g_cfg_host = nullptr;
static RouteCfg* g_cfg_dev = nullptr;
static int g_lo_default[3] = {0, 0, 0};   // the floors of the environment, restored for a request that gives none
// STRATA_STANDIN_PARTS=1 turns the per-part floors on (a request's floors, the thinking / fence tracker). Off (the default): ONE
// floor, the environment's, for every request and every part of an answer; a request's floors and the tracker are ignored.
static bool g_parts = false;
bool route_cfg_parts() { return g_parts; }
static int eff_floor(const RouteCfg* c, bool in_think, bool in_fence) {
    const int base = in_think ? c->lo[0] : c->lo[1];
    return in_fence ? (base > c->lo[2] ? base : c->lo[2]) : base;
}
void route_cfg_request(int lo_think, int lo_prose, int lo_code, bool in_think) {
    if (g_cfg_host == nullptr) return;     // the stand-in is off: nothing to set
    if (!g_parts) { lo_think = lo_prose = lo_code = -1; in_think = false; }   // one floor for everything: the request's are ignored
    g_cfg_host->lo[0] = lo_think >= 0 ? lo_think : g_lo_default[0];
    g_cfg_host->lo[1] = lo_prose >= 0 ? lo_prose : g_lo_default[1];
    g_cfg_host->lo[2] = lo_code >= 0 ? lo_code : g_lo_default[2];
    g_cfg_host->eff = eff_floor(g_cfg_host, in_think, false);
    __sync_synchronize();
}
void route_cfg_init(int lo_think, int lo_prose, int lo_code, int hi, float margin, float edge) {
    if (g_cfg_host == nullptr) {
        if (cudaHostAlloc((void**) &g_cfg_host, sizeof(RouteCfg), cudaHostAllocMapped) != cudaSuccess) { g_cfg_host = nullptr; return; }
        if (cudaHostGetDevicePointer((void**) &g_cfg_dev, g_cfg_host, 0) != cudaSuccess) { g_cfg_dev = nullptr; return; }
    }
    const char* parts = std::getenv("STRATA_STANDIN_PARTS");
    g_parts = parts != nullptr && std::atoi(parts) != 0;
    if (!g_parts) lo_think = lo_code = lo_prose;   // one floor for everything, whatever STRATA_ROUTE_FLOORS says
    g_cfg_host->lo[0] = g_lo_default[0] = lo_think;
    g_cfg_host->lo[1] = g_lo_default[1] = lo_prose;
    g_cfg_host->lo[2] = g_lo_default[2] = lo_code;
    g_cfg_host->hi = hi;
    g_cfg_host->margin = margin;
    g_cfg_host->edge = edge;
    g_cfg_host->eff = eff_floor(g_cfg_host, false, false);   // the answer's floor until a request or the tracker says otherwise
    __sync_synchronize();
}
void route_cfg_set_state(bool in_think, bool in_fence) {
    if (g_cfg_host == nullptr) return;
    g_cfg_host->eff = eff_floor(g_cfg_host, in_think, in_fence);
    __sync_synchronize();
}
const RouteCfg* route_cfg_device() { return g_cfg_dev; }
void native_route_resident(const float* logits, int32_t* ids, float* weights, const int32_t* res_layer, int n_tok,
                           unsigned long long* stats, void* stream, float* want) {
    if (!stream || n_tok < 1 || !res_layer || g_cfg_dev == nullptr) throw std::invalid_argument("native_route_resident: bad arguments");
    route_resident_k<<<(unsigned) ((n_tok + 31) / 32), 32, 0, static_cast<cudaStream_t>(stream)>>>(
        logits, ids, weights, res_layer, n_tok, g_cfg_dev, stats, want, n_tok <= 8 ? g_detail : nullptr);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void route_detail_alloc() {
    if (g_detail != nullptr) return;
    if (cudaMalloc((void**) &g_detail, 48 * sizeof(unsigned long long)) != cudaSuccess) { g_detail = nullptr; return; }
    cudaMemset(g_detail, 0, 48 * sizeof(unsigned long long));
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
}
float* route_want_layer(int64_t layer, int64_t n_expert) { return g_want ? g_want + (size_t) (layer * n_expert) : nullptr; }
bool route_want_active() { return g_want != nullptr; }
// A pick counted between the copy and the clear is lost: the counts are approximate by nature.
void route_want_drain(float* usage, size_t n) {
    if (g_want == nullptr || usage == nullptr || n < g_want_n) return;
    std::vector<float> h(g_want_n);
    if (cudaMemcpy(h.data(), g_want, g_want_n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) return;
    cudaMemset(g_want, 0, g_want_n * sizeof(float));
    for (size_t i = 0; i < g_want_n; ++i) usage[i] += h[i];
}
}
