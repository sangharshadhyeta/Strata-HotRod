// src/kernels/cuda/moe_group.cu - see include/strata/kernels/moe_group.hpp.
// The router: from the DeepSeek branch's route_kernel (src/dsv41/ds4_kernels.cu, upstream-dsv41), with a score mode, a row stride of n, none of its DS4_MASK / vector-length
// extras, and a `taken` flag (the original marks a pick with -FLT_MAX, which ties with the scan's start value, so with non-finite scores the same expert came out repeatedly).
#include "strata/kernels/moe_group.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels::moe {
namespace {

constexpr int kMaxN = 1024, kMaxK = 16;

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

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

__global__ void k_route(const float* __restrict__ logits, const float* __restrict__ bias, int n, int k, float route_scale, int score, int32_t* idx, float* w) {
    __shared__ float sc[kMaxN], bs[kMaxN];
    __shared__ float bv[32];
    __shared__ int bi[32];
    __shared__ int pick[kMaxK];
    __shared__ unsigned char taken[kMaxN];
    const int t = blockIdx.x;
    logits += (size_t) t * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float l = logits[i];
        sc[i] = score == 0 ? sigmoidf_(l) : sqrtf(l > 20.0f ? l : log1pf(expf(l)));
        bs[i] = sc[i] + bias[i];   // the selection key; the weights use sc alone
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

// ---- the grouped experts
/// UE4M3 (the NVFP4 sub-scale) with the 0.5 factor the doubled E2M1 table needs; 0 and 0x7F read as 0.
__device__ __forceinline__ float ue4m3_half(unsigned x) {
    if (x == 0u || x == 0x7Fu) return 0.0f;
    const unsigned e = (x >> 3) & 0xFu, m = x & 7u;
    return e == 0u ? (float) m * (1.0f / 1024.0f) : __uint_as_float(((e + 119u) << 23) | (m << 20));
}

/// The doubled E2M1 value of a 4-bit code (sign in bit 3): 0 1 2 3 4 6 8 12, by integer arithmetic (a table in constant memory serializes divergent lookups).
__device__ __forceinline__ float e2m1d(unsigned code) {
    const unsigned mag = code & 7u;
    const float v = mag < 2u ? (float) mag : (float) ((2u + (mag & 1u)) << ((mag >> 1) - 1u));
    return (code & 8u) ? -v : v;
}

/// A warp's dot of one NVFP4 row (36-byte blocks of 64 values: 4 sub-scales, 32 code bytes; a byte holds values j (low nibble) and 8 + j (high) of its sub-block) with xs [n] (shared).
/// Wide loads (the weights may sit across PCIe, where narrow reads cost): lane = 8 blk + w reads the 4 code bytes at 4 w of block blk (w 0..7), four blocks a step, one 32-bit load
/// a lane; sub-block w / 2, values 4 (w & 1) + j and 8 more.  Rows are 4-byte aligned (36-byte blocks, 4-byte header).
__device__ __forceinline__ float dot_nvfp4(const uint8_t* row, const float* xs, int n) {
    const int lane = threadIdx.x & 31, w = lane & 7, sbi = w >> 1, jo = 4 * (w & 1);
    const int nb = n >> 6;
    float acc = 0.0f;
#pragma unroll 4   // several blocks' loads in flight: the loop is memory-latency bound
    for (int b0 = 0; b0 < nb; b0 += 4) {
        const int b = b0 + (lane >> 3);
        if (b < nb) {
            const uint8_t* blk = row + (size_t) b * 36;
            const float d = ue4m3_half(blk[sbi]);
            const uint32_t q = *reinterpret_cast<const uint32_t*>(blk + 4 + 4 * w);
            const float* x = xs + b * 64 + sbi * 16 + jo;
            float a = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const unsigned byte = (q >> (8 * j)) & 0xFFu;
                a += e2m1d(byte & 15u) * x[j] + e2m1d(byte >> 4) * x[8 + j];
            }
            acc += d * a;
        }
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    return acc;
}
/// The same for a Q8_0 row (34-byte blocks: half scale + 32 int8; 2-byte aligned): lane = 16 blk + w, two blocks a step, a 16-bit load (2 values) a lane.
__device__ __forceinline__ float dot_q8_0(const uint8_t* row, const float* xs, int n) {
    const int lane = threadIdx.x & 31, w = lane & 15;
    const int nb = n >> 5;
    float acc = 0.0f;
#pragma unroll 4
    for (int b0 = 0; b0 < nb; b0 += 2) {
        const int b = b0 + (lane >> 4);
        if (b < nb) {
            const uint8_t* blk = row + (size_t) b * 34;
            const float d = __half2float(*reinterpret_cast<const __half*>(blk));
            const uint16_t q = *reinterpret_cast<const uint16_t*>(blk + 2 + 2 * w);
            acc += d * ((float) (signed char) (q & 0xFFu) * xs[b * 32 + 2 * w] + (float) (signed char) (q >> 8) * xs[b * 32 + 2 * w + 1]);
        }
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    return acc;
}
__device__ __forceinline__ float dot_row(int ty, const uint8_t* row, const float* xs, int n) { return ty == 40 ? dot_nvfp4(row, xs, n) : dot_q8_0(row, xs, n); }
__device__ __forceinline__ size_t row_bytes_dev(int ty, int n) { return ty == 40 ? (size_t) (n / 64) * 36 : (size_t) (n / 32) * 34; }

__device__ __forceinline__ float activate(float g, float u, int kind, float p0, float p1) {
    if (kind == 1) { g = fminf(g, p0); u = fminf(fmaxf(u, -p0), p0); return g * sigmoidf_(g) * u; }
    if (kind == 2) return tanhf(g / p0) * p0 * sigmoidf_(g) * tanhf(u / p1) * p1;
    return g * sigmoidf_(g) * u;
}

__global__ void k_prepare(const int32_t* idx, const float* rw, const uint8_t* gu_base, size_t gu_plane, const uint8_t* dn_base, size_t dn_plane, const float* gs, const float* ds, int K,
                          const uint8_t** gu_ptr, const uint8_t** dn_ptr, float* sg, float* wd) {
    const int s = threadIdx.x;
    if (s >= K) return;
    const int e = idx[s];
    gu_ptr[s] = gu_base + (size_t) e * gu_plane;
    dn_ptr[s] = dn_base + (size_t) e * dn_plane;
    sg[s] = gs ? gs[e] : 1.0f;
    wd[s] = rw[s] * (ds ? ds[e] : 1.0f);
}

__global__ void k_scales(const int32_t* idx, const float* rw, const float* gs, const float* ds, int K, float* sg, float* wd) {
    const int s = threadIdx.x;
    if (s >= K) return;
    const int e = idx[s];
    sg[s] = gs ? gs[e] : 1.0f;
    wd[s] = rw[s] * (ds ? ds[e] : 1.0f);
}

constexpr int kEW = 8, kER = 2;   // warps a block, rows a warp

// grid (ceil(FF / (kEW kER)), K), kEW * 32 threads, C floats of shared memory.
__global__ void k_gate_up(const uint8_t* const* gu_ptr, int ty, size_t up_off, const float* x, int C, int FF, const float* sg, int kind, float p0, float p1, float* h) {
    extern __shared__ float xs[];
    const int s = blockIdx.y;
    for (int i = threadIdx.x; i < C; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    const int warp = threadIdx.x >> 5;
    const uint8_t* plane = gu_ptr[s];
    const size_t rb = row_bytes_dev(ty, C);
    const float scale = sg[s];
    for (int j = 0; j < kER; ++j) {
        const int r = (blockIdx.x * kEW + warp) * kER + j;
        if (r >= FF) break;
        const float g = dot_row(ty, plane + (size_t) r * rb, xs, C) * scale;
        const float u = dot_row(ty, plane + up_off + (size_t) r * rb, xs, C) * scale;
        if ((threadIdx.x & 31) == 0) h[(size_t) s * FF + r] = activate(g, u, kind, p0, p1);
    }
}
// grid (ceil(C / (kEW kER)), K), FF floats of shared memory.
__global__ void k_down(const uint8_t* const* dn_ptr, int ty, const float* h, int C, int FF, float* y) {
    extern __shared__ float xs[];
    const int s = blockIdx.y;
    for (int i = threadIdx.x; i < FF; i += blockDim.x) xs[i] = h[(size_t) s * FF + i];
    __syncthreads();
    const int warp = threadIdx.x >> 5;
    const uint8_t* w = dn_ptr[s];
    const size_t rb = row_bytes_dev(ty, FF);
    for (int j = 0; j < kER; ++j) {
        const int r = (blockIdx.x * kEW + warp) * kER + j;
        if (r >= C) break;
        const float v = dot_row(ty, w + (size_t) r * rb, xs, FF);
        if ((threadIdx.x & 31) == 0) y[(size_t) s * C + r] = v;
    }
}
__global__ void k_combine(const float* y, const float* wd, int K, int C, float* out) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float a = 0.0f;
    for (int s = 0; s < K; ++s) a += wd[s] * y[(size_t) s * C + c];
    out[c] = a;
}


// ---- the int8-activation path: the activation quantized to q8_1 blocks (half scale, half sum, 32 int8: 36 bytes), the weights decoded by a byte-permute table lookup (the doubled
// E2M1 values as int8) and dotted with __dp4a: ~4x fewer ALU ops per weight than the exact FP32 path above.  The lookup is the one the engine's iq_kernels.cu uses (ggml's
// get_int_from_table_16, MIT); the NVFP4 word layout is the same as there (word w of a block: sub-block w / 2; low nibbles -> activation ints 2 (w & 1) .. , high nibbles -> +2).
__constant__ uint32_t kFp4Tab[4] = {0x03020100u, 0x0C080604u, 0xFDFEFF00u, 0xF4F8FAFCu};   // 0 1 2 3 4 6 8 12, 0 -1 -2 -3 -4 -6 -8 -12 (int8)
__device__ __forceinline__ int2 tbl16(uint32_t q4) {
    uint32_t tmp[2];
    const uint32_t sel = 0x32103210u | ((q4 & 0x88888888u) >> 1);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t sh = 16 * i;
        const uint32_t lo = __byte_perm(kFp4Tab[0], kFp4Tab[1], q4 >> sh);
        const uint32_t hi = __byte_perm(kFp4Tab[2], kFp4Tab[3], q4 >> sh);
        tmp[i] = __byte_perm(lo, hi, sel >> sh);
    }
    return make_int2((int) __byte_perm(tmp[0], tmp[1], 0x6420), (int) __byte_perm(tmp[0], tmp[1], 0x7531));
}

/// A warp's dot of one NVFP4 row with the q8_1-quantized activation xq [n / 32][36 bytes] (shared).
__device__ __forceinline__ float dot_nvfp4_q8(const uint8_t* row, const uint8_t* xq, int n) {
    const int lane = threadIdx.x & 31, w = lane & 7, is = w >> 1, i8 = (is & 1) * 4 + (w & 1);
    const int nb = n >> 6;
    float acc = 0.0f;
#pragma unroll 4
    for (int b0 = 0; b0 < nb; b0 += 4) {
        const int b = b0 + (lane >> 3);
        if (b < nb) {
            const uint8_t* blk = row + (size_t) b * 36;
            const uint8_t* xb = xq + (size_t) (b * 2 + (is >> 1)) * 36;
            const int2 v = tbl16(*reinterpret_cast<const uint32_t*>(blk + 4 + 4 * w));
            const int* xi = reinterpret_cast<const int*>(xb + 4);
            int sm = __dp4a(v.x, xi[i8], 0);
            sm = __dp4a(v.y, xi[i8 + 2], sm);
            acc += ue4m3_half(blk[is]) * __half2float(*reinterpret_cast<const __half*>(xb)) * (float) sm;
        }
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    return acc;
}
/// The same for a Q8_0 row (34-byte blocks, 2-byte aligned): lane = 8 blk + w, four blocks a step, 4 int8 a lane.
__device__ __forceinline__ float dot_q8_0_q8(const uint8_t* row, const uint8_t* xq, int n) {
    const int lane = threadIdx.x & 31, w = lane & 7;
    const int nb = n >> 5;
    float acc = 0.0f;
#pragma unroll 4
    for (int b0 = 0; b0 < nb; b0 += 4) {
        const int b = b0 + (lane >> 3);
        if (b < nb) {
            const uint8_t* blk = row + (size_t) b * 34;
            const uint8_t* xb = xq + (size_t) b * 36;
            const uint32_t wq = (uint32_t) *reinterpret_cast<const uint16_t*>(blk + 2 + 4 * w) | ((uint32_t) *reinterpret_cast<const uint16_t*>(blk + 4 + 4 * w) << 16);
            const int sm = __dp4a((int) wq, reinterpret_cast<const int*>(xb + 4)[w], 0);
            acc += __half2float(*reinterpret_cast<const __half*>(blk)) * __half2float(*reinterpret_cast<const __half*>(xb)) * (float) sm;
        }
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    return acc;
}
__device__ __forceinline__ float dot_row_q8(int ty, const uint8_t* row, const uint8_t* xq, int n) { return ty == 40 ? dot_nvfp4_q8(row, xq, n) : dot_q8_0_q8(row, xq, n); }

// one warp a 32-value block: d = max|x| / 127, q = round(x / d)
__global__ void k_quant(const float* x, int nblk, uint8_t* xq) {
    const int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (wid >= nblk) return;
    const float v = x[(size_t) wid * 32 + lane];
    float am = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
    const float d = am / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
    uint8_t* b = xq + (size_t) wid * 36;
    reinterpret_cast<signed char*>(b + 4)[lane] = (signed char) __float2int_rn(v * id);
    if (lane == 0) { *reinterpret_cast<__half*>(b) = __float2half(d); *reinterpret_cast<__half*>(b + 2) = __float2half(0.0f); }
}

__global__ void k_gate_up_q8(const uint8_t* const* gu_ptr, int ty, size_t up_off, const uint8_t* xq, int C, int FF, const float* sg, int kind, float p0, float p1, float* h) {
    extern __shared__ __align__(16) uint8_t xs8[];
    const int s = blockIdx.y;
    const int xb = (C >> 5) * 36;
    for (int i = threadIdx.x; i < xb / 4; i += blockDim.x) reinterpret_cast<int*>(xs8)[i] = reinterpret_cast<const int*>(xq)[i];
    __syncthreads();
    const int warp = threadIdx.x >> 5;
    const uint8_t* plane = gu_ptr[s];
    const size_t rb = row_bytes_dev(ty, C);
    const float scale = sg[s];
    for (int j = 0; j < kER; ++j) {
        const int r = (blockIdx.x * kEW + warp) * kER + j;
        if (r >= FF) break;
        const float g = dot_row_q8(ty, plane + (size_t) r * rb, xs8, C) * scale;
        const float u = dot_row_q8(ty, plane + up_off + (size_t) r * rb, xs8, C) * scale;
        if ((threadIdx.x & 31) == 0) h[(size_t) s * FF + r] = activate(g, u, kind, p0, p1);
    }
}
__global__ void k_down_q8(const uint8_t* const* dn_ptr, int ty, const uint8_t* hq, int C, int FF, float* y) {
    extern __shared__ __align__(16) uint8_t xs8[];
    const int s = blockIdx.y;
    const int xb = (FF >> 5) * 36;
    for (int i = threadIdx.x; i < xb / 4; i += blockDim.x) reinterpret_cast<int*>(xs8)[i] = reinterpret_cast<const int*>(hq + (size_t) s * xb)[i];
    __syncthreads();
    const int warp = threadIdx.x >> 5;
    const uint8_t* w = dn_ptr[s];
    const size_t rb = row_bytes_dev(ty, FF);
    for (int j = 0; j < kER; ++j) {
        const int r = (blockIdx.x * kEW + warp) * kER + j;
        if (r >= C) break;
        const float v = dot_row_q8(ty, w + (size_t) r * rb, xs8, FF);
        if ((threadIdx.x & 31) == 0) y[(size_t) s * C + r] = v;
    }
}

}  // namespace

bool type_supported(int ty) { return ty == 40 || ty == 8; }

void route(const float* logits, const float* bias, int T, int n, int k, float route_scale, Score score, int32_t* idx, float* w, void* stream) {
    if (T <= 0) return;
    if (n <= 0 || n > kMaxN || k <= 0 || k > kMaxK || k > n) {
        std::fprintf(stderr, "moe::route: n=%d k=%d outside 1..%d / 1..%d\n", n, k, kMaxN, kMaxK);
        std::exit(1);
    }
    k_route<<<T, n <= 128 ? 128 : 512, 0, (cudaStream_t) stream>>>(logits, bias, n, k, route_scale, (int) score, idx, w);
}
void experts_prepare(const int32_t* idx, const float* route_w, const uint8_t* gu_base, size_t gu_plane, const uint8_t* dn_base, size_t dn_plane, const float* gate_up_scale,
                     const float* down_scale, int K, const uint8_t** gu_ptr, const uint8_t** dn_ptr, float* sg, float* wd, void* stream) {
    k_prepare<<<1, 32, 0, (cudaStream_t) stream>>>(idx, route_w, gu_base, gu_plane, dn_base, dn_plane, gate_up_scale, down_scale, K, gu_ptr, dn_ptr, sg, wd);
}
void experts_scales(const int32_t* idx, const float* route_w, const float* gate_up_scale, const float* down_scale, int K, float* sg, float* wd, void* stream) {
    k_scales<<<1, 32, 0, (cudaStream_t) stream>>>(idx, route_w, gate_up_scale, down_scale, K, sg, wd);
}
void experts_gate_up(const uint8_t* const* gu_ptr, int ty, size_t up_off, const float* x, int C, int FF, const float* sg, Act act, float* h, int K, void* stream) {
    if (!type_supported(ty)) { std::fprintf(stderr, "moe::experts_gate_up: ggml type %d has no decode here\n", ty); std::exit(1); }
    k_gate_up<<<dim3((FF + kEW * kER - 1) / (kEW * kER), K), kEW * 32, (size_t) C * 4, (cudaStream_t) stream>>>(gu_ptr, ty, up_off, x, C, FF, sg, (int) act.kind, act.p0, act.p1, h);
}
void experts_down(const uint8_t* const* dn_ptr, int ty, const float* h, int C, int FF, float* y, int K, void* stream) {
    if (!type_supported(ty)) { std::fprintf(stderr, "moe::experts_down: ggml type %d has no decode here\n", ty); std::exit(1); }
    k_down<<<dim3((C + kEW * kER - 1) / (kEW * kER), K), kEW * 32, (size_t) FF * 4, (cudaStream_t) stream>>>(dn_ptr, ty, h, C, FF, y);
}
void quantize_act(const float* x, int n, int rows, void* xq, void* stream) {
    const int nblk = rows * (n / 32);
    k_quant<<<(nblk * 32 + 255) / 256, 256, 0, (cudaStream_t) stream>>>(x, nblk, (uint8_t*) xq);
}
void experts_gate_up_q8(const uint8_t* const* gu_ptr, int ty, size_t up_off, const void* xq, int C, int FF, const float* sg, Act act, float* h, int K, void* stream) {
    if (!type_supported(ty)) { std::fprintf(stderr, "moe::experts_gate_up_q8: ggml type %d has no decode here\n", ty); std::exit(1); }
    k_gate_up_q8<<<dim3((FF + kEW * kER - 1) / (kEW * kER), K), kEW * 32, (size_t) (C / 32) * 36, (cudaStream_t) stream>>>(gu_ptr, ty, up_off, (const uint8_t*) xq, C, FF, sg, (int) act.kind, act.p0, act.p1, h);
}
void experts_down_q8(const uint8_t* const* dn_ptr, int ty, const void* hq, int C, int FF, float* y, int K, void* stream) {
    if (!type_supported(ty)) { std::fprintf(stderr, "moe::experts_down_q8: ggml type %d has no decode here\n", ty); std::exit(1); }
    k_down_q8<<<dim3((C + kEW * kER - 1) / (kEW * kER), K), kEW * 32, (size_t) (FF / 32) * 36, (cudaStream_t) stream>>>(dn_ptr, ty, (const uint8_t*) hq, C, FF, y);
}
void experts_combine(const float* y, const float* wd, int K, int C, float* out, void* stream) {
    k_combine<<<(C + 255) / 256, 256, 0, (cudaStream_t) stream>>>(y, wd, K, C, out);
}

}  // namespace strata::kernels::moe
