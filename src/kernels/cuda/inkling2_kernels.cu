// src/kernels/cuda/inkling2_kernels.cu - see include/strata/kernels/inkling2_kernels.hpp.  Float32, correctness first.
#include "strata/kernels/inkling2_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels::ik {
namespace {

constexpr int kT = 128;   // threads of the one-block-a-row kernels (D = 128, and a multiple of 32)

/// The block's sum of `v` (every thread gets it); `sm` holds 32 floats.
__device__ __forceinline__ float block_sum(float v, float* sm) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) sm[w] = v;
    __syncthreads();
    float s = 0.0f;
    for (int i = 0; i < (int) (blockDim.x >> 5); ++i) s += sm[i];
    return s;
}
__device__ __forceinline__ float block_max(float v, float* sm) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) sm[w] = v;
    __syncthreads();
    float m = -FLT_MAX;
    for (int i = 0; i < (int) (blockDim.x >> 5); ++i) m = fmaxf(m, sm[i]);
    return m;
}
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

__global__ void k_matmul(const float* __restrict__ W, const float* __restrict__ x, float* __restrict__ y, int in, int out) {
    __shared__ float sm[32];
    const int o = blockIdx.x, t = blockIdx.y;
    const float* w = W + (size_t) o * in;
    const float* xx = x + (size_t) t * in;
    float s = 0.0f;
    for (int i = threadIdx.x; i < in; i += blockDim.x) s += w[i] * xx[i];
    s = block_sum(s, sm);
    if (threadIdx.x == 0) y[(size_t) t * out + o] = s;
}

// grid (H + KV, T), D threads.  Blocks b < H: query head b (and its r); the rest: key / value head b - H.
__global__ void k_qkv_prep(float* q, float* k, float* v, float* r, const float* qclip, const float* rclip, int H, int KV, int D, int R, float qk_clip, float rel_clip,
                           float eps, int full, float alpha, float log_floor, int pos0) {
    __shared__ float sm[32];
    const int t = blockIdx.y, b = blockIdx.x, tid = threadIdx.x;
    const float tau = full ? 1.0f + alpha * logf(fmaxf(1.0f, (float) (pos0 + t + 1) / log_floor)) : 1.0f;
    if (b < H) {
        float* qh = q + ((size_t) t * H + b) * D;
        const float val = qh[tid] * qclip[(size_t) b * D + tid];
        const float ss = block_sum(val * val, sm);
        qh[tid] = val * fminf(1.0f, qk_clip / (sqrtf(ss) + 1e-6f)) * tau;
        float* rh = r + ((size_t) t * H + b) * R;
        const float rv = tid < R ? rh[tid] * rclip[(size_t) b * R + tid] : 0.0f;
        const float rs = block_sum(rv * rv, sm);
        if (tid < R) rh[tid] = rv * fminf(1.0f, rel_clip / (sqrtf(rs) + 1e-6f)) * tau;
    } else {
        const int h = b - H;
        float* kh = k + ((size_t) t * KV + h) * D;
        const float kv = kh[tid];
        const float ks = block_sum(kv * kv, sm);
        kh[tid] = kv * fminf(1.0f, qk_clip / (sqrtf(ks) + 1e-6f));
        float* vh = v + ((size_t) t * KV + h) * D;
        const float vv = vh[tid];
        const float vs = block_sum(vv * vv, sm);
        vh[tid] = vv * rsqrtf(vs / (float) D + eps);
    }
}

__global__ void k_conv_dyn(float* state, const float* x, const float* f, const float* dyn, float* out, int C, int taps, int heads) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int h = c / (C / heads);
    float* st = state + (size_t) c * (taps - 1);   // st[0] the oldest of the last taps - 1 inputs, st[taps - 2] the newest
    const float* fw = f + (size_t) c * taps;
    const float xc = x[c];
    float acc = 0.0f;
    for (int j = 0; j < taps - 1; ++j) acc += fw[j] * st[j];
    acc += fw[taps - 1] * xc;
    acc += dyn[h * taps] * xc;                                              // lag 0: the current token
    for (int l = 1; l < taps; ++l) acc += dyn[h * taps + l] * st[taps - 1 - l];   // lag l: x[t - l]
    out[c] = acc;
    for (int j = 0; j < taps - 2; ++j) st[j] = st[j + 1];
    st[taps - 2] = xc;
}

// grid H, kT threads.
__global__ void k_attn(const float* q, const float* r, const float* gate, const float* sink, const float* Kc, const float* Vc, const float* rel_bias, float* out,
                       float* scratch, int pos0, int window, int extent, int cap, int H, int KV, int D, int R) {
    __shared__ float sm[32];
    const int h = blockIdx.x, kvh = h / (H / KV), tk = blockIdx.y, pos = pos0 + tk;
    q += (size_t) tk * H * D; r += (size_t) tk * H * R; gate += (size_t) tk * H * D; sink += (size_t) tk * H; out += (size_t) tk * H * D; scratch += (size_t) tk * H * cap;
    const int s0 = window > 0 ? max(0, pos - window + 1) : 0;
    const int n = pos - s0 + 1;
    float* lg = scratch + (size_t) h * cap;
    const float* qh = q + (size_t) h * D;
    const float* rh = r + (size_t) h * R;
    const float inv = rsqrtf((float) D);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int s = s0 + i, dist = pos - s;
        const float* kk = Kc + ((size_t) s * KV + kvh) * D;
        float d = 0.0f;
        for (int j = 0; j < D; ++j) d += qh[j] * kk[j];
        float bias = 0.0f;
        if (dist < extent) {
            const float* pr = rel_bias + (size_t) dist * R;
            for (int j = 0; j < R; ++j) bias += rh[j] * pr[j];
        }
        lg[i] = d * inv + bias;
    }
    __syncthreads();
    float mx = sink[h];
    for (int i = threadIdx.x; i < n; i += blockDim.x) mx = fmaxf(mx, lg[i]);
    mx = block_max(mx, sm);
    float z = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) { lg[i] = expf(lg[i] - mx); z += lg[i]; }
    z = block_sum(z, sm) + expf(sink[h] - mx);
    __syncthreads();
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < n; ++i) acc += lg[i] * Vc[((size_t) (s0 + i) * KV + kvh) * D + d];
        out[(size_t) h * D + d] = acc / z * sigmoidf_(gate[(size_t) h * D + d]);
    }
}

// grid nq, 256 threads: states [n][C] (n <= 16).
__global__ void k_attn_res(const float* states, int n, const float* q, float* out, int C, size_t st_tok, size_t out_tok, size_t out_q) {
    __shared__ float sm[32];
    __shared__ float rs[16], p[16];
    const int k = blockIdx.x;
    states += (size_t) blockIdx.y * st_tok;
    const float* qq = q + (size_t) k * C;
    for (int i = 0; i < n; ++i) {
        const float* s = states + (size_t) i * C;
        float ss = 0.0f, dt = 0.0f;
        for (int c = threadIdx.x; c < C; c += blockDim.x) { ss += s[c] * s[c]; dt += qq[c] * s[c]; }
        ss = block_sum(ss, sm);
        dt = block_sum(dt, sm);
        const float r = rsqrtf(ss / (float) C + 1e-6f);
        if (threadIdx.x == 0) { rs[i] = r; p[i] = dt * r / sqrtf((float) C); }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float mx = -FLT_MAX;
        for (int i = 0; i < n; ++i) mx = fmaxf(mx, p[i]);
        float z = 0.0f;
        for (int i = 0; i < n; ++i) { p[i] = expf(p[i] - mx); z += p[i]; }
        for (int i = 0; i < n; ++i) p[i] /= z;
    }
    __syncthreads();
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < n; ++i) acc += p[i] * states[(size_t) i * C + c] * rs[i];
        out[(size_t) blockIdx.y * out_tok + (size_t) k * out_q + c] = acc;
    }
}

__global__ void k_control_cat(const float* x, const float* msg, float* cat, int C) {
    __shared__ float sm[32];
    x += (size_t) blockIdx.x * C; msg += (size_t) blockIdx.x * C; cat += (size_t) blockIdx.x * 2 * C;
    float sx = 0.0f, sg = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) { sx += x[c] * x[c]; sg += msg[c] * msg[c]; }
    sx = block_sum(sx, sm);
    sg = block_sum(sg, sm);
    const float rx = rsqrtf(sx / (float) C + 1e-6f), rg = rsqrtf(sg / (float) C + 1e-6f);
    for (int c = threadIdx.x; c < C; c += blockDim.x) { cat[c] = x[c] * rx; cat[C + c] = msg[c] * rg; }
}
__global__ void k_control_mix(const float* x, const float* msg, const float* fg, const float* gn, float* y, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < C) y[c] = (1.0f - tanhf(fg[c])) * x[c] + tanhf(gn[c]) * msg[c];
}
__global__ void k_situ(const float* g, const float* u, float* h, int n, float s, float beta, float gamma) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gt = g[i] * s, up = u[i] * s;
    h[i] = tanhf(gt / beta) * beta * sigmoidf_(gt) * tanhf(up / gamma) * gamma;
}
__global__ void k_axpy(float* acc, const float* y, float a, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) acc[i] += a * y[i];
}
__global__ void k_add(const float* a, const float* b, float* o, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = a[i] + b[i];
}
__global__ void k_softcap(float* x, int n, float cap) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = tanhf(x[i] / cap) * cap;
}
__global__ void k_deq_q8_0(const unsigned char* row, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned char* b = row + (size_t) (i / 32) * 34;
    const float d = __half2float(*reinterpret_cast<const __half*>(b));
    out[i] = d * (float) reinterpret_cast<const signed char*>(b + 2)[i % 32];
}

inline int blocks(int n, int t = 256) { return (n + t - 1) / t; }

}  // namespace

void matmul_f32(const float* W, const float* x, float* y, int T, int in, int out, void* stream) {
    k_matmul<<<dim3(out, T), kT, 0, (cudaStream_t) stream>>>(W, x, y, in, out);
}
void qkv_prep(float* q, float* k, float* v, float* r, const float* qclip, const float* rclip, int T, int H, int KV, int D, int R, float qk_clip, float rel_clip, float eps,
              bool full, float alpha, float log_floor, int pos0, void* stream) {
    if (D != kT) { std::fprintf(stderr, "ik::qkv_prep: head_dim %d (the kernel runs one thread a dimension: %d)\n", D, kT); std::exit(1); }
    k_qkv_prep<<<dim3(H + KV, T), kT, 0, (cudaStream_t) stream>>>(q, k, v, r, qclip, rclip, H, KV, D, R, qk_clip, rel_clip, eps, full ? 1 : 0, alpha, log_floor, pos0);
}
void conv_dynamic(float* state, const float* x, const float* f, const float* dyn, float* out, int C, int taps, int heads, void* stream) {
    k_conv_dyn<<<blocks(C), 256, 0, (cudaStream_t) stream>>>(state, x, f, dyn, out, C, taps, heads);
}
void kv_append(float* Kc, float* Vc, const float* k, const float* v, int pos, int KV, int D, void* stream, int T) {
    cudaMemcpyAsync(Kc + (size_t) pos * KV * D, k, (size_t) T * KV * D * 4, cudaMemcpyDeviceToDevice, (cudaStream_t) stream);
    cudaMemcpyAsync(Vc + (size_t) pos * KV * D, v, (size_t) T * KV * D * 4, cudaMemcpyDeviceToDevice, (cudaStream_t) stream);
}
void attention(const float* q, const float* r, const float* gate, const float* sink, const float* Kc, const float* Vc, const float* rel_bias, float* out, float* scratch, int pos,
               int window, int extent, int cap, int H, int KV, int D, int R, void* stream, int T) {
    k_attn<<<dim3(H, T), kT, 0, (cudaStream_t) stream>>>(q, r, gate, sink, Kc, Vc, rel_bias, out, scratch, pos, window, extent, cap, H, KV, D, R);
}
void attn_res(const float* states, int n, const float* q, int nq, float* out, int C, void* stream, int T, size_t st_tok, size_t out_tok, size_t out_q) {
    if (n > 16) { std::fprintf(stderr, "ik::attn_res: %d states (at most 16)\n", n); std::exit(1); }
    k_attn_res<<<dim3(nq, T), 256, 0, (cudaStream_t) stream>>>(states, n, q, out, C, st_tok, out_tok ? out_tok : (size_t) nq * C, out_q ? out_q : (size_t) C);
}
void control_cat(const float* x, const float* msg, float* cat, int C, void* stream, int T) { k_control_cat<<<T, 256, 0, (cudaStream_t) stream>>>(x, msg, cat, C); }
void control_mix(const float* x, const float* msg, const float* forget, const float* gain, float* y, int C, void* stream) {
    k_control_mix<<<blocks(C), 256, 0, (cudaStream_t) stream>>>(x, msg, forget, gain, y, C);
}
void situ(const float* g, const float* u, float* h, int n, float s, float beta, float gamma, void* stream) { k_situ<<<blocks(n), 256, 0, (cudaStream_t) stream>>>(g, u, h, n, s, beta, gamma); }
void axpy(float* acc, const float* y, float a, int n, void* stream) { k_axpy<<<blocks(n), 256, 0, (cudaStream_t) stream>>>(acc, y, a, n); }
void add(const float* a, const float* b, float* out, int n, void* stream) { k_add<<<blocks(n), 256, 0, (cudaStream_t) stream>>>(a, b, out, n); }
void softcap(float* x, int n, float cap, void* stream) { k_softcap<<<blocks(n), 256, 0, (cudaStream_t) stream>>>(x, n, cap); }
void dequant_q8_0_row(const void* row, float* out, int n, void* stream) {
    k_deq_q8_0<<<blocks(n), 256, 0, (cudaStream_t) stream>>>(reinterpret_cast<const unsigned char*>(row), out, n);
}

}  // namespace strata::kernels::ik
