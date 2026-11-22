// src/inkling2/drafter.cu - see include/strata/inkling2/drafter.hpp.
#include "strata/inkling2/drafter.hpp"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <vector>

namespace strata::inkling2 {
namespace {

constexpr int kBlock = 8, kHD = 128, kLayers = 3, kTaps = 4, kR = 256, kMixRows = 512;
constexpr int kTapVisits[3] = {5, 23, 41};

void cu(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("drafter: ") + what + ": " + cudaGetErrorString(e));
}
void cb(cublasStatus_t e, const char* what) {
    if (e != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string("drafter: ") + what + ": cuBLAS " + std::to_string((int) e));
}
template <class T>
T* dalloc(size_t n) {
    T* p = nullptr;
    cu(cudaMalloc(&p, std::max<size_t>(1, n) * sizeof(T)), "cudaMalloc");
    return p;
}
__host__ __device__ inline float bf(uint16_t b) { const uint32_t u = (uint32_t) b << 16; float f; memcpy(&f, &u, 4); return f; }

// ---- safetensors
struct StTensor { std::string dtype; std::vector<int64_t> shape; uint64_t begin = 0, end = 0; };
struct StFile { std::string path; uint64_t data0 = 0; std::map<std::string, StTensor> t; };
StFile st_open(const std::string& path) {
    StFile f;
    f.path = path;
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) throw std::runtime_error("drafter: cannot open " + path);
    uint64_t n = 0;
    if (std::fread(&n, 8, 1, fp) != 1 || n > (64u << 20)) throw std::runtime_error("drafter: not a safetensors file");
    std::string h(n, '\0');
    if (std::fread(&h[0], 1, n, fp) != n) throw std::runtime_error("drafter: short header");
    std::fclose(fp);
    f.data0 = 8 + n;
    size_t i = h.find('{') + 1;
    while (true) {
        const size_t q0 = h.find('"', i);
        if (q0 == std::string::npos) break;
        const size_t q1 = h.find('"', q0 + 1);
        const std::string name = h.substr(q0 + 1, q1 - q0 - 1);
        const size_t ob = h.find('{', q1), oe = h.find('}', ob);
        const std::string body = h.substr(ob, oe - ob);
        i = oe + 1;
        if (name == "__metadata__") continue;
        StTensor t;
        const size_t d0 = body.find("\"dtype\"");
        const size_t d1 = body.find('"', body.find(':', d0) + 1), d2 = body.find('"', d1 + 1);
        t.dtype = body.substr(d1 + 1, d2 - d1 - 1);
        auto ints = [&](const char* key) {
            std::vector<int64_t> v;
            const size_t a = body.find(key), b0 = body.find('[', a), b1 = body.find(']', b0);
            const std::string list = body.substr(b0 + 1, b1 - b0 - 1);
            for (const char* p = list.c_str(); *p;) {
                while (*p == ' ' || *p == ',') ++p;
                if (!*p) break;
                v.push_back(std::strtoll(p, (char**) &p, 10));
            }
            return v;
        };
        t.shape = ints("\"shape\"");
        const auto off = ints("\"data_offsets\"");
        if (off.size() != 2) throw std::runtime_error("drafter: bad entry " + name);
        t.begin = (uint64_t) off[0], t.end = (uint64_t) off[1];
        f.t[name] = t;
    }
    return f;
}
std::vector<uint16_t> st_bf16(const StFile& f, const std::string& name, const std::vector<int64_t>& shape) {
    auto it = f.t.find(name);
    if (it == f.t.end()) throw std::runtime_error("drafter: no tensor " + name);
    if (it->second.dtype != "BF16") throw std::runtime_error("drafter: " + name + " is " + it->second.dtype);
    if (it->second.shape != shape) throw std::runtime_error("drafter: " + name + ": unexpected shape");
    const size_t n = (it->second.end - it->second.begin) / 2;
    std::vector<uint16_t> v(n);
    const int fd = ::open(f.path.c_str(), O_RDONLY);
    size_t done = 0;
    while (done < n * 2) {
        const ssize_t r = ::pread(fd, (char*) v.data() + done, n * 2 - done, (off_t) (f.data0 + it->second.begin + done));
        if (r <= 0) throw std::runtime_error("drafter: read failed");
        done += (size_t) r;
    }
    ::close(fd);
    return v;
}

// ---- kernels
__device__ __forceinline__ float bsum(float v, float* sh) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = threadIdx.x < (blockDim.x >> 5) ? sh[threadIdx.x] : 0.0f;
    if (threadIdx.x < 32) for (int o = 16; o > 0; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
    __syncthreads();
    if (threadIdx.x == 0) sh[0] = t;
    __syncthreads();
    const float r = sh[0];
    __syncthreads();
    return r;
}
__global__ void k_to_bf16(const float* x, size_t n, uint16_t* y) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __bfloat16_as_ushort(__float2bfloat16_rn(x[i]));
}
// y[row] = x[row] * rsqrt(mean(x^2) + eps) * w   (row of D values; in place allowed)
__global__ void k_rms(const float* x, const float* w, int D, float eps, float* y) {
    __shared__ float sh[32];
    const float* xr = x + (size_t) blockIdx.x * D;
    float a = 0.0f;
    for (int c = threadIdx.x; c < D; c += blockDim.x) a += xr[c] * xr[c];
    const float r = rsqrtf(bsum(a, sh) / (float) D + eps);
    float* yr = y + (size_t) blockIdx.x * D;
    for (int c = threadIdx.x; c < D; c += blockDim.x) yr[c] = xr[c] * r * w[c];
}
// out[row] = rms(sum_k alpha[k] x[k][row]) * w
__global__ void k_mix_rms(const float* x, int n_tap, size_t tap_stride, int C, const float* alpha, const float* w, float eps, float* out) {
    __shared__ float sh[32];
    const int row = blockIdx.x;
    float a = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float m = 0.0f;
        for (int t = 0; t < n_tap; ++t) m += alpha[t] * x[(size_t) t * tap_stride + (size_t) row * C + c];
        out[(size_t) row * C + c] = m;
        a += m * m;
    }
    const float r = rsqrtf(bsum(a, sh) / (float) C + eps);
    for (int c = threadIdx.x; c < C; c += blockDim.x) out[(size_t) row * C + c] *= r * w[c];
}
__global__ void k_rope(float* x, int heads, const int* pos, float theta) {
    const int row = blockIdx.x, h = blockIdx.y, i = threadIdx.x;   // i < 64
    const float inv = 1.0f / powf(theta, (float) (2 * i) / 128.0f);
    const float ang = (float) pos[row] * inv;
    float sn, cs;
    sincosf(ang, &sn, &cs);
    float* p = x + ((size_t) row * heads + h) * 128;
    const float a = p[i], b = p[i + 64];
    p[i] = a * cs - b * sn;
    p[i + 64] = b * cs + a * sn;
}
__global__ void k_add(const float* a, const float* b, size_t n, float* y) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}
__global__ void k_silu_mul(const float* gu, int F, uint16_t* out) {
    const int t = blockIdx.y, f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= F) return;
    const float g = gu[(size_t) t * 2 * F + f], u = gu[(size_t) t * 2 * F + F + f];
    out[(size_t) t * F + f] = __bfloat16_as_ushort(__float2bfloat16_rn(g / (1.0f + expf(-g)) * u));
}
__global__ void k_iota(int* out, int start, int n) { const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) out[i] = start + i; }
__global__ void k_init_block(const float* anchor_h, const float* mask, float* res, int* pos, int* toks, int C, int P, int anchor) {
    const int t = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (t == 0 && blockIdx.x == 0) {
        if (threadIdx.x < kBlock) pos[threadIdx.x] = P + threadIdx.x;
        if (threadIdx.x == 0) toks[0] = anchor;
    }
    if (c < C) res[(size_t) t * C + c] = (t == 0 ? anchor_h : mask)[c];
}
// the drafter's attention: grid (H, kBlock queries), 128 threads.  Keys: the context positions [k_lo, k_hi) (rows of a ring of cap: p % cap) then the block's own kBlock keys.
__global__ void k_attn(const float* q, const float* kc, const float* vc, int k_lo, int k_hi, int cap, const float* kb, const float* vb, int H, int KV, float scale, float* scratch, float* out) {
    __shared__ float sh[32];
    const int h = blockIdx.x, qi = blockIdx.y, kvh = h / (H / KV);
    const int nctx = k_hi - k_lo, total = nctx + kBlock, KD = KV * 128;
    float* lg = scratch + ((size_t) h * kBlock + qi) * (size_t) (total + 8 > 0 ? (size_t) (k_hi - k_lo) + kBlock + 8 : 0);
    const float* qh = q + ((size_t) qi * H + h) * 128;
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        const float* kk = i < nctx ? kc + (size_t) ((k_lo + i) % cap) * KD + kvh * 128 : kb + (size_t) (i - nctx) * KD + kvh * 128;
        float d = 0.0f;
        for (int j = 0; j < 128; ++j) d += qh[j] * kk[j];
        lg[i] = d * scale;
    }
    __syncthreads();
    float mx = -3.0e38f;
    for (int i = threadIdx.x; i < total; i += blockDim.x) mx = fmaxf(mx, lg[i]);
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = mx;
    __syncthreads();
    mx = sh[0];
    for (int w = 1; w < (int) (blockDim.x >> 5); ++w) mx = fmaxf(mx, sh[w]);
    __syncthreads();
    float z = 0.0f;
    for (int i = threadIdx.x; i < total; i += blockDim.x) { lg[i] = expf(lg[i] - mx); z += lg[i]; }
    z = bsum(z, sh);
    for (int d = threadIdx.x; d < 128; d += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < total; ++i) {
            const float* vv = i < nctx ? vc + (size_t) ((k_lo + i) % cap) * KD + kvh * 128 : vb + (size_t) (i - nctx) * KD + kvh * 128;
            acc += lg[i] * vv[d];
        }
        out[((size_t) qi * H + h) * 128 + d] = acc / z;
    }
}
__global__ void k_rnn_inputs(const float* state, const uint16_t* emb, const int* prev, const float* hidden, const float* f, int R, int C, uint16_t* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, W = 2 * R + 2 * C;
    if (i >= W) return;
    float v;
    if (i < R) v = state[i];
    else if (i < 2 * R) v = bf(emb[(size_t) (*prev) * R + (i - R)]);
    else if (i < 2 * R + C) v = hidden[i - 2 * R];
    else v = f[i - 2 * R - C];
    out[i] = __bfloat16_as_ushort(__float2bfloat16_rn(v));
}
__global__ void k_rnn_gates(const float* proj, const float* bias, int R, float* state, float* latent) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= R) return;
    const float gr = proj[i] + bias[i], cr = proj[R + i] + bias[R + i], orw = proj[2 * R + i] + bias[2 * R + i];
    const float gate = 1.0f / (1.0f + expf(-gr));
    state[i] = gate * state[i] + (1.0f - gate) * tanhf(cr);
    latent[i] = tanhf(orw);
}
// argmax over v < n of base[v] + bias[v] + sum_r W[r * ld + v] x[r]  (W FP16 [R][ld]); ties: the lower index.  Two stages.
__global__ void k_argmax_part(const uint16_t* W, int ld, const float* x, const float* base, const uint16_t* bias, int n, float* pv, int* pi, float* ps) {
    __shared__ float xs[kR];
    for (int r = threadIdx.x; r < kR; r += blockDim.x) xs[r] = x[r];
    __syncthreads();
    float best = -3.0e38f, sum = 0.0f;   // best = the running max; sum = sum of exp(score - best) so far (the softmax normaliser, online)
    int bi = 0x7fffffff;
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < n; v += gridDim.x * blockDim.x) {
        float s = base[v] + __half2float(__ushort_as_half(bias[v]));
        for (int r = 0; r < kR; ++r) s += __half2float(__ushort_as_half(W[(size_t) r * ld + v])) * xs[r];
        if (s > best) { sum = sum * expf(best - s) + 1.0f; best = s; bi = v; }
        else { sum += expf(s - best); if (s == best && v < bi) bi = v; }
    }
    for (int o = 16; o > 0; o >>= 1) {
        const float ob = __shfl_xor_sync(0xffffffffu, best, o), os = __shfl_xor_sync(0xffffffffu, sum, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        const float m = fmaxf(best, ob);
        sum = sum * expf(best - m) + os * expf(ob - m);
        if (ob > best || (ob == best && oi < bi)) bi = oi;
        best = m;
    }
    __shared__ float sv[32], ss[32];
    __shared__ int si[32];
    if ((threadIdx.x & 31) == 0) { sv[threadIdx.x >> 5] = best; si[threadIdx.x >> 5] = bi; ss[threadIdx.x >> 5] = sum; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float b = sv[0], z = ss[0]; int ix = si[0];
        for (int w = 1; w < (int) (blockDim.x >> 5); ++w) {
            const float m = fmaxf(b, sv[w]);
            z = z * expf(b - m) + ss[w] * expf(sv[w] - m);
            if (sv[w] > b || (sv[w] == b && si[w] < ix)) ix = si[w];
            b = m;
        }
        pv[blockIdx.x] = b; pi[blockIdx.x] = ix; ps[blockIdx.x] = z;
    }
}
__global__ void k_argmax_final(const float* pv, const int* pi, const float* ps, int n, int* out, float* prob) {
    float b = pv[0], z = ps[0]; int ix = pi[0];
    for (int i = 1; i < n; ++i) {
        const float m = fmaxf(b, pv[i]);
        z = z * expf(b - m) + ps[i] * expf(pv[i] - m);
        if (pv[i] > b || (pv[i] == b && pi[i] < ix)) ix = pi[i];
        b = m;
    }
    *out = ix;
    *prob = 1.0f / z;   // the drafted token's probability
}
// the target's Q8_0 head rows [row0, row0 + m) as floats
__global__ void k_deq_head(const uint8_t* head, size_t row_bytes, int row0, int C, float* out) {
    const int r = blockIdx.x;
    const uint8_t* row = head + (size_t) (row0 + r) * row_bytes;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        const uint8_t* b = row + (size_t) (c / 32) * 34;
        out[(size_t) r * C + c] = __half2float(*reinterpret_cast<const __half*>(b)) * (float) reinterpret_cast<const signed char*>(b + 2)[c % 32];
    }
}
__global__ void k_pack_fold(const float* src, int m, int n_rows, int r0, uint16_t* dst) {   // src [R][m] -> dst [R][n_rows]
    const int i = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if (i < m) dst[(size_t) r * n_rows + r0 + i] = __half_as_ushort(__float2half_rn(src[(size_t) r * m + i]));
}
__global__ void k_to_f16(const float* x, size_t n, uint16_t* y) { const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = __half_as_ushort(__float2half_rn(x[i])); }

}  // namespace

struct Drafter::Impl {
    struct DLayer {
        bool local = false;
        float *in_norm = nullptr, *post_norm = nullptr, *dflare = nullptr, *q_norm = nullptr, *k_norm = nullptr, *k_norm_ctx = nullptr;
        uint16_t *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr, *wkc = nullptr, *wvc = nullptr, *wgu = nullptr, *wd = nullptr;
        float *kc = nullptr, *vc = nullptr;   // the context caches [cap][KV * 128] F32: a ring of the last cap positions
        int cap = 0;
    };
    int C = 2304, H = 24, KV = 8, F = 9216, V = 0, ctx = 0, n_rows = 0, n_vocab = 0;
    float eps = 1e-6f, theta = 1e6f;
    std::vector<DLayer> L;
    float *alpha = nullptr, *norm = nullptr, *mask = nullptr, *mk_dflare = nullptr, *rnn_b = nullptr;
    uint16_t *w1 = nullptr, *rnn_w = nullptr, *fold_w = nullptr, *fold_b = nullptr;
    std::vector<uint16_t> hout_w, hout_b;
    cublasHandle_t blas = nullptr;
    cudaStream_t s = nullptr;
    size_t vram = 0;
    float *x = nullptr, *res = nullptr, *h = nullptr, *q = nullptr, *kk = nullptr, *vv = nullptr, *o = nullptr, *gu = nullptr, *f_rnn = nullptr, *state = nullptr, *latent = nullptr, *proj = nullptr,
          *scratch = nullptr, *mix = nullptr, *base = nullptr, *ctx_tmp = nullptr, *pv = nullptr;
    uint16_t *hb = nullptr, *hb2 = nullptr, *rnn_in = nullptr;
    int *pos = nullptr, *ctx_pos = nullptr, *toks = nullptr, *pi = nullptr, *amax = nullptr;
    float *ps = nullptr, *probs = nullptr;
    int* h_out = nullptr;
    float* h_prob = nullptr;
    size_t scratch_floats = 0;

    uint16_t* up16(const std::vector<uint16_t>& v) {
        uint16_t* p = dalloc<uint16_t>(v.size());
        cu(cudaMemcpy(p, v.data(), v.size() * 2, cudaMemcpyHostToDevice), "upload");
        vram += v.size() * 2;
        return p;
    }
    float* up32(const std::vector<uint16_t>& v) {
        std::vector<float> f(v.size());
        for (size_t i = 0; i < v.size(); ++i) f[i] = bf(v[i]);
        float* p = dalloc<float>(f.size());
        cu(cudaMemcpy(p, f.data(), f.size() * 4, cudaMemcpyHostToDevice), "upload");
        vram += f.size() * 4;
        return p;
    }
    void gemm_bf(const uint16_t* W, int N, int K, const uint16_t* X, int T, float* Y, int ldy) {
        const float one = 1.0f, zero = 0.0f;
        cb(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, N, T, K, &one, W, CUDA_R_16BF, K, X, CUDA_R_16BF, K, &zero, Y, CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), "gemm_bf");
    }
    void to_bf16(const float* x, size_t n, uint16_t* y) { k_to_bf16<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(x, n, y); }
    void rms(const float* x, const float* w, int rows, int D, float* y) { k_rms<<<rows, 256, 0, s>>>(x, w, D, eps, y); }
    void rope(float* x, int rows, int heads, const int* pos_) { k_rope<<<dim3(rows, heads), 64, 0, s>>>(x, heads, pos_, theta); }

    void layer_attn(DLayer& l, int P, float* out_) {
        const int T = kBlock, HD = H * 128, KD = KV * 128;
        to_bf16(h, (size_t) T * C, hb);
        gemm_bf(l.wq, HD, C, hb, T, q, HD);
        gemm_bf(l.wk, KD, C, hb, T, kk, KD);
        gemm_bf(l.wv, KD, C, hb, T, vv, KD);
        rms(q, l.q_norm, T * H, 128, q);
        rms(kk, l.k_norm, T * KV, 128, kk);
        rope(q, T, H, pos);
        rope(kk, T, KV, pos);
        const int k_lo = l.local ? std::max(0, P - 1023) : std::max(0, P - l.cap);
        k_attn<<<dim3(H, T), 128, 0, s>>>(q, l.kc, l.vc, k_lo, P, l.cap, kk, vv, H, KV, 1.0f / std::sqrt(128.0f), scratch, o);
        to_bf16(o, (size_t) T * HD, hb);
        gemm_bf(l.wo, C, HD, hb, T, out_, C);
    }
    void layer_mlp(DLayer& l, float* out_) {
        const int T = kBlock;
        to_bf16(h, (size_t) T * C, hb);
        gemm_bf(l.wgu, 2 * F, C, hb, T, gu, 2 * F);
        k_silu_mul<<<dim3((F + 255) / 256, T), 256, 0, s>>>(gu, F, hb2);
        gemm_bf(l.wd, C, F, hb2, T, out_, C);
    }
};

Drafter::Drafter(const std::string& path, int C, int ctx, void* stream) : p_(new Impl) {
    Impl& d = *p_;
    if (C != 2304) throw std::runtime_error("drafter: only the Mini width (2304) is supported");
    d.C = C; d.ctx = ctx; d.s = (cudaStream_t) stream;
    cb(cublasCreate(&d.blas), "cublasCreate");
    cb(cublasSetStream(d.blas, d.s), "cublasSetStream");
    const StFile f = st_open(path);
    const int64_t H = d.H, KV = d.KV, F = d.F, R = kR;
    {
        auto it = f.t.find("markov_head.markov_w1.weight");
        if (it == f.t.end() || it->second.shape.size() != 2) throw std::runtime_error("drafter: no markov_w1");
        d.V = (int) it->second.shape[0];
    }
    {   // fuse_weight [layers + 1][taps] -> softmax over each row
        const auto fw = st_bf16(f, "fuse_weight", {kLayers + 1, kTaps});
        std::vector<float> a(fw.size());
        for (int r = 0; r <= kLayers; ++r) {
            float m = -1e30f, sum = 0;
            for (int t = 0; t < kTaps; ++t) m = std::max(m, bf(fw[r * kTaps + t]));
            for (int t = 0; t < kTaps; ++t) sum += (a[r * kTaps + t] = std::exp(bf(fw[r * kTaps + t]) - m));
            for (int t = 0; t < kTaps; ++t) a[r * kTaps + t] /= sum;
        }
        d.alpha = dalloc<float>(a.size());
        cu(cudaMemcpy(d.alpha, a.data(), a.size() * 4, cudaMemcpyHostToDevice), "alpha");
    }
    d.L.resize(kLayers);
    for (int i = 0; i < kLayers; ++i) {
        Impl::DLayer& l = d.L[(size_t) i];
        const std::string b = "layers." + std::to_string(i) + ".";
        l.local = i != 1;   // Mini: layers 0 and 2 see the last 1024 positions, layer 1 all of them
        l.in_norm = d.up32(st_bf16(f, b + "input_layernorm.weight", {C}));
        l.post_norm = d.up32(st_bf16(f, b + "post_attention_layernorm.weight", {C}));
        l.dflare = d.up32(st_bf16(f, b + "dflare_norm.weight", {C}));
        l.q_norm = d.up32(st_bf16(f, b + "self_attn.q_norm.weight", {128}));
        l.k_norm = d.up32(st_bf16(f, b + "self_attn.k_norm.weight", {128}));
        l.k_norm_ctx = d.up32(st_bf16(f, b + "self_attn.k_norm_ctx.weight", {128}));
        l.wq = d.up16(st_bf16(f, b + "self_attn.q_proj.weight", {H * 128, C}));
        l.wk = d.up16(st_bf16(f, b + "self_attn.k_proj.weight", {KV * 128, C}));
        l.wv = d.up16(st_bf16(f, b + "self_attn.v_proj.weight", {KV * 128, C}));
        l.wo = d.up16(st_bf16(f, b + "self_attn.o_proj.weight", {C, H * 128}));
        l.wkc = d.up16(st_bf16(f, b + "self_attn.k_proj_ctx.weight", {KV * 128, C}));
        l.wvc = d.up16(st_bf16(f, b + "self_attn.v_proj_ctx.weight", {KV * 128, C}));
        {
            auto g = st_bf16(f, b + "mlp.gate_proj.weight", {F, C});
            const auto u = st_bf16(f, b + "mlp.up_proj.weight", {F, C});
            g.insert(g.end(), u.begin(), u.end());   // gate's rows, then up's
            l.wgu = d.up16(g);
        }
        l.wd = d.up16(st_bf16(f, b + "mlp.down_proj.weight", {C, F}));
        l.cap = l.local ? std::min(ctx, 2048) : ctx;
        l.kc = dalloc<float>((size_t) l.cap * KV * 128);
        l.vc = dalloc<float>((size_t) l.cap * KV * 128);
        d.vram += (size_t) l.cap * KV * 128 * 8;
    }
    d.norm = d.up32(st_bf16(f, "norm.weight", {C}));
    d.mask = d.up32(st_bf16(f, "mask_embed", {C}));
    d.mk_dflare = d.up32(st_bf16(f, "markov_head.dflare_norm.weight", {C}));
    d.w1 = d.up16(st_bf16(f, "markov_head.markov_w1.weight", {d.V, R}));
    d.rnn_w = d.up16(st_bf16(f, "markov_head.rnn_proj.weight", {3 * R, 2 * R + 2 * C}));
    d.rnn_b = d.up32(st_bf16(f, "markov_head.rnn_proj.bias", {3 * R}));
    d.hout_w = st_bf16(f, "markov_head.hidden_out.weight", {C, R});
    d.hout_b = st_bf16(f, "markov_head.hidden_out.bias", {C});
    const size_t T = kBlock;
    d.x = dalloc<float>(T * C); d.res = dalloc<float>(T * C); d.h = dalloc<float>(T * C);
    d.q = dalloc<float>(T * H * 128); d.o = dalloc<float>(T * H * 128);
    d.kk = dalloc<float>(std::max<size_t>(T, kMixRows) * KV * 128); d.vv = dalloc<float>(T * KV * 128);
    d.gu = dalloc<float>(T * 2 * F);
    d.hb = dalloc<uint16_t>(std::max<size_t>(T * std::max<int64_t>(C, H * 128), (size_t) kMixRows * C));
    d.hb2 = dalloc<uint16_t>(T * F);
    d.ctx_tmp = dalloc<float>((size_t) 2 * kMixRows * KV * 128);
    size_t cmax = 0;
    for (auto& l : d.L) cmax = std::max<size_t>(cmax, (size_t) l.cap);
    d.scratch_floats = (size_t) H * kBlock * (cmax + kBlock + 8);
    d.scratch = dalloc<float>(d.scratch_floats);
    d.mix = dalloc<float>((size_t) kMixRows * C);
    d.f_rnn = dalloc<float>(C);
    d.state = dalloc<float>(R); d.latent = dalloc<float>(R); d.proj = dalloc<float>(3 * R);
    d.rnn_in = dalloc<uint16_t>(2 * R + 2 * C);
    d.pos = dalloc<int>(T); d.ctx_pos = dalloc<int>(kMixRows); d.toks = dalloc<int>(kBlock);
    d.pv = dalloc<float>(256); d.pi = dalloc<int>(256); d.ps = dalloc<float>(256); d.probs = dalloc<float>(kBlock); d.amax = dalloc<int>(1);
    cu(cudaHostAlloc(&d.h_out, kBlock * sizeof(int), cudaHostAllocDefault), "h_out");
    cu(cudaHostAlloc(&d.h_prob, kBlock * sizeof(float), cudaHostAllocDefault), "h_prob");
    d.vram += d.scratch_floats * 4 + 4u * T * C * 4 + (size_t) T * 2 * F * 4;
}

Drafter::~Drafter() {
    Impl& d = *p_;
    for (auto& l : d.L) {
        for (void* q : {(void*) l.in_norm, (void*) l.post_norm, (void*) l.dflare, (void*) l.q_norm, (void*) l.k_norm, (void*) l.k_norm_ctx, (void*) l.wq, (void*) l.wk, (void*) l.wv, (void*) l.wo, (void*) l.wkc,
                        (void*) l.wvc, (void*) l.wgu, (void*) l.wd, (void*) l.kc, (void*) l.vc}) cudaFree(q);
    }
    for (void* q : {(void*) d.alpha, (void*) d.norm, (void*) d.mask, (void*) d.mk_dflare, (void*) d.rnn_b, (void*) d.w1, (void*) d.rnn_w, (void*) d.fold_w, (void*) d.fold_b, (void*) d.x, (void*) d.res,
                    (void*) d.h, (void*) d.q, (void*) d.o, (void*) d.kk, (void*) d.vv, (void*) d.gu, (void*) d.hb, (void*) d.hb2, (void*) d.ctx_tmp, (void*) d.scratch, (void*) d.mix, (void*) d.f_rnn,
                    (void*) d.state, (void*) d.latent, (void*) d.proj, (void*) d.rnn_in, (void*) d.pos, (void*) d.ctx_pos, (void*) d.toks, (void*) d.pv, (void*) d.pi, (void*) d.ps, (void*) d.probs, (void*) d.amax, (void*) d.base}) cudaFree(q);
    if (d.h_out) cudaFreeHost(d.h_out);
    if (d.h_prob) cudaFreeHost(d.h_prob);
    if (d.blas) cublasDestroy(d.blas);
}

void Drafter::fold(const uint8_t* head_q8, int n_rows, int n_vocab) {
    Impl& d = *p_;
    const int C = d.C, R = kR;
    d.n_rows = n_rows; d.n_vocab = n_vocab;
    const size_t row_bytes = (size_t) (C / 32) * 34;
    std::vector<float> ht((size_t) R * C), hb(C);
    for (int c = 0; c < C; ++c) for (int r = 0; r < R; ++r) ht[(size_t) r * C + c] = bf(d.hout_w[(size_t) c * R + r]);
    for (int c = 0; c < C; ++c) hb[c] = bf(d.hout_b[c]);
    float *dht = dalloc<float>(ht.size()), *dhb = dalloc<float>(hb.size());
    cu(cudaMemcpy(dht, ht.data(), ht.size() * 4, cudaMemcpyHostToDevice), "fold");
    cu(cudaMemcpy(dhb, hb.data(), hb.size() * 4, cudaMemcpyHostToDevice), "fold");
    const int chunk = std::min(4096, std::max(1, n_rows));
    float* dlm = dalloc<float>((size_t) chunk * C);
    float* dfw = dalloc<float>((size_t) R * chunk);
    float* dfb = dalloc<float>(chunk);
    d.fold_w = dalloc<uint16_t>((size_t) R * n_rows);
    d.fold_b = dalloc<uint16_t>(n_rows);
    d.base = dalloc<float>((size_t) (kBlock - 1) * n_rows);
    const float one = 1.0f, zero = 0.0f;
    for (int r0 = 0; r0 < n_rows; r0 += chunk) {
        const int m = std::min(chunk, n_rows - r0);
        k_deq_head<<<m, 256, 0, d.s>>>(head_q8, row_bytes, r0, C, dlm);
        cb(cublasSgemm(d.blas, CUBLAS_OP_T, CUBLAS_OP_N, m, R, C, &one, dlm, C, dht, C, &zero, dfw, m), "fold w");
        cb(cublasSgemv(d.blas, CUBLAS_OP_T, C, m, &one, dlm, C, dhb, 1, &zero, dfb, 1), "fold b");
        k_pack_fold<<<dim3((m + 255) / 256, R), 256, 0, d.s>>>(dfw, m, n_rows, r0, d.fold_w);
        k_to_f16<<<(m + 255) / 256, 256, 0, d.s>>>(dfb, (size_t) m, d.fold_b + r0);
    }
    cu(cudaStreamSynchronize(d.s), "fold");
    cudaFree(dht); cudaFree(dhb); cudaFree(dlm); cudaFree(dfw); cudaFree(dfb);
    d.vram += ((size_t) R + 1) * n_rows * 2 + (size_t) (kBlock - 1) * n_rows * 4;
}

void Drafter::write_context(const float* taps, int ld, int r0, int m, int pos0) {
    Impl& d = *p_;
    const int C = d.C, KD = d.KV * 128;
    for (size_t li = 0; li < d.L.size(); ++li) {
        Impl::DLayer& l = d.L[li];
        for (int a = 0; a < m;) {
            const int at = (pos0 + a) % l.cap;
            const int n = std::min({kMixRows, m - a, l.cap - at});   // a ring: a chunk ends where it wraps
            k_iota<<<(n + 255) / 256, 256, 0, d.s>>>(d.ctx_pos, pos0 + a, n);
            k_mix_rms<<<n, 256, 0, d.s>>>(taps + (size_t) (r0 + a) * C, kTaps, (size_t) ld * C, C, d.alpha + li * kTaps, l.dflare, d.eps, d.mix);
            d.to_bf16(d.mix, (size_t) n * C, d.hb);
            float* kd = l.kc + (size_t) at * KD;
            float* vd = l.vc + (size_t) at * KD;
            d.gemm_bf(l.wkc, KD, C, d.hb, n, kd, KD);
            d.gemm_bf(l.wvc, KD, C, d.hb, n, vd, KD);
            d.rms(kd, l.k_norm_ctx, n * d.KV, 128, kd);
            d.rope(kd, n, d.KV, d.ctx_pos);
            a += n;
        }
    }
}

void Drafter::set_f_rnn(const float* taps, int ld, int row) {
    Impl& d = *p_;
    k_mix_rms<<<1, 256, 0, d.s>>>(taps + (size_t) row * d.C, kTaps, (size_t) ld * d.C, d.C, d.alpha + kLayers * kTaps, d.mk_dflare, d.eps, d.f_rnn);
}

void Drafter::propose(int anchor, const float* anchor_h, int P, int proposals, const HeadFn& head, int* out, float* probs) {
    Impl& d = *p_;
    const int C = d.C, T = kBlock;
    if (d.n_rows == 0) throw std::runtime_error("drafter: fold() was not called");
    proposals = std::min(std::max(proposals, 1), kBlock - 1);
    k_init_block<<<dim3((C + 255) / 256, T), 256, 0, d.s>>>(anchor_h, d.mask, d.res, d.pos, d.toks, C, P, anchor);
    const size_t TC = (size_t) T * C;
    for (size_t i = 0; i < d.L.size(); ++i) {
        Impl::DLayer& l = d.L[i];
        if (i > 0) k_add<<<(unsigned) ((TC + 255) / 256), 256, 0, d.s>>>(d.x, d.res, TC, d.res);
        d.rms(d.res, l.in_norm, T, C, d.h);
        d.layer_attn(l, P, d.x);
        k_add<<<(unsigned) ((TC + 255) / 256), 256, 0, d.s>>>(d.x, d.res, TC, d.res);
        d.rms(d.res, l.post_norm, T, C, d.h);
        d.layer_mlp(l, d.x);
    }
    k_add<<<(unsigned) ((TC + 255) / 256), 256, 0, d.s>>>(d.x, d.res, TC, d.res);
    d.rms(d.res, d.norm, T, C, d.h);   // the draft hidden, rows 1 .. 7 used
    head(d.h + C, proposals, d.base);  // base logits of rows 1 .. proposals through the target's head
    cu(cudaMemsetAsync(d.state, 0, kR * 4, d.s), "state");
    for (int t = 0; t < proposals; ++t) {
        k_rnn_inputs<<<(2 * kR + 2 * C + 255) / 256, 256, 0, d.s>>>(d.state, d.w1, d.toks + t, d.h + (size_t) (t + 1) * C, d.f_rnn, kR, C, d.rnn_in);
        d.gemm_bf(d.rnn_w, 3 * kR, 2 * kR + 2 * C, d.rnn_in, 1, d.proj, 3 * kR);
        k_rnn_gates<<<(kR + 255) / 256, 256, 0, d.s>>>(d.proj, d.rnn_b, kR, d.state, d.latent);
        const int blocks = 128;
        k_argmax_part<<<blocks, 256, 0, d.s>>>(d.fold_w, d.n_rows, d.latent, d.base + (size_t) t * d.n_rows, d.fold_b, d.n_vocab, d.pv, d.pi, d.ps);
        k_argmax_final<<<1, 1, 0, d.s>>>(d.pv, d.pi, d.ps, blocks, d.toks + t + 1, d.probs + t);
    }
    cu(cudaMemcpyAsync(d.h_out, d.toks + 1, (size_t) proposals * 4, cudaMemcpyDeviceToHost, d.s), "tokens");
    cu(cudaMemcpyAsync(d.h_prob, d.probs, (size_t) proposals * 4, cudaMemcpyDeviceToHost, d.s), "probs");
    cu(cudaStreamSynchronize(d.s), "propose");
    std::memcpy(out, d.h_out, (size_t) proposals * sizeof(int));
    if (probs) std::memcpy(probs, d.h_prob, (size_t) proposals * sizeof(float));
}

int Drafter::n_taps() const { return kTaps; }
int Drafter::n_rows() const { return p_->n_rows; }
size_t Drafter::bytes() const { return p_->vram; }
const int* Drafter::tap_visits(int& n) { n = 3; return kTapVisits; }

}  // namespace strata::inkling2
