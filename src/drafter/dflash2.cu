// src/drafter/dflash2.cu - the native DFlash 2 drafter (see dflash2.hpp; the reference is z-lab's dflash/model.py, 22 Oct 2026).
//
// Reference step -> here:
//   fc, hidden_norm (target_hidden)          feed(): bf16 -> f32, fc via native_mmvq (Q4_K), rmsnorm_rows
//   k_proj/v_proj of the context, k_norm,    feed(): per layer, mmvq, head_norm_rope (norm + rope at the position) into the
//   rope, cache update                       ring; v scattered into the ring.  The ring is the draft cache: 2048 slots, the
//                                            window needs 2048.  A position's k/v is made once, when it is fed.
//   noise_embedding = embed(ids) * scale     propose(): the engine's EmbedFn (scale is 1.0: the config has none)
//   input_layernorm, attention_conv.prepare  rmsnorm_rows, mmvq (attn_conv_proj: 640 = 2 taps x K x groups), conv_kernel
//   q/k/v proj, q_norm/k_norm, rope          mmvq, head_norm_rope (queries: no ring; noise keys: their own scratch)
//   sdpa with the sliding mask               attention_kernel: keys within 2047 positions either side of the query (the mask
//                                            is not causal: the block's rows see forward positions too), GQA 32 -> 8 heads
//   o_proj, attention_conv.finish            mmvq, conv_kernel with the second tap (offset 320)
//   post_attention_layernorm, mlp_conv,      as above; gate/up mmvq, silu_mul, down mmvq
//   mlp (silu(gate) * up -> down)
//   norm (the output norm), propose()        rmsnorm_rows over the block; rows 1.. feed the selector
//   selector: hidden_projection, the head's  mmvq (selector_hidden), HeadFn, topk_kernel (top-16 of the logits),
//   top-16, the codebook path                selector_kernel (Q4_K or Q8_0 codebook rows dequantized in the kernel)
#include "dflash2.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>

#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/native_mmvq.hpp"

namespace strata::drafter {

namespace {

constexpr int kRing = 2048;          // ring slots per layer: exactly the window (a propose reads the kWindow positions before its start)
constexpr int kQ4kBytes = 144;       // one Q4_K block: 256 values
constexpr int kQ8Bytes = 34;         // one Q8_0 block: fp16 scale + 32 int8 values
constexpr int kType8_0 = 8;          // GGML type ids of the selector codebooks (the loader accepts these two)
constexpr int kType4K = 12;
constexpr int kTaps = 2;             // conv_kernel_size
constexpr int kGroupSize = 16;       // conv_group_size
constexpr int kConvOut = 640;        // kernel_projection: 2 taps x kTaps x (kHidden / kGroupSize) = 2 x 2 x 160
constexpr int kKvDim = Dflash2::kKvHeads * Dflash2::kHeadDim;      // 1024
constexpr int kQDim = Dflash2::kHeads * Dflash2::kHeadDim;         // 4096

// Sum over a block of 128 or 256 threads.  Every thread returns the sum; the trailing barrier keeps the shared slot from
// being reused before every thread has read it.
__device__ float block_sum(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) red[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = lane < (int) (blockDim.x >> 5) ? red[lane] : 0.f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) red[0] = v;
    }
    __syncthreads();
    const float total = red[0];
    __syncthreads();
    return total;
}

// One Q4_K value at index r (0..255) of a block: the ggml dequantize_row_q4_K arithmetic, element by element.
__device__ float q4k_elem(const uint8_t* blk, int r) {
    const float d = __half2float(*reinterpret_cast<const __half*>(blk));
    const float dmin = __half2float(*reinterpret_cast<const __half*>(blk + 2));
    const uint8_t* sc = blk + 4;
    const uint8_t* qs = blk + 16;
    const int g = r >> 6;
    const int l = r & 63;
    const int sub = l < 32 ? 2 * g : 2 * g + 1;
    const int nib = l < 32 ? (qs[g * 32 + l] & 0xF) : (qs[g * 32 + (l - 32)] >> 4);
    int s, m;
    if (sub < 4) {
        s = sc[sub] & 63;
        m = sc[sub + 4] & 63;
    } else {
        s = (sc[sub + 4] & 0xF) | ((sc[sub - 4] >> 6) << 4);
        m = (sc[sub + 4] >> 4) | ((sc[sub] >> 6) << 4);
    }
    return d * s * nib - dmin * m;
}

// One value at index r (0..255) of a codebook row of GGML type `type`: Q4_K (one 256-value block) or Q8_0 (eight 32-value
// blocks, each an fp16 scale then 32 int8 values, dequantized as d * q as ggml's dequantize_row_q8_0 does).
__device__ float codebook_elem(const uint8_t* row, int type, int r) {
    if (type == kType8_0) {
        const uint8_t* blk = row + (r >> 5) * kQ8Bytes;
        return __half2float(*reinterpret_cast<const __half*>(blk)) * (float) reinterpret_cast<const int8_t*>(blk + 2)[r & 31];
    }
    return q4k_elem(row, r);
}

__global__ void bf16_to_f32_kernel(const uint16_t* src, float* dst, int64_t n) {
    const int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __uint_as_float(((uint32_t) src[i]) << 16);
}

// y[row] = w * x[row] / rms(x[row]); grid = rows, block = 256.
__global__ void rmsnorm_rows_kernel(const float* x, const float* w, float* y, int D, float eps) {
    __shared__ float red[8];
    const float* xr = x + (size_t) blockIdx.x * D;
    float* yr = y + (size_t) blockIdx.x * D;
    float s = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) s += xr[i] * xr[i];
    s = block_sum(s, red);
    const float inv = rsqrtf(s / D + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x) yr[i] = w[i] * (xr[i] * inv);
}

// One head of one row: RMS norm over the head's 128 values (weight w), then rope at position pos0 + row (the rotate-half
// form of z-lab's apply_rotary_pos_emb, theta from the config).  The row is written at dst = (pos & mask) when mask >= 0
// (a ring slot), else at the row itself.  in and out have heads * 128 values per row.  grid = (rows, heads), block = 128.
__global__ void head_norm_rope_kernel(const float* in, float* out, const float* w, int pos0, int mask, float eps,
                                      float theta) {
    __shared__ float red[8];
    __shared__ float sv[Dflash2::kHeadDim];
    const int row = blockIdx.x;
    const int hd = blockIdx.y;
    const int t = threadIdx.x;
    const int stride = gridDim.y * Dflash2::kHeadDim;
    float v = in[(size_t) row * stride + hd * Dflash2::kHeadDim + t];
    const float s = block_sum(v * v, red);
    v = w[t] * (v * rsqrtf(s / Dflash2::kHeadDim + eps));
    sv[t] = v;
    __syncthreads();
    const int pos = pos0 + row;
    const int dst = mask >= 0 ? (pos & mask) : row;
    const int j = t & 63;
    const float inv_freq = 1.0f / powf(theta, (float) (2 * j) / (float) Dflash2::kHeadDim);
    const float ang = (float) pos * inv_freq;
    const float c = cosf(ang);
    const float sn = sinf(ang);
    const float o = t < 64 ? sv[t] * c - sv[t + 64] * sn : sv[t] * c + sv[t - 64] * sn;
    out[(size_t) dst * stride + hd * Dflash2::kHeadDim + t] = o;
}

// v rows (kKvDim values each) into the ring at (pos0 + row) & mask.  grid = rows, block = 256.
__global__ void scatter_rows_kernel(const float* in, float* ring, int pos0, int mask) {
    const int dst = (pos0 + blockIdx.x) & mask;
    for (int i = threadIdx.x; i < kKvDim; i += blockDim.x)
        ring[(size_t) dst * kKvDim + i] = in[(size_t) blockIdx.x * kKvDim + i];
}

// The grouped dynamic causal conv over the block's rows (z-lab's _grouped_dynamic_convolve):
//   y[t][c] = sum_o (base[o][c] + dyn[t][off + o * G + c / gs]) * x[t - o][c],   x[t - o] = 0 before the block.
// dyn rows have 640 values: the two taps' projections, tap j at offset j * K * G.  grid = (rows, kHidden / 256), block 256.
__global__ void conv_kernel(const float* x, const float* dyn, int dyn_off, const float* base, float* y, int K, int G,
                            int gs) {
    const int t = blockIdx.x;
    const int c = blockIdx.y * blockDim.x + threadIdx.x;
    const int H = Dflash2::kHidden;
    float acc = 0.f;
    for (int o = 0; o < K; o++) {
        const int tp = t - o;
        if (tp < 0) break;
        const float xv = x[(size_t) tp * H + c];
        const float kb = base[o * H + c];
        const float dv = dyn[(size_t) t * kConvOut + dyn_off + o * G + c / gs];
        acc += kb * xv;
        acc += dv * xv;
    }
    y[(size_t) t * H + c] = acc;
}

// Sliding attention for query row i of head hd (grid = (rows, heads), block = 128).  Keys: positions
// [max(0, start + i - 2047), start + rows - 1]; a position below start is in the ring, the block's own are in nk / nv.
__global__ void attention_kernel(const float* q, const float* ring_k, const float* ring_v, const float* nk,
                                 const float* nv, float* out, int start, int rows) {
    extern __shared__ float smem[];
    float* sq = smem;
    float* sc = smem + Dflash2::kHeadDim;
    __shared__ float inv_sum;
    const int i = blockIdx.x;
    const int hd = blockIdx.y;
    const int t = threadIdx.x;
    const int kvh = hd / (Dflash2::kHeads / Dflash2::kKvHeads);
    sq[t] = q[(size_t) i * kQDim + hd * Dflash2::kHeadDim + t];
    __syncthreads();
    const int qp = start + i;
    const int lo = max(0, qp - (Dflash2::kWindow - 1));
    const int hi = start + rows - 1;
    const int n = hi - lo + 1;
    const float scale = rsqrtf((float) Dflash2::kHeadDim);
    for (int idx = t; idx < n; idx += blockDim.x) {
        const int p = lo + idx;
        const float* kp = p < start ? ring_k + (size_t) (p & (kRing - 1)) * kKvDim + kvh * Dflash2::kHeadDim
                                    : nk + (size_t) (p - start) * kKvDim + kvh * Dflash2::kHeadDim;
        float d = 0.f;
        for (int e = 0; e < Dflash2::kHeadDim; e++) d += sq[e] * kp[e];
        sc[idx] = d * scale;
    }
    __syncthreads();
    if (t == 0) {
        float m = -INFINITY;
        for (int idx = 0; idx < n; idx++) m = fmaxf(m, sc[idx]);
        float s = 0.f;
        for (int idx = 0; idx < n; idx++) {
            sc[idx] = expf(sc[idx] - m);
            s += sc[idx];
        }
        inv_sum = 1.0f / s;
    }
    __syncthreads();
    float acc = 0.f;
    for (int idx = 0; idx < n; idx++) {
        const int p = lo + idx;
        const float* vp = p < start ? ring_v + (size_t) (p & (kRing - 1)) * kKvDim + kvh * Dflash2::kHeadDim
                                    : nv + (size_t) (p - start) * kKvDim + kvh * Dflash2::kHeadDim;
        acc += sc[idx] * vp[t];
    }
    out[(size_t) i * kQDim + hd * Dflash2::kHeadDim + t] = acc * inv_sum;
}

__global__ void silu_mul_kernel(const float* g, const float* u, float* a, int64_t n) {
    const int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x;
    if (i < n) a[i] = (g[i] / (1.0f + expf(-g[i]))) * u[i];
}

__global__ void add_kernel(const float* a, const float* b, float* out, int64_t n) {
    const int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

// The top kTopK logits of each row, largest first (ties: the smaller id).  grid = rows, block = 256.
__global__ void topk_kernel(const float* logits, int32_t* cand, float* val) {
    __shared__ float sv[256];
    __shared__ int si[256];
    __shared__ int chosen[Dflash2::kTopK];
    const float* lr = logits + (size_t) blockIdx.x * Dflash2::kVocab;
    for (int r = 0; r < Dflash2::kTopK; r++) {
        float bv = -INFINITY;
        int bi = -1;
        for (int idx = threadIdx.x; idx < Dflash2::kVocab; idx += blockDim.x) {
            bool taken = false;
            for (int k = 0; k < r; k++) taken = taken || chosen[k] == idx;
            if (taken) continue;
            const float v = lr[idx];
            if (bi < 0 || v > bv) {
                bv = v;
                bi = idx;
            }
        }
        sv[threadIdx.x] = bv;
        si[threadIdx.x] = bi;
        __syncthreads();
        if (threadIdx.x == 0) {
            float best = sv[0];
            int bidx = si[0];
            for (int j = 1; j < 256; j++)
                if (si[j] >= 0 && (bidx < 0 || sv[j] > best || (sv[j] == best && si[j] < bidx))) {
                    best = sv[j];
                    bidx = si[j];
                }
            chosen[r] = bidx;
            cand[blockIdx.x * Dflash2::kTopK + r] = bidx;
            val[blockIdx.x * Dflash2::kTopK + r] = best;
        }
        __syncthreads();
    }
}

// The candidate selector's path (z-lab's CandidateSelector.select at temperature 0).  One block of 256 threads (= the
// rank) walks the rows in order: score_k = logit_k + <pred_codebook[pred] * hp[row], succ_codebook[cand_k]>, pred = the
// best candidate.  The codebooks are Q4_K or Q8_0 tables (type): one row (token) is 256 values, dequantized element by element.
__global__ void selector_kernel(const int32_t* cand, const float* val, const float* hp, const uint8_t* pred_tab,
                                const uint8_t* succ_tab, int type, int anchor, int rows, int32_t* path) {
    __shared__ int pred;
    __shared__ float sc[Dflash2::kTopK];
    __shared__ float red[8];
    const int t = threadIdx.x;
    const size_t row_bytes = type == kType8_0 ? (size_t) (Dflash2::kRank / 32) * kQ8Bytes : (size_t) kQ4kBytes;
    if (t == 0) pred = anchor;
    __syncthreads();
    for (int s = 0; s < rows; s++) {
        const float qv = codebook_elem(pred_tab + (size_t) pred * row_bytes, type, t) * hp[s * Dflash2::kRank + t];
        for (int k = 0; k < Dflash2::kTopK; k++) {
            const int c = cand[s * Dflash2::kTopK + k];
            const float tot = block_sum(qv * codebook_elem(succ_tab + (size_t) c * row_bytes, type, t), red);
            if (t == 0) sc[k] = val[s * Dflash2::kTopK + k] + tot;
        }
        __syncthreads();
        if (t == 0) {
            int bk = 0;
            for (int k = 1; k < Dflash2::kTopK; k++)
                if (sc[k] > sc[bk]) bk = k;
            pred = cand[s * Dflash2::kTopK + bk];
            path[s] = pred;
        }
        __syncthreads();
    }
}

bool cuda_ok(cudaError_t e, std::string& err, const char* what) {
    if (e == cudaSuccess) return true;
    err = std::string(what) + ": " + cudaGetErrorString(e);
    return false;
}

}  // namespace

struct Dflash2::Impl {
    struct QW {
        const uint8_t* p = nullptr;
        int type = -1;
        int n_in = 0;
        int n_out = 0;
    };
    struct Layer {
        float* attn_norm = nullptr;
        float* q_norm = nullptr;
        float* k_norm = nullptr;
        float* ffn_norm = nullptr;
        float* conv_attn = nullptr;   // [2 taps][K][kHidden]
        float* conv_ffn = nullptr;
        QW q, k, v, o, conv_attn_proj, conv_ffn_proj, gate, up, down;
        float* ring_k = nullptr;      // [kRing][kKvDim]
        float* ring_v = nullptr;
    };

    bool loaded = false;
    Layer layers[kLayers];
    float* hidden_norm = nullptr;
    float* out_norm = nullptr;
    QW fc, sel_hidden;
    const uint8_t* sel_pred = nullptr;
    const uint8_t* sel_succ = nullptr;
    int sel_type = kType4K;           // the codebooks' GGML type: Q4_K or Q8_0
    std::vector<void*> allocs;

    void* xq = nullptr;               // native_q8_1 activation scratch
    size_t xq_bytes = 0;
    float *feat = nullptr, *fc_out = nullptr, *th = nullptr, *kraw = nullptr, *vraw = nullptr;
    float *h = nullptr, *res = nullptr, *xn = nullptr, *dyn = nullptr, *xc = nullptr, *qraw = nullptr, *qb = nullptr;
    float *nk = nullptr, *att = nullptr, *o_buf = nullptr, *oc = nullptr, *x2 = nullptr, *g = nullptr, *u = nullptr;
    float *a = nullptr, *m = nullptr, *fin = nullptr, *hp = nullptr, *logits = nullptr, *val = nullptr;
    int32_t *ids = nullptr, *cand = nullptr, *path = nullptr;

    int fed_hi = 0;
    int last_start = 0;

    ~Impl() {
        for (void* p : allocs) cudaFree(p);
    }

    bool alloc(size_t bytes, void** p, std::string& err) {
        if (!cuda_ok(cudaMalloc(p, bytes), err, "dflash2 alloc")) return false;
        allocs.push_back(*p);
        return true;
    }

    template <typename T>
    bool alloc_n(size_t n, T** p, std::string& err) {
        void* raw = nullptr;
        if (!alloc(n * sizeof(T), &raw, err)) return false;
        *p = static_cast<T*>(raw);
        return true;
    }

    bool upload(const strata::TensorInfo* t, const strata::GgufFile& gf, void** dst, std::string& err) {
        const uint64_t bytes = strata::tensor_payload_bytes(*t);
        if (!alloc(bytes, dst, err)) return false;
        return cuda_ok(cudaMemcpy(*dst, gf.tensor_data(*t), bytes, cudaMemcpyHostToDevice), err, "dflash2 upload");
    }

    // mmvq: y[c][n_out] = W . x[c] for ncols columns; x must have been quantized by quant() for this n_in.
    void quant(const float* x, int n_in, int ncols, void* st) {
        strata::kernels::native_quantize_q8_1(x, xq, n_in, ncols, st);
    }
    void mm(const QW& w, float* y, int ncols, void* st) {
        strata::kernels::native_mmvq(w.type, w.p, xq, y, w.n_in, w.n_out, ncols, st);
    }
};

Dflash2::Dflash2() : p_(new Impl()) {}

Dflash2::~Dflash2() = default;

bool Dflash2::loaded() const { return p_->loaded; }

int Dflash2::fed() const { return p_->fed_hi; }

int Dflash2::last_start() const { return p_->last_start; }

void Dflash2::unload() { p_.reset(new Impl()); }

void Dflash2::reset() {
    p_->fed_hi = 0;
    p_->last_start = 0;
}

bool Dflash2::load(const std::string& gguf_path, std::string& err) {
    unload();
    Impl& I = *p_;
    try {
        strata::GgufFile gf(gguf_path);
        bool ok = true;
        // A tensor with the given shape (GGUF order) and type, or an error naming it.
        auto need = [&](const std::string& name, std::vector<uint64_t> shape, uint32_t type) -> const strata::TensorInfo* {
            const strata::TensorInfo* t = gf.find(name);
            if (!t) {
                err = "dflash2: tensor " + name + " is missing";
                ok = false;
                return nullptr;
            }
            if (t->shape != shape || t->type != type) {
                err = "dflash2: tensor " + name + " has the wrong shape or type";
                ok = false;
                return nullptr;
            }
            return t;
        };
        auto qw = [&](const std::string& name, int n_in, int n_out, Impl::QW& w) -> bool {
            const strata::TensorInfo* t = gf.find(name);
            if (!t) {
                err = "dflash2: tensor " + name + " is missing";
                return false;
            }
            if (t->shape.size() != 2 || (int) t->shape[0] != n_in || (int) t->shape[1] != n_out ||
                !strata::kernels::native_mmvq_supported((int) t->type)) {
                err = "dflash2: tensor " + name + " has the wrong shape or an unsupported type";
                return false;
            }
            void* dev = nullptr;
            if (!I.upload(t, gf, &dev, err)) return false;
            w.p = static_cast<const uint8_t*>(dev);
            w.type = (int) t->type;
            w.n_in = n_in;
            w.n_out = n_out;
            return true;
        };
        auto f32 = [&](const std::string& name, std::vector<uint64_t> shape, float** dst) -> bool {
            const strata::TensorInfo* t = need(name, shape, 0);
            if (!t) return false;
            void* dev = nullptr;
            if (!I.upload(t, gf, &dev, err)) return false;
            *dst = static_cast<float*>(dev);
            return true;
        };
        const int H = kHidden, K = kTaps, F = kFfn, Q = kQDim, KV = kKvDim;
        if (!qw("fc.weight", kCtxFeat, H, I.fc)) return false;
        if (!f32("enc.output_norm.weight", {(uint64_t) H}, &I.hidden_norm)) return false;
        if (!f32("output_norm.weight", {(uint64_t) H}, &I.out_norm)) return false;
        if (!qw("selector_hidden.weight", H, kRank, I.sel_hidden)) return false;
        // The codebooks: one row per token, Q4_K or Q8_0 (both files of the drafter), the same type for both.
        auto codebook = [&](const std::string& name) -> const strata::TensorInfo* {
            const strata::TensorInfo* t = gf.find(name);
            if (!t) {
                err = "dflash2: tensor " + name + " is missing";
                ok = false;
                return nullptr;
            }
            if (t->shape != std::vector<uint64_t>{(uint64_t) kRank, (uint64_t) kVocab} || (t->type != kType4K && t->type != kType8_0)) {
                err = "dflash2: tensor " + name + " has the wrong shape or type";
                ok = false;
                return nullptr;
            }
            return t;
        };
        const strata::TensorInfo* sp = codebook("selector_predecessor.weight");
        const strata::TensorInfo* ss = codebook("selector_successor.weight");
        if (!sp || !ss) return false;
        if (sp->type != ss->type) {
            err = "dflash2: the selector codebooks have different types";
            return false;
        }
        I.sel_type = (int) sp->type;
        void* dev = nullptr;
        if (!I.upload(sp, gf, &dev, err)) return false;
        I.sel_pred = static_cast<const uint8_t*>(dev);
        if (!I.upload(ss, gf, &dev, err)) return false;
        I.sel_succ = static_cast<const uint8_t*>(dev);
        for (int l = 0; l < kLayers && ok; l++) {
            Impl::Layer& L = I.layers[l];
            auto name = [&](const char* suffix) { return "blk." + std::to_string(l) + "." + suffix; };
            if (!f32(name("attn_norm.weight"), {(uint64_t) H}, &L.attn_norm)) return false;
            if (!f32(name("attn_q_norm.weight"), {(uint64_t) kHeadDim}, &L.q_norm)) return false;
            if (!f32(name("attn_k_norm.weight"), {(uint64_t) kHeadDim}, &L.k_norm)) return false;
            if (!f32(name("ffn_norm.weight"), {(uint64_t) H}, &L.ffn_norm)) return false;
            if (!f32(name("attn_conv_base"), {(uint64_t) H, (uint64_t) K, 2}, &L.conv_attn)) return false;
            if (!f32(name("ffn_conv_base"), {(uint64_t) H, (uint64_t) K, 2}, &L.conv_ffn)) return false;
            if (!qw(name("attn_conv_proj.weight"), H, kConvOut, L.conv_attn_proj)) return false;
            if (!qw(name("ffn_conv_proj.weight"), H, kConvOut, L.conv_ffn_proj)) return false;
            if (!qw(name("attn_q.weight"), H, Q, L.q)) return false;
            if (!qw(name("attn_k.weight"), H, KV, L.k)) return false;
            if (!qw(name("attn_v.weight"), H, KV, L.v)) return false;
            if (!qw(name("attn_output.weight"), Q, H, L.o)) return false;
            if (!qw(name("ffn_gate.weight"), H, F, L.gate)) return false;
            if (!qw(name("ffn_up.weight"), H, F, L.up)) return false;
            if (!qw(name("ffn_down.weight"), F, H, L.down)) return false;
            if (!I.alloc((size_t) kRing * KV * sizeof(float), (void**) &L.ring_k, err)) return false;
            if (!I.alloc((size_t) kRing * KV * sizeof(float), (void**) &L.ring_v, err)) return false;
        }
        if (!ok) return false;
        // Scratch: the largest mmvq input is the fc's 12800 (ncols up to the block); the rows are at most kBlock.
        I.xq_bytes = strata::kernels::native_q8_1_bytes(kCtxFeat, kBlock);
        if (!I.alloc(I.xq_bytes, &I.xq, err)) return false;
        const size_t B = kBlock;
        if (!I.alloc_n(B * kCtxFeat, &I.feat, err) || !I.alloc_n(B * H, &I.fc_out, err) ||
            !I.alloc_n(B * H, &I.th, err) || !I.alloc_n(B * KV, &I.kraw, err) || !I.alloc_n(B * KV, &I.vraw, err) ||
            !I.alloc_n(B * H, &I.h, err) || !I.alloc_n(B * H, &I.res, err) || !I.alloc_n(B * H, &I.xn, err) ||
            !I.alloc_n(B * kConvOut, &I.dyn, err) || !I.alloc_n(B * H, &I.xc, err) ||
            !I.alloc_n(B * Q, &I.qraw, err) || !I.alloc_n(B * Q, &I.qb, err) || !I.alloc_n(B * KV, &I.nk, err) ||
            !I.alloc_n(B * Q, &I.att, err) || !I.alloc_n(B * H, &I.o_buf, err) || !I.alloc_n(B * H, &I.oc, err) ||
            !I.alloc_n(B * H, &I.x2, err) || !I.alloc_n(B * F, &I.g, err) || !I.alloc_n(B * F, &I.u, err) ||
            !I.alloc_n(B * F, &I.a, err) || !I.alloc_n(B * H, &I.m, err) || !I.alloc_n(B * H, &I.fin, err) ||
            !I.alloc_n((B - 1) * kRank, &I.hp, err) || !I.alloc_n((B - 1) * (size_t) kVocab, &I.logits, err) ||
            !I.alloc_n((B - 1) * kTopK, &I.val, err) || !I.alloc_n(B, &I.ids, err) ||
            !I.alloc_n((B - 1) * kTopK, &I.cand, err) || !I.alloc_n(B - 1, &I.path, err))
            return false;
    } catch (const std::exception& e) {
        err = std::string("dflash2: ") + e.what();
        unload();
        return false;
    }
    I.loaded = true;
    reset();
    return true;
}

bool Dflash2::feed(const uint16_t* feats_dev, int ctx_lo, int ctx_hi, void* stream, std::string& err, Dflash2Trace* trace) {
    Impl& I = *p_;
    if (!I.loaded) {
        err = "dflash2: not loaded";
        return false;
    }
    if (ctx_lo != I.fed_hi) {
        err = "dflash2: feed at " + std::to_string(ctx_lo) + ", the next position is " + std::to_string(I.fed_hi);
        return false;
    }
    if (trace) trace->th_norm.clear();
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    const int H = kHidden;
    int p = ctx_lo;
    while (p < ctx_hi) {
        // A chunk of at most kBlock positions that stays inside one ring wrap.
        const int c = std::min(std::min(kBlock, ctx_hi - p), kRing - (p & (kRing - 1)));
        const uint16_t* src = feats_dev + (size_t) (p - ctx_lo) * kCtxFeat;
        const int64_t n = (int64_t) c * kCtxFeat;
        bf16_to_f32_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, st>>>(src, I.feat, n);
        I.quant(I.feat, kCtxFeat, c, stream);
        I.mm(I.fc, I.fc_out, c, stream);
        rmsnorm_rows_kernel<<<c, 256, 0, st>>>(I.fc_out, I.hidden_norm, I.th, H, 1e-6f);
        if (trace) {
            const size_t at = trace->th_norm.size();
            trace->th_norm.resize(at + (size_t) c * H);
            if (!cuda_ok(cudaMemcpyAsync(trace->th_norm.data() + at, I.th, (size_t) c * H * sizeof(float),
                                         cudaMemcpyDeviceToHost, st), err, "dflash2 trace")) return false;
            if (!cuda_ok(cudaStreamSynchronize(st), err, "dflash2 trace")) return false;
        }
        for (int l = 0; l < kLayers; l++) {
            Impl::Layer& L = I.layers[l];
            I.quant(I.th, H, c, stream);
            I.mm(L.k, I.kraw, c, stream);
            I.mm(L.v, I.vraw, c, stream);
            head_norm_rope_kernel<<<dim3(c, kKvHeads), kHeadDim, 0, st>>>(I.kraw, L.ring_k, L.k_norm, p, kRing - 1, 1e-6f,
                                                                       10000.f);
            scatter_rows_kernel<<<c, 256, 0, st>>>(I.vraw, L.ring_v, p, kRing - 1);
        }
        if (!cuda_ok(cudaGetLastError(), err, "dflash2 feed")) return false;
        p += c;
    }
    I.fed_hi = ctx_hi;
    return true;
}

bool Dflash2::propose(int start, int32_t anchor, int block, const EmbedFn& embed, const HeadFn& head,
                      std::vector<int32_t>& draft, void* stream, std::string& err, Dflash2Trace* trace) {
    Impl& I = *p_;
    if (!I.loaded) {
        err = "dflash2: not loaded";
        return false;
    }
    if (block < 2 || block > kBlock) {
        err = "dflash2: block " + std::to_string(block) + " is outside 2.." + std::to_string(kBlock);
        return false;
    }
    if (start < 1 || start < I.last_start) {
        err = "dflash2: start " + std::to_string(start) + " is not after " + std::to_string(I.last_start);
        return false;
    }
    if (start > I.fed_hi) {
        err = "dflash2: positions up to " + std::to_string(start) + " are not fed (fed " + std::to_string(I.fed_hi) + ")";
        return false;
    }
    if (I.fed_hi > start) {   // the ring is the window: a position fed past start would overwrite a key this block reads
        err = "dflash2: " + std::to_string(I.fed_hi - start) + " positions fed beyond the start (the ring holds only the window)";
        return false;
    }
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    const int B = block;
    const int nb = B - 1;
    const int H = kHidden, Q = kQDim, F = kFfn;
    const float eps = 1e-6f;

    int32_t ids_h[kBlock];
    ids_h[0] = anchor;
    for (int i = 1; i < B; i++) ids_h[i] = kMaskId;
    if (!cuda_ok(cudaMemcpy(I.ids, ids_h, sizeof(int32_t) * B, cudaMemcpyHostToDevice), err, "dflash2 ids")) return false;
    if (!embed(I.ids, B, I.h, stream, err)) return false;

    if (trace) {
        trace->layer.assign(kLayers, std::vector<float>());
    }
    for (int l = 0; l < kLayers; l++) {
        Impl::Layer& L = I.layers[l];
        const size_t bh = (size_t) B * H;
        cudaMemcpyAsync(I.res, I.h, bh * sizeof(float), cudaMemcpyDeviceToDevice, st);
        rmsnorm_rows_kernel<<<B, 256, 0, st>>>(I.h, L.attn_norm, I.xn, H, eps);
        I.quant(I.xn, H, B, stream);
        I.mm(L.conv_attn_proj, I.dyn, B, stream);
        conv_kernel<<<dim3(B, H / 256), 256, 0, st>>>(I.xn, I.dyn, 0, L.conv_attn, I.xc, kTaps, kHidden / kGroupSize,
                                                    kGroupSize);
        I.quant(I.xc, H, B, stream);
        I.mm(L.q, I.qraw, B, stream);
        I.mm(L.k, I.kraw, B, stream);
        I.mm(L.v, I.vraw, B, stream);
        head_norm_rope_kernel<<<dim3(B, kHeads), kHeadDim, 0, st>>>(I.qraw, I.qb, L.q_norm, start, -1, eps, 10000.f);
        head_norm_rope_kernel<<<dim3(B, kKvHeads), kHeadDim, 0, st>>>(I.kraw, I.nk, L.k_norm, start, -1, eps, 10000.f);
        const size_t smem = (size_t) (kHeadDim + kWindow + kBlock) * sizeof(float);
        attention_kernel<<<dim3(B, kHeads), kHeadDim, smem, st>>>(I.qb, L.ring_k, L.ring_v, I.nk, I.vraw, I.att, start, B);
        I.quant(I.att, Q, B, stream);
        I.mm(L.o, I.o_buf, B, stream);
        conv_kernel<<<dim3(B, H / 256), 256, 0, st>>>(I.o_buf, I.dyn, kConvOut / 2, L.conv_attn + (size_t) kTaps * H,
                                                    I.oc, kTaps, kHidden / kGroupSize, kGroupSize);
        add_kernel<<<(unsigned) ((bh + 255) / 256), 256, 0, st>>>(I.res, I.oc, I.h, (int64_t) bh);
        cudaMemcpyAsync(I.res, I.h, bh * sizeof(float), cudaMemcpyDeviceToDevice, st);
        rmsnorm_rows_kernel<<<B, 256, 0, st>>>(I.h, L.ffn_norm, I.x2, H, eps);
        I.quant(I.x2, H, B, stream);
        I.mm(L.conv_ffn_proj, I.dyn, B, stream);
        conv_kernel<<<dim3(B, H / 256), 256, 0, st>>>(I.x2, I.dyn, 0, L.conv_ffn, I.xc, kTaps, kHidden / kGroupSize,
                                                    kGroupSize);
        I.quant(I.xc, H, B, stream);
        I.mm(L.gate, I.g, B, stream);
        I.mm(L.up, I.u, B, stream);
        silu_mul_kernel<<<(unsigned) ((B * (int64_t) F + 255) / 256), 256, 0, st>>>(I.g, I.u, I.a, (int64_t) B * F);
        I.quant(I.a, F, B, stream);
        I.mm(L.down, I.m, B, stream);
        conv_kernel<<<dim3(B, H / 256), 256, 0, st>>>(I.m, I.dyn, kConvOut / 2, L.conv_ffn + (size_t) kTaps * H, I.o_buf,
                                                    kTaps, kHidden / kGroupSize, kGroupSize);
        add_kernel<<<(unsigned) ((bh + 255) / 256), 256, 0, st>>>(I.res, I.o_buf, I.h, (int64_t) bh);
        if (!cuda_ok(cudaGetLastError(), err, "dflash2 propose")) return false;
        if (trace) {
            trace->layer[l].resize(bh);
            if (!cuda_ok(cudaMemcpyAsync(trace->layer[l].data(), I.h, bh * sizeof(float), cudaMemcpyDeviceToHost, st), err,
                         "dflash2 trace")) return false;
            if (!cuda_ok(cudaStreamSynchronize(st), err, "dflash2 trace")) return false;
        }
    }
    rmsnorm_rows_kernel<<<B, 256, 0, st>>>(I.h, I.out_norm, I.fin, H, eps);
    // The selector reads rows 1.. (the mask rows): the first row's output is not a draft.
    float* rows = I.fin + H;
    I.quant(rows, H, nb, stream);
    I.mm(I.sel_hidden, I.hp, nb, stream);
    if (!head(rows, nb, I.logits, stream, err)) return false;
    topk_kernel<<<nb, 256, 0, st>>>(I.logits, I.cand, I.val);
    selector_kernel<<<1, kRank, 0, st>>>(I.cand, I.val, I.hp, I.sel_pred, I.sel_succ, I.sel_type, anchor, nb, I.path);
    if (!cuda_ok(cudaGetLastError(), err, "dflash2 selector")) return false;
    std::vector<int32_t> path_h(nb);
    if (!cuda_ok(cudaMemcpyAsync(path_h.data(), I.path, sizeof(int32_t) * nb, cudaMemcpyDeviceToHost, st), err,
                 "dflash2 draft")) return false;
    if (trace) {
        trace->final.resize((size_t) nb * H);
        cudaMemcpyAsync(trace->final.data(), rows, (size_t) nb * H * sizeof(float), cudaMemcpyDeviceToHost, st);
        trace->cand.resize((size_t) nb * kTopK);
        cudaMemcpyAsync(trace->cand.data(), I.cand, (size_t) nb * kTopK * sizeof(int32_t), cudaMemcpyDeviceToHost, st);
        trace->cand_logit.resize((size_t) nb * kTopK);
        cudaMemcpyAsync(trace->cand_logit.data(), I.val, (size_t) nb * kTopK * sizeof(float), cudaMemcpyDeviceToHost, st);
    }
    if (!cuda_ok(cudaStreamSynchronize(st), err, "dflash2 sync")) return false;
    draft = path_h;
    I.last_start = start;
    return true;
}

}  // namespace strata::drafter
