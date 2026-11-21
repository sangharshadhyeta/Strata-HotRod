// include/strata/inkling2/ops_cpu.hpp - Inkling 2's ops on the CPU in plain float (double accumulation): M2a of docs/INKLING-PORT.md.
//
// Not a fast path.  Each op follows the formulas of the float64 reference (Maya's ref/inkling2.py, the specification) and is checked
// against that reference's seams by tests/core/inkling2_ops_test.cpp, so the GPU kernels that come next have a second, native
// oracle that reads the GGUF the way the engine will (Q8_0 / BF16 / F32 rows, NVFP4 experts with their per-expert scale).
// Layouts: activations are row-major [T][C]; a weight tensor of GGUF shape [in, out] is read as `out` rows of `in` values.
#pragma once

#include "strata/artifact/dequant.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/inkling2.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata {
namespace inkling2 {
namespace cpu {

using Mat = std::vector<float>;   // row-major [rows][cols]; the shape is carried by the caller

// ---- NVFP4: 64 values from a 36-byte block: four UE4M3 sub-scales (a 0.5 factor inside), then 32 bytes of E2M1 codes.
// Sub-block s (16 values) = codes qs[8 s .. 8 s + 7]: the low nibbles are its values 0-7, the high nibbles its values 8-15.
// A code's value is kvalues[code] (the E2M1 values doubled: 0 1 2 3 4 6 8 12, then their negatives) times the sub-scale.
inline float ue4m3_to_fp32(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    const int exp = (x >> 3) & 0xF, man = x & 7;
    const float raw = exp == 0 ? (float) man * (1.0f / 512.0f) : (1.0f + (float) man / 8.0f) * std::ldexp(1.0f, exp - 7);
    return raw * 0.5f;
}
inline void dequantize_nvfp4(const uint8_t* block, float* out) {
    static const int8_t kv[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    for (int s = 0; s < 4; ++s) {
        const float d = ue4m3_to_fp32(block[s]);
        const uint8_t* qs = block + 4 + 8 * s;
        for (int j = 0; j < 8; ++j) {
            out[16 * s + j] = d * (float) kv[qs[j] & 0xF];
            out[16 * s + 8 + j] = d * (float) kv[qs[j] >> 4];
        }
    }
}

/// Bytes of one row of `n` values in type `ty` (Q8_0, BF16, F32, NVFP4).
inline uint64_t row_bytes(uint32_t ty, int64_t n) {
    switch (ty) {
    case kQ8_0: return (uint64_t) (n / 32) * 34;
    case kBF16: return (uint64_t) n * 2;
    case kF32: return (uint64_t) n * 4;
    case kNVFP4: return (uint64_t) (n / 64) * 36;
    default: throw std::runtime_error("inkling2 cpu: unsupported tensor type " + std::to_string(ty));
    }
}

/// Row `row` (of `n` values) of the 2-D slice at `plane` (0 for a plain matrix; the expert index for a fused expert tensor).
inline void dequant_row(const GgufFile& g, const TensorInfo& t, int64_t plane, int64_t row, float* out) {
    const int64_t n = (int64_t) t.shape[0], rows = (int64_t) t.shape[1];
    const uint64_t rb = row_bytes(t.type, n);
    const uint8_t* p = g.tensor_data(t) + ((uint64_t) plane * (uint64_t) rows + (uint64_t) row) * rb;
    switch (t.type) {
    case kQ8_0: for (int64_t b = 0; b < n / 32; ++b) dequantize_q8_0(p + b * 34, out + b * 32); break;
    case kBF16: dequantize_bf16(p, out, (int) n); break;
    case kF32: dequantize_f32(p, out, (int) n); break;
    case kNVFP4: for (int64_t b = 0; b < n / 64; ++b) dequantize_nvfp4(p + b * 36, out + b * 64); break;
    default: throw std::runtime_error("inkling2 cpu: unsupported tensor type");
    }
}

/// The whole 2-D slice as floats: [rows][n].
inline Mat dequant_matrix(const GgufFile& g, const TensorInfo& t, int64_t plane = 0, int64_t row0 = 0, int64_t nrows = -1) {
    const int64_t n = (int64_t) t.shape[0], rows = nrows < 0 ? (int64_t) t.shape[1] - row0 : nrows;
    Mat m((size_t) (n * rows));
    for (int64_t r = 0; r < rows; ++r) dequant_row(g, t, plane, row0 + r, m.data() + r * n);
    return m;
}

inline double dot(const float* a, const float* b, int64_t n) {
    double s = 0;
    for (int64_t i = 0; i < n; ++i) s += (double) a[i] * (double) b[i];
    return s;
}

/// y[T][out] = x[T][in] @ W^T for W given as floats [out][in].
inline Mat matmul_t(const Mat& x, int64_t T, int64_t in, const Mat& W, int64_t out) {
    Mat y((size_t) (T * out));
    for (int64_t t = 0; t < T; ++t)
        for (int64_t o = 0; o < out; ++o) y[(size_t) (t * out + o)] = (float) dot(&x[(size_t) (t * in)], &W[(size_t) (o * in)], in);
    return y;
}
/// The same for a weight tensor read from the file.
inline Mat matmul_t(const GgufFile& g, const TensorInfo& w, const Mat& x, int64_t T) {
    return matmul_t(x, T, (int64_t) w.shape[0], dequant_matrix(g, w), (int64_t) w.shape[1]);
}

/// A 1-D (or [C]) F32 / BF16 tensor as floats.
inline Mat vec(const GgufFile& g, const TensorInfo& t) {
    Mat m((size_t) t.elements());
    const int64_t n = (int64_t) t.elements();
    if (t.type == kF32) dequantize_f32(g.tensor_data(t), m.data(), (int) n);
    else if (t.type == kBF16) dequantize_bf16(g.tensor_data(t), m.data(), (int) n);
    else throw std::runtime_error("inkling2 cpu: vec() takes F32 / BF16");
    return m;
}

/// x * rsqrt(mean(x^2) + eps) (* w) on each of T rows of C.
inline void rms(float* x, int64_t T, int64_t C, const float* w, float eps) {
    for (int64_t t = 0; t < T; ++t) {
        float* r = x + t * C;
        double ss = 0;
        for (int64_t i = 0; i < C; ++i) ss += (double) r[i] * (double) r[i];
        const float s = (float) (1.0 / std::sqrt(ss / (double) C + (double) eps));
        for (int64_t i = 0; i < C; ++i) r[i] = r[i] * s * (w ? w[i] : 1.0f);
    }
}

/// x * min(1, th / (||x|| + 1e-6)) on each of `n` vectors of `d`.
inline void l2clip(float* x, int64_t n, int64_t d, float th) {
    for (int64_t i = 0; i < n; ++i) {
        float* r = x + i * d;
        double ss = 0;
        for (int64_t j = 0; j < d; ++j) ss += (double) r[j] * (double) r[j];
        const float s = std::min(1.0f, (float) ((double) th / (std::sqrt(ss) + 1e-6)));
        for (int64_t j = 0; j < d; ++j) r[j] *= s;
    }
}

/// The short conv's weights folded as the reference folds master's layout: w [C][taps] * sc, the last tap (the current token) += sc,
/// sc = 1 / sqrt(taps' + 1) with taps' = taps (static) or 2 * taps (a full layer's k / v).  `w` is [C][taps].
inline Mat fold_conv(const Mat& w, int64_t C, int64_t taps, bool dynamic) {
    const float sc = (float) (1.0 / std::sqrt((double) ((dynamic ? 2 : 1) * taps + 1)));
    Mat f(w.size());
    for (int64_t c = 0; c < C; ++c)
        for (int64_t j = 0; j < taps; ++j) f[(size_t) (c * taps + j)] = w[(size_t) (c * taps + j)] * sc + (j == taps - 1 ? sc : 0.0f);
    return f;
}

/// y[t][c] = sum_j f[c][j] x[t - (taps - 1) + j][c] (zeros before the first token); a full layer's `dyn` [T][heads * taps] adds
/// sum_lag dyn[t][head(c) * taps + lag] x[t - lag][c], head(c) = c / (C / heads), lag 0 the current token.
inline Mat conv(const Mat& x, int64_t T, int64_t C, const Mat& f, int64_t taps, const Mat* dyn = nullptr, int64_t heads = 0) {
    Mat y((size_t) (T * C), 0.0f);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t c = 0; c < C; ++c) {
            double s = 0;
            for (int64_t j = 0; j < taps; ++j) {
                const int64_t src = t - (taps - 1) + j;
                if (src >= 0) s += (double) f[(size_t) (c * taps + j)] * (double) x[(size_t) (src * C + c)];
            }
            if (dyn != nullptr) {
                const int64_t h = c / (C / heads);
                for (int64_t lag = 0; lag < taps; ++lag) {
                    const int64_t src = t - lag;
                    if (src >= 0) s += (double) (*dyn)[(size_t) (t * heads * taps + h * taps + lag)] * (double) x[(size_t) (src * C + c)];
                }
            }
            y[(size_t) (t * C + c)] = (float) s;
        }
    return y;
}

inline float sigmoidf(float x) { return (float) (1.0 / (1.0 + std::exp(-(double) x))); }

/// The attention residual: states = blocks + [x] ([n][T][C]), each RMS-normed without weight (eps 1e-6); for each of `nq` queries q
/// ([nq][C]): softmax over the n states of (q . s) / sqrt(C), the normed states weighted by it.  Returns [nq][T][C].
inline std::vector<Mat> attn_res(const std::vector<const Mat*>& states, int64_t T, int64_t C, const float* q, int64_t nq) {
    const size_t n = states.size();
    std::vector<Mat> s(n);
    for (size_t i = 0; i < n; ++i) { s[i] = *states[i]; rms(s[i].data(), T, C, nullptr, 1e-6f); }
    std::vector<Mat> out((size_t) nq, Mat((size_t) (T * C), 0.0f));
    for (int64_t t = 0; t < T; ++t)
        for (int64_t k = 0; k < nq; ++k) {
            std::vector<double> lg(n);
            double mx = -1e300;
            for (size_t i = 0; i < n; ++i) { lg[i] = dot(q + k * C, &s[i][(size_t) (t * C)], C) / std::sqrt((double) C); mx = std::max(mx, lg[i]); }
            double z = 0;
            for (size_t i = 0; i < n; ++i) { lg[i] = std::exp(lg[i] - mx); z += lg[i]; }
            for (size_t i = 0; i < n; ++i) {
                const float p = (float) (lg[i] / z);
                for (int64_t c = 0; c < C; ++c) out[(size_t) k][(size_t) (t * C + c)] += p * s[i][(size_t) (t * C + c)];
            }
        }
    return out;
}

struct AttnSeams {   // what the reference records inside the attention (for the tests)
    Mat k_conv, v_conv, q, k, v, r, gated, out;
};

/// A layer's attention over positions 0..T-1 (no loop messages): q, r, gate and sink come from xq, k from xk, v from xv (a sliding layer
/// passes the same normed input three times; a full layer its three separately normed inputs).  A full layer's k / v short convs add
/// per-token dynamic taps, its q and r are scaled by tau(pos), and its attention has no window.  Fills `seams` (the reference's names:
/// k_conv, v_conv before the clips; q, k, v, r after; attn_gated; and `out` = after attn_output, before the post conv).
inline Mat attention(const GgufFile& g, const Config& c, const LayerTensors& L, const Mat& xq, const Mat& xk, const Mat& xv, int64_t T, AttnSeams& sm) {
    const int64_t H = c.n_head, KV = c.n_head_kv, D = c.head_dim, R = c.rel_states;
    const bool sliding = !L.full;
    Mat q = matmul_t(g, *L.attn_q, xq, T), k = matmul_t(g, *L.attn_k, xk, T), v = matmul_t(g, *L.attn_v, xv, T);
    Mat r = matmul_t(g, *L.attn_rel, xq, T), gate = matmul_t(g, *L.attn_gate, xq, T), sink = matmul_t(g, *L.attn_sink_proj, xq, T);
    if (sliding) {
        const Mat kw = fold_conv(vec(g, *L.attn_k_conv), KV * D, c.conv_taps, false), vw = fold_conv(vec(g, *L.attn_v_conv), KV * D, c.conv_taps, false);
        k = conv(k, T, KV * D, kw, c.conv_taps);
        v = conv(v, T, KV * D, vw, c.conv_taps);
    } else {   // dynamic taps: xk @ k_conv_proj^T (heads x taps per token), scaled as master's layout leaves them unscaled
        const float dsc = (float) (1.0 / std::sqrt((double) (2 * c.conv_taps + 1)));
        Mat kd = matmul_t(g, *L.attn_k_conv_proj, xk, T), vd = matmul_t(g, *L.attn_v_conv_proj, xv, T);
        for (float& z : kd) z *= dsc;
        for (float& z : vd) z *= dsc;
        const Mat kw = fold_conv(vec(g, *L.attn_k_conv), KV * D, c.conv_taps, true), vw = fold_conv(vec(g, *L.attn_v_conv), KV * D, c.conv_taps, true);
        k = conv(k, T, KV * D, kw, c.conv_taps, &kd, KV);
        v = conv(v, T, KV * D, vw, c.conv_taps, &vd, KV);
    }
    sm.k_conv = k; sm.v_conv = v;
    const Mat qclip = vec(g, *L.attn_q_clip), rclip = vec(g, *L.attn_rel_clip);   // [H][D], [H][R]
    for (int64_t t = 0; t < T; ++t)
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t d = 0; d < D; ++d) q[(size_t) ((t * H + h) * D + d)] *= qclip[(size_t) (h * D + d)];
            for (int64_t i = 0; i < R; ++i) r[(size_t) ((t * H + h) * R + i)] *= rclip[(size_t) (h * R + i)];
        }
    l2clip(q.data(), T * H, D, c.qk_clip);
    l2clip(k.data(), T * KV, D, c.qk_clip);
    rms(v.data(), T * KV, D, nullptr, c.rms_eps);
    l2clip(r.data(), T * H, R, c.rel_clip);
    if (!sliding) {   // the full layers' log scaling of q and r: tau = 1 + alpha ln max(1, (pos + 1) / floor)
        for (int64_t t = 0; t < T; ++t) {
            const float tau = (float) (1.0 + (double) c.log_scale_alpha * std::log(std::max(1.0, (double) (t + 1) / (double) c.log_scale_floor)));
            for (int64_t i = 0; i < H * D; ++i) q[(size_t) (t * H * D + i)] *= tau;
            for (int64_t i = 0; i < H * R; ++i) r[(size_t) (t * H * R + i)] *= tau;
        }
    }
    sm.q = q; sm.k = k; sm.v = v; sm.r = r;
    const int64_t extent = sliding ? c.window : c.rel_extent;          // a sliding layer's extent is its window
    const Mat proj = vec(g, *L.attn_rel_bias);                          // [extent][R]
    Mat o((size_t) (T * H * D), 0.0f);
    const int64_t grp = H / KV;
    for (int64_t t = 0; t < T; ++t)
        for (int64_t h = 0; h < H; ++h) {
            const int64_t kvh = h / grp;
            std::vector<double> lg;
            std::vector<int64_t> src;
            for (int64_t s = sliding ? std::max<int64_t>(0, t - c.window + 1) : 0; s <= t; ++s) {
                const int64_t dist = t - s;
                double bias = 0;
                if (dist < extent) bias = dot(&r[(size_t) ((t * H + h) * R)], &proj[(size_t) (dist * R)], R);
                lg.push_back(dot(&q[(size_t) ((t * H + h) * D)], &k[(size_t) ((s * KV + kvh) * D)], D) / std::sqrt((double) D) + bias);
                src.push_back(s);
            }
            const double sk = sink[(size_t) (t * H + h)];
            double mx = sk;
            for (double l : lg) mx = std::max(mx, l);
            double z = std::exp(sk - mx);
            for (double& l : lg) { l = std::exp(l - mx); z += l; }
            for (size_t i = 0; i < lg.size(); ++i) {
                const float p = (float) (lg[i] / z);
                for (int64_t d = 0; d < D; ++d) o[(size_t) ((t * H + h) * D + d)] += p * v[(size_t) ((src[i] * KV + kvh) * D + d)];
            }
        }
    for (size_t i = 0; i < o.size(); ++i) o[i] *= sigmoidf(gate[i]);
    sm.gated = o;
    Mat out = matmul_t(g, *L.attn_output, o, T);
    sm.out = out;
    return out;
}

struct MoeSeams { std::vector<int> picks; };   // [T][k] sorted expert indices

/// The MoE on x [T][E] (already ffn-normed): sigmoid router, top-k of (score + bias), weights normalized x route scale, SwiGLU with
/// softcaps, NVFP4 experts with their per-expert scales (absent scale = 1).
inline Mat moe(const GgufFile& g, const Config& c, const LayerTensors& L, const Mat& x, int64_t T, MoeSeams* sm) {
    const int64_t E = c.n_embd, NE = c.n_expert, K = c.n_expert_used, FF = c.n_ff_exp;
    const Mat router = dequant_matrix(g, *L.router), bias = vec(g, *L.router_bias);
    Mat scores = matmul_t(x, T, E, router, NE);
    for (float& s : scores) s = sigmoidf(s);
    std::vector<std::vector<int>> pick((size_t) T);
    std::vector<std::vector<float>> wts((size_t) T);
    if (sm) sm->picks.assign((size_t) (T * K), 0);
    for (int64_t t = 0; t < T; ++t) {
        std::vector<int> idx((size_t) NE);
        for (int e = 0; e < NE; ++e) idx[(size_t) e] = e;
        std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return scores[(size_t) (t * NE + a)] + bias[(size_t) a] > scores[(size_t) (t * NE + b)] + bias[(size_t) b]; });
        idx.resize((size_t) K);
        double z = 0;
        for (int e : idx) z += scores[(size_t) (t * NE + e)];
        for (int e : idx) wts[(size_t) t].push_back((float) ((double) scores[(size_t) (t * NE + e)] / (z + 1e-20) * (double) c.expert_weight_scale));
        pick[(size_t) t] = idx;
        if (sm) { std::vector<int> srt = idx; std::sort(srt.begin(), srt.end()); for (int64_t j = 0; j < K; ++j) sm->picks[(size_t) (t * K + j)] = srt[(size_t) j]; }
    }
    const Mat gs = L.gate_up_scale ? vec(g, *L.gate_up_scale) : Mat(), ds = L.down_scale ? vec(g, *L.down_scale) : Mat();
    Mat out((size_t) (T * E), 0.0f);
    std::vector<char> used((size_t) NE, 0);
    for (int64_t t = 0; t < T; ++t) for (int e : pick[(size_t) t]) used[(size_t) e] = 1;
    for (int e = 0; e < NE; ++e) {
        if (!used[(size_t) e]) continue;
        const Mat gu = dequant_matrix(g, *L.gate_up_exps, e, 0, 2 * FF);   // [2 FF][E]: gate rows first
        const Mat dn = dequant_matrix(g, *L.down_exps, e, 0, E);           // [E][FF]
        const float sg = gs.empty() ? 1.0f : gs[(size_t) e], sd = ds.empty() ? 1.0f : ds[(size_t) e];
        for (int64_t t = 0; t < T; ++t) {
            int j = -1;
            for (int jj = 0; jj < K; ++jj) if (pick[(size_t) t][(size_t) jj] == e) j = jj;
            if (j < 0) continue;
            std::vector<float> h((size_t) FF);
            for (int64_t f = 0; f < FF; ++f) {
                const double gt = dot(&x[(size_t) (t * E)], &gu[(size_t) (f * E)], E) * sg;
                const double up = dot(&x[(size_t) (t * E)], &gu[(size_t) ((FF + f) * E)], E) * sg;
                h[(size_t) f] = (float) (std::tanh(gt / c.situ_beta) * c.situ_beta * (1.0 / (1.0 + std::exp(-gt))) * std::tanh(up / c.situ_linear_beta) * c.situ_linear_beta);
            }
            for (int64_t o = 0; o < E; ++o)
                out[(size_t) (t * E + o)] += (float) (dot(h.data(), &dn[(size_t) (o * FF)], FF) * sd * (double) wts[(size_t) t][(size_t) j]);
        }
    }
    return out;
}

struct VisitSeams { AttnSeams attn; Mat attn_out, ffn_out, out, loop_q_in, loop_ffn_in, moe_in; std::vector<int> picks; };   // moe_in: the MoE's normed input

/// A core layer's loop control on its input x [T][E] with the pass-before's message `msg`: module 0 attn (q's), 1 k, 2 v, 3 mlp.
///   g = [rms(x); rms(msg)] @ down^T (rank 64);  x' = (1 - tanh(g forget^T)) x + tanh(g gain^T) msg     (weightless norms, eps 1e-6)
inline Mat control(const GgufFile& gf, const LayerTensors& L, int module, const Mat& x, const Mat& msg, int64_t T, int64_t E) {
    Mat xn = x, mn = msg;
    rms(xn.data(), T, E, nullptr, 1e-6f);
    rms(mn.data(), T, E, nullptr, 1e-6f);
    Mat cat((size_t) (T * 2 * E));
    for (int64_t t = 0; t < T; ++t) {
        std::memcpy(&cat[(size_t) (t * 2 * E)], &xn[(size_t) (t * E)], (size_t) E * 4);
        std::memcpy(&cat[(size_t) (t * 2 * E + E)], &mn[(size_t) (t * E)], (size_t) E * 4);
    }
    const Mat gg = matmul_t(gf, *L.loop_down[module], cat, T);
    Mat forget = matmul_t(gf, *L.loop_forget[module], gg, T), gain = matmul_t(gf, *L.loop_gain[module], gg, T);
    Mat y(x.size());
    for (size_t i = 0; i < y.size(); ++i) y[i] = (1.0f - std::tanh(forget[i])) * x[i] + std::tanh(gain[i]) * msg[i];
    return y;
}

/// One layer visit of stored layer `l`.  h: the stream [T][E]; blocks: the block states so far (the reference appends the layer's input at a
/// block boundary, stored index % block == 0, and the layer's own FFN residual already sees it).  `msgs`: the four messages (q's, k's,
/// v's, the MoE's) of a core layer in a pass after the first, else null.  Returns the new stream (r + the MoE's output); `blocks` is updated.
inline Mat layer_visit(const GgufFile& g, const Config& c, const Layout& lay, int64_t l, const Mat& h, std::vector<Mat>& blocks, int64_t T, VisitSeams& vs, const Mat* msgs = nullptr) {
    const LayerTensors& L = lay.layers[(size_t) l];
    const int64_t E = c.n_embd;
    Mat ins[3];
    if (l == 0) {
        ins[0] = ins[1] = ins[2] = h;
    } else {   // the attention residual over the block states and the stream, with this layer's query (a full layer has three)
        std::vector<const Mat*> st;
        for (const Mat& b : blocks) st.push_back(&b);
        st.push_back(&h);
        const Mat q = vec(g, *L.attn_res_query);
        const int64_t nq = L.full ? 3 : 1;
        std::vector<Mat> o = attn_res(st, T, E, q.data(), nq);
        for (int i = 0; i < 3; ++i) ins[i] = nq == 1 ? o[0] : o[(size_t) i];
    }
    if (msgs != nullptr) {   // before the input norms: a sliding layer then norms three different inputs
        for (int j = 0; j < 3; ++j) ins[j] = control(g, L, j, ins[j], msgs[j], T, E);
        vs.loop_q_in = ins[0];
    }
    Mat xs[3];
    if (!L.full) {
        const Mat w = vec(g, *L.attn_norm);
        for (int j = 0; j < 3; ++j) {
            if (j > 0 && msgs == nullptr) { xs[j] = xs[0]; continue; }
            xs[j] = ins[j];
            rms(xs[j].data(), T, E, w.data(), c.rms_eps);
        }
    } else {
        const TensorInfo* nw[3] = {L.attn_q_inp_norm, L.attn_k_inp_norm, L.attn_v_inp_norm};
        for (int i = 0; i < 3; ++i) {
            xs[i] = ins[i];
            const Mat w = vec(g, *nw[i]);
            rms(xs[i].data(), T, E, w.data(), c.rms_eps);
        }
    }
    Mat a = attention(g, c, L, xs[0], xs[1], xs[2], T, vs.attn);
    a = conv(a, T, E, fold_conv(vec(g, *L.attn_post_conv), E, c.conv_taps, false), c.conv_taps);
    vs.attn_out = a;
    std::vector<Mat> bl = blocks;   // the blocks this layer's FFN residual sees
    Mat r(h.size());
    if (l % c.attn_res_block == 0) {
        bl.push_back(h);
        r = a;
    } else {
        for (size_t i = 0; i < r.size(); ++i) r[i] = h[i] + a[i];
    }
    std::vector<const Mat*> st;
    for (const Mat& b : bl) st.push_back(&b);
    st.push_back(&r);
    const Mat fq = vec(g, *L.ffn_res_query);
    Mat m_in = attn_res(st, T, E, fq.data(), 1)[0];
    if (msgs != nullptr) { m_in = control(g, L, 3, m_in, msgs[3], T, E); vs.loop_ffn_in = m_in; }
    { const Mat w = vec(g, *L.ffn_norm); rms(m_in.data(), T, E, w.data(), c.rms_eps); }
    vs.moe_in = m_in;
    MoeSeams ms;
    Mat m = moe(g, c, L, m_in, T, &ms);
    vs.picks = ms.picks;
    m = conv(m, T, E, fold_conv(vec(g, *L.ffn_post_conv), E, c.conv_taps, false), c.conv_taps);
    vs.ffn_out = m;
    Mat out(h.size());
    for (size_t i = 0; i < out.size(); ++i) out[i] = r[i] + m[i];
    vs.out = out;
    if (l % c.attn_res_block == 0) blocks.push_back(h);
    return out;
}

/// The embedding of `toks`: Q8_0 rows of the token table, RMS-normed with token_embd_norm.
inline Mat embed(const GgufFile& g, const Config& c, const Layout& lay, const std::vector<int64_t>& toks) {
    const int64_t T = (int64_t) toks.size(), E = c.n_embd;
    Mat h((size_t) (T * E));
    for (int64_t t = 0; t < T; ++t) dequant_row(g, *lay.token_embd, 0, toks[(size_t) t], &h[(size_t) (t * E)]);
    const Mat w = vec(g, *lay.token_embd_norm);
    rms(h.data(), T, E, w.data(), c.rms_eps);
    return h;
}

/// The whole model on `toks` (teacher-forced, positions 0..T-1): the prelude once, the core `num_loops` times (a pass after the first reads
/// the one before it through the messages), the rest once, then the output attention residual, norm, head, softcap and log-softmax over the
/// real vocabulary.  Returns logprobs [T][n_vocab_real].  `on_visit(v, seams)` sees every visit (v counts visits, 0..n_visits-1).
template <class F> inline Mat forward(const GgufFile& g, const Config& c, const Layout& lay, const std::vector<int64_t>& toks, F&& on_visit) {
    const int64_t T = (int64_t) toks.size(), E = c.n_embd;
    Mat h = embed(g, c, lay, toks);
    std::vector<Mat> blocks;
    int64_t visit = 0;
    const auto run = [&](int64_t l, const Mat* msgs) {
        VisitSeams vs;
        h = layer_visit(g, c, lay, l, h, blocks, T, vs, msgs);
        on_visit(visit++, vs);
    };
    for (int64_t l = 0; l < c.core_start; ++l) run(l, nullptr);
    const Mat pre_h = h;
    const std::vector<Mat> pre_blocks = blocks;
    const Mat lrq = vec(g, *lay.loop_res_query), lkq = vec(g, *lay.loop_kv_res_query);   // [2 core_len][E]
    for (int64_t p = 0; p < c.num_loops; ++p) {
        std::vector<Mat> bank;
        Mat last;
        if (p > 0) {   // the pass before: the blocks it added after its first, and its final stream; this pass restarts from the prelude
            bank.assign(blocks.begin() + (int64_t) pre_blocks.size() + 1, blocks.end());
            last = h;
            h = pre_h;
            blocks = pre_blocks;
        }
        for (int64_t l = c.core_start; l < c.core_end; ++l) {
            if (p == 0) { run(l, nullptr); continue; }
            const int64_t i = l - c.core_start;
            std::vector<const Mat*> st;
            for (const Mat& b : bank) st.push_back(&b);
            st.push_back(&last);
            std::vector<Mat> mq = attn_res(st, T, E, &lrq[(size_t) (2 * i * E)], 2);    // [q's message, the MoE's message]
            std::vector<Mat> mk = attn_res(st, T, E, &lkq[(size_t) (2 * i * E)], 2);    // [k's message, v's message]
            const Mat msgs[4] = {mq[0], mk[0], mk[1], mq[1]};
            run(l, msgs);
        }
    }
    for (int64_t l = c.core_end; l < c.n_stored; ++l) run(l, nullptr);
    // the head: the output attention residual (its query), the output norm, the logits (softcapped), log-softmax over the real rows
    std::vector<const Mat*> st;
    for (const Mat& b : blocks) st.push_back(&b);
    st.push_back(&h);
    const Mat oq = vec(g, *lay.output_res_query);
    Mat x = attn_res(st, T, E, oq.data(), 1)[0];
    { const Mat w = vec(g, *lay.output_norm); rms(x.data(), T, E, w.data(), c.rms_eps); }
    const int64_t NV = c.n_vocab_real;
    Mat lp((size_t) (T * NV));
    std::vector<float> row((size_t) E);
    for (int64_t v = 0; v < NV; ++v) {   // one row of the head at a time: the whole matrix as floats would be 1.9 GB
        dequant_row(g, *lay.output, 0, v, row.data());
        for (int64_t t = 0; t < T; ++t) {
            const double z = dot(&x[(size_t) (t * E)], row.data(), E) / (double) c.final_softcap;
            lp[(size_t) (t * NV + v)] = (float) (std::tanh(z) * (double) c.final_softcap);
        }
    }
    for (int64_t t = 0; t < T; ++t) {
        float* r = &lp[(size_t) (t * NV)];
        double mx = -1e300;
        for (int64_t v = 0; v < NV; ++v) mx = std::max(mx, (double) r[v]);
        double z = 0;
        for (int64_t v = 0; v < NV; ++v) z += std::exp((double) r[v] - mx);
        const double lse = mx + std::log(z);
        for (int64_t v = 0; v < NV; ++v) r[v] = (float) ((double) r[v] - lse);
    }
    return lp;
}

}  // namespace cpu
}  // namespace inkling2
}  // namespace strata
