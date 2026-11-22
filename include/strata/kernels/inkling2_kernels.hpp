// include/strata/kernels/inkling2_kernels.hpp - Inkling 2's new device kernels (docs/INKLING-PORT.md F2): only what the engine does not already have.
//
// Reused from the engine, not here: gdn_conv_step (the static short convs), iq_mmvq + quantize_q8_1_rows (Q8_0 / NVFP4 matvecs), rms_norm_weighted,
// and the router (inkling2_router.hpp, from the DeepSeek branch).  Float32 throughout and written for correctness first: every kernel is checked against
// the CPU oracle (include/strata/inkling2/ops_cpu.hpp) and the reference's seams.  One token a call unless stated (decode style; a prompt is a loop of tokens).
#pragma once

#include <cstdint>

namespace strata::kernels::ik {

/// y[t][o] = sum_i W[o][i] x[t][i]; W float [out][in] (the small tensors: sink, conv_proj, router, loop control, residual queries).
void matmul_f32(const float* W, const float* x, float* y, int T, int in, int out, void* stream);

/// Q / K / V / r preparation for T tokens (positions pos0 .. pos0 + T - 1), in place:
///   q [T][H][D] *= qclip[H][D], L2 clip (qk_clip), x tau;   k [T][KV][D] L2 clip (qk_clip);   v [T][KV][D] weightless RMS (eps);
///   r [T][H][R] *= rclip[H][R], L2 clip (rel_clip), x tau;  tau = full ? 1 + alpha ln max(1, (pos + 1) / floor) : 1.
void qkv_prep(float* q, float* k, float* v, float* r, const float* qclip, const float* rclip, int T, int H, int KV, int D, int R, float qk_clip,
              float rel_clip, float eps, bool full, float alpha, float log_floor, int pos0, void* stream);

/// A full layer's short conv with dynamic taps, one token: state [C][taps-1] (oldest first, slides), f [C][taps] (the folded static taps),
/// dyn [heads][taps] (lag 0 = the current token).  out[c] = sum_j f[c][j] x[t - taps + 1 + j] + sum_lag dyn[head(c)][lag] x[t - lag].
void conv_dynamic(float* state, const float* x, const float* f, const float* dyn, float* out, int C, int taps, int heads, void* stream);

/// Append one token's k, v ([KV][D] each) at position `pos` of a cache [cap][KV][D].
void kv_append(float* Kc, float* Vc, const float* k, const float* v, int pos, int KV, int D, void* stream, int T = 1);   // T tokens at pos .. pos + T - 1

/// One token's attention over the cache (positions 0 .. pos, the current already appended): logits q.k / sqrt(D) + bias(dist) with
/// bias = r[h] . rel_bias[dist] for dist < extent (else 0), the window (0 = none) limiting the positions, a sink logit per head in the softmax
/// (value zero), the output times sigmoid(gate).  scratch: [H][cap] floats.  out [H][D].
void attention(const float* q, const float* r, const float* gate, const float* sink, const float* Kc, const float* Vc, const float* rel_bias, float* out,
               float* scratch, int pos, int window, int extent, int cap, int H, int KV, int D, int R, void* stream, int T = 1);   // T tokens at pos .. pos + T - 1: q / r / gate / sink / out packed per token, scratch [T][H][cap]

/// The attention residual for one token: states [n][C] (the blocks and the stream), nq queries q [nq][C]; per query: softmax over the n states of
/// (q . rms(s)) / sqrt(C) (weightless RMS, eps 1e-6), out[k] = sum_n p_n rms(s_n).  out [nq][C].
/// T tokens (grid y): token t reads states + t * st_tok and writes out[t * out_tok + k * out_q + c] (0 = packed [nq][C] per token).
void attn_res(const float* states, int n, const float* q, int nq, float* out, int C, void* stream, int T = 1, size_t st_tok = 0, size_t out_tok = 0, size_t out_q = 0);

/// The loop control gate of a core layer, one token: cat = [rms(x); rms(msg)] (weightless, eps 1e-6) -> [2C];  then, after the caller has run
/// cat @ down^T -> g (rank) and g @ forget^T, g @ gain^T (matmul_f32), mix:  y = (1 - tanh(forget)) x + tanh(gain) msg.
void control_cat(const float* x, const float* msg, float* cat, int C, void* stream, int T = 1);   // T rows
void control_mix(const float* x, const float* msg, const float* forget, const float* gain, float* y, int C, void* stream);

/// SITU: h = tanh(g s / beta) beta sigmoid(g s) tanh(u s / gamma) gamma, s = the expert's gate/up scale; n values.
void situ(const float* g, const float* u, float* h, int n, float s, float beta, float gamma, void* stream);

/// acc[i] += a * y[i];  out[i] = a[i] + b[i];  x[i] = tanh(x[i] / cap) * cap.
void axpy(float* acc, const float* y, float a, int n, void* stream);
void add(const float* a, const float* b, float* out, int n, void* stream);
void softcap(float* x, int n, float cap, void* stream);

// (The grouped experts and the router are model-independent and live in moe_group.hpp, shared with the other engines.)

/// A Q8_0 row (ggml block_q8_0: half scale + 32 int8) of `n` values to float (the token embedding).
void dequant_q8_0_row(const void* row, float* out, int n, void* stream);

}  // namespace strata::kernels::ik
