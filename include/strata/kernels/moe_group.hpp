// include/strata/kernels/moe_group.hpp - model-independent MoE kernels: a router with a selectable score, and the grouped experts of a layer visit.
//
// Written for Inkling 2 (docs/INKLING-PORT.md) but with no model's numbers in them, so the Qwen, Edge0 and DeepSeek paths can call them behind their own switches:
// shapes (C, FF, K, the number of experts) are run-time, the expert weight type (NVFP4 or Q8_0 today; a new type is one decode function) and the activation
// (plain SwiGLU, DeepSeek's clamped SwiGLU, Inkling's softcapped SITU) are parameters, per-expert scales are optional, and the experts are reached through a
// pointer array, so any tier that can say where an expert's bytes are (VRAM slot, pinned RAM arena, a zero-copy mapping) can drive them.
// Grouped = one launch for all K picks' gate + up + activation, one for down, one to combine: Maya's idea (maya-ik2's grouped expert kernels), our code.
// The router is the DeepSeek branch's biased top-k (src/dsv41 route_kernel), unified here with a score mode (the "one kernel, a scoring mode" of the plan).
// Float32 activations: no int8 quantization of the input, the weights decoded exactly.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::moe {

enum class Score { Sigmoid = 0, SqrtSoftplus = 1 };

/// One block a token.  Experts are PICKED by score + bias (the selection bias never enters the weights), the k best, the lowest index winning a tie; the weights are the picked
/// scores' share of their sum times `route_scale`.  logits [T][n]; n <= 1024, k <= 16.  idx [T][k] (best first), w [T][k].  A non-finite input still yields k valid, distinct ids.
void route(const float* logits, const float* bias, int T, int n, int k, float route_scale, Score score, int32_t* idx, float* w, void* stream);

enum class ActKind { Silu = 0, ClampedSilu = 1, Situ = 2 };
/// Silu: silu(g) u.   ClampedSilu(p0 = limit): g = min(g, lim), u = clamp(u, -lim, lim), then silu(g) u (DeepSeek).
/// Situ(p0 = beta, p1 = gamma): tanh(g / beta) beta sigmoid(g) tanh(u / gamma) gamma (Inkling 2).
struct Act { ActKind kind = ActKind::Silu; float p0 = 0.0f, p1 = 0.0f; };

/// From the router's picks (device): slot s's gate+up plane and down plane addresses (layer bases + expert * plane bytes), its gate/up scale and its combine weight
/// (route weight x down scale).  Either scale array may be null (= 1).
void experts_prepare(const int32_t* idx, const float* route_w, const uint8_t* gu_base, size_t gu_plane, const uint8_t* dn_base, size_t dn_plane, const float* gate_up_scale,
                     const float* down_scale, int K, const uint8_t** gu_ptr, const uint8_t** dn_ptr, float* sg, float* wd, void* stream);
/// The same without the addresses, for a tier that places experts itself (a VRAM slot cache: the host writes gu_ptr / dn_ptr): only the scales and combine weights.
void experts_scales(const int32_t* idx, const float* route_w, const float* gate_up_scale, const float* down_scale, int K, float* sg, float* wd, void* stream);
/// h[s][r] = act(gate_r . x, up_r . x) for r < FF: slot s's plane holds the gate rows, then (at up_off bytes) the up rows; `ty` the ggml type (40 NVFP4, 8 Q8_0); x [C] float.
void experts_gate_up(const uint8_t* const* gu_ptr, int ty, size_t up_off, const float* x, int C, int FF, const float* sg, Act act, float* h, int K, void* stream);
/// y[s][c] = down_c . h[s] for c < C (h [K][FF]; slot s's down rows at dn_ptr[s]).
void experts_down(const uint8_t* const* dn_ptr, int ty, const float* h, int C, int FF, float* y, int K, void* stream);
/// The int8-activation path (~4x fewer ALU ops a weight; the activation is quantized to q8_1 blocks - 36 bytes a 32 values - so the result carries int8 activation noise, as the
/// engine's other quantized matvecs do): quantize_act turns `rows` rows of n floats (n a multiple of 32) into xq [rows][n / 32][36]; the _q8 kernels take it in place of x / h.
void quantize_act(const float* x, int n, int rows, void* xq, void* stream);
void experts_gate_up_q8(const uint8_t* const* gu_ptr, int ty, size_t up_off, const void* xq, int C, int FF, const float* sg, Act act, float* h, int K, void* stream);
void experts_down_q8(const uint8_t* const* dn_ptr, int ty, const void* hq, int C, int FF, float* y, int K, void* stream);
/// out[c] = sum_s wd[s] y[s][c].
void experts_combine(const float* y, const float* wd, int K, int C, float* out, void* stream);

/// Whether the weight type has a decode here.
bool type_supported(int ggml_type);

}  // namespace strata::kernels::moe
