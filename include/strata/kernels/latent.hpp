// Adapted from Edge0's edge0-strata (https://github.com/Edge0-AI/Edge0, MIT licence: a fork of Niko1221's Strata), branch hotrod-edge0.
// Taken as it is.
// include/strata/kernels/latent.hpp - the latent MoE's engine-side kernels (new file).
//
// The A part reuses native_expert_grouped (40,40,ELL,I); these two kernels cover the
// only latent-specific data movement left in the verify graph:
//   * latent_plan_rows: rewrite the GPU plan entries' activation rows so A reads the
//     q8_1 rows of xl (row = token*G + label[expert]) instead of the plain token rows.
//   * latent_u_grouped: U for every routed entry: y[r] = yl[r] @ U[labels[ids[r]]].T.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// tok[j] = (base_token + dst[j]/K) * G + labels[ids[dst[j]]] for j < counts[1] (all GPU entries).
/// `ids` points at this window group's ids block; labels = the layer's 512 group labels (device).
void latent_plan_rows(int32_t* tok, const int32_t* dst, const int32_t* counts, const int32_t* ids,
                      const int32_t* labels_dev, int64_t E, int base_token, int G, int K, void* stream);

/// One block per routed entry (cap_entries of them): quantized yl rows (q8_1) times the route's
/// U[group] slice (Q8_0 or Q4_0 rows [G*H, ELL], fmt 8 or 4); the H-wide result lands at out[r*H .. r*H+H].
void latent_u_grouped(const uint8_t* u_q8, const uint8_t* yl_q8, const int32_t* ids, const int32_t* labels_dev,
                      int64_t E, int64_t cap_entries, int ELL, int H, float* out, void* stream, int fmt = 8, int G = 0, bool packed = false);
/// hotrod-edge0: packed = the device U was repacked by latent_u_repack_q4 (Q4_0 only, G = the group count, latent_u_pack_q4_ok(ELL)): the
/// instruction-bound per-entry kernel is replaced by one that loads a block's nibbles as one int4.  Same integer dot; only the float add order differs.
bool latent_u_pack_q4_ok(int ELL);
/// The D matrix [rows, cols] in its raw Q4_0 (fmt 4) or Q8_0 (fmt 8) blocks to f16 (the prompt path's GEMM over a chunk); cols % 32 == 0.
void latent_dequant_f16(const uint8_t* src, int64_t rows, int cols, int fmt, uint16_t* dst, void* stream);
/// The same for U rows that latent_u_repack_q4 repacked (Q4_0 only).
void latent_dequant_packed_f16(const uint8_t* src, int64_t rows, int cols, uint16_t* dst, void* stream);
void latent_u_repack_q4(uint8_t* u, int64_t rows, int ELL, void* stream);   ///< in place: [nb x 18 B blocks] -> [nb x 16 B nibbles][nb x f16 scales] per row

/// knife-0 (STRATA_LATENT_U_GPU): the CPU share's U computed on the GPU into parts_ directly.
/// One block per routed entry; a GPU row (r in dst[0..count)) copies hit_out, a CPU row computes U[labels[ids[r]]] . hq[r].
void latent_u_matvec(const uint8_t* u_q8, const uint8_t* hq, const int32_t* ids, const int32_t* labels_dev,
                     const int32_t* dst, const int32_t* count, int64_t E, int64_t n_entries, int ELL, int H,
                     int G, float* parts, void* stream, int fmt = 8);

}  // namespace strata::kernels