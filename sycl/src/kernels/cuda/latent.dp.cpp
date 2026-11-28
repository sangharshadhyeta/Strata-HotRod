// Adapted from Edge0's edge0-strata (https://github.com/Edge0-AI/Edge0, MIT licence: a fork of Niko1221's Strata), branch hotrod-edge0.
// Taken as it is.
// src/kernels/cuda/latent.cu - the latent MoE's engine-side kernels (new file).
//
// Bit-identical dot to iq_mmvq(8)'s vec_dot_q8_0_q8_1: per block
//   acc += dx * dw * sum_k(xs[k] * ws[k])
// where dx is the q8_1 block's f16 scale (struct {half ds[2]; int8 qs[32]}: the first half,
// bytes 0..1; the values follow at bytes 4..35) and dw the q8_0 block's f16 scale
// ({half d; int8 qs[32]}: bytes 0..1; values at 2..33).  The per-block 32-lane sum is
// computed directly (same integer arithmetic as the dp4a lanes, no alignment assumptions).
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_math.hpp"
#include "strata/sycl_queue.hpp"
#include "strata/kernels/latent.hpp"
#include "strata/kernels/dp4a.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

__dpct_inline__ void latent_plan_rows_k(int32_t *tok, const int32_t *dst,
                                        const int32_t *counts,
                                        const int32_t *ids,
                                        const int32_t *labels, int E,
                                        int base_token, int G, int K) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int nj =
        counts[1]; // all GPU entries (the VRAM share + the PCIe share)
    for (int j = item_ct1.get_local_id(2); j < nj;
         j += item_ct1.get_local_range(2)) {
        const int d = dst[j];
        const int32_t e = ids[d];
        const int g = (e >= 0 && e < E) ? labels[e] : 0;
        tok[j] = (base_token + d / K) * G + g;
    }
}

void latent_plan_rows(int32_t* tok, const int32_t* dst, const int32_t* counts, const int32_t* ids,
                      const int32_t* labels_dev, int64_t E, int base_token, int G, int K, void* stream) {
    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::use_root_sync};

    strata::q_of(stream)
        ->parallel_for<dpct_kernel_name<class latent_plan_rows_k_ed98b9>>(
            sycl::nd_range<3>(sycl::range(1, 1, 64), sycl::range(1, 1, 64)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                latent_plan_rows_k(tok, dst, counts, ids, labels_dev, (int)E,
                                   base_token, G, K);
            });
}

// latent_u_k optimized: coalesced dp4a, one warp per output row.
// Old version: one THREAD per row with scalar byte loads and no reuse -> ~25x read
// amplification, ~0.84 ms/layer. New: lanes stride the ELL words of a row as int32
// (built from 2-byte-aligned ushort pairs, q8_0 stores {half d; i8 qs[32]} at +2),
// dp4a per word, warp-shuffle reduction. U[g] rows are read once per (route,row)
// and shared across routes through L2 as before, but now at stream bandwidth.
__dpct_inline__ int latent_ld4_i8(const uint8_t *p) {
    const unsigned short a = *(const unsigned short*) p;
    const unsigned short b = *(const unsigned short*) (p + 2);
    return (int) ((unsigned) a | ((unsigned) b << 16));   // a = p0|p1<<8, b<<16 = p2<<16|p3<<24
}


// Q4DU: the weight block's dot with a q8_1 activation block, fmt 8 (Q8_0) or 4 (Q4_0).
// The q8_1 block is always {f16 d; f16 s; int8 qs[32]} (36 B, values at +4); the weight block
// is {f16 d; int8 qs[32]} (34 B) for fmt 8, {f16 d; uint8 nib[16]} (18 B) for fmt 4 with
// element j in byte j's low nibble and element j+16 in its high one, value (code - 8) * d -
// the same centred codes as iq_kernels.cu's vec_dot_q4_0_q8_1 (q4_0_centred).
__dpct_inline__ float lat_blk_dot_q4(const uint8_t *wb, const uint8_t *xb,
                                     int j) {
    // nibble words are built from byte loads: a Q4_0 block is only 2-byte aligned inside its
    // row (18 B stride), a raw int dereference at +2 is misaligned and faults on sm_120.
    const float dw = sycl::vec<sycl::half, 1>(*(const sycl::half *)wb)
                         .convert<float, sycl::rounding_mode::automatic>()[0];
    const float dx = sycl::vec<sycl::half, 1>(*(const sycl::half *)xb)
                         .convert<float, sycl::rounding_mode::automatic>()[0];
    // 4 bytes at wb+2+j: each byte's low nibble is element (j+k), its high nibble element (j+k+16).
    int w0 = 0, w16 = 0, xw = 0, xw16 = 0;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int byte_ = (int) wb[2 + j + k];
        w0 |= (((byte_ & 0xF) - 8) & 0xFF) << (8 * k);          // mask: a negative code's sign
        w16 |= ((((byte_ >> 4) & 0xF) - 8) & 0xFF) << (8 * k);  // bits must not bleed into the
        xw |= (int) (uint32_t) xb[4 + j + k] << (8 * k);        // neighbouring byte of the word
        xw16 |= (int) (uint32_t) xb[4 + j + k + 16] << (8 * k); // (dp4a reads s8 itself)
    }
    return dx * dw * (float)(strata::dp4a(xw, w0, 0) + strata::dp4a(xw16, w16, 0));
}

__dpct_inline__ void latent_u_k(const uint8_t *__restrict__ u_q8,
                                const uint8_t *__restrict__ ylq,
                                const int32_t *__restrict__ ids,
                                const int32_t *__restrict__ labels, int E,
                                int cap_entries, int ELL, int H,
                                float *__restrict__ out, int fmt) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int r = item_ct1.get_group(2);
    if (r >= cap_entries) return;
    const int32_t e = ids[r];
    const int g = (e >= 0 && e < E) ? labels[e] : 0;
    const int nb = ELL / 32;            // value blocks per row
    const int wrow = nb * (fmt == 4 ? 18 : 34);   // Q4DU: the weight row's bytes
    const uint8_t* w = u_q8 + (size_t) g * (size_t) H * (size_t) wrow;
    const uint8_t* x = ylq + (size_t) r * (size_t) (nb * 36);
    float* o = out + (size_t) r * (size_t) H;
    const int warp = item_ct1.get_local_id(2) >> 5,
              lane = item_ct1.get_local_id(2) & 31;
    const int h =
        (int)(item_ct1.get_group(1) * (item_ct1.get_local_range(2) >> 5)) +
        warp;
    if (h >= H) return;
    const uint8_t* wb = w + (size_t) h * (size_t) wrow;
    float facc = 0.0f;
    if (fmt == 4) {
        for (int wi = lane; wi < nb * 4; wi += 32) {   // 4 int4 words per block (16 nibble bytes)
            const int b = wi >> 2, j = (wi & 3) * 4;
            facc += lat_blk_dot_q4(wb + (size_t) b * 18, x + (size_t) b * 36, j);
        }
    } else {
    for (int wi = lane; wi < nb * 8; wi += 32) {
        const int b = wi >> 3, j = (wi & 7) * 4;
        const float dx =
            sycl::vec<sycl::half, 1>(*(const sycl::half *)(x + (size_t)b * 36))
                .convert<float, sycl::rounding_mode::automatic>()[0];
        const float dw =
            sycl::vec<sycl::half, 1>(*(const sycl::half *)(wb + (size_t)b * 34))
                .convert<float, sycl::rounding_mode::automatic>()[0];
        const int xw = *(const int*) (x + (size_t) b * 36 + 4 + j);
        const int ww = latent_ld4_i8(wb + (size_t) b * 34 + 2 + j);
        facc += dx * dw * (float)strata::dp4a(xw, ww, 0);
    }
    }
    #pragma unroll
    /*
    DPCT1108: '__shfl_down_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    for (int off = 16; off > 0; off >>= 1) facc +=
        dpct::experimental::shift_sub_group_left(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
            facc, off);
    if (lane == 0) o[h] = facc;
}

// latent-experts branch: packed Q4_0 U.  The per-entry kernel's Q4 dot builds every dp4a word from byte loads (a Q4_0 block is 18 B: only 2-byte
// aligned), ~60 instructions per 8 values; nsys on the 145B: 385 us a layer-window, instruction-bound (a by-group variant that read each
// row once was SLOWER, 518 us, which is how that was found out).  The device copy of U is repacked ONCE at upload into
// [nb x 16 B nibbles][nb x f16 scales] per row (the same bytes per row), so a lane loads a block's 16 nibble bytes as one int4 and
// unpacks them with bit operations.  The integer dot is exact, so only the float accumulation order differs from the per-entry kernel.
__dpct_inline__ void latent_u_repack_q4_k(uint8_t *__restrict__ u, int64_t rows,
                                          int nb, uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto sm_all = (uint8_t *)dpct_local;
    const int warp = item_ct1.get_local_id(2) >> 5,
              lane = item_ct1.get_local_id(2) & 31;
    const int64_t row =
        (int64_t)item_ct1.get_group(2) * (item_ct1.get_local_range(2) >> 5) +
        warp;
    if (row >= rows) return;
    const int wrow = nb * 18;
    uint8_t* sm = sm_all + (size_t) warp * (size_t) wrow;
    uint8_t* gp = u + (size_t) row * (size_t) wrow;
#pragma unroll
    for (int i = lane; i < wrow / 4; i += 32)((int *)sm)[i] =
        ((const int *)gp)[i];
    sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
    for (int b = lane; b < nb; b += 32) {
        const uint8_t* blk = sm + (size_t) b * 18;
        gp[(size_t) nb * 16 + (size_t) b * 2] = blk[0];
        gp[(size_t) nb * 16 + (size_t) b * 2 + 1] = blk[1];
        #pragma unroll
        for (int k = 0; k < 16; ++k) gp[(size_t) b * 16 + k] = blk[2 + k];
    }
}

constexpr int kUPackRows = 4;   // output rows a warp keeps in registers: each x block is loaded once for all of them
/*
DPCT1110: The total declared local variable size in device function
latent_u_p4_k exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void latent_u_p4_k(const uint8_t *__restrict__ u,
                                   const uint8_t *__restrict__ ylq,
                                   const int32_t *__restrict__ ids,
                                   const int32_t *__restrict__ labels, int E,
                                   int cap_entries, int ELL, int H,
                                   float *__restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int g = item_ct1.get_group(2);
    const int nb = ELL / 32, wrow = nb * 18;
    const int warp = item_ct1.get_local_id(2) >> 5,
              lane = item_ct1.get_local_id(2) & 31;
    const int h0 =
        ((int)item_ct1.get_group(1) * (item_ct1.get_local_range(2) >> 5) +
         warp) *
        kUPackRows;
    if (h0 >= H) return;
    bool any = false;
    for (int r = 0; r < cap_entries; ++r) {
        const int32_t e = ids[r];
        any |= ((e >= 0 && e < E) ? labels[e] : 0) == g;
    }
    if (!any) return;
    sycl::int4 nib[kUPackRows][2];
    float dw[kUPackRows][2];
    #pragma unroll
    for (int i = 0; i < kUPackRows; ++i) {
        const int hh =
            sycl::min(h0 + i, H - 1); // a row past H repeats the last one (its
                                      // result is not written)
        const uint8_t* row = u + ((size_t) g * (size_t) H + (size_t) hh) * (size_t) wrow;
        #pragma unroll
        for (int s = 0; s < 2; ++s) {
            const int b = lane + 32 * s;
            if (b < nb) {
                nib[i][s] = *(const sycl::int4 *)(row + (size_t)b * 16);
                dw[i][s] =
                    sycl::vec<sycl::half, 1>(
                        *(const sycl::half *)(row + (size_t)nb * 16 +
                                              (size_t)b * 2))
                        .convert<float, sycl::rounding_mode::automatic>()[0];
            } else {
                nib[i][s] = sycl::int4(0, 0, 0, 0);
                dw[i][s] = 0.0f;
            }
        }
    }
    for (int r = 0; r < cap_entries; ++r) {
        const int32_t e = ids[r];
        if (((e >= 0 && e < E) ? labels[e] : 0) != g) continue;
        const uint8_t* x = ylq + (size_t) r * (size_t) (nb * 36);
        float acc[kUPackRows] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int s = 0; s < 2; ++s) {
            const int b = lane + 32 * s;
            if (b >= nb) break;
            const uint8_t* xb = x + (size_t) b * 36;
            const float dx =
                sycl::vec<sycl::half, 1>(*(const sycl::half *)xb)
                    .convert<float, sycl::rounding_mode::automatic>()[0];
            int xq[8];
            #pragma unroll
            for (int k = 0; k < 8; ++k) xq[k] = *(const int*) (xb + 4 + 4 * k);
            #pragma unroll
            for (int i = 0; i < kUPackRows; ++i) {
                const unsigned w[4] = {
                    (unsigned)nib[i][s].x(), (unsigned)nib[i][s].y(),
                    (unsigned)nib[i][s].z(), (unsigned)nib[i][s].w()};
                int sumi = 0;
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const unsigned lo = dpct::vectorized_binary<sycl::uchar4>(
                        w[k] & 0x0F0F0F0Fu, 0x08080808u,
                        std::minus<>()); // elements 4k .. 4k+3, centred (code -
                                         // 8)
                    const unsigned hi = dpct::vectorized_binary<sycl::uchar4>(
                        (w[k] >> 4) & 0x0F0F0F0Fu, 0x08080808u,
                        std::minus<>()); // elements 16+4k .. 16+4k+3
                    sumi = strata::dp4a(xq[k], (int)lo, sumi);
                    sumi = strata::dp4a(xq[4 + k], (int)hi, sumi);
                }
                acc[i] += dx * dw[i][s] * (float) sumi;
            }
        }
        #pragma unroll
        for (int i = 0; i < kUPackRows; ++i) {
            float v = acc[i];
            #pragma unroll
            /*
            DPCT1108: '__shfl_down_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            for (int off = 16; off > 0; off >>= 1) v +=
                dpct::experimental::shift_sub_group_left(
                    0xffffffffu,
                    sycl::ext::oneapi::this_work_item::get_sub_group(), v, off);
            if (lane == 0 && h0 + i < H) out[(size_t) r * (size_t) H + (size_t) (h0 + i)] = v;
        }
    }
}

// latent-experts branch (prompt path): the D matrix [rows, cols] (Q4_0 18-byte or Q8_0 34-byte blocks, raw layout) to f16, for one GEMM over a chunk.
__dpct_inline__ void latent_dequant_f16_k(const uint8_t *__restrict__ src,
                                          int64_t n_blocks, int fmt,
                                          uint16_t *__restrict__ dst) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t b =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (b >= n_blocks) return;
    const uint8_t* blk = src + (size_t) b * (fmt == 4 ? 18 : 34);
    const float d = sycl::vec<sycl::half, 1>(*(const sycl::half *)blk)
                        .convert<float, sycl::rounding_mode::automatic>()[0];
    sycl::half *o = (sycl::half *)dst + (size_t)b * 32;
    if (fmt == 4) {
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int byte_ = blk[2 + j];
            o[j] =
                sycl::vec<float, 1>(d * (float)((byte_ & 0xF) - 8))
                    .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
            o[j + 16] =
                sycl::vec<float, 1>(d * (float)((byte_ >> 4) - 8))
                    .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
        }
    } else {
        #pragma unroll
        for (int j = 0; j < 32; ++j) o[j] =
            sycl::vec<float, 1>(d * (float)(int8_t)blk[2 + j])
                .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    }
}

// the same for U rows repacked by latent_u_repack_q4: per row [nb x 16 B nibbles][nb x f16 scales]
__dpct_inline__ void
latent_dequant_packed_f16_k(const uint8_t *__restrict__ src, int64_t rows,
                            int nb, uint16_t *__restrict__ dst) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t t =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (t >= rows * (int64_t) nb) return;
    const int64_t row = t / nb;
    const int b = (int) (t - row * nb);
    const uint8_t* rp = src + (size_t) row * (size_t) (nb * 18);
    const float d =
        sycl::vec<sycl::half, 1>(
            *(const sycl::half *)(rp + (size_t)nb * 16 + (size_t)b * 2))
            .convert<float, sycl::rounding_mode::automatic>()[0];
    const uint8_t* nib = rp + (size_t) b * 16;
    sycl::half *o = (sycl::half *)dst + (size_t)t * 32;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const int byte_ = nib[j];
        o[j] = sycl::vec<float, 1>(d * (float)((byte_ & 0xF) - 8))
                   .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
        o[j + 16] =
            sycl::vec<float, 1>(d * (float)((byte_ >> 4) - 8))
                .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    }
}

void latent_dequant_packed_f16(const uint8_t* src, int64_t rows, int cols, uint16_t* dst, void* stream) {
    const int nb = cols / 32;
    const int64_t n = rows * (int64_t) nb;
    if (n <= 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(
            strata::q_of(stream)->get_device(),
            {sycl::aspect::fp16});

        strata::q_of(stream)
            ->parallel_for<
                dpct_kernel_name<class latent_dequant_packed_f16_k_c4255f>>(
                sycl::nd_range<3>(
                    sycl::range(1, 1, (unsigned)((n + 127) / 128)) *
                        sycl::range(1, 1, 128),
                    sycl::range(1, 1, 128)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    latent_dequant_packed_f16_k(src, rows, nb, dst);
                });
    }
}

__dpct_inline__ void latent_scale_rows_k(float *__restrict__ x,
                                         const float *__restrict__ s,
                                         int64_t rows, int cols) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= rows * (int64_t) cols) return;
    x[i] *= s[i / cols];
}

void latent_scale_rows(float* x, const float* s, int64_t rows, int cols, void* stream) {
    const int64_t n = rows * (int64_t) cols;
    if (n <= 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class latent_scale_rows_k_1019bf>>(
                sycl::nd_range<3>(
                    sycl::range(1, 1, (unsigned)((n + 255) / 256)) *
                        sycl::range(1, 1, 256),
                    sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    latent_scale_rows_k(x, s, rows, cols);
                });
    }
}

void latent_dequant_f16(const uint8_t* src, int64_t rows, int cols, int fmt, uint16_t* dst, void* stream) {
    const int64_t nb = rows * (int64_t) (cols / 32);
    if (nb <= 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(
            strata::q_of(stream)->get_device(),
            {sycl::aspect::fp16});

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class latent_dequant_f16_k_7f9bae>>(
                sycl::nd_range<3>(
                    sycl::range(1, 1, (unsigned)((nb + 127) / 128)) *
                        sycl::range(1, 1, 128),
                    sycl::range(1, 1, 128)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    latent_dequant_f16_k(src, nb, fmt, dst);
                });
    }
}

bool latent_u_pack_q4_ok(int ELL) { return ELL > 0 && ELL % 32 == 0 && ((ELL / 32) * 18) % 16 == 0 && (ELL / 32) <= 64; }

void latent_u_repack_q4(uint8_t* u, int64_t rows, int ELL, void* stream) {
    if (rows <= 0) return;
    const int nb = ELL / 32;
    const int warps = 4;
    const size_t smem = (size_t) warps * (size_t) nb * 18;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range(smem), cgh);

                cgh.parallel_for<
                    dpct_kernel_name<class latent_u_repack_q4_k_a4ee6b>>(
                    sycl::nd_range<3>(
                        sycl::range(1, 1,
                                    (unsigned)((rows + warps - 1) / warps)) *
                            sycl::range(1, 1, warps * 32),
                        sycl::range(1, 1, warps * 32)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        latent_u_repack_q4_k(
                            u, rows, nb,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
            });
    }
}

void latent_u_grouped(const uint8_t* u_q8, const uint8_t* ylq, const int32_t* ids, const int32_t* labels_dev,
                      int64_t E, int64_t cap_entries, int ELL, int H, float* out, void* stream, int fmt, int G, bool packed) {
    if (cap_entries <= 0) return;
    if (u_q8 == nullptr || ylq == nullptr || ids == nullptr || labels_dev == nullptr || out == nullptr) {
        std::fprintf(stderr, "latent_u_grouped NULL ptr: u=%p ylq=%p ids=%p labels=%p out=%p E=%lld cap=%lld ELL=%d H=%d\n",
                     (const void*) u_q8, (const void*) ylq, (const void*) ids, (const void*) labels_dev,
                     (const void*) out, (long long) E, (long long) cap_entries, ELL, H);
        std::fflush(stderr);
        std::abort();
    }
    const int warps_per_block = 8;   // 256 threads
    const int rows_per_block = warps_per_block;
    if (packed && fmt == 4 && G > 0 && latent_u_pack_q4_ok(ELL)) {   // the device U was repacked at upload (latent_u_repack_q4)
        dpct::dim3 gp((unsigned)G,
                      (unsigned)((H + warps_per_block * kUPackRows - 1) /
                                 (warps_per_block * kUPackRows)));
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};
            dpct::has_capability_or_fail(
                strata::q_of(stream)->get_device(),
                {sycl::aspect::fp16});

            strata::q_of(stream)
                ->parallel_for<dpct_kernel_name<class latent_u_p4_k_233515>>(
                    sycl::nd_range<3>(
                        gp * sycl::range(1, 1, warps_per_block * 32),
                        sycl::range(1, 1, warps_per_block * 32)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            latent_u_p4_k(u_q8, ylq, ids, labels_dev, (int)E,
                                          (int)cap_entries, ELL, H, out);
                        });
        }
        return;
    }
    dpct::dim3 grid((unsigned)cap_entries,
                    (unsigned)((H + rows_per_block - 1) / rows_per_block));
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(
            strata::q_of(stream)->get_device(),
            {sycl::aspect::fp16});

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class latent_u_k_346094>>(
                sycl::nd_range<3>(grid *
                                      sycl::range(1, 1, warps_per_block * 32),
                                  sycl::range(1, 1, warps_per_block * 32)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        latent_u_k(u_q8, ylq, ids, labels_dev, (int)E,
                                   (int)cap_entries, ELL, H, out, fmt);
                    });
    }
}

// knife-0 (STRATA_LATENT_U_GPU): the CPU share's U computed on the GPU and written straight into parts_.
// One block per routed row r (like latent_u_k), 8 warps.  A GPU row (r in dst[0..count)) copies hit_out;
// a CPU row computes U[labels[ids[r]]] . hq[r] with the same dp4a dot as latent_u_k (bit-compatible to ~1e-7).
__dpct_inline__ void latent_u_matvec_k(
    const uint8_t *__restrict__ u_q8, const uint8_t *__restrict__ hq,
    const int32_t *__restrict__ ids, const int32_t *__restrict__ labels,
    const int32_t *__restrict__ dst, const int32_t *__restrict__ count, int E,
    int n_entries, int ELL, int H, float *__restrict__ parts, int fmt) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int g = item_ct1.get_group(2);
    const int warp = item_ct1.get_local_id(2) >> 5,
              lane = item_ct1.get_local_id(2) & 31;
    const int hout =
        (int)(item_ct1.get_group(1) * (item_ct1.get_local_range(2) >> 5)) +
        warp;
    const int nb = ELL / 32, wrow = nb * (fmt == 4 ? 18 : 34);   // Q4DU
    auto &c_shared = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &hq_s =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<uint8_t[2016]>(
            sycl::ext::oneapi::this_work_item::get_work_group<
                3>()); // nb*36 = 56*36, ELL=1792 (latent) fixed
    if (item_ct1.get_local_id(2) == 0) c_shared = *count;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    const int c = c_shared;
    if (hout >= H) return;
    const uint8_t* wb = u_q8 + (size_t) g * (size_t) H * (size_t) wrow + (size_t) hout * (size_t) wrow;
    for (int r = 0; r < n_entries; ++r) {
        const int32_t e = ids[r];
        if (e < 0 || e >= E || labels[e] != g) continue;
        bool gpu = false;
#pragma unroll
        for (int i = 0; i < c; ++i) if (dst[i] == r) { gpu = true; break; }
        if (gpu) continue;
        const uint8_t* x = hq + (size_t) r * (size_t) (nb * 36);
#pragma unroll
        for (int i = item_ct1.get_local_id(2); i < nb * 36;
             i += item_ct1.get_local_range(2)) hq_s[i] = x[i]; // once per block
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        float facc = 0.0f;
        if (fmt == 4) {
            for (int wi = lane; wi < nb * 4; wi += 32) {
                const int b = wi >> 2, j = (wi & 3) * 4;
                facc += lat_blk_dot_q4(wb + (size_t) b * 18, hq_s + (size_t) b * 36, j);
            }
        } else {
        for (int wi = lane; wi < nb * 8; wi += 32) {
            const int b = wi >> 3, j = (wi & 7) * 4;
            const float dx =
                sycl::vec<sycl::half, 1>(
                    *(const sycl::half *)(hq_s + (size_t)b * 36))
                    .convert<float, sycl::rounding_mode::automatic>()[0];
            /*
            DPCT1098: The '*' expression is used instead of the __ldg call.
            These two expressions do not provide the exact same functionality.
            Check the generated code for potential precision and/or performance
            issues.
            */
            const float dw =
                sycl::vec<sycl::half, 1>(*(wb + (size_t)b * 34))
                    .convert<float, sycl::rounding_mode::automatic>()[0];
            const int xw = *(const int*) (hq_s + (size_t) b * 36 + 4 + j);
            const int ww = latent_ld4_i8(wb + (size_t) b * 34 + 2 + j);
            facc += dx * dw * (float)strata::dp4a(xw, ww, 0);
        }
        }
        #pragma unroll
        /*
        DPCT1108: '__shfl_down_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        for (int off = 16; off > 0; off >>= 1) facc +=
            dpct::experimental::shift_sub_group_left(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                facc, off);
        if (lane == 0) parts[(size_t) r * (size_t) H + hout] = facc;
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }
}

void latent_u_matvec(const uint8_t* u_q8, const uint8_t* hq, const int32_t* ids, const int32_t* labels_dev,
                     const int32_t* dst, const int32_t* count, int64_t E, int64_t n_entries, int ELL, int H,
                     int G, float* parts, void* stream, int fmt) {
    if (n_entries <= 0 || G <= 0) return;
    const int warps_per_block = 8;
    dpct::dim3 grid((unsigned)G,
                    (unsigned)((H + warps_per_block - 1) / warps_per_block));
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(
            strata::q_of(stream)->get_device(),
            {sycl::aspect::fp16});

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class latent_u_matvec_k_8d6618>>(
                sycl::nd_range<3>(grid *
                                      sycl::range(1, 1, warps_per_block * 32),
                                  sycl::range(1, 1, warps_per_block * 32)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        latent_u_matvec_k(u_q8, hq, ids, labels_dev, dst, count,
                                          (int)E, (int)n_entries, ELL, H, parts,
                                          fmt);
                    });
    }
}

}  // namespace strata::kernels