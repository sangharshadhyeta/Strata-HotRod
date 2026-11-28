// src/kernels/cuda/dequant_bf16.cu - see include/strata/kernels/dequant_bf16.hpp.
//
// Arithmetic transcribed from ggml/src/ggml-quants.c at the pinned llama.cpp (MIT License, Copyright (c) 2023-2026
// The ggml authors): dequantize_row_q2_0/q4_0/q5_0/q8_0/q3_K/q4_K/q5_K/q6_K/iq4_nl/iq4_xs.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/core/tc_sm75.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

__dpct_inline__ float h2f(const uint8_t *p) {
    return sycl::vec<sycl::half, 1>(sycl::bit_cast<sycl::half, unsigned short>(
                                        (uint16_t)(p[0] | (p[1] << 8))))
        .convert<float, sycl::rounding_mode::automatic>()[0];
}
__dpct_inline__ uint16_t f2bf(float f) {
    uint32_t u = sycl::bit_cast<unsigned int>(f);
    // a NaN (a NaN scale in the block) stays a quiet NaN, as in ggml_compute_fp32_to_bf16 and `bf16_from_f32`:
    // the rounding add below would carry it into -0 or inf
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 64u);
    u += 0x7fffu + ((u >> 16) & 1u);          // round to nearest even
    return (uint16_t) (u >> 16);
}
__dpct_inline__ void put(uint16_t *o, int i, float v) { o[i] = f2bf(v); }
struct H16 { uint16_t v; };
__dpct_inline__ void put(H16 *o, int i, float v) {
    o[i].v = sycl::bit_cast<unsigned short, sycl::half>(
        sycl::vec<float, 1>(v)
            .convert<sycl::half, sycl::rounding_mode::rte>()[0]);
}
__dpct_inline__ void put(float *o, int i, float v) { o[i] = v; }

const int8_t kv_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                             1,    13,   25,  38,  53,  69,  89,  113};

__dpct_inline__ void scale_min_k4(int j, const uint8_t *q, int &d, int &m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

// One 32-element group `g` (row-major over the whole slice); `out` points at that group's 32 outputs.
template <int TYPE, typename T>
/*
DPCT1110: The total declared local variable size in device function group32
exceeds 128 bytes and may cause high register pressure. Consult with your
hardware vendor to find the total register size available and adjust the code,
or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void group32(const uint8_t *row_blocks, int gi_in_row, T *out) {
    if constexpr (TYPE == 42) {                                   // Q2_0: 64 per block of 18 B
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 2) * 18;
        const float d = h2f(b);
        const int e0 = (gi_in_row % 2) * 32;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            const int e = e0 + j;
            const int q = (b[2 + e / 4] >> ((e % 4) * 2)) & 3;
            put(out, j, (float) (q - 1) * d);
        }
    } else if constexpr (TYPE == 2) {                              // Q4_0
        const uint8_t* b = row_blocks + (size_t) gi_in_row * 18;
        const float d = h2f(b);
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            put(out, j, (float) ((b[2 + j] & 0x0F) - 8) * d);
            put(out, j + 16, (float) ((b[2 + j] >> 4) - 8) * d);
        }
    } else if constexpr (TYPE == 6) {                              // Q5_0
        const uint8_t* b = row_blocks + (size_t) gi_in_row * 22;
        const float d = h2f(b);
        const uint32_t qh = (uint32_t) b[2] | ((uint32_t) b[3] << 8) | ((uint32_t) b[4] << 16) | ((uint32_t) b[5] << 24);
        for (int j = 0; j < 16; ++j) {
            const int xh0 = ((qh >> j) << 4) & 0x10;
            const int xh1 = (qh >> (j + 12)) & 0x10;
            put(out, j, (float) (((b[6 + j] & 0x0F) | xh0) - 16) * d);
            put(out, j + 16, (float) (((b[6 + j] >> 4) | xh1) - 16) * d);
        }
    } else if constexpr (TYPE == 7) {                              // Q5_1
        const uint8_t* b = row_blocks + (size_t) gi_in_row * 24;
        const float d = h2f(b), m = h2f(b + 2);
        const uint32_t qh = (uint32_t) b[4] | ((uint32_t) b[5] << 8) |
                            ((uint32_t) b[6] << 16) | ((uint32_t) b[7] << 24);
        for (int j = 0; j < 16; ++j) {
            const int xh0 = ((qh >> j) & 1) << 4;
            const int xh1 = ((qh >> (j + 16)) & 1) << 4;
            put(out, j, (float) ((b[8 + j] & 15) | xh0) * d + m);
            put(out, j + 16, (float) ((b[8 + j] >> 4) | xh1) * d + m);
        }
    } else if constexpr (TYPE == 8) {                              // Q8_0
        const uint8_t* b = row_blocks + (size_t) gi_in_row * 34;
        const float d = h2f(b);
#pragma unroll
        for (int j = 0; j < 32; ++j) put(out, j, (float)(int8_t)b[2 + j] * d);
    } else if constexpr (TYPE == 20) {                             // IQ4_NL
        const uint8_t* b = row_blocks + (size_t) gi_in_row * 18;
        const float d = h2f(b);
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            put(out, j, d * (float) kv_iq4nl[b[2 + j] & 0xf]);
            put(out, j + 16, d * (float) kv_iq4nl[b[2 + j] >> 4]);
        }
    } else if constexpr (TYPE == 11) {                             // Q3_K: hmask[32] qs[64] scales[12] d
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 110;
        const int gi = gi_in_row % 8, n = gi / 4, jj = gi % 4;
        const uint8_t* hm = b;
        const uint8_t* q = b + 32 + n * 32;
        const uint8_t* sc = b + 96;
        const float d_all = h2f(b + 108);
        uint32_t aux[4];
        memcpy(aux, sc, 12);
        const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu, tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t* scales = reinterpret_cast<const int8_t*>(aux);
        const int shift = 2 * jj;
        const uint8_t m = (uint8_t) (1u << (n * 4 + jj));
        for (int t = 0; t < 32; ++t) {
            const int is = n * 8 + jj * 2 + (t >= 16 ? 1 : 0);
            const float dl = d_all * (float) (scales[is] - 32);
            put(out, t, dl * (float) ((int) ((q[t] >> shift) & 3) - ((hm[t] & m) ? 0 : 4)));
        }
    } else if constexpr (TYPE == 12) {                             // Q4_K: d dmin scales[12] qs[128]
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 144;
        const int gi = gi_in_row % 8, j64 = gi / 2, hi = gi % 2;
        const float d = h2f(b), dmin = h2f(b + 2);
        int sc, m;
        scale_min_k4(gi, b + 4, sc, m);
        const float d1 = d * (float) sc, m1 = dmin * (float) m;
        const uint8_t* q = b + 16 + 32 * j64;
#pragma unroll
        for (int l = 0; l < 32; ++l)
            put(out, l, d1 * (float)(hi ? (q[l] >> 4) : (q[l] & 0xF)) - m1);
    } else if constexpr (TYPE == 13) {                             // Q5_K: d dmin scales[12] qh[32] qs[128]
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 176;
        const int gi = gi_in_row % 8, j64 = gi / 2, hi = gi % 2;
        const float d = h2f(b), dmin = h2f(b + 2);
        int sc, m;
        scale_min_k4(gi, b + 4, sc, m);
        const float d1 = d * (float) sc, m1 = dmin * (float) m;
        const uint8_t* qh = b + 16;
        const uint8_t* ql = b + 48 + 32 * j64;
        const uint8_t u = (uint8_t) (1u << (2 * j64 + hi));
        for (int l = 0; l < 32; ++l) {
            const int nib = hi ? (ql[l] >> 4) : (ql[l] & 0xF);
            put(out, l, d1 * (float) (nib + ((qh[l] & u) ? 16 : 0)) - m1);
        }
    } else if constexpr (TYPE == 14) {                             // Q6_K: ql[128] qh[64] scales[16] d
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 210;
        const int gi = gi_in_row % 8, n = gi / 4, qu = gi % 4;
        const uint8_t* ql = b + 64 * n;
        const uint8_t* qh = b + 128 + 32 * n;
        const int8_t* sc = reinterpret_cast<const int8_t*>(b + 192) + 8 * n;
        const float d = h2f(b + 208);
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            int q;
            if (qu == 0) q = (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4);
            else if (qu == 1) q = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
            else if (qu == 2) q = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
            else q = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
            put(out, l, d * (float) sc[is + 2 * qu] * (float) (q - 32));
        }
    } else if constexpr (TYPE == 23) {                             // IQ4_XS: d scales_h scales_l[4] qs[128]
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 136;
        const int ib = gi_in_row % 8;
        const float d = h2f(b);
        const uint16_t scales_h = (uint16_t) (b[2] | (b[3] << 8));
        const int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xf) | (((scales_h >> (2 * ib)) & 3) << 4);
        const float dl = d * (float) (ls - 32);
        const uint8_t* qs = b + 8 + 16 * ib;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            put(out, j, dl * (float) kv_iq4nl[qs[j] & 0xf]);
            put(out, j + 16, dl * (float) kv_iq4nl[qs[j] >> 4]);
        }
    }
}

/// `VEC` (fork, Daily be4cf1c, sm_75 only): the group is dequantised into a register buffer and flushed with 16-byte
/// stores.  The plain form writes each element where it is computed, which for a warp means 32 store instructions
/// whose lanes are 64 bytes apart - 32 two-byte transactions per instruction.  **The values are identical** (the
/// same arithmetic, the same order), only the stores are wider; the Turing build measured this kernel at 18% of the
/// row's bandwidth.  (0.1.42's Q8_0 kernel below already writes coalesced and is left as it is.)
template <int TYPE, typename T, bool VEC>
/*
DPCT1110: The total declared local variable size in device function
dequant_kernel exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void dequant_kernel(const uint8_t *__restrict__ blocks,
                                    int64_t row_bytes, int64_t row0,
                                    int64_t rows, int64_t groups_per_row,
                                    int64_t ld, T *__restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t g =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (g >= rows * groups_per_row) return;
    const int64_t r = g / groups_per_row, gi = g % groups_per_row;
    T* dst = out + r * ld + gi * 32;
    if constexpr (!VEC) {
        group32<TYPE>(blocks + (row0 + r) * row_bytes, (int) gi, dst);
    } else {
        __dpct_align__(16)
            T buf[32]; // the uint4 flush below reads it as 16-byte vectors
        group32<TYPE>(blocks + (row0 + r) * row_bytes, (int) gi, buf);
        // 32 elements: 64 B for the 16-bit forms (4 x uint4), 128 B for the float one (8 x uint4)
        constexpr int CHUNKS = (sizeof(T) == 2 ? 4 : 8);
#pragma unroll
        for (int c = 0; c < CHUNKS; ++c) reinterpret_cast<sycl::uint4 *>(
            dst)[c] = reinterpret_cast<const sycl::uint4 *>(buf)[c];
    }
}

<<<<<<< sycl
||||||| base
// ---- the wide kernels: a thread per 8 values (Pascal, sm_61) --------------------------------------------------------
// dequant_kernel's thread reads its 32-value group a byte at a time and writes 32 two-byte values one by one, 64 bytes
// from its neighbour's.  On a Tesla P40 (sm_61) that is 7.2 ms for a 10240 x 2560 Q6_K weight (10 GB/s), and the IQ3_S
// pack's dense weights cost every prompt chunk ~0.5 s of it, whatever the chunk's length (a fifth of a 560-token prompt).
// Here a thread takes 8 consecutive values: its code bytes as 16-bit loads (the GGUF blocks are 2-byte aligned) and
// one 16-byte store (two for FP32), so a warp writes 512 consecutive bytes.  The arithmetic per value is group32's -
// the same expression on the same operands - so the output is bitwise the same (dequant_wide_parity).  On the P40:
// Q6_K 7.2 -> 0.33 ms (227 GB/s), Q4_K 1.67 -> 0.32 ms, Q5_K 1.24 -> 0.33 ms, IQ4_XS 1.21 -> 0.34 ms for that shape.
// Q4_K, Q5_K, Q6_K and IQ4_XS.  On by default on compute capability 6.1, where it was measured;
// STRATA_DEQUANT_WIDE=0 / =1 turns it off / on for every card.
__dpct_inline__ uint64_t
ld8(const uint8_t *p) { // 8 bytes at a 2-byte aligned address
    const uint16_t* q = reinterpret_cast<const uint16_t*>(p);
    return (uint64_t) q[0] | ((uint64_t) q[1] << 16) | ((uint64_t) q[2] << 32) | ((uint64_t) q[3] << 48);
}
// values [8 piece, 8 piece + 8) of the 32-element group `gi_in_row`, as group32 computes them
template <int TYPE, typename T>
/*
DPCT1110: The total declared local variable size in device function wide8
exceeds 128 bytes and may cause high register pressure. Consult with your
hardware vendor to find the total register size available and adjust the code,
or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void wide8(const uint8_t *row_blocks, int gi_in_row, int piece,
                           T *v) {
    const int l0 = 8 * piece;
    if constexpr (TYPE == 12) {                                    // Q4_K: d dmin scales[12] qs[128]
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 144;
        const int gi = gi_in_row % 8, j64 = gi / 2, hi = gi % 2;
        const float d = h2f(b), dmin = h2f(b + 2);
        int sc, m;
        scale_min_k4(gi, b + 4, sc, m);
        const float d1 = d * (float) sc, m1 = dmin * (float) m;
        const uint64_t q = ld8(b + 16 + 32 * j64 + l0);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int lb = (int) (q >> (8 * k)) & 0xFF;
            put(v, k, d1 * (float) (hi ? (lb >> 4) : (lb & 0xF)) - m1);
        }
    } else if constexpr (TYPE == 13) {                             // Q5_K: d dmin scales[12] qh[32] qs[128]
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 176;
        const int gi = gi_in_row % 8, j64 = gi / 2, hi = gi % 2;
        const float d = h2f(b), dmin = h2f(b + 2);
        int sc, m;
        scale_min_k4(gi, b + 4, sc, m);
        const float d1 = d * (float) sc, m1 = dmin * (float) m;
        const uint64_t qh = ld8(b + 16 + l0), ql = ld8(b + 48 + 32 * j64 + l0);
        const int ub = 2 * j64 + hi;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int lb = (int) (ql >> (8 * k)) & 0xFF, hb = (int) (qh >> (8 * k)) & 0xFF;
            const int nib = hi ? (lb >> 4) : (lb & 0xF);
            put(v, k, d1 * (float) (nib + (((hb >> ub) & 1) ? 16 : 0)) - m1);
        }
    } else if constexpr (TYPE == 14) {                             // Q6_K: ql[128] qh[64] scales[16] d
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 210;
        const int gi = gi_in_row % 8, n = gi / 4, qu = gi % 4;
        const uint64_t ql = ld8(b + 64 * n + ((qu & 1) ? 32 : 0) + l0);
        const uint64_t qh = ld8(b + 128 + 32 * n + l0);
        const int8_t* sc = reinterpret_cast<const int8_t*>(b + 192) + 8 * n;
        const float d = h2f(b + 208);
        const float ds = d * (float) sc[l0 / 16 + 2 * qu];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int lb = (int) (ql >> (8 * k)) & 0xFF, hb = (int) (qh >> (8 * k)) & 0xFF;
            const int q = ((qu >= 2) ? (lb >> 4) : (lb & 0xF)) | (((hb >> (2 * qu)) & 3) << 4);
            put(v, k, ds * (float) (q - 32));
        }
    } else if constexpr (TYPE == 23) {                             // IQ4_XS: d scales_h scales_l[4] qs[128]
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 136;
        const int ib = gi_in_row % 8;
        const float d = h2f(b);
        const uint16_t scales_h = (uint16_t) (b[2] | (b[3] << 8));
        const int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xf) | (((scales_h >> (2 * ib)) & 3) << 4);
        const float dl = d * (float) (ls - 32);
        const uint64_t qs = ld8(b + 8 + 16 * ib + 8 * (piece & 1));   // pieces 0, 1: the low nibbles; 2, 3: the high ones
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int lb = (int) (qs >> (8 * k)) & 0xFF;
            put(v, k, dl * (float) kv_iq4nl[piece >= 2 ? (lb >> 4) : (lb & 0xf)]);
        }
    }
}
template <int TYPE, typename T>
__dpct_inline__ void dequant_wide_kernel(const uint8_t *__restrict__ blocks,
                                         int64_t row_bytes, int64_t row0,
                                         uint32_t rows, uint32_t groups_per_row,
                                         int64_t ld, T *__restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const uint32_t i =
        item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2); // launch(): 4 * rows * groups_per_row < 2^31
    const uint32_t g = i >> 2, r = g / groups_per_row, gi = g - r * groups_per_row;
    if (r >= rows) return;
    const int piece = (int) (i & 3);
    __dpct_align__(16) T v[8];
    wide8<TYPE>(blocks + (row0 + (int64_t) r) * row_bytes, (int) gi, piece, v);
    sycl::uint4 *o =
        (sycl::uint4 *)(out + (int64_t)r * ld + (int64_t)gi * 32 + 8 * piece);
#pragma unroll
    for (int k = 0; k < (int)(8 * sizeof(T) / 16); ++k)
        o[k] = ((const sycl::uint4 *)v)[k];
}
bool wide_type(int type) { return type == 12 || type == 13 || type == 14 || type == 23; }
int g_dequant_wide = -1;   // dequant_set_wide (the parity test); -1 = STRATA_DEQUANT_WIDE, else the card's default
bool dequant_wide_on() {
    static const int env = [] { const char* v = std::getenv("STRATA_DEQUANT_WIDE"); return v != nullptr && v[0] != '\0' ? (std::atoi(v) != 0 ? 1 : 0) : -1; }();
    if (g_dequant_wide >= 0) return g_dequant_wide != 0;
    if (env >= 0) return env != 0;
    static int per_dev[64];   // 0 unknown, 1 no, 2 yes: sm_61, measured on a Tesla P40
    int dev = 0;
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    if (DPCT_CHECK_ERROR(dev = dpct::get_current_device_id()) != 0 || dev < 0 ||
        dev >= 64) {
        (void)0; return false;
    }
    if (!per_dev[dev]) {
        int major = 0, minor = 0;
        if (DPCT_CHECK_ERROR(
                major = dpct::get_device(dev).get_major_version()) != 0 ||
            /*
            DPCT1010: SYCL uses exceptions to report errors and does not use
            the error codes. The cudaGetLastError function call was replaced
            with 0. You need to rewrite this code.
            */
            DPCT_CHECK_ERROR(
                minor = dpct::get_device(dev).get_minor_version()) != 0) {
            (void)0; return false;
        }
        per_dev[dev] = major == 6 && minor == 1 ? 2 : 1;
    }
    return per_dev[dev] == 2;
}

// Q8_0 with four threads per block, 8 values each: the warp reads 8 consecutive blocks and writes 512 consecutive
// bytes (a thread per block wrote 2-byte values 64 bytes apart: ~110 GB/s, a quarter of a prompt projection's time)
template <typename T>
__dpct_inline__ void dequant_q8_0_kernel(const uint8_t *__restrict__ blocks,
                                         int64_t row_bytes, int64_t row0,
                                         int64_t rows, int64_t groups_per_row,
                                         T *__restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= rows * groups_per_row * 4) return;
    const int64_t g = i >> 2, r = g / groups_per_row, gi = g % groups_per_row;
    const int piece = (int) (i & 3);
    const uint8_t* b = blocks + (row0 + r) * row_bytes + gi * 34;
    const float d = h2f(b);
    const uint16_t* q = (const uint16_t*) (b + 2 + 8 * piece);   // blocks are 2-byte aligned
    __dpct_align__(16) T v[8];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint16_t w = q[k];
        put(v, 2 * k, (float) (int8_t) (w & 0xFF) * d);
        put(v, 2 * k + 1, (float) (int8_t) (w >> 8) * d);
    }
    sycl::uint4 *o = (sycl::uint4 *)(out + g * 32 + 8 * piece);
#pragma unroll
    for (int k = 0; k < (int)(8 * sizeof(T) / 16); ++k)
        o[k] = ((const sycl::uint4 *)v)[k];
}

=======
// ---- the wide kernels: a thread per 8 values (Pascal, sm_61) --------------------------------------------------------
// dequant_kernel's thread reads its 32-value group a byte at a time and writes 32 two-byte values one by one, 64 bytes
// from its neighbour's.  On a Tesla P40 (sm_61) that is 7.2 ms for a 10240 x 2560 Q6_K weight (10 GB/s), and the IQ3_S
// pack's dense weights cost every prompt chunk ~0.5 s of it, whatever the chunk's length (a fifth of a 560-token prompt).
// Here a thread takes 8 consecutive values: its code bytes as 16-bit loads (the GGUF blocks are 2-byte aligned) and
// one 16-byte store (two for FP32), so a warp writes 512 consecutive bytes.  The arithmetic per value is group32's -
// the same expression on the same operands - so the output is bitwise the same (dequant_wide_parity).  On the P40:
// Q6_K 7.2 -> 0.33 ms (227 GB/s), Q4_K 1.67 -> 0.32 ms, Q5_K 1.24 -> 0.33 ms, IQ4_XS 1.21 -> 0.34 ms for that shape.
// Q4_K, Q5_K, Q6_K and IQ4_XS.  On by default on compute capability 6.1, where it was measured;
// STRATA_DEQUANT_WIDE=0 / =1 turns it off / on for every card.
__dpct_inline__ uint64_t
ld8(const uint8_t *p) { // 8 bytes at a 2-byte aligned address
    const uint16_t* q = reinterpret_cast<const uint16_t*>(p);
    return (uint64_t) q[0] | ((uint64_t) q[1] << 16) | ((uint64_t) q[2] << 32) | ((uint64_t) q[3] << 48);
}
// values [8 piece, 8 piece + 8) of the 32-element group `gi_in_row`, as group32 computes them
template <int TYPE, typename T>
/*
DPCT1110: The total declared local variable size in device function wide8
exceeds 128 bytes and may cause high register pressure. Consult with your
hardware vendor to find the total register size available and adjust the code,
or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void wide8(const uint8_t *row_blocks, int gi_in_row, int piece,
                           T *v) {
    const int l0 = 8 * piece;
    if constexpr (TYPE == 12) {                                    // Q4_K: d dmin scales[12] qs[128]
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 144;
        const int gi = gi_in_row % 8, j64 = gi / 2, hi = gi % 2;
        const float d = h2f(b), dmin = h2f(b + 2);
        int sc, m;
        scale_min_k4(gi, b + 4, sc, m);
        const float d1 = d * (float) sc, m1 = dmin * (float) m;
        const uint64_t q = ld8(b + 16 + 32 * j64 + l0);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int lb = (int) (q >> (8 * k)) & 0xFF;
            put(v, k, d1 * (float) (hi ? (lb >> 4) : (lb & 0xF)) - m1);
        }
    } else if constexpr (TYPE == 13) {                             // Q5_K: d dmin scales[12] qh[32] qs[128]
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 176;
        const int gi = gi_in_row % 8, j64 = gi / 2, hi = gi % 2;
        const float d = h2f(b), dmin = h2f(b + 2);
        int sc, m;
        scale_min_k4(gi, b + 4, sc, m);
        const float d1 = d * (float) sc, m1 = dmin * (float) m;
        const uint64_t qh = ld8(b + 16 + l0), ql = ld8(b + 48 + 32 * j64 + l0);
        const int ub = 2 * j64 + hi;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int lb = (int) (ql >> (8 * k)) & 0xFF, hb = (int) (qh >> (8 * k)) & 0xFF;
            const int nib = hi ? (lb >> 4) : (lb & 0xF);
            put(v, k, d1 * (float) (nib + (((hb >> ub) & 1) ? 16 : 0)) - m1);
        }
    } else if constexpr (TYPE == 14) {                             // Q6_K: ql[128] qh[64] scales[16] d
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 210;
        const int gi = gi_in_row % 8, n = gi / 4, qu = gi % 4;
        const uint64_t ql = ld8(b + 64 * n + ((qu & 1) ? 32 : 0) + l0);
        const uint64_t qh = ld8(b + 128 + 32 * n + l0);
        const int8_t* sc = reinterpret_cast<const int8_t*>(b + 192) + 8 * n;
        const float d = h2f(b + 208);
        const float ds = d * (float) sc[l0 / 16 + 2 * qu];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int lb = (int) (ql >> (8 * k)) & 0xFF, hb = (int) (qh >> (8 * k)) & 0xFF;
            const int q = ((qu >= 2) ? (lb >> 4) : (lb & 0xF)) | (((hb >> (2 * qu)) & 3) << 4);
            put(v, k, ds * (float) (q - 32));
        }
    } else if constexpr (TYPE == 23) {                             // IQ4_XS: d scales_h scales_l[4] qs[128]
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 136;
        const int ib = gi_in_row % 8;
        const float d = h2f(b);
        const uint16_t scales_h = (uint16_t) (b[2] | (b[3] << 8));
        const int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xf) | (((scales_h >> (2 * ib)) & 3) << 4);
        const float dl = d * (float) (ls - 32);
        const uint64_t qs = ld8(b + 8 + 16 * ib + 8 * (piece & 1));   // pieces 0, 1: the low nibbles; 2, 3: the high ones
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int lb = (int) (qs >> (8 * k)) & 0xFF;
            put(v, k, dl * (float) kv_iq4nl[piece >= 2 ? (lb >> 4) : (lb & 0xf)]);
        }
    }
}
template <int TYPE, typename T>
__dpct_inline__ void dequant_wide_kernel(const uint8_t *__restrict__ blocks,
                                         int64_t row_bytes, int64_t row0,
                                         uint32_t rows, uint32_t groups_per_row,
                                         int64_t ld, T *__restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const uint32_t i =
        item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2); // launch(): 4 * rows * groups_per_row < 2^31
    const uint32_t g = i >> 2, r = g / groups_per_row, gi = g - r * groups_per_row;
    if (r >= rows) return;
    const int piece = (int) (i & 3);
    __dpct_align__(16) T v[8];
    wide8<TYPE>(blocks + (row0 + (int64_t) r) * row_bytes, (int) gi, piece, v);
    sycl::uint4 *o =
        (sycl::uint4 *)(out + (int64_t)r * ld + (int64_t)gi * 32 + 8 * piece);
#pragma unroll
    for (int k = 0; k < (int)(8 * sizeof(T) / 16); ++k)
        o[k] = ((const sycl::uint4 *)v)[k];
}
bool wide_type(int type) { return type == 12 || type == 13 || type == 14 || type == 23; }
int g_dequant_wide = -1;   // dequant_set_wide (the parity test); -1 = STRATA_DEQUANT_WIDE, else the card's default
bool dequant_wide_on() {
    static const int env = [] { const char* v = std::getenv("STRATA_DEQUANT_WIDE"); return v != nullptr && v[0] != '\0' ? (std::atoi(v) != 0 ? 1 : 0) : -1; }();
    if (g_dequant_wide >= 0) return g_dequant_wide != 0;
    if (env >= 0) return env != 0;
    static int per_dev[64];   // 0 unknown, 1 no, 2 yes: sm_61 (measured on a Tesla P40) and sm_86 (RTX 3060 / A4000, 0.1.45: short-prompt TTFT -12..-17%, bitwise)
    int dev = 0;
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    if (DPCT_CHECK_ERROR(dev = dpct::get_current_device_id()) != 0 || dev < 0 ||
        dev >= 64) {
        (void)0; return false;
    }
    if (!per_dev[dev]) {
        int major = 0, minor = 0;
        if (DPCT_CHECK_ERROR(
                major = dpct::get_device(dev).get_major_version()) != 0 ||
            /*
            DPCT1010: SYCL uses exceptions to report errors and does not use
            the error codes. The cudaGetLastError function call was replaced
            with 0. You need to rewrite this code.
            */
            DPCT_CHECK_ERROR(
                minor = dpct::get_device(dev).get_minor_version()) != 0) {
            (void)0; return false;
        }
        per_dev[dev] = (major == 6 && minor == 1) || (major == 8 && minor == 6) ? 2 : 1;
    }
    return per_dev[dev] == 2;
}

// Q8_0 with four threads per block, 8 values each: the warp reads 8 consecutive blocks and writes 512 consecutive
// bytes (a thread per block wrote 2-byte values 64 bytes apart: ~110 GB/s, a quarter of a prompt projection's time)
template <typename T>
__dpct_inline__ void dequant_q8_0_kernel(const uint8_t *__restrict__ blocks,
                                         int64_t row_bytes, int64_t row0,
                                         int64_t rows, int64_t groups_per_row,
                                         T *__restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= rows * groups_per_row * 4) return;
    const int64_t g = i >> 2, r = g / groups_per_row, gi = g % groups_per_row;
    const int piece = (int) (i & 3);
    const uint8_t* b = blocks + (row0 + r) * row_bytes + gi * 34;
    const float d = h2f(b);
    const uint16_t* q = (const uint16_t*) (b + 2 + 8 * piece);   // blocks are 2-byte aligned
    __dpct_align__(16) T v[8];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint16_t w = q[k];
        put(v, 2 * k, (float) (int8_t) (w & 0xFF) * d);
        put(v, 2 * k + 1, (float) (int8_t) (w >> 8) * d);
    }
    sycl::uint4 *o = (sycl::uint4 *)(out + g * 32 + 8 * piece);
#pragma unroll
    for (int k = 0; k < (int)(8 * sizeof(T) / 16); ++k)
        o[k] = ((const sycl::uint4 *)v)[k];
}

>>>>>>> ours-now
bool geometry(int type, int& block_elems, int& block_bytes) {
    switch (type) {
    case 2: block_elems = 32; block_bytes = 18; return true;
    case 6: block_elems = 32; block_bytes = 22; return true;
    case 7: block_elems = 32; block_bytes = 24; return true;
    case 8: block_elems = 32; block_bytes = 34; return true;
    case 20: block_elems = 32; block_bytes = 18; return true;
    case 11: block_elems = 256; block_bytes = 110; return true;
    case 12: block_elems = 256; block_bytes = 144; return true;
    case 13: block_elems = 256; block_bytes = 176; return true;
    case 14: block_elems = 256; block_bytes = 210; return true;
    case 23: block_elems = 256; block_bytes = 136; return true;
    case 42: block_elems = 64; block_bytes = 18; return true;
    default: return false;
    }
}

template <typename T>
void launch(int type, const void *blocks, int64_t row0, int64_t rows,
            int64_t cols, T *out, void *stream, int64_t ld = 0) try {
    int be = 0, bb = 0;
    if (!geometry(type, be, bb) || cols % be != 0 || rows <= 0) {
        std::fprintf(stderr, "dequant: unsupported type %d or shape %lld x %lld\n", type, (long long) rows,
                     (long long) cols);
        std::exit(1);
    }
    const int64_t row_bytes = cols / be * bb, gpr = cols / 32, total = rows * gpr;
    if (ld <= 0) ld = cols;
    const unsigned grid = (unsigned) ((total + 255) / 256);
    const uint8_t* p = (const uint8_t*) blocks;
    dpct::queue_ptr st = strata::q_of(stream);
<<<<<<< sycl
||||||| base
    if (type == 8 && ld == cols && (reinterpret_cast<uintptr_t>(blocks) & 1) == 0 &&
        (reinterpret_cast<uintptr_t>(out) & 15) == 0) {   // contiguous rows only (the kernel writes out + g * 32)
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            st->parallel_for<
                dpct_kernel_name<class dequant_q8_0_kernel_c58f6a, T>>(
                sycl::nd_range<3>(
                    sycl::range(1, 1, (unsigned)((4 * total + 255) / 256)) *
                        sycl::range(1, 1, 256),
                    sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    dequant_q8_0_kernel<T>(p, row_bytes, row0, rows, gpr, out);
                });
        }
        /*
        DPCT1010: SYCL uses exceptions to report errors and does not use the
        error codes. The cudaGetLastError function call was replaced with 0. You
        need to rewrite this code.
        */
        const dpct::err0 e = 0;
        /*
        DPCT1009: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */

        return;
    }
    // the wide kernels: even block addresses, 16-byte stores at every row (contiguous or a stride of whole stores)
    if (wide_type(type) && (reinterpret_cast<uintptr_t>(blocks) & 1) == 0 && (reinterpret_cast<uintptr_t>(out) & 15) == 0 &&
        ((size_t) ld * sizeof(T)) % 16 == 0 && 4 * total < (1ll << 31) && dequant_wide_on()) {
        const unsigned wgrid = (unsigned) ((4 * total + 255) / 256);
#define STRATA_DQW(TY)                                                         \
    {                                                                          \
        auto exp_props = sycl::ext::oneapi::experimental::properties{          \
            sycl::ext::oneapi::experimental::use_root_sync};                   \
                                                                               \
        st->submit([&](sycl::handler &cgh) {                                   \
            auto p_ct0 = p;                                                    \
            auto row_bytes_ct1 = row_bytes;                                    \
            auto row0_ct2 = row0;                                              \
            auto rows_ct3 = (uint32_t)rows;                                    \
            auto gpr_ct4 = (uint32_t)gpr;                                      \
            auto ld_ct5 = ld;                                                  \
            auto out_ct6 = out;                                                \
                                                                               \
            cgh.parallel_for<dpct_kernel_name<                                 \
                class dequant_wide_kernel_dea121, dpct_kernel_scalar<TY>, T>>( \
                sycl::nd_range<3>(sycl::range(1, 1, wgrid) *                   \
                                      sycl::range(1, 1, 256),                  \
                                  sycl::range(1, 1, 256)),                     \
                exp_props, [=](sycl::nd_item<3> item_ct1) {                    \
                    dequant_wide_kernel<TY, T>(p_ct0, row_bytes_ct1, row0_ct2, \
                                               rows_ct3, gpr_ct4, ld_ct5,      \
                                               out_ct6);                       \
                });                                                            \
        });                                                                    \
    } break
        switch (type) {
        case 12: STRATA_DQW(12);
        case 13: STRATA_DQW(13);
        case 14: STRATA_DQW(14);
        case 23: STRATA_DQW(23);
        }
#undef STRATA_DQW
        /*
        DPCT1010: SYCL uses exceptions to report errors and does not use the
        error codes. The cudaGetLastError function call was replaced with 0. You
        need to rewrite this code.
        */
        const dpct::err0 e = 0;
        /*
        DPCT1009: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */

        return;
    }
=======
    if (type == 8 && ld == cols && (reinterpret_cast<uintptr_t>(blocks) & 1) == 0 &&
        (reinterpret_cast<uintptr_t>(out) & 15) == 0) {   // contiguous rows only (the kernel writes out + g * 32)
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            st->parallel_for<
                dpct_kernel_name<class dequant_q8_0_kernel_fff089, T>>(
                sycl::nd_range<3>(
                    sycl::range(1, 1, (unsigned)((4 * total + 255) / 256)) *
                        sycl::range(1, 1, 256),
                    sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    dequant_q8_0_kernel<T>(p, row_bytes, row0, rows, gpr, out);
                });
        }
        /*
        DPCT1010: SYCL uses exceptions to report errors and does not use the
        error codes. The cudaGetLastError function call was replaced with 0. You
        need to rewrite this code.
        */
        const dpct::err0 e = 0;
        /*
        DPCT1009: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */

        return;
    }
    // the wide kernels: even block addresses, 16-byte stores at every row (contiguous or a stride of whole stores)
    if (wide_type(type) && (reinterpret_cast<uintptr_t>(blocks) & 1) == 0 && (reinterpret_cast<uintptr_t>(out) & 15) == 0 &&
        ((size_t) ld * sizeof(T)) % 16 == 0 && 4 * total < (1ll << 31) && dequant_wide_on()) {
        const unsigned wgrid = (unsigned) ((4 * total + 255) / 256);
#define STRATA_DQW(TY)                                                         \
    {                                                                          \
        auto exp_props = sycl::ext::oneapi::experimental::properties{          \
            sycl::ext::oneapi::experimental::use_root_sync};                   \
                                                                               \
        st->submit([&](sycl::handler &cgh) {                                   \
            auto p_ct0 = p;                                                    \
            auto row_bytes_ct1 = row_bytes;                                    \
            auto row0_ct2 = row0;                                              \
            auto rows_ct3 = (uint32_t)rows;                                    \
            auto gpr_ct4 = (uint32_t)gpr;                                      \
            auto ld_ct5 = ld;                                                  \
            auto out_ct6 = out;                                                \
                                                                               \
            cgh.parallel_for<dpct_kernel_name<                                 \
                class dequant_wide_kernel_60da01, dpct_kernel_scalar<TY>, T>>( \
                sycl::nd_range<3>(sycl::range(1, 1, wgrid) *                   \
                                      sycl::range(1, 1, 256),                  \
                                  sycl::range(1, 1, 256)),                     \
                exp_props, [=](sycl::nd_item<3> item_ct1) {                    \
                    dequant_wide_kernel<TY, T>(p_ct0, row_bytes_ct1, row0_ct2, \
                                               rows_ct3, gpr_ct4, ld_ct5,      \
                                               out_ct6);                       \
                });                                                            \
        });                                                                    \
    } break
        switch (type) {
        case 12: STRATA_DQW(12);
        case 13: STRATA_DQW(13);
        case 14: STRATA_DQW(14);
        case 23: STRATA_DQW(23);
        }
#undef STRATA_DQW
        /*
        DPCT1010: SYCL uses exceptions to report errors and does not use the
        error codes. The cudaGetLastError function call was replaced with 0. You
        need to rewrite this code.
        */
        const dpct::err0 e = 0;
        /*
        DPCT1009: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */

        return;
    }
    // sm_75 (and only cc 7.5) takes the vectorised-store form when every group's 32 outputs start on 16 bytes (an
    // aligned `out` and a row stride of whole 16-byte units); every other SM model keeps the plain kernel
    int dev = 0, cmaj = 0, cmin = 0;
    const bool vec =
        (reinterpret_cast<uintptr_t>(out) & 15) == 0 &&
        ((uint64_t)ld * sizeof(T)) % 16 == 0 &&
        DPCT_CHECK_ERROR(dev = dpct::get_current_device_id()) == 0 &&
        DPCT_CHECK_ERROR(cmaj = dpct::get_device(dev).get_major_version()) ==
            0 &&
        DPCT_CHECK_ERROR(cmin = dpct::get_device(dev).get_minor_version()) ==
            0 &&
        strata::core::tc_sm75_enabled(cmaj, cmin);
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    if (!vec) 0;
>>>>>>> ours-now
#define STRATA_DQ(TY)                                                          \
    if (vec) {                                                                 \
        auto exp_props = sycl::ext::oneapi::experimental::properties{          \
            sycl::ext::oneapi::experimental::use_root_sync};                   \
                                                                               \
        st->submit([&](sycl::handler &cgh) {                                   \
            auto p_ct0 = p;                                                    \
            auto row_bytes_ct1 = row_bytes;                                    \
            auto row0_ct2 = row0;                                              \
            auto rows_ct3 = rows;                                              \
            auto gpr_ct4 = gpr;                                                \
            auto ld_ct5 = ld;                                                  \
            auto out_ct6 = out;                                                \
                                                                               \
            cgh.parallel_for<dpct_kernel_name<class dequant_kernel_ec9311,     \
                                              dpct_kernel_scalar<TY>, T,       \
                                              dpct_kernel_scalar<true>>>(      \
                sycl::nd_range<3>(sycl::range(1, 1, grid) *                    \
                                      sycl::range(1, 1, 256),                  \
                                  sycl::range(1, 1, 256)),                     \
                exp_props, [=](sycl::nd_item<3> item_ct1) {                    \
                    dequant_kernel<TY, T, true>(p_ct0, row_bytes_ct1,          \
                                                row0_ct2, rows_ct3, gpr_ct4,   \
                                                ld_ct5, out_ct6);              \
                });                                                            \
        });                                                                    \
    } else {                                                                   \
        auto exp_props = sycl::ext::oneapi::experimental::properties{          \
            sycl::ext::oneapi::experimental::use_root_sync};                   \
                                                                               \
        st->submit([&](sycl::handler &cgh) {                                   \
            auto p_ct0 = p;                                                    \
            auto row_bytes_ct1 = row_bytes;                                    \
            auto row0_ct2 = row0;                                              \
            auto rows_ct3 = rows;                                              \
            auto gpr_ct4 = gpr;                                                \
            auto ld_ct5 = ld;                                                  \
            auto out_ct6 = out;                                                \
                                                                               \
            cgh.parallel_for<dpct_kernel_name<class dequant_kernel_ec9311,     \
                                              dpct_kernel_scalar<TY>, T,       \
                                              dpct_kernel_scalar<false>>>(     \
                sycl::nd_range<3>(sycl::range(1, 1, grid) *                    \
                                      sycl::range(1, 1, 256),                  \
                                  sycl::range(1, 1, 256)),                     \
                exp_props, [=](sycl::nd_item<3> item_ct1) {                    \
                    dequant_kernel<TY, T, false>(p_ct0, row_bytes_ct1,         \
                                                 row0_ct2, rows_ct3, gpr_ct4,  \
                                                 ld_ct5, out_ct6);             \
                });                                                            \
        });                                                                    \
    }                                                                          \
    break
    switch (type) {
    case 2: STRATA_DQ(2);
    case 6: STRATA_DQ(6);
    case 7: STRATA_DQ(7);
    case 8: STRATA_DQ(8);
    case 11: STRATA_DQ(11);
    case 12: STRATA_DQ(12);
    case 13: STRATA_DQ(13);
    case 14: STRATA_DQ(14);
    case 20: STRATA_DQ(20);
    case 23: STRATA_DQ(23);
    case 42: STRATA_DQ(42);
    }
#undef STRATA_DQ
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;
    /*
    DPCT1009: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

}  // namespace

bool dequant_bf16_supported(int ggml_type) noexcept {
    int a, b;
    return geometry(ggml_type, a, b);
}

namespace {
// plan v0.3 P6: the i-quant formats (llama.cpp's dequantizers, iq_kernels.cu)
bool iq_only(int t) { return t == 16 || t == 17 || t == 18 || t == 21 || t == 22 || t == 29; }
}  // namespace

void dequant_bf16(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, uint16_t* out,
                  void* stream) {
    launch<uint16_t>(ggml_type, blocks, row0, rows, cols, out, stream);
}

void dequant_f16(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, uint16_t* out,
                 void* stream) {
    if (iq_only(ggml_type)) {
        iq_dequant_f16(ggml_type, (const uint8_t*) blocks + (size_t) row0 * iq_row_bytes(ggml_type, cols), rows * cols,
                       out, stream);
        return;
    }
    launch<H16>(ggml_type, blocks, row0, rows, cols, reinterpret_cast<H16*>(out), stream);
}

bool dequant_f16_ld(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, int64_t ld,
                    uint16_t* out, void* stream) {
    int be = 0, bb = 0;
    if (iq_only(ggml_type) || !geometry(ggml_type, be, bb) || cols % be != 0 || rows <= 0 || ld < cols || ld % 8 != 0)
        return false;
    launch<H16>(ggml_type, blocks, row0, rows, cols, reinterpret_cast<H16*>(out), stream, ld);
    return true;
}

void dequant_f32(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, float* out, void* stream) {
    if (iq_only(ggml_type)) {
        iq_dequant_f32(ggml_type, (const uint8_t*) blocks + (size_t) row0 * iq_row_bytes(ggml_type, cols), rows * cols,
                       out, stream);
        return;
    }
    launch<float>(ggml_type, blocks, row0, rows, cols, out, stream);
}

}  // namespace strata::kernels
