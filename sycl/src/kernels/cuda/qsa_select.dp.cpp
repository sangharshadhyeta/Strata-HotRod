// src/kernels/cuda/qsa_select.cu - see include/strata/kernels/qsa_select.hpp.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/core/emulate.hpp"
#include "strata/core/sm75_series.hpp"
#include "strata/core/geometry.hpp"
#include <cstdlib>
#include <cstring>
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/gfx_arch.hpp"

#include <atomic>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::kernels {
namespace {

constexpr int IDX_DIM = 128, IDX_HEADS = 4, R = 4;
static_assert(IDX_DIM == strata::core::kCompiledGeometry.idx_key_dim && IDX_HEADS == strata::core::kCompiledGeometry.idx_q_heads && R == strata::core::kCompiledGeometry.idx_block,
              "qsa_select is compiled for core/geometry.hpp");
constexpr int SCORE_WARPS = 8;
constexpr int TOPK_T = 256;

__dpct_inline__ uint32_t order_key(float s) {
    const float v = s + 0.0f;
    if (!(v == v)) return 0u;
    const uint32_t b = sycl::bit_cast<unsigned int>(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__dpct_inline__ void block_scores_kernel(const float *__restrict__ pooled,
                                         const float *__restrict__ dead,
                                         const float *__restrict__ q_idx,
                                         const int32_t *__restrict__ steps,
                                         int64_t max_blocks,
                                         float *__restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t qi = item_ct1.get_group(1);
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid];
    const int64_t b = (int64_t)item_ct1.get_group(2) * SCORE_WARPS +
                      (item_ct1.get_local_id(2) >> 5);
    if (b > n_bid || b >= max_blocks) return;
    const int lane = item_ct1.get_local_id(2) & 31;
    const float* key = (b == n_bid) ? dead : pooled + b * IDX_DIM;
    const sycl::float4 k4 =
        *reinterpret_cast<const sycl::float4 *>(key + lane * 4);
    const float* q = q_idx + qi * IDX_HEADS * IDX_DIM + lane * 4;
    float score = 0.0f;
#pragma unroll
    for (int h = 0; h < IDX_HEADS; ++h) {
        const sycl::float4 q4 =
            *reinterpret_cast<const sycl::float4 *>(q + h * IDX_DIM);
        float d = k4.x() * q4.x() + k4.y() * q4.y() + k4.z() * q4.z() +
                  k4.w() * q4.w();
#pragma unroll
        /*
        DPCT1108: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        for (int o = 16; o > 0; o >>= 1) d +=
            dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                d, o);
        score += d > 0.0f ? d : 0.0f;
    }
    if (lane == 0) {
        if (b == n_bid && n_kv % R != 0) score += 1e9f;
        out[qi * max_blocks + b] = score;
    }
}

/*
DPCT1110: The total declared local variable size in device function
block_topk_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void block_topk_kernel(const float *__restrict__ scores,
                                       const int32_t *__restrict__ steps,
                                       int64_t max_blocks, int64_t cap,
                                       int32_t *__restrict__ ids) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &hist = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[256]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_a =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[TOPK_T]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_b =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[TOPK_T]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_digit = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_above = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int64_t qi = item_ct1.get_group(2);
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + qi * cap;
    const int t = item_ct1.get_local_id(2);
    if (n_kv <= width) {                               // everything is selected: the identity, ascending
#pragma unroll
        for (int64_t j = t; j < n_kv; j += TOPK_T) out[j] = (int32_t)j;
        return;
    }
    const float* sc = scores + qi * max_blocks;
    const int64_t nb = n_bid + 1;                      // blocks 0..n_bid, the last possibly empty
    const int64_t per = (nb + TOPK_T - 1) / TOPK_T;
    const int64_t b0 = (int64_t) t * per, b1 = (b0 + per < nb) ? b0 + per : nb;
    auto weight = [&](int64_t b) -> int { return b < n_bid ? R : (int) (n_kv - n_bid * R); };
    // ---- radix select: the largest key thr with (cells with key >= thr) >= width, 8 bits at a time
    uint32_t prefix = 0;
    int above = 0;                                     // cells strictly above the digits fixed so far
    for (int shift = 24; shift >= 0; shift -= 8) {
#pragma unroll
        for (int i = t; i < 256; i += TOPK_T) hist[i] = 0;
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
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        for (int64_t b = b0; b < b1; ++b) {
            const int w = weight(b);
            if (w == 0) continue;
            const uint32_t k = order_key(sc[b]);
            if ((k & hi_mask) == (prefix & hi_mask)) dpct::atomic_fetch_add<
                sycl::access::address_space::generic_space>(
                &hist[(k >> shift) & 255], w);
        }
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
        if (t == 0) {
            int cum = above, d = 255;
#pragma unroll
            for (; d > 0; --d) {
                if (cum + hist[d] >= width) break;
                cum += hist[d];
            }
            s_digit = d;
            s_above = cum;
        }
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
        prefix |= (uint32_t) s_digit << shift;
        above = s_above;
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
    const uint32_t thr = prefix;
    const int64_t eq_budget = width - above;          // cells equal to thr that fit, lowest index first
    // ---- per-thread counts of cells above and at the threshold, then their exclusive prefixes
    int gt = 0, eq = 0;
    for (int64_t b = b0; b < b1; ++b) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if (k > thr) gt += w;
        else if (k == thr) eq += w;
    }
    s_a[t] = gt;
    s_b[t] = eq;
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (t == 0) {
        int ag = 0, ae = 0;
        for (int i = 0; i < TOPK_T; ++i) {
            const int g = s_a[i], e = s_b[i];
            s_a[i] = ag; s_b[i] = ae;
            ag += g; ae += e;
        }
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const int64_t eq_before = s_b[t];
    int64_t my_eq = eq_budget - eq_before;
    if (my_eq < 0) my_eq = 0;
    if (my_eq > eq) my_eq = eq;
    const int sel = gt + (int) my_eq;
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    s_a[t] = sel;
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (t == 0) {
        int a = 0;
        for (int i = 0; i < TOPK_T; ++i) { const int c = s_a[i]; s_a[i] = a; a += c; }
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    int64_t wpos = s_a[t];
    int64_t eq_left = my_eq;
    for (int64_t b = b0; b < b1; ++b) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if (k > thr) {
#pragma unroll
            for (int c = 0; c < w; ++c) out[wpos++] = (int32_t)(b * R + c);
        } else if (k == thr) {
#pragma unroll
            for (int c = 0; c < w && eq_left > 0; ++c, --eq_left)
                out[wpos++] = (int32_t)(b * R + c);
        }
    }
}


// ---- QSA select on tensor cores (perf-review, after D-1): the block scores of many queries are one GEMM,
// rows (query, indexer head) x columns (blocks), K = 128, with relu per head summed. 3xTF32 (each operand split
// into a TF32 hi and lo part, hi*hi + hi*lo + lo*hi) keeps FP32-level accuracy; the summation order differs from
// the warp kernel, so a score can move in its last bits and a near-tie can select differently (not bitwise).
// Blocks < n_bid only; the tail block n_bid (the `dead` key, +1e9) is scored by the warp kernel's own code.
constexpr int TC_QT = 16;                 // queries per CTA (one m16 tile per indexer head)
constexpr int TC_NB = 32;                 // blocks per tile (4 warps x n8)
constexpr int TC_ITER = 4;                // tiles per CTA (the query tile is loaded once)
constexpr int TC_QS = IDX_HEADS * IDX_DIM + 4;   // query row stride in floats
constexpr int TC_KS = IDX_DIM + 4;               // key row stride

// TF32 conversion and MMA need sm_80: below it they compile to a trap and qsa_block_scores_tc refuses the device
#if defined(__HIPCC__)          // AMD: no mma.sync / cp.async; the host keeps the warp kernel (below)
#define STRATA_SEL_SM80 0
#elif 0   // SYCL: inline PTX (mma/ldmatrix/cp.async) - the XMX port is pending; see tools/fixups.py
#define STRATA_SEL_SM80 1
#else
#define STRATA_SEL_SM80 0
#endif
__dpct_inline__ uint32_t tf32_hi(float x) {
#if STRATA_SEL_SM80
    uint32_t r;
    /*
    DPCT1053: Migration of device assembly code is not supported.
    */
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
    return r;
#else
    return sycl::bit_cast<uint32_t>(x);
#endif
}
__dpct_inline__ void mma_tf32(float *c, const uint32_t *a, const uint32_t *b) {
#if !STRATA_SEL_SM80
    /* unreachable on SYCL: the launcher refuses this device */
#else
    /*
    DPCT1053: Migration of device assembly code is not supported.
    */
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
                   "r"(b[1]));
#endif
}

/*
DPCT1110: The total declared local variable size in device function
block_scores_tc_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void block_scores_tc_kernel(
    const float *__restrict__ pooled, const float *__restrict__ q_idx,
    const int32_t *__restrict__ steps, int64_t nq, int64_t max_blocks,
    int64_t reach, float *__restrict__ out, uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto sm = (float *)dpct_local;
    float* sQ = sm;                                   // [TC_QT][TC_QS]
    float* sK = sm + TC_QT * TC_QS;                   // [TC_NB][TC_KS]
    auto &s_nbid =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[TC_QT]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5,
              gid = lane >> 2, tig = lane & 3;
    const int64_t q0 = (int64_t)item_ct1.get_group(1) * TC_QT;
    for (int i = t; i < TC_QT * IDX_HEADS * IDX_DIM / 4; i += 128) {
        const int r = i / (IDX_HEADS * IDX_DIM / 4), c = i % (IDX_HEADS * IDX_DIM / 4);
        sycl::float4 v = sycl::float4(0.f, 0.f, 0.f, 0.f);
        if (q0 + r < nq) v = reinterpret_cast<const sycl::float4 *>(
            q_idx + (q0 + r) * IDX_HEADS * IDX_DIM)[c];
        *reinterpret_cast<sycl::float4 *>(sQ + r * TC_QS + c * 4) = v;
    }
    if (t < TC_QT) s_nbid[t] = q0 + t < nq ? steps[(q0 + t) * kStepCount + kStepNBid] : 0;
    int lo_nbid = 0x7fffffff, hi_nbid = 0;
    item_ct1.barrier(sycl::access::fence_space::local_space);
#pragma unroll
    for (int i = 0; i < TC_QT; ++i) {
        if (q0 + i >= nq) break;
        lo_nbid = sycl::min(lo_nbid, s_nbid[i]);
        hi_nbid = sycl::max(hi_nbid, s_nbid[i]);
    }
    for (int it = 0; it < TC_ITER; ++it) {
        const int64_t b0 =
            ((int64_t)item_ct1.get_group(2) * TC_ITER + it) * TC_NB;
        if (b0 >= reach || b0 >= hi_nbid) break;      // blocks >= every query's n_bid: nothing to score
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier(); // the previous tile's reads are done
        for (int i = t; i < TC_NB * IDX_DIM / 4; i += 128) {
            const int r = i / (IDX_DIM / 4), c = i % (IDX_DIM / 4);
            sycl::float4 v = sycl::float4(0.f, 0.f, 0.f, 0.f);
            if (b0 + r < hi_nbid) v = reinterpret_cast<const sycl::float4 *>(
                pooled + (b0 + r) * IDX_DIM)[c];
            *reinterpret_cast<sycl::float4 *>(sK + r * TC_KS + c * 4) = v;
        }
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
        float acc[IDX_HEADS][4];
#pragma unroll
        for (int h = 0; h < IDX_HEADS; ++h) acc[h][0] = acc[h][1] = acc[h][2] = acc[h][3] = 0.f;
        const float* kr = sK + (warp * 8 + gid) * TC_KS;
#pragma unroll 4
        for (int k0 = 0; k0 < IDX_DIM; k0 += 8) {
            const float kx0 = kr[k0 + tig], kx1 = kr[k0 + tig + 4];
            uint32_t bh[2], bl[2];
            bh[0] = tf32_hi(kx0);
            bh[1] = tf32_hi(kx1);
            bl[0] = tf32_hi(kx0 - sycl::bit_cast<float>(bh[0]));
            bl[1] = tf32_hi(kx1 - sycl::bit_cast<float>(bh[1]));
#pragma unroll
            for (int h = 0; h < IDX_HEADS; ++h) {
                const float* qa = sQ + h * IDX_DIM + k0 + tig;
                const float x0 = qa[gid * TC_QS], x1 = qa[(gid + 8) * TC_QS];
                const float x2 = qa[gid * TC_QS + 4], x3 = qa[(gid + 8) * TC_QS + 4];
                uint32_t ah[4], al[4];
                ah[0] = tf32_hi(x0); ah[1] = tf32_hi(x1); ah[2] = tf32_hi(x2); ah[3] = tf32_hi(x3);
                al[0] = tf32_hi(x0 - sycl::bit_cast<float>(ah[0]));
                al[1] = tf32_hi(x1 - sycl::bit_cast<float>(ah[1]));
                al[2] = tf32_hi(x2 - sycl::bit_cast<float>(ah[2]));
                al[3] = tf32_hi(x3 - sycl::bit_cast<float>(ah[3]));
                mma_tf32(acc[h], al, bh);
                mma_tf32(acc[h], ah, bl);
                mma_tf32(acc[h], ah, bh);
            }
        }
        // relu per head, heads added in order (as the warp kernel), written where the block completed for the query
        const int64_t bc = b0 + warp * 8 + 2 * tig;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int qr = gid + half * 8;
            const int64_t qi = q0 + qr;
            if (qi >= nq) continue;
            const int nb = s_nbid[qr];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int64_t b = bc + j;
                if (b >= nb || b >= max_blocks) continue;
                float score = 0.0f;
#pragma unroll
                for (int h = 0; h < IDX_HEADS; ++h) {
                    const float d = acc[h][half * 2 + j];
                    score += d > 0.0f ? d : 0.0f;
                }
                out[qi * max_blocks + b] = score;
            }
        }
    }
}

#if defined(__HIPCC__)
// ---- gfx12 (RDNA4: gfx1200 / gfx1201) scorer on v_wmma_f32_16x16x16_bf16 (wave32).  Same GEMM as above: for a tile of 16
// queries, every block's score is sum over the 4 indexer heads of relu(q_h . k), K = 128.  FP32-level accuracy from a
// three-way bf16 split of both operands (x = hi + mid + lo EXACTLY - the top 8, next 8 and last 8 bits of the fp32
// mantissa, so bf16's fp32-sized exponent range needs no scaling) and the six products of order <= 2, smallest first;
// like the TF32 kernel it sums in another order than the warp kernel (not bitwise).  The tail block n_bid is the warp
// kernel's arithmetic (block_scores_tail_kernel).
//   D = A x B with A = 16 key blocks (row l%16 of a lane, k = 8*(l/16)+i), B = 16 queries (column l%16, same k):
//   a lane ends up with 8 consecutive blocks of ONE query, written as one run.  No LDS for keys: a lane reads its
//   32-byte slice of the key row from global (the 16 query tiles of a launch re-read them from L2); the queries are
//   split once per CTA into LDS.
// gfx11 (RDNA3 / RDNA3.5) has the same instruction with another fragment layout: a lane holds all 16 k of its row
// (A) or column (B) - lanes 16..31 repeat lanes 0..15 - and D[2i + l/16][l%16]; SEL_KL is the k per lane, SEL_KOFF(g)
// the first, SEL_ROW(i, g) the D row of element i.  Same products, same order per element as on gfx12.
#if defined(__gfx1200__) || defined(__gfx1201__)
#define STRATA_SEL_GFX12 1
#define STRATA_SEL_GFX11 0
#elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1150__) || \
    defined(__gfx1151__)
#define STRATA_SEL_GFX12 0
#define STRATA_SEL_GFX11 1
#else
#define STRATA_SEL_GFX12 0
#define STRATA_SEL_GFX11 0
#endif
#if STRATA_SEL_GFX11
constexpr int SEL_KL = 16;
#define SEL_KOFF(g) 0
#define SEL_ROW(i, g) (2 * (i) + (g))
#else
constexpr int SEL_KL = 8;
#define SEL_KOFF(g) ((g) * 8)
#define SEL_ROW(i, g) ((g) * 8 + (i))
#endif
typedef short sel_s8 __attribute__((ext_vector_type(SEL_KL)));
typedef float sel_f8 __attribute__((ext_vector_type(8)));
typedef uint32_t sel_u4 __attribute__((ext_vector_type(SEL_KL / 2)));
constexpr int WQT = 16;                    // queries per CTA (the N of one WMMA)
constexpr int WITER = 4;                   // key tiles per warp (the CTA covers 4 warps * WITER * 16 blocks)
constexpr int WQS = IDX_DIM + 8;           // bf16 elements per LDS row: 272 bytes, conflict-free 16-byte reads

__device__ __forceinline__ sel_f8 wmma_bf16(const sel_s8& a, const sel_s8& b, const sel_f8& c) {
#if STRATA_SEL_GFX12
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
#elif STRATA_SEL_GFX11
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
#else
    /* unreachable on SYCL: the launcher refuses this device */
    return c;
#endif
}
// the bf16 pieces of x as fp32 bit patterns whose upper 16 bits are the bf16 (lower 16 are zero)
__device__ __forceinline__ void split3(float x, uint32_t& hi, uint32_t& mid, uint32_t& lo) {
    hi = sycl::bit_cast<uint32_t>(x) & 0xffff0000u;
    const float r1 = x - __uint_as_float(hi);                   // exact
    mid = sycl::bit_cast<uint32_t>(r1) & 0xffff0000u;
    lo = sycl::bit_cast<uint32_t>(r1 - __uint_as_float(mid)) & 0xffff0000u;   // exact: at most 8 significant bits left
}
__device__ __forceinline__ uint32_t pack_bf16x2(uint32_t a, uint32_t b) {   // low half = a's bf16, high half = b's
#if STRATA_SEL_GFX12 || STRATA_SEL_GFX11
    return __builtin_amdgcn_perm(b, a, 0x07060302u);
#else
    return (a >> 16) | (b & 0xffff0000u);
#endif
}

__global__ void __launch_bounds__(128) block_scores_wmma_kernel(const float* __restrict__ pooled,
                                                                const float* __restrict__ q_idx,
                                                                const int32_t* __restrict__ steps, int64_t nq,
                                                                int64_t max_blocks, int64_t reach,
                                                                float* __restrict__ out) {
    __shared__ __align__(16) uint16_t sq[3][IDX_HEADS][WQT][WQS];   // hi / mid / lo, [head][query][dim]
    __shared__ int s_nbid[WQT];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5, l16 = lane & 15, g = lane >> 4;
    const int64_t q0 = (int64_t) blockIdx.y * WQT;
    for (int i = t; i < WQT * IDX_HEADS * IDX_DIM / 4; i += 128) {
        const int row = i / (IDX_DIM / 4), c = i % (IDX_DIM / 4);
        const int qr = row / IDX_HEADS, h = row % IDX_HEADS;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (q0 + qr < nq) v = reinterpret_cast<const float4*>(q_idx + (q0 + qr) * IDX_HEADS * IDX_DIM)[h * (IDX_DIM / 4) + c];
        const float x[4] = {v.x, v.y, v.z, v.w};
        uint32_t hb[4], mb[4], lb[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) split3(x[j], hb[j], mb[j], lb[j]);
        *reinterpret_cast<uint2*>(&sq[0][h][qr][c * 4]) = make_uint2(pack_bf16x2(hb[0], hb[1]), pack_bf16x2(hb[2], hb[3]));
        *reinterpret_cast<uint2*>(&sq[1][h][qr][c * 4]) = make_uint2(pack_bf16x2(mb[0], mb[1]), pack_bf16x2(mb[2], mb[3]));
        *reinterpret_cast<uint2*>(&sq[2][h][qr][c * 4]) = make_uint2(pack_bf16x2(lb[0], lb[1]), pack_bf16x2(lb[2], lb[3]));
    }
    if (t < WQT) s_nbid[t] = q0 + t < nq ? steps[(q0 + t) * kStepCount + kStepNBid] : 0;
    __syncthreads();
    int hi_nbid = 0;
#pragma unroll
    for (int i = 0; i < WQT; ++i) hi_nbid = max(hi_nbid, s_nbid[i]);
    const int64_t qi = q0 + l16;
    const int nb_q = s_nbid[l16];
    for (int it = 0; it < WITER; ++it) {
        const int64_t b0 = (((int64_t) blockIdx.x * WITER + it) * 4 + warp) * 16;
        if (b0 >= reach || b0 >= hi_nbid) break;          // warp-uniform; later tiles start higher
        const int64_t row = b0 + l16;
        const bool rv = row < hi_nbid && row < max_blocks;
        const float* kp = pooled + (rv ? row : 0) * IDX_DIM + SEL_KOFF(g);
        // acc: the hi*hi products; cor: the five smaller ones (<= 2^-8 of it). Kept apart and added once at the end: a
        // correction summed into the big accumulator is rounded to ITS ulp at every one of the 48 steps (measured: 8e-7
        // of the score scale; apart, near the warp kernel's)
        sel_f8 acc[IDX_HEADS], cor[IDX_HEADS];
#pragma unroll
        for (int h = 0; h < IDX_HEADS; ++h) acc[h] = cor[h] = sel_f8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll 2
        for (int kk = 0; kk < IDX_DIM / 16; ++kk) {
            float x[SEL_KL];
#pragma unroll
            for (int j = 0; j < SEL_KL / 4; ++j) {
                float4 kv = make_float4(0.f, 0.f, 0.f, 0.f);
                if (rv) kv = *reinterpret_cast<const float4*>(kp + kk * 16 + 4 * j);
                x[4 * j] = kv.x; x[4 * j + 1] = kv.y; x[4 * j + 2] = kv.z; x[4 * j + 3] = kv.w;
            }
            uint32_t hb[SEL_KL], mb[SEL_KL], lb[SEL_KL];
#pragma unroll
            for (int j = 0; j < SEL_KL; ++j) split3(x[j], hb[j], mb[j], lb[j]);
            sel_u4 ah, am, al;
#pragma unroll
            for (int j = 0; j < SEL_KL / 2; ++j) {
                ah[j] = pack_bf16x2(hb[2 * j], hb[2 * j + 1]);
                am[j] = pack_bf16x2(mb[2 * j], mb[2 * j + 1]);
                al[j] = pack_bf16x2(lb[2 * j], lb[2 * j + 1]);
            }
            const sel_s8 Ah = __builtin_bit_cast(sel_s8, ah), Am = __builtin_bit_cast(sel_s8, am),
                         Al = __builtin_bit_cast(sel_s8, al);
#pragma unroll
            for (int h = 0; h < IDX_HEADS; ++h) {
                const sel_s8 Bh = *reinterpret_cast<const sel_s8*>(&sq[0][h][l16][kk * 16 + SEL_KOFF(g)]);
                const sel_s8 Bm = *reinterpret_cast<const sel_s8*>(&sq[1][h][l16][kk * 16 + SEL_KOFF(g)]);
                const sel_s8 Bl = *reinterpret_cast<const sel_s8*>(&sq[2][h][l16][kk * 16 + SEL_KOFF(g)]);
                cor[h] = wmma_bf16(Al, Bh, cor[h]);
                cor[h] = wmma_bf16(Ah, Bl, cor[h]);
                cor[h] = wmma_bf16(Am, Bm, cor[h]);
                cor[h] = wmma_bf16(Ah, Bm, cor[h]);
                cor[h] = wmma_bf16(Am, Bh, cor[h]);
                acc[h] = wmma_bf16(Ah, Bh, acc[h]);
            }
        }
        // relu per head, heads added in order (as the warp kernel); this lane: query qi, blocks b0 + SEL_ROW(i, g)
        if (qi < nq) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int64_t b = b0 + SEL_ROW(i, g);
                if (b >= nb_q || b >= max_blocks) continue;
                float score = 0.0f;
#pragma unroll
                for (int h = 0; h < IDX_HEADS; ++h) {
                    const float d = acc[h][i] + cor[h][i];
                    score += d > 0.0f ? d : 0.0f;
                }
                out[qi * max_blocks + b] = score;
            }
        }
    }
}

// the WMMA kernel needs gfx12 (gfx1200/gfx1201) or gfx11 (gfx110x/gfx115x: the other fragment layout) code objects
bool sel_gfx12_device() {
    static int ok[64] = {};   // per device: 0 unknown, 1 yes, 2 no
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
    if (ok[dev] == 0) {
        cudaDeviceProp prop;
        ok[dev] = (cudaGetDeviceProperties(&prop, dev) == cudaSuccess &&
                   (std::strncmp(prop.gcnArchName, "gfx1200", 7) == 0 || std::strncmp(prop.gcnArchName, "gfx1201", 7) == 0 ||
                    strata::kernels::gfx_arch_is_gfx11_wmma(prop.gcnArchName)))
                      ? 1 : 2;
        cudaGetLastError();
    }
    return ok[dev] == 1;
}

// RDNA2 (gfx1030-gfx1036): the FP32 tiled scorer's tiling measured there
bool sel_gfx103_device() {
    static int ok[64] = {};   // per device: 0 unknown, 1 yes, 2 no
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
    if (ok[dev] == 0) {
        cudaDeviceProp prop;
        ok[dev] = (cudaGetDeviceProperties(&prop, dev) == cudaSuccess && std::strncmp(prop.gcnArchName, "gfx103", 6) == 0) ? 1 : 2;
        cudaGetLastError();
    }
    return ok[dev] == 1;
}
#endif  // __HIPCC__

// the tail block n_bid of each query: exactly block_scores_kernel's arithmetic for that block
__dpct_inline__ void block_scores_tail_kernel(const float *__restrict__ dead,
                                              const float *__restrict__ q_idx,
                                              const int32_t *__restrict__ steps,
                                              int64_t max_blocks,
                                              float *__restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t qi = item_ct1.get_group(2);
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid];
    if (n_bid >= max_blocks) return;
    const int lane = item_ct1.get_local_id(2) & 31;
    const sycl::float4 k4 =
        *reinterpret_cast<const sycl::float4 *>(dead + lane * 4);
    const float* q = q_idx + qi * IDX_HEADS * IDX_DIM + lane * 4;
    float score = 0.0f;
#pragma unroll
    for (int h = 0; h < IDX_HEADS; ++h) {
        const sycl::float4 q4 =
            *reinterpret_cast<const sycl::float4 *>(q + h * IDX_DIM);
        float d = k4.x() * q4.x() + k4.y() * q4.y() + k4.z() * q4.z() +
                  k4.w() * q4.w();
#pragma unroll
        /*
        DPCT1108: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        for (int o = 16; o > 0; o >>= 1) d +=
            dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                d, o);
        score += d > 0.0f ? d : 0.0f;
    }
    if (lane == 0) {
        if (n_kv % R != 0) score += 1e9f;
        out[qi * max_blocks + n_bid] = score;
    }
}

// ---- the block scores as an FP32 tiled GEMM for cards without a matrix-core scorer (CUDA below sm_80; HIP, opt-in):
// rows (query, indexer head) x columns (blocks), K = 128 through shared memory, 2 queries x 4 heads x TN blocks per
// thread, relu per head summed in registers.  The warp kernel spends 5 shuffles per 4 products and reached ~1 TFLOPS
// on an RTX 2080 Ti; this one ~6.9 (256 queries: 4.5x at 4K, 6.8x at 120K-250K).  FP32 like the warp kernel in another
// summation order, so not bitwise (as the tensor-core scorer).  Blocks < n_bid only; the tail block is
// block_scores_tail_kernel's.
//
// The tiling (QT queries and NB blocks per CTA, KC of K per shared-memory pass, TN blocks per thread) only decides
// which thread computes which score: every score is still the fmaf chain over k = 0..127 in order and the same relu sum
// over the heads, so every tiling gives the same bytes (an RX 6900 XT, qsa_select_bench at 32K and 128K, 14 tilings,
// also with 60 and 72 of its 80 CUs: all byte-identical).  CUDA keeps the tiling it was written with (32/64/32/4);
// gfx103x runs 32/128/16/8, 15% faster there at 128K (0.695 -> 0.587 ms per 256 queries).
template <int QT, int NB, int KC, int TN>
/*
DPCT1110: The total declared local variable size in device function
block_scores_simt_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void block_scores_simt_kernel(const float *__restrict__ pooled,
                                              const float *__restrict__ q_idx,
                                              const int32_t *__restrict__ steps,
                                              int64_t nq, int64_t max_blocks,
                                              int64_t reach,
                                              float *__restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int ROWS = QT * IDX_HEADS, QS = ROWS + 4, KS = NB + 4,
                  CX = NB / TN;
    static_assert((ROWS / 8) * CX == 256, "256 threads, 8 rows (2 queries x 4 heads) each");
    static_assert(TN % 4 == 0 && KC % 4 == 0 && IDX_DIM % KC == 0, "float4 loads");
    auto &Qs =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[KC][QS]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &Ks =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[KC][KS]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int64_t q0 = (int64_t)item_ct1.get_group(1) * QT,
                  b0 = (int64_t)item_ct1.get_group(2) * NB;
    const int64_t qlast = (q0 + QT < nq ? q0 + QT : nq) - 1;
    if (b0 >= steps[qlast * kStepCount + kStepNBid]) return;   // n_bid rises with the position
    const int tid = item_ct1.get_local_id(2), tx = tid % CX, ty = tid / CX;
    float acc[8][TN] = {};
    for (int kc = 0; kc < IDX_DIM; kc += KC) {
        for (int idx = tid; idx < ROWS * (KC / 4); idx += 256) {   // the query rows: ROWS x KC/4 float4
            const int r = idx / (KC / 4), k4 = idx % (KC / 4);
            sycl::float4 v = sycl::float4(0.f, 0.f, 0.f, 0.f);
            if (q0 + (r >> 2) < nq)
                v = *reinterpret_cast<const sycl::float4 *>(
                    q_idx + (q0 * IDX_HEADS + r) * IDX_DIM + kc + k4 * 4);
            Qs[k4 * 4 + 0][r] = v.x(); Qs[k4 * 4 + 1][r] = v.y();
                Qs[k4 * 4 + 2][r] = v.z(); Qs[k4 * 4 + 3][r] = v.w();
        }
        for (int idx = tid; idx < NB * (KC / 4); idx += 256) {   // the pooled keys: NB x KC/4 float4
            const int c = idx / (KC / 4), k4 = idx % (KC / 4);
            sycl::float4 v = sycl::float4(0.f, 0.f, 0.f, 0.f);
            if (b0 + c < reach) v = *reinterpret_cast<const sycl::float4 *>(
                pooled + (b0 + c) * IDX_DIM + kc + k4 * 4);
            Ks[k4 * 4 + 0][c] = v.x(); Ks[k4 * 4 + 1][c] = v.y();
                Ks[k4 * 4 + 2][c] = v.z(); Ks[k4 * 4 + 3][c] = v.w();
        }
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        item_ct1.barrier(sycl::access::fence_space::local_space);
#pragma unroll 8
        for (int k = 0; k < KC; ++k) {
            float qr[8], kr[TN];
            *reinterpret_cast<sycl::float4 *>(&qr[0]) =
                *reinterpret_cast<const sycl::float4 *>(&Qs[k][ty * 8]);
            *reinterpret_cast<sycl::float4 *>(&qr[4]) =
                *reinterpret_cast<const sycl::float4 *>(&Qs[k][ty * 8 + 4]);
#pragma unroll
            for (int t = 0; t < TN; t += 4)
                *reinterpret_cast<sycl::float4 *>(&kr[t]) =
                    *reinterpret_cast<const sycl::float4 *>(
                        &Ks[k][tx * TN + t]);
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] = sycl::fma(qr[i], kr[j], acc[i][j]);
        }
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int64_t q = q0 + ty * 2 + i;
        if (q >= nq) continue;
        const int64_t n_bid = steps[q * kStepCount + kStepNBid];
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int64_t b = b0 + tx * TN + j;
            if (b >= n_bid || b >= max_blocks) continue;
            float sc = 0.0f;
#pragma unroll
            for (int h = 0; h < IDX_HEADS; ++h) {
                const float d = acc[i * 4 + h][j];
                sc += d > 0.0f ? d : 0.0f;
            }
            out[q * max_blocks + b] = sc;
        }
    }
}

// The FP32 tiled scores and the tail block; false when the grid does not fit.  `rdna2_tiles`: gfx103x's tiling.
static bool block_scores_simt(const float *pooled, const float *dead,
                              const float *q_idx, const int32_t *steps,
                              int64_t nq, int64_t max_blocks, int64_t reach,
                              float *scores, dpct::queue_ptr stream,
                              bool rdna2_tiles) {
    const int64_t qt = 32, nb = rdna2_tiles ? 128 : 64;
    const dpct::dim3 grid((unsigned)((reach + nb - 1) / nb),
                          (unsigned)((nq + qt - 1) / qt));
    if (grid.y > 65535) return false;
    if (rdna2_tiles)
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        stream->parallel_for<
            dpct_kernel_name<class block_scores_simt_kernel_84fe18,
                             dpct_kernel_scalar<32>, dpct_kernel_scalar<128>,
                             dpct_kernel_scalar<16>, dpct_kernel_scalar<8>>>(
            sycl::nd_range<3>(grid * sycl::range(1, 1, 256),
                              sycl::range(1, 1, 256)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                block_scores_simt_kernel<32, 128, 16, 8>(
                    pooled, q_idx, steps, nq, max_blocks, reach, scores);
            });
    } else {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        stream->parallel_for<
            dpct_kernel_name<class block_scores_simt_kernel_b12507,
                             dpct_kernel_scalar<32>, dpct_kernel_scalar<64>,
                             dpct_kernel_scalar<32>, dpct_kernel_scalar<4>>>(
            sycl::nd_range<3>(grid * sycl::range(1, 1, 256),
                              sycl::range(1, 1, 256)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                block_scores_simt_kernel<32, 64, 32, 4>(
                    pooled, q_idx, steps, nq, max_blocks, reach, scores);
            });
    }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        stream->parallel_for<
            dpct_kernel_name<class block_scores_tail_kernel_42ca31>>(
            sycl::nd_range<3>(sycl::range(1, 1, (unsigned)nq) *
                                  sycl::range(1, 1, 32),
                              sycl::range(1, 1, 32)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                block_scores_tail_kernel(dead, q_idx, steps, max_blocks,
                                         scores);
            });
    }
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

    return true;
}

// block_scores_simt_kernel with larger tiles (QT queries x NB blocks, RI x CJ products per thread) and the next K
// chunk's loads in registers while the current one is used.  Every (row, block) dot is the same fmaf chain over
// k = 0..127 from 0, and the relu sum over the heads in the same order: the same bits as block_scores_simt_kernel.
// STRATA_SELECT_SIMT=2.
constexpr int S2_KC = 32;
template <int QT, int NB>
/*
DPCT1110: The total declared local variable size in device function
block_scores_simt2_kernel exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
__dpct_inline__ void block_scores_simt2_kernel(
    const float *__restrict__ pooled, const float *__restrict__ q_idx,
    const int32_t *__restrict__ steps, int64_t nq, int64_t max_blocks,
    int64_t reach, float *__restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int ROWS = QT * IDX_HEADS, RI = ROWS / 16, CJ = NB / 16,
                  QS = ROWS + 4, KS = NB + 4;
    constexpr int QL = ROWS * S2_KC / 4 / 256, KL = NB * S2_KC / 4 / 256;   // float4 loads per thread
    static_assert(RI % 4 == 0 && CJ % 4 == 0 && QL >= 1 && KL >= 1, "tile");
    auto &Qs =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[S2_KC][QS]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &Ks =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[S2_KC][KS]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int64_t q0 = (int64_t)item_ct1.get_group(1) * QT,
                  b0 = (int64_t)item_ct1.get_group(2) * NB;
    const int64_t qlast = (q0 + QT < nq ? q0 + QT : nq) - 1;
    if (b0 >= steps[qlast * kStepCount + kStepNBid]) return;   // n_bid rises with the position
    const int tid = item_ct1.get_local_id(2), tx = tid & 15, ty = tid >> 4;
    float acc[RI][CJ];
#pragma unroll
    for (int i = 0; i < RI; ++i)
#pragma unroll
        for (int j = 0; j < CJ; ++j) acc[i][j] = 0.0f;
    sycl::float4 qv[QL], kv[KL];
    auto load = [&](int kc) {
#pragma unroll
        for (int i = 0; i < QL; ++i) {
            const int idx = tid + i * 256, r = idx >> 3, k4 = idx & 7;
            qv[i] = sycl::float4(0.f, 0.f, 0.f, 0.f);
            if (q0 + (r >> 2) < nq)
                qv[i] = *reinterpret_cast<const sycl::float4 *>(
                    q_idx + (q0 * IDX_HEADS + r) * IDX_DIM + kc + k4 * 4);
        }
#pragma unroll
        for (int i = 0; i < KL; ++i) {
            const int idx = tid + i * 256, c = idx >> 3, k4 = idx & 7;
            kv[i] = sycl::float4(0.f, 0.f, 0.f, 0.f);
            if (b0 + c < reach) kv[i] = *reinterpret_cast<const sycl::float4 *>(
                pooled + (b0 + c) * IDX_DIM + kc + k4 * 4);
        }
    };
    load(0);
#pragma unroll 1
    for (int kc = 0; kc < IDX_DIM; kc += S2_KC) {
#pragma unroll
        for (int i = 0; i < QL; ++i) {
            const int idx = tid + i * 256, r = idx >> 3, k4 = idx & 7;
            Qs[k4 * 4 + 0][r] = qv[i].x(); Qs[k4 * 4 + 1][r] = qv[i].y();
                Qs[k4 * 4 + 2][r] = qv[i].z(); Qs[k4 * 4 + 3][r] = qv[i].w();
        }
#pragma unroll
        for (int i = 0; i < KL; ++i) {
            const int idx = tid + i * 256, c = idx >> 3, k4 = idx & 7;
            Ks[k4 * 4 + 0][c] = kv[i].x(); Ks[k4 * 4 + 1][c] = kv[i].y();
                Ks[k4 * 4 + 2][c] = kv[i].z(); Ks[k4 * 4 + 3][c] = kv[i].w();
        }
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        item_ct1.barrier(sycl::access::fence_space::local_space);
        if (kc + S2_KC < IDX_DIM) load(kc + S2_KC);   // in flight during this chunk's products
#pragma unroll 4
        for (int k = 0; k < S2_KC; ++k) {
            float qr[RI], kr[CJ];
#pragma unroll
            for (int m = 0; m < RI / 4; ++m) {
                const sycl::float4 a = *reinterpret_cast<const sycl::float4 *>(
                    &Qs[k][ty * RI + 4 * m]);
                qr[4 * m] = a.x(); qr[4 * m + 1] = a.y(); qr[4 * m + 2] = a.z();
                    qr[4 * m + 3] = a.w();
            }
#pragma unroll
            for (int g = 0; g < CJ / 4; ++g) {
                const sycl::float4 b = *reinterpret_cast<const sycl::float4 *>(
                    &Ks[k][g * 64 + tx * 4]);
                kr[4 * g] = b.x(); kr[4 * g + 1] = b.y(); kr[4 * g + 2] = b.z();
                    kr[4 * g + 3] = b.w();
            }
#pragma unroll
            for (int i = 0; i < RI; ++i)
#pragma unroll
                for (int j = 0; j < CJ; ++j)
                    acc[i][j] = sycl::fma(qr[i], kr[j], acc[i][j]);
        }
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }
#pragma unroll
    for (int qi = 0; qi < RI / 4; ++qi) {
        const int64_t q = q0 + ty * (RI / 4) + qi;
        if (q >= nq) continue;
        const int64_t n_bid = steps[q * kStepCount + kStepNBid];
#pragma unroll
        for (int j = 0; j < CJ; ++j) {
            const int64_t b = b0 + (j >> 2) * 64 + tx * 4 + (j & 3);
            if (b >= n_bid || b >= max_blocks) continue;
            float sc = 0.0f;
#pragma unroll
            for (int h = 0; h < IDX_HEADS; ++h) {
                const float d = acc[qi * 4 + h][j];
                sc += d > 0.0f ? d : 0.0f;
            }
            out[q * max_blocks + b] = sc;
        }
    }
}

// ---- the same top-k with each query's keys read once: 1,024 threads hold up to TK_PER consecutive blocks' keys in
// registers (contexts up to 4 * 1024 * TK_PER cells), per-warp histograms, block-wide scans. The selection rule is
// block_topk_kernel's (radix threshold, ties to the lowest index, cells ascending): identical ids.
constexpr int TK_T = 1024;
constexpr int TK_PER = 33;
#if defined(__HIPCC__)
// AMD (RDNA, wave32): a 1,024-thread block may use 192 VGPRs per lane, so the keys of 66 blocks per thread fit - contexts
// up to 4 * 1024 * 66 = 270,336 cells (the 262,144 --max-context) keep the register kernel. NVIDIA's 64 registers per
// thread at 1,024 threads allow only TK_PER.
constexpr int TK_PER_MAX = 66;
#else
constexpr int TK_PER_MAX = TK_PER;
#endif

__dpct_inline__ int block_excl_scan(int v, int *s_warp, int &total) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = item_ct1.get_local_id(2) & 31,
              warp = item_ct1.get_local_id(2) >> 5;
    int x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        /*
        DPCT1108: '__shfl_up_sync' was migrated with the experimental feature
        masked sub_group function which may not be supported by all compilers or
        runtimes. You may need to adjust the code.
        */
        const int y = dpct::experimental::shift_sub_group_right(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), x,
            o);
        if (lane >= o) x += y;
    }
    if (lane == 31) s_warp[warp] = x;
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (warp == 0) {
        int w = s_warp[lane];
        int z = w;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            /*
            DPCT1108: '__shfl_up_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            const int y = dpct::experimental::shift_sub_group_right(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                z, o);
            if (lane >= o) z += y;
        }
        s_warp[lane] = z - w;               // exclusive per warp
        if (lane == 31) s_warp[32] = z;     // total
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const int r = s_warp[warp] + x - v;
    total = s_warp[32];
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    return r;
}

template <int PER>
/*
DPCT1110: The total declared local variable size in device function
block_topk_reg_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void block_topk_reg_kernel(const float *__restrict__ scores,
                                           const int32_t *__restrict__ steps,
                                           int64_t max_blocks, int64_t cap,
                                           int32_t *__restrict__ ids) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &hist =
    *sycl::ext::oneapi::group_local_memory_for_overwrite<int[TK_T / 32][256]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_warp =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[33]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_digit = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_above = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int64_t qi = item_ct1.get_group(2);
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + qi * cap;
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    if (n_kv <= width) {
#pragma unroll
        for (int64_t j = t; j < n_kv; j += TK_T) out[j] = (int32_t)j;
        return;
    }
    const float* sc = scores + qi * max_blocks;
    const int64_t nb = n_bid + 1;
    const int64_t per = (nb + TK_T - 1) / TK_T;       // <= PER (the caller checks)
    const int64_t b0 = (int64_t) t * per, b1 = (b0 + per < nb) ? b0 + per : nb;
    uint32_t key[PER];
#pragma unroll
    for (int j = 0; j < PER; ++j) key[j] = (b0 + j < b1) ? order_key(sc[b0 + j]) : 0u;
    auto weight = [&](int64_t b) -> int { return b < n_bid ? R : (int) (n_kv - n_bid * R); };
    uint32_t prefix = 0;
    int above = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
#pragma unroll
        for (int i = lane; i < 256; i += 32) hist[warp][i] = 0;
        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int64_t b = b0 + j;
            if (b >= b1) break;
            const int w = weight(b);
            if (w == 0) continue;
            if ((key[j] & hi_mask) == (prefix & hi_mask))
                dpct::atomic_fetch_add<
                    sycl::access::address_space::generic_space>(
                    &hist[warp][(key[j] >> shift) & 255], w);
        }
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
        if (t < 256) {                                // fold the warps' histograms into warp 0's
            int s = 0;
#pragma unroll
            for (int w2 = 0; w2 < TK_T / 32; ++w2) s += hist[w2][t];
            hist[0][t] = s;
        }
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
        // the digit: the first bin from the top where the count above reaches `width` (bin 0 when none of 255..1
        // does).  Threads 0-255 (warps 0-7) prefix-sum the 256 bins in descending order instead of one thread
        // walking them: the same digit and count above.
        int v = 0, x = 0;
        if (t < 256) {
            v = hist[0][255 - t];
            x = v;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                /*
                DPCT1108: '__shfl_up_sync' was migrated with the
                experimental feature masked sub_group function which may not be
                supported by all compilers or runtimes. You may need to adjust
                the code.
                */
                const int y = dpct::experimental::shift_sub_group_right(
                    0xffffffffu,
                    sycl::ext::oneapi::this_work_item::get_sub_group(), x, o);
                if (lane >= o) x += y;
            }
            if (lane == 31) s_warp[warp] = x;
        }
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
        if (t < 256) {
            int inc = x;
#pragma unroll
            for (int w2 = 0; w2 < warp; ++w2) inc += s_warp[w2];
            const int before = above + inc - v;
            if (before < width && (above + inc >= width || t == 255)) {
                s_digit = 255 - t;
                s_above = before;
            }
        }
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
        prefix |= (uint32_t) s_digit << shift;
        above = s_above;
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
    const uint32_t thr = prefix;
    const int64_t eq_budget = width - above;
    int gt = 0, eq = 0;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int64_t b = b0 + j;
        if (b >= b1) break;
        const int w = weight(b);
        if (w == 0) continue;
        if (key[j] > thr) gt += w;
        else if (key[j] == thr) eq += w;
    }
    int tot;
    const int eq_before = block_excl_scan(eq, s_warp, tot);
    int64_t my_eq = eq_budget - eq_before;
    if (my_eq < 0) my_eq = 0;
    if (my_eq > eq) my_eq = eq;
    const int sel = gt + (int) my_eq;
    int64_t wpos = block_excl_scan(sel, s_warp, tot);
    int64_t eq_left = my_eq;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int64_t b = b0 + j;
        if (b >= b1) break;
        const int w = weight(b);
        if (w == 0) continue;
        if (key[j] > thr) {
#pragma unroll
            for (int c = 0; c < w; ++c) out[wpos++] = (int32_t)(b * R + c);
        } else if (key[j] == thr) {
#pragma unroll
            for (int c = 0; c < w && eq_left > 0; ++c, --eq_left)
                out[wpos++] = (int32_t)(b * R + c);
        }
    }
}

// ---- a query with more blocks than the register kernel holds (contexts past 4 * 1024 * TK_PER cells): the register
// kernel's threads, per-warp histograms and warp scans, with each key read from memory again on every pass.  A
// histogram has no order, so on the four radix passes thread t reads blocks t, t + 1024, ...: a warp reads 32
// neighbours at a time.  The cells are emitted ascending in the order (warp, row, lane): warp w holds the `per` rows
// of 32 consecutive blocks from block w * 32 * per.  block_topk_kernel's selection rule (radix threshold, ties to the
// lowest index): identical ids.
// campaign3: also compiled on HIP (gfx11: plain shared/atomic code, no sm_90 features; the dispatch still needs
// reach > fit, i.e. the deep contexts where the ref radix was the only option on AMD).
constexpr int TS_MAXQ = 4096;
inline dpct::global_memory<int, 1>
    g_tk_fail(TS_MAXQ); // block_topk_sample_kernel: 1 = the query is
                        // block_topk_wide_kernel<true>'s
template <bool ONLY_FAILED = false>
/*
DPCT1110: The total declared local variable size in device function
block_topk_wide_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void
block_topk_wide_kernel(const float *__restrict__ scores,
                       const int32_t *__restrict__ steps, int64_t max_blocks,
                       int64_t cap, int32_t *__restrict__ ids, int *g_tk_fail) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    if (ONLY_FAILED && !g_tk_fail[item_ct1.get_group(2)]) return;
    auto &hist = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        int[TK_T / 32][256]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_gt =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[TK_T / 32]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_eq =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[TK_T / 32]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_digit = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_above = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int64_t qi = item_ct1.get_group(2);
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + qi * cap;
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    if (n_kv <= width) {
#pragma unroll
        for (int64_t j = t; j < n_kv; j += TK_T) out[j] = (int32_t)j;
        return;
    }
    const float* sc = scores + qi * max_blocks;
    const int64_t nb = n_bid + 1;
    const int64_t per = (nb + TK_T - 1) / TK_T;       // rows of 32 blocks per warp
    auto weight = [&](int64_t b) -> int { return b < n_bid ? R : (int) (n_kv - n_bid * R); };
    uint32_t prefix = 0;
    int above = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
#pragma unroll
        for (int i = lane; i < 256; i += 32) hist[warp][i] = 0;
        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        for (int64_t b = t; b < nb; b += TK_T) {
            const int w = weight(b);
            if (w == 0) continue;
            const uint32_t k = order_key(sc[b]);
            if ((k & hi_mask) == (prefix & hi_mask)) dpct::atomic_fetch_add<
                sycl::access::address_space::generic_space>(
                &hist[warp][(k >> shift) & 255], w);
        }
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
        if (t < 256) {                                // fold the warps' histograms into warp 0's
            int sum = 0;
#pragma unroll
            for (int w2 = 0; w2 < TK_T / 32; ++w2) sum += hist[w2][t];
            hist[0][t] = sum;
        }
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
        if (t == 0) {
            int cum = above, d = 255;
#pragma unroll
            for (; d > 0; --d) {
                if (cum + hist[0][d] >= width) break;
                cum += hist[0][d];
            }
            s_digit = d;
            s_above = cum;
        }
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
        prefix |= (uint32_t) s_digit << shift;
        above = s_above;
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
    const uint32_t thr = prefix;
    const int eq_budget = (int) (width - above);      // cells equal to thr that fit, lowest index first
    const int64_t w0 = (int64_t) warp * 32 * per;
    int gt = 0, eq = 0;
    for (int64_t b = w0 + lane; b < nb && b < w0 + 32 * per; b += 32) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if (k > thr) gt += w;
        else if (k == thr) eq += w;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        /*
        DPCT1108: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        gt += dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), gt,
            o);
        /*
        DPCT1108: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        eq += dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), eq,
            o);
    }
    if (lane == 0) { s_gt[warp] = gt; s_eq[warp] = eq; }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (t == 0) {                                     // per warp: the selected cells and the tied cells before it
        int eb = 0, sb = 0;
        for (int w2 = 0; w2 < TK_T / 32; ++w2) {
            const int g2 = s_gt[w2], e2 = s_eq[w2];
            const int take = eq_budget - eb < 0 ? 0 : (eq_budget - eb > e2 ? e2 : eq_budget - eb);
            s_gt[w2] = sb;
            s_eq[w2] = eb;
            sb += g2 + take;
            eb += e2;
        }
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    int run_sel = s_gt[warp], run_eq = s_eq[warp];
    for (int64_t r0 = w0; r0 < nb && r0 < w0 + 32 * per; r0 += 32) {
        const int64_t b = r0 + lane;
        const int w = b < nb ? weight(b) : 0;
        const uint32_t k = w ? order_key(sc[b]) : 0u;
        const int my_gt = (w && k > thr) ? w : 0, my_eq = (w && k == thr) ? w : 0;
        if (sycl::reduce_over_group(
                sycl::ext::oneapi::this_work_item::get_sub_group(),
                (0xffffffffu &
                 (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                             .get_local_linear_id())) &&
                        (my_gt | my_eq) != 0
                    ? (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                                  .get_local_linear_id())
                    : 0,
                sycl::ext::oneapi::plus<>()) ==
            0u) continue; // a row without a selected block
        int pe = my_eq;                               // tied cells up to and with this lane
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            /*
            DPCT1108: '__shfl_up_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            const int y = dpct::experimental::shift_sub_group_right(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                pe, o);
            if (lane >= o) pe += y;
        }
        const int left = eq_budget - (run_eq + pe - my_eq);
        const int take = left < 0 ? 0 : (left > my_eq ? my_eq : left);
        const int my_sel = my_gt + take;
        int ps = my_sel;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            /*
            DPCT1108: '__shfl_up_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            const int y = dpct::experimental::shift_sub_group_right(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                ps, o);
            if (lane >= o) ps += y;
        }
        int32_t* dst = out + run_sel + ps - my_sel;
#pragma unroll
        for (int c = 0; c < my_sel; ++c) dst[c] = (int32_t)(b * R + c);
        /*
        DPCT1108: '__shfl_sync' was migrated with the experimental feature
        masked sub_group function which may not be supported by all compilers or
        runtimes. You may need to adjust the code.
        */
        run_eq += dpct::experimental::select_from_sub_group(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), pe,
            31);
        /*
        DPCT1108: '__shfl_sync' was migrated with the experimental feature
        masked sub_group function which may not be supported by all compilers or
        runtimes. You may need to adjust the code.
        */
        run_sel += dpct::experimental::select_from_sub_group(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), ps,
            31);
    }
}


// Block scores with every key block read ONCE for all of a call's queries (block_scores_kernel's grid is
// (max_blocks / 8) x nq: ~24,600 mostly-idle blocks per layer at a decode window, each key re-read per query).  A fixed
// grid strides over the blocks; per (block, query) the same arithmetic in the same order as block_scores_kernel.
// qsa_block_scores takes it for every call without an active-block count and at most MQ queries: the captured decode
// window, the uncaptured decode, and prefill's pooled16 call.
constexpr int MQ = 8;
/*
DPCT1110: The total declared local variable size in device function
block_scores_multi_body exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void block_scores_multi_body(
    const float *__restrict__ pooled, const float *__restrict__ dead,
    const float *__restrict__ q_idx, const int32_t *__restrict__ steps, int nq,
    int64_t max_blocks, float *__restrict__ out, bool early_exit,
    const QsaScoreRows &srow) {
    /*
    DPCT1115: The sycl::ext::oneapi::group_local_memory_for_overwrite is
    used to allocate group-local memory at the none kernel functor scope of a
    work-group data parallel kernel. You may need to adjust the code.
    */
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &qs = *sycl::ext::oneapi::group_local_memory_for_overwrite<
    float[MQ * IDX_HEADS * IDX_DIM]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    /*
    DPCT1115: The sycl::ext::oneapi::group_local_memory_for_overwrite is
    used to allocate group-local memory at the none kernel functor scope of a
    work-group data parallel kernel. You may need to adjust the code.
    */
    auto &s_nkv =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int64_t[MQ]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_nbid =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int64_t[MQ]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    if (!early_exit)
#pragma unroll
        for (int i = item_ct1.get_local_id(2); i < nq * IDX_HEADS * IDX_DIM;
             i += item_ct1.get_local_range(2)) qs[i] = q_idx[i];
    if (item_ct1.get_local_id(2) < nq) {
        s_nkv[item_ct1.get_local_id(2)] =
            steps[item_ct1.get_local_id(2) * kStepCount + kStepNKv];
        s_nbid[item_ct1.get_local_id(2)] =
            steps[item_ct1.get_local_id(2) * kStepCount + kStepNBid];
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    int64_t top = 0;
#pragma unroll
    for (int q = 0; q < nq; ++q) top = s_nbid[q] > top ? s_nbid[q] : top;
    if (early_exit) {
        // (#783, stuchapin909) a CTA whose first block is past the batch's largest n_bid has no work: the grid is a
        // fixed 256 CTAs and a decode window reaches only a few of them, so those return before staging the queries.
        // Every thread of the CTA takes the same branch (top and blockIdx are CTA-uniform), so the barrier is safe.
        if ((int64_t)item_ct1.get_group(2) * SCORE_WARPS > top) return;
#pragma unroll
        for (int i = item_ct1.get_local_id(2); i < nq * IDX_HEADS * IDX_DIM;
             i += item_ct1.get_local_range(2)) qs[i] = q_idx[i];
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
    const int lane = item_ct1.get_local_id(2) & 31;
    const int64_t wstride = (int64_t)item_ct1.get_group_range(2) * SCORE_WARPS;
    for (int64_t b = (int64_t)item_ct1.get_group(2) * SCORE_WARPS +
                     (item_ct1.get_local_id(2) >> 5);
         b <= top && b < max_blocks; b += wstride) {
        // srow.n > 0: each query reads its own sequence's keys; else one set of keys for every query
        const bool own = srow.n > 0;
        const sycl::float4 kp = own ? sycl::float4{}
                                    : *reinterpret_cast<const sycl::float4 *>(
                                          pooled + b * IDX_DIM + lane * 4);
        const sycl::float4 kd =
            own ? sycl::float4{}
                : *reinterpret_cast<const sycl::float4 *>(dead + lane * 4);
        for (int qi = 0; qi < nq; ++qi) {
            const int64_t n_bid = s_nbid[qi];
            if (b > n_bid) continue;
            const sycl::float4 k4 =
                !own           ? ((b == n_bid) ? kd : kp)
                : (b == n_bid) ? *reinterpret_cast<const sycl::float4 *>(
                                     srow.dead[qi] + lane * 4)
                               : *reinterpret_cast<const sycl::float4 *>(
                                     srow.pooled[qi] + b * IDX_DIM + lane * 4);
            const float* q = qs + qi * IDX_HEADS * IDX_DIM + lane * 4;
            float score = 0.0f;
#pragma unroll
            for (int h = 0; h < IDX_HEADS; ++h) {
                const sycl::float4 q4 =
                    *reinterpret_cast<const sycl::float4 *>(q + h * IDX_DIM);
                float d = k4.x() * q4.x() + k4.y() * q4.y() + k4.z() * q4.z() +
                          k4.w() * q4.w();
#pragma unroll
                /*
                DPCT1108: '__shfl_xor_sync' was migrated with the
                experimental feature masked sub_group function which may not be
                supported by all compilers or runtimes. You may need to adjust
                the code.
                */
                for (int o = 16; o > 0; o >>= 1) d +=
                    dpct::experimental::permute_sub_group_by_xor(
                        0xffffffffu,
                        sycl::ext::oneapi::this_work_item::get_sub_group(), d,
                        o);
                score += d > 0.0f ? d : 0.0f;
            }
            if (lane == 0) {
                if (b == n_bid && s_nkv[qi] % R != 0) score += 1e9f;
                out[qi * max_blocks + b] = score;
            }
        }
    }
}

__dpct_inline__ void block_scores_multi_kernel(
    const float *__restrict__ pooled, const float *__restrict__ dead,
    const float *__restrict__ q_idx, const int32_t *__restrict__ steps, int nq,
    int64_t max_blocks, float *__restrict__ out, bool early_exit,
    QsaScoreRows srow = {}) {
    block_scores_multi_body(pooled, dead, q_idx, steps, nq, max_blocks, out, early_exit, srow);
}

// Queries of different slots, each with its own indexer keys: blockIdx.y is the query, and its CTAs run exactly the
// single-query call of block_scores_multi_kernel (qsa_block_scores_slots).
struct ScoreSlotKeys {
    const float* pooled[MQ];
    const float* dead[MQ];
};
__dpct_inline__ void
block_scores_slots_kernel(ScoreSlotKeys keys, const float *__restrict__ q_idx,
                          const int32_t *__restrict__ steps, int64_t max_blocks,
                          float *__restrict__ out, bool early_exit) {
    const unsigned z =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group(1);
    // the slot's pointers picked by unrolled selects: indexing the argument array by z would copy it to local memory
    const float* pooled = keys.pooled[0];
    const float* dead = keys.dead[0];
#pragma unroll
    for (unsigned i = 1; i < MQ; ++i)
        if (z == i) { pooled = keys.pooled[i]; dead = keys.dead[i]; }
    block_scores_multi_body(pooled, dead, q_idx + (size_t) z * IDX_HEADS * IDX_DIM,
                            steps + (size_t) z * kStepCount, 1, max_blocks, out + (size_t) z * max_blocks, early_exit,
                            QsaScoreRows{});
}

// QSAROWS (docs/QSAROWS.md, STRATA_QSA_ROWS=1): block_scores_multi_kernel's work for NQ <= 4 queries with less issue
// pressure per (block, query).  nsys of a 131k decode window (RTX 2080 Ti + RTX 3060): the multi kernel costs ~20 us more a
// call for every row of the window on both cards - per (block, query) it reads the four query slices from shared memory
// (4 x LDS.128) and reduces each head's lane products with its own five shuffles (20 a block and query), and those two
// pipes bound it, not the key reads (the keys are read once for all the queries, as there).  Here (1) the lane's slices
// of the NQ queries (4 heads x 4 floats each) live in registers for the whole kernel, (2) the four heads' lane products
// are reduce-scattered with warp_sum's pairing (xor 16, 8, then 4, 2, 1: six shuffles for the four heads) and gathered
// to lane 0 in head order (three more).  Every sum adds the same two operands as block_scores_kernel's warp_sum at the
// same level (float add commutes), the dot expression is the same text in the same file, the relu-and-add runs over the
// heads in the same order: the same score bits.  The tail block and the +1e9 are as in the multi kernel.
template <int NQ>
/*
DPCT1110: The total declared local variable size in device function
block_scores_rows_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void block_scores_rows_kernel(const float *__restrict__ pooled,
                                              const float *__restrict__ dead,
                                              const float *__restrict__ q_idx,
                                              const int32_t *__restrict__ steps,
                                              int nq, int64_t max_blocks,
                                              float *__restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &s_nkv = *sycl::ext::oneapi::group_local_memory_for_overwrite<int64_t[NQ]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_nbid =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int64_t[NQ]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    if (item_ct1.get_local_id(2) < nq) {
        s_nkv[item_ct1.get_local_id(2)] =
            steps[item_ct1.get_local_id(2) * kStepCount + kStepNKv];
        s_nbid[item_ct1.get_local_id(2)] =
            steps[item_ct1.get_local_id(2) * kStepCount + kStepNBid];
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    int64_t top = 0;
#pragma unroll
    for (int q = 0; q < nq; ++q) top = s_nbid[q] > top ? s_nbid[q] : top;
    if ((int64_t)item_ct1.get_group(2) * SCORE_WARPS >
        top) return; // the multi kernel's early exit (CTA-uniform)
    const int lane = item_ct1.get_local_id(2) & 31;
    sycl::float4 qr[NQ][IDX_HEADS];
#pragma unroll
    for (int qi = 0; qi < NQ; ++qi)
#pragma unroll
        for (int h = 0; h < IDX_HEADS; ++h)
            qr[qi][h] = qi < nq ? *reinterpret_cast<const sycl::float4 *>(
                                      q_idx + qi * IDX_HEADS * IDX_DIM +
                                      h * IDX_DIM + lane * 4)
                                : sycl::float4(0.f, 0.f, 0.f, 0.f);
    const sycl::float4 kd =
        *reinterpret_cast<const sycl::float4 *>(dead + lane * 4);
    const bool hi16 = (lane & 16) != 0, hi8 = (lane & 8) != 0;
    const int64_t wstride = (int64_t)item_ct1.get_group_range(2) * SCORE_WARPS;
    for (int64_t b = (int64_t)item_ct1.get_group(2) * SCORE_WARPS +
                     (item_ct1.get_local_id(2) >> 5);
         b <= top && b < max_blocks; b += wstride) {
        const sycl::float4 kp = *reinterpret_cast<const sycl::float4 *>(
            pooled + b * IDX_DIM + lane * 4);
#pragma unroll
        for (int qi = 0; qi < NQ; ++qi) {
            if (qi >= nq) break;
            const int64_t n_bid = s_nbid[qi];
            if (b > n_bid) continue;
            const sycl::float4 k4 = (b == n_bid) ? kd : kp;
            float d[IDX_HEADS];
#pragma unroll
            for (int h = 0; h < IDX_HEADS; ++h) {
                const sycl::float4 q4 = qr[qi][h];
                d[h] = k4.x() * q4.x() + k4.y() * q4.y() + k4.z() * q4.z() +
                       k4.w() * q4.w();
            }
            // reduce-scatter: after xor 16 the lane keeps heads {0,1} (bit 4 clear) or {2,3}; after xor 8 one head
            float r2[2];
#pragma unroll
            for (int i = 0; i < 2; ++i)
                /*
                DPCT1108: '__shfl_xor_sync' was migrated with the
                experimental feature masked sub_group function which may not be
                supported by all compilers or runtimes. You may need to adjust
                the code.
                */
                r2[i] = (hi16 ? d[i + 2] : d[i]) +
                        dpct::experimental::permute_sub_group_by_xor(
                            0xffffffffu,
                            sycl::ext::oneapi::this_work_item::get_sub_group(),
                            hi16 ? d[i] : d[i + 2], 16);
            /*
            DPCT1108: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            float r = (hi8 ? r2[1] : r2[0]) +
                      dpct::experimental::permute_sub_group_by_xor(
                          0xffffffffu,
                          sycl::ext::oneapi::this_work_item::get_sub_group(),
                          hi8 ? r2[0] : r2[1], 8);
            /*
            DPCT1108: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            r += dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                r, 4);
            /*
            DPCT1108: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            r += dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                r, 2);
            /*
            DPCT1108: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            r += dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                r, 1);
            // lane 8h holds head h's full sum: gather to lane 0 and add the relus in head order, as the warp kernel
            /*
            DPCT1108: '__shfl_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            const float s1 = dpct::experimental::select_from_sub_group(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                r, 8);
            /*
            DPCT1108: '__shfl_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            const float s2 = dpct::experimental::select_from_sub_group(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                r, 16);
            /*
            DPCT1108: '__shfl_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            const float s3 = dpct::experimental::select_from_sub_group(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                r, 24);
            if (lane == 0) {
                float score = 0.0f;
                score += r > 0.0f ? r : 0.0f;
                score += s1 > 0.0f ? s1 : 0.0f;
                score += s2 > 0.0f ? s2 : 0.0f;
                score += s3 > 0.0f ? s3 : 0.0f;
                if (b == n_bid && s_nkv[qi] % R != 0) score += 1e9f;
                out[qi * max_blocks + b] = score;
            }
        }
    }
}

#if defined(__HIPCC__)
// Optional gfx1100 port of PR1825 (14bf3215): same FP32 expression and reduction tree.
// Decode and the original top-k stay unchanged. No additional persistent weight/state allocation.
constexpr int TS_Q = 64;     // queries per CTA (8 per wave, four lanes per query)
constexpr int TS_KB = 32;    // key blocks per shared-memory tile
constexpr int TS_BR = 512;   // key blocks per CTA

__device__ __forceinline__ float ts_part(const float4 k4, const float4 q4) {
    return k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;   // block_scores_kernel's per-lane expression
}

__global__ void __launch_bounds__(256, 1) block_scores_thread_kernel(const float* __restrict__ pooled,
                                                                    const float* __restrict__ q_idx,
                                                                    const int32_t* __restrict__ steps, int64_t nq,
                                                                    int64_t max_blocks, int64_t reach,
                                                                    float* __restrict__ out) {
    __shared__ __align__(16) float4 ks[TS_KB][IDX_DIM / 4];
    __shared__ float so[TS_Q][TS_KB + 1];
    __shared__ int64_t s_nbid[TS_Q];
    // Four lanes per query, lane j (0..3) one quarter of the tree for all four heads: the partials c[j] of lane 0's
    // xor-shuffle sum combine groups j, j+16, j+8, j+24, j+4, j+20, j+12, j+28, and only then do the quarters meet
    // (c0 + c2, c1 + c3, then their sum) - two shuffles here.  A key load now feeds four heads: four times fewer
    // shared-memory loads per FMA. The MIO-pipe motivation was measured by PR1825 on NVIDIA.
    constexpr int QG[8] = {0, 16, 8, 24, 4, 20, 12, 28};
    const int t = threadIdx.x, lane = t & 31;
    const int ql = (t >> 5) * 8 + (lane >> 2), j4 = lane & 3;   // this lane's query (in the CTA) and quarter
    const int64_t q0 = (int64_t) blockIdx.y * TS_Q, qi = q0 + ql;
    if (t < TS_Q) s_nbid[t] = q0 + t < nq ? steps[(q0 + t) * kStepCount + kStepNBid] : -1;
    float4 qv[IDX_HEADS][8];
    {
        const float4* src = reinterpret_cast<const float4*>(q_idx + (qi < nq ? qi : 0) * IDX_HEADS * IDX_DIM);
#pragma unroll
        for (int h = 0; h < IDX_HEADS; ++h)
#pragma unroll
            for (int m = 0; m < 8; ++m) qv[h][m] = src[h * (IDX_DIM / 4) + j4 + QG[m]];
    }
    __syncthreads();
    int64_t top = -1;   // the largest n_bid of the CTA's queries: blocks at or past it are never written here
    for (int i = 0; i < TS_Q; ++i) top = s_nbid[i] > top ? s_nbid[i] : top;
    const int64_t b_lo = (int64_t) blockIdx.x * TS_BR;
    int64_t b_hi = b_lo + TS_BR;
    if (b_hi > reach) b_hi = reach;
    if (b_hi > top) b_hi = top;
    if (b_hi > max_blocks) b_hi = max_blocks;
    for (int64_t b0 = b_lo; b0 < b_hi; b0 += TS_KB) {
        const int nb = (int) (b_hi - b0 < TS_KB ? b_hi - b0 : TS_KB);
        __syncthreads();   // the previous tile's reads of ks and so are done
        for (int i = t; i < nb * (IDX_DIM / 4); i += blockDim.x)
            ks[i / (IDX_DIM / 4)][i % (IDX_DIM / 4)] = reinterpret_cast<const float4*>(pooled + (b0 + i / (IDX_DIM / 4)) * IDX_DIM)[i % (IDX_DIM / 4)];
        __syncthreads();
        for (int j = 0; j < nb; ++j) {
            float4 kk[8];
#pragma unroll
            for (int m = 0; m < 8; ++m) kk[m] = ks[j][j4 + QG[m]];
            float e[IDX_HEADS];
#pragma unroll
            for (int h = 0; h < IDX_HEADS; ++h) {
                const float c = ((ts_part(kk[0], qv[h][0]) + ts_part(kk[1], qv[h][1])) +
                                 (ts_part(kk[2], qv[h][2]) + ts_part(kk[3], qv[h][3]))) +
                                ((ts_part(kk[4], qv[h][4]) + ts_part(kk[5], qv[h][5])) +
                                 (ts_part(kk[6], qv[h][6]) + ts_part(kk[7], qv[h][7])));
                const float d = c + __shfl_xor_sync(0xffffffffu, c, 2);   // lane 0: c0 + c2, lane 1: c1 + c3
                e[h] = d + __shfl_xor_sync(0xffffffffu, d, 1);           // lane 0: (c0 + c2) + (c1 + c3)
            }
            if (j4 == 0) {
                float score = 0.0f;
#pragma unroll
                for (int h = 0; h < IDX_HEADS; ++h) score += e[h] > 0.0f ? e[h] : 0.0f;
                so[ql][j] = score;
            }
        }
        __syncthreads();
        for (int i = t; i < TS_Q * nb; i += blockDim.x) {
            const int q = i / nb, j = i % nb;
            if (b0 + j < s_nbid[q]) out[(q0 + q) * max_blocks + b0 + j] = so[q][j];
        }
    }
}

// The exact scorer runs on gfx1100 (the PR's own card) when STRATA_SCORES_THREAD=1, and on gfx12 (RDNA4) unless STRATA_SCORES_THREAD=0:
// there it is bitwise equal to the warp scorer on an R9700 (qsa_scores_gfx1100_parity --full, 35 cases; exact_pp residual dumps at 70K and
// 131K tokens) and 3.6-4.4x faster in the scorer stage, +8% (131K) to +15% (230K) on a prompt read.  Other wave32 parts have no measurement.
// Returns 0 (not offered), 1 (gfx1100: only on request), 2 (gfx12: the default).
int scores_thread_arch() {
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) { cudaGetLastError(); return 0; }
    static thread_local int cached_device = -1;
    static thread_local int arch = 0;
    if (device != cached_device) {
        cudaDeviceProp properties{};
        arch = 0;
        if (cudaGetDeviceProperties(&properties, device) == cudaSuccess)
            arch = gfx_arch_is_gfx12(properties.gcnArchName) ? 2 : gfx_arch_is(properties.gcnArchName, "gfx1100") ? 1 : 0;
        cached_device = device;
        cudaGetLastError();
    }
    return arch;
}
#endif

#if !defined(__HIPCC__)
// ---- prompt calls on cards without the tensor-core scorer (Turing): block_scores_kernel's scores, BITWISE, with a
// thread per (query, head) instead of a warp per (query, block).  There a warp spends 20 shuffles and five 512-byte
// loads per score; here a lane holds its query head (128 floats) in registers, every lane of the warp reads the same
// key from shared memory (a broadcast), and the 4-value partials are added in the tree lane 0 of the xor-shuffle sum
// computes: partial g covers values 4g..4g+3, level o adds g and g + o.  Float addition is commutative, so the same
// pairs in the same tree give the same bits.  The heads are then summed 0..3 from 0.0f as there.  Blocks < n_bid only:
// the tail block n_bid (the `dead` key, +1e9) is block_scores_tail_kernel's.
constexpr int TS_Q = 64;     // queries per CTA (8 per warp, a lane per query head)
constexpr int TS_KB = 32;    // key blocks per shared-memory tile
constexpr int TS_BR = 512;   // key blocks per CTA

__dpct_inline__ float ts_part(const sycl::float4 k4, const sycl::float4 q4) {
    return k4.x() * q4.x() + k4.y() * q4.y() + k4.z() * q4.z() +
           k4.w() * q4.w(); // block_scores_kernel's per-lane expression
}

/*
DPCT1110: The total declared local variable size in device function
block_scores_thread_kernel exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
__dpct_inline__ void block_scores_thread_kernel(
    const float *__restrict__ pooled, const float *__restrict__ q_idx,
    const int32_t *__restrict__ steps, int64_t nq, int64_t max_blocks,
    int64_t reach, float *__restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &ks = *sycl::ext::oneapi::group_local_memory_for_overwrite<
    sycl::float4[TS_KB][IDX_DIM / 4]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &so = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[TS_Q][TS_KB + 1]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_nbid =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int64_t[TS_Q]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    // Four lanes per query, lane j (0..3) one quarter of the tree for all four heads: the partials c[j] of lane 0's
    // xor-shuffle sum combine groups j, j+16, j+8, j+24, j+4, j+20, j+12, j+28, and only then do the quarters meet
    // (c0 + c2, c1 + c3, then their sum) - two shuffles here.  A key load now feeds four heads: four times fewer
    // shared-memory loads per FMA (the kernel was bound by the MIO pipe, ncu: mio_throttle first).
    constexpr int QG[8] = {0, 16, 8, 24, 4, 20, 12, 28};
    const int t = item_ct1.get_local_id(2), lane = t & 31;
    const int ql = (t >> 5) * 8 + (lane >> 2), j4 = lane & 3;   // this lane's query (in the CTA) and quarter
    const int64_t q0 = (int64_t)item_ct1.get_group(1) * TS_Q, qi = q0 + ql;
    if (t < TS_Q) s_nbid[t] = q0 + t < nq ? steps[(q0 + t) * kStepCount + kStepNBid] : -1;
    sycl::float4 qv[IDX_HEADS][8];
    {
        const sycl::float4 *src = reinterpret_cast<const sycl::float4 *>(
            q_idx + (qi < nq ? qi : 0) * IDX_HEADS * IDX_DIM);
#pragma unroll
        for (int h = 0; h < IDX_HEADS; ++h)
#pragma unroll
            for (int m = 0; m < 8; ++m) qv[h][m] = src[h * (IDX_DIM / 4) + j4 + QG[m]];
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    int64_t top = -1;   // the largest n_bid of the CTA's queries: blocks at or past it are never written here
#pragma unroll
    for (int i = 0; i < TS_Q; ++i) top = s_nbid[i] > top ? s_nbid[i] : top;
    const int64_t b_lo = (int64_t)item_ct1.get_group(2) * TS_BR;
    int64_t b_hi = b_lo + TS_BR;
    if (b_hi > reach) b_hi = reach;
    if (b_hi > top) b_hi = top;
    if (b_hi > max_blocks) b_hi = max_blocks;
    for (int64_t b0 = b_lo; b0 < b_hi; b0 += TS_KB) {
        const int nb = (int) (b_hi - b0 < TS_KB ? b_hi - b0 : TS_KB);
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier(); // the previous tile's reads of ks and so are done
#pragma unroll
        for (int i = t; i < nb * (IDX_DIM / 4);
             i += item_ct1.get_local_range(2))
            ks[i / (IDX_DIM / 4)][i % (IDX_DIM / 4)] =
                reinterpret_cast<const sycl::float4 *>(
                    pooled +
                    (b0 + i / (IDX_DIM / 4)) * IDX_DIM)[i % (IDX_DIM / 4)];
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
        for (int j = 0; j < nb; ++j) {
            sycl::float4 kk[8];
#pragma unroll
            for (int m = 0; m < 8; ++m) kk[m] = ks[j][j4 + QG[m]];
            float e[IDX_HEADS];
#pragma unroll
            for (int h = 0; h < IDX_HEADS; ++h) {
                const float c = ((ts_part(kk[0], qv[h][0]) + ts_part(kk[1], qv[h][1])) +
                                 (ts_part(kk[2], qv[h][2]) + ts_part(kk[3], qv[h][3]))) +
                                ((ts_part(kk[4], qv[h][4]) + ts_part(kk[5], qv[h][5])) +
                                 (ts_part(kk[6], qv[h][6]) + ts_part(kk[7], qv[h][7])));
                /*
                DPCT1108: '__shfl_xor_sync' was migrated with the
                experimental feature masked sub_group function which may not be
                supported by all compilers or runtimes. You may need to adjust
                the code.
                */
                const float d =
                    c + dpct::experimental::permute_sub_group_by_xor(
                            0xffffffffu,
                            sycl::ext::oneapi::this_work_item::get_sub_group(),
                            c, 2); // lane 0: c0 + c2, lane 1: c1 + c3
                /*
                DPCT1108: '__shfl_xor_sync' was migrated with the
                experimental feature masked sub_group function which may not be
                supported by all compilers or runtimes. You may need to adjust
                the code.
                */
                e[h] =
                    d + dpct::experimental::permute_sub_group_by_xor(
                            0xffffffffu,
                            sycl::ext::oneapi::this_work_item::get_sub_group(),
                            d, 1); // lane 0: (c0 + c2) + (c1 + c3)
            }
            if (j4 == 0) {
                float score = 0.0f;
#pragma unroll
                for (int h = 0; h < IDX_HEADS; ++h) score += e[h] > 0.0f ? e[h] : 0.0f;
                so[ql][j] = score;
            }
        }
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
        for (int i = t; i < TS_Q * nb; i += item_ct1.get_local_range(2)) {
            const int q = i / nb, j = i % nb;
            if (b0 + j < s_nbid[q]) out[(q0 + q) * max_blocks + b0 + j] = so[q][j];
        }
    }
}
#endif

#if !defined(__HIPCC__)
// ---- the decode top-k on a thread-block cluster (sm_90+; S19).  One CTA per query (block_topk_reg_kernel, or
// block_topk_kernel above 33,792 blocks: a --max-context over ~135K) makes four radix passes and two scans over up to
// 65,538 blocks on ONE SM while the rest of the GPU idles - a decode window has 1-5 queries.  Here a cluster of CL_N
// CTAs shares a query: CTA r holds the keys of blocks [r * per, (r + 1) * per) in shared memory, builds the digit
// histogram of its keys, and PUSHES it into slot r of every CTA's `hin` (distributed shared memory); after one cluster
// barrier each CTA sums the CL_N slots itself, so all of them take the same digit.  The emit pass needs each CTA's
// cells above / at the threshold: pushed the same way, then each CTA offsets its own by the ranks before it.
// RTX 5070, per call at capacity = context (decode_cluster_parity --bench): 21.9 -> 15.6 us at 32K, 58 -> 18 at 128K,
// 200 -> 22 at 262K (the one-CTA kernels' digit search was serial, too: one thread over 256 bins per pass).
//
// IDENTICAL IDS, by construction rather than by luck: the threshold thr (the width-th largest key, cells counted with
// their weights) and `above` (cells with a larger key) are pure functions of the multiset of (key, weight) - integer
// histograms, summed in any order - and the digit rule is block_topk_kernel's, written as a scan (the largest digit d
// whose cells at or above it reach `need`; 0 when none). The cells emitted are then fixed: every cell of a block with
// key > thr, and the first eq_budget = width - above cells at thr in ascending order (a block at thr may be cut),
// written in ascending cell order. Position of a thread's first cell = (cells above thr before it) + min(cells at thr
// before it, eq_budget): the telescoped sum of the reference's per-thread clamp. NaN keys are 0 (order_key) as there.
//
// Barriers: barrier.cluster arrive / wait (each CTA's threads all take part; it also orders the CTA's own shared
// memory, so it doubles as __syncthreads). Phase 0 is a relaxed arrive at entry, waited before the first remote store
// (a CTA's shared memory exists once it runs). Pushes go out before the arrive that releases them, and nothing reads
// another CTA's memory after the last wait - so a CTA may exit without a closing barrier (a wait counts the threads
// that have not exited). `hin` alternates by pass parity: CTA x reads slot buffer p&1 in pass p before arriving at
// pass p+1's barrier, and nobody writes buffer p&1 again (pass p+2) before waiting on that barrier.
constexpr int CL_N = 8;        // CTAs per query (the portable cluster size)
constexpr int CL_T = 1024;     // threads per CTA
constexpr int CL_MAXQ = 16;    // larger calls (prefill sub-batches) fill the GPU with one CTA per query already
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP >= 900
#define STRATA_SEL_CLUSTER 1
#else
#define STRATA_SEL_CLUSTER 0   // an older target's code is a trap; qsa_block_topk_cluster never launches it there
#endif

#if STRATA_SEL_CLUSTER
__dpct_inline__ void cl_arrive_relaxed() {
    /*
    DPCT1053: Migration of device assembly code is not supported.
    */
    asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");
}
/*
DPCT1053: Migration of device assembly code is not supported.
*/
__dpct_inline__ void cl_arrive() {
    asm volatile("barrier.cluster.arrive.release.aligned;\n" :: : "memory");
}
/*
DPCT1053: Migration of device assembly code is not supported.
*/
__dpct_inline__ void cl_wait() {
    asm volatile("barrier.cluster.wait.acquire.aligned;\n" :: : "memory");
}
__dpct_inline__ unsigned cl_rank() {
    unsigned r;
    /*
    DPCT1053: Migration of device assembly code is not supported.
    */
    asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r));
    return r;
}
template <typename T>
__dpct_inline__ T *cl_map(T *p,
                          unsigned rank) { // p in CTA `rank`'s shared memory
    uint64_t o;
    /*
    DPCT1053: Migration of device assembly code is not supported.
    */
    asm volatile("mapa.u64 %0, %1, %2;\n"
                 : "=l"(o)
                 : "l"((uint64_t)p), "r"(rank));
    return reinterpret_cast<T*>(o);
}
// exclusive block scan of a 64-bit value over CL_T threads; `total` = the CTA's sum
__dpct_inline__ unsigned long long cl_excl_scan64(unsigned long long v,
                                                  unsigned long long *s_warp,
                                                  unsigned long long &total) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = item_ct1.get_local_id(2) & 31,
              warp = item_ct1.get_local_id(2) >> 5;
    unsigned long long x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        /*
        DPCT1108: '__shfl_up_sync' was migrated with the experimental feature
        masked sub_group function which may not be supported by all compilers or
        runtimes. You may need to adjust the code.
        */
        const unsigned long long y = dpct::experimental::shift_sub_group_right(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), x,
            o);
        if (lane >= o) x += y;
    }
    if (lane == 31) s_warp[warp] = x;
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (warp == 0) {
        const unsigned long long w = s_warp[lane];
        unsigned long long z = w;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            /*
            DPCT1108: '__shfl_up_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            const unsigned long long y =
                dpct::experimental::shift_sub_group_right(
                    0xffffffffu,
                    sycl::ext::oneapi::this_work_item::get_sub_group(), z, o);
            if (lane >= o) z += y;
        }
        s_warp[lane] = z - w;
        if (lane == 31) s_warp[32] = z;
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    total = s_warp[32];
    return s_warp[warp] + x - v;
}
#endif

// grid (CL_N, nq), cluster (CL_N, 1, 1), CL_T threads, dynamic shared memory: ceil(max_blocks / CL_N) keys
/*
DPCT1110: The total declared local variable size in device function
block_topk_cluster_kernel exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
void block_topk_cluster_kernel(const float *__restrict__ scores,
                               const int32_t *__restrict__ steps,
                               int64_t max_blocks, int64_t cap,
                               int32_t *__restrict__ ids, uint8_t *dpct_local) {
#if STRATA_SEL_CLUSTER
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto keys = (uint32_t *)dpct_local; // this CTA's blocks' keys
    auto &hin =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[2][CL_N][256]>(
            sycl::ext::oneapi::this_work_item::get_work_group<
                3>()); // the cluster's histograms, slot = the pushing rank
    auto &hloc =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[2][256]>(
            sycl::ext::oneapi::this_work_item::get_work_group<
                3>()); // this CTA's histogram
    auto &cnt = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        unsigned long long[CL_N]>(
        sycl::ext::oneapi::this_work_item::get_work_group<
            3>()); // per rank: its cells above thr (low 32 bits) and at it
                   // (high)
    auto &s_warp = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        unsigned long long[33]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_prefix =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<uint32_t>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_above = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const unsigned rank = cl_rank();
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    cl_arrive_relaxed();                                // phase 0: this CTA runs
    const int64_t qi = item_ct1.get_group(1);
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + qi * cap;
    if (n_kv <= width) {                                // the identity, split over the cluster; no remote access
#pragma unroll
        for (int64_t j = (int64_t)rank * CL_T + t; j < n_kv;
             j += (int64_t)CL_N * CL_T) out[j] = (int32_t)j;
        return;
    }
    // the tail block n_bid weighs its cells (0..3); with none it is not a candidate (the reference skips w == 0)
    const int wtail = (int) (n_kv - n_bid * R);
    const int64_t nbe = n_bid + (wtail > 0 ? 1 : 0);
    const int64_t per = (nbe + CL_N - 1) / CL_N;       // <= ceil(max_blocks / CL_N), the keys the host sized
    const int64_t lo = (int64_t) rank * per < nbe ? (int64_t) rank * per : nbe;
    const int64_t hi = lo + per < nbe ? lo + per : nbe;
    const int n = (int) (hi - lo);
    const int tail_i = (wtail > 0 && n_bid >= lo && n_bid < hi) ? (int) (n_bid - lo) : -1;
    const float* sc = scores + qi * max_blocks + lo;
#pragma unroll
    for (int i = t; i < n; i += CL_T) keys[i] = order_key(sc[i]);
#pragma unroll
    for (int i = t; i < 2 * 256; i += CL_T)(&hloc[0][0])[i] = 0;
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    uint32_t prefix = 0;
    int above = 0;                                      // cells strictly above the digits fixed so far
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        const uint32_t hmask = pass == 0 ? 0u : (0xffffffffu << (shift + 8));
        int* hl = hloc[pass & 1];
        // warp-aggregated: the scores share their top bits, so plain atomics would queue on a few bins
        for (int i0 = 0; i0 < n; i0 += CL_T) {
            const int i = i0 + t;
            int bin = -1;
            if (i < n) {
                const uint32_t k = keys[i];
                if ((k & hmask) == prefix) bin = (int) ((k >> shift) & 255);
            }
            const unsigned same = dpct::match_any_over_sub_group(
                sycl::ext::oneapi::this_work_item::get_sub_group(), 0xffffffffu,
                bin);
            if (bin >= 0 && lane == dpct::ffs<int>(same) - 1)
                dpct::atomic_fetch_add<
                    sycl::access::address_space::generic_space>(
                    &hl[bin], sycl::popcount(same) * R);
            if (bin >= 0 && i == tail_i) dpct::atomic_fetch_add<
                sycl::access::address_space::generic_space>(&hl[bin],
                                                            wtail - R);
        }
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
        if (pass == 0) cl_wait();                       // phase 0 done: every CTA's shared memory is there
        for (int i = t; i < CL_N * 64; i += CL_T) {     // the histogram into slot `rank` of every CTA, as int4
            const int dst = i >> 6, c = i & 63;
            *cl_map(reinterpret_cast<sycl::int4 *>(&hin[pass & 1][rank][0]) + c,
                    (unsigned)dst) =
                reinterpret_cast<const sycl::int4 *>(hl)[c];
        }
#pragma unroll
        for (int i = t; i < 256; i += CL_T) hloc[(pass + 1) & 1][i] = 0;
        cl_arrive();
        cl_wait();
        if (warp == 0) {
            // lane L owns digits 255 - 8L down to 248 - 8L; the digit holding the need-th cell from the top
            const int need = (int) width - above;
            int tot[8], part = 0;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const int d = 255 - 8 * lane - k;
                int v = 0;
#pragma unroll
                for (int r = 0; r < CL_N; ++r) v += hin[pass & 1][r][d];
                tot[k] = v;
                part += v;
            }
            int incl = part;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                /*
                DPCT1108: '__shfl_up_sync' was migrated with the experimental
                feature masked sub_group function which may not be supported by
                all compilers or runtimes. You may need to adjust the code.
                */
                const int y = dpct::experimental::shift_sub_group_right(
                    0xffffffffu,
                    sycl::ext::oneapi::this_work_item::get_sub_group(), incl,
                    o);
                if (lane >= o) incl += y;
            }
            const int excl = incl - part;
            const unsigned hit = sycl::reduce_over_group(
                sycl::ext::oneapi::this_work_item::get_sub_group(),
                (0xffffffffu &
                 (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                             .get_local_linear_id())) &&
                        excl < need && incl >= need
                    ? (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                                  .get_local_linear_id())
                    : 0,
                sycl::ext::oneapi::plus<>());
            // no hit cannot happen (the cells total n_kv > width); the reference would then take digit 0
            const int who = hit ? dpct::ffs<int>(hit) - 1 : 31;
            if (lane == who) {
                int acc = excl, k = 0;
#pragma unroll
                for (; k < 7; ++k) {
                    if (acc + tot[k] >= need) break;
                    acc += tot[k];
                }
                s_prefix = prefix | ((uint32_t) (255 - 8 * lane - k) << shift);
                s_above = above + acc;
            }
        }
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
        prefix = s_prefix;
        above = s_above;
    }
    const uint32_t thr = prefix;
    const int64_t eq_budget = width - above;           // cells at thr that fit, lowest index first
    // each thread a contiguous run of the CTA's keys; an odd run length keeps the shared reads conflict-free
    int seg = (n + CL_T - 1) / CL_T;
    if ((seg & 1) == 0 && seg > 0) ++seg;
    const int s0 = t * seg < n ? t * seg : n, s1 = s0 + seg < n ? s0 + seg : n;
    uint32_t gt = 0, eq = 0;
    for (int i = s0; i < s1; ++i) {
        const uint32_t k = keys[i], w = i == tail_i ? (uint32_t) wtail : (uint32_t) R;
        if (k > thr) gt += w;
        else if (k == thr) eq += w;
    }
    unsigned long long cta;
    const unsigned long long before =
        cl_excl_scan64((unsigned long long) gt | ((unsigned long long) eq << 32), s_warp, cta);
    if (t < CL_N) *cl_map(&cnt[rank], (unsigned) t) = cta;
    cl_arrive();
    cl_wait();
    int64_t gt_off = 0, eq_off = 0;
#pragma unroll
    for (unsigned r = 0; r < rank; ++r) {
        gt_off += (int64_t) (cnt[r] & 0xffffffffu);
        eq_off += (int64_t) (cnt[r] >> 32);
    }
    const int64_t eq_before = eq_off + (int64_t) (before >> 32);
    int64_t wpos = gt_off + (int64_t) (before & 0xffffffffu) + (eq_before < eq_budget ? eq_before : eq_budget);
    int64_t eq_left = eq_budget - eq_before;
    if (eq_left < 0) eq_left = 0;
    for (int i = s0; i < s1; ++i) {
        const uint32_t k = keys[i];
        const int w = i == tail_i ? wtail : R;
        const int64_t b = lo + i;
        if (k > thr) {
#pragma unroll
            for (int c = 0; c < w; ++c) out[wpos++] = (int32_t)(b * R + c);
        } else if (k == thr) {
#pragma unroll
            for (int c = 0; c < w && eq_left > 0; ++c, --eq_left)
                out[wpos++] = (int32_t)(b * R + c);
        }
    }
#else
    (void) scores; (void) steps; (void) max_blocks; (void) cap; (void) ids;
    /* unreachable on SYCL: the launcher refuses this device */
#endif
}
#endif  // !__HIPCC__
bool scores_multi_on() {
    static const bool on = [] { const char* v = std::getenv("STRATA_SCORES_MULTI"); return v == nullptr || std::atoi(v) != 0; }();
    return on;
}
// STRATA_QSA_EARLY_EXIT=0 keeps the old staging (every CTA loads the queries first); the scores are the same bits
bool scores_early_exit() {
    static const bool on = [] { const char* v = std::getenv("STRATA_QSA_EARLY_EXIT"); return v == nullptr || std::atoi(v) != 0; }();
    return on;
}
}  // namespace

bool qsa_block_scores_slots(const float* const* pooled, const float* const* dead, const float* q_idx,
                            const int32_t* steps, int64_t nq, int64_t max_blocks, const QsaShapes& s, float* scores,
                            void* stream) {
    if (nq <= 0) return true;
    if (s.idx_dim != IDX_DIM || s.idx_n_head != IDX_HEADS || s.idx_block != R) {
        std::fprintf(stderr, "qsa_block_scores_slots: unsupported indexer geometry\n");
        std::exit(1);
    }
    if (nq > MQ || !scores_multi_on()) return false;   // the single-query call would not run the multi kernel
    ScoreSlotKeys keys{};
    for (int64_t z = 0; z < nq; ++z) {
        keys.pooled[z] = pooled[z];
        keys.dead[z] = dead[z];
    }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->submit([&](sycl::handler &cgh) {
                auto scores_early_exit_ct5 = scores_early_exit();

                cgh.parallel_for<
                    dpct_kernel_name<class block_scores_slots_kernel_9af779>>(
                    sycl::nd_range<3>(sycl::range(1, (unsigned)nq, 256) *
                                          sycl::range(1, 1, SCORE_WARPS * 32),
                                      sycl::range(1, 1, SCORE_WARPS * 32)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            block_scores_slots_kernel(keys, q_idx, steps,
                                                      max_blocks, scores,
                                                      scores_early_exit_ct5);
                        });
            });
    }
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

    return true;
}

// block_scores_multi_kernel with the warp sums as one recursive-halving reduction.  A (query, head) dot is a lane's
// 4-term product summed over the 32 lanes by an xor butterfly (offsets 16, 8, 4, 2, 1): 5 shuffles per value, 20 per
// (block, query) - on Turing the decode scorer's time grew with the window's queries (420K cells: 117 / 265 / 509 us
// for 1 / 3 / 6 queries), the shuffles, not the keys.  Here the V = 4 x queries values (padded to a power of 2) are
// reduced together: at offset o each lane keeps half of its values and adds its partner's copy of them - the pair
// (lane l, lane l ^ o) the butterfly adds for that value at that offset, so every value's additions are the
// butterfly's, operand for operand (float addition commutes): the same bits, with V - 1 shuffles plus the plain
// butterfly's remaining stages instead of 5 V.  Lane l ends with value (l >> (5 - log2 V)); the query's 4 heads are
// then gathered by its head-0 lane and summed in block_scores_multi_kernel's order.  STRATA_SCORES_HALVE=0: the old kernel.
template <int V>
/*
DPCT1110: The total declared local variable size in device function
block_scores_halve_kernel exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
__dpct_inline__ void block_scores_halve_kernel(
    const float *__restrict__ pooled, const float *__restrict__ dead,
    const float *__restrict__ q_idx, const int32_t *__restrict__ steps, int nq,
    int64_t max_blocks, float *__restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int NQ = V / IDX_HEADS; // query slots (nq <= NQ)
    constexpr int LV = V == 4 ? 2 : V == 8 ? 3 : V == 16 ? 4 : 5, SH = 5 - LV;
    static_assert(V == 4 || V == 8 || V == 16 || V == 32, "V");
    auto &qs = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[NQ * IDX_HEADS * IDX_DIM]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_nkv =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int64_t[NQ]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_nbid =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int64_t[NQ]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    if (item_ct1.get_local_id(2) < NQ) {
        s_nkv[item_ct1.get_local_id(2)] =
            item_ct1.get_local_id(2) < nq
                ? steps[item_ct1.get_local_id(2) * kStepCount + kStepNKv]
                : 0;
        s_nbid[item_ct1.get_local_id(2)] =
            item_ct1.get_local_id(2) < nq
                ? steps[item_ct1.get_local_id(2) * kStepCount + kStepNBid]
                : -1;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    int64_t top = 0;
#pragma unroll
    for (int q = 0; q < nq; ++q) top = s_nbid[q] > top ? s_nbid[q] : top;
    if ((int64_t)item_ct1.get_group(2) * SCORE_WARPS >
        top) return; // CTA-uniform (as block_scores_multi_kernel's early exit)
#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < NQ * IDX_HEADS * IDX_DIM;
         i += item_ct1.get_local_range(2))
        qs[i] = i < nq * IDX_HEADS * IDX_DIM ? q_idx[i] : 0.0f;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    const int lane = item_ct1.get_local_id(2) & 31;
    const int64_t wstride = (int64_t)item_ct1.get_group_range(2) * SCORE_WARPS;
    const sycl::float4 kd =
        *reinterpret_cast<const sycl::float4 *>(dead + lane * 4);
    // up to 4 queries: the lane's query values in registers (read once, not once per block from shared memory: the
    // shared-memory reads were the kernel's limit, ncu mio_throttle)
    constexpr bool QREG = V <= 16;
    sycl::float4 qr[QREG ? V : 1];
    if constexpr (QREG) {
#pragma unroll
        for (int i = 0; i < V; ++i)
            qr[i] = *reinterpret_cast<const sycl::float4 *>(
                qs + (i >> 2) * IDX_HEADS * IDX_DIM + (i & 3) * IDX_DIM +
                lane * 4);
    }
    int s_nbid_r[NQ];                               // n_bid < 2^31
#pragma unroll
    for (int qi = 0; qi < NQ; ++qi) s_nbid_r[qi] = (int) s_nbid[qi];
    for (int64_t b = (int64_t)item_ct1.get_group(2) * SCORE_WARPS +
                     (item_ct1.get_local_id(2) >> 5);
         b <= top && b < max_blocks; b += wstride) {
        const sycl::float4 kp = *reinterpret_cast<const sycl::float4 *>(
            pooled + b * IDX_DIM + lane * 4);
        float v[V];
#pragma unroll
        for (int qi = 0; qi < NQ; ++qi) {
            const sycl::float4 k4 = (b == s_nbid_r[qi]) ? kd : kp;
            const float* q = qs + qi * IDX_HEADS * IDX_DIM + lane * 4;
#pragma unroll
            for (int h = 0; h < IDX_HEADS; ++h) {
                sycl::float4 q4;
                if constexpr (QREG) q4 = qr[qi * IDX_HEADS + h];
                else q4 =
                    *reinterpret_cast<const sycl::float4 *>(q + h * IDX_DIM);
                v[qi * IDX_HEADS + h] = k4.x() * q4.x() + k4.y() * q4.y() +
                                        k4.z() * q4.z() + k4.w() * q4.w();
            }
        }
        // recursive halving over offsets 16, 8, ... while more than one value is left, then the plain butterfly
        int cnt = V;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            if (cnt > 1) {
                const int half = cnt >> 1;
                const bool hi = (lane & o) != 0;
#pragma unroll
                for (int i = 0; i < V / 2; ++i) {
                    if (i < half) {
                        const float send = hi ? v[i] : v[i + half];
                        const float keep = hi ? v[i + half] : v[i];
                        /*
                        DPCT1108: '__shfl_xor_sync' was migrated with the
                        experimental feature masked sub_group function which may
                        not be supported by all compilers or runtimes. You may
                        need to adjust the code.
                        */
                        const float recv =
                            dpct::experimental::permute_sub_group_by_xor(
                                0xffffffffu,
                                sycl::ext::oneapi::this_work_item::
                                    get_sub_group(),
                                send, o);
                        v[i] = keep + recv;
                    }
                }
                cnt = half;
            } else {
                /*
                DPCT1108: '__shfl_xor_sync' was migrated with the
                experimental feature masked sub_group function which may not be
                supported by all compilers or runtimes. You may need to adjust
                the code.
                */
                v[0] += dpct::experimental::permute_sub_group_by_xor(
                    0xffffffffu,
                    sycl::ext::oneapi::this_work_item::get_sub_group(), v[0],
                    o);
            }
        }
        // lane l holds value j = l >> SH = qi * 4 + h; the head-0 lane of each query gathers heads 1-3
        const int j = lane >> SH, qi = j >> 2;
        /*
        DPCT1108: '__shfl_sync' was migrated with the experimental feature
        masked sub_group function which may not be supported by all compilers or
        runtimes. You may need to adjust the code.
        */
        const float d1 = dpct::experimental::select_from_sub_group(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
            v[0], (lane + (1 << SH)) & 31);
        /*
        DPCT1108: '__shfl_sync' was migrated with the experimental feature
        masked sub_group function which may not be supported by all compilers or
        runtimes. You may need to adjust the code.
        */
        const float d2 = dpct::experimental::select_from_sub_group(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
            v[0], (lane + (2 << SH)) & 31);
        /*
        DPCT1108: '__shfl_sync' was migrated with the experimental feature
        masked sub_group function which may not be supported by all compilers or
        runtimes. You may need to adjust the code.
        */
        const float d3 = dpct::experimental::select_from_sub_group(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
            v[0], (lane + (3 << SH)) & 31);
        if ((j & 3) == 0 && (lane & ((1 << SH) - 1)) == 0 && qi < nq && b <= s_nbid_r[qi & (NQ - 1)]) {
            const float d[4] = {v[0], d1, d2, d3};
            float score = 0.0f;
#pragma unroll
            for (int h = 0; h < IDX_HEADS; ++h) score += d[h] > 0.0f ? d[h] : 0.0f;
            if (b == s_nbid_r[qi & (NQ - 1)] && s_nkv[qi] % R != 0) score += 1e9f;
            out[qi * max_blocks + b] = score;
        }
    }
}

// #1825: the prompt path's thread-per-head scorer (bitwise the warp kernel's scores).  Default: Turing with
// STRATA_SM75_PERF=1 (the P100 gain measured here is +1..3% at 30-60K tokens, under the 3% bar for a new default);
// STRATA_SCORES_THREAD=1 turns it on, =0 off, on any CUDA card
bool scores_thread_on() {
    static const int env = [] { const char* v = std::getenv("STRATA_SCORES_THREAD"); return v ? (std::atoi(v) != 0 ? 1 : 0) : -1; }();
    if (env >= 0) return env != 0;
    static int ok[64] = {};   // 0 unknown, 1 yes, 2 no
    int dev = 0;
    /*
    DPCT1026: The call to cudaGetLastError was removed because this
    functionality is redundant in SYCL.
    */
    if (DPCT_CHECK_ERROR(dev = dpct::get_current_device_id()) != 0 || dev < 0 ||
        dev >= 64) {
        ; return false;
    }
    if (ok[dev] == 0) {
        int major = 0, minor = 0;
        major = dpct::get_device(dev).get_major_version();
        minor = dpct::get_device(dev).get_minor_version();
        ok[dev] = strata::sm75_default(major, minor) ? 1 : 2;   // Turing, STRATA_SM75_PERF=1
    }
    return ok[dev] == 1;
}

bool scores_halve_on() {
    static const int env = [] { const char* v = std::getenv("STRATA_SCORES_HALVE"); return v ? (std::atoi(v) != 0 ? 1 : 0) : -1; }();
    if (env >= 0) return env != 0;
    static int ok[64] = {};   // 0 unknown, 1 yes, 2 no
    int dev = 0;
    /*
    DPCT1026: The call to cudaGetLastError was removed because this
    functionality is redundant in SYCL.
    */
    if (DPCT_CHECK_ERROR(dev = dpct::get_current_device_id()) != 0 || dev < 0 ||
        dev >= 64) {
        ; return false;
    }
    if (ok[dev] == 0) {
        int major = 0, minor = 0;
        major = dpct::get_device(dev).get_major_version();
        minor = dpct::get_device(dev).get_minor_version();
        ok[dev] = strata::sm75_default(major, minor) ? 1 : 2;   // Turing, STRATA_SM75_PERF=1
    }
    return ok[dev] == 1;
}

void qsa_block_scores(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                      int64_t max_blocks, const QsaShapes& s, float* scores, void* stream, int64_t active_blocks) {
    if (nq <= 0) return;
    if (s.idx_dim != IDX_DIM || s.idx_n_head != IDX_HEADS || s.idx_block != R || nq > 65535) {
        std::fprintf(stderr, "qsa_block_scores: unsupported indexer geometry\n");
        std::exit(1);
    }
    // a block past a query's n_bid returns at once: the grid need only reach the batch's largest n_bid (C-1)
    if (scores_multi_on() && nq <= MQ && active_blocks <= 0) {   // no active count: decode (captured or not) and prefill's pooled16
        // QSAROWS: STRATA_QSA_ROWS=1 (exactly) takes block_scores_rows_kernel for calls of up to 4 queries - the same bits
        static const bool rows_env = [] { const char* v = std::getenv("STRATA_QSA_ROWS"); return v != nullptr && v[0] == '1' && v[1] == '\0'; }();
        if (rows_env && qsa_block_scores_rows(pooled, dead, q_idx, steps, nq, max_blocks, s, scores, stream)) return;
        if (scores_halve_on()) {   // Turing: block_scores_halve_kernel, the same bits
            const dpct::queue_ptr st = strata::q_of(stream);
            if (nq <= 1) {
                auto exp_props = sycl::ext::oneapi::experimental::properties{
                    sycl::ext::oneapi::experimental::use_root_sync};

                st->parallel_for<
                    dpct_kernel_name<class block_scores_halve_kernel_bd1ef2,
                                     dpct_kernel_scalar<4>>>(
                    sycl::nd_range<3>(sycl::range(1, 1, 256) *
                                          sycl::range(1, 1, SCORE_WARPS * 32),
                                      sycl::range(1, 1, SCORE_WARPS * 32)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            block_scores_halve_kernel<4>(pooled, dead, q_idx,
                                                         steps, (int)nq,
                                                         max_blocks, scores);
                        });
            } else if (nq <= 2) {
                auto exp_props = sycl::ext::oneapi::experimental::properties{
                    sycl::ext::oneapi::experimental::use_root_sync};

                st->parallel_for<
                    dpct_kernel_name<class block_scores_halve_kernel_df11e1,
                                     dpct_kernel_scalar<8>>>(
                    sycl::nd_range<3>(sycl::range(1, 1, 256) *
                                          sycl::range(1, 1, SCORE_WARPS * 32),
                                      sycl::range(1, 1, SCORE_WARPS * 32)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            block_scores_halve_kernel<8>(pooled, dead, q_idx,
                                                         steps, (int)nq,
                                                         max_blocks, scores);
                        });
            } else if (nq <= 4) {
                auto exp_props = sycl::ext::oneapi::experimental::properties{
                    sycl::ext::oneapi::experimental::use_root_sync};

                st->parallel_for<
                    dpct_kernel_name<class block_scores_halve_kernel_16679c,
                                     dpct_kernel_scalar<16>>>(
                    sycl::nd_range<3>(sycl::range(1, 1, 256) *
                                          sycl::range(1, 1, SCORE_WARPS * 32),
                                      sycl::range(1, 1, SCORE_WARPS * 32)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            block_scores_halve_kernel<16>(pooled, dead, q_idx,
                                                          steps, (int)nq,
                                                          max_blocks, scores);
                        });
            } else { // 5-8 queries: two calls of up to 4 (the register form;
                     // each query's scores are its own)
                {
                    auto exp_props =
                        sycl::ext::oneapi::experimental::properties{
                            sycl::ext::oneapi::experimental::use_root_sync};

                    st->parallel_for<
                        dpct_kernel_name<class block_scores_halve_kernel_fb1ef4,
                                         dpct_kernel_scalar<16>>>(
                        sycl::nd_range<3>(
                            sycl::range(1, 1, 256) *
                                sycl::range(1, 1, SCORE_WARPS * 32),
                            sycl::range(1, 1, SCORE_WARPS * 32)),
                        exp_props,
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(32)]] {
                                block_scores_halve_kernel<16>(
                                    pooled, dead, q_idx, steps, 4, max_blocks,
                                    scores);
                            });
                }
                const int rest = (int) nq - 4;
                const float* q2 = q_idx + 4 * IDX_HEADS * IDX_DIM;
                const int32_t* st2 = steps + 4 * kStepCount;
                float* sc2 = scores + 4 * max_blocks;
                if (rest <= 1) {
                    auto exp_props =
                        sycl::ext::oneapi::experimental::properties{
                            sycl::ext::oneapi::experimental::use_root_sync};

                    st->parallel_for<
                        dpct_kernel_name<class block_scores_halve_kernel_701dee,
                                         dpct_kernel_scalar<4>>>(
                        sycl::nd_range<3>(
                            sycl::range(1, 1, 256) *
                                sycl::range(1, 1, SCORE_WARPS * 32),
                            sycl::range(1, 1, SCORE_WARPS * 32)),
                        exp_props,
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(32)]] {
                                block_scores_halve_kernel<4>(pooled, dead, q2,
                                                             st2, rest,
                                                             max_blocks, sc2);
                            });
                } else if (rest <= 2) {
                    auto exp_props =
                        sycl::ext::oneapi::experimental::properties{
                            sycl::ext::oneapi::experimental::use_root_sync};

                    st->parallel_for<
                        dpct_kernel_name<class block_scores_halve_kernel_8c63c4,
                                         dpct_kernel_scalar<8>>>(
                        sycl::nd_range<3>(
                            sycl::range(1, 1, 256) *
                                sycl::range(1, 1, SCORE_WARPS * 32),
                            sycl::range(1, 1, SCORE_WARPS * 32)),
                        exp_props,
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(32)]] {
                                block_scores_halve_kernel<8>(pooled, dead, q2,
                                                             st2, rest,
                                                             max_blocks, sc2);
                            });
                } else {
                    auto exp_props =
                        sycl::ext::oneapi::experimental::properties{
                            sycl::ext::oneapi::experimental::use_root_sync};

                    st->parallel_for<
                        dpct_kernel_name<class block_scores_halve_kernel_a36aa3,
                                         dpct_kernel_scalar<16>>>(
                        sycl::nd_range<3>(
                            sycl::range(1, 1, 256) *
                                sycl::range(1, 1, SCORE_WARPS * 32),
                            sycl::range(1, 1, SCORE_WARPS * 32)),
                        exp_props,
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(32)]] {
                                block_scores_halve_kernel<16>(pooled, dead, q2,
                                                              st2, rest,
                                                              max_blocks, sc2);
                            });
                }
            }
            /*
            DPCT1010: SYCL uses exceptions to report errors and does not
            use the error codes. The cudaGetLastError function call was replaced
            with 0. You need to rewrite this code.
            */
            const dpct::err0 e = 0;
            /*
            DPCT1009: SYCL reports errors using exceptions and does not use
            error codes. Please replace the "get_error_string_dummy(...)" with a
            real error-handling function.
            */

            return;
        }
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            strata::q_of(stream)
                ->submit([&](sycl::handler &cgh) {
                    auto scores_early_exit_ct7 = scores_early_exit();

                    cgh.parallel_for<dpct_kernel_name<
                        class block_scores_multi_kernel_a5908d>>(
                        sycl::nd_range<3>(
                            sycl::range(1, 1, 256) *
                                sycl::range(1, 1, SCORE_WARPS * 32),
                            sycl::range(1, 1, SCORE_WARPS * 32)),
                        exp_props,
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(32)]] {
                                block_scores_multi_kernel(
                                    pooled, dead, q_idx, steps, (int)nq,
                                    max_blocks, scores, scores_early_exit_ct7,
                                    {});
                            });
                });
        }
        /*
        DPCT1010: SYCL uses exceptions to report errors and does not use
        the error codes. The cudaGetLastError function call was replaced with 0.
        You need to rewrite this code.
        */
        const dpct::err0 e = 0;
        /*
        DPCT1009: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */

        return;
    }
    const int64_t reach = active_blocks > 0 && active_blocks < max_blocks ? active_blocks : max_blocks;
#if defined(__HIPCC__)
    // STRATA_SCORES_THREAD: 1 = on (gfx1100, gfx12), 0 = off, unset = on for gfx12 only (gfx1100: off, nothing measured here)
    static const int thread_env = [] {
        const char* value = std::getenv("STRATA_SCORES_THREAD");
        return value == nullptr ? -1 : (std::atoi(value) == 1 ? 1 : 0);
    }();
    // The bound is the ACTUAL live extent. A reserved 256K context never enables this by itself.
    if (thread_env != 0 && active_blocks >= 16385 && active_blocks <= max_blocks && nq >= 64 &&
        (scores_thread_arch() == 2 || (thread_env == 1 && scores_thread_arch() == 1))) {
        static thread_local bool logged = false;
        if (!logged && (thread_env == -1 || std::getenv("STRATA_SCORES_THREAD_TRACE"))) {
            std::fprintf(stderr, "qsa exact thread: the prompt's block scorer runs as the four-lanes-per-query kernel (nq=%lld active_blocks=%lld "
                                 "capacity=%lld; the same scores bit for bit; STRATA_SCORES_THREAD=0 turns it off)\n",
                         (long long)nq, (long long)active_blocks, (long long)max_blocks);
            logged = true;
        }
        const dim3 tgrid((unsigned)((reach + TS_BR - 1) / TS_BR), (unsigned)((nq + TS_Q - 1) / TS_Q));
        block_scores_thread_kernel<<<tgrid, 256, 0, (cudaStream_t)stream>>>(pooled, q_idx, steps, nq,
                                                                         max_blocks, reach, scores);
        block_scores_tail_kernel<<<(unsigned)nq, 32, 0, (cudaStream_t)stream>>>(dead, q_idx, steps,
                                                                            max_blocks, scores);
        const cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::fprintf(stderr, "qsa exact thread: %s\n", cudaGetErrorString(error));
            std::exit(1);
        }
        return;
    }
#else
    // prompt calls (an active-block count): the thread-per-head kernel, the same bits (opt-in: scores_thread_on); else the
    // warp kernel below
    if (active_blocks > 0 && scores_thread_on()) {
        const dpct::dim3 tgrid((unsigned)((reach + TS_BR - 1) / TS_BR),
                               (unsigned)((nq + TS_Q - 1) / TS_Q));
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            strata::q_of(stream)
                ->parallel_for<
                    dpct_kernel_name<class block_scores_thread_kernel_952779>>(
                    sycl::nd_range<3>(tgrid * sycl::range(1, 1, 256),
                                      sycl::range(1, 1, 256)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            block_scores_thread_kernel(pooled, q_idx, steps, nq,
                                                       max_blocks, reach,
                                                       scores);
                        });
        }
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            strata::q_of(stream)
                ->parallel_for<
                    dpct_kernel_name<class block_scores_tail_kernel_2831df>>(
                    sycl::nd_range<3>(sycl::range(1, 1, (unsigned)nq) *
                                          sycl::range(1, 1, 32),
                                      sycl::range(1, 1, 32)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            block_scores_tail_kernel(dead, q_idx, steps,
                                                     max_blocks, scores);
                        });
        }
        /*
        DPCT1010: SYCL uses exceptions to report errors and does not use
        the error codes. The cudaGetLastError function call was replaced with 0.
        You need to rewrite this code.
        */
        const dpct::err0 e = 0;
        /*
        DPCT1009: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */

        return;
    }
#endif
    const dpct::dim3 grid((unsigned)((reach + SCORE_WARPS - 1) / SCORE_WARPS),
                          (unsigned)nq);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class block_scores_kernel_756a1c>>(
                sycl::nd_range<3>(grid * sycl::range(1, 1, SCORE_WARPS * 32),
                                  sycl::range(1, 1, SCORE_WARPS * 32)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        block_scores_kernel(pooled, dead, q_idx, steps,
                                            max_blocks, scores);
                    });
    }
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

bool qsa_block_scores_rows(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                           int64_t max_blocks, const QsaShapes& s, float* scores, void* stream) {
    if (nq <= 0) return true;
    if (nq > 4 || s.idx_dim != IDX_DIM || s.idx_n_head != IDX_HEADS || s.idx_block != R) return false;
    dpct::queue_ptr st = strata::q_of(stream);
    const int n = (int) nq;
    if (n == 1) {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        st->parallel_for<dpct_kernel_name<class block_scores_rows_kernel_9d3a36,
                                          dpct_kernel_scalar<1>>>(
            sycl::nd_range<3>(sycl::range(1, 1, 256) *
                                  sycl::range(1, 1, SCORE_WARPS * 32),
                              sycl::range(1, 1, SCORE_WARPS * 32)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                block_scores_rows_kernel<1>(pooled, dead, q_idx, steps, n,
                                            max_blocks, scores);
            });
    } else if (n == 2) {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        st->parallel_for<dpct_kernel_name<class block_scores_rows_kernel_696e37,
                                          dpct_kernel_scalar<2>>>(
            sycl::nd_range<3>(sycl::range(1, 1, 256) *
                                  sycl::range(1, 1, SCORE_WARPS * 32),
                              sycl::range(1, 1, SCORE_WARPS * 32)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                block_scores_rows_kernel<2>(pooled, dead, q_idx, steps, n,
                                            max_blocks, scores);
            });
    } else if (n == 3) {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        st->parallel_for<dpct_kernel_name<class block_scores_rows_kernel_99f2cd,
                                          dpct_kernel_scalar<3>>>(
            sycl::nd_range<3>(sycl::range(1, 1, 256) *
                                  sycl::range(1, 1, SCORE_WARPS * 32),
                              sycl::range(1, 1, SCORE_WARPS * 32)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                block_scores_rows_kernel<3>(pooled, dead, q_idx, steps, n,
                                            max_blocks, scores);
            });
    } else {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        st->parallel_for<dpct_kernel_name<class block_scores_rows_kernel_7a8162,
                                          dpct_kernel_scalar<4>>>(
            sycl::nd_range<3>(sycl::range(1, 1, 256) *
                                  sycl::range(1, 1, SCORE_WARPS * 32),
                              sycl::range(1, 1, SCORE_WARPS * 32)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                block_scores_rows_kernel<4>(pooled, dead, q_idx, steps, n,
                                            max_blocks, scores);
            });
    }
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

    return true;
}

void qsa_block_scores_rows(const QsaScoreRows& rows, const float* q_idx, const int32_t* steps, int64_t nq,
                           int64_t max_blocks, const QsaShapes& s, float* scores, void* stream) {
    if (nq <= 0) return;
    if (s.idx_dim != IDX_DIM || s.idx_n_head != IDX_HEADS || s.idx_block != R || nq > MQ || rows.n != nq) {
        std::fprintf(stderr, "qsa_block_scores_rows: unsupported geometry\n");
        std::exit(1);
    }
    static const bool early = [] { const char* v = std::getenv("STRATA_QSA_EARLY_EXIT"); return v == nullptr || std::atoi(v) != 0; }();
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->submit([&](sycl::handler &cgh) {
                auto early_ct7 = early;

                cgh.parallel_for<
                    dpct_kernel_name<class block_scores_multi_kernel_1489d0>>(
                    sycl::nd_range<3>(sycl::range(1, 1, 256) *
                                          sycl::range(1, 1, SCORE_WARPS * 32),
                                      sycl::range(1, 1, SCORE_WARPS * 32)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            block_scores_multi_kernel(
                                nullptr, nullptr, q_idx, steps, (int)nq,
                                max_blocks, scores, early_ct7, rows);
                        });
            });
    }
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

bool qsa_block_scores_tc(const float *pooled, const float *dead,
                         const float *q_idx, const int32_t *steps, int64_t nq,
                         int64_t max_blocks, const QsaShapes &s, float *scores,
                         void *stream, int64_t active_blocks) try {
    if (nq <= 0) return true;
    if (s.idx_dim != IDX_DIM || s.idx_n_head != IDX_HEADS || s.idx_block != R || nq > 65535 * TC_QT) return false;
#if defined(__HIPCC__)
    // AMD: the gfx12 (RDNA4) WMMA scorer, opt-in (STRATA_SELECT_WMMA=1): it selects slightly differently from the warp
    // kernel (254/256 queries the same), so the default keeps the warp kernel; every other target keeps it too (false)
    static const bool wmma_on = [] {
        const char* v = std::getenv("STRATA_SELECT_WMMA");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    // any HIP card: the FP32 tiled scorer, opt-in as on CUDA (STRATA_SELECT_SIMT=1; another summation order than the
    // warp kernel, so the default keeps the warp kernel).  RX 6900 XT, 256 queries at 128K: 2.71 -> 0.587 ms
    static const bool simt_on = [] {
        const char* v = std::getenv("STRATA_SELECT_SIMT");
        return v != nullptr && v[0] == '1';
    }();
    if (simt_on && !(wmma_on && sel_gfx12_device())) {
        const int64_t reach = active_blocks > 0 && active_blocks < max_blocks ? active_blocks : max_blocks;
        const bool rdna2 = sel_gfx103_device();
        static std::atomic<bool> said{false};
        if (!said.exchange(true))
            std::fprintf(stderr, "qsa select: FP32 tiled scorer (STRATA_SELECT_SIMT=1), tiles %s\n",
                         rdna2 ? "32/128/16/8 (gfx103x)" : "32/64/32/4");
        return block_scores_simt(pooled, dead, q_idx, steps, nq, max_blocks, reach, scores, (cudaStream_t) stream, rdna2);
    }
    if (!wmma_on || !sel_gfx12_device()) return false;
    {
        const int64_t reach = active_blocks > 0 && active_blocks < max_blocks ? active_blocks : max_blocks;
        const int64_t per = (int64_t) 4 * WITER * 16;
        const dim3 grid((unsigned) ((reach + per - 1) / per), (unsigned) ((nq + WQT - 1) / WQT));
        block_scores_wmma_kernel<<<grid, 128, 0, (cudaStream_t) stream>>>(pooled, q_idx, steps, nq, max_blocks, reach, scores);
        block_scores_tail_kernel<<<(unsigned) nq, 32, 0, (cudaStream_t) stream>>>(dead, q_idx, steps, max_blocks, scores);
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_scores_tc: %s\n", cudaGetErrorString(e)); std::exit(1); }
        return true;
    }
#else
    {   // sm_80 or newer (TF32 MMA); an older card keeps the warp kernel
        static int cc_major[64] = {};
        int dev = 0;
        /*
        DPCT1026: The call to cudaGetLastError was removed because this
        functionality is redundant in SYCL.
        */
        if (DPCT_CHECK_ERROR(dev = dpct::get_current_device_id()) != 0 ||
            dev < 0 || dev >= 64) {
            ; return false;
        }
        if (cc_major[dev] == 0) {
            int major = 0;
            if (DPCT_CHECK_ERROR(
                    major = dpct::get_device(dev).get_major_version()) != 0) {
                /*
                DPCT1026: The call to cudaGetLastError was removed because
                this functionality is redundant in SYCL.
                */
                return false;
            }
            // STRATA_QSA_WARP=1|select (an A/B arm): the pre-sm_80 kernels on any card, as RTX 20 runs them
            const char* w = std::getenv("STRATA_QSA_WARP");
            (void) w; (void) major;
            cc_major[dev] = 7;   // SYCL port: no TF32 MMA - the pre-sm_80 branch (the warp kernel, or STRATA_SELECT_SIMT's tiled one)
        }
        if (cc_major[dev] < 8) {
            // no TF32 tensor cores: the FP32 tiled kernel, opt-in (STRATA_SELECT_SIMT=1): its sum order differs from the
            // warp kernel's, so the default stays the caller's warp kernel and the output bytes stay the same
            // STRATA_SELECT_SIMT=2: block_scores_simt2_kernel (the same bits as =1, larger tiles)
            static const int simt = [] {
                const char* v = std::getenv("STRATA_SELECT_SIMT");
                return v == nullptr ? 0 : v[0] == '1' ? 1 : v[0] == '2' ? 2 : 0;
            }();
            if (!simt) return false;
            const int64_t reach = active_blocks > 0 && active_blocks < max_blocks ? active_blocks : max_blocks;
            if (simt == 2) {
                // STRATA_SIMT2_TILE=2: 64 x 64 tiles (measured slower: 400K 3.00 vs 2.82 ms)
                static const int tile = [] { const char* v = std::getenv("STRATA_SIMT2_TILE"); return v ? std::atoi(v) : 1; }();
                if (tile != 2) {
                    const dpct::dim3 grid((unsigned)((reach + 127) / 128),
                                          (unsigned)((nq + 31) / 32));
                    if (grid.y > 65535) return false;
                    {
                        auto exp_props =
                            sycl::ext::oneapi::experimental::properties{
                                sycl::ext::oneapi::experimental::use_root_sync};

                        strata::q_of(stream)
                            ->parallel_for<dpct_kernel_name<
                                class block_scores_simt2_kernel_e66788,
                                dpct_kernel_scalar<32>,
                                dpct_kernel_scalar<128>>>(
                                sycl::nd_range<3>(grid * sycl::range(1, 1, 256),
                                                  sycl::range(1, 1, 256)),
                                exp_props, [=](sycl::nd_item<3> item_ct1) {
                                    block_scores_simt2_kernel<32, 128>(
                                        pooled, q_idx, steps, nq, max_blocks,
                                        reach, scores);
                                });
                    }
                } else {
                    const dpct::dim3 grid((unsigned)((reach + 63) / 64),
                                          (unsigned)((nq + 63) / 64));
                    if (grid.y > 65535) return false;
                    {
                        auto exp_props =
                            sycl::ext::oneapi::experimental::properties{
                                sycl::ext::oneapi::experimental::use_root_sync};

                        strata::q_of(stream)
                            ->parallel_for<dpct_kernel_name<
                                class block_scores_simt2_kernel_122814,
                                dpct_kernel_scalar<64>,
                                dpct_kernel_scalar<64>>>(
                                sycl::nd_range<3>(grid * sycl::range(1, 1, 256),
                                                  sycl::range(1, 1, 256)),
                                exp_props, [=](sycl::nd_item<3> item_ct1) {
                                    block_scores_simt2_kernel<64, 64>(
                                        pooled, q_idx, steps, nq, max_blocks,
                                        reach, scores);
                                });
                    }
                }
                {
                    auto exp_props =
                        sycl::ext::oneapi::experimental::properties{
                            sycl::ext::oneapi::experimental::use_root_sync};

                    strata::q_of(stream)
                        ->parallel_for<dpct_kernel_name<
                            class block_scores_tail_kernel_52ba40>>(
                            sycl::nd_range<3>(sycl::range(1, 1, (unsigned)nq) *
                                                  sycl::range(1, 1, 32),
                                              sycl::range(1, 1, 32)),
                            exp_props,
                            [=](sycl::nd_item<3> item_ct1)
                                [[sycl::reqd_sub_group_size(32)]] {
                                    block_scores_tail_kernel(
                                        dead, q_idx, steps, max_blocks, scores);
                                });
                }
                /*
                DPCT1010: SYCL uses exceptions to report errors and does
                not use the error codes. The cudaGetLastError function call was
                replaced with 0. You need to rewrite this code.
                */
                const dpct::err0 e = 0;
                /*
                DPCT1009: SYCL reports errors using exceptions and does not
                use error codes. Please replace the
                "get_error_string_dummy(...)" with a real error-handling
                function.
                */

                return true;
            }
            return block_scores_simt(pooled, dead, q_idx, steps, nq, max_blocks,
                                     reach, scores, strata::q_of(stream),
                                     false);
        }
    }
    static bool attr[64] = {};   // the shared-memory opt-in is per device (a layer split runs it on several)
    int adev = 0;
    adev = dpct::get_current_device_id();
    /*
    DPCT1083: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    const int bytes = (TC_QT * TC_QS + TC_NB * TC_KS) * (int)sizeof(float);
    if (!attr[adev]) {
        /*
        DPCT1027: The call to cudaFuncSetAttribute was replaced with 0
        because SYCL currently does not support corresponding setting.
        */
        if (0 != 0) {
            /*
            DPCT1026: The call to cudaGetLastError was removed because this
            functionality is redundant in SYCL.
            */
            return false;
        }
        attr[adev] = true;
    }
    const int64_t reach = active_blocks > 0 && active_blocks < max_blocks ? active_blocks : max_blocks;
    const int64_t per = (int64_t) TC_NB * TC_ITER;
    const dpct::dim3 grid((unsigned)((reach + per - 1) / per),
                          (unsigned)((nq + TC_QT - 1) / TC_QT));
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range(bytes), cgh);

                cgh.parallel_for<
                    dpct_kernel_name<class block_scores_tc_kernel_f4f1fd>>(
                    sycl::nd_range<3>(grid * sycl::range(1, 1, 128),
                                      sycl::range(1, 1, 128)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        block_scores_tc_kernel(
                            pooled, q_idx, steps, nq, max_blocks, reach, scores,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
            });
    }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<
                dpct_kernel_name<class block_scores_tail_kernel_9ade95>>(
                sycl::nd_range<3>(sycl::range(1, 1, (unsigned)nq) *
                                      sycl::range(1, 1, 32),
                                  sycl::range(1, 1, 32)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        block_scores_tail_kernel(dead, q_idx, steps, max_blocks,
                                                 scores);
                    });
    }
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

    return true;
#endif
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void qsa_block_topk_ref(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                        const QsaShapes& s, int32_t* ids, void* stream) {
    if (nq <= 0) return;
    if (s.idx_block != R || cap < qsa_selection_width(kTopkMaxCells, s)) {
        std::fprintf(stderr, "qsa_block_topk: unsupported geometry or cap\n");
        std::exit(1);
    }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class block_topk_kernel_79eb96>>(
                sycl::nd_range<3>(sycl::range(1, 1, (unsigned)nq) *
                                      sycl::range(1, 1, TOPK_T),
                                  sycl::range(1, 1, TOPK_T)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    block_topk_kernel(scores, steps, max_blocks, cap, ids);
                });
    }
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

#if !defined(__HIPCC__)
// Only Turing has a retained model measurement for this CUDA dispatch. Other CUDA devices keep the capacity rule
// (the RTX 5070 regression below). Cache the properties per calling thread; layer-split device switches are checked.
static bool topk_active_turing_device() try {
    int dev = 0;
    if (DPCT_CHECK_ERROR(dev = dpct::get_current_device_id()) !=
        0) return false;
    static thread_local int cached_device = -1;
    static thread_local bool turing = false;
    if (dev != cached_device) {
        dpct::device_info prop{};
        if (DPCT_CHECK_ERROR(dpct::get_device(dev).get_device_info(prop)) !=
            0) return false;
        /*
        DPCT1005: The SYCL device version is different from CUDA Compute
        Compatibility. You may need to rewrite this code.
        */
        turing = prop.get_major_version() == 7 && prop.get_minor_version() == 5;
        cached_device = dev;
    }
    return turing;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
#endif

bool qsa_block_topk_cluster(const float *scores, const int32_t *steps,
                            int64_t nq, int64_t max_blocks, int64_t cap,
                            const QsaShapes &s, int32_t *ids,
                            void *stream) try {
#if 1   // SYCL port: no thread-block clusters (sm_90): the caller takes the plain top-k
    (void) scores; (void) steps; (void) nq; (void) max_blocks; (void) cap; (void) s; (void) ids; (void) stream;
    return false;
#else
    if (nq <= 0) return true;
    if (s.idx_block != R || cap < qsa_selection_width(kTopkMaxCells, s) || nq > 65535 || max_blocks <= 0) return false;
    const size_t smem = (size_t) ((max_blocks + CL_N - 1) / CL_N) * sizeof(uint32_t);
    // Per device (a layer split runs on several): 1 the cluster kernel runs here, 2 it does not. It needs sm_90+ (the
    // card's, or STRATA_EMULATE_CC's) AND code built for it: a build with only older code JIT-compiles their PTX, whose
    // copy of this kernel is a trap - the function's PTX version says which. `opt`: the dynamic shared memory opted in.
    static int ok[64] = {};
    static size_t opt[64] = {};
    int dev = 0;
    /*
    DPCT1026: The call to cudaGetLastError was removed because this
    functionality is redundant in SYCL.
    */
    if (DPCT_CHECK_ERROR(dev = dpct::get_current_device_id()) != 0 || dev < 0 ||
        dev >= 64) {
        ; return false;
    }
    if (ok[dev] == 0) {
        int major = 0;
        dpct::kernel_function_info fa{};
        const bool code =
            DPCT_CHECK_ERROR(
                major = dpct::get_device(dev).get_major_version()) == 0 &&
            strata::cc_major_of(major) >= 9 &&
            DPCT_CHECK_ERROR(dpct::get_kernel_function_info(
                &fa, (const void *)block_topk_cluster_kernel)) == 0 &&
            fa.ptxVersion >= 90 && fa.binaryVersion >= 90;
        /*
        DPCT1026: The call to cudaGetLastError was removed because this
        functionality is redundant in SYCL.
        */
        ok[dev] = code ? 1 : 2;
    }
    if (ok[dev] != 1) return false;
    if (smem > opt[dev]) {   // a larger capacity: opt in, and check that a cluster of CL_N such CTAs can be resident
        int clusters = 0;
        cudaLaunchConfig_t q{};
        cudaLaunchAttribute qa[1];
        qa[0].id = cudaLaunchAttributeClusterDimension;
        qa[0].val.clusterDim.x = CL_N;
        qa[0].val.clusterDim.y = 1;
        qa[0].val.clusterDim.z = 1;
        q.gridDim = dpct::dim3(CL_N, 1, 1);
        q.blockDim = dpct::dim3(CL_T, 1, 1);
        q.dynamicSmemBytes = smem;
        q.attrs = qa;
        q.numAttrs = 1;
        /*
        DPCT1027: The call to cudaFuncSetAttribute was replaced with 0
        because SYCL currently does not support corresponding setting.
        */
        if (0 != 0 ||
            /*
            DPCT1007: Migration of cudaOccupancyMaxActiveClusters is not
            supported.
            */
            cudaOccupancyMaxActiveClusters(&clusters, block_topk_cluster_kernel,
                                           &q) != 0 ||
            clusters < 1) {
            /*
            DPCT1026: The call to cudaGetLastError was removed because this
            functionality is redundant in SYCL.
            */
            return false; // this capacity takes the one-CTA kernels; a smaller
                          // one may still fit
        }
        opt[dev] = smem;
    }
    cudaLaunchConfig_t cfg{};
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = CL_N;
    at[0].val.clusterDim.y = 1;
    at[0].val.clusterDim.z = 1;
    cfg.gridDim = dpct::dim3(CL_N, (unsigned)nq, 1);
    cfg.blockDim = dpct::dim3(CL_T, 1, 1);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = strata::q_of(stream);
    cfg.attrs = at;
    cfg.numAttrs = 1;
    const dpct::err0 e = cudaLaunchKernelEx(
        &cfg, block_topk_cluster_kernel, scores, steps, max_blocks, cap, ids);
    /*
    DPCT1009: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */

    return true;
#endif
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

#if !defined(__HIPCC__)
// ---- the decode top-k split over many CTAs, for cards without thread-block clusters (Turing: a decode window's 1-6
// queries ran block_topk_wide_kernel on 1-6 SMs, ~230 us a call at 420K cells, RTX 2080 Ti).  A query's blocks are
// cut into chunks of SP_CH; six launches: four radix passes (each chunk adds its 8-bit histogram to the query's global
// one), a count of each chunk's cells above / at the threshold, and the emit.  Each pass's CTAs take the previous
// digit from the complete histogram with block_topk_kernel's rule (the largest digit whose cells at or above it reach
// the width; 0 when none) - pure functions of integer counts, so every CTA finds the same digit, and the threshold,
// `above` and the emitted cells are block_topk_kernel's: identical ids.  STRATA_TOPK_SPLIT=0 keeps the wide kernel.
constexpr int SP_T = 256, SP_W = SP_T / 32, SP_CH = 2048, SP_MAXQ = 16, SP_MAXCH = 512;
inline dpct::global_memory<int, 3> g_sp_hist(SP_MAXQ, 4, 256);
inline dpct::global_memory<int, 3>
    g_sp_state(5, SP_MAXQ, 2); // after p digits: prefix, above
inline dpct::global_memory<int, 3>
    g_sp_cnt(SP_MAXQ, SP_MAXCH,
             2); // per chunk: cells above the threshold, cells at it

// the digit of pass `p` from its complete histogram: block_topk_kernel's scan as a parallel suffix sum
inline void sp_digit(const int *__restrict__ h, int width, int above,
                     int &digit, int &above_out, int *s_buf) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_local_id(2);
    // s_buf[d] = sum of h[d'] for d' > d (exclusive suffix), d in 0..255
    const int v = h[t];
    int x = v;                                     // inclusive suffix scan over t = 255..0: reverse index r = 255 - t
    const int lane = t & 31, w = t >> 5;
    /*
    DPCT1115: The sycl::ext::oneapi::group_local_memory_for_overwrite is
    used to allocate group-local memory at the none kernel functor scope of a
    work-group data parallel kernel. You may need to adjust the code.
    */
    auto &s_w = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[SP_W]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    // scan in reverse order: thread t holds element r = 255 - t
    int e = h[255 - t];
    /*
    DPCT1108: '__shfl_up_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    for (int o = 1; o < 32; o <<= 1) {
        const int y = dpct::experimental::shift_sub_group_right(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), e,
            o);
        if (lane >= o) e += y;
    }
    if (lane == 31) s_w[w] = e;
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    int add = 0;
#pragma unroll
    for (int i = 0; i < w; ++i) add += s_w[i];
    e += add;                                      // inclusive suffix sum of h from 255 down to r = 255 - t
    (void) x; (void) v;
    s_buf[255 - t] = e;                            // s_buf[r] = sum_{d >= r} h[d]
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // the scan picks the largest d in 1..255 with above + sum_{d' >= d} h >= width, else 0
    /*
    DPCT1115: The sycl::ext::oneapi::group_local_memory_for_overwrite is
    used to allocate group-local memory at the none kernel functor scope of a
    work-group data parallel kernel. You may need to adjust the code.
    */
    auto &s_best = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    if (t == 0) s_best = 0;
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (t >= 1 && above + s_buf[t] >= width)
        dpct::atomic_fetch_max<sycl::access::address_space::generic_space>(
            &s_best, t);
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const int d = s_best;
    digit = d;
    above_out = above + (d == 255 ? 0 : s_buf[d + 1]);   // cells strictly above digit d (the scan's cum)
    if (d == 0) above_out = above + s_buf[1];
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
}

template <int P>
/*
DPCT1110: The total declared local variable size in device function
topk_split_hist_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void
topk_split_hist_kernel(const float *__restrict__ scores,
                       const int32_t *__restrict__ steps, int64_t max_blocks,
                       dpct::accessor<int, dpct::global, 3> g_sp_hist,
                       dpct::accessor<int, dpct::global, 3> g_sp_state) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &hist =
    *sycl::ext::oneapi::group_local_memory_for_overwrite<int[SP_W][256]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_buf =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[256]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int q = item_ct1.get_group(1), t = item_ct1.get_local_id(2),
              lane = t & 31, warp = t >> 5;
    const int32_t* st = steps + (int64_t) q * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    if (n_kv <= width) return;
    const int64_t nb = n_bid + 1, b0 = (int64_t)item_ct1.get_group(2) * SP_CH;
    uint32_t prefix = 0;
    int above = 0;
    if (P > 0) {
        const int pa = g_sp_state[P - 1][q][0], ab = g_sp_state[P - 1][q][1];
        int d, ab2;
        sp_digit(g_sp_hist[q][P - 1], (int) width, ab, d, ab2, s_buf);
        prefix = (uint32_t) pa | ((uint32_t) d << (24 - 8 * (P - 1)));
        above = ab2;
        if (item_ct1.get_group(2) == 0 && t == 0) {
            g_sp_state[P][q][0] = (int)prefix; g_sp_state[P][q][1] = above;
        }
    } else if (item_ct1.get_group(2) == 0 && t == 0) {
        g_sp_state[0][q][0] = 0; g_sp_state[0][q][1] = 0;
    }
    if (P == 4 || b0 >= nb) return;
    constexpr int shift = 24 - 8 * (P < 4 ? P : 0);
    const uint32_t hi_mask = P == 0 ? 0u : (0xffffffffu << (shift + 8));
#pragma unroll
    for (int i = lane; i < 256; i += 32) hist[warp][i] = 0;
    sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
    const float* sc = scores + (int64_t) q * max_blocks;
    const int64_t b1 = b0 + SP_CH < nb ? b0 + SP_CH : nb;
    for (int64_t b = b0 + t; b < b1; b += SP_T) {
        const int w = b < n_bid ? R : (int) (n_kv - n_bid * R);
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if ((k & hi_mask) == (prefix & hi_mask))
            dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                &hist[warp][(k >> shift) & 255], w);
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    int s = 0;
#pragma unroll
    for (int w2 = 0; w2 < SP_W; ++w2) s += hist[w2][t];
    if (s) dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        &g_sp_hist[q][P][t], s);
}

// the threshold from pass 3's histogram, then each chunk's cells above / at it
/*
DPCT1110: The total declared local variable size in device function
topk_split_count_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void
topk_split_count_kernel(const float *__restrict__ scores,
                        const int32_t *__restrict__ steps, int64_t max_blocks,
                        dpct::accessor<int, dpct::global, 3> g_sp_hist,
                        dpct::accessor<int, dpct::global, 3> g_sp_state,
                        dpct::accessor<int, dpct::global, 3> g_sp_cnt) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &s_buf = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[256]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_r =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[2][SP_W]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int q = item_ct1.get_group(1), t = item_ct1.get_local_id(2),
              lane = t & 31, warp = t >> 5;
    const int32_t* st = steps + (int64_t) q * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    if (n_kv <= width) return;
    const int pa = g_sp_state[3][q][0], ab = g_sp_state[3][q][1];
    int d, ab2;
    sp_digit(g_sp_hist[q][3], (int) width, ab, d, ab2, s_buf);
    const uint32_t thr = (uint32_t) pa | (uint32_t) d;
    if (item_ct1.get_group(2) == 0 && t == 0) {
        g_sp_state[4][q][0] = (int)thr; g_sp_state[4][q][1] = ab2;
    }
    const int64_t nb = n_bid + 1, b0 = (int64_t)item_ct1.get_group(2) * SP_CH;
    const int64_t b1 = b0 + SP_CH < nb ? b0 + SP_CH : nb;
    const float* sc = scores + (int64_t) q * max_blocks;
    int gt = 0, eq = 0;
    for (int64_t b = b0 + t; b < b1; b += SP_T) {
        const int w = b < n_bid ? R : (int) (n_kv - n_bid * R);
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if (k > thr) gt += w;
        else if (k == thr) eq += w;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        /*
        DPCT1108: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        gt += dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), gt,
            o);
        /*
        DPCT1108: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        eq += dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), eq,
            o);
    }
    if (lane == 0) { s_r[0][warp] = gt; s_r[1][warp] = eq; }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (t == 0) {
        int g2 = 0, e2 = 0;
#pragma unroll
        for (int w2 = 0; w2 < SP_W; ++w2) {
            g2 += s_r[0][w2]; e2 += s_r[1][w2];
        }
        g_sp_cnt[q][item_ct1.get_group(2)][0] = g2;
        g_sp_cnt[q][item_ct1.get_group(2)][1] = e2;
    }
}

// the emit: block_topk_wide_kernel's within a chunk, offset by the chunks before it
/*
DPCT1110: The total declared local variable size in device function
topk_split_emit_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void
topk_split_emit_kernel(const float *__restrict__ scores,
                       const int32_t *__restrict__ steps, int64_t max_blocks,
                       int64_t cap, int32_t *__restrict__ ids,
                       dpct::accessor<int, dpct::global, 3> g_sp_state,
                       dpct::accessor<int, dpct::global, 3> g_sp_cnt) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &s_gt = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[SP_W]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_eq =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[SP_W]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_sel0 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_eqb0 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int q = item_ct1.get_group(1), t = item_ct1.get_local_id(2),
              lane = t & 31, warp = t >> 5;
    const int32_t* st = steps + (int64_t) q * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + (int64_t) q * cap;
    const int64_t nb = n_bid + 1, b0 = (int64_t)item_ct1.get_group(2) * SP_CH;
    if (n_kv <= width) {                               // everything is selected: the identity, ascending
        const int64_t c0 = b0 * R, c1 = (b0 + SP_CH) * R < n_kv ? (b0 + SP_CH) * R : n_kv;
#pragma unroll
        for (int64_t j = c0 + t; j < c1; j += SP_T) out[j] = (int32_t)j;
        return;
    }
    if (b0 >= nb) return;
    const uint32_t thr = (uint32_t) g_sp_state[4][q][0];
    const int eq_budget = (int) (width - g_sp_state[4][q][1]);
    if (t == 0) {                                      // the cells the chunks before this one select / hold at thr
        int sel = 0, eqb = 0;
        for (int c = 0; c < (int)item_ct1.get_group(2); ++c) {
            const int g2 = g_sp_cnt[q][c][0], e2 = g_sp_cnt[q][c][1];
            const int take = eq_budget - eqb < 0 ? 0 : (eq_budget - eqb > e2 ? e2 : eq_budget - eqb);
            sel += g2 + take;
            eqb += e2;
        }
        s_sel0 = sel;
        s_eqb0 = eqb;
    }
    const int64_t b1 = b0 + SP_CH < nb ? b0 + SP_CH : nb;
    const int64_t per = (SP_CH / 32) / SP_W;           // rows of 32 blocks per warp
    const int64_t w0 = b0 + (int64_t) warp * 32 * per, w1 = w0 + 32 * per < b1 ? w0 + 32 * per : b1;
    const float* sc = scores + (int64_t) q * max_blocks;
    auto weight = [&](int64_t b) -> int { return b < n_bid ? R : (int) (n_kv - n_bid * R); };
    int gt = 0, eq = 0;
    for (int64_t b = w0 + lane; b < w1; b += 32) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if (k > thr) gt += w;
        else if (k == thr) eq += w;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        /*
        DPCT1108: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        gt += dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), gt,
            o);
        /*
        DPCT1108: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        eq += dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), eq,
            o);
    }
    if (lane == 0) { s_gt[warp] = gt; s_eq[warp] = eq; }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (t == 0) {
        int eb = s_eqb0, sb = s_sel0;
        for (int w2 = 0; w2 < SP_W; ++w2) {
            const int g2 = s_gt[w2], e2 = s_eq[w2];
            const int take = eq_budget - eb < 0 ? 0 : (eq_budget - eb > e2 ? e2 : eq_budget - eb);
            s_gt[w2] = sb;
            s_eq[w2] = eb;
            sb += g2 + take;
            eb += e2;
        }
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    int run_sel = s_gt[warp], run_eq = s_eq[warp];
    for (int64_t r0 = w0; r0 < w1; r0 += 32) {
        const int64_t b = r0 + lane;
        const int w = b < w1 ? weight(b) : 0;
        const uint32_t k = w ? order_key(sc[b]) : 0u;
        const int my_gt = (w && k > thr) ? w : 0, my_eq = (w && k == thr) ? w : 0;
        if (sycl::reduce_over_group(
                sycl::ext::oneapi::this_work_item::get_sub_group(),
                (0xffffffffu &
                 (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                             .get_local_linear_id())) &&
                        (my_gt | my_eq) != 0
                    ? (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                                  .get_local_linear_id())
                    : 0,
                sycl::ext::oneapi::plus<>()) == 0u) continue;
        int pe = my_eq;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            /*
            DPCT1108: '__shfl_up_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            const int y = dpct::experimental::shift_sub_group_right(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                pe, o);
            if (lane >= o) pe += y;
        }
        const int left = eq_budget - (run_eq + pe - my_eq);
        const int take = left < 0 ? 0 : (left > my_eq ? my_eq : left);
        const int my_sel = my_gt + take;
        int ps = my_sel;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            /*
            DPCT1108: '__shfl_up_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            const int y = dpct::experimental::shift_sub_group_right(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                ps, o);
            if (lane >= o) ps += y;
        }
        int32_t* dst = out + run_sel + ps - my_sel;
#pragma unroll
        for (int c = 0; c < my_sel; ++c) dst[c] = (int32_t)(b * R + c);
        /*
        DPCT1108: '__shfl_sync' was migrated with the experimental feature
        masked sub_group function which may not be supported by all compilers or
        runtimes. You may need to adjust the code.
        */
        run_eq += dpct::experimental::select_from_sub_group(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), pe,
            31);
        /*
        DPCT1108: '__shfl_sync' was migrated with the experimental feature
        masked sub_group function which may not be supported by all compilers or
        runtimes. You may need to adjust the code.
        */
        run_sel += dpct::experimental::select_from_sub_group(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), ps,
            31);
    }
}

bool topk_split_device() {
    static int ok[64] = {};   // 0 unknown, 1 yes, 2 no
    static const int env = [] { const char* v = std::getenv("STRATA_TOPK_SPLIT"); return v ? (std::atoi(v) != 0 ? 1 : 0) : -1; }();
    if (env == 0) return false;
    int dev = 0;
    /*
    DPCT1026: The call to cudaGetLastError was removed because this
    functionality is redundant in SYCL.
    */
    if (DPCT_CHECK_ERROR(dev = dpct::get_current_device_id()) != 0 || dev < 0 ||
        dev >= 64) {
        ; return false;
    }
    if (ok[dev] == 0) {
        int major = 0, minor = 0;
        major = dpct::get_device(dev).get_major_version();
        minor = dpct::get_device(dev).get_minor_version();
        ok[dev] = (env == 1 || strata::sm75_default(major, minor)) ? 1 : 2;   // Turing, STRATA_SM75_PERF=1
    }
    return ok[dev] == 1;
}

bool qsa_block_topk_split(const float *scores, const int32_t *steps, int64_t nq,
                          int64_t max_blocks, int64_t cap, int32_t *ids,
                          dpct::queue_ptr s) try {
    const int64_t nch = (max_blocks + SP_CH - 1) / SP_CH;
    if (nq < 1 || nq > SP_MAXQ || nch > SP_MAXCH) return false;
    static void* hist_ptr[64] = {};
    int dev = 0;
    /*
    DPCT1026: The call to cudaGetLastError was removed because this
    functionality is redundant in SYCL.
    */
    if (DPCT_CHECK_ERROR(dev = dpct::get_current_device_id()) != 0 || dev < 0 ||
        dev >= 64) {
        ; return false;
    }
    if (hist_ptr[dev] == nullptr && (hist_ptr[dev] = g_sp_hist.get_ptr()) == nullptr) {   // (cudaGetSymbolAddress)
        /*
        DPCT1026: The call to cudaGetLastError was removed because this
        functionality is redundant in SYCL.
        */
        hist_ptr[dev] = nullptr;
        return false;
    }
    s->memset(hist_ptr[dev], 0, (size_t) nq * 4 * 256 * sizeof(int));   // (cudaMemsetAsync; SYCL reports by exception)
    const dpct::dim3 grid((unsigned)nch, (unsigned)nq);
    {
        g_sp_hist.init(*s);
        g_sp_state.init(*s);

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {
            auto g_sp_hist_acc_ct1 = g_sp_hist.get_access(cgh);
            auto g_sp_state_acc_ct1 = g_sp_state.get_access(cgh);

            cgh.parallel_for<dpct_kernel_name<
                class topk_split_hist_kernel_41b92a, dpct_kernel_scalar<0>>>(
                sycl::nd_range<3>(grid * sycl::range(1, 1, SP_T),
                                  sycl::range(1, 1, SP_T)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        topk_split_hist_kernel<0>(scores, steps, max_blocks,
                                                  g_sp_hist_acc_ct1,
                                                  g_sp_state_acc_ct1);
                    });
        });
    }
    {
        g_sp_hist.init(*s);
        g_sp_state.init(*s);

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {
            auto g_sp_hist_acc_ct1 = g_sp_hist.get_access(cgh);
            auto g_sp_state_acc_ct1 = g_sp_state.get_access(cgh);

            cgh.parallel_for<dpct_kernel_name<
                class topk_split_hist_kernel_882804, dpct_kernel_scalar<1>>>(
                sycl::nd_range<3>(grid * sycl::range(1, 1, SP_T),
                                  sycl::range(1, 1, SP_T)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        topk_split_hist_kernel<1>(scores, steps, max_blocks,
                                                  g_sp_hist_acc_ct1,
                                                  g_sp_state_acc_ct1);
                    });
        });
    }
    {
        g_sp_hist.init(*s);
        g_sp_state.init(*s);

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {
            auto g_sp_hist_acc_ct1 = g_sp_hist.get_access(cgh);
            auto g_sp_state_acc_ct1 = g_sp_state.get_access(cgh);

            cgh.parallel_for<dpct_kernel_name<
                class topk_split_hist_kernel_a1e7d0, dpct_kernel_scalar<2>>>(
                sycl::nd_range<3>(grid * sycl::range(1, 1, SP_T),
                                  sycl::range(1, 1, SP_T)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        topk_split_hist_kernel<2>(scores, steps, max_blocks,
                                                  g_sp_hist_acc_ct1,
                                                  g_sp_state_acc_ct1);
                    });
        });
    }
    {
        g_sp_hist.init(*s);
        g_sp_state.init(*s);

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {
            auto g_sp_hist_acc_ct1 = g_sp_hist.get_access(cgh);
            auto g_sp_state_acc_ct1 = g_sp_state.get_access(cgh);

            cgh.parallel_for<dpct_kernel_name<
                class topk_split_hist_kernel_08181d, dpct_kernel_scalar<3>>>(
                sycl::nd_range<3>(grid * sycl::range(1, 1, SP_T),
                                  sycl::range(1, 1, SP_T)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        topk_split_hist_kernel<3>(scores, steps, max_blocks,
                                                  g_sp_hist_acc_ct1,
                                                  g_sp_state_acc_ct1);
                    });
        });
    }
    {
        g_sp_hist.init(*s);
        g_sp_state.init(*s);
        g_sp_cnt.init(*s);

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {
            auto g_sp_hist_acc_ct1 = g_sp_hist.get_access(cgh);
            auto g_sp_state_acc_ct1 = g_sp_state.get_access(cgh);
            auto g_sp_cnt_acc_ct1 = g_sp_cnt.get_access(cgh);

            cgh.parallel_for<
                dpct_kernel_name<class topk_split_count_kernel_ff1a5f>>(
                sycl::nd_range<3>(grid * sycl::range(1, 1, SP_T),
                                  sycl::range(1, 1, SP_T)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        topk_split_count_kernel(
                            scores, steps, max_blocks, g_sp_hist_acc_ct1,
                            g_sp_state_acc_ct1, g_sp_cnt_acc_ct1);
                    });
        });
    }
    {
        g_sp_state.init(*s);
        g_sp_cnt.init(*s);

        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        s->submit([&](sycl::handler &cgh) {
            auto g_sp_state_acc_ct1 = g_sp_state.get_access(cgh);
            auto g_sp_cnt_acc_ct1 = g_sp_cnt.get_access(cgh);

            cgh.parallel_for<
                dpct_kernel_name<class topk_split_emit_kernel_efccdd>>(
                sycl::nd_range<3>(grid * sycl::range(1, 1, SP_T),
                                  sycl::range(1, 1, SP_T)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        topk_split_emit_kernel(scores, steps, max_blocks, cap,
                                               ids, g_sp_state_acc_ct1,
                                               g_sp_cnt_acc_ct1);
                    });
        });
    }
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
#endif

#if !defined(__HIPCC__)
// ---- the prompt path's top-k with one full read of the scores (Turing).  block_topk_wide_kernel reads a query's scores
// six times (four radix passes, a count, the emit); past ~90K cells that is DRAM traffic (256 queries at 400K: 0.6 GB a
// layer, ~1.5 ms on an RTX 2080 Ti).  Here a CTA first reads every 32nd block and takes the KS-th largest of those keys
// as a bound LB, then reads all blocks once and keeps the ones with key >= LB in shared memory.  When those hold at
// least `width` cells, the width-th largest cell key (block_topk_kernel's threshold) is >= LB, so every block at or
// above the threshold is a candidate: the threshold, `above` and the tie budget computed over the candidates with
// block_topk_kernel's rule are the reference's, and the candidates sorted by block emit the reference's cells in its
// order.  Otherwise (too few cells above LB, or more than SM_CAP candidates) the query is flagged and
// block_topk_wide_kernel<true> runs it.  STRATA_TOPK_SAMPLE=0 keeps the wide kernel alone.
constexpr int TS_STRIDE = 32, TS_CAP = 4096;

__dpct_inline__ int
ts_block_scan(int v, int *s_w) { // exclusive scan over the CTA's TK_T threads
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_local_id(2), lane = t & 31, w = t >> 5;
    int x = v;
#pragma unroll
    /*
    DPCT1108: '__shfl_up_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    for (int o = 1; o < 32; o <<= 1) {
        const int y = dpct::experimental::shift_sub_group_right(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), x,
            o);
        if (lane >= o) x += y;
    }
    if (lane == 31) s_w[w] = x;
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (w == 0) {
        int y = lane < TK_T / 32 ? s_w[lane] : 0;
#pragma unroll
        /*
        DPCT1108: '__shfl_up_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        for (int o = 1; o < 32; o <<= 1) {
            const int z = dpct::experimental::shift_sub_group_right(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                y, o);
            if (lane >= o) y += z;
        }
        if (lane < TK_T / 32) s_w[lane] = y;   // inclusive warp totals
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const int base = w > 0 ? s_w[w - 1] : 0;
    const int r = base + x - v;
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    return r;
}

/*
DPCT1110: The total declared local variable size in device function
block_topk_sample_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void block_topk_sample_kernel(const float *__restrict__ scores,
                                              const int32_t *__restrict__ steps,
                                              int64_t max_blocks, int64_t cap,
                                              int32_t *__restrict__ ids,
                                              int *g_tk_fail) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &s_key =
    *sycl::ext::oneapi::group_local_memory_for_overwrite<uint32_t[TS_CAP]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_idx =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int32_t[TS_CAP]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &hist = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[256]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_w =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[TK_T / 32]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_cnt = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_cells = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_digit = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_above = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int64_t qi = item_ct1.get_group(2);
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + qi * cap;
    const int t = item_ct1.get_local_id(2);
    if (n_kv <= width) {
        if (t == 0) g_tk_fail[qi] = 0;
#pragma unroll
        for (int64_t j = t; j < n_kv; j += TK_T) out[j] = (int32_t)j;
        return;
    }
    const float* sc = scores + qi * max_blocks;
    const int64_t nb = n_bid + 1;
    auto weight = [&](int64_t b) -> int { return b < n_bid ? R : (int) (n_kv - n_bid * R); };
    const int ns = (int) ((nb + TS_STRIDE - 1) / TS_STRIDE);
    if (ns > TS_CAP) { if (t == 0) g_tk_fail[qi] = 1; return; }
    // 1. the bound: the KS-th largest of every 32nd block's key (a block without cells counts as the smallest key)
    for (int i = t; i < ns; i += TK_T) {
        const int64_t b = (int64_t) i * TS_STRIDE;
        s_key[i] = weight(b) ? order_key(sc[b]) : 0u;
    }
    const int ks_want = (int) (2 * ((width + R * TS_STRIDE - 1) / (R * TS_STRIDE)) + 16);
    int need = ks_want < ns ? ks_want : ns;
    uint32_t prefix = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
#pragma unroll
        for (int i = t; i < 256; i += TK_T) hist[i] = 0;
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
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        for (int i = t; i < ns; i += TK_T) {
            const uint32_t k = s_key[i];
            if ((k & hi_mask) == (prefix & hi_mask)) dpct::atomic_fetch_add<
                sycl::access::address_space::generic_space>(
                &hist[(k >> shift) & 255], 1);
        }
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
        if (t == 0) {
            int cum = 0, d = 255;
#pragma unroll
            for (; d > 0; --d) {
                if (cum + hist[d] >= need) break;
                cum += hist[d];
            }
            s_digit = d;
            s_above = need - cum;
        }
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
        prefix |= (uint32_t) s_digit << shift;
        need = s_above;
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
    const uint32_t lb = prefix;
    // 2. one read of every block: the candidates (key >= LB) and their cells
    if (t == 0) { s_cnt = 0; s_cells = 0; }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    for (int64_t b = t; b < nb; b += TK_T) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if (k >= lb) {
            const int pos = dpct::atomic_fetch_add<
                sycl::access::address_space::generic_space>(&s_cnt, 1);
            if (pos < TS_CAP) { s_key[pos] = k; s_idx[pos] = (int32_t) b; }
            dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                &s_cells, w);
        }
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const int n = s_cnt;
    if (n > TS_CAP || s_cells < width) {   // the bound was too low or too high: the wide kernel takes this query
        if (t == 0) g_tk_fail[qi] = 1;
        return;
    }
    if (t == 0) g_tk_fail[qi] = 0;
    // 3. block_topk_kernel's threshold over the candidates (weighted)
    prefix = 0;
    int above = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
#pragma unroll
        for (int i = t; i < 256; i += TK_T) hist[i] = 0;
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
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        for (int i = t; i < n; i += TK_T) {
            const uint32_t k = s_key[i];
            if ((k & hi_mask) == (prefix & hi_mask)) dpct::atomic_fetch_add<
                sycl::access::address_space::generic_space>(
                &hist[(k >> shift) & 255], weight(s_idx[i]));
        }
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
        if (t == 0) {
            int cum = above, d = 255;
#pragma unroll
            for (; d > 0; --d) {
                if (cum + hist[d] >= width) break;
                cum += hist[d];
            }
            s_digit = d;
            s_above = cum;
        }
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
        prefix |= (uint32_t) s_digit << shift;
        above = s_above;
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
    const uint32_t thr = prefix;
    const int eq_budget = (int) (width - above);
    // 4. the candidates in block order (bitonic sort; padding sorts last)
    int n2 = 1;
    while (n2 < n) n2 <<= 1;
#pragma unroll
    for (int i = n + t; i < n2; i += TK_T) {
        s_idx[i] = 0x7fffffff; s_key[i] = 0u;
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    for (int kk = 2; kk <= n2; kk <<= 1) {
        for (int j = kk >> 1; j > 0; j >>= 1) {
            for (int i = t; i < n2; i += TK_T) {
                const int p = i ^ j;
                if (p > i) {
                    const bool up = (i & kk) == 0;
                    const int a = s_idx[i], b = s_idx[p];
                    if ((a > b) == up) {
                        s_idx[i] = b; s_idx[p] = a;
                        const uint32_t ka = s_key[i]; s_key[i] = s_key[p]; s_key[p] = ka;
                    }
                }
            }
            /*
            DPCT1118: SYCL group functions and algorithms must be
            encountered in converged control flow. You may need to adjust the
            code.
            */
            /*
            DPCT1065: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
        }
    }
    // 5. the emit: cells above thr, and the first eq_budget cells at thr, in block order
    constexpr int PER = TS_CAP / TK_T;              // candidates per thread, consecutive
    int eqc[PER], gtc[PER];
    int my_eq = 0;
#pragma unroll
    for (int r = 0; r < PER; ++r) {
        const int i = t * PER + r;
        const int w = i < n ? weight(s_idx[i]) : 0;
        const uint32_t k = s_key[i < n ? i : 0];
        gtc[r] = (i < n && k > thr) ? w : 0;
        eqc[r] = (i < n && k == thr) ? w : 0;
        my_eq += eqc[r];
    }
    int eq_before = ts_block_scan(my_eq, s_w);
    int sel[PER], my_sel = 0;
#pragma unroll
    for (int r = 0; r < PER; ++r) {
        const int left = eq_budget - eq_before;
        const int take = left < 0 ? 0 : (left > eqc[r] ? eqc[r] : left);
        sel[r] = gtc[r] + take;
        my_sel += sel[r];
        eq_before += eqc[r];
    }
    int pos = ts_block_scan(my_sel, s_w);
#pragma unroll
    for (int r = 0; r < PER; ++r) {
        const int i = t * PER + r;
        if (sel[r] > 0) {
            const int64_t b = s_idx[i];
#pragma unroll
            for (int c = 0; c < sel[r]; ++c)
                out[pos + c] = (int32_t)(b * R + c);
        }
        pos += sel[r];
    }
}

// The prompt path's one-read top-k: STRATA_TOPK_SAMPLE=0 / 1 forces it off / on for any card; unset it is a Turing kernel
// of the series (STRATA_SM75_PERF=1).  (As submitted in #2093 it was on for every CUDA card; same ids, but only measured
// on an RTX 2080 Ti, so the other cards keep the wide kernel.)
bool topk_sample_on() {
    static const int env = [] { const char* v = std::getenv("STRATA_TOPK_SAMPLE"); return v ? (std::atoi(v) != 0 ? 1 : 0) : -1; }();
    if (env >= 0) return env != 0;
    static int ok[64] = {};   // 0 unknown, 1 yes, 2 no
    int dev = 0;
    /*
    DPCT1026: The call to cudaGetLastError was removed because this
    functionality is redundant in SYCL.
    */
    if (DPCT_CHECK_ERROR(dev = dpct::get_current_device_id()) != 0 || dev < 0 ||
        dev >= 64) {
        ; return false;
    }
    if (ok[dev] == 0) {
        int major = 0, minor = 0;
        major = dpct::get_device(dev).get_major_version();
        minor = dpct::get_device(dev).get_minor_version();
        ok[dev] = strata::sm75_default(major, minor) ? 1 : 2;
    }
    return ok[dev] == 1;
}
#endif

void qsa_block_topk(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                    const QsaShapes& s, int32_t* ids, void* stream, int64_t active_blocks) {
    // keys in registers when every query's blocks fit (contexts up to ~135K cells), else (CUDA) the same threads
    // reading them from memory; the same ids. STRATA_TOPK_OLD=1: the original kernel
    static const bool old = std::getenv("STRATA_TOPK_OLD") != nullptr;
    if (nq <= 0) return;
#if !defined(__HIPCC__)
    // sm_90+: a cluster of CL_N CTAs per query for the calls of a few queries (decode windows); the same ids.
    // STRATA_QSA_CLUSTER=0: the one-CTA kernels below
    static const bool cluster = [] {
        const char* v = std::getenv("STRATA_QSA_CLUSTER");
        return !v || std::atoi(v) != 0;
    }();
    if (cluster && !old && nq <= CL_MAXQ && qsa_block_topk_cluster(scores, steps, nq, max_blocks, cap, s, ids, stream))
        return;
#endif
#if !defined(__HIPCC__)
    // Turing decode (a window's few queries, no block count): the split kernels above; the same ids
    if (!old && active_blocks <= 0 && max_blocks >= 16384 && s.idx_block == R &&
        cap >= qsa_selection_width(kTopkMaxCells, s) && topk_split_device() &&
        qsa_block_topk_split(scores, steps, nq, max_blocks, cap, ids,
                             strata::q_of(stream))) {
        /*
        DPCT1010: SYCL uses exceptions to report errors and does not use
        the error codes. The cudaGetLastError function call was replaced with 0.
        You need to rewrite this code.
        */
        const dpct::err0 e = 0;
        /*
        DPCT1009: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */

        return;
    }
#endif
    // the blocks a query can have: the call's active count when the caller knows it (the prompt path), else the capacity.
    // Decode (no count) keeps the capacity rule and the original register width: nothing changes there.
#if defined(__HIPCC__)
    const bool counted = active_blocks > 0;
#else
    // Turing: --max-context 262144 makes the stride 65538, even while a 131K prompt's active blocks fit in
    // TK_T * TK_PER registers. Use the prefill bound on sm_75, keeping max_blocks as the score-row stride.
    // Other CUDA devices keep 0.1.32's capacity rule: #337 was measured on RDNA4, and RTX 5070 64K prompts were
    // 1-3% slower. Decode/captured graphs omit the bound and never query the device here.
    static const bool capacity_guard = std::getenv("STRATA_TOPK_CAPACITY_GUARD") != nullptr;
    // STRATA_TOPK_ACTIVE_ANY=1 (tests): the Turing dispatch on any CUDA card, so qsa_topk_active_parity checks it
    // on whatever card runs the tests (the kernels are the same on every architecture)
    static const bool any_card = [] { const char* v = std::getenv("STRATA_TOPK_ACTIVE_ANY"); return v && v[0] == '1'; }();
    const bool counted = !capacity_guard && active_blocks > 0 && active_blocks <= max_blocks &&
                         (any_card || topk_active_turing_device());
#endif
    const int64_t reach = counted && active_blocks < max_blocks ? active_blocks : max_blocks;
    const int64_t fit = (int64_t) TK_T * (counted ? TK_PER_MAX : TK_PER);
#if defined(__HIPCC__)
    // gfx1201: below ~28K cells (7,168 blocks) the 1,024-thread kernel's fixed cost loses to the ref.  STRATA_TOPK_REG_MIN_BLOCKS
    // moves the line (0: the register kernel always); gfx1151's table sets 4096, the crossover with the cooperative scan
    static const int64_t kRegMinBlocks = [] {
        const char* v = std::getenv("STRATA_TOPK_REG_MIN_BLOCKS");
        return v != nullptr ? (int64_t) std::atoll(v) : (int64_t) 7168;
    }();
    const bool too_small = counted && reach < kRegMinBlocks;
    // campaign3: `reach > fit` no longer forces the ref radix - the wide kernel below takes it (same ids).
    if (old || too_small) {
#else
    constexpr int64_t kWideMinBlocks = 4608;   // RTX 3060: a prompt batch below ~18K cells is faster on the ref
    const bool too_small = reach > fit && active_blocks > 0 && active_blocks < kWideMinBlocks;
    if (old || too_small) {
#endif
        qsa_block_topk_ref(scores, steps, nq, max_blocks, cap, s, ids, stream);
        return;
    }
    if (s.idx_block != R || cap < qsa_selection_width(kTopkMaxCells, s)) {
        std::fprintf(stderr, "qsa_block_topk: unsupported geometry or cap\n");
        std::exit(1);
    }
    // Turing prefill (the `counted` bound) above ~90K cells (22,528 blocks): the wide kernel, which reads the keys
    // coalesced, instead of the register kernel's uncoalesced per-thread runs (PR #743: 131K, 1.03 -> 0.75 ms); the
    // same ids.  STRATA_TOPK_STREAM=0 restores the register kernel there.
    // campaign3: on HIP too (was CUDA-only; gfx11 has no sm_90 cluster path and fell to the ref radix past the
    // register kernel's reach - the deep-decode top-k cost).  Same ids as the ref kernel.
    static const bool turing_wide = [] {
        const char* v = std::getenv("STRATA_TOPK_STREAM");
        return v == nullptr || v[0] != '0';
    }();
    if (reach > fit || (turing_wide && counted && reach > 22528)) {   // past the register kernel's reach, or Turing's band
#if !defined(__HIPCC__)
        if (counted && nq <= TS_MAXQ && reach > 28672 && topk_sample_on()) {   // one read of the scores (below ~115K cells the wide kernel is as fast)
            /*
            DPCT1049: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            {
                g_tk_fail.init(*strata::q_of(stream));

                auto exp_props = sycl::ext::oneapi::experimental::properties{
                    sycl::ext::oneapi::experimental::use_root_sync};

                strata::q_of(stream)
                    ->submit([&](sycl::handler &cgh) {
                        auto g_tk_fail_ptr_ct1 = g_tk_fail.get_ptr();

                        cgh.parallel_for<dpct_kernel_name<
                            class block_topk_sample_kernel_f2805c>>(
                            sycl::nd_range<3>(sycl::range(1, 1, (unsigned)nq) *
                                                  sycl::range(1, 1, TK_T),
                                              sycl::range(1, 1, TK_T)),
                            exp_props,
                            [=](sycl::nd_item<3> item_ct1)
                                [[sycl::reqd_sub_group_size(32)]] {
                                    block_topk_sample_kernel(
                                        scores, steps, max_blocks, cap, ids,
                                        g_tk_fail_ptr_ct1);
                                });
                    });
            }
            /*
            DPCT1049: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            {
                g_tk_fail.init(*strata::q_of(stream));

                auto exp_props = sycl::ext::oneapi::experimental::properties{
                    sycl::ext::oneapi::experimental::use_root_sync};

                strata::q_of(stream)
                    ->submit([&](sycl::handler &cgh) {
                        auto g_tk_fail_ptr_ct1 = g_tk_fail.get_ptr();

                        cgh.parallel_for<dpct_kernel_name<
                            class block_topk_wide_kernel_b155ef,
                            dpct_kernel_scalar<true>>>(
                            sycl::nd_range<3>(sycl::range(1, 1, (unsigned)nq) *
                                                  sycl::range(1, 1, TK_T),
                                              sycl::range(1, 1, TK_T)),
                            exp_props,
                            [=](sycl::nd_item<3> item_ct1)
                                [[sycl::reqd_sub_group_size(32)]] {
                                    block_topk_wide_kernel<true>(
                                        scores, steps, max_blocks, cap, ids,
                                        g_tk_fail_ptr_ct1);
                                });
                    });
            }
        } else
#endif
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            g_tk_fail.init(*strata::q_of(stream));

            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            strata::q_of(stream)
                ->submit([&](sycl::handler &cgh) {
                    auto g_tk_fail_ptr_ct1 = g_tk_fail.get_ptr();

                    cgh.parallel_for<
                        dpct_kernel_name<class block_topk_wide_kernel_7b2d76,
                                         dpct_kernel_scalar<false>>>(
                        sycl::nd_range<3>(sycl::range(1, 1, (unsigned)nq) *
                                              sycl::range(1, 1, TK_T),
                                          sycl::range(1, 1, TK_T)),
                        exp_props,
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(32)]] {
                                block_topk_wide_kernel<false>(
                                    scores, steps, max_blocks, cap, ids,
                                    g_tk_fail_ptr_ct1);
                            });
                });
        }
    } else
    if (reach <= (int64_t) TK_T * TK_PER)
        /*
        DPCT1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class block_topk_reg_kernel_310715,
                                            dpct_kernel_scalar<TK_PER>>>(
                sycl::nd_range<3>(sycl::range(1, 1, (unsigned)nq) *
                                      sycl::range(1, 1, TK_T),
                                  sycl::range(1, 1, TK_T)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        block_topk_reg_kernel<TK_PER>(scores, steps, max_blocks,
                                                      cap, ids);
                    });
    } else
    /*
    DPCT1049: The work-group size passed to the SYCL kernel may exceed
    the limit. To get the device limit, query
    info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class block_topk_reg_kernel_287022,
                                            dpct_kernel_scalar<TK_PER_MAX>>>(
                sycl::nd_range<3>(sycl::range(1, 1, (unsigned)nq) *
                                      sycl::range(1, 1, TK_T),
                                  sycl::range(1, 1, TK_T)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        block_topk_reg_kernel<TK_PER_MAX>(scores, steps,
                                                          max_blocks, cap, ids);
                    });
    }
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

}  // namespace strata::kernels
