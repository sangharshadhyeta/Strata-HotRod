// src/dsv41/ds4_lalearn.cu - the DeepSeek engine's learned look-ahead (see ds4_lalearn.cuh).  Ported from Strata's native_router.cu
// (STRATA_LA_LEARN): the same slots, distances, counters, decision rules, AdamW and file format, with these changes: the learner
// lives on the device of the trunk's first layer (no scratch device), the real expert count E is passed in (the rows are padded
// to 512 with experts that count as resident and are never scored; their logits are -1e30 and their gradients zero), the
// resident flags are the layers' own vram_tab tables (refreshed once per window by a kernel), and the batch's timing profile is
// not ported.
#include "ds4_lalearn.cuh"
#include "ds4_launch.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace strata::ds4::la {
namespace {

constexpr int kSlots = 4;        // slot 0: the plain router; slots 1-3: the corrections of rank 4, 8 and 16
constexpr int kNSlots = kSlots - 1;
constexpr int kRows = 28;        // the window's projection rows: slot 1 rows 0-3, slot 2 rows 4-11, slot 3 rows 12-27
constexpr int kE = 512;          // the row stride and the block size of the batch (the real experts, then padding)
constexpr int kTok = 8;          // the most tokens a window holds
constexpr int kRoute = 1024;     // the logits' row stride (the engine's gate GEMV output)
constexpr int kCnt = 18;         // counters per (s,d): [0] picks since change, [1] picks in the recent 500, [2] picks all,
                                 // [3+c] hits since change, [7+c] hits recent, [11+c] hits all, [15] changes, [16] resets,
                                 // [17] recent blocks in a row where the plain router beat the leader
constexpr int kMinPicks = 200, kFreezePicks = 2000, kRecPicks = 500, kLeaderEvery = 4;
constexpr float kLr = 1e-3f, kB1 = 0.9f, kB2 = 0.999f, kEps = 1e-8f, kWd = 1e-3f, kZ90 = 1.6448536f;
constexpr float kPad = -1e30f;   // the logit of a padding expert (never a pick, zero gradient)

int dist_index(int dist) {
    for (int i = 0; i < kNd; ++i)
        if (kDist[i] == dist) return i;
    return -1;
}
__host__ __device__ inline int la_rank(int c) { return c == 1 ? 4 : c == 2 ? 8 : c == 3 ? 16 : 0; }
__host__ __device__ inline int la_row_off(int c) { return c == 1 ? 0 : c == 2 ? 4 : c == 3 ? 12 : 0; }
// uniform in [-sqrt(3)e-3, sqrt(3)e-3]: standard deviation 1e-3 (the initial draws and the reset draws)
__host__ __device__ inline float la_rand_pm(unsigned long long key) {
    key ^= key >> 33; key *= 0xff51afd7ed558ccdULL; key ^= key >> 33; key *= 0xc4ceb9fe1a85ec53ULL; key ^= key >> 33;
    return (2.0f * (float) (key >> 40) * (1.0f / 16777216.0f) - 1.0f) * 1.7320508e-3f;
}
__host__ __device__ inline unsigned long long la_key(int sd, int c, long long i, unsigned long long nre) {
    return (unsigned long long) (sd * kSlots + c) * 0x9E3779B97F4A7C15ULL + (unsigned long long) i * 0xD1B54A32D192ED03ULL +
           nre * 0x2545F4914F6CDD1DULL + 1ULL;
}
__device__ __forceinline__ void la_adam(float& p, float& m, float& v, float g, int t) {
    m = kB1 * m + (1.0f - kB1) * g;
    v = kB2 * v + (1.0f - kB2) * g * g;
    const float c1 = 1.0f - powf(kB1, (float) t), c2 = 1.0f - powf(kB2, (float) t);
    p -= kLr * ((m / c1) / (sqrtf(v / c2) + kEps) + kWd * p);   // AdamW: the decay is decoupled from the gradient
}
// the sum over the block's threads, returned to all of them; sh: 32 floats
__device__ __forceinline__ float la_block_sum(float v, float* sh) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int w = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) sh[w] = v;
    __syncthreads();
    if (w == 0) {
        float s = threadIdx.x < (blockDim.x >> 5) ? sh[threadIdx.x] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (threadIdx.x == 0) sh[0] = s;
    }
    __syncthreads();
    const float r = sh[0];
    __syncthreads();   // every thread has read sh[0] before a following call writes sh[w] (a race on the second sum otherwise)
    return r;
}

// forward (per source s and dist d): u = A x for the 28 rows (a block per row, all n tokens), then each slot's logits = plain + B u
// (a thread per token and expert); the leader's go back into the plain logits (route)
struct LaFwdArgs {
    const float* x;              // the source's MoE input, n x H
    float* xsave;                // the first dist: the window's copy of x for the training batch (else nullptr)
    float* u;                    // n x 28
    float* pred;                 // 4 x 8 x 512: the slots' logits (slot 0: the plain router)
    float* route;                // the plain logits on entry (row stride 1024, E valid), the leader's on exit
    const float* A[kSlots];
    const float* B[kSlots];
    const int* lead;
    int n, H, E;
};
__global__ void la_fwd_u_k(LaFwdArgs a) {   // grid 28, block 256
    const int k = blockIdx.x, c = k < 4 ? 1 : (k < 12 ? 2 : 3);
    const float* row = a.A[c] + (size_t) (k - la_row_off(c)) * a.H;
    __shared__ float sh[32];
    if (a.xsave != nullptr && k == 0)
        for (int i = threadIdx.x; i < a.n * a.H; i += blockDim.x) a.xsave[i] = a.x[i];
    for (int t = 0; t < a.n; ++t) {
        float s = 0.0f;
        for (int h = threadIdx.x; h < a.H; h += blockDim.x) s += row[h] * a.x[(size_t) t * a.H + h];
        s = la_block_sum(s, sh);
        if (threadIdx.x == 0) a.u[t * kRows + k] = s;
    }
}
// grid ceil(n * 512 / 256), block 256.  The padding experts (e >= E) get -1e30 in every slot and leave the route alone.
__global__ void la_fwd_pred_k(LaFwdArgs a) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= a.n * kE) return;
    const int t = i / kE, e = i % kE;
    if (e >= a.E) {
        for (int c = 0; c < kSlots; ++c) a.pred[c * kTok * kE + i] = kPad;
        return;
    }
    const int lead = *a.lead;
    const float p0 = a.route[(size_t) t * kRoute + e];
    float sel = p0;
    a.pred[i] = p0;
    for (int c = 1; c < kSlots; ++c) {
        const int r = la_rank(c);
        const float* b = a.B[c] + (size_t) e * r;
        const float* uu = a.u + t * kRows + la_row_off(c);
        float v = p0;
        for (int q = 0; q < r; ++q) v += b[q] * uu[q];
        a.pred[c * kTok * kE + i] = v;
        if (lead == c) sel = v;
    }
    a.route[(size_t) t * kRoute + e] = sel;
}

// the batch (one window, after its graph): every (source s, dist d, slot c) at once.  sd = s * kNd + dd is the (s,d) index, its target
// layer is s + kDist[dd] (skipped when past the last layer), and z = sd * kNSlots + (c - 1) the trained set.
struct LaBatch {
    float* A[kSlots]; float* B[kSlots]; float* mA[kSlots]; float* vA[kSlots]; float* mB[kSlots]; float* vB[kSlots];
    const float* x;              // [L][8][H]: each source's MoE input (this window)
    const float* u;              // [Nsd][8][28]
    const float* pred;           // [Nsd][4][8][512]: the slots' logits from the source's forward (this window)
    const float* lab;            // [L][8][512]: each layer's router logits (this window), padding -1e30
    const int* res;              // [L][512]: the resident flags (>= 0: resident, padding included)
    int* rk;                     // [L][8][512]: the true rank of each expert
    float* thr;                  // [L][8]: the true 32nd logit
    float* g;                    // [Nset][8][512]: d loss / d logit, batch-averaged
    float* du;                   // [Nset][8][16]: g B for the set's rows
    int* act;                    // [Nset]: 1 when the set trains this window
    int* tstep;                  // [Nset]: this window's Adam step
    int* steps;                  // [Nsd][4]
    int* live;                   // [Nsd][4]
    int* lead;                   // [Nsd]
    int* frozen;                 // [Nsd]
    int* reset;                  // [Nsd]
    int* tick;                   // [Nsd]: windows seen by the (s,d)
    unsigned long long* cnt;     // [Nsd][kCnt]
    unsigned long long* nwin;    // the windows whose batch ran
    unsigned long long* rinfo;   // [Nsd][4]: the last reset's recent picks, plain hits, old leader's hits, old leader's slot
    int dist[kNd];
    int L, H, lo, n, Nd, Nsd, Nset;
};
// the true rank of each expert (ties by index) and the true 32nd logit: grid (n, L), a thread per padded expert
__global__ void la_b_rank_k(LaBatch a) {
    __shared__ float s[kE];
    const int t = blockIdx.x, m = blockIdx.y, e = threadIdx.x;
    const size_t row = (size_t) m * kTok + t;
    const float v = a.lab[row * kE + e];
    s[e] = v;
    __syncthreads();
    int c = 0;
    for (int f = 0; f < kE; ++f) c += (s[f] > v) || (s[f] == v && f < e);
    a.rk[row * kE + e] = c;
    if (c == 31) a.thr[row] = v;
}
// the prequential score: grid (n, Nsd, 4), a thread per expert that is a true pick below the floor and not resident; a hit when fewer
// than 2 * lo non-resident predictions of the slot are above it.  A non-live candidate is not scored.
__global__ void la_b_score_k(LaBatch a) {
    const int t = blockIdx.x, sd = blockIdx.y, c = blockIdx.z, e = threadIdx.x;
    const int s = sd / a.Nd, m = s + a.dist[sd % a.Nd];
    if (m >= a.L) return;
    if (c > 0 && !a.live[sd * kSlots + c]) return;
    const size_t row = (size_t) m * kTok + t;
    if (a.rk[row * kE + e] >= a.lo || a.res[(size_t) m * kE + e] >= 0) return;
    const float* p = a.pred + ((size_t) sd * kSlots + c) * kTok * kE + (size_t) t * kE;
    const float pe = p[e];
    int above = 0;
    for (int f = 0; f < kE; ++f) above += (a.res[(size_t) m * kE + f] < 0) && (p[f] > pe || (p[f] == pe && f < e));
    unsigned long long* cc = a.cnt + (size_t) sd * kCnt;
    if (c == 0) {
        atomicAdd(cc + 0, 1ull);
        atomicAdd(cc + 1, 1ull);
        atomicAdd(cc + 2, 1ull);
    }
    if (above < 2 * a.lo) {
        atomicAdd(cc + 3 + c, 1ull);
        atomicAdd(cc + 7 + c, 1ull);
        atomicAdd(cc + 11 + c, 1ull);
    }
}
// which sets train this window: the (s,d) has a target layer; a live candidate trains; a frozen (s,d) trains its leader only every 4th window
__global__ void la_b_prep_k(LaBatch a) {   // grid ceil(Nset / 64), block 64
    const int z = blockIdx.x * blockDim.x + threadIdx.x;
    if (z >= a.Nset) return;
    const int sd = z / kNSlots, c = z % kNSlots + 1;
    const int s = sd / a.Nd;
    int ok = s + a.dist[sd % a.Nd] < a.L && a.live[sd * kSlots + c];
    if (a.frozen[sd] && (a.lead[sd] != c || a.tick[sd] % kLeaderEvery != 0)) ok = 0;
    a.act[z] = ok;
    if (ok) a.tstep[z] = a.steps[sd * kSlots + c] + 1;
}
// the gradient: block per (token, set), thread per padded expert.  Loss per token: the centred MSE over the true top 32 (the centre is the
// mean over those 32; its gradient is 2/32 (pc - tc)) plus the hinge mean over the other 480 (gradient 1/480 where the logit is above
// thr - 0.1).  Padding has logit -1e30 on both sides: no gradient.  du = g B for the set's rows (before B changes).
__global__ void la_b_grad_k(LaBatch a) {   // grid (n, Nset), block 512
    const int t = blockIdx.x, z = blockIdx.y, e = threadIdx.x;
    if (!a.act[z]) return;
    const int sd = z / kNSlots, c = z % kNSlots + 1, r = la_rank(c);
    const int s = sd / a.Nd, m = s + a.dist[sd % a.Nd];
    __shared__ float sh[32];
    __shared__ float P[16][kE];
    const size_t row = (size_t) m * kTok + t;
    const float p = a.pred[((size_t) sd * kSlots + c) * kTok * kE + (size_t) t * kE + e];
    const float tv = a.lab[row * kE + e];
    const bool top = a.rk[row * kE + e] < 32;
    const float mP = la_block_sum(top ? p : 0.0f, sh) / 32.0f;
    const float mT = la_block_sum(top ? tv : 0.0f, sh) / 32.0f;
    float g = top ? (2.0f / 32.0f) * ((p - mP) - (tv - mT)) : (p > a.thr[row] - 0.1f ? 1.0f / 480.0f : 0.0f);
    g /= (float) a.n;
    a.g[((size_t) z * kTok + t) * kE + e] = g;
    for (int k = 0; k < r; ++k) P[k][e] = g * a.B[c][(size_t) sd * kE * r + e * r + k];
    __syncthreads();
    const int w = e >> 5;
    if (w < r) {
        float sum = 0.0f;
        for (int i = e & 31; i < kE; i += 32) sum += P[w][i];
        for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
        if ((e & 31) == 0) a.du[((size_t) z * kTok + t) * 16 + w] = sum;
    }
}
__global__ void la_b_upd_b_k(LaBatch a) {   // grid (32, Nset), block 256: 512 x 16 entries per set
    const int z = blockIdx.y;
    if (!a.act[z]) return;
    const int sd = z / kNSlots, c = z % kNSlots + 1, r = la_rank(c), off = la_row_off(c);
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int e = idx / 16, k = idx % 16;
    if (e >= kE || k >= r) return;
    float g = 0.0f;
    for (int t = 0; t < a.n; ++t) g += a.g[((size_t) z * kTok + t) * kE + e] * a.u[((size_t) sd * kTok + t) * kRows + off + k];
    const size_t i = (size_t) sd * kE * r + (size_t) e * r + k;
    la_adam(a.B[c][i], a.mB[c][i], a.vB[c][i], g, a.tstep[z]);
}
// the A update, tiled: a thread per column h updates all r rows of the set, so each x element is read once per window (not once per row);
// du for the set sits in shared memory.  grid (ceil(H / 256), Nset), block 256.
__global__ void la_b_upd_a_k(LaBatch a) {
    const int z = blockIdx.y;
    if (!a.act[z]) return;
    const int sd = z / kNSlots, c = z % kNSlots + 1, r = la_rank(c);
    const int s = sd / a.Nd;
    __shared__ float du_s[kTok][16];
    for (int i = threadIdx.x; i < kTok * 16; i += blockDim.x) du_s[i / 16][i % 16] = a.du[(size_t) z * kTok * 16 + i];
    __syncthreads();
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h < a.H) {
        float xt[kTok];
        for (int t = 0; t < a.n; ++t) xt[t] = a.x[((size_t) s * kTok + t) * a.H + h];
        for (int k = 0; k < r; ++k) {
            float g = 0.0f;
            for (int t = 0; t < a.n; ++t) g += du_s[t][k] * xt[t];
            const size_t i = ((size_t) sd * r + k) * a.H + h;
            la_adam(a.A[c][i], a.mA[c][i], a.vA[c][i], g, a.tstep[z]);
        }
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) a.steps[sd * kSlots + c] = a.tstep[z];
}

// the window's decision, one thread per (s,d): the leader (a candidate beats the leader by a Newcombe Wilson 90% interval on >= 200 picks
// since the last change), the freeze (2000 picks without a change), the reset (the plain router beats a leader over the recent 500)
__host__ __device__ inline void la_wilson(float p, float n, float& lo, float& hi) {
    const float z2 = kZ90 * kZ90, den = 1.0f + z2 / n;
    const float c = (p + z2 / (2.0f * n)) / den;
    const float h = kZ90 * sqrtf(p * (1.0f - p) / n + z2 / (4.0f * n * n)) / den;
    lo = c - h;
    hi = c + h;
}
// Newcombe's lower bound for p1 - p2 from two Wilson 90% intervals over n1 and n2 picks
__device__ __forceinline__ float la_diff_lo(float p1, float n1, float p2, float n2) {
    float l1, u1, l2, u2;
    la_wilson(p1, n1, l1, u1);
    la_wilson(p2, n2, l2, u2);
    return (p1 - p2) - sqrtf((p1 - l1) * (p1 - l1) + (u2 - p2) * (u2 - p2));
}
__global__ void la_b_decide_k(LaBatch a) {   // grid ceil(Nsd / 64), block 64
    const int sd = blockIdx.x * blockDim.x + threadIdx.x;
    if (sd == 0) atomicAdd(a.nwin, 1ull);
    if (sd >= a.Nsd) return;
    unsigned long long* c = a.cnt + (size_t) sd * kCnt;
    a.reset[sd] = 0;
    a.tick[sd] += 1;
    int L = a.lead[sd];
    const float nS = (float) c[0];
    if (!a.frozen[sd] && nS >= kMinPicks) {
        int best = -1;
        float bestp = -1.0f;
        const float pL = (float) c[3 + L] / nS;
        for (int k = 0; k < kSlots; ++k) {
            if (k == L) continue;
            const float p = (float) c[3 + k] / nS;
            if (p > bestp && la_diff_lo(p, nS, pL, nS) > 0.0f) { best = k; bestp = p; }
        }
        if (best >= 0) {
            L = best;
            a.lead[sd] = L;
            c[15] += 1;
            c[0] = 0;
            for (int k = 0; k < kSlots; ++k) c[3 + k] = 0;
            for (int k = 1; k < kSlots; ++k) a.live[sd * kSlots + k] = 1;
        }
    }
    if (!a.frozen[sd] && c[0] >= kFreezePicks) {   // 2000 picks without a change: the non-leader candidates stop training
        a.frozen[sd] = 1;
        for (int k = 1; k < kSlots; ++k)
            if (k != L) a.live[sd * kSlots + k] = 0;
    }
    const float nR = (float) c[1];
    if (L != 0 && nR >= kRecPicks) {   // the plain router beats the leader over the recent picks: back to the plain router
        const float p0 = (float) c[7] / nR, pL = (float) c[7 + L] / nR;
        // two failing blocks in a row: one 90% test per 500 picks over 144 (layer, head) pairs fires by chance about one time in 20
        c[17] = la_diff_lo(p0, nR, pL, nR) > 0.0f ? c[17] + 1 : 0;
        if (c[17] >= 2) {
            c[17] = 0;
            a.lead[sd] = 0;
            a.frozen[sd] = 0;
            a.reset[sd] = 1;
            c[16] += 1;
            // why it fired (the host logs the last one): the recent picks, the plain router's and the old leader's hits, the old leader's slot
            a.rinfo[sd * kSlots + 0] = (unsigned long long) nR;
            a.rinfo[sd * kSlots + 1] = c[7];
            a.rinfo[sd * kSlots + 2] = c[7 + L];
            a.rinfo[sd * kSlots + 3] = (unsigned long long) L;
            c[0] = 0;
            for (int k = 0; k < kSlots; ++k) c[3 + k] = 0;
            for (int k = 1; k < kSlots; ++k) {
                a.live[sd * kSlots + k] = 1;
                a.steps[sd * kSlots + k] = 0;
            }
        }
    }
    if (nR >= kRecPicks) {
        c[1] = 0;
        for (int k = 0; k < kSlots; ++k) c[7 + k] = 0;
    }
}
__global__ void la_b_reinit_k(LaBatch a) {   // grid (160, Nsd), block 256: a reset slot starts fresh (A random, B and moments zero)
    const int sd = blockIdx.y;
    if (!a.reset[sd]) return;
    const unsigned long long nre = a.cnt[(size_t) sd * kCnt + 16];
    const int stride = gridDim.x * blockDim.x;
    for (int c = 1; c < kSlots; ++c) {
        const int r = la_rank(c);
        const size_t oa = (size_t) sd * r * a.H, ob = (size_t) sd * kE * r;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < r * a.H; i += stride) {
            a.A[c][oa + i] = la_rand_pm(la_key(sd, c, i, nre));
            a.mA[c][oa + i] = 0.0f;
            a.vA[c][oa + i] = 0.0f;
        }
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < kE * r; i += stride) {
            a.B[c][ob + i] = 0.0f;
            a.mB[c][ob + i] = 0.0f;
            a.vB[c][ob + i] = 0.0f;
        }
    }
}
// the resident flags, from each layer's vram_tab: grid L, block 512.  Padding (e >= E) counts as resident, so it is never scored.
__global__ void la_res_k(const unsigned long long* const* tabs, int* res, int E) {
    const int m = blockIdx.x, e = threadIdx.x;
    res[m * kE + e] = (e >= E || tabs[m][e] != 0ull) ? 0 : -1;
}

struct LaLearn {
    int L = 0, H = 0, E = 0, lo = 3, Nd = kNd, Nsd = 0, Nset = 0;
    bool pending = false;        // a batch was launched and its done event is not yet waited on by a launch
    bool tabs_set = false;       // set_resident_tables was called
    cudaStream_t side = nullptr; // the batch's stream (non-blocking): never the stream the host waits on at a window's end
    cudaEvent_t ev_end = nullptr, ev_done = nullptr;
    float* A[kSlots] = {};       // slot c: [Nsd][r_c][H]
    float* B[kSlots] = {};       // slot c: [Nsd][512][r_c]
    float* mA[kSlots] = {};
    float* vA[kSlots] = {};
    float* mB[kSlots] = {};
    float* vB[kSlots] = {};
    float* x = nullptr;          // [L][8][H]
    float* u = nullptr;          // [Nsd][8][28]
    float* pred = nullptr;       // [Nsd][4][8][512]
    float* lab = nullptr;        // [L][8][512]
    int* rk = nullptr;           // [L][8][512]
    float* thr = nullptr;        // [L][8]
    float* g = nullptr;          // [Nset][8][512]
    float* du = nullptr;         // [Nset][8][16]
    int* act = nullptr;          // [Nset]
    int* tstep = nullptr;        // [Nset]
    int* steps = nullptr;        // [Nsd][4]
    int* tick = nullptr;         // [Nsd]
    int* live = nullptr;         // [Nsd][4]
    int* lead = nullptr;         // [Nsd]
    int* frozen = nullptr;       // [Nsd]
    int* reset = nullptr;        // [Nsd]
    int* res = nullptr;          // [L][512]: the resident flags (refreshed each window)
    const unsigned long long** tabs = nullptr;   // [L]: device pointers to each layer's vram_tab (device memory)
    unsigned long long* cnt = nullptr;   // [Nsd][kCnt]
    unsigned long long* nwin = nullptr;
    unsigned long long* rinfo = nullptr; // [Nsd][4]: the last reset's facts (LaBatch::rinfo)
    // host: the windows that found the batch pending when their launch came (the batch was not done)
    unsigned long long win_pending = 0, win_waited = 0;
    std::vector<unsigned long long> resets_seen;   // host: per (s,d), the resets already logged
    std::vector<char> logged;    // host: the (s,d) whose chosen r was logged
    size_t bytes = 0;
};
LaLearn* g_la = nullptr;

// frees every allocation of s (the ones that were made) and deletes it
void free_learner(LaLearn* s) {
    for (int c = 1; c < kSlots; ++c) {
        cudaFree(s->A[c]); cudaFree(s->B[c]); cudaFree(s->mA[c]); cudaFree(s->vA[c]); cudaFree(s->mB[c]); cudaFree(s->vB[c]);
    }
    cudaFree(s->x); cudaFree(s->u); cudaFree(s->pred); cudaFree(s->lab); cudaFree(s->rk); cudaFree(s->thr); cudaFree(s->g);
    cudaFree(s->du); cudaFree(s->act); cudaFree(s->tstep); cudaFree(s->steps); cudaFree(s->tick); cudaFree(s->live);
    cudaFree(s->lead); cudaFree(s->frozen); cudaFree(s->reset); cudaFree(s->res); cudaFree(s->tabs); cudaFree(s->cnt);
    cudaFree(s->nwin); cudaFree(s->rinfo);
    if (s->side != nullptr) cudaStreamDestroy(s->side);
    if (s->ev_end != nullptr) cudaEventDestroy(s->ev_end);
    if (s->ev_done != nullptr) cudaEventDestroy(s->ev_done);
    (void) cudaGetLastError();
    delete s;
}

}  // namespace

bool alloc(int n_layers, int n_embd, int n_exp, int lo, std::string& msg) {
    if (g_la != nullptr) { msg = "already on"; return true; }
    if (n_layers <= 0 || n_embd <= 0 || n_exp <= 0 || n_exp > kE) { msg = "bad shape"; return false; }
    LaLearn* s = new LaLearn();
    s->L = n_layers;
    s->H = n_embd;
    s->E = n_exp;
    s->lo = lo;
    s->Nsd = kNd * s->L;
    s->Nset = s->Nsd * kNSlots;
    bool ok = true;
    auto dalloc = [&](void** p, size_t bytes) {
        *p = nullptr;
        if (!ok) return;
        if (cudaMalloc(p, bytes) != cudaSuccess) { (void) cudaGetLastError(); ok = false; return; }
        cudaMemset(*p, 0, bytes);
        s->bytes += bytes;
    };
    for (int c = 1; c < kSlots; ++c) {
        const size_t ra = (size_t) s->Nsd * la_rank(c) * s->H * sizeof(float), rb = (size_t) s->Nsd * kE * la_rank(c) * sizeof(float);
        dalloc((void**) &s->A[c], ra);
        dalloc((void**) &s->B[c], rb);
        dalloc((void**) &s->mA[c], ra);
        dalloc((void**) &s->vA[c], ra);
        dalloc((void**) &s->mB[c], rb);
        dalloc((void**) &s->vB[c], rb);
    }
    dalloc((void**) &s->x, (size_t) s->L * kTok * s->H * sizeof(float));
    dalloc((void**) &s->u, (size_t) s->Nsd * kTok * kRows * sizeof(float));
    dalloc((void**) &s->pred, (size_t) s->Nsd * kSlots * kTok * kE * sizeof(float));
    dalloc((void**) &s->lab, (size_t) s->L * kTok * kE * sizeof(float));
    dalloc((void**) &s->rk, (size_t) s->L * kTok * kE * sizeof(int));
    dalloc((void**) &s->thr, (size_t) s->L * kTok * sizeof(float));
    dalloc((void**) &s->g, (size_t) s->Nset * kTok * kE * sizeof(float));
    dalloc((void**) &s->du, (size_t) s->Nset * kTok * 16 * sizeof(float));
    dalloc((void**) &s->act, (size_t) s->Nset * sizeof(int));
    dalloc((void**) &s->tstep, (size_t) s->Nset * sizeof(int));
    dalloc((void**) &s->steps, (size_t) s->Nsd * kSlots * sizeof(int));
    dalloc((void**) &s->tick, (size_t) s->Nsd * sizeof(int));
    dalloc((void**) &s->live, (size_t) s->Nsd * kSlots * sizeof(int));
    dalloc((void**) &s->lead, (size_t) s->Nsd * sizeof(int));
    dalloc((void**) &s->frozen, (size_t) s->Nsd * sizeof(int));
    dalloc((void**) &s->reset, (size_t) s->Nsd * sizeof(int));
    dalloc((void**) &s->res, (size_t) s->L * kE * sizeof(int));
    dalloc((void**) &s->tabs, (size_t) s->L * sizeof(void*));
    dalloc((void**) &s->cnt, (size_t) s->Nsd * kCnt * sizeof(unsigned long long));
    dalloc((void**) &s->nwin, sizeof(unsigned long long));
    dalloc((void**) &s->rinfo, (size_t) s->Nsd * kSlots * sizeof(unsigned long long));
    if (ok && (cudaStreamCreateWithFlags(&s->side, cudaStreamNonBlocking) != cudaSuccess ||
               cudaEventCreateWithFlags(&s->ev_end, cudaEventDisableTiming) != cudaSuccess ||
               cudaEventCreateWithFlags(&s->ev_done, cudaEventDisableTiming) != cudaSuccess)) {
        (void) cudaGetLastError();
        ok = false;
    }
    if (!ok) {
        msg = "no room on the GPU for it (or no stream and events)";
        free_learner(s);
        return false;
    }
    // the padding of the label rows: -1e30 for the experts past E (labels are copied into the first E of each row only)
    std::vector<float> hl((size_t) s->L * kTok * kE, kPad);
    cudaMemcpy(s->lab, hl.data(), hl.size() * sizeof(float), cudaMemcpyHostToDevice);
    // the initial draws (the reset's function, nothing learned yet): A random, B zero; slot 0 has no parameters
    for (int c = 1; c < kSlots; ++c) {
        const int r = la_rank(c);
        std::vector<float> hA((size_t) s->Nsd * r * s->H);
        for (int sd = 0; sd < s->Nsd; ++sd)
            for (int i = 0; i < r * s->H; ++i) hA[(size_t) sd * r * s->H + i] = la_rand_pm(la_key(sd, c, i, 0));
        cudaMemcpy(s->A[c], hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice);
    }
    std::vector<int> live((size_t) s->Nsd * kSlots, 1);
    cudaMemcpy(s->live, live.data(), live.size() * sizeof(int), cudaMemcpyHostToDevice);
    s->logged.assign((size_t) s->Nsd, 0);
    s->resets_seen.assign((size_t) s->Nsd, 0);
    g_la = s;
    char buf[160];
    std::snprintf(buf, sizeof buf, "%d layers, %d experts, %zu MiB of VRAM", n_layers, n_exp, s->bytes >> 20);
    msg = buf;
    return true;
}

bool active() { return g_la != nullptr; }

void release() {
    if (g_la == nullptr) return;
    free_learner(g_la);
    g_la = nullptr;
}

bool set_resident_tables(const unsigned long long* const* host_ptrs, int L) {
    if (!active() || L != g_la->L || host_ptrs == nullptr) return false;
    LaLearn& s = *g_la;
    if (cudaMemcpy(s.tabs, host_ptrs, (size_t) L * sizeof(void*), cudaMemcpyHostToDevice) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    s.tabs_set = true;
    return true;
}

// the forward, on cs (in the window's graph)
void forward(int s, int dist, const float* x, float* gl, int n, cudaStream_t cs) {
    const int di = dist_index(dist);
    if (!active() || n < 1 || n > kTok || di < 0 || s < 0 || s + dist >= g_la->L) return;
    LaLearn& S = *g_la;
    const int sd = s * kNd + di;
    LaFwdArgs a{};
    a.x = x;
    a.xsave = di == 0 ? S.x + (size_t) s * kTok * S.H : nullptr;   // the first distance saves the input
    a.u = S.u + (size_t) sd * kTok * kRows;
    a.pred = S.pred + (size_t) sd * kSlots * kTok * kE;
    a.route = gl;
    a.lead = S.lead + sd;
    a.n = n;
    a.H = S.H;
    a.E = S.E;
    for (int c = 1; c < kSlots; ++c) {
        a.A[c] = S.A[c] + (size_t) sd * la_rank(c) * S.H;
        a.B[c] = S.B[c] + (size_t) sd * kE * la_rank(c);
    }
    strata_ds4_launch(la_fwd_u_k, dim3(kRows), 256, 0, cs, a);
    strata_ds4_launch(la_fwd_pred_k, dim3((n * kE + 255) / 256), 256, 0, cs, a);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

// the label of layer l for this window (on cs, in the graph): its router logits, kept for the batch
void label(int l, const float* glog, int n, cudaStream_t cs) {
    if (!active() || n < 1 || n > kTok || l < 0 || l >= g_la->L) return;
    LaLearn& S = *g_la;
    if (cudaMemcpy2DAsync(S.lab + (size_t) l * kTok * kE, kE * sizeof(float), glog, kRoute * sizeof(float), (size_t) S.E * sizeof(float),
                          (size_t) n, cudaMemcpyDeviceToDevice, cs) != cudaSuccess) {
        const auto error = cudaGetLastError();
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

// before a window's graph is launched on cs: the previous window's batch must be done (its weights, its inputs)
void pre_launch(cudaStream_t cs) {
    if (!active() || !g_la->pending) return;
    LaLearn& s = *g_la;
    s.win_pending += 1;
    if (cudaEventQuery(s.ev_done) != cudaSuccess) {   // cudaErrorNotReady: the batch is still running
        (void) cudaGetLastError();
        s.win_waited += 1;
    }
    cudaStreamWaitEvent(cs, s.ev_done, 0);
}

// after a learning window's graph is launched on cs: the batch on the side stream, which waits for the window's end and records done
void post_launch(cudaStream_t cs, int n) {
    if (!active() || n < 1 || n > kTok || !g_la->tabs_set) return;
    LaLearn& s = *g_la;
    cudaEventRecord(s.ev_end, cs);
    cudaStreamWaitEvent(s.side, s.ev_end, 0);
    cudaStream_t sd_s = s.side;
    strata_ds4_launch(la_res_k, dim3(s.L), kE, 0, sd_s, s.tabs, s.res, s.E);
    LaBatch a{};
    for (int c = 1; c < kSlots; ++c) {
        a.A[c] = s.A[c]; a.B[c] = s.B[c]; a.mA[c] = s.mA[c]; a.vA[c] = s.vA[c]; a.mB[c] = s.mB[c]; a.vB[c] = s.vB[c];
    }
    a.x = s.x; a.u = s.u; a.pred = s.pred; a.lab = s.lab; a.res = s.res;
    a.rk = s.rk; a.thr = s.thr; a.g = s.g; a.du = s.du; a.act = s.act; a.tstep = s.tstep;
    a.steps = s.steps; a.live = s.live; a.lead = s.lead; a.frozen = s.frozen; a.reset = s.reset; a.tick = s.tick;
    a.cnt = s.cnt; a.nwin = s.nwin; a.rinfo = s.rinfo;
    for (int i = 0; i < kNd; ++i) a.dist[i] = kDist[i];
    a.L = s.L; a.H = s.H; a.lo = s.lo; a.n = n; a.Nd = kNd; a.Nsd = s.Nsd; a.Nset = s.Nset;
    strata_ds4_launch(la_b_rank_k, dim3(n, s.L), kE, 0, sd_s, a);
    strata_ds4_launch(la_b_score_k, dim3(n, s.Nsd, kSlots), kE, 0, sd_s, a);
    strata_ds4_launch(la_b_prep_k, dim3((s.Nset + 63) / 64), 64, 0, sd_s, a);
    strata_ds4_launch(la_b_grad_k, dim3(n, s.Nset), kE, 0, sd_s, a);
    strata_ds4_launch(la_b_upd_b_k, dim3(32, s.Nset), 256, 0, sd_s, a);
    strata_ds4_launch(la_b_upd_a_k, dim3((s.H + 255) / 256, s.Nset), 256, 0, sd_s, a);
    strata_ds4_launch(la_b_decide_k, dim3((s.Nsd + 63) / 64), 64, 0, sd_s, a);
    strata_ds4_launch(la_b_reinit_k, dim3(160, s.Nsd), 256, 0, sd_s, a);
    cudaEventRecord(s.ev_done, sd_s);
    s.pending = true;
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

// the host's reads, between requests: the counters, the leaders and the freezes
static bool read_state(LaLearn& s, std::vector<unsigned long long>& cnt, std::vector<int>& lead, std::vector<int>& frozen) {
    if (s.pending) cudaEventSynchronize(s.ev_done);   // the last batch (side stream) is not waited by cudaMemcpy: wait for it here
    cnt.assign((size_t) s.Nsd * kCnt, 0);
    lead.assign((size_t) s.Nsd, 0);
    frozen.assign((size_t) s.Nsd, 0);
    return cudaMemcpy(cnt.data(), s.cnt, cnt.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost) == cudaSuccess &&
           cudaMemcpy(lead.data(), s.lead, lead.size() * sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess &&
           cudaMemcpy(frozen.data(), s.frozen, frozen.size() * sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess;
}

bool counters(int l, int dist, int& leader, unsigned long long& picks, unsigned long long hits[4]) {
    const int di = dist_index(dist);
    if (!active() || di < 0 || l < 0 || l >= g_la->L) return false;
    LaLearn& s = *g_la;
    std::vector<unsigned long long> cnt;
    std::vector<int> lead, frozen;
    if (!read_state(s, cnt, lead, frozen)) { (void) cudaGetLastError(); return false; }
    const size_t sd = (size_t) l * kNd + di;
    leader = lead[sd];
    picks = cnt[sd * kCnt + 2];
    for (int c = 0; c < kSlots; ++c) hits[c] = cnt[sd * kCnt + 11 + c];
    return true;
}

std::string report() {
    if (g_la == nullptr) return "";
    LaLearn& s = *g_la;
    std::vector<unsigned long long> cnt;
    std::vector<int> lead, frozen;
    if (!read_state(s, cnt, lead, frozen)) { (void) cudaGetLastError(); return ""; }
    std::string out;
    double hit_l[kNd] = {}, hit_p[kNd] = {}, tot[kNd] = {};
    unsigned long long resets = 0;
    for (int sd = 0; sd < s.Nsd; ++sd) {
        const int dd = sd % kNd, L = lead[sd];
        const unsigned long long* c = cnt.data() + (size_t) sd * kCnt;
        resets += c[16];
        tot[dd] += (double) c[0];
        hit_l[dd] += (double) c[3 + L];
        hit_p[dd] += (double) c[3];
        if (frozen[sd] && !s.logged[sd]) {   // the chosen r of each layer, logged once
            s.logged[sd] = 1;
            out += "ds4: look-ahead learner: layer " + std::to_string(sd / kNd) + " d=" + std::to_string(kDist[dd]) + " chose r=" +
                   std::to_string(la_rank(L)) + " (" + std::to_string(c[15]) + " changes, " + std::to_string(c[2]) + " picks)\n";
        }
    }
    auto hist_of = [&](int dd) {
        std::string h = "{";
        for (int r : {0, 4, 8, 16}) {
            int k = 0;
            for (int sd = dd; sd < s.Nsd; sd += kNd) k += la_rank(lead[sd]) == r;
            if (k > 0) h += (h.size() > 1 ? "," : "") + std::to_string(r) + ":" + std::to_string(k);
        }
        return h + "}";
    };
    auto pct = [](double a, double b) {
        if (!(b > 0)) return std::string("-");
        char buf[16];
        std::snprintf(buf, sizeof buf, "%.1f%%", 100.0 * a / b);
        return std::string(buf);
    };
    out += "ds4: look-ahead learner:";
    for (int dd = 0; dd < kNd; ++dd)
        out += (dd > 0 ? ", " : " layers ") + std::string("d=") + std::to_string(kDist[dd]) + " r=" + hist_of(dd);
    out += ", leader hit rate";
    for (int dd = 0; dd < kNd; ++dd)
        out += (dd > 0 ? ", " : " ") + std::string("d=") + std::to_string(kDist[dd]) + " " + pct(hit_l[dd], tot[dd]) + " vs plain " +
               pct(hit_p[dd], tot[dd]);
    out += " (non-resident ranks < lo; picks since each layer's last change), resets " + std::to_string(resets) + "\n";
    // the resets: each (s,d)'s last one, logged when its count has grown (the recent picks, the plain router's hits, the old leader's)
    if (s.resets_seen.size() != (size_t) s.Nsd) s.resets_seen.assign((size_t) s.Nsd, 0);
    std::vector<unsigned long long> ri((size_t) s.Nsd * kSlots, 0);
    if (cudaMemcpy(ri.data(), s.rinfo, ri.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost) != cudaSuccess) (void) cudaGetLastError();
    for (int sd = 0; sd < s.Nsd; ++sd) {
        const unsigned long long n_rs = cnt[(size_t) sd * kCnt + 16];
        if (n_rs == s.resets_seen[(size_t) sd]) continue;
        s.resets_seen[(size_t) sd] = n_rs;
        const unsigned long long* q = ri.data() + (size_t) sd * kSlots;
        out += "ds4: look-ahead learner: layer " + std::to_string(sd / kNd) + " d=" + std::to_string(kDist[sd % kNd]) +
               " reset (" + std::to_string(n_rs) + " so far): at " + std::to_string(q[0]) + " recent picks the plain router had " +
               std::to_string(q[1]) + " hits, the leader r=" + std::to_string(la_rank((int) q[3])) + " " + std::to_string(q[2]) + "\n";
    }
    char wbuf[200];
    std::snprintf(wbuf, sizeof wbuf, "ds4: look-ahead learner: the main stream found the batch pending in %llu of %llu windows\n",
                  s.win_waited, s.win_pending);
    return out + wbuf;
}

// The file: "LAL1", version 4, n_layers, n_embd, n_exp, the number of distances, the distances, the three correction ranks.  Then per
// (s,d) the leader's rank (0, 4, 8 or 16; | 256 when the (s,d) is frozen); then for each (s,d) with a rank, the leader's A (r x n_embd)
// and B (512 x r) as fp16.  Written to a temporary file and renamed.
static std::vector<uint32_t> file_header(const LaLearn& s) {
    std::vector<uint32_t> h = {0x314C414Cu, 4u, (uint32_t) s.L, (uint32_t) s.H, (uint32_t) s.E, (uint32_t) kNd};
    for (int i = 0; i < kNd; ++i) h.push_back((uint32_t) kDist[i]);
    for (int c = 1; c < kSlots; ++c) h.push_back((uint32_t) la_rank(c));
    return h;
}
bool save(const char* path, std::string& msg) {
    if (!active()) { msg = "off"; return false; }
    LaLearn& s = *g_la;
    std::vector<unsigned long long> cnt;
    std::vector<int> lead, frozen;
    if (!read_state(s, cnt, lead, frozen)) { (void) cudaGetLastError(); msg = "cannot read the GPU state"; return false; }
    const std::string tmp = std::string(path) + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (f == nullptr) { msg = "cannot write " + tmp; return false; }
    const std::vector<uint32_t> hdr = file_header(s);
    bool ok = std::fwrite(hdr.data(), sizeof(uint32_t), hdr.size(), f) == hdr.size();
    for (int sd = 0; ok && sd < s.Nsd; ++sd) {
        const uint32_t r = (uint32_t) la_rank(lead[sd]) | (frozen[sd] ? 256u : 0u);
        ok = std::fwrite(&r, sizeof r, 1, f) == 1;
    }
    for (int sd = 0; ok && sd < s.Nsd; ++sd) {
        const int c = lead[sd], r = la_rank(c);
        if (c == 0) continue;
        std::vector<float> fa((size_t) r * s.H), fb((size_t) kE * r);
        std::vector<uint16_t> ha(fa.size()), hb(fb.size());
        ok = cudaMemcpy(fa.data(), s.A[c] + (size_t) sd * r * s.H, fa.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess &&
             cudaMemcpy(fb.data(), s.B[c] + (size_t) sd * kE * r, fb.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
        for (size_t i = 0; i < fa.size(); ++i) ha[i] = __half_as_ushort(__float2half(fa[i]));
        for (size_t i = 0; i < fb.size(); ++i) hb[i] = __half_as_ushort(__float2half(fb[i]));
        ok = ok && std::fwrite(ha.data(), sizeof(uint16_t), ha.size(), f) == ha.size() &&
             std::fwrite(hb.data(), sizeof(uint16_t), hb.size(), f) == hb.size();
    }
    const bool closed = std::fclose(f) == 0;
    if (!ok || !closed) { std::remove(tmp.c_str()); msg = "write failed"; return false; }
    if (std::rename(tmp.c_str(), path) != 0) { std::remove(tmp.c_str()); msg = "rename failed"; return false; }
    return true;
}

// Loads a file saved for this learner's shape (the header must match).  Each (s,d) takes the file's leader in its rank's slot; the other
// slots start from the leader's first rows and columns (the rest random / zero), all live, with fresh moments.  Called between requests,
// before any window.
bool load(const char* path, std::string& msg) {
    if (!active()) { msg = "off"; return false; }
    LaLearn& s = *g_la;
    FILE* f = std::fopen(path, "rb");
    if (f == nullptr) { msg = "no file"; return false; }
    const std::vector<uint32_t> want = file_header(s);
    std::vector<uint32_t> hdr(want.size(), 0u);
    bool ok = std::fread(hdr.data(), sizeof(uint32_t), hdr.size(), f) == hdr.size() && hdr == want;
    if (!ok) { std::fclose(f); msg = "not a LAL1 file for this learner (layers, n_embd, n_exp, distances or ranks differ)"; return false; }
    std::vector<int> rl((size_t) s.Nsd, 0), fzv((size_t) s.Nsd, 0);
    for (int sd = 0; ok && sd < s.Nsd; ++sd) {
        uint32_t w = 0;
        ok = std::fread(&w, sizeof w, 1, f) == 1;
        const uint32_t r = w & 255u;
        ok = ok && (r == 0 || r == 4 || r == 8 || r == 16) && (w >> 9) == 0;
        rl[(size_t) sd] = (int) r;
        fzv[(size_t) sd] = (int) ((w >> 8) & 1u);
    }
    std::vector<float> hA[kSlots], hB[kSlots];
    for (int c = 1; c < kSlots; ++c) {
        const int r = la_rank(c);
        hA[c].assign((size_t) s.Nsd * r * s.H, 0.0f);
        hB[c].assign((size_t) s.Nsd * kE * r, 0.0f);
        for (int sd = 0; sd < s.Nsd; ++sd)
            for (int i = 0; i < r * s.H; ++i) hA[c][(size_t) sd * r * s.H + i] = la_rand_pm(la_key(sd, c, i, 0));
    }
    std::vector<int> lead((size_t) s.Nsd, 0);
    for (int sd = 0; ok && sd < s.Nsd; ++sd) {   // the data follow the words in the same order
        const int rL = rl[(size_t) sd];
        if (rL == 0) continue;
        const int cL = rL == 4 ? 1 : (rL == 8 ? 2 : 3);
        lead[(size_t) sd] = cL;
        std::vector<uint16_t> ra((size_t) rL * s.H), rb((size_t) kE * rL);
        ok = std::fread(ra.data(), sizeof(uint16_t), ra.size(), f) == ra.size() && std::fread(rb.data(), sizeof(uint16_t), rb.size(), f) == rb.size();
        if (!ok) break;
        std::vector<float> al(ra.size()), bl(rb.size());
        for (size_t i = 0; i < ra.size(); ++i) al[i] = __half2float(__ushort_as_half(ra[i]));
        for (size_t i = 0; i < rb.size(); ++i) bl[i] = __half2float(__ushort_as_half(rb[i]));
        for (int c = 1; c < kSlots; ++c) {
            const int r = la_rank(c), m = r < rL ? r : rL;
            for (int kk = 0; kk < m; ++kk) {
                for (int h = 0; h < s.H; ++h) hA[c][((size_t) sd * r + kk) * s.H + h] = al[(size_t) kk * s.H + h];
                for (int e = 0; e < kE; ++e) hB[c][((size_t) sd * kE + e) * r + kk] = bl[(size_t) e * rL + kk];
            }
        }
    }
    std::fclose(f);
    if (!ok) { msg = "the file is cut short or damaged"; return false; }
    for (int c = 1; c < kSlots; ++c) {
        const size_t na = hA[c].size() * sizeof(float), nb = hB[c].size() * sizeof(float);
        cudaMemcpy(s.A[c], hA[c].data(), na, cudaMemcpyHostToDevice);
        cudaMemcpy(s.B[c], hB[c].data(), nb, cudaMemcpyHostToDevice);
        cudaMemset(s.mA[c], 0, na);
        cudaMemset(s.vA[c], 0, na);
        cudaMemset(s.mB[c], 0, nb);
        cudaMemset(s.vB[c], 0, nb);
    }
    std::vector<int> live((size_t) s.Nsd * kSlots, 1), zero((size_t) s.Nsd * kSlots, 0), zsd((size_t) s.Nsd, 0);
    // a frozen (s,d) keeps only its leader training (every 4th window); its candidates are not restarted
    for (int sd = 0; sd < s.Nsd; ++sd)
        if (fzv[(size_t) sd])
            for (int k = 1; k < kSlots; ++k) live[(size_t) sd * kSlots + k] = lead[(size_t) sd] == k ? 1 : 0;
    cudaMemcpy(s.lead, lead.data(), lead.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(s.live, live.data(), live.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(s.steps, zero.data(), zero.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(s.frozen, fzv.data(), fzv.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(s.reset, zsd.data(), zsd.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(s.tick, zsd.data(), zsd.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(s.cnt, 0, (size_t) s.Nsd * kCnt * sizeof(unsigned long long));
    s.logged.assign((size_t) s.Nsd, 0);
    s.resets_seen.assign((size_t) s.Nsd, 0);
    if (cudaGetLastError() != cudaSuccess) { msg = "the copy to the GPU failed"; return false; }
    int nfrozen = 0;
    for (int v : fzv) nfrozen += v;
    msg = "loaded, " + std::to_string(nfrozen) + " of " + std::to_string(s.Nsd) + " frozen";
    return true;
}

}  // namespace strata::ds4::la
