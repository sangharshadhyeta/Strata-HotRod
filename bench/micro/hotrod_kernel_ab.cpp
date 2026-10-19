// bench/micro/hotrod_kernel_ab.cpp - A/B bit-exactness of three hot-rod kernel switches on seeded random inputs.
//
// One process runs one case under the environment it was started with; the switches are read once per process
// (static caches), so bench/micro/hotrod_kernel_ab.sh runs each setting in its own process and compares the raw
// output files byte for byte.  No llama.cpp oracle: synthetic geometry from the engine's own call sites.
//
//   --case gr      fused_gr_read_multi (the hyper-connection read the decode uses): n_embd 2560, hc 4, hc_lr 320,
//                  T = 4 tokens, BF16 weights.  Three reads: layer read with the pending write and injection,
//                  the same without the pending write, and the final mixer (no injection).  Env: STRATA_GR_DOWN_WPC
//                  (8, 4, 2); the variant is whatever fused_gr_check chose on this card (printed; 3 = staged).
//   --case router  the 512 x 2560 BF16 router GEMV + top-10 for T = 4 tokens, fp32 input with ties (seven identical
//                  expert rows, and one all-zero token that ties all 512 logits).  Env: STRATA_ROUTER_FUSED=1 takes the
//                  one-launch kernel, unset the two launches (bf16_gemv_fp32_mmvf_multi + native_router_top10_multi).
//                  Output: fp32 logits, ids, weights.
//   --case topk    qsa_block_topk (the QSA decode selection, no block count, selection width 2051) at max_blocks 16384,
//                  33792 and 65538, four queries, three score sets (continuous, 64 levels, all equal).  Env:
//                  STRATA_TOPK_SPLIT=1 takes the split kernel, unset the default.  Output: the selected ids of each
//                  query in their order.  Also checks the dispatched ids against qsa_block_topk_ref in-process.
//
// Usage: hotrod_kernel_ab --case gr|router|topk --seed N --out FILE
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;

namespace {

[[noreturn]] void die(const char* what, cudaError_t e) {
    std::fprintf(stderr, "hotrod_kernel_ab: %s: %s\n", what, cudaGetErrorString(e));
    std::exit(2);
}
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) die(what, e);
}

uint16_t bf16_of(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t) (u >> 16);
}

template <typename T>
T* upload(const std::vector<T>& h) {
    void* p = nullptr;
    ck(cudaMalloc(&p, h.size() * sizeof(T) + 64), "malloc");
    ck(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return (T*) p;
}

template <typename T>
T* zeros(size_t n) {
    void* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 64), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 64), "memset");
    return (T*) p;
}

template <typename T>
void dump(FILE* f, const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    std::fwrite(h.data(), sizeof(T), n, f);
}

void sync(cudaStream_t st, const char* what) {
    ck(cudaStreamSynchronize(st), what);
    ck(cudaGetLastError(), what);
}

// ---------------------------------------------------------------- gr
int run_gr(uint32_t seed, FILE* out) {
    const int64_t N = 2560, HC = 4, LR = 320, D = HC * N;
    const int T = 4;
    const float eps = 1e-6f;
    if (!k::fused_gr_supported(N, HC, LR)) {
        std::fprintf(stderr, "fused_gr_supported says no for this geometry\n");
        return 2;
    }
    std::mt19937 rng(seed);
    std::normal_distribution<float> g(0.f, 1.f);
    std::vector<float> wn((size_t) D);
    for (auto& x : wn) x = 1.0f + 0.1f * g(rng);
    std::vector<uint16_t> wd((size_t) (LR * D)), wu((size_t) (D * LR)), wi((size_t) (HC * D));
    for (auto& x : wd) x = bf16_of(g(rng) * 0.05f);
    for (auto& x : wu) x = bf16_of(g(rng) * 0.05f);
    for (auto& x : wi) x = bf16_of(g(rng) * 0.02f);
    std::vector<float> R((size_t) (T * D)), bo((size_t) (T * N)), ip((size_t) (T * HC));
    for (int t = 0; t < T; ++t)
        for (int64_t c = 0; c < HC; ++c)
            for (int64_t d = 0; d < N; ++d) R[(size_t) (t * D + c * N + d)] = g(rng) * (float) (1 << (2 * c));
    for (auto& x : bo) x = g(rng);
    for (auto& x : ip) x = 3.0f * g(rng);

    float* d_wn = upload(wn);
    uint16_t *d_wd = upload(wd), *d_wu = upload(wu), *d_wi = upload(wi);
    float *d_R = upload(R), *d_bo = upload(bo), *d_ip = upload(ip);
    float* d_Rout = zeros<float>((size_t) (T * D));
    float* d_mixed = zeros<float>((size_t) (T * N));
    float* d_lo = zeros<float>((size_t) (T * LR));
    float* d_rs = zeros<float>((size_t) (T * HC));
    float* d_inj = zeros<float>((size_t) (T * HC));
    float* d_xn = zeros<float>((size_t) (T * D));
    cudaStream_t st = nullptr;
    ck(cudaStreamCreate(&st), "stream");

    k::fused_gr_check();   // runs every variant against the plain read on this card and keeps the newest that agrees
    const int var = k::fused_gr_variant();
    std::printf("gr variant %d (1 plain, 2 split, 3 staged) DOWN_WPC=%s\n", var,
                std::getenv("STRATA_GR_DOWN_WPC") ? std::getenv("STRATA_GR_DOWN_WPC") : "(unset)");

    for (int pass = 0; pass < 3; ++pass) {
        const bool apply = pass != 1;
        const bool inject = pass != 2;
        ck(cudaMemset(d_Rout, 0, (size_t) (T * D) * 4), "memset");
        ck(cudaMemset(d_mixed, 0, (size_t) (T * N) * 4), "memset");
        ck(cudaMemset(d_lo, 0, (size_t) (T * LR) * 4), "memset");
        ck(cudaMemset(d_rs, 0, (size_t) (T * HC) * 4), "memset");
        ck(cudaMemset(d_inj, 0, (size_t) (T * HC) * 4), "memset");
        k::FusedGrArgs a[k::kFusedGrMaxT];
        for (int t = 0; t < T; ++t) {
            k::FusedGrArgs& x = a[t];
            x.R = d_R + (size_t) t * D;
            x.R_out = d_Rout + (size_t) t * D;
            x.apply = apply;
            x.bo_prev = d_bo + (size_t) t * N;
            x.inj_prev = d_ip + (size_t) t * HC;
            x.w_norm = d_wn;
            x.w_down = d_wd;
            x.w_up = d_wu;
            x.w_inject = inject ? d_wi : nullptr;
            x.eps = eps;
            x.lo = d_lo + (size_t) t * LR;
            x.rs = d_rs + (size_t) t * HC;
            x.inject_out = d_inj + (size_t) t * HC;
            x.mixed = d_mixed + (size_t) t * N;
        }
        k::fused_gr_read_multi(a, T, d_xn, st);
        sync(st, "fused_gr_read_multi");
        dump(out, d_mixed, (size_t) (T * N));
        dump(out, d_lo, (size_t) (T * LR));
        dump(out, d_rs, (size_t) (T * HC));
        dump(out, d_inj, (size_t) (T * HC));
        dump(out, d_Rout, (size_t) (T * D));
    }
    cudaStreamDestroy(st);
    cudaFree(d_wn); cudaFree(d_wd); cudaFree(d_wu); cudaFree(d_wi); cudaFree(d_R); cudaFree(d_bo); cudaFree(d_ip);
    cudaFree(d_Rout); cudaFree(d_mixed); cudaFree(d_lo); cudaFree(d_rs); cudaFree(d_inj); cudaFree(d_xn);
    return 0;
}

// ---------------------------------------------------------------- router
int run_router(uint32_t seed, FILE* out) {
    const int64_t NI = 2560, NO = 512;
    const int T = 4, K = 10;
    std::mt19937 rng(seed);
    std::normal_distribution<float> g(0.f, 1.f);
    std::vector<float> x((size_t) (T * NI));
    for (auto& v : x) v = g(rng);
    for (int64_t d = 0; d < NI; ++d) x[(size_t) (2 * NI + d)] = 0.0f;   // token 2: every logit is exactly 0
    std::vector<uint16_t> w((size_t) (NO * NI));
    for (auto& v : w) v = bf16_of(g(rng) * 0.05f);
    // seven experts with the bits of expert 0's row: exact ties in the logits for every token
    for (int64_t e : {17, 40, 41, 200, 333, 511})
        std::memcpy(&w[(size_t) (e * NI)], &w[0], (size_t) NI * sizeof(uint16_t));

    float* d_x = upload(x);
    uint16_t* d_w = upload(w);
    float* d_logits = zeros<float>((size_t) (T * NO));
    int32_t* d_ids = zeros<int32_t>((size_t) (T * K));
    float* d_wt = zeros<float>((size_t) (T * K));
    cudaStream_t st = nullptr;
    ck(cudaStreamCreate(&st), "stream");

    if (k::native_router_fused_on()) {
        k::native_router_fused_init();
        if (!k::native_router_fused_supported(NI, NO, T, d_x, NI, d_w)) {
            std::fprintf(stderr, "native_router_fused_supported says no\n");
            return 2;
        }
        std::printf("router path: fused (STRATA_ROUTER_FUSED=1)\n");
        k::native_router_gemv_top10_fused(d_x, NI, d_w, d_logits, d_ids, d_wt, NI, NO, T, st);
    } else {
        std::printf("router path: two launches\n");
        k::bf16_gemv_fp32_mmvf_multi(d_x, NI, d_w, d_logits, NO, NI, NO, T, st);
        k::native_router_top10_multi(d_logits, d_ids, d_wt, T, st);
    }
    sync(st, "router");
    dump(out, d_logits, (size_t) (T * NO));
    dump(out, d_ids, (size_t) (T * K));
    dump(out, d_wt, (size_t) (T * K));
    cudaStreamDestroy(st);
    cudaFree(d_x); cudaFree(d_w); cudaFree(d_logits); cudaFree(d_ids); cudaFree(d_wt);
    return 0;
}

// ---------------------------------------------------------------- topk
int run_topk(uint32_t seed, FILE* out) {
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t nq = 4;
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);   // the decode's cap_ (verify.cpp)
    // capacities giving max_blocks = capacity / idx_block + 2 = 16384, 33792 and 65538
    const int64_t capacities[3] = {65528, 135160, 262144};
    int bad = 0;
    for (int64_t capacity : capacities) {
        const int64_t max_blocks = capacity / s.idx_block + 2;
        const int64_t ctx = capacity;   // the four queries sit at the end of a full context
        std::vector<int32_t> steps((size_t) (nq * k::kStepCount));
        for (int64_t i = 0; i < nq; ++i) {
            int32_t* st = steps.data() + i * k::kStepCount;
            const int64_t pos = ctx - nq + i;
            st[k::kStepPos] = (int32_t) pos;
            st[k::kStepNKv] = (int32_t) (pos + 1);
            st[k::kStepNBid] = (int32_t) ((pos + 1) / s.idx_block);
            st[k::kStepWidth] = (int32_t) k::qsa_selection_width(pos + 1, s);
        }
        std::mt19937 rng((uint32_t) (seed * 1000003u + (uint32_t) capacity));
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> cont((size_t) (nq * max_blocks)), lev(cont.size()), eq(cont.size(), 0.25f);
        for (int64_t i = 0; i < nq; ++i) {
            const int32_t* st = steps.data() + i * k::kStepCount;
            float* row = cont.data() + i * max_blocks;
            for (int64_t b = 0; b < max_blocks; ++b) row[b] = nd(rng);
            if (st[k::kStepNKv] % s.idx_block != 0) row[st[k::kStepNBid]] += 1e9f;   // the tail block, as the scorers leave it
        }
        for (size_t j = 0; j < cont.size(); ++j)   // 64 levels: many equal scores around the threshold
            lev[j] = cont[j] > 1e8f ? cont[j] : std::floor(cont[j] * 8.0f) / 8.0f;

        int32_t* d_steps = upload(steps);
        float* d_scores = upload(cont);
        int32_t* d_ids = zeros<int32_t>((size_t) (nq * cap));
        int32_t* d_ref = zeros<int32_t>((size_t) (nq * cap));
        cudaStream_t st = nullptr;
        ck(cudaStreamCreate(&st), "stream");
        const std::vector<float>* sets[3] = {&cont, &lev, &eq};
        const char* names[3] = {"continuous", "64 levels", "all equal"};
        for (int si = 0; si < 3; ++si) {
            float* d_sc = si == 0 ? d_scores : upload(*sets[si]);
            // decode call: no block count, as verify.cpp's decode window makes it
            k::qsa_block_topk(d_sc, d_steps, nq, max_blocks, cap, s, d_ids, st);
            k::qsa_block_topk_ref(d_sc, d_steps, nq, max_blocks, cap, s, d_ref, st);
            sync(st, "qsa_block_topk");
            std::vector<int32_t> a((size_t) (nq * cap)), b(a.size());
            ck(cudaMemcpy(a.data(), d_ids, a.size() * 4, cudaMemcpyDeviceToHost), "download");
            ck(cudaMemcpy(b.data(), d_ref, b.size() * 4, cudaMemcpyDeviceToHost), "download");
            int64_t same = 0;
            for (int64_t i = 0; i < nq; ++i) {
                const int64_t w = steps[(size_t) (i * k::kStepCount + k::kStepWidth)];
                same += std::equal(a.begin() + i * cap, a.begin() + i * cap + w, b.begin() + i * cap);
                // the raw output: the selected ids of this query, in the order the kernel wrote them
                std::fwrite(a.data() + i * cap, sizeof(int32_t), (size_t) w, out);
            }
            if (same != nq) ++bad;
            std::printf("capacity %lld max_blocks %lld %-10s: dispatched == ref for %lld/%lld queries (width %d)\n",
                        (long long) capacity, (long long) max_blocks, names[si], (long long) same, (long long) nq,
                        (int) steps[k::kStepWidth]);
            if (si > 0) cudaFree(d_sc);
        }
        cudaStreamDestroy(st);
        cudaFree(d_steps); cudaFree(d_scores); cudaFree(d_ids); cudaFree(d_ref);
    }
    if (bad) std::printf("topk: dispatched ids differ from the reference in %d set(s)\n", bad);
    return bad ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::string which, out_path;
    unsigned seed = 0;
    bool have_seed = false;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--case") which = argv[i + 1];
        else if (a == "--seed") { seed = (unsigned) std::strtoul(argv[i + 1], nullptr, 10); have_seed = true; }
        else if (a == "--out") out_path = argv[i + 1];
        else break;
    }
    if (which.empty() || out_path.empty() || !have_seed || argc != 7) {
        std::fprintf(stderr, "usage: hotrod_kernel_ab --case gr|router|topk --seed N --out FILE\n");
        return 2;
    }
    FILE* out = std::fopen(out_path.c_str(), "wb");
    if (out == nullptr) {
        std::fprintf(stderr, "hotrod_kernel_ab: cannot write %s\n", out_path.c_str());
        return 2;
    }
    int rc;
    if (which == "gr") rc = run_gr(seed, out);
    else if (which == "router") rc = run_router(seed, out);
    else if (which == "topk") rc = run_topk(seed, out);
    else {
        std::fprintf(stderr, "hotrod_kernel_ab: unknown case %s\n", which.c_str());
        rc = 2;
    }
    std::fclose(out);
    return rc;
}
