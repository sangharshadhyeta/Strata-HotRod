// src/kernels/inkling2_router_parity.cpp - the GPU router (inkling2_route, hand-ported from the DeepSeek branch's biased top-k) against the reference.
//
// For stored layers 0-5 on the oracle prompt (the CPU forward gives each layer's real MoE input): logits = x @ router^T on the CPU, then the kernel
// picks the top-8 of sigmoid(logit) + bias and weights them.  Checks, over every token of those 6 visits:
//   (a) the picked set, sorted, equals the float64 reference's own picks seam (blk.V.picks): exactly;
//   (b) the weights equal  score / sum(picked scores) * 8  computed on the CPU in double, for the same experts;
// and a non-finite row (all NaN) yields valid, distinct expert ids (the guard).  Needs a GPU and STRATA_INKLING2_GGUF / _ORACLE / _TOKENS; else skipped.
#include "strata/inkling2/npy.hpp"
#include "strata/inkling2/ops_cpu.hpp"
#include "strata/kernels/inkling2_router.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <sstream>

namespace {
namespace I = strata::inkling2;
namespace C = strata::inkling2::cpu;
namespace K = strata::kernels;
}  // namespace

int main() {
    const char *gp = std::getenv("STRATA_INKLING2_GGUF"), *op = std::getenv("STRATA_INKLING2_ORACLE"), *tp = std::getenv("STRATA_INKLING2_TOKENS");
    int devs = 0;
    if (cudaGetDeviceCount(&devs) != cudaSuccess || devs == 0) { std::printf("(skipped: no CUDA device)\n"); return 0; }
    if (!gp || !op || !tp) { std::printf("(skipped: STRATA_INKLING2_GGUF / _ORACLE / _TOKENS are not all set)\n"); return 0; }
    const std::string dir = op;
    std::vector<int64_t> toks;
    { std::stringstream ss(tp); std::string t; while (std::getline(ss, t, ',')) toks.push_back(std::stoll(t)); }
    const int64_t T = (int64_t) toks.size();
    const strata::GgufFile g(gp);
    I::Config c;
    I::Layout lay;
    std::string e = I::parse_config(g.metadata(), c);
    if (e.empty()) e = I::load_layout(g, c, lay);
    if (!e.empty()) { std::printf("FAIL: %s\n", e.c_str()); return 1; }
    const int64_t E = c.n_embd, NE = c.n_expert, K8 = c.n_expert_used;
    int fails = 0;
    std::vector<C::Mat> blocks;
    C::Mat h = C::embed(g, c, lay, toks);
    for (int64_t l = 0; l < 6; ++l) {
        C::VisitSeams vs;
        h = C::layer_visit(g, c, lay, l, h, blocks, T, vs);
        const I::LayerTensors& L = lay.layers[(size_t) l];
        const C::Mat router = C::dequant_matrix(g, *L.router), bias = C::vec(g, *L.router_bias);
        const C::Mat logits = C::matmul_t(vs.moe_in, T, E, router, NE);
        float *d_l = nullptr, *d_b = nullptr, *d_w = nullptr;
        int32_t* d_i = nullptr;
        cudaMalloc(&d_l, logits.size() * 4); cudaMalloc(&d_b, bias.size() * 4);
        cudaMalloc(&d_i, (size_t) (T * K8) * 4); cudaMalloc(&d_w, (size_t) (T * K8) * 4);
        cudaMemcpy(d_l, logits.data(), logits.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, bias.data(), bias.size() * 4, cudaMemcpyHostToDevice);
        K::inkling2_route(d_l, d_b, (int) T, (int) NE, (int) K8, c.expert_weight_scale, d_i, d_w, nullptr);
        std::vector<int32_t> idx((size_t) (T * K8));
        C::Mat w((size_t) (T * K8));
        cudaMemcpy(idx.data(), d_i, idx.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(w.data(), d_w, w.size() * 4, cudaMemcpyDeviceToHost);
        cudaFree(d_l); cudaFree(d_b); cudaFree(d_i); cudaFree(d_w);
        const C::Mat ref = I::load_npy(dir + "/blk." + std::to_string(l) + ".picks.npy");
        int bad_sets = 0;
        double wmax = 0;
        for (int64_t t = 0; t < T; ++t) {
            std::vector<int> got(idx.begin() + t * K8, idx.begin() + (t + 1) * K8);
            std::sort(got.begin(), got.end());
            for (int64_t j = 0; j < K8; ++j) bad_sets += got[(size_t) j] != (int) ref[(size_t) (t * K8 + j)];
            double z = 0;
            for (int64_t j = 0; j < K8; ++j) z += 1.0 / (1.0 + std::exp(-(double) logits[(size_t) (t * NE + idx[(size_t) (t * K8 + j)])]));
            for (int64_t j = 0; j < K8; ++j) {
                const double s = 1.0 / (1.0 + std::exp(-(double) logits[(size_t) (t * NE + idx[(size_t) (t * K8 + j)])]));
                wmax = std::max(wmax, std::fabs(s / (z + 1e-20) * (double) c.expert_weight_scale - (double) w[(size_t) (t * K8 + j)]));
            }
        }
        const bool ok = bad_sets == 0 && wmax < 1e-5;
        if (!ok) ++fails;
        std::printf("%s layer %lld: %lld tokens, picks differing from the reference's seam: %d of %lld; max weight error %.2e\n", ok ? "ok  " : "FAIL", (long long) l, (long long) T, bad_sets,
                    (long long) (T * K8), wmax);
    }
    {   // the guard: an all-NaN row still yields valid, distinct ids
        std::vector<float> nan_row((size_t) NE, std::nanf("")), zb((size_t) NE, 0.0f);
        float *d_l = nullptr, *d_b = nullptr, *d_w = nullptr;
        int32_t* d_i = nullptr;
        cudaMalloc(&d_l, NE * 4); cudaMalloc(&d_b, NE * 4); cudaMalloc(&d_i, K8 * 4); cudaMalloc(&d_w, K8 * 4);
        cudaMemcpy(d_l, nan_row.data(), NE * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, zb.data(), NE * 4, cudaMemcpyHostToDevice);
        K::inkling2_route(d_l, d_b, 1, (int) NE, (int) K8, 8.0f, d_i, d_w, nullptr);
        std::vector<int32_t> idx((size_t) K8);
        cudaMemcpy(idx.data(), d_i, K8 * 4, cudaMemcpyDeviceToHost);
        cudaFree(d_l); cudaFree(d_b); cudaFree(d_i); cudaFree(d_w);
        std::set<int> uniq(idx.begin(), idx.end());
        bool in_range = true;
        for (int v : idx) in_range = in_range && v >= 0 && v < NE;
        const bool ok = in_range && uniq.size() == (size_t) K8;
        if (!ok) ++fails;
        std::printf("%s non-finite row: %zu distinct ids, all inside 0..%lld (the guard)\n", ok ? "ok  " : "FAIL", uniq.size(), (long long) NE - 1);
    }
    std::printf("%s\n", fails == 0 ? "the ported router matches the reference's picks and the weights" : "FAILED");
    return fails == 0 ? 0 : 1;
}
