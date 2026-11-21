// src/kernels/moe_group_parity.cpp - the model-independent MoE kernels (include/strata/kernels/moe_group.hpp) against the CPU oracle.
//
// What makes them reusable by other models is what is tested: both weight types (NVFP4 and Q8_0), all three activation kinds, both router score modes, shapes taken at run time.
//   1. a whole layer's MoE for real tokens: layer 0 (NVFP4 experts, the real MoE input of the oracle prompt) and layer 20 (Q8_0 experts, a synthetic input): the grouped
//      route -> prepare -> gate-up -> down -> combine against the CPU `moe` (exact dequant, double dots); the activations stay float, so the error is float rounding.
//   2. the three activation kinds (plain SwiGLU, DeepSeek's clamped, Inkling's SITU) on one real expert against their host formulas.
//   3. the SqrtSoftplus router (DeepSeek's score) against a host top-k.
// Needs a GPU and STRATA_INKLING2_GGUF / _ORACLE / _TOKENS; otherwise skipped (exit 0).
#include "strata/inkling2/ops_cpu.hpp"
#include "strata/kernels/moe_group.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <sstream>

namespace I = strata::inkling2;
namespace C = strata::inkling2::cpu;
namespace M = strata::kernels::moe;

namespace {
int g_fail = 0;
void* dup(const void* src, size_t n) { void* d = nullptr; cudaMalloc(&d, n); cudaMemcpy(d, src, n, cudaMemcpyHostToDevice); return d; }
double rel(const C::Mat& a, const C::Mat& b) {
    double mx = 0, md = 0;
    for (size_t i = 0; i < b.size(); ++i) { mx = std::max(mx, (double) std::fabs(b[i])); md = std::max(md, (double) std::fabs(a[i] - b[i])); }
    return md / std::max(mx, 1e-12);
}
void report(const std::string& what, double err, double bound) {
    const bool ok = err < bound;
    if (!ok) ++g_fail;
    std::printf("%s %-62s rel %.2e (bound %.0e)\n", ok ? "ok  " : "FAIL", what.c_str(), err, bound);
}

struct DevLayer { uint8_t *gu = nullptr, *dn = nullptr; float *gs = nullptr, *ds = nullptr, *router_bias = nullptr; size_t gu_plane = 0, dn_plane = 0, up_off = 0; int ty = 0; };
DevLayer upload(const strata::GgufFile& g, const I::LayerTensors& L, const I::Config& c) {
    DevLayer d;
    d.gu = (uint8_t*) dup(g.tensor_data(*L.gate_up_exps), strata::tensor_payload_bytes(*L.gate_up_exps));
    d.dn = (uint8_t*) dup(g.tensor_data(*L.down_exps), strata::tensor_payload_bytes(*L.down_exps));
    const uint64_t rb = C::row_bytes(L.gate_up_exps->type, c.n_embd), rbd = C::row_bytes(L.down_exps->type, c.n_ff_exp);
    d.gu_plane = (size_t) 2 * c.n_ff_exp * rb; d.dn_plane = (size_t) c.n_embd * rbd; d.up_off = (size_t) c.n_ff_exp * rb;
    d.ty = L.expert_nvfp4 ? 40 : 8;
    if (L.expert_nvfp4) { const C::Mat gs = C::vec(g, *L.gate_up_scale), ds = C::vec(g, *L.down_scale); d.gs = (float*) dup(gs.data(), gs.size() * 4); d.ds = (float*) dup(ds.data(), ds.size() * 4); }
    const C::Mat b = C::vec(g, *L.router_bias);
    d.router_bias = (float*) dup(b.data(), b.size() * 4);
    return d;
}
/// The grouped MoE of one token on the GPU.
C::Mat gpu_moe(const I::Config& c, const DevLayer& d, const C::Mat& logits_row, const C::Mat& x, M::Act act, M::Score score = M::Score::Sigmoid) {
    const int E = (int) c.n_embd, FF = (int) c.n_ff_exp, K = (int) c.n_expert_used, NE = (int) c.n_expert;
    float *d_l = (float*) dup(logits_row.data(), (size_t) NE * 4), *d_x = (float*) dup(x.data(), (size_t) E * 4), *d_w = nullptr, *d_sg = nullptr, *d_wd = nullptr, *d_h = nullptr, *d_y = nullptr, *d_o = nullptr;
    int32_t* d_i = nullptr;
    const uint8_t **d_gp = nullptr, **d_dp = nullptr;
    cudaMalloc(&d_w, 64); cudaMalloc(&d_i, 64); cudaMalloc(&d_sg, 64); cudaMalloc(&d_wd, 64); cudaMalloc(&d_h, (size_t) K * FF * 4); cudaMalloc(&d_y, (size_t) K * E * 4); cudaMalloc(&d_o, (size_t) E * 4);
    cudaMalloc(&d_gp, 16 * sizeof(void*)); cudaMalloc(&d_dp, 16 * sizeof(void*));
    M::route(d_l, d.router_bias, 1, NE, K, c.expert_weight_scale, score, d_i, d_w, nullptr);
    M::experts_prepare(d_i, d_w, d.gu, d.gu_plane, d.dn, d.dn_plane, d.gs, d.ds, K, d_gp, d_dp, d_sg, d_wd, nullptr);
    M::experts_gate_up(d_gp, d.ty, d.up_off, d_x, E, FF, d_sg, act, d_h, K, nullptr);
    M::experts_down(d_dp, d.ty, d_h, E, FF, d_y, K, nullptr);
    M::experts_combine(d_y, d_wd, K, E, d_o, nullptr);
    C::Mat out((size_t) E);
    cudaMemcpy(out.data(), d_o, (size_t) E * 4, cudaMemcpyDeviceToHost);
    for (void* p : {(void*) d_l, (void*) d_x, (void*) d_w, (void*) d_i, (void*) d_sg, (void*) d_wd, (void*) d_h, (void*) d_y, (void*) d_o, (void*) d_gp, (void*) d_dp}) cudaFree(p);
    return out;
}
}  // namespace

int main() {
    const char *gp = std::getenv("STRATA_INKLING2_GGUF"), *tp = std::getenv("STRATA_INKLING2_TOKENS");
    int devs = 0;
    if (cudaGetDeviceCount(&devs) != cudaSuccess || devs == 0) { std::printf("(skipped: no CUDA device)\n"); return 0; }
    if (!gp || !tp) { std::printf("(skipped: STRATA_INKLING2_GGUF / _TOKENS are not set)\n"); return 0; }
    std::vector<int64_t> toks;
    { std::stringstream ss(tp); std::string t; while (std::getline(ss, t, ',')) toks.push_back(std::stoll(t)); }
    const int64_t T = (int64_t) toks.size();
    const strata::GgufFile g(gp);
    I::Config c;
    I::Layout lay;
    std::string e = I::parse_config(g.metadata(), c);
    if (e.empty()) e = I::load_layout(g, c, lay);
    if (!e.empty()) { std::printf("FAIL: %s\n", e.c_str()); return 1; }
    const int64_t E = c.n_embd, NE = c.n_expert, FF = c.n_ff_exp;
    const M::Act situ{M::ActKind::Situ, c.situ_beta, c.situ_linear_beta};

    // 1a. layer 0: NVFP4 experts, the real MoE input of the oracle prompt
    std::vector<C::Mat> blocks;
    C::VisitSeams vs;
    (void) C::layer_visit(g, c, lay, 0, C::embed(g, c, lay, toks), blocks, T, vs);
    const I::LayerTensors& L0 = lay.layers[0];
    const DevLayer d0 = upload(g, L0, c);
    const C::Mat cpu0 = C::moe(g, c, L0, vs.moe_in, T, nullptr);
    const C::Mat rt0 = C::dequant_matrix(g, *L0.router);
    const C::Mat lg0 = C::matmul_t(vs.moe_in, T, E, rt0, NE);
    double worst = 0;
    for (int64_t t : {0, 13, 31, 47}) {
        if (t >= T) continue;
        const C::Mat x(vs.moe_in.begin() + t * E, vs.moe_in.begin() + (t + 1) * E), lg(lg0.begin() + t * NE, lg0.begin() + (t + 1) * NE);
        const C::Mat got = gpu_moe(c, d0, lg, x, situ), ref(cpu0.begin() + t * E, cpu0.begin() + (t + 1) * E);
        worst = std::max(worst, rel(got, ref));
    }
    report("layer 0 (NVFP4 experts), 4 real tokens: grouped MoE vs the CPU oracle", worst, 2e-4);
    // 1b. layer 20: Q8_0 experts, a synthetic normalized input
    const I::LayerTensors& L20 = lay.layers[20];
    const DevLayer d20 = upload(g, L20, c);
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    C::Mat xs((size_t) (2 * E));
    for (float& z : xs) z = nd(rng);
    C::rms(xs.data(), 2, E, nullptr, 1e-6f);
    const C::Mat cpu20 = C::moe(g, c, L20, xs, 2, nullptr);
    const C::Mat rt20 = C::dequant_matrix(g, *L20.router), lg20 = C::matmul_t(xs, 2, E, rt20, NE);
    worst = 0;
    for (int64_t t = 0; t < 2; ++t) {
        const C::Mat x(xs.begin() + t * E, xs.begin() + (t + 1) * E), lg(lg20.begin() + t * NE, lg20.begin() + (t + 1) * NE);
        worst = std::max(worst, rel(gpu_moe(c, d20, lg, x, situ), C::Mat(cpu20.begin() + t * E, cpu20.begin() + (t + 1) * E)));
    }
    report("layer 20 (Q8_0 experts), 2 synthetic tokens: grouped MoE vs the CPU oracle", worst, 2e-4);

    // 2. the activation kinds on one real expert: gate/up by the CPU, the activation by its host formula, the down rows by the CPU; the weights x1 (one pick)
    {
        const int ex = 5;
        const C::Mat x(vs.moe_in.begin() + 7 * E, vs.moe_in.begin() + 8 * E);
        const C::Mat gu = C::dequant_matrix(g, *L0.gate_up_exps, ex, 0, 2 * FF);
        const float sg = C::vec(g, *L0.gate_up_scale)[(size_t) ex];
        auto act_ref = [&](double gt, double up, M::ActKind k) {
            const double sil = gt / (1.0 + std::exp(-gt));
            if (k == M::ActKind::Silu) return sil * up;
            if (k == M::ActKind::ClampedSilu) { const double lim = 7.0, gg = std::min(gt, lim), uu = std::min(std::max(up, -lim), lim); return gg / (1.0 + std::exp(-gg)) * uu; }
            return std::tanh(gt / c.situ_beta) * c.situ_beta * (1.0 / (1.0 + std::exp(-gt))) * std::tanh(up / c.situ_linear_beta) * c.situ_linear_beta;
        };
        float *d_x = (float*) dup(x.data(), (size_t) E * 4), *d_h = nullptr, *d_sg = (float*) dup(&sg, 4);
        cudaMalloc(&d_h, (size_t) FF * 4);
        const uint8_t* plane = d0.gu + (size_t) ex * d0.gu_plane;
        const uint8_t* d_gp = nullptr;
        cudaMalloc((void**) &d_gp, sizeof(void*));
        cudaMemcpy((void*) d_gp, &plane, sizeof(void*), cudaMemcpyHostToDevice);
        const char* names[3] = {"plain SwiGLU", "clamped SwiGLU (limit 7)", "SITU (softcaps)"};
        const M::Act acts[3] = {{M::ActKind::Silu, 0, 0}, {M::ActKind::ClampedSilu, 7.0f, 0}, situ};
        for (int a = 0; a < 3; ++a) {
            M::experts_gate_up((const uint8_t* const*) d_gp, 40, d0.up_off, d_x, (int) E, (int) FF, d_sg, acts[a], d_h, 1, nullptr);
            C::Mat h((size_t) FF), ref((size_t) FF);
            cudaMemcpy(h.data(), d_h, (size_t) FF * 4, cudaMemcpyDeviceToHost);
            for (int64_t f = 0; f < FF; ++f)
                ref[(size_t) f] = (float) act_ref(C::dot(x.data(), &gu[(size_t) (f * E)], E) * sg, C::dot(x.data(), &gu[(size_t) ((FF + f) * E)], E) * sg, acts[a].kind);
            report(std::string("activation: ") + names[a] + " on a real NVFP4 expert", rel(h, ref), 2e-4);
        }
    }
    // 3. the SqrtSoftplus router: random logits, a host top-k by sqrt(softplus(l)) + bias
    {
        const int n = 128, k = 8;
        C::Mat lg((size_t) n), bias((size_t) n);
        for (float& z : lg) z = nd(rng) * 2.0f;
        for (float& z : bias) z = nd(rng) * 0.05f;
        float *d_l = (float*) dup(lg.data(), n * 4), *d_b = (float*) dup(bias.data(), n * 4), *d_w = nullptr;
        int32_t* d_i = nullptr;
        cudaMalloc(&d_w, 64); cudaMalloc(&d_i, 64);
        M::route(d_l, d_b, 1, n, k, 2.5f, M::Score::SqrtSoftplus, d_i, d_w, nullptr);
        std::vector<int32_t> idx(k);
        C::Mat w(k);
        cudaMemcpy(idx.data(), d_i, k * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(w.data(), d_w, k * 4, cudaMemcpyDeviceToHost);
        std::vector<double> sc(n);
        std::vector<int> order(n);
        for (int i = 0; i < n; ++i) { const double l = lg[(size_t) i]; sc[(size_t) i] = std::sqrt(l > 20 ? l : std::log1p(std::exp(l))); order[(size_t) i] = i; }
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return sc[(size_t) a] + bias[(size_t) a] > sc[(size_t) b] + bias[(size_t) b]; });
        double z = 0;
        for (int j = 0; j < k; ++j) z += sc[(size_t) order[(size_t) j]];
        bool same = true;
        double wmax = 0;
        for (int j = 0; j < k; ++j) { same = same && idx[(size_t) j] == order[(size_t) j]; wmax = std::max(wmax, std::fabs(sc[(size_t) order[(size_t) j]] / (z + 1e-20) * 2.5 - (double) w[(size_t) j])); }
        if (!same) ++g_fail;
        std::printf("%s router, SqrtSoftplus score (DeepSeek's): picks %s the host top-k\n", same ? "ok  " : "FAIL", same ? "equal" : "DIFFER FROM");
        report("router, SqrtSoftplus score: the weights", wmax, 1e-5);
    }
    std::printf("%s\n", g_fail == 0 ? "the shared MoE kernels match the oracle on both weight types, all three activations and both router scores" : "FAILED");
    return g_fail == 0 ? 0 : 1;
}
