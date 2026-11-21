// src/kernels/inkling2_nvfp4_parity.cpp - REUSE check: the engine's NVFP4 kernels (Edge0's) run Inkling 2's NVFP4 experts as they are.
//
// iq_mmvq(40, ...) is the engine's CUDA NVFP4 mat-vec (llama.cpp's vec_dot_nvfp4_q8_1: 64 values a block, four UE4M3 sub-scales, E2M1 codes
// split-half; activations quantized to q8_1).  Inkling's layers 0-19 hold their experts in exactly that format, with a per-expert F32 scale
// beside each tensor (Edge0's blobs carry the same idea as {s_gate, s_up, s_down}).  For real picks of layer 0 on the 48-token oracle this runs
//   gate, up  = iq_mmvq(NVFP4, the expert's gate / up rows, q8_1(x))  * s_gate_up        (the engine's kernel)
//   h         = 4 tanh(g/4) sigmoid(g) * 25 tanh(u/25)                                     (the one NEW op: SITU, on the host here)
//   out       = iq_mmvq(NVFP4, the expert's down rows, q8_1(h)) * s_down                   (the engine's kernel)
// and compares the vector with the CPU oracle's exact-dequant double dot.  The GPU path quantizes activations to int8 (q8_1), as every expert
// of every model on this engine does, so the bound is that path's usual envelope, not float rounding.
// Needs a GPU and STRATA_INKLING2_GGUF / _ORACLE / _TOKENS; otherwise skipped (exit 0).
#include "strata/inkling2/ops_cpu.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <sstream>

namespace {

namespace I = strata::inkling2;
namespace C = strata::inkling2::cpu;
namespace K = strata::kernels;

int g_fail = 0;

void* dev_copy(const void* src, size_t n) {
    void* d = nullptr;
    cudaMalloc(&d, n);
    cudaMemcpy(d, src, n, cudaMemcpyHostToDevice);
    return d;
}

/// y[o] = W[o] . x for `n_out` rows of NVFP4 weights starting at `w` (host pointer into the mapped file), x a float vector of `n_in`, through the engine's kernel.
C::Mat nvfp4_matvec(const uint8_t* w, int n_in, int n_out, const C::Mat& x) {
    const size_t rb = K::iq_row_bytes(40, n_in);
    void* d_w = dev_copy(w, rb * (size_t) n_out);
    float* d_x = (float*) dev_copy(x.data(), (size_t) n_in * 4);
    void* d_xq = nullptr;
    cudaMalloc(&d_xq, (size_t) (n_in / 32) * 36);   // q8_1 block: 2 half + 32 int8 = 36 bytes
    float* d_y = nullptr;
    cudaMalloc(&d_y, (size_t) n_out * 4);
    K::quantize_q8_1_rows(d_x, 1, n_in, d_xq, nullptr);
    K::iq_mmvq(40, d_w, d_xq, d_y, n_in, n_out, 1, nullptr);
    C::Mat y((size_t) n_out);
    cudaMemcpy(y.data(), d_y, (size_t) n_out * 4, cudaMemcpyDeviceToHost);
    cudaFree(d_w); cudaFree(d_x); cudaFree(d_xq); cudaFree(d_y);
    return y;
}

double rel_err(const C::Mat& got, const C::Mat& ref) {
    double mx = 0, md = 0;
    for (size_t i = 0; i < ref.size(); ++i) { mx = std::max(mx, (double) std::fabs(ref[i])); md = std::max(md, (double) std::fabs(got[i] - ref[i])); }
    return md / std::max(mx, 1e-12);
}

}  // namespace

int main() {
    const char *gp = std::getenv("STRATA_INKLING2_GGUF"), *op = std::getenv("STRATA_INKLING2_ORACLE"), *tp = std::getenv("STRATA_INKLING2_TOKENS");
    int devs = 0;
    if (cudaGetDeviceCount(&devs) != cudaSuccess || devs == 0) { std::printf("(skipped: no CUDA device)\n"); return 0; }
    if (!gp || !op || !tp) { std::printf("(skipped: STRATA_INKLING2_GGUF / _ORACLE / _TOKENS are not all set)\n"); return 0; }
    (void) op;
    std::vector<int64_t> toks;
    { std::stringstream ss(tp); std::string t; while (std::getline(ss, t, ',')) toks.push_back(std::stoll(t)); }
    const int64_t T = (int64_t) toks.size();
    const strata::GgufFile g(gp);
    I::Config c;
    I::Layout lay;
    std::string e = I::parse_config(g.metadata(), c);
    if (e.empty()) e = I::load_layout(g, c, lay);
    if (!e.empty()) { std::printf("FAIL: %s\n", e.c_str()); return 1; }
    if (!K::iq_supported(40)) { std::printf("FAIL: the engine's kernels do not list NVFP4 (40)\n"); return 1; }
    const int64_t E = c.n_embd, FF = c.n_ff_exp;
    if (K::iq_row_bytes(40, E) != C::row_bytes(I::kNVFP4, E) || K::iq_row_bytes(40, FF) != C::row_bytes(I::kNVFP4, FF)) { std::printf("FAIL: NVFP4 row sizes differ between the engine and the reader\n"); return 1; }
    // layer 0 on the oracle prompt (CPU), for the MoE's real input and the router's real picks
    std::vector<C::Mat> blocks;
    C::VisitSeams vs;
    (void) C::layer_visit(g, c, lay, 0, C::embed(g, c, lay, toks), blocks, T, vs);
    const I::LayerTensors& L = lay.layers[0];
    const C::Mat gs = C::vec(g, *L.gate_up_scale), ds = C::vec(g, *L.down_scale);
    const uint64_t rb_gu = C::row_bytes(I::kNVFP4, E), rb_d = C::row_bytes(I::kNVFP4, FF);
    std::printf("tokens %lld; NVFP4 rows %llu B (gate/up) and %llu B (down), the engine's and the reader's agree\n", (long long) T, (unsigned long long) rb_gu, (unsigned long long) rb_d);
    const int64_t test_t[4] = {0, 13, 31, 47};
    for (int64_t t : test_t) {
        const int ex = vs.picks[(size_t) (t * c.n_expert_used)];                 // the lowest-numbered pick of token t
        C::Mat x(vs.moe_in.begin() + t * E, vs.moe_in.begin() + (t + 1) * E);
        // CPU oracle: exact dequant, double dots, as ops_cpu::moe does for one expert
        const C::Mat gu = C::dequant_matrix(g, *L.gate_up_exps, ex, 0, 2 * FF), dn = C::dequant_matrix(g, *L.down_exps, ex, 0, E);
        C::Mat h_cpu((size_t) FF), out_cpu((size_t) E), g_cpu((size_t) FF), u_cpu((size_t) FF);
        for (int64_t f = 0; f < FF; ++f) {
            const double gt = C::dot(x.data(), &gu[(size_t) (f * E)], E) * gs[(size_t) ex], up = C::dot(x.data(), &gu[(size_t) ((FF + f) * E)], E) * gs[(size_t) ex];
            g_cpu[(size_t) f] = (float) gt; u_cpu[(size_t) f] = (float) up;
            h_cpu[(size_t) f] = (float) (std::tanh(gt / c.situ_beta) * c.situ_beta * (1.0 / (1.0 + std::exp(-gt))) * std::tanh(up / c.situ_linear_beta) * c.situ_linear_beta);
        }
        for (int64_t o = 0; o < E; ++o) out_cpu[(size_t) o] = (float) (C::dot(h_cpu.data(), &dn[(size_t) (o * FF)], FF) * ds[(size_t) ex]);
        // the engine's kernel: the expert's gate rows [0, FF), up rows [FF, 2 FF) of its plane, then its down rows
        const uint8_t* plane = g.tensor_data(*L.gate_up_exps) + (uint64_t) ex * (uint64_t) (2 * FF) * rb_gu;
        C::Mat gg = nvfp4_matvec(plane, (int) E, (int) FF, x), uu = nvfp4_matvec(plane + (uint64_t) FF * rb_gu, (int) E, (int) FF, x);
        C::Mat h((size_t) FF);
        for (int64_t f = 0; f < FF; ++f) {
            const double gt = (double) gg[(size_t) f] * gs[(size_t) ex], up = (double) uu[(size_t) f] * gs[(size_t) ex];
            gg[(size_t) f] = (float) gt; uu[(size_t) f] = (float) up;
            h[(size_t) f] = (float) (std::tanh(gt / c.situ_beta) * c.situ_beta * (1.0 / (1.0 + std::exp(-gt))) * std::tanh(up / c.situ_linear_beta) * c.situ_linear_beta);
        }
        C::Mat out = nvfp4_matvec(g.tensor_data(*L.down_exps) + (uint64_t) ex * (uint64_t) E * rb_d, (int) FF, (int) E, h);
        for (float& z : out) z *= ds[(size_t) ex];
        const double eg = rel_err(gg, g_cpu), eu = rel_err(uu, u_cpu), eh = rel_err(h, h_cpu), eo = rel_err(out, out_cpu);
        const bool ok = eg < 3e-2 && eu < 3e-2 && eo < 3e-2;
        if (!ok) ++g_fail;
        std::printf("%s token %2lld expert %3d: gate rel %.2e, up rel %.2e, h rel %.2e, expert output rel %.2e (ref max %.3g)\n", ok ? "ok  " : "FAIL", (long long) t, ex, eg, eu, eh, eo,
                    (double) *std::max_element(out_cpu.begin(), out_cpu.end(), [](float a, float b) { return std::fabs(a) < std::fabs(b); }));
    }
    std::printf("%s\n", g_fail == 0 ? "the engine's NVFP4 kernel runs Inkling's NVFP4 experts as it is (activations through q8_1, the engine's usual path)" : "FAILED");
    return g_fail == 0 ? 0 : 1;
}
