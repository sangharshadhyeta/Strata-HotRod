// src/kernels/inkling2_conv_parity.cpp - REUSE check: Qwen's GDN short-conv kernel (gdn_conv_step) runs Inkling 2's static short convs.
//
// Inkling's folded short conv is y[t][c] = sum_j f[c][j] x[t - 3 + j][c] with zeros before the first token; gdn_conv_step is
// out[c] = sum_i [state | x][i][c] * kW[c * d_conv + i] with the state sliding one row a call: the same operation, with kW = the folded
// weights [C][taps].  This runs the ENGINE's kernel, token by token as decode will, on layer 0's k conv (1,024 channels), v conv (1,024) and
// post-attention conv (2,304), and compares each output row with the CPU oracle (ops_cpu) and with the float64 reference's seam
// (blk.0.k_conv, blk.0.v_conv, blk.0.attn_out).  Needs a GPU and STRATA_INKLING2_GGUF / _ORACLE / _TOKENS; otherwise it is skipped (exit 0).
#include "strata/inkling2/ops_cpu.hpp"
#include "strata/kernels/gdn.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace {

namespace I = strata::inkling2;
namespace C = strata::inkling2::cpu;

int g_fail = 0;

C::Mat load_npy(const std::string& path) {   // float32 C-order .npy, version 1 or 2
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    char magic[10];
    f.read(magic, 10);
    uint32_t hl = (unsigned char) magic[8] | ((unsigned char) magic[9] << 8);
    if ((unsigned char) magic[6] >= 2) { char more[2]; f.read(more, 2); hl |= ((unsigned char) more[0] << 16) | ((unsigned char) more[1] << 24); }
    std::string hdr(hl, ' ');
    f.read(&hdr[0], hl);
    const size_t a = hdr.find('('), b = hdr.find(')');
    std::stringstream ss(hdr.substr(a + 1, b - a - 1));
    std::string tok;
    int64_t n = 1;
    while (std::getline(ss, tok, ',')) if (tok.find_first_of("0123456789") != std::string::npos) n *= std::stoll(tok);
    C::Mat m((size_t) n);
    f.read(reinterpret_cast<char*>(m.data()), (std::streamsize) (n * 4));
    return m;
}

double rel_err(const C::Mat& got, const C::Mat& ref) {
    double mx = 0, md = 0;
    for (size_t i = 0; i < ref.size(); ++i) { mx = std::max(mx, (double) std::fabs(ref[i])); md = std::max(md, (double) std::fabs(got[i] - ref[i])); }
    return md / std::max(mx, 1e-12);
}

/// The engine's kernel over T tokens of `x` [T][C] with the folded weights `f` [C][taps], one call a token, the state starting at zeros.
C::Mat run_gpu(const C::Mat& x, int64_t T, int64_t Ch, const C::Mat& f, int64_t taps) {
    float *d_state = nullptr, *d_x = nullptr, *d_w = nullptr, *d_out = nullptr;
    cudaMalloc(&d_state, (size_t) (Ch * (taps - 1)) * 4);
    cudaMemset(d_state, 0, (size_t) (Ch * (taps - 1)) * 4);
    cudaMalloc(&d_x, (size_t) Ch * 4);
    cudaMalloc(&d_w, f.size() * 4);
    cudaMalloc(&d_out, (size_t) Ch * 4);
    cudaMemcpy(d_w, f.data(), f.size() * 4, cudaMemcpyHostToDevice);
    C::Mat y((size_t) (T * Ch));
    for (int64_t t = 0; t < T; ++t) {
        cudaMemcpy(d_x, &x[(size_t) (t * Ch)], (size_t) Ch * 4, cudaMemcpyHostToDevice);
        strata::kernels::gdn_conv_step(d_state, d_x, d_w, d_out, Ch, taps, nullptr);
        cudaMemcpy(&y[(size_t) (t * Ch)], d_out, (size_t) Ch * 4, cudaMemcpyDeviceToHost);
    }
    cudaFree(d_state); cudaFree(d_x); cudaFree(d_w); cudaFree(d_out);
    return y;
}

void check(const std::string& name, const C::Mat& gpu, const C::Mat& cpu, const C::Mat& seam) {
    const double e_cpu = rel_err(gpu, cpu), e_seam = rel_err(gpu, seam);
    const bool ok = e_cpu < 1e-5 && e_seam < 1e-5;
    if (!ok) ++g_fail;
    std::printf("%s %-22s GPU (the engine's gdn_conv_step) vs CPU oracle rel %.2e, vs reference seam rel %.2e\n", ok ? "ok  " : "FAIL", name.c_str(), e_cpu, e_seam);
}

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
    const int64_t E = c.n_embd, KVD = c.n_head_kv * c.head_dim, taps = c.conv_taps;
    const I::LayerTensors& L = lay.layers[0];
    // layer 0's attention inputs, on the CPU (these are the conv's inputs: the raw k / v projections and the attention output)
    C::Mat xs = C::embed(g, c, lay, toks);
    { const C::Mat w = C::vec(g, *L.attn_norm); C::rms(xs.data(), T, E, w.data(), c.rms_eps); }
    const C::Mat k_raw = C::matmul_t(g, *L.attn_k, xs, T), v_raw = C::matmul_t(g, *L.attn_v, xs, T);
    C::AttnSeams sm;
    (void) C::attention(g, c, L, xs, xs, xs, T, sm);   // sm.out = attention's output before the post conv
    const C::Mat kf = C::fold_conv(C::vec(g, *L.attn_k_conv), KVD, taps, false), vf = C::fold_conv(C::vec(g, *L.attn_v_conv), KVD, taps, false);
    const C::Mat pf = C::fold_conv(C::vec(g, *L.attn_post_conv), E, taps, false);
    std::printf("tokens %lld, taps %lld\n", (long long) T, (long long) taps);
    check("k conv (1024 ch)", run_gpu(k_raw, T, KVD, kf, taps), C::conv(k_raw, T, KVD, kf, taps), load_npy(dir + "/blk.0.k_conv.npy"));
    check("v conv (1024 ch)", run_gpu(v_raw, T, KVD, vf, taps), C::conv(v_raw, T, KVD, vf, taps), load_npy(dir + "/blk.0.v_conv.npy"));
    check("post-attention (2304)", run_gpu(sm.out, T, E, pf, taps), C::conv(sm.out, T, E, pf, taps), load_npy(dir + "/blk.0.attn_out.npy"));
    std::printf("%s\n", g_fail == 0 ? "the engine's GDN conv kernel runs Inkling's static short convs as it is" : "FAILED");
    return g_fail == 0 ? 0 : 1;
}
