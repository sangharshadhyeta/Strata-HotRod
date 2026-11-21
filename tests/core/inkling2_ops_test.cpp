// tests/core/inkling2_ops_test.cpp - Inkling 2's CPU ops (include/strata/inkling2/ops_cpu.hpp) against the float64 reference's seams
// (Maya's ref/inkling2.py dumps, docs/INKLING-PORT.md M0): the oracle for each op, one seam at a time.
//
//   STRATA_INKLING2_GGUF    the Inkling-2-Mini GGUF            STRATA_INKLING2_ORACLE   the reference's --dump-dir (blk.V.<seam>.npy)
//   STRATA_INKLING2_TOKENS  the comma-separated token ids the reference ran on
// Unset = skipped (exit 0).  An op passes when its largest error, relative to the seam's largest value, is below the bound printed.
#include "strata/inkling2/ops_cpu.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace {

namespace I = strata::inkling2;
namespace C = strata::inkling2::cpu;

int g_fail = 0, g_ok = 0;

/// A float32 C-order .npy file (version 1/2 header): its values and shape.
C::Mat load_npy(const std::string& path, std::vector<int64_t>& shape) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    char magic[10];
    f.read(magic, 10);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) throw std::runtime_error(path + ": not an npy file");
    const int major = (unsigned char) magic[6];
    uint32_t hl = (unsigned char) magic[8] | ((unsigned char) magic[9] << 8);
    if (major >= 2) { char more[2]; f.read(more, 2); hl = (unsigned char) magic[8] | ((unsigned char) magic[9] << 8) | ((unsigned char) more[0] << 16) | ((unsigned char) more[1] << 24); }
    std::string hdr(hl, ' ');
    f.read(&hdr[0], hl);
    if (hdr.find("'<f4'") == std::string::npos || hdr.find("False") == std::string::npos) throw std::runtime_error(path + ": not little-endian float32 C-order: " + hdr);
    shape.clear();
    const size_t a = hdr.find('('), b = hdr.find(')');
    std::stringstream ss(hdr.substr(a + 1, b - a - 1));
    std::string tok;
    while (std::getline(ss, tok, ',')) if (tok.find_first_of("0123456789") != std::string::npos) shape.push_back(std::stoll(tok));
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    C::Mat m((size_t) n);
    f.read(reinterpret_cast<char*>(m.data()), (std::streamsize) (n * 4));
    return m;
}

void compare(const std::string& name, const C::Mat& got, const C::Mat& ref, double bound) {
    if (got.size() != ref.size()) { ++g_fail; std::printf("FAIL %-22s size %zu vs %zu\n", name.c_str(), got.size(), ref.size()); return; }
    double mx = 0, md = 0, sum2 = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        mx = std::max(mx, (double) std::fabs(ref[i]));
        md = std::max(md, (double) std::fabs(got[i] - ref[i]));
        sum2 += (double) (got[i] - ref[i]) * (got[i] - ref[i]);
    }
    const double rel = md / std::max(mx, 1e-12);
    const bool ok = rel < bound;
    (ok ? g_ok : g_fail)++;
    std::printf("%s %-22s max|d| %.3g (ref max %.4g, rel %.2e, bound %.0e)\n", ok ? "ok  " : "FAIL", name.c_str(), md, mx, rel, bound);
}

C::Mat seam(const std::string& dir, int visit, const std::string& name) {
    std::vector<int64_t> sh;
    return load_npy(dir + "/blk." + std::to_string(visit) + "." + name + ".npy", sh);
}

}  // namespace

int main() {
    const char *gp = std::getenv("STRATA_INKLING2_GGUF"), *op = std::getenv("STRATA_INKLING2_ORACLE"), *tp = std::getenv("STRATA_INKLING2_TOKENS");
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
    std::printf("tokens %lld, visit 0\n", (long long) T);
    const int64_t E = c.n_embd;

    // ---- the embedding: Q8_0 rows of the token table, RMS-normed with token_embd_norm
    C::Mat h((size_t) (T * E));
    for (int64_t t = 0; t < T; ++t) C::dequant_row(g, *lay.token_embd, 0, toks[(size_t) t], &h[(size_t) (t * E)]);
    { const C::Mat w = C::vec(g, *lay.token_embd_norm); C::rms(h.data(), T, E, w.data(), c.rms_eps); }
    {   std::vector<int64_t> sh; const C::Mat ref = load_npy(dir + "/embd.npy", sh); compare("embd", h, ref, 1e-4); }

    // ---- layer 0 (stored layer 0, visit 0): sliding, no residual query, NVFP4 experts
    const I::LayerTensors& L = lay.layers[0];
    C::Mat xs = h;
    { const C::Mat w = C::vec(g, *L.attn_norm); C::rms(xs.data(), T, E, w.data(), c.rms_eps); }
    C::AttnSeams sm;
    C::Mat a = C::attention_sliding(g, c, L, xs, T, sm);
    compare("k_conv", sm.k_conv, seam(dir, 0, "k_conv"), 1e-4);
    compare("v_conv", sm.v_conv, seam(dir, 0, "v_conv"), 1e-4);
    compare("q", sm.q, seam(dir, 0, "q"), 1e-4);
    compare("k", sm.k, seam(dir, 0, "k"), 1e-4);
    compare("v", sm.v, seam(dir, 0, "v"), 1e-4);
    compare("r", sm.r, seam(dir, 0, "r"), 1e-4);
    compare("attn_gated", sm.gated, seam(dir, 0, "attn_gated"), 1e-4);
    const C::Mat post = C::fold_conv(C::vec(g, *L.attn_post_conv), E, c.conv_taps, false);
    a = C::conv(a, T, E, post, c.conv_taps);
    compare("attn_out", a, seam(dir, 0, "attn_out"), 1e-4);
    // visit 0 is a block boundary (0 % 6 == 0): blocks = [h], the stream restarts at a
    const C::Mat& r = a;
    const C::Mat fq = C::vec(g, *L.ffn_res_query);
    C::Mat m_in = C::attn_res({&h, &r}, T, E, fq.data(), 1)[0];
    { const C::Mat w = C::vec(g, *L.ffn_norm); C::rms(m_in.data(), T, E, w.data(), c.rms_eps); }
    C::MoeSeams ms;
    C::Mat m = C::moe(g, c, L, m_in, T, &ms);
    {   const C::Mat ref = seam(dir, 0, "picks");
        int bad = 0;
        for (size_t i = 0; i < ms.picks.size(); ++i) bad += (int) ms.picks[i] != (int) ref[i];
        (bad == 0 ? g_ok : g_fail)++;
        std::printf("%s %-22s %d of %zu picks differ\n", bad == 0 ? "ok  " : "FAIL", "picks", bad, ms.picks.size());
    }
    const C::Mat pw = C::fold_conv(C::vec(g, *L.ffn_post_conv), E, c.conv_taps, false);
    m = C::conv(m, T, E, pw, c.conv_taps);
    compare("ffn_out", m, seam(dir, 0, "ffn_out"), 1e-4);
    C::Mat out(r.size());
    for (size_t i = 0; i < out.size(); ++i) out[i] = r[i] + m[i];
    compare("out", out, seam(dir, 0, "out"), 1e-4);
    std::printf("%d ops matched, %d failed\n", g_ok, g_fail);
    return g_fail == 0 ? 0 : 1;
}
