// tests/core/inkling2_ops_test.cpp - Inkling 2's CPU forward (include/strata/inkling2/ops_cpu.hpp) against the float64 reference's seams
// (Maya's ref/inkling2.py dumps, docs/INKLING-PORT.md M0): every layer visit, one seam at a time, then the log-probabilities.
//
//   STRATA_INKLING2_GGUF    the Inkling-2-Mini GGUF            STRATA_INKLING2_ORACLE   the reference's --dump-dir (blk.V.<seam>.npy)
//   STRATA_INKLING2_TOKENS  the comma-separated token ids the reference ran on
// Unset = skipped (exit 0).  The forward is one chain (each visit takes this code's previous output, not the oracle's), so a drift would
// show; a seam passes when its largest error, relative to the seam's largest value, is below the bound printed.
#include "strata/inkling2/ops_cpu.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace {

namespace I = strata::inkling2;
namespace C = strata::inkling2::cpu;

int g_fail = 0, g_ok = 0;
const double kBound = 1e-4;

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

/// The seam's relative error (max |d| / max |ref|); -1 when the sizes differ.
double rel_err(const C::Mat& got, const C::Mat& ref) {
    if (got.size() != ref.size()) return -1;
    double mx = 0, md = 0;
    for (size_t i = 0; i < ref.size(); ++i) { mx = std::max(mx, (double) std::fabs(ref[i])); md = std::max(md, (double) std::fabs(got[i] - ref[i])); }
    return md / std::max(mx, 1e-12);
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
    std::printf("tokens %lld, %lld visits\n", (long long) T, (long long) c.n_visits);

    {   // the embedding
        std::vector<int64_t> sh;
        const double r = rel_err(C::embed(g, c, lay, toks), load_npy(dir + "/embd.npy", sh));
        const bool ok = r >= 0 && r < kBound;
        (ok ? g_ok : g_fail)++;
        std::printf("%s embd rel %.2e\n", ok ? "ok  " : "FAIL", r);
    }
    const auto on_visit = [&](int64_t v, const C::VisitSeams& vs) {
        struct S { const char* name; const C::Mat* got; };
        std::vector<S> seams = {{"k_conv", &vs.attn.k_conv}, {"v_conv", &vs.attn.v_conv}, {"q", &vs.attn.q}, {"k", &vs.attn.k}, {"v", &vs.attn.v},
                                {"r", &vs.attn.r}, {"attn_gated", &vs.attn.gated}, {"attn_out", &vs.attn_out}, {"ffn_out", &vs.ffn_out}, {"out", &vs.out}};
        if (!vs.loop_q_in.empty()) { seams.push_back({"loop_q_in", &vs.loop_q_in}); seams.push_back({"loop_ffn_in", &vs.loop_ffn_in}); }
        double worst = 0;
        std::string worst_name;
        bool ok = true;
        for (const S& s : seams) {
            const double r = rel_err(*s.got, seam(dir, (int) v, s.name));
            if (r < 0 || r >= kBound) { ok = false; std::printf("   FAIL visit %lld seam %s rel %.3e\n", (long long) v, s.name, r); }
            if (r > worst) { worst = r; worst_name = s.name; }
        }
        const C::Mat pr = seam(dir, (int) v, "picks");
        int bad = 0;
        for (size_t i = 0; i < vs.picks.size() && i < pr.size(); ++i) bad += vs.picks[i] != (int) pr[i];
        if (vs.picks.size() != pr.size() || bad != 0) ok = false;
        (ok ? g_ok : g_fail)++;
        const int64_t stored = c.stored_of_visit(v);
        std::printf("%s visit %2lld (layer %2lld, %s%s): worst seam %s rel %.2e, picks differ %d of %zu\n", ok ? "ok  " : "FAIL", (long long) v, (long long) stored,
                    lay.layers[(size_t) stored].full ? "full" : "sliding", vs.loop_q_in.empty() ? "" : ", loop msgs", worst_name.c_str(), worst, bad, pr.size());
    };
    const C::Mat lp = C::forward(g, c, lay, toks, on_visit);
    // the log-probabilities: a row a position over the real vocabulary
    std::vector<int64_t> sh;
    const C::Mat ref = load_npy(dir + "/logprobs.npy", sh);
    const int64_t NV = c.n_vocab_real;
    double md = 0;
    int top1_same = 0;
    for (int64_t t = 0; t < T; ++t) {
        int64_t a = 0, b = 0;
        for (int64_t v = 0; v < NV; ++v) {
            md = std::max(md, (double) std::fabs(lp[(size_t) (t * NV + v)] - ref[(size_t) (t * NV + v)]));
            if (lp[(size_t) (t * NV + v)] > lp[(size_t) (t * NV + a)]) a = v;
            if (ref[(size_t) (t * NV + v)] > ref[(size_t) (t * NV + b)]) b = v;
        }
        top1_same += a == b;
    }
    const bool ok = sh.size() == 2 && sh[0] == T && sh[1] == NV && md < 5e-3 && top1_same == (int) T;
    (ok ? g_ok : g_fail)++;
    std::printf("%s logprobs: max |dlogp| %.3e over %lld x %lld, top-1 the same at %d of %lld positions\n", ok ? "ok  " : "FAIL", md, (long long) T, (long long) NV, top1_same, (long long) T);
    std::printf("%d checks matched, %d failed\n", g_ok, g_fail);
    return g_fail == 0 ? 0 : 1;
}
