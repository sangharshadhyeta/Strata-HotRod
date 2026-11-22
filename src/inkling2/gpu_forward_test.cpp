// src/inkling2/gpu_forward_test.cpp - the Inkling 2 GPU engine (F2) against the float64 reference's oracle, teacher-forced on the oracle prompt.
//
//   STRATA_INKLING2_GGUF / _ORACLE / _TOKENS as in inkling2_ops_test; optional STRATA_INKLING2_WINDOW / _EXTENT / _FLOOR for the shrunken-value oracle.
// Reports, per visit, the relative error of the visit's `out` seam (against the reference's seam, the engine's own chain), how many expert picks differ, and at the
// end the log-probabilities: top-1 agreement, mean and max |dlogp|, mean KL(reference || engine).  The engine runs every matvec through the engine's q8_1 int8
// activation path (as every expert and dense weight on this engine does), so it differs from the float64 reference by that quantization, not by float rounding;
// the bounds below are that path's envelope and the numbers are printed so they can be read, not just checked.
#include "strata/inkling2/engine.hpp"
#include "strata/inkling2/npy.hpp"
#include "strata/inkling2/ops_cpu.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <sstream>

namespace I = strata::inkling2;
namespace C = strata::inkling2::cpu;

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
    if (std::getenv("STRATA_INKLING2_WINDOW")) c.window = std::atoll(std::getenv("STRATA_INKLING2_WINDOW"));
    if (std::getenv("STRATA_INKLING2_EXTENT")) c.rel_extent = std::atoll(std::getenv("STRATA_INKLING2_EXTENT"));
    if (std::getenv("STRATA_INKLING2_FLOOR")) c.log_scale_floor = std::atoll(std::getenv("STRATA_INKLING2_FLOOR"));
    const auto t0 = std::chrono::steady_clock::now();
    I::Engine eng(g, c, lay, (int) T + 1);
    const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("engine loaded in %.1f s: dense weights %.2f GB in VRAM, experts %.2f GB registered (zero-copy)\n", load_s, eng.dense_bytes() / 1e9, eng.expert_bytes() / 1e9);
    // the visit's `out` rows, every token, and the picks
    std::vector<std::vector<float>> out_rows((size_t) c.n_visits);
    std::vector<std::vector<int>> picks((size_t) c.n_visits);
    for (auto& v : out_rows) v.reserve((size_t) (T * c.n_embd));
    // every seam of a few visits (the first, a full-attention one, a loop pass): to see WHERE the drift starts
    std::map<std::string, std::vector<float>> seams_kept;
    const auto keep = [](int visit) { return visit == 0 || visit == 5 || visit == 18 || visit == 23; };
    std::vector<float> lp((size_t) (T * c.n_vocab_real));
    if (std::getenv("STRATA_INKLING2_BENCH")) {   // the engine alone: no seam copies, no host softmax; pass 1 starts with a cold cache, pass 2 with the same tokens warm (a floor: no misses)
        for (int pass = 1; pass <= 2; ++pass) {
            eng.reset();
            const auto tb = std::chrono::steady_clock::now();
            for (int64_t t = 0; t < T; ++t) eng.step(toks[(size_t) t], (int) t, nullptr, {});
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - tb).count();
            std::printf("BENCH pass %d (%s cache, no seam capture): %.2f tok/s, %.1f ms/token\n", pass, pass == 1 ? "cold" : "warm", (double) T / s, s / (double) T * 1e3);
        }
        eng.reset();
    }
    if (const char* bs = std::getenv("STRATA_INKLING2_BLOCK_CHECK")) {   // blocks of N tokens against single steps: the logits must be identical
        const int N = std::max(1, std::atoi(bs));
        std::vector<std::vector<float>> single((size_t) T);
        eng.reset();
        for (int64_t t = 0; t < T; ++t) eng.step(toks[(size_t) t], (int) t, &single[(size_t) t]);
        eng.reset();
        double worst = 0; int64_t diff = 0;
        const auto tb = std::chrono::steady_clock::now();
        for (int64_t t0 = 0; t0 < T; t0 += N) {
            const int n = (int) std::min<int64_t>(N, T - t0);
            std::vector<std::vector<float>> lg;
            eng.step_block(toks.data() + t0, n, (int) t0, &lg);
            for (int i = 0; i < n; ++i)
                for (size_t v = 0; v < lg[(size_t) i].size(); ++v) { const double d = std::fabs((double) lg[(size_t) i][v] - (double) single[(size_t) (t0 + i)][v]); worst = std::max(worst, d); diff += d > 0; }
        }
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - tb).count();
        std::printf("BLOCK CHECK N=%d: max |logit difference| %.3g, %lld of %lld logits differ; %.2f tok/s in blocks\n", N, worst, (long long) diff, (long long) (T * c.n_vocab_real), (double) T / s);
        eng.reset();
    }
    const auto t1 = std::chrono::steady_clock::now();
    for (int64_t t = 0; t < T; ++t) {
        std::vector<float> logits;
        eng.step(toks[(size_t) t], (int) t, &logits, [&](int visit, const char* name, const float* dev, int n) {
            const bool is_out = std::strcmp(name, "out") == 0;
            if (!is_out && !keep(visit)) return;
            std::vector<float> row((size_t) n);
            cudaMemcpy(row.data(), dev, (size_t) n * 4, cudaMemcpyDeviceToHost);
            if (is_out) out_rows[(size_t) visit].insert(out_rows[(size_t) visit].end(), row.begin(), row.end());
            if (keep(visit)) { auto& s = seams_kept[std::to_string(visit) + ":" + name]; s.insert(s.end(), row.begin(), row.end()); }
        });
        for (int v = 0; v < c.n_visits; ++v) picks[(size_t) v].insert(picks[(size_t) v].end(), eng.last_picks()[(size_t) v].begin(), eng.last_picks()[(size_t) v].end());
        double mx = -1e300, z = 0;
        for (float x : logits) mx = std::max(mx, (double) x);
        for (float x : logits) z += std::exp((double) x - mx);
        const double lse = mx + std::log(z);
        for (int64_t v = 0; v < c.n_vocab_real; ++v) lp[(size_t) (t * c.n_vocab_real + v)] = (float) ((double) logits[(size_t) v] - lse);
    }
    const double run_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    std::printf("%lld tokens teacher-forced in %.1f s (%.2f tok/s with every expert read zero-copy over PCIe, no cache, one launch a matvec)\n", (long long) T, run_s, (double) T / run_s);
    int fails = 0;
    if (!eng.profile().empty()) std::printf("%s\n", eng.profile().c_str());
    std::printf("seams of visits 0 / 5 (full) / 18 (first loop message) / 23 (a full layer in pass 2), relative error against the reference (max |d| / max |ref|):\n");
    for (const auto& kv : seams_kept) {
        const size_t colon = kv.first.find(':');
        const std::string vis = kv.first.substr(0, colon), nm = kv.first.substr(colon + 1);
        try {
            const C::Mat ref = I::load_npy(dir + "/blk." + vis + "." + nm + ".npy");
            if (ref.size() != kv.second.size()) { std::printf("    visit %2s %-12s size %zu vs %zu\n", vis.c_str(), nm.c_str(), kv.second.size(), ref.size()); continue; }
            double mx = 0, md = 0;
            for (size_t i = 0; i < ref.size(); ++i) { mx = std::max(mx, (double) std::fabs(ref[i])); md = std::max(md, (double) std::fabs(kv.second[i] - ref[i])); }
            std::printf("    visit %2s %-12s rel %.2e\n", vis.c_str(), nm.c_str(), md / std::max(mx, 1e-12));
        } catch (const std::exception&) { /* a seam the reference does not record */ }
    }
    double worst_visit = 0;
    int pick_diff_total = 0;
    for (int v = 0; v < c.n_visits; ++v) {
        const C::Mat ref = I::load_npy(dir + "/blk." + std::to_string(v) + ".out.npy");
        double mx = 0, md = 0;
        for (size_t i = 0; i < ref.size(); ++i) { mx = std::max(mx, (double) std::fabs(ref[i])); md = std::max(md, (double) std::fabs(out_rows[(size_t) v][i] - ref[i])); }
        const double rel = md / std::max(mx, 1e-12);
        const C::Mat pr = I::load_npy(dir + "/blk." + std::to_string(v) + ".picks.npy");
        int bad = 0;
        for (int64_t t = 0; t < T; ++t) {
            std::vector<int> got(picks[(size_t) v].begin() + t * c.n_expert_used, picks[(size_t) v].begin() + (t + 1) * c.n_expert_used);
            std::sort(got.begin(), got.end());
            for (int64_t j = 0; j < c.n_expert_used; ++j) bad += got[(size_t) j] != (int) pr[(size_t) (t * c.n_expert_used + j)];
        }
        pick_diff_total += bad;
        worst_visit = std::max(worst_visit, rel);
        if (v % 6 == 0 || v == c.n_visits - 1 || rel > 0.1) std::printf("  visit %2d (layer %2lld): `out` rel %.2e, picks differing %d of %lld\n", v, (long long) c.stored_of_visit(v), rel, bad, (long long) (T * c.n_expert_used));
    }
    std::printf("visits: worst `out` rel %.2e; picks differing in all: %d of %lld\n", worst_visit, pick_diff_total, (long long) (c.n_visits * T * c.n_expert_used));
    std::vector<int64_t> sh;
    const C::Mat ref = I::load_npy(dir + "/logprobs.npy", sh);
    const int64_t NV = c.n_vocab_real;
    int top1 = 0;
    double sum_abs = 0, max_abs = 0, kl = 0;
    for (int64_t t = 0; t < T; ++t) {
        int64_t a = 0, b = 0;
        double klt = 0;
        for (int64_t v = 0; v < NV; ++v) {
            const double d = std::fabs((double) lp[(size_t) (t * NV + v)] - (double) ref[(size_t) (t * NV + v)]);
            sum_abs += d; max_abs = std::max(max_abs, d);
            if (lp[(size_t) (t * NV + v)] > lp[(size_t) (t * NV + a)]) a = v;
            if (ref[(size_t) (t * NV + v)] > ref[(size_t) (t * NV + b)]) b = v;
            const double pr = std::exp((double) ref[(size_t) (t * NV + v)]);
            if (pr > 1e-12) klt += pr * ((double) ref[(size_t) (t * NV + v)] - (double) lp[(size_t) (t * NV + v)]);
        }
        top1 += a == b;
        kl += klt;
    }
    kl /= (double) T;
    std::printf("log-probabilities over %lld x %lld: top-1 the same at %d of %lld positions, mean |dlogp| %.3e, max |dlogp| %.3f, mean KL(reference || engine) %.4f\n", (long long) T, (long long) NV,
                top1, (long long) T, sum_abs / (double) (T * NV), max_abs, kl);
    const bool ok = worst_visit < 0.25 && top1 * 10 >= (int) T * 8 && kl < 0.1;
    if (!ok) ++fails;
    std::printf("%s (bounds: worst visit rel < 0.25, top-1 >= 80%%, mean KL < 0.1: the int8 activation path's envelope, to be tightened by what is measured)\n", ok ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
