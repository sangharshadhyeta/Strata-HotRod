// tools/subfit_fit/subfit_fit.cpp - SPEC-substitute-fit stage 1b: the offline fit of substitutes to real experts.  Standalone: no server, no GPU
// (CPU expert kernels only).  Reads the sample file (subfit_samples.hpp) and the pack, and reports, per layer band and per rank:
//   1. how close the substitute's output is to the true expert's output, as is and after the best scalar;
//   2. the weight nudge: ridge regression of the residual y_e - y_sub on the substitute's hidden state h_sub, rank-truncated (4, 8, 16, 32,
//      full), fitted per layer on one half of the samples and scored on the other (two-fold cross-fit);
//   3. the better substitute (an oracle: the best of up to --cands resident candidates chosen with y_e, so it is an upper bound);
//   4. per rank of the tag-2 probes: the same, with the router weight w in the error (error energy weighted by w^2, the combine's weight).
// Verdict: GO when the pooled rank-16 nudge (or the oracle best candidate) removes >= 40% of the error energy left after the scalar.
//
// usage: subfit_fit --samples FILE [--pack DIR] [--profile FILE] [--cands 64] [--max-samples 4000] [--seed 1]
//                   [--lambda 0.01,0.1,1,10,100] [--check-blob LAYER EXPERT] [--dump N]
//   --dump N: prints the first N records as stored (tag, layer, e, s, rank, w, norms of x, y_s, y_e) and exits (no pack needed).
#include "fit_math.hpp"

#include "strata/core/expert_source.hpp"
#include "strata/core/subfit_samples.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace subfit_fit;
using strata::core::subfit::Record;
using strata::core::subfit::SampleReader;

namespace {

constexpr int kExperts = 512;
constexpr int kK = 10;               // routed experts per row
const int kRanks[] = {4, 8, 16, 32};  // truncated nudge ranks; the full-rank map is the last column

struct Opts {
    std::string samples, pack = "/mnt/nvme/strata-data/packs/iq3_s", profile = "data/expert-profile.bin";
    int cands = 64;
    long long max_samples = 4000;
    uint64_t seed = 1;
    std::vector<double> lambdas = {0.01, 0.1, 1.0, 10.0, 100.0};   // relative to the mean feature energy
    bool check = false;
    int check_layer = 0, check_expert = 0;
    int dump = 0;                               // --dump N: the first N records as stored, then exit
    double cap_refusals = -1.0, moves = -1.0;   // the request-end lines: "shift cap: moves refused N" and the stand-in moves M (mode estimate)
};

// The record's data after the substitute and candidate evaluation (float storage).
struct Item {
    int tag = 0, layer = 0, e = 0, s = -1, rank = 0;
    float w = 1.0f;
    std::vector<float> x, ye, ys_file;   // ys_file: the logged y_s (tag 1 only)
    std::vector<float> hs, ys;           // the substitute s (tag 1): its hidden state and recomputed output
    std::vector<float> hb, yb;           // the best candidate (oracle) of the set: its hidden state and output
    int best = -1;                       // the best candidate's id
    double cos_s = 0, cos_best = 0;      // cosine of the substitute / of the best candidate against y_e
    bool beats_s = false;
};

double wt(const Item& it) { return it.tag == 2 ? (double) it.w * (double) it.w : 1.0; }

std::vector<float> to_f(const std::vector<double>& v) { return std::vector<float>(v.begin(), v.end()); }

void usage() {
    std::fprintf(stderr, "usage: subfit_fit --samples FILE [--pack DIR] [--profile FILE] [--cands 64] [--max-samples 4000] [--seed 1]\n"
                         "                   [--lambda 0.01,0.1,1,10,100] [--check-blob LAYER EXPERT] [--dump N]\n");
}

bool parse(int argc, char** argv, Opts& o) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](std::string& dst) {
            if (i + 1 >= argc) return false;
            dst = argv[++i];
            return true;
        };
        std::string v;
        if (a == "--samples" && next(o.samples)) continue;
        if (a == "--pack" && next(o.pack)) continue;
        if (a == "--profile" && next(o.profile)) continue;
        if (a == "--cands" && next(v)) { o.cands = std::max(1, std::atoi(v.c_str())); continue; }
        if (a == "--max-samples" && next(v)) { o.max_samples = std::atoll(v.c_str()); continue; }
        if (a == "--seed" && next(v)) { o.seed = (uint64_t) std::strtoull(v.c_str(), nullptr, 10); continue; }
        if (a == "--lambda" && next(v)) {
            o.lambdas.clear();
            std::stringstream ss(v);
            std::string t;
            while (std::getline(ss, t, ',')) o.lambdas.push_back(std::atof(t.c_str()));
            if (o.lambdas.empty()) return false;
            continue;
        }
        if (a == "--cap-refusals" && next(v)) { o.cap_refusals = std::atof(v.c_str()); continue; }
        if (a == "--moves" && next(v)) { o.moves = std::atof(v.c_str()); continue; }
        if (a == "--dump" && next(v)) { o.dump = std::max(0, std::atoi(v.c_str())); continue; }
        if (a == "--check-blob" && i + 2 < argc) {
            o.check = true;
            o.check_layer = std::atoi(argv[++i]);
            o.check_expert = std::atoi(argv[++i]);
            continue;
        }
        return false;
    }
    return o.check || !o.samples.empty();
}

// STRP profile: "STRP", then version, nl, ne, want, n_ranked (u32), then n_ranked pairs (layer u16, expert u16), in rank order.
bool read_profile(const std::string& path, int64_t n_layers, std::vector<std::vector<int>>& by_layer) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return false;
    char magic[4];
    uint32_t hdr[5];
    if (std::fread(magic, 1, 4, f) != 4 || std::fread(hdr, 4, 5, f) != 5 || std::memcmp(magic, "STRP", 4) != 0 || hdr[0] != 1) {
        std::fclose(f);
        return false;
    }
    if ((int64_t) hdr[1] != n_layers || (int) hdr[2] != kExperts) {
        std::fclose(f);
        std::fprintf(stderr, "subfit_fit: the profile is %ux%u, the samples %lldx%d\n", hdr[1], hdr[2], (long long) n_layers, kExperts);
        return false;
    }
    std::vector<uint16_t> raw((size_t) hdr[4] * 2);
    if (!raw.empty() && std::fread(raw.data(), 2, raw.size(), f) != raw.size()) {
        std::fclose(f);
        return false;
    }
    std::fclose(f);
    by_layer.assign((size_t) n_layers, {});
    for (size_t i = 0; i < hdr[4]; ++i) {
        const int l = raw[2 * i], e = raw[2 * i + 1];
        if (l < n_layers && e < kExperts) by_layer[(size_t) l].push_back(e);
    }
    return true;
}

// Reference for --check-blob: the gate/up/down rows dequantised by ggml's own to_float, in float.  Returns the largest relative error
// of the native forward's hidden state and output against it.
int check_blob(const strata::kernels::cpu::NativeFmt& f, const uint8_t* blob, int64_t layer, int64_t expert) {
    const int64_t NE = f.n_embd, NF = f.n_ff;
    std::vector<float> x((size_t) NE);
    std::mt19937 rng(7);
    std::normal_distribution<float> g(0.0f, 1.0f);
    for (float& v : x) v = g(rng);
    std::vector<float> h((size_t) NF), y((size_t) NE);
    strata::kernels::cpu::native_expert_forward(f, blob, x.data(), h.data(), y.data());
    const ggml_type_traits* tg = ggml_get_type_traits((ggml_type) f.gu_type);
    const ggml_type_traits* td = ggml_get_type_traits((ggml_type) f.d_type);
    std::vector<float> row((size_t) std::max(NE, NF));
    std::vector<float> hr((size_t) NF);
    for (int64_t r = 0; r < NF; ++r) {
        tg->to_float(blob + (size_t) r * f.gu_row, row.data(), NE);
        double gs = 0, us = 0;
        std::vector<float> ur((size_t) NE);
        tg->to_float(blob + f.up_off + (size_t) r * f.gu_row, ur.data(), NE);
        for (int64_t c = 0; c < NE; ++c) {
            gs += (double) row[(size_t) c] * x[(size_t) c];
            us += (double) ur[(size_t) c] * x[(size_t) c];
        }
        hr[(size_t) r] = (float) ((gs / (1.0 + std::exp(-gs))) * us);
    }
    std::vector<float> yr((size_t) NE, 0.0f);
    for (int64_t r = 0; r < NE; ++r) {
        td->to_float(blob + f.down_off + (size_t) r * f.d_row, row.data(), NF);
        double s = 0;
        for (int64_t c = 0; c < NF; ++c) s += (double) row[(size_t) c] * hr[(size_t) c];
        yr[(size_t) r] = (float) s;
    }
    double eh = 0, nh = 0, ey = 0, ny = 0;
    for (int64_t r = 0; r < NF; ++r) {
        eh += (h[(size_t) r] - hr[(size_t) r]) * (h[(size_t) r] - hr[(size_t) r]);
        nh += hr[(size_t) r] * hr[(size_t) r];
    }
    for (int64_t r = 0; r < NE; ++r) {
        ey += (y[(size_t) r] - yr[(size_t) r]) * (y[(size_t) r] - yr[(size_t) r]);
        ny += yr[(size_t) r] * yr[(size_t) r];
    }
    std::printf("check-blob layer %lld expert %lld: hidden rel err %.4f, output rel err %.4f (ggml to_float reference; the native path "
                "quantises the activation, so ~1%% is expected)\n", (long long) layer, (long long) expert, std::sqrt(eh / std::max(nh, 1e-30)),
                std::sqrt(ey / std::max(ny, 1e-30)));
    return std::sqrt(ey / std::max(ny, 1e-30)) < 0.05 ? 0 : 1;
}

// Accumulators of pooled error energies (each term weighted by wt()).  e_* are error energies; y2 is the energy of y_e.
struct Acc {
    double n = 0, ws = 0, y2 = 0, cos_sum = 0, beats = 0;
    double e_as = 0, e_sc = 0, e_nud[5] = {0, 0, 0, 0, 0};        // substitute: as is, scalar, nudges (4, 8, 16, 32, full)
    double e_or = 0, e_or_sc = 0;                                  // best candidate (oracle): as is, scalar
    void add(const Acc& o) {
        n += o.n; ws += o.ws; y2 += o.y2; cos_sum += o.cos_sum; beats += o.beats;
        e_as += o.e_as; e_sc += o.e_sc; e_or += o.e_or; e_or_sc += o.e_or_sc;
        for (int k = 0; k < 5; ++k) e_nud[k] += o.e_nud[k];
    }
};

double rel(double e, double y2) { return y2 > 0 ? std::sqrt(e / y2) : 0.0; }

struct Analysis {
    Acc band[4];        // by layer band
    Acc rank[kK];       // tag 2 only: by probe rank
    Acc all;
};

// One set (tag 1: sub = s, nudge and scalar on its residual; tag 2: sub = the best candidate) of one layer, two folds: the fit on one half,
// the score on the other.  The nudge is the ridge of the residual y_e - y_sub on h_sub, truncated to each rank.
void analyse_layer(const std::vector<Item*>& items, bool tag1, double lambda_rel, Analysis& out, uint64_t seed) {
    const int n = (int) items.size();
    if (n < 4) return;
    const int F = 640, N = 2560;
    const int band = std::min(3, items[0]->layer / 12);
    auto hof = [&](const Item* it) { return tag1 ? it->hs.data() : it->hb.data(); };
    auto yof = [&](const Item* it) { return tag1 ? it->ys.data() : it->yb.data(); };
    // the ridge scale: the mean per-feature energy times the count, so lambda_rel is relative
    double fe = 0;
    for (const Item* it : items)
        for (int a = 0; a < F; ++a) fe += (double) hof(it)[a] * hof(it)[a];
    const double lambda = lambda_rel * std::max(1e-12, fe / (double) n);
    for (int fold = 0; fold < 2; ++fold) {
        std::vector<double> Htr, Rtr;
        int ntr = 0;
        std::vector<const Item*> te;
        double sxy_s = 0, sxx_s = 0, sxy_b = 0, sxx_b = 0;
        for (int i = 0; i < n; ++i) {
            const Item* it = items[(size_t) i];
            if (half_of(i) == fold) {
                te.push_back(it);
                continue;
            }
            ++ntr;
            const float* h = hof(it);
            const float* ysub = yof(it);
            for (int a = 0; a < F; ++a) Htr.push_back(h[a]);
            for (int j = 0; j < N; ++j) Rtr.push_back((double) it->ye[(size_t) j] - (double) ysub[j]);
            // the scalars (training half): the substitute's and the oracle's
            // (a tag-2 item has no stand-in output: its ys is empty, so only the oracle's (yb) scalar is taken; the crash of 2026-11-09)
            for (int j = 0; j < N; ++j) {
                const double y = it->ye[(size_t) j];
                if (tag1) {
                    sxy_s += (double) it->ys[(size_t) j] * y;
                    sxx_s += (double) it->ys[(size_t) j] * it->ys[(size_t) j];
                }
                sxy_b += (double) it->yb[(size_t) j] * y;
                sxx_b += (double) it->yb[(size_t) j] * it->yb[(size_t) j];
            }
        }
        if (ntr < 2 || te.empty()) continue;
        const double c_s = sxx_s > 0 ? sxy_s / sxx_s : 0.0;
        const double c_b = sxx_b > 0 ? sxy_b / sxx_b : 0.0;
        const double c = tag1 ? c_s : c_b;   // the scalar of the set's own sub (tag 2: the oracle's)
        std::vector<double> M;
        if (!ridge_fit(Htr, Rtr, ntr, F, N, lambda, M)) continue;
        std::vector<double> Mr[5];
        for (int k = 0; k < 4; ++k) truncate_rank(M, N, F, kRanks[k], seed + (uint64_t) fold * 7 + (uint64_t) k, Mr[k], 10, 2);
        Mr[4] = M;   // full rank
        for (const Item* it : te) {
            const float* h = hof(it);
            const float* ysub = yof(it);
            const double wgt = wt(*it);
            Acc a;
            a.n = 1;
            a.ws = wgt;
            a.cos_sum = tag1 ? it->cos_s : it->cos_best;
            a.beats = it->beats_s ? 1.0 : 0.0;
            double y2 = 0, e_as = 0, e_sc = 0, e_or = 0, e_or_sc = 0;
            for (int j = 0; j < N; ++j) {
                const double y = it->ye[(size_t) j];
                y2 += y * y;
                e_as += (y - ysub[j]) * (y - ysub[j]);
                e_sc += (y - c * ysub[j]) * (y - c * ysub[j]);
                if (tag1) {   // the oracle column (best of s and the candidates): its own scalar on the training half
                    e_or += (y - it->yb[(size_t) j]) * (y - it->yb[(size_t) j]);
                    e_or_sc += (y - c_b * it->yb[(size_t) j]) * (y - c_b * it->yb[(size_t) j]);
                }
            }
            if (!tag1) {   // tag 2: the substitute IS the oracle, so its as-is and scalar columns are the oracle's
                e_or = e_as;
                e_or_sc = e_sc;
            }
            a.y2 = wgt * y2;
            a.e_as = wgt * e_as;
            a.e_sc = wgt * e_sc;
            a.e_or = wgt * e_or;
            a.e_or_sc = wgt * e_or_sc;
            for (int k = 0; k < 5; ++k) {
                double e = 0;
                for (int j = 0; j < N; ++j) {
                    double p = ysub[j];
                    for (int aa = 0; aa < F; ++aa) p += Mr[k][(size_t) j * F + aa] * h[aa];
                    const double d = it->ye[(size_t) j] - p;
                    e += d * d;
                }
                a.e_nud[k] = wgt * e;
            }
            out.band[band].add(a);
            out.all.add(a);
            if (!tag1) out.rank[std::min(kK - 1, it->rank)].add(a);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    Opts o;
    if (!parse(argc, argv, o)) {
        usage();
        return 2;
    }
    std::string err;
    if (o.check) {
        std::string lerr;
        if (!strata::kernels::cpu::expert_layout_load(o.pack, 48, kExperts, lerr)) {
            std::fprintf(stderr, "subfit_fit: %s\n", lerr.c_str());
            return 2;
        }
        strata::core::FileExpertSource src;
        if (!src.open(o.pack, 48, kExperts, lerr)) {
            std::fprintf(stderr, "subfit_fit: %s\n", lerr.c_str());
            return 2;
        }
        const auto& lay = strata::kernels::cpu::expert_layout();
        const uint8_t* blob = src.blob(o.check_layer, o.check_expert);
        return check_blob(lay.fmt[(size_t) o.check_layer], blob, o.check_layer, o.check_expert);
    }

    // the samples
    SampleReader rd;
    if (!rd.open(o.samples, err)) {
        std::fprintf(stderr, "subfit_fit: %s\n", err.c_str());
        return 2;
    }
    const auto hd = rd.header();
    if (hd.n_embd != 2560) {
        std::fprintf(stderr, "subfit_fit: this tool reads n_embd 2560 (the file has %u)\n", hd.n_embd);
        return 2;
    }
    if (o.dump > 0) {   // --dump N: the records as the reader stores them (the logger's check: no pack, no evaluation)
        auto norm = [](const std::vector<float>& v) {
            double q = 0.0;
            for (float f : v) q += (double) f * (double) f;
            return std::sqrt(q);
        };
        Record d;
        for (int k = 0; k < o.dump && rd.next(d, err); ++k)
            std::printf("#%d tag %d layer %d e %d s %d rank %d w %.6g  |x| %.6g  |y_s| %.6g  |y_e| %.6g  (x %zu, y_s %zu, y_e %zu)\n", k, d.tag,
                        d.layer, d.e, d.tag == 1 ? d.s : -1, d.rank, d.tag == 2 ? d.w : 0.0f, norm(d.x), norm(d.ys), norm(d.ye), d.x.size(),
                        d.ys.size(), d.ye.size());
        if (!err.empty()) {
            std::fprintf(stderr, "subfit_fit: %s\n", err.c_str());
            return 2;
        }
        return 0;
    }
    const int64_t n_layers = hd.n_layers;
    if (!strata::kernels::cpu::expert_layout_load(o.pack, n_layers, kExperts, err)) {
        std::fprintf(stderr, "subfit_fit: %s\n", err.c_str());
        return 2;
    }
    strata::core::FileExpertSource src;
    if (!src.open(o.pack, n_layers, kExperts, err)) {
        std::fprintf(stderr, "subfit_fit: %s\n", err.c_str());
        return 2;
    }
    const auto& lay = strata::kernels::cpu::expert_layout();
    std::vector<std::vector<int>> prof;
    if (!read_profile(o.profile, n_layers, prof)) {
        std::fprintf(stderr, "subfit_fit: cannot read the profile %s\n", o.profile.c_str());
        return 2;
    }

    // read the records (up to max-samples), evaluate the substitute and the candidates
    std::vector<Item> items;
    Record r;
    std::mt19937_64 rng(o.seed);
    double xc_num = 0, xc_den = 0;
    while ((long long) items.size() < o.max_samples && rd.next(r, err)) {
        Item it;
        it.tag = r.tag;
        it.layer = r.layer;
        it.e = r.e;
        it.s = r.tag == 1 ? r.s : -1;
        it.rank = r.rank;
        it.w = r.tag == 2 ? r.w : 1.0f;
        it.x.assign(r.x.begin(), r.x.end());
        it.ye.assign(r.ye.begin(), r.ye.end());
        const auto& fm = lay.fmt[(size_t) it.layer];
        const int NF = (int) fm.n_ff, NE = (int) fm.n_embd;
        std::vector<float> h((size_t) NF), y((size_t) NE);
        if (it.tag == 1) {
            it.ys_file = r.ys;
            strata::kernels::cpu::native_expert_forward(fm, src.blob(it.layer, it.s), it.x.data(), h.data(), y.data());
            it.hs = h;
            it.ys = y;
            it.cos_s = cosine(y.data(), r.ye.data(), NE);
            // cross-check: the recomputed y_s against the file's (the engine's, from the same x up to bf16 rounding)
            for (int j = 0; j < NE; ++j) {
                xc_num += (double) (y[(size_t) j] - r.ys[(size_t) j]) * (y[(size_t) j] - r.ys[(size_t) j]);
                xc_den += (double) r.ys[(size_t) j] * r.ys[(size_t) j];
            }
        }
        // candidates: the layer's profile experts (rank order; the top 4 always, the rest drawn at random), without e and s
        const std::vector<int>& pool0 = prof[(size_t) it.layer];
        std::vector<int> pool = pool0.empty() ? std::vector<int>() : pool0;
        if (pool.empty())
            for (int c = 0; c < kExperts; ++c) pool.push_back(c);
        std::vector<int> cand;
        for (size_t i = 0; i < pool.size() && i < 4; ++i) cand.push_back(pool[i]);
        std::vector<int> rest(pool.begin() + std::min<size_t>(4, pool.size()), pool.end());
        std::shuffle(rest.begin(), rest.end(), rng);
        for (int c : rest) {
            if ((int) cand.size() >= o.cands) break;
            cand.push_back(c);
        }
        double best_cos = -2.0;
        bool have_best = false;
        for (int c : cand) {
            if (c == it.e || c == it.s) continue;
            strata::kernels::cpu::native_expert_forward(fm, src.blob(it.layer, c), it.x.data(), h.data(), y.data());
            const double cc = cosine(y.data(), r.ye.data(), NE);
            // a non-finite cosine (a zero y_e: 0/0) never wins, but the first candidate still fills hb/yb (the set's arrays must exist:
            // the analysis reads them; 2026-11-09 the all-zero y_e of the stage-1 file segfaulted it here)
            const bool finite = std::isfinite(cc);
            if (!have_best || (finite && cc > best_cos)) {
                have_best = true;
                best_cos = finite ? cc : -2.0;
                it.best = c;
                it.hb = h;
                it.yb = y;
            }
        }
        // tag 1: the oracle is the best of the candidates and s itself
        if (it.tag == 1) {
            it.cos_best = it.cos_s;
            it.beats_s = best_cos > it.cos_s;
            if (it.beats_s) {
                it.cos_best = best_cos;
            } else {
                it.best = it.s;
                it.hb = it.hs;
                it.yb = it.ys;
            }
        } else {
            it.cos_best = best_cos;
        }
        if (it.yb.empty()) continue;   // no candidate at all
        items.push_back(std::move(it));
    }
    if (!err.empty()) std::fprintf(stderr, "subfit_fit: %s\n", err.c_str());
    std::printf("samples read: %zu (tag 1 %zu, tag 2 %zu); recomputed y_s vs the file's: relative difference %.4f\n", items.size(),
                (size_t) std::count_if(items.begin(), items.end(), [](const Item& i) { return i.tag == 1; }),
                (size_t) std::count_if(items.begin(), items.end(), [](const Item& i) { return i.tag == 2; }),
                std::sqrt(xc_num / std::max(xc_den, 1e-30)));

    std::vector<Item*> s1, s2;
    for (Item& it : items) (it.tag == 1 ? s1 : s2).push_back(&it);
    // group by layer, keeping the file order
    auto by_layer = [&](const std::vector<Item*>& v) {
        std::map<int, std::vector<Item*>> m;
        for (Item* it : v) m[it->layer].push_back(it);
        return m;
    };
    const auto g1 = by_layer(s1), g2 = by_layer(s2);

    // the lambda grid: the total set-A error at rank 16 decides the lambda (reported)
    double best_lambda = o.lambdas.front();
    Analysis best1, best2;
    double best_score = 1e300;
    for (double lam : o.lambdas) {
        Analysis a1, a2;
        for (auto& kv : g1) analyse_layer(kv.second, true, lam, a1, o.seed);
        for (auto& kv : g2) analyse_layer(kv.second, false, lam, a2, o.seed);
        const double score = a1.all.e_nud[2] / std::max(1e-30, a1.all.y2);   // rank 16, pooled, relative energy
        std::printf("lambda %.4g: tag-1 pooled rank-16 nudge relative error %.4f (scalar %.4f)\n", lam, std::sqrt(score),
                    std::sqrt(a1.all.e_sc / std::max(1e-30, a1.all.y2)));
        if (score < best_score) {
            best_score = score;
            best_lambda = lam;
            best1 = a1;
            best2 = a2;
        }
    }

    // the tables
    const char* bn[4] = {"0-11", "12-23", "24-35", "36-47"};
    std::printf("\nlambda %.4g (relative to the mean feature energy); two-fold cross-fit per layer; relative errors = sqrt(error energy / energy of y_e)\n",
                best_lambda);
    std::printf("\nstand-in substitutes (tag 1: sub = s)\n");
    std::printf("%-6s %6s %8s %8s %8s %8s %8s %8s %8s %8s %8s\n", "band", "n", "cos", "as-is", "scalar", "nud4", "nud8", "nud16", "nud32",
                "full", "beats-s");
    for (int b = 0; b < 4; ++b) {
        const Acc& a = best1.band[b];
        if (a.n == 0) continue;
        std::printf("%-6s %6.0f %8.3f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.3f\n", bn[b], a.n, a.cos_sum / a.n,
                    rel(a.e_as, a.y2), rel(a.e_sc, a.y2), rel(a.e_nud[0], a.y2), rel(a.e_nud[1], a.y2), rel(a.e_nud[2], a.y2),
                    rel(a.e_nud[3], a.y2), rel(a.e_nud[4], a.y2), a.beats / a.n);
    }
    std::printf("\nbest candidate (oracle: chosen with y_e) for tag-2 probes, by band (w^2 weighted)\n");
    std::printf("%-6s %6s %8s %8s\n", "band", "n", "oracle", "or-scl");
    for (int b = 0; b < 4; ++b) {
        const Acc& a = best2.band[b];
        if (a.n == 0) continue;
        std::printf("%-6s %6.0f %8.4f %8.4f\n", bn[b], a.n, rel(a.e_or, a.y2), rel(a.e_or_sc, a.y2));
    }
    std::printf("\nper rank of the tag-2 probes (w^2 weighted, held-out)\n");
    std::printf("%-5s %6s %8s %8s %8s\n", "rank", "n", "as-is", "scalar", "nud16");
    for (int k = 0; k < kK; ++k) {
        const Acc& a = best2.rank[k];
        if (a.n == 0) continue;
        std::printf("%-5d %6.0f %8.4f %8.4f %8.4f\n", k, a.n, rel(a.e_or, a.y2), rel(a.e_or_sc, a.y2), rel(a.e_nud[2], a.y2));
    }

    // verdict: the pooled tag-1 rank-16 nudge, or the oracle best candidate, against the scalar's remaining error
    const Acc& p = best1.all;
    const double remain = p.e_sc;
    const double nud_removed = remain > 0 ? 1.0 - p.e_nud[2] / remain : 0.0;
    const Acc& q = best1.all;
    const double oracle_removed = remain > 0 ? 1.0 - q.e_or_sc / remain : 0.0;
    std::printf("\nremoved after the scalar (tag 1, pooled): rank-16 nudge %.1f%%, oracle best candidate (scalar) %.1f%%\n",
                100.0 * nud_removed, 100.0 * oracle_removed);
    const bool go = nud_removed >= 0.4 || oracle_removed >= 0.4;
    std::printf("verdict: %s (stage 2 is built when a held-out removal reaches 40%%)\n", go ? "GO" : "NO-GO");
    // mode estimate: ALL against DYNAMIC, on the shift cap's live numbers (--cap-refusals, --moves) and the pooled gain g
    if (o.cap_refusals >= 0.0 && o.moves > 0.0) {
        const double g = nud_removed;
        const double f = moves_share(o.cap_refusals, o.moves);
        const double r_all = residual_all(g), r_dyn = residual_dynamic(f, g);
        std::printf("\nmode estimate (first order; pooled rank-16 gain g = %.3f; cap refused %.0f of %.0f moves)\n", g, o.cap_refusals, o.moves);
        std::printf("%-10s %10s %14s %12s\n", "mode", "residual", "moves corrected", "gain kept");
        std::printf("%-10s %10.4f %13.1f%% %11.1f%%\n", "ALL", r_all, 100.0, 100.0);
        const double kept = 1.0 - r_all > 0.0 ? (1.0 - r_dyn) / (1.0 - r_all) : 0.0;
        std::printf("%-10s %10.4f %13.1f%% %11.1f%%\n", "DYNAMIC", r_dyn, 100.0 * f, 100.0 * kept);
        std::printf("verdict: dynamic keeps %.0f%% of ALL's gain for %.0f%% of its corrections\n", 100.0 * kept, 100.0 * f);
    }
    return 0;
}
