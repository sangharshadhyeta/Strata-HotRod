// src/inkling2/serve_main.cpp - `inkling2_serve --serve --model <gguf> [--max-context N] [--eos a,b]`: the Inkling 2 engine behind serve/server.py's engine protocol (the same
// stdin / stdout lines the Qwen engine and Maya's maya-ik2 speak: INFO / READY, GEN, PP, T, DONE, STOP, QUIT), so the existing Python server (tokenizer, chat template, the
// OpenAI / Anthropic endpoints) drives it unchanged.  M4 of docs/INKLING-PORT.md.
//
// A conversation that continues the last one reuses its positions (the per-visit conv state is recurrent and cannot rewind: any other prompt starts again from position 0).
// The prompt is read one token at a time through the decode step (no batched prefill yet: docs/INKLING-PORT.md lists it); sampling is on the host from the step's logits
// (greedy, or temperature / top-k / top-p / min-p with the repetition, frequency and presence penalties over the last n tokens).
#include <cuda_runtime.h>
#include "strata/core/online_tuner.hpp"
#include "strata/inkling2/engine.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/spec/draft_policy.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace I = strata::inkling2;
#define CK_SYNC() cudaDeviceSynchronize()
static const int S_WIN = [] { const char* e = std::getenv("STRATA_INKLING2_SPEC"); const int v = e ? std::atoi(e) : 4; return std::min(8, std::max(2, v)); }();   // --spec: the window's tokens

namespace {
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

struct Sampling {
    float temperature = 0.0f, top_p = 1.0f, min_p = 0.0f, penalty_repeat = 1.0f, penalty_freq = 0.0f, penalty_present = 0.0f;
    int top_k = 64, penalty_last_n = 64;
    unsigned long long seed = 0;
    bool has_seed = false;
};

std::deque<std::string> g_lines;
std::mutex g_mu;
std::condition_variable g_cv;
std::atomic<bool> g_cancel{false}, g_quit{false};

void stdin_thread() {
    std::string l;
    while (std::getline(std::cin, l)) {
        if (l == "STOP" || l.rfind("STOP ", 0) == 0) { g_cancel = true; continue; }
        if (l == "QUIT") { g_quit = true; g_cancel = true; g_cv.notify_all(); break; }
        std::lock_guard<std::mutex> lk(g_mu);
        g_lines.push_back(l);
        g_cv.notify_one();
    }
    g_quit = true;
    g_cancel = true;
    g_cv.notify_all();
}

/// The next token from the logits (the real vocabulary).  `recent` = the tokens the penalties see.
int sample(std::vector<float>& lg, const Sampling& s, const std::vector<int>& recent, std::mt19937_64& rng) {
    const int V = (int) lg.size();
    if (s.penalty_repeat != 1.0f || s.penalty_freq != 0.0f || s.penalty_present != 0.0f) {
        const size_t from = recent.size() > (size_t) s.penalty_last_n ? recent.size() - (size_t) s.penalty_last_n : 0;
        std::vector<std::pair<int, int>> cnt;
        for (size_t i = from; i < recent.size(); ++i) {
            bool f = false;
            for (auto& c : cnt) if (c.first == recent[i]) { ++c.second; f = true; break; }
            if (!f) cnt.push_back({recent[i], 1});
        }
        for (auto& c : cnt) {
            float& x = lg[(size_t) c.first];
            if (s.penalty_repeat != 1.0f) x = x > 0 ? x / s.penalty_repeat : x * s.penalty_repeat;
            x -= s.penalty_freq * (float) c.second + s.penalty_present;
        }
    }
    if (s.temperature <= 0.0f) return (int) (std::max_element(lg.begin(), lg.end()) - lg.begin());
    const int k = std::min(std::max(s.top_k, 1), std::min(64, V));
    // the k best by one pass and a small min-heap: after the first k it is one compare a logit (a partial_sort over 200K candidates took ~1.5 ms a token)
    std::vector<std::pair<float, int>> heap;
    heap.reserve((size_t) k);
    const auto worse = [](const std::pair<float, int>& a, const std::pair<float, int>& b) { return a.first > b.first; };   // min-heap on the value
    for (int i = 0; i < V; ++i) {
        if ((int) heap.size() < k) { heap.push_back({lg[(size_t) i], i}); if ((int) heap.size() == k) std::make_heap(heap.begin(), heap.end(), worse); }
        else if (lg[(size_t) i] > heap.front().first) { std::pop_heap(heap.begin(), heap.end(), worse); heap.back() = {lg[(size_t) i], i}; std::push_heap(heap.begin(), heap.end(), worse); }
    }
    std::sort(heap.begin(), heap.end(), [](const std::pair<float, int>& a, const std::pair<float, int>& b) { return a.first > b.first; });
    std::vector<int> idx((size_t) k);
    for (int i = 0; i < k; ++i) idx[(size_t) i] = heap[(size_t) i].second;
    std::vector<double> p((size_t) k);
    const double mx = lg[(size_t) idx[0]];
    double z = 0;
    for (int i = 0; i < k; ++i) { p[(size_t) i] = std::exp(((double) lg[(size_t) idx[(size_t) i]] - mx) / (double) s.temperature); z += p[(size_t) i]; }
    for (double& x : p) x /= z;
    int keep = k;
    if (s.min_p > 0.0f) { int n = 0; while (n < keep && p[(size_t) n] >= (double) s.min_p * p[0]) ++n; keep = std::max(n, 1); }
    if (s.top_p < 1.0f) { double c = 0; int n = 0; while (n < keep) { c += p[(size_t) n]; ++n; if (c >= (double) s.top_p) break; } keep = std::max(n, 1); }
    double zz = 0;
    for (int i = 0; i < keep; ++i) zz += p[(size_t) i];
    double r = std::uniform_real_distribution<double>(0.0, zz)(rng), c = 0;
    for (int i = 0; i < keep; ++i) { c += p[(size_t) i]; if (r <= c) return idx[(size_t) i]; }
    return idx[(size_t) keep - 1];
}

void parse_keys(const std::vector<std::string>& toks, size_t from, size_t to, Sampling& s) {
    for (size_t i = from; i < to; ++i) {
        const size_t eq = toks[i].find('=');
        if (eq == std::string::npos) continue;
        const std::string k = toks[i].substr(0, eq), v = toks[i].substr(eq + 1);
        if (k == "temperature") s.temperature = std::stof(v);
        else if (k == "top_p") s.top_p = std::stof(v);
        else if (k == "top_k") s.top_k = std::stoi(v);
        else if (k == "min_p") s.min_p = std::stof(v);
        else if (k == "penalty_repeat") s.penalty_repeat = std::stof(v);
        else if (k == "penalty_freq") s.penalty_freq = std::stof(v);
        else if (k == "penalty_present") s.penalty_present = std::stof(v);
        else if (k == "penalty_last_n") s.penalty_last_n = std::stoi(v);
        else if (k == "seed") { s.seed = std::stoull(v); s.has_seed = true; }
    }
}
}  // namespace

int main(int argc, char** argv) {
    std::string model;
    int ctx = 32768;
    std::vector<int> eos;
    bool serve = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--serve") serve = true;
        else if (a == "--model" && i + 1 < argc) model = argv[++i];
        else if (a == "--max-context" && i + 1 < argc) ctx = std::atoi(argv[++i]);
        else if (a == "--eos" && i + 1 < argc) { std::stringstream ss(argv[++i]); std::string t; while (std::getline(ss, t, ',')) eos.push_back(std::atoi(t.c_str())); }
    }
    if (!serve || model.empty()) { std::fprintf(stderr, "usage: inkling2_serve --serve --model <gguf> [--max-context N] [--eos a,b]\n"); return 2; }
    try {
        const strata::GgufFile g(model);
        I::Config c;
        I::Layout lay;
        std::string e = I::parse_config(g.metadata(), c);
        if (e.empty()) e = I::load_layout(g, c, lay);
        if (!e.empty()) { std::printf("ERR %s\n", e.c_str()); std::fflush(stdout); return 1; }
        if (eos.empty()) {
            const auto& md = g.metadata();
            if (auto it = md.find("tokenizer.ggml.eos_token_id"); it != md.end()) eos.push_back((int) it->second.u);
            eos.push_back(199999);   // <|endoftext|>
        }
        I::Engine eng(g, c, lay, ctx);
        std::thread(stdin_thread).detach();
        std::printf("INFO engine=inkling2 engine_kind=inkling2 model=Inkling-2 ctx=%d\n", ctx);
        std::printf("READY %d stop\n", ctx);
        std::fflush(stdout);
        std::vector<int> fed;   // the tokens the engine's state holds (positions 0 .. fed.size() - 1)
        std::mt19937_64 rng(std::random_device{}());
        while (!g_quit) {
            std::string line;
            {
                std::unique_lock<std::mutex> lk(g_mu);
                g_cv.wait(lk, [] { return !g_lines.empty() || g_quit; });
                if (g_lines.empty()) break;
                line = g_lines.front();
                g_lines.pop_front();
            }
            if (line.rfind("GEN ", 0) != 0) continue;   // (VRAM, SAVE ... are not part of this engine)
            g_cancel = false;
            std::vector<std::string> toks;
            { std::stringstream ss(line); std::string t; while (ss >> t) toks.push_back(t); }
            if (toks.size() < 3) { std::printf("ERR malformed GEN\nDONE 0 0 0 0 cancel 0 0 0\n"); std::fflush(stdout); continue; }
            const int max_new = std::atoi(toks[1].c_str());
            Sampling sp;
            parse_keys(toks, 2, toks.size() - 1, sp);
            if (sp.has_seed) rng.seed(sp.seed);
            std::vector<int> ids;
            { std::stringstream ss(toks.back()); std::string t; while (std::getline(ss, t, ',')) if (!t.empty()) ids.push_back(std::atoi(t.c_str())); }
            if (ids.empty() || (int) ids.size() >= ctx) {
                std::printf("ERR the prompt of %zu tokens does not fit the context of %d\nDONE 0 %zu 0 0 cancel 0 0 0\n", ids.size(), ctx, ids.size());
                std::fflush(stdout);
                continue;
            }
            const int room = ctx - (int) ids.size();
            const int limit = std::min(max_new <= 0 ? room : max_new, room);
            // continue the conversation when the prompt extends what the state holds, else start again
            size_t start = 0;
            if (ids.size() > fed.size() && std::equal(fed.begin(), fed.end(), ids.begin())) start = fed.size();
            else { eng.reset(); fed.clear(); }
            if (start > 0) std::printf("RESUME %zu\n", start);
            const auto t0 = Clock::now();
            std::vector<float> logits;
            bool cancelled = false;
            for (size_t i = start; i < ids.size();) {   // the prompt in blocks of 8 (visit-major, bit-identical to single steps); logits only for the last token
                if (g_cancel) { cancelled = true; break; }
                const int n = (int) std::min<size_t>(8, ids.size() - i);
                std::vector<int64_t> tk(ids.begin() + (long) i, ids.begin() + (long) i + n);
                const bool last = i + (size_t) n == ids.size();
                std::vector<std::vector<float>> lg;
                if (n == 1) { std::vector<float> l1; eng.step(tk[0], (int) i, last ? &l1 : nullptr); if (last) lg.assign(1, l1); }
                else eng.step_block(tk.data(), n, (int) i, last ? &lg : nullptr);
                eng.spec_commit((int) i, n);
                fed.insert(fed.end(), ids.begin() + (long) i, ids.begin() + (long) i + n);
                if (last) logits = lg.back();
                i += (size_t) n;
                if ((i - start) % 16 == 0 || last) {
                    const double ms = ms_since(t0);
                    std::printf("PP %zu %zu %.0f %.1f\n", i, ids.size(), ms, (double) (i - start) / (ms / 1e3));
                    std::fflush(stdout);
                }
            }
            const double prompt_ms = ms_since(t0);
            const auto t1 = Clock::now();
            const char* reason = "length";
            int n_out = 0;
            std::vector<int> recent(ids.end() - std::min<long>((long) ids.size(), sp.penalty_last_n), ids.end());
            const bool spec_run = !cancelled && eng.spec_enabled() && sp.penalty_repeat == 1.0f && sp.penalty_freq == 0.0f && sp.penalty_present == 0.0f && std::getenv("STRATA_INKLING2_NO_SPEC") == nullptr;
            if (spec_run) {
                // The Qwen engine's rule, with its own code: `--spec S` windows of at most S tokens, the draft layer extends a window by another guess while its probability for the last one is >= spec_min_p
                // (mtp_draft), spec::DraftPolicy learns the cost and the tokens committed per window size, and the core OnlineTuner moves spec_min_p over the same levels (0.3 .. 0.7).  Greedy: the
                // longest agreeing prefix is kept and the target's own next token follows it.
                static strata::spec::DraftPolicy policy(S_WIN);
                static strata::core::OnlineTuner tuner = [] {
                    strata::core::OnlineTuner t;
                    const std::vector<double> lv = {0.3, 0.4, 0.5, 0.6, 0.7};
                    t.add_knob("spec_min_p", lv, 2, true);
                    return t;
                }();
                static long total_prop = 0, total_acc = 0, total_steps = 0, windows_by_t[9] = {0};
                static double ms_draft = 0, ms_verify = 0, ms_commit = 0;
                const auto argmax = [](const std::vector<float>& v) { return (int) (std::max_element(v.begin(), v.end()) - v.begin()); };
                bool stopped = false, consistent = true;
                const auto emit = [&](int t) {
                    ++n_out;
                    std::printf("T %d\n", t);
                    std::fflush(stdout);
                    if (std::find(eos.begin(), eos.end(), t) != eos.end()) { reason = "stop"; stopped = true; }
                    else if (n_out >= limit) stopped = true;
                    return !stopped;
                };
                // sampled requests: the pipeline's probabilistic acceptance (core/spec_prob.hpp via kernels::sample_tokens_spec); the drafter is deterministic, so its drafts are point masses
                const bool sampled = sp.temperature > 0.0f;
                const std::vector<int> no_recent;
                strata::kernels::SamplerParams spp;
                spp.top_k = sp.top_k; spp.top_p = sp.top_p; spp.min_p = sp.min_p; spp.temperature = sp.temperature;
                spp.seed = sp.has_seed ? sp.seed : (unsigned long long) std::random_device{}();
                int tok = sampled ? sample(logits, sp, no_recent, rng) : argmax(logits);
                emit(tok);
                while (!stopped) {
                    if (g_cancel) { reason = "cancel"; break; }
                    const auto tr = Clock::now();
                    const int P = (int) fed.size();
                    const double min_p = tuner.value(0);
                    int d[8];
                    float pr[8];
                    const int maxd = std::min({S_WIN - 1, 7, ctx - P - 2, std::max(1, limit - n_out)});
                    if (maxd < 1) { // no room for a window: a plain step
                        std::vector<float> l1;
                        eng.step(tok, P, &l1);
                        eng.spec_commit(P, 1);
                        fed.push_back(tok);
                        tok = sampled ? sample(l1, sp, no_recent, rng) : argmax(l1);
                        if (!emit(tok)) break;
                        continue;
                    }
                    const auto t_a = Clock::now();
                    eng.spec_propose(tok, P, maxd, d, pr);
                    ms_draft += ms_since(t_a);
                    int k = 1;                                     // the first guess always; each further one only while the last is sure enough (mtp_draft's rule)
                    while (k < maxd && pr[k - 1] >= (float) min_p) ++k;
                    const auto pk = policy.choose(k + 1, 0, 0);    // (no lookup source here: the policy keeps the MTP window and learns its cost)
                    (void) pk;
                    int blk[8];
                    blk[0] = tok;
                    for (int j = 0; j < k; ++j) blk[j + 1] = d[j];
                    std::vector<std::vector<float>> lg;
                    std::vector<int64_t> bt(blk, blk + k + 1);
                    const auto t_b = Clock::now();
                    eng.spec_begin_block();
                    eng.step_block(bt.data(), k + 1, P, &lg);
                    ms_verify += ms_since(t_b);
                    int m = 0, newtok = -1;
                    if (!sampled) { while (m < k && argmax(lg[(size_t) m]) == d[m]) ++m; newtok = argmax(lg[(size_t) m]); }
                    else {
                        spp.counter = (unsigned long long) P;   // the window's first position, as the pipeline keys its Philox draws
                        int outv[8];
                        if (eng.spec_sample_block((const int32_t*) d, k + 1, spp, outv)) { while (m < k && d[m] == outv[m]) ++m; newtok = outv[m]; }
                        else { while (m < k) { const int t = sample(lg[(size_t) m], sp, no_recent, rng); if (t != d[m]) { newtok = t; break; } ++m; } if (newtok < 0) newtok = sample(lg[(size_t) k], sp, no_recent, rng); }
                    }
                    const auto t_c = Clock::now();
                    eng.spec_end_block(m + 1);
                    eng.spec_commit(P, m + 1);
                    CK_SYNC();
                    ms_commit += ms_since(t_c);
                    fed.push_back(tok);
                    for (int j = 0; j < m; ++j) fed.push_back(d[j]);
                    const double round_ms = ms_since(tr);
                    policy.observe(false, k + 1, m, 0, round_ms);
                    tuner.window(round_ms, m + 1);
                    total_prop += k; total_acc += m; ++total_steps; ++windows_by_t[k + 1];
                    for (int j = 0; j < m && !stopped; ++j) emit(d[j]);
                    if (stopped) { if (m > 0) consistent = false; break; }
                    tok = newtok;
                    if (!emit(tok)) break;
                }
                if (!consistent) { fed.clear(); eng.reset(); }
                std::fprintf(stderr, "inkling2 serve: per window: draft %.1f ms, verify %.1f ms, commit %.1f ms\n", ms_draft / std::max(1L, total_steps), ms_verify / std::max(1L, total_steps), ms_commit / std::max(1L, total_steps));
                std::fprintf(stderr, "inkling2 serve: speculation (S=%d, min_p %.2f): %ld drafts proposed, %ld accepted over %ld windows; window cost ms by size:", S_WIN, tuner.value(0), total_prop, total_acc, total_steps);
                for (int t = 2; t <= S_WIN; ++t) std::fprintf(stderr, " %d: %.0f", t, policy.cost_ms(t));
                std::fprintf(stderr, "\n");
            } else
            if (cancelled) reason = "cancel";
            else {
                for (int i = 0; i < limit; ++i) {
                    if (g_cancel) { reason = "cancel"; break; }
                    const int tok = sample(logits, sp, recent, rng);
                    ++n_out;
                    std::printf("T %d\n", tok);
                    std::fflush(stdout);
                    if (std::find(eos.begin(), eos.end(), tok) != eos.end()) { reason = "stop"; break; }
                    recent.push_back(tok);
                    if (i + 1 == limit) break;
                    eng.step(tok, (int) fed.size(), &logits);
                    eng.spec_commit((int) fed.size(), 1);
                    fed.push_back(tok);
                }
            }
            std::printf("DONE %d %zu %.1f %.1f %s 0 0 %zu\n", n_out, ids.size(), prompt_ms, ms_since(t1), reason, start);
            std::fflush(stdout);
        }
    } catch (const std::exception& ex) {
        std::printf("ERR %s\n", ex.what());
        std::fflush(stdout);
        return 1;
    }
    return 0;
}
