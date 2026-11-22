// src/inkling2/serve_main.cpp - `inkling2_serve --serve --model <gguf> [--max-context N] [--eos a,b]`: the Inkling 2 engine behind serve/server.py's engine protocol (the same
// stdin / stdout lines the Qwen engine and Maya's maya-ik2 speak: INFO / READY, GEN, PP, T, DONE, STOP, QUIT), so the existing Python server (tokenizer, chat template, the
// OpenAI / Anthropic endpoints) drives it unchanged.  M4 of docs/INKLING-PORT.md.
//
// A conversation that continues the last one reuses its positions (the per-visit conv state is recurrent and cannot rewind: any other prompt starts again from position 0).
// The prompt is read one token at a time through the decode step (no batched prefill yet: docs/INKLING-PORT.md lists it); sampling is on the host from the step's logits
// (greedy, or temperature / top-k / top-p / min-p with the repetition, frequency and presence penalties over the last n tokens).
#include "strata/inkling2/engine.hpp"

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
                eng.step_block(tk.data(), n, (int) i, last ? &lg : nullptr);
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
