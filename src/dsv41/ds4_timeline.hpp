// Strata. src/dsv41/ds4_timeline.hpp - the Monitor's decode timeline for the DeepSeek engine (Strata-HotRod, ported from
// src/core/expert_source.cpp): the same rings and the same file (v 2), so the server and the page read both engines alike.
//
// STRATA_TIMELINE_FILE + STRATA_TIMELINE_CTL (the server sets both).  HOST times only: each layer the CPU lane serves (the
// routing seen, its picks built, the waits on the store's reads, its experts done) and each read the I/O threads make are
// stamped into fixed rings.  The GPU is not touched - its work shows as the time between the CPU lane's hand-offs.  On while
// the CTL file holds "1" (the server writes it at start; config "timeline": false leaves it off until switched on).  Off: one
// relaxed load per layer and per read.  A thread of its own reads the CTL file every 0.5 s and writes the windows finished
// since its last write as one chunk (times in us after base(), one clock for the whole run); a chunk also holds the windows of
// the write before it, so a reader that polls once a second gets every window.  Each ring entry carries a sequence number
// written last, so a chunk skips an entry a writer is still filling.
//
// The fields of a layer row as this engine fills them (the page's names): t0 the routing seen by the CPU lane; tp and t1 the
// same (the GPU computes its experts from the routing on, nothing to publish); t2 the picks built; tpf t2 (the store's reads
// start at once and run alongside); t3 t2 + the time the lane waited on reads; t4 the lane's experts done, the answer posted.
// misses: the lane's experts the store had to read for this layer; jobs: the lane's expert rows.
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strata::ds4::tl {

struct Layer { std::atomic<uint64_t> seq{0}; int64_t t[7]; int32_t layer, n_tok, misses, jobs; uint32_t win; uint8_t first; };
struct Read { std::atomic<uint64_t> seq{0}; int64_t t0, t1; int32_t layer, worker; uint32_t win; uint8_t demand; };
constexpr uint64_t kWindows = 64;                 // a window is one decode step: the rings hold the last 64 of them
constexpr uint64_t kLayers = 48 * kWindows;       // up to 48 layers a window (the trunk has 40)
constexpr uint64_t kReads = 65536;

struct State {
    Layer layers[kLayers];
    Read reads[kReads];
    std::atomic<uint64_t> layer_next{0}, read_next{0};
    std::atomic<bool> on{false};
    std::atomic<uint32_t> win{0};
    std::atomic<int32_t> last_layer{-1};
    std::atomic<int> workers{0};
};
inline State& S() {
    static State* s = new State();   // (never freed: the I/O threads may stamp until the process ends)
    return *s;
}

inline int64_t ns(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();
}
inline int64_t ns() { return ns(std::chrono::steady_clock::now()); }
inline int64_t base() {
    static const int64_t b = ns();   // the run's one clock: the chunks' times are us after it
    return b;
}
inline bool on() { return S().on.load(std::memory_order_relaxed); }

// the CPU lane, when a layer's answer is posted (t: the seven stamps above)
inline void layer(int32_t li, int32_t n_tok, int32_t misses, int32_t jobs, const int64_t* t) {
    State& s = S();
    const int32_t last = s.last_layer.exchange(li, std::memory_order_relaxed);
    if (li < last) s.win.fetch_add(1, std::memory_order_relaxed);
    const uint64_t i = s.layer_next.fetch_add(1, std::memory_order_relaxed);
    Layer& e = s.layers[i % kLayers];
    e.seq.store(0, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    for (int j = 0; j < 7; ++j) e.t[j] = t[j];
    e.layer = li;
    e.n_tok = n_tok;
    e.misses = misses;
    e.jobs = jobs;
    e.win = s.win.load(std::memory_order_relaxed);
    e.first = last < 0 || li < last;              // the window's first layer: the rings overwrite oldest first, so this marks a whole window
    e.seq.store(i + 1, std::memory_order_release);
}
// an I/O thread, after a read (a demand read the CPU lane waits for, or a guess)
inline void read(int32_t li, bool demand, int64_t t0) {
    State& s = S();
    static thread_local int worker = -1;
    if (worker < 0) worker = s.workers.fetch_add(1, std::memory_order_relaxed);
    const int64_t t1 = ns();
    const uint64_t i = s.read_next.fetch_add(1, std::memory_order_relaxed);
    Read& e = s.reads[i % kReads];
    e.seq.store(0, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    e.t0 = t0;
    e.t1 = t1;
    e.layer = li;
    e.worker = worker;
    e.win = s.win.load(std::memory_order_relaxed);
    e.demand = demand ? 1 : 0;
    e.seq.store(i + 1, std::memory_order_release);
}

// one round of the writer: the CTL file read, the windows finished since the last write written as a chunk
inline void tick(const char* ctl, const char* out) {
    State& s = S();
    char c = '0';
    if (FILE* f = std::fopen(ctl, "rb")) {
        if (std::fread(&c, 1, 1, f) != 1) c = '0';
        std::fclose(f);
    }
    const bool want = c == '1';
    // the newest whole window of the last write and of the write before it (-1: none); the entries from this switch-on on
    static int64_t hi_prev = -1, hi_prev2 = -1;
    static uint64_t from_layer = 0, from_read = 0;
    if (want != s.on.load(std::memory_order_relaxed)) {
        s.on.store(want, std::memory_order_relaxed);
        hi_prev = hi_prev2 = -1;
        if (want) {
            from_layer = s.layer_next.load(std::memory_order_acquire);
            from_read = s.read_next.load(std::memory_order_acquire);
        } else {
            std::error_code ec;
            std::filesystem::remove(out, ec);
        }
        return;                              // just switched: nothing to write yet
    }
    if (!want) return;
    // the windows the rings hold whole: the newest one may still be filling and waits for the next write
    struct L { int64_t t[7]; int32_t layer, n_tok, misses, jobs; uint32_t win; };
    std::vector<L> ls;
    uint32_t w0 = UINT32_MAX, wmax = 0;
    const uint64_t ln = s.layer_next.load(std::memory_order_acquire);
    for (uint64_t i = std::max(ln > kLayers ? ln - kLayers : 0, from_layer); i < ln; ++i) {
        const Layer& e = s.layers[i % kLayers];
        if (e.seq.load(std::memory_order_acquire) != i + 1) continue;
        L x;
        for (int j = 0; j < 7; ++j) x.t[j] = e.t[j];
        x.layer = e.layer; x.n_tok = e.n_tok; x.misses = e.misses; x.jobs = e.jobs; x.win = e.win;
        const bool first = e.first != 0;
        std::atomic_thread_fence(std::memory_order_acquire);
        if (e.seq.load(std::memory_order_relaxed) != i + 1) continue;   // overwritten while copied
        ls.push_back(x);
        wmax = std::max(wmax, x.win);
        if (first) w0 = std::min(w0, x.win);
    }
    if (w0 >= wmax) return;                  // no window whole yet (w0 is UINT32_MAX when none holds its first layer)
    const int64_t hi = (int64_t) wmax - 1;
    if (hi <= hi_prev) return;               // nothing new since the last write: the file stays as it is
    const int64_t lost = hi_prev >= 0 && (int64_t) w0 > hi_prev + 1 ? (int64_t) w0 - hi_prev - 1 : 0;   // overwritten before a write
    const int64_t start = std::max<int64_t>(hi_prev2 + 1, (int64_t) w0);
    auto us = [&](int64_t t) { return (double) (t - base()) / 1000.0; };
    char buf[256];
    std::snprintf(buf, sizeof buf, "{\"v\":2,\"time\":%lld,\"base_ns\":%lld,\"now_us\":%.1f,\"first\":%lld,\"last\":%lld",
                  (long long) std::time(nullptr), (long long) base(), us(ns()), (long long) start, (long long) hi);
    std::string js = buf;
    if (lost > 0) js += ",\"gap\":" + std::to_string((long long) lost);
    js += ",\"layers\":[";
    bool first = true;
    for (const L& x : ls) {
        if ((int64_t) x.win < start || (int64_t) x.win > hi) continue;
        std::snprintf(buf, sizeof buf, "%s[%u,%d,%d,%d,%d,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f]", first ? "" : ",", (unsigned) x.win,
                      x.layer, x.n_tok, x.misses, x.jobs, us(x.t[0]), us(x.t[1]), us(x.t[2]), us(x.t[3]), us(x.t[4]), us(x.t[5]), us(x.t[6]));
        js += buf;
        first = false;
    }
    js += "],\"reads\":[";
    first = true;
    const uint64_t rn = s.read_next.load(std::memory_order_acquire);
    for (uint64_t i = std::max(rn > kReads ? rn - kReads : 0, from_read); i < rn; ++i) {
        const Read& e = s.reads[i % kReads];
        if (e.seq.load(std::memory_order_acquire) != i + 1) continue;
        const int64_t t0 = e.t0, t1 = e.t1;
        const int32_t li = e.layer, worker = e.worker;
        const uint32_t win = e.win;
        const uint8_t demand = e.demand;
        std::atomic_thread_fence(std::memory_order_acquire);
        if (e.seq.load(std::memory_order_relaxed) != i + 1 || (int64_t) win < start || (int64_t) win > hi) continue;
        std::snprintf(buf, sizeof buf, "%s[%d,%d,%d,%.1f,%.1f]", first ? "" : ",", worker, li, (int) demand, us(t0), us(t1));
        js += buf;
        first = false;
    }
    js += "]}";
    const std::string tmp = std::string(out) + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        f << js;
    }
    std::error_code ec;
    std::filesystem::rename(tmp, out, ec);
    hi_prev2 = hi_prev;
    hi_prev = hi;
}

// the writer's thread: started at load when the server asked for a timeline, a round every 0.5 s (off the CPU lane)
class Writer {
public:
    void start() {
        const char* ctl = std::getenv("STRATA_TIMELINE_CTL");
        const char* out = std::getenv("STRATA_TIMELINE_FILE");
        if (ctl == nullptr || out == nullptr || ctl[0] == '\0' || out[0] == '\0' || th_.joinable()) return;
        base();
        ctl_ = ctl;
        out_ = out;
        th_ = std::thread([this] {
            std::unique_lock<std::mutex> lk(mu_);
            while (!quit_) {
                lk.unlock();
                tick(ctl_.c_str(), out_.c_str());
                lk.lock();
                cv_.wait_for(lk, std::chrono::milliseconds(500), [this] { return quit_; });
            }
        });
    }
    ~Writer() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            quit_ = true;
        }
        cv_.notify_all();
        if (th_.joinable()) th_.join();
    }

private:
    std::string ctl_, out_;
    std::thread th_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool quit_ = false;
};

}  // namespace strata::ds4::tl
