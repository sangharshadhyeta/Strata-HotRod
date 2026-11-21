// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/ds4/ds4_pool.hpp - the DeepSeek engine's CPU workers: run(n, fn) runs fn(0..n-1) on the workers and the caller and
// returns when every job is done.  The design is glmfast::Workers' (src/core/glm_fast_state.hpp): one 64-bit ticket
// (generation << 32 | next index) claimed by CAS, so a worker that wakes late for an older batch can never take a job
// of the current one; idle workers spin for a while, then sleep on a condition variable.
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "ds4_os.hpp"
#include "ds4_timeline.hpp"
#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace strata::ds4 {

inline void cpu_relax() {
#if defined(_MSC_VER)
    _mm_pause();
#elif defined(__x86_64__) || defined(_M_X64)
    __builtin_ia32_pause();
#endif
}

class Pool {
public:
    explicit Pool(int n, const std::vector<int>& cpus = {}, int spin_us = 200) : spin_us_(spin_us) {
        for (int i = 0; i < n; ++i) {
            th_.emplace_back([this] { loop(); });
            if (!cpus.empty()) os::pin_thread(th_.back(), cpus[(size_t) i % cpus.size()]);
        }
    }
    ~Pool() {
        quit_.store(true);
        {
            std::lock_guard<std::mutex> lk(mu_);
        }
        cv_.notify_all();
        for (auto& t : th_) t.join();
    }
    int size() const { return (int) th_.size() + 1; }

    void run(int n, const std::function<void(int)>& fn) {
        start(n, fn);
        help();
        wait();
    }
    /// The batch goes to the workers; the caller may help() and must wait() before fn goes out of scope.  (Two pools
    /// - one a NUMA node - run at once this way: start both, help one, wait for both.)
    void start(int n, const std::function<void(int)>& fn) {
        n_ = n;
        if (n <= 0) return;
        const uint64_t g = (gen_.load(std::memory_order_relaxed) + 1) & 0xffffffffu;
        ticket_.store((g << 32) | 0x7fffffffu, std::memory_order_seq_cst);
        fn_ = &fn;
        total_.store(n, std::memory_order_release);
        done_.store(0, std::memory_order_relaxed);
        ticket_.store(g << 32, std::memory_order_release);
        gen_.store(g, std::memory_order_release);
        if (sleeping_.load(std::memory_order_acquire) > 0) {
            std::lock_guard<std::mutex> lk(mu_);
            cv_.notify_all();
        }
    }
    void help() {
        if (n_ > 0) work(gen_.load(std::memory_order_acquire));
    }
    void wait() {
        if (n_ <= 0) return;
        while (done_.load(std::memory_order_acquire) < n_) cpu_relax();
        n_ = 0;
    }

private:
    void work(uint64_t g) {
        for (;;) {
            uint64_t t = ticket_.load(std::memory_order_acquire);
            if ((t >> 32) != g) return;
            const int n = total_.load(std::memory_order_acquire);
            if ((int) (t & 0xffffffffu) >= n) return;
            if (!ticket_.compare_exchange_weak(t, t + 1, std::memory_order_acq_rel)) continue;
            (*fn_)((int) (t & 0xffffffffu));
            done_.fetch_add(1, std::memory_order_acq_rel);
        }
    }
    void loop() {
        uint64_t seen = gen_.load();
        for (;;) {
            const auto t0 = std::chrono::steady_clock::now();
            uint64_t g;
            for (;;) {
                g = gen_.load(std::memory_order_acquire);
                if (g != seen || quit_.load(std::memory_order_relaxed)) break;
                cpu_relax();
                if (std::chrono::steady_clock::now() - t0 > std::chrono::microseconds(spin_us_)) {
                    std::unique_lock<std::mutex> lk(mu_);
                    sleeping_.fetch_add(1);
                    cv_.wait(lk, [&] { return gen_.load() != seen || quit_.load(); });
                    sleeping_.fetch_sub(1);
                }
            }
            if (quit_.load()) return;
            seen = g;
            work(g);
        }
    }

    std::vector<std::thread> th_;
    int spin_us_;
    int n_ = 0;   // the current batch's size (the caller's side)
    const std::function<void(int)>* fn_ = nullptr;
    alignas(64) std::atomic<uint64_t> ticket_{0};
    alignas(64) std::atomic<uint64_t> gen_{0};
    alignas(64) std::atomic<int> total_{0};
    alignas(64) std::atomic<int> done_{0};
    std::atomic<int> sleeping_{0};
    std::atomic<bool> quit_{false};
    std::mutex mu_;
    std::condition_variable cv_;
};

/// File reads on threads of their own, in the order they were pushed, the pusher never waiting: the prompt path's
/// streamed experts (each part of an expert a job), unbuffered where the job has a direct descriptor and its range is
/// not all in the page cache already (the disk straight into dst, the cache neither used nor disturbed), buffered
/// otherwise or when the unbuffered read fails - the bytes land at dst + off % os::kDirectAlign either way.  A job's `left` is
/// counted down when it is done (`bad` set if it failed); wait_any() sleeps until some job finished or the timeout.
class IoQueue {
public:
    struct Job {
        os::fd_t fd = os::kBadFd, dfd = os::kBadFd;
        uint64_t off = 0, len = 0;
        uint8_t* dst = nullptr;
        std::atomic<int>* left = nullptr;
        std::atomic<bool>* bad = nullptr;
        const uint8_t* mem = nullptr;   // non-null: a copy of len bytes from memory to dst instead of a read
        const void* map = nullptr;      // the range in the file's mapping: all in the page cache, it is read buffered
        uint8_t* keep = nullptr;        // non-null (a read): the len bytes read copied here as well
        bool exact = false;             // the len bytes land exactly at dst (a ring slot: no staging slack around them)
        std::atomic<bool>* cancel = nullptr;   // set before the job starts: it is dropped unread (bad is set, left counted down)
        int tl_layer = -1;              // the Monitor's timeline (ds4_timeline.hpp): the read's layer (-1: not stamped)
        bool tl_demand = false;         // a demand read (the CPU lane waits for it), else a guess
        std::chrono::steady_clock::time_point t_push{};   // (set by push: the demand reads' queue wait drives the auto cap)
    };
    explicit IoQueue(int n) {
        // DS4_IO_GUESS_MAX: at most N guess jobs (the low queue) on the drive at once, so a demand read always finds a thread free
        // instead of waiting behind guesses that took every thread - Qwen's 24 Oct lesson (STRATA_IO_PF_THREADS=2: its 8 read-ahead
        // threads' wasted reads queued in front of the decode's own; +8-15%).  An expert is 3 jobs (its parts).
        //   unset or "auto" (the default): the cap sets itself - it starts at 6 (two guessed experts) and every 64 demand reads looks at
        //     their mean wait in the queue: above 0.3 ms one expert less (3 jobs, not below 3), else - when the cap held a guess back
        //     meanwhile - one expert more (up to every thread's 3 jobs); otherwise it holds
        //   a number: that cap, fixed;  0: no cap (all threads may read guesses, as before 2 Nov)
        const char* v = std::getenv("DS4_IO_GUESS_MAX");
        if (v == nullptr || v[0] == '\0' || std::strcmp(v, "auto") == 0) {
            auto_ = true;
            low_max_ = 6;
            low_cap_hi_ = std::max(3, 3 * n);
        } else {
            low_max_ = std::max(0, std::atoi(v));
        }
        for (int i = 0; i < n; ++i) th_.emplace_back([this] { loop(); });
    }
    /// the guess cap now (0: none) - for the request line
    int guess_cap() {
        std::lock_guard<std::mutex> lk(mu_);
        return low_max_;
    }
    ~IoQueue() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            quit_ = true;
        }
        cv_.notify_all();
        for (auto& t : th_) t.join();
    }
    /// low: a read that can wait (a guess) - the jobs of the first queue are all taken before any of this one's
    void push(const Job* jobs, int n, bool low = false) {
        {
            const auto now = auto_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            std::lock_guard<std::mutex> lk(mu_);
            for (int i = 0; i < n; ++i) {
                (low ? ql_ : q_).push_back(jobs[i]);
                if (auto_) (low ? ql_ : q_).back().t_push = now;
            }
            if (auto_ && low && low_inflight_ >= low_max_) held_ = true;   // (the cap holds guesses back now: room to grow, if demand reads do not wait)
        }
        if (n == 1) cv_.notify_one();
        else cv_.notify_all();
    }
    void wait_any(int us) {
        std::unique_lock<std::mutex> lk(done_mu_);
        const uint64_t seen = done_;
        done_cv_.wait_for(lk, std::chrono::microseconds(us), [&] { return done_ != seen; });
    }
    /// Jobs queued and not yet taken by a thread (both queues; the reads in flight are not counted)
    size_t depth() {
        std::lock_guard<std::mutex> lk(mu_);
        return q_.size() + ql_.size();
    }
    double busy_ms() const { return busy_us_.load(std::memory_order_relaxed) / 1000.0; }
    uint64_t direct_bytes() const { return direct_.load(std::memory_order_relaxed); }
    uint64_t buffered_bytes() const { return buffered_.load(std::memory_order_relaxed); }

private:
    void loop() {
        for (;;) {
            Job j;
            bool low = false;
            {
                std::unique_lock<std::mutex> lk(mu_);
                // (a guess is taken only while fewer than low_max_ guesses are in flight; a demand read always)
                auto low_ok = [&] { return !ql_.empty() && (quit_ || low_max_ <= 0 || low_inflight_ < low_max_); };   // (no cap at quit)
                cv_.wait(lk, [&] { return quit_ || !q_.empty() || low_ok(); });
                if (quit_ && q_.empty() && ql_.empty()) return;
                if (q_.empty() && !low_ok()) continue;   // (quit with guesses left that the cap holds back: around again, the predicate ends it)
                low = q_.empty();
                std::deque<Job>& qq = low ? ql_ : q_;
                j = qq.front();
                qq.pop_front();
                if (auto_ && !low) auto_demand(j);   // (a demand read's wait in the queue: the auto cap's signal)
                low = low && low_max_ > 0;   // (counted only with a cap: without one the path is as before)
                if (low) ++low_inflight_;
            }
            if (j.cancel != nullptr && j.cancel->load(std::memory_order_acquire)) {   // (dropped unread)
                if (low) low_done();
                if (j.bad != nullptr) j.bad->store(true, std::memory_order_release);
                if (j.left != nullptr) j.left->fetch_sub(1, std::memory_order_acq_rel);
                {
                    std::lock_guard<std::mutex> lk(done_mu_);
                    ++done_;
                }
                done_cv_.notify_all();
                continue;
            }
            const auto t0 = std::chrono::steady_clock::now();
            const uint64_t lead = j.off % os::kDirectAlign;
            bool ok = false;
            if (j.mem != nullptr) {
                std::memcpy(j.dst, j.mem, (size_t) j.len);
                ok = true;
            } else if (j.exact) {
                // unbuffered through a bounce buffer (the disk reads aligned spans), then the bytes to where they belong
                thread_local std::vector<uint8_t> bounce;
                const uint64_t span = os::direct_span(j.off, j.len);
                if (bounce.size() < span + os::kDirectAlign) bounce.resize((size_t) span + 2 * os::kDirectAlign);
                uint8_t* bp = (uint8_t*) (((uintptr_t) bounce.data() + os::kDirectAlign - 1) & ~(uintptr_t) (os::kDirectAlign - 1));
                if (j.dfd != os::kBadFd && !(j.map != nullptr && os::resident(j.map, (size_t) j.len)) &&
                    os::pread_direct(j.dfd, bp, j.len, j.off)) {
                    std::memcpy(j.dst, bp + lead, (size_t) j.len);
                    ok = true;
                    direct_.fetch_add(j.len, std::memory_order_relaxed);
                } else if (os::pread_par(j.fd, j.dst, j.len, j.off)) {
                    ok = true;
                    buffered_.fetch_add(j.len, std::memory_order_relaxed);
                }
            } else if (j.dfd != os::kBadFd && !(j.map != nullptr && os::resident(j.map, (size_t) j.len)) &&
                       os::pread_direct(j.dfd, j.dst, j.len, j.off)) {   // (the page cache's ones: buffered, below)
                ok = true;
                direct_.fetch_add(j.len, std::memory_order_relaxed);
            } else if (os::pread_par(j.fd, j.dst + lead, j.len, j.off)) {
                ok = true;
                buffered_.fetch_add(j.len, std::memory_order_relaxed);
            }
            if (ok && j.keep != nullptr && j.mem == nullptr) std::memmove(j.keep, j.exact ? j.dst : j.dst + lead, (size_t) j.len);   // (keep may overlap dst: the store reads in place and moves up)
            if (j.tl_layer >= 0 && j.mem == nullptr && tl::on()) tl::read(j.tl_layer, j.tl_demand, tl::ns(t0));
            busy_us_.fetch_add((uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                                   std::chrono::steady_clock::now() - t0).count(),
                               std::memory_order_relaxed);
            if (low) low_done();
            if (!ok && j.bad != nullptr) j.bad->store(true, std::memory_order_release);
            if (j.left != nullptr) j.left->fetch_sub(1, std::memory_order_acq_rel);
            {
                std::lock_guard<std::mutex> lk(done_mu_);
                ++done_;
            }
            done_cv_.notify_all();
        }
    }
    std::vector<std::thread> th_;
    std::mutex mu_, done_mu_;
    std::condition_variable cv_, done_cv_;
    std::deque<Job> q_, ql_;
    uint64_t done_ = 0;
    bool quit_ = false;
    int low_max_ = 0, low_inflight_ = 0;   // DS4_IO_GUESS_MAX (0: no cap) and the guess jobs on the drive now (both under mu_)
    // the auto cap (under mu_): 64 demand reads' queue wait per look; held_ = the cap held a guess back since the last look
    bool auto_ = false, held_ = false;
    int low_cap_hi_ = 24, aw_n_ = 0, moves_ = 0, vote_last_ = 0, vote_run_ = 0;   // (vote_*: the looks in a row that agree)
    double aw_us_ = 0.0;
    std::chrono::steady_clock::time_point said_{};   // the last cap line (0: none yet)
    void auto_demand(const Job& j) {
        aw_us_ += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - j.t_push).count();
        if (++aw_n_ < 64) return;
        const double mean = aw_us_ / aw_n_;
        const int was = low_max_;
        // a dead band between the two: down above 0.3 ms, up only below 0.15 ms (6 Nov: with one threshold it hopped 6 <-> 9 at every
        // look, 459 times in 4 minutes)
        // ... and a move only after 3 looks in a row agree (6 Nov, live: still 104-119 moves in 30 s between 3 and 6 - every look's noise
        // moved the cap; a look is 64 demand reads, so 3 agreeing looks is ~190 reads of evidence)
        const int vote = mean > 300.0 ? -1 : (mean < 150.0 && held_) ? 1 : 0;
        vote_run_ = vote != 0 && vote == vote_last_ ? vote_run_ + 1 : (vote != 0 ? 1 : 0);
        vote_last_ = vote;
        if (vote_run_ >= 3) {
            if (vote < 0) low_max_ = std::max(3, low_max_ - 3);                          // demand reads wait: one guessed expert less
            else low_max_ = std::min(low_cap_hi_, low_max_ + 3);                         // they hardly do, and guesses wait: one more
            vote_run_ = 0;
        }
        aw_us_ = 0.0;
        aw_n_ = 0;
        held_ = false;
        if (low_max_ != was) ++moves_;
        // one line at most every 30 s: the cap now, its moves since the last line and the last look's wait
        const auto now = std::chrono::steady_clock::now();
        if (moves_ > 0 && now - said_ >= std::chrono::seconds(30)) {
            std::fprintf(stderr, "ds4: read-ahead cap: %d guess reads in flight (%d moves in the last %s; demand reads waited %.0f us)\n",
                         low_max_, moves_, said_.time_since_epoch().count() == 0 ? "start" : "30 s+", mean);
            moves_ = 0;
            said_ = now;
        }
        if (low_max_ > was) cv_.notify_all();   // (room for the guesses that waited)
    }
    // a guess job is done: its place for the next guess, and a thread that waits for one may take it (only with a cap: without one
    // nothing waits on the count)
    void low_done() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            --low_inflight_;
        }
        cv_.notify_one();
    }
    std::atomic<uint64_t> busy_us_{0}, direct_{0}, buffered_{0};
};

}  // namespace strata::ds4
