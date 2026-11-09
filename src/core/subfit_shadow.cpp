// src/core/subfit_shadow.cpp - see include/strata/core/subfit_shadow.hpp.
#include "strata/core/subfit_shadow.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace strata::core::subfit_shadow {

int per_cut_env(const char* v) {
    if (v == nullptr) return kDefaultPerCut;
    const int n = std::atoi(v);
    return std::max(0, std::min(256, n));
}

bool cpu_ye_env(const char* v) { return v != nullptr && std::atoi(v) != 0; }

Queue::~Queue() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    if (th_.joinable()) th_.join();
}

void Queue::set_eval(Eval eval, bool idle_prio) {
    std::lock_guard<std::mutex> lk(mu_);
    eval_ = std::move(eval);
    idle_prio_ = idle_prio;
}

void Queue::push(Item&& it) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (!eval_) {
            ++st_.dropped;
            return;
        }
        if (!th_.joinable()) th_ = std::thread([this] { run(); });
        if (in_.size() >= kMaxQueued) {
            in_.pop_front();
            ++st_.dropped;
        }
        in_.push_back(std::move(it));
        ++st_.queued;
    }
    cv_.notify_one();
}

std::vector<Item> Queue::take_done() {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<Item> out;
    out.reserve(done_.size());
    for (Item& it : done_) out.push_back(std::move(it));
    done_.clear();
    return out;
}

bool Queue::drain_for_tests(int ms) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
    while (std::chrono::steady_clock::now() < end) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (in_.empty() && !busy_) return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return false;
}

Stats Queue::take_stats() {
    std::lock_guard<std::mutex> lk(mu_);
    const Stats s = st_;
    st_ = Stats{};
    return s;
}

size_t Queue::pending() const {
    std::lock_guard<std::mutex> lk(mu_);
    return in_.size();
}

void Queue::run() {
#if defined(__linux__)
    bool idle = true;
    {
        std::lock_guard<std::mutex> lk(mu_);
        idle = idle_prio_;
    }
    if (idle) {
        sched_param sp{};
        sp.sched_priority = 0;
        (void) pthread_setschedparam(pthread_self(), SCHED_IDLE, &sp);   // the CPU only when nothing else wants it
        constexpr int kIoprioWhoProcess = 1, kIoprioClassIdle = 3, kIoprioClassShift = 13;
        (void) syscall(SYS_ioprio_set, kIoprioWhoProcess, 0, kIoprioClassIdle << kIoprioClassShift);   // this thread's reads: idle class
    }
#endif
    for (;;) {
        Item it;
        Eval ev;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [this] { return stop_ || !in_.empty(); });
            if (stop_) return;
            // oldest first
            it = std::move(in_.front());
            in_.pop_front();
            busy_ = true;
            ev = eval_;
        }
        it.ys.assign(it.x.size(), 0.0f);
        bool ok = ev && !it.x.empty() && ev(it.layer, it.s, it.x.data(), it.ys.data());
        if (ok && it.need_ye) {   // the true expert on the same input (the exact row did not run it at this layer)
            it.ye.assign(it.x.size(), 0.0f);
            ok = ev(it.layer, it.e, it.x.data(), it.ye.data());
        }
        {
            std::lock_guard<std::mutex> lk(mu_);
            busy_ = false;
            if (!ok) {
                ++st_.failed;
                continue;
            }
            if (done_.size() >= kMaxDone) {
                done_.pop_front();
                ++st_.dropped;
            }
            done_.push_back(std::move(it));
            ++st_.evaluated;
        }
    }
}

}  // namespace strata::core::subfit_shadow
