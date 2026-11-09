// include/strata/core/subfit_shadow.hpp - SPEC-substitute-fit stage 2, shadow samples: the substitute's output on the exact input,
// computed on the CPU in idle time.
//
// The exact successor of a cut gives, for every substitution its cut window listed (layer l, true pick e, stand-in s), the exact
// input x of layer l and the true expert's output y_e.  The stand-in's output on that same x is what the learner needs as y_s; the GPU
// does not compute it (the stand-in did not run on the exact x).  This queue computes it on a worker thread at idle priority (Linux:
// SCHED_IDLE for the CPU, the idle I/O class for the expert reads), so the decode path keeps the CPU and the drive.  The learner takes
// the finished samples at the request's end (take_done), on its own thread: the learner itself is never touched by the worker.
//
// The expert is evaluated by an `Eval` function (the engine passes one built on native_expert_forward over the pack's experts.bin;
// the tests pass a fake), called only from the worker thread.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace strata::core::subfit_shadow {

constexpr int kDefaultPerCut = 64;      ///< STRATA_SUBFIT_SHADOW default: substitutions listed per cut row (0 = no shadow samples)
constexpr size_t kMaxQueued = 4096;     ///< items waiting for the CPU (the oldest dropped past it)
constexpr size_t kMaxDone = 8192;       ///< finished samples waiting for the learner (the oldest dropped past it)

int per_cut_env(const char* v);
bool cpu_ye_env(const char* v);   ///< STRATA_SUBFIT_SHADOW_CPU_YE (opt-in, default off)

struct Item {
    int layer = 0, e = 0, s = 0;
    std::vector<float> x;    ///< the exact input [n]
    std::vector<float> ye;   ///< the true expert's output on it [n]
    std::vector<float> ys;   ///< the stand-in's output on it [n] (filled by the worker)
    bool need_ye = false;    ///< STRATA_SUBFIT_SHADOW_CPU_YE=1: the exact row did not pick e, so the worker computes y_e = e(x) too
};

/// eval(layer, s, x, ys_out): the stand-in's output on x; false when it cannot (the item is dropped)
using Eval = std::function<bool(int layer, int s, const float* x, float* ys)>;

struct Stats {
    uint64_t queued = 0, evaluated = 0, dropped = 0, failed = 0;
};

class Queue {
public:
    Queue() = default;
    ~Queue();
    Queue(const Queue&) = delete;
    Queue& operator=(const Queue&) = delete;

    /// The worker starts on the first push after this (idle priority unless `idle_prio` is false: the tests).
    void set_eval(Eval eval, bool idle_prio = true);
    void push(Item&& it);
    /// The finished samples since the last call (in finishing order).
    std::vector<Item> take_done();
    /// Waits until the queue is empty and the worker idle, or `ms` passed (the tests).
    bool drain_for_tests(int ms);
    /// This interval's counts (reset by the call).
    Stats take_stats();
    size_t pending() const;

private:
    void run();
    Eval eval_;
    bool idle_prio_ = true;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<Item> in_;
    std::deque<Item> done_;
    Stats st_;
    bool busy_ = false;
    bool stop_ = false;
    std::thread th_;
};

}  // namespace strata::core::subfit_shadow
