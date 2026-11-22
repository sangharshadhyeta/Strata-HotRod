// include/strata/kernels/moe_slot_cache.hpp - a model-independent VRAM slot cache for MoE experts (class B: the experts sit in pinned RAM, the hot ones in VRAM).
//
// Nothing here knows a model: the caller says how many layers and experts there are, how many bytes an expert's two planes take, and where an expert's bytes are on the
// host (a base pointer + plane size per layer and plane).  Per layer visit the host reads the router's picks, `place` copies the missing experts into least-recently-used
// slots on a copy stream (a hit costs nothing) and hands back the slot addresses for moe::experts_gate_up / experts_down; the compute stream waits on an event.
// Switches (read by `from_env`, shared by every engine that uses it): STRATA_MOE_SLOTS=N (0 = off), the reserve is the caller's (the Inkling engine: STRATA_VRAM_RESERVE_MIB, Qwen's --vram-reserve-mib default 700).
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace strata::kernels::moe {

class SlotCache {
public:
    SlotCache() = default;
    SlotCache(const SlotCache&) = delete;
    SlotCache& operator=(const SlotCache&) = delete;
    ~SlotCache() {
        if (mem_) cudaFree(mem_);
        if (cs_) cudaStreamDestroy(cs_);
        if (ev_) cudaEventDestroy(ev_);
        if (summary_ && hits_ + miss_ > 0) std::fprintf(stderr, "moe slot cache: %ld hits, %ld misses (%.1f%% hit), %ld prefetch copies\n", hits_, miss_, 100.0 * hits_ / (double) (hits_ + miss_), prefetched_);
    }
    /// n_slots < 0: automatic ((the free VRAM - reserve_bytes) / the slot's bytes, read after every other allocation, as the Qwen engine sizes its expert cache); 0: off.  Returns whether a cache exists (fewer than one visit's picks = none).
    bool init(long n_slots, size_t gu_bytes, size_t dn_bytes, int n_layers, int n_experts, int max_k, size_t reserve_bytes, const char* tag = "moe") {
        gu_ = gu_bytes; dn_ = dn_bytes; ne_ = n_experts; maxk_ = max_k;
        if (n_slots == 0 || gu_ + dn_ == 0) return false;
        if (n_slots < 0) {
            size_t freeb = 0, totalb = 0;
            if (cudaMemGetInfo(&freeb, &totalb) != cudaSuccess) return false;
            const size_t margin = reserve_bytes;   // what the caller knows will be allocated after the cache, plus its safety reserve (the Qwen engine's rule: free - reserve)
            n_slots = (long) ((freeb > margin ? freeb - margin : 0) / (gu_ + dn_));
        }
        n_slots = std::min<long>(n_slots, (long) n_layers * n_experts);
        if (n_slots < max_k) return false;
        n_ = (int) n_slots;
        if (cudaMalloc((void**) &mem_, (size_t) n_ * (gu_ + dn_)) != cudaSuccess) { mem_ = nullptr; n_ = 0; return false; }
        cudaStreamCreateWithFlags(&cs_, cudaStreamNonBlocking);
        cudaEventCreateWithFlags(&ev_, cudaEventDisableTiming);
        key_.assign((size_t) n_, -1); used_.assign((size_t) n_, 0);
        map_.assign((size_t) n_layers * n_experts, -1);
        summary_ = true;
        std::fprintf(stderr, "%s: VRAM slot cache of %d experts (%.1f GB; %.0f%% of the %ld experts)\n", tag, n_, (double) n_ * (gu_ + dn_) / 1e9, 100.0 * n_ / (double) ((long) n_layers * n_experts),
                     (long) n_layers * n_experts);
        return true;
    }
    /// STRATA_MOE_SLOTS: unset = automatic.
    static long slots_from_env() { const char* e = std::getenv("STRATA_MOE_SLOTS"); return e ? std::atol(e) : -1; }
    bool active() const { return n_ > 0; }
    int slots() const { return n_; }
    long hits() const { return hits_; }
    long misses() const { return miss_; }

    /// Place the K picks `idx` of layer `layer`: expert e's gate+up plane is at gu_src + e * gu_plane on the host, its down plane at dn_src + e * dn_plane (each <= the slot's
    /// planes).  Writes the slot addresses to gu_ptr[s] / dn_ptr[s] (host arrays).  The copies of misses go on the copy stream; `wait_on` (the compute stream) waits for them.
    void place(int layer, const int32_t* idx, int K, const uint8_t* gu_src, size_t gu_plane, const uint8_t* dn_src, size_t dn_plane, const uint8_t** gu_ptr, const uint8_t** dn_ptr,
               cudaStream_t wait_on) {
        ++clock_;
        bool copied = false;
        int slot_of[128];
        for (int s = 0; s < K && s < 128; ++s) {
            const int64_t k = (int64_t) layer * ne_ + idx[s];
            int sl = map_[(size_t) k];
            if (sl >= 0) ++hits_;
            else {
                ++miss_;
                int64_t best = INT64_MAX;
                for (int i = 0; i < n_; ++i)   // the least recently used slot (a free one has last use 0)
                    if (used_[(size_t) i] < best) best = used_[(size_t) i], sl = i;
                if (key_[(size_t) sl] >= 0) map_[(size_t) key_[(size_t) sl]] = -1;
                key_[(size_t) sl] = k; map_[(size_t) k] = sl;
                uint8_t* dst = mem_ + (size_t) sl * (gu_ + dn_);
                cudaMemcpyAsync(dst, gu_src + (size_t) idx[s] * gu_plane, gu_plane, cudaMemcpyHostToDevice, cs_);
                cudaMemcpyAsync(dst + gu_, dn_src + (size_t) idx[s] * dn_plane, dn_plane, cudaMemcpyHostToDevice, cs_);
                copied = true;
            }
            used_[(size_t) sl] = clock_;   // set now: a pick made earlier in this visit is never the victim
            slot_of[s] = sl;
        }
        for (int s = 0; s < K && s < 128; ++s) {
            uint8_t* base = mem_ + (size_t) slot_of[s] * (gu_ + dn_);
            gu_ptr[s] = base; dn_ptr[s] = base + gu_;
        }
        if (copied || unwaited_) { cudaEventRecord(ev_, cs_); cudaStreamWaitEvent(wait_on, ev_, 0); unwaited_ = false; }   // also covers earlier prefetch copies
    }

    /// Look-ahead: start copying the experts `idx[0..n)` of layer `layer` that are not in a slot yet (predicted picks of a visit still to come), on the copy stream, behind the
    /// copies `place` already waited for; the compute stream waits for them at the next `place`.  Never evicts a slot used by the current visit.  Returns the copies issued.
    int prefetch(int layer, const int32_t* idx, int n, const uint8_t* gu_src, size_t gu_plane, const uint8_t* dn_src, size_t dn_plane) {
        int issued = 0;
        for (int s = 0; s < n; ++s) {
            const int64_t k = (int64_t) layer * ne_ + idx[s];
            const int have = map_[(size_t) k];
            if (have >= 0) { used_[(size_t) have] = clock_; continue; }   // keep it as fresh as this visit's picks
            int sl = -1;
            int64_t best = INT64_MAX;
            for (int i = 0; i < n_; ++i)
                if (used_[(size_t) i] < best) best = used_[(size_t) i], sl = i;
            if (sl < 0 || best >= clock_) break;   // every slot belongs to this visit or to a fresher prefetch
            if (key_[(size_t) sl] >= 0) map_[(size_t) key_[(size_t) sl]] = -1;
            key_[(size_t) sl] = k; map_[(size_t) k] = sl; used_[(size_t) sl] = clock_;
            uint8_t* dst = mem_ + (size_t) sl * (gu_ + dn_);
            cudaMemcpyAsync(dst, gu_src + (size_t) idx[s] * gu_plane, gu_plane, cudaMemcpyHostToDevice, cs_);
            cudaMemcpyAsync(dst + gu_, dn_src + (size_t) idx[s] * dn_plane, dn_plane, cudaMemcpyHostToDevice, cs_);
            unwaited_ = true; ++issued; ++prefetched_;
        }
        return issued;
    }
    long prefetched() const { return prefetched_; }

    /// Start-up warm fill: puts expert (layer, e) into a free slot now.  `rank` orders the later eviction: a larger rank is evicted later (the most popular expert gets the
    /// largest), and `finish_warm` makes every later use newer than every warm entry.  False when no slot is free or the expert is already here.
    bool warm_insert(int layer, int e, int rank, const uint8_t* gu_src, size_t gu_len, const uint8_t* dn_src, size_t dn_len) {
        const int64_t k = (int64_t) layer * ne_ + e;
        if (n_ == 0 || map_[(size_t) k] >= 0) return false;
        int sl = -1;
        for (int i = 0; i < n_; ++i) if (key_[(size_t) i] < 0) { sl = i; break; }
        if (sl < 0) return false;
        key_[(size_t) sl] = k; map_[(size_t) k] = sl; used_[(size_t) sl] = rank + 1;
        uint8_t* dst = mem_ + (size_t) sl * (gu_ + dn_);
        cudaMemcpyAsync(dst, gu_src, gu_len, cudaMemcpyHostToDevice, cs_);
        cudaMemcpyAsync(dst + gu_, dn_src, dn_len, cudaMemcpyHostToDevice, cs_);
        warmed_ = std::max<long>(warmed_, rank + 1);
        return true;
    }
    void finish_warm() { cudaStreamSynchronize(cs_); clock_ = std::max<int64_t>(clock_, (int64_t) warmed_ + 1); }
    int free_slots() const { int n = 0; for (int i = 0; i < n_; ++i) n += key_[(size_t) i] < 0; return n; }

private:
    int n_ = 0, ne_ = 0, maxk_ = 0;
    size_t gu_ = 0, dn_ = 0;
    uint8_t* mem_ = nullptr;
    cudaStream_t cs_ = nullptr;
    cudaEvent_t ev_ = nullptr;
    std::vector<int64_t> key_, used_;
    std::vector<int> map_;
    int64_t clock_ = 0;
    long hits_ = 0, miss_ = 0, prefetched_ = 0, warmed_ = 0;
    bool summary_ = false, unwaited_ = false;
};

}  // namespace strata::kernels::moe
