"""rebuild_generate.py: the SYCL engine's generate.cpp = our real src/program/generate.cpp + the Intel pieces of the SYCL
copy.  Every step asserts its anchor."""
import os

W = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the repository
s = open(f"{W}/src/program/generate.cpp").read()


def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:90], s.count(old))
    s = s.replace(old, new)


# 1. the SYCL helpers, the prompt path's share hook
nl = s.index("\n") + 1
s = s[:nl] + ('#include "strata/sycl_queue.hpp"   // intel_gpu_driver, intel_gpu_gen (the SYCL engine)\n'
              "namespace strata::prefill { void set_nonresident_share(double share); }   // the SYCL engine's prefill.cpp\n") + s[nl:]

# 2. --stream-experts (the flag setup gives Arc on the xe driver): the GGUF stream path, as the old SYCL copy had it - the
#    experts are read from the model's shards on demand, and the experts outside the VRAM cache are mirrored in pinned host
#    memory (steps 9-10).  It does NOT set mmap_experts (that would take the mmap path and never reach the mirror).
sub("    bool mmap_experts = false;",
    "    bool stream_experts = false;  ///< SYCL engine: --stream-experts (setup's flag on Arc): the experts read from the GGUF\n"
    "    bool mmap_experts = false;")
sub('        else if (a == "--mmap-experts") o.mmap_experts = true;',
    '        else if (a == "--mmap-experts") o.mmap_experts = true;\n'
    '        else if (a == "--stream-experts") o.stream_experts = true;   // SYCL engine: the GGUF stream path (step 9)')

# 3. Arc on the xe driver: the GPU page-faults reading ordinary (pageable) host memory, so the PCIe share defaults to 0
sub("    auto pcie_balance = [&]() {",
"""    // SYCL engine, Arc on the xe driver (Arc Pro B60/B65/B70, B580): the GPU page-faults when a kernel reads ordinary
    // (pageable) host memory, and the card times the job out (a B70: Timedout job, Fault response -EINVAL).  The experts
    // outside the VRAM cache are in pageable memory without --stream-experts, so the PCIe share defaults to 0 (the CPU
    // computes those misses) and an explicit --pcie-frac is kept with a warning.  With --stream-experts the GPU reads the
    // pinned host mirror instead (steps 9-10), so the rule does not apply - as in the old copy.  `pcie_given` is the old
    // copy's `pcie_explicit` (set before the probe above).
    if (strata::intel_gpu_driver() == "xe" && !o.stream_experts) {
        if (!pcie_given) {
            o.pcie_frac = 0.0;
            std::fprintf(stderr, "strata generate: xe driver without --stream-experts: the GPU faults on pageable host "
                                 "memory, so --pcie-frac defaults to 0 here (the CPU computes the experts outside VRAM). "
                                 "Use --stream-experts, as setup configures it, to let the GPU read a pinned mirror\\n");
        } else if (o.pcie_frac > 0.0) {
            std::fprintf(stderr, "strata generate: WARNING: --pcie-frac %.2f on the xe driver without --stream-experts "
                                 "makes the GPU read pageable host memory; that page-faulted and hung an Arc Pro B70 "
                                 "(Timedout job, device lost). Use --stream-experts, as setup configures it, or "
                                 "--pcie-frac 0\\n", o.pcie_frac);
        }
    }
    auto pcie_balance = [&]() {""")

# 4. the prompt path's stream-all walk follows the pairs outside the VRAM cache (prefill.cpp ring_slots)
sub("        strata::prefill::Prefill::set_pinned_share(share);\n    }",
    "        strata::prefill::Prefill::set_pinned_share(share);\n"
    "        // SYCL engine: the pairs outside the VRAM cache decide the prompt path's stream-all walk (prefill.cpp ring_slots)\n"
    "        int64_t nonres = 0;\n        for (const int32_t r : host_res) nonres += r < 0;\n"
    "        strata::prefill::set_nonresident_share(host_res.empty() ? 1.0 : (double) nonres / (double) host_res.size());\n    }")

# 5. at exit: leave without unwinding the GPU objects (their destructors ran against a runtime already shutting down: 139)
sub("""        save_profile("exit");   // #477: QUIT, or the server closed stdin
#if defined(STRATA_GPU_METAL)""",
"""        save_profile("exit");   // #477: QUIT, or the server closed stdin
#if defined(STRATA_GPU_SYCL)
        // SYCL engine: the requests are done and their output written; leave without unwinding the GPU objects (the OS
        // reclaims them).  Their destructors ran against a Level Zero runtime already shutting down and aborted (exit 139).
        try { dpct::get_current_device().queues_wait_and_throw(); } catch (const std::exception&) {}
        std::fflush(stdout);
        std::fflush(stderr);
        std::_Exit(0);
#endif
#if defined(STRATA_GPU_METAL)""")

# 6. every window graph and the drafter's graphs captured once at start (STRATA_WARM_GRAPHS=0: on first use)
sub("        bool first_window = native_pack;\n",
"""        bool first_window = native_pack;
        // SYCL engine: every window graph and the drafter's graphs are captured here, once, instead of on first use (a
        // 2,400-node graph takes tens of ms to finalize on this backend).  STRATA_WARM_GRAPHS=0: as before.
        {
            static const bool warm_on = [] { const char* v = std::getenv("STRATA_WARM_GRAPHS"); return !(v && v[0] == '0'); }();
            if (warm_on) {
                const Clock::time_point tw = Clock::now();
                if (!ver.warm(err) || (use_mtp && !mtp.warm(err))) {
                    std::fprintf(stderr, "strata generate: warm-up capture: %s\\n", err.c_str());
                    return 1;
                }
                std::fprintf(stderr, "strata generate: window and draft graphs captured in %.0f ms\\n",
                             std::chrono::duration<double, std::milli>(Clock::now() - tw).count());
            }
        }
""")

# 7. logprobs=K (0..20): after each "T id" an "LP logprob id:logprob ..." line (the SYCL copy's request key)
i = s.index("            double req_pcie_frac = o.pcie_frac, req_spec_min_p = (tuner_on")
s = s[:i] + "            int req_logprobs = -1;   // SYCL engine: logprobs=K (an LP line after each T line)\n" + s[i:]
sub('                    else if (key == "pcie_frac") { req_pcie_frac = std::clamp((double) fv, 0.0, 1.0); req_pcie_set = true; }',
    '                    else if (key == "logprobs") req_logprobs = std::clamp(std::atoi(tok.c_str() + eq + 1), -1, 20);\n'
    '                    else if (key == "pcie_frac") { req_pcie_frac = std::clamp((double) fv, 0.0, 1.0); req_pcie_set = true; }')
sub("""                    else std::printf("T %d\\n", (int) outv[(size_t) i]);
                    strata::core::progress_beat();""",
"""                    else std::printf("T %d\\n", (int) outv[(size_t) i]);
                    if (!rb_on && req_logprobs >= 0) {   // SYCL engine: row i of the window's head logits is token i's distribution
                        static std::vector<float> lrow;
                        static std::vector<int32_t> lord;
                        lrow.resize((size_t) ver.vocab());
                        if (!ver.copy_logits(i, lrow.data())) {
                            std::printf("LP nan\\n");
                        } else {
                            float mx = -INFINITY;
                            for (const float v : lrow) mx = std::max(mx, v);
                            double se = 0.0;
                            for (const float v : lrow) se += std::exp((double) v - mx);
                            const double lse = (double) mx + std::log(se);
                            std::printf("LP %.6f", (double) lrow[(size_t) outv[(size_t) i]] - lse);
                            if (req_logprobs > 0) {
                                lord.resize(lrow.size());
                                for (size_t v = 0; v < lrow.size(); ++v) lord[v] = (int32_t) v;
                                std::partial_sort(lord.begin(), lord.begin() + req_logprobs, lord.end(),
                                                  [&](int32_t x1, int32_t x2) { return lrow[(size_t) x1] > lrow[(size_t) x2]; });
                                for (int j = 0; j < req_logprobs; ++j)
                                    std::printf(" %d:%.6f", lord[(size_t) j], (double) lrow[(size_t) lord[(size_t) j]] - lse);
                            }
                            std::printf("\\n");
                        }
                    }
                    strata::core::progress_beat();""")

# ---- Intel pieces restored from the old SYCL copy (fde1d795e:sycl/src/program/generate.cpp).  Old-copy line ranges are in
# the comments.  Each step asserts its anchor occurs exactly once.

# 8. (feature 1) the GGUF expert source and the device-built plan's mirror table: the headers the steps below use
sub('#include "strata/core/expert_source.hpp"\n',
    '#include "strata/core/expert_source.hpp"\n'
    '#include "strata/core/gguf_expert_source.hpp"       // SYCL engine: --stream-experts (the GGUF-read expert source)\n'
    '#include "strata/kernels/resident_plan_mirror.hpp"  // SYCL engine: the pinned host mirror in the device-built verify plan\n')

# 9. (feature 1) --stream-experts: the experts read from the GGUF on demand (old copy 4274-4290).  The mirror (10) is built
# from this source, so the flag takes this branch and not the mmap one (step 2 no longer sets mmap_experts).
sub("""    strata::core::ArenaExpertSource arena_src;
    strata::core::ExpertSource* srcp = nullptr;
    if (o.mmap_experts) {
""", r"""    strata::core::ArenaExpertSource arena_src;
    strata::core::GgufExpertSource gguf_src;
    strata::core::ExpertSource* srcp = nullptr;
    if (o.stream_experts && !o.mmap_experts) {
        // SYCL engine: a card that holds every expert needs no host copy of them, and a 23 GiB machine cannot hold one.
        // The profile fill reads each blob once from the shard; the experts no slot holds are the pinned host mirror's
        // (below), and a prompt's lent slots are refilled from here.
        if (!native_pack || o.native_preset.empty()) {
            std::fprintf(stderr, "strata generate: --stream-experts needs a native (IQ) pack and --native SHARD1\n");
            return 2;
        }
        if (!gguf_src.open(o.native_preset, g.n_layers, g.n_expert, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "strata generate: experts streamed from the GGUF on demand (--stream-experts): no host "
                             "arena; every expert must fit the VRAM cache\n");
        srcp = &gguf_src;
    } else if (o.mmap_experts) {
""")

# 10. (feature 1) the pinned host mirror (old copy 4888-4981): the experts that did not fit VRAM, read once into pinned host
# memory the GPU reads over PCIe; the lend candidates (STRATA_LEND_MIRROR) too; STRATA_MIRROR_MIB caps it.  Placed after the
# profile fill and before the layer-split stages, as in the old copy.  mirror_table_d is consumed by the verify plans (steps 12-13).
sub("""    for (auto& stp : stages) {
        GpuStage& st = *stp;
        const auto& lay = strata::kernels::cpu::expert_layout();
        // the drafter and the head are already allocated by now""", r"""    // SYCL engine, plan item 2: the experts that did not fit VRAM, mirrored once in pinned host memory the GPU reads
    // over PCIe (--stream-experts has no host copy otherwise: each routed miss was an SSD read). The share of misses
    // the GPU takes is --pcie-frac; STRATA_MIRROR_MIB caps the mirror (default: MemAvailable less 4 GiB), 0 = off.
    unsigned long long* mirror_table_d = nullptr;   // [n_layers][n_expert] device-readable mirror addresses (0 = none)
    int64_t unmirrored_misses = 0;
    if (o.stream_experts && srcp == &gguf_src && o.expert_cache > 0) {
        // #1054, #1440: with a layer split the mirror holds only the first GPU's layers. The pinned memory belongs to
        // this GPU's context, and the later stages' caches do not exist yet (every later-stage expert would count as a
        // miss: 20 GiB of RAM for experts the other card then holds, or one 39 GiB pinned allocation that fails).
        const int64_t mirror_end = multi_gpu && !split_at.empty() ? split_at[0] : g.n_layers;
        std::vector<std::pair<int64_t, int64_t>> miss;
        size_t n_true_miss = 0;   // the experts no slot holds; what follows them in `miss` are the lend candidates (below)
        for (const auto& pr : profile)   // the profile's order: the most-routed misses first, if the cap is reached
            if (pr.first < mirror_end && xcache.slot_of(pr.first, pr.second) == strata::core::kNotResident)
                miss.push_back({pr.first, pr.second});
        for (int64_t l = 0; l < mirror_end; ++l)                     // pairs the profile does not list at all
            for (int64_t e = 0; e < g.n_expert; ++e)
                if (xcache.slot_of(l, e) == strata::core::kNotResident &&
                    std::find(miss.begin(), miss.end(), std::pair<int64_t, int64_t>{l, e}) == miss.end())
                    miss.push_back({l, e});
        n_true_miss = miss.size();
        // 0.1.43: the experts in the last slots - the ones a prompt's loan takes (Prefill::relayout lends the TAIL of the
        // cache) - are mirrored too, after the real misses (the cap drops them first).  A lent expert is streamed during
        // the prompt and put back after it; from the GGUF that was a pread at ~2.3 GB/s per expert, from the pinned mirror
        // it is a DMA.  Their mirror entries are never read by a decode window: it only looks at a mirror address for an
        // expert that has no slot.
        // STRATA_LEND_MIRROR=0 turns it off; =N: the last N slots (default 10% of the cache with a loan, at most 1,200).
        if (!o.no_prefill_borrow && o.prefill_chunk > 0 && !multi_gpu) {
            const char* lm = std::getenv("STRATA_LEND_MIRROR");
            const int64_t want_tail = lm ? std::atoll(lm) : std::min<int64_t>(xcache.slots() / 10, 1200);
            if (want_tail > 0) {
                const int64_t first_tail = xcache.slots() - std::min<int64_t>(want_tail, xcache.slots());
                for (int64_t l = 0; l < mirror_end; ++l)
                    for (int64_t e = 0; e < g.n_expert; ++e)
                        if (const int64_t sl = xcache.slot_of(l, e); sl != strata::core::kNotResident && sl >= first_tail)
                            miss.push_back({l, e});
            }
        }
        uint64_t avail = 0;
        if (FILE* f = std::fopen("/proc/meminfo", "r")) {
            char key[64]; unsigned long long kb = 0;
            while (std::fscanf(f, "%63s %llu kB", key, &kb) == 2)
                if (std::strcmp(key, "MemAvailable:") == 0) { avail = kb << 10; break; }
            std::fclose(f);
        }
        const char* mv = std::getenv("STRATA_MIRROR_MIB");
        const uint64_t cap = mv ? (uint64_t) std::atoll(mv) << 20 : (avail > (4ull << 30) ? avail - (4ull << 30) : 0);
        if (!miss.empty() && cap > 0) {
            const auto tm = Clock::now();
            const int64_t got = gguf_src.mirror(miss, cap, 8, err);
            if (got < 0) {
                std::fprintf(stderr, "strata generate: mirroring the experts missing from VRAM: %s (they stay on the SSD)\n",
                             err.c_str());
                err.clear();
            } else {
                std::fprintf(stderr, "strata generate: %lld of %zu experts missing from VRAM (+ %zu in the prompt loan's slots) "
                                     "mirrored in pinned host memory (%.2f GiB, %.1f s); the GPU reads them over PCIe\n",
                             (long long) got, n_true_miss, miss.size() - n_true_miss,
                             (double) gguf_src.mirrored_bytes() / 1073741824.0,
                             std::chrono::duration<double>(Clock::now() - tm).count());
                // the device-built verify plan's view of it: each (layer, expert) -> its mirror address, 0 = none
                std::vector<unsigned long long> tab((size_t) (g.n_layers * g.n_expert), 0ull);
                for (int64_t l = 0; l < g.n_layers; ++l)
                    for (int64_t e = 0; e < g.n_expert; ++e)
                        if (gguf_src.pinned(l, e))
                            tab[(size_t) (l * g.n_expert + e)] = (unsigned long long) gguf_src.device_alias(l, e);
                mirror_table_d = sycl::malloc_device<unsigned long long>(tab.size(), dpct::get_in_order_queue());
                dpct::get_in_order_queue().memcpy(mirror_table_d, tab.data(), tab.size() * sizeof(unsigned long long)).wait();
            }
        }
        unmirrored_misses = (int64_t) n_true_miss - (int64_t) (gguf_src.mirrored_bytes() ? std::count_if(miss.begin(), miss.begin() + (ptrdiff_t) n_true_miss,
            [&](const std::pair<int64_t, int64_t>& pr) { return gguf_src.pinned(pr.first, pr.second); }) : 0);
        const bool no_host = [] { const char* v = std::getenv("STRATA_VERIFY_NO_HOST"); return v && *v && std::strcmp(v, "0") != 0; }();
        if (unmirrored_misses > 0 && no_host) {
            std::fprintf(stderr, "strata generate: REFUSED: %lld experts are neither in VRAM nor mirrored; with STRATA_VERIFY_NO_HOST "
                                 "the device plan cannot run them and generation would lack a safe host fallback - raise "
                                 "STRATA_MIRROR_MIB or the free RAM, or lower --max-context\n", (long long) unmirrored_misses);
            refuse_exit(2);
        }
        // The same rule after the start: the mirror holds the start's misses only, so an expert an adaptive swap moves
        // out of VRAM would be in neither, and the device plan would wait for a host that never serves it.
        if (!miss.empty() && no_host && o.adapt_every > 0 && o.adapt_swaps > 0) {
            std::fprintf(stderr, "strata generate: adaptive swaps off: with STRATA_VERIFY_NO_HOST an expert they move out of "
                                 "VRAM would be neither there nor mirrored\n");
            o.adapt_every = 0;
        }
    }

    for (auto& stp : stages) {
        GpuStage& st = *stp;
        const auto& lay = strata::kernels::cpu::expert_layout();
        // the drafter and the head are already allocated by now""")

# 11. (feature 3) refuse_exit (old copy 1655-1670): a start-up refusal ends the process without unwinding main's objects
sub("int main(int argc, char** argv) {\n", r"""// SYCL engine, #1980: a start-up refusal ends the process here instead of returning from main.  Returning ran main's
// destructors (the stages' caches, queues and Level Zero objects) in an order the UR loader did not survive: 2x Arc Pro
// B60, every REFUSED start segfaulted in libur_loader after its message, and a trimmed one also left an xe page fault
// (bcs CAT error) on the second card that hung the next start.  The device work queued so far finishes first; nothing is
// freed.
[[noreturn]] static void refuse_exit(int code) {
    std::fflush(stdout);
    std::fflush(stderr);
    try {
        for (unsigned int d = 0; d < dpct::device_count(); ++d) dpct::get_device(d).queues_wait_and_throw();
    } catch (...) {
    }
    std::fflush(stderr);
    std::_Exit(code);
}

int main(int argc, char** argv) {
""")

# 12. (feature 1) the serve path's verify plan reads the mirror table (old copy 7103-7113; no usage counts here - that is the
# old copy's --expert-profile-save under STRATA_VERIFY_NO_HOST, which is not part of this port)
sub("        vh.d_res = thits.d_res;\n        // STRATA_ROUTE_RESIDENT_MTP=1 (EXPERIMENTAL",
    "        vh.d_res = thits.d_res;\n"
    "        // SYCL engine: the pinned host mirror's entries in the device-built plan (no usage counts on this path)\n"
    "        if (mirror_table_d) strata::kernels::resident_plan_set_mirror(thits.d_res, mirror_table_d, nullptr, (long long) (g.n_layers * g.n_expert));\n"
    "        // STRATA_ROUTE_RESIDENT_MTP=1 (EXPERIMENTAL")

# 13. (feature 4, and the mirror for the --spec path) generate's verify plan (old copy 12268-12277): STRATA_USAGE_DUMP's device
# counts and the mirror table
sub("""        strata::core::VerifyHits vh;
        vh.d_res = thits.d_res;
        vh.h_res = host_res.empty() ? nullptr : host_res.data();
""", r"""        strata::core::VerifyHits vh;
        int32_t* usage_dev = nullptr;   // STRATA_USAGE_DUMP=PATH: the device plan counts every routed expert (measurement)
        const char* usage_dump = std::getenv("STRATA_USAGE_DUMP");
        if (mirror_table_d && usage_dump && *usage_dump) {
            usage_dev = sycl::malloc_device<int32_t>((size_t) (g.n_layers * g.n_expert), dpct::get_in_order_queue());
            dpct::get_in_order_queue().memset(usage_dev, 0, (size_t) (g.n_layers * g.n_expert) * 4).wait();
        }
        vh.d_res = thits.d_res;
        if (mirror_table_d) strata::kernels::resident_plan_set_mirror(thits.d_res, mirror_table_d, usage_dev, (long long) (g.n_layers * g.n_expert));
        vh.h_res = host_res.empty() ? nullptr : host_res.data();
""")

# 14. (feature 4) the usage dump itself (old copy 12724-12740), after the dispatch line and before the adaptive-tier line
sub("""        if (rounds > 0 && !drive.d.usage.empty())
            std::printf("%-24s %lld experts swapped into the VRAM tier""", r"""        if (usage_dev != nullptr) {   // STRATA_USAGE_DUMP: the window's routing as make_profile.py reads it (layer, k=1, id, weight)
            std::vector<int32_t> uh((size_t) (g.n_layers * g.n_expert));
            dpct::get_in_order_queue().memcpy(uh.data(), usage_dev, uh.size() * 4).wait();
            if (FILE* uf = std::fopen(usage_dump, "wb")) {
                int64_t total = 0;
                for (int64_t l = 0; l < g.n_layers; ++l)
                    for (int64_t e = 0; e < g.n_expert; ++e)
                        for (int32_t c = 0; c < uh[(size_t) (l * g.n_expert + e)]; ++c, ++total) {
                            const int32_t hdr[3] = {(int32_t) l, 1, (int32_t) e};
                            const float wgt[2] = {0.0f, 0.0f};
                            std::fwrite(hdr, 4, 3, uf);
                            std::fwrite(wgt, 4, 1, uf);   // (layer, k) + k ids + k weights: 8 + 8 bytes
                        }
                std::fclose(uf);
                std::printf("%-24s %s (%lld routed entries counted on the device)\n", "usage dumped", usage_dump,
                            (long long) total);
            }
        }
        if (rounds > 0 && !drive.d.usage.empty())
            std::printf("%-24s %lld experts swapped into the VRAM tier""")

# 15. (feature 5) STRATA_DBG_DRAFT (old copy 12612-12616): the window the drafter built and what the target said, in the --spec loop
sub("""            int a = 0;
            while (a < T - 1 && window[(size_t) a + 1] == outv[(size_t) a]) ++a;
            if (first_window) {
""", r"""            int a = 0;
            while (a < T - 1 && window[(size_t) a + 1] == outv[(size_t) a]) ++a;
            if (std::getenv("STRATA_DBG_DRAFT") != nullptr) {   // the window the drafter built and what the target said
                std::fprintf(stderr, "draftdbg: window"); for (int i = 0; i < T; ++i) std::fprintf(stderr, " %d", (int) window[(size_t) i]);
                std::fprintf(stderr, " | target"); for (int i = 0; i < T; ++i) std::fprintf(stderr, " %d", (int) outv[(size_t) i]);
                std::fprintf(stderr, " | accepted %d\n", a);
            }
            if (first_window) {
""")

# 16. (feature 2) split headroom (old copy 3681-3712): the scoring state, declared with the auto split's search
sub("""        std::vector<int64_t> stage_floors;   // #2644: each stage's share of the room, in the share pass below (empty: off)
""", r"""        std::vector<int64_t> stage_floors;   // #2644: each stage's share of the room, in the share pass below (empty: off)
        // #1440 (SYCL engine, old copy 3681-3712): of the placements within STRATA_SPLIT_HEADROOM_TOL (default 0.2%) of the
        // best predicted time, the one whose tightest card keeps the most VRAM beyond all its layers' experts
        // (split_rules::pick_with_headroom).  On by default (a full xe card moves buffers to system memory instead of
        // failing); STRATA_SPLIT_HEADROOM=0 keeps the first best, as before.  The candidates are the placements the search
        // priced and the startability gate allows; the pick is made after the search (see below).
        static const bool split_headroom = [] {
            const char* v = std::getenv("STRATA_SPLIT_HEADROOM");
            return v == nullptr || std::atoi(v) != 0;
        }();
        const double headroom_tol =
            std::getenv("STRATA_SPLIT_HEADROOM_TOL") ? std::atof(std::getenv("STRATA_SPLIT_HEADROOM_TOL")) : 0.002;
        std::vector<int64_t> need_prefix((size_t) g.n_layers + 1, 0);
        for (int64_t l = 0; l < g.n_layers; ++l)
            need_prefix[(size_t) l + 1] = need_prefix[(size_t) l] + (int64_t) g.n_expert * cost(l);
        // a range's session, and a later stage's weights, as stage_left prices them (stage 0 is the only range from 0)
        auto range_session = [&](int64_t lb, int64_t le) -> int64_t {
            return (int64_t) strata::core::session_bytes(g, o.max_context, K, lb, le) +
                   (lb == 0 ? 0 : (int64_t) carve_bytes(lb, le));
        };
        std::vector<strata::program::split_rules::Placement> scored;
        std::vector<double> scored_mass;
        std::vector<int64_t> scored_held;
        auto score = [&](const std::vector<int64_t>& cuts, double ms, double hm, int64_t held) {
            scored.push_back({cuts, ms, strata::program::split_rules::min_spare(cuts, g.n_layers, cap, need_prefix, range_session)});
            scored_mass.push_back(hm);
            scored_held.push_back(held);
        };
""")
sub("""            if (ns <= 3) priced.emplace_back(ms, at);
""", """            if (ns <= 3) priced.emplace_back(ms, at);
            if (split_headroom && (!gate_on || can_start(parts_of(at)))) score(at, ms, hm, held);   // #1440: a candidate for the pick
""")
sub("""        priced.clear();
""", """        priced.clear();
        scored.clear();
        scored_mass.clear();
        scored_held.clear();
""")
sub("""        split_at = best;
        std::string ks;
""", """        // #1440 (SYCL engine, old copy 3805-3826): the headroom pick among the placements the search priced
        if (split_headroom && scored.size() > 1) {
            const size_t first = strata::program::split_rules::fastest(scored);
            const size_t pick = strata::program::split_rules::pick_with_headroom(scored, headroom_tol);
            if (pick != first && scored[first].cuts == best) {
                auto fmt_k = [](const std::vector<int64_t>& v) {
                    std::string s;
                    for (const int64_t k : v) s += (s.empty() ? "" : ",") + std::to_string(k);
                    return s;
                };
                std::fprintf(stderr, "strata generate: layer split auto: K=%s instead of K=%s (predicted %.1f vs %.1f ms): its "
                                     "tightest card keeps %.2f GiB beyond all its layers' experts, against %.2f GiB "
                                     "(#1440: a full card can page to system memory); STRATA_SPLIT_HEADROOM=0 keeps K=%s\\n",
                             fmt_k(scored[pick].cuts).c_str(), fmt_k(best).c_str(), scored[pick].ms, best_ms,
                             (double) scored[pick].min_spare / 1073741824.0,
                             (double) scored[first].min_spare / 1073741824.0, fmt_k(best).c_str());
                best = scored[pick].cuts;
                best_ms = scored[pick].ms;
                best_mass = scored_mass[pick];
                best_held = scored_held[pick];
            }
        }
        split_at = best;
        std::string ks;
""")

# 17. (feature 3) the layer-split later-stage refusal (old copy 5057-5075, #1980): its own host mirror is not built here, so
# a later stage's expert that is neither in its VRAM cache nor served is refused with STRATA_VERIFY_NO_HOST
sub("""    if (multi_gpu)
        std::fprintf(stderr, "strata generate: layer split: CUDA0 runs layers 0-%lld\\n", (long long) (split_at[0] - 1));
""", r"""    if (multi_gpu)
        std::fprintf(stderr, "strata generate: layer split: CUDA0 runs layers 0-%lld\n", (long long) (split_at[0] - 1));
    // #1980 (SYCL engine): the host mirror covers the first stage only. A later stage whose cache does not hold all of its
    // layers' experts has no mirror entry for the rest, and with STRATA_VERIFY_NO_HOST no host answers for them: the device
    // plan reads a bad address and the card is lost at the first verify window (2x Arc Pro B60, xe CAT fault). Refuse at
    // the start with the same message, and say how to get a complete load.
    {
        const bool no_host_split = [] { const char* v = std::getenv("STRATA_VERIFY_NO_HOST"); return v && *v && std::strcmp(v, "0") != 0; }();
        int64_t later_misses = 0;
        for (auto& stp : stages) {
            const GpuStage& st = *stp;
            later_misses += std::max<int64_t>(0, (st.le - st.lb) * g.n_expert - (int64_t) st.cache.slots());
        }
        if (later_misses > 0 && no_host_split) {
            std::fprintf(stderr, "strata generate: REFUSED: %lld experts of the later layer-split stages are neither in VRAM nor mirrored "
                                 "(the host mirror covers the first stage only); with STRATA_VERIFY_NO_HOST the device plan cannot run them "
                                 "and the card is lost at the first verify window - set STRATA_STAGE_TRIM=1 (with --layer-split N) to free "
                                 "each card's duplicate dense weights, lower --max-context or --vram-reserve-mib, or unset "
                                 "STRATA_VERIFY_NO_HOST\n", (long long) later_misses);
            refuse_exit(2);
        }
        if (later_misses > 0)
            std::fprintf(stderr, "strata generate: layer split: %lld experts of the later stages are not in VRAM (served by the host)\n",
                         (long long) later_misses);
    }
""")

# 18. (feature 1) the shutdown releases the mirror before the exit (old copy 11816-11821)
sub("        try { dpct::get_current_device().queues_wait_and_throw(); } catch (const std::exception&) {}\n",
    "        try {\n"
    "            dpct::get_current_device().queues_wait_and_throw();\n"
    "            gguf_src.close();   // SYCL engine: release the direct-access USM host mirror before exit (the old copy did)\n"
    "        } catch (const std::exception&) {}\n")

# (review) the Q8_0 hyper-connection read is opt-in on SYCL: not switched on by itself for a Q8_0 file
sub("#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_GPU_METAL)\n"
    "        // CUDA, a file that carries",
    "#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_GPU_METAL) && \\\n"
    "    !defined(STRATA_GPU_SYCL)\n"
    "        // CUDA, a file that carries")

# (review) pin by need registers host ranges with the GPU driver, which this backend cannot do: ignored, with a message
sub("""        if (const char* pn = std::getenv("STRATA_PIN_BY_NEED"); pn != nullptr && pn[0] != '\\0' && pn[0] != '0') {
""", """        if (const char* pn = std::getenv("STRATA_PIN_BY_NEED"); pn != nullptr && pn[0] != '\\0' && pn[0] != '0')
            std::fprintf(stderr, "strata generate: STRATA_PIN_BY_NEED: no host registration on the SYCL engine - ignored\\n");
        if (const char* pn = nullptr; pn != nullptr) {
""")

# prompt tail under STRATA_VERIFY_NO_HOST: only a pair whose expert has a pinned mirror copy may give its slot away
sub("""                            if (slot_cap(pr.second) < nb) continue;
""", """                            if (slot_cap(pr.second) < nb) continue;
                            // SYCL engine, STRATA_VERIFY_NO_HOST: the GPU runs a non-resident expert only from the pinned
                            // mirror (the start's misses and the cache's last slots) and the host serves none. A pair may
                            // give its slot away only if the expert it held has a mirror copy - else that expert is in
                            // neither place (non-finite logits, issue #7).
                            static const bool tail_no_host = [] { const char* v = std::getenv("STRATA_VERIFY_NO_HOST"); return v && *v && std::strcmp(v, "0") != 0; }();
                            if (tail_no_host && mirror_table_d != nullptr && !gguf_src.pinned(pr.first / g.n_expert, pr.first % g.n_expert)) continue;
""")

# learning without the host (STRATA_VERIFY_NO_HOST): the adaptive tier moves experts out of VRAM, so every expert needs a
# mirror copy, and its usage counts must come from the device (STRATA_ROUTE_RESIDENT with STRATA_ROUTE_RESIDENT_COUNT=1)
sub("""        const uint64_t cap = mv ? (uint64_t) std::atoll(mv) << 20 : (avail > (4ull << 30) ? avail - (4ull << 30) : 0);
""", """        const uint64_t cap = mv ? (uint64_t) std::atoll(mv) << 20 : (avail > (4ull << 30) ? avail - (4ull << 30) : 0);
        const bool learn_counted = [] {   // the device counts what the router wants (route_want), read before each swap round
            const char* rr = std::getenv("STRATA_ROUTE_RESIDENT");
            const char* ct = std::getenv("STRATA_ROUTE_RESIDENT_COUNT");
            return rr != nullptr && std::atof(rr) > 0.0 && ct != nullptr && std::atoi(ct) != 0;
        }();
        {
            const char* nh = std::getenv("STRATA_VERIFY_NO_HOST");
            const char* all = std::getenv("STRATA_SYCL_MIRROR_ALL");   // =0: keep the mirror to the misses (no learning)
            if (nh != nullptr && *nh && std::strcmp(nh, "0") != 0 && learn_counted && !(all != nullptr && all[0] == '0') &&
                !multi_gpu && o.adapt_every > 0 && o.adapt_swaps > 0) {
                const auto& lay = strata::kernels::cpu::expert_layout();
                uint64_t need = 0;
                for (int64_t l = 0; l < mirror_end; ++l) need += (uint64_t) lay.blob_bytes(l) * (uint64_t) g.n_expert;
                if (need + need / 50 <= cap) {
                    std::vector<char> listed((size_t) (g.n_layers * g.n_expert), 0);
                    for (const auto& pr : miss) listed[(size_t) (pr.first * g.n_expert + pr.second)] = 1;
                    for (int64_t l = 0; l < mirror_end; ++l)
                        for (int64_t e = 0; e < g.n_expert; ++e)
                            if (!listed[(size_t) (l * g.n_expert + e)]) miss.push_back({l, e});
                } else {
                    std::fprintf(stderr, "strata generate: learning needs every expert mirrored in RAM (%.1f GiB, %.1f GiB "
                                         "allowed): only the misses are mirrored\\n", (double) need / 1073741824.0,
                                 (double) cap / 1073741824.0);
                }
            }
        }
""")
sub("""        if (!miss.empty() && no_host && o.adapt_every > 0 && o.adapt_swaps > 0) {""",
    """        bool all_mirrored = gguf_src.mirrored_bytes() > 0;
        for (int64_t l = 0; all_mirrored && l < g.n_layers; ++l)
            for (int64_t e = 0; all_mirrored && e < g.n_expert; ++e) all_mirrored = gguf_src.pinned(l, e);
        if (no_host && all_mirrored && learn_counted && o.adapt_every > 0 && o.adapt_swaps > 0)
            std::fprintf(stderr, "strata generate: learning on: every expert is mirrored, so the adaptive tier may move any "
                                 "of them; its usage counts come from the device\\n");
        if (!miss.empty() && no_host && !(all_mirrored && learn_counted) && o.adapt_every > 0 && o.adapt_swaps > 0) {""")
sub("""                std::fprintf(stderr, "strata generate: %lld of %zu experts missing from VRAM (+ %zu in the prompt loan's slots) "
                                     "mirrored in pinned host memory (%.2f GiB, %.1f s); the GPU reads them over PCIe\\n",
                             (long long) got, n_true_miss, miss.size() - n_true_miss,
                             (double) gguf_src.mirrored_bytes() / 1073741824.0,
                             std::chrono::duration<double>(Clock::now() - tm).count());
""", """                std::fprintf(stderr, "strata generate: %lld experts mirrored in pinned host memory (%.2f GiB, %.1f s), of "
                                     "%zu missing from VRAM and %zu that are in VRAM now; the GPU reads them over PCIe\\n",
                             (long long) got, (double) gguf_src.mirrored_bytes() / 1073741824.0,
                             std::chrono::duration<double>(Clock::now() - tm).count(), n_true_miss, miss.size() - n_true_miss);
""")

open(f"{W}/sycl/src/program/generate.cpp", "w").write(s)
print("generate.cpp rebuilt:", s.count("\n"), "lines")
