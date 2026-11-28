"""rebuild_generate.py: the SYCL engine's generate.cpp = our real src/program/generate.cpp + the Intel pieces of the SYCL
copy.  Every step asserts its anchor."""
W = "/home/Projects/Strata-sycl"
s = open(f"{W}/src/program/generate.cpp").read()


def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:90], s.count(old))
    s = s.replace(old, new)


# 1. the SYCL helpers, the prompt path's share hook
nl = s.index("\n") + 1
s = s[:nl] + ('#include "strata/sycl_queue.hpp"   // intel_gpu_driver, intel_gpu_gen (the SYCL engine)\n'
              "namespace strata::prefill { void set_nonresident_share(double share); }   // the SYCL engine's prefill.cpp\n") + s[nl:]

# 2. --stream-experts (the flag setup gives Arc on the xe driver): our GGUF-mapped expert source (--mmap-experts reads the
#    model's shards in place, no host arena), which is what it asked for; the SYCL copy's pinned host mirror is not here
sub("    bool mmap_experts = false;",
    "    bool stream_experts = false;  ///< SYCL engine: --stream-experts (setup's flag on Arc): the experts read from the GGUF\n"
    "    bool mmap_experts = false;")
sub('        else if (a == "--mmap-experts") o.mmap_experts = true;',
    '        else if (a == "--mmap-experts") o.mmap_experts = true;\n'
    '        else if (a == "--stream-experts") { o.stream_experts = true; o.mmap_experts = true; }   // SYCL engine: see Options')

# 3. Arc on the xe driver: the GPU page-faults reading ordinary (pageable) host memory, so the PCIe share defaults to 0
sub("    auto pcie_balance = [&]() {",
"""    // SYCL engine, Arc on the xe driver (Arc Pro B60/B65/B70, B580): the GPU page-faults when a kernel reads ordinary
    // (pageable) host memory, and the card times the job out (a B70: Timedout job, Fault response -EINVAL).  The experts
    // outside the VRAM cache are in pageable memory here (no pinned mirror in this build), so the PCIe share defaults to 0
    // (the CPU computes those misses) and an explicit --pcie-frac is kept with a warning.
    if (strata::intel_gpu_driver() == "xe") {
        if (!(o.pcie_frac >= 0.0)) {
            o.pcie_frac = 0.0;
            std::fprintf(stderr, "strata generate: xe driver: the GPU faults on pageable host memory, so --pcie-frac defaults "
                                 "to 0 here (the CPU computes the experts outside VRAM)\\n");
        } else if (o.pcie_frac > 0.0) {
            std::fprintf(stderr, "strata generate: WARNING: --pcie-frac %.2f on the xe driver makes the GPU read pageable host "
                                 "memory; that page-faulted and hung an Arc Pro B70 (Timedout job, device lost)\\n", o.pcie_frac);
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

open(f"{W}/sycl/src/program/generate.cpp", "w").write(s)
print("generate.cpp rebuilt:", s.count("\n"), "lines")
