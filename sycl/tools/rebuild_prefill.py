"""rebuild_prefill.py: the SYCL engine's prefill.cpp = our real src/prefill/prefill.cpp + the Intel pieces of the SYCL
copy (read from the committed copy, tmp/prefill_head.cpp).  Every step asserts its anchor; nothing is guessed."""
import re

W = "/home/Projects/Strata-sycl"
s = open(f"{W}/src/prefill/prefill.cpp").read()
H = open("/root/.claude/jobs/743230e4/tmp/prefill_head.cpp").read()


def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:80], s.count(old))
    s = s.replace(old, new)


# 1. the SYCL includes, and the mmq headers by path (this file lives in sycl/src/prefill)
nl = s.index("\n") + 1
s = s[:nl] + ("#define DPCT_PROFILING_ENABLED\n#include <sycl/sycl.hpp>\n#include <dpct/dpct.hpp>\n"
              "#include \"strata/sycl_queue.hpp\"\n#include \"strata/copy_queue.hpp\"\n#include \"strata/prefill/xmx_moe.hpp\"\n"
              "#include \"strata/core/gguf_expert_source.hpp\"\n#include \"strata/kernels/iq_dequant_expert.hpp\"\n") + s[nl:]
sub('#include "mmq_resident_sort.hpp"\n', '#include "../../../src/prefill/mmq_resident_sort.hpp"\n')
sub('#include "wmma_gemm.h"\n', '#include "../../../src/prefill/wmma_gemm.h"\n')

# 2. the stream-all rule (by the VRAM's share of pairs), the XMX staging sizes
sub("inline int ring_slots(size_t T) {",
    "// SYCL engine: the share of (layer, expert) pairs the VRAM cache does not hold (set_nonresident_share, from generate.cpp)\n"
    "double g_nonres_share = 0.0;\ninline int ring_slots(size_t T) {")
sub("""    if (v && r == STAGE) return STAGE; // Explicit opt-in to routed-only staging, including large chunks.
    const int big = r < 16 ? 16 : r > ring_cap() ? ring_cap() : r;""",
"""    if (v && r == STAGE) return STAGE; // Explicit opt-in to routed-only staging, including large chunks.
    // SYCL engine: the stream-all walk copies EVERY non-resident expert of a layer, routed or not: with a big host tier (the
    // IQ2_XS: a quarter of the experts in the pinned mirror) that is several times the routed ones, and a 2,184-token prompt
    // fell from ~570 to 254 tok/s on the B70. So by default only while the VRAM holds more than 90% of the pairs.
    // STRATA_PREFILL_STREAM_ALL=0 keeps routed-only staging, =1 forces the walk.
    static const int stream_all_env = [] { const char* e = std::getenv("STRATA_PREFILL_STREAM_ALL"); return e ? (e[0] == '1' ? 1 : 0) : -1; }();
    const bool stream_all_ok = stream_all_env == 1 || (stream_all_env == -1 && g_nonres_share < 0.10);
    if (!stream_all_ok) return STAGE;
    const int big = r < 16 ? 16 : r > ring_cap() ? ring_cap() : r;""")
sub("inline int stage_slots(size_t T) { return std::max(ring_slots(T), STAGE_GRP); }",
    "// A770 port: STRATA_PF_XMX=2 holds up to a group of staged experts until its launch, so routed-only staging needs\n"
    "// that many more slots for the lookahead to keep copying (the slots are only reused once the group's event is in)\n"
    "inline int stage_slots() { return strata::prefill::xmx::mode() == 2 ? STAGE + strata::prefill::xmx::kMaxGroup : STAGE; }\n"
    "inline int stage_slots(size_t T) { return std::max(std::max(ring_slots(T), STAGE_GRP), stage_slots()); }   // (and XMX=2's group)")
i = s.index("constexpr int DQ = 2;"); j = s.index("\n", i) + 1
s = s[:j] + ("// A770 port: STRATA_PF_XMX=1 dequantizes a whole group of experts, then one grouped XMX launch per product\n"
             "inline int dq_slots() { return strata::prefill::xmx::mode() == 1 ? strata::prefill::xmx::kMaxGroup : DQ; }\n") + s[j:]


# 3. the SYCL copy's Stager (one job at a time; GGUF reads by the stager's thread; DMA completion as a sequence number in
#    page-locked memory: the Level Zero v2 adapter did not survive host waits on queue events), with our batch knobs
def struct_block(text, name):
    m = re.search(r"^struct " + name + r" \{\n", text, re.M)
    a = m.start()
    b = text.index("\n};\n", a) + 4
    return a, b


a, b = struct_block(s, "Stager")
ha, hb = struct_block(H, "Stager")
st = H[ha:hb]
st = st.replace("struct Stager {\n", "struct Stager {\n"
                "    // (ours: batched source reads; the SYCL engine reads one job at a time, so these only keep the API)\n"
                "    static constexpr int kMaxBatch = 32;\n    int batch = 1;\n", 1)
s = s[:a] + st + s[b:]

# 4. Impl: the XMX=1 dequant group, the PLE sequence numbers
sub("    uint16_t* dq_gu[DQ] = {};\n    uint16_t* dq_d[DQ] = {};",
    "    uint16_t* dq_gu[strata::prefill::xmx::kMaxGroup > DQ ? strata::prefill::xmx::kMaxGroup : DQ] = {};   // A770: STRATA_PF_XMX=1 (dq_slots)\n"
    "    uint16_t* dq_d[strata::prefill::xmx::kMaxGroup > DQ ? strata::prefill::xmx::kMaxGroup : DQ] = {};")
a, b = struct_block(s, r"Prefill::Impl")
body = s[a:b]
body = body[:-4] + ("\n    // SYCL engine: the PLE uploads' completion, written by the queue into page-locked memory (see run_impl)\n"
                    "    uint64_t* ple_done_seq = nullptr;\n    uint64_t ple_want[2] = {};\n    uint64_t ple_seq = 0;\n};\n")
s = s[:a] + body + s[b:]
sub("        m.ple_rows[b].resize(T * strata::kernels::PLE_N_HEADS);",
    "        if (!m.ple_done_seq) {   // SYCL engine: the PLE uploads' sequence numbers (page-locked)\n"
    "            m.ple_done_seq = sycl::malloc_host<uint64_t>(2, dpct::get_in_order_queue());\n"
    "            if (m.ple_done_seq) m.ple_done_seq[0] = m.ple_done_seq[1] = 0;\n            else ok = false;\n        }\n"
    "        m.ple_rows[b].resize(T * strata::kernels::PLE_N_HEADS);")

# 5. run_impl: the PLE sequence numbers, the GGUF jobs
sub("""                    cudaSuccess ||
                cudaEventRecord(m.ple_copied[ple_buf], m.cs) != cudaSuccess) {""",
"""                    cudaSuccess ||
                // SYCL engine: the upload's completion is a sequence number the queue writes into page-locked memory after
                // it (the host waiting on queue events hung the B70 under the Level Zero v2 adapter, 2026-10-02)
                [&] {
                    try { m.ple_want[ple_buf] = ++m.ple_seq; m.cs->fill<uint64_t>(m.ple_done_seq + ple_buf, m.ple_want[ple_buf], 1); return false; }
                    catch (const sycl::exception&) { return true; }
                }()) {""")
sub("""                if (cudaEventSynchronize(m.ple_copied[ple_buf ^ 1]) != cudaSuccess) {
                    err = std::string("prefill: the PLE rows' upload failed: ") + cudaGetErrorString(cudaGetLastError());
                    return false;
                }""",
"""                while (*(volatile const uint64_t*) (m.ple_done_seq + (ple_buf ^ 1)) < m.ple_want[ple_buf ^ 1])
                    std::this_thread::yield();   // SYCL engine: the sequence number above""")
sub("""        if (stream_all) {
            seq_start.assign((size_t) g.n_layers + 1, 0);
            std::vector<Stager::Job> js;""",
"""        // SYCL engine, --stream-experts: a GGUF source's experts are read by the stager's thread itself (Stager::Job::gsrc);
        // STRATA_GGUF_PLAN_READ=1 reads them here instead
        static const bool plan_reads = [] { const char* v = std::getenv("STRATA_GGUF_PLAN_READ"); return v && v[0] == '1'; }();
        const core::GgufExpertSource* gsrc = plan_reads ? nullptr : dynamic_cast<const core::GgufExpertSource*>(m.src);
        if (stream_all) {
            seq_start.assign((size_t) g.n_layers + 1, 0);
            std::vector<Stager::Job> js;""")
sub("""                    if (m.src->prompt_copies(l, e)) {   // CS-T: copied by the source into the stager's buffer
                        job = (int) js.size();
                        js.push_back({nullptr, (size_t) lay0.blob_bytes(l), m.src, (int32_t) l, e});""",
"""                    if (gsrc != nullptr) {   // read by the stager's thread when its turn comes (see Stager::Job)
                        job = (int) js.size();
                        js.push_back({nullptr, (size_t) lay0.blob_bytes(l), nullptr, (int32_t) l, e, gsrc});
                    } else if (m.src->prompt_copies(l, e)) {   // CS-T: copied by the source into the stager's buffer
                        job = (int) js.size();
                        js.push_back({nullptr, (size_t) lay0.blob_bytes(l), m.src, (int32_t) l, e});""")
sub("""                                job_of[j] = (int) js.size();
                                if (m.src->prompt_copies(l, e)) {   // CS-T: copied by the source""",
"""                                job_of[j] = (int) js.size();
                                if (gsrc != nullptr) {   // SYCL engine, --stream-experts: the stager's thread reads it from the GGUF
                                    js.push_back({nullptr, (size_t) lay.blob_bytes(l), nullptr, (int32_t) l, e, gsrc});
                                    continue;
                                }
                                if (m.src->prompt_copies(l, e)) {   // CS-T: copied by the source""")

# 6. set_nonresident_share (generate.cpp calls it)
sub("void Prefill::set_pinned_share(double share) { g_pinned_share = share; }",
    "void Prefill::set_pinned_share(double share) { g_pinned_share = share; }\n"
    "void set_nonresident_share(double share) { g_nonres_share = share; }   // SYCL engine (see ring_slots)")

# 7. A770 port, STRATA_PF_XMX=2 (opt-in): the streamed walk's experts join an open group; the group's products are one
#    fused grouped XMX launch per product, decoding the quantized weights itself (xmx::iq_xmx_grouped).  Off by default.
#    Streamed walk only, no MMQ layer, no latent layer, and only where the layer's gate/up and down formats are the ones
#    the parity test covers.  The group's ring slots are released by one event after its down product (xflush).
sub(r"""                        // one expert's products from its blob on the device; `slot` (a ring slot, or -1 for a resident
""", r"""                        // A770 port, STRATA_PF_XMX=2 (streamed walk, native pack, FP16 path): the experts' quantized blobs are
                        // held until a group of kXmxGroupMax, then each product runs over the group in one launch that decodes
                        // the weights itself.  Their ring slots are released by ONE event after the group's down product, and
                        // nothing is given back to the issuer while any are held.  (The formats the parity test covers: gate/up
                        // IQ3_XXS / IQ3_S / IQ2_S / IQ4_XS, down IQ4_NL / Q2_0; a layer in another format keeps the dequantize +
                        // oneMKL path.)  Only in the streamed walk, never on an MMQ or latent layer, never with a tail (scales).
                        auto xmx2_fmt = [&]() {
                            if (!lay.native || lay.fmt[(size_t) l].tail_off) return false;
                            const int gt = lay.fmt[(size_t) l].gu_type, dt = lay.fmt[(size_t) l].d_type;
                            return (gt == 18 || gt == 21 || gt == 22 || gt == 23) && (dt == 20 || dt == 42);
                        };
                        const bool xmx2 = strata::prefill::xmx::mode() == 2 && stream_all && !use_mmq && !latent && xmx2_fmt();
                        if (strata::prefill::xmx::mode() == 2 && !xmx2) {
                            static bool said = false;
                            if (!said) {
                                said = true;
                                std::fprintf(stderr, "strata: STRATA_PF_XMX=2 not taken at layer %lld (stream_all %d, mmq %d, "
                                                     "native %d, formats %d/%d)\n", (long long) l, (int) stream_all,
                                             (int) use_mmq, (int) lay.native, lay.native ? lay.fmt[(size_t) l].gu_type : -1,
                                             lay.native ? lay.fmt[(size_t) l].d_type : -1);
                            }
                        }
                        const uint8_t* xg_g[strata::kernels::kXmxGroupMax];
                        const uint8_t* xg_u[strata::kernels::kXmxGroupMax];
                        const uint8_t* xg_d[strata::kernels::kXmxGroupMax];
                        int32_t xg_c[strata::kernels::kXmxGroupMax];
                        int xg_slots[strata::kernels::kXmxGroupMax];
                        int xg_n = 0, xg_ns = 0;
                        int64_t xg_r0 = 0;
                        auto xflush = [&]() {
                            if (xg_n == 0) return;
                            if (static bool said = false; !said) {
                                said = true;
                                std::fprintf(stderr, "strata: prompt experts on the fused grouped XMX (STRATA_PF_XMX=2)\n");
                            }
                            const auto& f = lay.fmt[(size_t) l];
                            int64_t nr = 0;
                            for (int i = 0; i < xg_n; ++i) nr += xg_c[i];
                            // measured (xmx_group_parity, 16 experts): the fused launch wins below ~200 rows per
                            // expert, dequantize + oneMKL above (gate/up 0.67-0.76x at 320); STRATA_PF_XMX_ROWS moves it
                            static const int xg_max = [] {
                                const char* v = std::getenv("STRATA_PF_XMX_ROWS");
                                return v ? std::atoi(v) : 200;
                            }();
                            if (nr / xg_n < xg_max) {
                                pt.mark(kPfGemmGU, cs);
                                strata::kernels::iq_xmx_grouped(f.gu_type, xg_g, xg_u, xg_c, xg_n, m.Xs + xg_r0 * N,
                                                                m.GU + xg_r0 * 1280, 1280, (int) N, m.cs);
                                swiglu_interleaved(m.GU + xg_r0 * 1280, m.Hh + xg_r0 * 640, nr, m.cs);
                                pt.mark(kPfGemmD, cs);
                                strata::kernels::iq_xmx_grouped(f.d_type, xg_d, nullptr, xg_c, xg_n, m.Hh + xg_r0 * 640,
                                                                m.Dm + xg_r0 * N, (int) N, 640, m.cs);
                            } else {
                                int64_t o = xg_r0;
                                for (int i = 0; i < xg_n; ++i) {
                                    const int q = i % DQ;
                                    pt.mark(kPfDequant, cs);
                                    strata::kernels::iq_dequant_gu_f16(f.gu_type, xg_g[i], xg_u[i], f.n_ff, f.n_embd,
                                                                       m.dq_gu[q], m.cs, nullptr);
                                    strata::kernels::iq_dequant_f16(f.d_type, xg_d[i], f.n_embd * f.n_ff, m.dq_d[q], m.cs, nullptr);
                                    pt.mark(kPfGemmGU, cs);
                                    m.gemm.f16(m.Xs + o * N, m.dq_gu[q], m.GU + o * 1280, xg_c[i], 1280, N);
                                    swiglu_interleaved(m.GU + o * 1280, m.Hh + o * 640, xg_c[i], m.cs);
                                    pt.mark(kPfGemmD, cs);
                                    m.gemm.f16(m.Hh + o * 640, m.dq_d[q], m.Dm + o * N, xg_c[i], N, 640);
                                    o += xg_c[i];
                                }
                            }
                            if (xg_ns > 0) {
                                const int rel = xg_slots[xg_ns - 1];
                                cudaEventRecord(m.used[rel], m.cs);
                                for (int i = 0; i < xg_ns; ++i) m.used_of[xg_slots[i]] = rel;
                            }
                            xg_n = 0;
                            xg_ns = 0;
                        };
                        // one expert's products from its blob on the device; `slot` (a ring slot, or -1 for a resident
""")

# 8. compute(): the STRATA_PF_XMX=2 group intake, and the dequant ring's size for STRATA_PF_XMX=1 (dq_slots: DQ otherwise)
sub("""                            const int q = (int) (j % DQ);
""", """                            // A770 port, STRATA_PF_XMX=2: a streamed-walk expert joins the open group (see xflush)
                            if (xmx2) {
                                const auto& f = lay.fmt[(size_t) l];
                                if (xg_n == 0) xg_r0 = m.off[(size_t) e];
                                xg_g[xg_n] = blob_dev;
                                xg_u[xg_n] = blob_dev + f.up_off;
                                xg_d[xg_n] = blob_dev + f.down_off;
                                xg_c[xg_n] = m.cnt[(size_t) e];
                                ++xg_n;
                                if (slot >= 0) xg_slots[xg_ns++] = slot;
                                if (xg_n == strata::kernels::kXmxGroupMax || j + 1 == order.size()) xflush();
                                return true;
                            }
                            // A770 port, STRATA_PF_XMX=1 (mode 0 and latent layers: unchanged): the group's experts sit in dq slots
                            const bool xmx_on = strata::prefill::xmx::mode() == 1 && !latent;
                            const int q = (int) (j % (size_t) dq_slots());
""")

# 9. compute(): STRATA_PF_XMX=1 - the dequantized group goes through one grouped XMX GEMM per product (prefill/xmx_moe.hpp)
sub("""                            const int64_t o0 = m.off[(size_t) e], ne = m.cnt[(size_t) e];
""", """                            if (xmx_on) {
                                if (static bool said = false; !said) {
                                    said = true;
                                    std::fprintf(stderr, "strata: prompt experts on the grouped XMX GEMM (STRATA_PF_XMX=1)\\n");
                                }
                                // A770 port: the group's experts sit in dq slots 0..q, their rows back to back from the first
                                // one's (the order is by id and so are the offsets)
                                if (q + 1 < dq_slots() && j + 1 < order.size()) return true;
                                const size_t j0 = j - (size_t) q;
                                const int ng = q + 1;
                                int32_t gc[strata::prefill::xmx::kMaxGroup];
                                const uint16_t* wg[strata::prefill::xmx::kMaxGroup];
                                const uint16_t* wd[strata::prefill::xmx::kMaxGroup];
                                int64_t nr = 0;
                                for (int i = 0; i < ng; ++i) {
                                    gc[i] = m.cnt[(size_t) order[j0 + (size_t) i]];
                                    wg[i] = m.dq_gu[i]; wd[i] = m.dq_d[i];
                                    nr += gc[i];
                                }
                                const int64_t r0 = m.off[(size_t) order[j0]];
                                pt.mark(kPfGemmGU, cs);
                                if (nr / ng < strata::prefill::xmx::gu_max_mean()) {
                                    strata::prefill::xmx::grouped_f16(m.Xs + r0 * N, wg, gc, ng, m.GU + r0 * 1280, 1280,
                                                                      (int) N, m.cs);
                                } else {
                                    int64_t o = r0;
                                    for (int i = 0; i < ng; ++i) {
                                        m.gemm.f16(m.Xs + o * N, wg[i], m.GU + o * 1280, gc[i], 1280, N);
                                        o += gc[i];
                                    }
                                }
                                swiglu_interleaved(m.GU + r0 * 1280, m.Hh + r0 * 640, nr, m.cs);
                                pt.mark(kPfGemmD, cs);
                                strata::prefill::xmx::grouped_f16(m.Hh + r0 * 640, wd, gc, ng, m.Dm + r0 * N, (int) N, 640,
                                                                  m.cs);
                                return true;
                            }
                            const int64_t o0 = m.off[(size_t) e], ne = m.cnt[(size_t) e];
""")

# 10. the streamed walk: the open XMX=2 group gets its event before a slot is released, and the last group is flushed
#     at the walk's end (give_back only when XMX=2 held slots, so the default walk is the same)
sub("""                                    if (gg_nslots > 0) flush(0);   // the open group's slots get their event first
""", """                                    if (gg_nslots > 0) flush(0);   // the open group's slots get their event first
                                    if (xg_n > 0) xflush();   // A770 port: the open STRATA_PF_XMX=2 group, likewise
""")
sub("""                                    if (!group_gather || gg_nslots == 0) give_back(consumed);   // its group was gathered
""", """                                    if ((!group_gather || gg_nslots == 0) && xg_ns == 0) give_back(consumed);   // its group was gathered
""")
sub("""                            release_to(m.g->n_expert);
""", """                            release_to(m.g->n_expert);
                            xflush();   // A770 port: the last STRATA_PF_XMX=2 group
                            if (xmx2) give_back(consumed);
""")

# 11. the dequantized experts' ring: dq_slots() deep (DQ unless STRATA_PF_XMX=1), in the buffers and in the byte count
sub("""    for (int i = 0; i < DQ; ++i) { m.dq_gu[i] = o.take<uint16_t>(1280 * 2560, ok); m.dq_d[i] = o.take<uint16_t>(2560 * 640, ok); }
""", """    for (int i = 0; i < dq_slots(); ++i) { m.dq_gu[i] = o.take<uint16_t>(1280 * 2560, ok); m.dq_d[i] = o.take<uint16_t>(2560 * 640, ok); }
""")
sub("""    for (int i = 0; i < DQ; ++i) { o.take<uint16_t>(1280 * 2560, ok); o.take<uint16_t>(2560 * 640, ok); }
""", """    for (int i = 0; i < dq_slots(); ++i) { o.take<uint16_t>(1280 * 2560, ok); o.take<uint16_t>(2560 * 640, ok); }
""")

# 12. STRATA_PREFILL_FIRST (opt-in name, old default): a short first chunk, so the GPU starts while the rest of the prompt's PLE
#     rows are read.  Every chunk after it is the full C.  The chunk index math of co-prefill (c0 / C), the layer-major path
#     (ci * C) and a peer / layer split / planned chunk all assume C-sized chunks, so the short one is taken only in the plain
#     chunk-major loop (first_on, set once the layer-major decision is known, before that loop).
sub("""    const int32_t prev0[2] = {prev[0], prev[1]};
    auto ple_gather = [this, &m, &ss, tokens, n, prev0, C](int64_t c0, int buf, std::string& e) -> bool {
        const int64_t T = multi_ != nullptr ? multi_->len[(size_t) (c0 / C)] : std::min(C, n - c0);
""", """    const int32_t prev0[2] = {prev[0], prev[1]};
    // SYCL port: a short first chunk so the GPU starts while the rest of the prompt's PLE rows are still being read
    // (27k random 4 KB reads per 2k tokens, 330 ms at the drive's ~85k IOPS, otherwise all before the first kernel).
    // STRATA_PREFILL_FIRST=<tokens> (0: off), default 256 when the prompt is longer than twice that - but only when the
    // cache holds every expert: every chunk streams the experts the cache does not hold (over PCIe), so with part of the
    // model in the host mirror a short first chunk costs a whole extra pass of that stream (an Arc Pro B70 on IQ3_S: a
    // 4,095-token prompt 784 -> 1,002 tok/s without it).  STRATA_PREFILL_FIRST=<tokens> decides explicitly.
    const bool all_cached = m.cache != nullptr && m.g != nullptr && m.cache->slots() >= m.g->n_layers * m.g->n_expert;
    static const char* const first_env = std::getenv("STRATA_PREFILL_FIRST");
    const int64_t first_chunk = first_env ? std::atoll(first_env) : (all_cached ? 256 : 0);
    bool first_on = false;   // set before the chunk-major loop (see below)
    auto chunk_len = [&](int64_t c0) -> int64_t {
        if (c0 == 0 && first_on && first_chunk < C && n > 2 * first_chunk) return first_chunk;
        return std::min(C, n - c0);
    };
    auto ple_gather = [this, &m, &ss, tokens, n, prev0, C, &chunk_len](int64_t c0, int buf, std::string& e) -> bool {
        const int64_t T = multi_ != nullptr ? multi_->len[(size_t) (c0 / C)] : chunk_len(c0);
""")
sub("""        const int64_t T = mj ? multi_->len[(size_t) ci] : std::min(C, n - c0), p0 = mj ? 0 : pos0 + c0;
""", """        const int64_t T = mj ? multi_->len[(size_t) ci] : chunk_len(c0), p0 = mj ? 0 : pos0 + c0;
""")
sub("""            if (c0 + C < n) {
""", """            if (c0 + chunk_len(c0) < n) {
""")
sub("""[&ple_gather, &ple_next_err, c1 = c0 + C, b = ple_buf ^ 1]""",
    """[&ple_gather, &ple_next_err, c1 = c0 + chunk_len(c0), b = ple_buf ^ 1]""")
sub("""    const int64_t streamed0 = stats_.experts_streamed;
""", """    // the short first chunk only in the chunk-major loop: not with the layer-major path (its ci * C), co-prefill (c0 / C),
    // a layer split (either end), a peer, a forced or planned chunk, or a helper stage
    first_on = !lm && first_chunk > 0 && multi_ == nullptr && hand_in_ == nullptr && next_ == nullptr && !helped &&
               forced == 0 && planned_C == 0 && m.pp == nullptr;
    const int64_t streamed0 = stats_.experts_streamed;
""")
sub("""        for (int64_t c0 = 0; c0 < n; c0 += C) {
""", """        for (int64_t c0 = 0; c0 < n; c0 += chunk_len(c0)) {
""")

# 13. STRATA_PREFILL_TIMING: the chunk's setup laps (the host time between them, as the old SYCL copy printed them)
sub("""        const auto tsetup = Clock::now();
""", """        const auto tsetup = Clock::now();
        auto tlap = tsetup;   // STRATA_PREFILL_TIMING=1: where the host's chunk setup goes
        auto lap = [&](const char* what) {
            if (pt.on) { std::fprintf(stderr, "strata prefill timing: setup %s %.1f ms\\n", what, ms_since(tlap)); tlap = Clock::now(); }
        };
""")
sub("""            pt.mark(kPfStart, cs);
            // ---- embeddings, broadcast to the four streams""", """            pt.mark(kPfStart, cs);
            if (pt.on) { cudaStreamSynchronize(cs); lap("work queued before the chunk"); }
            // ---- embeddings, broadcast to the four streams""")
sub("""        // ---- the PLE rows of the whole chunk, one batched SSD request on a thread (see ple_gather).  A later chunk's
""", """        if (setup) lap("embeddings");
        // ---- the PLE rows of the whole chunk, one batched SSD request on a thread (see ple_gather).  A later chunk's
""")
sub("""        if (setup) {
            if (mj) {
""", """        if (setup) lap("PLE rows");
        if (setup) {
            if (mj) {
""")
sub("""            if (phase == PH_SETUP)
                std::copy_n(m.steps_host.data(),""", """            lap("step records");
            if (phase == PH_SETUP)
                std::copy_n(m.steps_host.data(),""")
sub("""        } else if (stream_all) {
            issue_until((size_t) m.ring);""", """            lap("expert stream plan");
        } else if (stream_all) {
            issue_until((size_t) m.ring);""")

# 14. (review) the debug dump's last residual row: with the short first chunk the chunks no longer start at multiples of C
sub("""        const int64_t last = (n - 1) % C;
""", """        const int64_t last = chunk_len(0) < std::min(C, n) ? (n - 1 - chunk_len(0)) % C : (n - 1) % C;
""")

open(f"{W}/sycl/src/prefill/prefill.cpp", "w").write(s)
print("prefill.cpp rebuilt:", s.count("\n"), "lines")
