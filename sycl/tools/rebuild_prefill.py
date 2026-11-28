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

open(f"{W}/sycl/src/prefill/prefill.cpp", "w").write(s)
print("prefill.cpp rebuilt:", s.count("\n"), "lines")
