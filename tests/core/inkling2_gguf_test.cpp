// tests/core/inkling2_gguf_test.cpp - the `inkling2` reader (include/strata/artifact/inkling2.hpp), M1 of docs/INKLING-PORT.md.
//
// Part 1, synthetic: the configuration parsed from metadata a test builds (Inkling-2-Mini's real values), and each way it must be
// refused.  No file, no GPU.  Part 2, the real file: if STRATA_INKLING2_GGUF names an `inkling2` GGUF (unsloth's Inkling-2-Mini
// UD-Q4_K_XL), every tensor is resolved with its exact shape and type and the counts are checked; without it that part is skipped.
// Exit 0 pass, 1 fail.
#include "strata/artifact/inkling2.hpp"
#include "strata/artifact/model_family.hpp"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace {

int g_fail = 0, g_ok = 0;
void check(bool c, const std::string& what) {
    if (c) ++g_ok; else { ++g_fail; std::printf("FAIL: %s\n", what.c_str()); }
}

using strata::MetaType;
using strata::MetaValue;
using Meta = std::map<std::string, MetaValue>;

MetaValue U(uint64_t v) { MetaValue m; m.type = MetaType::U32; m.u = v; return m; }
MetaValue F(double v) { MetaValue m; m.type = MetaType::F32; m.f = v; return m; }
MetaValue S(const std::string& v) { MetaValue m; m.type = MetaType::STRING; m.s = v; return m; }
MetaValue A(const std::vector<uint64_t>& v) {
    MetaValue m; m.type = MetaType::ARRAY; m.elem = MetaType::U32; m.count = v.size();
    for (uint64_t x : v) m.items.push_back(U(x));
    return m;
}

/// Inkling-2-Mini's metadata, as read from the real file on 22 Nov 2026.
Meta mini() {
    Meta m;
    const std::string p = "inkling2.";
    m["general.architecture"] = S("inkling2");
    m[p + "block_count"] = U(48);
    m[p + "context_length"] = U(1048576);
    m[p + "embedding_length"] = U(2304);
    m[p + "feed_forward_length"] = U(1152);
    m[p + "expert_feed_forward_length"] = U(1152);
    m[p + "vocab_size"] = U(201024);
    m[p + "vocab_size_unpadded"] = U(200058);
    m[p + "attention.head_count"] = A(std::vector<uint64_t>(48, 24));
    m[p + "attention.head_count_kv"] = A(std::vector<uint64_t>(48, 8));
    m[p + "attention.layer_norm_rms_epsilon"] = F(9.99999997e-07);
    m[p + "expert_count"] = U(128);
    m[p + "expert_used_count"] = U(8);
    m[p + "attention.key_length"] = U(128);
    m[p + "attention.value_length"] = U(128);
    m[p + "num_loops"] = U(3);
    m[p + "loop.core_start"] = U(6);
    m[p + "loop.core_end"] = U(18);
    m[p + "loop.gate_rank"] = U(64);
    m[p + "attention.key_length_swa"] = U(128);
    m[p + "attention.value_length_swa"] = U(128);
    m[p + "rope.dimension_count"] = U(0);
    m[p + "attention.sliding_window"] = U(1024);
    std::vector<uint64_t> pat;
    for (int v = 0; v < 48; ++v) pat.push_back(v % 6 == 5 ? 0 : 1);
    m[p + "attention.sliding_window_pattern"] = A(pat);
    m[p + "shortconv.l_cache"] = U(4);
    m[p + "expert_weights_scale"] = F(8.0);
    m[p + "activation.situ_beta"] = F(4.0);
    m[p + "activation.situ_linear_beta"] = F(25.0);
    m[p + "attn_res.block_size"] = U(6);
    m[p + "final_logit_softcapping"] = F(30.0);
    m[p + "attention.relative_length"] = U(16);
    m[p + "attention.relative_extent"] = U(1024);
    m[p + "attention.qk_clip"] = F(22.627416610717773);
    m[p + "attention.relative_clip"] = F(8.0);
    m[p + "attention.log_scale_floor"] = U(8192);
    m[p + "attention.log_scale_alpha"] = F(0.10000000149011612);
    return m;
}

bool refused(Meta m, const std::string& needle, const std::string& what) {
    strata::inkling2::Config c;
    const std::string e = strata::inkling2::parse_config(m, c);
    const bool ok = !e.empty() && e.find(needle) != std::string::npos;
    check(ok, what + " (got: '" + e + "')");
    return ok;
}

void synthetic() {
    strata::inkling2::Config c;
    const std::string e = strata::inkling2::parse_config(mini(), c);
    check(e.empty(), "Inkling-2-Mini's metadata parses (" + e + ")");
    check(c.n_visits == 48 && c.n_stored == 24, "48 visits over 24 stored layers");
    check(c.n_embd == 2304 && c.n_ff_exp == 1152 && c.n_expert == 128 && c.n_expert_used == 8, "widths and experts");
    check(c.n_head == 24 && c.n_head_kv == 8 && c.head_dim == 128 && c.window == 1024, "attention geometry");
    check(c.rel_states == 16 && c.rel_extent == 1024 && c.conv_taps == 4 && c.attn_res_block == 6, "relative states, conv taps, residual block");
    check(c.num_loops == 3 && c.core_start == 6 && c.core_end == 18 && c.core_len() == 12 && c.gate_rank == 64, "the loop");
    check(c.expert_weight_scale == 8.0f && c.situ_beta == 4.0f && c.situ_linear_beta == 25.0f && c.final_softcap == 30.0f, "scales and softcaps");
    // the visit -> stored layer map: before the core once, the core three times, then the rest
    const int64_t want[48] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,   // visits 0-17: the prelude and pass 1
                              6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,                      // pass 2
                              6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,                      // pass 3
                              18, 19, 20, 21, 22, 23};                                          // the rest
    bool map_ok = true;
    for (int64_t v = 0; v < 48; ++v) map_ok = map_ok && c.stored_of_visit(v) == want[v];
    check(map_ok, "stored_of_visit: 0..17, the core twice more, then 18..23");
    check(c.full_layer(5) && c.full_layer(11) && c.full_layer(17) && c.full_layer(23) && !c.full_layer(0) && !c.full_layer(6), "full layers are 5, 11, 17, 23");
    check(c.visit_sliding.size() == 48 && c.visit_sliding[5] == 0 && c.visit_sliding[4] == 1 && c.visit_sliding[47] == 0, "per-visit sliding pattern kept");

    { Meta m = mini(); m["general.architecture"] = S("llama"); refused(m, "'llama'", "another architecture is refused, named"); }
    { Meta m = mini(); m.erase("inkling2.shortconv.l_cache"); refused(m, "shortconv.l_cache", "a missing key is named"); }
    { Meta m = mini(); m["inkling2.rope.dimension_count"] = U(64); refused(m, "no RoPE", "a model with RoPE is refused"); }
    { Meta m = mini(); m["inkling2.attention.key_length_swa"] = U(64); refused(m, "key / value lengths", "swa lengths that differ are refused"); }
    { Meta m = mini(); m["inkling2.expert_feed_forward_length"] = U(2304); refused(m, "expert_feed_forward_length", "an expert width unlike the dense one is refused"); }
    { Meta m = mini(); m["inkling2.loop.core_end"] = U(17); refused(m, "whole attention-residual blocks", "a core that is not whole blocks is refused"); }
    { Meta m = mini(); m["inkling2.loop.core_end"] = U(6); refused(m, "empty or reversed", "an empty core is refused"); }
    { Meta m = mini(); m["inkling2.block_count"] = U(54); refused(m, "per-visit array", "arrays that do not match block_count are refused"); }
    { Meta m = mini(); m["inkling2.block_count"] = U(40); refused(m, "too small", "a block_count too small for the loop is refused"); }
    { Meta m = mini(); m["inkling2.attention.head_count"] = A(std::vector<uint64_t>(47, 24)); refused(m, "per-visit array", "a short head_count array is refused"); }
    {   // a sliding pattern that disagrees with 'index % 6 == 5 is full'
        Meta m = mini();
        std::vector<uint64_t> pat;
        for (int v = 0; v < 48; ++v) pat.push_back(v % 6 == 5 ? 0 : 1);
        pat[11] = 1;
        m["inkling2.attention.sliding_window_pattern"] = A(pat);
        refused(m, "sliding_window_pattern disagrees", "a pattern that moves a full layer is refused");
    }
    { Meta m = mini(); m["inkling2.attention.head_count_kv"] = A(std::vector<uint64_t>(48, 7)); refused(m, "GQA", "heads not a GQA multiple are refused"); }
    // the family layer
    check(strata::model_family_from_arch("inkling2") == strata::ModelFamily::Inkling2, "the family is recognised");
    check(strata::model_family_runs(strata::ModelFamily::Inkling2) == false, "and not yet run");
    check(strata::model_family_not_supported(strata::ModelFamily::Inkling2).find("INKLING-PORT") != std::string::npos, "its refusal points at the plan");
    check(strata::model_family_from_arch("qwen4exp") == strata::ModelFamily::Qwen4Exp && strata::model_family_runs(strata::ModelFamily::Qwen4Exp), "Qwen4Exp is unchanged");
}

int real_file(const char* path) {
    std::printf("real file: %s\n", path);
    const strata::GgufFile g(path);
    strata::inkling2::Config c;
    std::string e = strata::inkling2::parse_config(g.metadata(), c);
    check(e.empty(), "the real metadata parses (" + e + ")");
    if (!e.empty()) return 1;
    strata::inkling2::Layout lay;
    e = strata::inkling2::load_layout(g, c, lay);
    check(e.empty(), "every tensor resolves with its shape and type (" + e + ")");
    if (!e.empty()) return 1;
    check(lay.layers.size() == 24, "24 stored layers");
    int nv = 0, nq = 0, nfull = 0, ncore = 0, nscale = 0;
    for (size_t l = 0; l < lay.layers.size(); ++l) {
        const auto& t = lay.layers[l];
        nv += t.expert_nvfp4;
        nq += !t.expert_nvfp4;
        nfull += t.full;
        ncore += t.core;
        nscale += (t.gate_up_scale != nullptr) + (t.down_scale != nullptr);
        check(t.expert_nvfp4 == (l < 20), "layer " + std::to_string(l) + ": NVFP4 experts exactly in layers 0-19");
        check((t.gate_up_scale != nullptr) == t.expert_nvfp4 && (t.down_scale != nullptr) == t.expert_nvfp4, "layer " + std::to_string(l) + ": a scale beside NVFP4 experts only");
        check((t.attn_res_query == nullptr) == (l == 0), "layer " + std::to_string(l) + ": a residual query except at layer 0");
        check((t.attn_k_conv_proj != nullptr) == t.full && (t.attn_q_inp_norm != nullptr) == t.full, "layer " + std::to_string(l) + ": dynamic conv tensors on full layers only");
        check((t.attn_norm != nullptr) == !t.full, "layer " + std::to_string(l) + ": attn_norm on sliding layers only");
        check((t.loop_down[0] != nullptr) == t.core, "layer " + std::to_string(l) + ": loop control tensors on core layers only");
    }
    check(nv == 20 && nq == 4 && nfull == 4 && ncore == 12 && nscale == 40, "20 NVFP4 + 4 Q8_0 expert layers, 4 full, 12 in the core, 40 scales");
    check(g.tensors().size() == 734, "734 tensors in the file");
    std::printf("  experts %.2f GB, the rest %.2f GB (file %.2f GB)\n", lay.expert_bytes / 1e9, lay.dense_bytes / 1e9, g.file_size() / 1e9);
    check(lay.expert_bytes > 15e9 && lay.expert_bytes < 16e9, "the routed experts are about 15.8 GB (11.5 NVFP4 + 4.3 Q8_0)");
    // the refusal path: the same file with one tensor taken away or a type changed is not reachable here, so check a wrong config instead
    strata::inkling2::Config bad = c;
    bad.n_expert = 64;
    strata::inkling2::Layout l2;
    check(!strata::inkling2::load_layout(g, bad, l2).empty(), "a config that disagrees with the file's shapes is refused");
    return 0;
}

}  // namespace

int main() {
    synthetic();
    const char* path = std::getenv("STRATA_INKLING2_GGUF");
    if (path != nullptr && path[0] != 0) real_file(path);
    else std::printf("(the real-file part is skipped: STRATA_INKLING2_GGUF is not set)\n");
    std::printf("%d checks passed, %d failed\n", g_ok, g_fail);
    return g_fail == 0 ? 0 : 1;
}
