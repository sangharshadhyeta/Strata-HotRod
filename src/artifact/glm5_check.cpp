// src/artifact/glm5_check.cpp - strata-glm5-check: opens a GLM-5.3-Flash (glm5-next) GGUF, headers only (no GPU, no
// weights read), and says what is in it: the family and geometry, the layer pattern, the layout check of every
// tensor against the table (core/glm5_layout.hpp), and the byte budget the streaming planner starts from.
//
//   strata-glm5-check MODEL.gguf [--head MTP.gguf]
//
// `--head` checks a standalone MTP-block file (blk.<n_layers>.* only, e.g. the Unsloth head cut out by
// tools/glm5/fetch_head.py) against the same geometry.  Exit 0 when every check passes.
#include "strata/core/glm5_layout.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <numeric>
#include <string>

namespace {
std::string gb(uint64_t b) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.2f GB", (double) b / 1e9);
    return buf;
}
std::string mb(uint64_t b) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.2f MB", (double) b / 1e6);
    return buf;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: strata-glm5-check MODEL.gguf [--head MTP.gguf]\n");
        return 2;
    }
    const char* head_path = nullptr;
    for (int i = 2; i + 1 < argc; ++i)
        if (!std::strcmp(argv[i], "--head")) head_path = argv[i + 1];
    try {
        strata::GgufModel m = strata::GgufModel::open(argv[1]);
        strata::core::Glm5Geometry g;
        std::string err;
        if (!strata::core::glm5_geometry_from_gguf(m.meta(), g, err)) {
            std::printf("geometry: REFUSED: %s\n", err.c_str());
            return 1;
        }
        std::printf("%s: %s (%s), %zu shard(s)\n", argv[1], strata::model_family_title(strata::model_family(m.meta())),
                    strata::model_family_arch(strata::model_family(m.meta())), m.size());
        std::printf("  trunk %lld layers (%lld KDA, %lld MLA) + %lld MTP block(s); hidden %lld, vocab %lld, %lld heads\n",
                    (long long) g.n_layers, (long long) g.n_kda_layers(), (long long) g.n_mla_layers(), (long long) g.n_mtp,
                    (long long) g.n_embd, (long long) g.n_vocab, (long long) g.n_head);
        std::printf("  MoE: %lld experts, %lld per token, %lld shared; %lld dense lead layers (ffn %lld), expert ffn %lld, %s router\n",
                    (long long) g.n_expert, (long long) g.n_used, (long long) g.n_shared, (long long) g.n_dense_lead,
                    (long long) g.n_ff_dense, (long long) g.n_ff_exp, g.gating_func == 2 ? "sigmoid" : "?");
        std::printf("  MLA: q_lora %lld, kv_lora %lld, head %lld (rope slice %lld); indexer %lld heads x %lld, top-k %lld, k-pool %lld\n",
                    (long long) g.q_lora, (long long) g.kv_lora, (long long) g.k_mla, (long long) g.n_rot, (long long) g.idx_heads,
                    (long long) g.idx_dim, (long long) g.idx_top_k, (long long) g.idx_kpool);
        std::string pattern;
        for (int64_t l = 0; l < g.n_layers; ++l) pattern += g.is_kda(l) ? 'K' : 'M';
        std::printf("  layer pattern (K = KDA, M = MLA): %s\n", pattern.c_str());

        strata::core::Glm5LayoutReport rep;
        const int64_t first = 0, last = m.find("blk." + std::to_string(g.n_layers) + ".attn_norm.weight") ? g.n_total_layers() : g.n_layers;
        if (!strata::core::check_glm5_layout(m, g, first, last, true, err, &rep)) {
            std::printf("layout: FAIL: %s\n", err.c_str());
            return 1;
        }
        std::printf("layout: OK, %zu tensors checked over layers 0..%lld, nothing unnamed\n", rep.checked, (long long) last - 1);
        std::printf("  total %s = routed experts %s (stream) + everything else %s (resident)\n", gb(rep.bytes_total).c_str(),
                    gb(rep.bytes_experts).c_str(), gb(rep.bytes_resident).c_str());
        for (const auto& [type, b] : rep.bytes_by_type) std::printf("    experts as %-6s %s\n", type.c_str(), gb(b).c_str());

        // per-token expert bytes: n_used experts of every MoE trunk layer, at that layer's per-expert size
        uint64_t per_token = 0, smallest = ~0ull, largest = 0;
        int64_t moe_layers = 0;
        for (int64_t l = 0; l < std::min<int64_t>(g.n_layers, (int64_t) rep.bytes_per_expert.size()); ++l) {
            if (g.is_dense(l)) continue;
            ++moe_layers;
            per_token += (uint64_t) g.n_used * rep.bytes_per_expert[(size_t) l];
            smallest = std::min(smallest, rep.bytes_per_expert[(size_t) l]);
            largest = std::max(largest, rep.bytes_per_expert[(size_t) l]);
        }
        std::printf("  one expert (gate+up+down): %s .. %s; a token reads %lld x %lld experts = %s from the routed set\n",
                    mb(smallest).c_str(), mb(largest).c_str(), (long long) moe_layers, (long long) g.n_used, gb(per_token).c_str());
        for (int64_t l = 0; l < std::min<int64_t>(g.n_layers, (int64_t) rep.expert_types.size()); ++l)
            if (!g.is_dense(l) && (l < g.n_dense_lead + 3 || l >= g.n_layers - 2))
                std::printf("    layer %2lld %s experts %s [%s]\n", (long long) l, g.is_kda(l) ? "KDA" : "MLA",
                            mb(rep.bytes_per_expert[(size_t) l]).c_str(), rep.expert_types[(size_t) l].c_str());

        if (head_path) {
            strata::GgufModel h = strata::GgufModel::open(head_path);
            strata::core::Glm5LayoutReport hr;
            if (!strata::core::check_glm5_layout(h, g, g.n_layers, g.n_layers + 1, false, err, &hr)) {
                std::printf("head %s: FAIL: %s\n", head_path, err.c_str());
                return 1;
            }
            std::printf("head %s: OK, %zu tensors, %s (experts %s)\n", head_path, hr.checked, gb(hr.bytes_total).c_str(),
                        gb(hr.bytes_experts).c_str());
        }
    } catch (const std::exception& e) {
        std::printf("error: %s\n", e.what());
        return 1;
    }
    return 0;
}
