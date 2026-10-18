// tests/core/glm5_layout_test.cpp - GLM-5.3-Flash (glm5-next): the guard, the geometry from metadata and the layout
// table, on tiny synthetic GGUFs: no model, no GPU.
//
//   * the guard accepts a complete glm5-next header and refuses a missing key, a head_count_kv list of the wrong length
//     and a hyper-connection count other than 4; the family is recognised and the engine refuses it by name;
//   * the geometry: layer kinds from head_count_kv (0 = KDA), the trunk/MTP split, the derived widths;
//   * the layout table over a tiny model: a complete file passes (and the report's byte split adds up), a wrong shape,
//     a missing tensor and a tensor the table does not name are each refused naming the tensor;
//   * a standalone MTP-block file (blk.<n_layers>.* only) passes the head check and has no hyper-connection tensors.
//
// `glm5_layout_test --real SHARD`: the same view of a real model file (headers only).
#include "gguf_fixture.hpp"

#include "strata/core/glm5_layout.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-96s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("strata-glm5-layout-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

using fixture::Kv;
using fixture::Tensor;
const uint32_t F32 = 0;

// a tiny glm5-next: 6 trunk layers (K K M K K M), 1 dense lead, 4 experts top-2, + optionally one MTP block
struct Tiny {
    uint64_t trunk = 6, mtp = 0;
    std::vector<uint64_t> kv_heads = {0, 0, 1, 0, 0, 1};
    uint64_t embd = 16, heads = 2, kda_dim = 4, conv = 4, q_lora = 4, kv_lora = 4, k_mla = 8, v_mla = 8, rot = 0, idx_heads = 2,
             idx_dim = 4, top_k = 8, kpool = 2, hc = 4, experts = 4, used = 2, shared = 1, dense_lead = 1, ff_dense = 12,
             ff_exp = 6, vocab = 10;
};

std::vector<Kv> metadata(const Tiny& t, const char* arch = "glm5-next") {
    const std::string p = std::string(arch) + ".";
    std::vector<uint64_t> kv = t.kv_heads;
    for (uint64_t i = 0; i < t.mtp; ++i) kv.push_back(1);   // llama.cpp lists every block, the MTP block last
    std::vector<Kv> out = {fixture::str("general.architecture", arch),
                           fixture::u32(p + "block_count", t.trunk + t.mtp),
                           fixture::u32(p + "embedding_length", t.embd),
                           fixture::u32(p + "feed_forward_length", t.ff_dense),
                           fixture::u32(p + "attention.head_count", t.heads),
                           fixture::u32s(p + "attention.head_count_kv", kv),
                           fixture::u32(p + "attention.q_lora_rank", t.q_lora),
                           fixture::u32(p + "attention.kv_lora_rank", t.kv_lora),
                           fixture::u32(p + "attention.key_length_mla", t.k_mla),
                           fixture::u32(p + "attention.value_length_mla", t.v_mla),
                           fixture::u32(p + "rope.dimension_count", t.rot),
                           fixture::u32(p + "kda.head_dim", t.kda_dim),
                           fixture::u32(p + "ssm.conv_kernel", t.conv),
                           fixture::u32(p + "expert_count", t.experts),
                           fixture::u32(p + "expert_used_count", t.used),
                           fixture::u32(p + "expert_shared_count", t.shared),
                           fixture::u32(p + "expert_feed_forward_length", t.ff_exp),
                           fixture::u32(p + "expert_gating_func", 2),
                           fixture::f32(p + "expert_weights_scale", 2.5f),
                           fixture::u32(p + "leading_dense_block_count", t.dense_lead),
                           fixture::u32(p + "attention.indexer.head_count", t.idx_heads),
                           fixture::u32(p + "attention.indexer.key_length", t.idx_dim),
                           fixture::u32(p + "attention.indexer.top_k", t.top_k),
                           fixture::u32(p + "attention.indexer.kpool", t.kpool),
                           fixture::u32(p + "hyper_connection.count", t.hc),
                           fixture::u32(p + "hyper_connection.sinkhorn_iterations", 20)};
    if (t.mtp) out.push_back(fixture::u32(p + "nextn_predict_layers", t.mtp));
    std::vector<std::string> tokens;
    for (uint64_t i = 0; i < t.vocab; ++i) tokens.push_back("t" + std::to_string(i));
    out.push_back(fixture::str_array("tokenizer.ggml.tokens", tokens));
    return out;
}

bool geometry_of(const fs::path& p, strata::core::Glm5Geometry& g, std::string& err) {
    const strata::GgufFile f(p.string());
    return strata::core::glm5_geometry_from_gguf(f, g, err);
}

/// The tensors the table of `g` names for layers [first, last), as F32 fixtures.
std::vector<Tensor> tensors_of(const strata::core::Glm5Geometry& g, int64_t first, int64_t last, bool globals) {
    std::vector<Tensor> ts;
    uint8_t seed = 1;
    auto add = [&](const std::string& name, const std::vector<int64_t>& shape) {
        ts.push_back({name, std::vector<uint64_t>(shape.begin(), shape.end()), F32, seed++});
    };
    if (globals)
        for (const auto& w : strata::core::glm5_global_tensors(g)) add(w.name, w.shape);
    for (int64_t l = first; l < last; ++l)
        for (const auto& w : strata::core::glm5_layer_tensors(g, l))
            if (!w.optional) add("blk." + std::to_string(l) + "." + w.name, w.shape);
    return ts;
}

int real_model(const char* shard) {
    try {
        const strata::GgufModel m = strata::GgufModel::open(shard);
        strata::core::Glm5Geometry g;
        std::string err;
        if (!strata::core::glm5_geometry_from_gguf(m.meta(), g, err)) {
            std::printf("  geometry refused: %s\n", err.c_str());
            return 1;
        }
        strata::core::Glm5LayoutReport rep;
        const bool ok = strata::core::check_glm5_layout(m, g, 0, g.n_layers, true, err, &rep);
        std::printf("  %lld layers (%lld KDA), layout: %s (%zu tensors)\n", (long long) g.n_layers, (long long) g.n_kda_layers(),
                    ok ? "ok" : err.c_str(), rep.checked);
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("  refused: %s\n", e.what());
        return 1;
    }
}
}  // namespace

int main(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "--real") return real_model(argv[2]);
    std::printf("glm5_layout_test\n");
    TempDir d;
    const Tiny tiny;
    strata::core::Glm5Geometry g;
    std::string err;

    {
        // the family, the guard
        const fs::path ok = d.path / "ok.gguf", missing = d.path / "missing.gguf", shortkv = d.path / "shortkv.gguf",
                       hc3 = d.path / "hc3.gguf";
        fixture::write(ok, metadata(tiny), {});
        std::vector<Kv> kv = metadata(tiny);
        for (size_t i = 0; i < kv.size(); ++i)
            if (kv[i].key == "glm5-next.kda.head_dim") kv.erase(kv.begin() + (long) i);
        fixture::write(missing, kv, {});
        Tiny t2 = tiny;
        t2.kv_heads = {0, 0, 1};   // three entries for six layers
        fixture::write(shortkv, metadata(t2), {});
        Tiny t3 = tiny;
        t3.hc = 3;
        fixture::write(hc3, metadata(t3), {});
        const strata::GgufFile a(ok.string()), b(missing.string()), c(shortkv.string()), e(hc3.string());
        check(strata::model_family(a) == strata::ModelFamily::Glm5Next, "glm5-next is recognised as the GLM-5.3-Flash family");
        check(strata::check_architecture(a).empty(), "the guard accepts a complete glm5-next header");
        check(strata::check_architecture(b).find("glm5-next.kda.head_dim") != std::string::npos, "a missing key is named");
        check(strata::check_architecture(c).find("head_count_kv") != std::string::npos, "a head_count_kv list of the wrong length is refused");
        check(strata::check_architecture(e).find("hyper_connection.count") != std::string::npos, "hyper_connection.count other than 4 is refused");
        const std::string why = strata::check_engine_architecture(a);
        check(why.find("GLM-5.3-Flash") != std::string::npos && why.find("not written yet") != std::string::npos,
              "the engine's guard refuses the family by name, saying the engine path is not written");
    }
    {
        // the geometry
        const fs::path p = d.path / "ok.gguf";
        check(geometry_of(p, g, err), "geometry reads from the metadata: " + err);
        check(g.n_layers == 6 && g.n_mtp == 0 && g.n_embd == 16 && g.n_vocab == 10, "layers, hidden size and vocabulary");
        check(g.is_kda(0) && g.is_kda(1) && !g.is_kda(2) && g.is_kda(3) && g.is_kda(4) && !g.is_kda(5), "layer kinds come from head_count_kv (0 = KDA)");
        check(g.n_kda_layers() == 4 && g.n_mla_layers() == 2, "4 KDA and 2 MLA layers");
        check(g.d_inner() == 8 && g.hc_mix() == 24, "d_inner = heads x kda head dim, hc mix = (2 + hc) x hc");
        check(g.is_dense(0) && !g.is_dense(1), "the leading dense layers");
        check(g.gating_func == 2 && g.weights_scale == 2.5, "sigmoid gating and the weight scale");
    }
    {
        // the layout table, trunk only (no MTP block)
        const fs::path p = d.path / "model.gguf";
        fixture::write(p, metadata(tiny), tensors_of(g, 0, g.n_layers, true));
        {
            const strata::GgufModel m = strata::GgufModel::open(p.string());
            strata::core::Glm5LayoutReport rep;
            check(strata::core::check_glm5_layout(m, g, 0, g.n_layers, true, err, &rep), "a complete tiny file passes: " + err);
            check(rep.checked == m.shard(0).tensors().size(), "every tensor of the file was checked");
            check(rep.bytes_total == rep.bytes_experts + rep.bytes_resident && rep.bytes_experts > 0, "the byte report adds up");
            check(rep.expert_bytes_per_layer.size() == 6 && rep.expert_bytes_per_layer[0] == 0 && rep.expert_bytes_per_layer[1] > 0,
                  "the dense layer has no routed experts, the MoE layers do");
        }
        auto build = [&](const char* name, void (*mutate)(std::vector<Tensor>&)) {
            std::vector<Tensor> ts = tensors_of(g, 0, g.n_layers, true);
            mutate(ts);
            const fs::path q = d.path / name;
            fixture::write(q, metadata(tiny), ts);
            const strata::GgufModel m = strata::GgufModel::open(q.string());
            std::string e;
            strata::core::check_glm5_layout(m, g, 0, g.n_layers, true, e);
            return e;
        };
        std::string e = build("wrong_shape.gguf", [](std::vector<Tensor>& ts) {
            for (Tensor& t : ts)
                if (t.name == "blk.2.attn_q_b.weight") t.shape[1] += 1;
        });
        check(e.find("blk.2.attn_q_b.weight") != std::string::npos && e.find("wants") != std::string::npos, "a wrong shape is refused, naming the tensor");
        e = build("missing.gguf", [](std::vector<Tensor>& ts) {
            for (size_t i = 0; i < ts.size(); ++i)
                if (ts[i].name == "blk.4.ssm_a") {
                    ts.erase(ts.begin() + (long) i);
                    break;
                }
        });
        check(e == "missing blk.4.ssm_a", "a missing tensor is refused, naming it");
        e = build("extra.gguf", [](std::vector<Tensor>& ts) { ts.push_back({"blk.1.surprise.weight", {4}, F32, 9}); });
        check(e.find("blk.1.surprise.weight") != std::string::npos && e.find("not in the GLM-5.3-Flash layout") != std::string::npos,
              "a tensor the table does not name is refused");
        e = build("conv4d.gguf", [](std::vector<Tensor>& ts) {
            for (Tensor& t : ts)
                if (t.name == "blk.0.ssm_conv1d_q.weight") t.shape = {4, 1, 8, 1};
        });
        check(e.empty(), "a 4-D conv1d tensor ([k, 1, d, 1]) is accepted as the 3-D table entry: " + e);
    }
    {
        // a model with an in-file MTP block, and a standalone head file
        Tiny t = tiny;
        t.mtp = 1;
        strata::core::Glm5Geometry gm;
        const fs::path meta = d.path / "mtp_meta.gguf", whole = d.path / "mtp_whole.gguf", head = d.path / "head.gguf";
        fixture::write(meta, metadata(t), {});
        check(geometry_of(meta, gm, err) && gm.n_layers == 6 && gm.n_mtp == 1 && gm.is_mtp(6) && !gm.is_kda(6) && gm.has_indexer(6),
              "nextn_predict_layers splits the trunk from the MTP block; the block is MLA with an indexer: " + err);
        fixture::write(whole, metadata(t), tensors_of(gm, 0, gm.n_total_layers(), true));
        const strata::GgufModel m = strata::GgufModel::open(whole.string());
        check(strata::core::check_glm5_layout(m, gm, 0, gm.n_total_layers(), true, err), "a file with the MTP block passes: " + err);
        bool hc_in_mtp = false;
        for (const auto& w : strata::core::glm5_layer_tensors(gm, 6)) hc_in_mtp |= w.name.rfind("hc_", 0) == 0;
        check(!hc_in_mtp, "the MTP block has no hyper-connection tensors");
        fixture::write(head, metadata(t), tensors_of(gm, 6, 7, false));
        const strata::GgufModel h = strata::GgufModel::open(head.string());
        strata::core::Glm5LayoutReport rep;
        check(strata::core::check_glm5_layout(h, gm, 6, 7, false, err, &rep) && rep.checked > 20,
              "a standalone head file (blk.6.* only) passes the head check: " + err);
    }
    std::printf(g_fail ? "FAILED: %d\n" : "all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
