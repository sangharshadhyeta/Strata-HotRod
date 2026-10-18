// include/strata/core/glm5_layout.hpp - GLM-5.3-Flash (`glm5-next`): the geometry read from a GGUF's metadata, and the
// tensor layout table checked against a model file's directory.  Header-only and GPU-free (the engine path is not
// written yet; this is the first step of the port: Strata can open the file and say what is in it).
//
// The model (llama.cpp PR 27773, src/models/glm5-next.cpp is the reference for every shape here):
//   * 45 trunk layers (+ an optional MTP/nextn block after them); layer kind from `attention.head_count_kv[l]`:
//     0 = KDA linear attention (kimi-delta: q/k/v + conv1d, low-rank f/g gates, beta, per-head decay `ssm_a`),
//     1 = nope-MLA with a k-pool DSA indexer (q_a/q_b, kv_a_mqa, k_b/v_b, no rotary slice: rope.dimension_count 0);
//   * the first `leading_dense_block_count` layers have a dense FFN, the rest a sigmoid-routed MoE with a shared expert;
//   * every trunk layer carries manifold hyper-connection (mHC) mixers around its attention and FFN (hc = 4 streams);
//   * the MTP block (llama.cpp loads it but throws "NextN graph not implemented yet") is an MLA + MoE layer with
//     nextn.eh_proj / enorm / hnorm and no hyper-connection tensors.
// Names are GGUF names (blk.N.<suffix>); shapes are in GGUF order (dim 0 contiguous).  Trailing dimensions of 1 are
// ignored when shapes are compared (the conv1d tensors are written 4-D or 3-D by different converters).
#pragma once

#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/model_family.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace strata::core {

struct Glm5Geometry {
    int64_t n_embd = 0;            ///< embedding_length (4096)
    int64_t n_layers = 0;          ///< trunk layers: block_count - nextn_predict_layers (45)
    int64_t n_mtp = 0;             ///< nextn_predict_layers (0 for the GSQ-RCO file; 1 in Unsloth's)
    int64_t n_vocab = 0;           ///< length of tokenizer.ggml.tokens (154880)
    int64_t n_head = 0;            ///< attention.head_count: KDA heads and MLA heads alike (64)
    int64_t kda_head_dim = 0;      ///< kda.head_dim (128): d_inner = n_head * kda_head_dim
    int64_t d_conv = 0;            ///< ssm.conv_kernel (4)
    int64_t q_lora = 0;            ///< attention.q_lora_rank (1536)
    int64_t kv_lora = 0;           ///< attention.kv_lora_rank (512)
    int64_t k_mla = 0;             ///< attention.key_length_mla (256); the nope slice is k_mla - n_rot
    int64_t v_mla = 0;             ///< attention.value_length_mla (256)
    int64_t n_rot = 0;             ///< rope.dimension_count (0: nope)
    int64_t idx_heads = 0;         ///< attention.indexer.head_count (32)
    int64_t idx_dim = 0;           ///< attention.indexer.key_length (128)
    int64_t idx_top_k = 0;         ///< attention.indexer.top_k (2048)
    int64_t idx_kpool = 0;         ///< attention.indexer.kpool (4): keys pooled per indexer entry
    bool idx_select_tail = false;  ///< attention.indexer.kpool_select_tail
    bool idx_share_mtp = false;    ///< attention.indexer.index_share_mtp
    int64_t hc = 0;                ///< hyper_connection.count (4)
    int64_t hc_iters = 0;          ///< hyper_connection.sinkhorn_iterations (20)
    double hc_eps = 0;             ///< hyper_connection.epsilon
    int64_t n_expert = 0;          ///< expert_count (288)
    int64_t n_used = 0;            ///< expert_used_count (8)
    int64_t n_shared = 0;          ///< expert_shared_count (1)
    int64_t n_dense_lead = 0;      ///< leading_dense_block_count (3)
    int64_t n_ff_dense = 0;        ///< feed_forward_length (12288): the dense layers' FFN
    int64_t n_ff_exp = 0;          ///< expert_feed_forward_length (2048): one expert, and the shared expert's unit
    int64_t gating_func = 0;       ///< expert_gating_func (2 = sigmoid)
    double weights_scale = 1;      ///< expert_weights_scale (2.5)
    bool weights_norm = false;     ///< expert_weights_norm
    double rms_eps = 0;            ///< attention.layer_norm_rms_epsilon
    std::vector<uint8_t> kv_heads; ///< attention.head_count_kv per trunk layer
    std::vector<uint8_t> idx_full; ///< attention.indexer.types per trunk layer (1 = the layer owns an indexer)

    int64_t d_inner() const { return n_head * kda_head_dim; }
    int64_t hc_mix() const { return (2 + hc) * hc; }
    int64_t n_total_layers() const { return n_layers + n_mtp; }
    bool is_kda(int64_t l) const { return l < n_layers && kv_heads[(size_t) l] == 0; }
    bool is_mtp(int64_t l) const { return l >= n_layers; }
    bool is_dense(int64_t l) const { return l < n_dense_lead; }
    /// The MTP block always owns an indexer; a trunk MLA layer owns one when attention.indexer.types says so.
    bool has_indexer(int64_t l) const { return !is_kda(l) && (is_mtp(l) || idx_full[(size_t) l]); }
    int64_t n_kda_layers() const {
        int64_t n = 0;
        for (int64_t l = 0; l < n_layers; ++l) n += is_kda(l);
        return n;
    }
    int64_t n_mla_layers() const { return n_layers - n_kda_layers(); }
};

/// Fills `g` from the metadata shard `meta`.  False with `err` when the guard refuses the file or a value is impossible.
inline bool glm5_geometry_from_gguf(const strata::GgufFile& meta, Glm5Geometry& g, std::string& err) {
    err = strata::check_architecture(meta);
    if (!err.empty()) return false;
    if (strata::model_family(meta) != strata::ModelFamily::Glm5Next) {
        err = meta.path() + ": not a glm5-next model";
        return false;
    }
    g = Glm5Geometry{};
    const std::string p = "glm5-next.";
    auto num = [&](const char* key, int64_t& dst) {
        if (const strata::MetaValue* v = meta.get(p + key))
            if (v->is_num()) dst = (int64_t) v->u;
    };
    auto real = [&](const char* key, double& dst) {
        if (const strata::MetaValue* v = meta.get(p + key))
            if (v->is_num()) dst = v->num();
    };
    int64_t block_count = 0, select_tail = 0, share_mtp = 0, norm = 0;
    num("block_count", block_count);
    num("nextn_predict_layers", g.n_mtp);
    num("embedding_length", g.n_embd);
    num("attention.head_count", g.n_head);
    num("kda.head_dim", g.kda_head_dim);
    num("ssm.conv_kernel", g.d_conv);
    num("attention.q_lora_rank", g.q_lora);
    num("attention.kv_lora_rank", g.kv_lora);
    num("attention.key_length_mla", g.k_mla);
    num("attention.value_length_mla", g.v_mla);
    num("rope.dimension_count", g.n_rot);
    num("attention.indexer.head_count", g.idx_heads);
    num("attention.indexer.key_length", g.idx_dim);
    num("attention.indexer.top_k", g.idx_top_k);
    num("attention.indexer.kpool", g.idx_kpool);
    num("attention.indexer.kpool_select_tail", select_tail);
    num("attention.indexer.index_share_mtp", share_mtp);
    num("hyper_connection.count", g.hc);
    num("hyper_connection.sinkhorn_iterations", g.hc_iters);
    real("hyper_connection.epsilon", g.hc_eps);
    num("expert_count", g.n_expert);
    num("expert_used_count", g.n_used);
    num("expert_shared_count", g.n_shared);
    num("leading_dense_block_count", g.n_dense_lead);
    num("feed_forward_length", g.n_ff_dense);
    num("expert_feed_forward_length", g.n_ff_exp);
    num("expert_gating_func", g.gating_func);
    real("expert_weights_scale", g.weights_scale);
    num("expert_weights_norm", norm);
    real("attention.layer_norm_rms_epsilon", g.rms_eps);
    g.idx_select_tail = select_tail != 0;
    g.idx_share_mtp = share_mtp != 0;
    g.weights_norm = norm != 0;
    g.n_layers = block_count - g.n_mtp;
    if (const strata::MetaValue* v = meta.get("tokenizer.ggml.tokens"))
        if (v->type == strata::MetaType::ARRAY && v->count > 0) g.n_vocab = (int64_t) v->count;

    // the per-layer lists; head_count_kv names every block of block_count (trunk + MTP), the trunk is the first n_layers
    const strata::MetaValue* kv = meta.get(p + "attention.head_count_kv");
    g.kv_heads.assign((size_t) std::max<int64_t>(g.n_layers, 0), 0);
    for (int64_t l = 0; l < g.n_layers && kv && (size_t) l < kv->items.size(); ++l) g.kv_heads[(size_t) l] = (uint8_t) kv->items[(size_t) l].u;
    g.idx_full.assign((size_t) std::max<int64_t>(g.n_layers, 0), 1);   // llama.cpp: every layer owns an indexer unless the list says no
    if (const strata::MetaValue* v = meta.get(p + "attention.indexer.types"))
        if (v->type == strata::MetaType::ARRAY)
            for (int64_t l = 0; l < g.n_layers && (size_t) l < v->items.size(); ++l) g.idx_full[(size_t) l] = v->items[(size_t) l].u != 0;

    char buf[256];
    auto bad = [&](const char* what) {
        std::snprintf(buf, sizeof buf, "%s: %s", meta.path().c_str(), what);
        err = buf;
        return false;
    };
    if (g.n_layers <= 0 || g.n_mtp < 0 || g.n_embd <= 0 || g.n_head <= 0 || g.kda_head_dim <= 0 || g.d_conv <= 0 ||
        g.q_lora <= 0 || g.kv_lora <= 0 || g.k_mla <= 0 || g.v_mla <= 0 || g.n_rot < 0 || g.n_rot >= g.k_mla ||
        g.idx_heads <= 0 || g.idx_dim <= 0 || g.idx_top_k <= 0 || g.idx_kpool < 2 || g.hc != 4 || g.n_expert <= 0 ||
        g.n_used <= 0 || g.n_used > g.n_expert || g.n_shared < 0 || g.n_dense_lead < 0 || g.n_dense_lead > g.n_layers ||
        g.n_ff_dense <= 0 || g.n_ff_exp <= 0)
        return bad("a glm5-next key is zero or out of range");
    if (g.idx_top_k % g.idx_kpool) return bad("attention.indexer.top_k is not a multiple of attention.indexer.kpool");
    if (g.gating_func != 2) return bad("expert_gating_func is not 2 (sigmoid): only the sigmoid router is known");
    if (!kv || kv->items.size() < (size_t) g.n_layers) return bad("attention.head_count_kv is shorter than the trunk");
    for (int64_t l = 0; l < g.n_layers; ++l)
        if (g.kv_heads[(size_t) l] > 1) return bad("attention.head_count_kv has an entry above 1 (the MLA cache is one latent head)");
    return true;
}

struct Glm5Want {
    std::string name;            ///< a blk.N. suffix for a layer's tensor, the full name for a global one
    std::vector<int64_t> shape;
    bool optional = false;
};

/// Every tensor of layer `l` (trunk 0..n_layers-1, then the MTP block).
inline std::vector<Glm5Want> glm5_layer_tensors(const Glm5Geometry& g, int64_t l) {
    const int64_t N = g.n_embd, NH = g.n_head, DI = g.d_inner();
    std::vector<Glm5Want> t;
    auto add = [&](const char* s, std::vector<int64_t> shape, bool optional = false) { t.push_back({s, std::move(shape), optional}); };
    add("attn_norm.weight", {N});
    add("ffn_norm.weight", {N});
    if (!g.is_mtp(l)) {
        add("hc_attn_fn.weight", {g.hc * N, g.hc_mix()});
        add("hc_attn_base.weight", {g.hc_mix()});
        add("hc_attn_scale.weight", {3});
        add("hc_ffn_fn.weight", {g.hc * N, g.hc_mix()});
        add("hc_ffn_base.weight", {g.hc_mix()});
        add("hc_ffn_scale.weight", {3});
    }
    if (g.is_kda(l)) {
        add("attn_q.weight", {N, DI});
        add("attn_k.weight", {N, DI});
        add("attn_v.weight", {N, DI});
        add("ssm_conv1d_q.weight", {g.d_conv, 1, DI});
        add("ssm_conv1d_k.weight", {g.d_conv, 1, DI});
        add("ssm_conv1d_v.weight", {g.d_conv, 1, DI});
        add("ssm_f_a.weight", {N, g.kda_head_dim});
        add("ssm_f_b.weight", {g.kda_head_dim, DI});
        add("ssm_beta.weight", {N, NH});
        add("ssm_a", {NH});
        add("ssm_dt.bias", {DI});
        add("ssm_g_a.weight", {N, g.kda_head_dim});
        add("ssm_g_b.weight", {g.kda_head_dim, DI});
        add("ssm_norm.weight", {g.kda_head_dim});
        add("attn_output.weight", {DI, N});
    } else {
        add("attn_q_a_norm.weight", {g.q_lora});
        add("attn_kv_a_norm.weight", {g.kv_lora});
        add("attn_q_a.weight", {N, g.q_lora});
        add("attn_q_b.weight", {g.q_lora, NH * g.k_mla});
        add("attn_kv_a_mqa.weight", {N, g.kv_lora + g.n_rot});
        add("attn_k_b.weight", {g.k_mla - g.n_rot, g.kv_lora, NH});
        add("attn_v_b.weight", {g.kv_lora, g.v_mla, NH});
        add("attn_output.weight", {NH * g.v_mla, N});
        if (g.has_indexer(l)) {
            add("indexer.k_norm.weight", {g.idx_dim});
            add("indexer.k_norm.bias", {g.idx_dim});
            add("indexer.proj.weight", {N, g.idx_heads});
            add("indexer.attn_k.weight", {N, g.idx_dim});
            add("indexer.attn_q_b.weight", {g.q_lora, g.idx_heads * g.idx_dim});
            add("indexer_compressor_gate.weight", {N, g.idx_dim});
            add("indexer_compressor_ape.weight", {g.idx_dim, g.idx_kpool});
        }
    }
    if (g.is_dense(l)) {
        add("ffn_gate.weight", {N, g.n_ff_dense});
        add("ffn_down.weight", {g.n_ff_dense, N});
        add("ffn_up.weight", {N, g.n_ff_dense});
    } else {
        add("ffn_gate_inp.weight", {N, g.n_expert});
        add("exp_probs_b.bias", {g.n_expert});
        add("ffn_gate_exps.weight", {N, g.n_ff_exp, g.n_expert});
        add("ffn_down_exps.weight", {g.n_ff_exp, N, g.n_expert});
        add("ffn_up_exps.weight", {N, g.n_ff_exp, g.n_expert});
        add("ffn_gate_shexp.weight", {N, g.n_ff_exp * g.n_shared});
        add("ffn_down_shexp.weight", {g.n_ff_exp * g.n_shared, N});
        add("ffn_up_shexp.weight", {N, g.n_ff_exp * g.n_shared});
    }
    if (g.is_mtp(l)) {
        add("nextn.eh_proj.weight", {2 * N, N});
        add("nextn.enorm.weight", {N});
        add("nextn.hnorm.weight", {N});
        add("nextn.shared_head_norm.weight", {N}, true);
        add("nextn.embed_tokens.weight", {N, g.n_vocab}, true);
        add("nextn.shared_head_head.weight", {N, g.n_vocab}, true);
    }
    return t;
}

/// The tensors outside the layers.
inline std::vector<Glm5Want> glm5_global_tensors(const Glm5Geometry& g) {
    return {{"token_embd.weight", {g.n_embd, g.n_vocab}, false},
            {"output.weight", {g.n_embd, g.n_vocab}, false},
            {"output_norm.weight", {g.n_embd}, false}};
}

/// What a layout check found: counts and bytes, so the planner's inputs (what streams, what stays resident) come from the file.
struct Glm5LayoutReport {
    size_t checked = 0;
    uint64_t bytes_total = 0;
    uint64_t bytes_experts = 0;                       ///< ffn_{gate,up,down}_exps: the part that streams
    uint64_t bytes_resident = 0;                      ///< everything else
    std::map<std::string, uint64_t> bytes_by_type;    ///< per ggml type name, experts only
    std::vector<uint64_t> expert_bytes_per_layer;     ///< routed-expert bytes of each checked layer (0 for dense / absent)
    std::vector<uint64_t> bytes_per_expert;           ///< per checked layer: gate+up+down bytes of ONE expert
    std::vector<std::string> expert_types;            ///< per checked layer: "Q2_K/Q2_K/Q3_K" (gate/up/down)
};

inline bool glm5_shape_equal(const std::vector<uint64_t>& got, const std::vector<int64_t>& want) {
    size_t a = got.size(), b = want.size();
    while (a > 0 && got[a - 1] == 1) --a;
    while (b > 0 && want[b - 1] == 1) --b;
    if (a != b) return false;
    for (size_t i = 0; i < a; ++i)
        if (got[i] != (uint64_t) want[i]) return false;
    return true;
}

/// Checks layers [first, last) of `g` (and the global tensors when `with_globals`) against model `m`: every table entry
/// present with its shape, inside its file, of a known type; and no tensor of the file outside the table.
inline bool check_glm5_layout(const strata::GgufModel& m, const Glm5Geometry& g, int64_t first, int64_t last, bool with_globals,
                              std::string& err, Glm5LayoutReport* report = nullptr) {
    std::set<std::string> named;
    Glm5LayoutReport local;
    Glm5LayoutReport& r = report ? *report : local;
    r = Glm5LayoutReport{};
    auto check = [&](const std::string& name, const std::vector<int64_t>& shape, bool optional, bool& found) {
        found = false;
        size_t s = 0;
        const strata::TensorInfo* t = m.find(name, &s);
        if (!t) {
            if (optional) return true;
            err = "missing " + name;
            return false;
        }
        named.insert(name);
        if (!glm5_shape_equal(t->shape, shape)) {
            std::string got, want;
            for (uint64_t d : t->shape) got += (got.empty() ? "" : ", ") + std::to_string(d);
            for (int64_t d : shape) want += (want.empty() ? "" : ", ") + std::to_string(d);
            err = name + " is [" + got + "], the GLM-5.3-Flash layout wants [" + want + "]";
            return false;
        }
        if (!m.in_bounds(*t, s)) {
            err = name + " (" + t->type_name() + ") has no known size inside " + m.shard(s).path();
            return false;
        }
        const uint64_t bytes = strata::tensor_payload_bytes(*t);
        ++r.checked;
        r.bytes_total += bytes;
        const bool exps = name.find("_exps.") != std::string::npos;
        (exps ? r.bytes_experts : r.bytes_resident) += bytes;
        if (exps) r.bytes_by_type[t->type_name()] += bytes;
        found = true;
        return true;
    };
    bool found = false;
    if (with_globals)
        for (const Glm5Want& w : glm5_global_tensors(g))
            if (!check(w.name, w.shape, w.optional, found)) return false;
    for (int64_t l = first; l < last; ++l) {
        uint64_t layer_exps = 0;
        std::string types;
        for (const Glm5Want& w : glm5_layer_tensors(g, l)) {
            const std::string full = "blk." + std::to_string(l) + "." + w.name;
            if (!check(full, w.shape, w.optional, found)) return false;
            if (found && w.name.find("_exps.") != std::string::npos) {
                const strata::TensorInfo* t = m.find(full);
                layer_exps += strata::tensor_payload_bytes(*t);
                types += (types.empty() ? "" : "/") + std::string(t->type_name());
            }
        }
        r.expert_bytes_per_layer.push_back(layer_exps);
        r.bytes_per_expert.push_back(layer_exps / (uint64_t) g.n_expert);
        r.expert_types.push_back(types);
    }
    for (size_t s = 0; s < m.size(); ++s)
        for (const strata::TensorInfo& t : m.shard(s).tensors())
            if (!named.count(t.name)) {
                err = t.name + " is not in the GLM-5.3-Flash layout (layers " + std::to_string(first) + ".." +
                      std::to_string(last - 1) + ")";
                return false;
            }
    return true;
}

}  // namespace strata::core
