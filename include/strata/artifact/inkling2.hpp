// include/strata/artifact/inkling2.hpp - the `inkling2` GGUF (Thinking Machines' Inkling 2, e.g. Inkling-2-Mini): its configuration
// from the file's metadata and its tensor layout, checked tensor by tensor.  M1 of docs/INKLING-PORT.md: no engine code runs from this yet.
//
// What the file holds (unsloth's Inkling-2-Mini-UD-Q4_K_XL, read 22 Nov 2026; llama.cpp master's inkling2 layout):
//   * `block_count` is the number of layer VISITS (48), not the stored layers (24): the looped core, layers [core_start, core_end),
//     runs num_loops times with shared weights, so  visits = stored + (num_loops - 1) * (core_end - core_start).
//   * a stored layer is a full-attention layer when its index % 6 == 5, else sliding-window (the metadata's per-visit
//     `sliding_window_pattern` says the same and is checked against it).
//   * experts are fused: `ffn_gate_up_exps.weight` [n_embd, 2 * n_ff_exp, n_expert] (gate rows first) and `ffn_down_exps.weight`
//     [n_ff_exp, n_embd, n_expert]; NVFP4 experts carry a per-expert F32 `.scale` beside them, Q8_0 / other types carry none.
//
// Dependencies: the GGUF reader only.  Header-only, no allocation beyond the structs.
#pragma once

#include "strata/artifact/gguf_reader.hpp"

#include <cstdint>
#include <initializer_list>
#include <map>
#include <string>
#include <vector>

namespace strata {
namespace inkling2 {

constexpr const char* kArch = "inkling2";

/// ggml type ids this layout uses (the reader's numbering).
constexpr uint32_t kF32 = 0, kBF16 = 30, kQ8_0 = 8, kNVFP4 = 40;

struct Config {
    // widths
    int64_t n_visits = 0;        ///< `block_count`: the layer visits of one token (48)
    int64_t n_stored = 0;        ///< the stored layers (24), derived
    int64_t n_embd = 0;          ///< 2304
    int64_t n_ff_dense = 0;      ///< `feed_forward_length` (1152; the experts' width too, `expert_feed_forward_length`)
    int64_t n_ff_exp = 0;        ///< 1152
    int64_t n_vocab = 0;         ///< 201024 padded rows
    int64_t n_vocab_real = 0;    ///< 200058
    int64_t n_ctx_train = 0;     ///< 1048576
    // attention
    int64_t n_head = 0;          ///< 24 query heads
    int64_t n_head_kv = 0;       ///< 8
    int64_t head_dim = 0;        ///< 128 (key_length = value_length, and the sliding layers' `_swa` ones)
    int64_t window = 0;          ///< sliding window 1024
    int64_t rel_states = 0;      ///< `relative_length` 16: per-head relative states mixed into a bias by distance
    int64_t rel_extent = 0;      ///< `relative_extent` 1024: the distances the bias covers
    int64_t conv_taps = 0;       ///< `shortconv.l_cache` 4
    int64_t attn_res_block = 0;  ///< `attn_res.block_size` 6
    int64_t full_every = 6;      ///< a stored layer is full when index % full_every == full_every - 1
    float rms_eps = 0;
    float qk_clip = 0;           ///< 22.627417
    float rel_clip = 0;          ///< 8.0
    float log_scale_alpha = 0;   ///< 0.1: the full layers' q / r scaling  tau = 1 + alpha * ln(max(1, (pos + 1) / floor))
    int64_t log_scale_floor = 0; ///< 8192
    float final_softcap = 0;     ///< 30
    // experts
    int64_t n_expert = 0;        ///< 128
    int64_t n_expert_used = 0;   ///< 8
    float expert_weight_scale = 0;  ///< 8.0: the picked scores are normalized then multiplied by this
    float situ_beta = 0;         ///< 4: gate softcap
    float situ_linear_beta = 0;  ///< 25: up softcap
    // loop
    int64_t num_loops = 1;       ///< 3
    int64_t core_start = 0;      ///< 6
    int64_t core_end = 0;        ///< 18 (exclusive)
    int64_t gate_rank = 0;       ///< 64: the loop control gates' rank
    // per visit (size n_visits)
    std::vector<uint8_t> visit_sliding;   ///< 1 = sliding window, 0 = full attention (the metadata's pattern)

    int64_t core_len() const { return core_end - core_start; }
    bool full_layer(int64_t stored) const { return stored % full_every == full_every - 1; }
    bool in_core(int64_t stored) const { return stored >= core_start && stored < core_end; }
    /// The stored layer a visit runs (visits 0..core_start-1 once, the core num_loops times, then the rest).
    int64_t stored_of_visit(int64_t v) const {
        if (v < core_start) return v;
        const int64_t in = v - core_start;
        if (in < num_loops * core_len()) return core_start + in % core_len();
        return core_end + (in - num_loops * core_len());
    }
};

namespace detail {
inline const MetaValue* find(const std::map<std::string, MetaValue>& m, const std::string& k) {
    auto it = m.find(k);
    return it == m.end() ? nullptr : &it->second;
}
inline bool num(const std::map<std::string, MetaValue>& m, const std::string& k, double& out, std::string& err) {
    const MetaValue* v = find(m, k);
    if (v == nullptr || !v->is_num()) { err = "metadata key '" + k + "' is missing or not a number"; return false; }
    out = v->num();
    return true;
}
}  // namespace detail

/// Fill `c` from the GGUF metadata (`meta` = GgufFile::metadata()).  Empty string = ok, else the reason (nothing is guessed:
/// a key the layout needs that is absent is an error).  Everything is cross-checked: the visit arithmetic, the per-visit arrays'
/// length, the sliding pattern against the "every 6th layer is full" rule.
inline std::string parse_config(const std::map<std::string, MetaValue>& meta, Config& c) {
    std::string err;
    const MetaValue* arch = detail::find(meta, "general.architecture");
    if (arch == nullptr || arch->s != kArch) return std::string("architecture is '") + (arch ? arch->s : "?") + "', not '" + kArch + "'";
    const std::string p = std::string(kArch) + ".";
    double v = 0;
    const auto I = [&](const char* key, int64_t& dst) { if (!detail::num(meta, p + key, v, err)) return false; dst = (int64_t) v; return true; };
    const auto F = [&](const char* key, float& dst) { if (!detail::num(meta, p + key, v, err)) return false; dst = (float) v; return true; };
    int64_t kl = 0, vl = 0, kls = 0, vls = 0, ffe = 0, rope = -1;
    if (!I("block_count", c.n_visits) || !I("embedding_length", c.n_embd) || !I("feed_forward_length", c.n_ff_dense) ||
        !I("expert_feed_forward_length", ffe) || !I("vocab_size", c.n_vocab) || !I("vocab_size_unpadded", c.n_vocab_real) ||
        !I("context_length", c.n_ctx_train) || !I("attention.key_length", kl) || !I("attention.value_length", vl) ||
        !I("attention.key_length_swa", kls) || !I("attention.value_length_swa", vls) || !I("attention.sliding_window", c.window) ||
        !I("attention.relative_length", c.rel_states) || !I("attention.relative_extent", c.rel_extent) ||
        !I("shortconv.l_cache", c.conv_taps) || !I("attn_res.block_size", c.attn_res_block) || !I("expert_count", c.n_expert) ||
        !I("expert_used_count", c.n_expert_used) || !I("num_loops", c.num_loops) || !I("loop.core_start", c.core_start) ||
        !I("loop.core_end", c.core_end) || !I("loop.gate_rank", c.gate_rank) || !I("attention.log_scale_floor", c.log_scale_floor) ||
        !I("rope.dimension_count", rope) ||
        !F("attention.layer_norm_rms_epsilon", c.rms_eps) || !F("attention.qk_clip", c.qk_clip) || !F("attention.relative_clip", c.rel_clip) ||
        !F("attention.log_scale_alpha", c.log_scale_alpha) || !F("final_logit_softcapping", c.final_softcap) ||
        !F("expert_weights_scale", c.expert_weight_scale) || !F("activation.situ_beta", c.situ_beta) ||
        !F("activation.situ_linear_beta", c.situ_linear_beta))
        return err;
    c.n_ff_exp = ffe;
    if (rope != 0) return "rope.dimension_count is " + std::to_string(rope) + ": Inkling 2 has no RoPE (0)";
    if (kl != vl || kls != vls || kl != kls) return "key / value lengths differ (" + std::to_string(kl) + "/" + std::to_string(vl) + ", swa " + std::to_string(kls) + "/" + std::to_string(vls) + ")";
    c.head_dim = kl;
    if (ffe != c.n_ff_dense) return "expert_feed_forward_length " + std::to_string(ffe) + " differs from feed_forward_length " + std::to_string(c.n_ff_dense);
    if (c.num_loops < 1 || c.core_start < 0 || c.core_end <= c.core_start) return "the loop's core range is empty or reversed";
    if ((c.core_end - c.core_start) % c.attn_res_block != 0) return "the loop core is not whole attention-residual blocks";
    c.n_stored = c.n_visits - (c.num_loops - 1) * c.core_len();
    if (c.n_stored < c.core_end) return "block_count " + std::to_string(c.n_visits) + " is too small for a " + std::to_string(c.num_loops) + "-pass core ending at " + std::to_string(c.core_end);
    c.full_every = c.attn_res_block;   // the full layers are every 6th, the same 6 as the residual blocks; the pattern below checks it visit by visit
    // the per-layer arrays: head counts (one number repeated, as the reader keeps the first 64 of an array)
    const MetaValue* hc = detail::find(meta, p + "attention.head_count");
    const MetaValue* hk = detail::find(meta, p + "attention.head_count_kv");
    const MetaValue* sp = detail::find(meta, p + "attention.sliding_window_pattern");
    if (hc == nullptr || hk == nullptr || sp == nullptr || hc->type != MetaType::ARRAY || hk->type != MetaType::ARRAY || sp->type != MetaType::ARRAY)
        return "head_count, head_count_kv or sliding_window_pattern is missing or not an array";
    if ((int64_t) hc->count != c.n_visits || (int64_t) hk->count != c.n_visits || (int64_t) sp->count != c.n_visits)
        return "a per-visit array has " + std::to_string(hc->count) + "/" + std::to_string(hk->count) + "/" + std::to_string(sp->count) + " entries, block_count is " + std::to_string(c.n_visits);
    if (hc->items.size() < (size_t) c.n_visits || hk->items.size() < (size_t) c.n_visits || sp->items.size() < (size_t) c.n_visits)
        return "a per-visit array is longer than the reader keeps (64 entries)";
    c.n_head = (int64_t) hc->items[0].u;
    c.n_head_kv = (int64_t) hk->items[0].u;
    c.visit_sliding.assign((size_t) c.n_visits, 0);
    for (int64_t i = 0; i < c.n_visits; ++i) {
        if ((int64_t) hc->items[(size_t) i].u != c.n_head || (int64_t) hk->items[(size_t) i].u != c.n_head_kv) return "head counts differ between visits (not supported)";
        c.visit_sliding[(size_t) i] = (uint8_t) (sp->items[(size_t) i].u != 0);
    }
    for (int64_t vi = 0; vi < c.n_visits; ++vi) {
        const bool want_full = c.full_layer(c.stored_of_visit(vi));
        if ((c.visit_sliding[(size_t) vi] == 0) != want_full)
            return "visit " + std::to_string(vi) + ": sliding_window_pattern disagrees with 'stored layer index % 6 == 5 is full'";
    }
    if (c.n_head <= 0 || c.n_head_kv <= 0 || c.n_head % c.n_head_kv != 0) return "head counts are not a GQA ratio";
    return {};
}

/// One stored layer's tensors (null = absent for this layer type).
struct LayerTensors {
    bool full = false;                 ///< full attention (dynamic k / v convs, 3 residual queries) vs sliding
    bool core = false;                 ///< in the looped core (has the loop control tensors)
    bool expert_nvfp4 = false;         ///< the experts are NVFP4 (with .scale tensors), else a plain type
    const TensorInfo *attn_norm = nullptr, *attn_q = nullptr, *attn_k = nullptr, *attn_v = nullptr, *attn_gate = nullptr, *attn_output = nullptr;
    const TensorInfo *attn_q_clip = nullptr, *attn_rel = nullptr, *attn_rel_bias = nullptr, *attn_rel_clip = nullptr, *attn_sink_proj = nullptr;
    const TensorInfo *attn_k_conv = nullptr, *attn_v_conv = nullptr, *attn_post_conv = nullptr;
    const TensorInfo *attn_k_conv_proj = nullptr, *attn_v_conv_proj = nullptr;   // full layers
    const TensorInfo *attn_q_inp_norm = nullptr, *attn_k_inp_norm = nullptr, *attn_v_inp_norm = nullptr;   // full layers
    const TensorInfo *attn_res_query = nullptr;     // none at layer 0; [n_embd] or, on a full layer, [n_embd, 3]
    const TensorInfo *ffn_norm = nullptr, *ffn_res_query = nullptr, *ffn_post_conv = nullptr;
    const TensorInfo *router = nullptr, *router_bias = nullptr;
    const TensorInfo *gate_up_exps = nullptr, *down_exps = nullptr, *gate_up_scale = nullptr, *down_scale = nullptr;
    // loop control (core layers): attn, k, v, mlp each with down [2 * n_embd, rank], forget / gain [rank, n_embd]
    const TensorInfo *loop_down[4] = {}, *loop_forget[4] = {}, *loop_gain[4] = {};
};

struct Layout {
    const TensorInfo *token_embd = nullptr, *token_embd_norm = nullptr, *output = nullptr, *output_norm = nullptr, *output_res_query = nullptr;
    const TensorInfo *loop_res_query = nullptr, *loop_kv_res_query = nullptr;
    std::vector<LayerTensors> layers;   ///< the stored layers
    uint64_t expert_bytes = 0;          ///< all routed-expert tensors (what the tiers will move)
    uint64_t dense_bytes = 0;           ///< everything else in the file
};

namespace detail {
inline std::string shape_str(const std::vector<uint64_t>& s) {
    std::string r = "[";
    for (size_t i = 0; i < s.size(); ++i) r += (i ? ", " : "") + std::to_string(s[i]);
    return r + "]";
}
/// The tensor `name`, which must have exactly `shape` (GGUF order) and, when `types` is non-empty, one of those types.
inline const TensorInfo* need(const GgufFile& g, const std::string& name, const std::vector<uint64_t>& shape, std::initializer_list<uint32_t> types, std::string& err) {
    const TensorInfo* t = g.find(name);
    if (t == nullptr) { err = "tensor '" + name + "' is missing"; return nullptr; }
    if (t->shape != shape) { err = "tensor '" + name + "' has shape " + shape_str(t->shape) + ", expected " + shape_str(shape); return nullptr; }
    if (types.size() != 0) {
        bool ok = false;
        for (uint32_t ty : types) ok = ok || t->type == ty;
        if (!ok) { err = "tensor '" + name + "' has type " + t->type_name() + ", which this layout does not take"; return nullptr; }
    }
    return t;
}
}  // namespace detail

/// Resolve every tensor the model needs, with its exact shape and an allowed type.  Empty string = ok, else the first problem.
/// Also checks nothing in the file is left over (an unknown tensor means a layout this code does not know).
inline std::string load_layout(const GgufFile& g, const Config& c, Layout& out) {
    std::string err;
    const uint64_t E = (uint64_t) c.n_embd, V = (uint64_t) c.n_vocab, QH = (uint64_t) (c.n_head * c.head_dim), KH = (uint64_t) (c.n_head_kv * c.head_dim);
    const uint64_t FF = (uint64_t) c.n_ff_exp, NE = (uint64_t) c.n_expert, R = (uint64_t) c.gate_rank;
    using L = std::initializer_list<uint32_t>;
    const L dense = {kQ8_0, kBF16, kF32};
    const L f32 = {kF32};
    uint64_t counted = 0;
    const auto N = [&](const std::string& n, std::vector<uint64_t> s, L ty) { const TensorInfo* t = detail::need(g, n, s, ty, err); if (t) ++counted; return t; };
    out = Layout{};
    out.token_embd = N("token_embd.weight", {E, V}, dense);
    out.token_embd_norm = N("token_embd_norm.weight", {E}, f32);
    out.output = N("output.weight", {E, V}, dense);
    out.output_norm = N("output_norm.weight", {E}, f32);
    out.output_res_query = N("output_res_query.weight", {E}, f32);
    out.loop_res_query = N("loop_res_query.weight", {E, (uint64_t) (2 * c.core_len())}, f32);
    out.loop_kv_res_query = N("loop_kv_res_query.weight", {E, (uint64_t) (2 * c.core_len())}, f32);
    if (!err.empty()) return err;
    out.layers.assign((size_t) c.n_stored, LayerTensors{});
    for (int64_t l = 0; l < c.n_stored; ++l) {
        LayerTensors& t = out.layers[(size_t) l];
        const std::string b = "blk." + std::to_string(l) + ".";
        t.full = c.full_layer(l);
        t.core = c.in_core(l);
        if (!t.full) t.attn_norm = N(b + "attn_norm.weight", {E}, f32);   // a full layer has three input norms (q, k, v) instead
        t.attn_q =N(b + "attn_q.weight", {E, QH}, dense);
        t.attn_k = N(b + "attn_k.weight", {E, KH}, dense);
        t.attn_v = N(b + "attn_v.weight", {E, KH}, dense);
        t.attn_gate = N(b + "attn_gate.weight", {E, QH}, dense);
        t.attn_output = N(b + "attn_output.weight", {QH, E}, dense);
        t.attn_q_clip = N(b + "attn_q_clip.weight", {(uint64_t) c.head_dim, (uint64_t) c.n_head}, f32);
        t.attn_rel = N(b + "attn_rel.weight", {E, (uint64_t) (c.n_head * c.rel_states)}, dense);
        t.attn_rel_bias = N(b + "attn_rel_bias.weight", {(uint64_t) c.rel_states, (uint64_t) c.rel_extent}, f32);
        t.attn_rel_clip = N(b + "attn_rel_clip.weight", {(uint64_t) c.rel_states, (uint64_t) c.n_head}, f32);
        t.attn_sink_proj = N(b + "attn_sink_proj.weight", {E, (uint64_t) c.n_head}, dense);
        t.attn_k_conv = N(b + "attn_k_conv.weight", {(uint64_t) c.conv_taps, KH}, f32);
        t.attn_v_conv = N(b + "attn_v_conv.weight", {(uint64_t) c.conv_taps, KH}, f32);
        t.attn_post_conv = N(b + "attn_post_conv.weight", {(uint64_t) c.conv_taps, E}, f32);
        t.ffn_post_conv = N(b + "ffn_post_conv.weight", {(uint64_t) c.conv_taps, E}, f32);
        if (t.full) {
            const uint64_t D = (uint64_t) (c.n_head_kv * c.conv_taps);   // per-token taps: heads x taps
            t.attn_k_conv_proj = N(b + "attn_k_conv_proj.weight", {E, D}, dense);
            t.attn_v_conv_proj = N(b + "attn_v_conv_proj.weight", {E, D}, dense);
            t.attn_q_inp_norm = N(b + "attn_q_inp_norm.weight", {E}, f32);
            t.attn_k_inp_norm = N(b + "attn_k_inp_norm.weight", {E}, f32);
            t.attn_v_inp_norm = N(b + "attn_v_inp_norm.weight", {E}, f32);
        }
        if (l > 0) t.attn_res_query = N(b + "attn_res_query.weight", t.full ? std::vector<uint64_t>{E, 3} : std::vector<uint64_t>{E}, f32);
        t.ffn_norm = N(b + "ffn_norm.weight", {E}, f32);
        t.ffn_res_query = N(b + "ffn_res_query.weight", {E}, f32);
        t.router = N(b + "ffn_gate_inp.weight", {E, NE}, f32);
        t.router_bias = N(b + "exp_probs_b.bias", {NE}, f32);
        t.gate_up_exps = N(b + "ffn_gate_up_exps.weight", {E, 2 * FF, NE}, {kNVFP4, kQ8_0});
        if (t.gate_up_exps != nullptr) {
            t.expert_nvfp4 = t.gate_up_exps->type == kNVFP4;
            t.down_exps = N(b + "ffn_down_exps.weight", {FF, E, NE}, t.expert_nvfp4 ? L{kNVFP4} : L{kQ8_0});
            if (t.expert_nvfp4) {   // an NVFP4 expert tensor without its scale would be wrong by its amax: refused
                t.gate_up_scale = N(b + "ffn_gate_up_exps.scale", {NE}, f32);
                t.down_scale = N(b + "ffn_down_exps.scale", {NE}, f32);
            } else if (g.find(b + "ffn_gate_up_exps.scale") != nullptr || g.find(b + "ffn_down_exps.scale") != nullptr) {
                err = b + ": a .scale tensor beside non-NVFP4 experts (not part of this layout)";
            }
        }
        if (t.core) {
            static const char* kNames[4] = {"attn", "k", "v", "mlp"};
            for (int i = 0; i < 4; ++i) {
                t.loop_down[i] = N(b + "loop_" + kNames[i] + "_down.weight", {(uint64_t) (2 * c.n_embd), R}, f32);
                t.loop_forget[i] = N(b + "loop_" + kNames[i] + "_forget.weight", {R, E}, f32);
                t.loop_gain[i] = N(b + "loop_" + kNames[i] + "_gain.weight", {R, E}, f32);
            }
        }
        if (!err.empty()) return err;
        for (const TensorInfo* e : {t.gate_up_exps, t.down_exps}) if (e) { out.expert_bytes += tensor_payload_bytes(*e); }
    }
    if (!err.empty()) return err;
    // nothing unaccounted: every tensor in the file was claimed above
    if (counted != g.tensors().size()) {
        return "the file has " + std::to_string(g.tensors().size()) + " tensors, this layout accounts for " + std::to_string(counted) + " (an unknown tensor means an unknown layout)";
    }
    uint64_t all = 0;
    for (const auto& t : g.tensors()) all += tensor_payload_bytes(t);
    out.dense_bytes = all - out.expert_bytes;
    return {};
}

}  // namespace inkling2
}  // namespace strata
