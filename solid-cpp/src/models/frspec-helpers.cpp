#include "models.h"

#include "llama.h"

#include <fstream>

// See llama_frspec_set_pending_vocab_map() in llama.h for the full
// rationale. Thread-local rather than a plain global: model loading isn't
// guaranteed single-threaded across every caller in this codebase, and a
// hint meant for "the very next load on this thread" leaking across
// threads would be a real, hard-to-diagnose correctness bug (silently
// trimming the wrong model's vocab).
static thread_local std::string g_frspec_pending_vocab_map;

void llama_frspec_set_pending_vocab_map(const char * path) {
    g_frspec_pending_vocab_map = path ? path : "";
}

std::string llm_frspec_take_pending_vocab_map() {
    std::string path = std::move(g_frspec_pending_vocab_map);
    g_frspec_pending_vocab_map.clear();
    return path;
}

// Plain text, no JSON/header dependency deliberately: this file lives in
// libllama proper (src/models/), which doesn't link nlohmann::json (only
// tools/server does) - and a d2t mapping is nothing more than an ordered
// list of integers, so a dependency-free format is the honest choice, not
// a workaround. One token id per line, trimmed-vocab position = line
// index. The producer side (tools/server, which does have json available)
// writes this same plain format so both sides agree without needing to
// share a schema across the library boundary.
std::vector<int64_t> llm_frspec_load_d2t_sidecar(const std::string & path) {
    std::vector<int64_t> d2t;
    if (path.empty()) {
        return d2t;
    }
    std::ifstream f(path);
    if (!f.good()) {
        return d2t;
    }
    int64_t tok;
    while (f >> tok) {
        d2t.push_back(tok);
    }
    if (!f.eof()) {
        // a genuine parse error partway through is worse than "no trim
        // configured" - a truncated/corrupt file should never silently
        // apply a partial, wrong mapping
        return {};
    }
    return d2t;
}

ggml_tensor * llm_frspec_scatter_to_full_vocab(
        ggml_context * ctx0, ggml_tensor * cur, ggml_tensor * d2t, int64_t n_vocab_full) {
    const int64_t n_draft_vocab = cur->ne[0];
    const int64_t n_outputs     = cur->ne[1];

    GGML_ASSERT(d2t->type == GGML_TYPE_I32);
    GGML_ASSERT(d2t->ne[0] == n_draft_vocab);

    ggml_tensor * logits = ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_vocab_full, n_outputs), -INFINITY);
    cur = ggml_set_rows(ctx0, logits,
            ggml_reshape_3d(ctx0, cur, 1,             n_draft_vocab, n_outputs),
            ggml_reshape_3d(ctx0, d2t, n_draft_vocab, 1,             1));
    return ggml_reshape_2d(ctx0, cur, n_vocab_full, n_outputs);
}

void llm_graph_input_frspec_d2t::set_input(const llama_ubatch * /*ubatch*/) {
    // Constant for the model's whole lifetime - same data on every graph
    // build, unlike ordinary per-ubatch inputs. Cheap enough (a few KB at
    // most for any realistic trim size) that redoing this on every call
    // rather than tracking an "already set" flag isn't worth the extra
    // state.
    //
    // Stored/loaded as int64_t (ids, from the plain-text sidecar) but the
    // tensor itself is I32 (ggml_get_rows() requires it) - real vocab
    // sizes are always far under INT32_MAX, so this narrowing is exact.
    std::vector<int32_t> ids32(ids.begin(), ids.end());
    ggml_backend_tensor_set(d2t, ids32.data(), 0, ids32.size() * sizeof(int32_t));
}

bool llm_graph_input_frspec_d2t::can_reuse(const llm_graph_params & /*params*/) {
    // d2t itself never changes once loaded - always safe to reuse
    // whatever graph this was last built into, regardless of ubatch
    // shape or other graph params.
    return true;
}
