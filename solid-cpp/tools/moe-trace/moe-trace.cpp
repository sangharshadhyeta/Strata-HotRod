// Dump the MoE expert-selection trace: which expert is required, at which layer,
// for which token. Everything about tiering -- how long an expert must stay
// resident, what cache size is worth buying -- follows from this sequence.
// Optionally (MOE_TRACE_HIDDEN set) also dumps the exact hidden state each
// layer's router consumes, so a cross-layer lookahead approximation can be
// checked against what a later layer's router actually needed, rather than
// assumed from residual-stream-continuity folklore.
#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <unordered_map>

struct trace_ctx {
    FILE * f  = nullptr;
    FILE * hf = nullptr;    // hidden states: int32 il, int32 tok, int32 n_dim, float[n_dim]
    long long recs = 0;

    // Signal 3: the router's own score for each expert, not just its ranking.
    // The ranking alone says which pick was lost to a cache miss; the score
    // says what substituting the best resident alternative would actually
    // cost. Off unless MOE_TRACE_SCORES is set - the default line format is
    // unchanged for anything already parsing this file.
    bool scores = false;
    // ffn_moe_probs for the layer currently being computed, keyed by layer.
    // probs is produced before the argsort that ranks it (the argsort's input
    // derives from it), so by the time a layer's ranking arrives its scores
    // are already here. Cleared as each layer is emitted.
    std::unordered_map<int, std::vector<float>> probs;
    std::unordered_map<int, int64_t>            probs_n_expert;
};

// True only for "<prefix><digits>" - ggml appends suffixes like " (reshaped)"
// to views of a tensor, and those views share the base name. Matching on the
// prefix alone let ffn_moe_probs-0 (reshaped), a [1, ...] view, overwrite the
// real [n_expert, n_tokens] probs and silently suppress every score.
static bool name_is_indexed(const char * name, const char * prefix, size_t plen) {
    if (!name || strncmp(name, prefix, plen) != 0 || name[plen] == '\0') {
        return false;
    }
    for (const char * p = name + plen; *p; p++) {
        if (*p < '0' || *p > '9') {
            return false;
        }
    }
    return true;
}

static bool trace_cb(ggml_tensor * t, bool ask, void * ud) {
    trace_ctx * d = (trace_ctx *) ud;
    const bool want_topk = t->name && strncmp(t->name, "ffn_moe_argsort", 15) == 0 && t->type == GGML_TYPE_I32;
    const bool want_hid  = d->hf && t->name && strncmp(t->name, "ffn_moe_router_in", 17) == 0 && t->type == GGML_TYPE_F32;
    // Trailing '-' matters: it selects ffn_moe_probs-<il> and not the
    // ffn_moe_probs_biased/_masked variants, which are the selection-time
    // scores. The plain probs are the ones that become the expert weights
    // (weights = get_rows(probs, selected_experts) in build_moe_ffn), so
    // they are what a substitution's cost has to be measured against.
    const bool want_probs = d->scores &&
        name_is_indexed(t->name, "ffn_moe_probs-", 14) && t->type == GGML_TYPE_F32;
    if (ask) {
        // MOE_TRACE_NAMES=1 lists every MoE tensor the scheduler offers, so a
        // name that silently stops matching is visible rather than guessed at.
        if (getenv("MOE_TRACE_NAMES") && t->name && strstr(t->name, "ffn_moe")) {
            fprintf(stderr, "[moe-trace-name] %s type=%d ne0=%lld\n",
                    t->name, (int) t->type, (long long) t->ne[0]);
        }
        return want_topk || want_hid || want_probs;
    }
    if (want_probs) {
        int il = -1;
        if (const char * dash = strrchr(t->name, '-')) {
            il = atoi(dash + 1);
        }
        std::vector<float> buf(ggml_nelements(t));
        ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
        d->probs_n_expert[il] = t->ne[0];
        d->probs[il]          = std::move(buf);
        return true;
    }
    if (want_hid) {
        int il = -1;
        if (const char * dash = strrchr(t->name, '-')) {
            il = atoi(dash + 1);
        }
        const int64_t n_dim = t->ne[0];
        const int64_t n_tok = ggml_nelements(t) / n_dim;
        std::vector<float> buf(ggml_nelements(t));
        ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
        for (int64_t tok = 0; tok < n_tok; tok++) {
            int32_t hdr[3] = { il, (int32_t) tok, (int32_t) n_dim };
            fwrite(hdr, sizeof(int32_t), 3, d->hf);
            fwrite(buf.data() + tok * n_dim, sizeof(float), (size_t) n_dim, d->hf);
        }
        return true;
    }
    if (!want_topk) {
        return true;
    }
    // name is "ffn_moe_argsort-<il>"; recover the layer index
    int il = -1;
    if (const char * dash = strrchr(t->name, '-')) {
        il = atoi(dash + 1);
    }
    const int64_t n_expert = t->ne[0];          // full ranking per token
    const int64_t n_tokens = ggml_nelements(t) / n_expert;
    const char * envk = getenv("MOE_TOPK");
    const int64_t n_used = envk ? atoll(envk) : 6;   // leading entries are the selection
    std::vector<int32_t> buf(ggml_nelements(t));
    ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
    // one line per (layer, token): the selected expert ids, each optionally
    // followed by ':' and the router's score for it.
    const auto pit = d->probs.find(il);
    const bool have_scores = d->scores && pit != d->probs.end();
    const int64_t n_prob_expert = have_scores ? d->probs_n_expert[il] : 0;
    for (int64_t tok = 0; tok < n_tokens; tok++) {
        fprintf(d->f, "%d %lld", il, (long long) tok);
        for (int64_t k = 0; k < n_used && k < n_expert; k++) {
            const int32_t id = buf[tok * n_expert + k];
            fprintf(d->f, " %d", id);
            if (have_scores && id >= 0 && id < n_prob_expert) {
                const size_t off = (size_t) tok * (size_t) n_prob_expert + (size_t) id;
                if (off < pit->second.size()) {
                    fprintf(d->f, ":%.6g", pit->second[off]);
                }
            }
        }
        fputc('\n', d->f);
        d->recs++;
    }
    if (have_scores) {
        d->probs.erase(il);   // consumed; the next batch recomputes it
    }
    return true;
}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    trace_ctx d;
    const char * out = getenv("MOE_TRACE") ? getenv("MOE_TRACE") : "moe-trace.txt";
    d.f = fopen(out, "w");
    if (!d.f) { LOG_ERR("cannot open %s\n", out); return 1; }
    d.scores = getenv("MOE_TRACE_SCORES") != nullptr;
    const char * hout = getenv("MOE_TRACE_HIDDEN");
    if (hout) {
        d.hf = fopen(hout, "wb");
        if (!d.hf) { LOG_ERR("cannot open %s\n", hout); return 1; }
    }

    params.cb_eval           = trace_cb;
    params.cb_eval_user_data = &d;
    params.warmup            = false;

    auto llama_init = common_init_from_params(params);
    if (!llama_init || !llama_init->context()) { LOG_ERR("failed to load model\n"); return 1; }
    llama_context * ctx = llama_init->context();
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true, true);
    if (tokens.empty()) { LOG_ERR("no input tokens - pass -p/-f\n"); return 1; }
    LOG_INF("moe-trace: %zu tokens\n", tokens.size());

    const int nb = (int) params.n_batch;
    for (size_t i = 0; i < tokens.size(); i += nb) {
        const int n = (int) std::min((size_t) nb, tokens.size() - i);
        if (llama_decode(ctx, llama_batch_get_one(tokens.data() + i, n))) {
            LOG_ERR("decode failed\n"); break;
        }
    }

    // A "job" is a prompt plus its generated response - trace both, not just
    // prefill, or the measurement only reflects how varied the input text was.
    const int n_predict = getenv("MOE_TRACE_NPREDICT") ? atoi(getenv("MOE_TRACE_NPREDICT")) : 150;
    struct common_sampler * smpl = common_sampler_init(model, params.sampling);
    int n_cur = (int) tokens.size();
    for (int i = 0; i < n_predict; i++) {
        llama_token id = common_sampler_sample(smpl, ctx, -1);
        common_sampler_accept(smpl, id, true);
        if (llama_vocab_is_eog(vocab, id)) {
            break;
        }
        if (llama_decode(ctx, llama_batch_get_one(&id, 1))) {
            LOG_ERR("decode failed during generation\n"); break;
        }
        n_cur++;
    }
    common_sampler_free(smpl);
    LOG_INF("moe-trace: prefill %zu + generated %d = %d total tokens\n",
            tokens.size(), n_cur - (int) tokens.size(), n_cur);

    fclose(d.f);
    if (d.hf) { fclose(d.hf); }
    LOG_INF("moe-trace: wrote %lld (layer,token) records to %s\n", (long long) d.recs, out);
    llama_backend_free();
    return 0;
}
