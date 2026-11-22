#include "arg.h"
#include "common.h"
#include "debug.h"
#include "log.h"
#include "llama.h"

#include <clocale>
#include <string>
#include <vector>

static bool run(llama_context * ctx, const common_params & params, const std::string & prompt) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> tokens = common_tokenize(ctx, prompt, add_bos, true);

    if (tokens.empty()) {
        LOG_ERR("%s : there are not input tokens to process - (try to provide a prompt with '-p')\n", __func__);
        return false;
    }

    LOG_INF("number of input tokens = %zu\n", tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        LOG_INF("  %d\n", tokens[i]);
    }

    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
        LOG_ERR("%s : failed to eval\n", __func__);
        return false;
    }

    // TEMPORARY diagnostic: the moe-cache corruption reproducer needs actual
    // token-by-token GENERATION (decode with n_tokens=1, reading from the KV
    // cache) - a single prefill decode() above was verified completely clean
    // through every layer, so the bug must live specifically in the decode
    // step's compute path, not prefill's. Greedy-sample and decode a handful
    // of tokens to reach it under the same abort_on_nan instrumentation.
    const llama_model * m = llama_get_model(ctx);
    const llama_vocab * v = llama_model_get_vocab(m);
    const int32_t n_vocab = llama_vocab_n_tokens(v);
    for (int step = 0; step < 40; step++) {
        float * logits = llama_get_logits(ctx);
        llama_token best = 0;
        float best_val = logits[0];
        for (int32_t i = 1; i < n_vocab; i++) {
            if (logits[i] > best_val) { best_val = logits[i]; best = i; }
        }
        LOG("%s: greedy step %d -> token %d\n", __func__, step, best);
        if (llama_vocab_is_eog(v, best)) {
            break;
        }
        if (llama_decode(ctx, llama_batch_get_one(&best, 1))) {
            LOG_ERR("%s : failed to eval decode step %d\n", __func__, step);
            return false;
        }
    }

    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    // TEMPORARY diagnostic: abort at the first tensor whose graph output sums
    // to NaN, printing every tensor name/op up to that point - localizes the
    // moe-cache corruption bug's root NaN to the exact op, not just the layer.
    common_debug_cb_user_data cb_data(params, {}, /*abort_on_nan=*/true);

    llama_backend_init();
    llama_numa_init(params.numa);

    // pass the callback to the backend scheduler
    // it will be executed for each node during the graph computation
    params.cb_eval = common_debug_cb_eval;
    params.cb_eval_user_data = &cb_data;
    params.warmup = false;

    // init
    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
        LOG_INF("\n");
    }

    // TEMPORARY diagnostic: the moe-cache corruption reproducer needs TWO
    // sequential, non-concurrent requests where the second uses DIFFERENT
    // content than the first (server-verified: identical prompt repeated is
    // clean, a different second prompt corrupts, single slot, no
    // concurrency) - reset the KV cache between them the same way a fresh
    // request would start, and see whether a bare CLI decode loop (no
    // server slot machinery, no HTTP layer) reproduces it too.
    bool OK = run(ctx, params, params.prompt);
    if (!OK) {
        return 1;
    }
    if (const char * p2 = getenv("EVAL_CALLBACK_PROMPT2")) {
        // seq_id=-1 (wildcard) hit an unrelated GGML_ASSERT in
        // llama_memory_recurrent::find_slot for this hybrid-memory model -
        // a real bug in that reset path, but not the one being chased here.
        // Removing the SPECIFIC sequence (0) instead is what llama-server
        // actually does to release a slot between requests.
        llama_memory_seq_rm(llama_get_memory(ctx), 0, 0, -1);
        LOG("\n%s: --- second prompt, fresh KV cache, same session ---\n", __func__);
        OK = run(ctx, params, p2);
        if (!OK) {
            return 1;
        }
    }

    LOG("\n");
    llama_perf_context_print(ctx);

    llama_backend_free();

    return 0;
}
