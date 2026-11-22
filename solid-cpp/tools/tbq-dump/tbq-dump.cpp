// Dump real V-cache vectors (v_cur) to a flat f32 file, so a KV quantization
// scheme can be evaluated against the data it will actually see rather than
// against a synthetic model of it.
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <strings.h>
#include <vector>

struct dump_ctx {
    FILE * f    = nullptr;
    int64_t hd  = 0;    // head_dim (row length V is quantized along)
    int64_t rows = 0;
    int64_t cap  = 0;
};

static bool dump_cb(ggml_tensor * t, bool ask, void * ud) {
    dump_ctx * d = (dump_ctx *) ud;
    const bool want = t->name && strncasecmp(t->name, "Vcur", 4) == 0 && t->type == GGML_TYPE_F32;
    if (ask) {
        return want;                       // ask phase: say whether we want this tensor
    }
    if (!want || d->rows >= d->cap) {
        return true;
    }
    const int64_t hd = t->ne[0];           // ne0 = head_dim, rest = heads * tokens
    const int64_t n  = ggml_nelements(t);
    if (d->hd == 0) {
        d->hd = hd;
        fwrite(&hd, sizeof(hd), 1, d->f);  // header: one int64 head_dim
    }
    if (hd != d->hd) {
        return true;
    }
    std::vector<float> buf(n);
    ggml_backend_tensor_get(t, buf.data(), 0, n * sizeof(float));
    int64_t nrow = n / hd;
    if (d->rows + nrow > d->cap) {
        nrow = d->cap - d->rows;
    }
    fwrite(buf.data(), sizeof(float), (size_t) (nrow * hd), d->f);
    d->rows += nrow;
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

    dump_ctx d;
    d.cap = 200000;
    const char * out = getenv("TBQ_DUMP") ? getenv("TBQ_DUMP") : "v_cur.f32";
    d.f = fopen(out, "wb");
    if (!d.f) { LOG_ERR("cannot open %s\n", out); return 1; }

    params.cb_eval           = dump_cb;
    params.cb_eval_user_data = &d;
    params.warmup            = false;

    auto llama_init = common_init_from_params(params);
    if (!llama_init || !llama_init->context()) { LOG_ERR("failed to load model\n"); return 1; }
    llama_context * ctx = llama_init->context();

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true, true);
    if (tokens.empty()) { LOG_ERR("no input tokens - pass -p/-f\n"); return 1; }
    LOG_INF("tbq-dump: %zu tokens\n", tokens.size());

    // decode in chunks so a long prompt does not exceed n_batch
    const int nb = (int) params.n_batch;
    for (size_t i = 0; i < tokens.size() && d.rows < d.cap; i += nb) {
        const int n = (int) std::min((size_t) nb, tokens.size() - i);
        if (llama_decode(ctx, llama_batch_get_one(tokens.data() + i, n))) {
            LOG_ERR("decode failed\n"); break;
        }
    }
    fclose(d.f);
    LOG_INF("tbq-dump: wrote %lld rows of head_dim=%lld to %s\n",
            (long long) d.rows, (long long) d.hd, out);
    llama_backend_free();
    return 0;
}
