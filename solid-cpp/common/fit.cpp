#include "fit.h"
#include "gguf.h"

#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <sys/stat.h>

#include "log.h"

#include "../src/llama-ext.h"

#include <array>
#include <cassert>
#include <stdexcept>
#include <cinttypes>
#include <fstream>
#include <set>
#include <string>
#include <vector>

// this enum is only used in llama_params_fit_impl but needs to be defined outside of it to fix a Windows compilation issue
// enum to identify part of a layer for distributing its tensors:
enum common_layer_fraction_t {
    LAYER_FRACTION_NONE = 0, // nothing
    LAYER_FRACTION_ATTN = 1, // attention
    LAYER_FRACTION_UP   = 2, // attention + up
    LAYER_FRACTION_GATE = 3, // attention + up + gate
    LAYER_FRACTION_MOE  = 4, // everything but sparse MoE weights
};

class common_params_fit_exception : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

static std::vector<llama_device_memory_data> common_get_device_memory_data_impl(
        const char * path_model,
        const llama_model_params * mparams,
        const llama_context_params * cparams,
        std::vector<ggml_backend_dev_t> & devs,
        uint32_t & hp_ngl,
        uint32_t & hp_n_ctx_train,
        uint32_t & hp_n_expert,
        ggml_log_level log_level) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } original_logger;
        ggml_log_level min_level; // prints below this log level go to debug log
    };
    user_data_t ud;
    llama_log_get(&ud.original_logger.callback, &ud.original_logger.user_data);
    ud.min_level = log_level;

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        const ggml_log_level level_eff = level >= ud->min_level ? level : GGML_LOG_LEVEL_DEBUG;
        ud->original_logger.callback(level_eff, text, ud->original_logger.user_data);
    }, &ud);

    llama_model_params mparams_copy = *mparams;
    mparams_copy.no_alloc  = true;
    mparams_copy.load_mode = LLAMA_LOAD_MODE_NONE;

    llama_model * model = llama_model_load_from_file(path_model, mparams_copy);
    if (model == nullptr) {
        llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);
        throw std::runtime_error("failed to load model");
    }

    llama_context * ctx = llama_init_from_model(model, *cparams);
    if (ctx == nullptr) {
        llama_model_free(model);
        llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);
        throw std::runtime_error("failed to create llama_context from model");
    }

    const size_t nd = llama_model_n_devices(model);
    std::vector<llama_device_memory_data> ret(nd + 1);

    llama_memory_breakdown memory_breakdown = llama_get_memory_breakdown(ctx);

    for (const auto & [buft, mb] : memory_breakdown) {
        if (ggml_backend_buft_is_host(buft)) {
            ret.back().mb.model   += mb.model;
            ret.back().mb.context += mb.context;
            ret.back().mb.compute += mb.compute;
            continue;
        }

        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (!dev) {
            continue;
        }
        for (size_t i = 0; i < nd; i++) {
            if (dev == llama_model_get_device(model, i)) {
                ret[i].mb.model   += mb.model;
                ret[i].mb.context += mb.context;
                ret[i].mb.compute += mb.compute;
                break;
            }
        }
    }

    {
        ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu_dev == nullptr) {
            throw std::runtime_error("no CPU backend found");
        }
        size_t free;
        size_t total;
        ggml_backend_dev_memory(cpu_dev, &free, &total);

        // ggml's CPU device reports installed capacity for *both* free and
        // total, not what's actually free - on a host with other things
        // resident it silently overstates how much RAM is available. Prefer
        // /proc/meminfo's MemAvailable (already reclaimable page cache
        // included, so our own mmap'd model weights don't count against
        // themselves), falling back to the ggml figure where it can't be
        // read. Same fix as llama-model.cpp's mmap-prefetch decision and
        // server-context.cpp's prompt-cache sizing - keep these three in
        // sync if the approach ever changes.
        size_t avail = 0;
        {
            std::ifstream meminfo("/proc/meminfo");
            std::string key, unit;
            size_t value_kib = 0;
            while (meminfo >> key >> value_kib >> unit) {
                if (key == "MemAvailable:") {
                    avail = value_kib * 1024ULL;
                    break;
                }
            }
        }
        ret.back().free  = avail > 0 ? avail : free;
        ret.back().total = total;
    }
    for (size_t i = 0; i < nd; i++) {
        ggml_backend_dev_t dev = llama_model_get_device(model, i);

        size_t free;
        size_t total;
        ggml_backend_dev_memory(dev, &free, &total);

        // Some non-GPU accelerator backends, such as BLAS, report 0/0 and rely on
        // the host-memory fallback. For GPU-like backends, keep 0/0 so --fit does
        // not assign anything to a device with an unknown memory budget.
        if (free == 0 && total == 0) {
            const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                LOG_WRN("%s: device %s did not report memory; --fit will not use it\n",
                        __func__, ggml_backend_dev_name(dev));
            } else {
                free  = ret.back().free;
                total = ret.back().total;
            }
        }
        ret[i].free  = free;
        ret[i].total = total;
    }

    devs.clear();
    for (int i = 0; i < llama_model_n_devices(model); i++) {
        devs.push_back(llama_model_get_device(model, i));
    }

    hp_ngl         = llama_model_n_layer(model);
    if (mparams->load_mtp) {
        hp_ngl    += llama_model_n_layer_nextn(model);
    }
    hp_n_ctx_train = llama_model_n_ctx_train(model);
    hp_n_expert    = llama_model_n_expert(model);

    common_memory_breakdown_print(ctx);

    llama_free(ctx);
    llama_model_free(model);
    llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);

    return ret;
}

common_probe_context::~common_probe_context() {
    if (ctx) {
        llama_free(ctx);
        ctx = nullptr;
    }
    if (model) {
        llama_model_free(model);
        model = nullptr;
    }
}

void common_make_probe_context(
        const char * path_model,
        const llama_model_params * mparams,
        const llama_context_params * cparams,
        common_probe_context & out,
        ggml_log_level log_level) {
    struct probe_log_t {
        ggml_log_callback callback;
        void * user_data;
        ggml_log_level min_level;
    };
    probe_log_t pl;
    llama_log_get(&pl.callback, &pl.user_data);
    pl.min_level = log_level;

    // the probe is expected to be noisy (and to fail on some models); keep its
    // output out of the way exactly like common_get_device_memory_data does.
    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const probe_log_t * pl = (const probe_log_t *) user_data;
        const ggml_log_level level_eff = level >= pl->min_level ? level : GGML_LOG_LEVEL_DEBUG;
        pl->callback(level_eff, text, pl->user_data);
    }, &pl);

    llama_model_params mparams_copy = *mparams;
    mparams_copy.no_alloc  = true;
    mparams_copy.load_mode = LLAMA_LOAD_MODE_NONE;

    out.model = llama_model_load_from_file(path_model, mparams_copy);
    if (out.model != nullptr) {
        out.ctx = llama_init_from_model(out.model, *cparams);
        if (out.ctx == nullptr) {
            llama_model_free(out.model);
            out.model = nullptr;
        }
    }

    llama_log_set(pl.callback, pl.user_data);
}


common_device_memory_data_vec common_get_device_memory_data(
        const char * path_model,
        const llama_model_params * mparams,
        const llama_context_params * cparams,
        std::vector<ggml_backend_dev_t> & devs,
        uint32_t & hp_ngl,
        uint32_t & hp_n_ctx_train,
        uint32_t & hp_n_expert,
        ggml_log_level log_level) {
    std::vector<llama_device_memory_data> impl = common_get_device_memory_data_impl(
            path_model, mparams, cparams, devs, hp_ngl, hp_n_ctx_train, hp_n_expert, log_level);

    common_device_memory_data_vec ret(impl.size());
    for (size_t i = 0; i < impl.size(); i++) {
        ret[i].total   = impl[i].total;
        ret[i].free    = impl[i].free;
        ret[i].model   = impl[i].mb.model;
        ret[i].context = impl[i].mb.context;
        ret[i].compute = impl[i].mb.compute;
    }
    return ret;
}

// Per-expert-tensor byte sizes for a MoE model's FFN expert weights, scanned
// directly from GGUF tensor metadata (no weights loaded). Distinguishes the
// gate_up-fused layout (Gemma-4, GLM-5.2, DeepSeek2, Cohere2MoE, Qwen3.5MoE
// all use this) from the separate gate/up layout other architectures use,
// since moe-cache sizes a distinct VRAM pool per tensor *kind* and each must
// independently clear its own minimum-pool-size gate.
struct common_moe_cache_expert_sizes {
    long gate_up_or_gate_bytes = -1; // ffn_gate_up_exps (fused), or ffn_gate_exps (separate)
    long up_bytes               = -1; // ffn_up_exps, only set when the layout is NOT fused
    long down_bytes             = -1; // ffn_down_exps

    bool valid() const { return gate_up_or_gate_bytes > 0 && down_bytes > 0; }
    // conservative floor: both pools (gate_up-or-gate, and down; plus up if
    // separate) must independently clear moe-cache's `expert_size * 64`
    // minimum, so the budget needs to clear the sum of all found kinds.
    long combined_bytes() const {
        long sum = 0;
        if (gate_up_or_gate_bytes > 0) sum += gate_up_or_gate_bytes;
        if (up_bytes > 0)              sum += up_bytes;
        if (down_bytes > 0)            sum += down_bytes;
        return sum;
    }
};

static long moe_cache_tensor_expert_bytes(
        struct gguf_context * gctx, int64_t n_tensors, const char * pattern, int64_t n_expert) {
    if (n_expert <= 0) {
        return -1;
    }
    for (int64_t i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gctx, i);
        if (strstr(name, pattern)) {
            return (long)(gguf_get_tensor_size(gctx, i) / n_expert);
        }
    }
    return -1;
}

// scans one gguf file; returns a result with valid()==false if no routed-expert
// tensors were found in this specific file (common for later shards of a
// split GGUF that only hold dense layers, or vice versa)
static common_moe_cache_expert_sizes common_moe_cache_expert_sizes_in_file(
        const char * path, int64_t n_expert) {
    common_moe_cache_expert_sizes out;
    struct gguf_init_params ip = { /*no_alloc=*/ true, /*ctx=*/ nullptr };
    struct gguf_context * gctx = gguf_init_from_file(path, ip);
    if (!gctx) {
        return out;
    }
    const int64_t n_tensors = gguf_get_n_tensors(gctx);

    out.gate_up_or_gate_bytes = moe_cache_tensor_expert_bytes(gctx, n_tensors, "ffn_gate_up_exps", n_expert);
    if (out.gate_up_or_gate_bytes < 0) {
        out.gate_up_or_gate_bytes = moe_cache_tensor_expert_bytes(gctx, n_tensors, "ffn_gate_exps", n_expert);
        out.up_bytes              = moe_cache_tensor_expert_bytes(gctx, n_tensors, "ffn_up_exps", n_expert);
    }
    out.down_bytes = moe_cache_tensor_expert_bytes(gctx, n_tensors, "ffn_down_exps", n_expert);

    gguf_free(gctx);
    return out;
}

// handles split GGUFs (e.g. "-00001-of-00011.gguf") by scanning parts in
// order until expert tensors are found, since a MoE model's expert weights
// are not guaranteed to live in the first shard
static common_moe_cache_expert_sizes common_moe_cache_scan_expert_sizes(
        const char * path_model, int64_t n_expert) {
    common_moe_cache_expert_sizes result = common_moe_cache_expert_sizes_in_file(path_model, n_expert);
    if (result.valid()) {
        return result;
    }

    std::string path(path_model);
    const char * tag = strstr(path_model, "-00001-of-");
    if (!tag) {
        return result;
    }
    const int n_parts = atoi(tag + strlen("-00001-of-"));
    for (int part = 2; part <= n_parts && !result.valid(); part++) {
        std::string p(path_model);
        char rep[16];
        snprintf(rep, sizeof(rep), "-%05d-of-", part);
        p.replace(tag - path_model, strlen("-00001-of-"), rep);
        result = common_moe_cache_expert_sizes_in_file(p.c_str(), n_expert);
    }
    return result;
}

// Prints a recommended --moe-cache budget plus the GGML_CUDA_MOE_CACHE_*
// env vars needed for it to actually engage, computed from a real no-alloc
// device-memory probe (same one common_fit_print uses) plus a GGUF tensor
// metadata scan for the model's per-expert byte sizes. Static, one-time
// calculation - no running server needed, matches the moe-cache tuning
// knobs being fixed facts about the hardware+model pairing, not something
// that changes mid-session (see docs/moe-cache-colibri-notes.md for why
// this is scoped as static rather than dynamic).
void common_fit_print_moe_cache(
        const char * path_model,
        llama_model_params * mparams,
        llama_context_params * cparams) {
    std::vector<ggml_backend_dev_t> devs;
    uint32_t hp_ngl = 0;
    uint32_t hp_nct = 0;
    uint32_t hp_nex = 0;

    auto dmd = common_get_device_memory_data_impl(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, GGML_LOG_LEVEL_ERROR);
    GGML_ASSERT(dmd.size() == devs.size() + 1);

    if (hp_nex == 0) {
        printf("# not a MoE model (expert_count=0) - moe-cache does not apply\n");
        return;
    }

    common_moe_cache_expert_sizes sizes = common_moe_cache_scan_expert_sizes(path_model, hp_nex);
    if (!sizes.valid()) {
        printf("# WARNING: could not find routed-expert tensors in this GGUF's metadata\n"
               "#   (checked ffn_gate_up_exps, ffn_gate_exps/ffn_up_exps, ffn_down_exps)\n"
               "#   - unrecognized tensor naming for this architecture, no recommendation possible\n");
        return;
    }

    const long combined_bytes = sizes.combined_bytes();
    // matches moe-cache.cu's own `minimum_pool = expert_size * 64` gate, summed
    // across every distinct tensor kind found (each must independently clear it)
    const size_t minimum_needed_bytes =
        (size_t)((sizes.gate_up_or_gate_bytes > 0 ? sizes.gate_up_or_gate_bytes * 64 : 0) +
                 (sizes.up_bytes              > 0 ? sizes.up_bytes              * 64 : 0) +
                 (sizes.down_bytes            > 0 ? sizes.down_bytes            * 64 : 0));

    printf("# expert sizes: %s%ld KiB/expert",
        sizes.up_bytes > 0 ? "gate=" : "gate_up=", sizes.gate_up_or_gate_bytes / 1024);
    if (sizes.up_bytes > 0) {
        printf(", up=%ld KiB/expert", sizes.up_bytes / 1024);
    }
    printf(", down=%ld KiB/expert, combined=%ld KiB/expert\n", sizes.down_bytes / 1024, combined_bytes / 1024);

    if (devs.size() == 0) {
        printf("# no GPU devices detected - moe-cache requires at least one CUDA device\n");
        return;
    }

    // a modest, fixed safety margin for v1 - see docs/moe-cache-planner-scope.md's
    // "open question" note on tuning this policy with more real hardware data
    const size_t reserve_mib = 256;

    for (size_t id = 0; id < devs.size(); id++) {
        const long long expected_free = dmd[id].free - (long long) dmd[id].mb.total();
        const long long available = expected_free - (long long)(reserve_mib << 20);

        printf("\n# %s: free=%lldMiB total=%lldMiB projected_model+ctx+compute=%zuMiB expected_free_after_load=%lldMiB\n",
            ggml_backend_dev_name(devs[id]),
            (long long) dmd[id].free / 1024 / 1024, (long long) dmd[id].total / 1024 / 1024,
            dmd[id].mb.total() / 1024 / 1024, expected_free / 1024 / 1024);

        if (available <= 0 || (size_t) available < minimum_needed_bytes) {
            printf("# WARNING: %s has only ~%lldMiB available after a %zuMiB reserve, but needs at least "
                   "~%zuMiB for one cache slot per expert kind (64x the largest expert tensor).\n"
                   "#   moe-cache will stay dormant on this device with any budget. Free VRAM first:\n"
                   "#   lower -ngl / -ncmoe, use a smaller context, or reduce --parallel slots.\n",
                ggml_backend_dev_name(devs[id]), std::max<long long>(available, 0), reserve_mib,
                minimum_needed_bytes / 1024 / 1024);
            continue;
        }

        const long long budget_mib = available / 1024 / 1024;
        // absolute count, not a percentage: total expert count across all layers
        // isn't available from this pass without loading the full model, and an
        // unqualified percentage risks reading as a promise rather than a floor
        // (see docs/moe-cache-colibri-notes.md's routing-skew section on why real
        // hit rate depends on routing concentration, not just capacity)
        const long long cacheable_experts = combined_bytes > 0 ? available / combined_bytes : 0;

        printf("--moe-cache %lld\n", budget_mib);
        printf("GGML_CUDA_MOE_CACHE_MAX_BATCH=8\n");
        printf("GGML_CUDA_MOE_CACHE_RESERVE_MB=%zu\n", reserve_mib);
        printf("# ~%lld expert-slot-equivalents cacheable at this budget (combined gate/up/down per slot)\n", cacheable_experts);
        if (devs.size() == 1) {
            printf("# NOTE: 1 GPU detected - --moe-cache auto/on never engage on a single GPU regardless of\n"
                   "#   free VRAM (moe-cache.cu: automatic && devices < 2 forces dormant). The explicit\n"
                   "#   numeric budget above is required, not optional, on this hardware.\n");
        }
    }

    // --- placement preference: is full CPU-offload + cache likely better than
    // a static partial GPU placement? ---
    //
    // Adapted from leloch/moe-cache-pr's own common_moe_cache_prefers_cpu_moe(),
    // which lived in this file (commit e9de20cf1) before the final rework
    // removed it as "the unsafe auto-fit gguf scanning placement rule" - not
    // because the economics were wrong (their own measured numbers: a 754B
    // model went 19.2 vs 14.0 tok/s, a 397B model 33.3 vs 28.2 tok/s, full
    // CPU-offload+cache beating static partial placement both times), but
    // because a heuristic calibrated on 6 configs is risky to ship broadly to
    // unknown users. We don't have that constraint - this tool is being
    // validated against our own hardware as we go, starting with one
    // confirmed data point (RTX 3060 + Gemma-4-26B-A4B: full CPU-offload +
    // cache measured +54% over the CPU-offload-without-cache baseline). The
    // partial-GPU-placement-without-cache alternative has NOT yet been
    // measured on our hardware, so the thresholds below are leloch's
    // calibration, not yet independently re-confirmed by us - see
    // docs/moe-cache-planner-scope.md before trusting this on new hardware.
    {
        size_t model_bytes = 0;
        {
            const char * tag = strstr(path_model, "-00001-of-");
            int n_parts = 1;
            if (tag) {
                n_parts = atoi(tag + strlen("-00001-of-"));
            }
            for (int part = 1; part <= (n_parts > 0 ? n_parts : 1); part++) {
                std::string p(path_model);
                if (tag) {
                    char rep[16];
                    snprintf(rep, sizeof(rep), "-%05d-of-", part);
                    p.replace(tag - path_model, strlen("-00001-of-"), rep);
                }
                struct stat st;
                if (stat(p.c_str(), &st) == 0) {
                    model_bytes += (size_t) st.st_size;
                }
            }
        }

        size_t usable_vram = 0;
        for (size_t id = 0; id < devs.size(); id++) {
            usable_vram += (size_t) std::max<int64_t>(dmd[id].free, 0);
        }

        const double spill_ratio = usable_vram > 0 ? (double) model_bytes / (double) usable_vram : 99.0;
        const long largest_expert_kib =
            std::max({sizes.gate_up_or_gate_bytes, sizes.up_bytes, sizes.down_bytes}) / 1024;

        printf("\n# --- placement preference (leloch's calibration, not yet independently\n"
               "#     re-validated on this hardware - see docs/moe-cache-planner-scope.md) ---\n");
        printf("# model_bytes=%zuMiB usable_vram=%zuMiB spill_ratio=%.2f largest_expert=%ldKiB devices=%zu\n",
            model_bytes / 1024 / 1024, usable_vram / 1024 / 1024, spill_ratio, largest_expert_kib, devs.size());

        bool prefer_cpu_moe;
        const char * reason;
        if (spill_ratio < 1.8) {
            prefer_cpu_moe = false;
            reason = "spill_ratio < 1.8: model mostly fits, static placement should already do well";
        } else if (devs.size() < 2 && largest_expert_kib < 2048) {
            prefer_cpu_moe = false;
            reason = "single device + small experts: leloch's data showed per-device dispatch-chain\n"
                      "#   serialization costing 6-18% on one-GPU configs with small experts";
        } else {
            prefer_cpu_moe = true;
            reason = "large spill + (multi-device or big-enough experts): leloch's measured configs\n"
                      "#   favored full CPU-offload + cache here (754B: 19.2 vs 14.0 t/s; 397B: 33.3 vs 28.2 t/s)";
        }

        printf("# RECOMMENDATION: %s full CPU-offload (--cpu-moe / high --n-cpu-moe) + moe-cache\n",
            prefer_cpu_moe ? "PREFER" : "DO NOT prefer");
        printf("#   reason: %s\n", reason);
        if (!prefer_cpu_moe) {
            printf("#   -> try --fit-target / a partial --n-cpu-moe placement instead, and compare\n"
                   "#      against the full-offload+cache config above empirically before trusting either.\n");
        }
    }
}

static void common_params_fit_impl(
        const char * path_model, struct llama_model_params * mparams, struct llama_context_params * cparams,
        float * tensor_split, struct llama_model_tensor_buft_override * tensor_buft_overrides,
        size_t * margins_s, uint32_t n_ctx_min, enum ggml_log_level log_level) {
    if (mparams->split_mode == LLAMA_SPLIT_MODE_TENSOR) {
        throw common_params_fit_exception("llama_params_fit is not implemented for SPLIT_MODE_TENSOR, abort");
    }
    constexpr int64_t MiB = 1024*1024;
    typedef std::vector<llama_device_memory_data> dmds_t;
    const llama_model_params default_mparams = llama_model_default_params();

    std::vector<ggml_backend_dev_t> devs;
    uint32_t hp_ngl = 0; // hparams.n_gpu_layers
    uint32_t hp_nct = 0; // hparams.n_ctx_train
    uint32_t hp_nex = 0; // hparams.n_expert

    // step 0: an explicitly-requested context size can be so large that even the *measurement* pass
    // below (which creates a real llama_context to simulate allocations) hard-aborts the whole process
    // on CUDA OOM before any of the reduction logic further down gets a chance to run - ggml aborts hard
    // on CUDA errors, which no try/catch can intercept, so this has to be prevented rather than caught.
    // The n_ctx==0 (auto) path is already safe because unset context resolves to the model's own trained
    // context internally before anything is allocated. A large *explicit* request has no such protection,
    // and - confirmed by testing - even "the model's own trained context" is not a safe stand-in ceiling
    // by itself (a 256K-trained model can still exceed a 12GB card just to *measure* that size), and a
    // single two-point extrapolation from tiny probes turned out to be too crude too (the true cost-per-
    // token isn't uniform all the way from a few thousand tokens up to hundreds of thousands, so a rate
    // measured near the bottom under- or overshoots badly near the top). Search by doubling instead: each
    // step is only ever 2x the size of the last one that was *actually measured* to fit, so the risk of
    // any single probe overshooting stays bounded, and the final number is a real, checked measurement,
    // not a guess extrapolated from a distant sample. Scoped to the common single-device (or host-only)
    // case, where the free-memory budget is unambiguous; multi-device setups fall back to the one size
    // that's unconditionally safe everywhere.
    //
    // The requested context size is the priority, not n_seq_max (concurrent slots): the doubling search
    // below always runs at n_seq_max=1 first, since a single slot is the most permissive case for fitting
    // the full requested context. Only if the request doesn't fit even at one slot does context itself
    // get reduced (as a last resort); if it *does* fit at one slot, n_seq_max is grown back up afterwards
    // (see the block below this one) to recover as much concurrency as the now-confirmed-safe context
    // size leaves room for - not the other way around, where a fixed slot count silently eats into
    // context the user explicitly asked for.
    const uint32_t requested_n_ctx = cparams->n_ctx;
    const uint32_t requested_n_seq_max = cparams->n_seq_max;
    bool ctx_clamped_by_fit = false;
    bool ctx_kept_full_by_fit = false; // true: only n_seq_max was reduced, context itself is untouched
    // This whole search deliberately probes right up to the edge of what margins_s[0] allows, which is
    // fine everywhere else in this file (those paths only ever *reduce* usage below a value that already
    // measured safe). Here it is not fine: this is the one search whose final answer gets used to load
    // the *real* model with real weights, and two gaps confirmed by direct testing both eat into that
    // same margin. First, the probe measures a no_alloc model with weights never actually read from
    // disk - real loading measured ~400 MiB more in-use at steady state than the probe predicted, on top
    // of margins_s[0] already being fully spent by construction. Second, CUDA graph capture allocates
    // driver-side graph state lazily on the first real decode of each new batch *shape*, entirely outside
    // of what this probe measures (it only ever constructs a context, never decodes) - confirmed by
    // testing: a context --fit measured as safe at load time still hard-aborted the whole process
    // (uncatchable CUDA OOM, not a request that can be retried) the first time a real prompt of a new
    // shape triggered a fresh graph capture, well after startup. Neither gap has a precise size - it
    // depends on the graph's node count, which depends on the model/backend/speculative config - so
    // rather than guess a number, this search targets extra headroom on top of the configured margin
    // specifically here, where a wrong answer means a hard crash instead of a merely suboptimal one.
    // GGML_SCHED_PREFETCH_EXPERTS's double-buffer (ggml-backend.cpp) allocates
    // lazily, on first use, exactly like the CUDA-graph-capture gap above - a
    // fit search that approves a placement without knowing this buffer is
    // coming can leave too little room for it to grow into later, on a real
    // prompt of a new shape, well after this probe already ran. Measured
    // directly (llama-bench, pp2048, Gemma-4-26B-A4B, -ncmoe 15): 820 MiB net
    // extra peak VRAM with the feature on vs off (10402 -> 11222 MiB), for
    // its own real +13.2% pp2048 win. Reserved here, rounded up for margin,
    // ONLY when the env var will actually enable it - checked with the exact
    // same condition ggml-backend.cpp itself uses (set AND atoi() > 0), so a
    // launch that never turns this on pays nothing extra.
    int64_t prefetch_margin = 0;
    {
        const char * prefetch_env = getenv("GGML_SCHED_PREFETCH_EXPERTS");
        if (prefetch_env && atoi(prefetch_env) > 0) {
            prefetch_margin = 1024ll * 1024 * 1024; // 1 GiB, rounded up from the measured 820 MiB
        }
    }
    const int64_t step0_margin = 3 * (int64_t) margins_s[0] + prefetch_margin;
    if (requested_n_ctx > 4 * n_ctx_min) {
        llama_context_params cparams_probe = *cparams;
        cparams_probe.n_ctx = n_ctx_min;
        cparams_probe.n_seq_max = 1;

        std::vector<ggml_backend_dev_t> probe_devs;
        uint32_t probe_hp_ngl = 0, probe_hp_nct = 0, probe_hp_nex = 0;
        try {
            const dmds_t dmds_first = common_get_device_memory_data_impl(
                path_model, mparams, &cparams_probe, probe_devs, probe_hp_ngl, probe_hp_nct, probe_hp_nex, log_level);

            if (probe_devs.size() > 1) {
                LOG_WRN("%s: requested context of %" PRIu32 " is large and multiple devices are in use - "
                        "probing at the minimum context size of %" PRIu32 " first instead of risking a hard "
                        "crash on the raw request\n", __func__, requested_n_ctx, n_ctx_min);
                cparams->n_ctx = n_ctx_min;
                cparams->n_seq_max = 1;
                ctx_clamped_by_fit = true;
            } else if (probe_hp_nct > 0 && (int64_t) dmds_first[0].mb.total() + step0_margin > (int64_t) dmds_first[0].free) {
                // does not even fit at the minimum context size with a single slot (e.g. another process -
                // an embedding server, another llama.cpp instance, anything else sharing the device - is
                // already holding most of the device's memory). Step 1/2 below only actively reduces
                // context for the n_ctx==0 (auto) path; for an *explicit* request like this one it just
                // leaves cparams->n_ctx untouched (fit.cpp's "context size set by user -> no change"
                // branch), which would carry the full, already-proven-unsafe raw request all the way to
                // real model loading. Clamp here instead of relying on that fallthrough.
                cparams->n_ctx = n_ctx_min;
                cparams->n_seq_max = 1;
                ctx_clamped_by_fit = true;
            } else {
                const uint32_t ceiling = std::min(requested_n_ctx, probe_hp_nct);
                uint32_t safe_n_ctx = n_ctx_min;
                uint32_t next_probe = std::min(2 * n_ctx_min, ceiling);
                while (true) {
                    llama_context_params cparams_step = *cparams;
                    cparams_step.n_ctx = next_probe;
                    cparams_step.n_seq_max = 1;
                    const dmds_t dmds_step = common_get_device_memory_data_impl(
                        path_model, mparams, &cparams_step, probe_devs, probe_hp_ngl, probe_hp_nct, probe_hp_nex, log_level);
                    const bool fits = (int64_t) dmds_step[0].mb.total() + step0_margin <= (int64_t) dmds_step[0].free;
                    if (!fits) {
                        break;
                    }
                    safe_n_ctx = next_probe;
                    if (safe_n_ctx >= ceiling) {
                        break;
                    }
                    next_probe = std::min(2 * safe_n_ctx, ceiling);
                }
                if (safe_n_ctx < requested_n_ctx) {
                    LOG_WRN("%s: requested context of %" PRIu32 " does not fit even with a single slot "
                            "(model's trained context is %" PRIu32 ") - probed and confirmed %" PRIu32
                            " fits instead of risking a hard crash on the raw request\n",
                            __func__, requested_n_ctx, probe_hp_nct, safe_n_ctx);
                    cparams->n_ctx = safe_n_ctx;
                    cparams->n_seq_max = 1;
                    ctx_clamped_by_fit = true;
                } else if (requested_n_seq_max > 1) {
                    // the full requested context fits at one slot, and this size has already been
                    // measured safe by the doubling search above - now recover as much concurrency as
                    // actually fits at that *fixed* context size, largest first. Each remaining probe is
                    // a small, bounded increment (one more slot at an already-proven-safe context), not
                    // a second unbounded search, so it doesn't reintroduce the original crash risk.
                    uint32_t best_n_seq_max = 1;
                    for (uint32_t candidate = requested_n_seq_max; candidate >= 2; candidate--) {
                        llama_context_params cparams_slots = *cparams;
                        cparams_slots.n_ctx = requested_n_ctx;
                        cparams_slots.n_seq_max = candidate;
                        const dmds_t dmds_slots = common_get_device_memory_data_impl(
                            path_model, mparams, &cparams_slots, probe_devs, probe_hp_ngl, probe_hp_nct, probe_hp_nex, log_level);
                        const bool fits = (int64_t) dmds_slots[0].mb.total() + step0_margin <= (int64_t) dmds_slots[0].free;
                        if (fits) {
                            best_n_seq_max = candidate;
                            break;
                        }
                    }
                    cparams->n_ctx = requested_n_ctx;
                    cparams->n_seq_max = best_n_seq_max;
                    if (best_n_seq_max < requested_n_seq_max) {
                        LOG_WRN("%s: kept the full requested context of %" PRIu32 " and reduced concurrent "
                                "slots from %" PRIu32 " to %" PRIu32 " instead - not the other way around\n",
                                __func__, requested_n_ctx, requested_n_seq_max, best_n_seq_max);
                    }
                    ctx_clamped_by_fit = true; // skip the (context-shrinking) reduction search below - this path already fits
                    ctx_kept_full_by_fit = true; // ...and if remeasurement still disagrees, degrade slots first, not context
                }
            }
        } catch (const std::runtime_error &) {
            // even the small safe probes failed for some unrelated reason (e.g. bad model path) -
            // let the normal step 1 measurement below surface that error the usual way
        }
    }

    // step 1: get data for default parameters and check whether any changes are necessary in the first place

    LOG_TRC("%s: getting device memory data for initial parameters:\n", __func__);
    const dmds_t dmds_full = common_get_device_memory_data_impl(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);
    const size_t nd = devs.size(); // number of devices

    std::vector<int64_t> margins; // this function uses int64_t rather than size_t for memory sizes to more conveniently handle deficits
    margins.reserve(nd);
    if (nd == 0) {
        margins.push_back(margins_s[0]);
    } else {
        for (size_t id = 0; id < nd; id++) {
            margins.push_back(margins_s[id]);
        }
    }

    std::vector<std::string> dev_names;
    {
        dev_names.reserve(nd);
        size_t max_length = 0;
        for (const auto & dev : devs) {
            std::string name = ggml_backend_dev_name(dev);
            name += " (";
            name += ggml_backend_dev_description(dev);
            name += ")";
            dev_names.push_back(name);
            max_length = std::max(max_length, name.length());
        }
        for (std::string & dn : dev_names) {
            dn.insert(dn.end(), max_length - dn.length(), ' ');
        }
    }

    int64_t sum_free            = 0;
    int64_t sum_projected_free  = 0;
    int64_t sum_projected_used  = 0;
    int64_t sum_projected_model = 0;
    std::vector<int64_t> projected_free_per_device;
    projected_free_per_device.reserve(nd);

    if (nd == 0) {
        sum_projected_used = dmds_full.back().mb.total();
        sum_free           = dmds_full.back().total;
        sum_projected_free = sum_free - sum_projected_used;
        LOG_TRC("%s: projected to use %" PRId64 " MiB of host memory vs. %" PRId64 " MiB of total host memory\n",
            __func__, sum_projected_used/MiB, sum_free/MiB);
        if (sum_projected_free >= margins[0]) {
            LOG_TRC("%s: will leave %" PRId64 " >= %" PRId64 " MiB of system memory, no changes needed\n",
                __func__, sum_projected_free/MiB, margins[0]/MiB);
            return;
        }
    } else {
        if (nd > 1) {
            LOG_TRC("%s: projected memory use with initial parameters [MiB]:\n", __func__);
        }
        for (size_t id = 0; id < nd; id++) {
            const llama_device_memory_data & dmd = dmds_full[id];

            const int64_t projected_used = dmd.mb.total();
            const int64_t projected_free = dmd.free - projected_used;
            projected_free_per_device.push_back(projected_free);

            sum_free            += dmd.free;
            sum_projected_used  += projected_used;
            sum_projected_free  += projected_free;
            sum_projected_model += dmd.mb.model;

            if (nd > 1) {
                LOG_TRC("%s:   - %s: %6" PRId64 " total, %6" PRId64 " used, %6" PRId64 " free vs. target of %6" PRId64 "\n",
                    __func__, dev_names[id].c_str(), dmd.total/MiB, projected_used/MiB, projected_free/MiB, margins[id]/MiB);
            }
        }
        assert(sum_free >= 0 && sum_projected_used >= 0);
        LOG_TRC("%s: projected to use %" PRId64 " MiB of device memory vs. %" PRId64 " MiB of free device memory\n",
            __func__, sum_projected_used/MiB, sum_free/MiB);
        if (nd == 1) {
            if (projected_free_per_device[0] >= margins[0]) {
                LOG_TRC("%s: will leave %" PRId64 " >= %" PRId64 " MiB of free device memory, no changes needed\n",
                    __func__, projected_free_per_device[0]/MiB, margins[0]/MiB);
                return;
            }
        } else {
            bool changes_needed = false;
            for (size_t id = 0; id < nd; id++) {
                if (projected_free_per_device[id] < margins[id]) {
                    changes_needed = true;
                    break;
                }
            }
            if (!changes_needed) {
                LOG_TRC("%s: targets for free memory can be met on all devices, no changes needed\n", __func__);
                return;
            }
        }
    }

    // step 2: try reducing memory use by reducing the context size

    {
        int64_t global_surplus = sum_projected_free;
        if (nd == 0) {
            global_surplus -= margins[0];
        } else {
            for (size_t id = 0; id < nd; id++) {
                global_surplus -= margins[id];
            }
        }
        if (global_surplus < 0) {
            if (nd <= 1) {
                LOG_TRC("%s: cannot meet free memory target of %" PRId64 " MiB, need to reduce device memory by %" PRId64 " MiB\n",
                    __func__, margins[0]/MiB, -global_surplus/MiB);
            } else {
                LOG_TRC(
                    "%s: cannot meet free memory targets on all devices, need to use %" PRId64 " MiB less in total\n",
                    __func__, -global_surplus/MiB);
            }
            if (ctx_clamped_by_fit) {
                // step 0 already searched for and validated a configuration that fits, with the same
                // margin required here - reaching this point despite that means the real measurement
                // below came out slightly different from the last probe (e.g. driver-level allocator
                // variance right at the boundary step 0 deliberately searched right up to). The
                // interpolation math below assumes dmds_full was measured at hp_nct, which is no longer
                // true once step 0 has touched things, so reusing it here would produce a bogus (likely
                // far too large) result rather than a merely-suboptimal one.
                if (ctx_kept_full_by_fit && cparams->n_seq_max > 1) {
                    // context itself is the priority and was already kept at the full request - degrade
                    // slots first, all the way to one if needed, before ever touching context.
                    LOG_WRN("%s: %" PRIu32 " concurrent slots at the full requested context of %" PRIu32
                            " did not leave the required margin on remeasurement - falling back to a "
                            "single slot at the same (unreduced) context instead\n",
                            __func__, cparams->n_seq_max, cparams->n_ctx);
                    cparams->n_seq_max = 1;
                    return;
                }
                // context needs to shrink - fall back to the one size every other path in this function
                // already trusts unconditionally.
                LOG_WRN("%s: context size of %" PRIu32 " (found safe by step 0) did not leave the required "
                        "margin on remeasurement - falling back to the minimum context size of %" PRIu32 "\n",
                        __func__, cparams->n_ctx, n_ctx_min);
                cparams->n_ctx = n_ctx_min;
                cparams->n_seq_max = 1;
                if (nd <= 1) {
                    return;
                }
                // nd > 1: fall through to step 3 below, same as the n_ctx==0 path does after reducing
                // context - layer placement across devices still needs to run using this context size.
            }
            if (cparams->n_ctx == 0) {
                if (hp_nct > n_ctx_min) {
                    int64_t sum_used_target = sum_free;
                    if (nd == 0) {
                        sum_used_target -= margins[0];
                    } else {
                        for (size_t id = 0; id < nd; id++) {
                            sum_used_target -= margins[id];
                        }
                    }
                    if (nd > 1) {
                        // for multiple devices we need to be more conservative in terms of how much context we think can fit:
                        //   - for dense models only whole layers can be assigned to devices
                        //   - for MoE models only whole tensors can be assigned to devices, which we estimate to be <= 1/3 of a layer
                        //   - on average we expect a waste of 0.5 layers/tensors per device
                        //   - use slightly more than the expected average for nd devices to be safe
                        const int64_t model_per_layer = sum_projected_model / std::min(uint32_t(mparams->n_gpu_layers), hp_ngl);
                        sum_used_target -= (nd + 1) * model_per_layer / (hp_nex == 0 ? 2 : 6);
                    }

                    int64_t sum_projected_used_min_ctx = 0;
                    cparams->n_ctx = n_ctx_min;
                    const dmds_t dmds_min_ctx = common_get_device_memory_data_impl(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);
                    if (nd == 0) {
                        sum_projected_used_min_ctx = dmds_min_ctx.back().mb.total();
                    } else {
                        for (size_t id = 0; id < nd; id++) {
                            sum_projected_used_min_ctx += dmds_min_ctx[id].mb.total();
                        }
                    }
                    if (sum_used_target > sum_projected_used_min_ctx) {
                        // linear interpolation between minimum and maximum context size:
                        cparams->n_ctx += (hp_nct - n_ctx_min) * (sum_used_target - sum_projected_used_min_ctx)
                            / (sum_projected_used - sum_projected_used_min_ctx);
                        cparams->n_ctx = std::max(cparams->n_ctx - cparams->n_ctx % 256, n_ctx_min); // round down context for CUDA backend

                        const int64_t bytes_per_ctx = (sum_projected_used - sum_projected_used_min_ctx) / (hp_nct - n_ctx_min);
                        const int64_t memory_reduction = (hp_nct - cparams->n_ctx) * bytes_per_ctx;
                        LOG_TRC("%s: context size reduced from %" PRIu32 " to %" PRIu32 " -> need %" PRId64 " MiB less memory in total\n",
                            __func__, hp_nct, cparams->n_ctx, memory_reduction/MiB);
                        if (nd <= 1) {
                            LOG_TRC("%s: entire model can be fit by reducing context\n", __func__);
                            return;
                        }
                        LOG_TRC("%s: entire model should be fit across devices by reducing context\n", __func__);
                    } else {
                        const int64_t memory_reduction = sum_projected_used - sum_projected_used_min_ctx;
                        LOG_TRC("%s: context size reduced from %" PRIu32 " to %" PRIu32 " -> need %" PRId64 " MiB less memory in total\n",
                            __func__, hp_nct, cparams->n_ctx, memory_reduction/MiB);
                    }
                } else {
                    if (n_ctx_min == UINT32_MAX) {
                        LOG_TRC("%s: user has requested full context size of %" PRIu32 " -> no change\n", __func__, hp_nct);
                    } else {
                        LOG_TRC("%s: default model context size is %" PRIu32 " which is <= the min. context size of %" PRIu32 " -> no change\n",
                            __func__, hp_nct, n_ctx_min);
                    }
                }
            } else {
                LOG_TRC("%s: context size set by user to %" PRIu32 " -> no change\n", __func__, cparams->n_ctx);
            }
        }
    }
    if (nd == 0) {
        throw common_params_fit_exception("was unable to fit model into system memory by reducing context, abort");
    }

    if (mparams->n_gpu_layers != default_mparams.n_gpu_layers) {
        throw common_params_fit_exception("n_gpu_layers already set by user to " + std::to_string(mparams->n_gpu_layers) + ", abort");
    }
    if (nd > 1) {
        if (!tensor_split) {
            throw common_params_fit_exception("did not provide a buffer to write the tensor_split to, abort");
        }
        if (mparams->tensor_split) {
            for (size_t id = 0; id < nd; id++) {
                if (mparams->tensor_split[id] != 0.0f) {
                    throw common_params_fit_exception("model_params::tensor_split already set by user, abort");
                }
            }
        }
        if (mparams->split_mode == LLAMA_SPLIT_MODE_ROW) {
            throw common_params_fit_exception("changing weight allocation for LLAMA_SPLIT_MODE_ROW not implemented, abort");
        }
    }
    if (!tensor_buft_overrides) {
        throw common_params_fit_exception("did not provide buffer to set tensor_buft_overrides, abort");
    }
    if (mparams->tensor_buft_overrides && (mparams->tensor_buft_overrides->pattern || mparams->tensor_buft_overrides->buft)) {
        throw common_params_fit_exception("model_params::tensor_buft_overrides already set by user, abort");
    }

    // step 3: iteratively fill the back to front with "dense" layers
    //   - for a dense model simply fill full layers, giving each device a contiguous slice of the model
    //   - for a MoE model, same as dense model but with all MoE tensors in system memory

    // utility function that returns a static C string matching the tensors for a specific layer index and layer fraction:
    auto get_overflow_pattern = [&](const size_t il, const common_layer_fraction_t lf) -> const char * {
        constexpr size_t n_strings = 1000;
        if (il >= n_strings) {
            throw std::runtime_error("at most " + std::to_string(n_strings) + " model layers are supported");
        }
        switch (lf) {
            case LAYER_FRACTION_ATTN: {
                static std::array<std::string, n_strings> patterns;
                if (patterns[il].empty()) {
                    patterns[il] = "blk\\." + std::to_string(il) + "\\.ffn_(gate|up|gate_up|down).*";
                }
                return patterns[il].c_str();
            }
            case LAYER_FRACTION_UP: {
                static std::array<std::string, n_strings> patterns;
                if (patterns[il].empty()) {
                    patterns[il] = "blk\\." + std::to_string(il) + "\\.ffn_(gate|gate_up|down).*";
                }
                return patterns[il].c_str();
            }
            case LAYER_FRACTION_GATE: {
                static std::array<std::string, n_strings> patterns;
                if (patterns[il].empty()) {
                    patterns[il] = "blk\\." + std::to_string(il) + "\\.ffn_down.*";
                }
                return patterns[il].c_str();
            }
            case LAYER_FRACTION_MOE: {
                static std::array<std::string, n_strings> patterns;
                if (patterns[il].empty()) {
                    patterns[il] = "blk\\." + std::to_string(il) + "\\.ffn_(up|down|gate_up|gate)_(ch|)exps";
                }
                return patterns[il].c_str();
            }
            default:
                GGML_ABORT("fatal error");
        }
    };

    struct ngl_t {
        uint32_t n_layer = 0; // number of total layers
        uint32_t n_part  = 0; // number of partial layers, <= n_layer

        // for the first partial layer varying parts can overflow, all further layers use LAYER_FRACTION_MOE:
        common_layer_fraction_t overflow_type = LAYER_FRACTION_MOE;

        uint32_t n_full() const {
            assert(n_layer >= n_part);
            return n_layer - n_part;
        }
    };

    const size_t ntbo = llama_max_tensor_buft_overrides();

    // utility function to set n_gpu_layers and tensor_split
    auto set_ngl_tensor_split_tbo = [&](
            const std::vector<ngl_t> & ngl_per_device,
            const std::vector<ggml_backend_buffer_type_t> & overflow_bufts,
            llama_model_params & mparams) {
        mparams.n_gpu_layers = 0;
        for (size_t id = 0; id < nd; id++) {
            mparams.n_gpu_layers += ngl_per_device[id].n_layer;
            if (nd > 1) {
                tensor_split[id] = ngl_per_device[id].n_layer;
            }
        }
        assert(uint32_t(mparams.n_gpu_layers) <= hp_ngl + 1);
        uint32_t il0 = hp_ngl + 1 - mparams.n_gpu_layers; // start index for tensor buft overrides

        mparams.tensor_split = tensor_split;

        size_t itbo = 0;
        for (size_t id = 0; id < nd; id++) {
            il0 += ngl_per_device[id].n_full();
            for (uint32_t il = il0; il < il0 + ngl_per_device[id].n_part; il++) {
                if (itbo + 1 >= ntbo) {
                    tensor_buft_overrides[itbo].pattern = nullptr;
                    tensor_buft_overrides[itbo].buft    = nullptr;
                    itbo++;
                    mparams.tensor_buft_overrides = tensor_buft_overrides;
                    throw common_params_fit_exception("llama_max_tensor_buft_overrides() == "
                        + std::to_string(ntbo) + " is insufficient for model");
                }
                tensor_buft_overrides[itbo].pattern = get_overflow_pattern(il, il == il0 ? ngl_per_device[id].overflow_type : LAYER_FRACTION_MOE);
                tensor_buft_overrides[itbo].buft = il == il0 ? overflow_bufts[id] : ggml_backend_cpu_buffer_type();
                itbo++;
            }
            il0 += ngl_per_device[id].n_part;
        }
        tensor_buft_overrides[itbo].pattern = nullptr;
        tensor_buft_overrides[itbo].buft    = nullptr;
        itbo++;
        mparams.tensor_buft_overrides = tensor_buft_overrides;
    };

    // utility function that returns the memory use per device for given numbers of layers per device
    auto get_memory_for_layers = [&](
            const char * func_name,
            const std::vector<ngl_t> & ngl_per_device,
            const std::vector<ggml_backend_buffer_type_t> & overflow_bufts) -> std::vector<int64_t> {
        llama_model_params mparams_copy = *mparams;
        set_ngl_tensor_split_tbo(ngl_per_device, overflow_bufts, mparams_copy);

        const dmds_t dmd_nl = common_get_device_memory_data_impl(
            path_model, &mparams_copy, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);

        LOG_TRC("%s: memory for test allocation by device:\n", func_name);
        for (size_t id = 0; id < nd; id++) {
            const ngl_t & n = ngl_per_device[id];
            LOG_TRC(
                "%s: id=%zu, n_layer=%2" PRIu32 ", n_part=%2" PRIu32 ", overflow_type=%d, mem=%6" PRId64 " MiB\n",
                func_name, id, n.n_layer, n.n_part, int(n.overflow_type), dmd_nl[id].mb.total()/MiB);
        }

        std::vector<int64_t> ret;
        ret.reserve(nd);
        for (size_t id = 0; id < nd; id++) {
            ret.push_back(dmd_nl[id].mb.total());
        }
        return ret;
    };

    int64_t global_surplus_cpu_moe = 0;
    if (hp_nex > 0) {
        const static std::string pattern_moe_all = "blk\\.\\d+\\.ffn_(up|down|gate_up|gate)_(ch|)exps"; // matches all MoE tensors
        ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
        tensor_buft_overrides[0] = {pattern_moe_all.c_str(), cpu_buft};
        tensor_buft_overrides[1] = {nullptr, nullptr};
        mparams->tensor_buft_overrides = tensor_buft_overrides;

        LOG_TRC("%s: getting device memory data with all MoE tensors moved to system memory:\n", __func__);
        const dmds_t dmds_cpu_moe = common_get_device_memory_data_impl(
            path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);

        for (size_t id = 0; id < nd; id++) {
            global_surplus_cpu_moe += dmds_cpu_moe[id].free;
            global_surplus_cpu_moe -= int64_t(dmds_cpu_moe[id].mb.total()) + margins[id];
        }

        if (global_surplus_cpu_moe > 0) {
            LOG_TRC("%s: with only dense weights in device memory there is a total surplus of %" PRId64 " MiB\n",
                __func__, global_surplus_cpu_moe/MiB);
        } else {
            LOG_TRC("%s: with only dense weights in device memory there is still a total deficit of %" PRId64 " MiB\n",
                __func__, -global_surplus_cpu_moe/MiB);
        }

        // reset
        tensor_buft_overrides[0] = {nullptr, nullptr};
        mparams->tensor_buft_overrides = tensor_buft_overrides;
    }

    std::vector<int64_t> targets; // maximum acceptable memory use per device
    targets.reserve(nd);
    for (size_t id = 0; id < nd; id++) {
        targets.push_back(dmds_full[id].free - margins[id]);
        LOG_TRC("%s: id=%zu, target=%" PRId64 " MiB\n", __func__, id, targets[id]/MiB);
    }

    std::vector<ggml_backend_buffer_type_t> overflow_bufts; // which bufts the first partial layer of a device overflows to:
    overflow_bufts.reserve(nd);
    for (size_t id = 0; id < nd; id++) {
        overflow_bufts.push_back(ggml_backend_cpu_buffer_type());
    }

    std::vector<ngl_t> ngl_per_device(nd);
    std::vector<int64_t> mem = get_memory_for_layers(__func__, ngl_per_device, overflow_bufts);

    // optimize the number of layers per device using the method of false position:
    //   - ngl_per_device has 0 layers for each device, lower bound
    //   - try a "high" configuration where a device is given all unassigned layers
    //   - interpolate the memory use / layer between low and high linearly to get a guess where it meets our target
    //   - check memory use of our guess, replace either the low or high bound
    //   - once we only have a difference of a single layer, stop and return the lower bound that just barely still fits
    //   - the last device has the output layer, which cannot be a partial layer
    if (hp_nex == 0) {
        LOG_TRC("%s: filling dense layers back-to-front:\n", __func__);
    } else {
        LOG_TRC("%s: filling dense-only layers back-to-front:\n", __func__);
    }
    for (int id = nd - 1; id >= 0; id--) {
        uint32_t n_unassigned = hp_ngl + 1;
        for (size_t jd = id + 1; jd < nd; ++jd) {
            assert(n_unassigned >= ngl_per_device[jd].n_layer);
            n_unassigned -= ngl_per_device[jd].n_layer;
        }

        std::vector<ngl_t> ngl_per_device_high = ngl_per_device;
        ngl_per_device_high[id].n_layer = n_unassigned;
        if (hp_nex > 0) {
            ngl_per_device_high[id].n_part = size_t(id) < nd - 1 ? ngl_per_device_high[id].n_layer : ngl_per_device_high[id].n_layer - 1;
        }
        if (ngl_per_device_high[id].n_layer > 0) {
            std::vector<int64_t> mem_high = get_memory_for_layers(__func__, ngl_per_device_high, overflow_bufts);
            if (mem_high[id] > targets[id]) {
                assert(ngl_per_device_high[id].n_layer > ngl_per_device[id].n_layer);
                uint32_t delta = ngl_per_device_high[id].n_layer - ngl_per_device[id].n_layer;
                LOG_TRC("%s: start filling device %" PRIu32 ", delta=%" PRIu32 "\n", __func__, id, delta);
                while (delta > 1) {
                    uint32_t step_size = int64_t(delta) * (targets[id] - mem[id]) / (mem_high[id] - mem[id]);
                    step_size = std::max(step_size, uint32_t(1));
                    step_size = std::min(step_size, delta - 1);

                    std::vector<ngl_t> ngl_per_device_test = ngl_per_device;
                    ngl_per_device_test[id].n_layer += step_size;
                    if (hp_nex) {
                        ngl_per_device_test[id].n_part += size_t(id) == nd - 1 && ngl_per_device_test[id].n_part == 0 ?
                            step_size - 1 : step_size; // the first layer is the output layer which must always be full
                    }
                    const std::vector<int64_t> mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts);

                    if (mem_test[id] <= targets[id]) {
                        ngl_per_device = ngl_per_device_test;
                        mem            = mem_test;
                        LOG_TRC("%s: set ngl_per_device[%d].n_layer=%" PRIu32 "\n", __func__, id, ngl_per_device[id].n_layer);
                    } else {
                        ngl_per_device_high = ngl_per_device_test;
                        mem_high            = mem_test;
                        LOG_TRC("%s: set ngl_per_device_high[%d].n_layer=%" PRIu32 "\n", __func__, id, ngl_per_device_high[id].n_layer);
                    }
                    delta = ngl_per_device_high[id].n_layer - ngl_per_device[id].n_layer;
                }
            } else {
                assert(ngl_per_device_high[id].n_layer == n_unassigned);
                ngl_per_device = ngl_per_device_high;
                mem            = mem_high;
                LOG_TRC("%s: set ngl_per_device[%d].n_layer=%" PRIu32 "\n", __func__, id, ngl_per_device[id].n_layer);
            }
        }

        const int64_t projected_margin = dmds_full[id].free - mem[id];
        LOG_TRC(
            "%s:   - %s: %2" PRIu32 " layers, %6" PRId64 " MiB used, %6" PRId64 " MiB free\n",
            __func__, dev_names[id].c_str(), ngl_per_device[id].n_layer, mem[id]/MiB, projected_margin/MiB);
    }
    if (hp_nex == 0 || global_surplus_cpu_moe <= 0) {
        set_ngl_tensor_split_tbo(ngl_per_device, overflow_bufts, *mparams);
        return;
    }

    // step 4: for a MoE model where all dense tensors fit,
    //     convert the dense-only layers in the back to full layers in the front until all devices are full
    // essentially the same procedure as for the dense-only layers except front-to-back
    // also, try fitting at least part of one more layer to reduce waste for "small" GPUs with e.g. 24 GiB VRAM

    size_t id_dense_start = nd;
    for (int id = nd - 1; id >= 0; id--) {
        if (ngl_per_device[id].n_layer > 0) {
            id_dense_start = id;
            continue;
        }
        break;
    }
    assert(id_dense_start < nd);

    LOG_TRC("%s: converting dense-only layers to full layers and filling them front-to-back with overflow to next device/system memory:\n", __func__);
    for (size_t id = 0; id <= id_dense_start && id_dense_start < nd; id++) {
        std::vector<ngl_t> ngl_per_device_high = ngl_per_device;
        for (size_t jd = id_dense_start; jd < nd; jd++) {
            const uint32_t n_layer_move = jd < nd - 1 ? ngl_per_device_high[jd].n_layer : ngl_per_device_high[jd].n_layer - 1;
            ngl_per_device_high[id].n_layer += n_layer_move;
            ngl_per_device_high[jd].n_layer -= n_layer_move;
            ngl_per_device_high[jd].n_part = 0;
        }
        size_t id_dense_start_high = nd - 1;
        std::vector<int64_t> mem_high = get_memory_for_layers(__func__, ngl_per_device_high, overflow_bufts);

        if (mem_high[id] > targets[id]) {
            assert(ngl_per_device_high[id].n_full() >= ngl_per_device[id].n_full());
            uint32_t delta = ngl_per_device_high[id].n_full() - ngl_per_device[id].n_full();
            while (delta > 1) {
                uint32_t step_size = int64_t(delta) * (targets[id] - mem[id]) / (mem_high[id] - mem[id]);
                step_size = std::max(step_size, uint32_t(1));
                step_size = std::min(step_size, delta - 1);

                std::vector<ngl_t> ngl_per_device_test = ngl_per_device;
                size_t id_dense_start_test = id_dense_start;
                uint32_t n_converted_test = 0;
                for (;id_dense_start_test < nd; id_dense_start_test++) {
                    const uint32_t n_convert_jd = std::min(step_size - n_converted_test, ngl_per_device_test[id_dense_start_test].n_part);
                    ngl_per_device_test[id_dense_start_test].n_layer -= n_convert_jd;
                    ngl_per_device_test[id_dense_start_test].n_part -= n_convert_jd;
                    ngl_per_device_test[id].n_layer += n_convert_jd;
                    n_converted_test += n_convert_jd;

                    if (ngl_per_device_test[id_dense_start_test].n_part > 0) {
                        break;
                    }
                }
                const std::vector<int64_t> mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts);

                if (mem_test[id] <= targets[id]) {
                    ngl_per_device = ngl_per_device_test;
                    mem            = mem_test;
                    id_dense_start = id_dense_start_test;
                    LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part)=(%" PRIu32 ", %" PRIu32 "), id_dense_start=%zu\n",
                        __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);
                } else {
                    ngl_per_device_high = ngl_per_device_test;
                    mem_high            = mem_test;
                    id_dense_start_high = id_dense_start_test;
                    LOG_TRC("%s: set ngl_per_device_high[%zu].(n_layer, n_part)=(%" PRIu32 ", %" PRIu32 "), id_dense_start_high=%zu\n",
                        __func__, id, ngl_per_device_high[id].n_layer, ngl_per_device_high[id].n_part, id_dense_start_high);
                }
                assert(ngl_per_device_high[id].n_full() >= ngl_per_device[id].n_full());
                delta = ngl_per_device_high[id].n_full() - ngl_per_device[id].n_full();
            }
        } else {
            ngl_per_device = ngl_per_device_high;
            mem            = mem_high;
            id_dense_start = id_dense_start_high;
            LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part)=(%" PRIu32 ", %" PRIu32 "), id_dense_start=%zu\n",
                __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);
        }

        // try to fit at least part of one more layer
        if (ngl_per_device[id_dense_start].n_layer > (id < nd - 1 ? 0 : 1)) {
            std::vector<ngl_t> ngl_per_device_test = ngl_per_device;
            size_t id_dense_start_test = id_dense_start;
            ngl_per_device_test[id_dense_start_test].n_layer--;
            ngl_per_device_test[id_dense_start_test].n_part--;
            ngl_per_device_test[id].n_layer++;
            ngl_per_device_test[id].n_part++;
            if (ngl_per_device_test[id_dense_start_test].n_part == 0) {
                id_dense_start_test++;
            }
            ngl_per_device_test[id].overflow_type = LAYER_FRACTION_UP;
            std::vector<ggml_backend_buffer_type_t> overflow_bufts_test = overflow_bufts;
            if (id < nd - 1) {
                overflow_bufts_test[id] = ggml_backend_dev_buffer_type(devs[id + 1]);
            }
            LOG_TRC("%s: trying to fit one extra layer with overflow_type=LAYER_FRACTION_UP\n", __func__);
            std::vector<int64_t> mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts_test);
            if (mem_test[id] < targets[id] && (id + 1 == nd || mem_test[id + 1] < targets[id + 1])) {
                ngl_per_device = ngl_per_device_test;
                overflow_bufts = overflow_bufts_test;
                mem            = mem_test;
                id_dense_start = id_dense_start_test;
                LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part, overflow_type)=(%" PRIu32 ", %" PRIu32 ", UP), id_dense_start=%zu\n",
                    __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);

                ngl_per_device_test[id].overflow_type = LAYER_FRACTION_GATE;
                LOG_TRC("%s: trying to fit one extra layer with overflow_type=LAYER_FRACTION_GATE\n", __func__);
                mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts_test);
                if (mem_test[id] < targets[id] && (id + 1 == nd || mem_test[id + 1] < targets[id + 1])) {
                    ngl_per_device = ngl_per_device_test;
                    overflow_bufts = overflow_bufts_test;
                    mem            = mem_test;
                    id_dense_start = id_dense_start_test;
                    LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part, overflow_type)=(%" PRIu32 ", %" PRIu32 ", GATE), id_dense_start=%zu\n",
                        __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);
                }
            } else {
                ngl_per_device_test[id].overflow_type = LAYER_FRACTION_ATTN;
                LOG_TRC("%s: trying to fit one extra layer with overflow_type=LAYER_FRACTION_ATTN\n", __func__);
                mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts_test);
                if (mem_test[id] < targets[id] && (id + 1 == nd || mem_test[id + 1] < targets[id + 1])) {
                    ngl_per_device = ngl_per_device_test;
                    overflow_bufts = overflow_bufts_test;
                    mem            = mem_test;
                    id_dense_start = id_dense_start_test;
                    LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part, overflow_type)=(%" PRIu32 ", %" PRIu32 ", ATTN), id_dense_start=%zu\n",
                        __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);
                }
            }
        }

        const int64_t projected_margin = dmds_full[id].free - mem[id];
        LOG_TRC(
            "%s:   - %s: %2" PRIu32 " layers (%2" PRIu32 " overflowing), %6" PRId64 " MiB used, %6" PRId64 " MiB free\n",
            __func__, dev_names[id].c_str(), ngl_per_device[id].n_layer, ngl_per_device[id].n_part, mem[id]/MiB, projected_margin/MiB);
    }

    // print info for devices that were not changed during the conversion from dense only to full layers:
    for (size_t id = id_dense_start + 1; id < nd; id++) {
        const int64_t projected_margin = dmds_full[id].free - mem[id];
        LOG_TRC(
            "%s:   - %s: %2" PRIu32 " layers (%2" PRIu32 " overflowing), %6" PRId64 " MiB used, %6" PRId64 " MiB free\n",
            __func__, dev_names[id].c_str(), ngl_per_device[id].n_layer, ngl_per_device[id].n_part, mem[id]/MiB, projected_margin/MiB);
    }

    set_ngl_tensor_split_tbo(ngl_per_device, overflow_bufts, *mparams);
}

enum common_params_fit_status common_fit_params(
        const char * path_model,
        llama_model_params * mparams,
        llama_context_params * cparams,
        float * tensor_split,
        llama_model_tensor_buft_override * tensor_buft_overrides,
        size_t * margins,
        uint32_t n_ctx_min,
        ggml_log_level log_level) {
    const int64_t t0_us = llama_time_us();
    common_params_fit_status status = COMMON_PARAMS_FIT_STATUS_SUCCESS;
    try {
        common_params_fit_impl(path_model, mparams, cparams, tensor_split, tensor_buft_overrides, margins, n_ctx_min, log_level);
        LOG_TRC("%s: successfully fit params to free device memory\n", __func__);
    } catch (const common_params_fit_exception & e) {
        LOG_WRN("%s: failed to fit params to free device memory: %s\n", __func__, e.what());
        status = COMMON_PARAMS_FIT_STATUS_FAILURE;
    } catch (const std::runtime_error & e) {
        LOG_ERR("%s: encountered an error while trying to fit params to free device memory: %s\n", __func__, e.what());
        status = COMMON_PARAMS_FIT_STATUS_ERROR;
    }
    const int64_t t1_us = llama_time_us();
    LOG_TRC("%s: fitting params to free memory took %.2f seconds\n", __func__, (t1_us - t0_us) * 1e-6);
    return status;
}

void common_memory_breakdown_print(const struct llama_context * ctx) {
    //const auto & devices = ctx->get_model().devices;
    const auto * model = llama_get_model(ctx);

    std::vector<ggml_backend_dev_t> devices;
    for (int i = 0; i < llama_model_n_devices(model); i++) {
        devices.push_back(llama_model_get_device(model, i));
    }

    llama_memory_breakdown memory_breakdown = llama_get_memory_breakdown(ctx);

    std::vector<std::array<std::string, 9>> table_data;
    table_data.reserve(devices.size());
    const std::string template_header = "%s: | %s | %s   %s    %s   %s   %s   %s    %s |\n";
    const std::string template_gpu    = "%s: | %s | %s = %s + (%s = %s + %s + %s) + %s |\n";
    const std::string template_other  = "%s: | %s | %s   %s    %s = %s + %s + %s    %s |\n";

    table_data.push_back({template_header, "memory breakdown [MiB]", "total", "free", "self", "model", "context", "compute", "unaccounted"});

    constexpr size_t MiB = 1024 * 1024;
    const std::vector<std::string> desc_prefixes_strip = {"NVIDIA ", "GeForce ", "Tesla ", "AMD ", "Radeon ", "Instinct "};

    // track seen buffer types to avoid double counting:
    std::set<ggml_backend_buffer_type_t> seen_buffer_types;

    // accumulative memory breakdown for each device and for host:
    std::vector<llama_memory_breakdown_data> mb_dev(devices.size());
    llama_memory_breakdown_data              mb_host;

    for (const auto & buft_mb : memory_breakdown) {
        ggml_backend_buffer_type_t          buft = buft_mb.first;
        const llama_memory_breakdown_data & mb   = buft_mb.second;
        if (ggml_backend_buft_is_host(buft)) {
            mb_host.model   += mb.model;
            mb_host.context += mb.context;
            mb_host.compute += mb.compute;
            seen_buffer_types.insert(buft);
            continue;
        }
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (dev) {
            int i_dev = -1;
            for (size_t i = 0; i < devices.size(); i++) {
                if (devices[i] == dev) {
                    i_dev = i;
                    break;
                }
            }
            if (i_dev != -1) {
                mb_dev[i_dev].model   += mb.model;
                mb_dev[i_dev].context += mb.context;
                mb_dev[i_dev].compute += mb.compute;
                seen_buffer_types.insert(buft);
                continue;
            }
        }
    }

    // print memory breakdown for each device:
    for (size_t i = 0; i < devices.size(); i++) {
        ggml_backend_dev_t dev = devices[i];
        llama_memory_breakdown_data mb = mb_dev[i];

        const std::string name = ggml_backend_dev_name(dev);
        std::string desc = ggml_backend_dev_description(dev);
        for (const std::string & prefix : desc_prefixes_strip) {
            if (desc.length() >= prefix.length() && desc.substr(0, prefix.length()) == prefix) {
                desc = desc.substr(prefix.length());
            }
        }

        size_t free, total;
        ggml_backend_dev_memory(dev, &free, &total);

        const size_t self = mb.model + mb.context + mb.compute;
        const int64_t unaccounted = static_cast<int64_t>(total) - static_cast<int64_t>(free) - static_cast<int64_t>(self);

        table_data.push_back({
            template_gpu,
            "  - " + name + " (" + desc + ")",
            std::to_string(total / MiB),
            std::to_string(free / MiB),
            std::to_string(self / MiB),
            std::to_string(mb.model / MiB),
            std::to_string(mb.context / MiB),
            std::to_string(mb.compute / MiB),
            std::to_string(unaccounted / static_cast<int64_t>(MiB))});
    }

    // print memory breakdown for host:
    {
        const size_t self = mb_host.model + mb_host.context + mb_host.compute;
        table_data.push_back({
            template_other,
            "  - Host",
            "", // total
            "", // free
            std::to_string(self / MiB),
            std::to_string(mb_host.model / MiB),
            std::to_string(mb_host.context / MiB),
            std::to_string(mb_host.compute / MiB),
            ""}); // unaccounted
    }

    // print memory breakdown for all remaining buffer types:
    for (const auto & buft_mb : memory_breakdown) {
        ggml_backend_buffer_type_t          buft = buft_mb.first;
        const llama_memory_breakdown_data & mb   = buft_mb.second;
        if (seen_buffer_types.count(buft) == 1) {
            continue;
        }
        const std::string name = ggml_backend_buft_name(buft);
        const size_t self = mb.model + mb.context + mb.compute;
        table_data.push_back({
            template_other,
            "  - " + name,
            "", // total
            "", // free
            std::to_string(self / MiB),
            std::to_string(mb.model / MiB),
            std::to_string(mb.context / MiB),
            std::to_string(mb.compute / MiB),
            ""}); // unaccounted
        seen_buffer_types.insert(buft);
    }

    for (size_t j = 1; j < table_data[0].size(); j++) {
        size_t max_len = 0;
        for (const auto & td : table_data) {
            max_len = std::max(max_len, td[j].length());
        }
        for (auto & td : table_data) {
            td[j].insert(j == 1 ? td[j].length() : 0, max_len - td[j].length(), ' ');
        }
    }
    for (const auto & td : table_data) {
        LOG_TRC(td[0].c_str(),
            __func__, td[1].c_str(), td[2].c_str(), td[3].c_str(), td[4].c_str(), td[5].c_str(),
            td[6].c_str(), td[7].c_str(), td[8].c_str());
    }
}

void common_fit_print(
        const char * path_model,
        llama_model_params * mparams,
        llama_context_params * cparams) {
    std::vector<ggml_backend_dev_t> devs;
    uint32_t hp_ngl = 0; // hparams.n_gpu_layers
    uint32_t hp_nct = 0; // hparams.n_ctx_train
    uint32_t hp_nex = 0; // hparams.n_expert

    auto dmd = common_get_device_memory_data_impl(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, GGML_LOG_LEVEL_ERROR);
    GGML_ASSERT(dmd.size() == devs.size() + 1);

    for (size_t id = 0; id < devs.size(); id++) {
        printf("%s ",  ggml_backend_dev_name(devs[id]));
        printf("%zu ", dmd[id].mb.model/1024/1024);
        printf("%zu ", dmd[id].mb.context/1024/1024);
        printf("%zu ", dmd[id].mb.compute/1024/1024);
        printf("\n");
    }

    printf("Host ");
    printf("%zu ", dmd.back().mb.model/1024/1024);
    printf("%zu ", dmd.back().mb.context/1024/1024);
    printf("%zu ", dmd.back().mb.compute/1024/1024);
    printf("\n");
}
