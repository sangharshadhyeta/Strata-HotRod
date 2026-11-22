#pragma once

#include "ggml.h"
#include "llama.h"

#include <vector>

enum common_params_fit_status {
    COMMON_PARAMS_FIT_STATUS_SUCCESS = 0, // found allocations that are projected to fit
    COMMON_PARAMS_FIT_STATUS_FAILURE = 1, // could not find allocations that are projected to fit
    COMMON_PARAMS_FIT_STATUS_ERROR   = 2, // a hard error occurred, e.g. because no model could be found at the specified path
};

// fits mparams and cparams to free device memory (assumes system memory is unlimited)
//   - returns true if the parameters could be successfully modified to fit device memory
//   - this function is NOT thread safe because it modifies the global llama logger state
//   - only parameters that have the same value as in llama_default_model_params are modified
//     with the exception of the context size which is modified if and only if equal to 0
common_params_fit_status common_fit_params(
                         const char * path_model,
                 llama_model_params * mparams,
               llama_context_params * cparams,
                              float * tensor_split,          // writable buffer for tensor split, needs at least llama_max_devices elements
   llama_model_tensor_buft_override * tensor_buft_overrides, // writable buffer for overrides, needs at least llama_max_tensor_buft_overrides elements
                             size_t * margins,               // margins of memory to leave per device in bytes
                           uint32_t   n_ctx_min,             // minimum context size to set when trying to reduce memory use
                     ggml_log_level   log_level);            // minimum log level to print during fitting, lower levels go to debug log

// print estimated memory to stdout
void common_fit_print(
                         const char * path_model,
                 llama_model_params * mparams,
               llama_context_params * cparams);

void common_memory_breakdown_print(const llama_context * ctx);

// print a recommended --moe-cache budget plus the GGML_CUDA_MOE_CACHE_*
// env vars needed for it to actually engage, given a model + hardware.
// No-op (prints a short note and returns) if the model isn't a MoE model.
void common_fit_print_moe_cache(
                         const char * path_model,
                 llama_model_params * mparams,
               llama_context_params * cparams);

struct common_device_memory_data {
    int64_t total;
    int64_t free;
    size_t  model;
    size_t  context;
    size_t  compute;
};

using common_device_memory_data_vec = std::vector<common_device_memory_data>;

// A no_alloc model + context pair, kept alive only to stand in as
// cparams.ctx_other while something else is measured.
//
// A draft head that ships without its own embeddings / LM head (qwen4exp MTP,
// eagle3, dflash) borrows them from its target, and cannot build a context -
// let alone a graph - without one. Measuring such a draft before the real
// target context exists therefore needs a stand-in, and a no_alloc context is
// exactly that: it reads metadata and allocates nothing, so it is cheap enough
// to build and throw away inside a fitting pass.
struct common_probe_context {
    llama_model   * model = nullptr;
    llama_context * ctx   = nullptr;

    common_probe_context() = default;
    common_probe_context(const common_probe_context &) = delete;
    common_probe_context & operator=(const common_probe_context &) = delete;
    ~common_probe_context();
};

// Build a no_alloc context for path_model, for use as cparams.ctx_other.
// Returns a pair whose ctx is nullptr if the model could not provide one; the
// caller should then measure without it rather than treat this as fatal.
void common_make_probe_context(
                         const char * path_model,
           const llama_model_params * mparams,
         const llama_context_params * cparams,
              common_probe_context  & out,
                     ggml_log_level   log_level);

// Load a model + context with no_alloc and return the per-device memory breakdown.
common_device_memory_data_vec common_get_device_memory_data(
                         const char * path_model,
           const llama_model_params * mparams,
         const llama_context_params * cparams,
    std::vector<ggml_backend_dev_t> & devs,
                           uint32_t & hp_ngl,
                           uint32_t & hp_n_ctx_train,
                           uint32_t & hp_n_expert,
                     ggml_log_level   log_level);
