// Test-only translation unit: production scorer plus the unchanged warp dispatch.
// The production library does not compile this reference wrapper.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "qsa_select.dp.cpp"

namespace strata::kernels {
void qsa_block_scores_reference(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                      int64_t max_blocks, const QsaShapes& s, float* scores, void* stream, int64_t active_blocks) {
    if (nq <= 0) return;
    if (s.idx_dim != IDX_DIM || s.idx_n_head != IDX_HEADS || s.idx_block != R || nq > 65535) {
        std::fprintf(stderr, "qsa_block_scores: unsupported indexer geometry\n");
        std::exit(1);
    }
    // a block past a query's n_bid returns at once: the grid need only reach the batch's largest n_bid (C-1)
    static const bool multi = [] { const char* v = std::getenv("STRATA_SCORES_MULTI"); return v == nullptr || std::atoi(v) != 0; }();
    if (multi && nq <= MQ && active_blocks <= 0) {   // no active count: decode (captured or not) and prefill's pooled16
        // STRATA_QSA_EARLY_EXIT=0 keeps the old staging (every CTA loads the queries first); the scores are the same bits
        static const bool early = [] { const char* v = std::getenv("STRATA_QSA_EARLY_EXIT"); return v == nullptr || std::atoi(v) != 0; }();
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            strata::q_of(stream)
                ->submit([&](sycl::handler &cgh) {
                    auto early_ct7 = early;

                    cgh.parallel_for<dpct_kernel_name<
                        class block_scores_multi_kernel_1d7117>>(
                        sycl::nd_range<3>(
                            sycl::range(1, 1, 256) *
                                sycl::range(1, 1, SCORE_WARPS * 32),
                            sycl::range(1, 1, SCORE_WARPS * 32)),
                        exp_props,
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(32)]] {
                                block_scores_multi_kernel(
                                    pooled, dead, q_idx, steps, (int)nq,
                                    max_blocks, scores, early_ct7, {});
                            });
                });
        }
        /*
        DPCT1010: SYCL uses exceptions to report errors and does not use
        the error codes. The cudaGetLastError function call was replaced with 0.
        You need to rewrite this code.
        */
        const dpct::err0 e = 0;
        /*
        DPCT1009: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */

        return;
    }
    const int64_t reach = active_blocks > 0 && active_blocks < max_blocks ? active_blocks : max_blocks;
    const dpct::dim3 grid((unsigned)((reach + SCORE_WARPS - 1) / SCORE_WARPS),
                          (unsigned)nq);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class block_scores_kernel_86943d>>(
                sycl::nd_range<3>(grid * sycl::range(1, 1, SCORE_WARPS * 32),
                                  sycl::range(1, 1, SCORE_WARPS * 32)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        block_scores_kernel(pooled, dead, q_idx, steps,
                                            max_blocks, scores);
                    });
    }
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;
    /*
    DPCT1009: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
}

}
