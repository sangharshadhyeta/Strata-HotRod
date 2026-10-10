// src/kernels/cuda/standin_mask.cu - see include/strata/kernels/standin_mask.hpp.
#include "strata/kernels/standin_mask.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <cstdio>

namespace strata::kernels {

namespace {
const int32_t* g_table = nullptr;
int g_n_expert = 0;
constexpr int kMaxRows = 8;   // a decode window; larger is a prompt chunk, routed exactly
}  // namespace

// One block per row. Counts the layer's resident experts; masks only when at least k are resident, so the router always
// finds k real experts.
__global__ void standin_mask_kernel(float* logits, const int32_t* res, int n_expert, int k) {
    float* row = logits + (size_t) blockIdx.x * n_expert;
    __shared__ int count;
    if (threadIdx.x == 0) count = 0;
    __syncthreads();
    int local = 0;
    for (int e = threadIdx.x; e < n_expert; e += blockDim.x) local += res[e] >= 0 ? 1 : 0;
    atomicAdd(&count, local);
    __syncthreads();
    if (count < k) return;
    for (int e = threadIdx.x; e < n_expert; e += blockDim.x)
        if (res[e] < 0) row[e] = -INFINITY;
}

void standin_set_table(const int32_t* d_res, int n_expert) {
    g_table = d_res;
    g_n_expert = n_expert;
}

bool standin_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_STANDIN");
        return v != nullptr && std::atoi(v) != 0;
    }();
    return on && g_table != nullptr;
}

int standin_max_rows() { return kMaxRows; }

void standin_mask(float* logits, int n_rows, int n_expert, int k, int64_t layer, void* stream) {
    if (!standin_enabled() || n_rows <= 0 || n_rows > kMaxRows || n_expert != g_n_expert) return;
    standin_mask_kernel<<<n_rows, 512, 0, (cudaStream_t) stream>>>(logits, g_table + (size_t) layer * n_expert, n_expert, k);
}

}  // namespace strata::kernels
