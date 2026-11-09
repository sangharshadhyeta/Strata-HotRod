// src/kernels/cuda/subfit_capture.cu - the device side of STRATA_SUBFIT_LOG=1 (see include/strata/kernels/subfit_capture.hpp).
#include "strata/kernels/subfit_capture.hpp"

namespace strata::kernels {
namespace {

constexpr int kThreads = 256;

__global__ void k_capture_x(const float* __restrict__ mixed, const int32_t* __restrict__ rec, const int32_t* __restrict__ gate,
                            int layer, float* __restrict__ x, int64_t N, int n_rows) {
    const int row = blockIdx.y;
    if (row >= n_rows || gate[0] != 1 || rec[row * 4] != layer) return;
    for (int64_t c = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; c < N; c += (int64_t) gridDim.x * blockDim.x)
        x[row * N + c] = mixed[row * N + c];
}

__global__ void k_probe_pre(const int32_t* __restrict__ ids, const int32_t* __restrict__ rec, const int32_t* __restrict__ gate, int layer,
                            int K, int32_t* __restrict__ pre, int32_t* __restrict__ clean, int32_t* __restrict__ flag) {
    if (gate[3] != 1) return;
    const int k = threadIdx.x;
    if (k < K) {
        pre[layer * K + k] = ids[k];
        flag[layer * K + k] = 0;
    }
    if (k == 0) clean[layer] = rec[0] < 0 ? 1 : 0;   // rec[0]: row 0's first stand-in layer (-1 none), before this layer's route-resident
}

__global__ void k_probe_post(const int32_t* __restrict__ ids, const float* __restrict__ w, const int32_t* __restrict__ res,
                             const int32_t* __restrict__ pre, const int32_t* __restrict__ clean, const int32_t* __restrict__ gate,
                             const float* __restrict__ mixed0, int layer, int K, int64_t N, float* __restrict__ x,
                             float* __restrict__ w_out, int32_t* __restrict__ flag) {
    if (gate[3] != 1) return;
    for (int64_t c = threadIdx.x; c < N; c += blockDim.x) x[layer * N + c] = mixed0[c];
    const int k = threadIdx.x;
    if (k < K) {
        const int32_t e = ids[k];
        // the pick must be the router's own (route-resident did not replace it), its expert resident (d_res >= 0), row 0 clean
        const bool ok = clean[layer] != 0 && pre[layer * K + k] == e && e >= 0 && res[e] >= 0;
        flag[layer * K + k] = ok ? 1 : 0;
        w_out[layer * K + k] = w[k];
    }
}

__global__ void k_probe_y(const float* __restrict__ parts, const int32_t* __restrict__ gate, int64_t K_N, float* __restrict__ y) {
    if (gate[3] != 1) return;
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < K_N; i += (int64_t) gridDim.x * blockDim.x) y[i] = parts[i];
}

}  // namespace

void subfit_capture_x(const float* mixed, const int32_t* rec, const int32_t* gate, int layer, float* x, int64_t N, int n_rows,
                      cudaStream_t stream) {
    if (n_rows <= 0) return;
    const dim3 grid((unsigned) ((N + kThreads - 1) / kThreads), (unsigned) n_rows);
    k_capture_x<<<grid, kThreads, 0, stream>>>(mixed, rec, gate, layer, x, N, n_rows);
}

void subfit_probe_pre(const int32_t* ids, const int32_t* rec, const int32_t* gate, int layer, int K, int32_t* pre, int32_t* clean,
                      int32_t* flag, cudaStream_t stream) {
    k_probe_pre<<<1, 32, 0, stream>>>(ids, rec, gate, layer, K, pre, clean, flag);
}

void subfit_probe_post(const int32_t* ids, const float* w, const int32_t* res, const int32_t* pre, const int32_t* clean,
                       const int32_t* gate, const float* mixed0, int layer, int K, int64_t N, float* x, float* w_out, int32_t* flag,
                       cudaStream_t stream) {
    k_probe_post<<<1, kThreads, 0, stream>>>(ids, w, res, pre, clean, gate, mixed0, layer, K, N, x, w_out, flag);
}

void subfit_probe_y(const float* parts, const int32_t* gate, int64_t K_N, float* y, cudaStream_t stream) {
    const int64_t blocks = (K_N + kThreads - 1) / kThreads;
    k_probe_y<<<(unsigned) (blocks < 1 ? 1 : blocks), kThreads, 0, stream>>>(parts, gate, K_N, y);
}

}  // namespace strata::kernels
