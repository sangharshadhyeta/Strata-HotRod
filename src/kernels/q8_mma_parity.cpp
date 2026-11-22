// src/kernels/q8_mma_parity.cpp - the tensor-core Q8_0 matmul (kernels::mma::q8_matmul) against a CPU reference built from the same quantized bytes, for the engine's real shapes and 1 .. 8 columns.
// Synthetic data, a GPU, no model.
#include "strata/kernels/moe_group.hpp"
#include "strata/kernels/q8_mma.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

int main() {
    int devs = 0;
    if (cudaGetDeviceCount(&devs) != cudaSuccess || devs == 0) { std::printf("(skipped: no CUDA device)\n"); return 0; }
    namespace M = strata::kernels::moe;
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    struct Shape { int n_in, n_out; };
    int fails = 0;
    for (const Shape sh : {Shape{2304, 3072}, Shape{2304, 1024}, Shape{3072, 2304}, Shape{2304, 384}, Shape{2304, 201024}}) {
        const int nb = sh.n_in / 32;
        std::vector<uint8_t> W((size_t) sh.n_out * nb * 34);
        for (size_t r = 0; r < (size_t) sh.n_out * nb; ++r) {
            const __half d = __float2half(0.002f + 0.01f * std::fabs(nd(rng)));
            std::memcpy(&W[r * 34], &d, 2);
            for (int j = 0; j < 32; ++j) W[r * 34 + 2 + j] = (uint8_t) (int8_t) std::max(-127, std::min(127, (int) std::lround(nd(rng) * 40.0f)));
        }
        uint8_t* dW = nullptr; float *dx = nullptr, *dy = nullptr; void* dq = nullptr;
        cudaMalloc((void**) &dW, W.size()); cudaMemcpy(dW, W.data(), W.size(), cudaMemcpyHostToDevice);
        cudaMalloc(&dx, (size_t) 8 * sh.n_in * 4); cudaMalloc(&dq, (size_t) 8 * nb * 36); cudaMalloc(&dy, (size_t) 8 * sh.n_out * 4);
        for (int cols = 1; cols <= 8; ++cols) {
            std::vector<float> x((size_t) cols * sh.n_in);
            for (float& v : x) v = nd(rng) * 2.0f;
            cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
            M::quantize_act(dx, sh.n_in, cols, dq, nullptr);
            if (!strata::kernels::mma::q8_matmul(dW, dq, dy, sh.n_in, sh.n_out, cols, nullptr)) { std::printf("FAIL: q8_matmul refused %d x %d\n", sh.n_in, sh.n_out); ++fails; continue; }
            std::vector<uint8_t> xq((size_t) cols * nb * 36);
            std::vector<float> y((size_t) cols * sh.n_out);
            cudaMemcpy(xq.data(), dq, xq.size(), cudaMemcpyDeviceToHost);
            cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost);
            double worst = 0, mx = 0;
            const int rows_check = std::min(sh.n_out, 600);   // a spread of rows (the head has 201024)
            for (int ri = 0; ri < rows_check; ++ri) {
                const int r = sh.n_out == rows_check ? ri : (int) ((long long) ri * (sh.n_out - 1) / (rows_check - 1));
                for (int c = 0; c < cols; ++c) {
                    double ref = 0;
                    for (int b = 0; b < nb; ++b) {
                        __half dw, dxh; std::memcpy(&dw, &W[((size_t) r * nb + b) * 34], 2); std::memcpy(&dxh, &xq[((size_t) c * nb + b) * 36], 2);
                        long isum = 0;
                        for (int j = 0; j < 32; ++j) isum += (long) (int8_t) W[((size_t) r * nb + b) * 34 + 2 + j] * (long) (int8_t) xq[((size_t) c * nb + b) * 36 + 4 + j];
                        ref += (double) __half2float(dw) * (double) __half2float(dxh) * (double) isum;
                    }
                    mx = std::max(mx, std::fabs(ref));
                    worst = std::max(worst, std::fabs(ref - (double) y[(size_t) c * sh.n_out + r]));
                }
            }
            const bool ok = worst <= 1e-5 * std::max(mx, 1e-9);
            if (!ok) ++fails;
            if (!ok || cols == 8 || cols == 1) std::printf("%s %6d x %-6d cols %d: max |diff| %.2e (max |ref| %.2e)\n", ok ? "ok  " : "FAIL", sh.n_in, sh.n_out, cols, worst, mx);
        }
        cudaFree(dW); cudaFree(dx); cudaFree(dq); cudaFree(dy);
    }
    std::printf("%s\n", fails == 0 ? "the tensor-core Q8_0 matmul matches the reference for every shape and column count" : "FAILED");
    return fails == 0 ? 0 : 1;
}
