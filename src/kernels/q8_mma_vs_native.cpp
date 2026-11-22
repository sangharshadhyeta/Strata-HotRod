// src/kernels/q8_mma_vs_native.cpp - Qwen's Q8_0 matvec (native_mmvq, ggml type 8) against the tensor-core kernel (STRATA_Q8_MMA path), on Qwen's real Q8_0 shapes and 2 - 8 columns:
// the largest difference between the two outputs, and the time of each (cudaEvent, 200 calls).  Synthetic data, a GPU, no model.
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace K = strata::kernels;

int main(int argc, char** argv) {
    int devs = 0;
    if (cudaGetDeviceCount(&devs) != cudaSuccess || devs == 0) { std::printf("(skipped: no CUDA device)\n"); return 0; }
    struct Shape { int n_in, n_out; };
    std::vector<Shape> shapes = {{10240, 320}, {320, 10240}, {2560, 640}, {640, 2560}, {6144, 2560}, {2560, 10240}, {2560, 6144}, {2560, 512}, {2560, 12288}, {2560, 2560}, {2560, 248320}};
    if (argc > 1) { shapes.clear(); for (int i = 1; i < argc; ++i) { int a = 0, b = 0; if (std::sscanf(argv[i], "%d:%d", &a, &b) == 2) shapes.push_back({a, b}); } }
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    int fails = 0;
    cudaStream_t S; cudaStreamCreate(&S);
    std::printf("%-14s %s\n", "shape", "cols: max rel diff | native ms -> mma ms (x speed-up)");
    for (const Shape sh : shapes) {
        const int nb = sh.n_in / 32;
        std::vector<uint8_t> W((size_t) sh.n_out * nb * 34);
        for (size_t r = 0; r < (size_t) sh.n_out * nb; ++r) {
            const __half d = __float2half(0.002f + 0.01f * std::fabs(nd(rng)));
            std::memcpy(&W[r * 34], &d, 2);
            for (int j = 0; j < 32; ++j) W[r * 34 + 2 + j] = (uint8_t) (int8_t) std::max(-127, std::min(127, (int) std::lround(nd(rng) * 40.0f)));
        }
        uint8_t* dW = nullptr; float *dx = nullptr, *y0 = nullptr, *y1 = nullptr; void* dq = nullptr;
        if (cudaMalloc((void**) &dW, W.size()) != cudaSuccess || cudaMalloc(&dx, (size_t) 8 * sh.n_in * 4) != cudaSuccess || cudaMalloc(&dq, (size_t) 8 * nb * 36) != cudaSuccess ||
            cudaMalloc(&y0, (size_t) 8 * sh.n_out * 4) != cudaSuccess || cudaMalloc(&y1, (size_t) 8 * sh.n_out * 4) != cudaSuccess) {
            std::printf("FAIL %d:%d: a device allocation failed (GPU memory in use?)\n", sh.n_in, sh.n_out); ++fails; (void) cudaGetLastError(); continue;
        }
        cudaMemcpy(dW, W.data(), W.size(), cudaMemcpyHostToDevice);
        std::vector<float> x((size_t) 8 * sh.n_in);
        for (float& v : x) v = nd(rng) * 2.0f;
        cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
        K::quantize_q8_1_rows(dx, 8, sh.n_in, dq, S);
        std::printf("%5d x %-7d", sh.n_in, sh.n_out);
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        for (int cols = 2; cols <= 8; ++cols) {
            K::native_q8_mma_set(0);
            K::native_mmvq(8, dW, dq, y0, sh.n_in, sh.n_out, cols, S);
            K::native_q8_mma_set(1);
            K::native_mmvq(8, dW, dq, y1, sh.n_in, sh.n_out, cols, S);
            cudaStreamSynchronize(S);
            std::vector<float> a((size_t) cols * sh.n_out), b((size_t) cols * sh.n_out);
            cudaMemcpy(a.data(), y0, a.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(b.data(), y1, b.size() * 4, cudaMemcpyDeviceToHost);
            double mx = 0, md = 0;
            for (size_t i = 0; i < a.size(); ++i) { mx = std::max(mx, (double) std::fabs(a[i])); md = std::max(md, (double) std::fabs(a[i] - b[i])); }
            if (!(mx > 0) || md > 1e-5 * mx) ++fails;
            float t[2] = {0, 0};
            for (int mode = 0; mode < 2; ++mode) {
                K::native_q8_mma_set(mode);
                for (int i = 0; i < 20; ++i) K::native_mmvq(8, dW, dq, y0, sh.n_in, sh.n_out, cols, S);
                cudaEventRecord(e0, S);
                for (int i = 0; i < 200; ++i) K::native_mmvq(8, dW, dq, y0, sh.n_in, sh.n_out, cols, S);
                cudaEventRecord(e1, S); cudaEventSynchronize(e1);
                cudaEventElapsedTime(&t[mode], e0, e1);
            }
            std::printf("  %d: %.0e %5.3f->%5.3f (%.2fx)", cols, md / std::max(mx, 1e-30), t[0] / 200, t[1] / 200, t[0] / t[1]);
        }
        std::printf("\n");
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        cudaFree(dW); cudaFree(dx); cudaFree(dq); cudaFree(y0); cudaFree(y1);
    }
    K::native_q8_mma_set(-1);
    std::printf("%s\n", fails == 0 ? "the two kernels agree on every shape and column count" : "FAILED (a difference above 1e-5 relative, a zero reference, or an allocation)");
    return fails == 0 ? 0 : 1;
}
