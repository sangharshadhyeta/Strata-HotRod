// src/kernels/cuda/q8_mma.cu - see include/strata/kernels/q8_mma.hpp.
#include "strata/kernels/q8_mma.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace strata::kernels::mma {
namespace {

__device__ __forceinline__ uint32_t ld32_u16(const uint8_t* p) {   // 2-byte aligned (Q8_0's int8 data starts 2 bytes into a 34-byte block)
    return (uint32_t) *reinterpret_cast<const uint16_t*>(p) | ((uint32_t) *reinterpret_cast<const uint16_t*>(p + 2) << 16);
}
__device__ __forceinline__ float h2f(const uint8_t* p) { return __half2float(*reinterpret_cast<const __half*>(p)); }

// One warp = 16 weight rows x all k, against up to 8 tokens.  Fragment layouts of mma.m16n8k32 (s8):
//   A (16 x 32, row): a0 = row g, k 4t..4t+3; a1 = row g + 8, same k; a2 = row g, k 16 + 4t..; a3 = row g + 8, k 16 + 4t..       (g = lane / 4, t = lane % 4)
//   B (32 x 8, col):  b0 = k 4t..4t+3, token g; b1 = k 16 + 4t.., token g
//   C (16 x 8):       c0, c1 = row g, tokens 2t, 2t + 1;  c2, c3 = row g + 8, tokens 2t, 2t + 1
__global__ void k_q8_mma(const uint8_t* __restrict__ W, const uint8_t* __restrict__ xq, float* __restrict__ y, int nb, int n_out, int cols) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    const int r0 = warp * 16;
    if (r0 >= n_out) return;
    const int g = lane >> 2, t = lane & 3;
    const size_t wrow = (size_t) nb * 34, xrow = (size_t) nb * 36;
    const uint8_t* w0 = W + (size_t) (r0 + g) * wrow;
    const uint8_t* w1 = W + (size_t) (r0 + g + 8) * wrow;
    const uint8_t* xb = xq + (size_t) (g < cols ? g : cols - 1) * xrow;          // B column: token g (a padded column reads a real token's data; its C column is dropped)
    const uint8_t* xc0 = xq + (size_t) (2 * t < cols ? 2 * t : cols - 1) * xrow;  // the scales of the C columns this lane holds
    const uint8_t* xc1 = xq + (size_t) (2 * t + 1 < cols ? 2 * t + 1 : cols - 1) * xrow;
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
#pragma unroll 4
    for (int b = 0; b < nb; ++b) {
        const uint8_t* q0 = w0 + (size_t) b * 34;
        const uint8_t* q1 = w1 + (size_t) b * 34;
        const uint32_t a0 = ld32_u16(q0 + 2 + 4 * t), a1 = ld32_u16(q1 + 2 + 4 * t), a2 = ld32_u16(q0 + 2 + 16 + 4 * t), a3 = ld32_u16(q1 + 2 + 16 + 4 * t);
        const uint8_t* xk = xb + (size_t) b * 36 + 4;
        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(xk + 4 * t), b1 = *reinterpret_cast<const uint32_t*>(xk + 16 + 4 * t);
        int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                     : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(0), "r"(0), "r"(0), "r"(0));
        const float dw0 = h2f(q0), dw1 = h2f(q1), dx0 = h2f(xc0 + (size_t) b * 36), dx1 = h2f(xc1 + (size_t) b * 36);
        acc0 += dw0 * dx0 * (float) d0;
        acc1 += dw0 * dx1 * (float) d1;
        acc2 += dw1 * dx0 * (float) d2;
        acc3 += dw1 * dx1 * (float) d3;
    }
    if (2 * t < cols) { y[(size_t) (2 * t) * n_out + r0 + g] = acc0; y[(size_t) (2 * t) * n_out + r0 + g + 8] = acc2; }
    if (2 * t + 1 < cols) { y[(size_t) (2 * t + 1) * n_out + r0 + g] = acc1; y[(size_t) (2 * t + 1) * n_out + r0 + g + 8] = acc3; }
}

}  // namespace

bool q8_matmul(const uint8_t* W, const void* xq, float* y, int n_in, int n_out, int cols, void* stream) {
    if (cols < 1 || cols > 8 || (n_out & 15) != 0 || (n_in & 31) != 0) return false;
    const int warps = n_out / 16, threads = 128;
    k_q8_mma<<<(warps * 32 + threads - 1) / threads, threads, 0, (cudaStream_t) stream>>>(W, (const uint8_t*) xq, y, n_in / 32, n_out, cols);
    return true;
}

}  // namespace strata::kernels::mma
