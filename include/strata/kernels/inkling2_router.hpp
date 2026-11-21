// include/strata/kernels/inkling2_router.hpp - Inkling 2's router on the GPU (docs/INKLING-PORT.md).
//
// One block a token.  scores = sigmoid(logits); the experts are PICKED by scores + bias (the selection bias is for the pick only, never in the
// weights), the k best, the lowest index winning a tie; the weights are the picked scores' share of their sum times `route_scale` (8.0).
// Hand-ported from the DeepSeek branch's route_kernel (src/dsv41/ds4_kernels.cu, upstream-dsv41): the same biased top-k, with sigmoid in place of
// DeepSeek's sqrt-softplus score, a row stride of n (that kernel's is a fixed 1,024), and none of its DS4_MASK / vector-length extras.  Its
// non-finite-input guard is kept: a pick that never came out becomes the lowest expert not yet picked, so nothing reads outside the tables.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// logits [T][n] (row-major), bias [n]; n <= 1024, k <= 16.  Writes idx [T][k] (pick order: best score first) and w [T][k].
void inkling2_route(const float* logits, const float* bias, int T, int n, int k, float route_scale, int32_t* idx, float* w, void* stream);

}  // namespace strata::kernels
