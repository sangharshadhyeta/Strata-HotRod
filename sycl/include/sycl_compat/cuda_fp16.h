#pragma once
// SYCL build: CUDA's half type for host code is sycl::half (the migrated kernels use sycl::half too), so a buffer of
// __half filled on the host has the bytes the kernels expect.
#include <sycl/sycl.hpp>
using __half = sycl::half;
inline __half __float2half(float f) { return sycl::half(f); }
inline float __half2float(__half h) { return static_cast<float>(h); }
inline __half __float2half_rn(float f) { return sycl::half(f); }
