#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer_rw(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

#ifdef  __cplusplus
}
#endif

// Lazily-committed (VMM) buffers: see ggml_backend_cuda_vmm_buffer_context in
// ggml-cuda.cu. Both are no-ops/0 for ordinary buffers, so callers need not
// know which kind they hold.
GGML_BACKEND_API bool   ggml_backend_cuda_buffer_is_vmm(ggml_backend_buffer_t buffer);
// Ensure the leading `fraction` of each of the `n_stream` equal parts of every
// tensor in the buffer is backed by physical memory (a KV tensor holds one part
// per sequence stream, [.., kv_size, n_stream]). Returns false if the mapping
// failed (out of memory).
GGML_BACKEND_API bool   ggml_backend_cuda_vmm_commit_fraction(ggml_backend_buffer_t buffer, double fraction, int n_stream);
GGML_BACKEND_API size_t ggml_backend_cuda_vmm_committed_bytes(ggml_backend_buffer_t buffer);
// Reserved-but-not-yet-committed bytes across every lazily-committed buffer: the VRAM the KV cache
// can still claim as contexts grow.
GGML_BACKEND_API size_t ggml_backend_cuda_vmm_uncommitted_bytes(void);

// The expert cache gives pools back until `bytes_needed` are free on the device; returns the bytes released.
GGML_BACKEND_API size_t ggml_backend_cuda_moe_cache_yield_vram(int device, size_t bytes_needed);

// Residency snapshot of the expert cache for the router bias: see moe-cache.cu.
GGML_BACKEND_API size_t ggml_backend_cuda_moe_cache_resident_counts(const void * const * bases, const int32_t * layer_of,
        int n_bases, int n_layers, int64_t n_expert, uint8_t * out);
// Mark the next buffer allocation on this thread as eligible for lazy commit.
// Only meaningful for allocations whose use lags their reservation (the KV
// cache); weights are read in full at load and must never be lazy.
GGML_BACKEND_API void ggml_backend_cuda_vmm_next_alloc(bool enable);
// Whether lazy commit applies to the next KV allocation when GGML_CUDA_VMM_KV is not set (the KV cache passes: window >= 131072 cells).
GGML_BACKEND_API void ggml_backend_cuda_vmm_kv_auto(bool enable);
