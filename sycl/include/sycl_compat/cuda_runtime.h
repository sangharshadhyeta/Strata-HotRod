#pragma once
// Included only by the SYCL build: sycl/CMakeLists.txt puts sycl/include/sycl_compat first on the include path, as
// metal/CMakeLists.txt does with metal/include/metal_compat (the same design: the host files are shared, unchanged).
// The CUDA runtime's host API as the engine's host files use it, implemented over SYCL (USM, in-order queues, events,
// the oneAPI graph extension), so those files compile unchanged for Intel GPUs: the same idea as hip_compat, except
// that SYCL is not CUDA-shaped, so the calls are written out here instead of renamed.
//
// Scope: host code only.  Kernels are separate (.dp.cpp, migrated); a host file never launches one itself, it calls
// a wrapper declared in include/strata/kernels/*.hpp, whose cudaStream_t is a sycl::queue* (dpct::queue_ptr) here.
// A CUDA call with no SYCL meaning returns cudaErrorNotSupported (the engine treats it like an older driver).
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#define STRATA_SYCL_COMPAT 1
#ifndef CUDART_VERSION
#define CUDART_VERSION 12000   // the feature level these shims answer to; newer CUDA-only paths stay off
#endif

// ---- errors --------------------------------------------------------------------------------------------------
enum cudaError_t : int {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorNotReady = 600,
    cudaErrorPeerAccessAlreadyEnabled = 704,
    cudaErrorNotSupported = 801,
    cudaErrorStreamCaptureUnsupported = 900,
    cudaErrorUnknown = 999,
};
using cudaError = cudaError_t;

namespace strata_sycl_compat {
inline cudaError_t &last_error() {
    static thread_local cudaError_t e = cudaSuccess;
    return e;
}
inline cudaError_t fail(cudaError_t e) {
    last_error() = e;
    return e;
}
// Runs f; a SYCL exception becomes an error code (CUDA's calls do not throw).
template <class F> inline cudaError_t guard(F &&f, cudaError_t on_error = cudaErrorUnknown) {
    try {
        f();
        return cudaSuccess;
    } catch (const sycl::exception &) {
        return fail(on_error);
    } catch (const std::bad_alloc &) {
        return fail(cudaErrorMemoryAllocation);
    }
}
}  // namespace strata_sycl_compat

inline const char *cudaGetErrorString(cudaError_t e) {
    switch (e) {
        case cudaSuccess: return "no error";
        case cudaErrorInvalidValue: return "invalid argument";
        case cudaErrorMemoryAllocation: return "out of memory";
        case cudaErrorNotReady: return "device not ready";
        case cudaErrorPeerAccessAlreadyEnabled: return "peer access is already enabled";
        case cudaErrorNotSupported: return "operation not supported (SYCL build)";
        case cudaErrorStreamCaptureUnsupported: return "stream capture not supported (SYCL build)";
        default: return "unknown error (a SYCL exception)";
    }
}
inline cudaError_t cudaGetLastError() {
    const cudaError_t e = strata_sycl_compat::last_error();
    strata_sycl_compat::last_error() = cudaSuccess;
    return e;
}
inline cudaError_t cudaPeekAtLastError() { return strata_sycl_compat::last_error(); }

// ---- devices -------------------------------------------------------------------------------------------------
struct cudaDeviceProp {
    char name[256];
    size_t totalGlobalMem;
    int multiProcessorCount;
    int major, minor;              // 0.0: no compute capability on Intel; code that needs one takes its fallback
    int clockRate;                 // kHz
    int integrated;
    int warpSize;                  // the sub-group size the kernels are built for (32)
    size_t sharedMemPerBlock;
    int l2CacheSize;               // bytes (the device's global memory cache)
    int maxThreadsPerBlock;
    int canMapHostMemory;
    int unifiedAddressing;
    int pciBusID, pciDeviceID, pciDomainID;
};
enum cudaDeviceAttr : int {
    cudaDevAttrMultiProcessorCount,
    cudaDevAttrClockRate,
    cudaDevAttrComputeCapabilityMajor,
    cudaDevAttrComputeCapabilityMinor,
    cudaDevAttrIntegrated,
    cudaDevAttrMaxSharedMemoryPerBlockOptin,
};
inline cudaError_t cudaGetDeviceCount(int *n) {
    return strata_sycl_compat::guard([&] { *n = (int) dpct::dev_mgr::instance().device_count(); });
}
inline cudaError_t cudaSetDevice(int d) {
    return strata_sycl_compat::guard([&] { dpct::select_device((unsigned) d); }, cudaErrorInvalidValue);
}
inline cudaError_t cudaInitDevice(int d, unsigned = 0, unsigned = 0) { return cudaSetDevice(d); }
inline cudaError_t cudaGetDevice(int *d) {
    return strata_sycl_compat::guard([&] { *d = (int) dpct::get_current_device_id(); });
}
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp *p, int d) {
    return strata_sycl_compat::guard([&] {
        std::memset(p, 0, sizeof(*p));
        dpct::device_ext &dev = dpct::dev_mgr::instance().get_device((unsigned) d);
        const std::string nm = dev.get_info<sycl::info::device::name>();
        std::strncpy(p->name, nm.c_str(), sizeof(p->name) - 1);
        p->totalGlobalMem = dev.get_info<sycl::info::device::global_mem_size>();
        p->multiProcessorCount = (int) dev.get_info<sycl::info::device::max_compute_units>();
        p->clockRate = (int) dev.get_info<sycl::info::device::max_clock_frequency>() * 1000;
        p->integrated = dev.get_info<sycl::info::device::host_unified_memory>() ? 1 : 0;
        p->warpSize = 32;
        p->sharedMemPerBlock = dev.get_info<sycl::info::device::local_mem_size>();
        p->l2CacheSize = (int) dev.get_info<sycl::info::device::global_mem_cache_size>();
        p->maxThreadsPerBlock = (int) dev.get_info<sycl::info::device::max_work_group_size>();
        p->canMapHostMemory = 1;
        p->unifiedAddressing = 1;
    }, cudaErrorInvalidValue);
}
inline cudaError_t cudaDeviceGetAttribute(int *v, cudaDeviceAttr a, int d) {
    cudaDeviceProp p;
    const cudaError_t e = cudaGetDeviceProperties(&p, d);
    if (e != cudaSuccess) return e;
    switch (a) {
        case cudaDevAttrMultiProcessorCount: *v = p.multiProcessorCount; break;
        case cudaDevAttrClockRate: *v = p.clockRate; break;
        case cudaDevAttrComputeCapabilityMajor: *v = p.major; break;
        case cudaDevAttrComputeCapabilityMinor: *v = p.minor; break;
        case cudaDevAttrIntegrated: *v = p.integrated; break;
        case cudaDevAttrMaxSharedMemoryPerBlockOptin: *v = (int) p.sharedMemPerBlock; break;
    }
    return cudaSuccess;
}
inline cudaError_t cudaDeviceSynchronize() {
    return strata_sycl_compat::guard([] { dpct::get_current_device().queues_wait_and_throw(); });
}
inline cudaError_t cudaMemGetInfo(size_t *free_b, size_t *total_b) {
    return strata_sycl_compat::guard([&] { dpct::get_current_device().get_memory_info(*free_b, *total_b); });
}
inline cudaError_t cudaRuntimeGetVersion(int *v) { *v = CUDART_VERSION; return cudaSuccess; }
inline cudaError_t cudaDriverGetVersion(int *v) { *v = CUDART_VERSION; return cudaSuccess; }
// Peer access: Level Zero devices of one context reach each other's USM; nothing to enable.
inline cudaError_t cudaDeviceCanAccessPeer(int *can, int a, int b) {
    return strata_sycl_compat::guard([&] {
        auto &da = dpct::dev_mgr::instance().get_device((unsigned) a);
        auto &db = dpct::dev_mgr::instance().get_device((unsigned) b);
        *can = da.ext_oneapi_can_access_peer(db) ? 1 : 0;
    });
}
inline cudaError_t cudaDeviceEnablePeerAccess(int, unsigned = 0) { return cudaSuccess; }
inline cudaError_t cudaDeviceDisablePeerAccess(int) { return cudaSuccess; }
inline cudaError_t cudaDeviceGetStreamPriorityRange(int *lo, int *hi) { *lo = 0; *hi = 0; return cudaSuccess; }
#define cudaDeviceScheduleSpin 0x1
#define cudaDeviceMapHost 0x8
inline cudaError_t cudaSetDeviceFlags(unsigned) { return cudaSuccess; }

// ---- streams: an in-order queue per stream; stream 0 is the device's default in-order queue ------------------
using cudaStream_t = dpct::queue_ptr;   // sycl::queue*
#define cudaStreamNonBlocking 0x1
#define cudaStreamDefault 0x0
#define cudaStreamLegacy ((cudaStream_t) nullptr)
namespace strata_sycl_compat {
inline sycl::queue &q(cudaStream_t s) { return s ? *s : dpct::get_in_order_queue(); }
}
inline cudaError_t cudaStreamCreate(cudaStream_t *s) {
    return strata_sycl_compat::guard([&] { *s = dpct::get_current_device().create_in_order_queue(); });
}
inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t *s, unsigned) { return cudaStreamCreate(s); }
inline cudaError_t cudaStreamCreateWithPriority(cudaStream_t *s, unsigned, int) { return cudaStreamCreate(s); }
inline cudaError_t cudaStreamDestroy(cudaStream_t s) {
    return strata_sycl_compat::guard([&] { if (s) dpct::get_current_device().destroy_queue(s); });
}
inline cudaError_t cudaStreamSynchronize(cudaStream_t s) {
    return strata_sycl_compat::guard([&] { strata_sycl_compat::q(s).wait_and_throw(); });
}
inline cudaError_t cudaStreamQuery(cudaStream_t s) {
    try {
        return strata_sycl_compat::q(s).ext_oneapi_empty() ? cudaSuccess : cudaErrorNotReady;
    } catch (const sycl::exception &) {
        return strata_sycl_compat::fail(cudaErrorUnknown);
    }
}

// ---- events: the barrier a record submits; timing from the host clock at completion -------------------------
// A record is a barrier on the queue (it completes when everything submitted before it has).  Timing: SYCL gives a
// barrier no profiling info unless the queue was made with profiling on, which the engine's queues are not, so an
// event keeps the host time at which it was first seen complete (Synchronize/Query); ElapsedTime subtracts those.
// That is coarse (it includes the host's polling delay) - the engine uses event times for diagnostics only.
struct strata_sycl_event {
    sycl::event ev;
    bool recorded = false;
    std::optional<std::chrono::steady_clock::time_point> done;
};
using cudaEvent_t = strata_sycl_event *;
#define cudaEventDefault 0x0
#define cudaEventBlockingSync 0x1
#define cudaEventDisableTiming 0x2
#define cudaEventRecordDefault 0x0
inline cudaError_t cudaEventCreate(cudaEvent_t *e) {
    *e = new strata_sycl_event();
    return cudaSuccess;
}
inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t *e, unsigned) { return cudaEventCreate(e); }
inline cudaError_t cudaEventDestroy(cudaEvent_t e) {
    delete e;
    return cudaSuccess;
}
inline cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t s = nullptr) {
    if (!e) return strata_sycl_compat::fail(cudaErrorInvalidValue);
    return strata_sycl_compat::guard([&] {
        e->ev = strata_sycl_compat::q(s).ext_oneapi_submit_barrier();
        e->recorded = true;
        e->done.reset();
    });
}
inline cudaError_t cudaEventRecordWithFlags(cudaEvent_t e, cudaStream_t s, unsigned) { return cudaEventRecord(e, s); }
inline cudaError_t cudaEventSynchronize(cudaEvent_t e) {
    if (!e) return strata_sycl_compat::fail(cudaErrorInvalidValue);
    return strata_sycl_compat::guard([&] {
        if (e->recorded) e->ev.wait_and_throw();
        if (!e->done) e->done = std::chrono::steady_clock::now();
    });
}
inline cudaError_t cudaEventQuery(cudaEvent_t e) {
    if (!e) return strata_sycl_compat::fail(cudaErrorInvalidValue);
    if (!e->recorded) return cudaSuccess;
    try {
        const auto st = e->ev.get_info<sycl::info::event::command_execution_status>();
        if (st != sycl::info::event_command_status::complete) return cudaErrorNotReady;
        if (!e->done) e->done = std::chrono::steady_clock::now();
        return cudaSuccess;
    } catch (const sycl::exception &) {
        return strata_sycl_compat::fail(cudaErrorUnknown);
    }
}
inline cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t a, cudaEvent_t b) {
    if (!a || !b || !a->done || !b->done) return strata_sycl_compat::fail(cudaErrorNotReady);
    *ms = std::chrono::duration<float, std::milli>(*b->done - *a->done).count();
    return cudaSuccess;
}
inline cudaError_t cudaStreamWaitEvent(cudaStream_t s, cudaEvent_t e, unsigned = 0) {
    if (!e) return strata_sycl_compat::fail(cudaErrorInvalidValue);
    if (!e->recorded) return cudaSuccess;
    return strata_sycl_compat::guard([&] { strata_sycl_compat::q(s).ext_oneapi_submit_barrier({e->ev}); });
}
using cudaHostFn_t = void (*)(void *);
inline cudaError_t cudaLaunchHostFunc(cudaStream_t s, cudaHostFn_t fn, void *data) {
    return strata_sycl_compat::guard([&] {
        strata_sycl_compat::q(s).submit([&](sycl::handler &h) { h.host_task([fn, data] { fn(data); }); });
    });
}

// ---- memory: device USM, host USM for pinned/mapped memory -----------------------------------------------------
enum cudaMemcpyKind : int {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4,
};
#define cudaHostAllocDefault 0x0
#define cudaHostAllocPortable 0x1
#define cudaHostAllocMapped 0x2
#define cudaHostAllocWriteCombined 0x4
#define cudaHostRegisterDefault 0x0
#define cudaHostRegisterPortable 0x1
#define cudaHostRegisterMapped 0x2
#define cudaHostRegisterReadOnly 0x8
template <class T> inline cudaError_t cudaMalloc(T **p, size_t n) {
    void *r = nullptr;
    const cudaError_t e = strata_sycl_compat::guard(
        [&] { r = sycl::malloc_device(n ? n : 1, dpct::get_in_order_queue()); }, cudaErrorMemoryAllocation);
    if (e != cudaSuccess) return e;
    if (!r) return strata_sycl_compat::fail(cudaErrorMemoryAllocation);
    *p = static_cast<T *>(r);
    return cudaSuccess;
}
inline cudaError_t cudaFree(void *p) {
    return strata_sycl_compat::guard([&] { if (p) sycl::free(p, dpct::get_in_order_queue()); });
}
// Host USM is page-locked and reachable from the device at the same address: every flag gets the same memory.
template <class T> inline cudaError_t cudaHostAlloc(T **p, size_t n, unsigned = 0) {
    void *r = nullptr;
    const cudaError_t e = strata_sycl_compat::guard(
        [&] { r = sycl::malloc_host(n ? n : 1, dpct::get_in_order_queue()); }, cudaErrorMemoryAllocation);
    if (e != cudaSuccess) return e;
    if (!r) return strata_sycl_compat::fail(cudaErrorMemoryAllocation);
    *p = static_cast<T *>(r);
    return cudaSuccess;
}
template <class T> inline cudaError_t cudaMallocHost(T **p, size_t n) { return cudaHostAlloc(p, n); }
inline cudaError_t cudaFreeHost(void *p) { return cudaFree(p); }
template <class T> inline cudaError_t cudaHostGetDevicePointer(T **dev, void *host, unsigned) {
    *dev = static_cast<T *>(host);   // host USM: the same address on the device
    return cudaSuccess;
}
// Registering ordinary memory: the oneAPI extension pins it for faster copies.  It does not make the memory
// reachable from kernels, so code that maps registered memory must use host USM instead (cudaHostAlloc).
inline cudaError_t cudaHostRegister(void *p, size_t n, unsigned) {
    return strata_sycl_compat::guard([&] {
        sycl::ext::oneapi::experimental::prepare_for_device_copy(p, n, dpct::get_in_order_queue());
    }, cudaErrorNotSupported);
}
// The engine asks this only to tell registered/host memory from ordinary memory (dma_batch.hpp h2d_split).
inline cudaError_t cudaHostGetFlags(unsigned *flags, void *p) {
    try {
        const auto t = sycl::get_pointer_type(p, dpct::get_in_order_queue().get_context());
        if (t != sycl::usm::alloc::host) return strata_sycl_compat::fail(cudaErrorInvalidValue);
        *flags = cudaHostAllocMapped;
        return cudaSuccess;
    } catch (const sycl::exception &) {
        return strata_sycl_compat::fail(cudaErrorInvalidValue);
    }
}
inline cudaError_t cudaHostUnregister(void *p) {
    return strata_sycl_compat::guard([&] {
        sycl::ext::oneapi::experimental::release_from_device_copy(p, dpct::get_in_order_queue());
    }, cudaErrorNotSupported);
}
inline cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t n, cudaMemcpyKind = cudaMemcpyDefault,
                                   cudaStream_t s = nullptr) {
    if (n == 0) return cudaSuccess;
    return strata_sycl_compat::guard([&] { strata_sycl_compat::q(s).memcpy(dst, src, n); });
}
inline cudaError_t cudaMemcpy(void *dst, const void *src, size_t n, cudaMemcpyKind = cudaMemcpyDefault) {
    if (n == 0) return cudaSuccess;
    return strata_sycl_compat::guard([&] { dpct::get_in_order_queue().memcpy(dst, src, n).wait_and_throw(); });
}
inline cudaError_t cudaMemsetAsync(void *dst, int v, size_t n, cudaStream_t s = nullptr) {
    if (n == 0) return cudaSuccess;
    return strata_sycl_compat::guard([&] { strata_sycl_compat::q(s).memset(dst, v, n); });
}
inline cudaError_t cudaMemset(void *dst, int v, size_t n) {
    if (n == 0) return cudaSuccess;
    return strata_sycl_compat::guard([&] { dpct::get_in_order_queue().memset(dst, v, n).wait_and_throw(); });
}
inline cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width,
                                     size_t height, cudaMemcpyKind = cudaMemcpyDefault, cudaStream_t s = nullptr) {
    if (width == 0 || height == 0) return cudaSuccess;
    return strata_sycl_compat::guard([&] {
        sycl::queue &qq = strata_sycl_compat::q(s);
        if (dpitch == width && spitch == width) {
            qq.memcpy(dst, src, width * height);
            return;
        }
        qq.ext_oneapi_memcpy2d(dst, dpitch, src, spitch, width, height);
    });
}
// USM addresses are unique across the devices of the context: a peer copy is an ordinary copy on the queue.
inline cudaError_t cudaMemcpyPeerAsync(void *dst, int, const void *src, int, size_t n, cudaStream_t s = nullptr) {
    return cudaMemcpyAsync(dst, src, n, cudaMemcpyDefault, s);
}
inline cudaError_t cudaMemcpyPeer(void *dst, int, const void *src, int, size_t n) {
    return cudaMemcpy(dst, src, n, cudaMemcpyDefault);
}

// ---- graphs: stream capture -> the oneAPI graph extension (record and replay on an in-order queue) -------------
namespace syclex = sycl::ext::oneapi::experimental;
struct strata_sycl_graph {
    std::optional<syclex::command_graph<syclex::graph_state::modifiable>> g;
};
struct strata_sycl_graph_exec {
    std::optional<syclex::command_graph<syclex::graph_state::executable>> g;
};
using cudaGraph_t = strata_sycl_graph *;
using cudaGraphExec_t = strata_sycl_graph_exec *;
using cudaGraphNode_t = void *;
enum cudaStreamCaptureMode : int {
    cudaStreamCaptureModeGlobal = 0,
    cudaStreamCaptureModeThreadLocal = 1,
    cudaStreamCaptureModeRelaxed = 2,
};
enum cudaGraphNodeType : int {
    cudaGraphNodeTypeKernel = 0,
    cudaGraphNodeTypeMemcpy = 1,
    cudaGraphNodeTypeMemset = 2,
    cudaGraphNodeTypeEmpty = 6,
};
namespace strata_sycl_compat {
inline std::unordered_map<sycl::queue *, strata_sycl_graph *> &capturing() {
    static std::unordered_map<sycl::queue *, strata_sycl_graph *> m;
    return m;
}
inline std::mutex &capture_mutex() {
    static std::mutex m;
    return m;
}
}  // namespace strata_sycl_compat
inline cudaError_t cudaStreamBeginCapture(cudaStream_t s, cudaStreamCaptureMode) {
    sycl::queue &qq = strata_sycl_compat::q(s);
    auto *g = new strata_sycl_graph();
    const cudaError_t e = strata_sycl_compat::guard([&] {
        g->g.emplace(qq.get_context(), qq.get_device());
        g->g->begin_recording(qq);
    }, cudaErrorStreamCaptureUnsupported);
    if (e != cudaSuccess) {
        delete g;
        return e;
    }
    std::lock_guard<std::mutex> lk(strata_sycl_compat::capture_mutex());
    strata_sycl_compat::capturing()[&qq] = g;
    return cudaSuccess;
}
inline cudaError_t cudaStreamEndCapture(cudaStream_t s, cudaGraph_t *out) {
    sycl::queue &qq = strata_sycl_compat::q(s);
    strata_sycl_graph *g = nullptr;
    {
        std::lock_guard<std::mutex> lk(strata_sycl_compat::capture_mutex());
        auto it = strata_sycl_compat::capturing().find(&qq);
        if (it == strata_sycl_compat::capturing().end()) return strata_sycl_compat::fail(cudaErrorInvalidValue);
        g = it->second;
        strata_sycl_compat::capturing().erase(it);
    }
    const cudaError_t e = strata_sycl_compat::guard([&] { g->g->end_recording(qq); },
                                                    cudaErrorStreamCaptureUnsupported);
    if (e != cudaSuccess) {
        delete g;
        return e;
    }
    *out = g;
    return cudaSuccess;
}
inline cudaError_t cudaGraphInstantiate(cudaGraphExec_t *x, cudaGraph_t g, unsigned long long = 0) {
    auto *ex = new strata_sycl_graph_exec();
    const cudaError_t e = strata_sycl_compat::guard([&] { ex->g.emplace(g->g->finalize()); },
                                                    cudaErrorStreamCaptureUnsupported);
    if (e != cudaSuccess) {
        delete ex;
        return e;
    }
    *x = ex;
    return cudaSuccess;
}
inline cudaError_t cudaGraphInstantiate(cudaGraphExec_t *x, cudaGraph_t g, cudaGraphNode_t *, char *, size_t) {
    return cudaGraphInstantiate(x, g, 0);
}
inline cudaError_t cudaGraphLaunch(cudaGraphExec_t x, cudaStream_t s) {
    if (!x || !x->g) return strata_sycl_compat::fail(cudaErrorInvalidValue);
    return strata_sycl_compat::guard([&] { strata_sycl_compat::q(s).ext_oneapi_graph(*x->g); });
}
inline cudaError_t cudaGraphUpload(cudaGraphExec_t, cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaGraphDestroy(cudaGraph_t g) {
    delete g;
    return cudaSuccess;
}
inline cudaError_t cudaGraphExecDestroy(cudaGraphExec_t x) {
    delete x;
    return cudaSuccess;
}
// Node inspection (diagnostics only): not available on the SYCL graph.
inline cudaError_t cudaGraphGetNodes(cudaGraph_t, cudaGraphNode_t *, size_t *n) {
    if (n) *n = 0;
    return strata_sycl_compat::fail(cudaErrorNotSupported);
}
inline cudaError_t cudaGraphNodeGetType(cudaGraphNode_t, cudaGraphNodeType *) {
    return strata_sycl_compat::fail(cudaErrorNotSupported);
}
struct cudaKernelNodeParams {
    void *func = nullptr;
};
inline cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t, cudaKernelNodeParams *) {
    return strata_sycl_compat::fail(cudaErrorNotSupported);
}
enum cudaGraphDependencyType : int { cudaGraphDependencyTypeDefault = 0, cudaGraphDependencyTypeProgrammatic = 1 };
struct cudaGraphEdgeData {
    unsigned char from_port = 0, to_port = 0, type = 0;
};
inline cudaError_t cudaGraphGetEdges_v2(cudaGraph_t, cudaGraphNode_t *, cudaGraphNode_t *, cudaGraphEdgeData *, size_t *n) {
    if (n) *n = 0;
    return strata_sycl_compat::fail(cudaErrorNotSupported);
}
inline cudaError_t cudaGraphGetEdges(cudaGraph_t g, cudaGraphNode_t *a, cudaGraphNode_t *b, cudaGraphEdgeData *e, size_t *n) {
    return cudaGraphGetEdges_v2(g, a, b, e, n);
}
