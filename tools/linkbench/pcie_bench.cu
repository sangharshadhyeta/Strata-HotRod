// pcie_bench.cu - the RAM <-> VRAM link and the NVMe -> pinned RAM -> VRAM chain, measured with the transfer shapes the
// engine uses (one expert is 0.9 to 2.7 MiB; a swap round is up to 96 of them).
//   pcie_bench                     host/device copies by size, pinned and pageable, both directions
//   pcie_bench FILE THREADS KIB    the chain: THREADS workers each pread (O_DIRECT, random) KIB from FILE into a pinned
//                                  buffer and copy it to the GPU; reports the rate of the whole chain
#define _GNU_SOURCE
#include <cuda_runtime.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>
#include <chrono>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static double copy_gbps(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, double min_total_gb) {
    cudaStream_t s; CK(cudaStreamCreate(&s));
    int iters = (int) (min_total_gb * 1e9 / bytes) + 1;
    for (int i = 0; i < 3; i++) CK(cudaMemcpyAsync(dst, src, bytes, kind, s));
    CK(cudaStreamSynchronize(s));
    double t0 = now();
    for (int i = 0; i < iters; i++) CK(cudaMemcpyAsync(dst, src, bytes, kind, s));
    CK(cudaStreamSynchronize(s));
    double dt = now() - t0;
    CK(cudaStreamDestroy(s));
    return (double) bytes * iters / dt / 1e9;
}

static void link_state(const char* when) {
    FILE* f = popen("nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv,noheader", "r");
    char b[128] = {0};
    if (f) { if (fgets(b, sizeof b, f)) { b[strcspn(b, "\n")] = 0; } pclose(f); }
    printf("GPU link %s: PCIe gen,width = %s\n", when, b);
}

int main(int argc, char** argv) {
    CK(cudaSetDevice(0));
    const size_t MAXB = 512u << 20;
    void* d; CK(cudaMalloc(&d, MAXB));
    if (argc >= 4) {                                   // the chain
        const char* path = argv[1]; int T = atoi(argv[2]); size_t chunk = (size_t) (atof(argv[3]) * 1024);
        int fd = open(path, O_RDONLY | O_DIRECT); if (fd < 0) { perror("open"); return 1; }
        struct stat st; fstat(fd, &st); size_t span = (size_t) st.st_size & ~(size_t) 4095;
        std::vector<std::thread> th; std::vector<double> bytes(T, 0.0);
        double dur = 6.0, t0 = now();
        for (int i = 0; i < T; i++) th.emplace_back([&, i]() {
            void* h; CK(cudaHostAlloc(&h, chunk, cudaHostAllocDefault));
            void* dv; CK(cudaMalloc(&dv, chunk));
            cudaStream_t s; CK(cudaStreamCreate(&s));
            uint64_t r = 0x9E3779B97F4A7C15ull * (i + 1);
            while (now() - t0 < dur) {
                r ^= r << 13; r ^= r >> 7; r ^= r << 17;
                off_t off = (off_t) (r % (span / 4096 - chunk / 4096)) * 4096;
                if (pread(fd, h, chunk, off) != (ssize_t) chunk) break;
                CK(cudaMemcpyAsync(dv, h, chunk, cudaMemcpyHostToDevice, s));
                CK(cudaStreamSynchronize(s));
                bytes[i] += chunk;
            }
        });
        for (auto& t : th) t.join();
        double el = now() - t0, tot = 0; for (double b : bytes) tot += b;
        printf("chain NVMe(O_DIRECT)->pinned->VRAM  threads=%-2d chunk=%.0f KiB | %.2f GB/s\n", T, chunk / 1024.0, tot / el / 1e9);
        return 0;
    }
    void* hp; CK(cudaHostAlloc(&hp, MAXB, cudaHostAllocDefault));
    void* hg = malloc(MAXB); memset(hg, 1, MAXB); memset(hp, 1, MAXB);
    // raise the link out of its idle state and read what it trains to
    cudaStream_t s0; CK(cudaStreamCreate(&s0));
    for (int i = 0; i < 40; i++) CK(cudaMemcpyAsync(d, hp, MAXB, cudaMemcpyHostToDevice, s0));
    link_state("under load");
    CK(cudaStreamSynchronize(s0));
    printf("%-12s %12s %12s %12s\n", "bytes", "pinned H2D", "pageable H2D", "pinned D2H");
    size_t sizes[] = {512u << 10, 896u << 10, 2u << 20, 8u << 20, 64u << 20, 256u << 20, 512u << 20};
    for (size_t b : sizes) {
        double a = copy_gbps(d, hp, b, cudaMemcpyHostToDevice, 2.0);
        double p = copy_gbps(d, hg, b, cudaMemcpyHostToDevice, 1.0);
        double r = copy_gbps(hp, d, b, cudaMemcpyDeviceToHost, 2.0);
        printf("%-12.2f %9.2f GB/s %9.2f GB/s %9.2f GB/s   (%.2f MiB)\n", (double) b, a, p, r, b / 1048576.0);
    }
    // a swap round: 96 experts of 2 MiB, one stream, one synchronisation
    cudaStream_t s; CK(cudaStreamCreate(&s));
    for (int rep = 0; rep < 3; rep++) {
        double t0 = now();
        for (int i = 0; i < 96; i++) CK(cudaMemcpyAsync((char*) d + (size_t) i * (2u << 20), (char*) hp + (size_t) i * (2u << 20), 2u << 20, cudaMemcpyHostToDevice, s));
        CK(cudaStreamSynchronize(s));
        double ms = (now() - t0) * 1e3;
        if (rep == 2) printf("one swap round, 96 x 2 MiB pinned H2D, one stream: %.2f ms (%.2f GB/s)\n", ms, 96.0 * 2 * 1048576 / (ms * 1e-3) / 1e9);
    }
    // latency of a single synchronous 2 MiB copy
    double t0 = now();
    for (int i = 0; i < 200; i++) { CK(cudaMemcpy(d, hp, 2u << 20, cudaMemcpyHostToDevice)); }
    printf("single synchronous 2 MiB pinned H2D: %.0f us each\n", (now() - t0) / 200 * 1e6);
    return 0;
}
