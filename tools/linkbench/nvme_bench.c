// nvme_bench.c - the storage and RAM links of this machine, measured with the access shapes the engine uses.
//   nvme_bench FILE seq    SECONDS                         one thread, O_DIRECT, 1 MiB chunks in order
//   nvme_bench FILE rand   SECONDS THREADS CHUNK_KIB       O_DIRECT random aligned reads, THREADS in flight
//   nvme_bench FILE mmap   SECONDS THREADS CHUNK_KIB ADV   mmap random reads of CHUNK_KIB (ADV 0 normal, 1 MADV_RANDOM);
//                                                          reports the device bytes read against the bytes used
//   nvme_bench - ram       SECONDS THREADS                 memcpy bandwidth (256 MiB per thread)
#define _GNU_SOURCE
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static unsigned long long sectors_read(void) {
    FILE* f = fopen("/proc/diskstats", "r");
    char line[512];
    unsigned long long s = 0;
    while (f && fgets(line, sizeof line, f)) {
        unsigned a, b; char name[64]; unsigned long long rd, rdm, rds;
        if (sscanf(line, "%u %u %63s %llu %llu %llu", &a, &b, name, &rd, &rdm, &rds) == 6 && !strcmp(name, "nvme0n1")) s = rds;
    }
    if (f) fclose(f);
    return s;
}

typedef struct {
    int fd, id, adv;
    size_t chunk, span;
    double dur;
    const uint8_t* map;
    uint64_t bytes, ops, sink;
    double lat, lat_max;
} Arg;

static uint64_t rnd(uint64_t* s) { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; return *s; }

static void* t_direct(void* p) {
    Arg* a = p; void* buf; posix_memalign(&buf, 4096, a->chunk);
    uint64_t s = 0x9E3779B97F4A7C15ull * (a->id + 1);
    double end = now() + a->dur;
    while (now() < end) {
        off_t off = (off_t) (rnd(&s) % (a->span / 4096)) * 4096;
        double t0 = now();
        ssize_t r = pread(a->fd, buf, a->chunk, off);
        double dt = now() - t0;
        if (r != (ssize_t) a->chunk) break;
        a->bytes += r; a->ops++; a->lat += dt; if (dt > a->lat_max) a->lat_max = dt;
    }
    free(buf); return NULL;
}

static void* t_seq(void* p) {
    Arg* a = p; void* buf; posix_memalign(&buf, 4096, a->chunk);
    off_t off = 0; double end = now() + a->dur;
    while (now() < end && (size_t) off + a->chunk < a->span) {
        ssize_t r = pread(a->fd, buf, a->chunk, off);
        if (r != (ssize_t) a->chunk) break;
        off += r; a->bytes += r; a->ops++;
    }
    free(buf); return NULL;
}

static void* t_mmap(void* p) {
    Arg* a = p; uint8_t* buf = malloc(a->chunk);
    uint64_t s = 0xD1B54A32D192ED03ull * (a->id + 1), sink = 0;
    double end = now() + a->dur;
    while (now() < end) {
        size_t off = (size_t) (rnd(&s) % ((a->span - a->chunk) / 4096)) * 4096;
        double t0 = now();
        memcpy(buf, a->map + off, a->chunk);
        double dt = now() - t0;
        sink += buf[0] + buf[a->chunk - 1];
        a->bytes += a->chunk; a->ops++; a->lat += dt; if (dt > a->lat_max) a->lat_max = dt;
    }
    a->sink = sink; free(buf); return NULL;
}

static void* t_ram(void* p) {
    Arg* a = p; size_t n = 256u << 20;
    uint8_t* x = malloc(n); uint8_t* y = malloc(n); memset(x, 1, n); memset(y, 2, n);
    double end = now() + a->dur;
    while (now() < end) { memcpy(y, x, n); a->bytes += n; a->ops++; y[0] ^= 1; }
    a->sink = y[0]; free(x); free(y); return NULL;
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: see the header of nvme_bench.c\n"); return 2; }
    const char* path = argv[1]; const char* mode = argv[2];
    double dur = atof(argv[3]);
    int threads = argc > 4 ? atoi(argv[4]) : 1;
    size_t chunk = argc > 5 ? (size_t) (atof(argv[5]) * 1024) : (1u << 20);
    int adv = argc > 6 ? atoi(argv[6]) : 0;
    int fd = -1; size_t span = 0; const uint8_t* map = NULL;
    if (strcmp(path, "-")) {
        int flags = O_RDONLY | (strcmp(mode, "mmap") ? O_DIRECT : 0);
        fd = open(path, flags);
        if (fd < 0) { perror("open"); return 1; }
        struct stat st; fstat(fd, &st); span = (size_t) st.st_size & ~(size_t) 4095;
        if (!strcmp(mode, "mmap")) {
            posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);          // start cold
            map = mmap(NULL, span, PROT_READ, MAP_SHARED, fd, 0);
            if (map == MAP_FAILED) { perror("mmap"); return 1; }
            madvise((void*) map, span, adv ? MADV_RANDOM : MADV_NORMAL);
        }
    }
    if (!strcmp(mode, "seq")) { threads = 1; chunk = 1u << 20; }
    pthread_t th[64]; Arg a[64];
    unsigned long long s0 = sectors_read(); double t0 = now();
    for (int i = 0; i < threads; i++) {
        a[i] = (Arg){.fd = fd, .id = i, .adv = adv, .chunk = chunk, .span = span, .dur = dur, .map = map};
        void* (*fn)(void*) = !strcmp(mode, "seq") ? t_seq : !strcmp(mode, "rand") ? t_direct : !strcmp(mode, "mmap") ? t_mmap : t_ram;
        pthread_create(&th[i], NULL, fn, &a[i]);
    }
    uint64_t bytes = 0, ops = 0; double lat = 0, lat_max = 0;
    for (int i = 0; i < threads; i++) { pthread_join(th[i], NULL); bytes += a[i].bytes; ops += a[i].ops; lat += a[i].lat; if (a[i].lat_max > lat_max) lat_max = a[i].lat_max; }
    double el = now() - t0; unsigned long long s1 = sectors_read();
    double dev = (double) (s1 - s0) * 512.0;
    printf("%-5s thr=%-2d chunk=%-6.0fKiB adv=%d | %.2f GB/s  %.0f ops/s", mode, threads, chunk / 1024.0, adv, bytes / el / 1e9, ops / el);
    if (ops && lat > 0) printf("  mean %.0f us  max %.0f us", lat / ops * 1e6, lat_max * 1e6);
    if (!strcmp(mode, "mmap")) printf(" | device read %.2f GB/s = %.2fx the bytes used", dev / el / 1e9, dev / (double) bytes);
    printf("\n");
    return 0;
}
