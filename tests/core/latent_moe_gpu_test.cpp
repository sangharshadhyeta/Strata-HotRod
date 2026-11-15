// tests/core/latent_moe_gpu_test.cpp - one layer of Edge0's latent MoE (D -> A -> U -> combine) on the GPU against a CPU
// reference that shares no kernel with it.  Branch hotrod-edge0, step 2.
//
//   latent_moe_gpu_test <Edge0 NVFP4 folder> [layer=0] [T=4]
//
// The folder is read only.  Real weights of the chosen layer (D/U from latent_dense_q8.bin, the experts from experts.bin, the
// group labels from labels.bin), T tokens of deterministic pseudo-random hidden states, K=10 distinct experts per token.
//
// Reference (CPU, double accumulation, no CUDA): D and U are Q8_0 matvecs in plain C++; each expert is our CPU NVFP4 path
// (native_expert_forward, ggml-cpu's dot products) fed the float latent vector, so the gate/up/down arithmetic is another
// implementation than the GPU kernels.  The GPU path quantizes its activations to int8 (q8_1) before D, A and U, the CPU
// reference does not, so the two differ by that quantization (about 1% in total); a wrong layout, group, offset or scale would
// differ by order 1.  Pass thresholds, fixed before the first run: relative RMS error under 5% at each of the three stages
// (the D output, the A output, the final combined output).
#include "strata/core/latent_dense.hpp"
#include "strata/core/latent_tab.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/latent.hpp"
#include "strata/latent/latent_moe.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-100s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "latent_moe_gpu_test: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

float half_to_float(uint16_t h) {
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    float v;
    if (e == 0) v = std::ldexp((float) m, -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = std::ldexp((float) (m | 0x400), (int) e - 25);
    return s ? -v : v;
}

// one Q8_0 row ({f16 d; int8 qs[32]} blocks, 34 bytes) dotted with a float vector
double q8_dot(const uint8_t* row, const float* x, int n) {
    double acc = 0;
    for (int b = 0; b < n / 32; ++b) {
        const uint8_t* blk = row + (size_t) b * 34;
        uint16_t hh;
        std::memcpy(&hh, blk, 2);
        const double d = half_to_float(hh);
        const int8_t* qs = (const int8_t*) (blk + 2);
        double s = 0;
        for (int i = 0; i < 32; ++i) s += (double) qs[i] * (double) x[b * 32 + i];
        acc += d * s;
    }
    return acc;
}

struct Rng {
    uint64_t s;
    uint32_t next() { s = s * 6364136223846793005ull + 1442695040888963407ull; return (uint32_t) (s >> 33); }
    float uniform() { return (float) next() / 2147483648.0f; }                 // [0, 1)
    float normal() { float a = 0; for (int i = 0; i < 12; ++i) a += uniform(); return a - 6.0f; }
};

struct Err { double rel_rms = 0, max_rel = 0; };
Err compare(const std::vector<float>& gpu, const std::vector<float>& ref) {
    double num = 0, den = 0, mx = 0, mr = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = (double) gpu[i] - (double) ref[i];
        num += d * d;
        den += (double) ref[i] * (double) ref[i];
        mx = std::max(mx, std::fabs(d));
        mr = std::max(mr, std::fabs((double) ref[i]));
    }
    return {std::sqrt(num / std::max(den, 1e-30)), mx / std::max(mr, 1e-30)};
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: latent_moe_gpu_test <Edge0 NVFP4 folder> [layer=0] [T=4]\n");
        return 2;
    }
    const std::string dir = argv[1];
    const int layer = argc > 2 ? std::atoi(argv[2]) : 0;
    const int T = argc > 3 ? std::atoi(argv[3]) : 4;
    std::printf("latent_moe_gpu_test: %s, layer %d, %d tokens\n", dir.c_str(), layer, T);

    std::string err;
    namespace kc = strata::kernels::cpu;
    if (!kc::expert_layout_load(dir, 48, 512, err)) { std::printf("layout refused: %s\n", err.c_str()); return 2; }
    const kc::ExpertLayout& lay = kc::expert_layout();
    if (!lay.latent) { std::printf("not a latent pack\n"); return 2; }
    const int64_t H = 2560, ELL = lay.expert_embd, I = lay.expert_ff;
    const int G = 8, E = (int) lay.n_expert, K = 10;
    const strata::latent::Geometry geo{H, ELL, I, G, E, K};
    const int cap = T * K;

    strata::core::LatentDense dense;
    if (!dense.open(dir + "/latent_dense_q8.bin", 48, G, H, ELL, err)) { std::printf("latent dense: %s\n", err.c_str()); return 2; }
    strata::core::LatentA apack;
    const int64_t blob = (int64_t) lay.blob_bytes(layer);
    if (!apack.open(dir + "/experts.bin", 48, E, blob, err)) { std::printf("experts.bin: %s\n", err.c_str()); return 2; }
    std::vector<int32_t> labels_all((size_t) 48 * E);
    {
        std::ifstream f(dir + "/labels.bin", std::ios::binary);
        f.read((char*) labels_all.data(), (std::streamsize) (labels_all.size() * 4));
        if (!f) { std::printf("labels.bin is too small\n"); return 2; }
    }
    const int32_t* labels = labels_all.data() + (size_t) layer * E;
    {
        bool in_range = true;
        std::vector<int> seen(G, 0);
        for (int e = 0; e < E; ++e) { in_range = in_range && labels[e] >= 0 && labels[e] < G; if (in_range) ++seen[labels[e]]; }
        std::string per;
        for (int g2 = 0; g2 < G; ++g2) per += (g2 ? "," : "") + std::to_string(seen[g2]);
        check(in_range, "layer " + std::to_string(layer) + "'s group labels are all in [0, 8) (experts per group: " + per + ")");
        if (!in_range) return 1;
    }

    // ---- the session-level table (LatentTab): everything read off the pack, nothing hardcoded to this model or this card
    strata::core::LatentTab tab;
    {
        namespace fs = std::filesystem;
        const uint64_t q8_bytes = fs::file_size(dir + "/latent_dense_q8.bin"), q4_bytes = fs::file_size(dir + "/latent_dense_q4.bin");
        int f = 0, g = 0;
        check(strata::core::latent_dense_identify(q8_bytes, 48, H, ELL, f, g) && f == 8 && g == G, "the Q8 file's size says: Q8_0, 8 groups");
        check(strata::core::latent_dense_identify(q4_bytes, 48, H, ELL, f, g) && f == 4 && g == G, "the Q4 file's size says: Q4_0, 8 groups");
        check(!strata::core::latent_dense_identify(q8_bytes + 1, 48, H, ELL, f, g), "a file of another size is not taken for a D/U file");
        check(!strata::core::latent_dense_identify(q8_bytes, 48, H, ELL + 32, f, g), "nor is it at another latent width");
        const uint64_t GiB = 1ull << 30;
        strata::core::LatentDenseChoice c;
        unsetenv("STRATA_LATENT_DENSE_FMT");
        unsetenv("STRATA_LATENT_DENSE");
        std::string e2;
        check(strata::core::latent_dense_choose(dir, 48, H, ELL, 12 * GiB, 0, c, e2) && c.fmt == 8, "12 GiB free: the Q8 file (it takes under half)");
        check(strata::core::latent_dense_choose(dir, 48, H, ELL, 5 * GiB, 0, c, e2) && c.fmt == 4, "5 GiB free: the Q4 file (Q8 would take more than half)");
        check(!strata::core::latent_dense_choose(dir, 48, H, ELL, 1 * GiB, 0, c, e2) && e2.find("GiB") != std::string::npos,
              "1 GiB free: refused, saying how much is needed (\"" + e2.substr(0, 60) + "...\")");
        check(strata::core::latent_dense_choose(dir, 48, H, ELL, 5 * GiB, 80, c, e2) && c.fmt == 8, "the share is a flag: 80% of 5 GiB takes the Q8 file");
        setenv("STRATA_LATENT_DENSE_FMT", "4", 1);
        check(strata::core::latent_dense_choose(dir, 48, H, ELL, 12 * GiB, 0, c, e2) && c.fmt == 4, "STRATA_LATENT_DENSE_FMT=4 forces the Q4 file");
        unsetenv("STRATA_LATENT_DENSE_FMT");
        size_t fb = 0, tb = 0;
        ck(cudaSetDevice(0), "cudaSetDevice");
        ck(cudaMemGetInfo(&fb, &tb), "cudaMemGetInfo");
        std::string e3;
        const bool loaded = tab.load(dir, 48, H, ELL, E, (uint64_t) fb, e3);
        check(loaded, "LatentTab loads: " + (loaded ? std::string("Q") + (tab.fmt() == 8 ? "8" : "4") + ", " + std::to_string(tab.groups()) + " groups, " +
                                                       std::to_string((double) tab.device_bytes() / 1073741824.0).substr(0, 4) + " GiB on the GPU" : e3));
        if (!loaded) return 1;
        check(tab.groups() == G && tab.labels().size() == (size_t) 48 * E, "the group count and the labels come out of the files");
        check(strata::core::latent_shared_d() == tab.d() && strata::core::latent_shared_u() == tab.u(), "the device pair is published for the verify and prompt paths");
        // the upload is the file: layer 0 and the last layer, D and U, compared byte for byte with the mapped file
        bool same = true;
        for (const int64_t l : {int64_t(0), int64_t(47)}) {
            std::vector<uint8_t> hd(tab.dense().d_bytes()), hu(tab.dense().u_bytes());
            ck(cudaMemcpy(hd.data(), tab.d() + (size_t) l * tab.dense().d_bytes(), hd.size(), cudaMemcpyDeviceToHost), "tab D back");
            ck(cudaMemcpy(hu.data(), tab.u() + (size_t) l * tab.dense().u_bytes(), hu.size(), cudaMemcpyDeviceToHost), "tab U back");
            same = same && std::memcmp(hd.data(), tab.dense().d(l), hd.size()) == 0 && std::memcmp(hu.data(), tab.dense().u(l), hu.size()) == 0;
        }
        check(same, "layers 0 and 47 of D and U on the GPU are the file's bytes");
    }

    // inputs: hidden states, K distinct experts per token, positive weights summing to 1
    Rng rng{0x9e3779b97f4a7c15ull ^ (uint64_t) layer};
    std::vector<float> x((size_t) T * H);
    for (auto& v : x) v = 0.7f * rng.normal();
    std::vector<int32_t> topk((size_t) T * K);
    std::vector<float> w((size_t) T * K);
    for (int t = 0; t < T; ++t) {
        std::vector<int32_t> pool(E);
        for (int e = 0; e < E; ++e) pool[e] = e;
        for (int j = 0; j < K; ++j) { std::swap(pool[j], pool[j + rng.next() % (E - j)]); topk[(size_t) t * K + j] = pool[j]; }
        float s = 0;
        for (int j = 0; j < K; ++j) { w[(size_t) t * K + j] = 0.1f + rng.uniform(); s += w[(size_t) t * K + j]; }
        for (int j = 0; j < K; ++j) w[(size_t) t * K + j] /= s;
    }

    // ---------------------------------------------------------------- the CPU reference
    std::vector<float> ref_xl((size_t) T * G * ELL), ref_yl((size_t) cap * ELL), ref_y((size_t) cap * H), ref_out((size_t) T * H, 0.0f);
    {
        const uint8_t* D = dense.d(layer);
        const uint8_t* U = dense.u(layer);
        const size_t drow = dense.d_row_bytes(), urow = dense.u_row_bytes();
        for (int t = 0; t < T; ++t)
            for (int g = 0; g < G; ++g)
                for (int r = 0; r < ELL; ++r)
                    ref_xl[((size_t) t * G + g) * ELL + r] = (float) q8_dot(D + ((size_t) g * ELL + r) * drow, x.data() + (size_t) t * H, (int) H);
        std::vector<float> h((size_t) lay.fmt[(size_t) layer].n_ff);
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < K; ++j) {
                const int r = t * K + j, e = topk[r], g = labels[e];
                kc::native_expert_forward(lay.fmt[(size_t) layer], apack.blob(layer, e), ref_xl.data() + ((size_t) t * G + g) * ELL, h.data(),
                                          ref_yl.data() + (size_t) r * ELL);
                for (int o = 0; o < H; ++o)
                    ref_y[(size_t) r * H + o] = (float) q8_dot(U + ((size_t) g * H + o) * urow, ref_yl.data() + (size_t) r * ELL, (int) ELL);
            }
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < K; ++j)
                for (int o = 0; o < H; ++o) ref_out[(size_t) t * H + o] += w[(size_t) t * K + j] * ref_y[((size_t) t * K + j) * H + o];
    }
    std::printf("  CPU reference done (|xl| max %.3f, |out| max %.3f)\n", (double) *std::max_element(ref_xl.begin(), ref_xl.end()),
                (double) *std::max_element(ref_out.begin(), ref_out.end()));

    // ---------------------------------------------------------------- the GPU path
    ck(cudaSetDevice(0), "cudaSetDevice");
    cudaStream_t st;
    ck(cudaStreamCreate(&st), "stream");
    auto dalloc = [&](size_t n, const char* what) { void* p = nullptr; ck(cudaMalloc(&p, n), what); return p; };
    float* dX = (float*) dalloc(x.size() * 4, "x");
    ck(cudaMemcpy(dX, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x copy");
    uint8_t* dD = (uint8_t*) dalloc(dense.d_bytes(), "D");
    uint8_t* dU = (uint8_t*) dalloc(dense.u_bytes(), "U");
    ck(cudaMemcpy(dD, dense.d(layer), dense.d_bytes(), cudaMemcpyHostToDevice), "D copy");
    ck(cudaMemcpy(dU, dense.u(layer), dense.u_bytes(), cudaMemcpyHostToDevice), "U copy");
    std::vector<unsigned long long> ptrs((size_t) E, 0ull);
    std::vector<void*> blobs;
    for (int r = 0; r < cap; ++r) {
        const int e = topk[r];
        if (ptrs[e]) continue;
        void* p = dalloc((size_t) blob, "expert blob");
        ck(cudaMemcpy(p, apack.blob(layer, e), (size_t) blob, cudaMemcpyHostToDevice), "blob copy");
        ptrs[e] = (unsigned long long) p;
        blobs.push_back(p);
    }
    // native_expert_grouped's gate/up kernel reads each group's NVFP4 tail scales at `pointer + tail_off` before it looks at the group's entry
    // count (compute-sanitizer: a read at 0x1d8800 = tail_off of a null pointer), so a group with no entries still needs a VALID blob
    // address.  The groups no token routes to point at the first used blob; their entry count is 0, so nothing of it is used.
    for (int e = 0; e < E; ++e)
        if (!ptrs[e]) ptrs[e] = (unsigned long long) blobs.front();
    const strata::latent::Plan plan = strata::latent::build_plan(geo, topk.data(), labels, T);
    auto* dPtr = (unsigned long long*) dalloc((size_t) E * 8, "ptr");
    auto* dStart = (int32_t*) dalloc(plan.start.size() * 4, "start");
    auto* dN = (int32_t*) dalloc(4, "n");
    auto* dDst = (int32_t*) dalloc((size_t) cap * 4, "dst");
    auto* dTok = (int32_t*) dalloc((size_t) cap * 4, "tok");
    const int32_t n_groups = E;
    ck(cudaMemcpy(dPtr, ptrs.data(), (size_t) E * 8, cudaMemcpyHostToDevice), "ptr copy");
    ck(cudaMemcpy(dStart, plan.start.data(), plan.start.size() * 4, cudaMemcpyHostToDevice), "start copy");
    ck(cudaMemcpy(dN, &n_groups, 4, cudaMemcpyHostToDevice), "n copy");
    ck(cudaMemcpy(dDst, plan.dst.data(), (size_t) cap * 4, cudaMemcpyHostToDevice), "dst copy");
    ck(cudaMemcpy(dTok, plan.tok.data(), (size_t) cap * 4, cudaMemcpyHostToDevice), "tok copy");
    const strata::latent::DevicePlan dp{dPtr, dStart, dN, dDst, dTok, cap, E};

    strata::latent::Buffers b;
    b.xq = dalloc((size_t) T * (H / 32) * 36, "xq");
    b.xlq = dalloc((size_t) T * G * (ELL / 32) * 36, "xlq");
    b.uq = dalloc((size_t) cap * (ELL / 32) * 36, "uq");
    b.scratch = dalloc(strata::kernels::native_expert_scratch_bytes(cap, I), "scratch");
    b.xl = (float*) dalloc((size_t) T * G * ELL * 4, "xl");
    b.yl = (float*) dalloc((size_t) cap * ELL * 4, "yl");
    b.uin = (float*) dalloc((size_t) cap * ELL * 4, "uin");
    b.uout = (float*) dalloc((size_t) cap * H * 4, "uout");
    b.y = (float*) dalloc((size_t) cap * H * 4, "y");
    ck(cudaMemset(b.xl, 0, (size_t) T * G * ELL * 4), "xl clear");
    ck(cudaMemset(b.yl, 0, (size_t) cap * ELL * 4), "yl clear");
    ck(cudaMemset(b.y, 0, (size_t) cap * H * 4), "y clear");

    strata::latent::Inputs in;
    in.d_q8 = dD;
    in.u_q8 = dU;
    in.x = dX;
    in.topk = topk.data();
    in.labels = labels;
    in.w = w.data();
    in.T = T;
    in.fmt = dense.fmt();
    const bool ran = strata::latent::run_layer(geo, in, dp, b, (void*) st);
    ck(cudaStreamSynchronize(st), "run_layer sync");
    ck(cudaGetLastError(), "run_layer");
    check(ran, "run_layer returned true (the native NVFP4 expert layout at 1792/640 is supported)");
    if (!ran) return 1;

    std::vector<float> gpu_xl(ref_xl.size()), gpu_yl(ref_yl.size()), gpu_y(ref_y.size()), gpu_out((size_t) T * H);
    ck(cudaMemcpy(gpu_xl.data(), b.xl, gpu_xl.size() * 4, cudaMemcpyDeviceToHost), "xl back");
    ck(cudaMemcpy(gpu_yl.data(), b.yl, gpu_yl.size() * 4, cudaMemcpyDeviceToHost), "yl back");
    ck(cudaMemcpy(gpu_y.data(), b.y, gpu_y.size() * 4, cudaMemcpyDeviceToHost), "y back");
    strata::latent::combine(geo, w.data(), gpu_y.data(), gpu_out.data(), T);

    auto report = [&](const char* name, const std::vector<float>& a, const std::vector<float>& r) {
        const Err e = compare(a, r);
        char line[200];
        std::snprintf(line, sizeof line, "%s: relative RMS error %.3e, largest error / largest value %.3e", name, e.rel_rms, e.max_rel);
        check(e.rel_rms < 0.05, line);
    };
    report("D   (hidden -> 8 latent vectors per token)", gpu_xl, ref_xl);
    report("A   (the NVFP4 experts at the latent width)", gpu_yl, ref_yl);
    report("U   (per-group up-projection, per route)", gpu_y, ref_y);
    report("out (weighted sum over the 10 routed experts)", gpu_out, ref_out);

    // ---- the decode path's own form (verify.cpp): the plan's activation rows come in as plain token rows, the device rewrites them to
    //      token*G + group (latent_plan_rows), A runs on those rows, then U runs for every routed entry (latent_u_grouped)
    {
        std::vector<int32_t> tok_plain((size_t) cap);
        for (int q = 0; q < cap; ++q) tok_plain[(size_t) q] = plan.dst[(size_t) q] / K;
        const int32_t counts[4] = {0, cap, 0, 0};   // counts[1] = the GPU entries
        auto* dCounts = (int32_t*) dalloc(16, "counts");
        auto* dIds = (int32_t*) dalloc((size_t) cap * 4, "ids");
        auto* dLabels = (int32_t*) dalloc((size_t) E * 4, "labels");
        auto* dTok2 = (int32_t*) dalloc((size_t) cap * 4, "tok2");
        ck(cudaMemcpy(dCounts, counts, 16, cudaMemcpyHostToDevice), "counts copy");
        ck(cudaMemcpy(dIds, topk.data(), (size_t) cap * 4, cudaMemcpyHostToDevice), "ids copy");
        ck(cudaMemcpy(dLabels, labels, (size_t) E * 4, cudaMemcpyHostToDevice), "labels copy");
        ck(cudaMemcpy(dTok2, tok_plain.data(), (size_t) cap * 4, cudaMemcpyHostToDevice), "tok2 copy");
        strata::kernels::latent_plan_rows(dTok2, dDst, dCounts, dIds, dLabels, E, 0, G, K, (void*) st);
        ck(cudaStreamSynchronize(st), "latent_plan_rows sync");
        std::vector<int32_t> tok_gpu((size_t) cap);
        ck(cudaMemcpy(tok_gpu.data(), dTok2, (size_t) cap * 4, cudaMemcpyDeviceToHost), "tok2 back");
        check(tok_gpu == plan.tok, "latent_plan_rows: every entry's activation row is token*8 + group, exactly the host plan's");

        auto* yl2 = (float*) dalloc((size_t) cap * ELL * 4, "yl2");
        auto* y2 = (float*) dalloc((size_t) cap * H * 4, "y2");
        void* uq2 = dalloc((size_t) cap * (ELL / 32) * 36, "uq2");
        const strata::kernels::NativeExpertLayout La = strata::kernels::native_expert_layout(40, 40, (int) ELL, (int) I);
        strata::kernels::native_expert_grouped(La, dPtr, dStart, dN, dDst, dTok2, E, cap, b.xlq, b.scratch, yl2, (void*) st, E);
        strata::kernels::quantize_q8_1_rows(yl2, cap, ELL, uq2, (void*) st);
        strata::kernels::latent_u_grouped(dU, (const uint8_t*) uq2, dIds, dLabels, E, cap, (int) ELL, (int) H, y2, (void*) st, dense.fmt());
        ck(cudaStreamSynchronize(st), "decode form sync");
        ck(cudaGetLastError(), "decode form");
        std::vector<float> yl2_h(ref_yl.size()), y2_h(ref_y.size());
        ck(cudaMemcpy(yl2_h.data(), yl2, yl2_h.size() * 4, cudaMemcpyDeviceToHost), "yl2 back");
        ck(cudaMemcpy(y2_h.data(), y2, y2_h.size() * 4, cudaMemcpyDeviceToHost), "y2 back");
        report("A   in the decode form (rows rewritten on the device)", yl2_h, ref_yl);
        report("U   via latent_u_grouped (the decode path's U kernel)", y2_h, ref_y);
        check(compare(y2_h, gpu_y).rel_rms < 0.05, "latent_u_grouped agrees with run_layer's per-group U within the same threshold");
        {   // the by-group kernel (G > 0) must give the per-entry kernel's bits
            auto* y3 = (float*) dalloc((size_t) cap * H * 4, "y3");
            ck(cudaMemset(y3, 0xFF, (size_t) cap * H * 4), "y3 clear");
            strata::kernels::latent_u_grouped(dU, (const uint8_t*) uq2, dIds, dLabels, E, cap, (int) ELL, (int) H, y3, (void*) st, dense.fmt(), 8);
            ck(cudaStreamSynchronize(st), "by-group sync");
            ck(cudaGetLastError(), "by-group");
            std::vector<float> y3_h(y2_h.size());
            ck(cudaMemcpy(y3_h.data(), y3, y3_h.size() * 4, cudaMemcpyDeviceToHost), "y3 back");
            size_t diff = 0;
            for (size_t i = 0; i < y3_h.size(); ++i) diff += std::memcmp(&y3_h[i], &y2_h[i], 4) != 0;
            std::printf("  U   by-group kernel vs per-entry kernel: %zu of %zu values differ in their bits\n", diff, y3_h.size());
            check(diff == 0, "latent_u_grouped by group gives the per-entry kernel's bits");
            cudaFree(y3);
        }
    }

    for (void* p : blobs) cudaFree(p);
    std::printf(g_fail ? "latent_moe_gpu_test: %d FAILED\n" : "latent_moe_gpu_test: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
