// tests/dsv41/lalearn_test.cu - the DeepSeek learned look-ahead (src/dsv41/ds4_lalearn.cu) on synthetic routers.  The true router of
// layer l is W[l] (E x H); the plain guess for layer m is P[m] = W[m] + a rank-4 error, so the plain top-k is imperfect.  Each window
// uses one input x (H values) for every layer, the labels are W[l] x, and the learner's forward for source s and distance d corrects
// the plain guess of layer s+d.  After ~3000 windows the leader's hits (all-time, slot counters) over the last 1000 windows must beat
// the plain router's for d = 1.  Needs a GPU, no model.  Built, not run by the build (ds4_lalearn_test).
#include "../../src/dsv41/ds4_lalearn.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace la = strata::ds4::la;

namespace {

constexpr int kL = 6, kH = 256, kE = 384, kRow = 1024;
constexpr int kWindows = 3000, kLast = 1000;
constexpr int kLo = 3;

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "lalearn_test: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

}  // namespace

int main() {
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    // the true routers W[l] (E x H, entries of variance 1/H: logits of unit inputs have variance 1), the plain routers P[m] = W[m] + the
    // rank-4 error U V / (2 sqrt(H)) (logit error variance 1): the plain guess of layer m is wrong in a structured way a rank-4 correction can undo
    std::vector<std::vector<float>> W(kL, std::vector<float>((size_t) kE * kH)), P(kL, std::vector<float>((size_t) kE * kH));
    const float sw = 1.0f / std::sqrt((float) kH);
    for (int m = 0; m < kL; ++m) {
        for (float& v : W[(size_t) m]) v = sw * nd(rng);
        std::vector<float> U((size_t) kE * 4), V((size_t) 4 * kH);
        for (float& v : U) v = nd(rng);
        for (float& v : V) v = nd(rng);
        for (int e = 0; e < kE; ++e)
            for (int h = 0; h < kH; ++h) {
                float acc = 0.0f;
                for (int k = 0; k < 4; ++k) acc += U[(size_t) e * 4 + k] * V[(size_t) k * kH + h];
                P[(size_t) m][(size_t) e * kH + h] = W[(size_t) m][(size_t) e * kH + h] + acc / (2.0f * std::sqrt((float) kH));
            }
    }
    // the learner, on the current device; nothing resident (every table entry zero)
    std::string msg;
    if (!la::alloc(kL, kH, kE, kLo, msg)) {
        std::fprintf(stderr, "lalearn_test: alloc: %s\n", msg.c_str());
        return 2;
    }
    std::vector<unsigned long long*> dtab((size_t) kL, nullptr);
    std::vector<unsigned long long> zeros((size_t) kE, 0ull);
    for (int l = 0; l < kL; ++l) {
        check(cudaMalloc(&dtab[(size_t) l], kE * sizeof(unsigned long long)), "table");
        check(cudaMemcpy(dtab[(size_t) l], zeros.data(), kE * sizeof(unsigned long long), cudaMemcpyHostToDevice), "table copy");
    }
    if (!la::set_resident_tables((const unsigned long long* const*) dtab.data(), kL)) {
        std::fprintf(stderr, "lalearn_test: set_resident_tables failed\n");
        return 2;
    }
    cudaStream_t cs = nullptr;
    check(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking), "stream");
    // device buffers: the input, one row of the truth (labels) and one plain row per distance
    float *dx = nullptr, *dtruth = nullptr, *dgl[la::kNd] = {};
    check(cudaMalloc(&dx, kH * sizeof(float)), "x");
    check(cudaMalloc(&dtruth, kRow * sizeof(float)), "truth");
    for (int dd = 0; dd < la::kNd; ++dd) check(cudaMalloc(&dgl[dd], kRow * sizeof(float)), "gl");
    std::vector<float> x((size_t) kH), truth((size_t) kE), plain((size_t) kE);
    auto matvec = [&](const std::vector<float>& M, std::vector<float>& out) {
        for (int e = 0; e < kE; ++e) {
            float acc = 0.0f;
            for (int h = 0; h < kH; ++h) acc += M[(size_t) e * kH + h] * x[(size_t) h];
            out[(size_t) e] = acc;
        }
    };
    unsigned long long hits0[4] = {}, hits1[4] = {};
    unsigned long long picks0 = 0, picks1 = 0;
    int leader0 = 0, leader1 = 0;
    for (int w = 0; w < kWindows; ++w) {
        if (w == kWindows - kLast) {
            if (!la::counters(0, 1, leader0, picks0, hits0)) {
                std::fprintf(stderr, "lalearn_test: counters\n");
                return 2;
            }
        }
        for (float& v : x) v = nd(rng);
        check(cudaMemcpyAsync(dx, x.data(), kH * sizeof(float), cudaMemcpyHostToDevice, cs), "x upload");
        la::pre_launch(cs);
        for (int l = 0; l < kL; ++l) {   // each layer's label: its true router on x
            matvec(W[(size_t) l], truth);
            std::vector<float> row(kRow, 0.0f);
            for (int e = 0; e < kE; ++e) row[(size_t) e] = truth[(size_t) e];
            check(cudaMemcpyAsync(dtruth, row.data(), kRow * sizeof(float), cudaMemcpyHostToDevice, cs), "truth upload");
            la::label(l, dtruth, 1, cs);
        }
        for (int s = 0; s < kL; ++s)   // each source's forward for the distances whose target layer exists
            for (int dd = 0; dd < la::kNd; ++dd) {
                const int dist = la::kDist[dd];
                if (s + dist >= kL) break;
                matvec(P[(size_t) (s + dist)], plain);
                std::vector<float> row(kRow, 0.0f);
                for (int e = 0; e < kE; ++e) row[(size_t) e] = plain[(size_t) e];
                check(cudaMemcpyAsync(dgl[dd], row.data(), kRow * sizeof(float), cudaMemcpyHostToDevice, cs), "gl upload");
                la::forward(s, dist, dx, dgl[dd], 1, cs);
            }
        la::post_launch(cs, 1);
    }
    check(cudaStreamSynchronize(cs), "sync");
    if (!la::counters(0, 1, leader1, picks1, hits1)) {
        std::fprintf(stderr, "lalearn_test: counters\n");
        return 2;
    }
    const unsigned long long plain_hits = hits1[0] - hits0[0];
    const unsigned long long lead_hits = hits1[leader1] - hits0[leader1];
    const unsigned long long n_picks = picks1 - picks0;
    std::printf("lalearn_test: d=1 over the last %d windows: %llu picks, plain %llu hits, leader (slot %d) %llu hits\n", kLast, n_picks,
                plain_hits, leader1, lead_hits);
    std::printf("%s", la::report().c_str());
    la::release();
    for (int l = 0; l < kL; ++l) cudaFree(dtab[(size_t) l]);
    cudaFree(dx);
    cudaFree(dtruth);
    for (int dd = 0; dd < la::kNd; ++dd) cudaFree(dgl[dd]);
    cudaStreamDestroy(cs);
    if (n_picks == 0 || lead_hits <= plain_hits) {
        std::printf("lalearn_test: FAILED (the leader did not beat the plain router)\n");
        return 1;
    }
    std::printf("lalearn_test: OK\n");
    return 0;
}
