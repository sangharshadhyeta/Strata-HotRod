// Strata. Stand-ins in the DeepSeek engine's router (route_publish with a Standin, ds4_kernels.cu; Strata-HotRod's rank rule):
// no Standin is the router as it was; ranks below floor - 1 never swap; rank floor - 1 swaps within `edge` logits; ranks from the floor
// swap when weight x logit gap <= budget; a swap takes the best card-resident expert not picked, never twice; the weights stay the
// picks' own; the counters and the token's used / risk flags.  Needs a GPU (no model).
#include "ds4_kernels.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace strata::ds4;

static int fails = 0;
#define CHECK(c)                                                                     \
    do {                                                                             \
        if (!(c)) {                                                                  \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);                 \
            ++fails;                                                                 \
        }                                                                            \
    } while (0)

constexpr int N = 384, K = 6, DIM = 8;
constexpr float kScale = 2.5f;

struct Out {
    int idx[K];
    float w[K];
    Standin st;
};

// the routing of one token: logits, bias 0, the card table `res` (ids on the card), with standin (or none)
static Out run(const std::vector<float>& logits, const std::vector<int>& res, const Standin* cfg) {
    std::vector<float> h_log(1024, -30.0f), h_bias(N, 0.0f), h_x(DIM, 0.0f);
    for (int i = 0; i < N; ++i) h_log[(size_t) i] = logits[(size_t) i];
    std::vector<unsigned long long> h_vram(N, 0ull);
    for (int e : res) h_vram[(size_t) e] = 0x1000ull + (unsigned long long) e * 64;
    float *d_log, *d_bias, *d_x, *d_w;
    int* d_idx;
    unsigned long long *d_vram, *d_plan;
    Standin* d_st = nullptr;
    cudaMalloc(&d_log, 1024 * 4);
    cudaMalloc(&d_bias, N * 4);
    cudaMalloc(&d_x, DIM * 4);
    cudaMalloc(&d_w, kPicks * 4);
    cudaMalloc(&d_idx, kPicks * 4);
    cudaMalloc(&d_vram, N * 8);
    cudaMalloc(&d_plan, kPicks * 8);
    cudaMemcpy(d_log, h_log.data(), 1024 * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias.data(), N * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), DIM * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vram, h_vram.data(), N * 8, cudaMemcpyHostToDevice);
    if (cfg != nullptr) {
        cudaMalloc(&d_st, sizeof(Standin));
        cudaMemcpy(d_st, cfg, sizeof(Standin), cudaMemcpyHostToDevice);
    }
    route_publish(d_log, d_bias, N, K, 1, kScale, d_vram, d_x, DIM, d_idx, d_w, d_plan, nullptr, 0, 0, nullptr, 0, 0, 0, d_st);
    Out o{};
    cudaMemcpy(o.idx, d_idx, K * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(o.w, d_w, K * 4, cudaMemcpyDeviceToHost);
    if (d_st != nullptr) cudaMemcpy(&o.st, d_st, sizeof(Standin), cudaMemcpyDeviceToHost);
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::printf("FAIL: CUDA %s\n", cudaGetErrorString(e));
        ++fails;
    }
    cudaFree(d_log); cudaFree(d_bias); cudaFree(d_x); cudaFree(d_w); cudaFree(d_idx); cudaFree(d_vram); cudaFree(d_plan);
    if (d_st != nullptr) cudaFree(d_st);
    return o;
}

static Standin cfg(int floor, float edge, float budget) {
    Standin s;
    std::memset(&s, 0, sizeof s);
    s.floor = floor;
    s.edge = edge;
    s.budget = budget;
    return s;
}

int main() {
    // the token's picks are experts 0..5 (logits 5.0 down to 4.5), none on the card; on the card: 10 (4.0), 20 (3.9), 30 (3.8),
    // 40 (3.7), 50 (1.0); the rest far below
    std::vector<float> lg(N, -10.0f);
    for (int i = 0; i < K; ++i) lg[(size_t) i] = 5.0f - 0.1f * (float) i;
    lg[10] = 4.0f; lg[20] = 3.9f; lg[30] = 3.8f; lg[40] = 3.7f; lg[50] = 1.0f;
    const std::vector<int> res = {10, 20, 30, 40, 50};

    const Out plain = run(lg, res, nullptr);
    for (int r = 0; r < K; ++r) CHECK(plain.idx[r] == r);

    {   // everything open: ranks 0-1 exact; rank 2 within the edge (gap 0.8 <= 1); ranks 3-5 take the next residents in score order
        const Standin c = cfg(3, 1.0f, 1e9f);
        const Out o = run(lg, res, &c);
        CHECK(o.idx[0] == 0 && o.idx[1] == 1);
        CHECK(o.idx[2] == 10 && o.idx[3] == 20 && o.idx[4] == 30 && o.idx[5] == 40);
        for (int r = 0; r < K; ++r) CHECK(std::fabs(o.w[r] - plain.w[r]) < 1e-6f);   // the weights stay the picks' own
        CHECK(o.st.offered[0] == 0 && o.st.offered[1] == 0);
        for (int r = 2; r < K; ++r) CHECK(o.st.offered[r] == 1 && o.st.taken[r] == 1);
        CHECK(o.st.used[0] == 1 && o.st.risk[0] > 0.0f);
    }
    {   // edge 0: rank 2 exact as well; budget 0: nothing swaps (every gap is > 0); offered counted, used 0
        const Standin c = cfg(3, 0.0f, 0.0f);
        const Out o = run(lg, res, &c);
        for (int r = 0; r < K; ++r) CHECK(o.idx[r] == r);
        CHECK(o.st.offered[2] == 0);
        for (int r = 3; r < K; ++r) CHECK(o.st.offered[r] == 1 && o.st.taken[r] == 0);
        CHECK(o.st.used[0] == 0 && o.st.risk[0] == 0.0f);
    }
    {   // a budget between rank 5's cost and rank 3's: rank 3 (heavier, gap 4.7 - 4.0) stays; the best resident then goes to the
        // first rank that passes; the rank after it takes the next resident
        const float g3 = 4.7f - 4.0f, g4 = 4.6f - 4.0f, g5 = 4.5f - 3.9f;
        const float c3 = plain.w[3] * g3, c4 = plain.w[4] * g4, c5 = plain.w[5] * g5;
        CHECK(c3 > c4 && c4 > c5);
        const Standin c = cfg(3, 0.0f, 0.5f * (c3 + c4));
        const Out o = run(lg, res, &c);
        CHECK(o.idx[3] == 3);
        CHECK(o.idx[4] == 10);
        CHECK(o.idx[5] == 20);
        CHECK(o.st.taken[3] == 0 && o.st.taken[4] == 1 && o.st.taken[5] == 1);
    }
    {   // a pick already on the card is never offered; no resident left to swap in: the pick stays
        std::vector<int> res2 = {3};
        const Standin c = cfg(3, 1.0f, 1e9f);
        const Out o = run(lg, res2, &c);
        CHECK(o.idx[3] == 3 && o.st.offered[3] == 0);
        CHECK(o.idx[4] == 4 && o.idx[5] == 5);   // the only resident is picked already
    }
    if (fails == 0) std::printf("standin_test: all passed\n");
    return fails == 0 ? 0 : 1;
}
