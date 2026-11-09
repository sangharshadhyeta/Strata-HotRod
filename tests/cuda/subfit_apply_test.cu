// tests/cuda/subfit_apply_test.cu - SPEC-substitute-fit stage 2 step 3: subfit_apply against a CPU reference (GPU; compiled, not run on the
// serving machine).  Random parts, x, A, B, c, LUT, orig, ids and flags; the kernel's output must equal the reference in both modes:
//   - mode all: every row/slot with orig >= 0, LUT >= 0 is corrected (flags ignored);
//   - mode dynamic: only the flagged ones.
// Rows with orig = -1, or LUT = -1, must come out untouched; orig and flag are reset to -1 / 0 by the launch.  Exit 77 without a CUDA device.
#include "strata/kernels/subfit_apply.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using strata::kernels::SubfitApplyArgs;

namespace {

int failures = 0;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); \
            ++failures;                                                        \
        }                                                                      \
    } while (0)

constexpr int kN = 256, kR = 16, kK = 10, kRows = 6, kSlots = 12, kLayer = 1, kLayers = 2, kE = 512;

uint16_t to16(float f) { return __half_as_ushort(__float2half(f)); }
float from16(uint16_t h) { return __half2float(__ushort_as_half(h)); }

// CPU reference of one layer's apply, exactly the kernel's rule
void reference(std::vector<float>& parts, const std::vector<float>& x, const std::vector<int32_t>& ids, const std::vector<int16_t>& orig,
               const std::vector<uint8_t>& flag, const std::vector<int16_t>& lut, const std::vector<uint16_t>& A, const std::vector<uint16_t>& B,
               const std::vector<float>& c, int mode) {
    for (int row = 0; row < kRows; ++row) {
        std::vector<float> z(kR, 0.0f);
        for (int m = 0; m < kR; ++m)
            for (int j = 0; j < kN; ++j) z[(size_t) m] += from16(A[(size_t) m * kN + j]) * x[(size_t) row * kN + j];
        for (int k = 0; k < kK; ++k) {
            const int idx = row * kK + k;
            const int e = orig[(size_t) idx], s = ids[(size_t) idx];
            int slot = -1;
            if (e >= 0 && e < kE && s >= 0 && s < kE) slot = lut[(size_t) e * kE + s];
            if (slot < 0) continue;
            if (!(mode == 2 || flag[(size_t) idx] != 0)) continue;
            float* p = &parts[((size_t) row * kK + k) * kN];
            for (int j = 0; j < kN; ++j) {
                float v = 0.0f;
                for (int m = 0; m < kR; ++m) v += from16(B[((size_t) slot * kN + j) * kR + m]) * z[(size_t) m];
                p[j] = c[(size_t) slot] * p[j] + v;
            }
        }
    }
}

bool run_case(int mode, uint32_t seed, bool fitted_layer) {
    std::mt19937 g(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);
    std::vector<float> parts((size_t) kRows * kK * kN), x((size_t) kRows * kN);
    for (float& v : parts) v = nd(g);
    for (float& v : x) v = nd(g);
    std::vector<uint16_t> A((size_t) kR * kN), B((size_t) kSlots * kN * kR);
    for (auto& v : A) v = to16(0.1f * nd(g));
    for (auto& v : B) v = to16(0.1f * nd(g));
    std::vector<float> c(kSlots), resid(kSlots, 0.5f);
    for (float& v : c) v = 0.5f + ud(g);
    // the full LUT of the layer (-1 everywhere but a few fitted pairs); the orig / ids / flags of the rows
    std::vector<int16_t> lut((size_t) kLayers * kE * kE, -1);
    std::vector<int16_t> lut_l((size_t) kE * kE, -1);
    for (int i = 0; i < kSlots; ++i) {
        const int e = (int) (g() % kE), s = (int) (g() % kE);
        lut_l[(size_t) e * kE + s] = (int16_t) i;
    }
    if (fitted_layer)
        for (size_t i = 0; i < lut_l.size(); ++i) lut[(size_t) kLayer * kE * kE + i] = lut_l[i];
    std::vector<int16_t> orig((size_t) kRows * kK, -1);
    std::vector<int32_t> ids((size_t) kRows * kK);
    std::vector<uint8_t> flag((size_t) kRows * kK, 0);
    for (int row = 0; row < kRows; ++row)
        for (int k = 0; k < kK; ++k) {
            const size_t idx = (size_t) row * kK + k;
            ids[idx] = (int32_t) (g() % kE);
            const int r = ud(g) < 0.6f ? (int) (g() % kE) : -1;   // some slots substituted (orig = the pick)
            orig[idx] = (int16_t) r;
            flag[idx] = (uint8_t) (ud(g) < 0.5f ? 1 : 0);
            // make some substituted slots fitted: place (orig, ids) on a LUT entry of the layer
            if (r >= 0 && ud(g) < 0.5f) {
                const int slot = (int) (g() % kSlots);
                ids[idx] = (int32_t) (g() % kE);
                lut_l[(size_t) r * kE + ids[idx]] = (int16_t) slot;
                if (fitted_layer) lut[(size_t) kLayer * kE * kE + (size_t) r * kE + ids[idx]] = (int16_t) slot;
            }
        }
    // the reference's view of the layer's LUT
    const std::vector<int16_t> ref_lut = fitted_layer ? lut_l : std::vector<int16_t>((size_t) kE * kE, (int16_t) -1);
    std::vector<float> want = parts;
    reference(want, x, ids, orig, flag, ref_lut, A, B, c, mode);
    // device
    float *d_parts = nullptr, *d_x = nullptr, *d_c = nullptr, *d_resid = nullptr;
    int32_t* d_ids = nullptr;
    int16_t *d_orig = nullptr, *d_lut = nullptr;
    uint8_t* d_flag = nullptr;
    uint16_t *d_A = nullptr, *d_B = nullptr;
    int32_t* d_nfit = nullptr;
    strata::kernels::SubfitTables t;
    bool ok = strata::kernels::subfit_tables_alloc(t, kLayers, kSlots, kN, kR);
    CHECK(ok, "the tables allocate");
    if (!ok) return false;
    cudaMalloc(&d_parts, parts.size() * sizeof(float));
    cudaMalloc(&d_x, x.size() * sizeof(float));
    cudaMalloc(&d_ids, ids.size() * sizeof(int32_t));
    cudaMalloc(&d_orig, orig.size() * sizeof(int16_t));
    cudaMalloc(&d_flag, flag.size());
    cudaMalloc(&d_lut, lut.size() * sizeof(int16_t));
    cudaMalloc(&d_nfit, sizeof(int32_t));
    cudaMemcpy(d_parts, parts.data(), parts.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ids, ids.data(), ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_orig, orig.data(), orig.size() * sizeof(int16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flag, flag.data(), flag.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lut, lut.data(), lut.size() * sizeof(int16_t), cudaMemcpyHostToDevice);
    const int32_t nf = fitted_layer ? 1 : 0;
    cudaMemcpy(d_nfit, &nf, sizeof nf, cudaMemcpyHostToDevice);
    // the table: A (layer 0 slot), B and c, resid
    cudaMemcpy(t.A + (size_t) kLayer * kR * kN, A.data(), A.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(t.B, B.data(), B.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(t.c, c.data(), c.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(t.lut + (size_t) kLayer * kE * kE, lut_l.data(), lut_l.size() * sizeof(int16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(t.nfit + kLayer, &nf, sizeof nf, cudaMemcpyHostToDevice);
    (void) d_resid;
    (void) d_c;
    SubfitApplyArgs a;
    a.parts = d_parts;
    a.x = d_x;
    a.ids = d_ids;
    a.orig = d_orig;
    a.flag = d_flag;
    a.lut = t.lut + (size_t) kLayer * kE * kE;
    a.A = t.A + (size_t) kLayer * kR * kN;
    a.B = t.B;
    a.c = t.c;
    a.nfit = t.nfit + kLayer;
    a.r = kR;
    a.N = kN;
    a.K = kK;
    a.n = kRows;
    a.mode = mode;
    strata::kernels::subfit_apply(a, nullptr);
    cudaError_t err = cudaDeviceSynchronize();
    CHECK(err == cudaSuccess, "the kernel runs");
    std::vector<float> got(parts.size());
    cudaMemcpy(got.data(), d_parts, got.size() * sizeof(float), cudaMemcpyDeviceToHost);
    std::vector<int16_t> orig_after(orig.size());
    std::vector<uint8_t> flag_after(flag.size());
    cudaMemcpy(orig_after.data(), d_orig, orig.size() * sizeof(int16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(flag_after.data(), d_flag, flag.size(), cudaMemcpyDeviceToHost);
    double worst = 0.0;
    for (size_t i = 0; i < got.size(); ++i) worst = std::fmax(worst, std::fabs((double) got[i] - (double) want[i]));
    CHECK(worst < 1e-3, "the device output equals the reference");
    // rows with orig = -1 or LUT = -1 are untouched: the reference left them, so the same comparison covers it; check directly too
    bool untouched = true;
    for (int row = 0; row < kRows; ++row)
        for (int k = 0; k < kK; ++k) {
            const int idx = row * kK + k;
            const int e = orig[(size_t) idx], s = ids[(size_t) idx];
            const bool fit = e >= 0 && e < kE && s >= 0 && s < kE && lut_l[(size_t) e * kE + s] >= 0 && fitted_layer;
            const bool use = fit && (mode == 2 || flag[(size_t) idx] != 0);
            if (!use)
                for (int j = 0; j < kN; ++j)
                    if (got[((size_t) row * kK + k) * kN + j] != parts[((size_t) row * kK + k) * kN + j]) untouched = false;
        }
    CHECK(untouched, "rows with no fitted pair (or not flagged in dynamic mode) are untouched");
    bool reset = true;
    for (int16_t v : orig_after) reset = reset && v == -1;
    for (uint8_t v : flag_after) reset = reset && v == 0;
    CHECK(reset, "orig and flag are consumed (reset)");
    cudaFree(d_parts);
    cudaFree(d_x);
    cudaFree(d_ids);
    cudaFree(d_orig);
    cudaFree(d_flag);
    cudaFree(d_lut);
    cudaFree(d_nfit);
    strata::kernels::subfit_tables_free(t);
    return true;
}

}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1) {
        std::printf("subfit_apply_test: no CUDA device, skipped\n");
        return 77;
    }
    run_case(2, 1, true);    // all
    run_case(1, 2, true);    // dynamic
    run_case(1, 3, false);   // dynamic, layer without fit: a no-op
    run_case(2, 4, false);   // all, layer without fit: a no-op
    if (failures == 0) std::printf("subfit_apply_test: all checks passed\n");
    return failures == 0 ? 0 : 1;
}
