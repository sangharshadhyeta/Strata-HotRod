// tests/tools/subfit_fit_math_test.cpp - SPEC-substitute-fit stage 1b: the fit's linear algebra on synthetic data (CPU only).
// Covers: Cholesky solve, the ridge regression recovers a known map, the Jacobi eigen-decomposition, the randomized rank truncation,
// the held-out split, the scalar fit, and the candidate search (best cosine, the sample's own expert excluded).
#include "fit_math.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace subfit_fit;

namespace {
int failures = 0;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); \
            ++failures;                                                        \
        }                                                                      \
    } while (0)

std::vector<double> randn(size_t n, std::mt19937_64& rng, double sd = 1.0) {
    std::normal_distribution<double> g(0.0, sd);
    std::vector<double> v(n);
    for (double& x : v) x = g(rng);
    return v;
}

double frob(const std::vector<double>& a) {
    double s = 0;
    for (double x : a) s += x * x;
    return std::sqrt(s);
}

double diff_frob(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i) s += (a[i] - b[i]) * (a[i] - b[i]);
    return std::sqrt(s);
}

// y = M h with M (N x F) row-major
std::vector<double> matvec(const std::vector<double>& M, int N, int F, const double* h) {
    std::vector<double> y((size_t) N, 0.0);
    for (int i = 0; i < N; ++i)
        for (int a = 0; a < F; ++a) y[(size_t) i] += M[(size_t) i * F + a] * h[a];
    return y;
}

void test_cholesky() {
    // A = [[4,2],[2,3]], B = A X with X = [[1,0],[0,2]]
    std::vector<double> A = {4, 2, 2, 3};
    std::vector<double> B = {4, 4, 2, 6};   // A X: column 0 = (4,2), column 1 = (4,6)
    CHECK(cholesky_solve(A, 2, B, 2), "SPD matrix solves");
    CHECK(std::fabs(B[0] - 1) < 1e-12 && std::fabs(B[1]) < 1e-12 && std::fabs(B[2]) < 1e-12 && std::fabs(B[3] - 2) < 1e-12,
          "Cholesky recovers X");
    std::vector<double> bad = {1, 2, 2, 1};   // indefinite
    std::vector<double> rhs = {1, 1};
    CHECK(!cholesky_solve(bad, 2, rhs, 1), "an indefinite matrix is refused");
}

void test_ridge_recovers_map() {
    std::mt19937_64 rng(7);
    const int n = 400, F = 6, N = 9;
    std::vector<double> M = randn((size_t) N * F, rng);
    std::vector<double> H = randn((size_t) n * F, rng);
    std::vector<double> R((size_t) n * N, 0.0);
    for (int i = 0; i < n; ++i) {
        const std::vector<double> y = matvec(M, N, F, &H[(size_t) i * F]);
        for (int j = 0; j < N; ++j) R[(size_t) i * N + j] = y[(size_t) j];
    }
    std::vector<double> Mhat;
    CHECK(ridge_fit(H, R, n, F, N, 1e-9, Mhat), "ridge solves");
    CHECK(diff_frob(Mhat, M) / frob(M) < 1e-6, "noise-free data: the known map is recovered");

    // with noise the fit is still close, and a large lambda shrinks it
    std::vector<double> Rn = R;
    std::normal_distribution<double> g(0.0, 0.5);
    for (double& v : Rn) v += g(rng);
    std::vector<double> M1, M2;
    CHECK(ridge_fit(H, Rn, n, F, N, 1e-6, M1), "ridge with noise");
    CHECK(ridge_fit(H, Rn, n, F, N, 1e4, M2), "ridge with a large lambda");
    CHECK(diff_frob(M1, M) / frob(M) < 0.1, "noisy data: the map is still close");
    CHECK(frob(M2) < 0.5 * frob(M1), "a large lambda shrinks the map");
}

void test_eig() {
    // symmetric 3x3 with known eigenvalues 1, 3, 5 (A = Q diag Q^T for a rotation Q): use [[3,1,0],[1,3,0],[0,0,5]] -> 4, 2, 5
    std::vector<double> A = {3, 1, 0, 1, 3, 0, 0, 0, 5};
    std::vector<double> ev, V;
    eig_sym(A, 3, ev, V);
    CHECK(std::fabs(ev[0] - 5) < 1e-9 && std::fabs(ev[1] - 4) < 1e-9 && std::fabs(ev[2] - 2) < 1e-9, "eigenvalues, descending");
    // A v = lambda v for each column
    bool ok = true;
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 3; ++i) {
            double av = 0;
            for (int k = 0; k < 3; ++k) av += A[(size_t) i * 3 + k] * V[(size_t) k * 3 + j];
            if (std::fabs(av - ev[(size_t) j] * V[(size_t) i * 3 + j]) > 1e-9) ok = false;
        }
    CHECK(ok, "eigenvectors satisfy A v = lambda v");
}

void test_truncate() {
    std::mt19937_64 rng(11);
    const int N = 40, F = 25;
    // an exact rank-3 matrix: truncation to 3 reproduces it, to 2 does not
    std::vector<double> P = randn((size_t) N * 3, rng), Q = randn((size_t) 3 * F, rng);
    std::vector<double> M((size_t) N * F, 0.0);
    for (int i = 0; i < N; ++i)
        for (int a = 0; a < F; ++a)
            for (int k = 0; k < 3; ++k) M[(size_t) i * F + a] += P[(size_t) i * 3 + k] * Q[(size_t) k * F + a];
    std::vector<double> T3, T2;
    truncate_rank(M, N, F, 3, 5, T3);
    truncate_rank(M, N, F, 2, 5, T2);
    CHECK(diff_frob(T3, M) / frob(M) < 1e-8, "rank-3 matrix is reproduced by a rank-3 truncation");
    CHECK(diff_frob(T2, M) / frob(M) > 1e-2, "rank-2 truncation of a rank-3 matrix loses something");

    // a full-rank matrix: the error of a rank-r truncation is close to the sum of the discarded singular values squared (eig of M^T M)
    std::vector<double> G((size_t) F * F, 0.0);
    std::vector<double> Mr = randn((size_t) N * F, rng);
    for (int a = 0; a < F; ++a)
        for (int b = 0; b < F; ++b) {
            double s = 0;
            for (int i = 0; i < N; ++i) s += Mr[(size_t) i * F + a] * Mr[(size_t) i * F + b];
            G[(size_t) a * F + b] = s;
        }
    std::vector<double> ev, V;
    eig_sym(G, F, ev, V);
    const int r = 5;
    double tail = 0;
    for (int j = r; j < F; ++j) tail += ev[(size_t) j];
    std::vector<double> Tr;
    truncate_rank(Mr, N, F, r, 3, Tr);
    const double err2 = diff_frob(Tr, Mr) * diff_frob(Tr, Mr);
    CHECK(err2 <= 1.25 * tail && err2 >= 0.8 * tail, "randomized rank-5 error is near the optimal tail energy");
    truncate_rank(Mr, N, F, F, 3, Tr);
    CHECK(diff_frob(Tr, Mr) < 1e-12, "full rank returns the matrix");
}

void test_scalar_and_split() {
    std::vector<double> ys = {1, 2, 3, 4};
    std::vector<double> ye = {2, 4, 6, 8};   // ye = 2 ys
    CHECK(std::fabs(best_scalar(ys, ye, 2, 2) - 2.0) < 1e-12, "best scalar of an exact 2x");
    std::vector<double> zero = {0, 0, 0, 0};
    CHECK(best_scalar(zero, ye, 2, 2) == 0.0, "no energy: scalar 0");
    CHECK(half_of(0) == 0 && half_of(1) == 1 && half_of(10) == 0 && half_of(7) == 1, "even / odd held-out split");
    CHECK(std::fabs(cosine(ys.data(), ye.data(), 4) - 1.0) < 1e-12, "cosine of parallel vectors");
}

// The candidate search: the best cosine against y_e over candidate outputs, the sample's own expert excluded.
// (The same rule the tool applies; here with plain vectors so the test does not need the pack.)
int best_candidate(const std::vector<std::vector<double>>& outs, const std::vector<int>& ids, const std::vector<double>& ye, int own) {
    int best = -1;
    double bc = -2.0;
    for (size_t i = 0; i < outs.size(); ++i) {
        if (ids[i] == own) continue;
        const double c = cosine(outs[i].data(), ye.data(), (int) ye.size());
        if (c > bc) {
            bc = c;
            best = ids[i];
        }
    }
    return best;
}

void test_candidates() {
    const int N = 8;
    std::vector<double> ye(N);
    for (int i = 0; i < N; ++i) ye[(size_t) i] = std::sin((double) i + 1.0);
    std::vector<std::vector<double>> outs;
    std::vector<int> ids;
    for (int c = 0; c < 6; ++c) {   // noise candidates with ids 10..15
        std::vector<double> o(N);
        for (int i = 0; i < N; ++i) o[(size_t) i] = std::cos((double) (c * 7 + i));
        outs.push_back(o);
        ids.push_back(10 + c);
    }
    outs.push_back(ye);   // candidate 99 matches y_e exactly
    ids.push_back(99);
    CHECK(best_candidate(outs, ids, ye, 5) == 99, "the best cosine candidate is chosen");
    CHECK(best_candidate(outs, ids, ye, 99) != 99, "the sample's own expert is excluded from the search");
}

void test_mode_arithmetic() {
    CHECK(std::fabs(residual_all(0.75) - 0.5) < 1e-12, "ALL: sqrt(1 - g)");
    CHECK(std::fabs(residual_all(0.0) - 1.0) < 1e-12, "no gain: residual 1");
    CHECK(std::fabs(residual_all(1.0)) < 1e-12, "full gain: residual 0");
    CHECK(std::fabs(residual_all(1.7)) < 1e-12 && std::fabs(residual_all(-0.3) - 1.0) < 1e-12, "g is clipped to [0, 1]");
    CHECK(std::fabs(residual_dynamic(0.8, 0.75) - 0.6) < 1e-12, "DYNAMIC: (1 - f) + f sqrt(1 - g) = 0.2 + 0.4");
    CHECK(std::fabs(residual_dynamic(1.0, 0.75) - residual_all(0.75)) < 1e-12, "f = 1 is ALL");
    CHECK(std::fabs(residual_dynamic(0.0, 0.75) - 1.0) < 1e-12, "f = 0 is no correction");
    // the gain kept by DYNAMIC against ALL is f
    const double ra = residual_all(0.6), rd = residual_dynamic(0.8, 0.6);
    CHECK(std::fabs((1.0 - rd) / (1.0 - ra) - 0.8) < 1e-12, "gain kept by DYNAMIC is f of ALL's");
    CHECK(std::fabs(moves_share(3, 10) - 0.3) < 1e-12, "moves share: refused over moves");
    CHECK(moves_share(12, 10) == 1.0, "moves share is at most 1");
    CHECK(moves_share(0, 0) == 0.0, "no moves: 0");
}

}  // namespace

int main() {
    test_mode_arithmetic();
    test_cholesky();
    test_ridge_recovers_map();
    test_eig();
    test_truncate();
    test_scalar_and_split();
    test_candidates();
    if (failures == 0) {
        std::printf("subfit_fit_math_test: all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "subfit_fit_math_test: %d check(s) failed\n", failures);
    return 1;
}
