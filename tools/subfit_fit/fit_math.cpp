// tools/subfit_fit/fit_math.cpp - the dense linear algebra of the offline fit (see fit_math.hpp).
#include "fit_math.hpp"

#include <algorithm>
#include <cmath>
#include <random>

namespace subfit_fit {

bool cholesky_solve(const std::vector<double>& A, int n, std::vector<double>& B, int m) {
    std::vector<double> L((size_t) n * n, 0.0);
    for (int j = 0; j < n; ++j) {
        double d = A[(size_t) j * n + j];
        for (int k = 0; k < j; ++k) d -= L[(size_t) j * n + k] * L[(size_t) j * n + k];
        if (!(d > 0.0) || !std::isfinite(d)) return false;
        const double ljj = std::sqrt(d);
        L[(size_t) j * n + j] = ljj;
        for (int i = j + 1; i < n; ++i) {
            double s = A[(size_t) i * n + j];
            for (int k = 0; k < j; ++k) s -= L[(size_t) i * n + k] * L[(size_t) j * n + k];
            L[(size_t) i * n + j] = s / ljj;
        }
    }
    // L y = b, then L^T x = y, for each column of B
    for (int c = 0; c < m; ++c) {
        for (int i = 0; i < n; ++i) {
            double s = B[(size_t) i * m + c];
            for (int k = 0; k < i; ++k) s -= L[(size_t) i * n + k] * B[(size_t) k * m + c];
            B[(size_t) i * m + c] = s / L[(size_t) i * n + i];
        }
        for (int i = n - 1; i >= 0; --i) {
            double s = B[(size_t) i * m + c];
            for (int k = i + 1; k < n; ++k) s -= L[(size_t) k * n + i] * B[(size_t) k * m + c];
            B[(size_t) i * m + c] = s / L[(size_t) i * n + i];
        }
    }
    return true;
}

bool ridge_fit(const std::vector<double>& H, const std::vector<double>& R, int n, int F, int N, double lambda, std::vector<double>& M) {
    std::vector<double> G((size_t) F * F, 0.0);
    std::vector<double> B((size_t) F * N, 0.0);
    for (int i = 0; i < n; ++i) {
        const double* h = &H[(size_t) i * F];
        const double* r = &R[(size_t) i * N];
        for (int a = 0; a < F; ++a) {
            const double ha = h[a];
            if (ha == 0.0) continue;
            double* grow = &G[(size_t) a * F];
            for (int b = 0; b < F; ++b) grow[b] += ha * h[b];
            double* brow = &B[(size_t) a * N];
            for (int j = 0; j < N; ++j) brow[j] += ha * r[j];
        }
    }
    for (int a = 0; a < F; ++a) G[(size_t) a * F + a] += lambda;
    if (!cholesky_solve(G, F, B, N)) return false;
    // B is X (F x N); M = X^T (N x F)
    M.assign((size_t) N * F, 0.0);
    for (int a = 0; a < F; ++a)
        for (int j = 0; j < N; ++j) M[(size_t) j * F + a] = B[(size_t) a * N + j];
    return true;
}

void eig_sym(std::vector<double> A, int n, std::vector<double>& evals, std::vector<double>& evecs) {
    std::vector<double> V((size_t) n * n, 0.0);
    for (int i = 0; i < n; ++i) V[(size_t) i * n + i] = 1.0;
    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < n; ++p)
            for (int q = p + 1; q < n; ++q) off += A[(size_t) p * n + q] * A[(size_t) p * n + q];
        if (off < 1e-30) break;
        for (int p = 0; p < n; ++p) {
            for (int q = p + 1; q < n; ++q) {
                const double apq = A[(size_t) p * n + q];
                if (std::fabs(apq) < 1e-300) continue;
                const double theta = (A[(size_t) q * n + q] - A[(size_t) p * n + p]) / (2.0 * apq);
                const double t = (theta >= 0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
                for (int k = 0; k < n; ++k) {   // A <- J^T A J, columns p and q
                    const double akp = A[(size_t) k * n + p], akq = A[(size_t) k * n + q];
                    A[(size_t) k * n + p] = c * akp - s * akq;
                    A[(size_t) k * n + q] = s * akp + c * akq;
                }
                for (int k = 0; k < n; ++k) {   // rows p and q
                    const double apk = A[(size_t) p * n + k], aqk = A[(size_t) q * n + k];
                    A[(size_t) p * n + k] = c * apk - s * aqk;
                    A[(size_t) q * n + k] = s * apk + c * aqk;
                }
                for (int k = 0; k < n; ++k) {   // eigenvectors
                    const double vkp = V[(size_t) k * n + p], vkq = V[(size_t) k * n + q];
                    V[(size_t) k * n + p] = c * vkp - s * vkq;
                    V[(size_t) k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    std::vector<int> order((size_t) n);
    for (int i = 0; i < n; ++i) order[(size_t) i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return A[(size_t) a * n + a] > A[(size_t) b * n + b]; });
    evals.assign((size_t) n, 0.0);
    evecs.assign((size_t) n * n, 0.0);
    for (int j = 0; j < n; ++j) {
        const int src = order[(size_t) j];
        evals[(size_t) j] = A[(size_t) src * n + src];
        for (int k = 0; k < n; ++k) evecs[(size_t) k * n + j] = V[(size_t) k * n + src];
    }
}

namespace {
// Orthonormalises the columns of A (rows x cols) in place by modified Gram-Schmidt, twice for stability.
void orth_cols(std::vector<double>& A, int rows, int cols) {
    for (int pass = 0; pass < 2; ++pass)
        for (int j = 0; j < cols; ++j) {
            for (int k = 0; k < j; ++k) {
                double d = 0.0;
                for (int i = 0; i < rows; ++i) d += A[(size_t) i * cols + k] * A[(size_t) i * cols + j];
                for (int i = 0; i < rows; ++i) A[(size_t) i * cols + j] -= d * A[(size_t) i * cols + k];
            }
            double nn = 0.0;
            for (int i = 0; i < rows; ++i) nn += A[(size_t) i * cols + j] * A[(size_t) i * cols + j];
            nn = std::sqrt(nn);
            if (nn > 0.0)
                for (int i = 0; i < rows; ++i) A[(size_t) i * cols + j] /= nn;
        }
}
}  // namespace

void truncate_rank(const std::vector<double>& M, int N, int F, int r, uint64_t seed, std::vector<double>& out, int oversample, int power) {
    const int full = std::min(N, F);
    if (r >= full) {
        out = M;
        return;
    }
    const int k = std::min(full, r + oversample);
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> gauss(0.0, 1.0);
    // Y = M Omega (N x k)
    std::vector<double> Om((size_t) F * k);
    for (double& v : Om) v = gauss(rng);
    std::vector<double> Y((size_t) N * k, 0.0);
    for (int i = 0; i < N; ++i)
        for (int a = 0; a < F; ++a) {
            const double m = M[(size_t) i * F + a];
            for (int j = 0; j < k; ++j) Y[(size_t) i * k + j] += m * Om[(size_t) a * k + j];
        }
    orth_cols(Y, N, k);
    for (int it = 0; it < power; ++it) {
        // Z = M^T Y (F x k), orthonormal; Y = M Z (N x k), orthonormal
        std::vector<double> Z((size_t) F * k, 0.0);
        for (int i = 0; i < N; ++i)
            for (int a = 0; a < F; ++a) {
                const double m = M[(size_t) i * F + a];
                for (int j = 0; j < k; ++j) Z[(size_t) a * k + j] += m * Y[(size_t) i * k + j];
            }
        orth_cols(Z, F, k);
        std::fill(Y.begin(), Y.end(), 0.0);
        for (int i = 0; i < N; ++i)
            for (int a = 0; a < F; ++a) {
                const double m = M[(size_t) i * F + a];
                for (int j = 0; j < k; ++j) Y[(size_t) i * k + j] += m * Z[(size_t) a * k + j];
            }
        orth_cols(Y, N, k);
    }
    // Q = Y (N x k); B = Q^T M (k x F)
    std::vector<double> B((size_t) k * F, 0.0);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < k; ++j) {
            const double q = Y[(size_t) i * k + j];
            for (int a = 0; a < F; ++a) B[(size_t) j * F + a] += q * M[(size_t) i * F + a];
        }
    // the top-r left singular vectors of B: eig of B B^T (k x k)
    std::vector<double> G((size_t) k * k, 0.0);
    for (int i = 0; i < k; ++i)
        for (int j = 0; j < k; ++j) {
            double d = 0.0;
            for (int a = 0; a < F; ++a) d += B[(size_t) i * F + a] * B[(size_t) j * F + a];
            G[(size_t) i * k + j] = d;
        }
    std::vector<double> ev, U;
    eig_sym(G, k, ev, U);
    // out = (Q Ur) (Ur^T B) with Ur the first r columns of U
    std::vector<double> C((size_t) r * F, 0.0);   // Ur^T B
    for (int j = 0; j < r; ++j)
        for (int i = 0; i < k; ++i) {
            const double u = U[(size_t) i * k + j];
            for (int a = 0; a < F; ++a) C[(size_t) j * F + a] += u * B[(size_t) i * F + a];
        }
    out.assign((size_t) N * F, 0.0);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < r; ++j) {
            double p = 0.0;   // (Q Ur)[i][j]
            for (int q = 0; q < k; ++q) p += Y[(size_t) i * k + q] * U[(size_t) q * k + j];
            for (int a = 0; a < F; ++a) out[(size_t) i * F + a] += p * C[(size_t) j * F + a];
        }
}

double best_scalar(const std::vector<double>& ys, const std::vector<double>& ye, int n, int N) {
    double sxy = 0.0, sxx = 0.0;
    for (size_t i = 0; i < (size_t) n * N; ++i) {
        sxy += ys[i] * ye[i];
        sxx += ys[i] * ys[i];
    }
    return sxx > 0.0 ? sxy / sxx : 0.0;
}

double sq_dist(const double* a, const double* b, int N) {
    double d = 0.0;
    for (int i = 0; i < N; ++i) d += (a[i] - b[i]) * (a[i] - b[i]);
    return d;
}

double sq_norm(const double* a, int N) {
    double d = 0.0;
    for (int i = 0; i < N; ++i) d += a[i] * a[i];
    return d;
}

double cosine(const double* a, const double* b, int N) {
    double ab = 0.0;
    for (int i = 0; i < N; ++i) ab += a[i] * b[i];
    const double na = sq_norm(a, N), nb = sq_norm(b, N);
    return na > 0.0 && nb > 0.0 ? ab / std::sqrt(na * nb) : 0.0;
}

static double clip01(double g) { return g < 0.0 ? 0.0 : (g > 1.0 ? 1.0 : g); }

double residual_all(double g) { return std::sqrt(1.0 - clip01(g)); }

double residual_dynamic(double f, double g) { return (1.0 - clip01(f)) + clip01(f) * residual_all(g); }

double moves_share(double refused, double moves) {
    if (!(moves > 0.0)) return 0.0;
    return clip01(refused / moves);
}

double cosine(const float* a, const float* b, int N) {
    double ab = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < N; ++i) {
        ab += (double) a[i] * b[i];
        na += (double) a[i] * a[i];
        nb += (double) b[i] * b[i];
    }
    return na > 0.0 && nb > 0.0 ? ab / std::sqrt(na * nb) : 0.0;
}

}  // namespace subfit_fit
