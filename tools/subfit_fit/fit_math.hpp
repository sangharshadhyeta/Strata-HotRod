// tools/subfit_fit/fit_math.hpp - SPEC-substitute-fit stage 1b: the dense linear algebra of the offline fit, in plain C++ (double).
// Matrices are row-major std::vector<double>.  Nothing here knows about experts; subfit_fit.cpp does that.
#pragma once

#include <cstdint>
#include <vector>

namespace subfit_fit {

/// Solves A X = B for a symmetric positive definite A (n x n) by Cholesky.  B (n x m) is overwritten with X.  false if A is not SPD.
bool cholesky_solve(const std::vector<double>& A, int n, std::vector<double>& B, int m);

/// Ridge regression of residuals on features: M (N x F) minimises sum_i ||r_i - M h_i||^2 + lambda ||M||^2, with H (n x F) the features
/// and R (n x N) the targets.  M = X^T with X = (H^T H + lambda I)^-1 H^T R.  false if the system is not SPD (lambda <= 0 and singular).
bool ridge_fit(const std::vector<double>& H, const std::vector<double>& R, int n, int F, int N, double lambda, std::vector<double>& M);

/// Jacobi eigen-decomposition of a symmetric n x n matrix A (copied).  evals descending; evecs (n x n) has the eigenvector of evals[j] in column j.
void eig_sym(std::vector<double> A, int n, std::vector<double>& evals, std::vector<double>& evecs);

/// Best rank-r approximation of M (N x F) by a randomized range finder: r + oversample test vectors, `power` power iterations, the
/// small problem solved by eig_sym.  out (N x F) = the approximation.  r >= min(N, F) returns M itself.
void truncate_rank(const std::vector<double>& M, int N, int F, int r, uint64_t seed, std::vector<double>& out, int oversample = 10,
                   int power = 2);

/// y_hat = c * ys with the least-squares c for samples (ys_i, ye_i): c = sum(ys.ye) / sum(ys.ys) (0 when the denominator is 0).
double best_scalar(const std::vector<double>& ys, const std::vector<double>& ye, int n, int N);

/// Squared distance and squared norm helpers.
double sq_dist(const double* a, const double* b, int N);
double sq_norm(const double* a, int N);
double cosine(const double* a, const double* b, int N);
double cosine(const float* a, const float* b, int N);

/// Mode estimate (first order, amplitudes; 1 = no correction).  g = the fraction of a move's error energy that the rank-16 nudge removes.
/// ALL gives every move its fitted variant: residual sqrt(1 - g) per move.  DYNAMIC corrects only the moves the shift cap refuses today
/// (the share f = refused / moves); the others run plain: residual (1 - f) + f sqrt(1 - g).  The gain kept by DYNAMIC against ALL is f
/// and its corrections are the share f of ALL's.
double residual_all(double g);
double residual_dynamic(double f, double g);
double moves_share(double refused, double moves);


/// Splits sample indices by parity: returns 0 for even indices, 1 for odd (the held-out split of the spec).
inline int half_of(int64_t index) { return (int) (index & 1); }

}  // namespace subfit_fit
