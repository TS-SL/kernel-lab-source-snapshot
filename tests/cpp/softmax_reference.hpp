// Double-precision softmax reference and the derived error bound
// (docs/softmax.md), shared by test_softmax and bench_softmax so the
// benchmark checks its outputs with exactly the tested criterion.
#pragma once

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "cuda/kernels.hpp"
#include "kernellens/kernellens.h"

namespace kl_test {

constexpr double kU = 5.9604644775390625e-08;  // 2^-24

// Absolute slack for results in the subnormal range, where relative error
// is meaningless: expf's 2 ulp there is 2 * 2^-149 absolute, the division
// adds half a subnormal ulp, and dividing by the sum (>= 1) cannot enlarge
// either; 2^-147 = 4 * 2^-149 covers both (docs/softmax.md). A build that
// flushed subnormal outputs to zero (FTZ) would exceed this.
constexpr double kAbsSlack = 7.006492321624085e-45;  // 2^-147

// Error-bound parameters of the implementation that will actually run:
// k elements per thread, L reduction levels that combine real data,
// online recurrence or not.
struct BoundParams {
    double k;
    double levels;
    bool online;
};

inline BoundParams bound_params(kl_softmax_impl impl, std::int64_t rows, std::int64_t cols) {
    const kl_softmax_impl r = kl::cuda::softmax_resolve(rows, cols, impl);
    const double threads = r == KL_SOFTMAX_FUSED_WARP ? 32.0 : 256.0;
    // Only threads that hold data contribute non-identity partials, and
    // combining with an identity partial is exact (x + 0; s*1 + 0*0), so an
    // element passes through at most ceil(log2(min(cols, T))) inexact levels.
    const double levels = std::ceil(std::log2(std::min(static_cast<double>(cols), threads)));
    return {std::ceil(static_cast<double>(cols) / threads), levels, r == KL_SOFTMAX_FUSED_ONLINE};
}

// The d_i and Dw terms come only from rounding in float subtractions of
// inputs (x - m, and the online rescale arguments m' - m''). When every
// difference between two inputs of the row is exactly representable, those
// subtractions are exact and the terms are 0 (exact_differences = true).
// Dw is the exp-weighted mean of d_j - the quantity the derivation needs.
inline double element_bound(const BoundParams& p, double d_i, double Dw, bool exact_differences) {
    const double base = exact_differences ? 0.0 : d_i + Dw;
    return kU * (p.online ? base + 6 * (p.k - 1 + p.levels) + 9 : base + p.k + p.levels + 8);
}

// Reference row: y_ref in double, plus whether the definition gives NaN.
struct RefRow {
    std::vector<double> y;
    bool nan_row = false;
    double m = 0;   // row max
    double Dw = 0;  // sum_j e_j (m - x_j) / sum_j e_j over finite entries
};

inline RefRow reference_row(const float* x, std::int64_t cols) {
    const float inf = std::numeric_limits<float>::infinity();
    RefRow r;
    r.y.assign(static_cast<std::size_t>(cols), 0.0);
    double m = -std::numeric_limits<double>::infinity();
    bool any_nan = false, any_pinf = false;
    for (std::int64_t j = 0; j < cols; ++j) {
        if (std::isnan(x[j])) any_nan = true;
        if (x[j] == inf) any_pinf = true;
        if (!std::isnan(x[j])) m = std::max(m, static_cast<double>(x[j]));
    }
    // max == -inf (all -inf), NaN, or +inf: inf - inf or NaN in the definition.
    if (any_nan || any_pinf || m == -std::numeric_limits<double>::infinity()) {
        r.nan_row = true;
        return r;
    }
    double s = 0, weighted_d = 0;
    for (std::int64_t j = 0; j < cols; ++j) {
        const double e = std::exp(static_cast<double>(x[j]) - m);  // exp(-inf) = 0
        r.y[static_cast<std::size_t>(j)] = e;
        s += e;
        if (std::isfinite(x[j])) weighted_d += e * (m - x[j]);
    }
    for (double& v : r.y) v /= s;
    r.m = m;
    r.Dw = weighted_d / s;
    return r;
}

struct CheckResult {
    double worst_ratio = 0;  // max |err| / allowed, over all elements
    std::int64_t violations = 0;
    std::int64_t nan_mismatches = 0;
    std::int64_t row_sum_violations = 0;
};

// |y - y_ref| <= 1.01 * B_i * y_ref + 2^-147 for every element, NaN where
// the definition gives NaN, 0 <= y <= 1 elsewhere, and |sum_j y_j - 1| no
// larger than the sum of the per-element allowances. The last check is
// implied by the per-element one (it can only fail if that check were
// broken); it is a consistency check of this code, not an independent test.
inline CheckResult check_softmax(kl_softmax_impl impl, std::int64_t rows, std::int64_t cols,
                                 const float* x, const float* y, bool exact_differences = false) {
    CheckResult res;
    const BoundParams p = bound_params(impl, rows, cols);
    for (std::int64_t r = 0; r < rows; ++r) {
        const float* xr = x + r * cols;
        const float* yr = y + r * cols;
        const RefRow ref = reference_row(xr, cols);
        if (ref.nan_row) {
            for (std::int64_t j = 0; j < cols; ++j) res.nan_mismatches += std::isnan(yr[j]) ? 0 : 1;
            continue;
        }
        double row_sum = 0, allowed_sum = 0;
        for (std::int64_t j = 0; j < cols; ++j) {
            const double want = ref.y[static_cast<std::size_t>(j)];
            const double got = yr[j];
            const double d_i = std::isfinite(xr[j]) ? ref.m - xr[j] : 0.0;
            const double allowed =
                1.01 * element_bound(p, d_i, ref.Dw, exact_differences) * want + kAbsSlack;
            const double err = std::isnan(got) ? std::numeric_limits<double>::infinity()
                                               : std::fabs(got - want);
            res.worst_ratio = std::max(res.worst_ratio, err / allowed);
            if (!(err <= allowed)) ++res.violations;
            if (!(got >= 0.0 && got <= 1.0)) ++res.violations;
            row_sum += got;
            allowed_sum += allowed;
        }
        if (!(std::fabs(row_sum - 1.0) <= allowed_sum)) ++res.row_sum_violations;
    }
    return res;
}

}  // namespace kl_test
