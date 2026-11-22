// Double-precision layer-norm reference and the derived error bound
// (docs/layernorm.md), shared by test_layernorm and the benchmark so
// benchmarked outputs are checked with exactly the tested criterion.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "cuda/kernels.hpp"
#include "kernellens/kernellens.h"

namespace kl_test {

namespace ln {

constexpr double kU = 5.9604644775390625e-08;           // 2^-24
constexpr double kAbsSlack = 7.006492321624085e-45;     // 2^-147
constexpr double kHalfMinSubnormal = 7.006492321624085e-46;  // 2^-150
// Validity domain of the bound (docs/layernorm.md): no intermediate
// overflows and the scale stays far from the subnormal range.
constexpr double kMinScaleSquared = 7.888609052210118e-31;  // 2^-100: var + eps >= this
// 2^126: sums stay below this (the computed sum of squares can exceed the
// exact one by the factor 1 + d_mu^2 / var <= ~2, so 2^126 keeps it finite).
constexpr double kMaxSum = 8.507059173023462e+37;

// Reduction structure of the implementation that runs: elements per thread
// k and levels L that combine real data, for T threads per row.
struct Structure {
    double k;
    double levels;
};

inline Structure structure_for(std::int64_t cols, double threads) {
    return {std::ceil(static_cast<double>(cols) / threads),
            std::ceil(std::log2(std::min(static_cast<double>(cols), threads)))};
}

inline Structure structure(kl_layernorm_impl impl, std::int64_t cols) {
    const kl_layernorm_impl r = kl::cuda::layernorm_resolve(cols, impl);
    return structure_for(cols, r == KL_LAYERNORM_FUSED_WARP ? 32.0 : 256.0);
}

struct RefRow {
    std::vector<double> y;  // y_ref (affine applied if gamma given)
    std::vector<double> z;  // (x - mu) / sig
    bool nan_row = false;
    bool in_domain = true;  // the bound's validity conditions hold
    double mu = 0, var = 0, sig = 0;
    double m1 = 0;  // mean |x_j|            (naive kernel's bound)
    double q = 0;   // mean x_j^2            (naive kernel's bound)
    double a = 0;   // mean |x_j - x_0|      (pivot-shifted mean)
    double dev = 0; // mean |x_j - mu|       (corrected mean)
};

// gamma/beta may be nullptr (no affine).
inline RefRow reference_row(const float* x, std::int64_t n, const float* gamma, const float* beta, float eps) {
    RefRow r;
    r.y.assign(static_cast<std::size_t>(n), 0.0);
    r.z.assign(static_cast<std::size_t>(n), 0.0);
    double s = 0, a = 0, m1 = 0, q = 0;
    for (std::int64_t j = 0; j < n; ++j) {
        if (!std::isfinite(x[j])) {
            r.nan_row = true;  // the definition gives NaN everywhere (inf - inf, NaN)
            return r;
        }
        s += x[j];
        a += std::fabs(static_cast<double>(x[j]) - x[0]);
        m1 += std::fabs(static_cast<double>(x[j]));
        q += static_cast<double>(x[j]) * x[j];
    }
    r.mu = s / n;
    r.a = a / n;
    r.m1 = m1 / n;
    r.q = q / n;
    double v = 0, dv = 0;
    for (std::int64_t j = 0; j < n; ++j) {
        v += (x[j] - r.mu) * (x[j] - r.mu);
        dv += std::fabs(x[j] - r.mu);
    }
    r.var = v / n;
    r.dev = dv / n;
    r.in_domain = a < kMaxSum && v < kMaxSum && r.var + static_cast<double>(eps) >= kMinScaleSquared;
    r.sig = std::sqrt(r.var + static_cast<double>(eps));
    if (!(r.sig > 0)) {
        r.nan_row = true;  // eps = 0 and constant row: 0 / 0
        return r;
    }
    for (std::int64_t j = 0; j < n; ++j) {
        const double z = (x[j] - r.mu) / r.sig;
        r.z[static_cast<std::size_t>(j)] = z;
        r.y[static_cast<std::size_t>(j)] = gamma ? static_cast<double>(gamma[j]) * z + beta[j] : z;
    }
    return r;
}

// E_i of docs/layernorm.md for the two-pass implementations:
//   e1 = (k + L + 1) u A + u |mu| + 2^-150          pivot-shifted mean m1
//   e2 = (k + L + 1) u (D + e1) + u |mu| + 2^-150   corrected mean (output)
// with A = mean |x_j - x_0|, D = mean |x_j - mu|. The variance is taken
// around m1, so e1 enters only through (e1 / sig)^2.
inline double two_pass_E(const Structure& s, const RefRow& r, double z_abs) {
    const double kl1 = s.k + s.levels + 1;
    const double e1 = kl1 * kU * r.a + kU * std::fabs(r.mu) + kHalfMinSubnormal;
    const double e2 = kl1 * kU * (r.dev + e1) + kU * std::fabs(r.mu) + kHalfMinSubnormal;
    const double c1 = e1 / r.sig;
    return e2 / r.sig + z_abs * ((s.k + s.levels + 3) * kU / 2 + 4.5 * kU + c1 * c1 / 2);
}

// The same for the internal naive E[x^2] - E[x]^2 kernel (no pivot; valid
// only while its variance error is small against var + eps).
inline double naive_E(const Structure& s, const RefRow& r, double z_abs) {
    const double dmu = (s.k + s.levels) * kU * r.m1;
    const double dvar = (s.k + s.levels + 1) * kU * r.q + 2 * std::fabs(r.mu) * dmu + dmu * dmu +
                        kU * r.mu * r.mu + kU * r.var;
    const double e_r = dvar / (2 * r.sig * r.sig) + 2.5 * kU;
    return dmu / r.sig + z_abs * (2 * kU + e_r);
}

struct CheckResult {
    double worst_ratio = 0;
    std::int64_t violations = 0;
    std::int64_t nan_mismatches = 0;
    std::int64_t out_of_domain_rows = 0;  // rows the bound does not cover (callers require 0)
};

// |y - y_ref| <= 1.01 * bound + (1 + |gamma_j|) 2^-147 per element; NaN rows
// must be NaN. naive = true checks against the naive kernel's own bound.
inline CheckResult check_layernorm(kl_layernorm_impl impl, std::int64_t rows, std::int64_t cols, const float* x,
                                   const float* gamma, const float* beta, float eps, const float* y,
                                   bool naive = false) {
    CheckResult res;
    const Structure st = naive ? structure_for(cols, 256.0) : structure(impl, cols);
    for (std::int64_t r = 0; r < rows; ++r) {
        const float* xr = x + r * cols;
        const float* yr = y + r * cols;
        const RefRow ref = reference_row(xr, cols, gamma, beta, eps);
        if (ref.nan_row) {
            for (std::int64_t j = 0; j < cols; ++j) res.nan_mismatches += std::isnan(yr[j]) ? 0 : 1;
            continue;
        }
        if (!ref.in_domain) {
            ++res.out_of_domain_rows;
            continue;
        }
        for (std::int64_t j = 0; j < cols; ++j) {
            const double want = ref.y[static_cast<std::size_t>(j)];
            const double got = yr[j];
            // Non-finite gamma/beta make the reference NaN or +-inf for that
            // column only: the result must then be the same special value.
            if (!std::isfinite(want)) {
                const bool same = std::isnan(want) ? std::isnan(got) : got == want;
                res.nan_mismatches += same ? 0 : 1;
                continue;
            }
            const double z_abs = std::fabs(ref.z[static_cast<std::size_t>(j)]);
            const double E = naive ? naive_E(st, ref, z_abs) : two_pass_E(st, ref, z_abs);
            const double g = gamma ? std::fabs(static_cast<double>(gamma[j])) : 0.0;
            const double bound = gamma ? g * E + kU * std::fabs(want) : E;
            // Absolute part: t's rounding in the subnormal range (scaled by
            // |gamma|) and the fma's (docs/layernorm.md).
            const double allowed = 1.01 * bound + (1.0 + g) * kAbsSlack;
            const double err = std::isnan(got) ? std::numeric_limits<double>::infinity() : std::fabs(got - want);
            res.worst_ratio = std::max(res.worst_ratio, err / allowed);
            if (!(err <= allowed)) ++res.violations;
        }
    }
    return res;
}

}  // namespace ln

}  // namespace kl_test
