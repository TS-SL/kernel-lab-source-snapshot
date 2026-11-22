// Double-precision GEMM reference and the derived error bound (docs/gemm.md),
// shared by test_gemm and the benchmark so benchmarked outputs are checked
// with exactly the tested criterion.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace kl_test {

namespace gemm {

constexpr double kU = 5.9604644775390625e-08;                // 2^-24
constexpr double kHalfMinSubnormal = 7.006492321624085e-46;  // 2^-150
constexpr double kRefU = 2.220446049250313e-16;              // 2^-52: per-term slack for the float64 reference
constexpr double kMaxS = 8.507059173023462e+37;              // 2^126: validity domain, S < 2^126

struct CheckResult {
    double worst_ratio = 0;
    std::int64_t violations = 0;       // finite reference, error above the bound
    std::int64_t special_mismatches = 0;  // non-finite reference, different special value
    std::int64_t out_of_domain = 0;    // finite reference with S >= 2^126 (callers require 0)
    std::int64_t inexact = 0;          // exact mode: outputs not bitwise equal to the reference
};

// Checks C (m x n) against the float64 reference of A (m x k) B (k x n).
// exact = true additionally requires C == float(reference) bit for bit
// (valid for integer inputs whose partial sums stay below 2^24).
inline CheckResult check_gemm(std::int64_t m, std::int64_t n, std::int64_t k, const float* a, const float* b,
                              const float* c, bool exact = false) {
    CheckResult res;
    const double gamma = static_cast<double>(k) * kU / (1.0 - static_cast<double>(k) * kU);
    std::vector<double> ref(static_cast<std::size_t>(n)), s(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < m; ++i) {
        std::fill(ref.begin(), ref.end(), 0.0);
        std::fill(s.begin(), s.end(), 0.0);
        for (std::int64_t p = 0; p < k; ++p) {
            const double av = a[i * k + p];
            const float* brow = b + p * n;
            for (std::int64_t j = 0; j < n; ++j) {
                ref[static_cast<std::size_t>(j)] += av * brow[j];
                s[static_cast<std::size_t>(j)] += std::fabs(av * brow[j]);
            }
        }
        for (std::int64_t j = 0; j < n; ++j) {
            const double want = ref[static_cast<std::size_t>(j)];
            const double got = c[i * n + j];
            if (!std::isfinite(want)) {
                const bool same = std::isnan(want) ? std::isnan(got) : got == want;
                res.special_mismatches += same ? 0 : 1;
                continue;
            }
            const double sj = s[static_cast<std::size_t>(j)];
            if (!(sj < kMaxS)) {
                ++res.out_of_domain;
                continue;
            }
            // The float64 reference itself errs by up to ~K 2^-53 S (sequential sum): K 2^-52 S covers it.
            // Underflow roundings can be amplified by later ones: K 2^-150 (1 + gamma).
            const double allowed = (gamma + static_cast<double>(k) * kRefU) * sj +
                                   static_cast<double>(k) * kHalfMinSubnormal * (1.0 + gamma);
            const double err = std::isnan(got) ? std::numeric_limits<double>::infinity() : std::fabs(got - want);
            if (allowed > 0) res.worst_ratio = std::max(res.worst_ratio, err / allowed);
            if (!(err <= allowed)) ++res.violations;
            if (exact && !(static_cast<float>(want) == c[i * n + j])) ++res.inexact;
        }
    }
    return res;
}

}  // namespace gemm

}  // namespace kl_test
