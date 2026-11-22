// C++ tests for row-wise layer normalization (docs/layernorm.md).
//
// Reference: the definition in double precision on the CPU. Tolerance: the
// derived per-element bound (tests/cpp/layernorm_reference.hpp),
//   |y - y_ref| <= 1.01 * bound + 2^-147.
// Workloads: at most 2^20 elements per case; launches are sub-millisecond.
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "common/device_buffer.hpp"
#include "cuda/kernels.hpp"
#include "harness.hpp"
#include "kernellens/kernellens.h"
#include "layernorm_reference.hpp"

namespace {

using kl_test::fill;
using kl_test::same_float;
using kl_test::skip_for_max_n;
using kl_test::ln::check_layernorm;
using kl_test::ln::CheckResult;

const float kInf = std::numeric_limits<float>::infinity();
const float kNaN = std::numeric_limits<float>::quiet_NaN();

const kl_layernorm_impl kImpls[] = {KL_LAYERNORM_AUTO, KL_LAYERNORM_MULTIPASS, KL_LAYERNORM_FUSED_BLOCK,
                                    KL_LAYERNORM_FUSED_SMEM, KL_LAYERNORM_FUSED_WARP};

const char* impl_name(kl_layernorm_impl impl) {
    switch (impl) {
        case KL_LAYERNORM_AUTO: return "auto";
        case KL_LAYERNORM_MULTIPASS: return "multipass";
        case KL_LAYERNORM_FUSED_BLOCK: return "fused_block";
        case KL_LAYERNORM_FUSED_WARP: return "fused_warp";
        case KL_LAYERNORM_FUSED_SMEM: return "fused_smem";
        default: return "?";
    }
}

bool impl_supports(kl_layernorm_impl impl, std::int64_t cols) {
    if (impl == KL_LAYERNORM_FUSED_WARP) return cols <= KL_LAYERNORM_WARP_MAX_COLS;
    if (impl == KL_LAYERNORM_FUSED_SMEM) return cols <= KL_LAYERNORM_SMEM_MAX_COLS;
    return true;
}

struct Shape {
    std::int64_t rows, cols;
};

// Around warp (32) and block (256) boundaries, the warp kernel's
// values-per-lane steps and its 1024 limit, up to the 2^20 maximum.
const Shape kShapes[] = {
    {37, 1},    {37, 2},    {37, 31},   {37, 32},   {37, 33},   {37, 63},    {37, 64},
    {37, 65},   {37, 127},  {37, 128},  {37, 129},  {37, 255},  {37, 256},   {37, 257},
    {37, 511},  {37, 512},  {37, 513},  {37, 1000}, {37, 1023}, {37, 1024},  {37, 1025},
    {37, 2048}, {37, 4097}, {5, 8191}, {5, 8192}, {5, 8193}, {3, 10007}, {3, 65536}, {1, 1 << 20},
};

std::vector<float> run_ln(kl_layernorm_impl impl, std::int64_t rows, std::int64_t cols, const std::vector<float>& x,
                          const float* gamma, const float* beta, float eps) {
    std::vector<float> y(x.size(), -12345.0f);
    EXPECT_STATUS(kl_layernorm_host(rows, cols, x.data(), gamma, beta, eps, y.data(), impl), KL_OK);
    return y;
}

void check_all(const char* label, std::int64_t rows, std::int64_t cols, const std::vector<float>& x,
               const float* gamma, const float* beta, float eps, double* worst = nullptr) {
    for (kl_layernorm_impl impl : kImpls) {
        if (!impl_supports(impl, cols)) continue;
        const std::vector<float> y = run_ln(impl, rows, cols, x, gamma, beta, eps);
        const CheckResult c = check_layernorm(impl, rows, cols, x.data(), gamma, beta, eps, y.data());
        if (c.violations || c.nan_mismatches) {
            std::printf("  %s %s %lldx%lld: %lld violations, %lld special-value mismatches, worst %.3g x bound\n",
                        label, impl_name(impl), static_cast<long long>(rows), static_cast<long long>(cols),
                        static_cast<long long>(c.violations), static_cast<long long>(c.nan_mismatches),
                        c.worst_ratio);
        }
        if (c.out_of_domain_rows) {
            std::printf("  %s %s: %lld rows outside the bound's validity domain\n", label, impl_name(impl),
                        static_cast<long long>(c.out_of_domain_rows));
        }
        EXPECT(c.violations == 0);
        EXPECT(c.nan_mismatches == 0);
        EXPECT(c.out_of_domain_rows == 0);
        if (worst) *worst = std::max(*worst, c.worst_ratio);
    }
}

// Runs every shape with x ~ U[x_lo, x_hi], optionally with affine params.
void run_distribution(const char* label, float x_lo, float x_hi, bool affine, float eps, std::uint64_t seed) {
    double worst = 0;
    for (const Shape& s : kShapes) {
        if (skip_for_max_n(s.rows * s.cols, label)) continue;
        const std::vector<float> x = fill(seed++, s.rows * s.cols, x_lo, x_hi);
        const std::vector<float> g = fill(seed++, s.cols, 0.5f, 1.5f);
        const std::vector<float> b = fill(seed++, s.cols, -1.0f, 1.0f);
        check_all(label, s.rows, s.cols, x, affine ? g.data() : nullptr, affine ? b.data() : nullptr, eps, &worst);
    }
    std::printf("  %s: worst error %.3f of the derived bound\n", label, worst);
}

void test_ln_uniform_affine() { run_distribution("U[-10,10] affine", -10.0f, 10.0f, true, 1e-5f, 100); }
void test_ln_uniform_no_affine() { run_distribution("U[-10,10] no affine", -10.0f, 10.0f, false, 1e-5f, 200); }
// |mean| / std ~ 1700: the bound grows with it, linearly for the two-pass kernels.
void test_ln_large_offset() { run_distribution("1000+U[-1,1]", 999.0f, 1001.0f, true, 1e-5f, 300); }
// Variance ~3e-7 against eps = 1e-5: eps dominates the scale.
void test_ln_eps_dominated() { run_distribution("U[-1e-3,1e-3] eps-dominated", -1e-3f, 1e-3f, true, 1e-5f, 400); }

void test_ln_eps_values() {
    const std::int64_t rows = 9, cols = 777;
    const std::vector<float> x = fill(500, rows * cols, -3.0f, 3.0f);
    for (float eps : {0.0f, 1e-12f, 1e-5f, 1.0f, 1000.0f}) check_all("eps sweep", rows, cols, x, nullptr, nullptr, eps);
}

// Constant rows: with the pivot-shifted mean every x_j - x_0 is exactly 0, so
// the computed mean is exactly c for ANY c (not only when n * c happens to be
// exact: 0.1 was the reviewer's counterexample for the unshifted mean), even
// near FLT_MAX where n * c would overflow. Then x - mean = +0 and
// y = fmaf(+0 * r, gamma, beta) = beta bit for bit (+0 without affine; beta
// values here are nonzero, a beta of -0 would give +0). With eps = 0 the row
// is 0/0 = NaN, as in PyTorch.
void test_ln_constant_rows() {
    const float values[] = {3.25f, 0.1f, 1.0f / 3.0f, 7.77f, -1e-3f, 1e36f, -2.5e30f, FLT_MAX, 1e-40f};
    for (float c : values) {
    for (const Shape& s : kShapes) {
        if (skip_for_max_n(s.rows * s.cols, "constant rows")) continue;
        if (c != 3.25f && c != 0.1f && s.rows * s.cols > 65536) continue;  // all shapes for two values
        const std::vector<float> x(static_cast<std::size_t>(s.rows * s.cols), c);
        const std::vector<float> g = fill(600 + s.cols, s.cols, 0.5f, 1.5f);
        const std::vector<float> b = fill(700 + s.cols, s.cols, -1.0f, 1.0f);
        for (kl_layernorm_impl impl : kImpls) {
            if (!impl_supports(impl, s.cols)) continue;
            const std::vector<float> ya = run_ln(impl, s.rows, s.cols, x, g.data(), b.data(), 1e-5f);
            const std::vector<float> yn = run_ln(impl, s.rows, s.cols, x, nullptr, nullptr, 1e-5f);
            const std::vector<float> y0 = run_ln(impl, s.rows, s.cols, x, nullptr, nullptr, 0.0f);
            std::int64_t wrong = 0;
            for (std::int64_t i = 0; i < s.rows * s.cols; ++i) {
                wrong += same_float(ya[static_cast<std::size_t>(i)], b[static_cast<std::size_t>(i % s.cols)]) ? 0 : 1;
                wrong += same_float(yn[static_cast<std::size_t>(i)], 0.0f) ? 0 : 1;
                wrong += std::isnan(y0[static_cast<std::size_t>(i)]) ? 0 : 1;
            }
            if (wrong) std::printf("  constant %g %s cols=%lld: %lld wrong\n", static_cast<double>(c), impl_name(impl),
                                   static_cast<long long>(s.cols), static_cast<long long>(wrong));
            EXPECT(wrong == 0);
        }
    }
    }
}

// Inside the bound's validity domain (docs/layernorm.md) but far from
// U[-10, 10]: huge and tiny magnitudes, subnormal inputs (the mean's division
// can round to a subnormal: the 2^-150 term), and subnormal outputs scaled by
// a large gamma (the (1 + |gamma|) 2^-147 term).
void test_ln_extreme_in_domain() {
    std::uint64_t seed = 2000;
    for (std::int64_t cols : {2, 33, 128}) {
        const std::int64_t rows = 16;
        const std::vector<float> g = fill(seed++, cols, 0.5f, 1.5f);
        const std::vector<float> b = fill(seed++, cols, -1.0f, 1.0f);
        std::vector<float> big = fill(seed++, rows * cols, -1.0f, 1.0f);
        for (float& v : big) v *= 1e18f;  // squares ~1e36, n * max^2 < 2^127
        check_all("1e18*U[-1,1]", rows, cols, big, g.data(), b.data(), 1e-5f);
        std::vector<float> off = fill(seed++, rows * cols, 0.999f, 1.001f);
        for (float& v : off) v *= 1e20f;  // |mean| / std ~ 1700 at 1e20
        check_all("1e20*U[0.999,1.001]", rows, cols, off, g.data(), b.data(), 1e-5f);
        std::vector<float> tiny = fill(seed++, rows * cols, -1.0f, 1.0f);
        for (float& v : tiny) v *= 1e-14f;  // var ~3e-29 >= 2^-100 with eps = 0
        // (not at cols = 2: two nearly equal draws give var < 2^-100, outside
        // the domain - the checker refuses such rows rather than pass them)
        if (cols > 2) check_all("1e-14*U[-1,1] eps=0", rows, cols, tiny, nullptr, nullptr, 0.0f);
        std::vector<float> sub(static_cast<std::size_t>(rows * cols));
        const std::vector<float> u = fill(seed++, rows * cols, 0.0f, 1000.0f);
        for (std::size_t i = 0; i < sub.size(); ++i) sub[i] = std::ldexp(std::floor(u[i]), -149);  // subnormals
        for (std::int64_t r = 0; r < rows; ++r) sub[static_cast<std::size_t>(r * cols)] = std::ldexp(1.0f, -149);
        check_all("subnormal x eps=1e-5", rows, cols, sub, nullptr, nullptr, 1e-5f);
        check_all("subnormal x eps=1e-12", rows, cols, sub, nullptr, nullptr, 1e-12f);
        std::vector<float> t = fill(seed++, rows * cols, -1.0f, 1.0f);
        for (float& v : t) v *= 1e-37f;
        const std::vector<float> g6(static_cast<std::size_t>(cols), 1e6f);
        check_all("subnormal t, gamma 1e6", rows, cols, t, g6.data(), b.data(), 1e4f);
    }
}

// Outside the validity domain the bound does not apply (docs/layernorm.md).
// Two documented examples, where KernelLens and PyTorch agree (checked in the
// Python tests): d^2 underflows to 0 with eps = 0 -> +-inf; d^2 overflows ->
// 1/sqrt(inf) = 0 -> +-0.
void test_ln_outside_domain_documented() {
    const std::vector<float> under = {1e-30f, -1e-30f};
    const std::vector<float> over = {1e20f, -1e20f};
    for (kl_layernorm_impl impl : kImpls) {
        const std::vector<float> yu = run_ln(impl, 1, 2, under, nullptr, nullptr, 0.0f);
        EXPECT(yu[0] == kInf && yu[1] == -kInf);
        const std::vector<float> yo = run_ln(impl, 1, 2, over, nullptr, nullptr, 1e-5f);
        EXPECT(yo[0] == 0.0f && yo[1] == 0.0f);
    }
}

// The largest grid: rows = 2^20 (one block per row for the block kernels,
// 2^18 blocks for the warp kernel).
// Rows whose first element (the pivot) is an outlier: the pivot-shifted mean
// m1 is then much less accurate (its error grows with mean |x_j - x_0|), and
// only the corrected mean keeps the output within a bound that depends on
// the spread around the mean, not on the pivot (without the correction
// these rows are ~45x outside it).
void test_ln_outlier_first_element() {
    for (std::int64_t cols : {1000, 4097}) {
        for (float outlier : {1e3f, 1e5f, -1e5f}) {
            const std::int64_t rows = 8;
            std::vector<float> x = fill(2500 + cols, rows * cols, -1.0f, 1.0f);
            for (std::int64_t r = 0; r < rows; ++r) x[static_cast<std::size_t>(r * cols)] = outlier;
            check_all("outlier first", rows, cols, x, nullptr, nullptr, 1e-5f);
        }
    }
}

// The benchmark-only hooks: padded launches compute exactly what the public
// kernels compute; the occupancy query accepts only what the launchers accept.
void test_ln_padded_launcher_and_occupancy() {
    using kl::detail::DeviceBuffer;
    const std::int64_t rows = 7, cols = 4096;
    const std::size_t bytes = static_cast<std::size_t>(rows * cols) * sizeof(float);
    const std::vector<float> x = fill(2600, rows * cols, -5.0f, 5.0f);
    DeviceBuffer dx, dy;
    EXPECT_STATUS(DeviceBuffer::allocate(bytes, dx), KL_OK);
    EXPECT_STATUS(DeviceBuffer::allocate(bytes, dy), KL_OK);
    EXPECT(cudaMemcpy(dx.get(), x.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess);
    struct Case {
        kl_layernorm_impl impl;
        std::size_t extra;
    };
    for (const Case& k : {Case{KL_LAYERNORM_FUSED_BLOCK, 0}, Case{KL_LAYERNORM_FUSED_BLOCK, 16384},
                          Case{KL_LAYERNORM_FUSED_SMEM, 0}, Case{KL_LAYERNORM_FUSED_SMEM, 16384}}) {
        EXPECT(cudaMemset(dy.get(), 0xFF, bytes) == cudaSuccess);
        EXPECT_STATUS(kl::cuda::launch_layernorm_padded(rows, cols, dx.as<float>(), nullptr, nullptr, 1e-5f,
                                                        dy.as<float>(), k.impl, k.extra, 0),
                      KL_OK);
        std::vector<float> y(x.size());
        EXPECT(cudaMemcpy(y.data(), dy.get(), bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
        const std::vector<float> want = run_ln(k.impl, rows, cols, x, nullptr, nullptr, 1e-5f);
        EXPECT(std::equal(y.begin(), y.end(), want.begin(), [](float p, float q) { return same_float(p, q); }));
    }
    float* px = dx.as<float>();
    float* py = dy.as<float>();
    EXPECT_STATUS(kl::cuda::launch_layernorm_padded(rows, cols, px, nullptr, nullptr, 1e-5f, py,
                                                    KL_LAYERNORM_MULTIPASS, 0, 0),
                  KL_ERR_INVALID_ARGUMENT);
    EXPECT_STATUS(kl::cuda::launch_layernorm_padded(rows, cols, px, nullptr, nullptr, 1e-5f, py,
                                                    KL_LAYERNORM_FUSED_SMEM, 40 * 1024 - 16384 + 4, 0),
                  KL_ERR_UNSUPPORTED);  // 16 KiB row + padding above 40 KiB
    EXPECT_STATUS(kl::cuda::launch_layernorm_padded(rows, cols, px, nullptr, nullptr, 1e-5f, py,
                                                    KL_LAYERNORM_FUSED_BLOCK, 40 * 1024 + 4, 0),
                  KL_ERR_UNSUPPORTED);
    EXPECT(cudaDeviceSynchronize() == cudaSuccess);

    int b = 0, b_pad = 0, b_smem = 0;
    EXPECT_STATUS(kl::cuda::layernorm_blocks_per_sm(4096, KL_LAYERNORM_FUSED_BLOCK, 0, false, &b), KL_OK);
    EXPECT_STATUS(kl::cuda::layernorm_blocks_per_sm(4096, KL_LAYERNORM_FUSED_BLOCK, 16384, true, &b_pad), KL_OK);
    EXPECT_STATUS(kl::cuda::layernorm_blocks_per_sm(4096, KL_LAYERNORM_FUSED_SMEM, 0, false, &b_smem), KL_OK);
    EXPECT(b >= 1 && b_pad >= 1 && b_pad <= b && b_smem == b_pad);  // same shared memory, same blocks per SM
    int w = 0;
    EXPECT_STATUS(kl::cuda::layernorm_blocks_per_sm(1024, KL_LAYERNORM_FUSED_WARP, 0, false, &w), KL_OK);
    EXPECT(w >= 1);
    int unused = 0;
    EXPECT_STATUS(kl::cuda::layernorm_blocks_per_sm(8193, KL_LAYERNORM_FUSED_SMEM, 0, false, &unused),
                  KL_ERR_UNSUPPORTED);
    EXPECT_STATUS(kl::cuda::layernorm_blocks_per_sm(1025, KL_LAYERNORM_FUSED_WARP, 0, false, &unused),
                  KL_ERR_UNSUPPORTED);
    EXPECT_STATUS(kl::cuda::layernorm_blocks_per_sm(1024, KL_LAYERNORM_FUSED_WARP, 4, false, &unused),
                  KL_ERR_UNSUPPORTED);  // no padded warp launcher
    EXPECT_STATUS(kl::cuda::layernorm_blocks_per_sm(1024, KL_LAYERNORM_FUSED_SMEM, 0, true, &unused),
                  KL_ERR_UNSUPPORTED);  // no control copy of FUSED_SMEM
    EXPECT_STATUS(kl::cuda::layernorm_blocks_per_sm(1024, KL_LAYERNORM_FUSED_BLOCK, 64, false, &unused),
                  KL_ERR_UNSUPPORTED);  // padding only for the control copy
    EXPECT_STATUS(kl::cuda::layernorm_blocks_per_sm(4096, KL_LAYERNORM_FUSED_BLOCK, 40 * 1024 + 4, true, &unused),
                  KL_ERR_UNSUPPORTED);
    EXPECT_STATUS(kl::cuda::layernorm_blocks_per_sm(4096, KL_LAYERNORM_MULTIPASS, 0, false, &unused),
                  KL_ERR_UNSUPPORTED);
}

void test_ln_max_rows() {
    const std::int64_t rows = KL_LAYERNORM_MAX_ROWS, cols = 2;
    if (skip_for_max_n(rows * cols, "max rows")) return;
    const std::vector<float> x = fill(2100, rows * cols, -3.0f, 3.0f);
    check_all("rows=2^20", rows, cols, x, nullptr, nullptr, 1e-5f);
}

void test_ln_special_values() {
    for (std::int64_t cols : {1, 33, 257, 1024, 3000}) {
        const std::int64_t rows = 6;
        std::vector<float> x = fill(800 + cols, rows * cols, -5.0f, 5.0f);
        x[0 * cols + cols / 2] = kNaN;   // row 0: NaN -> NaN row
        x[1 * cols + 0] = kInf;          // row 1: +inf -> NaN row
        x[2 * cols + cols - 1] = -kInf;  // row 2: -inf -> NaN row
        // rows 3-5 finite
        std::vector<float> g = fill(900 + cols, cols, 0.5f, 1.5f);
        std::vector<float> b = fill(950 + cols, cols, -1.0f, 1.0f);
        g[0] = kNaN;            // column 0: NaN in every row
        if (cols > 1) b[1] = kInf;  // column 1: +inf in every finite row
        check_all("special", rows, cols, x, g.data(), b.data(), 1e-5f);
    }
}

// An outlier shifts the mean by |x_j - mu| / n; a reduction that dropped
// the outlier's partial would be far outside the bound at these sizes.
void test_ln_outlier_positions() {
    for (std::int64_t cols : {33, 257, 1000, 1024, 4097}) {
        std::vector<std::int64_t> pos = {0, 31, 32, 63, 255, 256, cols - 1};
        pos.erase(std::remove_if(pos.begin(), pos.end(), [&](std::int64_t j) { return j >= cols; }), pos.end());
        const auto rows = static_cast<std::int64_t>(pos.size());
        std::vector<float> x = fill(1000 + cols, rows * cols, -1.0f, 1.0f);
        for (std::int64_t r = 0; r < rows; ++r) x[r * cols + pos[r]] = 1000.0f;
        check_all("outlier", rows, cols, x, nullptr, nullptr, 1e-5f);
    }
}

// Rows (+c, -c, +c, ...) with eps = 0 and few elements per thread (cols 2
// and 64: at most 2 per lane/thread) make every step but the scale path
// exact: partial sums are 0 or 2^j * c, the mean is 0, d = +-c, the sum of
// squares is 2^j * fl(c^2), sqrtf(fl(c^2)) rounds back to c. So
// y = fl(c * fl(1 / c)) = +-1 or +-(1 - 2^-24) exactly. This pins the
// scale computation more tightly than the general bound: 1/sqrtf
// (correctly rounded) passes, while rsqrtf (up to 2 ulp) can give other
// values. The c values have full 24-bit mantissas across [2^-20, 2^20].
void test_ln_scale_path_rounding() {
    const float below_one = std::nextafter(1.0f, 0.0f);
    for (std::int64_t cols : {2, 64}) {
        const std::int64_t rows = 4096;
        const std::vector<float> m = fill(1600 + cols, rows, 1.0f, 2.0f);
        const std::vector<float> e = fill(1700 + cols, rows, -20.0f, 20.0f);
        std::vector<float> x(static_cast<std::size_t>(rows * cols));
        for (std::int64_t r = 0; r < rows; ++r) {
            const float c = std::ldexp(m[static_cast<std::size_t>(r)],
                                       static_cast<int>(std::floor(e[static_cast<std::size_t>(r)])));
            for (std::int64_t j = 0; j < cols; ++j) x[static_cast<std::size_t>(r * cols + j)] = (j % 2) ? -c : c;
        }
        for (kl_layernorm_impl impl : kImpls) {
            const std::vector<float> y = run_ln(impl, rows, cols, x, nullptr, nullptr, 0.0f);
            std::int64_t wrong = 0;
            for (std::size_t i = 0; i < y.size(); ++i) {
                const float a = std::fabs(y[i]);
                const bool sign_ok = std::signbit(y[i]) == std::signbit(x[i]);
                wrong += ((a == 1.0f || a == below_one) && sign_ok) ? 0 : 1;
            }
            if (wrong) std::printf("  scale path %s cols=%lld: %lld outputs not +-1 or +-(1 - 2^-24)\n",
                                   impl_name(impl), static_cast<long long>(cols), static_cast<long long>(wrong));
            EXPECT(wrong == 0);
        }
    }
}

// FUSED_SMEM runs FUSED_BLOCK's arithmetic in the same order, only reading
// the row from shared memory: the outputs must be identical bit for bit
// (docs/layernorm.md), including special values.
void test_ln_smem_bitwise_equals_block() {
    std::uint64_t seed = 1800;
    for (std::int64_t cols : {1, 2, 255, 256, 257, 1000, 1025, 4096, 8191, 8192}) {
        const std::int64_t rows = 9;
        std::vector<float> x = fill(seed++, rows * cols, -50.0f, 50.0f);
        for (std::int64_t j = 0; j < cols; ++j) x[static_cast<std::size_t>(rows / 2 * cols + j)] += 1000.0f;
        x[static_cast<std::size_t>(cols - 1)] = kNaN;  // row 0
        const std::vector<float> g = fill(seed++, cols, 0.5f, 1.5f);
        const std::vector<float> b = fill(seed++, cols, -1.0f, 1.0f);
        for (bool affine : {true, false}) {
            const float* gp = affine ? g.data() : nullptr;
            const float* bp = affine ? b.data() : nullptr;
            const std::vector<float> ys = run_ln(KL_LAYERNORM_FUSED_SMEM, rows, cols, x, gp, bp, 1e-5f);
            const std::vector<float> yb = run_ln(KL_LAYERNORM_FUSED_BLOCK, rows, cols, x, gp, bp, 1e-5f);
            const bool same =
                std::equal(ys.begin(), ys.end(), yb.begin(), [](float p, float q) { return same_float(p, q); });
            if (!same) {
                std::printf("  smem != block at cols=%lld affine=%d\n", static_cast<long long>(cols),
                            static_cast<int>(affine));
            }
            EXPECT(same);
        }
    }
}

void test_ln_deterministic() {
    for (kl_layernorm_impl impl : {KL_LAYERNORM_MULTIPASS, KL_LAYERNORM_FUSED_BLOCK, KL_LAYERNORM_FUSED_SMEM,
                                   KL_LAYERNORM_FUSED_WARP}) {
        const std::int64_t rows = 64, cols = impl == KL_LAYERNORM_FUSED_WARP ? 1000 : 3001;
        const std::vector<float> x = fill(1100, rows * cols, -20.0f, 20.0f);
        const std::vector<float> a = run_ln(impl, rows, cols, x, nullptr, nullptr, 1e-5f);
        const std::vector<float> b = run_ln(impl, rows, cols, x, nullptr, nullptr, 1e-5f);
        EXPECT(std::equal(a.begin(), a.end(), b.begin(), [](float p, float q) { return same_float(p, q); }));
    }
}

void test_ln_inplace_host() {
    const std::int64_t rows = 5, cols = 777;
    std::vector<float> x = fill(1200, rows * cols, -3.0f, 3.0f);
    const std::vector<float> want = run_ln(KL_LAYERNORM_FUSED_BLOCK, rows, cols, x, nullptr, nullptr, 1e-5f);
    EXPECT_STATUS(kl_layernorm_host(rows, cols, x.data(), nullptr, nullptr, 1e-5f, x.data(), KL_LAYERNORM_FUSED_BLOCK),
                  KL_OK);
    EXPECT(std::equal(x.begin(), x.end(), want.begin(), [](float p, float q) { return same_float(p, q); }));
}

// Guard bands around the output and workspace (out-of-bounds writes would
// otherwise go unnoticed: memcheck is unavailable here). rows % 4 != 0.
void test_ln_guard_bands() {
    using kl::detail::DeviceBuffer;
    const std::int64_t kGuard = 4096;
    const std::uint32_t kPattern = 0x7fbadbadu;
    float sentinel;
    std::memcpy(&sentinel, &kPattern, sizeof(sentinel));
    for (const Shape& s : {Shape{37, 1}, Shape{37, 33}, Shape{37, 1024}, Shape{5, 1025}, Shape{5, 8192},
                           Shape{3, 10007}}) {
        const std::int64_t n = s.rows * s.cols;
        const std::vector<float> x = fill(1300 + s.cols, n, -5.0f, 5.0f);
        const std::vector<float> g = fill(1350 + s.cols, s.cols, 0.5f, 1.5f);
        const std::vector<float> b = fill(1360 + s.cols, s.cols, -1.0f, 1.0f);
        for (kl_layernorm_impl impl : kImpls) {
            if (!impl_supports(impl, s.cols)) continue;
            std::size_t ws_bytes = 0;
            EXPECT_STATUS(kl_layernorm_workspace_bytes(s.rows, s.cols, impl, &ws_bytes), KL_OK);
            const auto ws_floats = static_cast<std::int64_t>(ws_bytes / sizeof(float));
            const std::int64_t ty = n + 2 * kGuard, tw = ws_floats + 2 * kGuard;
            std::vector<float> hy(static_cast<std::size_t>(ty), sentinel), hw(static_cast<std::size_t>(tw), sentinel);
            DeviceBuffer dx, dg, db, dy, dw;
            EXPECT_STATUS(DeviceBuffer::allocate(static_cast<std::size_t>(n) * 4, dx), KL_OK);
            EXPECT_STATUS(DeviceBuffer::allocate(static_cast<std::size_t>(s.cols) * 4, dg), KL_OK);
            EXPECT_STATUS(DeviceBuffer::allocate(static_cast<std::size_t>(s.cols) * 4, db), KL_OK);
            EXPECT_STATUS(DeviceBuffer::allocate(static_cast<std::size_t>(ty) * 4, dy), KL_OK);
            EXPECT_STATUS(DeviceBuffer::allocate(static_cast<std::size_t>(tw) * 4, dw), KL_OK);
            EXPECT(cudaMemcpy(dx.get(), x.data(), static_cast<std::size_t>(n) * 4, cudaMemcpyHostToDevice) == cudaSuccess);
            EXPECT(cudaMemcpy(dg.get(), g.data(), static_cast<std::size_t>(s.cols) * 4, cudaMemcpyHostToDevice) == cudaSuccess);
            EXPECT(cudaMemcpy(db.get(), b.data(), static_cast<std::size_t>(s.cols) * 4, cudaMemcpyHostToDevice) == cudaSuccess);
            EXPECT(cudaMemcpy(dy.get(), hy.data(), static_cast<std::size_t>(ty) * 4, cudaMemcpyHostToDevice) == cudaSuccess);
            EXPECT(cudaMemcpy(dw.get(), hw.data(), static_cast<std::size_t>(tw) * 4, cudaMemcpyHostToDevice) == cudaSuccess);
            EXPECT_STATUS(kl_layernorm_device(s.rows, s.cols, dx.as<float>(), dg.as<float>(), db.as<float>(), 1e-5f,
                                              dy.as<float>() + kGuard, impl, ws_floats ? dw.as<float>() + kGuard : nullptr,
                                              nullptr),
                          KL_OK);
            EXPECT(cudaDeviceSynchronize() == cudaSuccess);
            EXPECT(cudaMemcpy(hy.data(), dy.get(), static_cast<std::size_t>(ty) * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
            EXPECT(cudaMemcpy(hw.data(), dw.get(), static_cast<std::size_t>(tw) * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
            auto band_ok = [&](const std::vector<float>& v, std::int64_t from, std::int64_t to) {
                for (std::int64_t i = from; i < to; ++i) {
                    if (kl_test::bits(v[static_cast<std::size_t>(i)]) != kPattern) return false;
                }
                return true;
            };
            const bool y_ok = band_ok(hy, 0, kGuard) && band_ok(hy, kGuard + n, ty);
            const bool w_ok = band_ok(hw, 0, kGuard) && band_ok(hw, kGuard + ws_floats, tw);
            if (!y_ok || !w_ok) std::printf("  guard band overwritten: %s %lldx%lld\n", impl_name(impl),
                                            static_cast<long long>(s.rows), static_cast<long long>(s.cols));
            EXPECT(y_ok);
            EXPECT(w_ok);
            const std::vector<float> want = run_ln(impl, s.rows, s.cols, x, g.data(), b.data(), 1e-5f);
            bool same = true;
            for (std::int64_t i = 0; i < n; ++i) {
                same = same && same_float(hy[static_cast<std::size_t>(kGuard + i)], want[static_cast<std::size_t>(i)]);
            }
            EXPECT(same);
        }
    }
}

void test_ln_invalid_and_unsupported() {
    const float kSentinel = -12345.0f;
    std::vector<float> x(16, 1.0f), y(16, kSentinel), g(4, 1.0f), b(4, 0.0f);
    const std::uint64_t mallocs = kl::detail::device_malloc_count();
    const auto bad = static_cast<kl_layernorm_impl>(7);
    const auto neg = static_cast<kl_layernorm_impl>(-1);
    EXPECT_STATUS(kl_layernorm_host(-1, 4, x.data(), nullptr, nullptr, 1e-5f, y.data(), KL_LAYERNORM_AUTO), KL_ERR_INVALID_ARGUMENT);
    EXPECT_STATUS(kl_layernorm_host(4, 4, x.data(), nullptr, nullptr, 1e-5f, y.data(), bad), KL_ERR_INVALID_ARGUMENT);
    EXPECT_STATUS(kl_layernorm_host(4, 4, x.data(), nullptr, nullptr, 1e-5f, y.data(), neg), KL_ERR_INVALID_ARGUMENT);
    for (float eps : {-1e-5f, kNaN, kInf, -kInf}) {
        EXPECT_STATUS(kl_layernorm_host(4, 4, x.data(), nullptr, nullptr, eps, y.data(), KL_LAYERNORM_AUTO),
                      KL_ERR_INVALID_ARGUMENT);
    }
    EXPECT_STATUS(kl_layernorm_host(4, 4, x.data(), g.data(), nullptr, 1e-5f, y.data(), KL_LAYERNORM_AUTO), KL_ERR_INVALID_ARGUMENT);
    EXPECT_STATUS(kl_layernorm_host(4, 4, x.data(), nullptr, b.data(), 1e-5f, y.data(), KL_LAYERNORM_AUTO), KL_ERR_INVALID_ARGUMENT);
    EXPECT_STATUS(kl_layernorm_host(4, 4, nullptr, nullptr, nullptr, 1e-5f, y.data(), KL_LAYERNORM_AUTO), KL_ERR_INVALID_ARGUMENT);
    EXPECT_STATUS(kl_layernorm_host(0, 4, nullptr, nullptr, nullptr, 1e-5f, nullptr, KL_LAYERNORM_AUTO), KL_OK);
    EXPECT_STATUS(kl_layernorm_host(1, KL_LAYERNORM_MAX_COLS + 1, x.data(), nullptr, nullptr, 1e-5f, y.data(), KL_LAYERNORM_AUTO),
                  KL_ERR_UNSUPPORTED);
    EXPECT_STATUS(kl_layernorm_host(65, 1 << 20, x.data(), nullptr, nullptr, 1e-5f, y.data(), KL_LAYERNORM_AUTO), KL_ERR_UNSUPPORTED);
    EXPECT_STATUS(kl_layernorm_host(1, 1025, x.data(), nullptr, nullptr, 1e-5f, y.data(), KL_LAYERNORM_FUSED_WARP), KL_ERR_UNSUPPORTED);
    EXPECT_STATUS(kl_layernorm_host(1, 8193, x.data(), nullptr, nullptr, 1e-5f, y.data(), KL_LAYERNORM_FUSED_SMEM), KL_ERR_UNSUPPORTED);
    kl_test::expect_last_error_mentions("cols <= 8192", KL_HERE, __LINE__);
    for (float v : y) EXPECT(same_float(v, kSentinel));
    EXPECT(kl::detail::device_malloc_count() == mallocs);
    EXPECT_STATUS(kl_layernorm_supported(64, 1 << 20, 0.0f, KL_LAYERNORM_AUTO), KL_OK);
    EXPECT_STATUS(kl_layernorm_supported(1, 1, kNaN, KL_LAYERNORM_AUTO), KL_ERR_INVALID_ARGUMENT);

    // Device entry point: overlap and alignment checks.
    using kl::detail::DeviceBuffer;
    DeviceBuffer d;
    EXPECT_STATUS(DeviceBuffer::allocate(512 * sizeof(float), d), KL_OK);
    EXPECT(cudaMemset(d.get(), 0, 512 * sizeof(float)) == cudaSuccess);
    float* p = d.as<float>();
    // x = p[0..64), y = p[64..128), gamma = p[128..144), beta = p[144..160), ws = p[160..168)
    EXPECT_STATUS(kl_layernorm_device(4, 16, p, p + 128, p + 144, 1e-5f, p + 64, KL_LAYERNORM_MULTIPASS, p + 160, nullptr), KL_OK);
    EXPECT_STATUS(kl_layernorm_device(4, 16, p, p + 128, p + 144, 1e-5f, p + 60, KL_LAYERNORM_FUSED_BLOCK, nullptr, nullptr),
                  KL_ERR_INVALID_ARGUMENT);  // y overlaps x
    EXPECT_STATUS(kl_layernorm_device(4, 16, p, p + 120, p + 144, 1e-5f, p + 64, KL_LAYERNORM_FUSED_BLOCK, nullptr, nullptr),
                  KL_ERR_INVALID_ARGUMENT);  // y overlaps gamma
    EXPECT_STATUS(kl_layernorm_device(4, 16, p, p + 128, p + 144, 1e-5f, p + 64, KL_LAYERNORM_MULTIPASS, p + 124, nullptr),
                  KL_ERR_INVALID_ARGUMENT);  // workspace overlaps y
    EXPECT_STATUS(kl_layernorm_device(4, 16, p, p + 128, p + 144, 1e-5f, p + 64, KL_LAYERNORM_MULTIPASS, p + 140, nullptr),
                  KL_ERR_INVALID_ARGUMENT);  // workspace overlaps gamma/beta
    EXPECT_STATUS(kl_layernorm_device(4, 16, p, p + 128, p + 144, 1e-5f, p + 64, KL_LAYERNORM_MULTIPASS, nullptr, nullptr),
                  KL_ERR_INVALID_ARGUMENT);  // missing workspace
    const auto mis = reinterpret_cast<float*>(reinterpret_cast<char*>(p + 64) + 2);
    EXPECT_STATUS(kl_layernorm_device(4, 16, p, nullptr, nullptr, 1e-5f, mis, KL_LAYERNORM_FUSED_BLOCK, nullptr, nullptr),
                  KL_ERR_INVALID_ARGUMENT);  // misaligned y
    auto skew = [](const float* q) { return reinterpret_cast<const float*>(reinterpret_cast<const char*>(q) + 2); };
    EXPECT_STATUS(kl_layernorm_device(4, 16, skew(p + 200), p + 128, p + 144, 1e-5f, p + 64, KL_LAYERNORM_FUSED_BLOCK,
                                      nullptr, nullptr),
                  KL_ERR_INVALID_ARGUMENT);  // misaligned x
    EXPECT_STATUS(kl_layernorm_device(4, 16, p, skew(p + 128), p + 144, 1e-5f, p + 64, KL_LAYERNORM_FUSED_BLOCK,
                                      nullptr, nullptr),
                  KL_ERR_INVALID_ARGUMENT);  // misaligned gamma
    EXPECT_STATUS(kl_layernorm_device(4, 16, p, p + 128, skew(p + 144), 1e-5f, p + 64, KL_LAYERNORM_FUSED_BLOCK,
                                      nullptr, nullptr),
                  KL_ERR_INVALID_ARGUMENT);  // misaligned beta
    EXPECT_STATUS(kl_layernorm_device(4, 16, p, p + 128, p + 144, 1e-5f, p + 64, KL_LAYERNORM_MULTIPASS,
                                      const_cast<float*>(skew(p + 160)), nullptr),
                  KL_ERR_INVALID_ARGUMENT);  // misaligned workspace
    EXPECT_STATUS(kl_layernorm_device(4, 16, p, p + 128, p + 144, 1e-5f, p + 64, KL_LAYERNORM_MULTIPASS, p + 60, nullptr),
                  KL_ERR_INVALID_ARGUMENT);  // workspace overlaps x (and y)
    EXPECT_STATUS(kl_layernorm_device(4, 16, p, p + 128, p + 144, 1e-5f, p + 64, KL_LAYERNORM_MULTIPASS, p + 56, nullptr),
                  KL_ERR_INVALID_ARGUMENT);  // workspace overlaps x only
    EXPECT_STATUS(kl_layernorm_device(4, 16, p, p + 128, nullptr, 1e-5f, p + 64, KL_LAYERNORM_FUSED_BLOCK, nullptr, nullptr),
                  KL_ERR_INVALID_ARGUMENT);  // gamma without beta
    // x may alias gamma (both read-only).
    EXPECT_STATUS(kl_layernorm_device(4, 16, p, p, p + 16, 1e-5f, p + 64, KL_LAYERNORM_FUSED_BLOCK, nullptr, nullptr), KL_OK);
    EXPECT(cudaDeviceSynchronize() == cudaSuccess);
}

void test_ln_auto_dispatch() {
    using kl::cuda::layernorm_resolve;
    EXPECT(layernorm_resolve(1, KL_LAYERNORM_AUTO) == KL_LAYERNORM_FUSED_WARP);
    EXPECT(layernorm_resolve(1024, KL_LAYERNORM_AUTO) == KL_LAYERNORM_FUSED_WARP);
    EXPECT(layernorm_resolve(1025, KL_LAYERNORM_AUTO) == KL_LAYERNORM_FUSED_SMEM);
    EXPECT(layernorm_resolve(8192, KL_LAYERNORM_AUTO) == KL_LAYERNORM_FUSED_SMEM);
    EXPECT(layernorm_resolve(8193, KL_LAYERNORM_AUTO) == KL_LAYERNORM_FUSED_BLOCK);
    EXPECT(layernorm_resolve(1 << 20, KL_LAYERNORM_AUTO) == KL_LAYERNORM_FUSED_BLOCK);
    EXPECT(layernorm_resolve(10, KL_LAYERNORM_MULTIPASS) == KL_LAYERNORM_MULTIPASS);
    EXPECT(kl::cuda::layernorm_workspace_floats(10, KL_LAYERNORM_MULTIPASS) == 20);
    EXPECT(kl::cuda::layernorm_workspace_floats(10, KL_LAYERNORM_FUSED_BLOCK) == 0);
    EXPECT(kl::cuda::layernorm_workspace_floats(10, KL_LAYERNORM_FUSED_SMEM) == 0);
}

// The study kernel: on well-conditioned rows it meets its own (derived)
// bound; on offset rows its variance error is quadratic in mean/std, so it
// is measurably worse than the two-pass kernels' bound there.
void test_ln_naive_instability_study() {
    using kl::detail::DeviceBuffer;
    auto run_naive = [&](std::int64_t rows, std::int64_t cols, const std::vector<float>& x) {
        DeviceBuffer dx, dy;
        const std::size_t bytes = static_cast<std::size_t>(rows * cols) * 4;
        EXPECT_STATUS(DeviceBuffer::allocate(bytes, dx), KL_OK);
        EXPECT_STATUS(DeviceBuffer::allocate(bytes, dy), KL_OK);
        EXPECT(cudaMemcpy(dx.get(), x.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess);
        EXPECT_STATUS(kl::cuda::launch_layernorm_naive(rows, cols, dx.as<float>(), nullptr, nullptr, 1e-5f, dy.as<float>(), 0),
                      KL_OK);
        std::vector<float> y(x.size());
        EXPECT(cudaMemcpy(y.data(), dy.get(), bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
        return y;
    };
    const std::int64_t rows = 16, cols = 4096;
    const std::vector<float> centered = fill(1400, rows * cols, -1.0f, 1.0f);
    const std::vector<float> yc = run_naive(rows, cols, centered);
    const CheckResult own = check_layernorm(KL_LAYERNORM_FUSED_BLOCK, rows, cols, centered.data(), nullptr, nullptr, 1e-5f,
                                            yc.data(), /*naive=*/true);
    EXPECT(own.violations == 0);
    std::printf("  naive, centered rows: worst %.3f of its own bound\n", own.worst_ratio);
    // Observation over 8 fixed seeds per offset (deterministic, not a derived
    // property): the naive kernel's worst error is >= 10x the two-pass bound
    // on every seed, while the two-pass kernel meets it.
    for (float offset : {100.0f, 1000.0f, 10000.0f}) {
        double min_ratio = std::numeric_limits<double>::infinity(), max_two_pass = 0;
        std::int64_t nan_out = 0;
        for (std::uint64_t seed = 1500; seed < 1508; ++seed) {
            const std::vector<float> xo = fill(seed, rows * cols, offset - 1.0f, offset + 1.0f);
            const std::vector<float> yn = run_naive(rows, cols, xo);
            const CheckResult vs_two_pass =
                check_layernorm(KL_LAYERNORM_FUSED_BLOCK, rows, cols, xo.data(), nullptr, nullptr, 1e-5f, yn.data());
            const std::vector<float> yt = run_ln(KL_LAYERNORM_FUSED_BLOCK, rows, cols, xo, nullptr, nullptr, 1e-5f);
            const CheckResult two_pass =
                check_layernorm(KL_LAYERNORM_FUSED_BLOCK, rows, cols, xo.data(), nullptr, nullptr, 1e-5f, yt.data());
            for (float v : yn) nan_out += std::isnan(v) ? 1 : 0;
            EXPECT(two_pass.violations == 0);
            min_ratio = std::min(min_ratio, vs_two_pass.worst_ratio);
            max_two_pass = std::max(max_two_pass, two_pass.worst_ratio);
        }
        std::printf("  offset %g, 8 seeds: two-pass worst %.3f of its bound; naive worst error >= %.3g x the "
                    "two-pass bound on every seed, %lld NaN outputs in total\n",
                    offset, max_two_pass, min_ratio, static_cast<long long>(nan_out));
        EXPECT(min_ratio >= 10.0);
    }
}

}  // namespace

int main(int argc, char** argv) {
    static const kl_test::TestCase tests[] = {
        {"ln_uniform_affine", test_ln_uniform_affine},
        {"ln_uniform_no_affine", test_ln_uniform_no_affine},
        {"ln_large_offset", test_ln_large_offset},
        {"ln_eps_dominated", test_ln_eps_dominated},
        {"ln_eps_values", test_ln_eps_values},
        {"ln_constant_rows", test_ln_constant_rows},
        {"ln_extreme_in_domain", test_ln_extreme_in_domain},
        {"ln_outside_domain_documented", test_ln_outside_domain_documented},
        {"ln_max_rows", test_ln_max_rows},
        {"ln_outlier_first_element", test_ln_outlier_first_element},
        {"ln_padded_launcher_and_occupancy", test_ln_padded_launcher_and_occupancy},
        {"ln_special_values", test_ln_special_values},
        {"ln_outlier_positions", test_ln_outlier_positions},
        {"ln_scale_path_rounding", test_ln_scale_path_rounding},
        {"ln_smem_bitwise_equals_block", test_ln_smem_bitwise_equals_block},
        {"ln_deterministic", test_ln_deterministic},
        {"ln_inplace_host", test_ln_inplace_host},
        {"ln_guard_bands", test_ln_guard_bands},
        {"ln_invalid_and_unsupported", test_ln_invalid_and_unsupported},
        {"ln_auto_dispatch", test_ln_auto_dispatch},
        {"ln_naive_instability_study", test_ln_naive_instability_study},
    };
    return kl_test::run_tests(argc, argv, tests);
}
