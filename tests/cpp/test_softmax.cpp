// C++ tests for row-wise softmax (docs/softmax.md).
//
// Reference: the definition evaluated in double precision on the CPU.
// Tolerance: the derived per-element bound B_i from docs/softmax.md,
//   |y - y_ref| <= 1.01 * B_i * y_ref + 2^-147.
// Every case reports the worst observed error as a fraction of its bound.
//
// Workloads: at most 2^20 elements per case; each launch is sub-millisecond.
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "common/device_buffer.hpp"
#include "cuda/kernels.hpp"
#include "harness.hpp"
#include "softmax_reference.hpp"
#include "kernellens/kernellens.h"

namespace {

using kl_test::fill;
using kl_test::same_float;
using kl_test::check_softmax;
using kl_test::CheckResult;
using kl_test::skip_for_max_n;

const float kInf = std::numeric_limits<float>::infinity();
const float kNaN = std::numeric_limits<float>::quiet_NaN();

const kl_softmax_impl kImpls[] = {KL_SOFTMAX_AUTO, KL_SOFTMAX_MULTIPASS, KL_SOFTMAX_FUSED_ONLINE,
                                  KL_SOFTMAX_FUSED_WARP};

const char* impl_name(kl_softmax_impl impl) {
    switch (impl) {
        case KL_SOFTMAX_AUTO: return "auto";
        case KL_SOFTMAX_MULTIPASS: return "multipass";
        case KL_SOFTMAX_FUSED_ONLINE: return "fused_online";
        case KL_SOFTMAX_FUSED_WARP: return "fused_warp";
    }
    return "?";
}

bool impl_supports(kl_softmax_impl impl, std::int64_t cols) {
    return impl != KL_SOFTMAX_FUSED_WARP || cols <= KL_SOFTMAX_WARP_MAX_COLS;
}

std::vector<float> run_softmax(kl_softmax_impl impl, std::int64_t rows, std::int64_t cols,
                               const std::vector<float>& x) {
    std::vector<float> y(x.size(), -12345.0f);
    EXPECT_STATUS(kl_softmax_host(rows, cols, x.data(), y.data(), impl), KL_OK);
    return y;
}

// Runs every implementation that supports `cols` and checks it.
void check_all_impls(const char* label, std::int64_t rows, std::int64_t cols,
                     const std::vector<float>& x, double* worst_seen = nullptr,
                     bool exact_differences = false) {
    for (kl_softmax_impl impl : kImpls) {
        if (!impl_supports(impl, cols)) continue;
        const std::vector<float> y = run_softmax(impl, rows, cols, x);
        const CheckResult c = check_softmax(impl, rows, cols, x.data(), y.data(), exact_differences);
        if (c.violations != 0 || c.nan_mismatches != 0) {
            std::printf("  %s %s rows=%lld cols=%lld: %lld violations, %lld NaN mismatches, worst "
                        "error %.3g x bound\n",
                        label, impl_name(impl), static_cast<long long>(rows),
                        static_cast<long long>(cols), static_cast<long long>(c.violations),
                        static_cast<long long>(c.nan_mismatches), c.worst_ratio);
        }
        EXPECT(c.violations == 0);
        EXPECT(c.nan_mismatches == 0);
        EXPECT(c.row_sum_violations == 0);
        if (worst_seen != nullptr) *worst_seen = std::max(*worst_seen, c.worst_ratio);
    }
}

struct Shape {
    std::int64_t rows, cols;
};

// Column counts around warp (32), block (256), the warp kernel's
// values-per-lane steps (32/64/128/256/512/1024) and its 1024 limit, plus
// large rows up to the 2^20 maximum.
const Shape kShapes[] = {
    {37, 1},    {37, 2},    {37, 31},   {37, 32},   {37, 33},    {37, 63},    {37, 64},
    {37, 65},   {37, 127},  {37, 128},  {37, 129},  {37, 255},   {37, 256},   {37, 257},
    {37, 511},  {37, 512},  {37, 513},  {37, 1000}, {37, 1023},  {37, 1024},  {37, 1025},
    {37, 2048}, {37, 4097}, {3, 10007}, {3, 65536}, {1, 1 << 20},
};

// ---------------------------------------------------------------------------

void run_distribution(const char* label, float lo, float hi, std::uint64_t seed_base) {
    double worst = 0;
    std::uint64_t seed = seed_base;
    for (const Shape& s : kShapes) {
        if (skip_for_max_n(s.rows * s.cols, label)) continue;
        const std::vector<float> x = fill(seed++, s.rows * s.cols, lo, hi);
        check_all_impls(label, s.rows, s.cols, x, &worst);
    }
    std::printf("  %s: worst error %.3f of the derived bound\n", label, worst);
}

void test_softmax_uniform() { run_distribution("uniform[-10,10]", -10.0f, 10.0f, 100); }

// Without the max subtraction, exp(1000) overflows to inf.
void test_softmax_large_offset() { run_distribution("uniform[1000,1010]", 1000.0f, 1010.0f, 200); }

// Many terms underflow (exp(-200) is 0 in float) or land in the subnormal
// range; exercises the 2^-147 absolute slack.
void test_softmax_wide_range() { run_distribution("uniform[-100,100]", -100.0f, 100.0f, 300); }

// Inputs on the grid k * 2^-10 with |x| <= 128: any difference of two inputs
// is a multiple of 2^-10 of magnitude <= 256, i.e. at most 18 significant
// bits, so every float subtraction of inputs is exact and the tighter bound
// (no d_i, D terms) applies. Verified per row below, not assumed.
std::vector<float> grid_values(std::uint64_t seed, std::int64_t n, float lo, float hi) {
    std::vector<float> v = fill(seed, n, lo, hi);
    for (float& f : v) f = std::round(f * 1024.0f) / 1024.0f;
    return v;
}

bool all_differences_exact(const float* x, std::int64_t cols) {
    float m = -kInf;
    for (std::int64_t j = 0; j < cols; ++j) m = std::max(m, x[j]);
    for (std::int64_t j = 0; j < cols; ++j) {
        // Every input is k * 2^-10 with |k| <= 2^17; then any pairwise
        // difference is exact. Check the premise directly.
        const double scaled = static_cast<double>(x[j]) * 1024.0;
        if (scaled != std::round(scaled) || std::fabs(x[j]) > 128.0f) return false;
    }
    (void)m;
    return true;
}

void test_softmax_grid_inputs_tight_bound() {
    double worst = 0;
    std::uint64_t seed = 800;
    for (const Shape& s : kShapes) {
        if (skip_for_max_n(s.rows * s.cols, "grid inputs")) continue;
        const std::vector<float> x = grid_values(seed++, s.rows * s.cols, -16.0f, 16.0f);
        bool premise = true;
        for (std::int64_t r = 0; r < s.rows; ++r) premise = premise && all_differences_exact(&x[r * s.cols], s.cols);
        EXPECT(premise);
        check_all_impls("grid[-16,16] tight", s.rows, s.cols, x, &worst, /*exact_differences=*/true);
    }
    std::printf("  grid[-16,16] (tight bound): worst error %.3f of the bound\n", worst);
}

// Two-element rows [0, -t]: y = (1, e^-t) / (1 + e^-t). With t on the 2^-10
// grid the subtractions are exact, so the result's error is dominated by
// one expf call; this isolates exp accuracy (e.g. a fast-math __expf would
// be ~1.17*t ulp off and fail).
void test_softmax_exp_accuracy() {
    const std::int64_t rows = 87 * 64;  // t = 0, 1/64, ..., ~87 (e^-87 is still a normal float)
    std::vector<float> x(static_cast<std::size_t>(rows * 2));
    for (std::int64_t r = 0; r < rows; ++r) {
        const float t = static_cast<float>(r) / 64.0f;
        x[2 * r] = (r % 2) ? 0.0f : -t;  // alternate the position of the maximum
        x[2 * r + 1] = (r % 2) ? -t : 0.0f;
    }
    double worst = 0;
    check_all_impls("exp accuracy [0,-t]", rows, 2, x, &worst, /*exact_differences=*/true);
    std::printf("  exp accuracy rows [0,-t]: worst error %.3f of the bound\n", worst);
}

// Equal entries: every term is exp(0) = 1, the sum is exactly cols (< 2^24),
// so the only rounding is the final division: y must equal float(1/cols)
// bit for bit, for every implementation.
void test_softmax_constant_rows() {
    for (const Shape& s : kShapes) {
        if (skip_for_max_n(s.rows * s.cols, "constant rows")) continue;
        const std::vector<float> x(static_cast<std::size_t>(s.rows * s.cols), 3.25f);
        const float want = 1.0f / static_cast<float>(s.cols);
        for (kl_softmax_impl impl : kImpls) {
            if (!impl_supports(impl, s.cols)) continue;
            const std::vector<float> y = run_softmax(impl, s.rows, s.cols, x);
            std::int64_t wrong = 0;
            for (float v : y) wrong += same_float(v, want) ? 0 : 1;
            if (wrong) std::printf("  constant %s cols=%lld: %lld elements != 1/cols\n",
                                   impl_name(impl), static_cast<long long>(s.cols),
                                   static_cast<long long>(wrong));
            EXPECT(wrong == 0);
        }
    }
}

// One large entry per row: y is ~1 there and exp(-80)-sized elsewhere
// (still normal floats), so relative accuracy of tiny outputs is checked.
void test_softmax_one_hot() {
    for (std::int64_t cols : {33, 1000, 5000}) {
        const std::int64_t rows = 9;
        std::vector<float> x = fill(400 + cols, rows * cols, -1.0f, 1.0f);
        for (std::int64_t r = 0; r < rows; ++r) x[r * cols + (r * 7) % cols] = 80.0f;
        check_all_impls("one-hot", rows, cols, x);
    }
}

// -inf entries (masking), and the rows the definition makes NaN.
void test_softmax_special_values() {
    for (std::int64_t cols : {1, 31, 32, 33, 257, 1024, 3000}) {
        const std::int64_t rows = 8;
        std::vector<float> x = fill(500 + cols, rows * cols, -5.0f, 5.0f);
        auto row = [&](std::int64_t r) { return x.data() + r * cols; };
        // row 0: every third entry -inf (finite entries remain)
        for (std::int64_t j = 0; j < cols; j += 3) row(0)[j] = -kInf;
        if (cols > 1) row(0)[1] = 0.5f;
        // row 1: all -inf -> NaN
        std::fill(row(1), row(1) + cols, -kInf);
        // row 2: one NaN -> NaN row
        row(2)[cols / 2] = kNaN;
        // row 3: one +inf -> NaN row
        row(3)[cols - 1] = kInf;
        // row 4: -inf everywhere except the LAST element (the online
        //        recurrence must not turn early -inf into NaN)
        std::fill(row(4), row(4) + cols, -kInf);
        row(4)[cols - 1] = 2.0f;
        // row 5: first element -inf, then finite
        row(5)[0] = -kInf;
        // rows 6-7: unchanged finite data
        check_all_impls("special", rows, cols, x);

        // Row 4 must be exactly one-hot: 1.0 at the end, +0 elsewhere.
        const std::vector<float> y = run_softmax(KL_SOFTMAX_FUSED_ONLINE, rows, cols, x);
        bool exact = same_float(y[4 * cols + cols - 1], 1.0f);
        for (std::int64_t j = 0; j + 1 < cols; ++j) exact = exact && same_float(y[4 * cols + j], 0.0f);
        EXPECT(exact);
    }
}

// A wrong row max mostly cancels out of softmax (exp(x - m) / sum exp(x - m)
// is the same for any m) unless it causes overflow. A +100 spike on a [-1, 1]
// background makes a reduction that drops the spike's partial overflow:
// exp(100 - ~1) = inf in float. One row per spike position, covering the
// first/last lane of warps 0 and 1, the first element of the second block
// stride, and the last element.
void test_softmax_max_spike_positions() {
    for (std::int64_t cols : {33, 257, 1000, 1024, 4097, 70000}) {
        std::vector<std::int64_t> positions = {0, 31, 32, 63, 255, 256, cols - 1};
        positions.erase(std::remove_if(positions.begin(), positions.end(),
                                       [&](std::int64_t j) { return j >= cols; }),
                        positions.end());
        const auto rows = static_cast<std::int64_t>(positions.size());
        std::vector<float> x = fill(1100 + cols, rows * cols, -1.0f, 1.0f);
        for (std::int64_t r = 0; r < rows; ++r) x[r * cols + positions[r]] = 100.0f;
        double worst = 0;
        check_all_impls("max spike", rows, cols, x, &worst);
    }
}

// Extreme finite inputs: FLT_MAX - (-FLT_MAX) overflows to inf in float, so
// the -FLT_MAX term must still come out as exactly 0; signed zeros and
// subnormal inputs behave like 0.
void test_softmax_extreme_finite_values() {
    const float sub = std::numeric_limits<float>::denorm_min();
    const std::vector<float> row = {FLT_MAX, -FLT_MAX, 0.0f, -0.0f, sub, -sub, 1.0f, -1.0f};
    const auto cols = static_cast<std::int64_t>(row.size());
    std::vector<float> x;
    for (int r = 0; r < 3; ++r) x.insert(x.end(), row.begin(), row.end());
    // rows 1, 2: without FLT_MAX (maximum 1.0)
    x[1 * cols + 0] = 0.25f;
    x[2 * cols + 0] = -FLT_MAX;
    check_all_impls("extreme finite", 3, cols, x);
    for (kl_softmax_impl impl : kImpls) {
        const std::vector<float> y = run_softmax(impl, 3, cols, x);
        EXPECT(same_float(y[0], 1.0f));  // FLT_MAX dominates completely
        for (std::int64_t j = 1; j < cols; ++j) EXPECT(same_float(y[j], 0.0f));
    }
}

// Out-of-bounds writes are invisible to the other tests (small overruns land
// in allocation padding) and memcheck is unavailable here, so each
// implementation writes into the middle of a larger buffer whose guard bands
// hold a sentinel pattern; the bands (and a guarded workspace) must survive.
// rows % 4 != 0 exercises the warp kernel's partial last block.
void test_softmax_guard_bands() {
    using kl::detail::DeviceBuffer;
    const std::int64_t kGuard = 4096;  // floats on each side
    const std::uint32_t kPattern = 0x7fbadbadu;  // a NaN payload kernels never produce
    float sentinel;
    std::memcpy(&sentinel, &kPattern, sizeof(sentinel));
    for (const Shape& s : {Shape{37, 1}, Shape{37, 33}, Shape{37, 1024}, Shape{5, 1025}, Shape{3, 10007}}) {
        const std::int64_t n = s.rows * s.cols;
        const std::vector<float> x = fill(1200 + s.cols, n, -5.0f, 5.0f);
        for (kl_softmax_impl impl : kImpls) {
            if (!impl_supports(impl, s.cols)) continue;
            std::size_t ws_bytes = 0;
            EXPECT_STATUS(kl_softmax_workspace_bytes(s.rows, s.cols, impl, &ws_bytes), KL_OK);
            const std::int64_t ws_floats = static_cast<std::int64_t>(ws_bytes / sizeof(float));
            const std::int64_t total_y = n + 2 * kGuard;
            const std::int64_t total_ws = ws_floats + 2 * kGuard;
            std::vector<float> host_y(static_cast<std::size_t>(total_y), sentinel);
            std::vector<float> host_ws(static_cast<std::size_t>(total_ws), sentinel);
            DeviceBuffer dx, dy, dws;
            EXPECT_STATUS(DeviceBuffer::allocate(static_cast<std::size_t>(n) * 4, dx), KL_OK);
            EXPECT_STATUS(DeviceBuffer::allocate(static_cast<std::size_t>(total_y) * 4, dy), KL_OK);
            EXPECT_STATUS(DeviceBuffer::allocate(static_cast<std::size_t>(total_ws) * 4, dws), KL_OK);
            EXPECT(cudaMemcpy(dx.get(), x.data(), static_cast<std::size_t>(n) * 4, cudaMemcpyHostToDevice) == cudaSuccess);
            EXPECT(cudaMemcpy(dy.get(), host_y.data(), static_cast<std::size_t>(total_y) * 4, cudaMemcpyHostToDevice) == cudaSuccess);
            EXPECT(cudaMemcpy(dws.get(), host_ws.data(), static_cast<std::size_t>(total_ws) * 4, cudaMemcpyHostToDevice) == cudaSuccess);
            EXPECT_STATUS(kl_softmax_device(s.rows, s.cols, dx.as<float>(), dy.as<float>() + kGuard, impl,
                                            ws_floats ? dws.as<float>() + kGuard : nullptr, nullptr),
                          KL_OK);
            EXPECT(cudaDeviceSynchronize() == cudaSuccess);
            EXPECT(cudaMemcpy(host_y.data(), dy.get(), static_cast<std::size_t>(total_y) * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
            EXPECT(cudaMemcpy(host_ws.data(), dws.get(), static_cast<std::size_t>(total_ws) * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
            auto band_ok = [&](const std::vector<float>& v, std::int64_t from, std::int64_t to) {
                for (std::int64_t i = from; i < to; ++i) {
                    if (kl_test::bits(v[static_cast<std::size_t>(i)]) != kPattern) return false;
                }
                return true;
            };
            const bool y_ok = band_ok(host_y, 0, kGuard) && band_ok(host_y, kGuard + n, total_y);
            const bool ws_ok = band_ok(host_ws, 0, kGuard) && band_ok(host_ws, kGuard + ws_floats, total_ws);
            if (!y_ok || !ws_ok) {
                std::printf("  guard band overwritten: %s %lldx%lld (y %d, workspace %d)\n", impl_name(impl),
                            static_cast<long long>(s.rows), static_cast<long long>(s.cols), y_ok, ws_ok);
            }
            EXPECT(y_ok);
            EXPECT(ws_ok);
            // The interior must be the same result the host API produces.
            const std::vector<float> want = run_softmax(impl, s.rows, s.cols, x);
            bool same = true;
            for (std::int64_t i = 0; i < n; ++i) {
                same = same && same_float(host_y[static_cast<std::size_t>(kGuard + i)], want[static_cast<std::size_t>(i)]);
            }
            EXPECT(same);
        }
    }
}

void test_softmax_device_api_validation() {
    using kl::detail::DeviceBuffer;
    DeviceBuffer d;
    EXPECT_STATUS(DeviceBuffer::allocate(256 * sizeof(float), d), KL_OK);
    // Initialized, so a future initcheck run has nothing to report here.
    EXPECT(cudaMemset(d.get(), 0, 256 * sizeof(float)) == cudaSuccess);
    float* p = d.as<float>();
    std::size_t bytes = 1;
    EXPECT_STATUS(kl_softmax_workspace_bytes(10, 2048, KL_SOFTMAX_MULTIPASS, &bytes), KL_OK);
    EXPECT(bytes == 2 * 10 * sizeof(float));
    EXPECT_STATUS(kl_softmax_workspace_bytes(10, 2048, KL_SOFTMAX_AUTO, &bytes), KL_OK);  // 10 rows -> multipass
    EXPECT(bytes == 2 * 10 * sizeof(float));
    EXPECT_STATUS(kl_softmax_workspace_bytes(100, 2048, KL_SOFTMAX_AUTO, &bytes), KL_OK);  // online
    EXPECT(bytes == 0);
    EXPECT_STATUS(kl_softmax_workspace_bytes(1, 1025, KL_SOFTMAX_FUSED_WARP, &bytes), KL_ERR_UNSUPPORTED);
    EXPECT_STATUS(kl_softmax_workspace_bytes(1, 1, KL_SOFTMAX_AUTO, nullptr), KL_ERR_INVALID_ARGUMENT);
    // Overlapping x and y (in-place included) are refused: kernels use __restrict__.
    EXPECT_STATUS(kl_softmax_device(4, 16, p, p, KL_SOFTMAX_FUSED_ONLINE, nullptr, nullptr), KL_ERR_INVALID_ARGUMENT);
    EXPECT_STATUS(kl_softmax_device(4, 16, p, p + 63, KL_SOFTMAX_FUSED_ONLINE, nullptr, nullptr), KL_ERR_INVALID_ARGUMENT);
    EXPECT_STATUS(kl_softmax_device(4, 16, p, p + 64, KL_SOFTMAX_FUSED_ONLINE, nullptr, nullptr), KL_OK);
    EXPECT_STATUS(kl_softmax_device(4, 16, p, p + 64, KL_SOFTMAX_MULTIPASS, nullptr, nullptr), KL_ERR_INVALID_ARGUMENT);
    EXPECT_STATUS(kl_softmax_device(4, 16, nullptr, p, KL_SOFTMAX_AUTO, nullptr, nullptr), KL_ERR_INVALID_ARGUMENT);
    EXPECT_STATUS(kl_softmax_device(0, 16, nullptr, nullptr, KL_SOFTMAX_AUTO, nullptr, nullptr), KL_OK);
    // MULTIPASS workspace (2 floats per row = 8 floats here) must not overlap x or y.
    EXPECT_STATUS(kl_softmax_device(4, 16, p, p + 64, KL_SOFTMAX_MULTIPASS, p + 64 + 60, nullptr),
                  KL_ERR_INVALID_ARGUMENT);
    EXPECT_STATUS(kl_softmax_device(4, 16, p, p + 64, KL_SOFTMAX_MULTIPASS, p + 60, nullptr),
                  KL_ERR_INVALID_ARGUMENT);
    EXPECT_STATUS(kl_softmax_device(4, 16, p, p + 64, KL_SOFTMAX_MULTIPASS, p + 128, nullptr), KL_OK);
    // Misaligned pointers would cause a sticky CUDA error; they are refused.
    const auto misaligned = reinterpret_cast<float*>(reinterpret_cast<char*>(p + 64) + 2);
    EXPECT_STATUS(kl_softmax_device(4, 16, p, misaligned, KL_SOFTMAX_FUSED_ONLINE, nullptr, nullptr),
                  KL_ERR_INVALID_ARGUMENT);
    // Negative implementation ids are in the enum's range and rejected.
    EXPECT_STATUS(kl_softmax_device(4, 16, p, p + 64, static_cast<kl_softmax_impl>(-1), nullptr, nullptr),
                  KL_ERR_INVALID_ARGUMENT);
    EXPECT(cudaDeviceSynchronize() == cudaSuccess);
}

void test_softmax_single_column_is_one() {
    const std::vector<float> x = {-3.0f, 0.0f, 1e30f, -1e-30f, 7.5f};
    for (kl_softmax_impl impl : kImpls) {
        const std::vector<float> y = run_softmax(impl, 5, 1, x);
        for (float v : y) EXPECT(same_float(v, 1.0f));
    }
}

// Reductions use a fixed order, so repeated runs must be bit-identical.
void test_softmax_deterministic() {
    const std::int64_t rows = 64, cols = 3001;
    const std::vector<float> x = fill(600, rows * cols, -20.0f, 20.0f);
    for (kl_softmax_impl impl : {KL_SOFTMAX_MULTIPASS, KL_SOFTMAX_FUSED_ONLINE}) {
        const std::vector<float> a = run_softmax(impl, rows, cols, x);
        const std::vector<float> b = run_softmax(impl, rows, cols, x);
        EXPECT(std::equal(a.begin(), a.end(), b.begin(), [](float p, float q) { return same_float(p, q); }));
    }
    const std::int64_t wcols = 1000;  // warp kernel range
    const std::vector<float> xw = fill(601, rows * wcols, -20.0f, 20.0f);
    const std::vector<float> a = run_softmax(KL_SOFTMAX_FUSED_WARP, rows, wcols, xw);
    const std::vector<float> b = run_softmax(KL_SOFTMAX_FUSED_WARP, rows, wcols, xw);
    EXPECT(std::equal(a.begin(), a.end(), b.begin(), [](float p, float q) { return same_float(p, q); }));
}

void test_softmax_inplace_host() {
    const std::int64_t rows = 5, cols = 777;
    std::vector<float> x = fill(700, rows * cols, -3.0f, 3.0f);
    const std::vector<float> x_in = x;
    const std::vector<float> want = run_softmax(KL_SOFTMAX_FUSED_ONLINE, rows, cols, x_in);
    EXPECT_STATUS(kl_softmax_host(rows, cols, x.data(), x.data(), KL_SOFTMAX_FUSED_ONLINE), KL_OK);
    EXPECT(std::equal(x.begin(), x.end(), want.begin(), [](float p, float q) { return same_float(p, q); }));
}

void test_softmax_invalid_and_unsupported() {
    const float kSentinel = -12345.0f;
    std::vector<float> x(16, 1.0f), y(16, kSentinel);
    const std::uint64_t mallocs = kl::detail::device_malloc_count();
    auto untouched = [&] {
        for (float v : y) if (!same_float(v, kSentinel)) return false;
        return true;
    };
    const auto bad_impl = static_cast<kl_softmax_impl>(7);
    EXPECT_STATUS(kl_softmax_host(-1, 4, x.data(), y.data(), KL_SOFTMAX_AUTO), KL_ERR_INVALID_ARGUMENT);
    EXPECT_STATUS(kl_softmax_host(4, -1, x.data(), y.data(), KL_SOFTMAX_AUTO), KL_ERR_INVALID_ARGUMENT);
    EXPECT_STATUS(kl_softmax_host(4, 4, x.data(), y.data(), bad_impl), KL_ERR_INVALID_ARGUMENT);
    EXPECT_STATUS(kl_softmax_host(4, 4, nullptr, y.data(), KL_SOFTMAX_AUTO), KL_ERR_INVALID_ARGUMENT);
    EXPECT_STATUS(kl_softmax_host(4, 4, x.data(), nullptr, KL_SOFTMAX_AUTO), KL_ERR_INVALID_ARGUMENT);
    // Empty matrices are no-ops, even with null pointers.
    EXPECT_STATUS(kl_softmax_host(0, 4, nullptr, nullptr, KL_SOFTMAX_AUTO), KL_OK);
    EXPECT_STATUS(kl_softmax_host(4, 0, nullptr, nullptr, KL_SOFTMAX_AUTO), KL_OK);
    // Outside the supported domain (checked before any allocation or read).
    const std::int64_t too_many_cols = std::int64_t{KL_SOFTMAX_MAX_COLS} + 1;
    const std::int64_t too_many_rows = std::int64_t{KL_SOFTMAX_MAX_ROWS} + 1;
    EXPECT_STATUS(kl_softmax_host(1, too_many_cols, x.data(), y.data(), KL_SOFTMAX_AUTO), KL_ERR_UNSUPPORTED);
    EXPECT_STATUS(kl_softmax_host(too_many_rows, 1, x.data(), y.data(), KL_SOFTMAX_AUTO), KL_ERR_UNSUPPORTED);
    EXPECT_STATUS(kl_softmax_host(1 << 13, 1 << 14, x.data(), y.data(), KL_SOFTMAX_AUTO), KL_ERR_UNSUPPORTED);
    EXPECT_STATUS(kl_softmax_host(1, 1025, x.data(), y.data(), KL_SOFTMAX_FUSED_WARP), KL_ERR_UNSUPPORTED);
    EXPECT_STATUS(kl_softmax_host(std::numeric_limits<std::int64_t>::max(), 2, x.data(), y.data(),
                                  KL_SOFTMAX_AUTO), KL_ERR_UNSUPPORTED);
    EXPECT(untouched());
    EXPECT(kl::detail::device_malloc_count() == mallocs);
    EXPECT(kl_device_live_bytes() == 0);

    // kl_softmax_supported agrees with the host entry point.
    EXPECT_STATUS(kl_softmax_supported(1, 1024, KL_SOFTMAX_FUSED_WARP), KL_OK);
    EXPECT_STATUS(kl_softmax_supported(1, 1025, KL_SOFTMAX_FUSED_WARP), KL_ERR_UNSUPPORTED);
    EXPECT_STATUS(kl_softmax_supported(1, 1 << 20, KL_SOFTMAX_AUTO), KL_OK);
    EXPECT_STATUS(kl_softmax_supported(64, 1 << 20, KL_SOFTMAX_AUTO), KL_OK);   // exactly 2^26
    EXPECT_STATUS(kl_softmax_supported(65, 1 << 20, KL_SOFTMAX_AUTO), KL_ERR_UNSUPPORTED);
    EXPECT_STATUS(kl_softmax_supported(0, 0, bad_impl), KL_ERR_INVALID_ARGUMENT);

    // The launcher refuses MULTIPASS without a workspace.
    kl::detail::DeviceBuffer d;
    EXPECT_STATUS(kl::detail::DeviceBuffer::allocate(64 * sizeof(float), d), KL_OK);
    EXPECT_STATUS(kl::cuda::launch_softmax(4, 4, d.as<float>(), d.as<float>() + 16,
                                           KL_SOFTMAX_MULTIPASS, nullptr, 0),
                  KL_ERR_INVALID_ARGUMENT);
}

void test_softmax_auto_dispatch() {
    using kl::cuda::softmax_resolve;
    EXPECT(softmax_resolve(1, 1024, KL_SOFTMAX_AUTO) == KL_SOFTMAX_FUSED_WARP);
    EXPECT(softmax_resolve(4096, 1024, KL_SOFTMAX_AUTO) == KL_SOFTMAX_FUSED_WARP);
    EXPECT(softmax_resolve(64, 1025, KL_SOFTMAX_AUTO) == KL_SOFTMAX_FUSED_ONLINE);
    EXPECT(softmax_resolve(63, 1025, KL_SOFTMAX_AUTO) == KL_SOFTMAX_MULTIPASS);
    EXPECT(softmax_resolve(4, 1 << 20, KL_SOFTMAX_AUTO) == KL_SOFTMAX_MULTIPASS);
    EXPECT(softmax_resolve(10, 10, KL_SOFTMAX_MULTIPASS) == KL_SOFTMAX_MULTIPASS);
    EXPECT(softmax_resolve(1, 10, KL_SOFTMAX_FUSED_ONLINE) == KL_SOFTMAX_FUSED_ONLINE);
    EXPECT(kl::cuda::softmax_workspace_floats(10, KL_SOFTMAX_MULTIPASS) == 20);
    EXPECT(kl::cuda::softmax_workspace_floats(10, KL_SOFTMAX_FUSED_ONLINE) == 0);
}

}  // namespace

int main(int argc, char** argv) {
    static const kl_test::TestCase tests[] = {
        {"softmax_uniform", test_softmax_uniform},
        {"softmax_large_offset", test_softmax_large_offset},
        {"softmax_wide_range", test_softmax_wide_range},
        {"softmax_grid_inputs_tight_bound", test_softmax_grid_inputs_tight_bound},
        {"softmax_exp_accuracy", test_softmax_exp_accuracy},
        {"softmax_constant_rows", test_softmax_constant_rows},
        {"softmax_one_hot", test_softmax_one_hot},
        {"softmax_special_values", test_softmax_special_values},
        {"softmax_max_spike_positions", test_softmax_max_spike_positions},
        {"softmax_extreme_finite_values", test_softmax_extreme_finite_values},
        {"softmax_guard_bands", test_softmax_guard_bands},
        {"softmax_device_api_validation", test_softmax_device_api_validation},
        {"softmax_single_column_is_one", test_softmax_single_column_is_one},
        {"softmax_deterministic", test_softmax_deterministic},
        {"softmax_inplace_host", test_softmax_inplace_host},
        {"softmax_invalid_and_unsupported", test_softmax_invalid_and_unsupported},
        {"softmax_auto_dispatch", test_softmax_auto_dispatch},
    };
    return kl_test::run_tests(argc, argv, tests);
}
