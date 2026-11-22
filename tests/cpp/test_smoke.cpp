// C++ tests: toolchain smoke test, reference comparison, error
// paths and the VRAM safety cap. Deliberately dependency-free (no test
// framework) so running it needs only the CUDA toolchain to run it.
//
// All workloads are small (largest: two 64 MiB buffers) and each kernel runs
// for well under a millisecond, far below the Windows 2 s watchdog.
//
// Usage: test_smoke [--exclude TEST]... [--max-n N]
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/device_buffer.hpp"
#include "cuda/kernels.hpp"
#include "harness.hpp"
#include "kernellens/kernellens.h"

namespace {

using kl_test::expect_last_error_mentions;
using kl_test::fill;
using kl_test::same_float;
using kl_test::skip_for_max_n;
using kl_test::ulp_distance;

// Reference 1 (primary): the C/C++ standard requires fmaf to return the
// exact a*x+y rounded once, which is exactly what the kernel's fmaf must
// produce. test_reference_self_check confirms the host implementation does
// this on a case where a naive implementation would not.
float saxpy_exact(float a, float x, float y) { return std::fmaf(a, x, y); }

// Reference 2 (independent bound): exact product in double (24x24-bit fits
// in 53 bits), one double rounding for the add, one rounding to float. This
// is within 1 ulp of the correctly rounded result (double rounding).
float saxpy_double(float a, float x, float y) {
    return static_cast<float>(static_cast<double>(a) * static_cast<double>(x) +
                              static_cast<double>(y));
}

struct CompareResult {
    std::int64_t mismatches_exact = 0;  // vs std::fmaf, bitwise
    std::int64_t max_ulp_double = 0;    // vs double reference
};

CompareResult compare(float a, const std::vector<float>& x, const std::vector<float>& y_in,
                      const std::vector<float>& got) {
    CompareResult r;
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (!same_float(got[i], saxpy_exact(a, x[i], y_in[i]))) ++r.mismatches_exact;
        const std::int64_t d = ulp_distance(got[i], saxpy_double(a, x[i], y_in[i]));
        if (d > r.max_ulp_double) r.max_ulp_double = d;
    }
    return r;
}

// Runs kl_saxpy_host on copies and checks both references.
void check_saxpy(float a, const std::vector<float>& x, const std::vector<float>& y_in,
                 const char* label) {
    std::vector<float> y = y_in;
    EXPECT_STATUS(kl_saxpy_host(static_cast<std::int64_t>(x.size()), a, x.data(), y.data()), KL_OK);
    const CompareResult r = compare(a, x, y_in, y);
    if (r.mismatches_exact != 0 || r.max_ulp_double > 1) {
        std::printf("  %s a=%g n=%zu: %lld differ from fmaf, max %lld ulp from double ref\n", label, a,
                    x.size(), static_cast<long long>(r.mismatches_exact),
                    static_cast<long long>(r.max_ulp_double));
    }
    EXPECT(r.mismatches_exact == 0);
    EXPECT(r.max_ulp_double <= 1);
}

// ---------------------------------------------------------------------------

void test_device_info() {
    kl_device_info info{};
    EXPECT_STATUS(kl_get_device_info(&info), KL_OK);
    std::printf("  device: %s, cc %d.%d, %d SMs x %d threads, watchdog=%d, driver %d, runtime %d, "
                "%.0f MiB free / %.0f MiB total\n",
                info.name, info.compute_major, info.compute_minor, info.multiprocessor_count,
                info.max_threads_per_multiprocessor, info.kernel_exec_timeout,
                info.driver_version, info.runtime_version, info.free_global_mem / 1048576.0,
                info.total_global_mem / 1048576.0);
    // The binary carries sm_86 SASS and compute_86 PTX; older GPUs cannot run it.
    EXPECT(info.compute_major * 10 + info.compute_minor >= 86);
    EXPECT(info.multiprocessor_count > 0);
    EXPECT(info.total_global_mem > 0);
    EXPECT_STATUS(kl_get_device_info(nullptr), KL_ERR_INVALID_ARGUMENT);
}

// a*x = 1 + 2^-24 exactly (24929 * 673 = 2^24 + 1), the midpoint between the
// floats 1 and 1 + 2^-23. With y = +2^-60 the exact sum is just above the
// midpoint, so correct rounding gives 1 + 2^-23; the double reference loses
// 2^-60 (below half a double ulp), lands exactly on the midpoint and
// rounds-to-even to 1.0. This is the double-rounding case the 1-ulp bound
// exists for.
const float kHardA = 24929.0f / 32768.0f;  // 24929 * 2^-15, exact
const float kHardX = 673.0f / 512.0f;      // 673 * 2^-9, exact
const float kHardY = std::ldexp(1.0f, -60);

void test_reference_self_check() {
    const float up = 1.0f + std::ldexp(1.0f, -23);
    EXPECT(static_cast<double>(kHardA) * kHardX == 1.0 + std::ldexp(1.0, -24));
    EXPECT(same_float(saxpy_exact(kHardA, kHardX, kHardY), up));       // host fmaf is correct
    EXPECT(same_float(saxpy_double(kHardA, kHardX, kHardY), 1.0f));    // double ref is 1 ulp off
    EXPECT(same_float(saxpy_exact(kHardA, kHardX, -kHardY), 1.0f));    // below midpoint
}

void test_saxpy_double_rounding_case() {
    std::vector<float> x = {kHardX, kHardX, -kHardX};
    std::vector<float> y = {kHardY, -kHardY, -kHardY};
    std::vector<float> y_in = y;
    EXPECT_STATUS(kl_saxpy_host(3, kHardA, x.data(), y.data()), KL_OK);
    EXPECT(same_float(y[0], 1.0f + std::ldexp(1.0f, -23)));  // GPU rounds correctly
    EXPECT(same_float(y[1], 1.0f));
    EXPECT(same_float(y[2], -1.0f - std::ldexp(1.0f, -23)));
    EXPECT(ulp_distance(y[0], saxpy_double(kHardA, x[0], y_in[0])) == 1);  // bound is tight
    check_saxpy(kHardA, x, y_in, "double-rounding");
}

void test_saxpy_random_sizes() {
    // Sizes straddle warp (32) and block (256) boundaries, and the last one
    // exceeds the 65535-block grid cap (65535*256 = 16,776,960) so the
    // grid-stride loop must take a second iteration.
    const std::int64_t sizes[] = {1, 2, 31, 32, 33, 255, 256, 257, 1000, 4194304, 16776961};
    const float alphas[] = {0.0f, 1.0f, -2.5f, 1.0e-3f};
    std::uint64_t seed = 1;
    for (std::int64_t n : sizes) {
        if (skip_for_max_n(n, "saxpy size")) continue;
        for (float a : alphas) {
            // Different magnitudes for x and y make many sums inexact in
            // float, so a kernel with the wrong rounding mode would differ
            // from fmaf on a large fraction of elements.
            const std::vector<float> x = fill(seed++, n, -100.0f, 100.0f);
            const std::vector<float> y = fill(seed++, n, -1.0e-3f, 1.0e-3f);
            check_saxpy(a, x, y, "random");
        }
    }
}

void test_saxpy_special_values() {
    const float denorm = std::numeric_limits<float>::denorm_min();
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float specials[] = {0.0f,    -0.0f, 1.0f, -1.0f,  inf,     -inf,
                              nan,     FLT_MAX, -FLT_MAX, FLT_MIN, denorm, -denorm};
    const float alphas[] = {0.0f, -0.0f, 1.0f, -2.5f, 3.0e38f, 1.0e-30f};
    for (float a : alphas) {
        std::vector<float> x, y;
        for (float xi : specials) {
            for (float yi : specials) {
                x.push_back(xi);
                y.push_back(yi);
            }
        }
        check_saxpy(a, x, y, "special");
    }
}

void test_saxpy_aliased_host_pointers() {
    std::vector<float> y = fill(99, 1000, -1.0f, 1.0f);
    const std::vector<float> y_in = y;
    EXPECT_STATUS(kl_saxpy_host(1000, 3.0f, y.data(), y.data()), KL_OK);
    EXPECT(compare(3.0f, y_in, y_in, y).mismatches_exact == 0);
}

void test_invalid_arguments() {
    const float kSentinel = -12345.0f;
    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float y[4] = {kSentinel, kSentinel, kSentinel, kSentinel};
    auto y_untouched = [&] {
        for (float v : y) {
            if (!same_float(v, kSentinel)) return false;
        }
        return true;
    };
    EXPECT_STATUS(kl_saxpy_host(-1, 1.0f, x, y), KL_ERR_INVALID_ARGUMENT);
    EXPECT_STATUS(kl_saxpy_host(4, 1.0f, nullptr, y), KL_ERR_INVALID_ARGUMENT);
    EXPECT_STATUS(kl_saxpy_host(4, 1.0f, x, nullptr), KL_ERR_INVALID_ARGUMENT);
    EXPECT_STATUS(kl_saxpy_host(0, 1.0f, nullptr, nullptr), KL_OK);  // n == 0 is a no-op
    EXPECT_STATUS(kl_saxpy_host(0, 1.0f, x, y), KL_OK);
    // Byte count would overflow size_t.
    EXPECT_STATUS(kl_saxpy_host(std::numeric_limits<std::int64_t>::max(), 1.0f, x, y),
                  KL_ERR_RESOURCE_LIMIT);
    EXPECT(y_untouched());

    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float bad_bounds[][2] = {{1.0f, 1.0f}, {2.0f, 1.0f}, {-inf, 0.0f}, {0.0f, inf},
                                   {nan, 1.0f},  {0.0f, nan}};
    for (const auto& b : bad_bounds) {
        EXPECT_STATUS(kl_fill_uniform(1, 4, b[0], b[1], y), KL_ERR_INVALID_ARGUMENT);
    }
    EXPECT_STATUS(kl_fill_uniform(1, -1, 0.0f, 1.0f, y), KL_ERR_INVALID_ARGUMENT);
    EXPECT_STATUS(kl_fill_uniform(1, 2, 0.0f, 1.0f, nullptr), KL_ERR_INVALID_ARGUMENT);
    EXPECT(y_untouched());
    // Extreme but finite bounds are valid and stay finite.
    EXPECT_STATUS(kl_fill_uniform(1, 4, -FLT_MAX, FLT_MAX, y), KL_OK);
    for (float v : y) EXPECT(std::isfinite(v));
}

void test_budget_policy() {
    using kl::detail::budget_check;
    const std::size_t cap = kl::detail::kDeviceAllocHardCap;
    const std::size_t huge_free = std::size_t{1} << 40;
    EXPECT_STATUS(budget_check(cap, 0, huge_free), KL_OK);
    EXPECT_STATUS(budget_check(cap + 1, 0, huge_free), KL_ERR_RESOURCE_LIMIT);
    EXPECT_STATUS(budget_check(1, cap, huge_free), KL_ERR_RESOURCE_LIMIT);
    EXPECT_STATUS(budget_check(cap / 2, cap / 2, huge_free), KL_OK);
    EXPECT_STATUS(budget_check(std::numeric_limits<std::size_t>::max(), 0, huge_free),
                  KL_ERR_RESOURCE_LIMIT);
    // Half-of-free-memory rule.
    EXPECT_STATUS(budget_check(500, 0, 1000), KL_OK);
    EXPECT_STATUS(budget_check(501, 0, 1000), KL_ERR_RESOURCE_LIMIT);
    EXPECT(std::string(kl_last_error_message()).find("half of free") != std::string::npos);
    EXPECT_STATUS(budget_check(1, 0, 1), KL_ERR_RESOURCE_LIMIT);
    // The two halves of the policy, which DeviceBuffer::allocate calls
    // directly (cap_check inside its reservation loop, free_check after it).
    EXPECT_STATUS(kl::detail::cap_check(cap, 0), KL_OK);
    EXPECT_STATUS(kl::detail::cap_check(1, cap), KL_ERR_RESOURCE_LIMIT);
    EXPECT_STATUS(kl::detail::cap_check(cap + 1, 0), KL_ERR_RESOURCE_LIMIT);
    EXPECT_STATUS(kl::detail::free_check(500, 1000), KL_OK);
    EXPECT_STATUS(kl::detail::free_check(501, 1000), KL_ERR_RESOURCE_LIMIT);
    EXPECT_STATUS(kl::detail::free_check(0, 0), KL_OK);
}

void test_vram_cap() {
    using kl::detail::DeviceBuffer;
    const std::size_t cap = kl_device_alloc_hard_cap();
    EXPECT(cap == (std::size_t{2} << 30));
    EXPECT(kl_device_live_bytes() == 0);

    // Over-cap requests through the public API are refused before any
    // cudaMalloc and before reading the (tiny) host arrays:
    //  - n_one: each buffer alone exceeds the cap;
    //  - n_two: each buffer fits, but the two together do not.
    float dummy[4] = {0};
    const std::int64_t n_one = static_cast<std::int64_t>(cap / sizeof(float)) + 1;
    const std::int64_t n_two = static_cast<std::int64_t>(cap / (2 * sizeof(float))) + 1;
    for (std::int64_t n : {n_one, n_two}) {
        const std::uint64_t mallocs = kl::detail::device_malloc_count();
        EXPECT_STATUS(kl_saxpy_host(n, 1.0f, dummy, dummy), KL_ERR_RESOURCE_LIMIT);
        EXPECT(kl::detail::device_malloc_count() == mallocs);
        EXPECT(kl_device_live_bytes() == 0);
    }

    // Cumulative cap: a small live buffer plus a request that would cross the
    // cap is refused without calling cudaMalloc for the large one.
    {
        DeviceBuffer small;
        EXPECT_STATUS(DeviceBuffer::allocate(std::size_t{1} << 20, small), KL_OK);
        EXPECT(kl_device_live_bytes() == (std::size_t{1} << 20));
        const std::uint64_t mallocs = kl::detail::device_malloc_count();
        DeviceBuffer big;
        EXPECT_STATUS(DeviceBuffer::allocate(cap - (std::size_t{1} << 20) + 1, big),
                      KL_ERR_RESOURCE_LIMIT);
        EXPECT(big.get() == nullptr);
        EXPECT(kl::detail::device_malloc_count() == mallocs);
        EXPECT(kl_device_live_bytes() == (std::size_t{1} << 20));
        // An absurd request must not disturb the counter either.
        EXPECT_STATUS(DeviceBuffer::allocate(std::numeric_limits<std::size_t>::max(), big),
                      KL_ERR_RESOURCE_LIMIT);
        EXPECT(kl_device_live_bytes() == (std::size_t{1} << 20));

        DeviceBuffer moved = std::move(small);
        EXPECT(small.get() == nullptr);
        EXPECT(moved.get() != nullptr);
        EXPECT(kl_device_live_bytes() == (std::size_t{1} << 20));
    }
    EXPECT(kl_device_live_bytes() == 0);  // RAII released everything
}

kl_status provoke_runtime_api_error() {
    KL_CUDA_TRY(cudaSetDevice(-1));  // always invalid
    return KL_OK;
}

void test_error_paths() {
    using kl::detail::DeviceBuffer;
    // 1. A failing runtime API call is translated, with a useful message.
    EXPECT_STATUS(provoke_runtime_api_error(), KL_ERR_CUDA);
    expect_last_error_mentions("cudaErrorInvalidDevice", KL_HERE, __LINE__);

    // 2. The stale error from (1) must not be blamed on the next launch
    //    (regression test for the cudaGetLastError reset in launch_saxpy).
    DeviceBuffer dx, dy;
    EXPECT_STATUS(DeviceBuffer::allocate(1024 * sizeof(float), dx), KL_OK);
    EXPECT_STATUS(DeviceBuffer::allocate(1024 * sizeof(float), dy), KL_OK);
    EXPECT(cudaMemset(dx.get(), 0, 1024 * sizeof(float)) == cudaSuccess);
    EXPECT(cudaMemset(dy.get(), 0, 1024 * sizeof(float)) == cudaSuccess);
    EXPECT_STATUS(provoke_runtime_api_error(), KL_ERR_CUDA);
    // Precondition of the regression test: the error really is pending.
    EXPECT(cudaPeekAtLastError() == cudaErrorInvalidDevice);
    EXPECT_STATUS(kl::cuda::launch_saxpy(1024, 1.0f, dx.as<float>(), dy.as<float>(),
                                         kl::cuda::kSaxpyBlockSize, 0),
                  KL_OK);
    EXPECT(cudaStreamSynchronize(0) == cudaSuccess);

    // 3. An invalid launch configuration (2048 > 1024 threads per block) is
    //    caught by KL_CHECK_LAUNCH; nothing executes. The test checks that
    //    the failure is attributed to the post-launch check, not which error
    //    code the runtime picks: CUDA 13.3 reports cudaErrorInvalidValue here
    //    (observed with CUDA 13.3 on the development machine), where older
    //    material often shows cudaErrorInvalidConfiguration.
    EXPECT_STATUS(kl::cuda::launch_saxpy(1024, 1.0f, dx.as<float>(), dy.as<float>(), 2048, 0),
                  KL_ERR_CUDA);
    expect_last_error_mentions("saxpy.cu", KL_HERE, __LINE__);
    expect_last_error_mentions("cudaGetLastError()", KL_HERE, __LINE__);
    std::printf("  invalid launch reported as: %s\n", kl_last_error_message());

    // 4. That error is not sticky: the context still works afterwards.
    EXPECT_STATUS(kl::cuda::launch_saxpy(1024, 1.0f, dx.as<float>(), dy.as<float>(),
                                         kl::cuda::kSaxpyBlockSize, 0),
                  KL_OK);
    EXPECT(cudaStreamSynchronize(0) == cudaSuccess);
}

void test_fill_uniform_determinism() {
    std::vector<float> a = fill(42, 4096, -3.0f, 5.0f);
    std::vector<float> b = fill(42, 4096, -3.0f, 5.0f);
    std::vector<float> c = fill(43, 4096, -3.0f, 5.0f);
    EXPECT(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
    EXPECT(std::memcmp(a.data(), c.data(), a.size() * sizeof(float)) != 0);
    bool in_range = true;
    for (float v : a) in_range = in_range && v >= -3.0f && v <= 5.0f;
    EXPECT(in_range);
}

}  // namespace

// --exclude exists for sanitizer runs: error_paths provokes CUDA API errors on
// purpose, which a sanitizer would (correctly) report. --max-n bounds kernel
// sizes for slow tool runs (sanitizers slow kernels 10-100x).
int main(int argc, char** argv) {
    static const kl_test::TestCase tests[] = {
        {"device_info", test_device_info},
        {"reference_self_check", test_reference_self_check},
        {"saxpy_double_rounding_case", test_saxpy_double_rounding_case},
        {"saxpy_random_sizes", test_saxpy_random_sizes},
        {"saxpy_special_values", test_saxpy_special_values},
        {"saxpy_aliased_host_pointers", test_saxpy_aliased_host_pointers},
        {"invalid_arguments", test_invalid_arguments},
        {"budget_policy", test_budget_policy},
        {"vram_cap", test_vram_cap},
        {"error_paths", test_error_paths},
        {"fill_uniform_determinism", test_fill_uniform_determinism},
    };
    return kl_test::run_tests(argc, argv, tests);
}
