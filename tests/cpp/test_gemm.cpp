// C++ tests for FP32 matrix multiplication (docs/gemm.md).
//
// Reference: float64 on the CPU. Tolerance: the derived bound
//   |C - C_ref| <= (gamma_K + 2^-50) S + K 2^-150,  S = sum |a| |b|
// (tests/cpp/gemm_reference.hpp); integer inputs must match bit for bit.
// cuBLAS (CUBLAS_DEFAULT_MATH: no TF32) is a second, independent reference.
// Workloads stay small (<= 2^26 elements, sub-second launches).
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include "common/device_buffer.hpp"
#include "cuda/kernels.hpp"
#include "gemm_reference.hpp"
#include "harness.hpp"
#include "kernellens/kernellens.h"

namespace {

using kl_test::fill;
using kl_test::same_float;
using kl_test::skip_for_max_n;
using kl_test::gemm::check_gemm;
using kl_test::gemm::CheckResult;

const float kInf = std::numeric_limits<float>::infinity();
const float kNaN = std::numeric_limits<float>::quiet_NaN();
const kl_gemm_impl kImpls[] = {KL_GEMM_AUTO, KL_GEMM_NAIVE, KL_GEMM_TILED, KL_GEMM_REGBLOCK};

const char* impl_name(kl_gemm_impl impl) {
    switch (impl) {
        case KL_GEMM_AUTO: return "auto";
        case KL_GEMM_NAIVE: return "naive";
        case KL_GEMM_TILED: return "tiled";
        case KL_GEMM_REGBLOCK: return "regblock";
        default: return "?";
    }
}

struct Dims {
    std::int64_t m, n, k;
};

// Work measure for --max-n (sanitizer runs cap kernel sizes: tools slow
// kernels 10-100x and the watchdog is 2 s): M N K / 64.
std::int64_t work(const Dims& d) { return d.m * d.n * d.k / 64; }

std::vector<float> run_gemm(kl_gemm_impl impl, const Dims& d, const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<float> c(static_cast<std::size_t>(d.m * d.n), -12345.0f);
    EXPECT_STATUS(kl_gemm_host(d.m, d.n, d.k, a.data(), b.data(), c.data(), impl), KL_OK);
    return c;
}

// cuBLAS, row-major via C^T = B^T A^T (column-major n x m result = row-major C).
std::vector<float> run_cublas(const Dims& d, const std::vector<float>& a, const std::vector<float>& b) {
    using kl::detail::DeviceBuffer;
    std::vector<float> c(static_cast<std::size_t>(d.m * d.n), -12345.0f);
    DeviceBuffer da, db, dc;
    const std::size_t na = static_cast<std::size_t>(d.m * d.k) * 4, nb = static_cast<std::size_t>(d.k * d.n) * 4,
                      nc = static_cast<std::size_t>(d.m * d.n) * 4;
    EXPECT_STATUS(DeviceBuffer::allocate(na, da), KL_OK);
    EXPECT_STATUS(DeviceBuffer::allocate(nb, db), KL_OK);
    EXPECT_STATUS(DeviceBuffer::allocate(nc, dc), KL_OK);
    EXPECT(cudaMemcpy(da.get(), a.data(), na, cudaMemcpyHostToDevice) == cudaSuccess);
    EXPECT(cudaMemcpy(db.get(), b.data(), nb, cudaMemcpyHostToDevice) == cudaSuccess);
    cublasHandle_t h = nullptr;
    EXPECT(cublasCreate(&h) == CUBLAS_STATUS_SUCCESS);
    EXPECT(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH) == CUBLAS_STATUS_SUCCESS);
    const float one = 1.0f, zero = 0.0f;
    EXPECT(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(d.n), static_cast<int>(d.m),
                       static_cast<int>(d.k), &one, db.as<float>(), static_cast<int>(d.n), da.as<float>(),
                       static_cast<int>(d.k), &zero, dc.as<float>(), static_cast<int>(d.n)) == CUBLAS_STATUS_SUCCESS);
    EXPECT(cudaDeviceSynchronize() == cudaSuccess);
    EXPECT(cublasDestroy(h) == CUBLAS_STATUS_SUCCESS);
    EXPECT(cudaMemcpy(c.data(), dc.get(), nc, cudaMemcpyDeviceToHost) == cudaSuccess);
    return c;
}

void expect_ok(const char* label, const Dims& d, const CheckResult& r, bool exact) {
    if (r.violations || r.special_mismatches || r.out_of_domain || (exact && r.inexact)) {
        std::printf("  %s %lldx%lldx%lld: %lld violations, %lld special mismatches, %lld out of domain, %lld inexact, "
                    "worst %.3g x bound\n",
                    label, static_cast<long long>(d.m), static_cast<long long>(d.n), static_cast<long long>(d.k),
                    static_cast<long long>(r.violations), static_cast<long long>(r.special_mismatches),
                    static_cast<long long>(r.out_of_domain), static_cast<long long>(r.inexact), r.worst_ratio);
    }
    EXPECT(r.violations == 0);
    EXPECT(r.special_mismatches == 0);
    EXPECT(r.out_of_domain == 0);
    if (exact) EXPECT(r.inexact == 0);
}

// Each dimension through the tile boundaries of every kernel (16, 32, 64),
// the others fixed; plus a diagonal.
std::vector<Dims> boundary_shapes() {
    const std::int64_t sizes[] = {1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129};
    std::vector<Dims> v;
    for (std::int64_t s : sizes) {
        v.push_back({s, 33, 64});
        v.push_back({33, s, 64});
        v.push_back({33, 64, s});
        v.push_back({s, s, s});
    }
    return v;
}

void run_all(const char* label, const Dims& d, const std::vector<float>& a, const std::vector<float>& b, bool exact,
             double* worst) {
    for (kl_gemm_impl impl : kImpls) {
        const std::vector<float> c = run_gemm(impl, d, a, b);
        const CheckResult r = check_gemm(d.m, d.n, d.k, a.data(), b.data(), c.data(), exact);
        expect_ok(impl_name(impl), d, r, exact);
        if (worst) *worst = std::max(*worst, r.worst_ratio);
    }
    (void)label;
}

void test_gemm_boundary_shapes_uniform() {
    double worst = 0;
    std::uint64_t seed = 100;
    for (const Dims& d : boundary_shapes()) {
        const std::vector<float> a = fill(seed++, d.m * d.k, -1.0f, 1.0f);
        const std::vector<float> b = fill(seed++, d.k * d.n, -1.0f, 1.0f);
        run_all("uniform", d, a, b, false, &worst);
    }
    std::printf("  U[-1,1], %zu shapes: worst error %.3f of the derived bound\n", boundary_shapes().size(), worst);
}

void test_gemm_wide_magnitudes() {
    double worst = 0;
    for (const Dims& d : {Dims{65, 129, 1000}, Dims{200, 70, 4097}}) {
        if (skip_for_max_n(work(d), "wide magnitudes")) continue;
        std::vector<float> a = fill(200 + d.k, d.m * d.k, -1.0f, 1.0f);
        std::vector<float> b = fill(300 + d.k, d.k * d.n, -1.0f, 1.0f);
        const std::vector<float> e = fill(400 + d.k, d.m * d.k, -30.0f, 30.0f);
        for (std::size_t i = 0; i < a.size(); ++i) a[i] = std::ldexp(a[i], static_cast<int>(e[i]));
        run_all("wide", d, a, b, false, &worst);
    }
    std::printf("  magnitudes 2^-30..2^30: worst error %.3f of the derived bound\n", worst);
}

// Integer inputs in [-8, 8]: every product and partial sum is an integer
// below 2^24, so every FP32 operation is exact - all implementations and
// cuBLAS must match the float64 reference bit for bit.
void test_gemm_integer_exact() {
    std::uint64_t seed = 500;
    std::vector<Dims> shapes = boundary_shapes();
    shapes.push_back({100, 100, 4096});
    shapes.push_back({257, 129, 16384});  // K = 2^14: max partial sum 64 * 2^14 = 2^20
    for (const Dims& d : shapes) {
        if (skip_for_max_n(work(d), "integer exact")) continue;
        std::vector<float> a = fill(seed++, d.m * d.k, -8.0f, 8.0f);
        std::vector<float> b = fill(seed++, d.k * d.n, -8.0f, 8.0f);
        for (float& v : a) v = std::round(v);
        for (float& v : b) v = std::round(v);
        run_all("integer", d, a, b, true, nullptr);
        const std::vector<float> c = run_cublas(d, a, b);
        expect_ok("cublas", d, check_gemm(d.m, d.n, d.k, a.data(), b.data(), c.data(), true), true);
    }
}

// Same operation sequence in every kernel -> bitwise identical results,
// including shapes too large for the CPU reference.
void test_gemm_bitwise_identical() {
    std::uint64_t seed = 700;
    for (const Dims& d : {Dims{1, 1, 1}, Dims{77, 1000, 333}, Dims{1024, 1024, 1024}, Dims{16, 16, 16384},
                          Dims{16384, 16, 16}, Dims{4096, 1000, 64}}) {
        if (skip_for_max_n(work(d), "bitwise")) continue;
        const std::vector<float> a = fill(seed++, d.m * d.k, -2.0f, 2.0f);
        const std::vector<float> b = fill(seed++, d.k * d.n, -2.0f, 2.0f);
        const std::vector<float> want = run_gemm(KL_GEMM_NAIVE, d, a, b);
        for (kl_gemm_impl impl : {KL_GEMM_TILED, KL_GEMM_REGBLOCK, KL_GEMM_AUTO}) {
            const std::vector<float> c = run_gemm(impl, d, a, b);
            const bool same =
                std::equal(c.begin(), c.end(), want.begin(), [](float p, float q) { return same_float(p, q); });
            if (!same) std::printf("  %s != naive at %lldx%lldx%lld\n", impl_name(impl), static_cast<long long>(d.m),
                                   static_cast<long long>(d.n), static_cast<long long>(d.k));
            EXPECT(same);
        }
        // The benchmark's L1-bypass control computes the same sequence.
        using kl::detail::DeviceBuffer;
        DeviceBuffer da, db, dc;
        EXPECT_STATUS(DeviceBuffer::allocate(a.size() * 4, da), KL_OK);
        EXPECT_STATUS(DeviceBuffer::allocate(b.size() * 4, db), KL_OK);
        EXPECT_STATUS(DeviceBuffer::allocate(want.size() * 4, dc), KL_OK);
        EXPECT(cudaMemcpy(da.get(), a.data(), a.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess);
        EXPECT(cudaMemcpy(db.get(), b.data(), b.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess);
        EXPECT_STATUS(kl::cuda::launch_gemm_naive_l2only(d.m, d.n, d.k, da.as<float>(), db.as<float>(), dc.as<float>(), 0),
                      KL_OK);
        std::vector<float> c(want.size());
        EXPECT(cudaMemcpy(c.data(), dc.get(), c.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
        EXPECT(std::equal(c.begin(), c.end(), want.begin(), [](float p, float q) { return same_float(p, q); }));
    }
}

// cuBLAS (unknown summation order) satisfies the same order-independent
// bound - a check of the bound's derivation as much as of cuBLAS.
void test_gemm_cublas_within_bound() {
    double worst = 0;
    std::uint64_t seed = 800;
    for (const Dims& d : {Dims{64, 64, 64}, Dims{1000, 1023, 257}, Dims{512, 512, 4096}}) {
        if (skip_for_max_n(work(d), "cublas bound")) continue;
        const std::vector<float> a = fill(seed++, d.m * d.k, -1.0f, 1.0f);
        const std::vector<float> b = fill(seed++, d.k * d.n, -1.0f, 1.0f);
        const std::vector<float> c = run_cublas(d, a, b);
        const CheckResult r = check_gemm(d.m, d.n, d.k, a.data(), b.data(), c.data());
        expect_ok("cublas", d, r, false);
        worst = std::max(worst, r.worst_ratio);
    }
    std::printf("  cuBLAS: worst error %.3f of the derived bound\n", worst);
}

void test_gemm_k_zero_and_one() {
    const std::vector<float> none;
    for (kl_gemm_impl impl : kImpls) {
        std::vector<float> c(static_cast<std::size_t>(7 * 9), -12345.0f);
        EXPECT_STATUS(kl_gemm_host(7, 9, 0, nullptr, nullptr, c.data(), impl), KL_OK);
        EXPECT(std::all_of(c.begin(), c.end(), [](float v) { return same_float(v, 0.0f); }));  // +0
        const std::vector<float> a = fill(900, 7, -3.0f, 3.0f), b = fill(901, 9, -3.0f, 3.0f);
        const std::vector<float> c1 = run_gemm(impl, {7, 9, 1}, a, b);
        bool ok = true;
        for (int i = 0; i < 7; ++i) {
            for (int j = 0; j < 9; ++j) ok = ok && c1[static_cast<std::size_t>(i * 9 + j)] == a[i] * b[j];  // one rounding
        }
        EXPECT(ok);
    }
    (void)none;
}

// NaN / inf propagate as in IEEE arithmetic of the sum (compared with the
// float64 reference's NaN / inf pattern); inf * 0 gives NaN.
void test_gemm_special_values() {
    for (const Dims& d : {Dims{5, 6, 7}, Dims{33, 65, 70}}) {
        std::vector<float> a = fill(1000 + d.k, d.m * d.k, -2.0f, 2.0f);
        std::vector<float> b = fill(1100 + d.k, d.k * d.n, -2.0f, 2.0f);
        a[0 * d.k + 1] = kNaN;            // row 0 of C: NaN
        a[1 * d.k + 2] = kInf;            // row 1: +-inf unless times 0
        b[2 * d.n + 3] = 0.0f;            // -> C[1][3] = inf * 0 = NaN
        b[4 * d.n + 4] = -kInf;           // column 4: -+inf; with row 1's +inf in another term: inf - inf?
        a[3 * d.k + 4] = 0.0f;            // C[3][4]: 0 * -inf = NaN
        for (kl_gemm_impl impl : kImpls) {
            const std::vector<float> c = run_gemm(impl, d, a, b);
            const CheckResult r = check_gemm(d.m, d.n, d.k, a.data(), b.data(), c.data());
            expect_ok(impl_name(impl), d, r, false);
        }
    }
}

// Products near 2^-150: the K 2^-150 term of the bound, subnormal partial
// sums, and sums that round to -0 (an underflowing fmaf of a negative
// product gives -0). Every kernel must give the same bits,
// signed zeros included, and stay within the bound.
void test_gemm_tiny_products_and_signed_zeros() {
    std::uint64_t seed = 1500;
    for (const Dims& d : {Dims{1, 1, 1}, Dims{33, 65, 70}, Dims{64, 64, 64}, Dims{17, 3, 200}}) {
        std::vector<float> a = fill(seed++, d.m * d.k, -1.0f, 1.0f);
        std::vector<float> b = fill(seed++, d.k * d.n, -1.0f, 1.0f);
        for (float& v : a) v = std::ldexp(v, -74);
        for (float& v : b) v = std::ldexp(v, -74);
        if (d.k == 1) {
            a[0] = -std::ldexp(1.0f, -80);  // product -2^-160: fmaf(-2^-80, 2^-80, +0) = -0
            b[0] = std::ldexp(1.0f, -80);
        }
        const std::vector<float> want = run_gemm(KL_GEMM_NAIVE, d, a, b);
        expect_ok("tiny naive", d, check_gemm(d.m, d.n, d.k, a.data(), b.data(), want.data()), false);
        if (d.k == 1) EXPECT(want[0] == 0.0f && std::signbit(want[0]));  // -0
        for (kl_gemm_impl impl : {KL_GEMM_TILED, KL_GEMM_REGBLOCK, KL_GEMM_AUTO}) {
            const std::vector<float> c = run_gemm(impl, d, a, b);
            const bool same =
                std::equal(c.begin(), c.end(), want.begin(), [](float p, float q) { return same_float(p, q); });
            if (!same) std::printf("  tiny products: %s differs from naive (signed zeros included)\n", impl_name(impl));
            EXPECT(same);
        }
    }
}

// Guard bands around C on the device path: no write outside C.
void test_gemm_guard_bands() {
    using kl::detail::DeviceBuffer;
    const std::int64_t kGuard = 4096;
    const std::uint32_t kPattern = 0x7fbadbadu;
    float sentinel;
    std::memcpy(&sentinel, &kPattern, sizeof(sentinel));
    for (const Dims& d : {Dims{1, 1, 1}, Dims{17, 33, 5}, Dims{65, 63, 129}, Dims{130, 70, 31}}) {
        const std::vector<float> a = fill(1200 + d.m, d.m * d.k, -1.0f, 1.0f);
        const std::vector<float> b = fill(1300 + d.n, d.k * d.n, -1.0f, 1.0f);
        for (kl_gemm_impl impl : kImpls) {
            const std::int64_t nc = d.m * d.n, total = nc + 2 * kGuard;
            std::vector<float> hc(static_cast<std::size_t>(total), sentinel);
            DeviceBuffer da, db, dc;
            EXPECT_STATUS(DeviceBuffer::allocate(a.size() * 4, da), KL_OK);
            EXPECT_STATUS(DeviceBuffer::allocate(b.size() * 4, db), KL_OK);
            EXPECT_STATUS(DeviceBuffer::allocate(static_cast<std::size_t>(total) * 4, dc), KL_OK);
            EXPECT(cudaMemcpy(da.get(), a.data(), a.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess);
            EXPECT(cudaMemcpy(db.get(), b.data(), b.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess);
            EXPECT(cudaMemcpy(dc.get(), hc.data(), hc.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess);
            EXPECT_STATUS(kl_gemm_device(d.m, d.n, d.k, da.as<float>(), db.as<float>(), dc.as<float>() + kGuard, impl,
                                         nullptr),
                          KL_OK);
            EXPECT(cudaDeviceSynchronize() == cudaSuccess);
            EXPECT(cudaMemcpy(hc.data(), dc.get(), hc.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
            bool bands = true;
            for (std::int64_t i = 0; i < kGuard; ++i) {
                bands = bands && kl_test::bits(hc[static_cast<std::size_t>(i)]) == kPattern &&
                        kl_test::bits(hc[static_cast<std::size_t>(kGuard + nc + i)]) == kPattern;
            }
            if (!bands) std::printf("  guard band overwritten: %s\n", impl_name(impl));
            EXPECT(bands);
            const CheckResult r = check_gemm(d.m, d.n, d.k, a.data(), b.data(), hc.data() + kGuard);
            expect_ok(impl_name(impl), d, r, false);
        }
    }
}

void test_gemm_invalid_and_unsupported() {
    const float kSentinel = -12345.0f;
    std::vector<float> a(16, 1.0f), b(16, 1.0f), c(16, kSentinel);
    const std::uint64_t mallocs = kl::detail::device_malloc_count();
    EXPECT_STATUS(kl_gemm_host(-1, 4, 4, a.data(), b.data(), c.data(), KL_GEMM_AUTO), KL_ERR_INVALID_ARGUMENT);
    EXPECT_STATUS(kl_gemm_host(4, 4, -1, a.data(), b.data(), c.data(), KL_GEMM_AUTO), KL_ERR_INVALID_ARGUMENT);
    EXPECT_STATUS(kl_gemm_host(4, 4, 4, a.data(), b.data(), c.data(), static_cast<kl_gemm_impl>(7)),
                  KL_ERR_INVALID_ARGUMENT);
    EXPECT_STATUS(kl_gemm_host(4, 4, 4, a.data(), b.data(), c.data(), static_cast<kl_gemm_impl>(-1)),
                  KL_ERR_INVALID_ARGUMENT);
    EXPECT_STATUS(kl_gemm_host(4, 4, 4, nullptr, b.data(), c.data(), KL_GEMM_AUTO), KL_ERR_INVALID_ARGUMENT);
    EXPECT_STATUS(kl_gemm_host(4, 4, 4, a.data(), b.data(), nullptr, KL_GEMM_AUTO), KL_ERR_INVALID_ARGUMENT);
    EXPECT_STATUS(kl_gemm_host(KL_GEMM_MAX_DIM + 1, 1, 1, a.data(), b.data(), c.data(), KL_GEMM_AUTO),
                  KL_ERR_UNSUPPORTED);
    EXPECT_STATUS(kl_gemm_host(1, 1, KL_GEMM_MAX_DIM + 1, a.data(), b.data(), c.data(), KL_GEMM_AUTO),
                  KL_ERR_UNSUPPORTED);
    EXPECT_STATUS(kl_gemm_host(1 << 14, 1 << 13, 1, a.data(), b.data(), c.data(), KL_GEMM_AUTO),
                  KL_ERR_UNSUPPORTED);  // C would have 2^27 elements
    kl_test::expect_last_error_mentions("supported domain", KL_HERE, __LINE__);
    for (float v : c) EXPECT(same_float(v, kSentinel));
    EXPECT(kl::detail::device_malloc_count() == mallocs);
    EXPECT_STATUS(kl_gemm_host(0, 4, 4, nullptr, nullptr, nullptr, KL_GEMM_AUTO), KL_OK);
    EXPECT_STATUS(kl_gemm_supported(1 << 14, 1 << 12, 1 << 12, KL_GEMM_AUTO), KL_OK);  // A, C: 2^26 elements
    EXPECT_STATUS(kl_gemm_supported(1 << 14, 1 << 12, 1 << 14, KL_GEMM_AUTO), KL_ERR_UNSUPPORTED);  // A: 2^28
    EXPECT(kl::cuda::gemm_resolve(KL_GEMM_AUTO) == KL_GEMM_REGBLOCK);

    // Device entry point: overlap and alignment.
    using kl::detail::DeviceBuffer;
    DeviceBuffer d;
    EXPECT_STATUS(DeviceBuffer::allocate(256 * sizeof(float), d), KL_OK);
    // Initialized: the valid launches below really run (Compute Sanitizer
    // initcheck flagged reads of this buffer while it was left unset).
    EXPECT(cudaMemset(d.get(), 0, 256 * sizeof(float)) == cudaSuccess);
    float* p = d.as<float>();
    // A = p[0..16) (4x4), B = p[16..32), C = p[32..48)
    EXPECT_STATUS(kl_gemm_device(4, 4, 4, p, p + 16, p + 32, KL_GEMM_TILED, nullptr), KL_OK);
    EXPECT_STATUS(kl_gemm_device(4, 4, 4, p, p, p + 32, KL_GEMM_TILED, nullptr), KL_OK);  // A == B is fine
    EXPECT_STATUS(kl_gemm_device(4, 4, 4, p, p + 16, p + 10, KL_GEMM_TILED, nullptr), KL_ERR_INVALID_ARGUMENT);
    EXPECT_STATUS(kl_gemm_device(4, 4, 4, p, p + 16, p + 20, KL_GEMM_TILED, nullptr), KL_ERR_INVALID_ARGUMENT);
    auto skew = [](float* q) { return reinterpret_cast<float*>(reinterpret_cast<char*>(q) + 2); };
    EXPECT_STATUS(kl_gemm_device(4, 4, 4, skew(p + 64), p + 16, p + 32, KL_GEMM_TILED, nullptr),
                  KL_ERR_INVALID_ARGUMENT);
    EXPECT_STATUS(kl_gemm_device(4, 4, 4, p, skew(p + 64), p + 32, KL_GEMM_TILED, nullptr), KL_ERR_INVALID_ARGUMENT);
    EXPECT_STATUS(kl_gemm_device(4, 4, 4, p, p + 16, skew(p + 128), KL_GEMM_TILED, nullptr), KL_ERR_INVALID_ARGUMENT);
    EXPECT(cudaDeviceSynchronize() == cudaSuccess);
}

// Host API: C may alias A (inputs are copied before C is written).
void test_gemm_host_aliasing() {
    const Dims d{8, 8, 8};
    std::vector<float> a = fill(1400, 64, -1.0f, 1.0f);
    const std::vector<float> b = fill(1401, 64, -1.0f, 1.0f);
    const std::vector<float> want = run_gemm(KL_GEMM_REGBLOCK, d, a, b);
    EXPECT_STATUS(kl_gemm_host(8, 8, 8, a.data(), b.data(), a.data(), KL_GEMM_REGBLOCK), KL_OK);
    EXPECT(std::equal(a.begin(), a.end(), want.begin(), [](float p, float q) { return same_float(p, q); }));
}

}  // namespace

int main(int argc, char** argv) {
    static const kl_test::TestCase tests[] = {
        {"gemm_boundary_shapes_uniform", test_gemm_boundary_shapes_uniform},
        {"gemm_wide_magnitudes", test_gemm_wide_magnitudes},
        {"gemm_integer_exact", test_gemm_integer_exact},
        {"gemm_bitwise_identical", test_gemm_bitwise_identical},
        {"gemm_cublas_within_bound", test_gemm_cublas_within_bound},
        {"gemm_k_zero_and_one", test_gemm_k_zero_and_one},
        {"gemm_special_values", test_gemm_special_values},
        {"gemm_tiny_products_and_signed_zeros", test_gemm_tiny_products_and_signed_zeros},
        {"gemm_guard_bands", test_gemm_guard_bands},
        {"gemm_invalid_and_unsupported", test_gemm_invalid_and_unsupported},
        {"gemm_host_aliasing", test_gemm_host_aliasing},
    };
    return kl_test::run_tests(argc, argv, tests);
}
