// GEMM benchmark, C++ harness. Protocol: bench_harness.hpp and
// docs/methodology.md section 2, with the batch size chosen from the work
// (docs/gemm.md, "Benchmark plan"). Compares the KernelLens implementations
// with each other and with cuBLAS (cublasSgemm, CUBLAS_DEFAULT_MATH: no TF32)
// in the same process, stream and batches (ABBA).
//
// Inputs x ~ U[-1, 1]. Correctness: before timing and after the series every
// output is poisoned and recomputed, and 32 evenly spaced rows (including the
// first and the last) are checked against the float64 reference with the
// tested bound (a full check of the largest shapes would take minutes on the
// CPU).
//
// Bounded: <= 2^26 elements per matrix, NAIVE only up to 2048^3 work, TILED
// only up to 4e10 FLOP, each
// launch well below the 2 s watchdog; run it through scripts/gpu_guard.py.
//
// Usage: bench_gemm --out FILE.csv [--process N] [--samples N] [--warmup N]
//                   [--target-ms X] [--commit SHA] [--quick]
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "bench_harness.hpp"
#include "common/device_buffer.hpp"
#include "cuda/kernels.hpp"
#include "gemm_reference.hpp"
#include "kernellens/kernellens.h"

namespace {

using kl::detail::DeviceBuffer;
using kl_bench::Context;
using kl_bench::die;
using kl_bench::Impl;
using kl_bench::Shape;

// Pre-registered in docs/gemm.md: Shape{rows = M, cols = N, depth = K}.
const Shape kShapes[] = {{256, 256, 256},   {512, 512, 512},    {1024, 1024, 1024}, {2048, 2048, 2048},
                         {4096, 4096, 4096}, {1000, 1000, 1000}, {1023, 1023, 1023}, {8192, 768, 768},
                         {8192, 3072, 768},  {8192, 768, 3072},  {64, 4096, 4096},   {4096, 4096, 64}};
const Shape kQuickShapes[] = {{512, 512, 512}, {1023, 1023, 1023}};
constexpr double kNaiveMaxWork = 2.0 * 2048.0 * 2048.0 * 2048.0;  // 2 M N K (pre-registered)
// Safety cap added after the dry run (docs/gemm.md): TILED took ~170 ms per
// launch at 4096^3, above the project's 100 ms launch target.
constexpr double kTiledMaxWork = 4e10;

#define BENCH_CUBLAS(expr)                                                                    \
    do {                                                                                      \
        const cublasStatus_t s_ = (expr);                                                     \
        if (s_ != CUBLAS_STATUS_SUCCESS) {                                                    \
            std::fprintf(stderr, "bench_gemm: %s failed: status %d\n", #expr, static_cast<int>(s_)); \
            std::exit(1);                                                                     \
        }                                                                                     \
    } while (0)

// Batch size from the work: one batch lasts about target_ms at an assumed
// 5 TFLOP/s (docs/gemm.md), the same for every implementation of a shape.
int gemm_batch_k(const Shape& s, double target_ms) {
    const double flops = 2.0 * static_cast<double>(s.rows) * static_cast<double>(s.cols) * static_cast<double>(s.depth);
    const double k = std::ceil(target_ms * 1e-3 * 5e12 / flops);
    return static_cast<int>(std::clamp(k, 1.0, 2000.0));
}

void bench_shape(Context& c, const Shape& s) {
    const std::int64_t m = s.rows, n = s.cols, k = s.depth;
    const std::size_t na = static_cast<std::size_t>(m * k) * 4, nb = static_cast<std::size_t>(k * n) * 4,
                      nc = static_cast<std::size_t>(m * n) * 4;
    // Enough (A, B, C) sets that one rotation touches >= 4x L2.
    const int sets = static_cast<int>(std::max<std::int64_t>(
        1, (4LL * c.l2_bytes + static_cast<std::int64_t>(na + nb + nc) - 1) / static_cast<std::int64_t>(na + nb + nc)));
    std::vector<DeviceBuffer> as(sets), bs(sets), cs(sets);
    std::vector<std::vector<float>> host_a(sets), host_b(sets);
    for (int p = 0; p < sets; ++p) {
        if (DeviceBuffer::allocate(na, as[p]) != KL_OK || DeviceBuffer::allocate(nb, bs[p]) != KL_OK ||
            DeviceBuffer::allocate(nc, cs[p]) != KL_OK) {
            die("allocate");
        }
        host_a[p].resize(static_cast<std::size_t>(m * k));
        host_b[p].resize(static_cast<std::size_t>(k * n));
        if (kl_fill_uniform(7000 + 31 * p + static_cast<std::uint64_t>(k), m * k, -1.0f, 1.0f, host_a[p].data()) !=
                KL_OK ||
            kl_fill_uniform(8000 + 31 * p + static_cast<std::uint64_t>(n), k * n, -1.0f, 1.0f, host_b[p].data()) !=
                KL_OK) {
            die("fill");
        }
        BENCH_CUDA(cudaMemcpy(as[p].get(), host_a[p].data(), na, cudaMemcpyHostToDevice));
        BENCH_CUDA(cudaMemcpy(bs[p].get(), host_b[p].data(), nb, cudaMemcpyHostToDevice));
    }
    cudaStream_t stream;
    BENCH_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cublasHandle_t handle = nullptr;
    BENCH_CUBLAS(cublasCreate(&handle));
    BENCH_CUBLAS(cublasSetStream(handle, stream));
    BENCH_CUBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));  // no TF32: FP32 means FP32
    {
        // Record what cuBLAS will use: its math mode, and the environment
        // variable that can force TF32 behind an application's back.
        cublasMath_t mode{};
        BENCH_CUBLAS(cublasGetMathMode(handle, &mode));
        char* env = nullptr;
        std::size_t env_len = 0;
        const bool have_env = _dupenv_s(&env, &env_len, "NVIDIA_TF32_OVERRIDE") == 0 && env != nullptr;
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "{\"type\": \"cublas_config\", \"rows\": %lld, \"cols\": %lld, \"depth\": %lld, \"math_mode\": %d, "
                      "\"tf32_override_env\": \"%s\"}",
                      static_cast<long long>(m), static_cast<long long>(n), static_cast<long long>(k),
                      static_cast<int>(mode), have_env ? kl_bench::json_escape(env).c_str() : "unset");
        std::free(env);
        kl_bench::meta_item(c, buf);
    }

    // 32 evenly spaced rows (first and last included), each checked with m = 1.
    auto check = [&](int p) {
        BENCH_CUDA(cudaStreamSynchronize(stream));
        std::vector<float> row(static_cast<std::size_t>(n));
        double worst = 0;
        long long bad = 0;
        const std::int64_t samples = std::min<std::int64_t>(32, m);
        for (std::int64_t q = 0; q < samples; ++q) {
            const std::int64_t i = samples == 1 ? 0 : q * (m - 1) / (samples - 1);
            BENCH_CUDA(cudaMemcpy(row.data(), cs[p].as<float>() + i * n, static_cast<std::size_t>(n) * 4,
                                  cudaMemcpyDeviceToHost));
            const kl_test::gemm::CheckResult r =
                kl_test::gemm::check_gemm(1, n, k, host_a[p].data() + i * k, host_b[p].data(), row.data());
            worst = std::max(worst, r.worst_ratio);
            bad += r.violations + r.special_mismatches + r.out_of_domain;
        }
        return std::make_pair(worst, bad);
    };
    auto kernellens = [&](const char* name, kl_gemm_impl impl) {
        return Impl{name,
                    [&, impl](int p) {
                        if (kl::cuda::launch_gemm(m, n, k, as[p].as<float>(), bs[p].as<float>(), cs[p].as<float>(), impl,
                                                  stream) != KL_OK) {
                            die("launch_gemm");
                        }
                    },
                    check};
    };
    std::vector<Impl> impls;
    if (2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k) <= kNaiveMaxWork) {
        impls.push_back(kernellens("naive", KL_GEMM_NAIVE));
        // Control (docs/gemm.md): the same naive kernel with L1-bypassing loads.
        impls.push_back(Impl{"naive_l2only",
                             [&](int p) {
                                 if (kl::cuda::launch_gemm_naive_l2only(m, n, k, as[p].as<float>(), bs[p].as<float>(),
                                                                        cs[p].as<float>(), stream) != KL_OK) {
                                     die("launch_gemm_naive_l2only");
                                 }
                             },
                             check});
    }
    if (2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k) <= kTiledMaxWork) {
        impls.push_back(kernellens("tiled", KL_GEMM_TILED));
    }
    impls.push_back(kernellens("regblock", KL_GEMM_REGBLOCK));
    const float one = 1.0f, zero = 0.0f;
    impls.push_back(Impl{"cublas",
                         [&](int p) {
                             // Row-major C = A B as column-major C^T = B^T A^T.
                             BENCH_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(n),
                                                      static_cast<int>(m), static_cast<int>(k), &one, bs[p].as<float>(),
                                                      static_cast<int>(n), as[p].as<float>(), static_cast<int>(k), &zero,
                                                      cs[p].as<float>(), static_cast<int>(n)));
                         },
                         check});

    kl_bench::run_protocol(
        c, s, sets, stream, impls,
        [&] {
            for (int p = 0; p < sets; ++p) BENCH_CUDA(cudaMemsetAsync(cs[p].get(), 0xFF, nc, stream));
        },
        gemm_batch_k(s, c.opt->target_ms));
    BENCH_CUBLAS(cublasDestroy(handle));
    BENCH_CUDA(cudaStreamDestroy(stream));
}

}  // namespace

int main(int argc, char** argv) {
    // Clock ramp: REGBLOCK launches on zeroed 1024^3 inputs.
    const std::int64_t r = 1024;
    const std::size_t rbytes = static_cast<std::size_t>(r * r) * 4;
    DeviceBuffer ra, rb, rc;
    auto ramp = [&] {
        if (!ra.get()) {
            if (DeviceBuffer::allocate(rbytes, ra) != KL_OK || DeviceBuffer::allocate(rbytes, rb) != KL_OK ||
                DeviceBuffer::allocate(rbytes, rc) != KL_OK) {
                die("ramp");
            }
            BENCH_CUDA(cudaMemset(ra.get(), 0, rbytes));
            BENCH_CUDA(cudaMemset(rb.get(), 0, rbytes));
        }
        for (int i = 0; i < 5; ++i) {
            if (kl::cuda::launch_gemm(r, r, r, ra.as<float>(), rb.as<float>(), rc.as<float>(), KL_GEMM_REGBLOCK, 0) !=
                KL_OK) {
                die("ramp launch");
            }
        }
        BENCH_CUDA(cudaDeviceSynchronize());
    };
    return kl_bench::bench_main(argc, argv, "bench_gemm", ramp, [&](Context& c) {
        ra = DeviceBuffer();
        rb = DeviceBuffer();
        rc = DeviceBuffer();
        if (c.opt->quick) {
            for (const Shape& s : kQuickShapes) bench_shape(c, s);
        } else {
            for (const Shape& s : kShapes) bench_shape(c, s);
        }
    });
}
