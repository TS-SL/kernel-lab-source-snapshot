// Profiling driver (docs/profiling.md): launches each kernel of interest
// once, at fixed shapes and seeded inputs, so Nsight Compute profiles exactly
// these launches. Nothing is timed here. Run it through
// scripts/run-profile.ps1 (ncu --clock-control none, under gpu_guard).
//
// Launch order (the profile CSV lists them in this order):
//   softmax 4 x 65536:   multipass (3 kernels), fused_online
//   layer norm, for cols 2048, 4096, 8192 at 2^22 elements:
//                        fused_block, fused_block_maxsmem (control copy,
//                        largest shared-memory carveout), fused_smem
//   GEMM 1024^3:         naive, naive_l2only, tiled, regblock, cuBLAS
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "common/device_buffer.hpp"
#include "cuda/kernels.hpp"
#include "kernellens/kernellens.h"

namespace {

using kl::detail::DeviceBuffer;

[[noreturn]] void die(const char* what) {
    std::fprintf(stderr, "profile_kernels: %s: %s\n", what, kl_last_error_message());
    std::exit(1);
}

void check(kl_status s, const char* what) {
    if (s != KL_OK) die(what);
}

void cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "profile_kernels: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

DeviceBuffer filled(std::int64_t n, std::uint64_t seed, float lo, float hi) {
    std::vector<float> h(static_cast<std::size_t>(n));
    check(kl_fill_uniform(seed, n, lo, hi, h.data()), "fill");
    DeviceBuffer d;
    check(DeviceBuffer::allocate(h.size() * sizeof(float), d), "allocate");
    cuda(cudaMemcpy(d.get(), h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice), "copy");
    return d;
}

}  // namespace

int main() {
    {
        const std::int64_t rows = 4, cols = 65536;
        DeviceBuffer x = filled(rows * cols, 11, -10.0f, 10.0f), y, ws;
        check(DeviceBuffer::allocate(static_cast<std::size_t>(rows * cols) * 4, y), "allocate");
        check(DeviceBuffer::allocate(static_cast<std::size_t>(2 * rows) * 4, ws), "allocate");
        check(kl::cuda::launch_softmax(rows, cols, x.as<float>(), y.as<float>(), KL_SOFTMAX_MULTIPASS, ws.as<float>(), 0),
              "softmax multipass");
        check(kl::cuda::launch_softmax(rows, cols, x.as<float>(), y.as<float>(), KL_SOFTMAX_FUSED_ONLINE, nullptr, 0),
              "softmax online");
        cuda(cudaDeviceSynchronize(), "sync");
    }
    for (std::int64_t cols : {2048, 4096, 8192}) {
        const std::int64_t rows = (std::int64_t{1} << 22) / cols;
        DeviceBuffer x = filled(rows * cols, 20 + static_cast<std::uint64_t>(cols), -10.0f, 10.0f), y;
        DeviceBuffer g = filled(cols, 30, 0.5f, 1.5f), b = filled(cols, 31, -1.0f, 1.0f);
        check(DeviceBuffer::allocate(static_cast<std::size_t>(rows * cols) * 4, y), "allocate");
        check(kl::cuda::launch_layernorm(rows, cols, x.as<float>(), g.as<float>(), b.as<float>(), 1e-5f, y.as<float>(),
                                         KL_LAYERNORM_FUSED_BLOCK, nullptr, 0),
              "fused_block");
        check(kl::cuda::launch_layernorm_padded(rows, cols, x.as<float>(), g.as<float>(), b.as<float>(), 1e-5f,
                                                y.as<float>(), KL_LAYERNORM_FUSED_BLOCK, 0, 0),
              "fused_block_maxsmem");
        check(kl::cuda::launch_layernorm(rows, cols, x.as<float>(), g.as<float>(), b.as<float>(), 1e-5f, y.as<float>(),
                                         KL_LAYERNORM_FUSED_SMEM, nullptr, 0),
              "fused_smem");
        cuda(cudaDeviceSynchronize(), "sync");
    }
    {
        const std::int64_t n = 1024;
        DeviceBuffer a = filled(n * n, 40, -1.0f, 1.0f), b = filled(n * n, 41, -1.0f, 1.0f), c;
        check(DeviceBuffer::allocate(static_cast<std::size_t>(n * n) * 4, c), "allocate");
        for (kl_gemm_impl impl : {KL_GEMM_NAIVE}) {
            check(kl::cuda::launch_gemm(n, n, n, a.as<float>(), b.as<float>(), c.as<float>(), impl, 0), "gemm naive");
        }
        check(kl::cuda::launch_gemm_naive_l2only(n, n, n, a.as<float>(), b.as<float>(), c.as<float>(), 0), "l2only");
        check(kl::cuda::launch_gemm(n, n, n, a.as<float>(), b.as<float>(), c.as<float>(), KL_GEMM_TILED, 0), "tiled");
        check(kl::cuda::launch_gemm(n, n, n, a.as<float>(), b.as<float>(), c.as<float>(), KL_GEMM_REGBLOCK, 0), "regblock");
        cublasHandle_t h = nullptr;
        if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS || cublasSetMathMode(h, CUBLAS_DEFAULT_MATH) != CUBLAS_STATUS_SUCCESS) {
            die("cublas init");
        }
        const float one = 1.0f, zero = 0.0f;
        const int ni = static_cast<int>(n);
        if (cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, ni, ni, ni, &one, b.as<float>(), ni, a.as<float>(), ni, &zero,
                        c.as<float>(), ni) != CUBLAS_STATUS_SUCCESS) {
            die("cublasSgemm");
        }
        cuda(cudaDeviceSynchronize(), "sync");
        cublasDestroy(h);
    }
    if (kl_device_live_bytes() != 0) die("device memory still held");
    std::printf("profile_kernels: done\n");
    return 0;
}
