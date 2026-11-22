// Smoke-test kernel. Its job is to prove that the toolchain
// compiles sm_86 code, that the kernel executes on the GPU, and that results
// match an independent reference - not to be fast.
#include <algorithm>
#include <cstdint>

#include "common/cuda_check.hpp"
#include "cuda/kernels.hpp"

namespace kl::cuda {

namespace {

// Grid-stride loop: correct for any n with any grid size, so the grid can be
// capped without a separate tail path. 64-bit indices because n can exceed
// INT_MAX in principle (the VRAM cap bounds it to 2^28 floats per buffer
// today, but the index math should not be the thing that breaks).
__global__ void saxpy_kernel(std::int64_t n, float a, const float* __restrict__ x,
                             float* __restrict__ y) {
    const std::int64_t stride = static_cast<std::int64_t>(blockDim.x) * gridDim.x;
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += stride) {
        // Explicit fmaf: one rounding of the exact a*x+y. Spelled out rather
        // than left to nvcc's default contraction so the reference tolerance
        // argument (docs/methodology.md) does not depend on compiler flags.
        y[i] = fmaf(a, x[i], y[i]);
    }
}

}  // namespace

kl_status launch_saxpy(std::int64_t n, float a, const float* x, float* y, int block_size,
                       cudaStream_t stream) {
    if (n < 0) {
        return detail::fail(KL_ERR_INVALID_ARGUMENT, "launch_saxpy: n must be >= 0");
    }
    if (n == 0) {
        return KL_OK;
    }
    if (x == nullptr || y == nullptr) {
        return detail::fail(KL_ERR_INVALID_ARGUMENT, "launch_saxpy: null device pointer");
    }
    if (block_size <= 0) {
        return detail::fail(KL_ERR_INVALID_ARGUMENT, "launch_saxpy: block_size must be > 0");
    }
    // Cap the grid; the grid-stride loop covers the remainder. 65535 blocks
    // of 256 threads is ~16.7M threads, far more than the 28 SMs * 1536
    // resident threads of an RTX 3060, so the cap never limits occupancy.
    const std::int64_t blocks_needed = (n + block_size - 1) / block_size;
    const unsigned grid = static_cast<unsigned>(std::min<std::int64_t>(blocks_needed, 65535));
    // cudaGetLastError() reports the most recent error from ANY runtime call,
    // not just launches. Every runtime call in KernelLens is checked, except
    // cudaFree in DeviceBuffer::reset (destructors cannot report), so a
    // pending non-sticky error here was either reported earlier or comes from
    // that cudaFree; clear it so KL_CHECK_LAUNCH below attributes only this
    // launch. A sticky error would make the launch itself fail, so no
    // context-breaking error is hidden.
    (void)cudaGetLastError();
    saxpy_kernel<<<grid, static_cast<unsigned>(block_size), 0, stream>>>(n, a, x, y);
    KL_CHECK_LAUNCH();
    return KL_OK;
}

}  // namespace kl::cuda
