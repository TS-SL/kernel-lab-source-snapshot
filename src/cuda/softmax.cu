// Row-wise FP32 softmax: one baseline and two fused implementations.
// Design, supported domain and the error bound: docs/softmax.md.
#include <cstdint>

#include "common/cuda_check.hpp"
#include "cuda/kernels.hpp"
#include "cuda/launch_checks.hpp"
#include "cuda/reduce.cuh"

namespace kl::cuda {

namespace {

constexpr int kWarpRowsPerBlock = 4;  // FUSED_WARP: one row per warp, 4 warps per block

// ---- MULTIPASS baseline: three kernels, intermediates in global memory ----

__global__ void __launch_bounds__(kReduceBlock)
    row_max_kernel(std::int64_t cols, const float* __restrict__ x, float* __restrict__ row_max) {
    const float* row = x + static_cast<std::int64_t>(blockIdx.x) * cols;
    float m = -CUDART_INF_F;
    for (std::int64_t j = threadIdx.x; j < cols; j += kReduceBlock) m = fmaxf(m, row[j]);
    m = block_reduce(m, MaxOp{}, -CUDART_INF_F);
    if (threadIdx.x == 0) row_max[blockIdx.x] = m;
}

__global__ void __launch_bounds__(kReduceBlock)
    row_sum_kernel(std::int64_t cols, const float* __restrict__ x,
                   const float* __restrict__ row_max, float* __restrict__ row_sum) {
    const float* row = x + static_cast<std::int64_t>(blockIdx.x) * cols;
    const float m = row_max[blockIdx.x];
    float s = 0.0f;
    for (std::int64_t j = threadIdx.x; j < cols; j += kReduceBlock) s += expf(row[j] - m);
    s = block_reduce(s, SumOp{}, 0.0f);
    if (threadIdx.x == 0) row_sum[blockIdx.x] = s;
}

__global__ void __launch_bounds__(kReduceBlock)
    normalize_kernel(std::int64_t cols, const float* __restrict__ x,
                     const float* __restrict__ row_max, const float* __restrict__ row_sum,
                     float* __restrict__ y) {
    const std::int64_t offset = static_cast<std::int64_t>(blockIdx.x) * cols;
    const float m = row_max[blockIdx.x];
    const float s = row_sum[blockIdx.x];
    for (std::int64_t j = threadIdx.x; j < cols; j += kReduceBlock) {
        y[offset + j] = expf(x[offset + j] - m) / s;
    }
}

// ---- FUSED_ONLINE: one kernel, block per row, two reads of x ----

__global__ void __launch_bounds__(kReduceBlock)
    softmax_online_kernel(std::int64_t cols, const float* __restrict__ x, float* __restrict__ y) {
    const std::int64_t offset = static_cast<std::int64_t>(blockIdx.x) * cols;
    MaxSum a = max_sum_identity();
    for (std::int64_t j = threadIdx.x; j < cols; j += kReduceBlock) max_sum_add(a, x[offset + j]);
    a = block_reduce_max_sum(a);  // contains __syncthreads; every thread gets (m, s)
    for (std::int64_t j = threadIdx.x; j < cols; j += kReduceBlock) {
        y[offset + j] = expf(x[offset + j] - a.m) / a.s;
    }
}

// ---- FUSED_WARP: one warp per row, row cached in registers, one read ----
//
// Lane l holds elements l, l + 32, l + 64, ... (coalesced across the warp).
// VPL (values per lane) is a compile-time constant so v[] stays in
// registers; cols <= 32 * VPL.

template <int VPL>
__global__ void __launch_bounds__(kWarpRowsPerBlock * kWarpSize)
    softmax_warp_kernel(std::int64_t rows, int cols, const float* __restrict__ x,
                        float* __restrict__ y) {
    const int lane = threadIdx.x % kWarpSize;
    const std::int64_t r =
        static_cast<std::int64_t>(blockIdx.x) * kWarpRowsPerBlock + threadIdx.x / kWarpSize;
    // r is the same for all lanes of a warp, so whole warps leave together
    // and the full-mask shuffles below always see 32 active lanes. There is
    // no __syncthreads in this kernel.
    if (r >= rows) return;
    const float* row = x + r * cols;
    float* out = y + r * cols;

    float v[VPL];
    float m = -CUDART_INF_F;
#pragma unroll
    for (int i = 0; i < VPL; ++i) {
        const int j = lane + i * kWarpSize;
        v[i] = j < cols ? row[j] : -CUDART_INF_F;
        m = fmaxf(m, v[i]);
    }
    m = warp_reduce(m, MaxOp{});

    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < VPL; ++i) {
        const int j = lane + i * kWarpSize;
        v[i] = j < cols ? expf(v[i] - m) : 0.0f;  // padding contributes exactly 0
        s += v[i];
    }
    s = warp_reduce(s, SumOp{});

#pragma unroll
    for (int i = 0; i < VPL; ++i) {
        const int j = lane + i * kWarpSize;
        if (j < cols) out[j] = v[i] / s;
    }
}

template <int VPL>
void launch_warp(std::int64_t rows, int cols, const float* x, float* y, cudaStream_t stream) {
    const auto blocks = static_cast<unsigned>((rows + kWarpRowsPerBlock - 1) / kWarpRowsPerBlock);
    softmax_warp_kernel<VPL><<<blocks, kWarpRowsPerBlock * kWarpSize, 0, stream>>>(rows, cols, x, y);
}

}  // namespace

kl_status softmax_check(std::int64_t rows, std::int64_t cols, kl_softmax_impl impl) {
    if (rows < 0 || cols < 0) {
        return detail::fail(KL_ERR_INVALID_ARGUMENT, "softmax: rows and cols must be >= 0");
    }
    if (impl != KL_SOFTMAX_AUTO && impl != KL_SOFTMAX_MULTIPASS && impl != KL_SOFTMAX_FUSED_ONLINE &&
        impl != KL_SOFTMAX_FUSED_WARP) {
        return detail::fail(KL_ERR_INVALID_ARGUMENT, "softmax: unknown implementation id");
    }
    if (rows == 0 || cols == 0) {
        return KL_OK;  // empty matrix: nothing to compute
    }
    if (cols > KL_SOFTMAX_MAX_COLS || rows > KL_SOFTMAX_MAX_ROWS ||
        rows * cols > KL_SOFTMAX_MAX_ELEMENTS) {  // both <= 2^20: no overflow
        return detail::fail(KL_ERR_UNSUPPORTED,
                            "softmax: shape outside the supported domain (docs/softmax.md)");
    }
    if (impl == KL_SOFTMAX_FUSED_WARP && cols > KL_SOFTMAX_WARP_MAX_COLS) {
        return detail::fail(KL_ERR_UNSUPPORTED, "softmax: FUSED_WARP supports cols <= 1024");
    }
    return KL_OK;
}

kl_softmax_impl softmax_resolve(std::int64_t rows, std::int64_t cols, kl_softmax_impl impl) {
    if (impl != KL_SOFTMAX_AUTO) return impl;
    if (cols <= KL_SOFTMAX_WARP_MAX_COLS) return KL_SOFTMAX_FUSED_WARP;
    // Measured (docs/softmax.md): with few long rows the
    // multipass kernels beat the online kernel; a rows sweep at 65536 cols
    // puts the crossover between 28 rows (online slower) and 64 rows (online
    // faster), equal at 56. Why the online kernel is slower there is a
    // hypothesis. The profile (docs/profiling.md, H1) shows the kernel
    // latency-bound on memory loads at low occupancy; it does not single out
    // the exp dependency on the previous running max.
    return rows >= kSoftmaxOnlineMinRows ? KL_SOFTMAX_FUSED_ONLINE : KL_SOFTMAX_MULTIPASS;
}

std::size_t softmax_workspace_floats(std::int64_t rows, kl_softmax_impl resolved) {
    return resolved == KL_SOFTMAX_MULTIPASS ? 2 * static_cast<std::size_t>(rows) : 0;
}

kl_status launch_softmax(std::int64_t rows, std::int64_t cols, const float* x, float* y,
                         kl_softmax_impl impl, float* workspace, cudaStream_t stream) {
    KL_TRY(softmax_check(rows, cols, impl));
    if (rows == 0 || cols == 0) return KL_OK;
    if (x == nullptr || y == nullptr) {
        return detail::fail(KL_ERR_INVALID_ARGUMENT, "launch_softmax: null device pointer");
    }
    {
        // Shared checks (src/cuda/launch_checks.hpp): misaligned pointers
        // would cause a sticky CUDA error; the kernels use __restrict__, so
        // x, y and the workspace must not overlap.
        using checks::float_aligned;
        using checks::overlap;
        if (!float_aligned(x) || !float_aligned(y) || !float_aligned(workspace)) {
            return detail::fail(KL_ERR_INVALID_ARGUMENT, "launch_softmax: pointers must be 4-byte aligned");
        }
        const std::size_t n = static_cast<std::size_t>(rows * cols) * sizeof(float);
        if (overlap(x, n, y, n)) {
            return detail::fail(KL_ERR_INVALID_ARGUMENT, "launch_softmax: x and y overlap");
        }
        const std::size_t wn = softmax_workspace_floats(rows, softmax_resolve(rows, cols, impl)) * sizeof(float);
        if (overlap(workspace, wn, x, n) || overlap(workspace, wn, y, n)) {
            return detail::fail(KL_ERR_INVALID_ARGUMENT, "launch_softmax: workspace overlaps x or y");
        }
    }
    const kl_softmax_impl resolved = softmax_resolve(rows, cols, impl);
    if (resolved == KL_SOFTMAX_MULTIPASS && workspace == nullptr) {
        return detail::fail(KL_ERR_INVALID_ARGUMENT, "launch_softmax: MULTIPASS needs a workspace");
    }
    (void)cudaGetLastError();  // see launch_saxpy: attribute errors to these launches only
    const auto grid = static_cast<unsigned>(rows);
    switch (resolved) {
        case KL_SOFTMAX_MULTIPASS: {
            float* row_max = workspace;
            float* row_sum = workspace + rows;
            row_max_kernel<<<grid, kReduceBlock, 0, stream>>>(cols, x, row_max);
            KL_CHECK_LAUNCH();
            row_sum_kernel<<<grid, kReduceBlock, 0, stream>>>(cols, x, row_max, row_sum);
            KL_CHECK_LAUNCH();
            normalize_kernel<<<grid, kReduceBlock, 0, stream>>>(cols, x, row_max, row_sum, y);
            KL_CHECK_LAUNCH();
            return KL_OK;
        }
        case KL_SOFTMAX_FUSED_ONLINE:
            softmax_online_kernel<<<grid, kReduceBlock, 0, stream>>>(cols, x, y);
            KL_CHECK_LAUNCH();
            return KL_OK;
        case KL_SOFTMAX_FUSED_WARP: {
            const int c = static_cast<int>(cols);  // <= 1024
            const int per_lane = (c + kWarpSize - 1) / kWarpSize;
            if (per_lane <= 1) launch_warp<1>(rows, c, x, y, stream);
            else if (per_lane <= 2) launch_warp<2>(rows, c, x, y, stream);
            else if (per_lane <= 4) launch_warp<4>(rows, c, x, y, stream);
            else if (per_lane <= 8) launch_warp<8>(rows, c, x, y, stream);
            else if (per_lane <= 16) launch_warp<16>(rows, c, x, y, stream);
            else launch_warp<32>(rows, c, x, y, stream);
            KL_CHECK_LAUNCH();
            return KL_OK;
        }
        default:
            return detail::fail(KL_ERR_INTERNAL, "launch_softmax: unresolved implementation");
    }
}

}  // namespace kl::cuda
