// Row-wise FP32 layer normalization: a 3-kernel baseline, a fused
// block-per-row kernel (three reads of the row), the same with the row
// cached in shared memory (one read), and a fused warp-per-row kernel, all
// with the two-pass variance; plus an internal naive E[x^2] - E[x]^2 kernel
// kept only to measure its instability. Design and the error bound:
// docs/layernorm.md.
#include <atomic>
#include <cmath>
#include <cstdint>

#include "common/cuda_check.hpp"
#include "cuda/kernels.hpp"
#include "cuda/launch_checks.hpp"
#include "cuda/reduce.cuh"

namespace kl::cuda {

namespace {

constexpr int kWarpRowsPerBlock = 4;  // FUSED_WARP: one row per warp, 4 warps per block

// y = t * gamma + beta with one rounding (fmaf), or y = t without affine.
// `affine` is uniform across the grid, so the branch never diverges.
__device__ __forceinline__ float affine_out(float t, const float* __restrict__ gamma,
                                            const float* __restrict__ beta, std::int64_t j) {
    return gamma != nullptr ? fmaf(t, gamma[j], beta[j]) : t;
}

// 1 / sqrt(var + eps) with correctly rounded sqrtf and division (not rsqrtf,
// which has 2 ulp error) - the bound in docs/layernorm.md assumes this.
__device__ __forceinline__ float inv_std(float var, float eps) { return 1.0f / sqrtf(var + eps); }

// Pivot-shifted mean (docs/layernorm.md): every kernel sums x_j - x_0 for its
// row and adds the pivot x_0 back. A constant row then sums exact zeros, so
// its mean is exact (y == beta bit for bit, NaN for eps = 0, as PyTorch), and
// the mean's error scales with the row's spread around x_0, not with |x|.
__device__ __forceinline__ float shifted_mean(float pivot, float shifted_sum, std::int64_t cols) {
    return pivot + shifted_sum / static_cast<float>(cols);
}

// Corrected two-pass mean (docs/layernorm.md): the variance pass also sums
// the deviations d_j = x_j - m1 from the pivot-shifted mean m1; in exact
// arithmetic sum(d) / n = mu - m1, so m1 + sum(d) / n removes m1's error to
// first order, whichever element is the pivot (an outlier pivot inflates
// m1's error, not this one's). Constant rows stay exact (every d is 0). The
// output uses this mean; the variance keeps sum(d^2) around m1, whose error
// enters only as (mu - m1)^2.
__device__ __forceinline__ float corrected_mean(float m1, float deviation_sum, std::int64_t cols) {
    return m1 + deviation_sum / static_cast<float>(cols);
}

// ---- MULTIPASS baseline: three kernels, stats in global memory ----

__global__ void __launch_bounds__(kReduceBlock)
    ln_mean_kernel(std::int64_t cols, const float* __restrict__ x, float* __restrict__ mean) {
    const float* row = x + static_cast<std::int64_t>(blockIdx.x) * cols;
    const float pivot = row[0];
    float s = 0.0f;
    for (std::int64_t j = threadIdx.x; j < cols; j += kReduceBlock) s += row[j] - pivot;
    s = block_reduce(s, SumOp{}, 0.0f);
    if (threadIdx.x == 0) mean[blockIdx.x] = shifted_mean(pivot, s, cols);
}

// Variance pass. Also sums the deviations d = x - m1 and replaces the stored
// mean by the corrected mean m1 + sum(d) / n (see corrected_mean). Every
// thread reads mean[row] before the reductions' barriers; thread 0 writes it
// after them, so no thread can read the corrected value here.
__global__ void __launch_bounds__(kReduceBlock)
    ln_rstd_kernel(std::int64_t cols, float eps, const float* __restrict__ x,
                   float* __restrict__ mean, float* __restrict__ rstd) {
    const float* row = x + static_cast<std::int64_t>(blockIdx.x) * cols;
    const float m = mean[blockIdx.x];
    float q = 0.0f, sd = 0.0f;
    for (std::int64_t j = threadIdx.x; j < cols; j += kReduceBlock) {
        const float d = row[j] - m;
        q += d * d;
        sd += d;
    }
    q = block_reduce(q, SumOp{}, 0.0f);
    sd = block_reduce(sd, SumOp{}, 0.0f);
    if (threadIdx.x == 0) {
        rstd[blockIdx.x] = inv_std(q / static_cast<float>(cols), eps);
        mean[blockIdx.x] = corrected_mean(m, sd, cols);
    }
}

__global__ void __launch_bounds__(kReduceBlock)
    ln_normalize_kernel(std::int64_t cols, const float* __restrict__ x, const float* __restrict__ mean,
                        const float* __restrict__ rstd, const float* __restrict__ gamma,
                        const float* __restrict__ beta, float* __restrict__ y) {
    const std::int64_t offset = static_cast<std::int64_t>(blockIdx.x) * cols;
    const float m = mean[blockIdx.x];
    const float r = rstd[blockIdx.x];
    for (std::int64_t j = threadIdx.x; j < cols; j += kReduceBlock) {
        y[offset + j] = affine_out((x[offset + j] - m) * r, gamma, beta, j);
    }
}

// ---- FUSED_BLOCK: one kernel, block per row, three passes over the row ----
//
// kInstance 0 is the public kernel. kInstance 1 is an identical copy used
// only by the benchmark's occupancy-matched control (launch_layernorm_padded):
// a separate function, so its shared-memory carveout preference can be set
// without changing the public kernel's L1/shared-memory split.

template <int kInstance>
__global__ void __launch_bounds__(kReduceBlock)
    ln_fused_block_kernel(std::int64_t cols, float eps, const float* __restrict__ x,
                          const float* __restrict__ gamma, const float* __restrict__ beta,
                          float* __restrict__ y) {
    const std::int64_t offset = static_cast<std::int64_t>(blockIdx.x) * cols;
    const float* row = x + offset;
    const float pivot = row[0];
    float s = 0.0f;
    for (std::int64_t j = threadIdx.x; j < cols; j += kReduceBlock) s += row[j] - pivot;
    const float m1 = shifted_mean(pivot, block_reduce(s, SumOp{}, 0.0f), cols);
    float q = 0.0f, sd = 0.0f;
    for (std::int64_t j = threadIdx.x; j < cols; j += kReduceBlock) {
        const float d = row[j] - m1;
        q += d * d;
        sd += d;
    }
    const float r = inv_std(block_reduce(q, SumOp{}, 0.0f) / static_cast<float>(cols), eps);
    const float m = corrected_mean(m1, block_reduce(sd, SumOp{}, 0.0f), cols);
    for (std::int64_t j = threadIdx.x; j < cols; j += kReduceBlock) {
        y[offset + j] = affine_out((row[j] - m) * r, gamma, beta, j);
    }
}

// ---- FUSED_SMEM: FUSED_BLOCK with the row cached in shared memory ----
//
// Same thread mapping and the same arithmetic in the same order as
// ln_fused_block_kernel (bitwise-identical results, same error bound), but x
// is read from global memory once: the mean pass also stores each element
// into dynamic shared memory (cols floats, cols <= 8192), and the variance
// and output passes read it back. Each thread reads back only the elements
// it stored itself (j = threadIdx.x + i * kReduceBlock), so program order is
// enough between the store and the loads; no extra barrier. Thread t
// accesses word t + 256 i: consecutive lanes, consecutive banks.

__global__ void __launch_bounds__(kReduceBlock)
    ln_fused_smem_kernel(std::int64_t cols, float eps, const float* __restrict__ x,
                         const float* __restrict__ gamma, const float* __restrict__ beta,
                         float* __restrict__ y) {
    extern __shared__ float row_s[];
    const std::int64_t offset = static_cast<std::int64_t>(blockIdx.x) * cols;
    const float* row = x + offset;
    const float pivot = row[0];
    float s = 0.0f;
    for (std::int64_t j = threadIdx.x; j < cols; j += kReduceBlock) {
        const float v = row[j];
        row_s[j] = v;
        s += v - pivot;
    }
    const float m1 = shifted_mean(pivot, block_reduce(s, SumOp{}, 0.0f), cols);
    float q = 0.0f, sd = 0.0f;
    for (std::int64_t j = threadIdx.x; j < cols; j += kReduceBlock) {
        const float d = row_s[j] - m1;
        q += d * d;
        sd += d;
    }
    const float r = inv_std(block_reduce(q, SumOp{}, 0.0f) / static_cast<float>(cols), eps);
    const float m = corrected_mean(m1, block_reduce(sd, SumOp{}, 0.0f), cols);
    for (std::int64_t j = threadIdx.x; j < cols; j += kReduceBlock) {
        y[offset + j] = affine_out((row_s[j] - m) * r, gamma, beta, j);
    }
}

// ---- FUSED_WARP: one warp per row, row in registers, one read ----
//
// Lane l holds elements l, l + 32, ... (coalesced) in registers; the
// deviations from m1 are formed on the fly for the variance and the mean
// correction, the output uses x - m with the corrected mean m.

template <int VPL>
__global__ void __launch_bounds__(kWarpRowsPerBlock * kWarpSize)
    ln_warp_kernel(std::int64_t rows, int cols, float eps, const float* __restrict__ x,
                   const float* __restrict__ gamma, const float* __restrict__ beta,
                   float* __restrict__ y) {
    const int lane = threadIdx.x % kWarpSize;
    const std::int64_t r =
        static_cast<std::int64_t>(blockIdx.x) * kWarpRowsPerBlock + threadIdx.x / kWarpSize;
    // r is warp-uniform: whole warps leave together, so the full-mask
    // shuffles below always see 32 active lanes. No __syncthreads here.
    if (r >= rows) return;
    const float* row = x + r * cols;
    float* out = y + r * cols;

    const float pivot = row[0];  // same address for all lanes: one broadcast load
    float v[VPL];
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < VPL; ++i) {
        const int j = lane + i * kWarpSize;
        v[i] = j < cols ? row[j] : 0.0f;
        s += j < cols ? v[i] - pivot : 0.0f;  // padding contributes exactly 0 to the sum
    }
    const float m1 = shifted_mean(pivot, warp_reduce(s, SumOp{}), cols);
    float q = 0.0f, sd = 0.0f;
#pragma unroll
    for (int i = 0; i < VPL; ++i) {
        const int j = lane + i * kWarpSize;
        const float d = j < cols ? v[i] - m1 : 0.0f;  // padding: deviation 0, adds exactly 0
        q += d * d;
        sd += d;
    }
    const float rs = inv_std(warp_reduce(q, SumOp{}) / static_cast<float>(cols), eps);
    const float m = corrected_mean(m1, warp_reduce(sd, SumOp{}), cols);
#pragma unroll
    for (int i = 0; i < VPL; ++i) {
        const int j = lane + i * kWarpSize;
        if (j < cols) out[j] = affine_out((v[i] - m) * rs, gamma, beta, j);
    }
}

template <int VPL>
void launch_warp(std::int64_t rows, int cols, float eps, const float* x, const float* gamma,
                 const float* beta, float* y, cudaStream_t stream) {
    const auto blocks = static_cast<unsigned>((rows + kWarpRowsPerBlock - 1) / kWarpRowsPerBlock);
    ln_warp_kernel<VPL><<<blocks, kWarpRowsPerBlock * kWarpSize, 0, stream>>>(rows, cols, eps, x, gamma,
                                                                              beta, y);
}

// ---- Internal naive single-pass kernel (study only, NOT in the public API) ----
//
// var = E[x^2] - E[x]^2 from one read: the textbook formula. Its variance
// error grows with (mean/std)^2 and var can come out negative (then the
// output is NaN). Used by tests and benchmarks to measure exactly that.

__global__ void __launch_bounds__(kReduceBlock)
    ln_naive_kernel(std::int64_t cols, float eps, const float* __restrict__ x,
                    const float* __restrict__ gamma, const float* __restrict__ beta,
                    float* __restrict__ y) {
    const std::int64_t offset = static_cast<std::int64_t>(blockIdx.x) * cols;
    const float* row = x + offset;
    float s1 = 0.0f, s2 = 0.0f;
    for (std::int64_t j = threadIdx.x; j < cols; j += kReduceBlock) {
        const float v = row[j];
        s1 += v;
        s2 += v * v;
    }
    const float n = static_cast<float>(cols);
    const float m = block_reduce(s1, SumOp{}, 0.0f) / n;
    const float ex2 = block_reduce(s2, SumOp{}, 0.0f) / n;
    const float r = inv_std(ex2 - m * m, eps);
    for (std::int64_t j = threadIdx.x; j < cols; j += kReduceBlock) {
        y[offset + j] = affine_out((row[j] - m) * r, gamma, beta, j);
    }
}

// Pointer checks common to the launchers: alignment, and no overlap between
// what is written (y, workspace) and anything else.
kl_status check_pointers(std::int64_t rows, std::int64_t cols, const float* x, const float* gamma,
                         const float* beta, const float* y, const float* workspace,
                         std::size_t workspace_floats) {
    using checks::float_aligned;
    using checks::overlap;
    if (x == nullptr || y == nullptr) {
        return detail::fail(KL_ERR_INVALID_ARGUMENT, "layernorm: null x or y");
    }
    if ((gamma == nullptr) != (beta == nullptr)) {
        return detail::fail(KL_ERR_INVALID_ARGUMENT, "layernorm: gamma and beta must both be given or both be NULL");
    }
    if (!float_aligned(x) || !float_aligned(y) || !float_aligned(gamma) || !float_aligned(beta) ||
        !float_aligned(workspace)) {
        return detail::fail(KL_ERR_INVALID_ARGUMENT, "layernorm: pointers must be 4-byte aligned");
    }
    const std::size_t n = static_cast<std::size_t>(rows * cols) * sizeof(float);
    const std::size_t c = static_cast<std::size_t>(cols) * sizeof(float);
    const std::size_t w = workspace_floats * sizeof(float);
    if (overlap(y, n, x, n) || overlap(y, n, gamma, c) || overlap(y, n, beta, c) ||
        overlap(y, n, workspace, w)) {
        return detail::fail(KL_ERR_INVALID_ARGUMENT, "layernorm: y overlaps an input or the workspace");
    }
    if (overlap(workspace, w, x, n) || overlap(workspace, w, gamma, c) || overlap(workspace, w, beta, c)) {
        return detail::fail(KL_ERR_INVALID_ARGUMENT, "layernorm: workspace overlaps an input");
    }
    return KL_OK;
}

// FUSED_SMEM's blocks per SM depend on the L1/shared-memory split the driver
// picks for the kernel. Ask for the largest shared-memory carveout (a
// per-function preference): with it, 5 blocks fit per SM at 4096 cols and 3
// at 8192 (docs/layernorm.md). Also set for the benchmark-control copy of
// the block kernel, never for the public FUSED_BLOCK kernel. Set once per
// process; retried if it failed. Two first callers may both set it (the
// call is idempotent). A preference only: it affects occupancy and speed,
// never results; after cudaDeviceReset it would not be re-applied.
kl_status prefer_max_shared_carveout() {
    static std::atomic<bool> done{false};
    if (done.load(std::memory_order_acquire)) return KL_OK;
    KL_CUDA_TRY(cudaFuncSetAttribute(ln_fused_smem_kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                     cudaSharedmemCarveoutMaxShared));
    KL_CUDA_TRY(cudaFuncSetAttribute(ln_fused_block_kernel<1>, cudaFuncAttributePreferredSharedMemoryCarveout,
                                     cudaSharedmemCarveoutMaxShared));
    done.store(true, std::memory_order_release);
    return KL_OK;
}

// Dynamic shared memory allowed for the padded benchmark controls (below the
// 48 KiB default per-block limit with room for the static reduction buffer).
constexpr std::size_t kMaxPaddedSmemBytes = 40 * 1024;

}  // namespace

kl_status layernorm_check(std::int64_t rows, std::int64_t cols, float eps, kl_layernorm_impl impl) {
    if (rows < 0 || cols < 0) {
        return detail::fail(KL_ERR_INVALID_ARGUMENT, "layernorm: rows and cols must be >= 0");
    }
    if (impl != KL_LAYERNORM_AUTO && impl != KL_LAYERNORM_MULTIPASS && impl != KL_LAYERNORM_FUSED_BLOCK &&
        impl != KL_LAYERNORM_FUSED_WARP && impl != KL_LAYERNORM_FUSED_SMEM) {
        return detail::fail(KL_ERR_INVALID_ARGUMENT, "layernorm: unknown implementation id");
    }
    if (!std::isfinite(eps) || eps < 0.0f) {
        return detail::fail(KL_ERR_INVALID_ARGUMENT, "layernorm: eps must be finite and >= 0");
    }
    if (rows == 0 || cols == 0) {
        return KL_OK;
    }
    if (cols > KL_LAYERNORM_MAX_COLS || rows > KL_LAYERNORM_MAX_ROWS ||
        rows * cols > KL_LAYERNORM_MAX_ELEMENTS) {
        return detail::fail(KL_ERR_UNSUPPORTED,
                            "layernorm: shape outside the supported domain (docs/layernorm.md)");
    }
    if (impl == KL_LAYERNORM_FUSED_WARP && cols > KL_LAYERNORM_WARP_MAX_COLS) {
        return detail::fail(KL_ERR_UNSUPPORTED, "layernorm: FUSED_WARP supports cols <= 1024");
    }
    if (impl == KL_LAYERNORM_FUSED_SMEM && cols > KL_LAYERNORM_SMEM_MAX_COLS) {
        return detail::fail(KL_ERR_UNSUPPORTED, "layernorm: FUSED_SMEM supports cols <= 8192");
    }
    return KL_OK;
}

// AUTO (docs/layernorm.md): one global read wherever possible - the warp
// kernel for short rows, the shared-memory kernel up to its 8192-column
// limit - and the three-pass block kernel beyond. Chosen from the traffic
// model after the dry run, before any measured series.
kl_layernorm_impl layernorm_resolve(std::int64_t cols, kl_layernorm_impl impl) {
    if (impl != KL_LAYERNORM_AUTO) return impl;
    if (cols <= KL_LAYERNORM_WARP_MAX_COLS) return KL_LAYERNORM_FUSED_WARP;
    return cols <= KL_LAYERNORM_SMEM_MAX_COLS ? KL_LAYERNORM_FUSED_SMEM : KL_LAYERNORM_FUSED_BLOCK;
}

std::size_t layernorm_workspace_floats(std::int64_t rows, kl_layernorm_impl resolved) {
    return resolved == KL_LAYERNORM_MULTIPASS ? 2 * static_cast<std::size_t>(rows) : 0;
}

kl_status launch_layernorm(std::int64_t rows, std::int64_t cols, const float* x, const float* gamma,
                           const float* beta, float eps, float* y, kl_layernorm_impl impl,
                           float* workspace, cudaStream_t stream) {
    KL_TRY(layernorm_check(rows, cols, eps, impl));
    if (rows == 0 || cols == 0) return KL_OK;
    const kl_layernorm_impl resolved = layernorm_resolve(cols, impl);
    const std::size_t ws_floats = layernorm_workspace_floats(rows, resolved);
    KL_TRY(check_pointers(rows, cols, x, gamma, beta, y, workspace, ws_floats));
    if (ws_floats > 0 && workspace == nullptr) {
        return detail::fail(KL_ERR_INVALID_ARGUMENT, "launch_layernorm: MULTIPASS needs a workspace");
    }
    (void)cudaGetLastError();  // see launch_saxpy: attribute errors to these launches only
    const auto grid = static_cast<unsigned>(rows);
    switch (resolved) {
        case KL_LAYERNORM_MULTIPASS: {
            float* mean = workspace;
            float* rstd = workspace + rows;
            ln_mean_kernel<<<grid, kReduceBlock, 0, stream>>>(cols, x, mean);
            KL_CHECK_LAUNCH();
            ln_rstd_kernel<<<grid, kReduceBlock, 0, stream>>>(cols, eps, x, mean, rstd);
            KL_CHECK_LAUNCH();
            ln_normalize_kernel<<<grid, kReduceBlock, 0, stream>>>(cols, x, mean, rstd, gamma, beta, y);
            KL_CHECK_LAUNCH();
            return KL_OK;
        }
        case KL_LAYERNORM_FUSED_BLOCK:
            ln_fused_block_kernel<0><<<grid, kReduceBlock, 0, stream>>>(cols, eps, x, gamma, beta, y);
            KL_CHECK_LAUNCH();
            return KL_OK;
        case KL_LAYERNORM_FUSED_SMEM: {
            // cols <= 8192 (checked): <= 32 KiB of dynamic shared memory, below
            // the 48 KiB default per-block limit, so no opt-in attribute.
            KL_TRY(prefer_max_shared_carveout());
            const std::size_t smem = static_cast<std::size_t>(cols) * sizeof(float);
            ln_fused_smem_kernel<<<grid, kReduceBlock, smem, stream>>>(cols, eps, x, gamma, beta, y);
            KL_CHECK_LAUNCH();
            return KL_OK;
        }
        case KL_LAYERNORM_FUSED_WARP: {
            const int c = static_cast<int>(cols);  // <= 1024
            const int per_lane = (c + kWarpSize - 1) / kWarpSize;
            if (per_lane <= 1) launch_warp<1>(rows, c, eps, x, gamma, beta, y, stream);
            else if (per_lane <= 2) launch_warp<2>(rows, c, eps, x, gamma, beta, y, stream);
            else if (per_lane <= 4) launch_warp<4>(rows, c, eps, x, gamma, beta, y, stream);
            else if (per_lane <= 8) launch_warp<8>(rows, c, eps, x, gamma, beta, y, stream);
            else if (per_lane <= 16) launch_warp<16>(rows, c, eps, x, gamma, beta, y, stream);
            else launch_warp<32>(rows, c, eps, x, gamma, beta, y, stream);
            KL_CHECK_LAUNCH();
            return KL_OK;
        }
        default:
            return detail::fail(KL_ERR_INTERNAL, "launch_layernorm: unresolved implementation");
    }
}

kl_status launch_layernorm_padded(std::int64_t rows, std::int64_t cols, const float* x, const float* gamma,
                                  const float* beta, float eps, float* y, kl_layernorm_impl impl,
                                  std::size_t extra_smem_bytes, cudaStream_t stream) {
    if (impl != KL_LAYERNORM_FUSED_BLOCK && impl != KL_LAYERNORM_FUSED_SMEM) {
        return detail::fail(KL_ERR_INVALID_ARGUMENT, "launch_layernorm_padded: FUSED_BLOCK or FUSED_SMEM only");
    }
    KL_TRY(layernorm_check(rows, cols, eps, impl));
    if (rows == 0 || cols == 0) return KL_OK;
    KL_TRY(check_pointers(rows, cols, x, gamma, beta, y, nullptr, 0));
    const std::size_t base = impl == KL_LAYERNORM_FUSED_SMEM ? static_cast<std::size_t>(cols) * sizeof(float) : 0;
    if (extra_smem_bytes > kMaxPaddedSmemBytes - base) {
        return detail::fail(KL_ERR_UNSUPPORTED, "launch_layernorm_padded: dynamic shared memory above 40 KiB");
    }
    KL_TRY(prefer_max_shared_carveout());
    (void)cudaGetLastError();
    const auto grid = static_cast<unsigned>(rows);
    if (impl == KL_LAYERNORM_FUSED_SMEM) {
        ln_fused_smem_kernel<<<grid, kReduceBlock, base + extra_smem_bytes, stream>>>(cols, eps, x, gamma, beta, y);
    } else {
        ln_fused_block_kernel<1><<<grid, kReduceBlock, extra_smem_bytes, stream>>>(cols, eps, x, gamma, beta, y);
    }
    KL_CHECK_LAUNCH();
    return KL_OK;
}

kl_status layernorm_blocks_per_sm(std::int64_t cols, kl_layernorm_impl impl, std::size_t extra_smem_bytes,
                                  bool control_copy, int* out) {
    if (out == nullptr || cols < 1) {
        return detail::fail(KL_ERR_INVALID_ARGUMENT, "layernorm_blocks_per_sm: bad arguments");
    }
    // Only configurations the launchers accept.
    KL_TRY(layernorm_check(1, cols, 0.0f, impl));
    const std::size_t base = impl == KL_LAYERNORM_FUSED_SMEM ? static_cast<std::size_t>(cols) * sizeof(float) : 0;
    const bool padded_impl = impl == KL_LAYERNORM_FUSED_BLOCK || impl == KL_LAYERNORM_FUSED_SMEM;
    if ((extra_smem_bytes > 0 && !padded_impl) || (control_copy && impl != KL_LAYERNORM_FUSED_BLOCK) ||
        (extra_smem_bytes > 0 && impl == KL_LAYERNORM_FUSED_BLOCK && !control_copy) ||
        extra_smem_bytes > kMaxPaddedSmemBytes - base) {
        return detail::fail(KL_ERR_UNSUPPORTED, "layernorm_blocks_per_sm: configuration no launcher accepts");
    }
    KL_TRY(prefer_max_shared_carveout());
    *out = 0;
    switch (impl) {
        case KL_LAYERNORM_FUSED_BLOCK:
            if (!control_copy) {
                KL_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(out, ln_fused_block_kernel<0>, kReduceBlock, 0));
            } else {
                KL_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(out, ln_fused_block_kernel<1>, kReduceBlock,
                                                                          extra_smem_bytes));
            }
            return KL_OK;
        case KL_LAYERNORM_FUSED_SMEM:
            KL_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                out, ln_fused_smem_kernel, kReduceBlock, static_cast<std::size_t>(cols) * sizeof(float) + extra_smem_bytes));
            return KL_OK;
        case KL_LAYERNORM_FUSED_WARP: {
            const int per_lane = static_cast<int>((cols + kWarpSize - 1) / kWarpSize);
            const int threads = kWarpRowsPerBlock * kWarpSize;
            cudaError_t e = cudaSuccess;
            if (per_lane <= 1) e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(out, ln_warp_kernel<1>, threads, 0);
            else if (per_lane <= 2) e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(out, ln_warp_kernel<2>, threads, 0);
            else if (per_lane <= 4) e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(out, ln_warp_kernel<4>, threads, 0);
            else if (per_lane <= 8) e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(out, ln_warp_kernel<8>, threads, 0);
            else if (per_lane <= 16) e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(out, ln_warp_kernel<16>, threads, 0);
            else if (per_lane <= 32) e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(out, ln_warp_kernel<32>, threads, 0);
            else return detail::fail(KL_ERR_UNSUPPORTED, "layernorm_blocks_per_sm: FUSED_WARP supports cols <= 1024");
            KL_CUDA_TRY(e);
            return KL_OK;
        }
        default:
            return detail::fail(KL_ERR_UNSUPPORTED, "layernorm_blocks_per_sm: FUSED_BLOCK, FUSED_SMEM or FUSED_WARP only");
    }
}

kl_status launch_layernorm_naive(std::int64_t rows, std::int64_t cols, const float* x, const float* gamma,
                                 const float* beta, float eps, float* y, cudaStream_t stream) {
    KL_TRY(layernorm_check(rows, cols, eps, KL_LAYERNORM_FUSED_BLOCK));
    if (rows == 0 || cols == 0) return KL_OK;
    KL_TRY(check_pointers(rows, cols, x, gamma, beta, y, nullptr, 0));
    (void)cudaGetLastError();
    ln_naive_kernel<<<static_cast<unsigned>(rows), kReduceBlock, 0, stream>>>(cols, eps, x, gamma, beta, y);
    KL_CHECK_LAUNCH();
    return KL_OK;
}

}  // namespace kl::cuda
