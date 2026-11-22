// Host-side launchers for KernelLens kernels. Launchers take device pointers,
// enqueue work on `stream`, check the launch, and do NOT synchronize.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "kernellens/kernellens.h"

namespace kl::cuda {

// y[i] = fmaf(a, x[i], y[i]) for 0 <= i < n.
// Preconditions: n >= 0; x and y are device pointers to n floats each and do
// NOT overlap (the kernel uses __restrict__). block_size is passed straight
// to the launch so tests can exercise the launch-error path; production
// callers use kSaxpyBlockSize.
inline constexpr int kSaxpyBlockSize = 256;
kl_status launch_saxpy(std::int64_t n, float a, const float* x, float* y, int block_size,
                       cudaStream_t stream);

// ---- Softmax (docs/softmax.md) ----

// Validates shape and implementation id; KL_OK also for the empty case.
kl_status softmax_check(std::int64_t rows, std::int64_t cols, kl_softmax_impl impl);

// Maps KL_SOFTMAX_AUTO to the concrete implementation; others unchanged:
// FUSED_WARP if cols <= 1024; else FUSED_ONLINE if rows >= 64; else MULTIPASS.
inline constexpr std::int64_t kSoftmaxOnlineMinRows = 64;
kl_softmax_impl softmax_resolve(std::int64_t rows, std::int64_t cols, kl_softmax_impl impl);

// Device workspace (in floats) a resolved implementation needs: 2 * rows for
// MULTIPASS (row max and row sum), 0 otherwise.
std::size_t softmax_workspace_floats(std::int64_t rows, kl_softmax_impl resolved);

// Enqueues softmax on `stream`; checks every launch; does not synchronize.
// Preconditions: x, y device pointers to rows*cols floats that do NOT
// overlap (kernels use __restrict__); workspace as above.
kl_status launch_softmax(std::int64_t rows, std::int64_t cols, const float* x, float* y,
                         kl_softmax_impl impl, float* workspace, cudaStream_t stream);

// ---- Layer normalization (docs/layernorm.md) ----

// Validates shape, eps and implementation id; KL_OK also for the empty case.
kl_status layernorm_check(std::int64_t rows, std::int64_t cols, float eps, kl_layernorm_impl impl);

// AUTO -> FUSED_WARP if cols <= 1024, FUSED_SMEM if cols <= 8192, else
// FUSED_BLOCK; others unchanged.
kl_layernorm_impl layernorm_resolve(std::int64_t cols, kl_layernorm_impl impl);

// Workspace in floats: 2 * rows for MULTIPASS (mean, 1/std), 0 otherwise.
std::size_t layernorm_workspace_floats(std::int64_t rows, kl_layernorm_impl resolved);

// Enqueues layer normalization on `stream`; checks every launch; does not
// synchronize. gamma/beta both nullptr (no affine) or both device arrays of
// cols floats. Refuses misaligned pointers and y/workspace overlapping
// anything (kernels use __restrict__).
kl_status launch_layernorm(std::int64_t rows, std::int64_t cols, const float* x, const float* gamma,
                           const float* beta, float eps, float* y, kl_layernorm_impl impl,
                           float* workspace, cudaStream_t stream);

// INTERNAL, study only: single-pass var = E[x^2] - E[x]^2 (numerically
// unstable for rows whose |mean| >> std; may return NaN rows). Not exposed
// in the public API; used by tests and benchmarks to measure the
// instability.
kl_status launch_layernorm_naive(std::int64_t rows, std::int64_t cols, const float* x, const float* gamma,
                                 const float* beta, float eps, float* y, cudaStream_t stream);

// INTERNAL, benchmark controls only: FUSED_BLOCK (a separate copy of its
// kernel) or FUSED_SMEM launched with `extra_smem_bytes` of unused dynamic
// shared memory, to lower the blocks per SM for occupancy-matched
// comparisons (docs/layernorm.md). Same arithmetic as the public kernels.
kl_status launch_layernorm_padded(std::int64_t rows, std::int64_t cols, const float* x, const float* gamma,
                                  const float* beta, float eps, float* y, kl_layernorm_impl impl,
                                  std::size_t extra_smem_bytes, cudaStream_t stream);

// INTERNAL: resident blocks per SM as reported by
// cudaOccupancyMaxActiveBlocksPerMultiprocessor (after the carveout
// preference is set) for a configuration a launcher accepts: FUSED_BLOCK
// (control_copy: the copy launch_layernorm_padded runs), FUSED_SMEM or
// FUSED_WARP at `cols`, with `extra_smem_bytes` as for launch_layernorm_padded.
kl_status layernorm_blocks_per_sm(std::int64_t cols, kl_layernorm_impl impl, std::size_t extra_smem_bytes,
                                  bool control_copy, int* out);


// ---- Matrix multiplication (docs/gemm.md) ----

// Validates shape and implementation id; KL_OK also for empty shapes.
kl_status gemm_check(std::int64_t m, std::int64_t n, std::int64_t k, kl_gemm_impl impl);

// AUTO -> REGBLOCK; others unchanged.
kl_gemm_impl gemm_resolve(kl_gemm_impl impl);

// Enqueues C = A B on `stream`; checks the launch; does not synchronize.
// Refuses misaligned pointers and C overlapping A or B.
kl_status launch_gemm(std::int64_t m, std::int64_t n, std::int64_t k, const float* a, const float* b, float* c,
                      kl_gemm_impl impl, cudaStream_t stream);

// INTERNAL, benchmark control only: the naive kernel with L1-bypassing loads
// (__ldcg). Same arithmetic as KL_GEMM_NAIVE (bitwise identical results).
kl_status launch_gemm_naive_l2only(std::int64_t m, std::int64_t n, std::int64_t k, const float* a, const float* b,
                                   float* c, cudaStream_t stream);

}  // namespace kl::cuda
