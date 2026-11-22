// FP32 matrix multiplication C = A B (row-major): a naive baseline, a
// shared-memory tiled kernel and a register-blocked kernel. All three compute
// every output with the same operation sequence (acc = 0, then
// acc = fmaf(A[i][p], B[p][j], acc) for p = 0..K-1), so their results are
// bitwise identical. Design and the error bound: docs/gemm.md.
#include <cstdint>

#include "common/cuda_check.hpp"
#include "cuda/kernels.hpp"
#include "cuda/launch_checks.hpp"

namespace kl::cuda {

namespace {

// ---- NAIVE baseline: one thread per output, everything from global memory ----

constexpr int kNaiveBlock = 16;

__global__ void __launch_bounds__(kNaiveBlock * kNaiveBlock)
    gemm_naive_kernel(int m, int n, int k, const float* __restrict__ a, const float* __restrict__ b,
                      float* __restrict__ c) {
    const int col = blockIdx.x * kNaiveBlock + threadIdx.x;
    const int row = blockIdx.y * kNaiveBlock + threadIdx.y;
    if (row >= m || col >= n) return;
    const float* ar = a + static_cast<std::int64_t>(row) * k;
    float acc = 0.0f;
    for (int p = 0; p < k; ++p) acc = fmaf(ar[p], b[static_cast<std::int64_t>(p) * n + col], acc);
    c[static_cast<std::int64_t>(row) * n + col] = acc;
}

// Benchmark control only (launch_gemm_naive_l2only): the naive kernel with
// loads that bypass L1 (__ldcg, cached in L2 only). Same arithmetic; if the
// naive kernel's reuse is served by L1, this copy is slower (docs/gemm.md).
__global__ void __launch_bounds__(kNaiveBlock * kNaiveBlock)
    gemm_naive_l2only_kernel(int m, int n, int k, const float* __restrict__ a, const float* __restrict__ b,
                             float* __restrict__ c) {
    const int col = blockIdx.x * kNaiveBlock + threadIdx.x;
    const int row = blockIdx.y * kNaiveBlock + threadIdx.y;
    if (row >= m || col >= n) return;
    const float* ar = a + static_cast<std::int64_t>(row) * k;
    float acc = 0.0f;
    for (int p = 0; p < k; ++p) acc = fmaf(__ldcg(ar + p), __ldcg(b + static_cast<std::int64_t>(p) * n + col), acc);
    c[static_cast<std::int64_t>(row) * n + col] = acc;
}

// ---- TILED: 32 x 32 output tile per block, A and B tiles in shared memory ----
//
// Thread (ty, tx) owns output (row0 + ty, col0 + tx). Each K step loads one
// element of each tile per thread (coalesced: tx runs along a row of A's tile
// and of B's tile), then accumulates over the tile. Out-of-range tile
// elements are loaded as 0 but never used: the inner loop stops at K, so no
// fmaf(0, 0, acc) is added (which could turn a -0 sum into +0) and the
// operation sequence equals the naive kernel's. In the inner loop a warp
// (fixed ty) reads one word of as (broadcast) and 32 consecutive words of bs.

constexpr int kTile = 32;

__global__ void __launch_bounds__(kTile * kTile)
    gemm_tiled_kernel(int m, int n, int k, const float* __restrict__ a, const float* __restrict__ b,
                      float* __restrict__ c) {
    __shared__ float as[kTile][kTile];
    __shared__ float bs[kTile][kTile];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * kTile + ty;
    const int col = blockIdx.x * kTile + tx;
    float acc = 0.0f;
    for (int t = 0; t < k; t += kTile) {
        const int pa = t + tx, pb = t + ty;
        as[ty][tx] = (row < m && pa < k) ? a[static_cast<std::int64_t>(row) * k + pa] : 0.0f;
        bs[ty][tx] = (pb < k && col < n) ? b[static_cast<std::int64_t>(pb) * n + col] : 0.0f;
        __syncthreads();  // tiles complete before anyone reads them
        const int pmax = min(kTile, k - t);
        for (int p = 0; p < pmax; ++p) acc = fmaf(as[ty][p], bs[p][tx], acc);
        __syncthreads();  // everyone is done with the tiles before the next load
    }
    if (row < m && col < n) c[static_cast<std::int64_t>(row) * n + col] = acc;
}

// ---- REGBLOCK: 64 x 64 output tile per block, 4 x 4 outputs per thread ----
//
// 256 threads (16 x 16). Thread (ty, tx) owns rows row0 + ty + 16 i and
// columns col0 + tx + 16 j (i, j = 0..3). Per K step of 16, each thread loads
// 4 elements of A's 64 x 16 tile and 4 of B's 16 x 64 tile; A's tile is
// stored transposed (as[p][r]) so the inner loop reads A values for a fixed
// p along r. Inner loop: 4 A values + 4 B values from shared memory feed 16
// FMAs (registers). A warp (tx = 0..15, two values of ty) reads 2 distinct
// as words (broadcast) and 16 consecutive bs words. as rows are padded to
// 65 words so that the transposing store (p varies fastest) is at most
// 2-way bank-conflicted. Same edge rule as TILED: the inner loop stops at K.

constexpr int kRbM = 64, kRbN = 64, kRbK = 16, kRbThreads = 256;

__global__ void __launch_bounds__(kRbThreads)
    gemm_regblock_kernel(int m, int n, int k, const float* __restrict__ a, const float* __restrict__ b,
                         float* __restrict__ c) {
    __shared__ float as[kRbK][kRbM + 1];
    __shared__ float bs[kRbK][kRbN];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * 16 + tx;
    const int row0 = blockIdx.y * kRbM, col0 = blockIdx.x * kRbN;
    float acc[4][4] = {};
    for (int t = 0; t < k; t += kRbK) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            // A tile: element e -> (r = e / 16, p = e % 16): 16 consecutive
            // threads read 16 consecutive floats of one row of A.
            const int e = tid + kRbThreads * q;
            const int r = e >> 4, p = e & 15;
            const int gr = row0 + r, gp = t + p;
            as[p][r] = (gr < m && gp < k) ? a[static_cast<std::int64_t>(gr) * k + gp] : 0.0f;
            // B tile: element e -> (p = e / 64, cc = e % 64): coalesced rows of B.
            const int pb = e >> 6, cc = e & 63;
            const int gpb = t + pb, gc = col0 + cc;
            bs[pb][cc] = (gpb < k && gc < n) ? b[static_cast<std::int64_t>(gpb) * n + gc] : 0.0f;
        }
        __syncthreads();  // tiles complete before anyone reads them
        const int pmax = min(kRbK, k - t);
        for (int p = 0; p < pmax; ++p) {
            float av[4], bv[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) av[i] = as[p][ty + 16 * i];
#pragma unroll
            for (int j = 0; j < 4; ++j) bv[j] = bs[p][tx + 16 * j];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
            }
        }
        __syncthreads();  // everyone is done with the tiles before the next load
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int r = row0 + ty + 16 * i;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int cc = col0 + tx + 16 * j;
            if (r < m && cc < n) c[static_cast<std::int64_t>(r) * n + cc] = acc[i][j];
        }
    }
}

unsigned blocks_for(int extent, int tile) { return static_cast<unsigned>((extent + tile - 1) / tile); }

}  // namespace

kl_status gemm_check(std::int64_t m, std::int64_t n, std::int64_t k, kl_gemm_impl impl) {
    if (m < 0 || n < 0 || k < 0) {
        return detail::fail(KL_ERR_INVALID_ARGUMENT, "gemm: M, N and K must be >= 0");
    }
    if (impl != KL_GEMM_AUTO && impl != KL_GEMM_NAIVE && impl != KL_GEMM_TILED && impl != KL_GEMM_REGBLOCK) {
        return detail::fail(KL_ERR_INVALID_ARGUMENT, "gemm: unknown implementation id");
    }
    if (m > KL_GEMM_MAX_DIM || n > KL_GEMM_MAX_DIM || k > KL_GEMM_MAX_DIM || m * k > KL_GEMM_MAX_ELEMENTS ||
        k * n > KL_GEMM_MAX_ELEMENTS || m * n > KL_GEMM_MAX_ELEMENTS) {  // each <= 2^14: no overflow
        return detail::fail(KL_ERR_UNSUPPORTED, "gemm: shape outside the supported domain (docs/gemm.md)");
    }
    return KL_OK;
}

kl_gemm_impl gemm_resolve(kl_gemm_impl impl) { return impl == KL_GEMM_AUTO ? KL_GEMM_REGBLOCK : impl; }

kl_status launch_gemm(std::int64_t m, std::int64_t n, std::int64_t k, const float* a, const float* b, float* c,
                      kl_gemm_impl impl, cudaStream_t stream) {
    KL_TRY(gemm_check(m, n, k, impl));
    if (m == 0 || n == 0) return KL_OK;
    if (c == nullptr || (k > 0 && (a == nullptr || b == nullptr))) {
        return detail::fail(KL_ERR_INVALID_ARGUMENT, "launch_gemm: null device pointer");
    }
    {
        using checks::float_aligned;
        using checks::overlap;
        if (!float_aligned(a) || !float_aligned(b) || !float_aligned(c)) {
            return detail::fail(KL_ERR_INVALID_ARGUMENT, "launch_gemm: pointers must be 4-byte aligned");
        }
        const std::size_t na = static_cast<std::size_t>(m * k) * sizeof(float);
        const std::size_t nb = static_cast<std::size_t>(k * n) * sizeof(float);
        const std::size_t nc = static_cast<std::size_t>(m * n) * sizeof(float);
        if (overlap(c, nc, a, na) || overlap(c, nc, b, nb)) {
            return detail::fail(KL_ERR_INVALID_ARGUMENT, "launch_gemm: C overlaps A or B");
        }
    }
    const int mi = static_cast<int>(m), ni = static_cast<int>(n), ki = static_cast<int>(k);
    (void)cudaGetLastError();  // see launch_saxpy: attribute errors to this launch only
    switch (gemm_resolve(impl)) {
        case KL_GEMM_NAIVE:
            gemm_naive_kernel<<<dim3(blocks_for(ni, kNaiveBlock), blocks_for(mi, kNaiveBlock)),
                                dim3(kNaiveBlock, kNaiveBlock), 0, stream>>>(mi, ni, ki, a, b, c);
            break;
        case KL_GEMM_TILED:
            gemm_tiled_kernel<<<dim3(blocks_for(ni, kTile), blocks_for(mi, kTile)), dim3(kTile, kTile), 0, stream>>>(
                mi, ni, ki, a, b, c);
            break;
        case KL_GEMM_REGBLOCK:
            gemm_regblock_kernel<<<dim3(blocks_for(ni, kRbN), blocks_for(mi, kRbM)), dim3(16, 16), 0, stream>>>(
                mi, ni, ki, a, b, c);
            break;
        default:
            return detail::fail(KL_ERR_INTERNAL, "launch_gemm: unresolved implementation");
    }
    KL_CHECK_LAUNCH();
    return KL_OK;
}

kl_status launch_gemm_naive_l2only(std::int64_t m, std::int64_t n, std::int64_t k, const float* a, const float* b,
                                   float* c, cudaStream_t stream) {
    // Same validation as the public path (shape, pointers, overlap).
    KL_TRY(gemm_check(m, n, k, KL_GEMM_NAIVE));
    if (m == 0 || n == 0) return KL_OK;
    if (c == nullptr || (k > 0 && (a == nullptr || b == nullptr)) || !checks::float_aligned(a) ||
        !checks::float_aligned(b) || !checks::float_aligned(c) ||
        checks::overlap(c, static_cast<std::size_t>(m * n) * 4, a, static_cast<std::size_t>(m * k) * 4) ||
        checks::overlap(c, static_cast<std::size_t>(m * n) * 4, b, static_cast<std::size_t>(k * n) * 4)) {
        return detail::fail(KL_ERR_INVALID_ARGUMENT, "launch_gemm_naive_l2only: bad pointers");
    }
    const int mi = static_cast<int>(m), ni = static_cast<int>(n), ki = static_cast<int>(k);
    (void)cudaGetLastError();
    gemm_naive_l2only_kernel<<<dim3(blocks_for(ni, kNaiveBlock), blocks_for(mi, kNaiveBlock)),
                               dim3(kNaiveBlock, kNaiveBlock), 0, stream>>>(mi, ni, ki, a, b, c);
    KL_CHECK_LAUNCH();
    return KL_OK;
}

}  // namespace kl::cuda
