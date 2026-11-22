// Warp- and block-level reductions shared by KernelLens kernels.
//
// Conventions: blocks that call block_reduce* have exactly kReduceBlock
// threads (a multiple of 32), every thread of the block calls the function
// (it contains __syncthreads), and every thread receives the result.
#pragma once

#include <math_constants.h>

namespace kl::cuda {

inline constexpr int kWarpSize = 32;
inline constexpr int kReduceBlock = 256;  // 8 warps
inline constexpr unsigned kFullMask = 0xffffffffu;

struct MaxOp {
    __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
};
struct SumOp {
    __device__ float operator()(float a, float b) const { return a + b; }
};

// Butterfly reduction: after 5 levels every lane holds the warp result.
// Requires all 32 lanes to be active (full mask).
template <typename Op>
__device__ __forceinline__ float warp_reduce(float v, Op op) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        v = op(v, __shfl_xor_sync(kFullMask, v, offset));
    }
    return v;
}

template <typename Op>
__device__ inline float block_reduce(float v, Op op, float identity) {
    __shared__ float partial[kReduceBlock / kWarpSize];
    __shared__ float result;
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    v = warp_reduce(v, op);
    if (lane == 0) partial[warp] = v;
    __syncthreads();  // all warp partials written
    if (warp == 0) {
        float w = lane < kReduceBlock / kWarpSize ? partial[lane] : identity;
        w = warp_reduce(w, op);
        if (lane == 0) result = w;
    }
    __syncthreads();  // result written
    const float r = result;
    __syncthreads();  // everyone has read result before a later call reuses it
    return r;
}

// ---------------------------------------------------------------------------
// Online softmax state: running maximum m and sum s of exp(x - m).

struct MaxSum {
    float m;
    float s;
};

__device__ __forceinline__ MaxSum max_sum_identity() { return {-CUDART_INF_F, 0.0f}; }

// Adds one element. Special values follow the reference definition
// softmax(x) = exp(x - max) / sum(exp(x - max)):
//  * x == -inf contributes exactly 0 (skipping it avoids exp(-inf - -inf) =
//    NaN while the running maximum is still -inf);
//  * x == +inf makes the sum NaN (the reference computes inf - inf);
//  * NaN propagates through expf.
__device__ __forceinline__ void max_sum_add(MaxSum& a, float x) {
    if (x > a.m) {
        a.s = a.s * expf(a.m - x) + (isinf(x) ? CUDART_NAN_F : 1.0f);
        a.m = x;
    } else if (x != -CUDART_INF_F) {
        a.s += expf(x - a.m);
    }
}

// Combines two partial states. When a side already has the maximum, its
// factor is exactly 1 (this also covers m == -inf on both sides: s stays 0).
__device__ __forceinline__ MaxSum max_sum_combine(MaxSum a, MaxSum b) {
    const float m = fmaxf(a.m, b.m);
    const float ra = (a.m == m) ? 1.0f : expf(a.m - m);
    const float rb = (b.m == m) ? 1.0f : expf(b.m - m);
    return {m, a.s * ra + b.s * rb};
}

__device__ __forceinline__ MaxSum warp_reduce_max_sum(MaxSum v) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        const MaxSum other{__shfl_xor_sync(kFullMask, v.m, offset),
                           __shfl_xor_sync(kFullMask, v.s, offset)};
        v = max_sum_combine(v, other);
    }
    return v;
}

__device__ inline MaxSum block_reduce_max_sum(MaxSum v) {
    __shared__ float partial_m[kReduceBlock / kWarpSize];
    __shared__ float partial_s[kReduceBlock / kWarpSize];
    __shared__ MaxSum result;
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    v = warp_reduce_max_sum(v);
    if (lane == 0) {
        partial_m[warp] = v.m;
        partial_s[warp] = v.s;
    }
    __syncthreads();
    if (warp == 0) {
        MaxSum w = lane < kReduceBlock / kWarpSize ? MaxSum{partial_m[lane], partial_s[lane]}
                                                   : max_sum_identity();
        w = warp_reduce_max_sum(w);
        if (lane == 0) result = w;
    }
    __syncthreads();
    const MaxSum r = result;
    __syncthreads();
    return r;
}

}  // namespace kl::cuda
