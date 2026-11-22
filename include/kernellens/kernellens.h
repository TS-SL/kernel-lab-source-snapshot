/*
 * KernelLens public C API.
 *
 * A plain C ABI so the same shared library can be driven from C++ tests,
 * benchmarks and Python (ctypes) without a binding generator.
 *
 * Conventions
 *  - Every fallible function returns kl_status. KL_OK means success.
 *  - On failure, kl_last_error_message() returns a human-readable detail
 *    string for the calling thread (valid until that thread's next KernelLens
 *    call).
 *  - "_host" functions take host pointers, copy to the device, run the
 *    kernel, synchronize, and copy back. They are for correctness testing,
 *    not for timing.
 *  - Only device 0 is used.
 */
#ifndef KERNELLENS_KERNELLENS_H
#define KERNELLENS_KERNELLENS_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(KL_BUILDING_DLL)
#    define KL_API __declspec(dllexport)
#  elif defined(KL_STATIC)
#    define KL_API
#  else
#    define KL_API __declspec(dllimport)
#  endif
#else
#  define KL_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum kl_status {
    KL_OK = 0,
    KL_ERR_INVALID_ARGUMENT = 1, /* null pointer, negative size, ... */
    KL_ERR_UNSUPPORTED = 2,      /* shape/layout/dtype outside the documented domain */
    KL_ERR_RESOURCE_LIMIT = 3,   /* request exceeds the project's VRAM safety cap,
                                    or a host memory allocation failed */
    KL_ERR_CUDA = 4,             /* a CUDA runtime call or kernel launch failed */
    KL_ERR_NO_DEVICE = 5,        /* no usable CUDA device */
    KL_ERR_INTERNAL = 6          /* unexpected C++ exception inside the library */
} kl_status;

typedef struct kl_device_info {
    char name[256];
    int compute_major;
    int compute_minor;
    int multiprocessor_count;
    int max_threads_per_multiprocessor;
    int kernel_exec_timeout; /* 1 if the OS watchdog (e.g. Windows TDR) applies */
    int driver_version;      /* e.g. 13030 for CUDA 13.3 */
    int runtime_version;     /* CUDA runtime the library was built against */
    size_t total_global_mem;
    size_t free_global_mem;
} kl_device_info;

/* Short constant name for a status code, e.g. "KL_ERR_CUDA". */
KL_API const char* kl_status_string(kl_status status);

/* Detail message for the most recent failure on this thread ("" if none). */
KL_API const char* kl_last_error_message(void);

/* Query device 0. */
KL_API kl_status kl_get_device_info(kl_device_info* out);

/*
 * Safety cap on device memory held by KernelLens at one time, in bytes.
 * An operation is refused (KL_ERR_RESOURCE_LIMIT), before it allocates
 * anything, if its total device memory would push the library's live total
 * above this hard cap or is larger than half of the device memory currently
 * free. With concurrent callers, a request that passes that up-front check
 * can still be refused part-way (each allocation reserves atomically); it
 * then frees what it took, and the cap is never exceeded.
 */
KL_API size_t kl_device_alloc_hard_cap(void);

/* Device bytes currently held by KernelLens allocations. */
KL_API size_t kl_device_live_bytes(void);

/*
 * Smoke-test operator: y[i] = fmaf(a, x[i], y[i]) for 0 <= i < n, FP32.
 * x and y are host arrays of n floats; y is updated in place. n == 0 is a
 * no-op. x may equal y (the device copies are separate buffers).
 */
KL_API kl_status kl_saxpy_host(int64_t n, float a, const float* x, float* y);

/*
 * Row-wise softmax over a contiguous row-major FP32 matrix x[rows][cols]:
 *   y[r][j] = exp(x[r][j] - max_r) / sum_k exp(x[r][k] - max_r)
 * Supported domain, special values (-inf/+inf/NaN) and error bounds:
 * docs/softmax.md. Shapes outside the domain return KL_ERR_UNSUPPORTED.
 */
#define KL_SOFTMAX_MAX_COLS (1 << 20)
#define KL_SOFTMAX_MAX_ROWS (1 << 20)
#define KL_SOFTMAX_MAX_ELEMENTS (1 << 26)
#define KL_SOFTMAX_WARP_MAX_COLS 1024

typedef enum kl_softmax_impl {
    KL_SOFTMAX_AUTO = 0,         /* FUSED_WARP if cols <= 1024; else FUSED_ONLINE if
                                    rows >= 64; else MULTIPASS (docs/softmax.md) */
    KL_SOFTMAX_MULTIPASS = 1,    /* baseline: 3 kernels (max, sum, normalize) */
    KL_SOFTMAX_FUSED_ONLINE = 2, /* 1 kernel, block per row, online max/sum */
    KL_SOFTMAX_FUSED_WARP = 3,   /* 1 kernel, warp per row, row in registers; cols <= 1024 */
    /* Not implementations: widen the enum's value range to all of int, so
     * callers (e.g. ctypes) passing an unknown id, negative ones included,
     * get KL_ERR_INVALID_ARGUMENT rather than undefined behaviour. */
    KL_SOFTMAX_IMPL_FORCE_INT_MIN = -0x7fffffff - 1,
    KL_SOFTMAX_IMPL_FORCE_INT = 0x7fffffff
} kl_softmax_impl;

/* Validates (rows, cols, impl) without touching the GPU: KL_OK if the call
 * would be accepted (including the empty no-op case). */
KL_API kl_status kl_softmax_supported(int64_t rows, int64_t cols, kl_softmax_impl impl);

/* x and y are host arrays of rows*cols floats; y may equal x. rows == 0 or
 * cols == 0 is a no-op. */
KL_API kl_status kl_softmax_host(int64_t rows, int64_t cols, const float* x, float* y,
                                 kl_softmax_impl impl);

/* Device workspace (bytes) kl_softmax_device needs for (rows, cols, impl);
 * 0 for implementations that need none. Validates like kl_softmax_supported. */
KL_API kl_status kl_softmax_workspace_bytes(int64_t rows, int64_t cols, kl_softmax_impl impl,
                                            size_t* out_bytes);

/*
 * Device-pointer entry point (for benchmarks and interop, e.g. PyTorch
 * tensors): enqueues softmax on `stream` (a cudaStream_t; NULL = legacy
 * default stream) and returns without synchronizing. x, y: 4-byte-aligned
 * device arrays of rows*cols floats; workspace: 4-byte-aligned device memory
 * of at least kl_softmax_workspace_bytes bytes (may be NULL if that is 0).
 * x, y and workspace must not overlap (checked: KL_ERR_INVALID_ARGUMENT).
 * Launch errors are reported here; errors during execution surface at the
 * caller's next synchronization. Allocates nothing.
 */
KL_API kl_status kl_softmax_device(int64_t rows, int64_t cols, const float* x, float* y,
                                   kl_softmax_impl impl, void* workspace, void* stream);

/*
 * Row-wise layer normalization over a contiguous row-major FP32 matrix,
 * matching torch.nn.functional.layer_norm(x, (cols,), gamma, beta, eps):
 *   y[r][j] = (x[r][j] - mean_r) / sqrt(var_r + eps) * gamma[j] + beta[j]
 * with the biased variance. gamma and beta (length cols) are both given or
 * both NULL (no affine). eps must be finite and >= 0. Domain, special values
 * and the derived error bound: docs/layernorm.md.
 */
#define KL_LAYERNORM_MAX_COLS (1 << 20)
#define KL_LAYERNORM_MAX_ROWS (1 << 20)
#define KL_LAYERNORM_MAX_ELEMENTS (1 << 26)
#define KL_LAYERNORM_WARP_MAX_COLS 1024
#define KL_LAYERNORM_SMEM_MAX_COLS 8192

typedef enum kl_layernorm_impl {
    KL_LAYERNORM_AUTO = 0,         /* FUSED_WARP if cols <= 1024, FUSED_SMEM if <= 8192, else FUSED_BLOCK */
    KL_LAYERNORM_MULTIPASS = 1,    /* baseline: 3 kernels (mean, variance, normalize) */
    KL_LAYERNORM_FUSED_BLOCK = 2,  /* 1 kernel, block per row, two-pass variance, 3 reads of the row */
    KL_LAYERNORM_FUSED_WARP = 3,   /* 1 kernel, warp per row, row in registers; cols <= 1024 */
    KL_LAYERNORM_FUSED_SMEM = 4,   /* FUSED_BLOCK with the row in shared memory, 1 read; cols <= 8192 */
    KL_LAYERNORM_IMPL_FORCE_INT_MIN = -0x7fffffff - 1, /* range widening, as kl_softmax_impl */
    KL_LAYERNORM_IMPL_FORCE_INT = 0x7fffffff
} kl_layernorm_impl;

/* Validates (rows, cols, eps, impl) without touching the GPU. */
KL_API kl_status kl_layernorm_supported(int64_t rows, int64_t cols, float eps, kl_layernorm_impl impl);

/* Device workspace (bytes) kl_layernorm_device needs; 0 if none. */
KL_API kl_status kl_layernorm_workspace_bytes(int64_t rows, int64_t cols, kl_layernorm_impl impl,
                                              size_t* out_bytes);

/* Host arrays: x and y of rows*cols floats (y may equal x); gamma, beta of
 * cols floats or both NULL. rows == 0 or cols == 0 is a no-op. */
KL_API kl_status kl_layernorm_host(int64_t rows, int64_t cols, const float* x, const float* gamma,
                                   const float* beta, float eps, float* y, kl_layernorm_impl impl);

/* Device-pointer entry point (as kl_softmax_device): enqueues on `stream`,
 * no synchronization, no allocation. All pointers 4-byte aligned; y and the
 * workspace must not overlap each other or any input (checked). */
KL_API kl_status kl_layernorm_device(int64_t rows, int64_t cols, const float* x, const float* gamma,
                                     const float* beta, float eps, float* y, kl_layernorm_impl impl,
                                     void* workspace, void* stream);

/*
 * FP32 matrix multiplication C = A B for contiguous row-major A[M][K],
 * B[K][N], C[M][N] (no scaling, C overwritten). M == 0 or N == 0 is a no-op;
 * K == 0 sets C to 0. Each dimension <= 2^14, each matrix <= 2^26 elements.
 * All implementations compute each output as the same sequential fmaf sum,
 * so their results are bitwise identical. Domain and error bound:
 * docs/gemm.md.
 */
#define KL_GEMM_MAX_DIM (1 << 14)
#define KL_GEMM_MAX_ELEMENTS (1 << 26)

typedef enum kl_gemm_impl {
    KL_GEMM_AUTO = 0,      /* REGBLOCK */
    KL_GEMM_NAIVE = 1,     /* baseline: one thread per output, no reuse */
    KL_GEMM_TILED = 2,     /* 32 x 32 shared-memory tiles */
    KL_GEMM_REGBLOCK = 3,  /* 64 x 64 shared-memory tiles, 4 x 4 outputs per thread */
    KL_GEMM_IMPL_FORCE_INT_MIN = -0x7fffffff - 1, /* range widening, as kl_softmax_impl */
    KL_GEMM_IMPL_FORCE_INT = 0x7fffffff
} kl_gemm_impl;

/* Validates (M, N, K, impl) without touching the GPU. */
KL_API kl_status kl_gemm_supported(int64_t m, int64_t n, int64_t k, kl_gemm_impl impl);

/* Host arrays (any overlap allowed: inputs are copied before C is written). */
KL_API kl_status kl_gemm_host(int64_t m, int64_t n, int64_t k, const float* a, const float* b, float* c,
                              kl_gemm_impl impl);

/* Device pointers, enqueued on `stream`, no synchronization or allocation.
 * Pointers 4-byte aligned; C must not overlap A or B (checked). */
KL_API kl_status kl_gemm_device(int64_t m, int64_t n, int64_t k, const float* a, const float* b, float* c,
                                kl_gemm_impl impl, void* stream);

/*
 * Deterministic test data: fills out[0..n) with floats in [lo, hi] from a
 * SplitMix64 stream seeded with `seed` (hi is reachable only through the
 * final rounding to float). Requires finite lo < hi (NaN or infinite bounds
 * are rejected; they would produce NaN). CPU only. Bit-identical to the
 * Python implementation in tests/python/kl.py (checked by a test).
 */
KL_API kl_status kl_fill_uniform(uint64_t seed, int64_t n, float lo, float hi, float* out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* KERNELLENS_KERNELLENS_H */
