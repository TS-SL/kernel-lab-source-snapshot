// C ABI implementation. Validates arguments, owns host<->device staging for
// the *_host entry points, and translates everything into kl_status.
//
// No C++ exception may cross the C boundary: every entry point that can
// allocate host memory runs its body through guarded().
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>

#include "common/cuda_check.hpp"
#include "common/device_buffer.hpp"
#include "common/prng.hpp"
#include "cuda/kernels.hpp"
#include "kernellens/kernellens.h"

using kl::detail::DeviceBuffer;

namespace {

// Runs `body` and converts any escaping exception into a status code.
template <typename Body>
kl_status guarded(const char* fn, Body&& body) noexcept {
    kl_status status = KL_ERR_INTERNAL;
    const char* what = "unexpected C++ exception";
    try {
        return body();
    } catch (const std::bad_alloc&) {
        status = KL_ERR_RESOURCE_LIMIT;
        what = "host memory allocation failed";
    } catch (...) {
    }
    try {
        kl::detail::set_last_error(std::string(fn) + ": " + what);
    } catch (...) {
        kl::detail::clear_last_error();  // cannot even build the message
    }
    return status;
}

kl_status select_device0() {
    int count = 0;
    const cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        kl::detail::cuda_failure(err, "cudaGetDeviceCount(&count)", __FILE__, __LINE__);
        return KL_ERR_NO_DEVICE;
    }
    if (count < 1) {
        return kl::detail::fail(KL_ERR_NO_DEVICE, "no CUDA device found");
    }
    KL_CUDA_TRY(cudaSetDevice(0));
    return KL_OK;
}

// n floats -> bytes, refusing sizes whose byte count would overflow size_t.
kl_status float_bytes(std::int64_t n, std::size_t& out) {
    if (n < 0) {
        return kl::detail::fail(KL_ERR_INVALID_ARGUMENT, "n must be >= 0");
    }
    const auto un = static_cast<std::uint64_t>(n);
    if (un > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        return kl::detail::fail(KL_ERR_RESOURCE_LIMIT, "n is too large");
    }
    out = static_cast<std::size_t>(un) * sizeof(float);
    return KL_OK;
}

}  // namespace

extern "C" {

const char* kl_status_string(kl_status status) {
    switch (status) {
        case KL_OK: return "KL_OK";
        case KL_ERR_INVALID_ARGUMENT: return "KL_ERR_INVALID_ARGUMENT";
        case KL_ERR_UNSUPPORTED: return "KL_ERR_UNSUPPORTED";
        case KL_ERR_RESOURCE_LIMIT: return "KL_ERR_RESOURCE_LIMIT";
        case KL_ERR_CUDA: return "KL_ERR_CUDA";
        case KL_ERR_NO_DEVICE: return "KL_ERR_NO_DEVICE";
        case KL_ERR_INTERNAL: return "KL_ERR_INTERNAL";
    }
    return "KL_STATUS_UNKNOWN";
}

const char* kl_last_error_message(void) { return kl::detail::last_error().c_str(); }

size_t kl_device_alloc_hard_cap(void) { return kl::detail::kDeviceAllocHardCap; }

size_t kl_device_live_bytes(void) { return kl::detail::live_device_bytes(); }

kl_status kl_get_device_info(kl_device_info* out) {
    return guarded(__func__, [&]() -> kl_status {
        kl::detail::clear_last_error();
        if (out == nullptr) {
            return kl::detail::fail(KL_ERR_INVALID_ARGUMENT, "kl_get_device_info: out is null");
        }
        std::memset(out, 0, sizeof(*out));
        KL_TRY(select_device0());

        cudaDeviceProp prop{};
        KL_CUDA_TRY(cudaGetDeviceProperties(&prop, 0));
        static_assert(sizeof(out->name) >= sizeof(prop.name), "name buffer too small");
        std::memcpy(out->name, prop.name, sizeof(prop.name));
        out->name[sizeof(out->name) - 1] = '\0';

        KL_CUDA_TRY(cudaDeviceGetAttribute(&out->compute_major, cudaDevAttrComputeCapabilityMajor, 0));
        KL_CUDA_TRY(cudaDeviceGetAttribute(&out->compute_minor, cudaDevAttrComputeCapabilityMinor, 0));
        KL_CUDA_TRY(cudaDeviceGetAttribute(&out->multiprocessor_count, cudaDevAttrMultiProcessorCount, 0));
        KL_CUDA_TRY(cudaDeviceGetAttribute(&out->max_threads_per_multiprocessor,
                                           cudaDevAttrMaxThreadsPerMultiProcessor, 0));
        KL_CUDA_TRY(cudaDeviceGetAttribute(&out->kernel_exec_timeout, cudaDevAttrKernelExecTimeout, 0));
        KL_CUDA_TRY(cudaDriverGetVersion(&out->driver_version));
        KL_CUDA_TRY(cudaRuntimeGetVersion(&out->runtime_version));
        KL_CUDA_TRY(cudaMemGetInfo(&out->free_global_mem, &out->total_global_mem));
        return KL_OK;
    });
}

kl_status kl_saxpy_host(int64_t n, float a, const float* x, float* y) {
    return guarded(__func__, [&]() -> kl_status {
        kl::detail::clear_last_error();
        std::size_t bytes = 0;
        KL_TRY(float_bytes(n, bytes));
        if (n == 0) {
            return KL_OK;
        }
        if (x == nullptr || y == nullptr) {
            return kl::detail::fail(KL_ERR_INVALID_ARGUMENT, "kl_saxpy_host: null host pointer");
        }
        if (bytes > std::numeric_limits<std::size_t>::max() / 2) {
            return kl::detail::fail(KL_ERR_RESOURCE_LIMIT, "kl_saxpy_host: n is too large");
        }
        KL_TRY(select_device0());

        // The whole operation's device footprint (two buffers) is checked
        // before the first allocation, and both buffers exist before any host
        // memory is read, so an over-cap request touches neither the GPU nor
        // x/y.
        KL_TRY(kl::detail::check_device_budget(2 * bytes));
        DeviceBuffer dx;
        DeviceBuffer dy;
        KL_TRY(DeviceBuffer::allocate(bytes, dx));
        KL_TRY(DeviceBuffer::allocate(bytes, dy));

        KL_CUDA_TRY(cudaMemcpy(dx.get(), x, bytes, cudaMemcpyHostToDevice));
        KL_CUDA_TRY(cudaMemcpy(dy.get(), y, bytes, cudaMemcpyHostToDevice));
        KL_TRY(kl::cuda::launch_saxpy(n, a, dx.as<float>(), dy.as<float>(),
                                      kl::cuda::kSaxpyBlockSize, /*stream=*/0));
        // Surface asynchronous execution errors here, attributed to the
        // kernel, rather than at the following memcpy.
        KL_CUDA_TRY(cudaStreamSynchronize(0));
        KL_CUDA_TRY(cudaMemcpy(y, dy.get(), bytes, cudaMemcpyDeviceToHost));
        return KL_OK;
    });
}

kl_status kl_softmax_supported(int64_t rows, int64_t cols, kl_softmax_impl impl) {
    return guarded(__func__, [&]() -> kl_status {
        kl::detail::clear_last_error();
        return kl::cuda::softmax_check(rows, cols, impl);
    });
}

kl_status kl_softmax_host(int64_t rows, int64_t cols, const float* x, float* y,
                          kl_softmax_impl impl) {
    return guarded(__func__, [&]() -> kl_status {
        kl::detail::clear_last_error();
        KL_TRY(kl::cuda::softmax_check(rows, cols, impl));
        if (rows == 0 || cols == 0) {
            return KL_OK;
        }
        if (x == nullptr || y == nullptr) {
            return kl::detail::fail(KL_ERR_INVALID_ARGUMENT, "kl_softmax_host: null host pointer");
        }
        KL_TRY(select_device0());

        // Within the domain rows*cols <= 2^26, so these cannot overflow.
        const std::size_t bytes = static_cast<std::size_t>(rows * cols) * sizeof(float);
        const kl_softmax_impl resolved = kl::cuda::softmax_resolve(rows, cols, impl);
        const std::size_t ws_bytes = kl::cuda::softmax_workspace_floats(rows, resolved) * sizeof(float);
        KL_TRY(kl::detail::check_device_budget(2 * bytes + ws_bytes));
        DeviceBuffer dx;
        DeviceBuffer dy;
        DeviceBuffer dws;
        KL_TRY(DeviceBuffer::allocate(bytes, dx));
        KL_TRY(DeviceBuffer::allocate(bytes, dy));
        KL_TRY(DeviceBuffer::allocate(ws_bytes, dws));

        KL_CUDA_TRY(cudaMemcpy(dx.get(), x, bytes, cudaMemcpyHostToDevice));
        KL_TRY(kl::cuda::launch_softmax(rows, cols, dx.as<float>(), dy.as<float>(), resolved,
                                        dws.as<float>(), /*stream=*/0));
        KL_CUDA_TRY(cudaStreamSynchronize(0));
        KL_CUDA_TRY(cudaMemcpy(y, dy.get(), bytes, cudaMemcpyDeviceToHost));
        return KL_OK;
    });
}

kl_status kl_softmax_workspace_bytes(int64_t rows, int64_t cols, kl_softmax_impl impl,
                                     size_t* out_bytes) {
    return guarded(__func__, [&]() -> kl_status {
        kl::detail::clear_last_error();
        if (out_bytes == nullptr) {
            return kl::detail::fail(KL_ERR_INVALID_ARGUMENT, "kl_softmax_workspace_bytes: out is null");
        }
        *out_bytes = 0;
        KL_TRY(kl::cuda::softmax_check(rows, cols, impl));
        if (rows == 0 || cols == 0) return KL_OK;
        const kl_softmax_impl resolved = kl::cuda::softmax_resolve(rows, cols, impl);
        *out_bytes = kl::cuda::softmax_workspace_floats(rows, resolved) * sizeof(float);
        return KL_OK;
    });
}

kl_status kl_softmax_device(int64_t rows, int64_t cols, const float* x, float* y,
                            kl_softmax_impl impl, void* workspace, void* stream) {
    return guarded(__func__, [&]() -> kl_status {
        kl::detail::clear_last_error();
        // launch_softmax validates shape, pointers and the workspace need.
        return kl::cuda::launch_softmax(rows, cols, x, y, impl, static_cast<float*>(workspace),
                                        static_cast<cudaStream_t>(stream));
    });
}

kl_status kl_layernorm_supported(int64_t rows, int64_t cols, float eps, kl_layernorm_impl impl) {
    return guarded(__func__, [&]() -> kl_status {
        kl::detail::clear_last_error();
        return kl::cuda::layernorm_check(rows, cols, eps, impl);
    });
}

kl_status kl_layernorm_workspace_bytes(int64_t rows, int64_t cols, kl_layernorm_impl impl,
                                       size_t* out_bytes) {
    return guarded(__func__, [&]() -> kl_status {
        kl::detail::clear_last_error();
        if (out_bytes == nullptr) {
            return kl::detail::fail(KL_ERR_INVALID_ARGUMENT, "kl_layernorm_workspace_bytes: out is null");
        }
        *out_bytes = 0;
        KL_TRY(kl::cuda::layernorm_check(rows, cols, 0.0f, impl));
        if (rows == 0 || cols == 0) return KL_OK;
        *out_bytes = kl::cuda::layernorm_workspace_floats(rows, kl::cuda::layernorm_resolve(cols, impl)) *
                     sizeof(float);
        return KL_OK;
    });
}

kl_status kl_layernorm_host(int64_t rows, int64_t cols, const float* x, const float* gamma,
                            const float* beta, float eps, float* y, kl_layernorm_impl impl) {
    return guarded(__func__, [&]() -> kl_status {
        kl::detail::clear_last_error();
        KL_TRY(kl::cuda::layernorm_check(rows, cols, eps, impl));
        if (rows == 0 || cols == 0) {
            return KL_OK;
        }
        if (x == nullptr || y == nullptr) {
            return kl::detail::fail(KL_ERR_INVALID_ARGUMENT, "kl_layernorm_host: null host pointer");
        }
        if ((gamma == nullptr) != (beta == nullptr)) {
            return kl::detail::fail(KL_ERR_INVALID_ARGUMENT,
                                    "kl_layernorm_host: gamma and beta must both be given or both be NULL");
        }
        KL_TRY(select_device0());

        // Within the domain rows*cols <= 2^26, so these cannot overflow.
        const std::size_t bytes = static_cast<std::size_t>(rows * cols) * sizeof(float);
        const std::size_t affine_bytes = gamma ? static_cast<std::size_t>(cols) * sizeof(float) : 0;
        const kl_layernorm_impl resolved = kl::cuda::layernorm_resolve(cols, impl);
        const std::size_t ws_bytes = kl::cuda::layernorm_workspace_floats(rows, resolved) * sizeof(float);
        KL_TRY(kl::detail::check_device_budget(2 * bytes + 2 * affine_bytes + ws_bytes));
        DeviceBuffer dx, dy, dg, db, dws;
        KL_TRY(DeviceBuffer::allocate(bytes, dx));
        KL_TRY(DeviceBuffer::allocate(bytes, dy));
        KL_TRY(DeviceBuffer::allocate(affine_bytes, dg));
        KL_TRY(DeviceBuffer::allocate(affine_bytes, db));
        KL_TRY(DeviceBuffer::allocate(ws_bytes, dws));

        KL_CUDA_TRY(cudaMemcpy(dx.get(), x, bytes, cudaMemcpyHostToDevice));
        if (gamma != nullptr) {
            KL_CUDA_TRY(cudaMemcpy(dg.get(), gamma, affine_bytes, cudaMemcpyHostToDevice));
            KL_CUDA_TRY(cudaMemcpy(db.get(), beta, affine_bytes, cudaMemcpyHostToDevice));
        }
        KL_TRY(kl::cuda::launch_layernorm(rows, cols, dx.as<float>(), dg.as<float>(), db.as<float>(), eps,
                                          dy.as<float>(), resolved, dws.as<float>(), /*stream=*/0));
        KL_CUDA_TRY(cudaStreamSynchronize(0));
        KL_CUDA_TRY(cudaMemcpy(y, dy.get(), bytes, cudaMemcpyDeviceToHost));
        return KL_OK;
    });
}

kl_status kl_layernorm_device(int64_t rows, int64_t cols, const float* x, const float* gamma,
                              const float* beta, float eps, float* y, kl_layernorm_impl impl,
                              void* workspace, void* stream) {
    return guarded(__func__, [&]() -> kl_status {
        kl::detail::clear_last_error();
        return kl::cuda::launch_layernorm(rows, cols, x, gamma, beta, eps, y, impl,
                                          static_cast<float*>(workspace), static_cast<cudaStream_t>(stream));
    });
}

kl_status kl_fill_uniform(uint64_t seed, int64_t n, float lo, float hi, float* out) {
    return guarded(__func__, [&]() -> kl_status {
        kl::detail::clear_last_error();
        if (n < 0) {
            return kl::detail::fail(KL_ERR_INVALID_ARGUMENT, "kl_fill_uniform: n must be >= 0");
        }
        if (n > 0 && out == nullptr) {
            return kl::detail::fail(KL_ERR_INVALID_ARGUMENT, "kl_fill_uniform: out is null");
        }
        // Infinite bounds would give NaN (inf - inf, inf * 0); NaN bounds
        // fail every comparison.
        if (!std::isfinite(lo) || !std::isfinite(hi) || !(lo < hi)) {
            return kl::detail::fail(KL_ERR_INVALID_ARGUMENT,
                                    "kl_fill_uniform: requires finite lo < hi");
        }
        kl::detail::SplitMix64 rng(seed);
        for (int64_t i = 0; i < n; ++i) {
            out[i] = kl::detail::uniform_float(rng, lo, hi);
        }
        return KL_OK;
    });
}

kl_status kl_gemm_supported(int64_t m, int64_t n, int64_t k, kl_gemm_impl impl) {
    return guarded(__func__, [&]() -> kl_status {
        kl::detail::clear_last_error();
        return kl::cuda::gemm_check(m, n, k, impl);
    });
}

kl_status kl_gemm_host(int64_t m, int64_t n, int64_t k, const float* a, const float* b, float* c,
                       kl_gemm_impl impl) {
    return guarded(__func__, [&]() -> kl_status {
        kl::detail::clear_last_error();
        KL_TRY(kl::cuda::gemm_check(m, n, k, impl));
        if (m == 0 || n == 0) return KL_OK;
        if (c == nullptr || (k > 0 && (a == nullptr || b == nullptr))) {
            return kl::detail::fail(KL_ERR_INVALID_ARGUMENT, "kl_gemm_host: null host pointer");
        }
        KL_TRY(select_device0());
        // Within the domain (each matrix <= 2^26 elements) these cannot overflow.
        const std::size_t na = static_cast<std::size_t>(m * k) * sizeof(float);
        const std::size_t nb = static_cast<std::size_t>(k * n) * sizeof(float);
        const std::size_t nc = static_cast<std::size_t>(m * n) * sizeof(float);
        KL_TRY(kl::detail::check_device_budget(na + nb + nc));
        DeviceBuffer da, db, dc;
        KL_TRY(DeviceBuffer::allocate(na, da));
        KL_TRY(DeviceBuffer::allocate(nb, db));
        KL_TRY(DeviceBuffer::allocate(nc, dc));
        if (k > 0) {
            KL_CUDA_TRY(cudaMemcpy(da.get(), a, na, cudaMemcpyHostToDevice));
            KL_CUDA_TRY(cudaMemcpy(db.get(), b, nb, cudaMemcpyHostToDevice));
        }
        KL_TRY(kl::cuda::launch_gemm(m, n, k, da.as<float>(), db.as<float>(), dc.as<float>(), impl, /*stream=*/0));
        KL_CUDA_TRY(cudaStreamSynchronize(0));
        KL_CUDA_TRY(cudaMemcpy(c, dc.get(), nc, cudaMemcpyDeviceToHost));
        return KL_OK;
    });
}

kl_status kl_gemm_device(int64_t m, int64_t n, int64_t k, const float* a, const float* b, float* c,
                         kl_gemm_impl impl, void* stream) {
    return guarded(__func__, [&]() -> kl_status {
        kl::detail::clear_last_error();
        // launch_gemm validates shape and pointers.
        return kl::cuda::launch_gemm(m, n, k, a, b, c, impl, static_cast<cudaStream_t>(stream));
    });
}

}  // extern "C"
