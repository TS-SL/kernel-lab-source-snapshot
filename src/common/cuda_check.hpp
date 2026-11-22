// Error handling shared by all KernelLens host code.
//
// CUDA runtime calls go through KL_CUDA_TRY, which converts a failure into
// KL_ERR_CUDA, records "file:line: expr -> cudaErrorName: description" for
// kl_last_error_message(), and returns from the enclosing function. The
// exceptions are spelled out where they occur: calls that must undo a
// reservation before returning call cuda_failure() by hand
// (device_buffer.cpp, select_device0), cudaFree in DeviceBuffer::reset cannot
// report (destructor), and launchers deliberately discard a stale error with
// (void)cudaGetLastError() before launching.
//
// Kernel launches are followed by KL_CHECK_LAUNCH(), which catches launch
// configuration errors (cudaGetLastError). Errors raised while the kernel
// runs are asynchronous; they surface at the next synchronizing call, which
// is also wrapped in KL_CUDA_TRY.
#pragma once

#include <cuda_runtime.h>

#include <string>

#include "kernellens/kernellens.h"

namespace kl::detail {

void set_last_error(std::string message);
void clear_last_error();
const std::string& last_error();

// Records a failed CUDA call and returns KL_ERR_CUDA.
kl_status cuda_failure(cudaError_t err, const char* expr, const char* file, int line);

// Records a non-CUDA failure (e.g. invalid argument) and returns `status`.
kl_status fail(kl_status status, const std::string& message);

// __FILE__ without the source-tree prefix (KL_SOURCE_DIR, set by CMake), so
// messages read "src\cuda\saxpy.cu:58" (MSVC keeps backslashes) and contain
// no user paths.
const char* relative_source_path(const char* file);

}  // namespace kl::detail

#define KL_CUDA_TRY(expr)                                                         \
    do {                                                                          \
        const cudaError_t kl_err_ = (expr);                                       \
        if (kl_err_ != cudaSuccess) {                                             \
            return ::kl::detail::cuda_failure(kl_err_, #expr, __FILE__, __LINE__); \
        }                                                                         \
    } while (0)

// Must be placed immediately after a <<<...>>> launch.
#define KL_CHECK_LAUNCH() KL_CUDA_TRY(cudaGetLastError())

// Propagates a non-OK kl_status from a helper call.
#define KL_TRY(expr)                                 \
    do {                                             \
        const kl_status kl_st_ = (expr);             \
        if (kl_st_ != KL_OK) {                       \
            return kl_st_;                           \
        }                                            \
    } while (0)
