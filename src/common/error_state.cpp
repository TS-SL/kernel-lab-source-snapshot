#include <cctype>
#include <cstddef>
#include <cstdio>
#include <string>

#include "common/cuda_check.hpp"

namespace kl::detail {

namespace {
thread_local std::string g_last_error;
}

void set_last_error(std::string message) { g_last_error = std::move(message); }

void clear_last_error() { g_last_error.clear(); }

const std::string& last_error() { return g_last_error; }

const char* relative_source_path(const char* file) {
#ifdef KL_SOURCE_DIR
    // KL_SOURCE_DIR uses '/' and ends with '/'; __FILE__ may use '\' (MSVC)
    // and differ in drive-letter case. Compare ASCII case-insensitively.
    const char* root = KL_SOURCE_DIR;
    std::size_t i = 0;
    for (; root[i] != '\0'; ++i) {
        char a = file[i];
        if (a == '\0') return file;
        if (a == '\\') a = '/';
        if (std::tolower(static_cast<unsigned char>(a)) !=
            std::tolower(static_cast<unsigned char>(root[i]))) {
            return file;
        }
    }
    return file + i;
#else
    return file;
#endif
}

kl_status cuda_failure(cudaError_t err, const char* expr, const char* file, int line) {
    char buf[1024];
    std::snprintf(buf, sizeof(buf), "%s:%d: %s -> %s: %s", relative_source_path(file), line, expr,
                  cudaGetErrorName(err), cudaGetErrorString(err));
    set_last_error(buf);
    return KL_ERR_CUDA;
}

kl_status fail(kl_status status, const std::string& message) {
    set_last_error(message);
    return status;
}

}  // namespace kl::detail
