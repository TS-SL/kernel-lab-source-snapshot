#include "common/device_buffer.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <string>
#include <utility>

#include "common/cuda_check.hpp"

namespace kl::detail {

namespace {
// Invariant: g_live_bytes <= kDeviceAllocHardCap at all times (reservations
// use a compare-exchange loop that never overshoots).
std::atomic<std::size_t> g_live_bytes{0};
std::atomic<std::uint64_t> g_malloc_count{0};

std::string cap_message(std::size_t request, std::size_t live) {
    return "device allocation of " + std::to_string(request) + " bytes would exceed the " +
           std::to_string(kDeviceAllocHardCap) + "-byte safety cap (live: " + std::to_string(live) +
           ")";
}

std::string free_message(std::size_t request, std::size_t free_bytes) {
    return "device allocation of " + std::to_string(request) +
           " bytes exceeds half of free device memory (" + std::to_string(free_bytes) +
           " bytes free)";
}
}  // namespace

std::size_t live_device_bytes() { return g_live_bytes.load(); }

std::uint64_t device_malloc_count() { return g_malloc_count.load(); }

kl_status cap_check(std::size_t request, std::size_t live) {
    if (request > kDeviceAllocHardCap || live > kDeviceAllocHardCap - request) {
        return fail(KL_ERR_RESOURCE_LIMIT, cap_message(request, live));
    }
    return KL_OK;
}

kl_status free_check(std::size_t request, std::size_t free_bytes) {
    if (request > free_bytes / 2) {
        return fail(KL_ERR_RESOURCE_LIMIT, free_message(request, free_bytes));
    }
    return KL_OK;
}

kl_status budget_check(std::size_t request, std::size_t live, std::size_t free_bytes) {
    KL_TRY(cap_check(request, live));
    return free_check(request, free_bytes);
}

kl_status check_device_budget(std::size_t total_bytes) {
    if (total_bytes == 0) {
        return KL_OK;
    }
    std::size_t free_bytes = 0;
    std::size_t total_mem = 0;
    KL_CUDA_TRY(cudaMemGetInfo(&free_bytes, &total_mem));
    return budget_check(total_bytes, g_live_bytes.load(), free_bytes);
}

DeviceBuffer::~DeviceBuffer() { reset(); }

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : ptr_(std::exchange(other.ptr_, nullptr)), bytes_(std::exchange(other.bytes_, 0)) {}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
        reset();
        ptr_ = std::exchange(other.ptr_, nullptr);
        bytes_ = std::exchange(other.bytes_, 0);
    }
    return *this;
}

void DeviceBuffer::reset() {
    if (ptr_ != nullptr) {
        (void)cudaFree(ptr_);  // see the comment on reset() in the header
        g_live_bytes.fetch_sub(bytes_);
        ptr_ = nullptr;
        bytes_ = 0;
    }
}

kl_status DeviceBuffer::allocate(std::size_t bytes, DeviceBuffer& out) {
    if (out.ptr_ != nullptr) {
        return fail(KL_ERR_INVALID_ARGUMENT, "DeviceBuffer::allocate: output buffer is not empty");
    }
    if (bytes == 0) {
        return KL_OK;  // empty buffer, nothing to allocate
    }
    // Reserve against the hard cap without ever overshooting it, so
    // concurrent callers cannot both pass and the counter is never wrong.
    // cap_check also rejects bytes > cap, so live + bytes cannot overflow.
    std::size_t live = g_live_bytes.load();
    do {
        KL_TRY(cap_check(bytes, live));
    } while (!g_live_bytes.compare_exchange_weak(live, live + bytes));

    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    const cudaError_t info_err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (info_err != cudaSuccess) {
        g_live_bytes.fetch_sub(bytes);
        return cuda_failure(info_err, "cudaMemGetInfo(&free_bytes, &total_bytes)", __FILE__, __LINE__);
    }
    if (const kl_status st = free_check(bytes, free_bytes); st != KL_OK) {
        g_live_bytes.fetch_sub(bytes);
        return st;
    }

    void* ptr = nullptr;
    const cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) {
        g_live_bytes.fetch_sub(bytes);
        return cuda_failure(err, "cudaMalloc(&ptr, bytes)", __FILE__, __LINE__);
    }
    g_malloc_count.fetch_add(1);
    out.ptr_ = ptr;
    out.bytes_ = bytes;
    return KL_OK;
}

}  // namespace kl::detail
