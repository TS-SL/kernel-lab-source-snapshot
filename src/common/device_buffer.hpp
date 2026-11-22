// Owning RAII wrapper for device memory, with the project's VRAM safety cap.
//
// Ownership rule: every device allocation in KernelLens is a DeviceBuffer.
// Raw cudaMalloc/cudaFree are not used elsewhere, so the live-byte counter
// below is the complete picture of what the library holds.
#pragma once

#include <cstddef>
#include <cstdint>

#include "kernellens/kernellens.h"

namespace kl::detail {

// Hard cap on device memory the library may hold at once. 2 GiB is far above
// what the planned workloads need (see docs/methodology.md) and well below
// half of a 12 GB RTX 3060.
inline constexpr std::size_t kDeviceAllocHardCap = std::size_t{2} << 30;

std::size_t live_device_bytes();

// Number of successful cudaMalloc calls so far (lets tests prove that a
// refused request allocated nothing).
std::uint64_t device_malloc_count();

// The allocation policy as pure functions, so every branch is unit-tested
// and DeviceBuffer::allocate uses the same code (no second copy of a rule).
// Each returns KL_OK or KL_ERR_RESOURCE_LIMIT.
kl_status cap_check(std::size_t request, std::size_t live);      // live + request <= hard cap
kl_status free_check(std::size_t request, std::size_t free_bytes);  // request <= free / 2
kl_status budget_check(std::size_t request, std::size_t live, std::size_t free_bytes);  // both

// Applies budget_check to the TOTAL device memory an operation is about to
// allocate, using the current live bytes and cudaMemGetInfo. Call it before
// the first DeviceBuffer::allocate of a multi-buffer operation so an over-cap
// request is refused before any memory is taken.
kl_status check_device_budget(std::size_t total_bytes);

class DeviceBuffer {
public:
    DeviceBuffer() = default;
    ~DeviceBuffer();
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

    // Allocates `bytes` of device memory into `out` (which must be empty).
    // Atomically reserves the bytes against kDeviceAllocHardCap and checks
    // them against half of free device memory before calling cudaMalloc;
    // refuses with KL_ERR_RESOURCE_LIMIT otherwise.
    static kl_status allocate(std::size_t bytes, DeviceBuffer& out);

    void* get() const { return ptr_; }
    template <typename T>
    T* as() const { return static_cast<T*>(ptr_); }
    std::size_t size_bytes() const { return bytes_; }

    // Frees the allocation (if any). A cudaFree failure cannot be reported
    // from here (destructors cannot fail); it only happens after an earlier,
    // already-reported error has broken the context.
    void reset();

private:
    void* ptr_ = nullptr;
    std::size_t bytes_ = 0;
};

}  // namespace kl::detail
