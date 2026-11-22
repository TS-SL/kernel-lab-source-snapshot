// Pointer checks shared by the device-pointer launchers.
//
// Device pointers share one (unified) address space, so integer comparisons
// of addresses are meaningful.
#pragma once

#include <cstddef>
#include <cstdint>

namespace kl::cuda::checks {

// A misaligned float pointer causes a STICKY cudaErrorMisalignedAddress that
// would also break any other runtime sharing the context (e.g. PyTorch in
// the same process), so launchers refuse it up front. nullptr is aligned.
inline bool float_aligned(const void* p) {
    return reinterpret_cast<std::uintptr_t>(p) % alignof(float) == 0;
}

// Do [a, a + la) and [b, b + lb) overlap? Written without a + la, which
// could wrap for (invalid) pointers near the top of the address space.
// Empty ranges and nullptr never overlap.
inline bool overlap(const void* a, std::size_t la, const void* b, std::size_t lb) {
    if (a == nullptr || b == nullptr || la == 0 || lb == 0) return false;
    const auto x = reinterpret_cast<std::uintptr_t>(a);
    const auto y = reinterpret_cast<std::uintptr_t>(b);
    return x <= y ? y - x < la : x - y < lb;
}

}  // namespace kl::cuda::checks
