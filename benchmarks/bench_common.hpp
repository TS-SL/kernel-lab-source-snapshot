// Benchmark rules shared by the C++ harness and (by documented formula) the
// Python harness, so every implementation of a shape uses the same batch.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace kl_bench {

// Batch size K for a shape: chosen so one sample of an implementation that
// moves the minimal 8 bytes/element at 300 GB/s lasts target_ms. Fixed per
// shape (not calibrated per implementation): slower implementations simply
// produce longer samples, and every implementation of a shape divides any
// fixed per-batch cost by the same K. Mirrored in bench_softmax_compare.py.
inline int batch_k(std::int64_t elements, double target_ms) {
    const double bytes = 8.0 * static_cast<double>(elements);
    const double k = std::ceil(target_ms * 1e-3 * 300e9 / bytes);
    return static_cast<int>(std::clamp(k, 1.0, 2000.0));
}

}  // namespace kl_bench
