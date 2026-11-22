// Deterministic, language-independent test data.
//
// SplitMix64 (Steele, Lea & Flood, "Fast Splittable Pseudorandom Number
// Generators", OOPSLA 2014; constants as in Vigna's public-domain
// splitmix64.c). Chosen because it is a few lines of integer arithmetic that
// can be reproduced exactly in Python, so C++ tests, Python tests and
// benchmarks can generate bit-identical inputs from the same seed.
#pragma once

#include <cstdint>

namespace kl::detail {

struct SplitMix64 {
    std::uint64_t state;

    explicit SplitMix64(std::uint64_t seed) : state(seed) {}

    std::uint64_t next() {
        std::uint64_t z = (state += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }

    // Uniform in [0, 1) with 24 bits of randomness: exactly representable
    // in float and double, so the mapping is unambiguous across languages.
    double next_unit() { return static_cast<double>(next() >> 40) * (1.0 / 16777216.0); }
};

// lo + (hi - lo) * u evaluated in double, then rounded once to float.
// x64 MSVC uses SSE2 double arithmetic without FMA contraction by default,
// which matches CPython's float semantics.
inline float uniform_float(SplitMix64& rng, double lo, double hi) {
    return static_cast<float>(lo + (hi - lo) * rng.next_unit());
}

}  // namespace kl::detail
