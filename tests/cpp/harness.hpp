// Minimal dependency-free test harness shared by the C++ test programs.
//
// Each program defines its tests and calls run_tests(argc, argv, tests).
// Command line: [--exclude TEST]... [--max-n N]
//   --exclude  skips a test (sanitizer runs exclude tests that provoke CUDA
//              errors on purpose)
//   --max-n    upper bound on elements per test case for slow tool runs;
//              tests report every case they skip because of it
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "common/cuda_check.hpp"
#include "kernellens/kernellens.h"

namespace kl_test {

inline int g_failures = 0;
inline std::int64_t g_max_n = std::numeric_limits<std::int64_t>::max();
inline int g_cases_skipped = 0;  // cases left out because of --max-n

#define KL_HERE ::kl::detail::relative_source_path(__FILE__)

#define EXPECT(cond)                                                          \
    do {                                                                      \
        if (!(cond)) {                                                        \
            ++::kl_test::g_failures;                                          \
            std::printf("  FAIL %s:%d: %s\n", KL_HERE, __LINE__, #cond);      \
        }                                                                     \
    } while (0)

#define EXPECT_STATUS(expr, expected)                                                     \
    do {                                                                                  \
        const kl_status st_ = (expr);                                                     \
        if (st_ != (expected)) {                                                          \
            ++::kl_test::g_failures;                                                      \
            std::printf("  FAIL %s:%d: %s returned %s, expected %s (%s)\n", KL_HERE,      \
                        __LINE__, #expr, kl_status_string(st_), kl_status_string(expected), \
                        kl_last_error_message());                                         \
        }                                                                                 \
    } while (0)

// True if a case of n elements must be skipped under --max-n (and counts it).
inline bool skip_for_max_n(std::int64_t n, const char* what) {
    if (n <= g_max_n) return false;
    ++g_cases_skipped;
    std::printf("  %s (%lld elements) skipped (--max-n %lld)\n", what, static_cast<long long>(n),
                static_cast<long long>(g_max_n));
    return true;
}

inline std::uint32_t bits(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

// Identical bit patterns (so +0 != -0), except that any NaN matches any NaN.
inline bool same_float(float a, float b) {
    if (std::isnan(a) || std::isnan(b)) return std::isnan(a) && std::isnan(b);
    return bits(a) == bits(b);
}

// ULP distance; +0/-0 equal; NaN == NaN; exactly one NaN -> huge.
inline std::int64_t ulp_distance(float a, float b) {
    if (std::isnan(a) || std::isnan(b)) {
        return (std::isnan(a) && std::isnan(b)) ? 0 : std::numeric_limits<std::int64_t>::max();
    }
    auto ordered = [](float f) {
        std::int32_t i;
        std::memcpy(&i, &f, sizeof(i));
        return i < 0 ? std::int64_t{INT32_MIN} - i : std::int64_t{i};
    };
    const std::int64_t d = ordered(a) - ordered(b);
    return d < 0 ? -d : d;
}

inline std::vector<float> fill(std::uint64_t seed, std::int64_t n, float lo, float hi) {
    std::vector<float> v(static_cast<std::size_t>(n));
    if (n > 0 && kl_fill_uniform(seed, n, lo, hi, v.data()) != KL_OK) {
        std::printf("  kl_fill_uniform failed: %s\n", kl_last_error_message());
        std::exit(2);
    }
    return v;
}

// Checks that the last error message mentions `text`; prints it if not.
inline void expect_last_error_mentions(const char* text, const char* file, int line) {
    const std::string msg = kl_last_error_message();
    if (msg.find(text) == std::string::npos) {
        ++g_failures;
        std::printf("  FAIL %s:%d: last error does not mention %s; it is: \"%s\"\n", file, line,
                    text, msg.c_str());
    }
}

struct TestCase {
    const char* name;
    void (*fn)();
};

template <std::size_t N>
int run_tests(int argc, char** argv, const TestCase (&tests)[N]) {
    // Unbuffered, so output written before a crash is not lost when stdout
    // is redirected to a log file.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::vector<std::string> excluded;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--exclude" && i + 1 < argc) {
            excluded.emplace_back(argv[++i]);
        } else if (arg == "--max-n" && i + 1 < argc) {
            const char* text = argv[++i];
            char* end = nullptr;
            g_max_n = std::strtoll(text, &end, 10);
            if (end == text || *end != '\0' || g_max_n < 1) {
                std::printf("--max-n needs a positive integer, got \"%s\"\n", text);
                return 2;
            }
        } else {
            std::printf("usage: %s [--exclude TEST]... [--max-n N]\n", argv[0]);
            return 2;
        }
    }
    for (const std::string& name : excluded) {
        bool known = false;
        for (const TestCase& t : tests) known = known || name == t.name;
        if (!known) {
            std::printf("unknown test name for --exclude: %s\n", name.c_str());
            return 2;
        }
    }
    int failed_tests = 0;
    int skipped_tests = 0;
    for (const TestCase& t : tests) {
        bool skip = false;
        for (const std::string& name : excluded) skip = skip || name == t.name;
        if (skip) {
            ++skipped_tests;
            std::printf("[ SKIP ] %s (excluded on the command line)\n", t.name);
            continue;
        }
        const int before = g_failures;
        std::printf("[ RUN  ] %s\n", t.name);
        t.fn();
        const bool ok = g_failures == before;
        failed_tests += ok ? 0 : 1;
        std::printf("[ %s ] %s\n", ok ? " OK " : "FAIL", t.name);
    }
    std::printf("%d of %d tests failed, %d skipped, %d cases skipped by --max-n\n", failed_tests,
                static_cast<int>(N) - skipped_tests, skipped_tests, g_cases_skipped);
    return failed_tests == 0 ? 0 : 1;
}

}  // namespace kl_test
