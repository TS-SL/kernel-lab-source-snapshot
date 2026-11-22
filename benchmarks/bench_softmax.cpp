// Softmax kernel benchmark, C++ harness. Protocol: bench_harness.hpp and
// docs/methodology.md section 2 (shared with bench_layernorm). Compares
// KernelLens implementations with each other without any Python overhead;
// bench_softmax_compare.py compares them with PyTorch inside one process.
//
// End-to-end: kl_softmax_host (pageable host memory, per-call allocation,
// H2D + kernel + D2H) timed with a host steady clock.
//
// Bounded by construction: <= 2^24 elements per shape, each launch < 5 ms,
// fixed sample counts. Run it through scripts/gpu_guard.py.
//
// Usage: bench_softmax --out FILE.csv [--process N] [--samples N] [--warmup N]
//                      [--target-ms X] [--commit SHA] [--quick]
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "bench_harness.hpp"
#include "common/device_buffer.hpp"
#include "cuda/kernels.hpp"
#include "kernellens/kernellens.h"
#include "softmax_reference.hpp"

namespace {

using kl::detail::DeviceBuffer;
using kl_bench::Context;
using kl_bench::die;
using kl_bench::Impl;
using kl_bench::Shape;

// 2^17 .. 2^24 elements: attention-like short rows (warp-kernel range)
// through long rows, then a rows sweep at fixed row length.
const Shape kShapes[] = {{4096, 32},  {4096, 128},  {4096, 512},  {4096, 1024}, {2048, 2048},
                         {1024, 4096}, {256, 16384}, {64, 65536},  {16, 262144}, {4, 1048576},
                         // Rows sweep at cols = 65536 (the RTX 3060 has 28 SMs; one block
                         // per row): how time grows with the number of rows/blocks.
                         {4, 65536},   {8, 65536},   {16, 65536},  {28, 65536},  {56, 65536},
                         {112, 65536}, {224, 65536}};
const Shape kQuickShapes[] = {{4096, 128}, {1024, 4096}};
const Shape kEndToEndShapes[] = {{4096, 128}, {1024, 4096}, {4, 1048576}};

const char* impl_name(kl_softmax_impl impl) {
    switch (impl) {
        case KL_SOFTMAX_AUTO: return "auto";
        case KL_SOFTMAX_MULTIPASS: return "multipass";
        case KL_SOFTMAX_FUSED_ONLINE: return "fused_online";
        case KL_SOFTMAX_FUSED_WARP: return "fused_warp";
        default: return "?";
    }
}

void bench_shape(Context& c, const Shape& s) {
    const std::int64_t elems = s.rows * s.cols;
    const std::size_t bytes = static_cast<std::size_t>(elems) * sizeof(float);
    const int pairs = kl_bench::rotation_pairs(bytes, c.l2_bytes);

    std::vector<DeviceBuffer> xs(pairs), ys(pairs);
    DeviceBuffer ws;
    std::vector<std::vector<float>> host_x(pairs);
    for (int p = 0; p < pairs; ++p) {
        if (DeviceBuffer::allocate(bytes, xs[p]) != KL_OK) die("allocate x");
        if (DeviceBuffer::allocate(bytes, ys[p]) != KL_OK) die("allocate y");
        host_x[p].resize(static_cast<std::size_t>(elems));
        if (kl_fill_uniform(1000 + 31 * p + static_cast<std::uint64_t>(s.cols), elems, -10.0f, 10.0f,
                            host_x[p].data()) != KL_OK) {
            die("fill");
        }
        BENCH_CUDA(cudaMemcpy(xs[p].get(), host_x[p].data(), bytes, cudaMemcpyHostToDevice));
    }
    if (DeviceBuffer::allocate(2 * static_cast<std::size_t>(s.rows) * sizeof(float), ws) != KL_OK) die("ws");

    cudaStream_t stream;
    BENCH_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    auto make = [&](kl_softmax_impl impl) {
        return Impl{impl_name(impl),
                    [&, impl](int p) {
                        if (kl::cuda::launch_softmax(s.rows, s.cols, xs[p].as<float>(), ys[p].as<float>(), impl,
                                                     ws.as<float>(), stream) != KL_OK) {
                            die("launch_softmax");
                        }
                    },
                    [&, impl](int p) {
                        std::vector<float> y(static_cast<std::size_t>(elems));
                        BENCH_CUDA(cudaMemcpy(y.data(), ys[p].get(), bytes, cudaMemcpyDeviceToHost));
                        const kl_test::CheckResult r =
                            kl_test::check_softmax(impl, s.rows, s.cols, host_x[p].data(), y.data());
                        // Any violation (bound, NaN pattern or row sum) aborts the run.
                        return std::make_pair(
                            r.worst_ratio, static_cast<long long>(r.violations + r.nan_mismatches + r.row_sum_violations));
                    }};
    };
    std::vector<Impl> impls = {make(KL_SOFTMAX_MULTIPASS), make(KL_SOFTMAX_FUSED_ONLINE)};
    if (s.cols <= KL_SOFTMAX_WARP_MAX_COLS) impls.push_back(make(KL_SOFTMAX_FUSED_WARP));

    // Poison = 0xFF bytes (NaN) in every output buffer.
    kl_bench::run_protocol(c, s, pairs, stream, impls, [&] {
        for (int p = 0; p < pairs; ++p) BENCH_CUDA(cudaMemsetAsync(ys[p].get(), 0xFF, bytes, stream));
    });
    BENCH_CUDA(cudaStreamDestroy(stream));
}

void bench_end_to_end(Context& c, const Shape& s) {
    const std::int64_t elems = s.rows * s.cols;
    std::vector<float> x(static_cast<std::size_t>(elems)), y(static_cast<std::size_t>(elems));
    if (kl_fill_uniform(77, elems, -10.0f, 10.0f, x.data()) != KL_OK) die("fill");
    const kl_softmax_impl resolved = kl::cuda::softmax_resolve(s.rows, s.cols, KL_SOFTMAX_AUTO);
    kl_bench::run_end_to_end(c, s, std::string("host_api_auto=") + impl_name(resolved), [&] {
        if (kl_softmax_host(s.rows, s.cols, x.data(), y.data(), KL_SOFTMAX_AUTO) != KL_OK) die("host");
    });
}

}  // namespace

int main(int argc, char** argv) {
    // Clock ramp: short warp-softmax launches on a zeroed 4096 x 1024 matrix.
    const std::int64_t rr = 4096, rc = 1024;
    const std::size_t rbytes = static_cast<std::size_t>(rr * rc) * sizeof(float);
    DeviceBuffer rx, ry;
    auto ramp = [&] {
        if (!rx.get()) {
            if (DeviceBuffer::allocate(rbytes, rx) != KL_OK || DeviceBuffer::allocate(rbytes, ry) != KL_OK) {
                die("ramp");
            }
            BENCH_CUDA(cudaMemset(rx.get(), 0, rbytes));
        }
        for (int i = 0; i < 20; ++i) {
            if (kl::cuda::launch_softmax(rr, rc, rx.as<float>(), ry.as<float>(), KL_SOFTMAX_FUSED_WARP, nullptr, 0) !=
                KL_OK) {
                die("ramp launch");
            }
        }
        BENCH_CUDA(cudaDeviceSynchronize());
    };
    return kl_bench::bench_main(argc, argv, "bench_softmax", ramp, [&](Context& c) {
        rx = DeviceBuffer();  // release the ramp buffers before measuring
        ry = DeviceBuffer();
        if (c.opt->quick) {
            for (const Shape& s : kQuickShapes) bench_shape(c, s);
        } else {
            for (const Shape& s : kShapes) bench_shape(c, s);
            for (const Shape& s : kEndToEndShapes) bench_end_to_end(c, s);
        }
    });
}
