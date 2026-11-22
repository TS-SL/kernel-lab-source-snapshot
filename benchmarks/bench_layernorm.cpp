// Layer-norm kernel benchmark, C++ harness. Protocol: bench_harness.hpp and
// docs/methodology.md section 2 (identical to bench_softmax). Compares the
// KernelLens implementations (multipass, fused_block, fused_smem for
// cols <= 8192, fused_warp for cols <= 1024) with each other, plus the internal naive
// E[x^2] - E[x]^2 kernel (a study: what the unstable formula would save);
// bench_layernorm_compare.py compares them with PyTorch in one process.
//
// All rows use the affine form (gamma, beta given), x ~ U[-10, 10]
// (well-conditioned, so the naive kernel also meets its own bound and can be
// checked like the others), eps = 1e-5.
//
// Bounded by construction: <= 2^24 elements per shape, each launch < 5 ms,
// fixed sample counts. Run it through scripts/gpu_guard.py.
//
// Usage: bench_layernorm --out FILE.csv [--process N] [--samples N] [--warmup N]
//                        [--target-ms X] [--commit SHA] [--quick]
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "bench_harness.hpp"
#include "common/device_buffer.hpp"
#include "cuda/kernels.hpp"
#include "kernellens/kernellens.h"
#include "layernorm_reference.hpp"

namespace {

using kl::detail::DeviceBuffer;
using kl_bench::Context;
using kl_bench::die;
using kl_bench::Impl;
using kl_bench::Shape;

constexpr float kEps = 1e-5f;

// Pre-registered in docs/layernorm.md ("Benchmark plan") before any
// measurement: the softmax shapes (comparable across operators), two
// transformer-like shapes, a cols sweep at 2^22 elements across the
// predicted L2 transition of FUSED_BLOCK, and the rows sweep at 65536 cols.
const Shape kShapes[] = {{4096, 32},   {4096, 128},  {4096, 512},  {4096, 1024}, {8192, 768},
                         {2048, 2048}, {1024, 4096}, {512, 8192},  {256, 16384}, {128, 32768},
                         {64, 65536},  {16, 262144}, {4, 1048576}, {4096, 4096}, {2048, 8192},
                         {4, 65536},   {8, 65536},   {16, 65536},  {28, 65536},  {56, 65536},
                         {112, 65536}, {224, 65536}};
const Shape kQuickShapes[] = {{4096, 128}, {1024, 4096}};
const Shape kEndToEndShapes[] = {{4096, 128}, {1024, 4096}, {4, 1048576}};

void bench_shape(Context& c, const Shape& s) {
    const std::int64_t elems = s.rows * s.cols;
    const std::size_t bytes = static_cast<std::size_t>(elems) * sizeof(float);
    const int pairs = kl_bench::rotation_pairs(bytes, c.l2_bytes);

    std::vector<DeviceBuffer> xs(pairs), ys(pairs);
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
    std::vector<float> host_g(static_cast<std::size_t>(s.cols)), host_b(static_cast<std::size_t>(s.cols));
    if (kl_fill_uniform(5000 + static_cast<std::uint64_t>(s.cols), s.cols, 0.5f, 1.5f, host_g.data()) != KL_OK ||
        kl_fill_uniform(6000 + static_cast<std::uint64_t>(s.cols), s.cols, -1.0f, 1.0f, host_b.data()) != KL_OK) {
        die("fill gamma/beta");
    }
    DeviceBuffer dg, db, ws;
    const std::size_t pbytes = static_cast<std::size_t>(s.cols) * sizeof(float);
    if (DeviceBuffer::allocate(pbytes, dg) != KL_OK || DeviceBuffer::allocate(pbytes, db) != KL_OK) die("gamma/beta");
    BENCH_CUDA(cudaMemcpy(dg.get(), host_g.data(), pbytes, cudaMemcpyHostToDevice));
    BENCH_CUDA(cudaMemcpy(db.get(), host_b.data(), pbytes, cudaMemcpyHostToDevice));
    if (DeviceBuffer::allocate(2 * static_cast<std::size_t>(s.rows) * sizeof(float), ws) != KL_OK) die("ws");

    cudaStream_t stream;
    BENCH_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    auto check = [&](kl_layernorm_impl impl, bool naive) {
        return [&, impl, naive](int p) {
            std::vector<float> y(static_cast<std::size_t>(elems));
            BENCH_CUDA(cudaMemcpy(y.data(), ys[p].get(), bytes, cudaMemcpyDeviceToHost));
            const kl_test::ln::CheckResult r = kl_test::ln::check_layernorm(
                impl, s.rows, s.cols, host_x[p].data(), host_g.data(), host_b.data(), kEps, y.data(), naive);
            // Bound violations, special-value mismatches and rows outside the
            // bound's validity domain all abort the run.
            return std::make_pair(r.worst_ratio, static_cast<long long>(r.violations + r.nan_mismatches +
                                                                        r.out_of_domain_rows));
        };
    };
    auto public_impl = [&](const char* name, kl_layernorm_impl impl) {
        return Impl{name,
                    [&, impl](int p) {
                        if (kl::cuda::launch_layernorm(s.rows, s.cols, xs[p].as<float>(), dg.as<float>(),
                                                       db.as<float>(), kEps, ys[p].as<float>(), impl, ws.as<float>(),
                                                       stream) != KL_OK) {
                            die("launch_layernorm");
                        }
                    },
                    check(impl, false)};
    };
    std::vector<Impl> impls = {public_impl("multipass", KL_LAYERNORM_MULTIPASS),
                               public_impl("fused_block", KL_LAYERNORM_FUSED_BLOCK)};
    if (s.cols <= KL_LAYERNORM_SMEM_MAX_COLS) impls.push_back(public_impl("fused_smem", KL_LAYERNORM_FUSED_SMEM));
    if (s.cols <= KL_LAYERNORM_WARP_MAX_COLS) impls.push_back(public_impl("fused_warp", KL_LAYERNORM_FUSED_WARP));
    impls.push_back(Impl{"naive_study",
                         [&](int p) {
                             if (kl::cuda::launch_layernorm_naive(s.rows, s.cols, xs[p].as<float>(), dg.as<float>(),
                                                                  db.as<float>(), kEps, ys[p].as<float>(),
                                                                  stream) != KL_OK) {
                                 die("launch_layernorm_naive");
                             }
                         },
                         check(KL_LAYERNORM_FUSED_BLOCK, true)});
    // Controls (docs/layernorm.md, "Controls and shapes added"): the same
    // arithmetic as the public kernels, through launch_layernorm_padded.
    //  fused_block_maxsmem: FUSED_BLOCK's control copy, largest shared-memory
    //                       carveout (small L1), no padding: same blocks per SM
    //                       as fused_block, so the pair isolates L1 capacity;
    //  fused_block_occ:     the same copy padded to FUSED_SMEM's shared memory:
    //                       vs fused_block_maxsmem it isolates occupancy, vs
    //                       fused_smem (same carveout and blocks per SM) one
    //                       read of x vs three;
    //  fused_smem_pad32k:   FUSED_SMEM padded to 32 KiB (3 blocks per SM, as at
    //                       8192 cols), to see what that occupancy costs.
    auto padded = [&](const char* name, kl_layernorm_impl impl, std::size_t extra) {
        return Impl{name,
                    [&, impl, extra](int p) {
                        if (kl::cuda::launch_layernorm_padded(s.rows, s.cols, xs[p].as<float>(), dg.as<float>(),
                                                              db.as<float>(), kEps, ys[p].as<float>(), impl, extra,
                                                              stream) != KL_OK) {
                            die("launch_layernorm_padded");
                        }
                    },
                    check(impl, false)};
    };
    const std::size_t row_bytes = static_cast<std::size_t>(s.cols) * sizeof(float);
    if (s.cols <= KL_LAYERNORM_SMEM_MAX_COLS) impls.push_back(padded("fused_block_maxsmem", KL_LAYERNORM_FUSED_BLOCK, 0));
    if (s.cols > KL_LAYERNORM_WARP_MAX_COLS && s.cols <= KL_LAYERNORM_SMEM_MAX_COLS) {
        impls.push_back(padded("fused_block_occ", KL_LAYERNORM_FUSED_BLOCK, row_bytes));
        if (row_bytes < 32768) impls.push_back(padded("fused_smem_pad32k", KL_LAYERNORM_FUSED_SMEM, 32768 - row_bytes));
    }
    // Resident blocks per SM of each kernel at this shape (API-reported), for the sidecar.
    auto record_occupancy = [&](const char* name, kl_layernorm_impl impl, std::size_t extra, bool control) {
        int blocks = 0;
        if (kl::cuda::layernorm_blocks_per_sm(s.cols, impl, extra, control, &blocks) != KL_OK) die("occupancy query");
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "{\"type\": \"occupancy\", \"impl\": \"%s\", \"rows\": %lld, \"cols\": %lld, "
                      "\"blocks_per_sm\": %d}",
                      name, static_cast<long long>(s.rows), static_cast<long long>(s.cols), blocks);
        kl_bench::meta_item(c, buf);
    };
    record_occupancy("fused_block", KL_LAYERNORM_FUSED_BLOCK, 0, false);
    if (s.cols <= KL_LAYERNORM_SMEM_MAX_COLS) {
        record_occupancy("fused_smem", KL_LAYERNORM_FUSED_SMEM, 0, false);
        record_occupancy("fused_block_maxsmem", KL_LAYERNORM_FUSED_BLOCK, 0, true);
    }
    if (s.cols <= KL_LAYERNORM_WARP_MAX_COLS) record_occupancy("fused_warp", KL_LAYERNORM_FUSED_WARP, 0, false);
    if (s.cols > KL_LAYERNORM_WARP_MAX_COLS && s.cols <= KL_LAYERNORM_SMEM_MAX_COLS) {
        record_occupancy("fused_block_occ", KL_LAYERNORM_FUSED_BLOCK, row_bytes, true);
        if (row_bytes < 32768) record_occupancy("fused_smem_pad32k", KL_LAYERNORM_FUSED_SMEM, 32768 - row_bytes, false);
    }

    kl_bench::run_protocol(c, s, pairs, stream, impls, [&] {
        for (int p = 0; p < pairs; ++p) BENCH_CUDA(cudaMemsetAsync(ys[p].get(), 0xFF, bytes, stream));
    });
    BENCH_CUDA(cudaStreamDestroy(stream));
}

void bench_end_to_end(Context& c, const Shape& s) {
    const std::int64_t elems = s.rows * s.cols;
    std::vector<float> x(static_cast<std::size_t>(elems)), y(static_cast<std::size_t>(elems));
    std::vector<float> g(static_cast<std::size_t>(s.cols)), b(static_cast<std::size_t>(s.cols));
    if (kl_fill_uniform(77, elems, -10.0f, 10.0f, x.data()) != KL_OK ||
        kl_fill_uniform(78, s.cols, 0.5f, 1.5f, g.data()) != KL_OK ||
        kl_fill_uniform(79, s.cols, -1.0f, 1.0f, b.data()) != KL_OK) {
        die("fill");
    }
    const kl_layernorm_impl resolved = kl::cuda::layernorm_resolve(s.cols, KL_LAYERNORM_AUTO);
    const char* name = resolved == KL_LAYERNORM_FUSED_WARP   ? "fused_warp"
                       : resolved == KL_LAYERNORM_FUSED_SMEM ? "fused_smem"
                                                             : "fused_block";
    kl_bench::run_end_to_end(c, s, std::string("host_api_auto=") + name, [&] {
        if (kl_layernorm_host(s.rows, s.cols, x.data(), g.data(), b.data(), kEps, y.data(), KL_LAYERNORM_AUTO) !=
            KL_OK) {
            die("kl_layernorm_host");
        }
    });
}

}  // namespace

int main(int argc, char** argv) {
    // Clock ramp: short layer-norm launches on a zeroed 4096 x 1024 matrix.
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
            if (kl::cuda::launch_layernorm(rr, rc, rx.as<float>(), nullptr, nullptr, kEps, ry.as<float>(),
                                           KL_LAYERNORM_FUSED_WARP, nullptr, 0) != KL_OK) {
                die("ramp launch");
            }
        }
        BENCH_CUDA(cudaDeviceSynchronize());
    };
    return kl_bench::bench_main(argc, argv, "bench_layernorm", ramp, [&](Context& c) {
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
