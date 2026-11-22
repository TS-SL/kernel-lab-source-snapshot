# Architecture

The infrastructure every operator uses, plus one smoke-test operator
(SAXPY). The operators (softmax, layer normalization, GEMM) slot into the
same layers.

```
 Python tests (ctypes)        C++ tests / benchmarks
        |                              |
        v                              v
 include/kernellens/kernellens.h   - C ABI: kl_status codes, *_host entry points
        |
 src/kernellens_api.cpp           - argument validation, device selection,
        |                          host<->device staging, synchronization
        v
 src/common/device_buffer.*      - RAII device memory + VRAM safety cap
 src/common/cuda_check.hpp       - KL_CUDA_TRY / KL_CHECK_LAUNCH / last error
        |
 src/cuda/kernels.hpp, *.cu      - launchers (device pointers, async, checked)
                                   and __global__ kernels
```

## Decisions

1. **C ABI shared library.** One `kernellens.dll` is driven by C++ tests,
   Python tests (ctypes) and, later, benchmarks. No binding generator, no
   PyTorch extension build, and no ABI coupling to PyTorch's CUDA version.
   The static CUDA runtime is linked, so no CUDA runtime DLL is needed; the
   DLL depends on the GPU driver (`nvcuda.dll`, loaded at run time) and the
   MSVC runtime (`VCRUNTIME140`, `VCRUNTIME140_1`, `MSVCP140`, UCRT).
2. **Status codes, not exceptions, across the boundary.** Every fallible
   call returns `kl_status`; a thread-local message gives `file:line`,
   the failing expression and the CUDA error name. Entry points run inside
   `guarded()`, so a C++ exception (e.g. `std::bad_alloc`) becomes a status
   code instead of crossing the C ABI.
3. **Single owner for device memory.** `DeviceBuffer` is the only place that
   calls `cudaMalloc`/`cudaFree`, so the live-byte counter is complete and
   the safety cap cannot be bypassed by accident. An operation's whole
   footprint is checked (`check_device_budget`) before its first allocation
   and before any host data is read; each allocation then reserves its bytes
   with a compare-exchange loop that never lets the counter exceed the cap.
   The policy itself is a pure function (`budget_check`) so every branch is
   unit-tested.
4. **Launchers are asynchronous; `_host` entry points synchronize.**
   Launchers check the launch (`cudaGetLastError`) but do not synchronize,
   so later benchmarks can time kernels with CUDA events. The `_host` test
   entry points call `cudaStreamSynchronize` right after the kernel so an
   execution error is attributed to the kernel, not to the next memcpy.
5. **Stale-error reset before launch.** `cudaGetLastError()` returns the most
   recent error from any runtime call. Every runtime call is checked except
   `cudaFree` in `DeviceBuffer::reset` (destructors cannot report), so
   launchers clear a pending non-sticky error before launching and the
   post-launch check reports only that launch (regression test:
   `error_paths` step 2, shown to fail without the reset).
6. **Object library.** Sources compile once into `kernellens_objs`, which is
   linked into both the DLL and `test_smoke.exe`, so C++ tests can exercise
   internal launchers (e.g. an invalid block size) without exporting test
   hooks from the public API.
7. **Deterministic inputs shared across languages.** SplitMix64 (C++ and an
   independent numpy implementation) with a fixed double-to-float mapping;
   a test checks that both produce bit-identical arrays.

## Operator layer

8. **Softmax as three interchangeable implementations** behind one
   `kl_softmax_impl` id (`src/cuda/softmax.cu`), sharing warp/block
   reductions in `src/cuda/reduce.cuh`. Shape validation
   (`softmax_check`) is one function used by every entry point, so the
   documented domain is enforced identically everywhere.
9. **Two entry points.** `kl_softmax_host` (copies, synchronizes; for
   tests) and `kl_softmax_device` (device pointers, caller's stream, no
   allocation, no synchronization; for benchmarks and interop). The device
   entry point is what lets PyTorch tensors be passed in directly, so
   KernelLens and torch.softmax can be timed in one process.
10. **Shared test/benchmark reference.** `tests/cpp/softmax_reference.hpp`
    (double reference + derived bound) is used by both `test_softmax` and
    `bench_softmax`, so benchmarked outputs are checked with exactly the
    tested criterion. `tests/cpp/harness.hpp` holds the small test harness
    used by both C++ test programs.

## Build

CMake 4.4 with Ninja and MSVC; `CMAKE_CUDA_ARCHITECTURES=86` (sm_86 SASS +
compute_86 PTX); C++17 for host and device; `-lineinfo` for tool source
correlation. Presets: `release`, `debug`. `ctest` runs the C++ tests and
the Python tests against the freshly built DLL; configuration fails if the
`.venv` is missing unless `-DKERNELLENS_REQUIRE_PYTHON_TESTS=OFF` is given.

## Operators beyond softmax

Layer normalization (`src/cuda/layernorm.cu`, docs/layernorm.md) and GEMM
(`src/cuda/gemm.cu`, docs/gemm.md) follow the same pattern as softmax:
kernels and launchers in `src/cuda`, C entry points in
`src/kernellens_api.cpp`, references and bounds in `tests/`.

## Design notes (defects found during development)

- **Stale CUDA errors.** `cudaGetLastError()` returns the most recent error
  from any runtime call, so an already-handled API failure could be
  misreported as a launch failure. Launchers clear pending errors before
  launching. Regression test: `error_paths` step 2; with the reset line
  removed it fails on a valid launch (`cudaGetLastError() ->
  cudaErrorInvalidDevice`), with it restored it passes (unarchived
  observation).
- **Error code for an over-large block.** CUDA 13.3 on this machine reports
  `cudaErrorInvalidValue` for a 2048-thread block, not the
  `cudaErrorInvalidConfiguration` older material shows. The test checks that
  the failure is attributed to the post-launch check, not which code the
  runtime picks, and prints the code it saw (`results/raw/m1-ctest.log`).
- **Benchmark clock drop.** In the first softmax series, shapes measured
  right after seconds of CPU-side correctness checking ran partly at
  240-870 MHz SM clock. Both harnesses now re-ramp with ~300 ms of the
  shape's own launches before warm-up; the superseded series is kept as the
  evidence (`results/processed/m2-run1-softmax-summary.md`).
- **PyTorch layer-norm baseline.** `aten.native_layer_norm.out` adds three
  device-to-device copies per call (an autogenerated out= variant), which
  would overstate PyTorch's time; `torch.nn.functional.layer_norm` launches
  only `vectorized_layer_norm_kernel` and is the baseline.
- **Sanitizer verdicts.** racecheck prints "RACECHECK SUMMARY" instead of
  "ERROR SUMMARY"; the first classifier called clean racecheck runs FAILED.
  The rules now live in `scripts/sanitizer_verdict.py` with unit tests. The
  first device-checked run also found a test defect (a GEMM test launching
  on an uninitialized buffer), fixed before the clean run.
- **Evidence privacy.** Error messages print repo-relative paths
  (`KL_SOURCE_DIR`), and the scripts replace the repository and profile
  paths in saved logs with `<repo>` / `<home>`.
