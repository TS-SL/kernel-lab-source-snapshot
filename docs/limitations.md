# Limitations

Every claim below names its tool, operator, shapes, precision and platform,
and says whether its evidence is archived (a saved log from a stated
development revision) or fresh (re-run for this release). The platform for
everything: one NVIDIA GeForce RTX 3060 12 GB (sm_86), driver 610.60,
CUDA 13.3.1, Windows 11 with the WDDM driver model. All operators are FP32,
row-major and contiguous.

## Memory safety (Compute Sanitizer)

| Tool | Test programs (operators) | Result | Evidence |
|---|---|---|---|
| memcheck (`--leak-check full`) | smoke (SAXPY; `error_paths` excluded), softmax, layer norm, GEMM | 0 errors | archived, `results/raw/m5-sanitizer-*-memcheck.log` |
| racecheck | the same four | 0 hazards (0 errors, 0 warnings) | archived, `results/raw/m5-sanitizer-*-racecheck.log` |
| initcheck | the same four | 0 errors | archived, `results/raw/m5-sanitizer-*-initcheck.log` |
| synccheck | the same four | 0 errors | archived, `results/raw/m5-sanitizer-*-synccheck.log` |

What these runs cover and what they don't:

- **Shapes are capped.** `scripts/run-sanitizer.ps1` passes
  `--max-n 1048576` because sanitizers slow kernels 10-100x and the Windows
  watchdog is 2 s. For GEMM the capped quantity is `M N K / 64`. Each log
  lists the cases the cap skipped:
  - SAXPY at 4,194,304 and 16,776,961 elements;
  - layer norm "max rows" (2,097,152 elements);
  - the GEMM bitwise, integer-exact and cuBLAS-bound cases at 4.1M-16.8M
    work units;
  - softmax skipped no case.

  The functional tests are the only check on these large cases.
- **The smoke program's `error_paths` test is excluded.** It provokes CUDA
  errors on purpose.
- **The leak summary is not evidence.** memcheck reports "0 bytes leaked in
  0 allocations" although the tests allocate device memory, so it did not
  track the library's allocations. The no-leak evidence is the library's own
  live-byte counter, which the tests check.
- **The verdict rules are unit-tested.** `scripts/sanitizer_verdict.py` has
  8 cases in `tests/python/test_sanitizer_verdict.py`, built from log shapes
  seen on this machine:
  - CLEAN memcheck and CLEAN racecheck;
  - a racecheck hazard and an initcheck report;
  - BLOCKED;
  - an abnormal exit, a test failure and a missing summary.

  They pass in this release's fresh full-suite run.
- **The tools need a system setting.** Device-side checking requires the
  WDDM debugger interface, a machine-wide registry value.
  - While it was disabled, every tool reported BLOCKED, and racecheck also
    exited abnormally after the tests passed (archived,
    `results/raw/m1-sanitizer-*.log`, `results/raw/m2-sanitizer-*.log`).
  - With it enabled, racecheck completed normally. The exact cause of the
    abnormal exit is UNVERIFIED.
  - The sanitizer was not re-run for this release.
- **The first device-checked run found a test defect** (archived): a GEMM
  test ran valid kernels on an uninitialized buffer (64 initcheck reports).
  The test now zeroes the buffer, and the clean logs come from after the
  fix.
- **Out-of-bounds reads that land in unused shared-memory slots can't be
  seen by functional tests.** An example is a dropped `row < M` or `p < K`
  guard in a GEMM tile load. memcheck found no such read on the capped test
  shapes, which include every tile boundary. Larger shapes are not
  tool-checked.

## Profiling

- **Nsight Compute needs GPU performance-counter access**, an NVIDIA
  Control Panel setting. Without it `ncu` fails with `ERR_NVGPUCTRPERM`
  (`results/raw/m2-ncu-probe.log`).
- **The archived profile is narrow** (`results/raw/m5-ncu-raw.csv`,
  development revision 79a9ab2):
  - one cold launch per kernel, so there is no variance estimate;
  - unlocked clocks (`--clock-control none`);
  - 18 chosen launches.

  Explanations for other shapes remain hypotheses unless a timing control
  supports them.

## Platform and tooling

- **One GPU, one machine, Windows only.** Compiler flags
  (`/W4 /permissive-`, `-Xcompiler=/W3`) and scripts (PowerShell) target
  Windows with MSVC. Linux builds have not been attempted. Only device 0 is
  used.
- **No continuous integration.** The repository has no CI workflow, and no
  hosted run has ever tested it. Standard hosted runners have no NVIDIA GPU,
  so at most they could compile. Every test result here comes from local
  runs on the machine above.
- **GPU guard.** On a stop, `scripts/gpu_guard.py` kills the command's whole
  process tree with `taskkill /T /F`. This release's fresh full-suite run
  tests that, grandchild included.
  - The test needs normal user rights over the processes it starts. A
    separate audit run in a restricted sandbox reported 1,055 of 1,056
    Python tests passing; the process-tree test failed there and then
    passed with normal permissions. That log is not in this repository.
  - The guard cannot find orphans whose parent has already exited.
  - It does not chase processes left behind after a command exits
    normally.
  - A nested Job Object does not work as an alternative, because the host
    shell's job allows silent breakaway.
- **Short path required.** Windows long paths were disabled on the
  development machine, and a venv created under a 180-character path got an
  incomplete pip. Keep the clone at a short path.
- **Dependency integrity.** Python packages are pinned to exact versions
  (`requirements-lock.txt`) but not to hashes. The CUDA installer itself was
  verified (MD5 and NVIDIA signature); the components it downloaded were
  verified only by the installer.
- **The `_host` API is for correctness tests.** It is synchronous and
  allocates on every call. Its timings (allocation + pageable copies +
  kernel) are reported separately and never mixed with kernel timings.
- **SAXPY is a toolchain smoke test.** It says nothing about operator
  performance.

## Softmax

- **The error-bound tests have blind spots** (docs/softmax.md, "What the
  tests can and cannot detect"):
  - a wrong row maximum is caught only when it causes overflow (the
    max-spike test);
  - dropping a single element of a long row is caught only by the bit-exact
    constant-row test.
- **Few long rows.** Every implementation assigns one block (or warp) per
  row, so shapes with few rows cannot use the whole GPU. PyTorch is up to
  5.2x faster there. Splitting a row across blocks is not implemented.
- **Launch-overhead-dominated shapes.** At 4096 x 32 the warp kernel's
  launch takes about as long as the host needs to enqueue it. Its time there
  is an upper bound on the kernel time, and comparisons at that shape are
  flagged. CUDA graphs are not used.
- **NVML clock readings lag** (they are driver-sampled), so per-sample
  clocks are only used as series-level conditions.

## Layer normalization

- **The error bound has a validity domain** (docs/layernorm.md): no
  intermediate overflow, and `var + eps >= 2^-100`.
  - Outside it no accuracy is claimed.
  - Tests pin two outside-domain cases; both match PyTorch.
  - A finite row whose `x_j - x_0` overflows comes out NaN.
- **An outlier first element still reaches the scale.** The corrected mean
  removes the pivot's effect on the mean. The variance, though, is taken
  around the pivot-shifted mean, so an outlier first element enters the
  scale quadratically. This is bounded and tested: up to ~3 % relative in
  the worst case at 2^20 columns (ESTIMATED). The corrected variance that
  would remove it is not implemented.
- **Special-value differences.**
  - With `gamma_j = +-inf`, the column is NaN where the computed deviation
    is exactly 0.
  - On some constant rows, PyTorch's CUDA kernel was seen to differ from
    its own CPU kernel and from KernelLens (no saved log; docs/layernorm.md).
    The tests compare with the CPU kernel.
- **FUSED_SMEM is limited to 8192 columns** (32 KiB of shared memory, below
  the 48 KiB default limit). Longer rows use the three-read FUSED_BLOCK.
  FUSED_SMEM's occupancy depends on a shared-memory carveout *preference*;
  the recorded occupancy is what the API reports, not what was observed.
- **Few long rows.** These also use one block per row, so 4 x 1048576 and
  16 x 262144 leave most SMs idle.
- **The naive `E[x^2] - E[x]^2` kernel is unstable by design.** It is
  internal, used only by tests and benchmarks.

## Matrix multiplication

- **TF32 detection.** The error bound would accept TF32 results at large K,
  and the integer tests are exact in TF32. TF32 is ruled out by:
  - the cuBLAS math mode recorded in every benchmark sidecar
    (`CUBLAS_DEFAULT_MATH`);
  - the `NVIDIA_TF32_OVERRIDE` environment check, also in every sidecar;
  - the K = 64 tests.
- **Benchmark checks sample 32 rows of each output.** The full-output tests
  are separate; a sporadic error outside the sampled rows would be missed.
- **Power limit.** Under GEMM load the board hit its power limit in 31 % of
  samples. Times are at the clocks the GPU chose (group medians
  1852-1935 MHz), not at a fixed clock.
- **Scope.** No transposes, no `alpha`/`beta`, FP32 only, and each
  dimension <= 2^14. REGBLOCK reaches 15-23 % of the ESTIMATED FP32 peak, and
  cuBLAS (FP32, no TF32) is 2.35-3.93x faster on every measured shape.
