# Development guide

Rules and procedures for changing KernelLens. They exist because the only
test machine is a desktop GPU that also drives the display.

## GPU safety (RTX 3060, display-attached, Windows TDR = 2 s)

- **Run every GPU workload through `scripts/gpu_guard.py`.** It stops the
  run at 80 C, on any thermal or hardware slowdown, when `nvidia-smi` fails,
  or at the time limit. The test, sanitizer, benchmark and profiling scripts
  already use it.
- **Keep each kernel launch far below the 2 s Windows watchdog.** The target
  is under 100 ms. Sanitizers and profilers slow kernels down, so use small
  shapes with them (`--max-n`).
- **Respect the device-memory cap.** The library refuses allocations that
  would exceed a 2 GiB total or half of the free VRAM
  (`src/common/device_buffer.hpp`).
- **One GPU-heavy job at a time.**
- **Never change GPU or OS settings from the project.** That covers drivers,
  power limits, clocks, voltages, firmware, TDR settings and counter
  permissions. Two tools need machine settings changed by an administrator:
  - Compute Sanitizer needs the WDDM debugger interface;
  - Nsight Compute needs GPU performance-counter access.

  The scripts only detect these settings and report BLOCKED. Profiles run
  with `ncu --clock-control none`, so clocks are never locked.
- **Stop and investigate** after a GPU reset, a persistent CUDA error or any
  instability.

## Build, test and check

In a new PowerShell window, allow local scripts first (this affects only
that window):

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then:

```powershell
. .\scripts\dev-env.ps1        # MSVC + CUDA + CMake + Ninja for this shell
.\scripts\clean-build.ps1      # fresh configure + build -> results\raw\latest-build.log
.\scripts\run-tests.ps1        # ctest (C++ + Python) under gpu_guard -> results\raw\latest-ctest.log
.\scripts\run-sanitizer.ps1    # FAILED / ERRORS / BLOCKED / CLEAN per tool and program
.\scripts\check-env.ps1        # read-only environment report
```

- Logs named `latest-*` (or any name other than an archived series) are
  gitignored.
- The scripts redact saved logs: the repository path becomes `<repo>` and
  the user profile becomes `<home>`.
- Toolchain versions are pinned in [hardware.md](hardware.md) and the
  `requirements*.txt` files.

## Code conventions

- The public API is C (`include/kernellens/kernellens.h`), and every function
  returns a `kl_status`.
- All device memory is owned by a `kl::detail::DeviceBuffer`. Every runtime
  call goes through `KL_CUDA_TRY`, and every kernel launch is followed by
  `KL_CHECK_LAUNCH()`.
- Unsupported shapes and layouts return an error, never silently wrong
  output.
- Test and benchmark inputs come from fixed seeds via `kl_fill_uniform`
  (C++) or `kl.fill_uniform` (Python); the two are tested bit-identical.
- Every bug fix gets a regression test that fails without the fix.
- Tolerances are derived from the algorithm ([methodology.md](methodology.md)).
  They are never widened to make a test pass.

## Evidence labels

The documentation uses these labels:

| Label | Meaning |
|---|---|
| VERIFIED | executed test or inspection |
| MEASURED | executed, documented measurement meeting the benchmark protocol |
| ESTIMATED | calculated, for example a peak from a datasheet |
| UNVERIFIED | stated but not checked |
| FAILED | ran and did not pass |
| BLOCKED | could not run (missing permission or tool) |

ESTIMATED values are never presented as MEASURED, and a check that did not
run is never reported as passing.

## Benchmark checklist

The full protocol is in [methodology.md](methodology.md). In short:

- **Before measuring:**
  - correctness tests pass for every implementation in the comparison;
  - every implementation gets an identical workload (shape, dtype, layout,
    seed);
  - the GPU is idle and cool;
  - the run goes through the guard.
- **Measuring:**
  - warm-up and clock re-ramp first;
  - CUDA events around fixed-size batches;
  - buffers rotate through more than 4x L2;
  - implementations interleave ABBA;
  - outputs are checked before and after timing;
  - clocks and throttle reasons are recorded.
- **Reporting:**
  - medians with hierarchical-bootstrap CIs over at least 3 processes on a
    clean commit;
  - kernel timings and host-API timings are never mixed;
  - predictions are written before the measured series and judged by the
    analysis script.
- **Profiling explains, it doesn't measure.** Nsight Compute durations are
  never mixed with event timings.
