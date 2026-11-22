# KernelLens

KernelLens is a study of FP32 CUDA operators on a single GPU. It implements
softmax, layer normalization and matrix multiplication (GEMM) several ways,
checks every implementation against float64 references with derived error
bounds, and measures how reductions, memory access, fusion and tiling change
their speed on one NVIDIA GeForce RTX 3060. The results are compared with
PyTorch and cuBLAS.

The library is a Windows DLL with a C API. One call from Python looks like
this. The output below is from an actual run of
[examples/softmax_example.py](examples/softmax_example.py):

```text
softmax status: KL_OK
softmax output:
 [[0.09003057 0.24472848 0.6652409  0.        ]
 [0.25       0.25       0.25       0.25      ]]
worst error / derived bound: 0.157
gemm status: KL_OK
gemm worst error / derived bound: 0.035
gemm with K = 32768: KL_ERR_UNSUPPORTED - gemm: shape outside the supported domain (docs/gemm.md)
```

What the run shows:

- A masked (`-inf`) entry comes out as an exact zero.
- A constant row comes out exactly uniform.
- A 96 x 80 x 70 GEMM stays within 3.5 % of its error bound.
- A shape outside the supported domain is rejected with a status code rather
  than computed wrongly.

## How it is built

```
include/kernellens/   public C API (kl_status return codes, kl_*_host / kl_*_device)
src/common/          error state, DeviceBuffer (the single owner of device memory, with a VRAM cap), PRNG
src/cuda/            kernels, launchers, shared warp/block reductions, dispatch (AUTO)
tests/cpp/           C++ tests: double-precision CPU references and derived error bounds
tests/python/        Python tests via ctypes: numpy float64 and PyTorch references
benchmarks/          C++ harness (KernelLens vs itself and cuBLAS) and in-process KernelLens-vs-PyTorch harness
scripts/             dev environment, GPU guard, test / sanitizer / benchmark / profile runners, analysis
examples/            the example above
results/raw/         archived evidence logs and raw benchmark samples (redacted copies)
results/processed/   summaries generated from results/raw by scripts
docs/                design, derivations, methodology, limitations, development guide
```

Each operator has two entry points:

- `kl_<op>_host` copies data, synchronizes and allocates on every call. It
  exists for tests.
- `kl_<op>_device` takes device pointers and the caller's stream, and never
  allocates or synchronizes. It is used for benchmarks, and lets PyTorch
  tensors be passed in directly.

All device memory goes through one RAII owner that refuses requests above a
2 GiB cap or above half of free VRAM. Every launch is checked, and an
unsupported shape returns `KL_ERR_UNSUPPORTED`. Details:
[docs/architecture.md](docs/architecture.md).

## Operators

### Softmax (row-wise, [docs/softmax.md](docs/softmax.md))

| Implementation | What it changes |
|---|---|
| `MULTIPASS` (baseline) | 3 kernels: row max, row sum, normalize; reads x 3 times, with per-row stats in global memory |
| `FUSED_ONLINE` | 1 kernel, block per row; max and sum in one read with the online-normalizer recurrence (Milakov & Gimelshein, 2018), then the write |
| `FUSED_WARP` | 1 warp per row, row held in registers, x read once; cols <= 1024 |

**Dispatch (`AUTO`).**

- cols <= 1024: `FUSED_WARP`.
- Otherwise, `FUSED_ONLINE` if rows >= 64, else `MULTIPASS`.
- The 64-row threshold was chosen on a dry run. A held-out rows sweep at
  65536 columns put the crossover between 28 and 64 rows.

### Layer normalization (row-wise, biased variance, [docs/layernorm.md](docs/layernorm.md))

| Implementation | What it changes |
|---|---|
| `MULTIPASS` (baseline) | 3 kernels: mean, variance, normalize |
| `FUSED_BLOCK` | 1 kernel, block per row; reads the row 3 times back to back, so the re-reads can hit the caches |
| `FUSED_SMEM` | `FUSED_BLOCK` with the row cached in shared memory: one global read; cols <= 8192 |
| `FUSED_WARP` | warp per row, row in registers; cols <= 1024 |

**Dispatch (`AUTO`).**

- cols <= 1024: `FUSED_WARP`.
- cols <= 8192: `FUSED_SMEM`.
- Otherwise: `FUSED_BLOCK`.

The mean is shifted by the row's first element and then corrected. This
keeps constant rows exact and bounds the error for rows with a large offset.

### Matrix multiplication (C = A B, row-major, [docs/gemm.md](docs/gemm.md))

| Implementation | What it changes |
|---|---|
| `NAIVE` (baseline) | one thread per output, operands read from global memory |
| `TILED` | 32 x 32 tiles of A and B staged in shared memory |
| `REGBLOCK` | 64 x 64 block tile, 4 x 4 outputs per thread, K in steps of 16 |

**Dispatch (`AUTO`).** Always `REGBLOCK`.

All three kernels use the same `fmaf` order, so their results are bitwise
identical. The error bound `gamma_K * sum |a||b|` does not depend on
summation order, so it also covers cuBLAS.

## Baselines

- **PyTorch 2.15.0+cu132.** The comparisons use `torch.softmax(x, -1, out=y)`
  and `torch.nn.functional.layer_norm(x, (cols,), gamma, beta, 1e-5)`. They
  run in one process, on the same tensors, stream, CUDA events and batch
  size as KernelLens's `kl_<op>_device`, interleaved ABBA.
  - `aten.native_layer_norm.out` was not used: it adds three device copies
    per call.
  - The kernels PyTorch actually launched were recorded with the torch
    profiler.
- **cuBLAS 13 `cublasSgemm`.** It runs with
  `cublasSetMathMode(CUBLAS_DEFAULT_MATH)`, so TF32 is off and FP32 means
  FP32. Every run records the math mode and checks that
  `NVIDIA_TF32_OVERRIDE` is unset. Handle creation is outside the timed
  region, and cuBLAS runs on the same stream and in the same batches as the
  custom kernels.

## Results

### How the numbers were measured

- **Timing:**
  - CUDA events around fixed-size batches of launches, with the same batch
    size for every implementation of a shape;
  - after warm-up and a clock re-ramp;
  - input buffers rotate through more than 4x the 2.3 MiB L2, so inputs
    are cold in L2 between launches.
- **Statistics:** 3 processes x 50 samples; medians with 95 %
  hierarchical-bootstrap CIs.
- **Checks:** outputs are poisoned and checked against the reference
  before and after every timed series.
- **Clocks** are not locked. SM clock and throttle reasons are recorded.

Full protocol: [docs/methodology.md](docs/methodology.md).

### Results by shape

All numbers below are kernel times. The ratio is comparator time /
KernelLens time, so a value above 1 means KernelLens is faster and below 1
means it is slower.

These are MEASURED values from three archived series, each at a development
revision of this code: softmax 64bbf58, layer norm b309644, GEMM 79a9ab2.
The kernels have not changed since; later commits changed only host-side
argument checks and comments. The series were not re-run for this release.

| Operator | Shape | KernelLens (AUTO choice) | Comparator (FP32) | KernelLens | Comparator | Ratio |
|---|---|---|---|---|---|---|
| softmax | 4096 x 128 | `FUSED_WARP` | `torch.softmax` | 15.5 us | 15.7 us | 1.01 |
| softmax | 1024 x 4096 | `FUSED_ONLINE` | `torch.softmax` | 158.4 us | 112.4 us | **0.71** |
| softmax | 4 x 65536 | `MULTIPASS` | `torch.softmax` | 72.1 us | 23.9 us | **0.33** |
| layer norm | 1024 x 4096 | `FUSED_SMEM` | `F.layer_norm` | 105.6 us | 157.0 us | 1.49 |
| layer norm | 256 x 16384 | `FUSED_BLOCK` | `F.layer_norm` | 205.9 us | 160.9 us | **0.78** |
| layer norm | 4 x 1048576 | `FUSED_BLOCK` | `F.layer_norm` | 567.4 us | 1684.1 us | 2.97 |
| GEMM | 1024 x 1024 x 1024 | `REGBLOCK` | cuBLAS, no TF32 | 0.820 ms (2.62 TFLOP/s) | 0.301 ms (7.13 TFLOP/s) | **0.37** |
| GEMM | 4096 x 4096 x 4096 | `REGBLOCK` | cuBLAS, no TF32 | 49.8 ms (2.76 TFLOP/s) | 16.9 ms (8.13 TFLOP/s) | **0.34** |
| GEMM | 64 x 4096 x 4096 | `REGBLOCK` | cuBLAS, no TF32 | 1.120 ms (1.92 TFLOP/s) | 0.285 ms (7.53 TFLOP/s) | **0.25** |

Sources: [softmax](results/processed/m2-softmax-summary.md),
[layer norm](results/processed/m3-layernorm-summary.md),
[GEMM](results/processed/m4-gemm-summary.md).

What the full series show:

- **Softmax:**
  - level with PyTorch (within 1.3 %) at 128-1024 columns;
  - slower at 2048-16384 columns (0.71-0.80x);
  - much slower with few long rows (0.27-0.58x), because every kernel uses
    one block per row and leaves most of the 28 SMs idle.
  - Against KernelLens's own baseline, `FUSED_WARP` is 2.0-8.3x faster than
    `MULTIPASS` at <= 1024 columns, at up to 322 GB/s effective. That is
    about 89 % of the ESTIMATED 360 GB/s DRAM peak.
- **Layer norm:**
  - 1.08-4.3x faster than PyTorch up to 8192 columns (the 4.3x at
    4096 x 32 is flagged as host-bound);
  - 0.74-0.89x for 16384-65536 columns with many rows, where
    `FUSED_BLOCK` reads each row three times;
  - faster again on very long rows.
- **GEMM:**
  - cuBLAS is 2.35-3.93x faster than `REGBLOCK` on every measured shape.
    `REGBLOCK` reaches 15-23 % of the ESTIMATED 12.7 TFLOP/s FP32 peak.
  - **Shared-memory tiling alone did not help.** `TILED` is 0.84-0.97x the
    speed of `NAIVE`, because L1 already serves the naive kernel's operand
    reuse: bypassing L1 slows `NAIVE` by 1.6-3.9x.
  - Register blocking is what helps: 2.65-3.74x over `TILED` from 1000^3
    up.
- **Host API vs kernel:**
  - `kl_softmax_host` at 1024 x 4096 takes 3.90 ms end to end (allocation,
    pageable copies, kernel), 24.7x its 0.158 ms kernel.
  - The host API is for tests. Its timings are never mixed with kernel
    timings.
- **Profiling** (Nsight Compute, one launch per kernel,
  [docs/profiling.md](docs/profiling.md)):
  - The three-read layer-norm kernel's re-reads hit the caches only while
    the rows in flight fit.
  - `REGBLOCK` is limited by shared-memory throttling, not DRAM.
  - The naive-vs-tiled explanation was INCONCLUSIVE under its pre-written
    rule.

**A fresh diagnostic** was run for this release: one process, 20 samples,
on the current code, not a full series. It agreed with the archived numbers:

- GEMM 1023^3: cuBLAS 0.31 ms, `REGBLOCK` 0.82 ms, `NAIVE` 2.54 ms, `TILED`
  2.80 ms.
- Softmax 1024 x 4096: `torch.softmax` 116 us vs `FUSED_ONLINE` 161 us.

All output checks passed. It is not evidence for the table above.

## Build, test and run

Requirements:

- Windows x64 (tested only on Windows 11) and an NVIDIA GPU with sm_86 or
  newer;
- a driver for CUDA >= 13.3;
- Visual Studio 2022 Build Tools with the C++ workload, and CUDA 13.3.1;
- CMake >= 3.26 and Ninja;
- Python 3.12.

Versions used and how they were verified: [docs/hardware.md](docs/hardware.md).
`scripts/dev-env.ps1` looks for CMake and Ninja under
`%LOCALAPPDATA%\Programs\kernellens-tools`; set `KERNELLENS_TOOLS` to use
another folder. Keep the clone at a short path if Windows long paths are
disabled.

PowerShell blocks scripts by default. Allow them for the current window
only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Create the Python environment. The torch wheel is about 2.1 GB:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --no-cache-dir -r requirements-lock.txt
```

Build:

```powershell
. .\scripts\dev-env.ps1
cmake --preset release
cmake --build --preset release
```

Run the C++ and Python tests under the GPU guard (log in
`results\raw\latest-ctest.log`):

```powershell
.\scripts\run-tests.ps1
```

Run the example from the top of this README:

```powershell
.venv\Scripts\python.exe scripts\gpu_guard.py --log results\raw\latest-example-gpu.csv -- .venv\Scripts\python.exe examples\softmax_example.py
```

Configuration fails if `.venv` is missing, so the Python tests cannot
silently drop out. Pass `-DKERNELLENS_REQUIRE_PYTHON_TESTS=OFF` to build
without them.

**Benchmarks** take 8-13 minutes per operator, under the guard. Absolute
times are specific to this GPU, driver and clock behaviour.

```powershell
.\scripts\run-bench.ps1 -Op softmax -Name mine
```

```powershell
.venv\Scripts\python.exe scripts\analyze_bench.py --op softmax --name mine
```

Use `-Op layernorm` / `--op layernorm` the same way. GEMM has its own
analysis script:

```powershell
.\scripts\run-bench.ps1 -Op gemm -Name mine
```

```powershell
.venv\Scripts\python.exe scripts\analyze_gemm.py --name mine
```

**Two optional tools need machine settings that only an administrator can
change** (docs/hardware.md):

- `.\scripts\run-sanitizer.ps1` needs the WDDM debugger interface enabled
  for Compute Sanitizer.
- `.\scripts\run-profile.ps1` needs GPU performance-counter access for
  Nsight Compute.

Without them the scripts report BLOCKED instead of passing.

Every GPU run goes through `scripts/gpu_guard.py`. It stops the run, killing
the command's whole process tree, at 80 C, on any thermal slowdown, on an
unreadable sensor, or at a time limit. Development rules:
[docs/development.md](docs/development.md).

## Verification status

- **Fresh, for this release:** a clean build from the staged public file
  set had 0 compiler warnings. All C++ test programs passed: smoke 11/11,
  softmax 17/17, layer norm 21/21, GEMM 11/11 groups. All 1056 Python tests
  passed with 0 skipped, including the GPU guard's process-tree kill. That
  test needs normal user rights over the processes it starts; see
  [docs/limitations.md](docs/limitations.md).
- **Archived Compute Sanitizer runs:** memcheck, racecheck, initcheck and
  synccheck reported 0 errors on all four test programs, at shapes capped
  for the 2 s watchdog (`results/raw/m5-sanitizer-*.log`). Not re-run for
  this release.
- **Archived mutation checks:** deliberately injected bugs are caught,
  9 of 9 for GEMM and 11 of 11 for layer norm
  (`results/raw/m4-mutation-gemm.txt`, `results/raw/m3-mutation-layernorm.txt`).
- **No CI.** There is no CI workflow, and no hosted run has tested this
  code. Hosted runners have no NVIDIA GPU.

## Limitations that affect interpretation

- **One machine:** one RTX 3060 (driver 610.60, CUDA 13.3.1, Windows 11,
  WDDM, GPU also drives the display). Numbers will differ elsewhere.
- **Clocks are not locked.** GEMM runs hit the board's power limit in 31 %
  of samples.
- **FP32, row-major, contiguous only.** GEMM has no transposes and no
  alpha/beta, and each dimension is <= 2^14.
- **Few long rows leave most SMs idle** for softmax and layer norm: one
  block per row, with no row splitting. Softmax is 0.27-0.58x PyTorch
  there.
- **The sanitizer and profiler evidence is narrow:** capped sizes, and one
  profiled launch per kernel.

Full list, with the tool, shapes and evidence for each claim:
[docs/limitations.md](docs/limitations.md).

## Credits and license

MIT license ([LICENSE](LICENSE)). Standard algorithms are credited where
used:

- SplitMix64 for the input generator;
- grid-stride loops;
- the online softmax normalizer (Milakov & Gimelshein, "Online normalizer
  calculation for softmax", 2018).

PyTorch and cuBLAS are used only as comparison baselines and test
references.
