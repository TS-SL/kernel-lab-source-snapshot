# Verification and benchmark methodology

## 1. Correctness

**References.** Each operator is compared with references that do not share
code with the kernel: a C++ host reference (`tests/cpp`) and numpy float64
and/or PyTorch (`tests/python`).

**Inputs.** Fixed seeds through SplitMix64 (`kl_fill_uniform` in C++,
`kl.fill_uniform` in Python - tested bit-identical over several seeds and
ranges, including near-FLT_MAX and tiny ranges). Shapes always include sizes
around warp (32) and block boundaries, n = 1, odd sizes, and a size that
needs more than one grid-stride pass. Special values: +/-0, +/-inf, NaN,
FLT_MAX, FLT_MIN, subnormals.

**Tolerances are derived, not tuned. SAXPY:**

- The kernel computes `fmaf(a, x, y)`: the exact `a*x + y` rounded once to
  float (round-to-nearest-even; no fast-math, subnormals kept).
- *Primary check - exact, bit for bit (signs of zero included, any NaN
  matches any NaN):*
  - C++: against host `std::fmaf`, which the C/C++ standard requires to round
    once. A self-check confirms the host implementation does so on a case
    where a naive one would not.
  - Python: against the float64 reference on every element where it is
    provably exact. The product of two floats is exact in double; TwoSum
    recovers the rounding error of the double addition, and where that
    error is 0 the double sum is the exact value, so converting it to float
    is the correctly rounded result. Random tests assert that >= 99 % of
    elements fall in this exact class, so the check has teeth.
  - Inputs use different magnitudes for x (+/-100) and y (+/-1e-3), so most
    float results are inexact; a kernel that rounded toward zero or up would
    fail on many elements.
- *Secondary bound - <= 1 ulp* against the float64 reference everywhere
  (one double rounding, then one float rounding: double rounding can be off
  by at most 1 ulp). The bound is shown to be tight by a hand-built case:
  `a = 24929 * 2^-15`, `x = 673 * 2^-9` give `a*x = 1 + 2^-24` exactly (the
  midpoint between 1 and 1 + 2^-23); with `y = 2^-60` the correct result is
  `1 + 2^-23`, the double reference gives `1.0`, and the GPU gives
  `1 + 2^-23` (tests `saxpy_double_rounding_case`,
  `test_saxpy_double_rounding_midpoint`).
- Comparing against an *unfused* float32 `a*x + y` would be wrong: under
  cancellation the separate rounding of the product can be off by many ulps
  of the result.
- *Softmax:* per-element bound derived from expf's 2 ulp,
  one rounding per operation and the reduction structure of each
  implementation; derivation, the tighter exact-differences variant, the
  subnormal slack and what the tests can and cannot detect are in
  `docs/softmax.md`.
- *Layer norm and GEMM:* derived bounds with stated validity domains,
  written before their tests (`docs/layernorm.md`, `docs/gemm.md`).

**ULP distance.** Float bit patterns are mapped to a monotonic integer line;
+0 and -0 are equal; NaN matches NaN only. Used only for the secondary
bound; the primary check is bitwise.

**Error paths.** Invalid arguments and over-cap requests must return the
documented status without touching output memory (checked with sentinel
values) and, for over-cap requests, without any `cudaMalloc` (checked with an
allocation counter). Launch failures must be reported with their source
location. Every regression test is shown to fail without its fix where
practical (done for the stale-error reset; docs/architecture.md, "Design
notes").

**Memory safety.** `scripts/run-sanitizer.ps1` runs memcheck (with leak
check), racecheck, initcheck and synccheck on the C++ tests with kernel sizes
capped at `--max-n 1048576` (sanitizers slow kernels 10-100x; the Windows
watchdog is 2 s), so the 16.7M-element grid-stride case is not run under the
sanitizer. Verdicts, checked in order: FAILED (program did not complete
cleanly), ERRORS (any sanitizer error besides the two known
debugger-interface start-up messages), BLOCKED (device interface
unavailable, nothing else reported), CLEAN.

## 2. Benchmark protocol

Implemented once and shared by every operator: `benchmarks/bench_harness.hpp`
(C++ harness) and `benchmarks/bench_protocol.py` (in-process PyTorch
comparison); per-operator files only supply shapes, buffers, launches and
checks (`bench_softmax.cpp`, `bench_layernorm.cpp`, `bench_*_compare.py`).
`scripts/run-bench.ps1 -Op <op>` runs a series and `scripts/analyze_bench.py
--op <op>` summarizes it.

**Preconditions.**
- Correctness tests pass for every implementation in the comparison.
- The GPU is idle: no other GPU-heavy process (check `nvidia-smi`
  utilization and the process list before starting; the GPU also drives the
  display, so desktop activity is a noise source to record, not remove).
- Runs go through `scripts/gpu_guard.py` (80 C stop, thermal slowdown stop,
  time limit); its 1 Hz log is kept with the results.

**Workloads.** Defined per operator before measuring: shape list, FP32,
row-major contiguous layout. Baseline and candidate get byte-identical inputs
(same seed), produce the same outputs, and are built with identical compiler
flags. Before timing and again after the timed series, every output buffer
is poisoned and one untimed pass of each implementation is checked against
the reference (the timed batches themselves are not re-read).
cuBLAS comparisons use the same shape, dtype and layout, with TF32 disabled
(`CUBLAS_DEFAULT_MATH`, no tensor-op math) so FP32 means FP32; handle
creation and workspace setup are outside the timed region.

**Bounds.** Each kernel launch well under 100 ms (Windows TDR is 2 s); total
device memory under the 2 GiB library cap; each series has a fixed
repetition count and a guard `--max-seconds` limit.

**Kernel timing.**
- CUDA events on the same stream, after warm-up launches (excludes module
  load and first-launch costs); `cudaEventSynchronize` before reading.
- Short kernels: time a batch of K back-to-back launches per event pair and
  divide by K (event resolution is about 0.5 us and WDDM batches
  submissions). K is a fixed function of the shape, the same for every
  implementation and both harnesses: chosen so a sample of an
  implementation moving the minimal 8 bytes/element at 300 GB/s lasts
  ~1 ms (`benchmarks/bench_common.hpp`). Calibrating K per implementation
  was tried first and rejected in review: a fixed per-batch cost is then
  divided by different K. A K-vs-2K linearity check estimates that cost.
- Host enqueue time is recorded per batch; a configuration whose host
  needs >= 80 % of the GPU time to enqueue is flagged as possibly
  host-bound.
- Comparisons with PyTorch run in ONE process (`bench_<op>_compare.py`):
  the PyTorch call and the KernelLens kernels (through `kl_<op>_device` on
  the same torch tensors and stream) interleaved ABBA with the same events
  and K, so both see the same clocks. Softmax uses
  `torch.softmax(x, -1, out=y)`; layer norm uses
  `torch.nn.functional.layer_norm` (its `out=` variant adds device copies,
  docs/layernorm.md) with results kept in a ring of output slots. Which
  kernels PyTorch launches is recorded with the torch profiler after the
  timed shapes. Comparisons between KernelLens implementations use the C++
  harness (no Python).
- Cache state is part of the workload: the device L2 here is 2304 KiB (CUDA
  device attribute, `results/raw/environment.txt`). A workload whose
  footprint fits in L2 is labelled *hot-cache*; to measure *cold-cache*
  (DRAM) behaviour, rotate across enough input buffers to exceed L2, or
  overwrite a scratch buffer larger than L2 between samples (outside the
  timed region). Reported numbers always say which.
- Clocks are not locked (changing clocks is outside this project's safety
  rules), so each series records SM clock and clock-event reasons
  (including power cap) from the guard log, alternates baseline and
  candidate (ABAB...) rather than running all of one first, and starts from
  the same idle temperature (cool-down pause between series).
- Nsight Compute durations are NOT mixed with event timings: by default
  `ncu` locks clocks to base and flushes caches between replays, so its
  times answer a different question. Profiler output is used to explain,
  event timing to measure.

**End-to-end timing.** Host-to-device copy + kernel + device-to-host copy
(plus allocation if the API allocates), host steady clock, reported
separately and labelled; host buffers are pinned (`cudaMallocHost`) unless
the measurement is explicitly about pageable memory.

**Repetitions and statistics.**
- At least 10 warm-up and 50 timed samples per configuration, repeated in
  at least 3 separate processes; a configuration whose coefficient of
  variation exceeds 5 % is reported as noisy, not silently re-run.
- Report median plus min, p10, p90, max, CV and IQR/median; a 95 %
  hierarchical-bootstrap confidence interval (resample processes, then
  samples within each) for the median and for each speedup, plus the range
  of per-process ratios. Samples within a process are autocorrelated, so
  pooling them as independent would overstate precision.
- A summary is labelled MEASURED only with >= 3 processes per harness on a
  clean commit (no uncommitted or untracked files) and, for every result
  group (implementation x shape), >= 3 processes, a sidecar from the same
  run as each CSV (same commit and start stamp) and both correctness checks
  in every process; otherwise the analysis script marks the group and the
  summary NOT EVIDENCE. The series runner deletes a run's old outputs
  before starting it.
- No outlier removal unless a rule is stated in advance; raw samples are
  always kept.
- `speedup = median(baseline) / median(candidate)` per shape, computed by a
  script from the raw file; a summary across shapes uses the geometric mean
  (over the shapes where the implementation runs) and lists the shapes
  where it is not shown faster (CI lower bound <= 1) and where it is slower
  (CI upper bound < 1). Shapes that leave most SMs idle (fewer than two rows
  per SM for one-block-per-row kernels) are reported separately.
- Predictions are written down before the measured series, with decision
  rules (PASS / FAIL / INCONCLUSIVE from the CI and the per-process ratios)
  that the analysis script applies; a prediction holds only if it is PASS
  on every shape where it is judged (layer norm and GEMM; the softmax
  predictions predate this rule).
- Achieved bandwidth = bytes the operator must move / median kernel time
  (MEASURED-derived): 8 bytes per element for softmax, 8 bytes per element
  + 8 per column for layer norm (gamma, beta). The RTX 3060's theoretical DRAM bandwidth (~360 GB/s
  from the 15 Gbps x 192-bit specification) is ESTIMATED and never presented
  as achieved.

**Recording.** Every sample goes to `results/raw/` (CSV/JSON) with operator,
implementation, shape, dtype, layout, GPU, driver, CUDA, compiler flags, git
commit, batch size K, repetition count, timing method, cache state, and
whether transfers or allocation are included.
