# Layer normalization

Written before the kernels; results are added only from executed runs.

## Definition and supported domain

Row-wise layer normalization of a contiguous, row-major FP32 matrix
`x[rows][cols]` over the last dimension, matching
`torch.nn.functional.layer_norm(x, (cols,), weight, bias, eps)`:

    mu_r     = (1/n) sum_j x[r][j]                          (n = cols)
    var_r    = (1/n) sum_j (x[r][j] - mu_r)^2               (biased, as PyTorch)
    y[r][j]  = (x[r][j] - mu_r) / sqrt(var_r + eps) * gamma[j] + beta[j]

`gamma` and `beta` are both given (length `cols`) or both NULL (then
`y = (x - mu) / sqrt(var + eps)`).

| Item | Supported | Otherwise |
|---|---|---|
| dtype / layout | FP32, contiguous row-major, normalization over the last dimension | - |
| rows, cols | `rows == 0` or `cols == 0`: no-op, `KL_OK` | negative: `KL_ERR_INVALID_ARGUMENT` |
| cols, rows, rows x cols | same limits as softmax: <= 2^20, <= 2^20, <= 2^26 | `KL_ERR_UNSUPPORTED` |
| `eps` | finite, `>= 0` | NaN, inf or negative: `KL_ERR_INVALID_ARGUMENT` |
| `gamma`, `beta` | both NULL or both non-NULL | one of them NULL: `KL_ERR_INVALID_ARGUMENT` |
| `KL_LAYERNORM_FUSED_WARP` | cols <= 1024 | `KL_ERR_UNSUPPORTED` |
| `KL_LAYERNORM_FUSED_SMEM` | cols <= 8192 | `KL_ERR_UNSUPPORTED` |

**Special values.** A row containing NaN or +-inf gives NaN for the whole
row; so does a finite row outside the bound's validity domain (below) whose
`x_j - x_0` overflows, e.g. `(-3e38, 3e38)`. A constant row gives NaN with
`eps = 0` (0/0) and exactly `beta` bit for bit with `eps > 0`, for any value
of the constant, because the mean (below) is exact for constant rows - with
one signed-zero exception: `fmaf(t, gamma_j, -0)` is +0 where `t * gamma_j`
is +0, so `y = +0` exactly where `beta_j = -0` and `t * gamma_j = +0` (`t`,
the normalized value, is +0 for elements equal to a nonzero constant and -0
for elements that are -0). Without affine a constant row gives `t`: +0, or
-0 at elements that are -0. NaN or +-inf in `beta`, or NaN in `gamma`, affects only that column.
With `gamma_j = +-inf`, column `j` is +-inf except where the computed
deviation `fl(x - mu_hat)` is exactly 0, which gives NaN (0 x inf) - also for
an element that differs from the exact mean by less than the mean's rounding.

**PyTorch 2.15**, probed by execution during development (*session
observations*): its CPU kernel gives the same results for constant rows
(NaN with `eps = 0`, exactly `beta` otherwise); its CUDA kernel gave 0, not
NaN, for a constant row of 1000 x 0.1 with `eps = 0` (NaN for 777 and 4096
elements), and -0 for a constant row of 1e36 (CPU: NaN; KernelLens: the exact
result, 0). The tests compare NaN and inf patterns with PyTorch's CPU kernel
on the special-value matrices, and with both kernels on two rows outside the
bound's validity domain (below), where all agree. The first version of these
kernels (unshifted mean) gave +-1 instead of NaN for constant rows such as
1000 x 0.1 with `eps = 0`, and `beta + O(u) gamma` instead of exactly `beta`
(found by review, see "Review corrections").

## Implementations

All four use the **two-pass** variance - mean first, then the sum of
squared deviations from it - which is the numerically stable choice
(below).

| Enum | Kernels | Global reads of x | Notes |
|---|---|---|---|
| `KL_LAYERNORM_MULTIPASS` (baseline) | 3: row mean, row variance, normalize | 3 | Mean and 1/sqrt(var + eps) pass through global memory. Block per row, 256 threads. |
| `KL_LAYERNORM_FUSED_BLOCK` | 1 | 3 (same block, back to back) | Block per row: mean, then variance, then normalize, with the stats in registers/shared memory. The second and third reads of a row follow the first immediately, so they may be served from L2 (the hypothesis to test). |
| `KL_LAYERNORM_FUSED_SMEM` | 1 | 1 | `FUSED_BLOCK` with the row cached in dynamic shared memory (cols x 4 bytes); cols <= 8192 (32 KiB). Added after the dry run (below). |
| `KL_LAYERNORM_FUSED_WARP` | 1 | 1 | Warp per row (4 rows per block), row in registers; cols <= 1024. |
| `KL_LAYERNORM_AUTO` | - | - | `FUSED_WARP` for cols <= 1024, `FUSED_SMEM` for cols <= 8192, else `FUSED_BLOCK`. Changed once after the dry run (below), before any measured series; the benchmark checks it. |

**Mean with a pivot shift.** Every public kernel forms the mean as
`mu_hat = x_0 + fl(sum_j (x_j - x_0)) / n`, with the row's first element as
the pivot (the shifted-data approach recommended by Chan, Golub & LeVeque,
1983). It costs one subtraction per element in the first pass and one
broadcast load per row. A constant row then sums exact zeros, so its mean is
exact. On its own the pivot makes the mean *less* accurate when `x_0` is an
outlier (its error grows with `mean |x_j - x_0|`, up to ~`n` times the
unshifted mean's), so it is combined with a **corrected two-pass mean**: the
variance pass, which already forms `d_j = x_j - m1` from the shifted mean
`m1`, also sums the `d_j` and the output uses `m = m1 + fl(sum_j d_j) / n`.
In exact arithmetic `sum_j d_j / n = mu - m1`, so the correction removes
`m1`'s error to first order whichever element is the pivot, and constant
rows stay exact (every `d_j` is 0). It costs one more block (or warp)
reduction per row. The variance itself is still taken around `m1` (its
error enters only squared). Both were added after review (see "Review
corrections"), so the dry-run timings below predate them.

**FUSED_SMEM invariant.** It uses `FUSED_BLOCK`'s thread mapping and the
same arithmetic in the same order, so its results are bitwise identical
(tested) and the same error bound applies. The first pass stores each
element it reads into shared memory; the variance and output passes read it
back from there. Each thread reads back only the elements it stored itself
(`j = threadIdx.x + i * 256`), so no barrier is needed between the store and
the loads (program order within a thread suffices), and thread `t` touches
word `t + 256 i`: consecutive lanes, consecutive banks, no bank conflicts.

**Not in the public API:** a *naive* kernel that computes the mean and
`E[x^2]` from one read, `var = E[x^2] - E[x]^2`, then reads the row again to
normalize (`kl::cuda::launch_layernorm_naive`, used only by tests and
benchmarks). It is the textbook formula whose error
grows with the square of `mean / std`; it is kept to *measure* that
instability, not to be used.

**Welford's algorithm** (single read, online mean/variance update) was
considered for the fused block kernel and not used: its published error
analyses (Chan, Golub & LeVeque, 1983) give the scaling but not constants
tight enough to serve as a test tolerance, and a crude derivation here gives
a bound quadratic in `mean / std` - useless as a test for the offset rows
that matter. The two-pass kernels have a bound derived below.

## Error bound (derived; used as the test tolerance)

Notation: `u = 2^-24`; `+ - * /` and `sqrtf` round correctly (IEEE, the
defaults `-prec-div=true`, `-prec-sqrt=true`; CUDA Programming Guide:
`sqrtf` 0 ulp); `1 / sqrt(...)` is computed as `1.0f / sqrtf(...)`, not with
`rsqrtf` (2 ulp). No fast-math flags. nvcc's default `-fmad=true` may
contract `q += d * d` into one `fmaf` (one rounding instead of two); every
step below allows the separate roundings, so the bound covers both.
`k` = elements per thread
(`ceil(n / T)`, `T` = 256 or 32), `L = ceil(log2(min(n, T)))` reduction levels
that combine real data (combining with an identity partial, `x + 0`, is
exact). For a row: `A = (1/n) sum |x_j - x_0|` (spread around the pivot),
`D = (1/n) sum |x_j - mu|` (spread around the mean), `sig = sqrt(var + eps)`,
`z_i = (x_i - mu) / sig`.

1. **Mean (pivot-shifted).** Each `x_j - x_0` rounds once (exactly when the
   result is subnormal), each shifted term then passes through at most
   `k - 1 + L` additions, the division by `n` (`n` exact as a float) rounds
   once - with an absolute error <= 2^-150 if the quotient is subnormal - and
   `fl(x_0 + m')` rounds once more. To first order
   `|m1 - mu| <= e1 = (k + L + 1) u A + u |mu| + 2^-150`. The `u |mu|` term
   is the cost of representing the mean in FP32, present in any FP32
   algorithm.
2. **Variance (two-pass, around `m1`).** Each term `fl(fl(x_j - m1)^2)` has
   relative error `<= 3u`; the positive terms pass through `k - 1 + L`
   additions; the division by `n` rounds once. Exactly,
   `(1/n) sum (x_j - m1)^2 = var + (m1 - mu)^2` (the deviations from `mu`
   sum to 0), so `var_hat = (var + (m1 - mu)^2)(1 + rho)` with
   `|rho| <= (k + L + 3) u`.
3. **Corrected mean.** The same `d_j = fl(x_j - m1)` (one rounding each)
   are summed through `k - 1 + L` additions and divided by `n` (one more
   rounding, absolute error <= 2^-150 if subnormal); `sum_j |d_j| / n
   <= D + e1`; exactly `(1/n) sum (x_j - m1) = mu - m1`; and `fl(m1 + c)`
   rounds once: `|m - mu| <= e2 = (k + L + 1) u (D + e1) + u |mu| + 2^-150`.
4. **Scale.** `fl(var_hat + eps)`, `sqrtf` and the division `1 / s` each
   round once; to first order `r_hat = (1 / sig)(1 + e_r)` with
   `|e_r| <= (k + L + 3) u / 2 + e1^2 / (2 sig^2) + 2.5 u`.
5. **Output.** `t = fl(fl(x_i - m) * r_hat)` adds two roundings and
   inherits the corrected mean's error: `|t - z_i| <= e2 / sig + |z_i| (2u + |e_r|)`.
   With affine, `y = fmaf(t, gamma, beta)` rounds once more.

Per element (first order; the `(e1 / sig)^2` term is kept because it is
computable and matters only for extreme rows):

    e1  = (k + L + 1) u A + u |mu| + 2^-150
    e2  = (k + L + 1) u (D + e1) + u |mu| + 2^-150
    E_i = e2 / sig + |z_i| ( (k + L + 3) u / 2 + 4.5 u + (e1 / sig)^2 / 2 )
    |y_i - y_i*| <= |gamma_i| E_i + u |y_i*|      (affine)
    |y_i - y_i*| <= E_i                           (no affine)

Tests allow `1.01 * bound + (1 + |gamma_i|) 2^-147`, with the reference in
float64: the factor covers second-order terms; the absolute term covers the
rounding of `t` in the subnormal range (<= 2^-150, scaled by `|gamma_i|`)
and of the fma (<= 2^-150).

**Validity domain.** The bound is derived, and claimed, only when
(a) no intermediate overflows: `sum_j |x_j - x_0| < 2^126` and
`sum_j (x_j - mu)^2 < 2^126` (the computed sum of squares around `m1`
exceeds the exact one by the factor `1 + (m1 - mu)^2 / var`; the rounding
part of `|m1 - mu|` is at most `sqrt(var)` and the accumulation part at most
`(k + L + 1) u (sqrt(n) + 1) sqrt(var)`, about `0.25 sqrt(var)` at
n = 2^20, so the factor is <= 1 + 1.5^2 = 3.25, below the factor 4 that
2^126 leaves under 2^128); and
(b) `var + eps >= 2^-100`, so the sum of
squares and the scale stay far from the subnormal range (subnormal squares
then perturb the variance by less than 2^-49 relative, inside the 1.01
factor). Both checkers (C++ and Python) refuse rows outside the domain
instead of passing them. Outside it no bound is claimed. Two documented
examples, pinned by tests and equal to PyTorch's CPU and CUDA results:
`(1e-30, -1e-30)` with `eps = 0` gives `(+inf, -inf)` (the squares underflow
to 0, so the scale is 1/0), and `(1e20, -1e20)` gives `(+0, -0)` (the
squares overflow, so the scale is 1/inf = 0). Tests inside the domain cover
magnitudes from 1e-14 to 1e20, subnormal inputs and subnormal outputs scaled
by `gamma = 1e6`.

**What this says about conditioning.** For a row with mean 1000 and spread
1, `e2 / sig ~ u |mu| / sig ~ 1000 u`: the cost of representing the mean,
present in any FP32 algorithm. The mean's own rounding scales with the
spread around the mean (`D / sig ~ 1`), whichever element is first; with
the unshifted mean of the first version it was `(k + L) u M1 / sig`
(`M1 = (1/n) sum |x_j|`), up to `k + L` times larger for offset rows, and
with the pivot alone (second version) it was `(k + L + 1) u A / sig`, up to
~`sqrt(n)` times larger for a row whose first element is an outlier. The
variance is still taken around `m1`, so an outlier pivot still reaches the
scale - quadratically, through the `(e1 / sig)^2 / 2` term the bound keeps:
negligible for typical rows, up to ~3 % relative in the worst case at
n = 2^20 (ESTIMATED). The standard corrected variance
`q / n - (sum_j d_j / n)^2` would remove it at the cost of one fma; not done
(future work). The two-pass variance keeps everything linear in
`mean / std`. The naive formula's variance error is
`~ (k + L) u (mean^2 + var) / var`, quadratic in `mean / std`, and
`var_hat` can come out negative (then NaN).

## Benchmark plan (pre-registered before any measured series)

Same protocol and harnesses as softmax (`benchmarks/bench_harness.hpp`,
`benchmarks/bench_protocol.py`, docs/methodology.md section 2): kernel-only
CUDA-event time per launch, fixed `K` per shape, buffers rotated over >= 4x
L2, ABBA interleaving, NVML clocks per batch, poisoned-output checks with
the tested bound before and after each shape, 3 processes per harness on a
clean commit. All rows use `x ~ U[-10, 10]`, `gamma ~ U[0.5, 1.5]`,
`beta ~ U[-1, 1]`, `eps = 1e-5`.

**Implementations measured:** `multipass`, `fused_block`, `fused_smem`
(cols <= 8192; added after the dry run, see below), `fused_warp`
(cols <= 1024), `naive_study` (C++ harness only: the internal
`E[x^2] - E[x]^2` kernel, to measure what the unstable formula would save)
and `torch_layer_norm` (Python harness only:
`torch.nn.functional.layer_norm` under `torch.no_grad()`; it launches one
`vectorized_layer_norm_kernel` per call for the shapes profiled during
development, and also writes per-row mean and rstd, 8 bytes per row. The
`aten.native_layer_norm.out` variant was rejected: the torch profiler shows
it adds three device-to-device copies per call).

**Shapes:** the 17 softmax shapes (comparable across operators), two
transformer-like shapes (8192 x 768, 4096 x 4096), and a cols sweep at 2^22
elements adding 512 x 8192 and 128 x 32768. End-to-end `kl_layernorm_host`
at 4096 x 128, 1024 x 4096 and 4 x 1048576.

**Traffic model and predictions (ESTIMATED, written before measuring).**
Minimum traffic is 8 bytes per element (read x, write y; gamma and beta are
8 bytes per column, reused by every row).
- `multipass` reads x three times in three kernels; every matrix above
  2 MiB is larger than the 2.25 MiB L2, so each read comes from DRAM:
  ~16 B/element plus three launches. Prediction: the slowest everywhere.
- `fused_warp` reads x once: ~8 B/element. Prediction: fastest of the
  KernelLens kernels wherever it applies (as for softmax).
- `fused_block` reads each row three times back to back. ptxas reports 40
  registers per thread (no spills), so 6 blocks of 256 threads fit per SM:
  up to 28 x 6 = 168 rows are in flight. Between two reads of an element,
  roughly the in-flight rows pass through L2, about
  `min(rows, 168) x cols x 4` bytes. Prediction: the re-reads mostly hit L2
  while that is well below 2.25 MiB (cols <= ~2048 at >= 168 rows: close to
  8 B/element) and mostly miss above it (cols >= 8192: close to
  16 B/element, i.e. no faster than `multipass` per byte; only the two saved
  launches and the stats round trip remain). The transition is predicted
  between 2048 and 8192 cols at 2^22 elements.
- `naive_study` reads twice: in the miss regime ~12 B/element, so
  predicted ~25 % faster than `fused_block` at cols >= 8192 and about equal
  where `fused_block`'s re-reads hit L2.
- `torch_layer_norm`: no prediction from its source (not inspected; how
  many times its kernel reads x is UNVERIFIED). The comparison is reported
  as measured.

**AUTO is fixed before the measured series**; the series checks whether its
choice is the fastest KernelLens implementation per shape.

### Change after the dry run (before any measured series)

A dry run of the C++ harness (one process, uncommitted code; *session
observations*, not evidence) measured `multipass`, `fused_block`,
`fused_warp` and `naive_study` on the shapes above. It matched the
`fused_block` prediction: about 290-320 GB/s effective at cols <= 2048,
about 200 GB/s at 4096 and about 160 GB/s from 8192 cols on (the same as
`multipass`); `naive_study` was about 1.3x faster than `fused_block` from
8192 cols on. In response `FUSED_SMEM` was added and AUTO changed from
"warp, else fused_block" to "warp, smem up to 8192 cols, else fused_block".
The AUTO limits come from the traffic model and the 8192-column shared-memory
limit, not from timings of `FUSED_SMEM`, which had not run yet.

**Predictions for `FUSED_SMEM` (ESTIMATED, written before it ran).** It reads
x once (~8 B/element). Shared memory limits the blocks per SM (100 KiB per
SM on sm_86, ~1 KiB reserved per block): 5 blocks at 4096 cols, 3 at 8192,
against 6 for `fused_block`. Prediction: about equal to `fused_block` at
cols <= 2048 (whose re-reads hit L2 there); at least 1.3x faster than
`fused_block` at 4096 and 8192 cols, where `fused_block` re-reads from DRAM,
unless the lower occupancy at 8192 limits its bandwidth (the
measurement decides). At cols <= 1024 `FUSED_WARP` stays the AUTO choice
(no prediction that `FUSED_SMEM` beats it).

### Review corrections and decision rules (after the dry run, before the measured series)

A static correctness and performance review of this plan and the code led
to the changes below
before the measured series. They are listed here, not edited into the
predictions above, which stay as first written.

**Code changes that affect timing.**
- The pivot-shifted mean (correctness review, round 1: constant rows whose
  sum is not exact in FP32, e.g. 1000 x 0.1, gave +-1 instead of NaN with
  `eps = 0` and not exactly `beta` otherwise). One extra subtraction per
  element and one broadcast load per row.
- The corrected two-pass mean (correctness review, round 2: with the pivot
  alone a row whose first element is an outlier had a mean error ~280-2750x
  the unshifted kernel's in a CPU emulation, still within the
  then-bound because it grew with `A`; a test now pins these rows). One
  extra addition per element in the variance pass and one more reduction
  per row. The dry-run timings predate both changes.
- `FUSED_SMEM` requests the largest shared-memory carveout
  (`cudaFuncAttributePreferredSharedMemoryCarveout`); without it its blocks
  per SM would depend on the driver's choice (5 blocks at 4096 cols and 3 at
  8192 need the 100 KiB carveout). The harness records the resulting
  occupancy (`cudaOccupancyMaxActiveBlocksPerMultiprocessor`) per kernel
  and shape.

**Corrections to the traffic model** (performance review; the predictions
above are not changed):
- The loads compile to `LDG.E.CONSTANT` (through L1). With 6 blocks per SM
  the rows in flight on one SM take 6 x cols x 4 bytes (48 KiB at 2048
  cols, 96 KiB at 4096), so some of `fused_block`'s re-reads may be served by
  L1 rather than L2. "L2 hit" is one of two hypotheses; the profiler that
  could tell them apart is BLOCKED.
- 168 x cols x 4 bytes equals the L2 size at about 3500 cols, and pass 3's
  writes and gamma/beta also occupy L2, so the transition is probably below
  that.
- "Every matrix above 2 MiB" does not cover 4096 x 32 (0.5 MiB) and
  4096 x 128 (2 MiB), where `multipass`'s re-reads can hit L2: "slowest
  everywhere" is not derived for those two shapes (the summary marks them).
- `naive_study`'s modelled advantage from 8192 cols on is a time ratio of
  16/12 = 1.33x (not "25 % faster").

**Decision rules.** Written in `scripts/analyze_bench.py` after the dry run
and before any measured series, then copied here. In between, a
one-process harness check (2 shapes, 6 samples; not evidence) ran, and its
summary - including `fused_smem`'s time at 1024 x 4096, a P5 shape - was
viewed, so P5 is not blind for that shape; the thresholds were not changed
afterwards, and code and rules are committed before the measured series.
Ratios are time(other) / time(candidate) from the C++ harness with 95 %
hierarchical-bootstrap CIs; few-row shapes (rows < 56, fewer than two rows
per SM) are reported separately and are not judged.
- **P1** `fused_warp` faster than each other public implementation where it
  applies: PASS if the CI lower bound > 1, FAIL if the upper bound < 1,
  else TIE.
- **P2** each other public implementation faster than `multipass` (same
  rule; marked "not derived" where x fits in L2).
- **P3a** `fused_block` / `multipass` time <= 0.80 at cols <= 2048 and
  rows >= 168: PASS if the CI upper bound <= 0.80, FAIL if the lower bound
  > 0.80, else INCONCLUSIVE. **P3b** >= 0.85 at cols >= 8192: PASS if the
  lower bound >= 0.85, FAIL if the upper bound < 0.85, else INCONCLUSIVE.
- **P4** `naive_study` >= 1.15x faster than `fused_block` at cols >= 8192:
  PASS if the lower bound >= 1.15, FAIL if the upper bound < 1.15, else
  INCONCLUSIVE.
- **P5** `fused_smem` >= 1.30x faster than `fused_block` at 4096 and 8192
  cols: PASS if the lower bound >= 1.30 and every per-process ratio
  >= 1.30, FAIL if the upper bound < 1.30, else INCONCLUSIVE.
- **P6** `fused_block` / `fused_smem` within [0.90, 1.10] at cols <= 2048:
  PASS if the whole CI lies inside, FAIL if it lies entirely outside, else
  INCONCLUSIVE.
- **P7** AUTO runs the fastest implementation, or one tied with it (CI of
  the ratio including 1), on every judged shape.

A prediction **holds** only if it is PASS on every shape where it is judged;
the summary prints the counts per prediction. Judged shapes exclude the
few-row shapes, the rows sweep (except 64 x 65536) and, for P2 and P3a, the
shapes whose x fits in L2 (4096 x 32, 4096 x 128), where those predictions
are not derived.

P1-P4 concern kernels the dry run had already measured, so for them the
series is a replication, not a test; P5-P7 concern `FUSED_SMEM` and AUTO,
whose thresholds were fixed before `FUSED_SMEM` was timed.

**Controls and shapes added** (C++ harness, for 1024 < cols <= 8192):
`fused_block_occ` runs `FUSED_BLOCK` with unused dynamic shared memory equal
to `FUSED_SMEM`'s, so both have the same blocks per SM and differ only in
reading x once or three times; `fused_smem_pad32k` pads `FUSED_SMEM` to
32 KiB (3 blocks per SM, as at 8192 cols) at 2048 and 4096 cols, to show
what that occupancy costs. Both use separate internal launchers
(`kl::cuda::launch_layernorm_padded`) and the same arithmetic as the public
kernels. Shape 2048 x 8192 was added: at 512 x 8192 `fused_block` runs about
3 waves of in-flight rows and `fused_smem` about 6, so tail effects differ;
at 2048 x 8192 they are 4x smaller. Not added: an `__ldcg` (L1-bypassing)
variant of `fused_block`; instead `fused_block_maxsmem` runs
`FUSED_BLOCK`'s control copy with the largest shared-memory carveout (small
L1) and no padding, at every shape with cols <= 8192: against `fused_block`
it isolates L1 capacity, against `fused_block_occ` (same carveout) it
isolates occupancy. The occupancy values are API-reported
(`cudaOccupancyMaxActiveBlocksPerMultiprocessor` after the carveout
preference): the carveout is a hint, and which split the driver applied is
not observable here.

**Harness and analysis changes.** The torch profiler call moved after the
last timed shape (it attaches to the process); torch's output is checked in
NaN-filled blocks handed to its allocator (the sidecar records whether the
checked output occupied one: a fresh write), and a loose sanity limit (10x
the `fused_block` bound) aborts the series if the baseline is broken; the
host cost of one rows = 0 call is recorded for both sides; effective
bandwidth for layer norm counts gamma and beta (8 bytes per element + 8
bytes per column); a result group is MEASURED only with 3 processes, every
sidecar and both correctness checks in every process; the "not faster" list
is split into "not shown faster" and "slower" (the softmax summary was
regenerated with these labels: no number changed and no softmax shape
moved between the lists). Later: the series runner deletes a run's old
CSV, sidecar and guard log before starting it, and the analysis accepts a
sidecar only if its commit and start stamp equal its CSV's (a crashed
process can no longer be certified by a sidecar from an earlier run); the
torch fresh-write flag is informational (in torch's small allocator pool,
outputs <= 1 MiB, a `False` can be a false negative).

**Dry-run data** were kept only in a scratch directory outside the
repository; the dry-run numbers above are unarchived observations.

## Results (MEASURED, commit b309644)

Source: `results/processed/m3-layernorm-summary.md` (generated by
`scripts/analyze_bench.py --op layernorm` from `results/raw/m3-bench-layernorm-*`:
3 C++ and 3 Python processes, 27,450 samples, status MEASURED; peak GPU
temperature 64 C, no guard stop). Times are kernel-only medians with 95 %
hierarchical-bootstrap CIs; GB/s is the minimal traffic (8 B/element +
8 B/column) divided by the median. Every benchmarked output passed the
derived bound (948 checks of KernelLens outputs, worst 0.33 of the bound,
0 violations; plus 132 checks of torch's output);
torch's output was verified as a fresh write in 132 of 132 checks.

**Predictions** (decision rules above; P1-P4 are replications of the dry
run, P5-P7 were not seen before the rules):

| Prediction | Judged shapes | PASS | FAIL | Holds |
|---|---|---|---|---|
| P1 fused_warp fastest where it applies | 15 | 13 | 2 | **no**: at 4096 x 1024 fused_block and fused_smem are 1-2 % faster |
| P2 multipass slowest | 22 | 22 | 0 | yes |
| P3a fused_block re-reads served from cache at cols <= 2048 | 4 | 4 | 0 | yes (fused_block / multipass time 0.50-0.62) |
| P3b fused_block no cheaper than multipass per byte from 8192 cols | 5 | 5 | 0 | yes (0.96-0.99) |
| P4 naive >= 1.15x faster than fused_block from 8192 cols | 5 | 5 | 0 | yes (1.31-1.32x; model 1.33x) |
| P5 fused_smem >= 1.30x faster than fused_block at 4096 / 8192 cols | 4 | 4 | 0 | yes (1.58-1.97x) |
| P6 fused_smem within 10 % of fused_block at cols <= 2048 | 6 | 5 | 1 | **no**: at 2048 x 2048 fused_smem is 1.11x faster (fused_block's re-reads already start to miss) |
| P7 AUTO runs the fastest (or tied) implementation | 13 | 12 | 1 | **no**: at 4096 x 1024 AUTO's fused_warp is 1.02x slower than fused_smem |

AUTO is not changed after the series (that would fit it to this data); the
4096 x 1024 shape is a candidate for a later, separately measured change.

**KernelLens implementations (C++ harness):**
- `fused_smem` reads x once and runs at 314-329 GB/s from 1024 to 8192 cols
  (~90 % of the ESTIMATED 360 GB/s peak), 1.98-2.01x faster than
  `multipass` there. `fused_warp` reaches 307-320 GB/s at 512-1024 cols and
  is 9.2x faster than `multipass` at 4096 x 32 (where multipass is mostly
  launch-bound).
- `fused_block` (three reads) matches `fused_smem` up to 1024 cols and falls
  from 289 GB/s (2048 cols) to 197-202 (4096) and 165-166 (8192), at 8192 cols the same as
  `multipass` per byte (159-165 GB/s) - the pre-registered L2/L1 transition, now with
  3 processes.
- The cost of numerical stability: from 8192 cols the unstable two-read
  `naive_study` is 1.31-1.32x faster than `fused_block`, but `fused_smem`
  (stable, one read) is 1.91-1.97x faster than `fused_block` at 8192 cols,
  so stability is not what costs time where `fused_smem` applies.

**Controls** (what the timing differences come from; profiler counters are
BLOCKED, so these are timing evidence, not cache measurements):
- *One read vs three at equal occupancy and carveout:* `fused_smem` is
  1.49x (2048 cols) to 1.92x (8192 cols) faster than `fused_block_occ`
  (ratio fused_smem/occ time 0.52-0.67). The read count, not occupancy,
  explains `fused_smem`'s advantage.
- *Occupancy:* reducing `fused_block` from 6 to 5 or 3 blocks per SM (same
  carveout) changes its time by 0-5 %, and padding `fused_smem` to 3 blocks
  per SM changes it by 0-1 %: 3 blocks per SM are enough for this
  streaming kernel.
- *L1 capacity:* the same `fused_block` with the largest shared-memory
  carveout (small L1, same 6 blocks per SM) is up to 1.33x slower at 2048
  cols (ratio 0.75), 1.16-1.20x slower at 4096 cols and about 1.03-1.16x at
  1024 cols (CI of a noisy group: CV 6.3 %),
  and unchanged at >= 8192 cols. So part of `fused_block`'s re-read hits at 1024-4096 cols are served
  by L1 - the hypothesis the review raised; L2's share is not separable
  without counters.

**Against PyTorch** (`torch.nn.functional.layer_norm`, same process, stream
and clocks; ratio torch time / KernelLens AUTO time):
- KernelLens is faster at every shape up to 8192 cols: 4.3x at 4096 x 32
  (both near launch overhead; flagged host-bound for KernelLens), 2.0x at
  4096 x 128, 1.08x at 4096 x 512, 1.24x at 8192 x 768, 1.34x at
  4096 x 1024, and 1.45-1.50x from 2048 to 8192 cols with `fused_smem`.
- PyTorch is faster for long rows with many rows: 1.28-1.31x at 16384 and
  32768 cols, 1.12-1.34x at 56-224 x 65536 (torch / KernelLens 0.74-0.89),
  where KernelLens falls back to the three-read `fused_block`.
- KernelLens is faster for few long rows (1.30-2.97x at 4-28 x 65536,
  16 x 262144 and 4 x 1048576); why PyTorch is slow there is UNVERIFIED
  (only its kernel name was recorded: `vectorized_layer_norm_kernel` for
  every shape).
- The two harnesses agree on KernelLens kernel times within 1 % (0.99-1.01).

**End-to-end** `kl_layernorm_host` (pageable copies + per-call allocation)
takes 9.7-53x the kernel time - it is a correctness API, not a fast path.

**Variability:** 7 of the groups have CV > 5 % (listed in the summary); all
have IQR/median <= 3.2 % except two (Python fused_warp 4096 x 32, a
launch-bound shape, and C++ fused_block_maxsmem 4096 x 1024); medians are
unaffected. Samples with low NVML clock readings have medians within 2 % of
the others in all 50 groups where both exist (post-batch idle snapshots).

**Harness incident** (*unarchived observation*; its files were not kept): a first attempt at this series was interrupted after
three of six runs when the controlling terminal session ended; its partial files were
deleted unanalyzed and the whole series was rerun.
